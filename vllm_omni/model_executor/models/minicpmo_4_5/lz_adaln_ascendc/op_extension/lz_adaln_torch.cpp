// SPDX-License-Identifier: Apache-2.0
/* P28: FusedAdaLN host-side launcher (kernel direct-invoke).
 *
 * Tiling is computed on the host and copied to a small device tensor. The
 * device tiling tensors are cached by value: steady-state calls (including
 * NPUGraph capture, which always follows an eager warmup in this repo) never
 * issue an H2D memcpy, which would be illegal under stream capture.
 */
#include <acl/acl.h>
#include <cstdint>
#include <map>
#include <tuple>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "../op_kernel/lz_adaln_tiling.h"

// Kernel entry generated from op_kernel/lz_adaln_gate_kernel.asc by bisheng.
extern "C" void lz_adaln_gate_kernel(uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
                                     uint8_t* x, uint8_t* shift, uint8_t* scale, uint8_t* gate,
                                     uint8_t* y, uint8_t* tiling);

namespace lz_adaln {
namespace {

using TilingKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>;

// Single threaded per vLLM stage-2 process; the cache only ever grows by a
// handful of 32-byte device tensors (one per distinct launch shape).
std::map<TilingKey, at::Tensor>& tilingCache()
{
    static std::map<TilingKey, at::Tensor> cache;
    return cache;
}

const at::Tensor& deviceTiling(const at::Tensor& like, const AdaLNTilingData& tiling)
{
    const TilingKey key{tiling.numBlocks, tiling.totalRows, tiling.rowsPerCore,
                        tiling.chunkT, tiling.paramStride, tiling.mode};
    auto& cache = tilingCache();
    auto it = cache.find(key);
    if (it == cache.end()) {
        at::Tensor t = at::empty({static_cast<int64_t>(sizeof(AdaLNTilingData))},
                                 like.options().dtype(at::kByte));
        auto ret = aclrtMemcpy(t.mutable_data_ptr(), sizeof(AdaLNTilingData), &tiling,
                                     sizeof(AdaLNTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(ret == ACL_SUCCESS, "lz_adaln: aclrtMemcpy for tiling failed: ", static_cast<int>(ret));
        it = cache.emplace(key, t).first;
    }
    return it->second;
}

void checkCommon(const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale)
{
    TORCH_CHECK(x.scalar_type() == at::kFloat, "lz_adaln: x must be fp32");
    TORCH_CHECK(shift.scalar_type() == at::kFloat, "lz_adaln: shift must be fp32");
    TORCH_CHECK(scale.scalar_type() == at::kFloat, "lz_adaln: scale must be fp32");
    TORCH_CHECK(x.is_privateuseone(), "lz_adaln: x must be on NPU");
    TORCH_CHECK(shift.is_privateuseone() && scale.is_privateuseone(),
                "lz_adaln: shift/scale must be on NPU");
    TORCH_CHECK(x.dim() == 3, "lz_adaln: x must be [B, T, 512], got dim ", x.dim());
    TORCH_CHECK(x.size(2) == LZ_ADALN_ROW_LEN,
                "lz_adaln: hidden size must be ", LZ_ADALN_ROW_LEN, ", got ", x.size(2));
    TORCH_CHECK(x.is_contiguous(), "lz_adaln: x must be contiguous");
    TORCH_CHECK(x.numel() > 0, "lz_adaln: x must not be empty");
    TORCH_CHECK(shift.size(-1) == LZ_ADALN_ROW_LEN && scale.size(-1) == LZ_ADALN_ROW_LEN,
                "lz_adaln: shift/scale last dim must be ", LZ_ADALN_ROW_LEN);
    TORCH_CHECK(shift.numel() == x.size(0) * LZ_ADALN_ROW_LEN &&
                    scale.numel() == x.size(0) * LZ_ADALN_ROW_LEN,
                "lz_adaln: shift/scale must carry one 512-wide row per batch element");
    TORCH_CHECK(shift.stride(-1) == 1 && scale.stride(-1) == 1,
                "lz_adaln: shift/scale last dim must be contiguous");
    const int64_t b = x.size(0);
    const int64_t minStride = b > 1 ? LZ_ADALN_ROW_LEN : 1;
    TORCH_CHECK(shift.stride(0) >= minStride && scale.stride(0) >= minStride,
                "lz_adaln: shift/scale batch stride too small (overlapping rows)");
    // The kernel reads all param rows with ONE stride (from shift): scale/gate
    // must share shift's row layout. The DiT chunk(9) path always does; the
    // Python wrapper normalizes any other layout before calling in.
    if (b > 1) {
        TORCH_CHECK(scale.stride(0) == shift.stride(0),
                    "lz_adaln: scale row stride (", scale.stride(0),
                    ") must match shift (", shift.stride(0), ")");
    }
    const auto checkAligned = [](const at::Tensor& t, const char* name) {
        const auto addr = reinterpret_cast<std::uintptr_t>(t.const_data_ptr());
        TORCH_CHECK(addr % 32 == 0, "lz_adaln: ", name, " base address not 32B aligned");
    };
    checkAligned(shift, "shift");
    checkAligned(scale, "scale");
}

at::Tensor launch(const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale,
                  const at::Tensor* gate, uint32_t mode)
{
    checkCommon(x, shift, scale);
    if (gate != nullptr) {
        TORCH_CHECK(gate->scalar_type() == at::kFloat, "lz_adaln: gate must be fp32");
        TORCH_CHECK(gate->is_privateuseone(), "lz_adaln: gate must be on NPU");
        TORCH_CHECK(gate->size(-1) == LZ_ADALN_ROW_LEN && gate->numel() == x.size(0) * LZ_ADALN_ROW_LEN,
                    "lz_adaln: gate must carry one 512-wide row per batch element");
        TORCH_CHECK(gate->stride(-1) == 1 && (x.size(0) == 1 || gate->stride(0) >= LZ_ADALN_ROW_LEN),
                    "lz_adaln: gate stride too small");
        if (x.size(0) > 1) {
            TORCH_CHECK(gate->stride(0) == shift.stride(0),
                        "lz_adaln: gate row stride (", gate->stride(0),
                        ") must match shift (", shift.stride(0), ")");
        }
        const auto addr = reinterpret_cast<std::uintptr_t>(gate->const_data_ptr());
        TORCH_CHECK(addr % 32 == 0, "lz_adaln: gate base address not 32B aligned");
    }

    at::Tensor y = at::empty_like(x);

    const int64_t batch = x.size(0);
    const int64_t chunkT = x.size(1);
    const int64_t totalRows64 = batch * chunkT;
    TORCH_CHECK(totalRows64 <= 0xffffffffll, "lz_adaln: too many rows");

    int32_t deviceId = -1;
    TORCH_CHECK(aclrtGetDevice(&deviceId) == ACL_SUCCESS, "lz_adaln: aclrtGetDevice failed");
    int64_t vectorCores = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCores) == ACL_SUCCESS &&
                    vectorCores > 0,
                "lz_adaln: failed to query vector core count");
    uint32_t numBlocks = static_cast<uint32_t>(std::min<int64_t>(vectorCores, totalRows64));
    uint32_t rowsPerCore =
        static_cast<uint32_t>((totalRows64 + numBlocks - 1) / numBlocks);
    numBlocks = static_cast<uint32_t>((totalRows64 + rowsPerCore - 1) / rowsPerCore);

    AdaLNTilingData tiling;
    tiling.numBlocks = numBlocks;
    tiling.totalRows = static_cast<uint32_t>(totalRows64);
    tiling.rowsPerCore = rowsPerCore;
    tiling.chunkT = static_cast<uint32_t>(chunkT);
    tiling.paramStride = static_cast<uint32_t>(shift.stride(0));
    tiling.mode = mode;
    tiling.reserved[0] = 0;
    tiling.reserved[1] = 0;
    TORCH_CHECK(shift.stride(0) <= 0xffffffffll, "lz_adaln: param stride overflow");

    const at::Tensor& tilingTensor = deviceTiling(x, tiling);

    // stream(true) drains torch_npu's pending task queue before the raw
    // launch, keeping ordering with preceding eager ops (see cannbot
    // direct-invoke template).
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    const uint8_t* gatePtr = gate != nullptr
                                 ? reinterpret_cast<const uint8_t*>(gate->const_data_ptr())
                                 : reinterpret_cast<const uint8_t*>(shift.const_data_ptr());

    lz_adaln_gate_kernel(
        numBlocks, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(x.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(const_cast<void*>(shift.const_data_ptr())),
        reinterpret_cast<uint8_t*>(const_cast<void*>(scale.const_data_ptr())),
        const_cast<uint8_t*>(gatePtr),
        reinterpret_cast<uint8_t*>(y.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));
    return y;
}

}  // namespace

at::Tensor fused_adaln_gate_torch(const at::Tensor& x, const at::Tensor& shift,
                                  const at::Tensor& scale, const at::Tensor& gate)
{
    return launch(x, shift, scale, &gate, LZ_ADALN_MODE_GATE_RESIDUAL);
}

at::Tensor fused_adaln_norm_torch(const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale)
{
    return launch(x, shift, scale, nullptr, LZ_ADALN_MODE_MODULATE);
}

}  // namespace lz_adaln

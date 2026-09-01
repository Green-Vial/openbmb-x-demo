// SPDX-License-Identifier: Apache-2.0
/* P28b: FusedQkNorm host-side launcher (kernel direct-invoke).
 *
 * Mirrors lz_adaln_torch.cpp: tiling computed on host, cached BY VALUE on the
 * device (steady-state calls and NPUGraph capture never issue an H2D memcpy).
 * New vs P28: two inputs (q, k) with EXPLICIT B/H/T element strides — the
 * transposed to_heads views are consumed in place, no contiguous() copy —
 * and two contiguous outputs.
 */
#include <acl/acl.h>
#include <cstdint>
#include <cstring>
#include <map>
#include <tuple>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include "../op_kernel/lz_qk_norm_tiling.h"

// Kernel entry generated from op_kernel/lz_qk_norm_kernel.asc by bisheng.
extern "C" void lz_qk_norm_kernel(uint32_t blockDim, void* l2Ctrl, aclrtStream stream,
                                  uint8_t* q, uint8_t* k, uint8_t* qw, uint8_t* qb,
                                  uint8_t* kw, uint8_t* kb, uint8_t* qOut, uint8_t* kOut,
                                  uint8_t* tiling);

namespace lz_qknorm {
namespace {

using TilingKey = std::tuple<uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t,
                             uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>;

std::map<TilingKey, at::Tensor>& tilingCache()
{
    static std::map<TilingKey, at::Tensor> cache;
    return cache;
}

void checkInput(const at::Tensor& x, const char* name)
{
    TORCH_CHECK(x.scalar_type() == at::kFloat, "lz_qk_norm: ", name, " must be fp32");
    TORCH_CHECK(x.is_privateuseone(), "lz_qk_norm: ", name, " must be on NPU");
    TORCH_CHECK(x.dim() == 4, "lz_qk_norm: ", name, " must be [B, nh, T, hd], got dim ", x.dim());
    TORCH_CHECK(x.size(3) % 8 == 0, "lz_qk_norm: headDim must be a multiple of 8 (32B rows), got ",
                x.size(3));
    TORCH_CHECK(x.stride(3) == 1, "lz_qk_norm: ", name, " last dim must be contiguous");
    // Every row's starting address must be 32B aligned for DataCopy: with a
    // 32B-aligned base that reduces to all strides being multiples of 8 elems.
    const auto aligned = [](int64_t s) { return s % 8 == 0; };
    TORCH_CHECK(aligned(x.stride(0)) && aligned(x.stride(1)) && aligned(x.stride(2)),
                "lz_qk_norm: ", name, " B/H/T strides must be multiples of 8 elements");
    const auto addr = reinterpret_cast<std::uintptr_t>(x.const_data_ptr());
    TORCH_CHECK(addr % 32 == 0, "lz_qk_norm: ", name, " base address not 32B aligned");
}

void checkParam(const at::Tensor& p, int64_t headDim, const char* name)
{
    TORCH_CHECK(p.scalar_type() == at::kFloat, "lz_qk_norm: ", name, " must be fp32");
    TORCH_CHECK(p.is_privateuseone(), "lz_qk_norm: ", name, " must be on NPU");
    TORCH_CHECK(p.numel() == headDim && p.stride(-1) == 1, "lz_qk_norm: ", name, " must be [",
                headDim, "] contiguous");
    const auto addr = reinterpret_cast<std::uintptr_t>(p.const_data_ptr());
    TORCH_CHECK(addr % 32 == 0, "lz_qk_norm: ", name, " base address not 32B aligned");
}

}  // namespace

std::tuple<at::Tensor, at::Tensor> fused_qk_norm_torch(const at::Tensor& q, const at::Tensor& k,
                                            const at::Tensor& qw, const at::Tensor& qb,
                                            const at::Tensor& kw, const at::Tensor& kb, double eps)
{
    checkInput(q, "q");
    checkInput(k, "k");
    TORCH_CHECK(q.sizes() == k.sizes(), "lz_qk_norm: q and k must have identical shapes");
    const int64_t headDim = q.size(3);
    checkParam(qw, headDim, "qw");
    checkParam(qb, headDim, "qb");
    checkParam(kw, headDim, "kw");
    checkParam(kb, headDim, "kb");

    const int64_t b = q.size(0);
    const int64_t nh = q.size(1);
    const int64_t dimT = q.size(2);
    TORCH_CHECK(b * nh * dimT > 0, "lz_qk_norm: empty input");
    // Work unit = one (b, t) position: the nh head rows must be CONTIGUOUS
    // (H stride == headDim) for the batched-heads kernel design. Real
    // to_heads views always satisfy this; anything else falls back to eager
    // via the Python wrapper.
    TORCH_CHECK(q.stride(1) == headDim && k.stride(1) == headDim,
                "lz_qk_norm: H stride must equal headDim (contiguous head rows)");
    TORCH_CHECK(nh <= 8, "lz_qk_norm: nh > 8 unsupported (scalar scratch sized for 8)");

    at::Tensor qOut = at::empty_like(q, q.options());  // contiguous output
    at::Tensor kOut = at::empty_like(k, k.options());

    const int64_t batches = b * dimT;
    // The kernel's unit space is [0, 2*batches): [0, batches) = q, the rest = k.
    // Tiling MUST split 2*batches units across cores, not batches — otherwise
    // every k unit goes unassigned and its output stays uninitialized.
    const int64_t totalUnits = batches * 2;
    int32_t deviceId = -1;
    TORCH_CHECK(aclrtGetDevice(&deviceId) == ACL_SUCCESS, "lz_qk_norm: aclrtGetDevice failed");
    int64_t vectorCores = 0;
    TORCH_CHECK(aclrtGetDeviceInfo(deviceId, ACL_DEV_ATTR_VECTOR_CORE_NUM, &vectorCores) == ACL_SUCCESS &&
                    vectorCores > 0,
                "lz_qk_norm: failed to query vector core count");
    uint32_t numBlocks = static_cast<uint32_t>(std::min<int64_t>(vectorCores, totalUnits));
    uint32_t unitsPerCore = static_cast<uint32_t>((totalUnits + numBlocks - 1) / numBlocks);
    numBlocks = static_cast<uint32_t>((totalUnits + unitsPerCore - 1) / unitsPerCore);

    LzQkNormTilingData tiling;
    tiling.numBlocks = numBlocks;
    tiling.batches = static_cast<uint32_t>(batches);
    tiling.unitsPerCore = unitsPerCore;
    tiling.headDim = static_cast<uint32_t>(headDim);
    tiling.nh = static_cast<uint32_t>(nh);
    tiling.dimT = static_cast<uint32_t>(dimT);
    tiling.qStrideB = static_cast<uint32_t>(q.stride(0));
    tiling.qStrideT = static_cast<uint32_t>(q.stride(2));
    tiling.kStrideB = static_cast<uint32_t>(k.stride(0));
    tiling.kStrideT = static_cast<uint32_t>(k.stride(2));
    tiling.eps = static_cast<float>(eps);
    tiling.reserved = 0;

    const TilingKey key{numBlocks,      tiling.batches, tiling.unitsPerCore, tiling.headDim,
                        tiling.nh,      tiling.dimT,    tiling.qStrideB,     tiling.qStrideT,
                        tiling.kStrideB, tiling.kStrideT,
                        static_cast<uint32_t>(tiling.eps * 1e9f)};
    auto& cache = tilingCache();
    auto it = cache.find(key);
    if (it == cache.end()) {
        at::Tensor t = at::empty({static_cast<int64_t>(sizeof(LzQkNormTilingData))},
                                 q.options().dtype(at::kByte));
        auto ret = aclrtMemcpy(t.mutable_data_ptr(), sizeof(LzQkNormTilingData), &tiling,
                               sizeof(LzQkNormTilingData), ACL_MEMCPY_HOST_TO_DEVICE);
        TORCH_CHECK(ret == ACL_SUCCESS, "lz_qk_norm: aclrtMemcpy for tiling failed: ",
                    static_cast<int>(ret));
        it = cache.emplace(key, t).first;
    }
    const at::Tensor& tilingTensor = it->second;

    // stream(true) drains torch_npu's pending task queue before the raw
    // launch, keeping ordering with preceding eager ops (see cannbot template).
    auto aclStream = c10_npu::getCurrentNPUStream().stream(true);

    lz_qk_norm_kernel(
        numBlocks, nullptr, aclStream,
        reinterpret_cast<uint8_t*>(q.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(k.mutable_data_ptr()),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(qw.const_data_ptr())),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(qb.const_data_ptr())),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(kw.const_data_ptr())),
        const_cast<uint8_t*>(reinterpret_cast<const uint8_t*>(kb.const_data_ptr())),
        reinterpret_cast<uint8_t*>(qOut.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(kOut.mutable_data_ptr()),
        reinterpret_cast<uint8_t*>(tilingTensor.mutable_data_ptr()));
    return {qOut, kOut};
}

}  // namespace lz_qknorm

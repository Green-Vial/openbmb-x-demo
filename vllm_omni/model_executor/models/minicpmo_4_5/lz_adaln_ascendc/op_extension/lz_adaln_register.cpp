// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <torch/library.h>
#include "lz_adaln_ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(lz_npu, m)
{
    m.def("fused_adaln_gate(Tensor x, Tensor shift, Tensor scale, Tensor gate) -> Tensor");
    m.def("fused_adaln_norm(Tensor x, Tensor shift, Tensor scale) -> Tensor");
}

TORCH_LIBRARY_IMPL(lz_npu, PrivateUse1, m)
{
    m.impl("fused_adaln_gate", TORCH_FN(lz_adaln::fused_adaln_gate_torch));
    m.impl("fused_adaln_norm", TORCH_FN(lz_adaln::fused_adaln_norm_torch));
}

at::Tensor fused_adaln_gate_meta(const at::Tensor& x, const at::Tensor& shift,
                                 const at::Tensor& scale, const at::Tensor& gate)
{
    return at::empty_like(x);
}

at::Tensor fused_adaln_norm_meta(const at::Tensor& x, const at::Tensor& shift, const at::Tensor& scale)
{
    return at::empty_like(x);
}

TORCH_LIBRARY_IMPL(lz_npu, Meta, m)
{
    m.impl("fused_adaln_gate", &fused_adaln_gate_meta);
    m.impl("fused_adaln_norm", &fused_adaln_norm_meta);
}

}  // namespace

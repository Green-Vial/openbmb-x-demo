// SPDX-License-Identifier: Apache-2.0
#include <torch/extension.h>
#include <tuple>
#include "lz_qk_norm_host.h"
#include <torch/library.h>
#include "lz_adaln_ops.h"

namespace {

TORCH_LIBRARY_FRAGMENT(lz_npu, m)
{
    m.def("fused_adaln_gate(Tensor x, Tensor shift, Tensor scale, Tensor gate) -> Tensor");
    m.def("fused_adaln_norm(Tensor x, Tensor shift, Tensor scale) -> Tensor");
    m.def("fused_qk_norm(Tensor q, Tensor k, Tensor qw, Tensor qb, Tensor kw, Tensor kb, float eps)"
          " -> (Tensor, Tensor)");
}

TORCH_LIBRARY_IMPL(lz_npu, PrivateUse1, m)
{
    m.impl("fused_adaln_gate", TORCH_FN(lz_adaln::fused_adaln_gate_torch));
    m.impl("fused_adaln_norm", TORCH_FN(lz_adaln::fused_adaln_norm_torch));
    m.impl("fused_qk_norm", TORCH_FN(lz_qknorm::fused_qk_norm_torch));
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

std::tuple<at::Tensor, at::Tensor> fused_qk_norm_meta(const at::Tensor& q, const at::Tensor& k,
                                           const at::Tensor& qw, const at::Tensor& qb,
                                           const at::Tensor& kw, const at::Tensor& kb, double eps)
{
    return {at::empty_like(q), at::empty_like(k)};
}

TORCH_LIBRARY_IMPL(lz_npu, Meta, m)
{
    m.impl("fused_adaln_gate", &fused_adaln_gate_meta);
    m.impl("fused_adaln_norm", &fused_adaln_norm_meta);
    m.impl("fused_qk_norm", &fused_qk_norm_meta);
}

}  // namespace

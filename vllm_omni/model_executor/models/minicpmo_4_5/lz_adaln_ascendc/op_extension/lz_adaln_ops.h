// SPDX-License-Identifier: Apache-2.0
#ifndef LZ_ADALN_OPS_H
#define LZ_ADALN_OPS_H

#include <torch/extension.h>

namespace lz_adaln {

// out = x + gate * (LayerNorm(x, eps=1e-6) * (1 + scale) + shift)
at::Tensor fused_adaln_gate_torch(const at::Tensor& x, const at::Tensor& shift,
                                  const at::Tensor& scale, const at::Tensor& gate);

// out = LayerNorm(x, eps=1e-6) * (1 + scale) + shift
at::Tensor fused_adaln_norm_torch(const at::Tensor& x, const at::Tensor& shift,
                                  const at::Tensor& scale);

}  // namespace lz_adaln

#endif  // LZ_ADALN_OPS_H

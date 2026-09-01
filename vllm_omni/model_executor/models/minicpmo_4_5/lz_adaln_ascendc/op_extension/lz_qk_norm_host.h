// SPDX-License-Identifier: Apache-2.0
#ifndef LZ_QK_NORM_HOST_H
#define LZ_QK_NORM_HOST_H

#include <tuple>
#include <torch/extension.h>

namespace lz_qknorm {

std::tuple<at::Tensor, at::Tensor> fused_qk_norm_torch(const at::Tensor& q, const at::Tensor& k,
                                            const at::Tensor& qw, const at::Tensor& qb,
                                            const at::Tensor& kw, const at::Tensor& kb, double eps);

}  // namespace lz_qknorm

#endif  // LZ_QK_NORM_HOST_H

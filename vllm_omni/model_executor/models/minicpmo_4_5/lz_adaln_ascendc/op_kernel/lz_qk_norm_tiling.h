// SPDX-License-Identifier: Apache-2.0
/* P28b: fused QK-norm tiling layout shared by the AscendC kernel and the host
 * launcher. Pure C/C++ - no __aicore__/__gm__ keywords here.
 *
 * Work unit = ONE (b, t) position = nh consecutive heads = nh * headDim
 * contiguous elements (guaranteed by host: H stride == headDim). Batching 8
 * heads per CopyIn/sync round cuts the per-row event-sync overhead by nh,
 * which is what makes a 64-wide row profitable at all. */
#ifndef LZ_QKNORM_TILING_H
#define LZ_QKNORM_TILING_H

#include <cstdint>

struct LzQkNormTilingData {
    uint32_t numBlocks;    // cores actually launched
    uint32_t batches;      // B * T work units per tensor (q and k identical)
    uint32_t unitsPerCore; // ceil(batches / numBlocks)  (unit = q AND k of one (b,t))
    uint32_t headDim;      // 64 for the CosyVoice2 DiT
    uint32_t nh;           // heads per work unit (8)
    uint32_t dimT;         // T
    uint32_t qStrideB;     // element strides of the transposed (B, nh, T, hd) views
    uint32_t qStrideT;
    uint32_t kStrideB;
    uint32_t kStrideT;
    float eps;             // 1e-5 (nn.LayerNorm default; affine weight+bias)
    uint32_t reserved;
};

#endif  // LZ_QKNORM_TILING_H

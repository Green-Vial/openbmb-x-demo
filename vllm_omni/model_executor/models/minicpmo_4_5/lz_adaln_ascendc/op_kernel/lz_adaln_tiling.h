// SPDX-License-Identifier: Apache-2.0
/* P28: FusedAdaLN tiling layout shared by the AscendC kernel and the host
 * launcher. Pure C/C++ - no __aicore__/__gm__ keywords here. */
#ifndef LZ_ADALN_TILING_H
#define LZ_ADALN_TILING_H

#include <cstdint>

// The kernel processes one 512-wide fp32 row per iteration; the CosyVoice2
// DiT hidden size is fixed at 512 and the host validates it.
constexpr uint32_t LZ_ADALN_ROW_LEN = 512;
constexpr uint32_t LZ_ADALN_ROW_BYTES = LZ_ADALN_ROW_LEN * sizeof(float);

// Computation mode:
//   1 - out = x + gate * (LayerNorm(x) * (1 + scale) + shift)   (gate-residual)
//   0 - out = LayerNorm(x) * (1 + scale) + shift                (LN+modulate)
constexpr uint32_t LZ_ADALN_MODE_GATE_RESIDUAL = 1;
constexpr uint32_t LZ_ADALN_MODE_MODULATE = 0;

struct AdaLNTilingData {
    uint32_t numBlocks;    // cores actually launched
    uint32_t totalRows;    // B * T
    uint32_t rowsPerCore;  // ceil(totalRows / numBlocks)
    uint32_t chunkT;       // T; param row for x row r is r / chunkT
    uint32_t paramStride;  // element stride between consecutive param rows
    uint32_t mode;         // LZ_ADALN_MODE_*
    uint32_t reserved[2];
};

#endif  // LZ_ADALN_TILING_H

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P28b: FusedQkNorm — correctness and parity for the fused QK-norm op.

The CosyVoice2 DiT attention applies two affine LayerNorm(head_dim) calls
(q_norm/k_norm, eps=1e-5) per block; the AscendC kernel fuses both into one
launch and consumes the transposed to_heads views in place (explicit B/T
strides, contiguous head rows). Host 计时对 eager 路径是负收益（device 串行
主导），图 replay 内 4.16x——默认关闭（OMNI_LZ_QKNORM_FUSED=1 opt-in）。
"""

from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.minicpmo_4_5 import batched_token2wav as b2w
from vllm_omni.model_executor.models.minicpmo_4_5.lz_qknorm_fused import (
    fused_qk_norm,
    is_available,
)

pytestmark = [pytest.mark.core_model, pytest.mark.npu]

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()), reason="requires Ascend NPU"
)

HIDDEN = 512


def _make_inputs(b, t, nh=8, hd=64):
    """Real to_heads layout: transposed views with contiguous head rows."""
    device = torch.device("npu")
    q = torch.randn(b, t, nh * hd, device=device, dtype=torch.float32).reshape(b, t, nh, hd).transpose(1, 2)
    k = torch.randn(b, t, nh * hd, device=device, dtype=torch.float32).reshape(b, t, nh, hd).transpose(1, 2)
    qw = torch.randn(hd, device=device, dtype=torch.float32)
    qb = torch.randn(hd, device=device, dtype=torch.float32)
    kw = torch.randn(hd, device=device, dtype=torch.float32)
    kb = torch.randn(hd, device=device, dtype=torch.float32)
    return q, k, qw, qb, kw, kb


def _eager(q, k, qw, qb, kw, kb, eps):
    return F.layer_norm(q, (q.shape[-1],), qw, qb, eps), F.layer_norm(k, (k.shape[-1],), kw, kb, eps)


@requires_npu
@pytest.mark.parametrize("t", [1, 44, 125])
def test_fused_qk_norm_correctness(t: int):
    torch.manual_seed(0)
    assert is_available(), "kernel must load/build on this NPU environment"
    q, k, qw, qb, kw, kb = _make_inputs(2, t)
    got = fused_qk_norm(q, k, qw, qb, kw, kb, 1e-5)
    ref = _eager(q, k, qw, qb, kw, kb, 1e-5)
    diff = max(float((g - r).abs().max()) for g, r in zip(got, ref))
    print(f"\n[fused_qk_norm] B=2 T={t} max abs diff = {diff:.3e}")
    assert diff < 1e-4, f"max abs diff {diff:.3e} >= 1e-4"


def _build_attention():
    from cosyvoice2.flow.decoder_dit import Attention

    torch.manual_seed(7)
    attn = Attention(HIDDEN, num_heads=8, head_dim=64, qkv_bias=True, qk_norm=True)
    return attn.to(torch.device("npu"), torch.float32).eval()


@requires_npu
def test_attention_forward_chunk_parity():
    """Original forward_chunk vs the fused drop-in, incl. a cache roundtrip."""
    from cosyvoice2.flow.decoder_dit import Attention

    original = Attention.forward_chunk
    attn = _build_attention()
    torch.manual_seed(3)
    x = torch.randn(2, 44, HIDDEN, device="npu", dtype=torch.float32)
    att_cache = torch.randn(2, 8, 16, 128, device="npu", dtype=torch.float32)

    with torch.no_grad():
        ref = original(attn, x, att_cache.clone(), None)
        b2w._apply_lz_qk_norm_fused_patch()
        got = original(attn, x, att_cache.clone(), None)

    diffs = [float((g - r).abs().max()) for g, r in zip(got, ref)]
    print(f"\n[attention parity] out/cache diffs = {['%.2e' % d for d in diffs]}")
    assert all(d < 1e-4 for d in diffs)

    # second chunk consumes the step-1 caches
    x2 = torch.randn(2, 44, HIDDEN, device="npu", dtype=torch.float32)
    with torch.no_grad():
        ref2 = original(attn, x2, ref[1], None)
        got2 = original(attn, x2, got[1], None)
    d2 = float((got2[0] - ref2[0]).abs().max())
    print(f"[attention parity] chunk2 (cache roundtrip) diff = {d2:.3e}")
    assert d2 < 1e-4


@requires_npu
def test_unsupported_layout_soft_fallback():
    """Contiguous (non-to_heads) inputs fall back to eager for that call only."""
    torch.manual_seed(0)
    assert is_available()
    device = torch.device("npu")
    q = torch.randn(2, 8, 16, 64, device=device, dtype=torch.float32)  # contiguous: H stride != hd
    k = torch.randn_like(q)
    qw, qb = torch.randn(64, device=device, dtype=torch.float32), torch.randn(64, device=device, dtype=torch.float32)
    kw, kb = torch.randn(64, device=device, dtype=torch.float32), torch.randn(64, device=device, dtype=torch.float32)
    got = fused_qk_norm(q, k, qw, qb, kw, kb, 1e-5)
    ref = _eager(q, k, qw, qb, kw, kb, 1e-5)
    diff = max(float((g - r).abs().max()) for g, r in zip(got, ref))
    assert diff == 0.0, "unsupported layout must take the bit-identical eager path"
    assert is_available(), "a layout fallback must not permanently disable the kernel"

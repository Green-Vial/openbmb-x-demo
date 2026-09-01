# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P28: FusedAdaLN AscendC kernel — correctness and microbenchmark.

Correctness compares the fused op against the eager reference chain
(F.layer_norm -> modulate -> gate residual) on npu:0 for B=2 and
T in {1, 128, 200}. fp32 layernorm reductions are not bit-exact across
reduction orders; the acceptance threshold is max abs diff < 1e-5 (the
measured values are printed for the record).

The microbenchmark times the eager 5-op chain vs the fused kernel
(100 iterations, torch.npu.synchronize-bracketed wall clock) for T in
{1, 128}; numbers are recorded, not asserted.
"""

from __future__ import annotations

import time

import pytest
import torch
import torch.nn.functional as F

from vllm_omni.model_executor.models.minicpmo_4_5 import batched_token2wav as b2w
from vllm_omni.model_executor.models.minicpmo_4_5.lz_adaln_fused import (
    fused_adaln_gate,
    fused_adaln_norm,
    is_available,
)

pytestmark = [pytest.mark.core_model, pytest.mark.npu]

requires_npu = pytest.mark.skipif(
    not (hasattr(torch, "npu") and torch.npu.is_available()), reason="requires Ascend NPU"
)

HIDDEN = 512
BATCH = 2
BENCH_ITERS = 100
BENCH_WARMUP = 20


def _make_inputs(batch: int, t: int, chunked_params: bool):
    """Random fp32 inputs; params mirror adaLN_modulation(c).chunk(9) views."""
    device = torch.device("npu")
    x = torch.randn(batch, t, HIDDEN, device=device, dtype=torch.float32)
    if chunked_params:
        cat = torch.randn(batch, 1, 9 * HIDDEN, device=device, dtype=torch.float32)
        shift, scale, gate = cat.chunk(9, dim=-1)[0:3]
        shift = shift + 0.0  # keep autograd-free explicit tensors
    else:
        shift = torch.randn(batch, 1, HIDDEN, device=device, dtype=torch.float32)
        scale = torch.randn(batch, 1, HIDDEN, device=device, dtype=torch.float32)
        gate = torch.randn(batch, 1, HIDDEN, device=device, dtype=torch.float32)
    return x, shift, scale, gate


def _eager_gate(x, shift, scale, gate):
    h = F.layer_norm(x, (HIDDEN,), None, None, 1e-6)
    return x + gate * (h * (1 + scale) + shift)


def _eager_norm(x, shift, scale):
    h = F.layer_norm(x, (HIDDEN,), None, None, 1e-6)
    return h * (1 + scale) + shift


@requires_npu
@pytest.mark.parametrize("t", [1, 128, 200])
@pytest.mark.parametrize("chunked_params", [True, False])
def test_fused_adaln_gate_correctness(t: int, chunked_params: bool):
    torch.manual_seed(0)
    assert is_available(), "kernel must load/build on this NPU environment"
    x, shift, scale, gate = _make_inputs(BATCH, t, chunked_params)
    got = fused_adaln_gate(x, shift, scale, gate)
    ref = _eager_gate(x, shift, scale, gate)
    diff = float((got - ref).abs().max())
    print(f"\n[fused_adaln_gate] B={BATCH} T={t} chunked={chunked_params} max abs diff = {diff:.3e}")
    assert diff < 1e-5, f"max abs diff {diff:.3e} >= 1e-5"


@requires_npu
@pytest.mark.parametrize("t", [1, 128, 200])
def test_fused_adaln_norm_correctness(t: int):
    torch.manual_seed(0)
    assert is_available(), "kernel must load/build on this NPU environment"
    x, shift, scale, _ = _make_inputs(BATCH, t, chunked_params=True)
    got = fused_adaln_norm(x, shift, scale)
    ref = _eager_norm(x, shift, scale)
    diff = float((got - ref).abs().max())
    print(f"\n[fused_adaln_norm] B={BATCH} T={t} max abs diff = {diff:.3e}")
    assert diff < 1e-5, f"max abs diff {diff:.3e} >= 1e-5"


def _build_dit_block():
    from cosyvoice2.flow.decoder_dit import DiTBlock

    torch.manual_seed(7)
    block = DiTBlock(HIDDEN, num_heads=8, head_dim=64)
    return block.to(torch.device("npu"), torch.float32).eval()


def _block_io(block, x, c, cnn_cache=None, att_cache=None):
    with torch.no_grad():
        return block.forward_chunk(x, c, cnn_cache, att_cache, None)


@requires_npu
def test_dit_block_forward_chunk_parity():
    """Original forward_chunk vs the fused drop-in, two steps incl. caches."""
    block = _build_dit_block()
    torch.manual_seed(3)
    x = torch.randn(4, 37, HIDDEN, device="npu", dtype=torch.float32)
    c = torch.randn(4, 1, HIDDEN, device="npu", dtype=torch.float32)

    x1_ref, cnn_ref, att_ref = _block_io(block, x, c)
    x1_fus, cnn_fus, att_fus = b2w._lz_fused_dit_block_forward_chunk(block, x, c)

    assert cnn_fus.shape == cnn_ref.shape, "cnn cache shape must match the original"
    assert att_fus.shape == att_ref.shape, "att cache shape must match the original"

    step1 = float((x1_fus - x1_ref).abs().max())
    print(f"\n[parity] step1 (no cache) max abs diff = {step1:.3e}")
    assert step1 < 1e-4

    # second chunk consumes the step-1 caches (same inputs for both impls)
    x2_ref, cnn2_ref, att2_ref = _block_io(block, x1_ref, c, cnn_ref, att_ref)
    x2_fus, cnn2_fus, att2_fus = b2w._lz_fused_dit_block_forward_chunk(
        block, x1_ref, c, cnn_fus, att_fus
    )
    assert cnn2_fus.shape == cnn2_ref.shape
    assert att2_fus.shape == att2_ref.shape
    step2 = float((x2_fus - x2_ref).abs().max())
    print(f"[parity] step2 (with caches) max abs diff = {step2:.3e}")
    assert step2 < 1e-4

    cache_diff = max(
        float((cnn2_fus - cnn2_ref).abs().max()),
        float((att2_fus - att2_ref).abs().max()),
    )
    print(f"[parity] cache max abs diff = {cache_diff:.3e}")
    assert cache_diff < 1e-4


@requires_npu
def test_omni_lz_adaln_fused_patch_gating():
    """The class patch must be applied only when _apply_... is called."""
    from cosyvoice2.flow.decoder_dit import DiTBlock

    block = _build_dit_block()
    torch.manual_seed(5)
    x = torch.randn(4, 9, HIDDEN, device="npu", dtype=torch.float32)
    c = torch.randn(4, 1, HIDDEN, device="npu", dtype=torch.float32)
    x_ref, cnn_ref, att_ref = _block_io(block, x, c)  # original implementation

    assert b2w._apply_lz_adaln_fused_patch() is True
    assert b2w._LZ_ADALN_PATCH_APPLIED is True

    x_fus, cnn_fus, att_fus = _block_io(block, x, c)  # now the patched method
    diff = float((x_fus - x_ref).abs().max())
    print(f"\n[patched-class parity] max abs diff = {diff:.3e}")
    assert diff < 1e-4
    assert cnn_fus.shape == cnn_ref.shape and att_fus.shape == att_ref.shape


@requires_npu
@pytest.mark.benchmark
def test_microbenchmark_eager_vs_fused():
    """Eager 5-op chain vs fused kernel, 100 iters, sync-bracketed timing."""
    device = torch.device("npu")

    def bench(fn, *args) -> float:
        for _ in range(BENCH_WARMUP):
            fn(*args)
        torch.npu.synchronize()
        start = time.perf_counter()
        for _ in range(BENCH_ITERS):
            fn(*args)
        torch.npu.synchronize()
        return (time.perf_counter() - start) / BENCH_ITERS * 1e6  # us per call

    rows = []
    for t in [1, 128]:
        x, shift, scale, gate = _make_inputs(BATCH, t, chunked_params=True)
        rows.append((t, "gate", "eager", bench(_eager_gate, x, shift, scale, gate)))
        rows.append((t, "gate", "fused", bench(fused_adaln_gate, x, shift, scale, gate)))
        rows.append((t, "norm", "eager", bench(_eager_norm, x, shift, scale)))
        rows.append((t, "norm", "fused", bench(fused_adaln_norm, x, shift, scale)))

    print("\n[microbenchmark] B=2, fp32, npu:0, 100 iters mean (us/call)")
    print(f"{'T':>4} {'op':>5} {'impl':>6} {'us/call':>10}")
    per_t: dict[int, dict[tuple[str, str], float]] = {}
    for t, op, impl, us in rows:
        print(f"{t:>4} {op:>5} {impl:>6} {us:>10.2f}")
        per_t.setdefault(t, {})[(op, impl)] = us
    for t, cell in per_t.items():
        for op in ("gate", "norm"):
            eager_us = cell[(op, "eager")]
            fused_us = cell[(op, "fused")]
            print(
                f"T={t:>3} {op}: eager {eager_us:.2f}us vs fused {fused_us:.2f}us "
                f"-> {eager_us / fused_us:.2f}x"
            )


@requires_npu
def test_fused_adaln_npu_graph_capture_replay():
    """The patched forward_chunk must be NPUGraph-capturable and replay-safe.

    Mirrors the P15 production pattern: eager warmup on static buffers first
    (fills the tiling cache so capture contains no H2D memcpy), capture,
    then replay with fresh inputs and compare against the eager reference.
    """
    from cosyvoice2.flow.decoder_dit import DiTBlock

    original = DiTBlock.forward_chunk
    b2w._apply_lz_adaln_fused_patch()
    try:
        block = _build_dit_block()
        torch.manual_seed(11)
        batch, chunk_t, past_t = 2, 64, 16
        x = torch.randn(batch, chunk_t, HIDDEN, device="npu", dtype=torch.float32)
        c = torch.randn(batch, 1, HIDDEN, device="npu", dtype=torch.float32)
        cnn_cache = torch.randn(batch, 1024, 2, device="npu", dtype=torch.float32)
        att_cache = torch.randn(batch, 8, past_t, 128, device="npu", dtype=torch.float32)

        sx, sc = torch.zeros_like(x), torch.zeros_like(c)
        s_cnn, s_att = torch.zeros_like(cnn_cache), torch.zeros_like(att_cache)
        out = [None, None, None]

        def _run():
            with torch.no_grad():
                out[0], out[1], out[2] = block.forward_chunk(sx, sc, s_cnn, s_att)

        sx.copy_(x); sc.copy_(c); s_cnn.copy_(cnn_cache); s_att.copy_(att_cache)
        _run()  # eager warmup: allocator settles, tiling cache fills
        torch.npu.synchronize()

        graph = torch.npu.NPUGraph()
        with torch.npu.graph(graph):
            _run()
        torch.npu.synchronize()

        # replay with fresh inputs
        torch.manual_seed(12)
        x2 = torch.randn_like(x)
        c2 = torch.randn_like(c)
        cnn2 = torch.randn_like(cnn_cache)
        att2 = torch.randn_like(att_cache)
        sx.copy_(x2); sc.copy_(c2); s_cnn.copy_(cnn2); s_att.copy_(att2)
        graph.replay()
        torch.npu.synchronize()
        got = tuple(t.clone() for t in out)

        DiTBlock.forward_chunk = original  # eager reference
        ref = _block_io(block, x2, c2, cnn2, att2)
        diff = max(float((g - r).abs().max()) for g, r in zip(got, ref))
        print(f"\n[graph replay] max abs diff vs eager = {diff:.3e}")
        assert diff < 1e-4
    finally:
        DiTBlock.forward_chunk = original

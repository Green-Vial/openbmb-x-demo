"""P27 unit tests: sampler-graph decision logic and bitwise math (CPU only).

No NPU is required here: the NPUGraph capture/replay itself is exercised on
the bench (see the P27 notes in KuaaMU_深入学习答疑.md). These tests pin down
the parts that must hold for the graph path to be bit-identical to eager:

* the graph bodies are the exact op sequences the eager path issues;
* masking EOS before vs after the top-k/top-p warp is equivalent, so the
  tail graph may own the min_tokens mask;
* signature keying, the combined head+tail graph budget and the permanent
  eager fallback behave as designed.
"""

import torch

from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_tts import (
    _LZ_SAMPLER_MAX_GRAPHS,
    MiniCPMO45OmniTTSForConditionalGeneration,
    _apply_top_k_top_p,
    _lz_gumbel_tail,
    _lz_head_code_logits,
)

_VOCAB = 2000
_EOS_ID = _VOCAB - 1


def _make_logits(seed: int, hot_eos: bool = False) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(1, _VOCAB, generator=g)
    if hot_eos:
        logits[..., _EOS_ID] = 5.0  # force EOS into the top-k candidate set
    return logits


# ---- graph bodies: bitwise identical to the eager reference expressions ----


def test_head_body_bitwise_vs_eager_expression():
    head = torch.nn.Linear(64, _VOCAB, bias=False)
    hidden = torch.randn(1, 64)
    temperature = 0.8
    via_body = _lz_head_code_logits(head, hidden, temperature)
    via_eager = head(hidden).float() / temperature
    assert torch.equal(via_body, via_eager)
    assert via_body.dtype == torch.float32


def test_gumbel_tail_no_mask_bitwise_vs_eager_expression():
    logits = _make_logits(1)
    g = torch.Generator().manual_seed(2)
    q = torch.empty(1, _VOCAB).exponential_(generator=g)
    assert torch.equal(_lz_gumbel_tail(logits, q), (logits - torch.log(q.clamp_min(1e-9))).argmax(dim=-1))


def test_gumbel_tail_eos_mask_bitwise_vs_eager_assign():
    logits = _make_logits(3, hot_eos=True)
    g = torch.Generator().manual_seed(4)
    q = torch.empty(1, _VOCAB).exponential_(generator=g)
    eos_mask = torch.zeros(1, _VOCAB, dtype=torch.bool)
    eos_mask[..., _EOS_ID] = True
    assert torch.equal(_lz_gumbel_tail(logits, q, eos_mask), (logits - torch.log(q.clamp_min(1e-9))).argmax(dim=-1))


# ---- semantic precondition: mask-after-warp == mask-before-warp ----


def test_eos_mask_order_equivalence_under_topk_topp_warp():
    """The tail graph masks EOS after the warp; eager masks before it.

    Equivalent because -inf carries zero softmax mass (cumsum unchanged) and
    never survives the top-k threshold — checked bitwise here over seeds that
    put EOS both inside and outside the candidate set.
    """
    for seed, hot_eos in [(5, False), (6, True), (7, True), (8, False)]:
        logits = _make_logits(seed, hot_eos=hot_eos)
        g = torch.Generator().manual_seed(seed + 100)
        q = torch.empty(1, _VOCAB).exponential_(generator=g)
        # eager order (mask before warp)
        eager_logits = logits.clone()
        eager_logits[..., _EOS_ID] = float("-inf")
        eager_logits = _apply_top_k_top_p(eager_logits, top_k=25, top_p=0.85, min_tokens_to_keep=3)
        eager_id = (eager_logits - torch.log(q.clamp_min(1e-9))).argmax(dim=-1)
        # graph order (mask inside the tail graph, after the warp)
        warped = _apply_top_k_top_p(logits, top_k=25, top_p=0.85, min_tokens_to_keep=3)
        eos_mask = torch.zeros(1, _VOCAB, dtype=torch.bool)
        eos_mask[..., _EOS_ID] = True
        graph_id = _lz_gumbel_tail(warped, q, eos_mask)
        assert torch.equal(eager_id, graph_id), f"seed={seed} hot_eos={hot_eos}"


# ---- signature / budget / permanent-fallback logic (fake capture) ----


def _bare_model() -> MiniCPMO45OmniTTSForConditionalGeneration:
    """Instance without __init__ (no vllm_config needed) + graph state attrs."""
    model = object.__new__(MiniCPMO45OmniTTSForConditionalGeneration)
    model._lz_sampler_graph_dead = False
    model._lz_sampler_head_graphs = {}
    model._lz_sampler_tail_graphs = {}
    model._lz_gumbel_sampling = True
    model._num_audio_tokens = _VOCAB
    return model


class _FakeBucket:
    pass


def test_tail_graph_captures_once_per_signature(monkeypatch):
    model = _bare_model()
    calls = []

    def fake_capture(logits, mask_eos):
        calls.append(bool(mask_eos))
        return _FakeBucket()

    monkeypatch.setattr(model, "_lz_capture_tail_graph", fake_capture)
    logits = _make_logits(9)
    first = model._lz_sampler_tail_graph(logits, True)
    second = model._lz_sampler_tail_graph(logits.clone(), True)
    assert isinstance(first, _FakeBucket) and first is second
    assert calls == [True]  # one capture, cached by (vocab, dtype, mask_eos)
    third = model._lz_sampler_tail_graph(logits, False)
    assert isinstance(third, _FakeBucket) and calls == [True, False]


def test_head_graph_gated_on_gumbel_and_dead_flag(monkeypatch):
    model = _bare_model()
    model._lz_gumbel_sampling = False
    assert model._lz_sampler_head_graph(torch.randn(1, 64)) is None
    model._lz_gumbel_sampling = True
    model._lz_sampler_graph_dead = True
    assert model._lz_sampler_head_graph(torch.randn(1, 64)) is None
    # tail lookups honor the dead flag too (no capture attempted)
    assert model._lz_sampler_tail_graph(_make_logits(10), True) is None


def test_budget_exhaustion_permanently_disables(monkeypatch):
    model = _bare_model()
    for mask_eos in (True, False):
        key = (_VOCAB, "torch.float32", mask_eos)
        model._lz_sampler_tail_graphs[key] = _FakeBucket()
    head_key = (64, _VOCAB, "torch.float32")
    model._lz_sampler_head_graphs[head_key] = _FakeBucket()
    # a second (unused) head signature fills the budget to _LZ_SAMPLER_MAX_GRAPHS
    model._lz_sampler_head_graphs[(32, _VOCAB, "torch.float32")] = _FakeBucket()
    assert len(model._lz_sampler_head_graphs) + len(model._lz_sampler_tail_graphs) == _LZ_SAMPLER_MAX_GRAPHS

    def fail_capture(logits, mask_eos):  # must never be reached
        raise AssertionError("capture attempted past the budget")

    monkeypatch.setattr(model, "_lz_capture_tail_graph", fail_capture)
    monkeypatch.setattr(model, "_lz_capture_head_graph", fail_capture)
    # a NEW signature past the budget: None + permanent dead flag
    assert model._lz_sampler_tail_graph(torch.randn(1, _VOCAB + 1), True) is None
    assert model._lz_sampler_graph_dead is True
    # even previously-cached signatures stop replaying from now on
    assert model._lz_sampler_head_graph(torch.randn(1, 64)) is None
    assert model._lz_sampler_tail_graph(torch.randn(1, _VOCAB + 1), False) is None


def test_head_graph_capture_and_cache(monkeypatch):
    model = _bare_model()
    calls = []

    def fake_capture(hidden):
        calls.append(tuple(hidden.shape))
        return _FakeBucket()

    monkeypatch.setattr(model, "_lz_capture_head_graph", fake_capture)
    first = model._lz_sampler_head_graph(torch.randn(1, 64))
    second = model._lz_sampler_head_graph(torch.randn(1, 64))
    assert isinstance(first, _FakeBucket) and first is second
    assert calls == [(1, 64)]


# ---- _lz_gumbel_row: P23 noise block handling (verbatim extraction) ----


def _model_with_request_state(model: MiniCPMO45OmniTTSForConditionalGeneration) -> None:
    model._request_audio_states = {"req": {}}
    model._request_generators = {}
    model._codec_seed = 42


def test_gumbel_row_returns_none_without_state():
    model = _bare_model()
    model._request_audio_states = {}
    model._request_generators = {}
    model._codec_seed = 42
    assert model._lz_gumbel_row("req", _make_logits(12), 0) is None


def test_gumbel_row_block_alloc_and_row_selection():
    model = _bare_model()
    _model_with_request_state(model)
    logits = _make_logits(13)
    row0 = model._lz_gumbel_row("req", logits, 0)
    state = model._request_audio_states["req"]
    noise = state["lz_gumbel"]
    assert noise.shape == (64, _VOCAB)
    assert torch.equal(row0, noise[0:1])
    # mid-block steps reuse the same block without redrawing
    row5 = model._lz_gumbel_row("req", logits, 5)
    assert model._request_audio_states["req"]["lz_gumbel"] is noise
    assert torch.equal(row5, noise[5:6])
    # wrapping past the block end redraws the whole block
    g = model._request_generator("req", logits.device)
    first_draw = noise[1:2].clone()
    row64 = model._lz_gumbel_row("req", logits, 64)
    assert torch.equal(row64, model._request_audio_states["req"]["lz_gumbel"][0:1])
    assert not torch.equal(first_draw, model._request_audio_states["req"]["lz_gumbel"][1:2])
    assert g is model._request_generator("req", logits.device)

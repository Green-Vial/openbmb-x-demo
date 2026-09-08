# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from:
# https://huggingface.co/openbmb/MiniCPM-o-4_5/blob/main/modeling_minicpmo.py
"""MiniCPM-o 4.5 native autoregressive Talker.

Pipeline:
  1. Receive thinker hidden_states + full token IDs via additional_information
  2. Extract tts_bos..tts_eos region
  3. Build condition: emb_text(tokens) + projector_semantic(hidden) (hidden_text_merge)
  4. Continuously generate request-aligned discrete audio-code deltas
"""

import os
from collections.abc import Iterable
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import LlamaConfig
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.models.interfaces import SupportsPP
from vllm.model_executor.models.llama import LlamaModel
from vllm.model_executor.models.utils import maybe_prefix
from vllm.v1.sample.sampler import Sampler

from vllm_omni.experimental.fullduplex.engine.intermediate import get_tts_handoff
from vllm_omni.model_executor.models.output_templates import OmniOutput
from vllm_omni.platforms import current_omni_platform

logger = init_logger(__name__)

_REPETITION_WINDOW = 16
# P23: rows of pre-drawn Exp(1) noise per request block (see _sample_audio_code).
_LZ_GUMBEL_BLOCK = 64
# P27: budget for captured sampler graphs (head + tail buckets summed over
# all signatures); a new signature past the budget permanently reverts the
# sampler to eager (see _lz_sampler_head_graph / _lz_sampler_tail_graph).
_LZ_SAMPLER_MAX_GRAPHS = 4
_LZ_SAMPLER_MISSING = object()
_MIN_AUDIO_TOKENS = 128
_MAX_AUDIO_TOKENS = 2048
_AUDIO_TOKENS_PER_TEXT_TOKEN = 10
# Flat margin added on top of the length estimate: prosody, leading silence
# and the unvoiced tail are not proportional to text length, so the pure
# ratio underestimates them for short prompts.
_AUDIO_TOKENS_TEXT_OVERHEAD = 48
# Codec-token sampling happens inside the model; vLLM sampling parameters
# only choose the Talker's binary continue/stop row.
_CODEC_SEED = 42
_CODEC_TEMPERATURE = 0.8
_CODEC_TOP_K = 25
_CODEC_TOP_P = 0.85
_CODEC_REPETITION_PENALTY = 1.05
_CODEC_MIN_TOKENS = 50
_DUPLEX_CODEC_TOKENS_PER_CHUNK = 26


def _max_audio_tokens(condition_tokens: int) -> int:
    """Bound codec generation with the checkpoint's native length heuristic.

    ``text_tokens * 10 + 48`` mirrors the native generation budget: the ratio
    covers the proportional body, the flat +48 covers prosody/silence tails.
    The 128 floor keeps short responses running past the 50-step EOS mask
    (below it, EOS is still ineligible when the cap hits); the 2048 ceiling
    matches the native default and stays within the Talker's 4096-position
    context.
    """
    return max(
        _MIN_AUDIO_TOKENS,
        min(
            _MAX_AUDIO_TOKENS,
            condition_tokens * _AUDIO_TOKENS_PER_TEXT_TOKEN + _AUDIO_TOKENS_TEXT_OVERHEAD,
        ),
    )


def _restore_weight_norm_weight(weight_g: torch.Tensor, weight_v: torch.Tensor) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    return torch._weight_norm(weight_v, weight_g, dim=0)


_PENALTY_BASE_CACHE: dict[tuple[str, torch.dtype], torch.Tensor] = {}


def _penalty_base(penalty: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Device-resident penalty scalar, cached per (device, dtype).

    ``torch.as_tensor(penalty, device=...)`` issues a host->device copy on
    every decode step; the cached scalar is identical and copy-free.
    """
    key = (str(device), dtype)
    cached = _PENALTY_BASE_CACHE.get(key)
    if cached is None:
        cached = torch.tensor(penalty, device=device, dtype=dtype)
        _PENALTY_BASE_CACHE[key] = cached
    return cached


def _apply_repetition_penalty(
    logits: torch.Tensor,
    history: torch.Tensor,
    *,
    penalty: float,
    window_size: int,
) -> torch.Tensor:
    """Match MiniCPMTTS' frequency-aware repetition penalty."""
    if penalty == 1.0 or history.numel() == 0:
        return logits
    recent = history.reshape(-1)[-window_size:].to(device=logits.device, dtype=torch.long)
    # torch.bincount must read max(recent) back to the host to size its
    # output, inserting a device synchronization into every decode step.
    # A scatter_add into a full-vocab buffer yields the identical integer
    # counts with a statically known shape, so the kernel-submission
    # pipeline stays fully asynchronous.
    frequencies = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.long)
    frequencies.scatter_add_(0, recent, torch.ones_like(recent))
    alpha = torch.pow(_penalty_base(penalty, logits.device, logits.dtype), frequencies.to(dtype=logits.dtype))
    return torch.where(logits < 0, logits * alpha, logits / alpha)


def _apply_top_k_top_p(
    logits: torch.Tensor,
    *,
    top_k: int | None,
    top_p: float | None,
    min_tokens_to_keep: int = 3,
) -> torch.Tensor:
    """Apply the same candidate floors as the upstream Transformers warpers."""
    filtered = logits.clone()
    vocab_size = filtered.shape[-1]
    # MiniCPM-o's gen_logits() appends TopPLogitsWarper before
    # TopKLogitsWarper. The order is observable for fixed-seed sampling.
    if top_p is not None and 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=False, dim=-1)
        cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        remove = cumulative_probs <= (1.0 - float(top_p))
        remove[..., -min_tokens_to_keep:] = False
        remove = remove.scatter(-1, sorted_indices, remove)
        filtered.masked_fill_(remove, float("-inf"))
    if top_k is not None and top_k > 0:
        keep = min(vocab_size, max(int(top_k), min_tokens_to_keep))
        threshold = torch.topk(filtered, keep, dim=-1).values[..., -1, None]
        filtered.masked_fill_(filtered < threshold, float("-inf"))
    return filtered


class _LZSamplerHeadBucket:
    """P27: static buffers + captured NPUGraph for the head-code linear.

    Graph body: ``head_code[0](h).float() / temperature`` — the deterministic
    ops between the talker hidden state and the raw codec logits. The linear
    weight is referenced by address and owned by the model, so it stays alive
    for the graph's lifetime; the bucket owns the static input/output
    buffers that ``copy_``/replay traffic flows through.
    """

    def __init__(self) -> None:
        self.graph: Any = None
        self.h_in: torch.Tensor | None = None
        self.logits_out: torch.Tensor | None = None


class _LZSamplerTailBucket:
    """P27: static buffers + captured NPUGraph for the Gumbel tail.

    Graph body: optional EOS mask (min_tokens state) → ``logits - log(q)`` →
    ``argmax``. Both inputs (warped logits, Exp(1) noise row) are pure data,
    copied into the static buffers before every replay; the codec id is read
    from the fixed output buffer and cloned by the caller before it can
    outlive the next replay.
    """

    def __init__(self) -> None:
        self.graph: Any = None
        self.logits_in: torch.Tensor | None = None
        self.q_in: torch.Tensor | None = None
        self.id_out: torch.Tensor | None = None


def _lz_head_code_logits(head: nn.Linear, hidden: torch.Tensor, temperature: float) -> torch.Tensor:
    """P27 head-graph body: codec logits from a [1, hidden] talker row.

    Shared by the capture path and the eager path so both issue the identical
    op sequence (linear → float → temperature scale) — the precondition for
    graph replay being bit-identical to eager.
    """
    return head(hidden).float() / temperature


def _lz_gumbel_tail(
    logits: torch.Tensor,
    q: torch.Tensor,
    eos_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """P27 tail-graph body: optional EOS mask, then ``logits - log(q)`` → argmax.

    Shared by the capture path and the eager fallback so both issue the
    identical op sequence. ``q`` is one Exp(1) noise row (P23); ``eos_mask``
    is a static [1, vocab] bool buffer (None = min_tokens already satisfied).
    """
    if eos_mask is not None:
        logits = logits.masked_fill(eos_mask, float("-inf"))
    return (logits - torch.log(q.clamp_min(1e-9))).argmax(dim=-1)


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible hidden-state projector used by MiniCPMTTS."""

    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class MiniCPMO45OmniTTSForConditionalGeneration(nn.Module, SupportsPP):
    """Runner-owned MiniCPM-o 4.5 Talker that emits codec tokens only."""

    requires_request_sample_eligibility = True

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        from vllm_omni.model_executor.models.minicpmo_4_5.minicpmo_4_5_omni_llm import MiniCPMOConfig

        config: MiniCPMOConfig = vllm_config.model_config.hf_config
        self.config = config
        self.vllm_config = vllm_config
        self._batch_stop_logits: torch.Tensor | None = None
        self._request_generators: dict[str, torch.Generator] = {}
        self._request_audio_states: dict[str, dict[str, Any]] = {}
        self._deferred_cleanup_ids: set[str] = set()
        # P12b: (device, dtype) -> (continue_row, stop_row) device templates.
        self._stop_row_templates: dict[tuple[torch.device, torch.dtype], tuple[torch.Tensor, torch.Tensor]] = {}
        # P27: NPUGraph replay of the deterministic sampler chain (head linear
        # + Gumbel tail). Rollback: OMNI_LZ_SAMPLER=0 — or any capture
        # failure — keeps the eager path, permanently and silently.
        # DEFAULT OFF (bench-verified): the per-replay fixed cost of two tiny
        # graphs exceeds the saved kernel launches on this chain (RTF
        # 0.31 -> 0.72 when on). Kept as an opt-in experiment behind
        # OMNI_LZ_SAMPLER=1.
        self._lz_sampler_graph_dead = os.environ.get("OMNI_LZ_SAMPLER", "0") not in {"1", "true", "yes", "on"}
        self._lz_sampler_head_graphs: dict[tuple[int, int, str], Any] = {}
        self._lz_sampler_tail_graphs: dict[tuple[int, str, bool], Any] = {}

        tts_config = getattr(config, "tts_config", None)
        if tts_config is None and getattr(config, "model_type", None) == "minicpmtts":
            tts_config = config
        if tts_config is not None:
            self._tts_config = tts_config
            self._tts_bos_id = getattr(tts_config, "audio_bos_token_id", 151687)
            self._text_eos_id = getattr(tts_config, "text_eos_token_id", 151692)
            self._num_audio_tokens = getattr(tts_config, "num_audio_tokens", 6562)
            self._hidden_size = getattr(tts_config, "hidden_size", 768)
            self._normalize = getattr(tts_config, "normalize_projected_hidden", True)
            self._codec_seed = int(getattr(tts_config, "seed", _CODEC_SEED))
            self._codec_temperature = float(getattr(tts_config, "temperature", _CODEC_TEMPERATURE))
            self._codec_top_k = int(getattr(tts_config, "top_k", _CODEC_TOP_K))
            self._codec_top_p = float(getattr(tts_config, "top_p", _CODEC_TOP_P))
            self._codec_repetition_penalty = float(getattr(tts_config, "repetition_penalty", _CODEC_REPETITION_PENALTY))
            self._codec_min_tokens = int(getattr(tts_config, "min_new_tokens", _CODEC_MIN_TOKENS))
            # P23: Gumbel-max sampling path (see _sample_audio_code). Rollback:
            # OMNI_LZ_GUMBEL=0 restores torch.multinomial.
            self._lz_gumbel_sampling = os.environ.get("OMNI_LZ_GUMBEL", "1") not in {"0", "false", "no", "off"}
        else:
            self._tts_config = None
            self._lz_gumbel_sampling = False

        self.has_preprocess = True
        self.has_postprocess = False
        self.gpu_resident_buffer_keys: set[tuple[str, str]] = {
            ("audio_codes", "current"),
            ("audio_codes", "accumulated"),
        }
        self._init_native_talker(prefix)

    def _init_native_talker(self, prefix: str) -> None:
        if self._tts_config is None:
            raise ValueError("MiniCPM-o continuous Talker requires tts_config")
        cfg = self._tts_config
        if int(getattr(cfg, "num_vq", 1)) != 1:
            raise ValueError(
                "MiniCPM-o continuous Talker currently requires num_vq=1; "
                f"checkpoint reports {getattr(cfg, 'num_vq', None)}"
            )
        llama_config = LlamaConfig(
            vocab_size=32000,
            hidden_size=int(cfg.hidden_size),
            intermediate_size=int(cfg.intermediate_size),
            num_hidden_layers=int(cfg.num_hidden_layers),
            num_attention_heads=int(cfg.num_attention_heads),
            num_key_value_heads=int(cfg.num_key_value_heads),
            hidden_act=getattr(cfg, "hidden_act", "silu"),
            max_position_embeddings=int(cfg.max_position_embeddings),
            rms_norm_eps=float(getattr(cfg, "rms_norm_eps", 1e-6)),
            tie_word_embeddings=False,
        )
        talker_config = self.vllm_config.with_hf_config(llama_config, architectures=["LlamaForCausalLM"])
        talker_config.model_config.hf_text_config = llama_config
        self.tts_model = LlamaModel(
            vllm_config=talker_config,
            prefix=maybe_prefix(prefix, "tts_obj.model"),
        )
        self.emb_text = nn.Embedding(int(cfg.num_text_tokens), int(cfg.hidden_size))
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg.llm_dim), int(cfg.hidden_size))
        self.emb_code = nn.ModuleList(
            [nn.Embedding(int(cfg.num_audio_tokens), int(cfg.hidden_size)) for _ in range(int(cfg.num_vq))]
        )
        self.head_code = nn.ModuleList(
            [nn.Linear(int(cfg.hidden_size), int(cfg.num_audio_tokens), bias=False) for _ in range(int(cfg.num_vq))]
        )
        self.make_empty_intermediate_tensors = self.tts_model.make_empty_intermediate_tensors

    def _boundary_embeddings(self) -> torch.Tensor:
        """Embed the ``<text_eos><audio_bos>`` tail every condition ends with."""
        ids = torch.tensor(
            [self._text_eos_id, self._tts_bos_id],
            device=self.emb_text.weight.device,
            dtype=torch.long,
        )
        return self.emb_text(ids)

    def _build_condition_embeddings(
        self,
        tts_token_ids: torch.Tensor,
        tts_hidden_states: torch.Tensor,
        *,
        native_duplex: bool = False,
    ) -> torch.Tensor:
        if tts_token_ids.numel() == 0 or tts_hidden_states.numel() == 0:
            # The thinker can legally emit an empty speech segment (<|tts_bos|>
            # immediately followed by a boundary token) when it decides not to
            # speak. Condition on the boundary tokens alone, which matches the
            # 2-token scheduler prompt the stage bridge builds for an empty
            # handoff.
            return self._boundary_embeddings()
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        token_ids = tts_token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = tts_hidden_states.to(device=device, dtype=dtype)
        if hidden.shape[0] != token_ids.shape[0] and token_ids.shape[0] != 1:
            raise ValueError(
                "MiniCPM-o Talker condition length mismatch: "
                f"token_ids={token_ids.shape[0]} hidden_states={hidden.shape[0]}"
            )
        text_embeds = self.emb_text(token_ids)
        hidden_embeds = self.projector_semantic(hidden)
        if self._normalize:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        audio_bos = self.emb_text(torch.tensor([self._tts_bos_id], device=device, dtype=torch.long))
        condition = text_embeds + hidden_embeds
        if native_duplex:
            # Match MiniCPMTTS.generate_chunk's streaming condition.
            return torch.cat([condition, audio_bos], dim=0)
        return torch.cat([condition, self._boundary_embeddings()], dim=0)

    def preprocess(
        self,
        input_ids: torch.Tensor,
        input_embeds: torch.Tensor | None,
        **info_dict: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any]]:
        """Build request-local prefill/decode embeddings for the vLLM runner."""
        del input_embeds
        span_len = int(input_ids.shape[0])
        is_prefill = bool(info_dict.get("_omni_is_prefill", False))
        state = info_dict.get("audio_state")
        first_call = not isinstance(state, dict)

        if is_prefill or first_call:
            token_ids, hidden_states = get_tts_handoff(info_dict)
            # Cross-process stage transport serializes CPU tensors as lists.
            # Normalize both local tensor handoffs and transported payloads
            # before validating/building the Talker condition.
            if isinstance(token_ids, (list, tuple)):
                token_ids = torch.as_tensor(token_ids, dtype=torch.long)
            if isinstance(hidden_states, (list, tuple)):
                hidden_states = torch.as_tensor(hidden_states, dtype=torch.float32)
            if not isinstance(token_ids, torch.Tensor) or not isinstance(hidden_states, torch.Tensor):
                available = sorted(key for key in info_dict if not key.startswith("_"))
                raise ValueError(
                    "MiniCPM-o Talker requires tensor tts_token_ids and "
                    "tts_hidden_states conditioning; "
                    f"received token_ids={type(token_ids).__name__}, "
                    f"hidden_states={type(hidden_states).__name__}, "
                    f"available_keys={available}"
                )
            # An empty condition means the thinker chose not to speak: finish the
            # request up front so it emits zero audio codes instead of killing
            # the stage engine.
            empty_condition = token_ids.numel() == 0 or hidden_states.numel() == 0
            if empty_condition:
                logger.warning_once(
                    "MiniCPM-o Talker received an empty condition (request %s); this request produces no audio.",
                    info_dict.get("request_id"),
                )
            native_duplex = bool(info_dict.get("native_duplex", False))
            full_embeds = self._build_condition_embeddings(
                token_ids,
                hidden_states,
                native_duplex=native_duplex,
            )
            offset = int(info_dict.get("_omni_num_computed_tokens", 0))
            request_id = str(info_dict.get("request_id", "0"))
            meta = info_dict.get("meta")
            # The handoff rebuilds only the tail-aligned Talker condition.
            # Materialize zero-token embeddings for any scheduler prompt
            # prefix so chunked prefill can slice from a non-zero offset.
            prompt_len = info_dict.get("_omni_prompt_len")
            target_len = int(prompt_len) if prompt_len is not None else offset + span_len
            prefix_len = target_len - full_embeds.shape[0]
            if prefix_len > 0:
                placeholder_ids = torch.zeros(
                    prefix_len,
                    dtype=torch.long,
                    device=self.emb_text.weight.device,
                )
                full_embeds = torch.cat([self.emb_text(placeholder_ids), full_embeds], dim=0)
            embeds = full_embeds[offset : offset + span_len]
            if embeds.shape[0] != span_len:
                raise ValueError(
                    "MiniCPM-o Talker prefill span exceeds condition: "
                    f"request_id={info_dict.get('request_id')} offset={offset} "
                    f"span={span_len} condition={full_embeds.shape[0]} "
                    f"tts_ids={token_ids.shape[0]} tts_hidden={hidden_states.shape[0]} "
                    f"prompt_len={info_dict.get('_omni_prompt_len')}"
                )
            duplex_boundary = isinstance(meta, dict) and (
                bool(meta.get("turn_start", False)) or bool(meta.get("turn_end", False))
            )
            if native_duplex:
                max_tokens = _DUPLEX_CODEC_TOKENS_PER_CHUNK
                min_tokens = 0 if duplex_boundary else _DUPLEX_CODEC_TOKENS_PER_CHUNK
            else:
                max_tokens = _max_audio_tokens(int(token_ids.numel()))
                min_tokens = self._codec_min_tokens
            state = {
                "step": 0,
                "max_tokens": max_tokens,
                "min_tokens": min_tokens,
                "finished": empty_condition,
            }
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            request_states[request_id] = state
            empty_codes = torch.empty(0, dtype=torch.long, device=embeds.device)
            return (
                input_ids,
                embeds,
                {
                    "audio_state": state,
                    "audio_codes": {
                        "current": empty_codes,
                        "accumulated": empty_codes,
                    },
                },
            )

        current = (info_dict.get("audio_codes", {}) or {}).get("current")
        if not isinstance(current, torch.Tensor) or current.numel() != 1:
            if state.get("finished"):
                # A request that finished before sampling any code can still be
                # scheduled for decode steps while sampling min_tokens masks the
                # stop token. make_omni_output ignores its hidden states, so any
                # shape-correct embedding will do.
                weight = self.emb_code[0].weight
                return input_ids, weight.new_zeros((span_len, weight.shape[1])), {}
            raise RuntimeError("MiniCPM-o Talker decode is missing the previous request-local audio code")
        code = current.to(device=self.emb_code[0].weight.device, dtype=torch.long).reshape(1)
        embeds = self.emb_code[0](code)
        return input_ids, embeds, {}

    def _request_generator(self, request_id: str, device: torch.device) -> torch.Generator:
        generator = self._request_generators.get(request_id)
        if generator is None:
            generator = torch.Generator(device=device)
            generator.manual_seed(self._codec_seed)
            self._request_generators[request_id] = generator
        return generator

    def _lz_sampler_head_graph(self, hidden_state: torch.Tensor) -> _LZSamplerHeadBucket | None:
        """P27: head-linear bucket for this signature, capturing lazily.

        Signature (hidden_size, vocab, dtype) — the talker samples one row per
        request, so a deployment holds a single entry. Returns None (eager)
        when disabled/dead or when the graph budget is exhausted.
        """
        if self._lz_sampler_graph_dead or not self._lz_gumbel_sampling:
            return None
        key = (int(hidden_state.shape[-1]), int(self._num_audio_tokens), str(hidden_state.dtype))
        bucket = self._lz_sampler_head_graphs.get(key, _LZ_SAMPLER_MISSING)
        if bucket is _LZ_SAMPLER_MISSING:
            if len(self._lz_sampler_head_graphs) + len(self._lz_sampler_tail_graphs) >= _LZ_SAMPLER_MAX_GRAPHS:
                self._lz_sampler_graph_dead = True
                logger.info("P27 sampler graph budget exhausted; sampler stays eager")
                return None
            bucket = self._lz_capture_head_graph(hidden_state)
            self._lz_sampler_head_graphs[key] = bucket
        return bucket

    def _lz_capture_head_graph(self, hidden_state: torch.Tensor) -> _LZSamplerHeadBucket | None:
        """Capture ``head_code linear + temperature``; eager forever on failure."""
        try:
            if not (hasattr(torch, "npu") and hasattr(torch.npu, "NPUGraph")):
                raise RuntimeError("torch.npu.NPUGraph is unavailable on this platform")
            device = hidden_state.device
            b = _LZSamplerHeadBucket()
            b.h_in = torch.zeros((1, int(hidden_state.shape[-1])), dtype=hidden_state.dtype, device=device)

            def _body(h: torch.Tensor) -> torch.Tensor:
                return _lz_head_code_logits(self.head_code[0], h, self._codec_temperature)

            with torch.no_grad():
                _body(b.h_in)  # allocator/kernel warmup on the static buffers
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.no_grad(), torch.npu.graph(graph):
                # Tensors allocated during capture live in the graph's private
                # pool and keep a stable address across replays (same pattern
                # as the CFM/HiFT buckets in batched_token2wav.py).
                b.logits_out = _body(b.h_in)
            b.graph = graph
            torch.npu.synchronize()
            logger.info("P27 sampler head graph captured: hidden=%d", b.h_in.shape[-1])
            return b
        except Exception:
            logger.exception("P27 sampler head graph capture failed; sampler stays eager")
            self._lz_sampler_graph_dead = True
            return None

    def _lz_sampler_tail_graph(self, logits: torch.Tensor, mask_eos: bool) -> _LZSamplerTailBucket | None:
        """P27: Gumbel-tail bucket for (vocab, dtype, min_tokens state)."""
        if self._lz_sampler_graph_dead:
            return None
        key = (int(logits.shape[-1]), str(logits.dtype), bool(mask_eos))
        bucket = self._lz_sampler_tail_graphs.get(key, _LZ_SAMPLER_MISSING)
        if bucket is _LZ_SAMPLER_MISSING:
            if len(self._lz_sampler_head_graphs) + len(self._lz_sampler_tail_graphs) >= _LZ_SAMPLER_MAX_GRAPHS:
                self._lz_sampler_graph_dead = True
                logger.info("P27 sampler graph budget exhausted; sampler stays eager")
                return None
            bucket = self._lz_capture_tail_graph(logits, bool(mask_eos))
            self._lz_sampler_tail_graphs[key] = bucket
        return bucket

    def _lz_capture_tail_graph(self, logits: torch.Tensor, mask_eos: bool) -> _LZSamplerTailBucket | None:
        """Capture ``eos mask + log(q) + sub + argmax`` for one min_tokens state."""
        try:
            if not (hasattr(torch, "npu") and hasattr(torch.npu, "NPUGraph")):
                raise RuntimeError("torch.npu.NPUGraph is unavailable on this platform")
            device = logits.device
            vocab = int(logits.shape[-1])
            b = _LZSamplerTailBucket()
            b.logits_in = torch.zeros((1, vocab), dtype=logits.dtype, device=device)
            b.q_in = torch.zeros((1, vocab), dtype=logits.dtype, device=device)
            # min_tokens EOS mask as a static input: the False variant is an
            # all-False no-op, the True variant pins -inf at the EOS column —
            # both bit-identical to the eager scalar assign.
            eos_mask = torch.zeros((1, vocab), dtype=torch.bool, device=device)
            if mask_eos:
                eos_mask[..., self._num_audio_tokens - 1] = True

            def _body() -> torch.Tensor:
                return _lz_gumbel_tail(b.logits_in, b.q_in, eos_mask if mask_eos else None)

            with torch.no_grad():
                _body()
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.no_grad(), torch.npu.graph(graph):
                b.id_out = _body()
            b.graph = graph
            torch.npu.synchronize()
            logger.info("P27 sampler tail graph captured: vocab=%d mask_eos=%s", vocab, mask_eos)
            return b
        except Exception:
            logger.exception("P27 sampler tail graph capture failed; sampler stays eager")
            self._lz_sampler_graph_dead = True
            return None

    def _lz_gumbel_row(self, request_id: str, logits: torch.Tensor, step: int) -> torch.Tensor | None:
        """This step's Exp(1) noise row (P23), or None without request state.

        Extracted verbatim from the original inline block so the generator's
        consumption order stays exactly the same (same block size, same redraw
        points); the caller falls back to ``multinomial`` on None.
        """
        request_states = getattr(self, "_request_audio_states", {})
        state = request_states.get(request_id) if isinstance(request_states, dict) else None
        if not isinstance(state, dict):
            return None
        device = logits.device
        vocab = logits.shape[-1]
        noise = state.get("lz_gumbel")
        if not isinstance(noise, torch.Tensor) or noise.device != device or noise.shape[-1] != vocab:
            noise = torch.empty((_LZ_GUMBEL_BLOCK, vocab), device=device)
        row = step % _LZ_GUMBEL_BLOCK
        if step == 0 or row == 0 or not isinstance(noise, torch.Tensor) or noise.shape[-1] != vocab:
            noise.exponential_(generator=self._request_generator(request_id, device))
            state["lz_gumbel"] = noise
        return noise[row : row + 1]

    def _sample_audio_code(
        self,
        hidden_state: torch.Tensor,
        history: torch.Tensor,
        request_id: str,
        step: int,
    ) -> torch.Tensor:
        # P27: replay the head linear + temperature scale from a graph when
        # available. The replay output is a fixed buffer holding the raw
        # (pre-warp) logits; the eager warps below only read it.
        head_bucket = self._lz_sampler_head_graph(hidden_state)
        if head_bucket is not None:
            head_bucket.h_in.copy_(hidden_state)
            head_bucket.graph.replay()
            logits = head_bucket.logits_out
        else:
            logits = self.head_code[0](hidden_state).float() / self._codec_temperature
        eos_id = self._num_audio_tokens - 1
        logits = _apply_repetition_penalty(
            logits,
            history,
            penalty=self._codec_repetition_penalty,
            window_size=_REPETITION_WINDOW,
        )
        request_states = getattr(self, "_request_audio_states", {})
        state = request_states.get(request_id)
        min_tokens = (
            int(state.get("min_tokens", self._codec_min_tokens)) if isinstance(state, dict) else self._codec_min_tokens
        )
        if self._lz_gumbel_sampling:
            q = self._lz_gumbel_row(request_id, logits, step)
            if q is not None:
                mask_eos = step < min_tokens
                # P27: replay the deterministic tail (eos mask + log q + sub +
                # argmax). The min_tokens EOS mask rides inside the tail graph
                # (hence the mask_eos signature key); masking after the warp is
                # equivalent to the eager mask-before-warp order because -inf
                # carries zero softmax mass and never survives top-k/top-p.
                tail_bucket = self._lz_sampler_tail_graph(logits, mask_eos)
                if tail_bucket is not None:
                    tail_bucket.logits_in.copy_(logits)
                    tail_bucket.q_in.copy_(q)
                    tail_bucket.graph.replay()
                    # The fixed output buffer is reused by every replay:
                    # clone before the id reaches longer-lived consumers.
                    return tail_bucket.id_out.clone().reshape(())
                if mask_eos:
                    logits[..., eos_id] = float("-inf")
                logits = _apply_top_k_top_p(
                    logits,
                    top_k=self._codec_top_k,
                    top_p=self._codec_top_p,
                    min_tokens_to_keep=3,
                )
                return (logits - torch.log(q.clamp_min(1e-9))).argmax(dim=-1).reshape(())
        if step < min_tokens:
            logits[..., eos_id] = float("-inf")
        logits = _apply_top_k_top_p(
            logits,
            top_k=self._codec_top_k,
            top_p=self._codec_top_p,
            min_tokens_to_keep=3,
        )
        # P23 Gumbel-max reparameterisation.  ``multinomial`` consumes device
        # RNG inside the sampling kernel chain (softmax + exp + cumdiv +
        # draw), which both costs extra per-step kernel launches and pins the
        # sampling op to eager (device RNG state cannot be replayed inside a
        # graph).  The exponential-race identity
        # ``argmax(l - log E) ~ categorical(softmax(l))`` for E ~ Exp(1)
        # moves the randomness into a *data* input: noise is drawn eagerly in
        # pre-generated per-request blocks (same generator, same consumption
        # order per step), leaving only ``add`` + ``argmax`` kernels on the
        # sampling path.  The drawn values are distribution-equal but not
        # bit-identical to ``multinomial`` (documented, WER/SIM gated).
        # The Gumbel branch (with the P27 graph replay of its deterministic
        # tail) is handled above; this remainder is the multinomial fallback
        # for OMNI_LZ_GUMBEL=0 or a missing request state.
        probabilities = torch.softmax(logits, dim=-1)
        return torch.multinomial(
            probabilities,
            num_samples=1,
            generator=self._request_generator(request_id, probabilities.device),
        ).reshape(())

    def _stop_row_templates_for(self, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Device-resident (continue_row, stop_row) logit templates.

        ``hidden.new_tensor([...])`` copies two floats host->device on every
        decode step; the cached rows are bit-identical, allocation-free, and
        only ever read (torch.stack copies them into the batch tensor).
        """
        key = (hidden.device, hidden.dtype)
        cached = self._stop_row_templates.get(key)
        if cached is None:
            cached = (
                hidden.new_tensor([0.0, float("-inf")]),
                hidden.new_tensor([float("-inf"), 0.0]),
            )
            self._stop_row_templates[key] = cached
        return cached

    def make_omni_output(
        self,
        model_outputs: torch.Tensor | OmniOutput,
        **kwargs: Any,
    ) -> OmniOutput:
        if isinstance(model_outputs, OmniOutput):
            return model_outputs
        hidden = model_outputs
        infos = kwargs.get("model_intermediate_buffer") or []
        spans = kwargs.get("request_token_spans")
        if spans is None or len(spans) != len(infos):
            raise RuntimeError("MiniCPM-o continuous Talker requires one request_token_span per request")
        sample_eligible = kwargs.get("request_sample_eligible")
        if sample_eligible is None:
            sample_eligible = [True] * len(infos)
        if len(sample_eligible) != len(infos):
            raise RuntimeError(
                f"MiniCPM-o continuous Talker received {len(sample_eligible)} sampling flags for {len(infos)} requests"
            )
        emit_duplex_metadata = any(isinstance(info, dict) and info.get("native_duplex") is True for info in infos)

        stop_rows: list[torch.Tensor] = []
        codec_deltas: list[torch.Tensor] = []
        terminal_flags: list[torch.Tensor] = []
        native_duplex_flags: list[torch.Tensor] = []
        duplex_epochs: list[torch.Tensor] = []
        duplex_turn_ids: list[torch.Tensor] = []
        segment_texts_utf8: list[torch.Tensor] = []
        turn_end_flags: list[torch.Tensor] = []
        empty_delta = hidden.new_empty((0, 1), dtype=torch.long)
        # P12b: device-resident templates, cached per (device, dtype).
        continue_row, stop_row_template = self._stop_row_templates_for(hidden)
        for index, info in enumerate(infos):
            info_dict = info if isinstance(info, dict) else {}
            native_duplex = info_dict.get("native_duplex") is True
            if emit_duplex_metadata:
                duplex_info = info_dict.get("duplex")
                if not isinstance(duplex_info, dict):
                    duplex_info = {}
                epoch = duplex_info.get("epoch", -1)
                turn_id = duplex_info.get("turn_id", -1)
                if native_duplex and not all(
                    isinstance(value, int) and not isinstance(value, bool) and value >= 0 for value in (epoch, turn_id)
                ):
                    raise RuntimeError(
                        "MiniCPM-o native duplex Talker requires non-negative integer "
                        f"epoch and turn_id, got epoch={epoch!r}, turn_id={turn_id!r}"
                    )
                meta_info = info_dict.get("meta")
                if not isinstance(meta_info, dict):
                    meta_info = {}
                segment_text = meta_info.get("native_duplex_segment_text", "") if native_duplex else ""
                if not isinstance(segment_text, str):
                    segment_text = ""
                turn_eos_id = meta_info.get("turn_eos_token_id")
                ids_info = info_dict.get("ids")
                tts_ids = ids_info.get("tts") if native_duplex and isinstance(ids_info, dict) else None
                if isinstance(tts_ids, torch.Tensor):
                    contains_turn_eos = isinstance(turn_eos_id, int) and bool(
                        torch.any(tts_ids.reshape(-1) == turn_eos_id).item()
                    )
                elif isinstance(tts_ids, (list, tuple)):
                    contains_turn_eos = isinstance(turn_eos_id, int) and turn_eos_id in tts_ids
                else:
                    contains_turn_eos = False
                native_duplex_flags.append(torch.tensor(native_duplex, dtype=torch.bool))
                duplex_epochs.append(torch.tensor(epoch if isinstance(epoch, int) else -1, dtype=torch.long))
                duplex_turn_ids.append(torch.tensor(turn_id if isinstance(turn_id, int) else -1, dtype=torch.long))
                segment_texts_utf8.append(
                    torch.tensor(
                        list(segment_text.encode("utf-8")),
                        dtype=torch.uint8,
                    )
                )
                turn_end_flags.append(torch.tensor(native_duplex and contains_turn_eos, dtype=torch.bool))

            if not isinstance(info, dict):
                stop_rows.append(continue_row)
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                continue
            start, end = spans[index]
            end = min(int(end), int(hidden.shape[0]))
            if int(start) >= end:
                stop_rows.append(continue_row)
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                continue
            request_id = str(info.get("request_id", index))
            request_states = getattr(self, "_request_audio_states", None)
            if request_states is None:
                request_states = {}
                self._request_audio_states = request_states
            state = request_states.get(request_id)
            if not isinstance(state, dict):
                state = dict(info.get("audio_state", {}) or {})
                request_states[request_id] = state
            if state.get("finished"):
                stop_rows.append(stop_row_template)
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                continue
            if not sample_eligible[index]:
                # vLLM computes a logit row for incomplete chunked prefills but
                # discards its sampled token. Advancing codec/RNG state here
                # would make output depend on prefill chunking and compaction.
                stop_rows.append(continue_row)
                codec_deltas.append(empty_delta)
                terminal_flags.append(torch.tensor(False, dtype=torch.bool))
                continue
            codes = state.get("codes")
            if not isinstance(codes, torch.Tensor):
                codes = (info.get("audio_codes", {}) or {}).get("accumulated")
            if not isinstance(codes, torch.Tensor):
                codes = torch.empty(0, dtype=torch.long, device=hidden.device)
            else:
                codes = codes.to(device=hidden.device, dtype=torch.long).reshape(-1)
            step = int(state.get("step", 0))
            sampled = self._sample_audio_code(hidden[end - 1 : end], codes, request_id, step)
            sampled_id = int(sampled.item())
            is_eos = sampled_id == self._num_audio_tokens - 1
            state["step"] = int(state.get("step", 0)) + 1
            reached_limit = int(state["step"]) >= int(state.get("max_tokens", 2048))
            finished = is_eos or reached_limit
            state["finished"] = finished
            # Equivalence probe (env-gated): one line per sampled codec token,
            # request-local and order-stable, so windowed and single-step runs
            # can be diffed line by line.  Audio bytes are NOT comparable
            # run-to-run because the HiFT SineGen draws unseeded noise.
            _dump = os.environ.get("OMNI_LZ_CODEC_DUMP")
            if _dump:
                with open(_dump, "a") as _f:
                    _f.write(f"{request_id} {state['step']} {sampled_id} {int(finished)}\n")
            # MiniCPMTTS.generate_chunk consumes the boundary sample but
            # returns only codes that were fed into the retained KV state.
            if not is_eos and not reached_limit:
                codes = torch.cat([codes[-(_REPETITION_WINDOW - 1) :], sampled.reshape(1)])
                delta = sampled.reshape(1, 1)
            else:
                delta = empty_delta
            state["codes"] = codes
            info["audio_state"] = state
            info["audio_codes"] = {
                "current": sampled.reshape(1),
                "accumulated": codes,
            }
            codec_deltas.append(delta)
            terminal_flags.append(torch.tensor(finished, dtype=torch.bool))
            stop_rows.append(stop_row_template if finished else continue_row)

        self._batch_stop_logits = torch.stack(stop_rows, dim=0) if stop_rows else hidden.new_empty((0, 2))
        # Lists are deliberate: the runner routes element i to request i,
        # preserving compaction alignment while emitting only this step's code.
        meta_outputs = {"finished": terminal_flags}
        if emit_duplex_metadata:
            meta_outputs.update(
                {
                    "native_duplex": native_duplex_flags,
                    "duplex_epoch": duplex_epochs,
                    "duplex_turn_id": duplex_turn_ids,
                    "llm_output_text_utf8": segment_texts_utf8,
                    "turn_end": turn_end_flags,
                }
            )
        multimodal_outputs: dict[str, Any] = {
            "codes": {"audio": codec_deltas},
            "meta": meta_outputs,
        }
        return OmniOutput(
            text_hidden_states=hidden,
            multimodal_outputs=multimodal_outputs,
        )

    def on_requests_finished(self, finished_req_ids: set[str] | list[str]) -> None:
        self._deferred_cleanup_ids.update(str(req_id) for req_id in finished_req_ids)

    def _flush_deferred_cleanup(self) -> None:
        request_audio_states = getattr(self, "_request_audio_states", {})
        for request_id in self._deferred_cleanup_ids:
            self._request_generators.pop(request_id, None)
            request_audio_states.pop(request_id, None)
        self._deferred_cleanup_ids.clear()

    def _dummy_hidden_states(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> torch.Tensor:
        """Shape-correct zero tensor for vllm KV cache profiling.

        vllm's gpu_model_runner._dummy_run takes forward()'s return value as
        ``hidden_states`` and does ``hidden_states[logit_indices_device]``;
        returning None on the dummy path crashes with
        ``TypeError: 'NoneType' object is not subscriptable``.
        """
        for ref in (input_ids, positions, inputs_embeds):
            if isinstance(ref, torch.Tensor):
                num_tokens = int(ref.shape[0]) if ref.ndim >= 1 else 1
                device = ref.device
                break
        else:
            num_tokens = 1
            device = current_omni_platform.get_torch_device()
        hidden_size = int(getattr(self, "_hidden_size", 768) or 768)
        return torch.zeros((num_tokens, hidden_size), device=device, dtype=torch.bfloat16)

    def forward(
        self,
        input_ids=None,
        positions=None,
        intermediate_tensors=None,
        inputs_embeds=None,
        **kwargs,
    ):
        self._flush_deferred_cleanup()
        if input_ids is None and inputs_embeds is None:
            return self._dummy_hidden_states(input_ids, positions, inputs_embeds)
        return self.tts_model(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
        )

    def compute_logits(self, hidden_states, *args, **kwargs):
        if not isinstance(hidden_states, torch.Tensor):
            return None
        if self._batch_stop_logits is None:
            return torch.zeros(
                hidden_states.shape[0],
                2,
                device=hidden_states.device,
                dtype=torch.float32,
            )
        logits = self._batch_stop_logits
        self._batch_stop_logits = None
        return logits

    def sample(self, logits, sampling_metadata):
        return Sampler()(logits, sampling_metadata)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return self._load_native_weights(weights)

    def _load_native_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g = head_v = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name[len("tts.") :]
            if stripped.startswith("model."):
                backbone_weights.append((stripped[len("model.") :], tensor))
                continue
            if stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            if stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            target = stripped
            parameter = direct_params.get(target)
            if parameter is None:
                continue
            parameter.data.copy_(tensor.to(device=parameter.device, dtype=parameter.dtype))
            loaded.add(target)

        for name in self.tts_model.load_weights(backbone_weights):
            loaded.add(f"tts_model.{name}")

        if head_g is None or head_v is None:
            raise ValueError("MiniCPM-o checkpoint is missing weight-norm Talker head parameters")
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code[0].weight.data.copy_(
            restored.to(
                device=self.head_code[0].weight.device,
                dtype=self.head_code[0].weight.dtype,
            )
        )
        loaded.add("head_code.0.weight")
        return loaded

    def get_input_embeddings(self, input_ids, multimodal_embeddings=None, **kwargs):
        if hasattr(self, "emb_text") and self.emb_text is not None:
            return self.emb_text(input_ids)
        return torch.zeros(input_ids.shape[0], 1)

    def embed_input_ids(self, input_ids, **kwargs):
        return self.get_input_embeddings(input_ids, **kwargs)

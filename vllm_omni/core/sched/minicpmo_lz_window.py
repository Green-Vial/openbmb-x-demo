# SPDX-License-Identifier: Apache-2.0
"""Runner-local K-window decode planning for the MiniCPM-o 4.5 Talker stage.

When an async-scheduling AR stage is in steady 1-token decode, the per-step
engine loop (schedule -> input prep -> forward -> sample -> reconcile -> IPC)
is dominated by fixed host cost.  This module lets the runner replay K decode
steps inside a single engine step by rewriting the scheduler output *before*
it reaches the worker:

1. ``num_scheduled_tokens`` for the whole batch is bumped from 1 to K.  The
   base scheduler already ran, so this is a post-schedule patch: the extra
   K-1 KV slots are allocated here (``allocate_slots``) and the new block ids
   are appended to ``scheduled_cached_reqs.new_block_ids`` so the worker's
   block table covers every slot the window will write.
2. ``request.num_output_placeholders`` and ``request.num_computed_tokens``
   are inflated by K-1.  The async-scheduling accounting then treats the
   window exactly like a step that samples K tokens per request: the K
   reported tokens drive ``num_output_placeholders`` back to zero, while
   ``update_from_output`` replays the shortfall if the runner produced fewer
   tokens than planned (early stop or single-step fallback), so the engine's
   confirmed ``num_computed_tokens`` always matches the KV slots actually
   written.
3. The inflated placeholder count doubles as the fence: while a window is in
   flight, ``num_tokens_with_spec + placeholders - computed == 0`` keeps the
   request out of subsequent schedules until its K tokens are reconciled.

Every refusal path is fail-closed: the patch is skipped entirely and the
normal single-step engine loop runs.  Rollback: ``OMNI_LZ_LOCAL_K=0`` makes
``resolve_local_k`` return 0 and disables the whole mechanism.
"""

from __future__ import annotations

import os
from typing import Any

from vllm.logger import init_logger
from vllm.v1.request import RequestStatus

logger = init_logger(__name__)

# Environment switch: "0" disables the window, positive values select K
# (clamped to MAX_LOCAL_K).  Unset falls back to DEFAULT_LOCAL_K.
ENV_LOCAL_K = "OMNI_LZ_LOCAL_K"
DEFAULT_LOCAL_K = 10
MAX_LOCAL_K = 16

# Stage-1 Talker architectures whose model-side sampling advances per-request
# codec state inside make_omni_output; the window relies on that in-place
# state chain, so other archs never enter a window.
WINDOW_MODEL_ARCHS = frozenset({
    "MiniCPMO45OmniTTSForConditionalGeneration",
    # Stage engines resolve architectures from the pipeline-level arch (the
    # tts stage declares no per-stage model_arch), so the scheduler sees the
    # wrapper-family name rather than the inner TTS class name.
    "MiniCPMO45OmniForConditionalGeneration",
})


def resolve_local_k(raw: str | int | None = None) -> int:
    """Parse the window size from ``OMNI_LZ_LOCAL_K``.

    Returns 0 when the window is disabled (explicit ``0``/negative values or
    unparseable input) and otherwise K clamped to ``[1, MAX_LOCAL_K]``.
    """
    if raw is None:
        raw = os.environ.get(ENV_LOCAL_K, "")
    if isinstance(raw, int):
        value = raw
    else:
        text = str(raw).strip()
        if not text:
            value = DEFAULT_LOCAL_K
        else:
            try:
                value = int(text, 10)
            except ValueError:
                logger.warning_once(
                    "Invalid %s=%r; falling back to K=0 (window disabled).",
                    ENV_LOCAL_K,
                    raw,
                )
                return 0
    if value <= 0:
        return 0
    return min(value, MAX_LOCAL_K)


def scheduler_allows_window(scheduler: Any) -> bool:
    """Static configuration gate evaluated once per ``schedule()`` call.

    Only single-rank (TP/PP/DP/PCP/DCP == 1), spec-decode-free, LoRA-free
    MiniCPM-o Talker stages on async scheduling are eligible.  Anything the
    runner cannot reproduce bit-identically (KV-transfer criteria, routed
    experts, encoder-decoder, mamba-aligned caches) is refused here.
    """
    vllm_config = getattr(scheduler, "vllm_config", None)
    if vllm_config is None:
        return False
    model_config = vllm_config.model_config
    archs = set(getattr(model_config, "architectures", None) or ())
    if not archs & WINDOW_MODEL_ARCHS:
        return False
    # The window relies on the Talker's request-local codec state chain; the
    # thinker stage of the same wrapper family shares the arch name and must
    # never open windows.
    if getattr(model_config, "model_stage", None) != "tts":
        return False
    if not getattr(scheduler.scheduler_config, "async_scheduling", False):
        return False
    parallel_config = vllm_config.parallel_config
    if (
        parallel_config.pipeline_parallel_size != 1
        or parallel_config.tensor_parallel_size != 1
        or parallel_config.data_parallel_size != 1
        or getattr(parallel_config, "pcp_size", 1) != 1
        or getattr(parallel_config, "dcp_size", 1) != 1
    ):
        return False
    if getattr(scheduler, "num_spec_tokens", 0) != 0:
        return False
    if getattr(scheduler, "kv_transfer_criteria", None) is not None:
        return False
    if getattr(scheduler, "lora_config", None) is not None:
        return False
    if getattr(model_config, "is_encoder_decoder", False):
        return False
    if getattr(model_config, "enable_return_routed_experts", False):
        return False
    cache_config = getattr(vllm_config, "cache_config", None)
    if getattr(cache_config, "mamba_cache_mode", "off") == "align":
        return False
    return True


def _request_admits_window(request: Any) -> bool:
    """Per-request eligibility for one window step."""
    if request is None or request.is_finished():
        return False
    if getattr(request, "status", None) != RequestStatus.RUNNING:
        return False
    # Decode phase only: the window writes K contiguous post-prompt slots and
    # cannot interleave with chunked prefill or prefix-cache (re)computation.
    if request.num_computed_tokens < request.num_prompt_tokens:
        return False
    if getattr(request, "has_encoder_inputs", False):
        return False
    if getattr(request, "use_structured_output", False):
        return False
    if getattr(request, "pooling_params", None) is not None:
        return False
    params = request.sampling_params
    if params is None:
        return False
    if params.logprobs is not None or params.prompt_logprobs is not None:
        return False
    if getattr(params, "bad_words_token_ids", None):
        return False
    if getattr(params, "allowed_token_ids", None):
        return False
    # Value-dependent logits processors must stay off: the window keeps the
    # placeholder (-1) bookkeeping in step, but only length-based processors
    # (min_tokens) are value-independent.
    if getattr(params, "frequency_penalty", 0.0) != 0.0:
        return False
    if getattr(params, "presence_penalty", 0.0) != 0.0:
        return False
    if getattr(params, "repetition_penalty", 1.0) != 1.0:
        return False
    return True


def _window_budget(request: Any, max_model_len: int) -> int:
    """Largest K this request can host: min(env K, max_tokens left, context)."""
    params = request.sampling_params
    max_tokens = int(getattr(params, "max_tokens", 0) or 0)
    remaining = max_tokens - len(request.output_token_ids) if max_tokens else 0
    context_room = max_model_len - request.num_tokens
    return min(remaining, context_room)


def plan_lz_window(scheduler: Any, scheduler_output: Any, local_k: int) -> bool:
    """Rewrite ``scheduler_output`` in place into a K-window decode step.

    Must be called right after the base ``schedule()`` produced a plain
    single-token decode step.  Returns True when the output was rewritten;
    on any refusal the output and the request accounting are left untouched
    so the caller falls back to the normal single-step path (the scheduler
    side of ``update_from_output`` reconciles the reservation shortfall if
    the worker ever declines a plan that was emitted).
    """
    if local_k < 2:
        return False
    num_scheduled = scheduler_output.num_scheduled_tokens
    if not num_scheduled:
        return False
    if scheduler_output.scheduled_new_reqs:
        return False
    if scheduler_output.scheduled_spec_decode_tokens:
        return False
    if scheduler_output.scheduled_encoder_inputs:
        return False
    if getattr(scheduler_output, "has_structured_output_requests", False):
        return False
    if getattr(scheduler_output, "pending_structured_output_tokens", False):
        return False

    cached_reqs = scheduler_output.scheduled_cached_reqs
    cached_index = {req_id: i for i, req_id in enumerate(cached_reqs.req_ids)}
    windowed: list[tuple[str, Any]] = []
    for req_id, num_tokens in num_scheduled.items():
        # The base scheduler must have scheduled exactly one decode token.
        if num_tokens != 1:
            return False
        request = scheduler.requests.get(req_id)
        if not _request_admits_window(request):
            return False
        if req_id not in cached_index:
            return False
        windowed.append((req_id, request))

    if not windowed:
        return False

    max_model_len = scheduler.max_model_len
    window_k = local_k
    for _, request in windowed:
        window_k = min(window_k, _window_budget(request, max_model_len))
    if window_k < 2:
        return False

    extra = window_k - 1
    extra_blocks: dict[str, Any] = {}
    for req_id, request in windowed:
        # Extend the block allocation to cover the K-1 extra KV slots the
        # window will write.  On failure allocate_slots leaves state intact
        # and returns None; the window is refused for the whole batch.
        blocks = scheduler.kv_cache_manager.allocate_slots(
            request, extra, num_lookahead_tokens=0
        )
        extra_blocks[req_id] = blocks
        if blocks is None:
            logger.debug(
                "LZ window refused: request %s cannot host %d extra KV slots",
                req_id,
                extra,
            )

    # Communicate whatever allocations succeeded as look-ahead block rows so
    # the scheduler/worker block tables never diverge, even when the window
    # itself is refused below.
    for i, req_id in enumerate(cached_reqs.req_ids):
        blocks = extra_blocks.get(req_id)
        if blocks is None:
            continue
        new_ids = blocks.get_block_ids()
        if not any(new_ids):
            continue
        current = cached_reqs.new_block_ids[i]
        if current is None:
            cached_reqs.new_block_ids[i] = new_ids
        else:
            cached_reqs.new_block_ids[i] = tuple(
                old + new for old, new in zip(current, new_ids)
            )

    if any(blocks is None for blocks in extra_blocks.values()):
        return False

    # Commit the window: K scheduled tokens per request and K in-flight
    # output placeholders so the engine-side accounting closes exactly when
    # the runner reports the K sampled tokens.
    for req_id, request in windowed:
        request.num_output_placeholders += extra
        request.num_computed_tokens += extra
        scheduler_output.num_scheduled_tokens[req_id] = window_k
        scheduler_output.lz_window_steps[req_id] = window_k
    scheduler_output.total_num_scheduled_tokens += extra * len(windowed)
    logger.debug(
        "LZ window scheduled: K=%d requests=%d", window_k, len(windowed)
    )
    return True


def reconcile_window_shortfall(
    request: Any, planned_k: int, reported_tokens: int
) -> None:
    """Close the accounting gap when a windowed step produced < K tokens.

    ``AsyncScheduler._update_request_with_output`` already consumed
    ``reported_tokens`` placeholders and its ``cache_blocks`` call used
    ``num_computed_tokens - num_output_placeholders``, which equals the true
    confirmed length in both the window and fallback cases.  The remaining
    K - reported reservations (placeholders + optimistic computed tokens)
    belong to steps that never ran and are rolled back here.
    """
    shortfall = planned_k - reported_tokens
    if shortfall <= 0:
        return
    # Placeholders hold exactly K - reported in steady state; clamp only to
    # stay defensive against unexpected interim adjustments, and roll back
    # computed_tokens by the same amount so the (computed - placeholders)
    # confirmed-length invariant is preserved.
    shortfall = min(shortfall, max(request.num_output_placeholders, 0))
    request.num_output_placeholders -= shortfall
    request.num_computed_tokens -= shortfall

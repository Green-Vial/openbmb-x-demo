# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Strict, state-explicit batching for MiniCPM-o 4.5 Token2wav."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from vllm.logger import init_logger

logger = init_logger(__name__)

_SILENCE_TOKEN = 4218

# P15: NPU graph capture of the CFM decode loop. The estimator launch storm
# (~900 small kernels per CFM step, 3 steps per chunk) is pure host overhead
# on a shared accelerator. att-cache lengths grow per chunk and vary per
# request, so graphs are bucketed by padded cache length (upstream cosyvoice
# uses the same padding+mask trick for its CUDA-graph path).
_CFM_GRAPH_ATT_GRAIN = 128
_CFM_GRAPH_ATT_MAX = 768
_CFM_GRAPH_MAX_ENTRIES = 12
_MISSING = object()


class _CFMGraphBucket:
    """Static buffers + captured NPUGraph for one (batch, width, att_bucket)."""

    __slots__ = (
        "graph",
        "x_in",
        "mu_in",
        "spk_in",
        "cond_in",
        "cnn_in",
        "att_in",
        "mask",
        "out_x",
        "out_cnn",
        "out_att",
        "t_embs",
        "dts",
        "width",
        "att_bucket",
    )


class _HiFTGraphBucket:
    """Static buffers + captured NPUGraph for one (batch, mel_width)."""

    __slots__ = (
        "graph",
        "mel_in",
        "cache_in",
        "out_mag",
        "out_phase",
        "out_src",
        "width",
    )


def _autocast_disabled(device: torch.device):
    """Disable any enclosing autocast region on ``device``.

    ``torch.amp.autocast`` resolves the autocast dtype for ``device_type``
    while constructing the context, which raises on accelerators (e.g. Ascend
    NPU) that never registered autocast support. Degrade to a no-op there: an
    enclosing region can only exist on a device type torch already knows.
    """
    try:
        return torch.amp.autocast(device.type, enabled=False)
    except (RuntimeError, TypeError, ValueError):
        return nullcontext()


def tensor_signature(value: torch.Tensor) -> tuple[tuple[int, ...], str, str]:
    return tuple(value.shape), str(value.dtype), value.device.type


def state_shape_signature(state: BatchedToken2WavState) -> tuple[Any, ...]:
    flow = tuple((name, tensor_signature(state.flow_cache[name])) for name in sorted(state.flow_cache))
    hift = tuple((name, tensor_signature(state.hift_cache[name])) for name in sorted(state.hift_cache))
    return flow, hift


@dataclass(frozen=True)
class PromptFeatures:
    speech_tokens: torch.Tensor
    speaker_embedding: torch.Tensor
    mels: torch.Tensor


@dataclass(frozen=True)
class BatchedToken2WavState:
    flow_cache: dict[str, torch.Tensor]
    hift_cache: dict[str, torch.Tensor]


class BatchedToken2Wav(nn.Module):
    """Drive Token2wav's modules with dynamically-sized, request-owned caches.

    This class intentionally never calls ``Token2wav.stream`` or
    ``Token2wav.__call__``. The upstream object is used only as a one-time
    asset loader and prompt feature extractor.
    """

    def __init__(self, token2wav: Any, cfm_graph: bool = True):
        super().__init__()
        self._token2wav = token2wav
        self.flow = token2wav.flow
        self.hift = token2wav.hift
        # The upstream streaming path preallocates fixed-size CFM and DiT
        # caches. This adapter never calls that path and supplies dynamically
        # sized request-owned buffers to ``blocks_forward_chunk`` instead.
        decoder = self.flow.decoder
        for module in (decoder, decoder.estimator):
            for buffer_name in ("att_cache_buffer", "cnn_cache_buffer"):
                if buffer_name in module._buffers:
                    setattr(module, buffer_name, None)
        hift_parameter = next(self.hift.parameters(), None)
        if hift_parameter is not None and hift_parameter.device.type == "cuda":
            # Prime the CUDA state used by HiFT during backend construction.
            # Otherwise, the first live audio chunk can fail when async stages
            # share one GPU.
            device = hift_parameter.device
            dtype = hift_parameter.dtype
            mel_channels = int(self.hift.conv_pre.in_channels)
            with (
                torch.inference_mode(),
                torch.random.fork_rng(devices=[device]),
                _autocast_disabled(device),
            ):
                # 50 mel frames match the default first streamed vocoder chunk.
                speech, source = self.hift(
                    torch.zeros((1, mel_channels, 50), device=device, dtype=dtype),
                    torch.zeros((1, 1, 0), device=device, dtype=dtype),
                )
            torch.accelerator.synchronize(device)
            del speech, source
            torch.accelerator.empty_cache()
        self.float16 = bool(token2wav.float16)
        self.n_timesteps = int(token2wav.n_timesteps)
        self.mel_cache_len = int(token2wav.mel_cache_len)
        self.source_cache_len = int(token2wav.source_cache_len)
        self.register_buffer(
            "speech_window",
            token2wav.speech_window.detach().clone(),
            persistent=False,
        )
        self._prompt_features: dict[tuple[str, str], PromptFeatures] = {}
        # P15: NPU graph capture of the CFM decode loop (see _decode_cfm_graphed).
        self._cfm_graph_enabled = bool(cfm_graph)
        self._cfm_graphs: dict[tuple[int, int, int], _CFMGraphBucket | None] = {}
        self._cfm_graph_pool: Any = None
        self._cfm_graph_dead = False
        self._cfm_modal_width: int | None = None
        # P16: same treatment for the HiFT vocoder (everything up to the
        # final torch.istft, which syncs on NPU and stays eager). The
        # steady-state mel width is mel_cache_len + chunk frames; the first
        # chunk of a stream is narrower and gets its own graph.
        self._hift_graph_enabled = bool(cfm_graph)
        self._hift_graphs: dict[tuple[int, int], _HiFTGraphBucket | None] = {}
        self._hift_graph_dead = False
        if self._hift_graph_enabled:
            self._patch_hift_for_graph()

    def prepare_prompt(self, prompt_cache_id: str, prompt_wav: str) -> PromptFeatures:
        cache_key = (prompt_cache_id, prompt_wav)
        cached = self._prompt_features.get(cache_key)
        if cached is None:
            # The generation runner may wrap model.forward in bf16 autocast,
            # and vLLM constructs the model under a bf16 default dtype, while
            # S3Tokenizer prompt extraction uses fp32 convolution weights.
            previous_dtype = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.float32)
                with _autocast_disabled(self.speech_window.device):
                    values = self._token2wav._prepare_prompt(prompt_wav)
            finally:
                torch.set_default_dtype(previous_dtype)
            cached = PromptFeatures(
                speech_tokens=values[0],
                speaker_embedding=values[2],
                mels=values[3],
            )
            self._prompt_features[cache_key] = cached
        return cached

    def evict_prompt(self, prompt_cache_id: str, prompt_wav: str) -> None:
        """Release request-owned prompt features after stream completion."""
        self._prompt_features.pop((prompt_cache_id, prompt_wav), None)

    @staticmethod
    def _repeat_prompt(features: PromptFeatures, batch_size: int) -> tuple[torch.Tensor, ...]:
        return (
            features.speech_tokens.expand(batch_size, -1),
            features.speaker_embedding.expand(batch_size, -1),
            features.mels.expand(batch_size, -1, -1),
        )

    def _autocast(self, device: torch.device):
        if device.type != "cuda":
            return nullcontext()
        if not self.float16:
            return torch.amp.autocast("cuda", enabled=False)
        return torch.amp.autocast(
            "cuda",
            dtype=torch.float16,
        )

    def _pre_lookahead_len(self) -> int | None:
        """Right-context width of the encoder's pre-lookahead convolution.

        ``None`` when the encoder does not expose one, so callers keep working
        against encoder implementations without that layer.
        """
        layer = getattr(self.flow.encoder, "pre_lookahead_layer", None)
        width = getattr(layer, "pre_lookahead_len", None)
        return int(width) if width is not None else None

    def _encode_chunk(
        self,
        tokens: torch.Tensor,
        *,
        last_chunk: bool,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedded = self.flow.input_embedding(tokens)
        hidden, new_cnn, new_att = self.flow.encoder.forward_chunk(
            xs=embedded,
            last_chunk=last_chunk,
            cnn_cache=cnn_cache,
            att_cache=att_cache,
        )
        return self.flow.encoder_proj(hidden), new_cnn, new_att

    @staticmethod
    def _estimator_buffers(
        estimator: nn.Module,
        x: torch.Tensor,
        old_att: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = estimator.blocks
        depth = len(blocks)
        batch_size = int(x.shape[0])
        chunk_size = int(x.shape[2])
        old_att_len = int(old_att.shape[3]) if old_att is not None else 0
        block0 = blocks[0]
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        heads = int(block0.attn.num_heads)
        att_width = int(block0.attn.head_dim * 2)
        cnn = x.new_empty((depth, batch_size, cnn_channels, cnn_width))
        att = x.new_empty((depth, batch_size, heads, old_att_len + chunk_size, att_width))
        return cnn, att

    def _estimator_step(
        self,
        estimator: nn.Module,
        *,
        x: torch.Tensor,
        mu: torch.Tensor,
        time: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
        mask: torch.Tensor | None = None,
        time_embedding: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if time_embedding is None:
            time_embedding = estimator.t_embedder(time).unsqueeze(1)
        width = int(x.shape[-1])
        speaker_features = speakers.unsqueeze(-1).expand(-1, -1, width)
        estimator_input = torch.cat((x, mu, speaker_features, cond), dim=1)
        cnn_out, att_out = self._estimator_buffers(estimator, estimator_input, att_cache)
        old_cnn: Any = cnn_cache if cnn_cache is not None else [None] * len(estimator.blocks)
        old_att: Any = att_cache if att_cache is not None else [None] * len(estimator.blocks)
        result = estimator.blocks_forward_chunk(
            estimator_input,
            time_embedding,
            mask,
            old_cnn,
            old_att,
            cnn_out,
            att_out,
        )
        return result, cnn_out, att_out

    def _cfm_graph_constants(self, batch: int, device: torch.device, dtype: torch.dtype):
        """Per-model constants of the CFM loop, computed eagerly once.

        ``batch`` is the request batch N; the returned timestep embeddings
        carry the CFG-doubled batch 2N (matching the eager path, which feeds
        ``cat((time, time))`` through the embedder).

        The timestep embedder creates its frequency table on CPU and copies
        it to device, which is illegal inside a graph capture; the timeline
        and per-step dt are also pure constants. Precompute them (identical
        values to the eager path's per-chunk recomputation).
        """
        decoder = self.flow.decoder
        estimator = decoder.estimator
        timeline = torch.linspace(0, 1, self.n_timesteps + 1, device=device, dtype=dtype)
        timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
        time = timeline[0].expand(batch)
        t_embs: list[torch.Tensor] = []
        dts: list[torch.Tensor] = []
        dt = timeline[1] - timeline[0]
        for step in range(self.n_timesteps):
            t_embs.append(
                estimator.t_embedder(torch.cat((time, time), dim=0)).unsqueeze(1).detach().clone()
            )
            dts.append(dt.detach().clone())
            time = time + dt
            if step + 1 < self.n_timesteps:
                dt = timeline[step + 2] - time[0]
        return t_embs, dts

    def _run_cfm_loop(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_in: torch.Tensor,
        att_in: torch.Tensor,
        mask: torch.Tensor | None,
        t_embs: list[torch.Tensor],
        dts: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The CFM integration loop shared by the eager and captured paths.

        Mirrors the original _decode_cfm body, except that the timestep
        embeddings / dt scalars are precomputed constants (capture-safe) and
        the attention mask (``None`` in the eager path) can exclude padded
        att-cache columns in the graph path.
        """
        decoder = self.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        next_cnn: list[torch.Tensor] = []
        next_att: list[torch.Tensor] = []
        for step in range(self.n_timesteps):
            estimate, step_cnn, step_att = self._estimator_step(
                estimator,
                x=torch.cat((x, x), dim=0),
                mu=mu_cfg,
                time=torch.zeros(0, device=mu.device, dtype=mu.dtype),
                speakers=speakers_cfg,
                cond=cond_cfg,
                cnn_cache=cnn_in[step],
                att_cache=att_in[step],
                mask=mask,
                time_embedding=t_embs[step],
            )
            conditional, unconditional = estimate.split(batch_size, dim=0)
            velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
            x = x + dts[step] * velocity
            next_cnn.append(step_cnn)
            next_att.append(step_att)
        return x, torch.stack(next_cnn), torch.stack(next_att)

    def _capture_cfm_graph(
        self,
        batch: int,
        width: int,
        att_bucket: int,
        mu: torch.Tensor,
        speakers: torch.Tensor,
    ) -> _CFMGraphBucket | None:
        """Capture one NPUGraph for (batch, width, att_bucket).

        Returns None (and permanently disables the feature) when capture is
        unsupported on this stack; callers then keep the eager path.
        """
        if self._cfm_graph_dead:
            return None
        device = mu.device
        dtype = mu.dtype
        n_steps = self.n_timesteps
        decoder = self.flow.decoder
        block0 = decoder.estimator.blocks[0]
        depth = len(decoder.estimator.blocks)
        heads = int(block0.attn.num_heads)
        att_width = int(block0.attn.head_dim * 2)
        cnn_channels = int(block0.conv.in_channels + block0.conv.out_channels)
        cnn_width = int(block0.conv.block[1].causal_padding[0])
        try:
            b = _CFMGraphBucket()
            b.width = width
            b.att_bucket = att_bucket
            mel_ch = int(decoder.rand_noise.shape[1])
            b.x_in = torch.zeros((batch, mel_ch, width), dtype=dtype, device=device)
            b.mu_in = torch.zeros_like(b.x_in)
            b.cond_in = torch.zeros_like(b.x_in)
            b.spk_in = torch.zeros((batch, speakers.shape[1]), dtype=dtype, device=device)
            b.cnn_in = torch.zeros((n_steps, depth, 2 * batch, cnn_channels, cnn_width), dtype=dtype, device=device)
            b.att_in = torch.zeros(
                (n_steps, depth, 2 * batch, heads, att_bucket, att_width), dtype=dtype, device=device
            )
            b.mask = torch.ones((2 * batch, width, att_bucket + width), dtype=torch.bool, device=device)
            if self._cfm_graph_pool is None:
                self._cfm_graph_pool = torch.npu.graph_pool_handle()
            # NOTE: graphs here deliberately do NOT share one memory pool —
            # sharing a pool across separately-captured graphs made the HiFT
            # capture fail with "Not allow to synchronize captured-stream"
            # on this stack (each graph gets its own pool instead; memory
            # overhead is a few hundred MB per graph at these shapes).
            # Timestep embeddings / dt are per-model constants; compute them
            # eagerly (the t_embedder does a host->device copy internally,
            # which is illegal under capture). They are captured by address,
            # so they MUST stay alive for the lifetime of the graph — park
            # them on the bucket.
            b.t_embs, b.dts = self._cfm_graph_constants(batch, device, dtype)
            t_embs, dts = b.t_embs, b.dts
            # Warmup on the static buffers (side-effect free; allocator settles).
            with torch.no_grad():
                self._run_cfm_loop(
                    b.x_in, b.mu_in, b.spk_in, b.cond_in, b.cnn_in, b.att_in, b.mask, t_embs, dts
                )
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph, pool=self._cfm_graph_pool):
                b.out_x, b.out_cnn, b.out_att = self._run_cfm_loop(
                    b.x_in, b.mu_in, b.spk_in, b.cond_in, b.cnn_in, b.att_in, b.mask, t_embs, dts
                )
            b.graph = graph
            torch.npu.synchronize()
            logger.info(
                "CFM graph captured: batch=%d width=%d att_bucket=%d steps=%d",
                batch,
                width,
                att_bucket,
                n_steps,
            )
            return b
        except Exception:
            logger.exception("CFM graph capture failed; falling back to eager permanently")
            self._cfm_graph_dead = True
            return None

    def _decode_cfm_graphed(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        cnn_cache: torch.Tensor,
        att_cache: torch.Tensor,
        offset: int,
        end: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """Graph-replay counterpart of the eager _decode_cfm body."""
        batch = int(mu.shape[0])
        width = int(mu.shape[2])
        att_len = int(att_cache.shape[4])
        att_bucket = ((att_len + _CFM_GRAPH_ATT_GRAIN - 1) // _CFM_GRAPH_ATT_GRAIN) * _CFM_GRAPH_ATT_GRAIN
        if att_bucket > _CFM_GRAPH_ATT_MAX:
            return None
        # Only the modal (standard full-chunk) width is worth a graph; tail
        # chunks have per-request widths and stay eager.
        if self._cfm_modal_width is None:
            self._cfm_modal_width = width
        elif width != self._cfm_modal_width:
            return None
        key = (batch, width, att_bucket)
        bucket = self._cfm_graphs.get(key, _MISSING)
        if bucket is _MISSING:
            if len(self._cfm_graphs) >= _CFM_GRAPH_MAX_ENTRIES:
                self._cfm_graphs[key] = None
                return None
            bucket = self._capture_cfm_graph(batch, width, att_bucket, mu, speakers)
            self._cfm_graphs[key] = bucket
        if bucket is None:
            return None
        decoder = self.flow.decoder
        x = decoder.rand_noise[:, :, offset:end].expand(batch, -1, -1).clone()
        bucket.x_in.copy_(x)
        bucket.mu_in.copy_(mu)
        bucket.spk_in.copy_(speakers)
        bucket.cond_in.copy_(cond)
        bucket.cnn_in.copy_(cnn_cache)
        bucket.att_in[..., att_len:, :].zero_()
        bucket.att_in[..., :att_len, :].copy_(att_cache)
        bucket.mask.zero_()
        bucket.mask[:, :, : att_len + width].fill_(True)
        bucket.graph.replay()
        out_x = bucket.out_x.clone()
        out_cnn = bucket.out_cnn.clone()
        out_att = bucket.out_att[..., : att_len + width, :].clone()
        return out_x, out_cnn, out_att

    def _decode_cfm(
        self,
        mu: torch.Tensor,
        speakers: torch.Tensor,
        cond: torch.Tensor,
        *,
        cnn_cache: torch.Tensor | None,
        att_cache: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        decoder = self.flow.decoder
        estimator = decoder.estimator
        batch_size = int(mu.shape[0])
        offset = int(att_cache.shape[4]) if att_cache is not None else 0
        end = offset + int(mu.shape[2])
        if end > int(decoder.rand_noise.shape[2]):
            raise RuntimeError(
                "MiniCPMO45Code2WavBatchError "
                f'{{"reason":"noise_capacity","required":{end},'
                f'"available":{int(decoder.rand_noise.shape[2])}}}'
            )
        if (
            self._cfm_graph_enabled
            and not self._cfm_graph_dead
            and cnn_cache is not None
            and att_cache is not None
        ):
            graphed = self._decode_cfm_graphed(mu, speakers, cond, cnn_cache, att_cache, offset, end)
            if graphed is not None:
                return graphed
        x = decoder.rand_noise[:, :, offset:end].expand(batch_size, -1, -1).clone()
        timeline = torch.linspace(
            0,
            1,
            self.n_timesteps + 1,
            device=mu.device,
            dtype=mu.dtype,
        )
        timeline = 1 - torch.cos(timeline * 0.5 * torch.pi)
        time = timeline[0].expand(batch_size)
        mu_cfg = torch.cat((mu, torch.zeros_like(mu)), dim=0)
        speakers_cfg = torch.cat((speakers, torch.zeros_like(speakers)), dim=0)
        cond_cfg = torch.cat((cond, torch.zeros_like(cond)), dim=0)
        next_cnn: list[torch.Tensor] = []
        next_att: list[torch.Tensor] = []
        dt = timeline[1] - timeline[0]
        for step in range(self.n_timesteps):
            old_cnn = cnn_cache[step] if cnn_cache is not None else None
            old_att = att_cache[step] if att_cache is not None else None
            estimate, step_cnn, step_att = self._estimator_step(
                estimator,
                x=torch.cat((x, x), dim=0),
                mu=mu_cfg,
                time=torch.cat((time, time), dim=0),
                speakers=speakers_cfg,
                cond=cond_cfg,
                cnn_cache=old_cnn,
                att_cache=old_att,
            )
            conditional, unconditional = estimate.split(batch_size, dim=0)
            velocity = (1.0 + decoder.inference_cfg_rate) * conditional - decoder.inference_cfg_rate * unconditional
            x = x + dt * velocity
            time = time + dt
            if step + 1 < self.n_timesteps:
                dt = timeline[step + 2] - time[0]
            next_cnn.append(step_cnn)
            next_att.append(step_att)
        return x, torch.stack(next_cnn), torch.stack(next_att)

    @staticmethod
    def _split_flow_cache(cache: dict[str, torch.Tensor], batch_size: int) -> list[dict[str, torch.Tensor]]:
        result: list[dict[str, torch.Tensor]] = []
        for row in range(batch_size):
            result.append(
                {
                    "conformer_cnn_cache": cache["conformer_cnn_cache"][row : row + 1].detach().clone(),
                    "conformer_att_cache": cache["conformer_att_cache"][:, row : row + 1].detach().clone(),
                    "estimator_cnn_cache": torch.cat(
                        (
                            cache["estimator_cnn_cache"][:, :, row : row + 1],
                            cache["estimator_cnn_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                    "estimator_att_cache": torch.cat(
                        (
                            cache["estimator_att_cache"][:, :, row : row + 1],
                            cache["estimator_att_cache"][:, :, batch_size + row : batch_size + row + 1],
                        ),
                        dim=2,
                    ).detach(),
                }
            )
        return result

    @staticmethod
    def _stack_flow_cache(states: list[BatchedToken2WavState]) -> dict[str, torch.Tensor]:
        flows = [state.flow_cache for state in states]
        conditional_cnn = [flow["estimator_cnn_cache"][:, :, 0:1] for flow in flows]
        unconditional_cnn = [flow["estimator_cnn_cache"][:, :, 1:2] for flow in flows]
        conditional_att = [flow["estimator_att_cache"][:, :, 0:1] for flow in flows]
        unconditional_att = [flow["estimator_att_cache"][:, :, 1:2] for flow in flows]
        return {
            "conformer_cnn_cache": torch.cat([flow["conformer_cnn_cache"] for flow in flows], dim=0),
            "conformer_att_cache": torch.cat([flow["conformer_att_cache"] for flow in flows], dim=1),
            "estimator_cnn_cache": torch.cat((*conditional_cnn, *unconditional_cnn), dim=2),
            "estimator_att_cache": torch.cat((*conditional_att, *unconditional_att), dim=2),
        }

    def setup_batch(
        self,
        features: PromptFeatures,
        batch_size: int,
    ) -> list[BatchedToken2WavState]:
        prompt_tokens, speakers, prompt_mels = self._repeat_prompt(features, batch_size)
        lookahead_width = self._pre_lookahead_len()
        lookahead = prompt_tokens.new_full(
            (batch_size, 3 if lookahead_width is None else lookahead_width),
            _SILENCE_TOKEN,
        )
        with self._autocast(prompt_tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                torch.cat((prompt_tokens, lookahead), dim=1),
                last_chunk=False,
                cnn_cache=None,
                att_cache=None,
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            _, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                prompt_mels.transpose(1, 2).contiguous(),
                cnn_cache=None,
                att_cache=None,
            )
        flow_cache = {
            "conformer_cnn_cache": conformer_cnn,
            "conformer_att_cache": conformer_att,
            "estimator_cnn_cache": estimator_cnn,
            "estimator_att_cache": estimator_att,
        }
        split = self._split_flow_cache(flow_cache, batch_size)
        mel_channels = int(prompt_mels.shape[2])
        return [
            BatchedToken2WavState(
                flow_cache=row,
                hift_cache={
                    "mel": prompt_mels.new_zeros((1, mel_channels, 0)),
                    "source": prompt_mels.new_zeros((1, 1, 0)),
                    "speech": prompt_mels.new_zeros((1, 0)),
                },
            )
            for row in split
        ]

    @staticmethod
    def _fade_in_out(
        speech: torch.Tensor,
        previous: torch.Tensor,
        window: torch.Tensor,
    ) -> torch.Tensor:
        overlap = min(
            int(window.shape[0] // 2),
            int(speech.shape[-1]),
            int(previous.shape[-1]),
        )
        result = speech.clone()
        if overlap > 0:
            result[..., :overlap] = (
                result[..., :overlap] * window[:overlap] + previous[..., -overlap:] * window[-overlap:]
            )
        return result

    def _patch_hift_for_graph(self) -> None:
        """Remove the per-call host constants that break NPU graph capture.

        - hift.stft_window is created on CPU; _stft/_istft do
          ``window.to(x.device)`` every call, which is a real H2D copy the
          first time and a captured-graph killer. Move it once at setup.
        - SineGen2.forward rebuilds ``torch.FloatTensor([[range(...)]])`` on
          host and copies it to device every chunk; keep the same values in a
          device-resident buffer.
        """
        if getattr(self, "_hift_patched_for_graph", False):
            return
        try:
            self.hift.stft_window = self.hift.stft_window.to(self.speech_window.device)
            sine_gen = self.hift.m_source.l_sin_gen
            harmonics = torch.FloatTensor([[range(1, sine_gen.harmonic_num + 2)]]).to(
                self.speech_window.device
            )
        except AttributeError as exc:
            logger.exception("HiFT graph patch: unexpected module layout; disabling")
            self._hift_graph_dead = True
            raise AttributeError from exc

        # Closure over the instance (SineGen2.forward is plain python; the
        # harmonics row is the only per-call host constant it rebuilds).
        def _forward_graph_safe(f0: torch.Tensor):
            fn = torch.multiply(f0, harmonics)
            sine_waves = sine_gen._f02sine(fn) * sine_gen.sine_amp
            uv = sine_gen._f02uv(f0)
            noise_amp = uv * sine_gen.noise_std + (1 - uv) * sine_gen.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)
            return sine_waves * uv + noise, uv, noise

        sine_gen.forward = _forward_graph_safe
        self._hift_patched_for_graph = True

    def _capture_hift_graph(self, batch: int, width: int, mel: torch.Tensor) -> _HiFTGraphBucket | None:
        """Capture f0->source->cache-inject->stft->decode (no istft) for (batch, width)."""
        if self._hift_graph_dead:
            return None
        try:
            b = _HiFTGraphBucket()
            b.width = width
            device = mel.device
            dtype = mel.dtype
            mel_ch = int(mel.shape[1])
            src_len = int(self.source_cache_len)
            b.mel_in = torch.zeros((batch, mel_ch, width), dtype=dtype, device=device)
            b.cache_in = torch.zeros((batch, 1, src_len), dtype=dtype, device=device)
            hift = self.hift

            def _run() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
                f0 = hift.f0_predictor(b.mel_in)
                s_up = hift.f0_upsamp(f0[:, None]).transpose(1, 2)
                s_t, _, _ = hift.m_source(s_up)
                s = s_t.transpose(1, 2).clone()
                s[:, :, :src_len] = b.cache_in
                s_stft_real, s_stft_imag = hift._stft(s.squeeze(1))
                s_stft = torch.cat([s_stft_real, s_stft_imag], dim=1)
                x = hift.conv_pre(b.mel_in)
                for i in range(hift.num_upsamples):
                    x = F.leaky_relu(x, hift.lrelu_slope)
                    x = hift.ups[i](x)
                    if i == hift.num_upsamples - 1:
                        x = hift.reflection_pad(x)
                    si = hift.source_downs[i](s_stft)
                    si = hift.source_resblocks[i](si)
                    x = x + si
                    xs = None
                    for j in range(hift.num_kernels):
                        if xs is None:
                            xs = hift.resblocks[i * hift.num_kernels + j](x)
                        else:
                            xs = xs + hift.resblocks[i * hift.num_kernels + j](x)
                    x = xs / hift.num_kernels
                x = F.leaky_relu(x)
                x = hift.conv_post(x)
                magnitude = torch.exp(x[:, : hift.istft_params["n_fft"] // 2 + 1, :])
                phase = torch.sin(x[:, hift.istft_params["n_fft"] // 2 + 1 :, :])
                return magnitude, phase, s

            with torch.no_grad():
                _run()  # allocator warmup on the static buffers
            torch.npu.synchronize()
            graph = torch.npu.NPUGraph()
            with torch.npu.graph(graph):
                b.out_mag, b.out_phase, b.out_src = _run()
            b.graph = graph
            torch.npu.synchronize()
            logger.info("HiFT graph captured: batch=%d mel_width=%d", batch, width)
            return b
        except Exception:
            logger.exception("HiFT graph capture failed; falling back to eager permanently")
            self._hift_graph_dead = True
            return None

    def _hift_graphed(self, mel: torch.Tensor, old_source: torch.Tensor):
        """Graph-replay counterpart of ``self.hift(mel, old_source)`` minus istft.

        Returns (speech, source) with speech reconstructed by the eager istft
        (0.7ms; torch.istft syncs internally and cannot be captured), or None
        when the shape is not worth/graphable.
        """
        if self._hift_graph_dead:
            return None
        batch, width = int(mel.shape[0]), int(mel.shape[2])
        if int(mel.shape[1]) != 80:
            return None
        if int(old_source.shape[-1]) != int(self.source_cache_len):
            return None
        key = (batch, width)
        bucket = self._hift_graphs.get(key, _MISSING)
        if bucket is _MISSING:
            if len(self._hift_graphs) >= _CFM_GRAPH_MAX_ENTRIES:
                self._hift_graphs[key] = None
                return None
            bucket = self._capture_hift_graph(batch, width, mel)
            self._hift_graphs[key] = bucket
        if bucket is None:
            return None
        bucket.mel_in.copy_(mel)
        bucket.cache_in.copy_(old_source)
        bucket.graph.replay()
        magnitude = bucket.out_mag.clone()
        phase = bucket.out_phase.clone()
        source = bucket.out_src.clone()
        speech = self.hift._istft(magnitude, phase)
        # hift.decode() ends with clamp(±audio_limit); our capture stops
        # before istft, so apply the same clamp here or overshoot samples
        # poison the speech/source caches of later chunks.
        limit = float(getattr(self.hift, "audio_limit", 1.0))
        speech = torch.clamp(speech, -limit, limit)
        return speech, source

    def decode_batch(
        self,
        tokens: torch.Tensor,
        features: PromptFeatures,
        states: list[BatchedToken2WavState],
        *,
        last_chunk: bool,
        flush_encoder: bool = False,
    ) -> tuple[list[torch.Tensor], list[BatchedToken2WavState]]:
        batch_size = int(tokens.shape[0])
        if batch_size != len(states):
            raise ValueError(f"tokens batch {batch_size} != state batch {len(states)}")
        # The encoder's pre-lookahead convolution consumes ``pre_lookahead_len``
        # frames of right context and keeps no left cache, so a non-final chunk
        # must carry at least one full kernel. Only the final chunk is allowed
        # to be shorter: ``forward_chunk`` zero-pads it by the lookahead width.
        lookahead = self._pre_lookahead_len()
        if lookahead is not None and not last_chunk:
            num_frames = int(tokens.shape[1])
            if num_frames <= lookahead:
                raise RuntimeError(
                    "MiniCPMO45Code2WavBatchError "
                    f'{{"reason":"chunk_below_lookahead_window","frames":{num_frames},'
                    f'"minimum":{lookahead + 1}}}'
                )
        flow_cache = self._stack_flow_cache(states)
        speakers = features.speaker_embedding.expand(batch_size, -1)
        with self._autocast(tokens.device):
            hidden, conformer_cnn, conformer_att = self._encode_chunk(
                tokens,
                last_chunk=last_chunk or flush_encoder,
                cnn_cache=flow_cache["conformer_cnn_cache"],
                att_cache=flow_cache["conformer_att_cache"],
            )
            projected_speakers = self.flow.spk_embed_affine_layer(F.normalize(speakers, dim=1))
            cond = torch.zeros_like(hidden).transpose(1, 2).contiguous()
            chunk_mel, estimator_cnn, estimator_att = self._decode_cfm(
                hidden.transpose(1, 2).contiguous(),
                projected_speakers,
                cond,
                cnn_cache=flow_cache["estimator_cnn_cache"],
                att_cache=flow_cache["estimator_att_cache"],
            )

        prompt_len = int(features.mels.shape[1])
        if estimator_att.shape[4] > prompt_len + 100:
            estimator_att = torch.cat(
                (estimator_att[..., :prompt_len, :], estimator_att[..., -100:, :]),
                dim=4,
            )
        if conformer_att.shape[3] > prompt_len + 100:
            conformer_att = torch.cat(
                (conformer_att[..., :prompt_len, :], conformer_att[..., -100:, :]),
                dim=3,
            )
        new_flow = self._split_flow_cache(
            {
                "conformer_cnn_cache": conformer_cnn,
                "conformer_att_cache": conformer_att,
                "estimator_cnn_cache": estimator_cnn,
                "estimator_att_cache": estimator_att,
            },
            batch_size,
        )
        old_mel = torch.cat([state.hift_cache["mel"] for state in states], dim=0)
        old_source = torch.cat([state.hift_cache["source"] for state in states], dim=0)
        old_speech = torch.cat([state.hift_cache["speech"] for state in states], dim=0)
        mel = torch.cat((old_mel, chunk_mel), dim=2)
        # P16: HiFT vocoder minus istft as one NPUGraph replay; the eager
        # fallback is the unmodified self.hift call.
        graphed = self._hift_graphed(mel, old_source)
        if graphed is not None:
            speech, source = graphed
        else:
            speech, source = self.hift(mel, old_source)
        if old_speech.shape[-1] > 0:
            window = self.speech_window.to(device=speech.device, dtype=speech.dtype)
            speech = self._fade_in_out(speech, old_speech, window)
        next_hift = {
            "mel": mel[..., -self.mel_cache_len :].detach(),
            "source": source[..., -self.source_cache_len :].detach(),
            "speech": speech[..., -self.source_cache_len :].detach(),
        }
        emitted = speech if last_chunk else speech[..., : -self.source_cache_len]
        next_states = [
            BatchedToken2WavState(
                flow_cache=new_flow[row],
                hift_cache={name: value[row : row + 1].detach().clone() for name, value in next_hift.items()},
            )
            for row in range(batch_size)
        ]
        audios = [emitted[row].reshape(-1).to(dtype=torch.float32) for row in range(batch_size)]
        return audios, next_states

    @torch.inference_mode()
    def warmup(self, prompt_wav: str, batch_sizes=(1,), chunk_frames: int = 28) -> None:
        """Exercise the full prepare->setup->decode path before the first live request.

        The bench client's ``--num-warmups`` only warms the *client* side; the
        Code2Wav stage otherwise pays its entire cold-start cost on the first
        real request:

        - first ``prepare_prompt``: s3tokenizer ONNX session, torchaudio
          resample kernels, mel filterbank construction (CPU);
        - first ``setup_batch``: flow encoder pre-lookahead/upsample kernels
          plus a full n_timesteps CFM pass over the prompt mel;
        - first ``decode_batch``: conformer/DiT/HiFT kernel autotuning on the
          accelerator for the streaming chunk shapes.

        Uses the model's own reference audio so no request data is needed.
        State is fully discarded afterwards (states are local; the prompt
        feature cache entry is evicted), so live requests are unaffected.

        Args:
            prompt_wav: reference audio to drive the warmup.
            batch_sizes: batch dimensions to exercise (match max_num_seqs).
            chunk_frames: codec frames per streamed chunk (25 data + 3 left
                context = 28 by default).
        """
        device = self.speech_window.device
        cache_id = "__warmup__"
        logger.info("Code2Wav warmup: start (batch_sizes=%s, chunk_frames=%d)", list(batch_sizes), chunk_frames)
        try:
            features = self.prepare_prompt(cache_id, prompt_wav)
            for batch_size in batch_sizes:
                states = self.setup_batch(features, batch_size)
                tokens = torch.zeros((batch_size, chunk_frames), dtype=torch.long, device=device)
                # One mid-stream chunk and one final chunk cover both decode
                # branches (non-final keeps the tail cache; final emits it).
                self.decode_batch(tokens, features, states, last_chunk=False)
                self.decode_batch(tokens, features, states, last_chunk=True)
            torch.accelerator.synchronize(device)
            logger.info("Code2Wav warmup: done")
        except Exception:
            # Warmup must never take the stage down: log and let the first
            # real request pay the cold-start cost instead.
            logger.exception("Code2Wav warmup failed (continuing without it)")
        finally:
            self.evict_prompt(cache_id, prompt_wav)
            torch.accelerator.empty_cache()

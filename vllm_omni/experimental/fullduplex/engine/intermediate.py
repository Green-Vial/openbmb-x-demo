from __future__ import annotations

import os
from typing import TypedDict

import torch


class DuplexIntermediateBuffer(TypedDict, total=False):
    """Structured keys carried through ``model_intermediate_buffer``.

    The buffer remains a dict for scheduler and msgspec compatibility, but
    duplex-specific producers and consumers should use the helpers in this
    module instead of scattering nested string keys across serving, runner, and
    model code.
    """

    request_id: str
    global_request_id: list[str]
    prompt_token_ids: list[int]
    llm_output_token_ids: list[int]
    llm_output_text: list[str]
    stream_output: bool
    native_duplex: bool
    ids: dict[str, object]
    hidden_states: dict[str, object]
    codes: dict[str, object]
    meta: dict[str, object]
    duplex: dict[str, object]
    omni_payload: object
    waveform: object
    mel_spec: object


def build_duplex_intermediate_buffer(
    *,
    request_id: str,
    prompt_token_ids: list[int] | None = None,
    output_token_ids: list[int] | None = None,
    output_text: str | None = None,
    stream_output: bool = False,
    native_duplex: bool = False,
) -> DuplexIntermediateBuffer:
    buffer: DuplexIntermediateBuffer = {
        "global_request_id": [str(request_id)],
        "ids": {},
    }
    if prompt_token_ids is not None:
        prompt_ids = [int(token_id) for token_id in prompt_token_ids]
        buffer["prompt_token_ids"] = prompt_ids
        buffer["ids"]["prompt"] = prompt_ids
    if output_token_ids is not None:
        output_ids = [int(token_id) for token_id in output_token_ids]
        buffer["llm_output_token_ids"] = output_ids
        buffer["ids"]["output"] = output_ids
    if output_text is not None:
        buffer["llm_output_text"] = [output_text]
    if stream_output:
        buffer["stream_output"] = True
    if native_duplex:
        buffer["native_duplex"] = True
    return buffer


def set_ref_audio(buffer: dict[str, object], waveform: object, sample_rate_hz: int) -> None:
    buffer.setdefault("codes", {})["ref"] = waveform
    buffer.setdefault("meta", {})["ref_audio_sr"] = int(sample_rate_hz)


def set_tts_handoff(buffer: dict[str, object], token_ids: object | None, hidden_states: object | None) -> None:
    """Store the AR-to-TTS handoff used by the full-duplex stage bridge."""
    if token_ids is not None:
        buffer.setdefault("ids", {})["tts"] = token_ids
    if hidden_states is not None:
        buffer.setdefault("hidden_states", {})["tts"] = hidden_states


# P19 counterpart of the sender-side sentinel in
# ``stage_input_processors/minicpmo_4_5_omni.py``: rebuild the tensor from the
# raw bytes payload.  ``torch.frombuffer`` on a ``bytearray`` keeps the copy
# local to this process (the incoming bytes object belongs to the message).
# Floating payloads widen to float32, matching the historical nested-list
# path (``tolist`` -> ``as_tensor(dtype=float32)``) bit-for-bit.
_LZ_TS_KEY = "__lzy_ts__"


def normalize_handoff_tensor(value: object) -> object:
    if not (isinstance(value, dict) and value.get(_LZ_TS_KEY) is not None):
        return value
    shape = [int(d) for d in value.get("shape", [])]
    data = value.get("data")
    if isinstance(data, memoryview):
        data = data.tobytes()
    dtype_name = str(value[_LZ_TS_KEY])
    if dtype_name == "bfloat16":
        tensor = torch.frombuffer(bytearray(data), dtype=torch.int16).view(torch.bfloat16)
    else:
        tensor = torch.frombuffer(bytearray(data), dtype=getattr(torch, dtype_name))
    tensor = tensor.reshape(shape).clone()
    if tensor.is_floating_point() and tensor.dtype != torch.float32:
        tensor = tensor.to(torch.float32)
    return tensor


def get_tts_handoff(info: dict[str, object]) -> tuple[object | None, object | None]:
    """Read the canonical handoff, including the legacy flat aliases."""
    ids_info = info.get("ids")
    hidden_info = info.get("hidden_states")
    token_ids = ids_info.get("tts") if isinstance(ids_info, dict) else None
    hidden_states = hidden_info.get("tts") if isinstance(hidden_info, dict) else None
    if os.environ.get("OMNI_LZ_HANDOFF_LIST_LEGACY", "0") != "1":
        token_ids = normalize_handoff_tensor(token_ids)
        hidden_states = normalize_handoff_tensor(hidden_states)
    return (
        info.get("tts_token_ids") if token_ids is None else token_ids,
        info.get("tts_hidden_states") if hidden_states is None else hidden_states,
    )


def get_stream_request_key(info: dict[str, object]) -> str:
    key = info.get("global_request_id") or info.get("request_id") or info.get("_omni_req_id")
    if isinstance(key, (list, tuple)):
        key = key[0] if key else None
    if isinstance(key, bytes):
        key = key.decode("utf-8", errors="replace")
    if key is None:
        raise ValueError(
            "Duplex streaming handoff requires a stable request id; "
            "expected global_request_id, request_id, or _omni_req_id."
        )
    return str(key)

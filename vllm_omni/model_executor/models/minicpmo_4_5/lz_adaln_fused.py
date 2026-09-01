# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P28: fused AdaLN-Zero AscendC custom kernel (FusedAdaLN).

The CosyVoice2 DiT adaLN-Zero block repeats the same 512-wide layernorm +
affine pattern 3 times per block (48 times per ``forward_chunk`` pass):

    h = LayerNorm(x, eps=1e-6, elementwise_affine=False)   # x: [B, T, 512] fp32
    y = h * (1 + scale) + shift                            # shift/scale: [B, 1, 512]
    out = x + gate * y                                     # gate: [B, 1, 512]

The eager chain is 4-6 tiny kernel launches per site; the AscendC kernel
(``lz_adaln_ascendc/``) collapses each site into one launch:

- ``fused_adaln_gate(x, shift, scale, gate)`` ->
  ``x + gate * (LayerNorm(x) * (1 + scale) + shift)``
- ``fused_adaln_norm(x, shift, scale)`` ->
  ``LayerNorm(x) * (1 + scale) + shift`` (integration path; in
  ``DiTBlock.forward_chunk`` the gate applies to the *submodule output*, so
  the residual+gate segment must stay eager to preserve semantics)

Both entry points degrade transparently: if the kernel library cannot be
loaded/built, or the inputs fall outside the kernel's contract, the exact
eager composition runs instead and the failure is logged once. Performance
may drop, but results stay bit-compatible with the fallback chain.
"""

from __future__ import annotations

import subprocess
import threading
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

logger = init_logger(__name__)

_KERNEL_DIR = Path(__file__).resolve().parent / "lz_adaln_ascendc"
_SO_PATH = _KERNEL_DIR / "build" / "liblz_adaln_ops.so"
_BUILD_TIMEOUT_S = 600

# 0 = not tried, 1 = ready, 2 = permanently failed (eager fallback)
_state = 0
_state_lock = threading.Lock()
_warned = False


def _warn_once(msg: str) -> None:
    global _warned
    if not _warned:
        _warned = True
        logger.warning("lz_adaln_fused: %s (falling back to eager permanently)", msg)


def _build_library() -> bool:
    """Compile the AscendC kernel + torch extension via cmake/bisheng."""
    try:
        proc = subprocess.run(
            ["bash", str(_KERNEL_DIR / "build.sh")],
            capture_output=True,
            text=True,
            timeout=_BUILD_TIMEOUT_S,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        _warn_once(f"build script failed to run: {exc}")
        return False
    if proc.returncode != 0:
        tail = "\n".join((proc.stdout + proc.stderr).splitlines()[-15:])
        _warn_once(f"kernel build failed:\n{tail}")
        return False
    return True


def _smoke_check() -> bool:
    """One tiny numeric check before declaring the kernel ready."""
    device = torch.device("npu")
    x = torch.randn(2, 3, 512, device=device, dtype=torch.float32)
    shift = torch.randn(2, 1, 512, device=device, dtype=torch.float32)
    scale = torch.randn(2, 1, 512, device=device, dtype=torch.float32)
    gate = torch.randn(2, 1, 512, device=device, dtype=torch.float32)
    got = torch.ops.lz_npu.fused_adaln_gate(x, shift, scale, gate)
    ref = x + gate * (F.layer_norm(x, (512,), None, None, 1e-6) * (1 + scale) + shift)
    diff = float((got - ref).abs().max())
    if diff > 1e-4:
        _warn_once(f"smoke check mismatch (max abs diff {diff:.3e})")
        return False
    return True


def _ensure_loaded() -> bool:
    global _state
    if _state != 0:
        return _state == 1
    with _state_lock:
        if _state != 0:
            return _state == 1
        try:
            if not _SO_PATH.exists() and not _build_library():
                _state = 2
                return False
            torch.ops.load_library(str(_SO_PATH))
            if not _smoke_check():
                _state = 2
                return False
            _state = 1
            logger.info("lz_adaln_fused: AscendC fused AdaLN kernel loaded (%s)", _SO_PATH)
            return True
        except Exception as exc:  # noqa: BLE001 - any failure must degrade to eager
            _warn_once(f"kernel library load failed: {exc}")
            _state = 2
            return False


def is_available() -> bool:
    """Whether the AscendC fused AdaLN kernel is loaded and usable."""
    return _ensure_loaded()


def _kernel_ready(x: torch.Tensor, *params: torch.Tensor) -> bool:
    if x.dtype != torch.float32 or x.device.type != "npu" or x.dim() != 3:
        return False
    if any(p.dtype != torch.float32 or p.device != x.device for p in params):
        return False
    return _ensure_loaded()


def _prep_param(p: torch.Tensor, batch: int, ref_stride: int) -> torch.Tensor:
    """Normalize one shift/scale/gate tensor to the kernel's layout contract.

    The kernel reads every param row with ONE stride (taken from shift), so
    scale/gate must share shift's row stride exactly. The real DiT path
    (``adaLN_modulation(c).chunk(9)``) always does -- the chunks are views of
    one contiguous tensor with identical strides. Any other layout (e.g.
    freshly-built params mixed with views, which pytest exercises) is
    normalized here with a tiny contiguous copy; a mismatched stride left
    in place would make the kernel read wrong rows -- or out of bounds when
    the reference stride exceeds another param's storage.
    """
    if (
        p.stride(-1) != 1
        or p.numel() != batch * p.size(-1)
        or (batch > 1 and p.stride(0) != ref_stride)
    ):
        p = p.contiguous()
    if p.data_ptr() % 32:
        p = p.clone()  # contiguous() would return self; clone re-allocates aligned
    return p


def _eager_adaln_norm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return F.layer_norm(x, (x.shape[-1],), None, None, 1e-6) * (1 + scale) + shift


def _eager_adaln_gate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    return x + gate * _eager_adaln_norm(x, shift, scale)


def _run(op_name: str, eager, x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor,
         gate: Optional[torch.Tensor] = None) -> torch.Tensor:
    if not _kernel_ready(x, shift, scale, *([gate] if gate is not None else [])):
        return eager(x, shift, scale, gate) if gate is not None else eager(x, shift, scale)
    try:
        x = x.contiguous()
        batch = x.shape[0]
        ref_stride = shift.stride(0)
        shift = _prep_param(shift, batch, ref_stride)
        scale = _prep_param(scale, batch, ref_stride)
        if gate is not None:
            gate = _prep_param(gate, batch, ref_stride)
            return torch.ops.lz_npu.fused_adaln_gate(x, shift, scale, gate)
        return torch.ops.lz_npu.fused_adaln_norm(x, shift, scale)
    except Exception as exc:  # noqa: BLE001 - never break the serving path
        _warn_once(f"kernel launch failed ({op_name}): {exc}; using eager")
        global _state
        _state = 2
        return eager(x, shift, scale, gate) if gate is not None else eager(x, shift, scale)


def fused_adaln_gate(
    x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor, gate: torch.Tensor
) -> torch.Tensor:
    """out = x + gate * (LayerNorm(x, eps=1e-6) * (1 + scale) + shift)."""
    return _run("fused_adaln_gate", _eager_adaln_gate, x, shift, scale, gate)


def fused_adaln_norm(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """out = LayerNorm(x, eps=1e-6) * (1 + scale) + shift (no residual/gate)."""
    return _run("fused_adaln_norm", _eager_adaln_norm, x, shift, scale)

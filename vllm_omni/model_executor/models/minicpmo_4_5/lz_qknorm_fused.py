# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P28b: fused QK-norm AscendC custom kernel (FusedQkNorm).

The CosyVoice2 DiT attention (qk_norm=True) applies one affine
``nn.LayerNorm(head_dim, eps=1e-5)`` to Q and one to K before the attention
math — two tiny kernels per block. The AscendC kernel in
``lz_adaln_ascendc/op_kernel/lz_qk_norm_kernel.asc`` collapses both into one
launch and consumes the transposed ``to_heads`` views IN PLACE (explicit
B/H/T element strides, no ``contiguous()`` copy):

    q_out, k_out = fused_qk_norm(q, k, qw, qb, kw, kb, eps)

Degradation contract mirrors P28 (``lz_adaln_fused``): any load/build/smoke/
launch failure permanently switches this op to the exact eager composition
below — performance may drop, results stay bit-compatible.
"""

from __future__ import annotations

import threading

import torch
import torch.nn.functional as F
from vllm.logger import init_logger

from vllm_omni.model_executor.models.minicpmo_4_5.lz_adaln_fused import (
    _ensure_loaded as _load_shared_so,  # shared build/load path (same .so)
    _warn_once,
)

logger = init_logger(__name__)

# 0 = not tried, 1 = ready, 2 = permanently failed (eager fallback)
_state = 0
_state_lock = threading.Lock()


def _eager(q: torch.Tensor, k: torch.Tensor, qw: torch.Tensor, qb: torch.Tensor,
           kw: torch.Tensor, kb: torch.Tensor, eps: float):
    q_out = F.layer_norm(q, (q.shape[-1],), qw, qb, eps)
    k_out = F.layer_norm(k, (k.shape[-1],), kw, kb, eps)
    return q_out, k_out


def _smoke_check() -> bool:
    device = torch.device("npu")
    torch.manual_seed(0)
    # real to_heads layout: transposed views with contiguous head rows
    q = torch.randn(2, 4, 8, 64, device=device, dtype=torch.float32).transpose(1, 2)
    k = torch.randn(2, 4, 8, 64, device=device, dtype=torch.float32).transpose(1, 2)
    # explicit fp32: the serving process may run under a bf16 default dtype
    # (torch.set_default_dtype), which would silently poison these tensors.
    qw = torch.randn(64, device=device, dtype=torch.float32)
    qb = torch.randn(64, device=device, dtype=torch.float32)
    kw = torch.randn(64, device=device, dtype=torch.float32)
    kb = torch.randn(64, device=device, dtype=torch.float32)
    got = torch.ops.lz_npu.fused_qk_norm(q, k, qw, qb, kw, kb, 1e-5)
    ref = _eager(q, k, qw, qb, kw, kb, 1e-5)
    diff = max(float((g - r).abs().max()) for g, r in zip(got, ref))
    if diff > 1e-4:
        _warn_once(f"qk_norm smoke check mismatch (max abs diff {diff:.3e})")
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
            if not _load_shared_so():
                # shared library failed to load/build at all
                _state = 2
                return False
            if not _smoke_check():
                _state = 2
                return False
            _state = 1
            logger.info("lz_qknorm_fused: AscendC fused QK-norm kernel ready")
            return True
        except Exception as exc:  # noqa: BLE001 - any failure must degrade to eager
            _warn_once(f"qk_norm kernel load failed: {exc}")
            _state = 2
            return False


def is_available() -> bool:
    """Whether the fused QK-norm kernel is loaded and usable."""
    return _ensure_loaded()


def fused_qk_norm(q: torch.Tensor, k: torch.Tensor, qw: torch.Tensor, qb: torch.Tensor,
                  kw: torch.Tensor, kb: torch.Tensor, eps: float):
    """Both LayerNorms in one launch; eager fallback on any contract breach.

    The batched-heads kernel requires contiguous head rows (H stride ==
    headDim) — the real to_heads layout. Other layouts (e.g. freshly built
    contiguous tensors) take the eager pair for THIS call only: an unsupported
    input shape is not a kernel failure, so the kernel stays enabled.
    """
    if not is_available():
        return _eager(q, k, qw, qb, kw, kb, eps)
    if q.stride(1) != q.size(3) or k.stride(1) != k.size(3):
        return _eager(q, k, qw, qb, kw, kb, eps)
    try:
        out = torch.ops.lz_npu.fused_qk_norm(q, k, qw, qb, kw, kb, eps)
        return out[0], out[1]
    except Exception as exc:  # noqa: BLE001 - never break the serving path
        _warn_once(f"qk_norm kernel launch failed: {exc}; using eager")
        global _state
        _state = 2
        return _eager(q, k, qw, qb, kw, kb, eps)

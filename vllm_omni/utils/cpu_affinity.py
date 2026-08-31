# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""P26: per-process CPU affinity partitioning for single-host deployments.

vllm-omni runs an orchestrator event loop plus one engine core per stage on
the same host. All of these loops are CPU-bound around each decode step, so
they contend for cores and evict each other's caches; on a busy box that
host-side jitter shows up directly as RTF variance.

The mitigation is two plain syscalls per process:

* ``os.setpriority(os.PRIO_PROCESS, 0, -10)`` — raise the scheduling nice
  level so external load (compiles, monitoring agents) preempts the hot loop
  less often. Best-effort: without CAP_SYS_NICE / RLIMIT_NICE this fails and
  is silently ignored.
* ``os.sched_setaffinity(0, mask)`` — pin the process to its own group of
  cores so the orchestrator and the stage engine cores stop competing with
  each other and stop migrating between cores (cold caches).

The usable core set is taken from ``os.sched_getaffinity(0)`` (cgroup-safe:
it reports the cores this process is actually allowed to run on, unlike
``os.cpu_count()``) and split evenly into four contiguous groups:

* group 0 — orchestrator event loop
* groups 1/2/3 — stage 0/1/2 engine cores (``stage_id + 1``)

Everything here is defensive: any failure leaves the process unmodified, so
service startup never depends on it. Rollback: set ``OMNI_LZ_CPU_AFFINITY=0``
to disable the whole feature.

Note on Linux semantics: ``sched_setaffinity``/``setpriority`` with ``who=0``
apply to the calling *thread* (a freshly forked/spawned engine-core process
has exactly one thread at hook time, so process == thread there; the
orchestrator hook runs in its dedicated event-loop thread, leaving other
threads of the serving process free-range on purpose).
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)

# Below this many usable cores, quartering the set would over-pack each group
# (and small machines don't have a contention problem worth the risk).
_MIN_AFFINITY_CORES = 16

# orchestrator = 0, stage0/1/2 = 1/2/3.
_NUM_GROUPS = 4

# Rollback switch: any of these values disables the feature entirely.
_ENV_ROLLBACK = "OMNI_LZ_CPU_AFFINITY"
_DISABLE_VALUES = {"0", "false", "no", "off"}

# Nice value applied to the hot loops (best-effort; needs privileges).
_PRIORITY_BOOST = -10


def _affinity_disabled_by_env() -> bool:
    """True when the rollback env var turns the feature off."""
    return os.environ.get(_ENV_ROLLBACK, "").strip().lower() in _DISABLE_VALUES


def _partition_cores(allowed: list[int], groups: int = _NUM_GROUPS) -> list[list[int]]:
    """Split a sorted core-id list into ``groups`` contiguous, near-equal parts.

    Pure function so tests can exercise the partitioning without touching the
    scheduler. The remainder is distributed to the leading groups so group 0
    (orchestrator) is never the small one.
    """
    total = len(allowed)
    base, rem = divmod(total, groups)
    parts: list[list[int]] = []
    start = 0
    for index in range(groups):
        size = base + (1 if index < rem else 0)
        parts.append(list(allowed[start : start + size]))
        start += size
    return parts


def apply_stage_cpu_affinity(group: int) -> bool:
    """Pin the calling process/thread to core ``group``'s partition.

    ``group`` follows the orchestrator/stage layout: 0 = orchestrator,
    1/2/3 = stage 0/1/2 engine cores. Returns True only when the affinity
    binding succeeded; every failure path returns False without raising.
    """
    if _affinity_disabled_by_env():
        return False
    if not hasattr(os, "sched_getaffinity") or not hasattr(os, "sched_setaffinity"):
        return False  # non-Linux platform
    try:
        group = int(group)
    except (TypeError, ValueError):
        return False
    if not 0 <= group < _NUM_GROUPS:
        return False
    try:
        allowed = sorted(os.sched_getaffinity(0))
    except OSError:
        return False
    if len(allowed) < _MIN_AFFINITY_CORES:
        return False
    mask = set(_partition_cores(allowed)[group])
    if not mask:
        return False
    try:
        # Priority first: even if the binding below fails, the hot loop still
        # loses fewer timeslices to external load. Failure is expected without
        # privileges and must stay silent.
        os.setpriority(os.PRIO_PROCESS, 0, _PRIORITY_BOOST)
    except (OSError, OverflowError, ValueError):
        pass
    try:
        os.sched_setaffinity(0, mask)
    except OSError:
        logger.debug("CPU affinity: sched_setaffinity failed for group %d", group, exc_info=True)
        return False
    logger.info(
        "CPU affinity: pinned to group %d (%d cores of %d usable; nice=%d attempted)",
        group,
        len(mask),
        len(allowed),
        _PRIORITY_BOOST,
    )
    return True

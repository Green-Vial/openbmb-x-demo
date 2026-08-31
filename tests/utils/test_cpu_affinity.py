"""P26 unit tests: core partitioning and rollback-env parsing.

Pure-logic tests only: the real syscalls (sched_getaffinity/sched_setaffinity/
setpriority) are always monkeypatched so the test process itself is never
re-bound and never needs privileges.
"""

import os

import pytest

from vllm_omni.utils.cpu_affinity import (
    _MIN_AFFINITY_CORES,
    _NUM_GROUPS,
    _affinity_disabled_by_env,
    _partition_cores,
    apply_stage_cpu_affinity,
)


# ---- _partition_cores: grouping correctness ----


@pytest.mark.parametrize("total", [16, 17, 18, 31, 32, 64, 96, 128])
def test_partition_sizes_are_balanced(total):
    cores = list(range(total))
    parts = _partition_cores(cores)
    assert len(parts) == _NUM_GROUPS
    sizes = [len(p) for p in parts]
    # near-equal: every size differs by at most one from every other
    assert max(sizes) - min(sizes) <= 1
    assert sum(sizes) == total
    assert all(size > 0 for size in sizes)


@pytest.mark.parametrize("total", [16, 18, 31, 32, 64, 96])
def test_partition_is_contiguous_disjoint_and_ordered(total):
    cores = list(range(total))
    parts = _partition_cores(cores)
    merged = [core for part in parts for core in part]
    # concatenation restores the original order => parts are contiguous,
    # disjoint, and collectively exhaustive
    assert merged == cores
    for part in parts:
        assert part == sorted(part)


def test_partition_group0_is_orchestrator_sized():
    # 32 cores -> 4 x 8; group 0 must be the first contiguous block
    parts = _partition_cores(list(range(32)))
    assert parts[0] == list(range(0, 8))
    assert parts[1] == list(range(8, 16))
    assert parts[2] == list(range(16, 24))
    assert parts[3] == list(range(24, 32))


def test_partition_remainder_goes_to_leading_groups():
    # 18 cores -> remainder 2 distributed to the first two groups
    parts = _partition_cores(list(range(18)))
    assert [len(p) for p in parts] == [5, 5, 4, 4]
    assert parts[0] == list(range(0, 5))


def test_partition_custom_group_count():
    parts = _partition_cores(list(range(16)), groups=2)
    assert parts == [list(range(8)), list(range(8, 16))]


# ---- env parsing / rollback switch ----


@pytest.mark.parametrize("value", ["0", "false", "no", "off", "OFF", "False", " 0 "])
def test_env_disable_values(monkeypatch, value):
    monkeypatch.setenv("OMNI_LZ_CPU_AFFINITY", value)
    assert _affinity_disabled_by_env() is True


@pytest.mark.parametrize("value", ["1", "true", "yes", "on", "", "anything"])
def test_env_enabled_values(monkeypatch, value):
    monkeypatch.setenv("OMNI_LZ_CPU_AFFINITY", value)
    assert _affinity_disabled_by_env() is False


def test_env_unset_is_enabled(monkeypatch):
    monkeypatch.delenv("OMNI_LZ_CPU_AFFINITY", raising=False)
    assert _affinity_disabled_by_env() is False


# ---- apply_stage_cpu_affinity: defensive behavior with fake syscalls ----


@pytest.fixture()
def fake_syscalls(monkeypatch):
    """Replace the three syscalls with recorders; default to 32 usable cores."""
    state = {
        "allowed": list(range(32)),
        "affinity_calls": [],
        "priority_calls": [],
        "affinity_error": None,
        "priority_error": None,
    }

    def fake_getaffinity(pid):
        assert pid == 0
        return set(state["allowed"])

    def fake_setaffinity(pid, mask):
        assert pid == 0
        if state["affinity_error"] is not None:
            raise state["affinity_error"]
        state["affinity_calls"].append(set(mask))

    def fake_setpriority(which, who, prio):
        if state["priority_error"] is not None:
            raise state["priority_error"]
        state["priority_calls"].append((which, who, prio))

    monkeypatch.setattr(os, "sched_getaffinity", fake_getaffinity, raising=False)
    monkeypatch.setattr(os, "sched_setaffinity", fake_setaffinity, raising=False)
    monkeypatch.setattr(os, "setpriority", fake_setpriority, raising=False)
    monkeypatch.delenv("OMNI_LZ_CPU_AFFINITY", raising=False)
    return state


def test_apply_binds_expected_group(fake_syscalls):
    assert apply_stage_cpu_affinity(0) is True
    assert apply_stage_cpu_affinity(2) is True
    calls = fake_syscalls["affinity_calls"]
    assert calls == [set(range(0, 8)), set(range(16, 24))]


def test_apply_boosts_priority(fake_syscalls):
    assert apply_stage_cpu_affinity(1) is True
    which, who, prio = fake_syscalls["priority_calls"][0]
    assert prio == -10
    assert who == 0


def test_apply_priority_failure_is_silent(fake_syscalls):
    fake_syscalls["priority_error"] = PermissionError("no CAP_SYS_NICE")
    assert apply_stage_cpu_affinity(3) is True  # binding still succeeds
    assert fake_syscalls["affinity_calls"] == [set(range(24, 32))]
    assert fake_syscalls["priority_calls"] == []


def test_apply_affinity_failure_returns_false(fake_syscalls):
    fake_syscalls["affinity_error"] = OSError("sandbox")
    assert apply_stage_cpu_affinity(1) is False
    assert fake_syscalls["affinity_calls"] == []


@pytest.mark.parametrize("group", [-1, 4, 7, 100])
def test_apply_out_of_range_group_is_rejected(fake_syscalls, group):
    assert apply_stage_cpu_affinity(group) is False
    assert fake_syscalls["affinity_calls"] == []
    # nothing was touched at all
    assert fake_syscalls["priority_calls"] == []


def test_apply_below_min_cores_is_noop(fake_syscalls):
    fake_syscalls["allowed"] = list(range(_MIN_AFFINITY_CORES - 1))
    assert apply_stage_cpu_affinity(1) is False
    assert fake_syscalls["affinity_calls"] == []
    assert fake_syscalls["priority_calls"] == []


def test_apply_at_min_cores_succeeds(fake_syscalls):
    fake_syscalls["allowed"] = list(range(_MIN_AFFINITY_CORES))
    assert apply_stage_cpu_affinity(1) is True
    assert fake_syscalls["affinity_calls"] == [set(range(4, 8))]


def test_env_rollback_disables_everything(monkeypatch, fake_syscalls):
    monkeypatch.setenv("OMNI_LZ_CPU_AFFINITY", "0")
    assert apply_stage_cpu_affinity(1) is False
    assert fake_syscalls["affinity_calls"] == []
    assert fake_syscalls["priority_calls"] == []


def test_missing_sched_syscalls_is_noop(monkeypatch, fake_syscalls):
    monkeypatch.delattr(os, "sched_getaffinity", raising=False)
    monkeypatch.delattr(os, "sched_setaffinity", raising=False)
    assert apply_stage_cpu_affinity(1) is False
    assert fake_syscalls["priority_calls"] == []


def test_non_int_group_is_rejected(fake_syscalls):
    assert apply_stage_cpu_affinity("stage0") is False  # type: ignore[arg-type]
    assert fake_syscalls["affinity_calls"] == []

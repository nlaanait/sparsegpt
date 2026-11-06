"""Refuse large GPU/host runs that risk bricking unified-memory hosts.

GB10-class nodes may not report VRAM via nvidia-smi. Gate on host
MemAvailable / used instead, and refuse when projected use would cross a
hard ceiling (default 100 GiB).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MemorySnapshot:
    mem_total_bytes: int
    mem_available_bytes: int
    mem_free_bytes: int
    rss_bytes: int

    @property
    def mem_used_bytes(self) -> int:
        return self.mem_total_bytes - self.mem_available_bytes

    @property
    def used_gib(self) -> float:
        return self.mem_used_bytes / 1024**3

    @property
    def available_gib(self) -> float:
        return self.mem_available_bytes / 1024**3


def _read_proc_status_rss(pid: int = 0) -> int:
    path = Path("/proc/self/status" if pid <= 0 else f"/proc/{pid}/status")
    for line in path.read_text().splitlines():
        if line.startswith("VmRSS:"):
            return int(line.split()[1]) * 1024
    return 0


def _read_meminfo() -> tuple[int, int, int]:
    values: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        key, raw, *_ = line.replace(":", " ").split()
        if key in {"MemTotal", "MemAvailable", "MemFree"}:
            values[key] = int(raw) * 1024
    return values["MemTotal"], values["MemAvailable"], values["MemFree"]


def snapshot(pid: int = 0) -> MemorySnapshot:
    total, available, free = _read_meminfo()
    return MemorySnapshot(
        mem_total_bytes=total,
        mem_available_bytes=available,
        mem_free_bytes=free,
        rss_bytes=_read_proc_status_rss(pid),
    )


def format_snapshot(state: MemorySnapshot) -> str:
    return (
        f"used={state.used_gib:.1f}GiB "
        f"available={state.available_gib:.1f}GiB "
        f"rss={state.rss_bytes / 1024**3:.2f}GiB"
    )


def ensure_host_memory_headroom(
    *,
    max_used_gib: float = 90.0,
    min_available_gib: float = 16.0,
) -> MemorySnapshot:
    """Raise RuntimeError if host memory looks unsafe for another model load."""
    state = snapshot()
    if state.available_gib < min_available_gib:
        raise RuntimeError(
            f"MemAvailable {state.available_gib:.1f} GiB below minimum "
            f"{min_available_gib:.1f} GiB ({format_snapshot(state)})"
        )
    if state.used_gib > max_used_gib:
        raise RuntimeError(
            f"host used {state.used_gib:.1f} GiB exceeds max_used "
            f"{max_used_gib:.1f} GiB ({format_snapshot(state)})"
        )
    return state


def assert_safe_to_run(
    *,
    estimated_extra_bytes: int,
    hard_ceiling_bytes: int = 100 * 1024**3,
    min_available_bytes: int = 16 * 1024**3,
) -> MemorySnapshot:
    """Raise RuntimeError if starting another workload looks unsafe."""
    state = snapshot()
    if state.mem_available_bytes < min_available_bytes:
        raise RuntimeError(
            f"MemAvailable {state.mem_available_bytes / 1024**3:.1f} GiB "
            f"below minimum {min_available_bytes / 1024**3:.1f} GiB"
        )
    projected = state.mem_used_bytes + max(0, estimated_extra_bytes)
    if projected > hard_ceiling_bytes:
        raise RuntimeError(
            f"projected host use {projected / 1024**3:.1f} GiB exceeds "
            f"hard ceiling {hard_ceiling_bytes / 1024**3:.1f} GiB "
            f"(used={state.mem_used_bytes / 1024**3:.1f} GiB, "
            f"extra={estimated_extra_bytes / 1024**3:.1f} GiB)"
        )
    return state

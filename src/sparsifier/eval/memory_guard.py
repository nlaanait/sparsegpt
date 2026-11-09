"""Contain experiment processes on the unified-memory GB10.

Use together with an exclusive gflow allocation and a systemd MemoryMax.
Host pressure uses MemAvailable (including reclaimable cache), not CUDA's
unusable global utilization counters on this machine.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import time

GIB = 1024**3
# GSM8K 9B SparseGPT peaks ~17–21 GiB reserved; local geometry needs more headroom
# for atlas + Hessian work. GB10 mem_get_info free often understates reclaimable
# cache, so allocator caps must follow host-validated requests, not CUDA free alone.
MIN_ALLOCATOR_CAP_GIB = 16.0
MAX_ALLOCATOR_CAP_GIB = 80.0
HOST_STOP_GIB = 96.0
HOST_MARGIN_GIB = 8.0
CUDA_FREE_FRACTION = 0.9
CUDA_FREE_RETRIES = 6
CUDA_FREE_RETRY_SLEEP_S = 5.0


def host_memory() -> dict[str, float]:
    values = {}
    for line in Path("/proc/meminfo").read_text().splitlines():
        name, value = line.split(":", 1)
        values[name] = int(value.strip().split()[0]) * 1024
    return {"total_gib": values["MemTotal"] / GIB,
            "available_gib": values["MemAvailable"] / GIB,
            "used_gib": (values["MemTotal"] - values["MemAvailable"]) / GIB}


def check_budget(used: float, allocation: float, stop: float = HOST_STOP_GIB) -> None:
    if not 0 < allocation <= MAX_ALLOCATOR_CAP_GIB or not 0 < stop <= HOST_STOP_GIB:
        raise ValueError(
            f"GPU cap must be <= {MAX_ALLOCATOR_CAP_GIB:g} GiB and host stop <= {HOST_STOP_GIB:g} GiB"
        )
    # Unified memory: do not add CUDA reservation onto host used. Require the
    # requested fraction itself to fit under stop, and current host used to leave
    # the usual margin for the watchdog.
    if allocation + HOST_MARGIN_GIB > stop:
        raise RuntimeError(
            f"allocation exceeds host stop: allocation={allocation}, stop={stop} GiB"
        )
    if used + HOST_MARGIN_GIB > stop:
        raise RuntimeError(
            f"insufficient headroom: used={used:.2f}, allocation={allocation}, stop={stop} GiB"
        )


def choose_allocator_cap(
    cap_gib: float,
    free_bytes: int,
    *,
    available_gib: float | None = None,
    min_cap_gib: float = MIN_ALLOCATOR_CAP_GIB,
) -> float:
    """Pick the process CUDA fraction after host ``check_budget`` has passed.

    On GB10, honor the host-validated request. CUDA free may understate
    reclaimable cache and must not shrink a local 80 GiB request down to ~20 GiB.
    """
    if not 0 < cap_gib <= MAX_ALLOCATOR_CAP_GIB or not 0 < min_cap_gib <= cap_gib:
        raise ValueError("cap_gib/min_cap_gib out of range")
    if free_bytes < 0:
        raise ValueError("free_bytes must be nonnegative")
    cuda_cap = free_bytes / GIB * CUDA_FREE_FRACTION
    if available_gib is None:
        actual_cap = min(cap_gib, cuda_cap)
    else:
        host_cap = max(0.0, float(available_gib) - HOST_MARGIN_GIB)
        actual_cap = min(cap_gib, host_cap)
    if actual_cap < min_cap_gib:
        raise RuntimeError(
            f"less than {min_cap_gib:g} GiB of conservative CUDA headroom "
            f"(free={free_bytes / GIB:.2f} GiB, available={available_gib}, cap={actual_cap:.2f} GiB)"
        )
    return actual_cap


def configure_cuda(cap_gib: float = 56.) -> dict:
    import torch
    host = host_memory()
    check_budget(host["used_gib"], cap_gib)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required")
    free, total = torch.cuda.mem_get_info()
    # GB10: host MemAvailable is the exclusivity signal; set the allocator
    # ceiling from the host-validated request instead of dropping caches.
    last_error: RuntimeError | None = None
    actual_cap = 0.0
    for attempt in range(CUDA_FREE_RETRIES):
        host = host_memory()
        try:
            actual_cap = choose_allocator_cap(
                cap_gib, free, available_gib=host["available_gib"]
            )
            last_error = None
            break
        except RuntimeError as error:
            last_error = error
            if attempt + 1 == CUDA_FREE_RETRIES:
                break
            time.sleep(CUDA_FREE_RETRY_SLEEP_S)
            free, total = torch.cuda.mem_get_info()
    if last_error is not None:
        raise last_error
    torch.cuda.set_per_process_memory_fraction(actual_cap * GIB / total)
    return {"allocator_cap_gib": actual_cap, "requested_cap_gib": cap_gib, "cuda_total_gib": total / GIB,
            "host_before": host_memory(), "cuda_free_gib": free / GIB}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu-cap-gib", type=float, default=56.)
    parser.add_argument("--host-stop-gib", type=float, default=96.)
    parser.add_argument("--telemetry", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("missing child command")
    check_budget(host_memory()["used_gib"], args.gpu_cap_gib, args.host_stop_gib)
    args.telemetry.parent.mkdir(parents=True, exist_ok=True)
    child = subprocess.Popen(command, start_new_session=True)
    def stop_child(signum, _frame):
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
        raise SystemExit(128 + signum)
    signal.signal(signal.SIGTERM, stop_child)
    signal.signal(signal.SIGINT, stop_child)
    peak, last_log = 0., 0.
    try:
        with args.telemetry.open("a", buffering=1) as out:
            while child.poll() is None:
                state = host_memory()
                peak = max(peak, state["used_gib"])
                now = time.time()
                if now - last_log >= 5 or state["used_gib"] >= args.host_stop_gib:
                    out.write(json.dumps({"time": now, "child_pid": child.pid,
                                          "peak_host_used_gib": peak, **state}) + "\n")
                    last_log = now
                if state["used_gib"] >= args.host_stop_gib:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
                    raise RuntimeError(f"memory guard stopped experiment at {state['used_gib']:.2f} GiB")
                time.sleep(.1)
            raise SystemExit(child.returncode)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()


if __name__ == "__main__":
    main()

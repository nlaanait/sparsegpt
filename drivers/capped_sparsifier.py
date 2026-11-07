"""Run one ``sparsifier`` prune under a CUDA cap that stays clear of 110 GiB.

The cap is the smallest of 4 GiB, the host cushion under the brick, and
all but 1.5 GiB of the CUDA-reported free memory. The 27B sweep can keep
the GPU; this process stops rather than allocating into that reserve.
"""

from __future__ import annotations

from sparsifier.eval.live_stages import BRICK_GIB, CUSHION_GIB, read_host_memory_gib


def install_cap() -> float:
    import torch

    total_host, available = read_host_memory_gib()
    used = total_host - available
    room_gib = BRICK_GIB - CUSHION_GIB - used
    if room_gib < 3.0:
        raise SystemExit(
            f"refusing to prune: host is using {used:.1f} GiB, leaving {room_gib:.1f} GiB under the brick"
        )
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    reserve = int(1.5 * 1024**3)
    cap = min(int(4 * 1024**3), int(room_gib * 1024**3), max(0, int(free_bytes) - reserve))
    floor = 3 * 1024**3
    if cap < floor or total_bytes <= 0:
        raise SystemExit(
            f"refusing to prune: cap would be {cap / 1024**3:.2f} GiB "
            f"with {free_bytes / 1024**3:.2f} GiB CUDA-free"
        )
    torch.cuda.set_per_process_memory_fraction(cap / total_bytes)
    print(
        f"CUDA cap {cap / 1024**3:.2f} GiB (free {free_bytes / 1024**3:.2f}, host used {used:.1f})",
        flush=True,
    )
    return cap / 1024**3


def main() -> None:
    install_cap()
    from sparsifier.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()

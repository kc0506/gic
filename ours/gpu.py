# coding=utf-8
"""GPU selection — EXPLICIT, no import side effects.

Call pick_gpu() at the TOP of an entrypoint, BEFORE importing torch/taichi, so
CUDA_VISIBLE_DEVICES is set before any CUDA context is created. Library modules
must NOT call this at import time (that was the old roundtrip_sim2sim foot-gun:
importing it for a helper silently pinned a GPU).
"""
import os
import subprocess


def _least_used_gpu() -> str:
    """Index (as str) of the GPU with the least used memory; prefers fully idle."""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
        text=True,
    )
    best_idx, best_used = None, None
    for line in out.strip().splitlines():
        idx, used = [int(t) for t in line.split(",")]
        if best_used is None or used < best_used:
            best_idx, best_used = idx, used
    assert best_idx is not None, "no GPU found"
    return str(best_idx)


def pick_gpu() -> str:
    """Set CUDA_VISIBLE_DEVICES to the least-used GPU unless already pinned.

    Returns the chosen device string. No-op if the env var is already set, so an
    explicit CUDA_VISIBLE_DEVICES from the caller (colocation) always wins.
    """
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = _least_used_gpu()
        print(f"[gpu] using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")
    return os.environ["CUDA_VISIBLE_DEVICES"]

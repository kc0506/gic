# coding=utf-8
"""Feasibility profiler for multi-instance JOINT estimation (shared E, per-instance v0).

NOT a science run. Takes ONE 8-frame telephone GT and *pretends* it is N samples
(all fit the same GT) purely to measure the cost of doing N forward+backward per
optimizer step through a SINGLE reused simulator:

  per iter:  zero_grad
             for k in range(N):  set init_vel=v0_k; forward; backward  (grads ACCUMULATE)
             step

The hypothesis under test (from reading the gic memory model):
  - gic stores only `cuda_chunk_size+1` substep slots on GPU (ring buffer); the full
    BPTT trajectory is offloaded to CPU `cached_states` and popped during backward.
  - => interleaved forward->backward per instance keeps BOTH gpu fields (reused) and
    cpu cached_states (filled then drained each instance) at ~1x single-instance.
  - => only wall time scales ~Nx.

We measure GPU used MB (nvidia-smi, our PID), CPU peak RSS, s/iter, and len(cached_states)
right after each forward (peak) and after each backward (drain proof) to confirm/refute.

Usage (gic env, gic repo root):
  python profile_joint.py            # 4 instances, 8 frames, 64 substeps, 5 iters
  python profile_joint.py --n_instances 4 --n_iters 5
"""

# importing roundtrip_ours_scene -> roundtrip_sim2sim picks a free GPU BEFORE cuda touch
from roundtrip_ours_scene import (
    AnchoredEstimator,
    forward_bounded,
    load_our_scene,
)
from roundtrip_sim2sim import rollout_collect_surfaces, set_params

import json
import os
import resource
import subprocess
import time
from argparse import ArgumentParser, Namespace

import taichi as ti
import torch
from train_dynamic import backward as gic_backward
from simulator import Estimator


def gpu_used_mb() -> float:
    """MB of GPU memory used by THIS process on the visible device (nvidia-smi)."""
    pid = os.getpid()
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory",
             "--format=csv,noheader,nounits"], encoding="utf-8")
    except Exception:
        return -1.0
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 2 and parts[0] == str(pid):
            return float(parts[1])
    return 0.0


def cpu_peak_mb() -> float:
    """Peak resident set size of this process in MB (ru_maxrss is KB on Linux)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


if __name__ == "__main__":
    t0 = time.time()
    ap = ArgumentParser(description="joint (shared-E) feasibility profiler")
    ap.add_argument("--config", default="config/ours/telephone.json")
    ap.add_argument("--scene_cache", default="/tmp2/b10401006/ev-project/generative-phys/"
                    "outputs/_scene_cache/telephone_ds0.1_g32_k8.pt")
    ap.add_argument("--n_instances", default=4, type=int)
    ap.add_argument("--n_frames", default=8, type=int)
    ap.add_argument("--mpm_iter_cnt", default=64, type=int)
    ap.add_argument("--n_iters", default=5, type=int)
    ap.add_argument("--gt_logE", default=5.0, type=float)
    ap.add_argument("--gt_nu", default=0.3, type=float)
    ap.add_argument("--anchor_mass_scale", default=1e4, type=float)
    ap.add_argument("--inject_pvol", action="store_true", default=True)
    ap.add_argument("--out", default="output/profile_joint/run.json")
    args = ap.parse_args()

    with open(args.config) as f:
        phys_args = Namespace(**json.load(f)["physics"])
    phys_args.mpm_iter_cnt = args.mpm_iter_cnt
    phys_args.n_frames = args.n_frames

    # 4 xy-plane directions at |v0|=0.5 (the verified-identifiable subspace post-rot68).
    import math
    m = 0.5 / math.sqrt(2)
    dirs = [(m, m, 0.0), (-m, m, 0.0), (m, -m, 0.0), (-m, -m, 0.0)]
    v0_dirs = [dirs[k % len(dirs)] for k in range(args.n_instances)]

    xyz, anchor_mask = load_our_scene(args.scene_cache)
    print(f"[prof] N={xyz.shape[0]}, anchors={int(anchor_mask.sum())}, "
          f"{args.n_instances} instances x {args.n_frames}f x {args.mpm_iter_cnt} substeps")

    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
    gpu_after_init = gpu_used_mb()

    dummy_gts = [xyz.clone() for _ in range(args.n_frames)]
    est = AnchoredEstimator(phys_args, "float32", dummy_gts, surface_index=None,
                            init_vol=xyz, dynamic_scene=None, image_scale=1.0,
                            pipeline=None, image_op=None)
    est.set_anchor(anchor_mask, args.anchor_mass_scale)
    if args.inject_pvol:
        cpv = torch.load(args.scene_cache, map_location="cpu", weights_only=False)
        ghost = (cpv["disc"]["sim_xyzs"] == 0).all(dim=1)
        pvol = torch.from_numpy(cpv["disc"]["points_vol"]).float()[~ghost]
        est.set_pvol(pvol)

    # ---- GT once: gic self-rollout at GT params; reused by all N instances ----
    est.max_f = args.n_frames
    set_params(est, args.gt_logE, args.gt_nu, [0.0, -0.5, 0.0])
    gts = rollout_collect_surfaces(est)
    est.gts = gts
    est.load_gts(gts)
    gpu_after_gt = gpu_used_mb()
    print(f"[prof] GT generated; gpu used: init {gpu_after_init:.0f}MB -> after GT {gpu_after_gt:.0f}MB")

    # ---- shared E/nu + N per-instance v0 params, ONE optimizer ----
    import torch.nn as nn
    v0_params = [nn.Parameter(torch.tensor(v, device=est.device)) for v in v0_dirs]
    set_params(est, args.gt_logE, args.gt_nu, list(v0_dirs[0]))  # E/nu near GT
    opt = torch.optim.Adam([
        {"params": est.E, "lr": 0.05, "name": "Youngs modulus"},
        {"params": est.nu, "lr": 0.01, "name": "Poisson ratio"},
        {"params": v0_params, "lr": 0.025, "name": "velocity"},
    ])
    est.set_stage(Estimator.physical_params_stage)

    # ---- profile loop ----
    iter_times, per_inst_times = [], []
    peak_chunks, drain_ok = 0, True
    gpu_peak, cpu_peak = gpu_after_gt, cpu_peak_mb()
    for it in range(args.n_iters):
        t_iter = time.time()
        opt.zero_grad()
        joint_loss = 0.0
        for k in range(args.n_instances):
            t_inst = time.time()
            est.init_vel = v0_params[k]          # initialize() reads self.init_vel
            est.max_f = args.n_frames
            forward_bounded(est)
            n_after_fwd = len(est.simulator.cached_states)
            peak_chunks = max(peak_chunks, n_after_fwd)
            joint_loss += float(est.loss[None])
            gic_backward(est)                    # accumulates E/nu/v0_k grads
            if len(est.simulator.cached_states) != 0:
                drain_ok = False
            per_inst_times.append(time.time() - t_inst)
            gpu_peak = max(gpu_peak, gpu_used_mb())
        opt.step()
        cpu_peak = max(cpu_peak, cpu_peak_mb())
        dt = time.time() - t_iter
        iter_times.append(dt)
        print(f"[prof] iter {it}: {dt:.1f}s  joint_loss {joint_loss:.5f}  "
              f"E {10**float(est.E):.3e}  peak_chunks/inst {peak_chunks}  drain_ok {drain_ok}")

    mean_inst = sum(per_inst_times) / len(per_inst_times)
    mean_iter = sum(iter_times[1:]) / max(len(iter_times) - 1, 1)  # drop iter0 (compile)
    summary = {
        "n_instances": args.n_instances, "n_frames": args.n_frames,
        "mpm_iter_cnt": args.mpm_iter_cnt, "n_iters": args.n_iters,
        "n_particles": int(xyz.shape[0]),
        "s_per_instance_fwdbwd_mean": mean_inst,
        "s_per_joint_iter_mean_excl_iter0": mean_iter,
        "s_per_joint_iter_iter0_compile": iter_times[0],
        "implied_single_iter_s": mean_inst,            # 1 instance == 1 single-fit iter
        "joint_over_single_ratio": mean_iter / mean_inst if mean_inst else None,
        "gpu_used_mb_after_tiinit": gpu_after_init,
        "gpu_used_mb_after_gt": gpu_after_gt,
        "gpu_used_mb_peak": gpu_peak,
        "cpu_peak_rss_mb": cpu_peak,
        "cached_states_peak_per_instance": peak_chunks,
        "cached_states_drain_ok": drain_ok,
        "total_wall_s": time.time() - t0,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=2)
    print("\n[prof] ===== SUMMARY =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"[prof] wrote {args.out}")

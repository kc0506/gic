# coding=utf-8
"""Sim-to-sim roundtrip robustness test for GIC.

Phase 1 of the "find an anchor that works" plan:
  1. Generate GT observations with GIC's *own* MPM simulator under chosen
     GT params (E*, nu*, v0*), collecting per-frame surface point clouds —
     the exact observation format GIC's estimator consumes.
  2. Re-fit from a given init with GIC's own two-stage loop
     (velocity stage -> physical params stage), geometry loss only.

Everything else (scene particles, substeps, CFL policy, loss, optimizer,
lr schedule) is GIC's original harness, untouched.

Note: alpha/img losses are disabled because the dataset's masks correspond
to the dataset's real GT params, not our synthetic ones.

Usage (gic conda env, from gic repo root):
  python roundtrip_sim2sim.py -c config/pacnerf/torus.json \
      -s data/pacnerf/torus -m output/pacnerf/torus \
      --gt_logE 6.0 --gt_nu 0.3 --init_logE 5.0 --init_nu 0.1 \
      --tag pilot
"""

import os
import subprocess


def _pick_free_gpu() -> str:
    """Pick the GPU with the least used memory (prefer fully idle)."""
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


if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = _pick_free_gpu()
    print(f"[roundtrip] using GPU {os.environ['CUDA_VISIBLE_DEVICES']}")

import json
import time
from argparse import ArgumentParser

import numpy as np
import taichi as ti
import torch

from arguments import ModelParams, OptimizationParams, PipelineParams, get_combined_args
from simulator import Estimator
from simulator.estimator import constraint_inv
from train_dynamic import prepare_gt, train
from utils.general_utils import safe_state
from utils.system_utils import draw_curve


def load_or_build_static_cache(model, gs_args, pipeline, phys_args, cache_path: str) -> dict:
    """Return {'vol': (N,3) f32, 'vol_surface': (S,) i64, 'gts_recon': list[(Sf,3)], ...} on CPU.

    Runs GIC's prepare_gt (GS + deform model density filling) once and caches
    the static volume + surface index; later runs skip the GS machinery.
    """
    if os.path.exists(cache_path):
        print(f"[roundtrip] loading static cache {cache_path}")
        return torch.load(cache_path, map_location="cpu")
    gts, vol, vol_densities, grid_size, vol_surface, _cam_info = prepare_gt(
        model.extract(gs_args), gs_args.iteration, pipeline.extract(gs_args), phys_args
    )
    cache = {
        "vol": vol.cpu(),                      # (N, 3) initial particle positions
        "vol_densities": vol_densities.cpu(),  # (N,)
        "grid_size": grid_size.cpu(),          # (1,)
        "vol_surface": vol_surface.cpu(),      # (S,) indices into vol
        "gts_recon": [g.cpu() for g in gts],   # list of (Sf, 3) reconstructed surfaces
    }
    torch.save(cache, cache_path)
    print(f"[roundtrip] saved static cache {cache_path}")
    return cache


def rollout_collect_surfaces(estimator: Estimator) -> list:
    """Forward-roll the simulator with current params; return per-frame surface points.

    Returns list of (S, 3) float32 cuda tensors, one per frame (S = sim surface count).
    Replicates train_dynamic.forward()'s CFL-halving retry loop.
    """
    estimator.set_stage(Estimator.physical_params_stage)
    saved_geo_loss = estimator.geo_loss
    estimator.geo_loss = False  # pure rollout: no matching, no loss
    dt = estimator.simulator.dt_ori[None]
    while True:
        surfaces = []
        for idx in range(estimator.max_f):
            if idx == 0:
                estimator.initialize()
                estimator.simulator.set_dt(dt)
            estimator.forward(idx, img_backward=False)
            pts, _color = estimator.get_surface_vertics(idx)  # (S, 3) f32 numpy
            surfaces.append(torch.from_numpy(pts).to(estimator.device))
        if not estimator.succeed():
            dt /= 2
            print(f"[roundtrip] gen: cfl dissatisfied, shrink dt to {dt}")
        else:
            break
    estimator.geo_loss = saved_geo_loss
    return surfaces


def set_params(estimator: Estimator, logE: float, nu: float, vel: list) -> None:
    """Overwrite estimator's learnable params in place (E in log10 space)."""
    dev = estimator.device
    estimator.E.data = torch.tensor(logE, device=dev)
    estimator.nu.data = constraint_inv(torch.tensor(nu, device=dev), estimator.nu_bound)
    estimator.init_vel.data = torch.tensor(vel, device=dev)


def save_rollout_gif(surfaces: list, path: str, max_pts: int = 2000) -> None:
    """Save a 3D scatter GIF of the rollout for human inspection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    pts0 = surfaces[0].cpu().numpy()
    sel = np.random.default_rng(0).permutation(pts0.shape[0])[:max_pts]
    all_pts = np.stack([s.cpu().numpy()[sel] for s in surfaces])  # (F, M, 3)
    mins, maxs = all_pts.reshape(-1, 3).min(0), all_pts.reshape(-1, 3).max(0)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")

    def update(f: int):
        ax.cla()
        p = all_pts[f]
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=1)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[2], maxs[2])
        ax.set_zlim(mins[1], maxs[1])
        ax.set_title(f"frame {f}")

    anim = FuncAnimation(fig, update, frames=len(surfaces))
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def plot_param_traj(e_s: list, gt_E: float, gt_nu: float, path: str) -> None:
    """Plot E (log y) and nu trajectories vs GT lines over fit iterations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Es = [d["Youngs modulus"] for d in e_s if "Youngs modulus" in d]
    nus = [d["Poisson ratio"] for d in e_s if "Poisson ratio" in d]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(Es, label="E (estimated)")
    axes[0].axhline(gt_E, color="r", ls="--", label="E (GT)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iter")
    axes[0].legend()
    axes[1].plot(nus, label="nu (estimated)")
    axes[1].axhline(gt_nu, color="r", ls="--", label="nu (GT)")
    axes[1].set_xlabel("iter")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="GIC sim-to-sim roundtrip")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    op = OptimizationParams(parser)
    parser.add_argument("--config_file", default="config/pacnerf/torus.json", type=str)
    parser.add_argument("--gt_logE", required=True, type=float, help="GT log10(E)")
    parser.add_argument("--gt_nu", default=0.3, type=float)
    parser.add_argument("--gt_vel", nargs=3, default=[-0.2, -0.5, -0.2], type=float)
    parser.add_argument("--init_logE", required=True, type=float, help="fit init log10(E)")
    parser.add_argument("--init_nu", default=0.1, type=float)
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--iter_cnt", default=None, type=int, help="override phys-stage iters")
    parser.add_argument("--vel_iter_cnt", default=None, type=int, help="override vel-stage iters")
    gs_args, phys_args = get_combined_args(parser)
    safe_state(gs_args.quiet)
    if gs_args.iter_cnt is not None:
        phys_args.iter_cnt = gs_args.iter_cnt
    if gs_args.vel_iter_cnt is not None:
        phys_args.vel_iter_cnt = gs_args.vel_iter_cnt

    # roundtrip overrides: geometry loss only, fit init from CLI
    phys_args.img_loss = False
    phys_args.w_img = 0.0
    phys_args.w_alp = 0.0
    phys_args.geo_loss = True
    phys_args.init_E = gs_args.init_logE
    phys_args.init_nu = gs_args.init_nu
    fit_init_vel = list(phys_args.init_vel)

    out_dir = os.path.join(gs_args.model_path, "roundtrip", gs_args.tag)
    os.makedirs(out_dir, exist_ok=True)

    cache_path = os.path.join(gs_args.model_path, "roundtrip_static_cache.pt")
    cache = load_or_build_static_cache(model, gs_args, pipeline, phys_args, cache_path)
    torch.cuda.empty_cache()

    vol = cache["vol"].cuda()                  # (N, 3)
    vol_surface = cache["vol_surface"].cuda()  # (S,)
    n_frames = min(len(cache["gts_recon"]), getattr(phys_args, "n_frames", len(cache["gts_recon"])))
    S = vol_surface.shape[0]
    print(f"[roundtrip] N={vol.shape[0]} particles, S={S} surface, F={n_frames} frames")

    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)

    # dummy gts only size the taichi buffers; replaced by synthetic GT below
    dummy_gts = [vol[vol_surface].clone() for _ in range(n_frames)]
    estimator = Estimator(
        phys_args, "float32", dummy_gts, surface_index=vol_surface, init_vol=vol,
        dynamic_scene=None, image_scale=1.0,
        pipeline=pipeline.extract(gs_args), image_op=op.extract(gs_args),
    )

    # ---- 1. generate synthetic GT with GIC's own simulator ----
    set_params(estimator, gs_args.gt_logE, gs_args.gt_nu, gs_args.gt_vel)
    gen_t0 = time.time()
    gts_synth = rollout_collect_surfaces(estimator)
    print(f"[roundtrip] GT rollout done in {time.time() - gen_t0:.1f}s")
    disp = (gts_synth[-1] - gts_synth[0]).norm(dim=-1)  # (S,) total displacement
    print(f"[roundtrip] GT motion: mean disp {disp.mean():.4f}, max {disp.max():.4f}")
    save_rollout_gif(gts_synth, os.path.join(out_dir, "gt_rollout.gif"))

    # ---- 2. swap in synthetic GT, reset params to init, fit ----
    estimator.gts = gts_synth
    estimator.load_gts(gts_synth)
    set_params(estimator, gs_args.init_logE, gs_args.init_nu, fit_init_vel)

    estimator.set_stage(Estimator.velocity_stage)
    losses_vel, e_s_vel = train(estimator, phys_args, phys_args.vel_estimation_frames)
    torch.cuda.empty_cache()

    estimator.set_stage(Estimator.physical_params_stage)
    losses_phys, e_s_phys = train(estimator, phys_args, n_frames)

    # ---- 3. export ----
    min_idx = losses_phys.index(min(losses_phys))
    best = e_s_phys[min_idx]
    gt_E = 10.0 ** gs_args.gt_logE
    rel_err_E = abs(best["Youngs modulus"] - gt_E) / gt_E
    result = {
        "gt": {"E": gt_E, "logE": gs_args.gt_logE, "nu": gs_args.gt_nu, "vel": gs_args.gt_vel},
        "init": {"logE": gs_args.init_logE, "nu": gs_args.init_nu, "vel": fit_init_vel},
        "best": {**best, "iter": min_idx, "loss": losses_phys[min_idx]},
        "final": e_s_phys[-1],
        "vel_estimated": estimator.init_vel.detach().cpu().numpy().tolist(),
        "rel_err_E": rel_err_E,
        "losses_vel": [float(l) for l in losses_vel],
        "losses_phys": [float(l) for l in losses_phys],
        "E_traj": [d.get("Youngs modulus") for d in e_s_phys],
        "nu_traj": [d.get("Poisson ratio") for d in e_s_phys],
        "gt_motion_mean_disp": float(disp.mean()),
        "wall_time_s": time.time() - start_time,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    draw_curve(losses_phys, out_dir, name="loss_phys")
    draw_curve(losses_vel, out_dir, name="loss_vel")
    plot_param_traj(e_s_phys, gt_E, gs_args.gt_nu, os.path.join(out_dir, "param_traj.png"))
    print(f"[roundtrip] DONE tag={gs_args.tag} GT E={gt_E:.3g} -> best E={best['Youngs modulus']:.3g} "
          f"(rel err {rel_err_E:.2%}), nu GT={gs_args.gt_nu} -> {best['Poisson ratio']:.3f}, "
          f"wall {time.time() - start_time:.0f}s")

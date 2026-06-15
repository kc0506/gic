#!/usr/bin/env python
# coding=utf-8
"""fit_traj_Efield: recover a per-particle log10-E FIELD from a trajectory, v0 fixed.

The E analogue of fit_traj_v0. GT material is a field (uniform / ramp along the
cord / circular branch-dependent on the two strands); v0 is fixed to a known
scalar direction (--obs). E is observable only where the motion produces STRAIN,
so error is reported weighted by a k-NN strain proxy and a recovered-vs-GT
scatter, not averaged away.

The E-field-only slice of the old efield_fit.py (joint v0+E -> fit_traj_joint).
Uses train_efield_grid (custom best-GRID restore loop), NOT train_ours.

Usage (gic env, gic repo root) -- run dir auto-placed at output/fit_traj_Efield/<NN>:
  python fit_traj_Efield.py --scene.cache <cache.pt> --scene.rot-z-deg 67.6 \
      --scene.inject-pvol --gt-kind ramp --gt-ramp 4.5 5.5 --run-label ramp
  python fit_traj_Efield.py --scene.cache <cache.pt> --scene.rot-z-deg 67.6 \
      --scene.inject-pvol --gt-kind circular --efield.res 16x16x16 --run-label circ
"""
from ours.gpu import pick_gpu

pick_gpu()  # pick a free GPU before torch/taichi create a CUDA context

import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Literal, Optional, Tuple

import numpy as np
import torch
import tyro

from utils.system_utils import draw_curve

from ours.config import FieldCfg, FramesCfg, GTCfg, SceneCfg, TrainCfg, make_phys_args
from ours.fields import (EVoxelField, eval_Egrid_at, fill_circular_grid, strain_proxy)
from ours.rundir import RunDir
from ours.scene import build_anchored_scene, rollout_collect_surfaces, set_params
from ours.train import train_efield_grid
from ours.viz import (plot_E_gt_vs_pred, plot_E_grid_nodes, plot_E_projections,
                      plot_E_z_heatmap, save_overlay_gif)

VDIR = {"xp": [0.5, 0, 0], "xm": [-0.5, 0, 0], "yp": [0, 0.5, 0], "ym": [0, -0.5, 0]}


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=GTCfg)
    train: TrainCfg = field(default_factory=lambda: TrainCfg(patience=24, min_iters=30, fine_lr=0.02))
    efield: FieldCfg = field(default_factory=FieldCfg)
    """E voxel-field options (res default 4x4x16, tv 1e-3)"""
    frames: FramesCfg = field(default_factory=FramesCfg)
    init_logE: float = 4.0
    """uniform init of the fit E-field (deliberately wrong)"""
    gt_kind: Literal["uniform", "ramp", "circular"] = "uniform"
    gt_ramp: Tuple[float, float] = (4.5, 5.5)
    """ramp/circular GT: log10 E from lo (anchor end) to hi (tip / arc end)"""
    obs: str = "ym"
    """fixed v0 direction (VDIR key xp/xm/yp/ym, or 'x,y,z')"""
    gt_frames: int = 14
    """GT rollout length (fit window = frames.n_frames, default 8)"""
    E_lr: float = 0.2
    estop_tol: float = 0.003
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config, rd: RunDir) -> None:
    t0 = time.time()
    v0 = VDIR[cfg.obs] if cfg.obs in VDIR else [float(x) for x in cfg.obs.split(",")]
    res = (tuple(int(p) for p in cfg.efield.res.lower().split("x"))
           if "x" in cfg.efield.res else (int(cfg.efield.res),) * 3)
    phys_args = make_phys_args(cfg.scene, cfg.train, cfg.frames, cfg.init_logE, cfg.gt.nu)
    phys_args.n_frames = cfg.gt_frames
    n_frames = 8 if cfg.frames.n_frames is None else cfg.frames.n_frames

    scene = build_anchored_scene(cfg.scene.cache, cfg.scene.rot_z_deg, phys_args,
                                 cfg.scene.anchor_mass_scale, cfg.gt_frames,
                                 inject_pvol=cfg.scene.inject_pvol,
                                 ti_mem_frac=cfg.scene.ti_mem_frac)
    est, xyz, free = scene["est"], scene["xyz"], scene["free"]
    aabb, z_lo, z_hi, flip_z = scene["aabb"], scene["z_lo"], scene["z_hi"], scene["flip_z"]

    # ---- GT E-field + rollout (v0 fixed scalar from obs) ----
    gt_field = EVoxelField(aabb.cpu(), res=res)
    if cfg.gt_kind == "uniform":
        gt_field.set_uniform_(cfg.gt.logE)
    elif cfg.gt_kind == "ramp":
        gt_field.fill_ramp_(cfg.gt_ramp[0], cfg.gt_ramp[1], z_lo, z_hi, flip_z)
    else:  # circular: branch-dependent E bucketed into the grid (grid-representable)
        fill_circular_grid(gt_field, xyz.cpu(), free.cpu(), z_lo, z_hi,
                           cfg.gt_ramp[0], cfg.gt_ramp[1])
    est.set_E_field(gt_field, xyz, lr=0.0)
    set_params(est, cfg.gt.logE, cfg.gt.nu, v0)
    gt_logE_p = gt_field(xyz).detach().cpu()
    est.max_f = cfg.gt_frames
    gts = rollout_collect_surfaces(est)
    strain = strain_proxy(gts[0], gts[-1]).cpu()
    print(f"[Efield] GT {cfg.gt_kind} logE [{gt_logE_p[free.cpu()].min():.2f},"
          f"{gt_logE_p[free.cpu()].max():.2f}] | free strain mean "
          f"{strain[free.cpu()].mean():.4f} max {strain[free.cpu()].max():.4f}")

    # ---- fit E-field (uniform wrong init, v0 fixed) ----
    fit_field = EVoxelField(aabb.cpu(), res=res)
    fit_field.set_uniform_(cfg.init_logE)
    n_starved = fit_field.freeze_starved_(xyz[free].cpu(), cfg.efield.starve_thresh)
    est.set_E_field(fit_field, xyz, lr=cfg.E_lr)   # builds est.optimizer = [E-grid]
    set_params(est, cfg.init_logE, cfg.gt.nu, v0)  # v0 FIXED at GT direction
    est.gts = gts
    est.load_gts(gts)
    fm, xyz_c, aabb_c = free.cpu(), xyz.cpu(), fit_field.aabb.cpu()
    gt_lp_free = gt_logE_p[fm]
    w = strain[fm].clamp(min=0).numpy(); w = w / max(w.sum(), 1e-12)
    est.max_f = n_frames
    print(f"[Efield] fit res {res} ({res[0]*res[1]*res[2]} nodes), starved {n_starved}, "
          f"init logE {cfg.init_logE}, tv {cfg.efield.tv}, obs {cfg.obs} v0={v0}")

    out = train_efield_grid(est, fit_field, gt_lp_free, fm, w, aabb_c, xyz_c,
                            iter_cnt=phys_args.iter_cnt if cfg.train.iter_cnt is None else cfg.train.iter_cnt,
                            tv_weight=cfg.efield.tv, fine_lr=cfg.train.fine_lr,
                            patience=cfg.train.patience, min_iters=cfg.train.min_iters,
                            estop_tol=cfg.estop_tol, v0_field=None)
    grid_traj, err_traj, errw_traj, loss_traj = (out["grid_traj"], out["err_traj"],
                                                 out["errw_traj"], out["loss_traj"])
    best_grid = out["best_grid"]

    # ---- export ----
    lp_best = eval_Egrid_at(best_grid, aabb_c, xyz_c)[fm]
    err_all = float((lp_best - gt_lp_free).abs().mean())
    err_w = float((np.abs((lp_best - gt_lp_free).numpy()) * w).sum())
    obs_mask = strain[fm].numpy() >= np.median(strain[fm].numpy())
    err_obs = float(np.abs((lp_best - gt_lp_free).numpy())[obs_mask].mean())
    v0_est = est.init_vel.detach().cpu().numpy().tolist()
    v0_rel = float(np.linalg.norm(np.array(v0_est) - np.array(v0)) / max(np.linalg.norm(v0), 1e-12))
    set_params(est, cfg.init_logE, cfg.gt.nu, v0)  # v0 known = GT for pred rollout
    est.max_f = cfg.gt_frames
    pred = rollout_collect_surfaces(est)
    save_overlay_gif(gts, pred, rd.path("overlay.gif"), fit_frames=n_frames)
    result = {
        "scenario": "ours_efield_traj",
        "gt_kind": cfg.gt_kind, "gt_logE": cfg.gt.logE, "gt_ramp": list(cfg.gt_ramp),
        "gt_nu": cfg.gt.nu, "init_logE": cfg.init_logE, "obs": cfg.obs, "v0": v0,
        "res": list(res), "n_starved": n_starved, "tv": cfg.efield.tv,
        "logE_err_all": err_all, "logE_err_strain_w": err_w,
        "logE_err_observable_half": err_obs,
        "logE_err_traj": err_traj, "logE_err_strain_w_traj": errw_traj,
        "losses": loss_traj, "n_frames": n_frames,
        "joint_v0": False, "v0_field_mode": False, "warmup_iters": 0,
        "v0_gt": v0, "v0_estimated": v0_est, "v0_rel_err": v0_rel,
        "wall_time_s": time.time() - t0,
    }
    with open(rd.path("result.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"best_grid": best_grid, "gt_grid": gt_field.grid.detach().cpu(),
                "aabb": aabb_c, "res": list(res), "grid_traj": grid_traj},
               rd.path("ckpt.pt"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    draw_curve(loss_traj, rd.root, name="loss")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(err_traj, color="0.7", label="all free (incl. low-strain dead zone)")
    ax.plot(errw_traj, color="tab:red", label="strain-weighted (observable)")
    ax.set_yscale("log"); ax.set_xlabel("iter"); ax.set_ylabel("mean |log10 E - GT|")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(rd.path("Eerr.png")); plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    sc = ax.scatter(strain[fm].numpy(), lp_best.numpy(), s=3, c=gt_lp_free.numpy(),
                    cmap="viridis", alpha=0.5)
    fig.colorbar(sc, label="GT log10 E")
    if cfg.gt_kind == "uniform":
        ax.axhline(cfg.gt.logE, color="r", ls="--", label="GT logE")
    ax.axhline(cfg.init_logE, color="k", ls=":", label="init logE")
    ax.set_xlabel("GT per-particle local strain"); ax.set_ylabel("recovered log10 E")
    ax.set_title(f"recovered E vs observability\nerr all {err_all:.3f} / "
                 f"observable-half {err_obs:.3f}", fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(rd.path("E_vs_strain.png")); plt.close(fig)
    plot_E_gt_vs_pred(gt_lp_free, lp_best, rd.path("E_gt_vs_pred.png"),
                      color=strain[fm], color_label="GT per-particle local strain",
                      title=f"recovered vs GT log10 E (err all {err_all:.3f} / obs-half {err_obs:.3f})")
    zt = ((xyz_c[fm][:, 2] - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0, 1)
    if flip_z:
        zt = 1.0 - zt
    if cfg.gt_kind == "circular":
        plot_E_z_heatmap(zt, gt_lp_free, lp_best, rd.path("E_z_heatmap.png"),
                         title="GT (z, logE) density + recovered points")
    else:
        order = np.argsort(zt.numpy())
        fig, ax = plt.subplots(figsize=(6.5, 4))
        ax.scatter(zt.numpy(), lp_best.numpy(), s=4, alpha=0.3, label="recovered")
        ax.plot(zt.numpy()[order], gt_lp_free.numpy()[order], "r", lw=1.5, label="GT")
        ax.set_xlabel("zt (0=anchor end)"); ax.set_ylabel("log10 E")
        ax.set_title("E profile along cord"); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(rd.path("profile_1d.png")); plt.close(fig)
    plot_E_projections(xyz_c[fm], lp_best, gt_lp_free, rd.path("E_proj.png"),
                       aabb=aabb_c, res=res, xyz_anchor=xyz_c[~fm])
    plot_E_grid_nodes(best_grid, gt_field.grid.detach().cpu(), aabb_c, xyz_c[fm],
                      rd.path("E_grid.png"))
    subprocess.run([sys.executable, "make_panel.py", "--per_run", "--runs", rd.root],
                   cwd=os.path.dirname(os.path.abspath(__file__)))
    from ours.imgloss import build_panel
    build_panel(rd.root)
    print(f"[Efield] DONE: logE err all {err_all:.3f} | observable-half {err_obs:.3f} "
          f"| strain-w {err_w:.3f}, wall {time.time()-t0:.0f}s -> {rd.root}")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

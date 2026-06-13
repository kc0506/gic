#!/usr/bin/env python
# coding=utf-8
"""fit_traj_joint: JOINTLY recover v0 and a log10-E FIELD from a trajectory.

v0 is learned (scalar via --vfield.res 0, or a voxel field) AND E is a field,
optimized together after a v0-only warmup at the (wrong) init E. GT material is a
field (uniform/ramp/circular); GT v0 is a scalar direction (--obs) or a
non-uniform field (--gt-v0-variant). geometry loss.

The joint slice of the old efield_fit (--joint_v0 scalar / --v0_field double).
Built-in DIAGNOSTIC (joint_diag.png): loss + E err + v0 xy-relL2 over a
continuous warmup->joint axis, so "v0 fit is bad" vs "v0 fine but E won't
recover" is readable -- the curve the old efield_fit never produced.

Usage (gic env, gic repo root) -- run dir auto-placed at output/fit_traj_joint/<NN>:
  # scalar v0 + E-field:
  python fit_traj_joint.py --scene.cache <c> --scene.rot-z-deg 67.6 --scene.inject-pvol \
      --gt-kind ramp --run-label j_scalar
  # double field (v0-field + E-field), non-uniform GT v0:
  python fit_traj_joint.py --scene.cache <c> --scene.rot-z-deg 67.6 --scene.inject-pvol \
      --vfield.res 4x4x16 --gt-v0-variant mid_kick --gt-v0-scale 10.0 --run-label j_mid
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
from ours.fields import (EVoxelField, V0VoxelField, eval_Egrid_at, eval_grid_at,
                         fill_circular_grid, fill_profile_grid, strain_proxy)
from ours.rundir import RunDir
from ours.scene import build_anchored_scene, rollout_collect_surfaces, set_params
from ours.train import train_efield_grid, warmup_v0
from ours.viz import (plot_E_gt_vs_pred, plot_E_grid_nodes, plot_E_projections,
                      plot_E_z_heatmap, plot_field_projections, plot_grid_nodes,
                      plot_joint_diagnostics, save_overlay_gif)

VDIR = {"xp": [0.5, 0, 0], "xm": [-0.5, 0, 0], "yp": [0, 0.5, 0], "ym": [0, -0.5, 0]}


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=GTCfg)
    train: TrainCfg = field(default_factory=lambda: TrainCfg(patience=24, min_iters=30, fine_lr=0.02))
    efield: FieldCfg = field(default_factory=FieldCfg)
    vfield: FieldCfg = field(default_factory=lambda: FieldCfg(res="0"))
    """v0 side: res='0' = scalar joint; else double field (v0-field + E-field)"""
    frames: FramesCfg = field(default_factory=FramesCfg)
    init_logE: float = 4.0
    gt_kind: Literal["uniform", "ramp", "circular"] = "uniform"
    gt_ramp: Tuple[float, float] = (4.5, 5.5)
    obs: str = "ym"
    gt_v0_variant: Optional[str] = None
    """non-uniform GT v0 field (mid_kick|true_bend|ramp_y|ramp_x); needs vfield.res"""
    gt_v0_scale: float = 1.0
    gt_frames: int = 14
    E_lr: float = 0.2
    v0_lr: float = 0.025
    warmup_iters: int = 40
    estop_tol: float = 0.003
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config, rd: RunDir) -> None:
    t0 = time.time()
    v0 = VDIR[cfg.obs] if cfg.obs in VDIR else [float(x) for x in cfg.obs.split(",")]
    res = (tuple(int(p) for p in cfg.efield.res.lower().split("x"))
           if "x" in cfg.efield.res else (int(cfg.efield.res),) * 3)
    field_v0 = cfg.vfield.res != "0"
    res_v0 = (tuple(int(p) for p in cfg.vfield.res.lower().split("x"))
              if "x" in cfg.vfield.res else (int(cfg.vfield.res),) * 3) if field_v0 else None
    phys_args = make_phys_args(cfg.scene, cfg.train, cfg.frames, cfg.init_logE, cfg.gt.nu)
    phys_args.n_frames = cfg.gt_frames
    n_frames = 8 if cfg.frames.n_frames is None else cfg.frames.n_frames

    scene = build_anchored_scene(cfg.scene.cache, cfg.scene.rot_z_deg, phys_args,
                                 cfg.scene.anchor_mass_scale, cfg.gt_frames,
                                 inject_pvol=cfg.scene.inject_pvol,
                                 ti_mem_frac=cfg.scene.ti_mem_frac)
    est, xyz, free = scene["est"], scene["xyz"], scene["free"]
    aabb, z_lo, z_hi, flip_z = scene["aabb"], scene["z_lo"], scene["z_hi"], scene["flip_z"]
    fm, xyz_c, aabb_c = free.cpu(), xyz.cpu(), aabb.cpu()

    # ---- GT E-field + (optional) GT v0-field + rollout ----
    gt_field = EVoxelField(aabb.cpu(), res=res)
    if cfg.gt_kind == "uniform":
        gt_field.set_uniform_(cfg.gt.logE)
    elif cfg.gt_kind == "ramp":
        gt_field.fill_ramp_(cfg.gt_ramp[0], cfg.gt_ramp[1], z_lo, z_hi, flip_z)
    else:
        fill_circular_grid(gt_field, xyz.cpu(), free.cpu(), z_lo, z_hi,
                           cfg.gt_ramp[0], cfg.gt_ramp[1])
    est.set_E_field(gt_field, xyz, lr=0.0)
    set_params(est, cfg.gt.logE, cfg.gt.nu, v0)
    gt_logE_p = gt_field(xyz).detach().cpu()
    gt_v0_pf = None
    if cfg.gt_v0_variant is not None:
        assert field_v0, "--gt-v0-variant requires --vfield.res"
        gt_v0f = V0VoxelField(aabb.cpu(), res=res_v0)
        fill_profile_grid(gt_v0f, cfg.gt_v0_variant, cfg.gt_v0_scale, z_lo, z_hi, flip=flip_z)
        est.set_v0_field(gt_v0f, xyz, lr=0.0)
        gt_v0_pf = (gt_v0f(xyz).detach() * free.float().unsqueeze(1)).cpu()
        print(f"[joint] GT v0 = '{cfg.gt_v0_variant}' x{cfg.gt_v0_scale}; free |v0| mean "
              f"{gt_v0_pf[fm].norm(dim=-1).mean():.3f}")
    est.max_f = cfg.gt_frames
    gts = rollout_collect_surfaces(est)
    strain = strain_proxy(gts[0], gts[-1]).cpu()
    gt_lp_free = gt_logE_p[fm]
    gt_v0_free = gt_v0_pf[fm].float() if gt_v0_pf is not None else \
        torch.tensor(v0, dtype=torch.float32).expand(int(fm.sum()), 3)
    print(f"[joint] GT {cfg.gt_kind} logE [{gt_lp_free.min():.2f},{gt_lp_free.max():.2f}] | "
          f"free strain mean {strain[fm].mean():.4f}")

    # ---- fit E-field + v0 (scalar or field), v0-only warmup, then joint ----
    fit_field = EVoxelField(aabb.cpu(), res=res)
    fit_field.set_uniform_(cfg.init_logE)
    n_starved = fit_field.freeze_starved_(xyz[free].cpu(), cfg.efield.starve_thresh)
    est.set_E_field(fit_field, xyz, lr=cfg.E_lr)
    v0_field = None
    if field_v0:
        v0_field = V0VoxelField(aabb.cpu(), res=res_v0)
        v0_field.randomize_(cfg.vfield.init_std, seed=cfg.vfield.seed)
        v0_field.freeze_starved_(xyz[free].cpu(), cfg.vfield.starve_thresh)
        est.set_v0_field(v0_field, xyz, lr=cfg.v0_lr)
        v0_param = v0_field.grid
    else:
        v0_param = est.init_vel
    set_params(est, cfg.init_logE, cfg.gt.nu, [0.0, 0.0, 0.0])  # v0 learned from zero
    est.gts = gts
    est.load_gts(gts)
    w = strain[fm].clamp(min=0).numpy(); w = w / max(w.sum(), 1e-12)
    est.max_f = n_frames
    print(f"[joint] E res {res} starved {n_starved}, v0 {'field '+str(res_v0) if field_v0 else 'scalar'}, "
          f"warmup {cfg.warmup_iters}, init logE {cfg.init_logE}")

    # phase 1: v0 warmup at wrong E (records diagnostic)
    warm_loss, warm_v0rel = warmup_v0(est, v0_param, cfg.warmup_iters, cfg.v0_lr,
                                      v0_field=v0_field, gt_v0_pf=gt_v0_free,
                                      fm=fm, aabb_c=aabb_c, xyz_c=xyz_c)
    # phase 2: joint {v0, E-grid}; nu excluded (frozen)
    est.optimizer = torch.optim.Adam([
        {"params": v0_param, "lr": cfg.v0_lr, "name": "velocity"},
        {"params": fit_field.grid, "lr": cfg.E_lr, "name": "Youngs modulus"}])
    out = train_efield_grid(est, fit_field, gt_lp_free, fm, w, aabb_c, xyz_c,
                            iter_cnt=phys_args.iter_cnt if cfg.train.iter_cnt is None else cfg.train.iter_cnt,
                            tv_weight=cfg.efield.tv, fine_lr=cfg.train.fine_lr,
                            patience=cfg.train.patience, min_iters=cfg.train.min_iters,
                            estop_tol=cfg.estop_tol, v0_field=v0_field, gt_v0_pf=gt_v0_free)
    best_grid = out["best_grid"]
    err_traj, loss_traj = out["err_traj"], out["loss_traj"]
    # joint-phase v0 relL2: field path records it; scalar path derive from v0_traj
    if field_v0:
        joint_v0rel = out["v0_rel_traj"]
    else:
        gsv = max(float(gt_v0_free.norm(dim=-1).mean()), 1e-12)
        joint_v0rel = [float(np.linalg.norm(np.array(v[:2]) - gt_v0_free[0, :2].numpy()) / gsv)
                       for v in out["v0_traj"]]

    # ---- export ----
    lp_best = eval_Egrid_at(best_grid, aabb_c, xyz_c)[fm]
    err_all = float((lp_best - gt_lp_free).abs().mean())
    err_w = float((np.abs((lp_best - gt_lp_free).numpy()) * w).sum())
    obs_mask = strain[fm].numpy() >= np.median(strain[fm].numpy())
    err_obs = float(np.abs((lp_best - gt_lp_free).numpy())[obs_mask].mean())
    gsv = max(float(gt_v0_free.norm(dim=-1).mean()), 1e-12)
    if field_v0:
        vrec = eval_grid_at(v0_field.grid.detach().cpu(), aabb_c, xyz_c)[fm]
        v0_est = vrec.mean(0).tolist()
        v0_field_relL2 = float((vrec - gt_v0_free)[:, :2].norm(dim=-1).mean() / gsv)
        v0_rel = float((vrec - gt_v0_free).norm(dim=-1).mean() / gsv)
    else:
        vrec = None
        v0_est = est.init_vel.detach().cpu().numpy().tolist()
        v0_field_relL2 = None
        v0_rel = float(np.linalg.norm(np.array(v0_est) - np.array(v0)) / max(np.linalg.norm(v0), 1e-12))
    est.max_f = cfg.gt_frames
    pred = rollout_collect_surfaces(est)
    save_overlay_gif(gts, pred, rd.path("overlay.gif"), fit_frames=n_frames)
    result = {
        "scenario": "ours_joint_v0E",
        "gt_kind": cfg.gt_kind, "gt_logE": cfg.gt.logE, "gt_ramp": list(cfg.gt_ramp),
        "gt_v0_variant": cfg.gt_v0_variant, "gt_v0_scale": cfg.gt_v0_scale,
        "gt_nu": cfg.gt.nu, "init_logE": cfg.init_logE, "obs": cfg.obs, "v0": v0,
        "res": list(res), "v0_field_mode": field_v0, "n_starved": n_starved,
        "warmup_iters": cfg.warmup_iters,
        "logE_err_all": err_all, "logE_err_strain_w": err_w, "logE_err_observable_half": err_obs,
        "logE_err_traj": err_traj, "losses": loss_traj, "n_frames": n_frames,
        "v0_gt": v0, "v0_estimated": v0_est, "v0_rel_err": v0_rel,
        "v0_field_relL2_xy": v0_field_relL2,
        "warmup_loss_traj": warm_loss, "warmup_v0rel_traj": warm_v0rel,
        "joint_v0rel_traj": joint_v0rel,
        "wall_time_s": time.time() - t0,
    }
    with open(rd.path("result.json"), "w") as f:
        json.dump(result, f, indent=2)
    torch.save({"best_grid": best_grid, "gt_grid": gt_field.grid.detach().cpu(),
                "aabb": aabb_c, "res": list(res),
                "v0_grid": (v0_field.grid.detach().cpu() if field_v0 else None),
                "gt_v0_grid": (gt_v0f.grid.detach().cpu() if cfg.gt_v0_variant else None)},
               rd.path("ckpt.pt"))

    draw_curve(loss_traj, rd.root, name="loss")
    # THE diagnostic the old efield_fit lacked: warmup->joint continuous curve
    plot_joint_diagnostics(warm_loss + loss_traj, err_traj,
                           warm_v0rel + joint_v0rel, cfg.warmup_iters,
                           rd.path("joint_diag.png"),
                           title=f"joint v0+E: loss / E err / v0 xy-relL2 "
                                 f"(E {err_obs:.3f}, v0 {v0_rel:.2%})")
    # E-field viz
    plot_E_gt_vs_pred(gt_lp_free, lp_best, rd.path("E_gt_vs_pred.png"),
                      color=strain[fm], color_label="GT local strain",
                      title=f"recovered vs GT log10 E (obs-half {err_obs:.3f})")
    zt = ((xyz_c[fm][:, 2] - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0, 1)
    if flip_z:
        zt = 1.0 - zt
    if cfg.gt_kind == "circular":
        plot_E_z_heatmap(zt, gt_lp_free, lp_best, rd.path("E_z_heatmap.png"),
                         title="GT (z, logE) density + recovered")
    plot_E_projections(xyz_c[fm], lp_best, gt_lp_free, rd.path("E_proj.png"),
                       aabb=aabb_c, res=res, xyz_anchor=xyz_c[~fm])
    plot_E_grid_nodes(best_grid, gt_field.grid.detach().cpu(), aabb_c, xyz_c[fm],
                      rd.path("E_grid.png"))
    # v0-field viz (double-field only)
    if field_v0:
        plot_field_projections(xyz_c[fm], vrec, gt_v0_free, rd.path("v0_proj.png"),
                               aabb=aabb_c, res=res_v0, xyz_anchor=xyz_c[~fm])
    subprocess.run([sys.executable, "make_panel.py", "--per_run", "--runs", rd.root],
                   cwd=os.path.dirname(os.path.abspath(__file__)))
    print(f"[joint] DONE: E obs-half {err_obs:.3f} | v0 rel {v0_rel:.2%}"
          + (f" (field xy {v0_field_relL2:.2%})" if field_v0 else "")
          + f", wall {time.time()-t0:.0f}s -> {rd.root}")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

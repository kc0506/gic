#!/usr/bin/env python
# coding=utf-8
"""fit_traj_v0: recover the initial velocity v0 from a trajectory, E held fixed.

v0 is a scalar 3-DOF vector (--vfield.res 0, default) or a voxel FIELD
(--vfield.res 4x4x16). E/nu are held at GT (--fit-logE default) OR at a wrong E
(--fit-logE <val>: the warmup-viability test -- can v0 still converge under a
mis-specified E?). GT is GIC-self-simulated, optionally with a non-uniform v0
field (--gt-v0-variant). Velocity stage only; geometry loss.

The fix-E / v0-recovery slice of the old roundtrip_ours_scene.

Usage (gic env, gic repo root) -- run dir auto-placed at output/fit_traj_v0/<NN>:
  python fit_traj_v0.py --scene.cache <cache.pt> --gt.logE 5.0 --run-label scalar
  python fit_traj_v0.py --scene.cache <cache.pt> --gt.logE 5.0 \
      --vfield.res 4x4x16 --gt-v0-variant mid_kick --gt-v0-scale 10.0 --run-label mid
"""
from ours.gpu import pick_gpu

pick_gpu()  # pick a free GPU before torch/taichi create a CUDA context

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import tyro

from simulator import Estimator
from utils.system_utils import draw_curve

from ours.config import FieldCfg, FramesCfg, GTCfg, SceneCfg, TrainCfg, make_phys_args
from ours.fields import V0VoxelField, eval_grid_at, fill_profile_grid
from ours.rundir import RunDir
from ours.scene import build_anchored_scene, rollout_collect_surfaces, set_params
from ours.train import train_ours
from ours.viz import (plot_axis_profiles, plot_field_projections, plot_grid_nodes,
                      plot_profile_1d, save_overlay_gif, save_rollout_gif)


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=GTCfg)
    train: TrainCfg = field(default_factory=TrainCfg)
    vfield: FieldCfg = field(default_factory=lambda: FieldCfg(res="0"))
    """v0 field options; res='0' (default) = scalar 3-DOF, else voxel field"""
    frames: FramesCfg = field(default_factory=FramesCfg)
    fit_logE: Optional[float] = None
    """E held fixed during the fit; None = gt.logE (fix-E-GT). A different value
    = wrong-E warmup-viability test."""
    gt_v0_variant: Optional[str] = None
    """mid_kick|true_bend|ramp_y|ramp_x: non-uniform GT v0 field (needs vfield.res)"""
    gt_v0_scale: float = 1.0
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config, rd: RunDir) -> None:
    start = time.time()
    fit_logE = cfg.gt.logE if cfg.fit_logE is None else cfg.fit_logE
    phys_args = make_phys_args(cfg.scene, cfg.train, cfg.frames, fit_logE, cfg.gt.nu)
    n_frames = phys_args.n_frames if cfg.frames.n_frames is None else cfg.frames.n_frames
    phys_args.n_frames = n_frames
    field_res = None
    if cfg.vfield.res != "0":
        field_res = (tuple(int(x) for x in cfg.vfield.res.lower().split("x"))
                     if "x" in cfg.vfield.res else (int(cfg.vfield.res),) * 3)

    scene = build_anchored_scene(cfg.scene.cache, cfg.scene.rot_z_deg, phys_args,
                                 cfg.scene.anchor_mass_scale, n_frames,
                                 inject_pvol=cfg.scene.inject_pvol,
                                 ti_mem_frac=cfg.scene.ti_mem_frac)
    est, xyz, free = scene["est"], scene["xyz"], scene["free"]
    aabb, z_lo, z_hi, flip_z = scene["aabb"], scene["z_lo"], scene["z_hi"], scene["flip_z"]
    anchor_mask = ~free

    # ---- 1. GT: GIC sim, optionally with a non-uniform v0 field ----
    set_params(est, cfg.gt.logE, cfg.gt.nu, list(cfg.gt.vel))
    gt_part = None
    if cfg.gt_v0_variant is not None:
        assert field_res is not None, "--gt-v0-variant requires --vfield.res"
        gt_field = V0VoxelField(aabb.cpu(), res=field_res)
        fill_profile_grid(gt_field, cfg.gt_v0_variant, cfg.gt_v0_scale, z_lo, z_hi, flip=flip_z)
        est.set_v0_field(gt_field, xyz, lr=0.0)
        gt_part = (gt_field(xyz).detach() * free.float().unsqueeze(1)).cpu()
        print(f"[v0] GT v0 = '{cfg.gt_v0_variant}' x{cfg.gt_v0_scale} on {field_res}; "
              f"free |v0| mean {gt_part[free.cpu()].norm(dim=-1).mean():.3f}")
    gts = rollout_collect_surfaces(est)
    disp = (gts[-1] - gts[0]).norm(dim=-1)
    print(f"[v0] GT motion: free mean disp {disp[free].mean():.5f}, max {disp[free].max():.5f}")
    save_rollout_gif(gts, rd.path("gt_rollout.gif"))

    # ---- 2. fit v0 (scalar or field) at the fixed (maybe wrong) E ----
    est.gts = gts
    est.load_gts(gts)
    v0_field, n_starved = None, 0
    if field_res is not None:
        v0_field = V0VoxelField(aabb.cpu(), res=field_res)
        v0_field.randomize_(cfg.vfield.init_std, seed=cfg.vfield.seed)
        if cfg.vfield.starve_thresh > 0:
            n_starved = v0_field.freeze_starved_(xyz[free].cpu(), thresh=cfg.vfield.starve_thresh)
        field_lr = cfg.vfield.lr if cfg.vfield.lr is not None else phys_args.vel_lr
        est.set_v0_field(v0_field, xyz, lr=field_lr)
        print(f"[v0] FIELD res={field_res} ({3*np.prod(field_res)} DOF), starved {n_starved}, lr {field_lr}")
    # scalar v0 starts from config init_vel (NOT gt.vel -- starting at the
    # solution gives near-zero loss and a meaningless "perfect" result, and risks
    # chamfer NaN on coincident points). field v0 starts from a randomized grid
    # (set above), here zeroed scalar. Mirrors roundtrip's fix_E path.
    fit_v0 = [0.0, 0.0, 0.0] if field_res is not None else list(phys_args.init_vel)
    set_params(est, fit_logE, cfg.gt.nu, fit_v0)
    est.set_stage(Estimator.velocity_stage)
    losses_vel, e_s_vel = train_ours(
        est, phys_args, phys_args.vel_estimation_frames, rd.root, gts=gts,
        patience=cfg.train.patience, min_iters=cfg.train.min_iters,
        ckpt_every=cfg.train.ckpt_every, overlay_every=cfg.train.overlay_every,
        param_stop_tol=cfg.train.param_stop_tol, tv_weight=cfg.vfield.tv)

    # ---- 3. export (v0-only) ----
    aabb_c, xyz_c, fm = aabb.cpu(), xyz.cpu(), free.cpu()
    v_best = None
    if v0_field is not None:
        gt_pf = gt_part[fm].float() if gt_part is not None else \
            torch.tensor(cfg.gt.vel, dtype=torch.float32).expand(int(fm.sum()), 3)
        gt_scale = max(float(gt_pf.norm(dim=-1).mean()), 1e-12)
        gt_mean_vec = gt_pf.mean(0)
        per_iter_v = [eval_grid_at(d["velocity"].float(), aabb_c, xyz_c)[fm] for d in e_s_vel]
        v0_traj = [v.mean(0).tolist() for v in per_iter_v]
        v0_traj_p05 = [v.quantile(0.05, dim=0).tolist() for v in per_iter_v]
        v0_traj_p95 = [v.quantile(0.95, dim=0).tolist() for v in per_iter_v]
        field_err_traj = [float((v - gt_pf).norm(dim=-1).mean() / gt_scale) for v in per_iter_v]
        field_err_xy_traj = [float((v - gt_pf)[:, :2].norm(dim=-1).mean() / gt_scale) for v in per_iter_v]
        best_idx = losses_vel.index(min(losses_vel))
        v_best = per_iter_v[best_idx]
        ax_err_best = (v_best - gt_pf).abs().mean(0)
        v0_est = v_best.mean(0).tolist()
    else:
        v0_traj = [[float(x) for x in d["velocity"]] for d in e_s_vel]
        v0_est = est.init_vel.detach().cpu().numpy().tolist()

    set_params(est, fit_logE, cfg.gt.nu, v0_est)
    est.max_f = n_frames
    pred = rollout_collect_surfaces(est)
    save_rollout_gif(pred, rd.path("pred_rollout.gif"))
    save_overlay_gif(gts, pred, rd.path("overlay.gif"),
                     fit_frames=phys_args.vel_estimation_frames)
    plot_axis_profiles(gts, pred, free, rd.path("axis_profile.png"),
                       fit_frames=phys_args.vel_estimation_frames)
    gt_v = gt_mean_vec.numpy() if (v0_field is not None and gt_part is not None) else np.array(cfg.gt.vel)
    err = np.linalg.norm(np.array(v0_est) - gt_v) / max(np.linalg.norm(gt_v), 1e-12)
    result = {
        "scenario": "ours_fixE_learn_v0",
        "fit_logE": fit_logE,
        "gt": {"E": 10.0 ** cfg.gt.logE, "nu": cfg.gt.nu, "vel": list(cfg.gt.vel),
               "v0_variant": cfg.gt_v0_variant, "v0_scale": cfg.gt_v0_scale},
        "v0_estimated": v0_est, "v0_rel_err": float(err),
        "losses_vel": [float(l) for l in losses_vel], "v0_traj": v0_traj,
        "gt_motion_free_mean_disp": float(disp[free].mean()),
        "wall_time_s": time.time() - start,
    }
    if v0_field is not None:
        per_err = (v_best - gt_pf).norm(dim=-1)
        result["v0_traj_p05"], result["v0_traj_p95"] = v0_traj_p05, v0_traj_p95
        result["v0_field"] = {
            "res": list(field_res), "init_std": cfg.vfield.init_std,
            "seed": cfg.vfield.seed, "lr": cfg.vfield.lr,
            "starve_thresh": cfg.vfield.starve_thresh, "n_starved_frozen": n_starved,
            "tv_weight": cfg.vfield.tv,
            "per_axis_err_best": ax_err_best.tolist(),
            "rel_l2_xy_best": field_err_xy_traj[best_idx], "rel_l2_xy_traj": field_err_xy_traj,
            "rel_l2_best": field_err_traj[best_idx], "rel_l2_traj": field_err_traj,
            "mean_vec_best": v0_est, "per_axis_std_best": v_best.std(0).tolist(),
            "per_particle_err_max": float(per_err.max()) / gt_scale,
            "per_particle_err_p95": float(per_err.quantile(0.95)) / gt_scale,
        }
    with open(rd.path("result.json"), "w") as f:
        json.dump(result, f, indent=2)
    draw_curve(losses_vel, rd.root, name="loss_vel")

    if v0_field is not None:
        plot_field_projections(xyz_c[fm], v_best, gt_pf, rd.path("field_proj.png"),
                               aabb=aabb_c, res=field_res, xyz_anchor=xyz_c[~fm])
        best_grid = e_s_vel[best_idx]["velocity"].float()
        gt_for_nodes = (gt_field.grid.data.detach().cpu() if gt_part is not None
                        else torch.tensor(cfg.gt.vel, dtype=torch.float32))
        plot_grid_nodes(best_grid, aabb_c, gt_for_nodes, xyz_c[fm],
                        rd.path("grid_quiver.png"), rd.path("grid_hist.png"))
        if gt_part is not None:
            zt = ((xyz_c[fm][:, 2] - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0, 1)
            if flip_z:
                zt = 1.0 - zt
            plot_profile_1d(zt, v_best, gt_pf, rd.path("profile_1d.png"))
        print(f"[v0] FIELD per-axis |err| {ax_err_best.numpy().round(3).tolist()} | "
              f"xy relL2 {field_err_xy_traj[best_idx]:.2%} (all-axes {field_err_traj[best_idx]:.2%})")
    print(f"[v0] DONE: GT v0={list(cfg.gt.vel)} -> {v0_est} (rel err {err:.2%}), "
          f"wall {time.time() - start:.0f}s -> {rd.root}")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

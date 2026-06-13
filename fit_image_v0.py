#!/usr/bin/env python
# coding=utf-8
"""fit_image_v0: recover the initial velocity v0 from IMAGE loss, E held fixed.

v0 is a scalar 3-DOF vector (--vfield.res 0, default) or a voxel FIELD
(--vfield.res 4x4x16). E/nu are pinned at the GT (--fit-logE default). The fit
runs in the PHYSICAL stage with the optimizer swapped to the velocity parameter
(image position-gradients are only injected there; the velocity stage would
chamfer against the dummy GT buffers), so E/nu never move. GT = GIC self-sim,
optionally with a non-uniform v0 field (--gt-v0-variant). With --render.bg-image
the supervision becomes full-RGB over a static background.

The image-loss / v0-recovery slice of the old image_fit_ours --mode
fit_v0scalar|fit_v0field. Trajectory v0 -> fit_traj_v0.

Usage (gic env, gic repo root):
  python fit_image_v0.py --scene.cache <cache.pt> --gt.logE 5.0 --run-label scalar
  python fit_image_v0.py --scene.cache <cache.pt> --gt.logE 5.0 \
      --vfield.res 4x4x16 --gt-v0-variant ramp_x --vfield.lr 0.025 --run-label rampx
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

from ours.config import FieldCfg, FramesCfg, GTCfg, RenderCfg, SceneCfg, TrainCfg, make_phys_args
from ours.fields import V0VoxelField, eval_grid_at
from ours.imgloss import build_panel, gt_pred_diff_gif, render_pred_frames, setup_image_scene
from ours.rundir import RunDir
from ours.scene import rollout_collect_surfaces, set_params
from ours.train import train_ours
from ours.viz import (plot_field_projections, plot_grid_nodes, plot_profile_1d,
                      plot_v0_traj, save_overlay_gif)


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=GTCfg)
    train: TrainCfg = field(default_factory=lambda: TrainCfg(patience=16, min_iters=25))
    vfield: FieldCfg = field(default_factory=lambda: FieldCfg(res="0"))
    """v0 field options; res='0' (default) = scalar 3-DOF, else voxel field"""
    frames: FramesCfg = field(default_factory=FramesCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    fit_logE: Optional[float] = None
    """E held fixed during the fit; None = gt.logE"""
    gt_v0_variant: Optional[str] = None
    """mid_kick|true_bend|ramp_y|ramp_x: non-uniform GT v0 field (needs vfield.res)"""
    gt_v0_scale: float = 1.0
    gt_frames: Optional[int] = None
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config, rd: RunDir) -> None:
    t0 = time.time()
    fit_logE = cfg.gt.logE if cfg.fit_logE is None else cfg.fit_logE
    phys_args = make_phys_args(cfg.scene, cfg.train, cfg.frames, fit_logE, cfg.gt.nu)
    n_frames = phys_args.n_frames if cfg.frames.n_frames is None else cfg.frames.n_frames
    gt_n = cfg.gt_frames if cfg.gt_frames is not None else n_frames
    field_res = None
    if cfg.vfield.res != "0":
        field_res = (tuple(int(x) for x in cfg.vfield.res.lower().split("x"))
                     if "x" in cfg.vfield.res else (int(cfg.vfield.res),) * 3)

    s = setup_image_scene(cfg.scene, cfg.gt, cfg.render, phys_args, gt_n,
                          gt_v0_variant=cfg.gt_v0_variant, gt_v0_scale=cfg.gt_v0_scale,
                          v0_field_res=field_res)
    est, gaussians, pipe = s["est"], s["gaussians"], s["pipe"]
    xyz, anchor_mask, aabb = s["xyz"], s["anchor_mask"], s["aabb"]
    free = ~anchor_mask
    z_lo, z_hi, flip_z, gt_part = s["z_lo"], s["z_hi"], s["flip_z"], s["gt_part"]

    # ---- fit v0 (scalar or field) at the fixed E, image loss, phys stage ----
    field_lr = cfg.vfield.lr if cfg.vfield.lr is not None else phys_args.vel_lr
    v0_field, n_starved = None, 0
    set_params(est, fit_logE, cfg.gt.nu, [0.0, 0.0, 0.0])
    if field_res is not None:
        v0_field = V0VoxelField(aabb.cpu(), res=field_res)
        v0_field.randomize_(cfg.vfield.init_std, seed=cfg.vfield.seed)
        if cfg.vfield.starve_thresh > 0:
            n_starved = v0_field.freeze_starved_(xyz[free].cpu(), thresh=cfg.vfield.starve_thresh)
        est.set_v0_field(v0_field, xyz, lr=field_lr)  # builds vel_optimizer
        est.optimizer = est.vel_optimizer             # phys stage steps the FIELD
        print(f"[img-v0] FIELD res={field_res} ({3*np.prod(field_res)} DOF), "
              f"starved {n_starved}, lr {field_lr}, tv {cfg.vfield.tv}")
    else:
        est.optimizer = torch.optim.Adam(
            [{"params": est.init_vel, "lr": field_lr, "name": "velocity"}])
        print(f"[img-v0] SCALAR 3-DOF, lr {field_lr}, cam {cfg.render.camera}")
    est.set_stage(Estimator.physical_params_stage)
    losses, e_s = train_ours(
        est, phys_args, n_frames, rd.root, gts=None,
        patience=cfg.train.patience, min_iters=cfg.train.min_iters,
        ckpt_every=cfg.train.ckpt_every, overlay_every=0,
        param_stop_tol=cfg.train.param_stop_tol, tv_weight=cfg.vfield.tv)

    # ---- export ----
    aabb_c, xyz_c, fm = aabb.cpu(), xyz.cpu(), free.cpu()
    v_best = None
    if v0_field is not None:
        gt_pf = gt_part[fm].float() if gt_part is not None else \
            torch.tensor(cfg.gt.vel, dtype=torch.float32).expand(int(fm.sum()), 3)
        gt_scale = max(float(gt_pf.norm(dim=-1).mean()), 1e-12)
        gt_mean_vec = gt_pf.mean(0)
        per_iter_v = [eval_grid_at(d["velocity"].float(), aabb_c, xyz_c)[fm] for d in e_s]
        v0_traj = [v.mean(0).tolist() for v in per_iter_v]
        rel_traj = [float((v - gt_pf).norm(dim=-1).mean() / gt_scale) for v in per_iter_v]
        rel_xy_traj = [float((v - gt_pf)[:, :2].norm(dim=-1).mean() / gt_scale) for v in per_iter_v]
        best_idx = losses.index(min(losses))
        v_best = per_iter_v[best_idx]
        ax_err_best = (v_best - gt_pf).abs().mean(0)
        v0_est = v_best.mean(0).tolist()
    else:
        v0_traj = [[float(x) for x in d["velocity"]] for d in e_s]
        v0_est = est.init_vel.detach().cpu().numpy().tolist()

    est.img_loss = False
    set_params(est, fit_logE, cfg.gt.nu, v0_est if v0_field is None else [0.0, 0.0, 0.0])
    est.max_f = gt_n
    pred_roll = rollout_collect_surfaces(est)
    est.set_stage(Estimator.physical_params_stage)
    pred_pngs = render_pred_frames(pred_roll, gaussians, pipe, s["pose_cam"],
                                   view=cfg.render.camera, bg_per_view=s["bg_per_view"], drive=s["drive"])
    gt_pred_diff_gif(s["gt_frames_png"], pred_pngs, rd.path("gt_pred_diff.gif"), n_frames)
    save_overlay_gif(s["gt_roll"], pred_roll, rd.path("overlay.gif"), fit_frames=n_frames)

    gt_v = (gt_mean_vec.numpy() if (v0_field is not None and gt_part is not None)
            else np.array(cfg.gt.vel))
    err = float(np.linalg.norm(np.array(v0_est) - gt_v) / max(np.linalg.norm(gt_v), 1e-12))
    result = {
        "scenario": "ours_image_fixE_learn_v0",
        "fit_logE": fit_logE,
        "gt": {"E": 10.0 ** cfg.gt.logE, "nu": cfg.gt.nu, "vel": list(cfg.gt.vel),
               "v0_variant": cfg.gt_v0_variant, "v0_scale": cfg.gt_v0_scale},
        "w_img": cfg.render.w_img, "w_alp": cfg.render.w_alp, "n_frames": n_frames,
        "camera": (cfg.render.cameras or cfg.render.camera), "bg_image": cfg.render.bg_image,
        "v0_estimated": v0_est, "v0_rel_err": err,
        "losses_phys": [float(l) for l in losses], "v0_traj": v0_traj,
        "wall_time_s": time.time() - t0,
    }
    if v0_field is not None:
        per_err = (v_best - gt_pf).norm(dim=-1)
        result["v0_field"] = {
            "res": list(field_res), "init_std": cfg.vfield.init_std, "seed": cfg.vfield.seed,
            "lr": field_lr, "tv_weight": cfg.vfield.tv, "n_starved_frozen": n_starved,
            "per_axis_err_best": ax_err_best.tolist(),
            "rel_l2_xy_best": rel_xy_traj[best_idx], "rel_l2_xy_traj": rel_xy_traj,
            "rel_l2_best": rel_traj[best_idx], "rel_l2_traj": rel_traj,
            "mean_vec_best": v0_est, "per_axis_std_best": v_best.std(0).tolist(),
            "per_particle_err_max": float(per_err.max()) / gt_scale,
            "per_particle_err_p95": float(per_err.quantile(0.95)) / gt_scale,
        }
    with open(rd.path("result.json"), "w") as f:
        json.dump(result, f, indent=2)
    draw_curve([float(l) for l in losses], rd.root, name="loss_phys")
    plot_v0_traj(v0_traj, list(cfg.gt.vel), rd.path("v0_traj.png"))

    if v0_field is not None:
        plot_field_projections(xyz_c[fm], v_best, gt_pf, rd.path("field_proj.png"),
                               aabb=aabb_c, res=field_res, xyz_anchor=xyz_c[~fm])
        gt_for_nodes = (s["gt_v0_grid"] if s["gt_v0_grid"] is not None
                        else torch.tensor(cfg.gt.vel, dtype=torch.float32))
        plot_grid_nodes(e_s[best_idx]["velocity"].float(), aabb_c, gt_for_nodes, xyz_c[fm],
                        rd.path("grid_quiver.png"), rd.path("grid_hist.png"))
        if gt_part is not None:
            zt = ((xyz_c[fm][:, 2] - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0, 1)
            if flip_z:
                zt = 1.0 - zt
            plot_profile_1d(zt, v_best, gt_pf, rd.path("profile_1d.png"))
        print(f"[img-v0] FIELD per-axis |err| {ax_err_best.numpy().round(3).tolist()} | "
              f"xy relL2 {rel_xy_traj[best_idx]:.2%} (all-axes {rel_traj[best_idx]:.2%})")
    build_panel(rd.root)
    print(f"[img-v0] DONE: GT v0={list(cfg.gt.vel)} -> {[round(x,3) for x in v0_est]} "
          f"(rel err {err:.2%}), wall {time.time()-t0:.0f}s -> {rd.root}")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

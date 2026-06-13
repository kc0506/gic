#!/usr/bin/env python
# coding=utf-8
"""fit_image_joint: joint recovery of scalar v0 AND scalar E from IMAGE loss.

Two-phase hybrid mirroring the traj-side joint: (1) a v0-only warmup at the
(wrong) init E -- a cold-start joint lets E wander decades while v0 is still ~0;
(2) a joint phase stepping E (scheduled lr) and v0 together, nu frozen. GT = GIC
self-sim at gt params, rendered identically. With --render.bg-image the
supervision becomes full-RGB over a static background.

The joint slice of the old image_fit_ours --mode fit_joint, plus a built-in
warmup->joint diagnostic (loss / logE-err / v0-relL2). Trajectory joint ->
fit_traj_joint.

Usage (gic env, gic repo root):
  python fit_image_joint.py --scene.cache <cache.pt> --gt.logE 5.0 \
      --init-logE 4.0 --warmup-iters 40 --run-label jscalar
"""
from ours.gpu import pick_gpu

pick_gpu()  # pick a free GPU before torch/taichi create a CUDA context

import json
import time
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import tyro

from simulator import Estimator
from utils.system_utils import draw_curve

from ours.config import FramesCfg, GTCfg, RenderCfg, SceneCfg, TrainCfg, make_phys_args
from ours.imgloss import build_panel, gt_pred_diff_gif, render_pred_frames, setup_image_scene
from ours.rundir import RunDir
from ours.scene import rollout_collect_surfaces, set_params
from ours.train import train_ours
from ours.viz import plot_joint_diagnostics, plot_v0_traj, save_overlay_gif


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=GTCfg)
    train: TrainCfg = field(default_factory=lambda: TrainCfg(patience=16, min_iters=25, fine_lr=0.02))
    frames: FramesCfg = field(default_factory=FramesCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    init_logE: float = 4.0
    warmup_iters: int = 40
    v0_lr: float = 0.025
    gt_frames: Optional[int] = None
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config, rd: RunDir) -> None:
    t0 = time.time()
    phys_args = make_phys_args(cfg.scene, cfg.train, cfg.frames, cfg.init_logE, cfg.gt.nu)
    n_frames = phys_args.n_frames if cfg.frames.n_frames is None else cfg.frames.n_frames
    gt_n = cfg.gt_frames if cfg.gt_frames is not None else n_frames

    s = setup_image_scene(cfg.scene, cfg.gt, cfg.render, phys_args, gt_n)
    est, gaussians, pipe = s["est"], s["gaussians"], s["pipe"]
    print(f"[img-joint] N={s['xyz'].shape[0]}, init logE {cfg.init_logE}, "
          f"warmup {cfg.warmup_iters}, {n_frames}f, cam {cfg.render.camera}")

    set_params(est, cfg.init_logE, cfg.gt.nu, [0.0, 0.0, 0.0])
    est.set_stage(Estimator.physical_params_stage)

    # ---- phase 1: v0 warmup at the wrong init E ----
    est.optimizer = torch.optim.Adam(
        [{"params": est.init_vel, "lr": cfg.v0_lr, "name": "velocity"}])
    saved_iter_cnt = phys_args.iter_cnt
    phys_args.iter_cnt = cfg.warmup_iters
    print(f"[img-joint] phase1 warmup: v0 lr {cfg.v0_lr}, E pinned 1e{cfg.init_logE}")
    l1, e1 = train_ours(est, phys_args, n_frames, rd.root, gts=None,
                        patience=cfg.train.patience, min_iters=10, ckpt_every=10,
                        overlay_every=0, fine_lr=0.0, param_stop_tol=cfg.train.param_stop_tol)
    phys_args.iter_cnt = saved_iter_cnt

    # ---- phase 2: joint E + v0 (nu frozen; pop scheduler so lr-0 sticks) ----
    e_lr = phys_args.params["Youngs modulus"]["init_lr"]
    est.optimizer = torch.optim.Adam([
        {"params": est.E, "lr": e_lr, "name": "Youngs modulus"},
        {"params": est.nu, "lr": 0.0, "name": "Poisson ratio"},
        {"params": est.init_vel, "lr": cfg.v0_lr, "name": "velocity"}])
    est.lr_schedulers.pop("Poisson ratio", None)
    print(f"[img-joint] phase2 joint: E lr {e_lr} (scheduled) + v0 lr {cfg.v0_lr}")
    l2, e2 = train_ours(est, phys_args, n_frames, rd.root, gts=None,
                        patience=cfg.train.patience, min_iters=cfg.train.min_iters,
                        ckpt_every=10, overlay_every=0, fine_lr=cfg.train.fine_lr,
                        param_stop_tol=cfg.train.param_stop_tol)

    # ---- export ----
    losses = [float(x) for x in (l1 + l2)]
    E_traj = [10.0 ** cfg.init_logE] * len(l1) + [d["Youngs modulus"] for d in e2]
    v0_traj = ([[float(x) for x in d["velocity"]] for d in e1]
               + [[float(x) for x in d["velocity"]] for d in e2])
    gt_E = 10.0 ** cfg.gt.logE
    gt_v = np.array(list(cfg.gt.vel))
    gsv = max(float(np.linalg.norm(gt_v)), 1e-12)
    e_err_traj = [abs(np.log10(max(e, 1.0)) - cfg.gt.logE) for e in E_traj]
    v0_rel_traj = [float(np.linalg.norm(np.array(v) - gt_v) / gsv) for v in v0_traj]
    best2 = l2.index(min(l2))
    best_E = e2[best2]["Youngs modulus"]
    v0_best = [float(x) for x in e2[best2]["velocity"]]
    rel_E = (best_E - gt_E) / gt_E
    rel_v = float(np.linalg.norm(np.array(v0_best) - gt_v) / gsv)

    est.img_loss = False
    est.E.data.copy_(torch.log10(torch.tensor(best_E, device=est.device)))
    est.init_vel = nn.Parameter(torch.tensor(v0_best, device=est.device))
    est.max_f = gt_n
    pred_roll = rollout_collect_surfaces(est)
    est.set_stage(Estimator.physical_params_stage)
    pred_pngs = render_pred_frames(pred_roll, gaussians, pipe, s["pose_cam"],
                                   view=cfg.render.camera, bg_per_view=s["bg_per_view"], drive=s["drive"])
    gt_pred_diff_gif(s["gt_frames_png"], pred_pngs, rd.path("gt_pred_diff.gif"), n_frames)
    save_overlay_gif(s["gt_roll"], pred_roll, rd.path("overlay.gif"), fit_frames=n_frames)

    result = {
        "scenario": "ours_image_joint_v0E",
        "gt": {"E": gt_E, "logE": cfg.gt.logE, "nu": cfg.gt.nu, "vel": list(cfg.gt.vel)},
        "init": {"logE": cfg.init_logE}, "warmup_iters": len(l1),
        "w_img": cfg.render.w_img, "w_alp": cfg.render.w_alp, "n_frames": n_frames,
        "camera": (cfg.render.cameras or cfg.render.camera), "bg_image": cfg.render.bg_image,
        "losses_phys": losses, "alt_phys_bounds": [len(l1), len(l1) + len(l2)],
        "E_traj": E_traj, "v0_traj": v0_traj,
        "best": {"Youngs modulus": best_E, "iter": len(l1) + best2, "loss": float(l2[best2])},
        "rel_err_E": rel_E, "v0_estimated": v0_best, "v0_rel_err": rel_v,
        "wall_time_s": time.time() - t0,
    }
    with open(rd.path("result.json"), "w") as f:
        json.dump(result, f, indent=2)
    draw_curve(losses, rd.root, name="loss_phys")
    plot_joint_diagnostics(losses, e_err_traj, v0_rel_traj, len(l1),
                           rd.path("joint_diag.png"),
                           title=f"image joint | GT logE {cfg.gt.logE} v0 {list(cfg.gt.vel)}")
    plot_v0_traj(v0_traj, list(cfg.gt.vel), rd.path("v0_traj.png"), warmup_iters=len(l1))
    build_panel(rd.root)
    print(f"[img-joint] DONE: E {best_E:.4g} ({rel_E:+.2%}) v0 {[round(x,3) for x in v0_best]} "
          f"(rel {rel_v:.2%}), wall {time.time()-t0:.0f}s -> {rd.root}")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

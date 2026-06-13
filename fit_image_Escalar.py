#!/usr/bin/env python
# coding=utf-8
"""fit_image_Escalar: recover a scalar global Young's modulus E from IMAGE loss.

v0 is FIXED to gt.vel; E(+nu) is fit from rendered-frame supervision through one
(or several) static cameras. GT = GIC self-sim at gt params, rendered identically
to the prediction (self-consistent roundtrip). With --render.bg-image the GT and
pred composite over the same static background and the loss becomes full-RGB --
the realistic with-bg setting.

The scalar-E slice of the old image_fit_ours --mode fit. v0 -> fit_image_v0;
joint v0+E -> fit_image_joint. Trajectory (geometry) loss -> fit_traj_Escalar.

Usage (gic env, gic repo root):
  python fit_image_Escalar.py --scene.cache <cache.pt> --gt.logE 5.0 \
      --init-logE 4.0 --run-label gtE5_initE4
  python fit_image_Escalar.py --scene.cache <cache.pt> --render.bg-image scene ...
"""
from ours.gpu import pick_gpu

pick_gpu()  # pick a free GPU before torch/taichi create a CUDA context

import json
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import tyro

from simulator import Estimator
from utils.system_utils import draw_curve

from ours.config import FramesCfg, GTCfg, RenderCfg, SceneCfg, TrainCfg, make_phys_args
from ours.imgloss import build_panel, gt_pred_diff_gif, render_pred_frames, setup_image_scene
from ours.rundir import RunDir
from ours.scene import rollout_collect_surfaces, set_params
from ours.train import train_ours
from ours.viz import plot_param_traj, save_overlay_gif


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=GTCfg)
    train: TrainCfg = field(default_factory=lambda: TrainCfg(patience=16, min_iters=25, fine_lr=0.02))
    frames: FramesCfg = field(default_factory=FramesCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    init_logE: float = 4.0
    init_nu: float = 0.1
    gt_frames: Optional[int] = None
    """GT render/rollout length (> n_frames => the tail is held-out extrapolation)"""
    E_lr: Optional[float] = None
    nu_lr: Optional[float] = None
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config, rd: RunDir) -> None:
    t0 = time.time()
    phys_args = make_phys_args(cfg.scene, cfg.train, cfg.frames,
                               cfg.init_logE, cfg.init_nu, cfg.E_lr, cfg.nu_lr)
    n_frames = phys_args.n_frames if cfg.frames.n_frames is None else cfg.frames.n_frames
    gt_n = cfg.gt_frames if cfg.gt_frames is not None else n_frames
    assert gt_n >= n_frames, (gt_n, n_frames)

    s = setup_image_scene(cfg.scene, cfg.gt, cfg.render, phys_args, gt_n)
    est, gaussians, pipe = s["est"], s["gaussians"], s["pipe"]
    print(f"[img-Escalar] N={s['xyz'].shape[0]}, fix v0={list(cfg.gt.vel)}, "
          f"init logE {cfg.init_logE}, {n_frames}f fit / {gt_n}f GT, cam {cfg.render.camera}")

    # ---- fit E(+nu) from image loss, v0 fixed ----
    set_params(est, cfg.init_logE, cfg.init_nu, list(cfg.gt.vel))
    est.set_stage(Estimator.physical_params_stage)
    losses, e_s = train_ours(
        est, phys_args, n_frames, rd.root, gts=None,
        patience=cfg.train.patience, min_iters=cfg.train.min_iters,
        ckpt_every=cfg.train.ckpt_every, overlay_every=0, fine_lr=cfg.train.fine_lr)

    # ---- export ----
    min_idx = losses.index(min(losses))
    best = e_s[min_idx]
    gt_E = 10.0 ** cfg.gt.logE
    rel = abs(best["Youngs modulus"] - gt_E) / gt_E

    est.img_loss = False
    set_params(est, math.log10(best["Youngs modulus"]), best["Poisson ratio"], list(cfg.gt.vel))
    est.max_f = gt_n
    pred_roll = rollout_collect_surfaces(est)
    est.set_stage(Estimator.physical_params_stage)
    pred_pngs = render_pred_frames(pred_roll, gaussians, pipe, s["pose_cam"],
                                   view=cfg.render.camera, bg_per_view=s["bg_per_view"], drive=s["drive"])
    gt_pred_diff_gif(s["gt_frames_png"], pred_pngs, rd.path("gt_pred_diff.gif"), n_frames)
    save_overlay_gif(s["gt_roll"], pred_roll, rd.path("overlay.gif"), fit_frames=n_frames)

    result = {
        "scenario": "ours_image_Escalar",
        "gt": {"E": gt_E, "logE": cfg.gt.logE, "nu": cfg.gt.nu, "vel": list(cfg.gt.vel)},
        "init": {"logE": cfg.init_logE, "nu": cfg.init_nu},
        "w_img": cfg.render.w_img, "w_alp": cfg.render.w_alp, "n_frames": n_frames,
        "camera": (cfg.render.cameras or cfg.render.camera),
        "bg_image": cfg.render.bg_image,
        "best": {**best, "iter": min_idx, "loss": float(losses[min_idx])},
        "final": e_s[-1],
        "rel_err_E": rel,
        "losses_phys": [float(l) for l in losses],
        "E_traj": [d.get("Youngs modulus") for d in e_s],
        "nu_traj": [d.get("Poisson ratio") for d in e_s],
        "wall_time_s": time.time() - t0,
    }
    with open(rd.path("result.json"), "w") as f:
        json.dump(result, f, indent=2)
    draw_curve([float(l) for l in losses], rd.root, name="loss_phys")
    plot_param_traj(e_s, gt_E, cfg.gt.nu, rd.path("param_traj.png"))
    build_panel(rd.root)
    print(f"[img-Escalar] DONE: GT E={gt_E:.3g} -> best {best['Youngs modulus']:.4g} "
          f"(rel err {rel:.2%}), wall {time.time() - t0:.0f}s -> {rd.root}")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

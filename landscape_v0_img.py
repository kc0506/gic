#!/usr/bin/env python
# coding=utf-8
"""landscape_v0_img: image-loss landscape over scalar v0, to diagnose observability.

When a v0 recovery fails (e.g. uniform-x on the front camera came back at 133%), the
question is WHY: is the loss landscape flat/degenerate along that axis (the camera
can't see the motion -> no gradient signal, any optimizer is doomed), or is there a
clean minimum at GT that the optimizer simply missed (lr / init)? This sweeps each
v0 axis through the GT value (others held at GT), recomputes the SAME image loss the
fit uses (no grad, forward rollout + render + MSE vs the GT frames), and plots the
three 1-D slices with a marker at GT. A clean V at GT = observable; a flat line =
not. Run for --render.gaussians fullres AND pseudo to also isolate the render.

Usage (gic env, gic repo root):
  python landscape_v0_img.py --scene.cache <cache.pt> --gt.vel 0.5 0 0 --render.camera front
"""
from ours.gpu import pick_gpu

pick_gpu()

import time
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

from ours.config import FramesCfg, GTCfg, RenderCfg, SceneCfg, TrainCfg, make_phys_args
from ours.imgloss import render_pred_frames, setup_image_scene
from ours.rundir import RunDir
from ours.scene import rollout_collect_surfaces, set_params


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=GTCfg)
    render: RenderCfg = field(default_factory=RenderCfg)
    frames: FramesCfg = field(default_factory=FramesCfg)
    lo: float = -0.6
    hi: float = 1.4
    n_pts: int = 21
    axes: str = "x"
    """which v0 axes to sweep (csv of x/y/z); default just x"""
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config, rd: RunDir) -> None:
    t0 = time.time()
    phys_args = make_phys_args(cfg.scene, TrainCfg(), cfg.frames, cfg.gt.logE, cfg.gt.nu)
    n_frames = phys_args.n_frames if cfg.frames.n_frames is None else cfg.frames.n_frames

    s = setup_image_scene(cfg.scene, cfg.gt, cfg.render, phys_args, n_frames)
    est, gaussians, pipe = s["est"], s["gaussians"], s["pipe"]
    gt_pngs = [g.astype(np.float32) for g in s["gt_frames_png"]]
    pose_cam, bg_per_view, drive = s["pose_cam"], s["bg_per_view"], s["drive"]
    gt_v = np.array(cfg.gt.vel, dtype=np.float64)
    print(f"[landscape] GT v0={gt_v.tolist()}, cam {cfg.render.camera}, "
          f"render {cfg.render.gaussians}, {n_frames}f, sweep [{cfg.lo},{cfg.hi}]x{cfg.n_pts}")

    def img_loss_at(v0) -> float:
        set_params(est, cfg.gt.logE, cfg.gt.nu, list(v0))
        est.img_loss = False
        est.max_f = n_frames
        roll = rollout_collect_surfaces(est)
        pred = render_pred_frames(roll, gaussians, pipe, pose_cam,
                                  view=cfg.render.camera, bg_per_view=bg_per_view, drive=drive)
        return float(np.mean([(p.astype(np.float32) - g) ** 2 for p, g in zip(pred, gt_pngs)]))

    vals = np.linspace(cfg.lo, cfg.hi, cfg.n_pts)
    name2ax = {"x": 0, "y": 1, "z": 2}
    sweep = [a.strip() for a in cfg.axes.split(",") if a.strip()]
    curves = {}
    for name in sweep:
        ax = name2ax[name]
        losses = []
        for v in vals:
            cand = gt_v.copy()
            cand[ax] = v
            losses.append(img_loss_at(cand))
        curves[name] = losses
        gt_loss = losses[int(np.argmin(np.abs(vals - gt_v[ax])))]
        print(f"[landscape] axis {name}: min@v={vals[int(np.argmin(losses))]:+.2f} "
              f"(loss {min(losses):.4g}); GT v={gt_v[ax]:+.2f} loss~{gt_loss:.4g}; "
              f"range {min(losses):.4g}..{max(losses):.4g}")

    fig, axs = plt.subplots(1, len(sweep), figsize=(5 * len(sweep), 4), squeeze=False)
    for i, name in enumerate(sweep):
        axs[0][i].plot(vals, curves[name], "-o", ms=3)
        axs[0][i].axvline(gt_v[name2ax[name]], color="r", ls="--",
                          label=f"GT={gt_v[name2ax[name]]:+.2f}")
        axs[0][i].set_title(f"sweep v0_{name} (others@GT)")
        axs[0][i].set_xlabel(f"v0_{name}")
        axs[0][i].set_yscale("log")
        axs[0][i].legend()
        axs[0][i].grid(alpha=0.3)
    axs[0][0].set_ylabel("image loss (MSE vs GT frames)")
    fig.suptitle(f"v0 image-loss landscape | {cfg.render.gaussians} | cam {cfg.render.camera} "
                 f"| GT {gt_v.tolist()}")
    fig.tight_layout()
    fig.savefig(rd.path("landscape.png"), dpi=110)
    np.savez(rd.path("landscape.npz"), vals=vals, gt_v=gt_v, **curves)
    print(f"[landscape] DONE wall {time.time()-t0:.0f}s -> {rd.root}/landscape.png")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

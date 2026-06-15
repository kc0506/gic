#!/usr/bin/env python
# coding=utf-8
"""diff_intensity_img: per-frame |pred-GT| intensity maps for a few v0 settings.

Sanity check for the front-with-bg +x failure: the forward landscape says v0_x=0.5 is
a deep minimum, yet the autograd fit stalls (x drifts negative, y drifts up). This
renders the SAME image loss as a set of per-frame absolute-difference intensity maps
(|pred-GT| summed over channels) for each probe v0, and prints the mean abs-diff per
frame -- so you can SEE/compare whether walking +x actually shrinks the residual more
than the y-drift the optimizer chose. No training, just forward rollout + render.

Usage (gic env, gic repo root):
  python diff_intensity_img.py --scene.cache <cache.pt> --render.bg-image scene \
      --render.camera front --gt.vel 0.5 0 0
"""
from ours.gpu import pick_gpu

pick_gpu()

import time
from dataclasses import dataclass, field
from typing import List, Optional

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
    probes: str = "0,0,0 ; 0,0.3,0 ; 0.5,0,0"
    """';'-separated v0 vectors to probe (default: no-motion / stalled-y / GT-x)"""
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

    probes = [[float(x) for x in p.strip().split(",")] for p in cfg.probes.split(";") if p.strip()]
    print(f"[diff] GT v0={list(cfg.gt.vel)}, cam {cfg.render.camera}, render {cfg.render.gaussians}, "
          f"{n_frames}f, probes={probes}")

    rows, labels, mean_curves = [], [], {}
    for v0 in probes:
        set_params(est, cfg.gt.logE, cfg.gt.nu, v0)
        est.img_loss = False
        est.max_f = n_frames
        roll = rollout_collect_surfaces(est)
        pred = render_pred_frames(roll, gaussians, pipe, pose_cam,
                                  view=cfg.render.camera, bg_per_view=bg_per_view, drive=drive)
        diffs = [np.abs(p.astype(np.float32) - g).sum(axis=2) for p, g in zip(pred, gt_pngs)]
        per_frame_mean = [float(d.mean()) for d in diffs]
        mean_curves[str(v0)] = per_frame_mean
        rows.append(diffs)
        labels.append(f"v0={v0}  mean|d|={np.mean(per_frame_mean):.2f}")
        print(f"[diff] v0={v0}: per-frame mean|diff| = "
              f"{[round(x, 1) for x in per_frame_mean]}  (avg {np.mean(per_frame_mean):.2f})")

    vmax = max(d.max() for row in rows for d in row) or 1.0
    nf = n_frames
    fig, axs = plt.subplots(len(rows), nf, figsize=(1.6 * nf, 1.8 * len(rows)), squeeze=False)
    for r, (diffs, lab) in enumerate(zip(rows, labels)):
        for f in range(nf):
            axs[r][f].imshow(diffs[f], cmap="inferno", vmin=0, vmax=vmax)
            axs[r][f].set_xticks([]); axs[r][f].set_yticks([])
            if f == 0:
                axs[r][f].set_ylabel(lab, fontsize=8)
            if r == 0:
                axs[r][f].set_title(f"f{f}", fontsize=8)
    fig.suptitle(f"per-frame |pred-GT| intensity | cam {cfg.render.camera} | "
                 f"bg={cfg.render.bg_image} | GT v0={list(cfg.gt.vel)}")
    fig.tight_layout()
    fig.savefig(rd.path("diff_intensity.png"), dpi=110)
    np.savez(rd.path("diff_intensity.npz"), **{str(k): np.array(v) for k, v in mean_curves.items()})
    print(f"[diff] DONE wall {time.time()-t0:.0f}s -> {rd.root}/diff_intensity.png")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

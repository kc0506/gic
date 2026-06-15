#!/usr/bin/env python
# coding=utf-8
"""poster_traj_cloud: matplotlib 3D point-cloud overlay of the GT rollout.

The trajectory counterpart to poster_overlay: scatter the simulated particles at T
timesteps in ONE 3D plot, later frames more opaque (and warmer-coloured) so motion
reads. z is the vertical (physics +z = up; matplotlib's 3rd coord is already vertical)
and the axes/box are hidden. Camera-independent (it's a 3D plot, not a render).

Usage (gic env, gic repo root):
  python poster_traj_cloud.py --scene.cache <cache.pt> --gt.vel 0.5 0 0
"""
from ours.gpu import pick_gpu

pick_gpu()

import os
import time
from dataclasses import dataclass, field
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

from ours.config import FramesCfg, GTCfg, RenderCfg, SceneCfg, TrainCfg, make_phys_args
from ours.imgloss import setup_image_scene
from ours.rundir import RunDir

ASSETS = "/tmp2/b10401006/ev-project/generative-phys/poster/assets"


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=lambda: GTCfg(vel=(0.5, 0.0, 0.0)))
    frames: FramesCfg = field(default_factory=FramesCfg)
    n_overlay: int = 6
    op_lo: float = 0.15
    """alpha of the earliest frame (latest is 1.0)"""
    cmap: str = "plasma"
    elev: float = 18.0
    azim: float = -70.0
    psize: float = 3.0
    out_name: str = "traj_cloud"
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config, rd: RunDir) -> None:
    t0 = time.time()
    phys_args = make_phys_args(cfg.scene, TrainCfg(), cfg.frames, cfg.gt.logE, cfg.gt.nu)
    n_frames = phys_args.n_frames if cfg.frames.n_frames is None else cfg.frames.n_frames
    render = RenderCfg(gaussians="pseudo")  # light: no 1M-splat load, we only need gt_roll
    s = setup_image_scene(cfg.scene, cfg.gt, render, phys_args, n_frames)
    roll = [p.detach().cpu().numpy() for p in s["gt_roll"]]   # T x (N,3) sim-space particles

    idx = [int(round(i)) for i in np.linspace(0, len(roll) - 1, cfg.n_overlay)]
    ops = np.linspace(cfg.op_lo, 1.0, len(idx))
    cmap = plt.get_cmap(cfg.cmap)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for k, t in enumerate(idx):
        P = roll[t]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=cfg.psize, linewidths=0,
                   color=cmap(k / max(len(idx) - 1, 1)), alpha=float(ops[k]),
                   depthshade=False)
    # equal aspect from data; z is the vertical axis (= physics +z up)
    allP = np.concatenate([roll[t] for t in idx], 0)
    ctr = allP.mean(0)
    r = (allP.max(0) - allP.min(0)).max() / 2.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r)
    ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=cfg.elev, azim=cfg.azim)
    ax.set_axis_off()
    fig.tight_layout(pad=0)
    for path in (rd.path(f"{cfg.out_name}.png"), os.path.join(ASSETS, f"{cfg.out_name}.png")):
        fig.savefig(path, dpi=150, transparent=True, bbox_inches="tight")
    print(f"[cloud] frames {idx} (alpha {cfg.op_lo}->1.0), cmap {cfg.cmap}, "
          f"view elev{cfg.elev}/azim{cfg.azim}, GT v0={list(cfg.gt.vel)}")
    print(f"[cloud] DONE wall {time.time()-t0:.0f}s -> {ASSETS}/{cfg.out_name}.png")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

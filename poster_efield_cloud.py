#!/usr/bin/env python
# coding=utf-8
"""poster_efield_cloud: 3D point-cloud overlay of a traj rolled out under an E-FIELD.

Same stroboscopic overlay as poster_traj_cloud, but the GT is rolled out with a
spatially-varying E field (uniform/ramp/circular, like fit_traj_joint's GT) so it can
visualize runs such as efcirc_uniformv0 (circular E + uniform v0). Later frames more
opaque + warmer; z vertical (physics +z up); axes hidden. Needs GPU (MPM rollout).

Usage (gic env, gic repo root):
  python poster_efield_cloud.py --scene.cache <tele.pt> --gt-kind circular \
      --gt-ramp 4.5 5.5 --efield.res 16x16x16 --obs ym
"""
from ours.gpu import pick_gpu

pick_gpu()

import os
import time
from dataclasses import dataclass, field
from typing import Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

from ours.config import FieldCfg, FramesCfg, GTCfg, SceneCfg, TrainCfg, make_phys_args
from ours.fields import EVoxelField, fill_circular_grid
from ours.scene import build_anchored_scene, rollout_collect_surfaces, set_params

ASSETS = "/tmp2/b10401006/ev-project/generative-phys/poster/assets"
VDIR = {"xp": [0.5, 0, 0], "xm": [-0.5, 0, 0], "yp": [0, 0.5, 0], "ym": [0, -0.5, 0]}


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=GTCfg)
    efield: FieldCfg = field(default_factory=lambda: FieldCfg(res="16x16x16"))
    frames: FramesCfg = field(default_factory=lambda: FramesCfg(n_frames=8))
    gt_kind: str = "circular"
    gt_ramp: Tuple[float, float] = (4.5, 5.5)
    obs: str = "ym"
    gt_frames: int = 8
    n_overlay: int = 6
    op_lo: float = 0.15
    cmap: str = "plasma"
    elev: float = 18.0
    azim: float = -70.0
    psize: float = 3.0
    out_name: str = "efcirc_cloud"
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config) -> None:
    t0 = time.time()
    v0 = VDIR[cfg.obs] if cfg.obs in VDIR else [float(x) for x in cfg.obs.split(",")]
    res = (tuple(int(p) for p in cfg.efield.res.lower().split("x"))
           if "x" in cfg.efield.res else (int(cfg.efield.res),) * 3)
    phys_args = make_phys_args(cfg.scene, TrainCfg(), cfg.frames, cfg.gt.logE, cfg.gt.nu)
    phys_args.n_frames = cfg.gt_frames
    scene = build_anchored_scene(cfg.scene.cache, cfg.scene.rot_z_deg, phys_args,
                                 cfg.scene.anchor_mass_scale, cfg.gt_frames,
                                 inject_pvol=cfg.scene.inject_pvol,
                                 ti_mem_frac=cfg.scene.ti_mem_frac)
    est, xyz, free = scene["est"], scene["xyz"], scene["free"]
    z_lo, z_hi = scene["z_lo"], scene["z_hi"]

    gt_field = EVoxelField(scene["aabb"].cpu(), res=res)
    if cfg.gt_kind == "uniform":
        gt_field.set_uniform_(cfg.gt.logE)
    elif cfg.gt_kind == "ramp":
        gt_field.fill_ramp_(cfg.gt_ramp[0], cfg.gt_ramp[1], z_lo, z_hi, scene["flip_z"])
    else:
        fill_circular_grid(gt_field, xyz.cpu(), free.cpu(), z_lo, z_hi,
                           cfg.gt_ramp[0], cfg.gt_ramp[1])
    est.set_E_field(gt_field, xyz, lr=0.0)
    set_params(est, cfg.gt.logE, cfg.gt.nu, v0)
    roll = [p.detach().cpu().numpy() for p in rollout_collect_surfaces(est)]

    idx = [int(round(i)) for i in np.linspace(0, len(roll) - 1, cfg.n_overlay)]
    ops = np.linspace(cfg.op_lo, 1.0, len(idx))
    cmap = plt.get_cmap(cfg.cmap)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    for k, t in enumerate(idx):
        P = roll[t]
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=cfg.psize, linewidths=0,
                   color=cmap(k / max(len(idx) - 1, 1)), alpha=float(ops[k]), depthshade=False)
    allP = np.concatenate([roll[t] for t in idx], 0)
    ctr = allP.mean(0); r = (allP.max(0) - allP.min(0)).max() / 2.0
    ax.set_xlim(ctr[0] - r, ctr[0] + r); ax.set_ylim(ctr[1] - r, ctr[1] + r)
    ax.set_zlim(ctr[2] - r, ctr[2] + r)
    ax.set_box_aspect((1, 1, 1)); ax.view_init(elev=cfg.elev, azim=cfg.azim); ax.set_axis_off()
    fig.tight_layout(pad=0)
    fig.savefig(os.path.join(ASSETS, f"{cfg.out_name}.png"), dpi=150,
                transparent=True, bbox_inches="tight")
    print(f"[efield-cloud] {cfg.gt_kind} E {cfg.gt_ramp} res{res} | v0 {v0} | frames {idx}")
    print(f"[efield-cloud] DONE wall {time.time()-t0:.0f}s -> {ASSETS}/{cfg.out_name}.png")


def main() -> None:
    cfg = tyro.cli(Config)
    from ours.rundir import RunDir
    RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg)


if __name__ == "__main__":
    main()

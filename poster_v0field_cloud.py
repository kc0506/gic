#!/usr/bin/env python
# coding=utf-8
"""poster_v0field_cloud: GT vs PRED v0-field rollout, each as a stroboscopic 3D cloud.

For a v0-field run (e.g. field_a16_bend_s6): rebuild the GT field analytically from
--gt-v0-variant, load the PRED field from the run ckpt's v0_field_grid, roll each out
under the fixed E, and emit ONE overlay cloud per side (later frames opaque+warm,
z up, axes hidden). Needs GPU.

Usage (gic env, gic repo root):
  python poster_v0field_cloud.py --scene.cache <tele.pt> --ckpt <run>/ckpt_latest_stage0.pt \
      --gt-v0-variant true_bend --gt-v0-scale 6 --scene.rot-z-deg 67.6 --fix-logE 4.0
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
import torch
import tyro

from ours.config import FramesCfg, GTCfg, SceneCfg, TrainCfg, make_phys_args
from ours.fields import V0VoxelField, fill_profile_grid
from ours.scene import build_anchored_scene, rollout_collect_surfaces, set_params

ASSETS = "/tmp2/b10401006/ev-project/generative-phys/poster/assets"


@dataclass
class Config:
    scene: SceneCfg
    ckpt: str
    gt: GTCfg = field(default_factory=GTCfg)
    frames: FramesCfg = field(default_factory=lambda: FramesCfg(n_frames=14))
    gt_v0_variant: str = "true_bend"
    gt_v0_scale: float = 6.0
    res: str = "4x4x16"
    fix_logE: float = 4.0
    gt_frames: int = 14
    n_overlay: int = 6
    op_lo: float = 0.15
    cmap: str = "plasma"
    elev: float = 18.0
    azim: float = -70.0
    psize: float = 3.0
    out_name: str = "field_a16_bend_s6_cloud"
    run_label: str = ""
    out: Optional[str] = None


def _cloud(roll, path, cfg):
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
    fig.savefig(path, dpi=150, transparent=True, bbox_inches="tight")
    plt.close(fig)
    return idx


def run(cfg: Config) -> None:
    t0 = time.time()
    res = tuple(int(p) for p in cfg.res.lower().split("x"))   # (rx,ry,rz)
    phys_args = make_phys_args(cfg.scene, TrainCfg(), cfg.frames, cfg.fix_logE, cfg.gt.nu)
    phys_args.n_frames = cfg.gt_frames
    scene = build_anchored_scene(cfg.scene.cache, cfg.scene.rot_z_deg, phys_args,
                                 cfg.scene.anchor_mass_scale, cfg.gt_frames,
                                 inject_pvol=cfg.scene.inject_pvol,
                                 ti_mem_frac=cfg.scene.ti_mem_frac)
    est, xyz = scene["est"], scene["xyz"]
    z_lo, z_hi, flip_z, aabb = scene["z_lo"], scene["z_hi"], scene["flip_z"], scene["aabb"]

    def rollout_with(grid):
        f = V0VoxelField(aabb.cpu(), res=res)
        with torch.no_grad():
            f.grid.copy_(grid)
        est.set_v0_field(f, xyz, lr=0.0)
        set_params(est, cfg.fix_logE, cfg.gt.nu, [0.0, 0.0, 0.0])
        return [p.detach().cpu().numpy() for p in rollout_collect_surfaces(est)]

    # GT field (analytic) + PRED field (ckpt)
    gt_f = V0VoxelField(aabb.cpu(), res=res)
    fill_profile_grid(gt_f, cfg.gt_v0_variant, cfg.gt_v0_scale, z_lo, z_hi, flip=flip_z)
    gt_roll = rollout_with(gt_f.grid.detach())
    ck = torch.load(cfg.ckpt, map_location="cpu", weights_only=False)
    pred_grid = ck["v0_field_grid"] if isinstance(ck, dict) else ck
    pred_roll = rollout_with(torch.as_tensor(pred_grid).float())

    gi = _cloud(gt_roll, os.path.join(ASSETS, f"{cfg.out_name}_gt.png"), cfg)
    _cloud(pred_roll, os.path.join(ASSETS, f"{cfg.out_name}_pred.png"), cfg)
    print(f"[v0field-cloud] variant {cfg.gt_v0_variant} x{cfg.gt_v0_scale}, res{res}, "
          f"E1e{cfg.fix_logE}, frames {gi}")
    print(f"[v0field-cloud] DONE wall {time.time()-t0:.0f}s -> "
          f"{ASSETS}/{cfg.out_name}_{{gt,pred}}.png")


def main() -> None:
    cfg = tyro.cli(Config)
    from ours.rundir import RunDir
    RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg)


if __name__ == "__main__":
    main()

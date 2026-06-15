#!/usr/bin/env python
# coding=utf-8
"""poster_overlay: stroboscopic multi-frame overlay of the moving object on ONE bg.

Renders the GT rollout, then composites T object-only frames (real texture, top_k
driven) over a single static background -> one image showing the motion trail. The
trajectory is per-frame separable, so this is just: bg once + object at each picked
frame, alpha-composited (opacity ramps faint->solid so motion direction reads).

Usage (gic env, gic repo root):
  python poster_overlay.py --scene.cache <cache.pt> --render.camera diag45 --gt.vel 0.5 0 0
"""
from ours.gpu import pick_gpu

pick_gpu()

import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
import tyro
from PIL import Image

from utils.general_utils import inverse_sigmoid

from ours.config import FramesCfg, GTCfg, RenderCfg, SceneCfg, TrainCfg, make_phys_args
from ours.imgloss import _png, render_drive_frame, setup_image_scene
from ours.rundir import RunDir

ASSETS = "/tmp2/b10401006/ev-project/generative-phys/poster/assets"


@dataclass
class Config:
    scene: SceneCfg
    gt: GTCfg = field(default_factory=lambda: GTCfg(vel=(0.5, 0.0, 0.0)))
    render: RenderCfg = field(default_factory=lambda: RenderCfg(camera="diag45"))
    frames: FramesCfg = field(default_factory=FramesCfg)
    n_overlay: int = 6
    """how many evenly-spaced frames to superimpose"""
    op_lo: float = 0.35
    """opacity of the earliest overlaid frame (latest is 1.0); ramps for motion sense"""
    out_name: str = "overlay_object"
    run_label: str = ""
    out: Optional[str] = None


def _zero(g, mask):  # opacity -> ~0 where mask True; returns a fresh opacity tensor
    op = g._opacity.detach().clone()
    op[mask] = inverse_sigmoid(torch.tensor(1e-6, device=op.device))
    return op


def run(cfg: Config, rd: RunDir) -> None:
    t0 = time.time()
    phys_args = make_phys_args(cfg.scene, TrainCfg(), cfg.frames, cfg.gt.logE, cfg.gt.nu)
    n_frames = phys_args.n_frames if cfg.frames.n_frames is None else cfg.frames.n_frames
    # force fullres + scene bg so we have the real object texture + a real background
    cfg.render.gaussians = "fullres"
    cfg.render.bg_image = "scene"
    s = setup_image_scene(cfg.scene, cfg.gt, cfg.render, phys_args, n_frames)
    gaussians, pipe, drive = s["gaussians"], s["pipe"], s["drive"]
    gt_roll, pose_cam = s["gt_roll"], s["pose_cam"]

    cache = torch.load(cfg.scene.cache, map_location="cpu", weights_only=False)
    sim_mask = cache["disc"]["sim_mask"].to(gaussians.get_xyz.device)

    base_op = gaussians._opacity.detach().clone()
    # --- static background: object opacity zeroed, render at rest pose ---
    gaussians._opacity = _zero(gaussians, sim_mask)
    bg_img, _ = render_drive_frame(gaussians, drive, gt_roll[0], pipe, pose_cam)
    bg = bg_img.clamp(0, 1)

    # --- object-only per frame: bg opacity zeroed ---
    gaussians._opacity = base_op.clone()
    gaussians._opacity = _zero(gaussians, ~sim_mask)

    idx = [int(round(i)) for i in np.linspace(0, len(gt_roll) - 1, cfg.n_overlay)]
    ops = np.linspace(cfg.op_lo, 1.0, len(idx))
    comp = bg.clone()
    for k, t in enumerate(idx):
        obj, alp = render_drive_frame(gaussians, drive, gt_roll[t], pipe, pose_cam)
        a = alp.clamp(0, 1) * float(ops[k])           # (1,H,W) blend weight
        comp = obj.clamp(0, 1) * a + comp * (1.0 - a)
    Image.fromarray(_png(comp)).save(rd.path(f"{cfg.out_name}.png"))
    # also drop a copy straight into poster/assets
    import os
    Image.fromarray(_png(comp)).save(os.path.join(ASSETS, f"{cfg.out_name}.png"))
    print(f"[overlay] frames {idx} (op {cfg.op_lo}->1.0), cam {cfg.render.camera}, "
          f"GT v0={list(cfg.gt.vel)}")
    print(f"[overlay] DONE wall {time.time()-t0:.0f}s -> {ASSETS}/{cfg.out_name}.png")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

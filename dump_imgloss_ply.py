#!/usr/bin/env python
# coding=utf-8
"""Dump the fit_image_* first-frame 3DGS as .ply for visual debugging.

Saves the rest-pose object gaussians (pseudo gaussians at the MPM particles) and
the 'scene' background gaussians (original PhysDreamer scene, moving-object
region removed, mapped into our normalized+rotated sim space) so they can be
loaded together in a viewer to check alignment/scale/orientation. Also prints the
static camera pose. No taichi / no rollout (frame 0 = rest = the cache xyz).

Usage (gic env, gic repo root):
  python dump_imgloss_ply.py --scene.cache <cache.pt> --scene.rot-z-deg 67.6
"""
from ours.gpu import pick_gpu

pick_gpu()

import os
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch
import tyro

from ours.config import SceneCfg
from ours.geom import rot_xyz as rot_z
from ours.imgloss import (_build_bg_gaussians, _load_orig, _orig_xyz_norm,
                          build_pseudo_gaussians, make_pose)
from ours.rundir import RunDir
from ours.scene import load_our_scene


@dataclass
class Config:
    scene: SceneCfg
    camera: str = "front"
    run_label: str = ""
    out: Optional[str] = None


def run(cfg: Config, rd: RunDir) -> None:
    xyz, anchor_mask = load_our_scene(cfg.scene.cache)
    if cfg.scene.rot_z_deg:
        xyz = rot_z(xyz, cfg.scene.rot_z_deg)
    cache = torch.load(cfg.scene.cache, map_location="cpu", weights_only=False)
    ghost = (cache["disc"]["sim_xyzs"] == 0).all(dim=1)
    pv = torch.from_numpy(cache["disc"]["points_vol"]).float()[~ghost]
    shift = cache["disc"]["shift"].reshape(-1)
    cscale = float(cache["disc"]["scale"])

    orig = _load_orig(cache["dataset_dir"])
    oxyz = _orig_xyz_norm(orig, shift, cscale, cfg.scene.rot_z_deg)
    pseudo, idx = build_pseudo_gaussians(xyz, pv, orig, oxyz)
    sim_mask = cache["disc"]["sim_mask"].to(oxyz.device)
    bg = _build_bg_gaussians(orig, oxyz, cscale, sim_mask, cfg.scene.rot_z_deg)

    pseudo.save_ply(rd.path("obj_frame0.ply"))
    bg.save_ply(rd.path("bg_scene.ply"))
    # also the untransformed original, for reference (raw PhysDreamer space)
    orig.save_ply(rd.path("orig_raw.ply"))

    def box(name, t):
        lo, hi = t.min(0).values, t.max(0).values
        print(f"  {name:10s} N={t.shape[0]:>7d}  "
              f"min[{lo[0]:+.3f},{lo[1]:+.3f},{lo[2]:+.3f}] "
              f"max[{hi[0]:+.3f},{hi[1]:+.3f},{hi[2]:+.3f}]")

    R, T = make_pose(cfg.camera)
    C = -R @ T  # camera centre in world (R here = R_w2c.T, so -R@T = C)
    print(f"[dump] camera '{cfg.camera}': centre {np.round(C, 3).tolist()}, "
          f"look +y/+x depending on view (see make_pose)")
    print("[dump] bounding boxes (normalized sim space):")
    box("obj(xyz)", xyz.detach().cpu())
    box("oxyz", oxyz.detach().cpu())
    box("orig_raw", orig._xyz.detach().cpu())
    print(f"[dump] cache shift={shift.tolist()}, scale={cscale:.4f}, rot_z={cfg.scene.rot_z_deg}")
    print(f"[dump] bg: removed {int(sim_mask.sum())} object gaussians (sim_mask) "
          f"of {oxyz.shape[0]} orig; rot_z quats={'yes' if cfg.scene.rot_z_deg else 'no'}")
    print(f"[dump] DONE -> {rd.root}")
    print(f"  obj_frame0.ply / bg_scene.ply : normalized+rotated sim space (camera renders THIS)")
    print(f"  orig_raw.ply                  : raw PhysDreamer space (pre shift/scale/rot)")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

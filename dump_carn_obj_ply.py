#!/usr/bin/env python
# coding=utf-8
"""Dump the REAL full-res object (and full scene) as .ply in normalized sim space.

For onboarding a new scene's rotation: the camera renders the normalized+rot_z sim
space, so to pick --scene.rot-z-deg you want to SEE the real anisotropic object in
that space at rot_z=0 and align its principal axes to x/y (like telephone -> -22.4).
This dumps the full-res object subset (sim_mask, real shape/texture) + the full
scene, both at rot_z=0, and prints the object's xy PCA principal angle as a start.

Usage (gic env, gic repo root):
  python dump_carn_obj_ply.py --scene.cache <cache.pt>
"""
from ours.gpu import pick_gpu

pick_gpu()

from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro

from scene.gaussian_model import GaussianModel

from ours.config import SceneCfg
from ours.geom import rot_xyz as rot_z
from ours.imgloss import _load_orig, _orig_xyz_norm, build_fullres_gaussians
from ours.rundir import RunDir
from ours.scene import load_our_scene


@dataclass
class Config:
    scene: SceneCfg
    run_label: str = ""
    out: Optional[str] = None


def _subset(g: GaussianModel, mask: torch.Tensor) -> GaussianModel:
    s = GaussianModel(g.max_sh_degree)
    s._xyz = g._xyz[mask].clone()
    s._features_dc = g._features_dc[mask].clone()
    s._features_rest = g._features_rest[mask].clone()
    s._scaling = g._scaling[mask].clone()
    s._rotation = g._rotation[mask].clone()
    s._opacity = g._opacity[mask].clone()
    s.active_sh_degree = g.active_sh_degree
    return s


def run(cfg: Config, rd: RunDir) -> None:
    xyz, anchor_mask = load_our_scene(cfg.scene.cache)  # particle space (rot applied below if any)
    cache = torch.load(cfg.scene.cache, map_location="cpu", weights_only=False)
    shift = cache["disc"]["shift"].reshape(-1)
    cscale = float(cache["disc"]["scale"])
    orig = _load_orig(cache["dataset_dir"])
    # rot_z=0: native orientation in normalized sim space
    oxyz = _orig_xyz_norm(orig, shift, cscale, 0.0)
    scene_g = build_fullres_gaussians(orig, oxyz, cscale, 0.0)
    sim_mask = cache["disc"]["sim_mask"]
    obj_g = _subset(scene_g, sim_mask)

    scene_g.save_ply(rd.path("carn_scene_norm.ply"))
    obj_g.save_ply(rd.path("carn_obj_norm.ply"))

    # PCA of the object xy positions -> principal angle (deg, CCW from +x)
    p = obj_g.get_xyz.detach().cpu().numpy()[:, :2]
    pc = p - p.mean(0)
    _, _, vt = np.linalg.svd(pc, full_matrices=False)
    ang = float(np.degrees(np.arctan2(vt[0, 1], vt[0, 0])))
    o = obj_g.get_xyz.detach().cpu()
    print(f"[dump] scene N={scene_g.get_xyz.shape[0]}, object(sim_mask) N={obj_g.get_xyz.shape[0]}")
    print(f"[dump] object bbox (norm sim space): "
          f"min{np.round(o.min(0).values.numpy(), 3).tolist()} "
          f"max{np.round(o.max(0).values.numpy(), 3).tolist()}")
    print(f"[dump] object xy PCA principal axis = {ang:.1f} deg CCW from +x")
    print(f"[dump]   -> to align that axis onto +x, try --scene.rot-z-deg {-ang:.1f} "
          f"(or {-ang + 90:.1f} to put it on +y); sign/quadrant: verify in viewer")
    print(f"[dump] DONE -> {rd.root}")
    print(f"  carn_obj_norm.ply   : real full-res OBJECT only (align THIS in the viewer)")
    print(f"  carn_scene_norm.ply : full scene for context (bg + object), same space")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

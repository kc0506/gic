#!/usr/bin/env python
# coding=utf-8
"""probe_carnation_cam: pick an observable+framed camera for a NEW scene (carnation).

Onboarding a scene to the image pipeline needs (rotation, view, fov, pan). Physics
reuses telephone.json; the excitation direction comes from the probe. This tool
renders the FULL-RES scene statically (no MPM) under a 3-axis Euler rotation of the
scene (positions AND gaussian quaternions, about the sim-space centre 0.5) from a
sweep of camera elevation x fov, each cell as two columns: t=0 and a synthetic +x
displacement of the free particles -- so you SEE both the framing and whether the
motion projects into the image plane. Eye-level often can't see a cluttered scene;
sweep elevation to look DOWN past the artefacts. Pick the cell, then port the chosen
rotation/camera into the fit_image_* pipeline.

Usage (gic env, gic repo root):
  python probe_carnation_cam.py --scene.cache <cache.pt> --euler -3.8 -4.32 21.55
"""
from ours.gpu import pick_gpu

pick_gpu()  # pick a free GPU before torch/taichi create a CUDA context

import math
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import torch
import tyro
from PIL import Image

from arguments import PipelineParams
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel

from ours.config import SceneCfg
from ours.gauss_drive import TopKDrive, matrix_to_quaternion, quaternion_multiply
from ours.imgloss import (HW, _load_orig, _orig_xyz_norm, _png, build_fullres_gaussians,
                          lookat_pose, render_drive_frame)
from ours.rundir import RunDir
from ours.scene import load_our_scene

CENTER = 0.5


def euler_matrix(ax: float, ay: float, az: float) -> torch.Tensor:
    """XYZ-extrinsic rotation matrix R = Rz @ Ry @ Rx (apply Rx, then Ry, then Rz), degrees."""
    rx, ry, rz = (math.radians(a) for a in (ax, ay, az))
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    return Rz @ Ry @ Rx


def rot_pos(xyz: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Rotate (N,3) positions about CENTER by R."""
    return (xyz - CENTER) @ R.t().to(xyz.device) + CENTER


def rot_quat(quats: torch.Tensor, R: torch.Tensor) -> torch.Tensor:
    """Left-multiply each [w,x,y,z] gaussian quaternion by the quaternion of R."""
    qR = matrix_to_quaternion(R.to(quats.device)).reshape(1, 4).expand(quats.shape[0], 4)
    return quaternion_multiply(qR, quats)


@dataclass
class Config:
    scene: SceneCfg
    euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """XYZ-extrinsic Euler rotation (deg) of the scene about centre (positions+quats)"""
    elevations: List[float] = field(default_factory=lambda: [30.0, 50.0, 70.0])
    """camera pitch-down angles (deg); 0=eye-level front, 90=straight above"""
    azimuths: List[float] = field(default_factory=lambda: [0.0])
    """CAMERA azimuth angles (deg) -- rotate the VIEWPOINT around the fixed scene
    (NOT the scene); 0 = camera at -y looking +y (front). Sweep to find the view
    that sees the wide face / observes the excitation. Grid rows = azimuth x elevation."""
    fovs: List[float] = field(default_factory=lambda: [0.12, 0.18])
    dist: float = 1.7
    disp: float = 0.12
    """synthetic +x displacement of free particles (sim units) for observability"""
    dump_ply: bool = False
    """also save the euler-rotated object/scene as .ply (same frame as the renders)"""
    run_label: str = ""
    out: str = ""


def _subset(g: GaussianModel, mask: torch.Tensor) -> GaussianModel:
    s = GaussianModel(g.max_sh_degree)
    for a in ("_xyz", "_features_dc", "_features_rest", "_scaling", "_rotation", "_opacity"):
        setattr(s, a, getattr(g, a)[mask].clone())
    s.active_sh_degree = g.active_sh_degree
    return s


def run(cfg: Config, rd: RunDir) -> None:
    from argparse import ArgumentParser
    _p = ArgumentParser()
    pipe = PipelineParams(_p).extract(_p.parse_args([]))

    xyz, anchor_mask = load_our_scene(cfg.scene.cache)
    cache = torch.load(cfg.scene.cache, map_location="cpu", weights_only=False)
    shift = cache["disc"]["shift"].reshape(-1)
    cscale = float(cache["disc"]["scale"])
    orig = _load_orig(cache["dataset_dir"])
    oxyz = _orig_xyz_norm(orig, shift, cscale, 0.0)        # rot=0; apply euler below
    gaussians = build_fullres_gaussians(orig, oxyz, cscale, 0.0)

    R = euler_matrix(*cfg.euler)
    gaussians._xyz = rot_pos(gaussians.get_xyz.detach(), R)
    gaussians._rotation = rot_quat(gaussians.get_rotation.detach(), R)
    xyz = rot_pos(xyz, R)

    sim_mask = cache["disc"]["sim_mask"].cuda()
    top_k_index = cache["disc"]["top_k_index"].cuda()
    drive = TopKDrive(gaussians.get_xyz.detach(), gaussians.get_rotation.detach(),
                      xyz.detach(), top_k_index, sim_mask)
    free = ~anchor_mask
    xyz_disp = xyz.clone()
    xyz_disp[free, 0] += cfg.disp
    print(f"[probe] {gaussians.get_xyz.shape[0]} splats, object={int(sim_mask.sum())}, "
          f"free={int(free.sum())}, euler={cfg.euler}, az={cfg.azimuths}, el={cfg.elevations}")

    if cfg.dump_ply:
        gaussians.save_ply(rd.path("scene_euler.ply"))
        _subset(gaussians, sim_mask.to(gaussians.get_xyz.device)).save_ply(rd.path("obj_euler.ply"))
        print(f"[probe] dumped scene_euler.ply / obj_euler.ply (euler {cfg.euler}, same frame as renders)")

    dummy_img, dummy_alp = torch.zeros(3, HW, HW), np.zeros((1, HW, HW), np.float32)
    rows = []
    for az in cfg.azimuths:
        for el in cfg.elevations:
            cells = []
            for fov in cfg.fovs:
                Rcam, T = lookat_pose(az, el, cfg.dist)
                cam = Camera(colmap_id=0, R=Rcam, T=T, FoVx=fov, FoVy=fov, image=dummy_img,
                             gt_alpha_mask=dummy_alp, image_name="probe", uid=0, fid=0)
                img0, _ = render_drive_frame(gaussians, drive, xyz, pipe, cam)
                img1, _ = render_drive_frame(gaussians, drive, xyz_disp, pipe, cam)
                pair = np.concatenate([_png(img0), _png(img1)], axis=1)
                cells.append(pair)
                Image.fromarray(pair).save(rd.path(f"az{az:g}_el{el:g}_fov{fov:g}.png"))
            rows.append(np.concatenate(cells, axis=1))
    Image.fromarray(np.concatenate(rows, axis=0)).save(rd.path("contact_sheet.png"))
    print(f"[probe] rows=azimuth x elevation {[(a, e) for a in cfg.azimuths for e in cfg.elevations]}, "
          f"cols=fov {cfg.fovs} (cell: t0 | +x disp)")
    print(f"[probe] DONE -> {rd.root}/contact_sheet.png")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out or None, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

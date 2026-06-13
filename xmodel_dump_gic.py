# coding=utf-8
"""gic-side fresh forward for the cross-model comparison (task 2026-06-11).

Alignment with the warp dump (reuse_mpm/explore/xmodel_dump.py):
  E=1e5, nu=0.3, rho=2000, v0=(0,-0.5,0) (zeroed on anchors), gravity=0,
  dx=1/32, dt=(1/30)/64 (mpm_iter_cnt=64, fps=30), same cache particles,
  per-particle p_vol injected from the cache (matches warp mass distribution).
Residual (non-alignable) differences, by design of the experiment:
  - constitutive model: gic elasticity = mu*F*F^T + lam*log(J)*I  vs
    warp 'jelly' = fixed corotated
  - freeze: heavy-anchor emulation (rho x1e4, v0=0) vs exact grid freeze BC
"""
import sys

sys.path.insert(0, "/tmp2/b10401006/ev-project/gic")

import argparse

_ap = argparse.ArgumentParser()
_ap.add_argument("--logE", type=float, default=5.0)
_ap.add_argument("--nu", type=float, default=0.3)
_ap.add_argument("--v0", type=float, nargs=3, default=[0.0, -0.5, 0.0])
_ap.add_argument("--label", type=str, default="tele_E1e5")
_ap.add_argument("--cache", type=str,
                 default="/tmp2/b10401006/ev-project/generative-phys/outputs/forward_gen/06_tele_E1e5/scene_cache.pt")
_ap.add_argument("--shift_cells", type=int, nargs=3, default=[0, 0, 0],
                 help="integer-cell translation (k*dx per axis), matching warp xmodel_dump")
_ap.add_argument("--n_frames", type=int, default=14)
_ap.add_argument("--rot_z_deg", type=float, default=0.0,
                 help="rotate scene about z through (0.5,0.5), matching warp xmodel_dump")
ARGS = _ap.parse_args()

from roundtrip_ours_scene import AnchoredEstimator, load_our_scene, set_params, rollout_collect_surfaces

import json
import time
from argparse import Namespace

import numpy as np
import taichi as ti
import torch

from simulator import Estimator

CACHE = ARGS.cache
OUT = f"/tmp2/b10401006/ev-project/gic/output/xmodel/{ARGS.label}"

phys_args = Namespace(**json.load(open("/tmp2/b10401006/ev-project/gic/config/ours/telephone.json"))["physics"])
phys_args.mpm_iter_cnt = 64   # dt = (1/30)/64, bit-identical to warp
phys_args.fps = 30
phys_args.voxel_size = 0.03125
phys_args.rho = 2000
phys_args.n_frames = ARGS.n_frames

xyz, anchor_mask = load_our_scene(CACHE)
if any(ARGS.shift_cells):
    xyz = xyz + torch.tensor(ARGS.shift_cells, dtype=xyz.dtype, device=xyz.device) * 0.03125
if ARGS.rot_z_deg:
    import math
    _t = math.radians(ARGS.rot_z_deg)
    _c, _s = math.cos(_t), math.sin(_t)
    _x, _y = xyz[:, 0] - 0.5, xyz[:, 1] - 0.5
    xyz = xyz.clone()
    xyz[:, 0] = _c * _x - _s * _y + 0.5
    xyz[:, 1] = _s * _x + _c * _y + 0.5
    print(f"[xmodel gic] rotated z {ARGS.rot_z_deg} deg")
cache = torch.load(CACHE, map_location="cpu", weights_only=False)
ghost = (cache["disc"]["sim_xyzs"] == 0).all(dim=1)
pvol = torch.from_numpy(cache["disc"]["points_vol"]).float()[~ghost].cuda()  # (N,)

ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.25)


class XModelEstimator(AnchoredEstimator):
    """AnchoredEstimator + per-particle p_vol injected from the warp cache."""

    def set_pvol(self, pvol_t: torch.Tensor) -> None:
        self._pvol = pvol_t

    def initialize(self):
        super().initialize()
        write_pvol(self._pvol.cpu().numpy(), self.num_particles[None])
        self.compute_particle_mass()


est = XModelEstimator(
    phys_args, "float32", [xyz.clone() for _ in range(phys_args.n_frames)], surface_index=None,
    init_vol=xyz, dynamic_scene=None, image_scale=1.0, pipeline=None, image_op=None,
)


@ti.kernel
def write_pvol(pv: ti.types.ndarray(), n: ti.i32):
    for p in range(n):
        est.simulator.p_vol[p] = pv[p]


est.set_anchor(anchor_mask, 1e4)
est.set_pvol(pvol)
set_params(est, ARGS.logE, ARGS.nu, list(ARGS.v0))
est.set_stage(Estimator.physical_params_stage)

t0 = time.time()
frames = rollout_collect_surfaces(est)  # list of (N,3) cuda
traj = np.stack([f.cpu().numpy() for f in frames])  # (T, N, 3)

import os

os.makedirs(OUT, exist_ok=True)
np.save(os.path.join(OUT, "gic_traj.npy"), traj)
meta = {
    "side": "gic taichi (elasticity: mu*F*F^T + lam*log(J)*I)",
    "cache": CACHE, "logE": ARGS.logE, "nu": ARGS.nu, "rho": 2000,
    "v0": list(ARGS.v0), "dt": (1 / 30) / 64, "substeps_per_frame": 64,
    "dx": 0.03125, "gravity": [0, 0, 0], "n_frames": ARGS.n_frames,
    "rot_z_deg": ARGS.rot_z_deg,
    "n_particles": int(traj.shape[1]),
    "p_vol": "per-particle, injected from cache (aligned with warp)",
    "shift_cells": list(ARGS.shift_cells),
    "freeze": "heavy-anchor emulation (rho x1e4, anchor v0=0)",
    "final_n_substeps": int(est.simulator.n_substeps[None]),
    "elapsed_s": round(time.time() - t0, 1),
}
with open(os.path.join(OUT, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"[xmodel gic] saved {traj.shape} -> {OUT} "
      f"(n_substeps final={meta['final_n_substeps']}, {meta['elapsed_s']}s)")

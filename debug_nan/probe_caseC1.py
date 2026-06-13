# coding=utf-8
"""Case C1 forward probe: carnation + heavy anchors + E=1e6, v0=(0,-0.5,0).

Steps the simulator substep-by-substep, tracking per-substep:
  - min/max det(F) over particles (elasticity stress has log(J): J<=0 -> NaN)
  - NaN count in v
Stops at first NaN / J<=0 and reports the offending particle.
"""
import sys

sys.path.insert(0, "/tmp2/b10401006/ev-project/gic")

from roundtrip_ours_scene import AnchoredEstimator, load_our_scene, set_params

import json
from argparse import ArgumentParser, Namespace

import taichi as ti
import torch

from simulator import Estimator

parser = ArgumentParser()
parser.add_argument("--cache", default="/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/carnations_ds0.1_g32_k8.pt")
parser.add_argument("--logE", default=6.0, type=float)
parser.add_argument("--nu", default=0.1, type=float)
parser.add_argument("--v0", nargs=3, default=[0.0, -0.5, 0.0], type=float)
parser.add_argument("--anchor_scale", default=1e4, type=float)
parser.add_argument("--max_substeps", default=600, type=int)
args = parser.parse_args()

with open("/tmp2/b10401006/ev-project/gic/config/ours/telephone.json") as f:
    phys_args = Namespace(**json.load(f)["physics"])

xyz, anchor_mask = load_our_scene(args.cache)
ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
est = AnchoredEstimator(
    phys_args, "float32", [xyz.clone() for _ in range(phys_args.n_frames)], surface_index=None,
    init_vol=xyz, dynamic_scene=None, image_scale=1.0, pipeline=None, image_op=None,
)
est.set_anchor(anchor_mask, args.anchor_scale)
set_params(est, args.logE, args.nu, args.v0)
est.set_stage(Estimator.physical_params_stage)
est.geo_loss = False
est.initialize()

sim = est.simulator
n = est.num_particles[None]

min_J = ti.field(ti.f32, shape=())
max_J = ti.field(ti.f32, shape=())
argmin_J = ti.field(ti.i32, shape=())
nan_v = ti.field(ti.i32, shape=())
max_v = ti.field(ti.f32, shape=())


@ti.kernel
def scan_state(col: ti.i32, n_p: ti.i32):
    min_J[None] = 1e30
    max_J[None] = -1e30
    nan_v[None] = 0
    max_v[None] = 0.0
    for p in range(n_p):
        J = sim.F[p, col].determinant()
        old = ti.atomic_min(min_J[None], J)
        if J < old:
            argmin_J[None] = p
        ti.atomic_max(max_J[None], J)
        vel = sim.v[p, col]
        for d in ti.static(range(3)):
            if ti.math.isnan(vel[d]):
                nan_v[None] += 1
            else:
                ti.atomic_max(max_v[None], ti.abs(vel[d]))


print(f"dt={sim.dt[None]:.3e}, n_substeps/frame={sim.n_substeps[None]}, "
      f"E=1e{args.logE:g}, anchor_scale={args.anchor_scale:g}")
chunk = sim.cuda_chunk_size
for s in range(args.max_substeps):
    sim.substep(s, cache=True)  # cache=False skips the col-100->col-0 roll at chunk wrap
    col = (s % chunk) + 1
    scan_state(col, n)
    bad = (nan_v[None] > 0) or (min_J[None] <= 0.0)
    if s % 20 == 0 or bad:
        p_bad = argmin_J[None]
        print(f"substep {s:4d}: minJ={min_J[None]:.4f} (p={p_bad}, anchor={bool(anchor_mask[p_bad])}) "
              f"maxJ={max_J[None]:.4f} nan_v={nan_v[None]} max|v|={max_v[None]:.3f} "
              f"cfl_ok={bool(sim.cfl_satisfy[None])}")
    if bad:
        print("FIRST BAD SUBSTEP — stopping")
        break
else:
    print("no NaN / inversion within probe window")

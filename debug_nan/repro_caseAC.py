# coding=utf-8
"""Cases A & C2 repro: gen GT with gic itself, then ONE staged backward at the
parameter point that historically NaN'd. Same instrumentation as repro_caseB.

  case A : --cache <tele 06 cache> --gen_v0 0 -0.5 0 --fit_v0 0.005721 -0.502585 -0.006430 --fit_logE 5.0
  case C2: --cache <carn cache>    --gen_v0 0 -0.5 0 --fit_v0 0 0 0             --fit_logE 6.0
"""
import sys

sys.path.insert(0, "/tmp2/b10401006/ev-project/gic")

from roundtrip_ours_scene import AnchoredEstimator, load_our_scene, set_params, rollout_collect_surfaces

import json
from argparse import ArgumentParser, Namespace

import taichi as ti
import torch

from simulator import Estimator
from train_dynamic import forward

parser = ArgumentParser()
parser.add_argument("--cache", required=True)
parser.add_argument("--gen_logE", default=5.0, type=float)
parser.add_argument("--gen_v0", nargs=3, default=[0.0, -0.5, 0.0], type=float)
parser.add_argument("--fit_logE", required=True, type=float)
parser.add_argument("--fit_v0", nargs=3, required=True, type=float)
parser.add_argument("--frames", default=4, type=int, help="vel_estimation_frames")
args = parser.parse_args()

with open("/tmp2/b10401006/ev-project/gic/config/ours/telephone.json") as f:
    phys_args = Namespace(**json.load(f)["physics"])

xyz, anchor_mask = load_our_scene(args.cache)
n_frames = phys_args.n_frames

ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
est = AnchoredEstimator(
    phys_args, "float32", [xyz.clone() for _ in range(n_frames)], surface_index=None,
    init_vol=xyz, dynamic_scene=None, image_scale=1.0, pipeline=None, image_op=None,
)
est.set_anchor(anchor_mask, 1e4)

# gen GT with gic itself
set_params(est, args.gen_logE, 0.3, args.gen_v0)
gts = rollout_collect_surfaces(est)
est.gts = gts
est.load_gts(gts)
# how many bit-exact pairs will the fit rollout see? (fit-side positions are
# only known after the fit forward; report gen-vs-init as a proxy at f>=1)
for f in range(1, args.frames):
    exact = (gts[f] == xyz).all(dim=1)
    print(f"gen gt frame {f}: bit-exact-to-init particles = {int(exact.sum())} "
          f"(anchors {int((exact & anchor_mask).sum())}/{int(anchor_mask.sum())})")

nan_cnt = ti.field(ti.i32, shape=())
max_abs = ti.field(ti.f32, shape=())


@ti.kernel
def scan(field: ti.template(), n_p: ti.i32, n_s: ti.i32):
    nan_cnt[None] = 0
    max_abs[None] = 0.0
    for p, s in ti.ndrange(n_p, n_s):
        v = field[p, s]
        for d in ti.static(range(3)):
            if ti.math.isnan(v[d]):
                nan_cnt[None] += 1
            else:
                ti.atomic_max(max_abs[None], ti.abs(v[d]))


def report(stage: str) -> None:
    chunk = est.simulator.cuda_chunk_size
    n = est.num_particles[None]
    scan(est.simulator.x.grad, n, chunk)
    xg = (nan_cnt[None], max_abs[None])
    scan(est.simulator.v.grad, n, chunk)
    vg = (nan_cnt[None], max_abs[None])
    print(f"{stage:<34} x.grad: nan={xg[0]:>7} max={xg[1]:.3e} | v.grad: nan={vg[0]:>7} max={vg[1]:.3e}")


est.set_stage(Estimator.velocity_stage)
est.max_f = args.frames
set_params(est, args.fit_logE, 0.1, args.fit_v0)
forward(est, img_backward=False)
print(f"forward done, geometry loss = {est.loss[None]:.6f}")

est.loss.grad[None] = 1
est.clear_grads()
for ri in range(est.max_f):
    f = est.max_f - 1 - ri
    local = (f * est.simulator.n_substeps[None]) % est.simulator.cuda_chunk_size
    est.compute_loss_sim2gt.grad(f, local)
    est.compute_loss_gt2sim.grad(f, local)
    report(f"frame {f}: after chamfer .grad")
    if f > 0:
        est.simulator.advance_grad(f - 1)
        report(f"frame {f}: after advance_grad")
print("done")

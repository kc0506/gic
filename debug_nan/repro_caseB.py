# coding=utf-8
"""Case B integration repro: ONE vel-stage forward+backward with per-stage NaN tracing.

Replicates xsim joint iter 0 exactly (v0=0, external warp gts, 4 frames), but
splits estimator.backward(f) into (chamfer-loss grads) and (advance_grad) and
dumps NaN counts / max-abs of x.grad & v.grad between every stage.

Run (gic env, gic repo root):  python debug_nan/repro_caseB.py
"""
import sys

sys.path.insert(0, "/tmp2/b10401006/ev-project/gic")

from roundtrip_ours_scene import AnchoredEstimator, load_our_scene, set_params  # GPU pick happens here

import json
from argparse import Namespace

import numpy as np
import taichi as ti
import torch

from simulator import Estimator

CACHE = "/tmp2/b10401006/ev-project/generative-phys/outputs/dataset_gen/04_tel_axisy_rest_T16/scene_cache.pt"
TRAJ = "/tmp2/b10401006/ev-project/generative-phys/outputs/dataset_gen/04_tel_axisy_rest_T16/sample_0000/mpm_xyz.npy"
CONFIG = "/tmp2/b10401006/ev-project/gic/config/ours/telephone.json"

with open(CONFIG) as f:
    phys_args = Namespace(**json.load(f)["physics"])

xyz, anchor_mask = load_our_scene(CACHE)
cache = torch.load(CACHE, map_location="cpu", weights_only=False)
disc = cache["disc"]
ghost = (disc["sim_xyzs"] == 0).all(dim=1)
traj = torch.from_numpy(np.load(TRAJ))
traj = (traj + float(disc["shift"])) / float(disc["scale"])
traj = traj[:, ~ghost]
traj[0] = xyz.cpu()
n_frames = phys_args.n_frames = traj.shape[0]
gts = [traj[t].float().cuda() for t in range(n_frames)]

ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
est = AnchoredEstimator(
    phys_args, "float32", gts, surface_index=None, init_vol=xyz,
    dynamic_scene=None, image_scale=1.0, pipeline=None, image_op=None,
)
est.set_anchor(anchor_mask, 1e4)

# --- instrumentation kernels -------------------------------------------------
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


# --- vel-stage iter 0: forward -----------------------------------------------
est.set_stage(Estimator.velocity_stage)
est.max_f = phys_args.vel_estimation_frames  # 4, same as train()
set_params(est, 6.0, 0.1, [0.0, 0.0, 0.0])   # init E=1e6, nu=0.1, v0=0 (xsim joint iter0)

from train_dynamic import forward

forward(est, img_backward=False)
print(f"forward done, geometry loss = {est.loss[None]:.6f}")

# --- backward, manually staged ------------------------------------------------
est.loss.grad[None] = 1
est.clear_grads()
report("after clear_grads")
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

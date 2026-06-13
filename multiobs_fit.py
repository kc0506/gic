# coding=utf-8
"""M0: multi-observation SHARED-E recovery (amortized material, traj loss).

Two observations of the SAME telephone scene under DIFFERENT v0 (e.g. +x and
-y), same GT material E. We recover ONE shared E from both, serialized through
a single Estimator (taichi fields can't be allocated twice): each iter runs a
forward+backward per observation WITHOUT zeroing E.grad between them, so
autograd accumulates dL_A/dE + dL_B/dE before the E step.

M0 fixes each v0 to its GT and learns ONLY the shared E (init off by a decade)
-- isolating the question "do two independent excitations sharpen / stabilise
the E basin vs a single observation". (M1 = also learn per-obs v0.)

Equal loss weighting (no per-obs normalization) for M0; per-obs final losses
are reported so we can tell if normalization is needed for M1.
"""
from roundtrip_ours_scene import (
    AnchoredEstimator, forward_bounded, load_our_scene, save_overlay_gif, CFLExhausted,
)
from roundtrip_sim2sim import rollout_collect_surfaces, set_params
from train_dynamic import backward as gic_backward

import json
import math
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import taichi as ti
import torch

from simulator import Estimator
from utils.system_utils import draw_curve

GEN = "/tmp2/b10401006/ev-project/generative-phys"
VDIR = {"xp": [0.5, 0.0, 0.0], "xm": [-0.5, 0.0, 0.0],
        "yp": [0.0, 0.5, 0.0], "ym": [0.0, -0.5, 0.0]}


def rot_xyz(xyz: torch.Tensor, deg: float) -> torch.Tensor:
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y = xyz[:, 0] - 0.5, xyz[:, 1] - 0.5
    q = xyz.clone()
    q[:, 0] = c * x - s * y + 0.5
    q[:, 1] = s * x + c * y + 0.5
    return q


def main() -> None:
    t0 = time.time()
    ap = ArgumentParser(description="multi-observation shared-E recovery")
    ap.add_argument("--scene_cache",
                    default=f"{GEN}/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt")
    ap.add_argument("--config", default="config/ours/telephone.json")
    ap.add_argument("--gt_logE", default=5.0, type=float)
    ap.add_argument("--gt_nu", default=0.3, type=float)
    ap.add_argument("--init_logE", default=4.0, type=float)
    ap.add_argument("--obs", nargs="+", default=["xp", "ym"],
                    help="observation v0 directions (keys of VDIR) or 'x,y,z' triples")
    ap.add_argument("--rot_z_deg", default=67.6, type=float)
    ap.add_argument("--n_frames", default=8, type=int)
    ap.add_argument("--gt_frames", default=14, type=int)
    ap.add_argument("--anchor_mass_scale", default=1e4, type=float)
    ap.add_argument("--mpm_iter_cnt", default=64, type=int)
    ap.add_argument("--iter_cnt", default=80, type=int)
    ap.add_argument("--E_lr", default=0.2, type=float)
    ap.add_argument("--fine_lr", default=0.02, type=float)
    ap.add_argument("--patience", default=20, type=int)
    ap.add_argument("--min_iters", default=20, type=int)
    ap.add_argument("--estop_tol", default=0.005, type=float,
                    help="early stop once |delta log10 E| < tol for `patience` iters")
    ap.add_argument("--ti_mem_frac", default=0.3, type=float)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_root", default="output/ours_multiobs", type=str)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    obs_vels = [VDIR[o] if o in VDIR else [float(x) for x in o.split(",")]
                for o in args.obs]
    print(f"[m0] observations: {list(zip(args.obs, obs_vels))}")

    phys_args = Namespace(**json.load(open(args.config))["physics"])
    phys_args.mpm_iter_cnt = args.mpm_iter_cnt
    phys_args.n_frames = args.gt_frames

    xyz, anchor_mask = load_our_scene(args.scene_cache)
    if args.rot_z_deg:
        xyz = rot_xyz(xyz, args.rot_z_deg)
    cache = torch.load(args.scene_cache, map_location="cpu", weights_only=False)
    ghost = (cache["disc"]["sim_xyzs"] == 0).all(dim=1)
    pvol = torch.from_numpy(cache["disc"]["points_vol"]).float()[~ghost]
    free = ~anchor_mask

    ti.init(arch=ti.cuda, debug=False, fast_math=False,
            device_memory_fraction=args.ti_mem_frac)
    dummy = [xyz.clone() for _ in range(args.gt_frames)]
    est = AnchoredEstimator(phys_args, "float32", dummy, surface_index=None,
                            init_vol=xyz, dynamic_scene=None, image_scale=1.0,
                            pipeline=None, image_op=None)
    est.set_anchor(anchor_mask, args.anchor_mass_scale)
    est.set_pvol(pvol)
    est.set_stage(Estimator.physical_params_stage)
    est.geo_loss = True

    # ---- GT per observation (gic rollout at GT E + that v0) ----
    gts_list = []
    for o, v in zip(args.obs, obs_vels):
        set_params(est, args.gt_logE, args.gt_nu, v)
        est.max_f = args.gt_frames
        roll = rollout_collect_surfaces(est)
        disp = (roll[-1] - roll[0]).norm(dim=-1)[free].mean()
        print(f"[m0] GT obs {o} v0={v}: free mean disp {disp:.5f}")
        gts_list.append(roll)

    # ---- fit shared E (v0 fixed to GT per obs) ----
    set_params(est, args.init_logE, args.gt_nu, obs_vels[0])
    opt = torch.optim.Adam([{"params": est.E, "lr": args.E_lr, "name": "E"}])
    gt_E = 10.0 ** args.gt_logE

    E_traj, loss_traj, per_obs_loss = [], [], []
    best_loss, best_E, last_improve, last_move, dropped = float("inf"), None, 0, 0, False
    for it in range(args.iter_cnt):
        E_cur = float(est.E.detach())
        opt.zero_grad()  # zero E.grad ONCE; the two obs backward()s accumulate
        tot, ploss, failed = 0.0, [], False
        for v, gts in zip(obs_vels, gts_list):
            est.init_vel.data = torch.tensor(v, device=est.device)
            est.gts = gts
            est.load_gts(gts)
            est.loss[None] = 0.0
            try:
                forward_bounded(est, max_halvings=3)
            except CFLExhausted:
                failed = True
                break
            l = float(est.loss[None])
            ploss.append(l)
            tot += l
            gic_backward(est)  # init_mu.backward -> accumulates into est.E.grad
        if failed:
            print(f"[m0] iter {it}: CFL exhausted, keeping best (E {best_E})")
            break
        opt.step()
        E_new = float(est.E.detach())
        E_lin = 10.0 ** E_cur
        E_traj.append(E_lin)
        loss_traj.append(tot)
        per_obs_loss.append(ploss)
        if tot < best_loss:
            best_loss, best_E = tot, E_cur
        if tot < best_loss * (1 + 1e-9) or tot <= best_loss * (1.02):
            pass
        if tot < best_loss * 1.02:
            last_improve = it
        if abs(E_new - E_cur) >= args.estop_tol:
            last_move = it
        print(f"[m0] iter {it} totloss {tot:.6f} (obs {[f'{p:.5f}' for p in ploss]}) "
              f"E {E_lin:.4g} -> {10.0**E_new:.4g} | best E {10.0**best_E:.4g} "
              f"({(10.0**best_E-gt_E)/gt_E:+.1%})")

        # two-phase: first plateau -> restore best + flat fine lr
        plateau = it + 1 >= args.min_iters and (it - last_improve) >= args.patience
        if plateau and not dropped:
            dropped = True
            est.E.data.copy_(torch.tensor(best_E, device=est.device))
            opt.state.pop(est.E, None)
            for g in opt.param_groups:
                g["lr"] = args.fine_lr
            last_improve = it
            print(f"[m0] plateau -> restore best (E {10.0**best_E:.4g}) + fine lr {args.fine_lr}")
            continue
        settled = (it - last_move) >= args.patience
        if plateau and dropped and settled:
            print(f"[m0] early stop at iter {it}")
            break

    # ---- export ----
    est.E.data.copy_(torch.tensor(best_E, device=est.device))
    rel_E = (10.0 ** best_E - gt_E) / gt_E
    # final rollout per obs (full GT length) for overlays
    for i, (o, v, gts) in enumerate(zip(args.obs, obs_vels, gts_list)):
        set_params(est, best_E, args.gt_nu, v)
        est.max_f = args.gt_frames
        roll = rollout_collect_surfaces(est)
        save_overlay_gif(gts, roll, os.path.join(out_dir, f"overlay_obs{i}_{o}.gif"),
                         fit_frames=args.n_frames)
    result = {
        "scenario": "ours_multiobs_shared_E",
        "gt": {"E": gt_E, "logE": args.gt_logE, "nu": args.gt_nu},
        "init_logE": args.init_logE,
        "observations": list(zip(args.obs, obs_vels)),
        "best_E": 10.0 ** best_E,
        "rel_err_E": rel_E,
        "E_traj": E_traj,
        "losses_total": loss_traj,
        "losses_per_obs": per_obs_loss,
        "n_frames": args.n_frames,
        "wall_time_s": time.time() - t0,
    }
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    draw_curve(loss_traj, out_dir, name="loss_total")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(E_traj, "-o", ms=3)
    ax.axhline(gt_E, color="k", ls="--", label="GT")
    ax.set_yscale("log")
    ax.set_xlabel("iter")
    ax.set_ylabel("shared E")
    ax.set_title(f"{args.tag}: shared E from {len(args.obs)} obs (err {rel_E:+.1%})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "E_traj.png"))
    print(f"[m0] DONE {args.tag}: best E {10.0**best_E:.4g} ({rel_E:+.2%}) "
          f"from obs {args.obs}, wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

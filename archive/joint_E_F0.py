# coding=utf-8
"""Joint (E, alpha) gradient-health test: learn BOTH Young's modulus and the
initial-deformation AMPLITUDE alpha (F0 = V0^alpha), pure gradient descent, no
restore-best -- to see whether the optimizer slides along the E*strain
degeneracy ridge or settles at the true (E*, alpha=1).

F0(alpha) = Qe diag(lambda^alpha) Qe^T from eigh(V0); differentiable in alpha.
Bridge: gic's manual BPTT leaves dL/dF0 in simulator.F.grad[:,0]; we read it and
call F0.backward(gradient=dL/dF0) to get alpha.grad. E.grad comes from gic's
existing init_mu(E) path. nu frozen at GT to isolate (E, alpha).

We report the SETTLED fixed point (mean of last iters), NOT a best-loss export
(restore-best / argmin is grid-search-flavoured, no generality -- see
feedback-convergence-rigor). Sweeps several inits to map basin/ridge behaviour.

Usage (gic env, gic repo root; co-locate GPU via CUDA_VISIBLE_DEVICES):
  python joint_E_F0.py --gt_traj .../warp_traj.npy --init_xyz .../init_xyz.npy \
      --v0_field .../f0_field.npy --gt_logE 5.0 --inits 5.0,1.0 5.301,1.1 4.699,0.9
"""
from roundtrip_ours_scene import (
    AnchoredEstimator, forward_bounded, load_our_scene, save_overlay_gif,
)
from roundtrip_sim2sim import rollout_collect_surfaces

import json
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import taichi as ti
import torch
import torch.nn as nn
from simulator import Estimator
from simulator.estimator import constraint_inv
from train_dynamic import backward as gic_backward


if __name__ == "__main__":
    t0 = time.time()
    ap = ArgumentParser(description="joint (E, alpha) gradient-fixed-point test")
    ap.add_argument("--config", default="config/ours/telephone.json")
    ap.add_argument("--scene_cache", required=True)
    ap.add_argument("--gt_traj", required=True, help="warp release traj (normalized)")
    ap.add_argument("--init_xyz", required=True, help="snapshot positions = t0")
    ap.add_argument("--v0_field", required=True, help="V0 (left stretch) field [n,3,3]")
    ap.add_argument("--gt_logE", default=5.0, type=float)
    ap.add_argument("--gt_nu", default=0.3, type=float)
    ap.add_argument("--n_frames", default=9, type=int)
    ap.add_argument("--mpm_iter_cnt", default=64, type=int)
    ap.add_argument("--anchor_mass_scale", default=1e4, type=float)
    ap.add_argument("--n_iters", default=70, type=int)
    ap.add_argument("--lr_E", default=0.04, type=float, help="log10-E step")
    ap.add_argument("--lr_alpha", default=0.04, type=float)
    ap.add_argument("--inits", nargs="+", default=["5.0,1.0", "5.301,1.1", "4.699,0.9"],
                    help="space-separated logE,alpha init points")
    ap.add_argument("--tag", default="joint_E_F0")
    ap.add_argument("--out_root", default="output/ours_telephone")
    ap.add_argument("--viz_only", action="store_true",
                    help="skip optimization; redraw plots/gifs from existing joint_result.json")
    ap.add_argument("--landscape_npz", default="/tmp2/b10401006/ev-project/generative-phys/"
                    "outputs/explore/f0_alpha_landscape/tele_alpha_logE_f8/landscape2d.npz",
                    help="(alpha,logE) landscape to draw the optimization paths on")
    args = ap.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(args.config) as f:
        phys_args = Namespace(**json.load(f)["physics"])
    phys_args.mpm_iter_cnt = args.mpm_iter_cnt
    phys_args.n_frames = args.n_frames
    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)

    # ---- scene: deformed snapshot positions as t0 ----
    xyz, anchor_mask = load_our_scene(args.scene_cache)
    init_xyz = torch.from_numpy(np.load(args.init_xyz)).float().cuda()
    assert init_xyz.shape == xyz.shape, (init_xyz.shape, xyz.shape)
    xyz = init_xyz
    n = xyz.shape[0]

    # ---- V0 eigendecomposition -> F0(alpha) = Qe diag(lambda^alpha) Qe^T ----
    V0 = torch.from_numpy(np.load(args.v0_field)).float().cuda()      # [n,3,3] SPD
    lam, Qe = torch.linalg.eigh(0.5 * (V0 + V0.transpose(-1, -2)))    # symmetrize for safety
    lam = lam.clamp_min(1e-6)                                        # [n,3] stretches
    print(f"[joint] N={n}; V0 stretch range [{lam.min():.3f},{lam.max():.3f}], "
          f"alpha=1 maxdev {(lam-1).abs().max():.3f}")

    def F0_of_alpha(alpha):
        return torch.einsum("nij,nj,nkj->nik", Qe, lam.pow(alpha), Qe)  # [n,3,3]

    # ---- warp GT trajectory (normalized), snap frame0, truncate ----
    traj = torch.from_numpy(np.load(args.gt_traj)).float()           # [T,n,3]
    assert traj.shape[1] == n, (traj.shape, n)
    drift = (traj[0].cuda() - xyz).norm(dim=-1).max()
    assert drift < 1e-3, f"traj frame0 vs init_xyz drift {drift}"
    traj[0] = xyz.cpu()
    gts = [traj[t].float().cuda() for t in range(min(args.n_frames, traj.shape[0]))]
    n_frames = len(gts)

    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.5)
    dummy = [xyz.clone() for _ in range(n_frames)]
    est = AnchoredEstimator(phys_args, "float32", dummy, surface_index=None, init_vol=xyz,
                            dynamic_scene=None, image_scale=1.0, pipeline=None, image_op=None)
    est.set_anchor(anchor_mask, args.anchor_mass_scale)
    cpv = torch.load(args.scene_cache, map_location="cpu", weights_only=False)
    ghost = (cpv["disc"]["sim_xyzs"] == 0).all(dim=1)
    est.set_pvol(torch.from_numpy(cpv["disc"]["points_vol"]).float()[~ghost])
    est.gts = gts
    est.load_gts(gts)
    est.set_stage(Estimator.physical_params_stage)
    est.init_vel.data = torch.zeros(3, device=est.device)   # v0 = 0 (pure release), fixed
    est.nu.data = constraint_inv(torch.tensor(args.gt_nu, device=est.device), est.nu_bound)  # frozen

    res_path = os.path.join(out_dir, "joint_result.json")
    if args.viz_only:
        results = json.load(open(res_path))["results"]
        print(f"[joint] viz_only: loaded {res_path}")
    else:
      g_buf = np.zeros((n, 3, 3), dtype=np.float32)
      results = {}
      for spec in args.inits:
        logE0, alpha0 = (float(x) for x in spec.split(","))
        est.E.data = torch.tensor(logE0, device=est.device)
        alpha = nn.Parameter(torch.tensor(alpha0, device=est.device))
        opt = torch.optim.Adam([{"params": [est.E], "lr": args.lr_E},
                                {"params": [alpha], "lr": args.lr_alpha}])
        E_tr, a_tr, L_tr = [], [], []
        for it in range(args.n_iters):
            F0 = F0_of_alpha(alpha)                  # differentiable in alpha
            est.set_F0(F0.detach())
            est.max_f = n_frames
            opt.zero_grad(); est.loss[None] = 0.0
            try:
                forward_bounded(est)
            except Exception as e:
                print(f"[joint] {spec} iter {it}: forward failed ({e}); stop"); break
            loss = float(est.loss[None])
            gic_backward(est)                        # -> est.E.grad + simulator.F.grad[:,0]
            est._read_F0_grad(g_buf, n)
            F0.backward(gradient=torch.from_numpy(g_buf).to(est.device))  # -> alpha.grad
            opt.step()
            E_tr.append(float(10.0 ** est.E)); a_tr.append(float(alpha)); L_tr.append(loss)
        tail = slice(max(0, len(E_tr) - 10), len(E_tr))
        E_set, a_set = float(np.mean(E_tr[tail])), float(np.mean(a_tr[tail]))
        results[spec] = {"E_traj": E_tr, "alpha_traj": a_tr, "loss_traj": L_tr,
                         "E_settled": E_set, "alpha_settled": a_set,
                         "E_rel_err": E_set / (10 ** args.gt_logE) - 1.0}
        print(f"[joint] init (logE {logE0}, a {alpha0}) -> settled E {E_set:.0f} "
              f"({results[spec]['E_rel_err']*100:+.1f}%), alpha {a_set:.3f}; "
              f"loss {L_tr[0]:.4f}->{L_tr[-1]:.4f}")

      with open(res_path, "w") as f:
        json.dump({"gt_logE": args.gt_logE, "gt_nu": args.gt_nu, "n_frames": n_frames,
                   "lr_E": args.lr_E, "lr_alpha": args.lr_alpha, "n_iters": args.n_iters,
                   "results": results}, f, indent=2)

    # ============================ visualizations ============================
    gt_logE = args.gt_logE
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(results), 3)))

    # (1) per-init trajectories: E, alpha, loss vs iter
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for c, (spec, d) in zip(colors, results.items()):
        it = range(len(d["E_traj"]))
        ax[0].plot(it, d["E_traj"], color=c, label=f"init {spec}")
        ax[1].plot(it, d["alpha_traj"], color=c, label=f"init {spec}")
        ax[2].plot(it, d["loss_traj"], color=c, label=f"init {spec}")
    ax[0].axhline(10 ** gt_logE, color="k", ls="--", lw=1, label="GT E")
    ax[0].set_title("E vs iter"); ax[0].set_yscale("log"); ax[0].set_ylabel("E"); ax[0].legend(fontsize=7)
    ax[1].axhline(1.0, color="k", ls="--", lw=1, label="GT alpha=1")
    ax[1].set_title("alpha (F0 amplitude) vs iter"); ax[1].legend(fontsize=7)
    ax[2].set_title("trajectory loss vs iter"); ax[2].set_yscale("log"); ax[2].set_xlabel("iter")
    for a in ax:
        a.set_xlabel("iter")
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "joint_traj.png"), dpi=120); plt.close(fig)

    # (2) THE money plot: optimization paths on the (alpha, logE) landscape
    if os.path.exists(args.landscape_npz):
        lp = np.load(args.landscape_npz)
        alphas, logEs, Lmap = lp["alphas"], lp["logEs"], lp["Lmap"]  # Lmap [alpha, logE]
        fig, axL = plt.subplots(figsize=(7.5, 6))
        cf = axL.contourf(logEs, alphas, np.log10(Lmap + 1e-18), levels=20, cmap="viridis")
        fig.colorbar(cf, ax=axL, label="log10 trajectory loss")
        for c, (spec, d) in zip(colors, results.items()):
            le = np.log10(np.array(d["E_traj"])); a = np.array(d["alpha_traj"])
            axL.plot(le, a, "-", color=c, lw=1.5, alpha=0.9)
            axL.scatter(le[0], a[0], color=c, marker="o", s=70, edgecolor="w",
                        zorder=5, label=f"init {spec}")
            axL.scatter(le[-1], a[-1], color=c, marker="X", s=110, edgecolor="k", zorder=6)
        axL.scatter([gt_logE], [1.0], color="red", marker="*", s=260, edgecolor="w",
                    zorder=7, label="GT (E*, 1)")
        axL.set_xlabel("log10 E"); axL.set_ylabel("alpha (F0 amplitude)")
        axL.set_title("joint (E, alpha) gradient paths on the degeneracy landscape\n"
                      "o=init  X=settled  *=truth")
        axL.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "joint_paths_on_landscape.png"), dpi=130); plt.close(fig)
        print(f"[joint] wrote joint_paths_on_landscape.png")

    # (3) overlay gif: GT (blue) vs prediction at each init's SETTLED (E, alpha)
    for spec, d in results.items():
        E_set, a_set = d["E_settled"], d["alpha_settled"]
        est.E.data = torch.tensor(float(np.log10(E_set)), device=est.device)
        est.set_F0(F0_of_alpha(torch.tensor(a_set, device=est.device)).detach())
        est.max_f = n_frames
        pred = rollout_collect_surfaces(est)
        est.set_stage(Estimator.physical_params_stage); est.max_f = n_frames
        safe = spec.replace(",", "_")
        save_overlay_gif(gts, pred, os.path.join(out_dir, f"overlay_init_{safe}.gif"))
        print(f"[joint] overlay_init_{safe}.gif  (E {E_set:.0f}, alpha {a_set:.3f})")

    print(f"[joint] DONE -> {out_dir} ({time.time()-t0:.0f}s)")

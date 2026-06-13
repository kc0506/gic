# coding=utf-8
"""Loss landscapes on the gic chamfer objective (forward-only, no backward).

Modes:
  E1d  : loss vs log10(E), one curve per fit window in --windows (nu, v0 = GT)
  Enu  : 2D (log10 E, nu) heatmap at v0 = GT; optionally overlay an optimizer
         path from a run's result.json (E_traj / nu_traj)
  v0xy : 2D (v0_x, v0_y) heatmap at E, nu = GT (window = --n_frames, mirrors
         the vel stage's 4-frame view by default)

Each grid point = one forward rollout + chamfer loss, NO dt-halving retry:
a CFL failure is recorded as NaN (halving would silently change the loss
definition across the grid).

Usage (gic env, gic repo root), e.g.:
  python loss_landscape.py --mode E1d --gt_traj .../warp_traj.npy --gt_traj_normalized \
      --gt_logE 4.0 --windows 8 14 --tag E1e4_windows
"""
from roundtrip_ours_scene import (  # noqa: E402  (this import picks a GPU)
    AnchoredEstimator,
    load_our_scene,
    set_params,
)

import json
import math
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import taichi as ti
import torch

from simulator import Estimator

GEN = "/tmp2/b10401006/ev-project/generative-phys"
OUT_ROOT = f"{GEN}/reports/20260612_gic_q2/landscape"


def forward_loss(est: Estimator, max_f: int) -> float:
    """One forward rollout; return scalar chamfer loss, NaN if CFL violated."""
    est.max_f = max_f
    est.zero_grad()
    est.loss[None] = 0.0
    dt = est.simulator.dt_ori[None]
    for idx in range(max_f):
        if idx == 0:
            est.initialize()
            est.simulator.set_dt(dt)
        est.forward(idx, img_backward=False)
    if not est.succeed():
        return float("nan")
    return float(est.loss[None])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--config", default="config/ours/telephone.json")
    ap.add_argument("--scene_cache", default=f"{GEN}/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt")
    ap.add_argument("--gt_traj", required=True)
    ap.add_argument("--gt_traj_normalized", action="store_true")
    ap.add_argument("--rot_z_deg", default=0.0, type=float)
    ap.add_argument("--gt_logE", required=True, type=float)
    ap.add_argument("--gt_nu", default=0.3, type=float)
    ap.add_argument("--gt_vel", nargs=3, default=[0.0, -0.5, 0.0], type=float)
    ap.add_argument("--anchor_mass_scale", default=1e4, type=float)
    ap.add_argument("--mpm_iter_cnt", default=64, type=int)
    ap.add_argument("--mode", required=True,
                    choices=["E1d", "Enu", "v0xy", "v0z1d", "vyvz", "Evz", "Evy", "vynu"])
    ap.add_argument("--n_frames", default=8, type=int, help="fit window (frames incl. frame0)")
    ap.add_argument("--windows", nargs="+", default=None, type=int, help="E1d: multiple windows")
    ap.add_argument("--grid_n", default=21, type=int)
    ap.add_argument("--logE_range", nargs=2, default=[3.5, 6.5], type=float)
    ap.add_argument("--nu_range", nargs=2, default=[0.02, 0.45], type=float)
    ap.add_argument("--vx_range", nargs=2, default=[-0.75, 0.75], type=float)
    ap.add_argument("--vy_range", nargs=2, default=[-0.75, 0.75], type=float)
    ap.add_argument("--vz_range", nargs=2, default=[-0.75, 0.75], type=float)
    ap.add_argument("--overlay_run", default=None,
                    help="Enu: result.json whose E_traj/nu_traj is drawn on the map")
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    os.makedirs(OUT_ROOT, exist_ok=True)
    t0 = time.time()

    phys_args = Namespace(**json.load(open(args.config))["physics"])
    phys_args.mpm_iter_cnt = args.mpm_iter_cnt

    xyz, anchor_mask = load_our_scene(args.scene_cache)
    if args.rot_z_deg:
        t = math.radians(args.rot_z_deg)
        c, s = math.cos(t), math.sin(t)
        x, y = xyz[:, 0] - 0.5, xyz[:, 1] - 0.5
        xyz = xyz.clone()
        xyz[:, 0] = c * x - s * y + 0.5
        xyz[:, 1] = s * x + c * y + 0.5

    cache = torch.load(args.scene_cache, map_location="cpu", weights_only=False)
    ghost = (cache["disc"]["sim_xyzs"] == 0).all(dim=1)
    traj = torch.from_numpy(np.load(args.gt_traj))
    if not args.gt_traj_normalized:
        traj = (traj + cache["disc"]["shift"].reshape(1, 1, -1)) / float(cache["disc"]["scale"])
    traj = traj[:, ~ghost]
    drift = (traj[0] - xyz.cpu()).norm(dim=-1).max()
    assert drift < 1e-3, f"GT frame0 vs (rotated) cache drift {drift}"
    traj[0] = xyz.cpu()
    gts = [traj[t].float().cuda() for t in range(traj.shape[0])]

    max_window = max(args.windows) if args.windows else args.n_frames
    assert max_window <= len(gts), (max_window, len(gts))
    phys_args.n_frames = len(gts)

    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.25)
    est = AnchoredEstimator(
        phys_args, "float32", gts, surface_index=None, init_vol=xyz,
        dynamic_scene=None, image_scale=1.0, pipeline=None, image_op=None,
    )
    est.set_anchor(anchor_mask, args.anchor_mass_scale)
    pvol = torch.from_numpy(cache["disc"]["points_vol"]).float()[~ghost]
    est.set_pvol(pvol)
    est.gts = gts
    est.load_gts(gts)
    est.set_stage(Estimator.physical_params_stage)
    est.geo_loss = True

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if args.mode == "E1d":
        windows = args.windows or [args.n_frames]
        logEs = np.linspace(*args.logE_range, args.grid_n)
        curves = {}
        for w in windows:
            L = []
            for le in logEs:
                set_params(est, float(le), args.gt_nu, list(args.gt_vel))
                L.append(forward_loss(est, w))
                print(f"[E1d w={w}] logE={le:.3f} loss={L[-1]:.6g}")
            curves[w] = np.array(L)
        np.savez(f"{OUT_ROOT}/{args.tag}_E1d.npz", logEs=logEs,
                 **{f"loss_w{w}": v for w, v in curves.items()})
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for w, v in curves.items():
            ax.plot(logEs, v, "-o", ms=3, label=f"window {w}f")
            i = int(np.nanargmin(v))
            ax.axvline(logEs[i], ls=":", lw=0.8,
                       color=ax.lines[-1].get_color())
        ax.axvline(args.gt_logE, color="k", ls="--", lw=1.2, label="GT")
        ax.set_yscale("log")
        ax.set_xlabel("log10 E")
        ax.set_ylabel("chamfer loss")
        ax.set_title(f"loss(logE) | {args.tag} (dotted = argmin per window)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUT_ROOT}/{args.tag}_E1d.png", dpi=130)

    elif args.mode == "Enu":
        logEs = np.linspace(*args.logE_range, args.grid_n)
        nus = np.linspace(*args.nu_range, args.grid_n)
        Z = np.full((args.grid_n, args.grid_n), np.nan)  # [nu, logE]
        for j, le in enumerate(logEs):
            for i, nu in enumerate(nus):
                set_params(est, float(le), float(nu), list(args.gt_vel))
                Z[i, j] = forward_loss(est, args.n_frames)
            print(f"[Enu] col {j+1}/{args.grid_n} (logE={le:.2f}) done "
                  f"min={np.nanmin(Z[:, j]):.5g}")
        np.savez(f"{OUT_ROOT}/{args.tag}_Enu.npz", logEs=logEs, nus=nus, loss=Z)
        fig, ax = plt.subplots(figsize=(7, 5.2))
        pc = ax.pcolormesh(logEs, nus, np.log10(Z), shading="auto", cmap="viridis")
        fig.colorbar(pc, label="log10 loss")
        ax.contour(logEs, nus, np.log10(Z), levels=12, colors="w", linewidths=0.4)
        ax.plot(args.gt_logE, args.gt_nu, "r*", ms=14, label="GT")
        ii, jj = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.plot(logEs[jj], nus[ii], "wx", ms=10, mew=2, label="grid argmin")
        if args.overlay_run:
            r = json.load(open(args.overlay_run))
            ax.plot(np.log10(r["E_traj"]), r["nu_traj"], "-", color="orange",
                    lw=1.2, alpha=0.9, label="optimizer path")
            ax.plot(np.log10(r["E_traj"][0]), r["nu_traj"][0], "o", color="orange", ms=6)
        ax.set_xlabel("log10 E")
        ax.set_ylabel("nu")
        ax.set_title(f"loss(logE, nu) | {args.tag} | window {args.n_frames}f")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT_ROOT}/{args.tag}_Enu.png", dpi=130)

    elif args.mode == "v0xy":
        vxs = np.linspace(*args.vx_range, args.grid_n)
        vys = np.linspace(*args.vy_range, args.grid_n)
        Z = np.full((args.grid_n, args.grid_n), np.nan)  # [vy, vx]
        for j, vx in enumerate(vxs):
            for i, vy in enumerate(vys):
                set_params(est, args.gt_logE, args.gt_nu, [float(vx), float(vy), 0.0])
                Z[i, j] = forward_loss(est, args.n_frames)
            print(f"[v0xy] col {j+1}/{args.grid_n} (vx={vx:.2f}) done "
                  f"min={np.nanmin(Z[:, j]):.5g}")
        np.savez(f"{OUT_ROOT}/{args.tag}_v0xy.npz", vxs=vxs, vys=vys, loss=Z)
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        pc = ax.pcolormesh(vxs, vys, np.log10(Z), shading="auto", cmap="viridis")
        fig.colorbar(pc, label="log10 loss")
        ax.contour(vxs, vys, np.log10(Z), levels=12, colors="w", linewidths=0.4)
        ax.plot(args.gt_vel[0], args.gt_vel[1], "r*", ms=14, label="GT v0")
        ii, jj = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.plot(vxs[jj], vys[ii], "wx", ms=10, mew=2, label="grid argmin")
        ax.set_xlabel("v0_x")
        ax.set_ylabel("v0_y")
        ax.set_aspect("equal")
        ax.set_title(f"loss(v0_x, v0_y) | {args.tag} | window {args.n_frames}f")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT_ROOT}/{args.tag}_v0xy.png", dpi=130)

    elif args.mode == "v0z1d":
        windows = args.windows or [args.n_frames]
        vzs = np.linspace(*args.vz_range, args.grid_n)
        curves = {}
        for w in windows:
            L = []
            for vz in vzs:
                set_params(est, args.gt_logE, args.gt_nu,
                           [args.gt_vel[0], args.gt_vel[1], float(vz)])
                L.append(forward_loss(est, w))
                print(f"[v0z1d w={w}] vz={vz:.3f} loss={L[-1]:.6g}")
            curves[w] = np.array(L)
        np.savez(f"{OUT_ROOT}/{args.tag}_v0z1d.npz", vzs=vzs,
                 **{f"loss_w{w}": v for w, v in curves.items()})
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for w, v in curves.items():
            ax.plot(vzs, v, "-o", ms=3, label=f"window {w}f")
        ax.axvline(args.gt_vel[2], color="k", ls="--", lw=1.2, label="GT v0_z")
        ax.set_yscale("log")
        ax.set_xlabel("v0_z")
        ax.set_ylabel("chamfer loss")
        ax.set_title(f"loss(v0_z) | {args.tag} (vx,vy fixed at GT)")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"{OUT_ROOT}/{args.tag}_v0z1d.png", dpi=130)

    elif args.mode == "vyvz":
        vys = np.linspace(*args.vy_range, args.grid_n)
        vzs = np.linspace(*args.vz_range, args.grid_n)
        Z = np.full((args.grid_n, args.grid_n), np.nan)  # [vz, vy]
        for j, vy in enumerate(vys):
            for i, vz in enumerate(vzs):
                set_params(est, args.gt_logE, args.gt_nu,
                           [args.gt_vel[0], float(vy), float(vz)])
                Z[i, j] = forward_loss(est, args.n_frames)
            print(f"[vyvz] col {j+1}/{args.grid_n} (vy={vy:.2f}) done "
                  f"min={np.nanmin(Z[:, j]):.5g}")
        np.savez(f"{OUT_ROOT}/{args.tag}_vyvz.npz", vys=vys, vzs=vzs, loss=Z)
        fig, ax = plt.subplots(figsize=(6.4, 5.6))
        pc = ax.pcolormesh(vys, vzs, np.log10(Z), shading="auto", cmap="viridis")
        fig.colorbar(pc, label="log10 loss")
        ax.contour(vys, vzs, np.log10(Z), levels=12, colors="w", linewidths=0.4)
        ax.plot(args.gt_vel[1], args.gt_vel[2], "r*", ms=14, label="GT v0")
        ii, jj = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.plot(vys[jj], vzs[ii], "wx", ms=10, mew=2, label="grid argmin")
        ax.set_xlabel("v0_y")
        ax.set_ylabel("v0_z")
        ax.set_aspect("equal")
        ax.set_title(f"loss(v0_y, v0_z) | {args.tag} | window {args.n_frames}f")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT_ROOT}/{args.tag}_vyvz.png", dpi=130)

    elif args.mode == "Evz":
        logEs = np.linspace(*args.logE_range, args.grid_n)
        vzs = np.linspace(*args.vz_range, args.grid_n)
        Z = np.full((args.grid_n, args.grid_n), np.nan)  # [vz, logE]
        for j, le in enumerate(logEs):
            for i, vz in enumerate(vzs):
                set_params(est, float(le), args.gt_nu,
                           [args.gt_vel[0], args.gt_vel[1], float(vz)])
                Z[i, j] = forward_loss(est, args.n_frames)
            print(f"[Evz] col {j+1}/{args.grid_n} (logE={le:.2f}) done "
                  f"min={np.nanmin(Z[:, j]):.5g}")
        np.savez(f"{OUT_ROOT}/{args.tag}_Evz.npz", logEs=logEs, vzs=vzs, loss=Z)
        fig, ax = plt.subplots(figsize=(7, 5.2))
        pc = ax.pcolormesh(logEs, vzs, np.log10(Z), shading="auto", cmap="viridis")
        fig.colorbar(pc, label="log10 loss")
        ax.contour(logEs, vzs, np.log10(Z), levels=12, colors="w", linewidths=0.4)
        ax.plot(args.gt_logE, args.gt_vel[2], "r*", ms=14, label="GT")
        ii, jj = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.plot(logEs[jj], vzs[ii], "wx", ms=10, mew=2, label="grid argmin")
        ax.set_xlabel("log10 E")
        ax.set_ylabel("v0_z")
        ax.set_title(f"loss(logE, v0_z) | {args.tag} | window {args.n_frames}f")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT_ROOT}/{args.tag}_Evz.png", dpi=130)

    elif args.mode == "Evy":
        # the missing 2D map for the wrong-E scalar failure: vy x logE
        logEs = np.linspace(*args.logE_range, args.grid_n)
        vys = np.linspace(*args.vy_range, args.grid_n)
        Z = np.full((args.grid_n, args.grid_n), np.nan)  # [vy, logE]
        for j, le in enumerate(logEs):
            for i, vy in enumerate(vys):
                set_params(est, float(le), args.gt_nu,
                           [args.gt_vel[0], float(vy), args.gt_vel[2]])
                Z[i, j] = forward_loss(est, args.n_frames)
            print(f"[Evy] col {j+1}/{args.grid_n} (logE={le:.2f}) done "
                  f"min={np.nanmin(Z[:, j]):.5g}")
        np.savez(f"{OUT_ROOT}/{args.tag}_Evy.npz", logEs=logEs, vys=vys, loss=Z)
        fig, ax = plt.subplots(figsize=(7, 5.2))
        pc = ax.pcolormesh(logEs, vys, np.log10(Z), shading="auto", cmap="viridis")
        fig.colorbar(pc, label="log10 loss")
        ax.contour(logEs, vys, np.log10(Z), levels=12, colors="w", linewidths=0.4)
        ax.plot(args.gt_logE, args.gt_vel[1], "r*", ms=14, label="GT")
        ii, jj = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.plot(logEs[jj], vys[ii], "wx", ms=10, mew=2, label="grid argmin")
        # per-column best vy: the bias path an optimizer with pinned-wrong E follows
        best_vy = vys[np.nanargmin(Z, axis=0)]
        ax.plot(logEs, best_vy, "r--", lw=1.0, alpha=0.8, label="best vy per E")
        ax.set_xlabel("log10 E")
        ax.set_ylabel("v0_y")
        ax.set_title(f"loss(logE, v0_y) | {args.tag} | window {args.n_frames}f")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT_ROOT}/{args.tag}_Evy.png", dpi=130)

    elif args.mode == "vynu":
        vys = np.linspace(*args.vy_range, args.grid_n)
        nus = np.linspace(*args.nu_range, args.grid_n)
        Z = np.full((args.grid_n, args.grid_n), np.nan)  # [nu, vy]
        for j, vy in enumerate(vys):
            for i, nu in enumerate(nus):
                set_params(est, args.gt_logE, float(nu),
                           [args.gt_vel[0], float(vy), args.gt_vel[2]])
                Z[i, j] = forward_loss(est, args.n_frames)
            print(f"[vynu] col {j+1}/{args.grid_n} (vy={vy:.2f}) done "
                  f"min={np.nanmin(Z[:, j]):.5g}")
        np.savez(f"{OUT_ROOT}/{args.tag}_vynu.npz", vys=vys, nus=nus, loss=Z)
        fig, ax = plt.subplots(figsize=(7, 5.2))
        pc = ax.pcolormesh(vys, nus, np.log10(Z), shading="auto", cmap="viridis")
        fig.colorbar(pc, label="log10 loss")
        ax.contour(vys, nus, np.log10(Z), levels=12, colors="w", linewidths=0.4)
        ax.plot(args.gt_vel[1], args.gt_nu, "r*", ms=14, label="GT")
        ii, jj = np.unravel_index(np.nanargmin(Z), Z.shape)
        ax.plot(vys[jj], nus[ii], "wx", ms=10, mew=2, label="grid argmin")
        ax.set_xlabel("v0_y")
        ax.set_ylabel("nu")
        ax.set_title(f"loss(v0_y, nu) | {args.tag} | window {args.n_frames}f")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(f"{OUT_ROOT}/{args.tag}_vynu.png", dpi=130)

    print(f"[landscape] {args.tag} ({args.mode}) done in {time.time() - t0:.0f}s -> {OUT_ROOT}")


if __name__ == "__main__":
    main()

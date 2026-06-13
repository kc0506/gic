# coding=utf-8
"""E-FIELD recovery (traj loss): learn a per-particle log10 E field, v0 fixed.

The E analog of the v0-field campaign. GT material is a field (uniform or a
ramp along the cord); we recover it from one trajectory observation with v0
fixed to GT. Runs in the phys stage with the optimizer swapped to the
EVoxelField grid (AnchoredEstimator.set_E_field injects per-particle mu/lam).

KEY DIFFERENCE FROM v0 FIELD: E is observable only where the motion produces
STRAIN. A single v0 direction illuminates only part of the field; low-strain
regions stay at init regardless of geometric support. We therefore report
error weighted by a real LOCAL-STRAIN proxy (neighbour-distance change) AND
a scatter of recovered log10 E vs that strain, so the dead-zone is visible rather
than averaged away.
"""
from roundtrip_ours_scene import (
    AnchoredEstimator, forward_bounded, load_our_scene, save_overlay_gif, CFLExhausted,
)
from roundtrip_sim2sim import rollout_collect_surfaces, set_params
from train_dynamic import backward as gic_backward
from v0_field_ours import EVoxelField, eval_Egrid_at

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
VDIR = {"xp": [0.5, 0, 0], "xm": [-0.5, 0, 0], "yp": [0, 0.5, 0], "ym": [0, -0.5, 0]}


def rot_xyz(xyz, deg):
    t = math.radians(deg); c, s = math.cos(t), math.sin(t)
    x, y = xyz[:, 0] - 0.5, xyz[:, 1] - 0.5
    q = xyz.clone(); q[:, 0] = c * x - s * y + 0.5; q[:, 1] = s * x + c * y + 0.5
    return q


def strain_proxy(x0: torch.Tensor, xT: torch.Tensor, k: int = 8) -> torch.Tensor:
    """Per-particle LOCAL strain proxy (N,): mean |relative-distance change| to
    the k nearest neighbours (fixed at frame 0) between frame 0 and frame T.

    Rigid motion (translation/rotation) preserves inter-particle distances ->
    ~0; only true stretch/shear registers. This is the RIGHT observability
    measure for E (raw displacement is anti-correlated for a cantilever: the
    free tip swings far but barely strains; the clamped base strains hardest
    but barely moves)."""
    d0 = torch.cdist(x0, x0)                                   # (N,N)
    knn = d0.topk(k + 1, largest=False).indices[:, 1:]         # (N,k) drop self
    nb0 = d0.gather(1, knn)                                    # (N,k) rest dists
    dT = (xT[:, None, :] - xT[knn]).norm(dim=-1)               # (N,k) deformed dists
    # clamp the denominator to a fraction of the median rest distance: the cord is
    # <1 voxel thick so some neighbour pairs are near-coincident (d0~0) and the raw
    # ratio explodes (artifact, not strain). median-relative floor kills the blowup.
    floor = 0.3 * float(nb0.median())
    return ((dT - nb0).abs() / nb0.clamp(min=floor)).mean(dim=1)  # (N,)


def fill_circular_grid(field, xyz_all, free_mask, z_lo, z_hi,
                       lo_logE=4.5, hi_logE=5.5, nb=24) -> None:
    """Fill an EVoxelField grid with branch-dependent (circular) log10 E.

    Per-particle: 2-means per z-bin labels the two strands (lower-x=branch0,
    higher-x=branch1, no crossing verified); arc length runs anchor(top)->down
    strand0->bottom->up strand1->top, logE = lo + (hi-lo)*arc. Then bucket the
    per-particle logE into the grid (trilinear-weight average) so the GT is
    grid-representable. xyz_all (N,3) cpu, free_mask (N,) bool cpu.
    """
    import torch.nn.functional as F
    P = xyz_all.numpy()
    free = free_mask.numpy()
    ze = np.linspace(z_lo, z_hi + 1e-6, nb + 1)
    branch = np.zeros(P.shape[0], dtype=int)
    for i in range(nb):
        m = free & (P[:, 2] >= ze[i]) & (P[:, 2] < ze[i + 1])
        idx = np.where(m)[0]
        if len(idx) < 6:
            continue
        xs = P[idx, 0]; c0, c1 = xs.min(), xs.max()
        for _ in range(40):
            lab = (np.abs(xs - c1) < np.abs(xs - c0)).astype(int)
            if (lab == 0).any(): c0 = xs[lab == 0].mean()
            if (lab == 1).any(): c1 = xs[lab == 1].mean()
        hi = max(c0, c1); lo = min(c0, c1)
        branch[idx] = (np.abs(xs - hi) < np.abs(xs - lo)).astype(int)
    sA = 0.5 * (z_hi - P[:, 2]) / (z_hi - z_lo)
    sB = 0.5 + 0.5 * (P[:, 2] - z_lo) / (z_hi - z_lo)
    arc = np.where(branch == 0, sA, sB)
    logE_pp = torch.tensor(lo_logE + (hi_logE - lo_logE) * arc, dtype=torch.float32)
    # bucket free-particle logE into the grid: node = trilinear-weight avg
    rz, ry, rx = field.grid.shape[2:]
    aabb = field.aabb
    q = (2.0 * (xyz_all[free_mask] - aabb[0]) / (aabb[1] - aabb[0] + 1e-8) - 1
         ).clamp(-1, 1).view(1, -1, 1, 1, 3)
    vals = logE_pp[free_mask]
    ws = torch.zeros(1, 1, rz, ry, rx, requires_grad=True)
    F.grid_sample(ws, q, mode="bilinear", align_corners=True).reshape(-1).dot(
        torch.ones_like(vals)).backward()
    W = ws.grad.clone()
    vs = torch.zeros(1, 1, rz, ry, rx, requires_grad=True)
    F.grid_sample(vs, q, mode="bilinear", align_corners=True).reshape(-1).dot(vals).backward()
    V = vs.grad.clone()
    with torch.no_grad():
        field.grid.copy_(torch.where(W > 1e-6, V / W.clamp(min=1e-6),
                                     torch.full_like(W, float(vals.mean()))))


def plot_E_projections(xyz_free, logE_rec, logE_gt, path, aabb=None, res=None,
                       xyz_anchor=None) -> None:
    """3 plane projections, 2 rows: recovered log10 E (top) and |err| (bottom).
    Grid lines = field voxel corners; grey x = anchors."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = xyz_free.numpy()
    rec, gt = logE_rec.numpy(), logE_gt.numpy()
    err = np.abs(rec - gt)
    res_xyz = (res, res, res) if isinstance(res, int) else res
    planes = (("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6))
    vlo, vhi = float(min(rec.min(), gt.min())), float(max(rec.max(), gt.max()))
    for row, (val, name, cmap, vmm) in enumerate((
            (rec, "recovered log10 E", "viridis", (vlo, vhi)),
            (err, "|log10 E - GT|", "inferno", (0.0, max(float(err.max()), 1e-6))))):
        order = np.argsort(val)
        for ax, (na, nb, a, b) in zip(axes[row], planes):
            if aabb is not None and res_xyz is not None:
                for t in np.linspace(float(aabb[0][a]), float(aabb[1][a]), res_xyz[a]):
                    ax.axvline(t, color="0.85", lw=0.6, zorder=0)
                for t in np.linspace(float(aabb[0][b]), float(aabb[1][b]), res_xyz[b]):
                    ax.axhline(t, color="0.85", lw=0.6, zorder=0)
            if xyz_anchor is not None:
                pa = xyz_anchor.numpy()
                ax.scatter(pa[:, a], pa[:, b], marker="x", s=10, c="0.45", lw=0.8, zorder=1)
            sc = ax.scatter(p[order, a], p[order, b], c=val[order], s=2,
                            cmap=cmap, vmin=vmm[0], vmax=vmm[1])
            ax.set_xlabel(na)
            ax.set_ylabel(nb if ax is not axes[row][0] else f"{name}\n{nb}")
            ax.set_aspect("equal")
        fig.colorbar(sc, ax=axes[row], shrink=0.85)
    fig.suptitle("E field at free particles — top: recovered log10 E, "
                 "bottom: |err| (dead zones light up)", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_E_grid_nodes(grid, gt_grid, aabb, xyz_free, path) -> None:
    """The learned DOF: scalar log10-E nodes, sliced along the smallest axis.
    Node color = recovered log10 E, size ~ support, red edge = starved."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    rz, ry, rx = grid.shape[2:]
    res_xyz = (rx, ry, rz)
    probe = torch.zeros(1, 1, rz, ry, rx, requires_grad=True)
    p = 2.0 * (xyz_free - aabb[0]) / (aabb[1] - aabb[0] + 1e-8) - 1.0
    F.grid_sample(probe, p.clamp(-1.0, 1.0).view(1, -1, 1, 1, 3),
                  mode="bilinear", align_corners=True).sum().backward()
    Sn = probe.grad[0, 0].numpy().transpose(2, 1, 0)            # (rx,ry,rz)
    Vn = grid[0, 0].detach().numpy().transpose(2, 1, 0)         # (rx,ry,rz) log10 E
    Gn = gt_grid[0, 0].numpy().transpose(2, 1, 0)
    coords = [np.linspace(float(aabb[0][k]), float(aabb[1][k]), res_xyz[k]) for k in range(3)]
    s_ax = int(np.argmin(res_xyz))
    rem = [k for k in (0, 1, 2) if k != s_ax]
    a, b = (rem if res_xyz[rem[0]] >= res_xyz[rem[1]] else rem[::-1])
    names = "xyz"
    n_p = res_xyz[s_ax]
    C0, C1 = np.meshgrid(coords[rem[0]], coords[rem[1]], indexing="ij")
    X = C0 if a == rem[0] else C1
    Y = C0 if b == rem[0] else C1
    fig, axes = plt.subplots(1, n_p, figsize=(3.4 * n_p, 3.6), sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    smax = float(Sn.max())
    vlo, vhi = float(min(Vn.min(), Gn.min())), float(max(Vn.max(), Gn.max()))
    for ks, ax in zip(range(n_p), axes):
        sl = [slice(None)] * 3; sl[s_ax] = ks
        V2, S2 = Vn[tuple(sl)], Sn[tuple(sl)]
        ec = ["red" if si < 1.0 else "0.3" for si in S2.ravel()]
        sc = ax.scatter(X.ravel(), Y.ravel(), c=V2.ravel(),
                        s=20 + 200 * S2.ravel() / max(smax, 1e-12),
                        cmap="viridis", vmin=vlo, vmax=vhi, edgecolors=ec, lw=0.8)
        ax.set_title(f"{names[s_ax]}={coords[s_ax][ks]:.3f}", fontsize=9)
        ax.set_xlabel(names[a]); ax.set_aspect("equal")
    axes[0].set_ylabel(names[b])
    fig.colorbar(sc, ax=axes, label="node log10 E", shrink=0.8)
    fig.suptitle(f"E grid NODES (learned DOF), sliced along {names[s_ax]} — "
                 "size ~ support, red edge = starved", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    t0 = time.time()
    ap = ArgumentParser(description="E-field recovery (traj)")
    ap.add_argument("--scene_cache",
                    default=f"{GEN}/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt")
    ap.add_argument("--config", default="config/ours/telephone.json")
    ap.add_argument("--gt_kind", default="uniform", choices=["uniform", "ramp", "circular"])
    ap.add_argument("--gt_logE", default=5.0, type=float, help="uniform GT log10 E")
    ap.add_argument("--gt_ramp", nargs=2, default=[4.5, 5.5], type=float,
                    help="ramp GT: log10 E from lo (anchor end) to hi (tip)")
    ap.add_argument("--gt_nu", default=0.3, type=float)
    ap.add_argument("--init_logE", default=4.0, type=float, help="uniform init of fit field")
    ap.add_argument("--obs", default="ym", help="v0 direction (VDIR key or x,y,z)")
    ap.add_argument("--res", default="4x4x16", type=str)
    ap.add_argument("--rot_z_deg", default=67.6, type=float)
    ap.add_argument("--n_frames", default=8, type=int)
    ap.add_argument("--gt_frames", default=14, type=int)
    ap.add_argument("--anchor_mass_scale", default=1e4, type=float)
    ap.add_argument("--mpm_iter_cnt", default=64, type=int)
    ap.add_argument("--iter_cnt", default=120, type=int)
    ap.add_argument("--E_lr", default=0.2, type=float)
    ap.add_argument("--fine_lr", default=0.02, type=float)
    ap.add_argument("--tv", default=1e-3, type=float)
    ap.add_argument("--patience", default=24, type=int)
    ap.add_argument("--min_iters", default=30, type=int)
    ap.add_argument("--estop_tol", default=0.003, type=float)
    ap.add_argument("--joint_v0", action="store_true",
                    help="ALSO learn scalar v0 (zero init): warmup v0 -> joint{v0,E-grid}")
    ap.add_argument("--v0_field", action="store_true",
                    help="rung-3 DOUBLE FIELD: v0 is also a voxel field; "
                         "warmup v0-field -> joint{v0-grid, E-grid}")
    ap.add_argument("--v0_field_init_std", default=0.05, type=float)
    ap.add_argument("--gt_v0_variant", default=None, type=str,
                    help="GT v0 as a non-uniform field (ramp_y|ramp_x|mid_kick|true_bend); "
                         "default = uniform scalar from --obs")
    ap.add_argument("--gt_v0_scale", default=0.5, type=float)
    ap.add_argument("--warmup_iters", default=40, type=int)
    ap.add_argument("--v0_lr", default=0.025, type=float)
    ap.add_argument("--ti_mem_frac", default=0.3, type=float)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out_root", default="output/ours_efield", type=str)
    args = ap.parse_args()

    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    res = (tuple(int(p) for p in args.res.lower().split("x"))
           if "x" in args.res else (int(args.res),) * 3)
    v0 = VDIR[args.obs] if args.obs in VDIR else [float(x) for x in args.obs.split(",")]

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
    pad = 2.0 * phys_args.voxel_size
    aabb = torch.stack([xyz.min(0).values - pad, xyz.max(0).values + pad])
    z_lo, z_hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    flip_z = bool(xyz[anchor_mask][:, 2].mean() > xyz[free][:, 2].mean())

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

    # ---- GT field + rollout ----
    gt_field = EVoxelField(aabb.cpu(), res=res)
    if args.gt_kind == "uniform":
        gt_field.set_uniform_(args.gt_logE)
    elif args.gt_kind == "ramp":
        gt_field.fill_ramp_(args.gt_ramp[0], args.gt_ramp[1], z_lo, z_hi, flip_z)
    elif args.gt_kind == "circular":
        # branch-dependent E along the cord arc length, BUCKETED into the grid so
        # the GT is exactly grid-representable (zero floor). Validated res16: the
        # two strands keep distinct E (interp-back err 0.004 dex).
        fill_circular_grid(gt_field, xyz.cpu(), (~anchor_mask).cpu(),
                           z_lo, z_hi, args.gt_ramp[0], args.gt_ramp[1])
    est.set_E_field(gt_field, xyz, lr=0.0)
    set_params(est, args.gt_logE, args.gt_nu, v0)  # E scalar unused now; sets v0/nu
    gt_logE_p = gt_field(xyz).detach().cpu()                        # (N,) per-particle
    # GT v0 field (rung-3 double NON-uniform): inject a v0-variant field for the GT
    # rollout too, so both GT material AND GT initial-velocity are non-uniform.
    gt_v0_pf = None
    if args.gt_v0_variant is not None:
        from v0_field_ours import V0VoxelField as _V0F, fill_profile_grid as _fillv0
        gt_v0f = _V0F(aabb.cpu(), res=res)
        _fillv0(gt_v0f, args.gt_v0_variant, args.gt_v0_scale, z_lo, z_hi, flip_z)
        est.set_v0_field(gt_v0f, xyz, lr=0.0)
        gt_v0_pf = (gt_v0f(xyz).detach() * (~anchor_mask).float().unsqueeze(1)).cpu()  # (N,3)
        print(f"[ef] GT v0 = '{args.gt_v0_variant}' x{args.gt_v0_scale} field; free |v0| "
              f"mean {gt_v0_pf[(~anchor_mask).cpu()].norm(dim=-1).mean():.3f}")
    est.max_f = args.gt_frames
    gts = rollout_collect_surfaces(est)
    strain = strain_proxy(gts[0], gts[-1]).cpu()                   # (N,) local strain
    print(f"[ef] GT {args.gt_kind} logE range [{gt_logE_p[free.cpu()].min():.2f},"
          f"{gt_logE_p[free.cpu()].max():.2f}] | free strain mean "
          f"{strain[free.cpu()].mean():.4f} max {strain[free.cpu()].max():.4f}")

    # ---- fit field (uniform init, wrong) ----
    fit_field = EVoxelField(aabb.cpu(), res=res)
    fit_field.set_uniform_(args.init_logE)
    n_starved = fit_field.freeze_starved_(xyz[free].cpu(), 1.0)
    est.set_E_field(fit_field, xyz, lr=args.E_lr)  # builds est.optimizer = [E-grid]
    # joint_v0: v0 is UNKNOWN (zero init, learned); else fixed to GT.
    # v0_field (rung-3, DOUBLE field): v0 is also a voxel field, not scalar.
    v0_field = None
    if args.v0_field:
        from v0_field_ours import V0VoxelField, eval_grid_at as eval_v0grid
        v0_field = V0VoxelField(aabb.cpu(), res=res)
        v0_field.randomize_(args.v0_field_init_std, seed=0)
        v0_field.freeze_starved_(xyz[free].cpu(), 1.0)
        est.set_v0_field(v0_field, xyz, lr=args.v0_lr)  # sets _v0_field + vel_optimizer
        v0_param = v0_field.grid
    fit_v0 = [0.0, 0.0, 0.0] if (args.joint_v0 or args.v0_field) else v0
    set_params(est, args.init_logE, args.gt_nu, fit_v0)
    est.gts = gts
    est.load_gts(gts)
    print(f"[ef] fit res {res} ({res[0]*res[1]*res[2]} nodes), starved {n_starved}, "
          f"init logE {args.init_logE}, tv {args.tv}, obs {args.obs} v0={v0} "
          f"{'JOINT v0+E (warmup '+str(args.warmup_iters)+')' if args.joint_v0 else 'E-only (v0 fixed)'}")

    fm = free.cpu()
    xyz_c, aabb_c = xyz.cpu(), fit_field.aabb.cpu()
    gt_lp_free = gt_logE_p[fm]
    gt_v = np.array(v0)
    grid_traj, err_traj, errw_traj, loss_traj, v0_traj = [], [], [], [], []
    best_loss, best_grid, last_improve, last_move, dropped = float("inf"), None, 0, 0, False
    w = strain[fm].clamp(min=0).numpy(); w = w / max(w.sum(), 1e-12)   # strain weight
    est.max_f = args.n_frames

    # ---- phase 1: v0 warmup at the (wrong, uniform) init E-field ----
    # v0 side param: the field grid (rung-3) or scalar init_vel (rung gate)
    if args.joint_v0 or args.v0_field:
        v0p = v0_field.grid if args.v0_field else est.init_vel
        warm_opt = torch.optim.Adam([{"params": v0p, "lr": args.v0_lr}])
        for wi in range(args.warmup_iters):
            warm_opt.zero_grad()
            est.loss[None] = 0.0
            try:
                forward_bounded(est, max_halvings=3)
            except CFLExhausted:
                print(f"[ef] warmup {wi}: CFL exhausted"); break
            gic_backward(est)
            warm_opt.step()
        if args.v0_field:
            vm = eval_v0grid(v0_field.grid.detach().cpu(), aabb_c, xyz_c)[fm].mean(0)
            print(f"[ef] warmup done: v0-field free mean {vm.numpy().round(3).tolist()} (GT {v0})")
        else:
            print(f"[ef] warmup done: v0 {est.init_vel.detach().cpu().numpy().round(3).tolist()} (GT {v0})")
        # joint optimizer: v0 (field grid or scalar) + E-grid; nu frozen (excluded)
        est.optimizer = torch.optim.Adam([
            {"params": v0p, "lr": args.v0_lr, "name": "velocity"},
            {"params": fit_field.grid, "lr": args.E_lr, "name": "Youngs modulus"}])
    for it in range(args.iter_cnt):
        prev_grid = fit_field.grid.detach().cpu().clone()
        est.optimizer.zero_grad()
        est.loss[None] = 0.0
        try:
            forward_bounded(est, max_halvings=3)
        except CFLExhausted:
            print(f"[ef] iter {it}: CFL exhausted, keep best"); break
        loss = float(est.loss[None])
        gic_backward(est)
        if args.tv > 0:
            tv = fit_field.regularization()
            if v0_field is not None:
                tv = tv + v0_field.regularization()
            (args.tv * tv).backward()
        est.optimizer.step()

        lp = eval_Egrid_at(fit_field.grid.detach().cpu(), aabb_c, xyz_c)[fm]
        e_all = float((lp - gt_lp_free).abs().mean())
        e_w = float((np.abs((lp - gt_lp_free).numpy()) * w).sum())  # strain-weighted
        grid_traj.append(fit_field.grid.detach().cpu().clone())
        err_traj.append(e_all); errw_traj.append(e_w); loss_traj.append(loss)
        if v0_field is not None:
            v0_traj.append(eval_v0grid(v0_field.grid.detach().cpu(), aabb_c, xyz_c)[fm]
                           .mean(0).numpy().tolist())
        else:
            v0_traj.append(est.init_vel.detach().cpu().numpy().tolist())
        if loss < best_loss:
            best_loss, best_grid = loss, fit_field.grid.detach().cpu().clone()
        if loss < best_loss * 1.02:
            last_improve = it
        move = (fit_field.grid.detach().cpu() - prev_grid).abs().max().item()
        if move >= args.estop_tol:
            last_move = it
        print(f"[ef] iter {it} loss {loss:.6f} | logE err all {e_all:.3f} "
              f"strain-w {e_w:.3f} | best loss {best_loss:.6f}")
        plateau = it + 1 >= args.min_iters and (it - last_improve) >= args.patience
        if plateau and not dropped:
            dropped = True
            fit_field.grid.data.copy_(best_grid.to(est.device))
            est.optimizer.state.pop(fit_field.grid, None)
            for g in est.optimizer.param_groups:
                g["lr"] = args.fine_lr
            last_improve = it
            print(f"[ef] plateau -> restore best + fine lr {args.fine_lr}")
            continue
        if plateau and dropped and (it - last_move) >= args.patience:
            print(f"[ef] early stop at iter {it}"); break

    # ---- export ----
    fit_field.grid.data.copy_(best_grid.to(est.device))
    lp_best = eval_Egrid_at(best_grid, aabb_c, xyz_c)[fm]
    err_all = float((lp_best - gt_lp_free).abs().mean())
    err_w = float((np.abs((lp_best - gt_lp_free).numpy()) * w).sum())
    # observable subset = top half by strain proxy
    obs_mask = strain[fm].numpy() >= np.median(strain[fm].numpy())
    err_obs = float(np.abs((lp_best - gt_lp_free).numpy())[obs_mask].mean())
    v0_field_relL2 = None
    if v0_field is not None:
        vrec = eval_v0grid(v0_field.grid.detach().cpu(), aabb_c, xyz_c)[fm]   # (M,3)
        # per-particle GT v0: ramp/variant field if given, else uniform gt_vel
        gt_vf = gt_v0_pf[fm] if gt_v0_pf is not None else \
            torch.tensor(v0, dtype=torch.float32).expand(int(fm.sum()), 3)
        gsv = max(float(gt_vf.norm(dim=-1).mean()), 1e-12)
        v0_est = vrec.mean(0).tolist()
        v0_field_relL2 = float((vrec - gt_vf)[:, :2].norm(dim=-1).mean() / gsv)  # xy-only
        v0_rel = float((vrec - gt_vf).norm(dim=-1).mean() / gsv)  # full per-particle
    else:
        v0_est = est.init_vel.detach().cpu().numpy().tolist()
        v0_rel = float(np.linalg.norm(np.array(v0_est) - gt_v) / max(np.linalg.norm(gt_v), 1e-12))
    if not (args.joint_v0 or args.v0_field):
        set_params(est, args.init_logE, args.gt_nu, v0)  # E-only: v0 known = GT
    # joint: keep the LEARNED v0 (scalar init_vel, or _v0_field grid) for pred rollout
    est.max_f = args.gt_frames
    pred = rollout_collect_surfaces(est)
    save_overlay_gif(gts, pred, os.path.join(out_dir, "overlay.gif"),
                     fit_frames=args.n_frames)
    result = {
        "scenario": "ours_efield_traj",
        "gt_kind": args.gt_kind, "gt_logE": args.gt_logE, "gt_ramp": args.gt_ramp,
        "gt_nu": args.gt_nu, "init_logE": args.init_logE, "obs": args.obs, "v0": v0,
        "res": list(res), "n_starved": n_starved, "tv": args.tv,
        "logE_err_all": err_all, "logE_err_strain_w": err_w,
        "logE_err_observable_half": err_obs,
        "logE_err_traj": err_traj, "logE_err_strain_w_traj": errw_traj,
        "losses": loss_traj, "n_frames": args.n_frames,
        "joint_v0": args.joint_v0, "v0_field_mode": args.v0_field,
        "warmup_iters": args.warmup_iters if (args.joint_v0 or args.v0_field) else 0,
        "v0_gt": v0, "v0_estimated": v0_est, "v0_rel_err": v0_rel,
        "v0_field_relL2_xy": v0_field_relL2,
        "v0_traj": v0_traj if (args.joint_v0 or args.v0_field) else None,
        "wall_time_s": time.time() - t0,
    }
    json.dump(result, open(os.path.join(out_dir, "result.json"), "w"), indent=2)
    # ckpt the recovered + GT grids so all viz is regenerable OFFLINE (no re-run)
    torch.save({"best_grid": best_grid, "gt_grid": gt_field.grid.detach().cpu(),
                "aabb": aabb_c, "res": list(res), "grid_traj": grid_traj},
               os.path.join(out_dir, "ckpt.pt"))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    draw_curve(loss_traj, out_dir, name="loss")
    # err curves
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(err_traj, color="0.7", label="all free (incl. low-strain dead zone)")
    ax.plot(errw_traj, color="tab:red", label="strain-weighted (observable)")
    ax.set_yscale("log"); ax.set_xlabel("iter"); ax.set_ylabel("mean |log10 E - GT|")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "Eerr.png")); plt.close(fig)
    # scatter: recovered logE vs strain proxy (shows dead zone directly)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    d = strain[fm].numpy()
    sc = ax.scatter(d, lp_best.numpy(), s=3, c=gt_lp_free.numpy(), cmap="viridis", alpha=0.5)
    fig.colorbar(sc, label="GT log10 E")
    if args.gt_kind == "uniform":
        ax.axhline(args.gt_logE, color="r", ls="--", label="GT logE")
    ax.axhline(args.init_logE, color="k", ls=":", label="init logE")
    ax.set_xlabel("GT per-particle local strain")
    ax.set_ylabel("recovered log10 E")
    ax.set_title(f"{args.tag}: recovered E vs observability\n"
                 f"err all {err_all:.3f} / observable-half {err_obs:.3f}", fontsize=9)
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "E_vs_strain.png")); plt.close(fig)
    # profile along z (the key view for ramp)
    zt = ((xyz_c[fm][:, 2] - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0, 1)
    if flip_z:
        zt = 1.0 - zt
    order = np.argsort(zt.numpy())
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.scatter(zt.numpy(), lp_best.numpy(), s=4, alpha=0.3, label="recovered")
    ax.plot(zt.numpy()[order], gt_lp_free.numpy()[order], "r", lw=1.5, label="GT")
    ax.set_xlabel("zt (0=anchor end)"); ax.set_ylabel("log10 E")
    ax.set_title(f"{args.tag}: E profile along cord"); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(out_dir, "profile_1d.png")); plt.close(fig)

    # field projections + grid-node views (the v0-field viz suite, E-scalar)
    plot_E_projections(xyz_c[fm], lp_best, gt_lp_free,
                       os.path.join(out_dir, "E_proj.png"),
                       aabb=aabb_c, res=res, xyz_anchor=xyz_c[~fm])
    plot_E_grid_nodes(best_grid, gt_field.grid.detach().cpu(), aabb_c, xyz_c[fm],
                      os.path.join(out_dir, "E_grid.png"))

    # panel.gif via the shared pipeline (make_panel recognises efield scenarios
    # and collects the pre-saved figures) -- no bespoke assembly here
    import subprocess, sys
    subprocess.run([sys.executable, "make_panel.py", "--per_run", "--runs", out_dir],
                   cwd=os.path.dirname(os.path.abspath(__file__)))

    v0msg =((f" | v0 {[round(x,3) for x in v0_est]} (mean err {v0_rel:.2%}"
              + (f", field xy {v0_field_relL2:.2%}" if v0_field_relL2 is not None else "") + ")")
             if (args.joint_v0 or args.v0_field) else "")
    print(f"[ef] DONE {args.tag}: logE err all {err_all:.3f} | observable-half "
          f"{err_obs:.3f} | strain-w {err_w:.3f}{v0msg}, wall {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()

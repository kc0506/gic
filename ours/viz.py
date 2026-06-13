# coding=utf-8
"""Visualization helpers: rollout/overlay gifs, field/grid/profile/param plots.

Pure plotting — no GPU, no simulator state. Each function takes already-collected
tensors/arrays and writes a file. matplotlib is imported lazily inside each
function (Agg backend) so importing this module is cheap.
"""
import numpy as np
import torch


def save_overlay_gif(gt_list: list, pred_list: list, path: str, max_pts: int = 1500,
                     labels: tuple = ("GT", "pred"), fit_frames: "int | None" = None) -> None:
    """A (blue) vs B (red) rollout overlay: 3D + xy/xz/yz projections per frame.

    fit_frames: frames >= this index were NOT seen by the fit (held-out
    extrapolation); pred turns ORANGE there and the title flags it.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    sel = np.random.default_rng(0).permutation(gt_list[0].shape[0])[:max_pts]
    gt = np.stack([g.cpu().numpy()[sel] for g in gt_list])      # (F, M, 3)
    pr = np.stack([p.cpu().numpy()[sel] for p in pred_list])    # (F, M, 3)
    allp = np.concatenate([gt, pr]).reshape(-1, 3)
    mins, maxs = allp.min(0), allp.max(0)
    fig = plt.figure(figsize=(14.5, 3.9))
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")
    axes2d = [fig.add_subplot(1, 4, k) for k in (2, 3, 4)]
    planes = (("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2))

    def update(f: int):
        extrap = fit_frames is not None and f >= fit_frames
        pcol = "tab:orange" if extrap else "tab:red"
        plab = f"{labels[1]} (EXTRAP)" if extrap else labels[1]
        ax3d.cla()
        ax3d.scatter(gt[f][:, 0], gt[f][:, 2], gt[f][:, 1], s=1, c="tab:blue", label=labels[0])
        ax3d.scatter(pr[f][:, 0], pr[f][:, 2], pr[f][:, 1], s=1, c=pcol, label=plab, alpha=0.6)
        ax3d.set_xlim(mins[0], maxs[0]); ax3d.set_ylim(mins[2], maxs[2]); ax3d.set_zlim(mins[1], maxs[1])
        # plotted as (x, z, y): sim y (main excitation) is VERTICAL, long axis z lies flat
        ax3d.set_xlabel("x"); ax3d.set_ylabel("z"); ax3d.set_zlabel("y")
        ax3d.legend(loc="upper right", fontsize=7)
        for ax, (na, nb, a, b) in zip(axes2d, planes):
            ax.cla()
            ax.scatter(gt[f][:, a], gt[f][:, b], s=1, c="tab:blue")
            ax.scatter(pr[f][:, a], pr[f][:, b], s=1, c=pcol, alpha=0.6)
            ax.set_xlim(mins[a], maxs[a]); ax.set_ylim(mins[b], maxs[b])
            ax.set_xlabel(na); ax.set_ylabel(nb)
            ax.set_aspect("equal")
        if extrap:
            fig.suptitle(f"frame {f} — EXTRAPOLATION (beyond {fit_frames}f fit window)",
                         fontsize=10, color="darkorange")
        else:
            fig.suptitle(f"frame {f}", fontsize=10, color="black")

    anim = FuncAnimation(fig, update, frames=len(gt_list))
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def plot_axis_profiles(gt_list: list, pred_list: list, free_mask: torch.Tensor,
                       path: str, fit_frames: "int | None" = None) -> None:
    """Per-frame mean +/- std of FREE particle positions per axis, GT vs pred.

    fit_frames: frames >= this are held-out extrapolation (orange shading).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fm = free_mask.cpu().numpy()
    gt = np.stack([g.cpu().numpy()[fm] for g in gt_list])    # (F, M, 3)
    pr = np.stack([p.cpu().numpy()[fm] for p in pred_list])  # (F, M, 3)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for k, c in enumerate("xyz"):
        ax = axes[k]
        for arr, name, col in ((gt, "GT", "tab:blue"), (pr, "pred", "tab:red")):
            m, s = arr[:, :, k].mean(1), arr[:, :, k].std(1)
            ax.plot(m, color=col, label=name)
            ax.fill_between(range(len(m)), m - s, m + s, color=col, alpha=0.15)
        if fit_frames is not None and fit_frames < len(gt):
            ax.axvspan(fit_frames - 0.5, len(gt) - 1, color="orange", alpha=0.08)
            ax.axvline(fit_frames - 0.5, color="darkorange", ls=":", lw=1,
                       label="fit|extrap")
        ax.set_title(f"{c} (free particles)")
        ax.set_xlabel("frame")
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_field_projections(xyz_free: torch.Tensor, v_field: torch.Tensor,
                           gt_v: torch.Tensor, path: str,
                           aabb: "torch.Tensor | None" = None, res=None,
                           xyz_anchor: "torch.Tensor | None" = None,
                           n_arrows: int = 250) -> None:
    """Three plane projections of the recovered v0 field at free particles.

    xyz_free (M,3) rest positions, v_field (M,3) recovered v0; gt_v (3,) uniform
    GT or (M,3) per-particle GT (non-uniform variants). Scatter colored by
    per-particle err / mean|gt|; quiver = recovered (black) vs GT (red) at the
    same subsampled positions. aabb (2,3) + res ((rx,ry,rz)): draw the field's
    voxel-corner grid lines in the background. xyz_anchor (A,3): frozen
    particles drawn as grey x.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = xyz_free.numpy()
    v = v_field.numpy()
    g = gt_v.numpy()
    if g.ndim == 1:
        g = np.broadcast_to(g, v.shape)
    gscale = max(float(np.linalg.norm(g, axis=-1).mean()), 1e-12)  # mean |gt|
    rel_full = np.linalg.norm(v - g, axis=-1) / gscale             # (M,) 3D err
    rel_xy = np.linalg.norm((v - g)[:, :2], axis=-1) / gscale      # (M,) z ignored
    sel = np.random.default_rng(0).permutation(p.shape[0])[:n_arrows]
    res_xyz = ((res, res, res) if isinstance(res, int) else res)
    planes = (("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2))
    vmax = max(float(rel_full.max()), 1e-6)       # shared color scale, both rows
    fig, axes = plt.subplots(2, 3, figsize=(14, 8.6))
    for row, (rel, row_name) in enumerate(((rel_full, "FULL 3D err"),
                                           (rel_xy, "XY-only err (z ignored)"))):
        # draw low->high err so overlapping (projected) particles show the MAX
        # err of the stack on top -- deterministic max-projection
        order = np.argsort(rel)
        p_s, rel_s = p[order], rel[order]
        for ax, (na, nb, a, b) in zip(axes[row], planes):
            if aabb is not None and res_xyz is not None:
                for t in np.linspace(float(aabb[0][a]), float(aabb[1][a]), res_xyz[a]):
                    ax.axvline(t, color="0.85", lw=0.6, zorder=0)
                for t in np.linspace(float(aabb[0][b]), float(aabb[1][b]), res_xyz[b]):
                    ax.axhline(t, color="0.85", lw=0.6, zorder=0)
            if xyz_anchor is not None:
                pa = xyz_anchor.numpy()
                ax.scatter(pa[:, a], pa[:, b], marker="x", s=10, c="0.45",
                           lw=0.8, zorder=1, label="anchor")
            sc = ax.scatter(p_s[:, a], p_s[:, b], c=rel_s, s=2, cmap="viridis",
                            vmin=0.0, vmax=vmax)
            qscale = 12.0 * gscale
            ax.quiver(p[sel, a], p[sel, b], v[sel, a], v[sel, b], color="k",
                      width=0.0025, alpha=0.55, scale=qscale, label="recovered")
            ax.quiver(p[sel, a], p[sel, b], g[sel, a], g[sel, b], color="red",
                      width=0.0025, alpha=0.45, scale=qscale, label="GT")
            ax.set_xlabel(na)
            ax.set_ylabel(nb if ax is not axes[row][0] else f"{row_name}\n{nb}")
            ax.set_aspect("equal")
            ax.legend(fontsize=7, loc="lower right")
    fig.colorbar(sc, ax=axes, label="err / mean|gt| (shared scale)", shrink=0.85)
    title = ("recovered v0 field at free particles (best iter) — "
             "stacked particles: MAX err on top")
    if res_xyz is not None:
        title += f" — grid lines = field voxel corners (res {res_xyz})"
    fig.suptitle(title, fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_profile_1d(zt_free: torch.Tensor, v_best: torch.Tensor,
                    gt_pf: torch.Tensor, path: str) -> None:
    """Recovered vs GT v0 as 1D profiles along zt (free particles).

    zt_free (M,) position along the cord (0 = anchor end), v_best (M,3)
    recovered, gt_pf (M,3) GT. The chosen non-uniform GTs vary only along z,
    so this is the most direct fidelity view.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    zt = zt_free.numpy()
    order = np.argsort(zt)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    for k, axn in enumerate("xyz"):
        ax = axes[k]
        ax.scatter(zt, v_best[:, k].numpy(), s=1.5, alpha=0.25, label="recovered")
        ax.plot(zt[order], gt_pf[order, k].numpy(), color="r", lw=1.3, label="GT")
        ax.set_title(f"v0_{axn}(zt)", fontsize=9)
        ax.set_xlabel("zt (0 = anchor end)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_grid_nodes(grid: torch.Tensor, aabb: torch.Tensor, gt: torch.Tensor,
                    xyz_free: torch.Tensor, path_quiver: str, path_hist: str) -> None:
    """Visualize the ACTUAL learned DOF: the grid-node vectors themselves.

    grid (1,3,rz,ry,rx) -- grid_sample layout: W<->x, H<->y, D<->z, i.e. node
    (ix,iy,iz) = grid[0,:,iz,iy,ix]. aabb (2,3); gt = (3,) uniform GT vector OR
    a (1,3,rz,ry,rx) GT grid (field-variant GT); xyz_free (M,3) free-particle
    rest positions.

    Per-node SUPPORT = total trilinear weight from free particles reaching the
    node (adjoint of the interp). Support ~ 0 => no gradient ever, init value
    persists. quiver fig: slices along the SMALLEST grid axis (anisotropic
    grids slice across the thin direction); node color = err vs GT, size ~
    support, red edge = starved, red arrows = GT nodes. hist fig: per-component
    node values, supported nodes only.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch.nn.functional as F

    rz, ry, rx = grid.shape[2:]
    res_xyz = (rx, ry, rz)
    probe = torch.zeros(1, 1, rz, ry, rx, requires_grad=True)
    p = 2.0 * (xyz_free - aabb[0]) / (aabb[1] - aabb[0] + 1e-8) - 1.0
    F.grid_sample(probe, p.clamp(-1.0, 1.0).view(1, -1, 1, 1, 3), mode="bilinear",
                  align_corners=True).sum().backward()
    # transpose everything to (rx,ry,rz[,3]) so axis k <-> spatial axis k
    Sn = probe.grad[0, 0].numpy().transpose(2, 1, 0)             # (rx,ry,rz)
    Vn = grid[0].detach().numpy().transpose(3, 2, 1, 0)          # (rx,ry,rz,3)
    if gt.dim() == 1:
        Gn = np.broadcast_to(gt.numpy(), Vn.shape)
    else:
        Gn = gt[0].detach().numpy().transpose(3, 2, 1, 0)        # (rx,ry,rz,3)
    gscale = max(float(np.linalg.norm(Gn.reshape(-1, 3), axis=-1).mean()), 1e-12)
    errn = np.linalg.norm(Vn - Gn, axis=-1) / gscale             # (rx,ry,rz)
    coords = [np.linspace(float(aabb[0][k]), float(aabb[1][k]), res_xyz[k])
              for k in range(3)]

    s_ax = int(np.argmin(res_xyz))                # slice along the smallest axis
    rem = [k for k in (0, 1, 2) if k != s_ax]
    a, b = (rem if res_xyz[rem[0]] >= res_xyz[rem[1]] else rem[::-1])  # a = wide axis
    names = "xyz"
    n_panels = res_xyz[s_ax]
    C0, C1 = np.meshgrid(coords[rem[0]], coords[rem[1]], indexing="ij")
    X = C0 if a == rem[0] else C1
    Y = C0 if b == rem[0] else C1
    panel_w = 3.1 * max(1.0, (coords[a][-1] - coords[a][0]) /
                        max(coords[b][-1] - coords[b][0], 1e-6) * 0.6)
    fig, axes = plt.subplots(1, n_panels, figsize=(panel_w * n_panels, 3.6),
                             sharex=True, sharey=True)
    axes = np.atleast_1d(axes)
    smax = float(Sn.max())
    qscale = 12.0 * gscale
    for k_s, ax in zip(range(n_panels), axes):
        sl = [slice(None)] * 3
        sl[s_ax] = k_s
        V2, S2, E2 = Vn[tuple(sl)], Sn[tuple(sl)], errn[tuple(sl)]  # (r0,r1[,3])
        G2 = Gn[tuple(sl)]
        ec = ["red" if si < 1.0 else "none" for si in S2.ravel()]
        sc = ax.scatter(X.ravel(), Y.ravel(), c=E2.ravel(),
                        s=15 + 185 * S2.ravel() / max(smax, 1e-12),
                        cmap="viridis", vmin=0.0, vmax=float(errn.max()),
                        edgecolors=ec, lw=0.8)
        ax.quiver(X.ravel(), Y.ravel(), V2[..., a].ravel(), V2[..., b].ravel(),
                  color="k", width=0.005, alpha=0.7, scale=qscale)
        ax.quiver(X.ravel(), Y.ravel(), G2[..., a].ravel(), G2[..., b].ravel(),
                  color="red", width=0.004, alpha=0.5, scale=qscale)
        ax.set_title(f"{names[s_ax]} = {coords[s_ax][k_s]:.3f}", fontsize=9)
        ax.set_xlabel(names[a])
        ax.set_aspect("equal")
    axes[0].set_ylabel(names[b])
    fig.colorbar(sc, ax=axes, label="node |v0 - gt| / mean|gt|", shrink=0.8)
    fig.suptitle(f"grid NODES (the learned DOF), sliced along {names[s_ax]} — "
                 "size ~ particle support, red edge = starved, red arrows = GT",
                 fontsize=10)
    fig.savefig(path_quiver, dpi=110, bbox_inches="tight")
    plt.close(fig)

    sup_mask = (Sn > 1.0).ravel()                 # >=1 particle-equivalent
    n_starved = int((~sup_mask).sum())
    flatV = Vn.reshape(-1, 3)
    flatG = Gn.reshape(-1, 3)
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2))
    for k, axn in enumerate("xyz"):
        axes[k].hist(flatV[sup_mask, k], bins=24, color="tab:blue", label="supported")
        if gt.dim() == 1:
            axes[k].axvline(float(gt[k]), color="r", ls="--", label="GT")
        else:  # non-uniform GT: outline hist of GT node values (supported set)
            axes[k].hist(flatG[sup_mask, k], bins=24, histtype="step",
                         color="r", label="GT nodes")
        axes[k].set_title(f"grid node v0_{axn}", fontsize=9)
        axes[k].legend(fontsize=7)
    fig.suptitle(f"supported nodes only ({n_starved} starved omitted)", fontsize=9)
    fig.tight_layout()
    fig.savefig(path_hist, dpi=110)
    plt.close(fig)


def save_rollout_gif(surfaces: list, path: str, max_pts: int = 2000) -> None:
    """Save a 3D scatter GIF of the rollout for human inspection."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    pts0 = surfaces[0].cpu().numpy()
    sel = np.random.default_rng(0).permutation(pts0.shape[0])[:max_pts]
    all_pts = np.stack([s.cpu().numpy()[sel] for s in surfaces])  # (F, M, 3)
    mins, maxs = all_pts.reshape(-1, 3).min(0), all_pts.reshape(-1, 3).max(0)
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(projection="3d")

    def update(f: int):
        ax.cla()
        p = all_pts[f]
        ax.scatter(p[:, 0], p[:, 2], p[:, 1], s=1)
        ax.set_xlim(mins[0], maxs[0])
        ax.set_ylim(mins[2], maxs[2])
        ax.set_zlim(mins[1], maxs[1])
        ax.set_title(f"frame {f}")

    anim = FuncAnimation(fig, update, frames=len(surfaces))
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def plot_param_traj(e_s: list, gt_E: float, gt_nu: float, path: str) -> None:
    """Plot E (log y) and nu trajectories vs GT lines over fit iterations."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Es = [d["Youngs modulus"] for d in e_s if "Youngs modulus" in d]
    nus = [d["Poisson ratio"] for d in e_s if "Poisson ratio" in d]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(Es, label="E (estimated)")
    axes[0].axhline(gt_E, color="r", ls="--", label="E (GT)")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("iter")
    axes[0].legend()
    axes[1].plot(nus, label="nu (estimated)")
    axes[1].axhline(gt_nu, color="r", ls="--", label="nu (GT)")
    axes[1].set_xlabel("iter")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _to_np(x):
    return x.detach().cpu().numpy() if hasattr(x, "detach") else (
        x.numpy() if hasattr(x, "numpy") else np.asarray(x))


def plot_E_gt_vs_pred(gt_logE, pred_logE, path, color=None, color_label="",
                      title="") -> None:
    """Per-particle recovered vs GT log10 E scatter; diagonal y=x = perfect.

    Parametrization-free, so it handles MULTI-VALUED fields (e.g. circular E,
    where two strands at the same z carry different E) cleanly -- unlike a 1D
    profile along z. gt_logE/pred_logE: (M,) tensors or arrays; color: optional
    (M,) hue (e.g. strain proxy or zt).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g, p = _to_np(gt_logE), _to_np(pred_logE)
    lo = float(min(g.min(), p.min()))
    hi = float(max(g.max(), p.max()))
    pad = 0.05 * (hi - lo + 1e-9)
    fig, ax = plt.subplots(figsize=(5.5, 5.2))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="0.5", ls="--",
            lw=1, label="y=x (perfect)", zorder=0)
    if color is not None:
        sc = ax.scatter(g, p, c=_to_np(color), s=5, cmap="viridis", alpha=0.55)
        fig.colorbar(sc, label=color_label)
    else:
        ax.scatter(g, p, s=5, alpha=0.4, color="tab:blue")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal")
    ax.set_xlabel("GT log10 E")
    ax.set_ylabel("recovered log10 E")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_E_z_heatmap(zt, gt_logE, pred_logE, path, bins: int = 40, title="") -> None:
    """2D histogram of GT (zt, log10 E) density (background) + recovered points.

    For circular E the GT splits into TWO bands (two strands at the same z carry
    different E); the recovered points (red) show whether the fit separates them.
    GT is the background density (bins), pred is the foreground particle layer --
    the bg-vs-particle distinction the user asked for. zt/gt_logE/pred_logE: (M,).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    z, g, p = _to_np(zt), _to_np(gt_logE), _to_np(pred_logE)
    elo = float(min(g.min(), p.min()))
    ehi = float(max(g.max(), p.max()))
    pad = 0.05 * (ehi - elo + 1e-9)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    h = ax.hist2d(z, g, bins=bins, range=[[0.0, 1.0], [elo - pad, ehi + pad]],
                  cmap="Blues")
    fig.colorbar(h[3], label="GT particle count per (z, logE) bin")
    ax.scatter(z, p, s=4, c="tab:red", alpha=0.35, label="recovered (per particle)")
    ax.set_xlabel("zt (0 = anchor end)")
    ax.set_ylabel("log10 E")
    ax.set_title(title, fontsize=9)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_E_projections(xyz_free, logE_rec, logE_gt, path, aabb=None, res=None,
                       xyz_anchor=None) -> None:
    """3 plane projections, 2 rows: recovered log10 E (top) and |err| (bottom).
    Grid lines = field voxel corners; grey x = anchors."""
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


def plot_joint_diagnostics(loss_traj, e_err_traj, v0_rel_traj, warmup_iters: int,
                           path, title="") -> None:
    """Joint v0+E diagnostic: loss, E-field err, and v0 xy-relL2 on ONE axis vs a
    CONTINUOUS iter index (warmup -> joint, dashed line at the boundary).

    This is the curve missing from the old efield_fit that made it impossible to
    tell "v0 fit is bad" from "v0 fit is fine but E still won't recover": if v0
    relL2 drops and stays low while E err stays high, it's the latter. All three
    are lists indexed over the SAME iters (warmup entries first, then joint).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = len(loss_traj)
    it = list(range(n))
    fig, axL = plt.subplots(figsize=(7.5, 4.5))
    axL.set_yscale("log")
    axL.set_xlabel("iter (warmup -> joint)")
    axL.set_ylabel("loss  /  E |log10 err|", color="tab:blue")
    l1, = axL.plot(it, loss_traj, color="tab:blue", lw=1.2, label="loss")
    lines = [l1]
    if e_err_traj is not None and len(e_err_traj):
        off = n - len(e_err_traj)  # E err only recorded in the joint phase
        l2, = axL.plot(range(off, n), e_err_traj, color="tab:purple", lw=1.2,
                       label="E |log10 err| (all free)")
        lines.append(l2)
    axR = axL.twinx()
    axR.set_ylabel("v0 xy relL2", color="tab:red")
    if v0_rel_traj is not None and len(v0_rel_traj):
        l3, = axR.plot(it[:len(v0_rel_traj)], v0_rel_traj, color="tab:red", lw=1.2,
                       label="v0 xy relL2")
        lines.append(l3)
    if 0 < warmup_iters < n:
        axL.axvline(warmup_iters - 0.5, color="0.4", ls="--", lw=1)
        axL.text(warmup_iters - 0.5, axL.get_ylim()[1], " joint", fontsize=8,
                 va="top", color="0.4")
    axL.legend(lines, [ln.get_label() for ln in lines], fontsize=8, loc="upper right")
    axL.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def plot_v0_traj(v0_traj: list, gt_vel, path: str, warmup_iters: int = 0) -> None:
    """Standalone scalar-v0 parameter curve: x/y/z vs iter with GT dashed lines.

    The velocity-param fixture the image entries were dropping. warmup_iters>0
    draws the warmup|joint boundary (joint entry). v0_traj: list of [x,y,z]."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    arr = np.asarray(v0_traj)  # (iters, 3)
    fig, ax = plt.subplots(figsize=(6, 4))
    for k, (c, col) in enumerate(zip("xyz", ["tab:blue", "tab:orange", "tab:green"])):
        ax.plot(arr[:, k], color=col, lw=1.3, label=f"v0_{c}")
        ax.axhline(gt_vel[k], color=col, ls="--", lw=0.8, alpha=0.6)
    if warmup_iters:
        ax.axvline(warmup_iters - 0.5, color="k", ls=":", lw=1, label="warmup|joint")
    ax.set_xlabel("iter")
    ax.set_ylabel("v0 component")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)

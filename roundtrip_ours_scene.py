# coding=utf-8
"""Phase 2a: our telephone scenario inside GIC's harness (pure GIC roundtrip).

Loads the reuse_mpm telephone scene cache (normalized [0,1]^3 particles,
freeze_mask anchors), emulates our freeze BC with heavy anchor particles
(mass x anchor_mass_scale, v0 zeroed), gravity off, then runs the same
generate-with-GT -> refit-from-init roundtrip as roundtrip_sim2sim.py.

Tests whether OUR regime (anchored no-gravity oscillation, our E range, our
particle count/resolution) is recoverable under GIC's optimizer/BPTT/loss.

Usage (gic env, gic repo root):
  python roundtrip_ours_scene.py \
      --scene_cache /tmp2/b10401006/ev-project/generative-phys/outputs/forward_gen/06_tele_E1e5/scene_cache.pt \
      --gt_logE 5.0 --init_logE 6.0 --tag tele_gtE5_initE6
"""

# NOTE: no `from __future__ import annotations` here -- stringified annotations
# break taichi 1.2's @ti.kernel argument parsing (ti.types.ndarray()).
# importing roundtrip_sim2sim picks a free GPU before torch/taichi touch CUDA
from roundtrip_sim2sim import (
    plot_param_traj,
    rollout_collect_surfaces,
    save_rollout_gif,
    set_params,
)

import json
import os
import time
from argparse import ArgumentParser, Namespace

import taichi as ti
import torch

from simulator import Estimator
from train_dynamic import train
from utils.system_utils import draw_curve


class AnchoredEstimator(Estimator):
    """Estimator with our freeze-BC emulated via heavy, zero-velocity anchors."""

    _pvol = None  # optional per-particle volume override (cross-model alignment)
    _F0 = None    # optional initial deformation gradient field (F0 sys-id experiments)
    _v0_field = None  # optional V0VoxelField replacing the 3-DOF scalar init_vel
    _E_field = None   # optional EVoxelField replacing the scalar log10 E

    def set_E_field(self, field, query_xyz: torch.Tensor, lr: float) -> None:
        """field: EVoxelField; query_xyz: (N,3) rest positions. initialize() then
        builds per-particle mu/lam from field(xyz) instead of scalar self.E; the
        gradient returns via init_mu.backward(gradient=mu_grad). The phys-stage
        optimizer is swapped to the grid (group name 'Youngs modulus' so the
        existing get_optimizer/step plumbing works)."""
        object.__setattr__(self, "_E_field", field.to(query_xyz.device))
        self._E_field_xyz = query_xyz
        self.optimizer = torch.optim.Adam(
            [{"params": self._E_field.grid, "lr": lr, "name": "Youngs modulus"}])

    def set_v0_field(self, field, query_xyz: torch.Tensor, lr: float) -> None:
        """field: V0VoxelField; query_xyz: (N,3) f32 cuda rest positions.

        Swaps the vel-stage optimizer to the field's grid params (same group name
        'velocity' so _record_params / panel plumbing keep working). initialize()
        then builds init_velocities from the field -- the existing
        init_velocities.backward(gradient=...) bridge routes grads to the grid.
        """
        # Estimator IS an nn.Module: plain assignment would stash the field in
        # self._modules, and the class-level `_v0_field = None` then shadows every
        # read (class attrs win over nn.Module.__getattr__). Bypass via __dict__.
        object.__setattr__(self, "_v0_field", field.to(query_xyz.device))
        self._field_xyz = query_xyz
        self.vel_optimizer = torch.optim.Adam(
            [{"params": self._v0_field.grid, "lr": lr, "name": "velocity"}])

    def set_pvol(self, pvol: torch.Tensor) -> None:
        """pvol: (N,) f32 — replaces gic's uniform (dx/2)^3 particle volume."""
        self._pvol = pvol.cpu().numpy()

    @ti.kernel
    def _write_pvol(self, pv: ti.types.ndarray(), n: ti.i32):
        for p in range(n):
            self.simulator.p_vol[p] = pv[p]

    def set_F0(self, F0: torch.Tensor) -> None:
        """F0: (N,3,3) f32 — initial deformation gradient (overrides gic's F=I at t0).

        For F0 sys-id: the new-t0 snapshot carries a non-identity F (a known,
        fixed initial deformation). Injected AFTER from_torch (which resets F=I)
        in initialize(). gic stores F per (particle, substep); we write slot 0.
        """
        self._F0 = F0.cpu().numpy()

    @ti.kernel
    def _write_F0(self, F0: ti.types.ndarray(), n: ti.i32):
        for p in range(n):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    self.simulator.F[p, 0][i, j] = F0[p, i, j]

    @ti.kernel
    def _read_F0_grad(self, g: ti.types.ndarray(), n: ti.i32):
        """dL/dF0 at slot 0 after a full backward -- bridges gic's manual BPTT to a
        torch F0(alpha) graph so alpha (initial-deformation amplitude) is learnable."""
        for p in range(n):
            for i in ti.static(range(3)):
                for j in ti.static(range(3)):
                    g[p, i, j] = self.simulator.F.grad[p, 0][i, j]

    def set_anchor(self, anchor_mask: torch.Tensor, mass_scale: float) -> None:
        """anchor_mask: (N,) bool cuda; mass_scale: rho multiplier for anchors."""
        self._anchor_mask = anchor_mask
        self._free_f32 = (~anchor_mask).float().unsqueeze(1)        # (N, 1)
        self._rho_scale = torch.where(
            anchor_mask,
            torch.full_like(anchor_mask, mass_scale, dtype=torch.float32),
            torch.ones_like(anchor_mask, dtype=torch.float32),
        )                                                            # (N,)

    def initialize(self):
        super().initialize()
        cnt = self.init_vol.shape[0]
        # v0 applies to free particles only (keeps autograd path to init_vel correct)
        if self._v0_field is not None:
            vel_t = self._v0_field(self._field_xyz) * self._free_f32         # (N, 3)
        else:
            vel_t = self.init_vel.repeat(cnt).reshape(cnt, -1) * self._free_f32  # (N, 3)
        self.init_velocities = vel_t
        rho_vec = self.global_rho.repeat(cnt) * self._rho_scale             # (N,)
        self.init_rhos = rho_vec
        if self._E_field is not None:
            # per-particle mu/lam from the log10-E field (replaces super()'s
            # scalar-E versions). grad: init_mu -> 10**logE -> field grid.
            nu = self.get_nu()
            E_lin = 10.0 ** self._E_field(self._E_field_xyz)                 # (N,)
            self.init_mu = E_lin / (2.0 * (1.0 + nu))                        # (N,)
            self.init_lam = E_lin * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))     # (N,)
        self.from_torch(
            self.init_vol.data.cpu().numpy(),
            vel_t.data.cpu().numpy(),
            rho_vec.data.cpu().numpy(),
            self.init_mu.data.cpu().numpy(),
            self.init_lam.data.cpu().numpy(),
        )
        if self._pvol is not None:
            self._write_pvol(self._pvol, cnt)
        self.compute_particle_mass()
        if self._F0 is not None:  # inject known initial deformation (after F=I reset)
            self._write_F0(self._F0, cnt)


def save_overlay_gif(gt_list: list, pred_list: list, path: str, max_pts: int = 1500,
                     labels: tuple = ("GT", "pred"), fit_frames: "int | None" = None) -> None:
    """A (blue) vs B (red) rollout overlay: 3D + xy/xz/yz projections per frame.

    fit_frames: frames >= this index were NOT seen by the fit (held-out
    extrapolation); pred turns ORANGE there and the title flags it.
    """
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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
    import numpy as np
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


def _record_params(estimator: Estimator) -> dict:
    """Snapshot current params (same extraction as train_dynamic.train)."""
    d = {}
    for params in estimator.get_optimizer().param_groups:
        name = params["name"]
        p = params["params"][0].detach().cpu()
        if name == "Poisson ratio":
            p = estimator.get_nu().detach().cpu()
        elif name in ["Youngs modulus", "Yield stress", "plastic viscosity",
                      "shear modulus", "bulk modulus"]:
            p = 10 ** p
        d[name] = p if name == "velocity" else p.item()
    return d


def save_ckpt(estimator: Estimator, losses: list, estimated_params: list,
              it: int, path: str) -> None:
    """Checkpoint raw params + optimizer state so a fit can be RESUMED exactly.

    Raw (unconstrained) tensors: E is log10-space, nu is pre-tanh; optimizer
    state matters for Adam moments, hence state_dict.
    """
    field = getattr(estimator, "_v0_field", None)
    torch.save({
        "iter": it,
        "E_raw": estimator.E.data.detach().cpu(),
        "nu_raw": estimator.nu.data.detach().cpu(),
        "init_vel": estimator.init_vel.data.detach().cpu(),
        "v0_field_grid": None if field is None else field.grid.data.detach().cpu(),
        "optimizer": estimator.get_optimizer().state_dict(),
        "stage": int(estimator.stage[None]),
        "losses": [float(l) for l in losses],
        "estimated_params": [
            {k: (v if not torch.is_tensor(v) else v.tolist()) for k, v in d.items()}
            for d in estimated_params],
    }, path)


class CFLExhausted(RuntimeError):
    """CFL retry budget exhausted: current params are in an unsimulable region."""


def forward_bounded(estimator: Estimator, max_halvings: int = 3) -> None:
    """train_dynamic.forward with a CAP on dt-halving retries.

    gic's forward() retries `while True`, which deadlocks when params (e.g. an
    E overshoot, or NaN) make CFL unsatisfiable at any dt -- that burned 2.2h
    on carnation once. After `max_halvings` failures raise CFLExhausted so the
    caller can stop gracefully and keep the best-so-far params.
    """
    dt = estimator.simulator.dt_ori[None]
    for _ in range(max_halvings + 1):
        for idx in range(estimator.max_f):
            if idx == 0:
                estimator.initialize()
                estimator.simulator.set_dt(dt)
            estimator.forward(idx, img_backward=True)
        if estimator.succeed():
            return
        dt /= 2
        print(f"[forward_bounded] cfl dissatisfied, dt -> {dt}")
    raise CFLExhausted(f"CFL unsatisfied after {max_halvings} dt halvings")


def train_ours(estimator: Estimator, phys_args, max_f: int, out_dir: str,
               gts: list = None, patience: int = 8, min_iters: int = 15,
               rel_improve: float = 0.02, ckpt_every: int = 10,
               overlay_every: int = 0, fine_lr: float = 0.0,
               param_stop_tol: float = 0.0, tv_weight: float = 0.0) -> tuple:
    """train_dynamic.train + early stop + checkpoints + periodic overlay gifs.

    Early stop: quit when `patience` iters pass without the best loss improving
    by >`rel_improve` relative (never before `min_iters`). Best-so-far params
    are what the caller exports (same min-loss convention as gic's train), so
    stopping can only truncate post-convergence oscillation, not hurt the best.

    fine_lr > 0 (phys stage only) enables a two-phase schedule: the FIRST time
    the plateau condition fires, restore the best-so-far raw params, reset the
    Adam state, and swap every lr scheduler to a FLAT lr (E = fine_lr, others
    scaled by the same ratio vs their init), then keep going; the SECOND
    plateau stops. Rationale (measured, E1e4 8f): coarse lr 0.2 is needed to
    traverse a decade but oscillates +-5-20% around the valley; flat lr/4
    starves (never arrives, +21.5%). Coarse-traverse + fine-refine fixes both.

    param_stop_tol > 0 (vel stage): early stop ALSO requires the velocity params
    to have stopped moving (per-iter max |delta| < tol for `patience` iters).
    Guards against the vyz failure mode: a slowly-crawling component (z moved
    +0.013/iter) gets killed by a pure loss-plateau criterion while still
    converging (Q2 report, corrected lesson 7).
    Returns (losses, estimated_params) exactly like train_dynamic.train.
    """
    from train_dynamic import backward as gic_backward

    stage = int(estimator.stage[None])
    iter_cnt = (phys_args.vel_iter_cnt if stage == Estimator.velocity_stage
                else phys_args.iter_cnt)
    estimator.max_f = max_f
    losses, estimated_params = [], []
    best_loss, last_sig_improve = float("inf"), 0
    last_big_move = 0  # last iter where velocity params moved >= param_stop_tol
    best_raw = None  # (E_raw, nu_raw, vel_raw) at the best loss so far
    dropped = False
    for i in range(iter_cnt):
        estimated_params.append(_record_params(estimator))
        if param_stop_tol > 0 and len(estimated_params) >= 2:
            # movement across ALL optimized params (J1 lesson: watching only
            # velocity let a loss-plateau stop kill joint mid E-rebound).
            # E compared in log10 (0.005 dex ~ 1.2%); velocity/nu in raw units.
            import math as _math
            prev, cur = estimated_params[-2], estimated_params[-1]
            delta = 0.0
            for k, v in cur.items():
                if k not in prev:
                    continue
                if torch.is_tensor(v):
                    delta = max(delta, float((v - prev[k]).abs().max()))
                elif k == "Youngs modulus":
                    delta = max(delta, abs(_math.log10(max(v, 1e-12))
                                           - _math.log10(max(prev[k], 1e-12))))
                else:
                    delta = max(delta, abs(v - prev[k]))
            if delta >= param_stop_tol:
                last_big_move = i
        estimator.zero_grad()
        estimator.loss[None] = 0.0
        try:
            forward_bounded(estimator)
        except CFLExhausted as e:
            estimated_params.pop()
            print(f"[train_ours] STOP at iter {i}: {e}; keeping best-so-far "
                  f"(loss {best_loss:.6f})")
            break
        loss = estimator.loss[None] + estimator.image_loss
        if loss < best_loss:  # params recorded pre-step => stash BEFORE step()
            best_raw = (estimator.E.data.clone(), estimator.nu.data.clone(),
                        estimator.init_vel.data.clone())
        losses.append(loss)
        gic_backward(estimator)
        field = getattr(estimator, "_v0_field", None)
        if field is not None and tv_weight > 0:
            # TV grads ACCUMULATE onto the data grads before step (pure torch
            # path; the starve grad-mask hook applies to these too)
            (tv_weight * field.regularization()).backward()
        estimator.step(i)

        if loss < best_loss * (1.0 - rel_improve):
            last_sig_improve = i
        best_loss = min(best_loss, loss)
        min_idx = losses.index(min(losses))
        best_show = {k: (f"grid{tuple(v.shape)} mean {v.mean(dim=(0, 2, 3, 4)).tolist()}"
                         if torch.is_tensor(v) and v.numel() > 3 else v)
                     for k, v in estimated_params[min_idx].items()}
        print(f"[train_ours] iter {i} loss {loss:.6f} | best {losses[min_idx]:.6f} "
              f"@ {min_idx} | {best_show}")

        if ckpt_every and ((i + 1) % ckpt_every == 0 or i == iter_cnt - 1):
            save_ckpt(estimator, losses, estimated_params, i,
                      os.path.join(out_dir, f"ckpt_latest_stage{stage}.pt"))
        if overlay_every and gts is not None and (i + 1) % overlay_every == 0:
            estimator.max_f = len(gts)                  # full-frame rollout even in
            pred = rollout_collect_surfaces(estimator)  # the 4-frame vel stage
            estimator.set_stage(stage)                  # rollout forces phys stage
            estimator.max_f = max_f
            save_overlay_gif(gts, pred, os.path.join(out_dir, f"overlay_iter{i + 1:03d}.gif"),
                             fit_frames=max_f)

        params_settled = (param_stop_tol == 0 or (i - last_big_move) >= patience)
        if i + 1 >= min_iters and (i - last_sig_improve) >= patience and params_settled:
            can_drop = (fine_lr > 0 and not dropped
                        and stage == Estimator.physical_params_stage)
            if can_drop:
                dropped = True
                estimator.E.data.copy_(best_raw[0])
                estimator.nu.data.copy_(best_raw[1])
                opt = estimator.get_optimizer()
                for p in (estimator.E, estimator.nu):
                    opt.state.pop(p, None)  # stale coarse-phase Adam moments
                scale = fine_lr / phys_args.params["Youngs modulus"]["init_lr"]
                for name, info in phys_args.params.items():
                    if name in estimator.lr_schedulers:
                        flat = info["init_lr"] * scale
                        estimator.lr_schedulers[name] = (lambda lr: (lambda it: lr))(flat)
                        for pg in opt.param_groups:
                            if pg["name"] == name:
                                pg["lr"] = flat
                last_sig_improve = i
                print(f"[train_ours] plateau at iter {i}: restored best "
                      f"(loss {best_loss:.6f}) and dropped to flat fine lr "
                      f"(E lr {fine_lr}); refining until next plateau")
                continue
            print(f"[train_ours] early stop at iter {i}: no >{rel_improve:.0%} best-loss "
                  f"improvement for {patience} iters (best {best_loss:.6f} @ {min_idx})")
            break

    if ckpt_every:
        save_ckpt(estimator, losses, estimated_params, len(losses) - 1,
                  os.path.join(out_dir, f"ckpt_latest_stage{stage}.pt"))
    field = getattr(estimator, "_v0_field", None)
    restore_vel = losses and (stage == Estimator.velocity_stage or field is not None)
    if restore_vel and "velocity" in estimated_params[losses.index(min(losses))]:
        import torch.nn as nn
        best = estimated_params[losses.index(min(losses))]
        if field is not None:  # "velocity" record = (1,3,rz,ry,rx) grid in field mode
            field.grid.data.copy_(best["velocity"].to(estimator.device))
        else:
            estimator.init_vel = nn.Parameter(best["velocity"].to(estimator.device))
    return losses, estimated_params


def load_our_scene(cache_path: str) -> tuple:
    """Return (xyz (N,3) f32 cuda, anchor_mask (N,) bool cuda) with origin ghosts removed."""
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    disc = cache["disc"]
    xyz = disc["sim_xyzs"]            # (N0, 3) normalized [0,1]^3
    freeze = disc["freeze_mask"]      # (N0,) bool
    ghost = (xyz == 0).all(dim=1)     # kmeans origin ghosts (all frozen, inert)
    keep = ~ghost
    print(f"[ours] cache {cache_path}: {xyz.shape[0]} particles, "
          f"{int(ghost.sum())} origin ghosts removed, "
          f"{int((freeze & keep).sum())} real anchors kept")
    return xyz[keep].float().cuda(), freeze[keep].cuda()


if __name__ == "__main__":
    start_time = time.time()
    parser = ArgumentParser(description="GIC roundtrip on our telephone scenario")
    parser.add_argument("--config", default="config/ours/telephone.json", type=str)
    parser.add_argument("--scene_cache", required=True, type=str)
    parser.add_argument("--gt_logE", required=True, type=float)
    parser.add_argument("--gt_nu", default=0.3, type=float)
    parser.add_argument("--gt_vel", nargs=3, default=[0.0, -0.5, 0.0], type=float)
    parser.add_argument("--init_logE", required=True, type=float)
    parser.add_argument("--init_nu", default=0.1, type=float)
    parser.add_argument("--anchor_mass_scale", default=1e4, type=float)
    parser.add_argument("--gt_traj", default=None, type=str,
                        help="(T,N0,3) .npy from our warp pipeline; if set, use as GT "
                             "instead of generating with GIC's simulator (cross-sim fit). "
                             "gt_logE/gt_nu/gt_vel then only label the known GT values.")
    parser.add_argument("--gt_traj_normalized", action="store_true",
                        help="gt_traj is already in normalized sim space (skip world->sim denorm)")
    parser.add_argument("--inject_pvol", action="store_true",
                        help="use the cache's per-particle points_vol instead of gic's uniform (dx/2)^3")
    parser.add_argument("--f0_npy", default=None, type=str,
                        help="(N,3,3) initial deformation gradient field to inject at t0 "
                             "(F0 sys-id: fixed known F0). Order must match the ghost-filtered cache.")
    parser.add_argument("--init_xyz_npy", default=None, type=str,
                        help="(N,3) normalized init positions overriding the cache rest state "
                             "(the deformed snapshot = new t0; pairs with --f0_npy and --gt_traj).")
    parser.add_argument("--mpm_iter_cnt", default=None, type=int, help="override substeps/frame")
    parser.add_argument("--voxel_size", default=None, type=float, help="override config dx")
    parser.add_argument("--iter_cnt", default=None, type=int)
    parser.add_argument("--vel_iter_cnt", default=None, type=int)
    parser.add_argument("--n_frames", default=None, type=int,
                        help="truncate the fit to the first N frames of the GT")
    parser.add_argument("--vel_frames", default=None, type=int,
                        help="override vel_estimation_frames (vel-stage fit window; "
                             "GT stays full length, the rest is held-out extrapolation)")
    parser.add_argument("--phys_frames", default=None, type=int,
                        help="phys-stage fit window (default: full GT length)")
    parser.add_argument("--alt_rounds", default=1, type=int,
                        help=">1 = ALTERNATING joint: vel(v0) <-> phys(E) for N rounds")
    parser.add_argument("--alt_vel_iters", default=40, type=int)
    parser.add_argument("--alt_phys_iters", default=40, type=int)
    parser.add_argument("--joint_v0E", action="store_true",
                        help="TRUE JOINT: E + v0 in one optimizer, phys stage only "
                             "(nu present with lr 0 = fixed at init)")
    parser.add_argument("--E_lr", default=None, type=float,
                        help="override Youngs-modulus init_lr (log10-E units; final_lr scales proportionally)")
    parser.add_argument("--nu_lr", default=None, type=float,
                        help="override Poisson-ratio init_lr (final_lr scales proportionally)")
    parser.add_argument("--rot_z_deg", default=0.0, type=float,
                        help="rotate scene about z through (0.5,0.5); must match the warp GT dump's rot_z_deg")
    parser.add_argument("--patience", default=8, type=int,
                        help="early stop: iters without significant best-loss improvement")
    parser.add_argument("--min_iters", default=15, type=int)
    parser.add_argument("--overlay_every", default=10, type=int,
                        help="save overlay_iterNNN.gif every N iters (0=off)")
    parser.add_argument("--ckpt_every", default=10, type=int)
    parser.add_argument("--fine_lr", default=0.0, type=float,
                        help="two-phase: on first plateau restore best + flat fine E lr (0=off)")
    parser.add_argument("--fix_v0_gt", action="store_true",
                        help="skip velocity stage and fix v0 to gt_vel (isolate E recovery)")
    parser.add_argument("--fix_E_gt", action="store_true",
                        help="fix E/nu to GT and run ONLY the velocity stage (isolate v0 recovery)")
    parser.add_argument("--fix_E_logE", default=None, type=float,
                        help="like --fix_E_gt but fix the FIT's E at this (wrong) log10 value "
                             "while the GT uses gt_logE (warmup-viability test)")
    parser.add_argument("--v0_field_res", default="0", type=str,
                        help="fit v0 as a voxel FIELD: '4' (cubic) or 'RXxRYxRZ' e.g. "
                             "'4x4x16' (anisotropic, z-refined); '0' = scalar path")
    parser.add_argument("--gt_v0_variant", default=None, type=str,
                        help="generate GT with this analytic profile node-sampled into a "
                             "same-res grid (mid_kick|true_bend|ramp_y|twist_xy) instead "
                             "of the uniform gt_vel (self-consistent field GT)")
    parser.add_argument("--gt_v0_scale", default=1.0, type=float,
                        help="amplitude multiplier for --gt_v0_variant")
    parser.add_argument("--v0_field_init_std", default=0.05, type=float,
                        help="zero-mean gaussian init std of the field grid (velocity units)")
    parser.add_argument("--v0_field_lr", default=None, type=float,
                        help="field grid lr (default: config vel_lr)")
    parser.add_argument("--v0_field_seed", default=0, type=int)
    parser.add_argument("--ti_mem_frac", default=0.3, type=float,
                        help="taichi device_memory_fraction (0.3 fits busy shared cards)")
    parser.add_argument("--v0_field_tv", default=0.0, type=float,
                        help="TV smoothness weight on the field grid (mask-aware; 0=off)")
    parser.add_argument("--v0_field_starve_thresh", default=1.0, type=float,
                        help="fix-to-0 + de-train grid nodes with < this many particle-"
                             "equivalents of trilinear support (GT-free, geometry only; 0=off)")
    parser.add_argument("--param_stop_tol", default=0.0, type=float,
                        help="vel stage: early stop also requires per-iter max |delta v| < tol "
                             "(0=off); guards slow components against loss-plateau stop")
    parser.add_argument("--tag", required=True, type=str)
    parser.add_argument("--out_root", default="output/ours_telephone", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        phys_args = Namespace(**json.load(f)["physics"])
    phys_args.init_E = args.init_logE
    phys_args.init_nu = args.init_nu
    if args.voxel_size is not None:
        phys_args.voxel_size = args.voxel_size
    if args.mpm_iter_cnt is not None:
        phys_args.mpm_iter_cnt = args.mpm_iter_cnt
    if args.iter_cnt is not None:
        phys_args.iter_cnt = args.iter_cnt
    if args.vel_iter_cnt is not None:
        phys_args.vel_iter_cnt = args.vel_iter_cnt
    if args.vel_frames is not None:
        phys_args.vel_estimation_frames = args.vel_frames
    for cli_lr, pname in ((args.E_lr, "Youngs modulus"), (args.nu_lr, "Poisson ratio")):
        if cli_lr is not None:
            info = phys_args.params[pname]
            ratio = cli_lr / info["init_lr"]
            info["init_lr"] = cli_lr
            info["final_lr"] = info["final_lr"] * ratio
            print(f"[ours] {pname} lr override: {info['init_lr']} -> {info['final_lr']}")
    fit_init_vel = list(phys_args.init_vel)
    if args.fix_v0_gt:
        fit_init_vel = list(args.gt_vel)
        phys_args.vel_iter_cnt = 0
    if args.fix_E_gt and args.fix_E_logE is not None:
        raise SystemExit("--fix_E_gt and --fix_E_logE are mutually exclusive")
    if args.fix_E_gt or args.fix_E_logE is not None:
        # v0-only mode; E fixed at GT (fix_E_gt) or at a deliberately WRONG
        # value (fix_E_logE: the warmup-viability test -- can v0 still converge
        # when the vel stage runs under a mis-specified E?)
        fit_logE = args.gt_logE if args.fix_E_gt else args.fix_E_logE
        args.init_logE = fit_logE
        args.init_nu = args.gt_nu
        phys_args.init_E = fit_logE
        phys_args.init_nu = args.gt_nu
    n_frames = phys_args.n_frames

    out_dir = os.path.join(args.out_root, args.tag)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config_used.json"), "w") as f:
        json.dump({**vars(phys_args), "cli": vars(args)}, f, indent=2, default=str)

    xyz, anchor_mask = load_our_scene(args.scene_cache)
    if args.init_xyz_npy is not None:  # new t0 = deformed snapshot positions
        import numpy as np
        init_xyz = torch.from_numpy(np.load(args.init_xyz_npy)).float().cuda()
        assert init_xyz.shape == xyz.shape, (init_xyz.shape, xyz.shape)
        drift = (init_xyz - xyz).norm(dim=-1)
        print(f"[ours] init_xyz override: {init_xyz.shape[0]} pts, mean shift vs cache "
              f"{drift.mean():.4f} (deformed snapshot t0)")
        xyz = init_xyz
    if args.rot_z_deg:
        import math
        _t = math.radians(args.rot_z_deg)
        _c, _s = math.cos(_t), math.sin(_t)
        _x, _y = xyz[:, 0] - 0.5, xyz[:, 1] - 0.5
        xyz = xyz.clone()
        xyz[:, 0] = _c * _x - _s * _y + 0.5
        xyz[:, 1] = _s * _x + _c * _y + 0.5
        print(f"[ours] rotated scene z {args.rot_z_deg} deg (match warp dump)")
    print(f"[ours] N={xyz.shape[0]}, anchors={int(anchor_mask.sum())}, "
          f"dx={phys_args.voxel_size}, bbox {xyz.min(0).values.tolist()} .. {xyz.max(0).values.tolist()}")

    ext_traj = None
    if args.gt_traj is not None:
        import numpy as np

        cache = torch.load(args.scene_cache, map_location="cpu", weights_only=False)
        disc = cache["disc"]
        ghost = (disc["sim_xyzs"] == 0).all(dim=1)
        traj = torch.from_numpy(np.load(args.gt_traj))  # (T, N0, 3)
        if not args.gt_traj_normalized:
            # dataset_gen saves mpm_xyz denormalized; map back: norm = (world + shift) / scale
            # shift is 0-dim (legacy caches) or [3] (norm v2, per-axis centering)
            traj = (traj + disc["shift"].reshape(1, 1, -1)) / float(disc["scale"])
        traj = traj[:, ~ghost]  # (T, N, 3)
        assert traj.shape[1] == xyz.shape[0], (traj.shape, xyz.shape)
        drift = (traj[0] - xyz.cpu()).norm(dim=-1).max()
        print(f"[ours] external GT traj {tuple(traj.shape)}, frame0 vs cache max drift {drift:.2e}")
        assert drift < 1e-3, f"traj frame0 does not match cache after denorm: {drift}"
        traj[0] = xyz.cpu()  # snap frame0 to exact init (kills near-zero chamfer distances)
        ext_traj = [traj[t].float().cuda() for t in range(traj.shape[0])]
        if args.n_frames is not None:
            assert args.n_frames <= len(ext_traj), (args.n_frames, len(ext_traj))
            ext_traj = ext_traj[:args.n_frames]
            print(f"[ours] fit truncated to first {args.n_frames} frames")
        n_frames = phys_args.n_frames = len(ext_traj)

    # 0.3 is enough for the telephone scene (6.9k particles, chunk 100) and fits on
    # busy shared cards; 0.5 OOMs when neighbours hold >12GB (image_fit lesson).
    ti.init(arch=ti.cuda, debug=False, fast_math=False,
            device_memory_fraction=args.ti_mem_frac)

    dummy_gts = [xyz.clone() for _ in range(n_frames)]
    estimator = AnchoredEstimator(
        phys_args, "float32", dummy_gts, surface_index=None, init_vol=xyz,
        dynamic_scene=None, image_scale=1.0, pipeline=None, image_op=None,
    )
    estimator.set_anchor(anchor_mask, args.anchor_mass_scale)
    if args.inject_pvol:
        cache_pv = torch.load(args.scene_cache, map_location="cpu", weights_only=False)
        ghost_pv = (cache_pv["disc"]["sim_xyzs"] == 0).all(dim=1)
        pvol = torch.from_numpy(cache_pv["disc"]["points_vol"]).float()[~ghost_pv]
        estimator.set_pvol(pvol)
        print(f"[ours] injected per-particle p_vol (mean {pvol.mean():.3e})")
    if args.f0_npy is not None:
        import numpy as np
        F0 = torch.from_numpy(np.load(args.f0_npy)).float()
        assert F0.shape[0] == xyz.shape[0] and F0.shape[1:] == (3, 3), F0.shape
        estimator.set_F0(F0)
        dev = torch.linalg.svdvals(F0)
        print(f"[ours] injected F0 field: maxdev {(dev-1).abs().max():.3f} "
              f"(known fixed initial deformation)")

    # field res: None = scalar path; (rx,ry,rz) = voxel field (possibly anisotropic)
    field_res = None
    if args.v0_field_res != "0":
        field_res = (tuple(int(p) for p in args.v0_field_res.lower().split("x"))
                     if "x" in args.v0_field_res
                     else (int(args.v0_field_res),) * 3)
    pad = 2.0 * phys_args.voxel_size
    aabb = torch.stack([xyz.min(0).values - pad, xyz.max(0).values + pad])  # (2,3)
    free = ~anchor_mask
    z_lo, z_hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    flip_z = bool(xyz[anchor_mask][:, 2].mean() > xyz[free][:, 2].mean())

    # ---- 1. GT: external warp trajectory, or generate with GIC's simulator ----
    gt_part = None  # (N,3) per-particle GT v0 when GT is a field variant
    if ext_traj is not None:
        gts_synth = ext_traj
    else:
        set_params(estimator, args.gt_logE, args.gt_nu, args.gt_vel)
        if args.gt_v0_variant is not None:
            from v0_field_ours import V0VoxelField, fill_profile_grid
            assert field_res is not None, "--gt_v0_variant requires --v0_field_res"
            gt_field = V0VoxelField(aabb.cpu(), res=field_res)
            fill_profile_grid(gt_field, args.gt_v0_variant, args.gt_v0_scale,
                              z_lo, z_hi, flip=flip_z)
            estimator.set_v0_field(gt_field, xyz, lr=0.0)
            gt_part = (gt_field(xyz).detach() * free.float().unsqueeze(1)).cpu()  # (N,3)
            print(f"[ours] GT v0 = '{args.gt_v0_variant}' x{args.gt_v0_scale} on "
                  f"{field_res} grid (zt=0 at anchor end); free |v0| mean "
                  f"{gt_part[free.cpu()].norm(dim=-1).mean():.3f}, "
                  f"max {gt_part.norm(dim=-1).max():.3f}")
        gts_synth = rollout_collect_surfaces(estimator)
    disp = (gts_synth[-1] - gts_synth[0]).norm(dim=-1)          # (N,)
    anchor_disp = disp[anchor_mask]
    print(f"[ours] GT motion: free mean disp {disp[free].mean():.5f}, max {disp[free].max():.5f}; "
          f"anchor mean disp {anchor_disp.mean():.6f} (should be ~0)")
    save_rollout_gif(gts_synth, os.path.join(out_dir, "gt_rollout.gif"))

    # ---- 2. refit from init ----
    estimator.gts = gts_synth
    estimator.load_gts(gts_synth)
    set_params(estimator, args.init_logE, args.init_nu, fit_init_vel)

    v0_field = None
    if field_res is not None:
        from v0_field_ours import V0VoxelField
        # fit field set AFTER GT generation: the GT rollout above used the scalar
        # path (or the GT field); from here on initialize() reads THIS field.
        v0_field = V0VoxelField(aabb.cpu(), res=field_res)
        v0_field.randomize_(args.v0_field_init_std, seed=args.v0_field_seed)
        n_starved = 0
        if args.v0_field_starve_thresh > 0:
            n_starved = v0_field.freeze_starved_(
                xyz[~anchor_mask].cpu(), thresh=args.v0_field_starve_thresh)
        field_lr = args.v0_field_lr if args.v0_field_lr is not None else phys_args.vel_lr
        estimator.set_v0_field(v0_field, xyz, lr=field_lr)
        n_nodes = field_res[0] * field_res[1] * field_res[2]
        print(f"[ours] v0 FIELD mode: res={field_res} ({3 * n_nodes} DOF), "
              f"init std {args.v0_field_init_std} (seed {args.v0_field_seed}), "
              f"lr {field_lr}, starved frozen {n_starved}/{n_nodes} "
              f"(thresh {args.v0_field_starve_thresh})")

    if args.joint_v0E or args.alt_rounds > 1:
        assert not (args.fix_E_gt or args.fix_E_logE is not None or args.fix_v0_gt), \
            "joint/alternating modes are exclusive with the fix_* isolation modes"
    if args.alt_rounds > 1 or phys_args.vel_iter_cnt == 0:
        # alt runs its own vel phases; vel_iter_cnt 0 = cold start (ablation)
        losses_vel, e_s_vel = [], []
    else:
        # SHARED velocity warmup: the same vel stage serves the two-stage path
        # AND joint mode (J1 showed cold-start joint lets E wander decades
        # before v0 grows in; warmup is the cure)
        estimator.set_stage(Estimator.velocity_stage)
        losses_vel, e_s_vel = train_ours(
            estimator, phys_args, phys_args.vel_estimation_frames, out_dir, gts=gts_synth,
            patience=args.patience, min_iters=args.min_iters,
            ckpt_every=args.ckpt_every, overlay_every=args.overlay_every,
            param_stop_tol=args.param_stop_tol, tv_weight=args.v0_field_tv)

    if args.fix_E_gt or args.fix_E_logE is not None:
        # v0-only mode: export vel-stage results and stop (no phys stage)
        import numpy as np

        field_err_traj, v_best = None, None
        if v0_field is not None:
            from v0_field_ours import eval_grid_at
            aabb_c, xyz_c, fm = v0_field.aabb.cpu(), xyz.cpu(), free.cpu()
            # per-particle GT at free particles: field variant or uniform gt_vel
            if gt_part is not None:
                gt_pf = gt_part[fm].float()                              # (M,3)
            else:
                gt_pf = torch.tensor(args.gt_vel, dtype=torch.float32
                                     ).expand(int(fm.sum()), 3)          # (M,3)
            gt_scale = max(float(gt_pf.norm(dim=-1).mean()), 1e-12)      # mean |gt|
            gt_mean_vec = gt_pf.mean(0)                                  # (3,)
            # per-iter recorded grids -> free-particle v0 (M,3) -> traj of stats
            per_iter_v = [eval_grid_at(d["velocity"].float(), aabb_c, xyz_c)[fm]
                          for d in e_s_vel]
            v0_traj = [v.mean(0).tolist() for v in per_iter_v]  # mean free-particle vec
            # per-axis spatial spread across particles (for uniform GT the bands
            # should collapse onto the mean; non-uniform GT has intrinsic spread)
            v0_traj_p05 = [v.quantile(0.05, dim=0).tolist() for v in per_iter_v]
            v0_traj_p95 = [v.quantile(0.95, dim=0).tolist() for v in per_iter_v]
            field_err_traj = [float((v - gt_pf).norm(dim=-1).mean() / gt_scale)
                              for v in per_iter_v]
            # xy-only = the OBSERVABLE-subspace metric (z is the known weak axis);
            # the all-axes number is kept for cross-comparison but is NOT the headline
            field_err_xy_traj = [float((v - gt_pf)[:, :2].norm(dim=-1).mean() / gt_scale)
                                 for v in per_iter_v]
            v_best = per_iter_v[losses_vel.index(min(losses_vel))]  # (M,3)
            ax_err_best = (v_best - gt_pf).abs().mean(0)            # (3,)
            v0_est = v_best.mean(0).tolist()
        else:
            v0_traj = [[float(x) for x in d["velocity"]] for d in e_s_vel]  # (iters, 3)
            v0_est = estimator.init_vel.detach().cpu().numpy().tolist()
        # pred rollout under the FIT's E (== gt_logE unless --fix_E_logE)
        set_params(estimator, args.init_logE, args.gt_nu, v0_est)
        estimator.max_f = n_frames  # vel-stage train() left it at vel_estimation_frames
        pred_roll = rollout_collect_surfaces(estimator)
        save_rollout_gif(pred_roll, os.path.join(out_dir, "pred_rollout.gif"))
        save_overlay_gif(gts_synth, pred_roll, os.path.join(out_dir, "overlay.gif"),
                         fit_frames=phys_args.vel_estimation_frames)
        plot_axis_profiles(gts_synth, pred_roll, free, os.path.join(out_dir, "axis_profile.png"),
                           fit_frames=phys_args.vel_estimation_frames)
        if v0_field is not None and gt_part is not None:
            gt_v = gt_mean_vec.numpy()      # variant GT: compare means against mean
        else:
            gt_v = np.array(args.gt_vel)
        err = np.linalg.norm(np.array(v0_est) - gt_v) / max(np.linalg.norm(gt_v), 1e-12)
        result = {
            "scenario": "ours_fixE_learn_v0",
            "fit_logE": args.init_logE,  # != gt logE in --fix_E_logE (wrong-E) mode
            "gt": {"E": 10.0 ** args.gt_logE, "nu": args.gt_nu, "vel": args.gt_vel,
                   "v0_variant": args.gt_v0_variant, "v0_scale": args.gt_v0_scale,
                   "v0_mean_vec": (gt_mean_vec.tolist()
                                   if v0_field is not None else args.gt_vel)},
            "v0_estimated": v0_est,
            "v0_rel_err": float(err),
            "losses_vel": [float(l) for l in losses_vel],
            "v0_traj": v0_traj,
            "gt_motion_free_mean_disp": float(disp[free].mean()),
            "wall_time_s": time.time() - start_time,
        }
        if v0_field is not None:
            result["v0_traj_p05"] = v0_traj_p05
            result["v0_traj_p95"] = v0_traj_p95
            best_idx = losses_vel.index(min(losses_vel))
            per_err = (v_best - gt_pf).norm(dim=-1)               # (M,)
            gt_norm = gt_scale
            result["v0_field"] = {
                "res": list(field_res),
                "init_std": args.v0_field_init_std,
                "seed": args.v0_field_seed,
                "lr": args.v0_field_lr,
                "starve_thresh": args.v0_field_starve_thresh,
                "n_starved_frozen": n_starved,
                "tv_weight": args.v0_field_tv,
                # HEADLINE metrics (observable subspace / per-axis):
                "per_axis_err_best": ax_err_best.tolist(),     # mean |err| per axis
                "rel_l2_xy_best": field_err_xy_traj[best_idx],
                "rel_l2_xy_traj": field_err_xy_traj,
                # cross-comparison only (inflated by unobservable axes):
                "rel_l2_best": field_err_traj[best_idx],   # mean per-particle err / |gt|
                "rel_l2_traj": field_err_traj,
                "mean_vec_best": v0_est,                    # mean over free particles
                "per_axis_std_best": v_best.std(0).tolist(),
                "per_particle_err_max": float(per_err.max()) / gt_norm,
                "per_particle_err_p95": float(per_err.quantile(0.95)) / gt_norm,
            }
        with open(os.path.join(out_dir, "result.json"), "w") as f:
            json.dump(result, f, indent=2)
        draw_curve(losses_vel, out_dir, name="loss_vel")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(6, 4))
        arr = np.array(v0_traj)  # (iters, 3)
        for k, c in enumerate("xyz"):
            ax.plot(arr[:, k], color=f"C{k}", label=f"v0_{c}")
            ax.axhline(gt_v[k], color=f"C{k}", ls="--", alpha=0.5)
        if v0_field is not None:
            lo_a, hi_a = np.array(v0_traj_p05), np.array(v0_traj_p95)  # (iters, 3)
            for k in range(3):
                ax.fill_between(range(len(arr)), lo_a[:, k], hi_a[:, k],
                                color=f"C{k}", alpha=0.15)
            title = "free-particle v0: mean line, p5-p95 band (AGGREGATE view)"
            if gt_part is not None:
                # non-uniform GT has INTRINSIC spread: recovered band should
                # MATCH the GT band (dotted), not collapse onto the mean
                gq05 = gt_pf.quantile(0.05, dim=0)
                gq95 = gt_pf.quantile(0.95, dim=0)
                for k in range(3):
                    ax.axhline(float(gq05[k]), color=f"C{k}", ls=":", alpha=0.6)
                    ax.axhline(float(gq95[k]), color=f"C{k}", ls=":", alpha=0.6)
                title += " — dotted = GT p5/p95"
            ax.set_title(title, fontsize=8)
        ax.set_xlabel("vel-stage iter")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "v0_traj.png"))
        if v0_field is not None:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(field_err_traj, color="0.7", lw=1.0,
                    label="all axes (incl. UNOBSERVABLE — cross-compare only)")
            ax.plot(field_err_xy_traj, color="tab:red", label="xy-only (observable)")
            ax.set_xlabel("vel-stage iter")
            ax.set_ylabel("field rel L2")
            ax.set_yscale("log")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "field_err.png"))
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.hist((per_err / gt_norm).numpy(), bins=60)
            ax.set_xlabel("per-particle |v0 - gt| / mean|gt| (best iter, free particles)")
            ax.set_ylabel("count")
            fig.tight_layout()
            fig.savefig(os.path.join(out_dir, "field_err_hist.png"))
            plot_field_projections(xyz_c[fm], v_best, gt_pf,
                                   os.path.join(out_dir, "field_proj.png"),
                                   aabb=aabb_c, res=field_res,
                                   xyz_anchor=xyz_c[~fm])
            best_grid = e_s_vel[best_idx]["velocity"].float()  # (1,3,rz,ry,rx)
            gt_for_nodes = (gt_field.grid.data.detach().cpu() if gt_part is not None
                            else torch.tensor(args.gt_vel, dtype=torch.float32))
            plot_grid_nodes(best_grid, aabb_c, gt_for_nodes, xyz_c[fm],
                            os.path.join(out_dir, "grid_quiver.png"),
                            os.path.join(out_dir, "grid_hist.png"))
            if gt_part is not None:  # 1D profile: the most direct view for these GTs
                zt_f = ((xyz_c[fm][:, 2] - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0, 1)
                if flip_z:
                    zt_f = 1.0 - zt_f
                plot_profile_1d(zt_f, v_best, gt_pf,
                                os.path.join(out_dir, "profile_1d.png"))
            print(f"[ours] FIELD per-axis |err| x {ax_err_best[0]:.3f} "
                  f"y {ax_err_best[1]:.3f} z {ax_err_best[2]:.3f} | "
                  f"xy relL2 {field_err_xy_traj[best_idx]:.2%} "
                  f"(all-axes {field_err_traj[best_idx]:.2%} — inflated by "
                  f"unobservable axes; judge by visuals)")
        print(f"[ours] DONE(v0-only) tag={args.tag} GT v0={args.gt_vel} -> {v0_est} "
              f"(rel err {err:.2%}), wall {time.time() - start_time:.0f}s")
        raise SystemExit(0)

    phys_fit_frames = args.phys_frames if args.phys_frames is not None else n_frames
    if args.joint_v0E:
        # E + v0 step TOGETHER on the phys-stage loss (AFTER the shared velocity
        # warmup above, unless --vel_iter_cnt 0); nu rides along with lr 0 so all
        # downstream record/plot plumbing (which expects the group) works.
        # NOTE: built after the vel stage's restore-best, which REBINDS init_vel.
        e_lr = phys_args.params["Youngs modulus"]["init_lr"]
        # v0 side: the FIELD grid (warmed up in the vel stage) if field mode, else
        # the scalar init_vel. initialize() already reads _v0_field for the
        # per-particle v0, so pointing the optimizer at the grid is all that's left.
        vel_param = v0_field.grid if v0_field is not None else estimator.init_vel
        estimator.optimizer = torch.optim.Adam([
            {"params": estimator.E, "lr": e_lr, "name": "Youngs modulus"},
            {"params": estimator.nu, "lr": 0.0, "name": "Poisson ratio"},
            {"params": vel_param, "lr": phys_args.vel_lr, "name": "velocity"},
        ])
        # update_learning_rate() overwrites group lr from lr_schedulers by NAME
        # every step -- without this pop the nu group's lr-0 freeze gets undone
        estimator.lr_schedulers.pop("Poisson ratio", None)
        print(f"[ours] JOINT {'v0field' if v0_field is not None else 'v0'}+E: "
              f"E lr {e_lr} (scheduled), v0 lr {phys_args.vel_lr} (flat), nu FROZEN, "
              f"window {phys_fit_frames}f")
        estimator.set_stage(Estimator.physical_params_stage)
        losses_phys, e_s_phys = train_ours(
            estimator, phys_args, phys_fit_frames, out_dir, gts=gts_synth,
            patience=args.patience, min_iters=args.min_iters,
            ckpt_every=args.ckpt_every, overlay_every=args.overlay_every,
            fine_lr=args.fine_lr, param_stop_tol=args.param_stop_tol,
            tv_weight=args.v0_field_tv)
    elif args.alt_rounds > 1:
        # block-coordinate: vel(v0 @ vel_frames) <-> phys(E @ phys_frames), N rounds
        losses_phys, e_s_phys = [], []
        alt_vel_bounds, alt_phys_bounds = [], []  # cumulative iters at round ends
        phys_args.vel_iter_cnt = args.alt_vel_iters
        phys_args.iter_cnt = args.alt_phys_iters
        for rd in range(args.alt_rounds):
            estimator.set_stage(Estimator.velocity_stage)
            lv, ev = train_ours(
                estimator, phys_args, phys_args.vel_estimation_frames, out_dir,
                gts=gts_synth, patience=args.patience,
                min_iters=min(args.min_iters, args.alt_vel_iters),
                ckpt_every=args.ckpt_every, overlay_every=0,
                param_stop_tol=args.param_stop_tol, tv_weight=args.v0_field_tv)
            losses_vel += lv
            e_s_vel += ev
            estimator.set_stage(Estimator.physical_params_stage)
            lp, ep = train_ours(
                estimator, phys_args, phys_fit_frames, out_dir, gts=gts_synth,
                patience=args.patience,
                min_iters=min(args.min_iters, args.alt_phys_iters),
                ckpt_every=args.ckpt_every, overlay_every=0, fine_lr=0.0)
            losses_phys += lp
            e_s_phys += ep
            alt_vel_bounds.append(len(losses_vel))
            alt_phys_bounds.append(len(losses_phys))
            cur = _record_params(estimator)
            print(f"[ours] ALT round {rd + 1}/{args.alt_rounds}: "
                  f"E {cur.get('Youngs modulus'):.4g}, "
                  f"v0 {[round(v, 3) for v in estimator.init_vel.detach().cpu().tolist()]}")
            # end-of-round state, rolled out to FULL GT length (incl. extrapolation)
            estimator.max_f = n_frames
            roll = rollout_collect_surfaces(estimator)
            save_overlay_gif(gts_synth, roll,
                             os.path.join(out_dir, f"overlay_round{rd + 1}.gif"),
                             fit_frames=phys_fit_frames)
    else:
        estimator.set_stage(Estimator.physical_params_stage)
        losses_phys, e_s_phys = train_ours(
            estimator, phys_args, phys_fit_frames, out_dir, gts=gts_synth,
            patience=args.patience, min_iters=args.min_iters,
            ckpt_every=args.ckpt_every, overlay_every=args.overlay_every,
            fine_lr=args.fine_lr)

    # ---- 3. export ----
    min_idx = losses_phys.index(min(losses_phys))
    best = e_s_phys[min_idx]
    gt_E = 10.0 ** args.gt_logE

    # rollout under the best estimate for visual comparison vs GT
    import math
    best_v0 = estimator.init_vel.detach().cpu().numpy().tolist()
    set_params(estimator, math.log10(best["Youngs modulus"]), best["Poisson ratio"], best_v0)
    estimator.max_f = n_frames
    pred_roll = rollout_collect_surfaces(estimator)
    save_rollout_gif(pred_roll, os.path.join(out_dir, "pred_rollout.gif"))
    save_overlay_gif(gts_synth, pred_roll, os.path.join(out_dir, "overlay.gif"),
                     fit_frames=phys_fit_frames)
    plot_axis_profiles(gts_synth, pred_roll, free, os.path.join(out_dir, "axis_profile.png"),
                       fit_frames=phys_fit_frames)
    rel_err_E = abs(best["Youngs modulus"] - gt_E) / gt_E
    if v0_field is not None:
        from v0_field_ours import eval_grid_at
    result = {
        "scenario": "ours_telephone_anchored_nogravity",
        "gt": {"E": gt_E, "logE": args.gt_logE, "nu": args.gt_nu, "vel": args.gt_vel},
        "init": {"logE": args.init_logE, "nu": args.init_nu, "vel": fit_init_vel},
        "anchor_mass_scale": args.anchor_mass_scale,
        "voxel_size": phys_args.voxel_size,
        "best": {**{k: (v.tolist() if torch.is_tensor(v) else v) for k, v in best.items()},
                 "iter": min_idx, "loss": float(losses_phys[min_idx])},
        "final": {k: (v.tolist() if torch.is_tensor(v) else v)
                  for k, v in e_s_phys[-1].items()},
        "vel_estimated": estimator.init_vel.detach().cpu().numpy().tolist(),
        "rel_err_E": rel_err_E,
        "losses_vel": [float(l) for l in losses_vel],
        "losses_phys": [float(l) for l in losses_phys],
        "E_traj": [d.get("Youngs modulus") for d in e_s_phys],
        "nu_traj": [d.get("Poisson ratio") for d in e_s_phys],
        # v0 trajectory: joint mode records it in the phys loop, alt in vel loops.
        # In v0-FIELD mode "velocity" is a (1,3,rz,ry,rx) grid -> log free-particle
        # MEAN per axis (full field is in the ckpt's v0_field_grid for offline viz).
        "v0_traj": ([(eval_grid_at(d["velocity"].float(), aabb.cpu(), xyz.cpu())[free.cpu()]
                       .mean(0).tolist() if torch.is_tensor(d["velocity"]) and d["velocity"].dim() == 5
                      else [float(x) for x in d["velocity"]])
                     for d in (e_s_phys if args.joint_v0E else e_s_vel)
                     if "velocity" in d] or None),
        "alt_rounds": args.alt_rounds if args.alt_rounds > 1 else None,
        "alt_vel_bounds": alt_vel_bounds if args.alt_rounds > 1 else None,
        "alt_phys_bounds": alt_phys_bounds if args.alt_rounds > 1 else None,
        "gt_motion_free_mean_disp": float(disp[free].mean()),
        "gt_motion_free_max_disp": float(disp[free].max()),
        "gt_motion_anchor_mean_disp": float(anchor_disp.mean()),
        "wall_time_s": time.time() - start_time,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as f:
        json.dump(result, f, indent=2)
    draw_curve(losses_phys, out_dir, name="loss_phys")
    if losses_vel:
        draw_curve(losses_vel, out_dir, name="loss_vel")
    plot_param_traj(e_s_phys, gt_E, args.gt_nu, os.path.join(out_dir, "param_traj.png"))
    print(f"[ours] DONE tag={args.tag} GT E={gt_E:.3g} -> best E={best['Youngs modulus']:.3g} "
          f"(rel err {rel_err_E:.2%}), wall {time.time() - start_time:.0f}s")

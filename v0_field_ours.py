"""Learnable initial-velocity FIELD v0(pos) for the gic roundtrip harness.

Voxel-grid port of reuse_mpm/v0field.py (kind="voxel" only): a (res^3, 3) grid of
velocity vectors, trilinearly interpolated at particle rest positions. Replaces the
3-DOF scalar `init_vel` in the vel stage; the gradient reaches the grid through the
SAME bridge gic already uses (train_dynamic.backward calls
`estimator.init_velocities.backward(gradient=velocity_grad)`, so any torch graph
from grid params -> init_velocities gets its grads for free).

First milestone (field-v0 step 1): GT is a UNIFORM field, random init, E fixed at
GT, vel stage only -- "does field parameterization break a known-solvable problem?".
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class V0VoxelField(nn.Module):
    """pos (n,3) normalized sim coords -> v0 (n,3), trilinear voxel grid.

    Args:
        aabb:    (2,3) [min; max] rows bounding the queried particles; maps query
                 positions into [-1,1] for grid_sample.
        res:     grid resolution per axis (DOF = 3 * res^3).
        v_clamp: hard clamp on per-component |v0| (CFL guard); None = off.
    """

    def __init__(self, aabb: Tensor, res=4, v_clamp: "float | None" = 20.0) -> None:
        super().__init__()
        # res: int (cubic) or (rx,ry,rz) -- ANISOTROPIC grids let a thin/long
        # object refine along its long axis without splitting x/y support
        # (measured: 4x4x16 median node support 123 vs iso-16's 16).
        self.res_xyz = (res, res, res) if isinstance(res, int) else tuple(res)
        rx, ry, rz = self.res_xyz
        self.register_buffer("aabb", aabb.detach().clone())  # (2,3)
        self.v_clamp = None if v_clamp is None else float(v_clamp)
        # (1,3,rz,ry,rx) grid of (vx,vy,vz) -- grid_sample layout (D,H,W)=(z,y,x);
        # zero init = start from rest.
        self.grid = nn.Parameter(torch.zeros(1, 3, rz, ry, rx))

    def randomize_(self, std: float, seed: int = 0) -> None:
        """In-place zero-mean gaussian init of the grid (the random-init arm)."""
        g = torch.Generator(device="cpu").manual_seed(seed)
        noise = torch.randn(self.grid.shape, generator=g)  # (1,3,R,R,R)
        with torch.no_grad():
            self.grid.copy_(noise * std)

    def freeze_starved_(self, xyz_free: Tensor, thresh: float = 1.0) -> int:
        """Fix gradient-starved nodes to 0 and make them non-trainable.

        A node's SUPPORT = total trilinear weight it receives from the free
        particles (xyz_free (M,3)) -- pure geometry, no GT needed. Nodes with
        support < thresh (particle-equivalents) get ~zero gradient and would
        just keep their init noise forever; instead we zero their value and
        mask their gradient (a hook), so they are exactly 0 and non-trainable.
        Returns the number of frozen nodes (out of res^3).
        """
        rx, ry, rz = self.res_xyz
        probe = torch.zeros(1, 1, rz, ry, rx, requires_grad=True)
        p = 2.0 * (xyz_free - self.aabb[0]) / (self.aabb[1] - self.aabb[0] + 1e-8) - 1.0
        F.grid_sample(probe, p.clamp(-1.0, 1.0).view(1, -1, 1, 1, 3),
                      mode="bilinear", align_corners=True).sum().backward()
        support = probe.grad[0, 0]                                  # (rz,ry,rx)
        mask = (support >= thresh).float().view(1, 1, rz, ry, rx)   # bcast over xyz
        self.register_buffer("support", support)
        self.register_buffer("grad_mask", mask)
        with torch.no_grad():
            self.grid.mul_(self.grad_mask)
        self.grid.register_hook(lambda g: g * self.grad_mask)       # buffer follows .to()
        return int((mask == 0).sum())

    def regularization(self) -> Tensor:
        """Total-variation smoothness on the grid. Scalar.

        Mask-aware: if freeze_starved_ was applied, only edges whose BOTH
        endpoints are trainable count -- otherwise TV would drag supported
        boundary nodes toward their frozen-at-0 starved neighbours. Each
        spatial axis contributes mean |diff| over its valid edges.
        """
        g = self.grid[0]                                    # (3,rz,ry,rx)
        m = getattr(self, "grad_mask", None)
        tv = g.new_zeros(())
        for d in (1, 2, 3):                                 # spatial dims of g
            n = g.shape[d] - 1
            if n < 1:
                continue
            dg = (g.narrow(d, 1, n) - g.narrow(d, 0, n)).abs()   # (3,...) edge diffs
            if m is not None:
                mm = m[0, 0]                                # (rz,ry,rx)
                w = mm.narrow(d - 1, 1, n) * mm.narrow(d - 1, 0, n)  # valid edges
                tv = tv + (dg * w.unsqueeze(0)).sum() / (3.0 * w.sum().clamp(min=1.0))
            else:
                tv = tv + dg.mean()
        return tv

    def _normalise(self, pos: Tensor) -> Tensor:
        """(n,3) sim coords -> (n,3) in [-1,1] (grid_sample convention)."""
        lo, hi = self.aabb[0], self.aabb[1]  # (3,), (3,)
        p = 2.0 * (pos - lo) / (hi - lo + 1e-8) - 1.0
        return p.clamp(-1.0, 1.0)

    def forward(self, pos: Tensor) -> Tensor:
        """pos (n,3) -> v0 (n,3), clamped."""
        n = pos.shape[0]
        p = self._normalise(pos)                       # (n,3) in [-1,1]
        gidx = p.view(1, n, 1, 1, 3)                   # (1,n,1,1,3) coords=(x,y,z)
        out = F.grid_sample(self.grid, gidx, mode="bilinear",
                            align_corners=True)        # (1,3,n,1,1)
        v = out.reshape(3, n).t()                      # (n,3)
        if self.v_clamp is not None:
            v = v.clamp(-self.v_clamp, self.v_clamp)
        return v


def variant_field(name: str, zt: Tensor) -> Tensor:
    """Analytic v0 profile: zt (N,) in [0,1] (0 = anchor end) -> v0 (N,3).

    The selected non-uniform GT designs (2026-06-12 preview rounds; centers
    nudged 0.025 toward the free tip per user review).
    """
    import math

    v = torch.zeros(zt.shape[0], 3, device=zt.device)
    if name == "uniform_y":
        v[:, 1] = -0.5
    elif name == "ramp_y":
        v[:, 1] = -0.5 * zt
    elif name == "ramp_x":
        v[:, 0] = -0.5 * zt
    elif name == "mid_kick":
        v[:, 1] = -0.5 * torch.exp(-(((zt - 0.525) / 0.12) ** 2))
    elif name == "true_bend":
        v[:, 1] = (0.5 * torch.exp(-(((zt - 0.475) / 0.15) ** 2))
                   - 0.5 * torch.exp(-(((zt - 1.0) / 0.18) ** 2)))
    elif name == "twist_xy":
        th = math.pi * zt
        v[:, 0] = 0.5 * torch.cos(th)
        v[:, 1] = 0.5 * torch.sin(th)
    else:
        raise ValueError(name)
    return v


def fill_profile_grid(field: V0VoxelField, name: str, scale: float,
                      z_lo: float, z_hi: float, flip: bool) -> None:
    """Write the node-SAMPLED analytic profile into the field grid (in place).

    z_lo/z_hi: particle z range defining zt; flip: True if anchors sit at high
    z. The GT rollout then uses this grid -- the fit (same class, same res) has
    ZERO representation floor by construction.
    """
    rx, ry, rz = field.res_xyz
    nodes_z = torch.linspace(float(field.aabb[0][2]), float(field.aabb[1][2]), rz)
    zt = ((nodes_z - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0.0, 1.0)
    if flip:
        zt = 1.0 - zt
    v = variant_field(name, zt) * scale                         # (rz,3)
    with torch.no_grad():
        field.grid.copy_(v.t().contiguous().view(1, 3, rz, 1, 1).expand(1, 3, rz, ry, rx))


def eval_grid_at(grid: Tensor, aabb: Tensor, pos: Tensor) -> Tensor:
    """Stateless trilinear eval: grid (1,3,R,R,R), aabb (2,3), pos (n,3) -> (n,3).

    Used to post-process per-iter recorded grids into per-particle v0 stats
    without rebuilding a module.
    """
    n = pos.shape[0]
    p = 2.0 * (pos - aabb[0]) / (aabb[1] - aabb[0] + 1e-8) - 1.0
    p = p.clamp(-1.0, 1.0).view(1, n, 1, 1, 3)         # (1,n,1,1,3)
    out = F.grid_sample(grid, p, mode="bilinear", align_corners=True)  # (1,3,n,1,1)
    return out.reshape(3, n).t()                       # (n,3)


class EVoxelField(nn.Module):
    """log10(E) voxel field, trilinear -> per-particle log10 E (clamped).

    The E analog of V0VoxelField. Stored in LOG10 space (matches gic's scalar
    self.E and keeps the optimization multiplicative). The estimator turns the
    per-particle log10 E into mu/lam in initialize() (AnchoredEstimator.set_E_field
    injection); the gradient returns via init_mu.backward(gradient=mu_grad), the
    same bridge the scalar E uses.

    Observability caveat (unlike v0): E only has gradient where the motion
    produces STRAIN -- low-strain regions are dead regardless of geometric
    support. Geometric starve-freeze still helps (no particles near a node ->
    no gradient) but does NOT cover the strain dead-zone; report by strain.
    """

    def __init__(self, aabb: Tensor, res=4,
                 logE_clamp: "tuple" = (4.0, 6.15)) -> None:
        super().__init__()
        self.res_xyz = (res, res, res) if isinstance(res, int) else tuple(res)
        rx, ry, rz = self.res_xyz
        self.register_buffer("aabb", aabb.detach().clone())  # (2,3)
        self.logE_clamp = tuple(logE_clamp)
        # (1,1,rz,ry,rx) log10 E grid; init filled by set_uniform_/fill_ramp_
        self.grid = nn.Parameter(torch.zeros(1, 1, rz, ry, rx))

    def set_uniform_(self, logE: float) -> None:
        with torch.no_grad():
            self.grid.fill_(float(logE))

    def fill_ramp_(self, lo_logE: float, hi_logE: float,
                   z_lo: float, z_hi: float, flip: bool) -> None:
        """Linear log10 E along z: lo at the anchor end (zt=0), hi at the tip."""
        rx, ry, rz = self.res_xyz
        nodes_z = torch.linspace(float(self.aabb[0][2]), float(self.aabb[1][2]), rz)
        zt = ((nodes_z - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0.0, 1.0)
        if flip:
            zt = 1.0 - zt
        vals = lo_logE + (hi_logE - lo_logE) * zt                  # (rz,)
        with torch.no_grad():
            self.grid.copy_(vals.view(1, 1, rz, 1, 1).expand(1, 1, rz, ry, rx))

    def freeze_starved_(self, xyz_free: Tensor, thresh: float = 1.0) -> int:
        """Mask grad of geometrically-starved nodes (value stays at init, NOT 0:
        a 0-log10-E node = E=1 would wreck CFL). Returns count frozen."""
        rx, ry, rz = self.res_xyz
        probe = torch.zeros(1, 1, rz, ry, rx, requires_grad=True)
        p = 2.0 * (xyz_free - self.aabb[0]) / (self.aabb[1] - self.aabb[0] + 1e-8) - 1.0
        F.grid_sample(probe, p.clamp(-1.0, 1.0).view(1, -1, 1, 1, 3),
                      mode="bilinear", align_corners=True).sum().backward()
        support = probe.grad[0, 0]                                  # (rz,ry,rx)
        mask = (support >= thresh).float().view(1, 1, rz, ry, rx)
        self.register_buffer("support", support)
        self.register_buffer("grad_mask", mask)
        self.grid.register_hook(lambda g: g * self.grad_mask)
        return int((mask == 0).sum())

    def regularization(self) -> Tensor:
        """Mask-aware TV on the log10-E grid (same form as V0VoxelField)."""
        g = self.grid[0]                                    # (1,rz,ry,rx)
        m = getattr(self, "grad_mask", None)
        tv = g.new_zeros(())
        for d in (1, 2, 3):
            n = g.shape[d] - 1
            if n < 1:
                continue
            dg = (g.narrow(d, 1, n) - g.narrow(d, 0, n)).abs()
            if m is not None:
                mm = m[0, 0]
                w = mm.narrow(d - 1, 1, n) * mm.narrow(d - 1, 0, n)
                tv = tv + (dg * w.unsqueeze(0)).sum() / w.sum().clamp(min=1.0)
            else:
                tv = tv + dg.mean()
        return tv

    def forward(self, pos: Tensor) -> Tensor:
        """pos (n,3) -> log10 E (n,), clamped to logE_clamp."""
        n = pos.shape[0]
        p = 2.0 * (pos - self.aabb[0]) / (self.aabb[1] - self.aabb[0] + 1e-8) - 1.0
        out = F.grid_sample(self.grid, p.clamp(-1.0, 1.0).view(1, n, 1, 1, 3),
                            mode="bilinear", align_corners=True)   # (1,1,n,1,1)
        return out.reshape(n).clamp(self.logE_clamp[0], self.logE_clamp[1])


def eval_Egrid_at(grid: Tensor, aabb: Tensor, pos: Tensor) -> Tensor:
    """Stateless log10-E eval: grid (1,1,rz,ry,rx), pos (n,3) -> (n,) log10 E."""
    n = pos.shape[0]
    p = 2.0 * (pos - aabb[0]) / (aabb[1] - aabb[0] + 1e-8) - 1.0
    out = F.grid_sample(grid, p.clamp(-1.0, 1.0).view(1, n, 1, 1, 3),
                        mode="bilinear", align_corners=True)
    return out.reshape(n)

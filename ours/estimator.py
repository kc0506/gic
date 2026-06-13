# coding=utf-8
"""AnchoredEstimator: our freeze-BC + field-injection extension of GIC's Estimator.

NOTE: no `from __future__ import annotations` here -- stringified annotations
break taichi's @ti.kernel argument parsing (ti.types.ndarray()).
"""
import taichi as ti
import torch

from simulator import Estimator


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

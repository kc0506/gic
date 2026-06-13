# coding=utf-8
"""Particle->gaussian drive for the full-resolution image render.

A faithful port of PhysDreamer's kmeans render path (physdreamer.gaussian_3d.
utils.rigid_body_utils + projects/inference/local_utils.interpolate_points_w_R):
the full-res object gaussians follow the MPM deformation by interpolating each
gaussian's motion from its top_k nearest particles (averaged offset + rigid-fit
rotation). This is the proven forward renderer -- we render the REAL object
(anisotropic gaussians, real SH) instead of isotropic blobs.

The position update (mean of the top_k particle offsets) is differentiable wrt
the particle displacement, so the image loss backprops to the particles -> E/v0.
The rotation (rigid-fit R via SVD) is appearance-only and computed detached: the
splat orientation is still correct, but its unstable SVD gradient is omitted (the
position path carries the fit signal).

gic env cannot import physdreamer (no module / needs jaxtyping), so the four
rigid_body_utils helpers are ported verbatim here (pure torch).
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# --- ported verbatim from physdreamer.gaussian_3d.utils.rigid_body_utils ---


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    ret = torch.zeros_like(x)
    positive_mask = x > 0
    ret[positive_mask] = torch.sqrt(x[positive_mask])
    return ret


def standardize_quaternion(quaternions: torch.Tensor) -> torch.Tensor:
    return torch.where(quaternions[..., 0:1] < 0, -quaternions, quaternions)


def quaternion_multiply(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    ow = aw * bw - ax * bx - ay * by - az * bz
    ox = aw * bx + ax * bw + ay * bz - az * by
    oy = aw * by - ax * bz + ay * bw + az * bx
    oz = aw * bz + ax * by - ay * bx + az * bw
    return standardize_quaternion(torch.stack((ow, ox, oy, oz), -1))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(
        matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(torch.stack([
        1.0 + m00 + m11 + m22, 1.0 + m00 - m11 - m22,
        1.0 - m00 + m11 - m22, 1.0 - m00 - m11 + m22], dim=-1))
    quat_by_rijk = torch.stack([
        torch.stack([q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01], dim=-1),
        torch.stack([m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20], dim=-1),
        torch.stack([m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21], dim=-1),
        torch.stack([m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2], dim=-1),
    ], dim=-2)
    flr = torch.tensor(0.1).to(dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(flr))
    return quat_candidates[
        F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5, :].reshape(batch_dim + (4,))


def get_rigid_transform(A: torch.Tensor, B: torch.Tensor) -> tuple:
    """Rigid (R,t) so that B ~= R @ A + t, batched over [..., N, 3]."""
    centroid_A = torch.mean(A, dim=-2, keepdim=True)
    centroid_B = torch.mean(B, dim=-2, keepdim=True)
    H = (A - centroid_A).transpose(-2, -1) @ (B - centroid_B)
    U, S, Vt = torch.linalg.svd(H)
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)
    flip = (torch.det(R) < 0) * -2.0 + 1.0
    pad = torch.stack([torch.ones_like(flip), torch.ones_like(flip), flip], dim=-1)
    Vt = Vt * pad[..., None]
    R = Vt.transpose(-2, -1) @ U.transpose(-2, -1)
    t = centroid_B - (R @ centroid_A.transpose(-2, -1)).transpose(-2, -1)
    return R, t.transpose(-2, -1)


# --- the drive ---


class TopKDrive:
    """PhysDreamer top_k interpolation: object gaussians follow the particles.

    gauss_xyz0/gauss_rot0: (Ng,3)/(Ng,4) canonical full-res gaussians (sim space).
    particle_xyz0: (Np,3) canonical MPM particle positions (sim space).
    top_k_index: (Nobj,k) particle indices per OBJECT gaussian (cache top_k_index).
    sim_mask: (Ng,) bool -- the object gaussians (driven); the rest stay static.
    """

    def __init__(self, gauss_xyz0: torch.Tensor, gauss_rot0: torch.Tensor,
                 particle_xyz0: torch.Tensor, top_k_index: torch.Tensor,
                 sim_mask: torch.Tensor) -> None:
        self.gauss_xyz0 = gauss_xyz0
        self.gauss_rot0 = gauss_rot0
        self.p0 = particle_xyz0
        self.top_k = top_k_index
        self.sim_mask = sim_mask
        self.query_rot = gauss_rot0[sim_mask]

    def __call__(self, particle_disp: torch.Tensor) -> tuple:
        """particle_disp (Np,3) -> (d_xyz (Ng,3) differentiable, rot_full (Ng,4)).

        d_xyz is zero outside sim_mask (static bg/foreground), the averaged top_k
        offset inside (carries gradient to the particles). rot_full is detached.
        """
        top_k_disp = particle_disp[self.top_k]            # (Nobj,k,3)
        avg_offsets = top_k_disp.mean(dim=1)              # (Nobj,3) differentiable
        d_xyz = torch.zeros_like(self.gauss_xyz0)
        d_xyz = d_xyz.index_put((self.sim_mask.nonzero(as_tuple=True)[0],), avg_offsets)
        with torch.no_grad():                             # rotation = appearance only
            src = self.p0[self.top_k]                     # (Nobj,k,3)
            R, _ = get_rigid_transform(src, src + top_k_disp.detach())
            new_rot = quaternion_multiply(matrix_to_quaternion(R), self.query_rot)
            rot_full = self.gauss_rot0.clone()
            rot_full[self.sim_mask] = new_rot
        return d_xyz, rot_full

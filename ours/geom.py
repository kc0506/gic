# coding=utf-8
"""Coordinate helpers shared across entrypoints."""
import math

import torch

CENTER = 0.5  # scenes are normalized into ~[0,1]^3; rotations are about the centre


def rot_xyz(xyz: torch.Tensor, deg: float) -> torch.Tensor:
    """Rotate (N,3) positions about the z axis through the center (0.5, 0.5).

    Used to align a scene with the warp GT dump's rot_z_deg. Only x,y change;
    z is untouched. Returns a clone (input unmodified).
    """
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y = xyz[:, 0] - 0.5, xyz[:, 1] - 0.5
    q = xyz.clone()
    q[:, 0] = c * x - s * y + 0.5
    q[:, 1] = s * x + c * y + 0.5
    return q


def euler_matrix(ax: float, ay: float, az: float) -> torch.Tensor:
    """XYZ-extrinsic rotation matrix R = Rz @ Ry @ Rx (apply Rx, then Ry, then Rz), degrees.

    euler_matrix(0,0,z) == the rotation rot_xyz applies, so rot_euler with (0,0,z)
    is bit-identical to rot_xyz(.,z) -- one code path for single-axis and 3-axis.
    """
    rx, ry, rz = (math.radians(a) for a in (ax, ay, az))
    cx, sx, cy, sy, cz, sz = (math.cos(rx), math.sin(rx), math.cos(ry),
                              math.sin(ry), math.cos(rz), math.sin(rz))
    Rx = torch.tensor([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=torch.float32)
    Ry = torch.tensor([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=torch.float32)
    Rz = torch.tensor([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=torch.float32)
    return Rz @ Ry @ Rx


def rot_euler(xyz: torch.Tensor, ax: float, ay: float, az: float) -> torch.Tensor:
    """Rotate (N,3) positions about CENTER by the XYZ-extrinsic Euler angles (deg)."""
    R = euler_matrix(ax, ay, az).to(xyz.device)
    return (xyz - CENTER) @ R.t() + CENTER


def rot_euler_quat(quats: torch.Tensor, ax: float, ay: float, az: float) -> torch.Tensor:
    """Left-multiply each [w,x,y,z] gaussian quaternion by the quaternion of euler_matrix.

    Anisotropic splats must rotate their ORIENTATION by the same rotation as the
    positions, or every splat points the wrong way (cf. _rotz_quat for the z-only case).
    """
    from ours.gauss_drive import matrix_to_quaternion, quaternion_multiply
    R = euler_matrix(ax, ay, az).to(quats.device)
    qR = matrix_to_quaternion(R).reshape(1, 4).expand(quats.shape[0], 4)
    return quaternion_multiply(qR, quats)

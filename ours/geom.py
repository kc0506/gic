# coding=utf-8
"""Coordinate helpers shared across entrypoints."""
import math

import torch


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

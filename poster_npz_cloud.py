#!/usr/bin/env python
# coding=utf-8
"""poster_npz_cloud: per-frame 3D point cloud from a saved traj.npz (e.g. f0 ybend).

Loads a trajectory .npz (traj (T,N,3) + optional ydef (T,N) deflection) and renders
ONE separate image per requested frame, upper-right matplotlib 3D view, z vertical
(physics +z up), axes hidden. Colours by deflection (shared scale across frames) so
the bend reads. CPU only.

Usage:
  python poster_npz_cloud.py --npz <traj.npz> --frames 1 4 8
"""
import os
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

ASSETS = "/tmp2/b10401006/ev-project/generative-phys/poster/assets"


@dataclass
class Config:
    npz: str
    frames: List[int] = field(default_factory=lambda: [1, 4, 8])
    color_key: str = "stretch"
    """'stretch' = per-particle local strain (strain_proxy, sequential 0..max);
    an npz field name (e.g. ydef/disp) = diverging shared scale; '' = solid colour"""
    cmap: str = "magma"
    elev: float = 22.0
    azim: float = -60.0
    aspect_y: float = 1.0
    """visual stretch of the y axis (box aspect); >1 elongates y to exaggerate the bend"""
    psize: float = 4.0
    combined: bool = False
    """also emit ONE image with all frames shifted along +x (shift_frac of x-width each),
    alpha ramping faint->solid so the marching progression reads"""
    shift_frac: float = 0.3
    op_lo: float = 0.3
    prefix: str = "ybend_cloud"


def run(cfg: Config) -> None:
    d = np.load(cfg.npz, allow_pickle=True)
    traj = d["traj"]                                   # (T,N,3)
    # colour values per requested frame: 'stretch' = local strain (sequential 0..max);
    # an npz field = diverging (+-vmax); else solid.
    colf, vmin, vmax = None, None, None
    if cfg.color_key == "stretch":
        import torch
        from ours.fields import strain_proxy
        Xr = torch.from_numpy(d["X_rest"]).float()
        colf = {f: strain_proxy(Xr, torch.from_numpy(traj[f]).float()).numpy() for f in cfg.frames}
        vmax = float(max(c.max() for c in colf.values())); vmin = 0.0
    elif cfg.color_key and cfg.color_key in d:
        arr = d[cfg.color_key]
        colf = {f: arr[f] for f in cfg.frames}
        vmax = float(np.abs(np.stack(list(colf.values()))).max()); vmin = -vmax

    def col(f):  # back-compat accessor
        return colf[f] if colf is not None else None

    allP = traj[cfg.frames].reshape(-1, 3)
    ctr = allP.mean(0); r = (allP.max(0) - allP.min(0)).max() / 2.0

    for f in cfg.frames:
        P = traj[f]
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        kw = dict(c=col(f), cmap=cfg.cmap, vmin=vmin, vmax=vmax) if colf is not None \
            else dict(color="#2c6fbb")
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=cfg.psize, linewidths=0,
                   depthshade=False, **kw)
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] - r, ctr[1] + r)
        ax.set_zlim(ctr[2] - r, ctr[2] + r)
        ax.set_box_aspect((1, cfg.aspect_y, 1))
        ax.view_init(elev=cfg.elev, azim=cfg.azim)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        name = f"{cfg.prefix}_f{f}.png"
        fig.savefig(os.path.join(ASSETS, name), dpi=150, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"[npz-cloud] frame {f} -> {ASSETS}/{name}")

    if cfg.combined:
        xw = float(allP[:, 0].max() - allP[:, 0].min())
        step = xw * cfg.shift_frac
        ops = np.linspace(cfg.op_lo, 1.0, len(cfg.frames))
        fig = plt.figure(figsize=(6 + 2 * len(cfg.frames), 6))
        ax = fig.add_subplot(111, projection="3d")
        shifted = []
        for k, f in enumerate(cfg.frames):
            P = traj[f].copy(); P[:, 0] += k * step
            shifted.append(P)
            kw = dict(c=col(f), cmap=cfg.cmap, vmin=vmin, vmax=vmax) if colf is not None \
                else dict(color="#2c6fbb")
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=cfg.psize, linewidths=0,
                       depthshade=False, alpha=float(ops[k]), **kw)
        allS = np.concatenate(shifted, 0)
        c2 = allS.mean(0)
        rx = (allS[:, 0].max() - allS[:, 0].min()) / 2.0
        ry = max((allS[:, 1].max() - allS[:, 1].min()) / 2.0, r)
        rz = max((allS[:, 2].max() - allS[:, 2].min()) / 2.0, r)
        ax.set_xlim(c2[0] - rx, c2[0] + rx)
        ax.set_ylim(c2[1] - ry, c2[1] + ry)
        ax.set_zlim(c2[2] - rz, c2[2] + rz)
        ax.set_box_aspect((rx, ry * cfg.aspect_y, rz))
        ax.view_init(elev=cfg.elev, azim=cfg.azim)
        ax.set_axis_off()
        fig.tight_layout(pad=0)
        cname = f"{cfg.prefix}_combined.png"
        fig.savefig(os.path.join(ASSETS, cname), dpi=150, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"[npz-cloud] combined ({len(cfg.frames)} frames, +x shift {cfg.shift_frac}x"
              f"={step:.3f}, alpha {cfg.op_lo}->1.0) -> {ASSETS}/{cname}")

    print(f"[npz-cloud] view elev{cfg.elev}/azim{cfg.azim}, color={cfg.color_key or 'solid'}, "
          f"shared vmax={vmax}")


if __name__ == "__main__":
    run(tyro.cli(Config))

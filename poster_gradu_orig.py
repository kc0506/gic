#!/usr/bin/env python
# coding=utf-8
"""poster_gradu_orig: replot f0_gradu_viz's ORIGINAL 3D cloud style for chosen frames.

Mirrors reuse_mpm/explore/f0_gradu_viz.py's 3d panel exactly: default matplotlib 3D
view + axes, colour = displacement-from-rest, viridis, vmin0 / vmax=quantile(disp,.98)
shared across frames. One separate image per frame. Reads the saved traj.npz (no sim).

Usage:  python poster_gradu_orig.py --npz <traj.npz> --frames 0 4 8
"""
import os
from dataclasses import dataclass, field
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tyro

ASSETS = "/tmp2/b10401006/ev-project/generative-phys/poster/assets"


@dataclass
class Config:
    npz: str
    """.npz with traj (T,N,3)+X_rest, OR a traj .npy (then --rest-npy / frame0 for ref)"""
    rest_npy: str = ""
    """rest reference for disp colouring; '' = npz X_rest, or frame0 for a .npy traj"""
    color_mode: str = "disp"
    """'disp' = |x - rest| per frame; 'ft' = per-frame local strain F(t) (strain_proxy,
    rigid-invariant, computed from neighbour offsets vs rest); 'f0' = ||F0-I||_F static"""
    f0_npy: str = ""
    """path to f0.npy (N,3,3) for --color-mode f0"""
    frames: List[int] = field(default_factory=lambda: [0, 4, 8])
    cmap: str = "viridis"
    psize: float = 4.0
    axes: bool = True
    """keep the default matplotlib 3D axes (the 'original' look); False = hide"""
    overlay: bool = False
    """superimpose all --frames in ONE image (time-coloured by --cmap + alpha ramp)
    instead of one image per frame"""
    op_lo: float = 0.2
    elev: float = 30.0
    azim: float = -60.0
    prefix: str = "ybend_gradu"


def run(cfg: Config) -> None:
    if cfg.npz.endswith(".npy"):
        traj = np.load(cfg.npz)                            # (T,N,3)
        Xr = np.load(cfg.rest_npy) if cfg.rest_npy else traj[0]
    else:
        d = np.load(cfg.npz, allow_pickle=True)
        traj = d["traj"]
        Xr = np.load(cfg.rest_npy) if cfg.rest_npy else d["X_rest"]
    disp = np.linalg.norm(traj - Xr[None], axis=2)         # (T,N)
    # colour value per frame: disp | F(t) local-strain | static F0 intensity
    if cfg.color_mode == "ft":
        import torch
        from ours.fields import strain_proxy
        ref = torch.from_numpy(np.asarray(Xr)).float()
        ftint = {f: strain_proxy(ref, torch.from_numpy(traj[f]).float()).numpy() for f in cfg.frames}
        cval = lambda f: ftint[f]
        vmax = float(max(np.quantile(v, 0.98) for v in ftint.values())) or 1e-3
    elif cfg.color_mode == "f0":
        f0int = np.linalg.norm(np.load(cfg.f0_npy) - np.eye(3), axis=(1, 2))   # (N,)
        cval = lambda f: f0int
        vmax = float(np.quantile(f0int, 0.98)) or 1e-3
    else:
        cval = lambda f: disp[f]
        vmax = float(np.quantile(disp, 0.98)) or 1e-3
    mins = traj.reshape(-1, 3).min(0); maxs = traj.reshape(-1, 3).max(0)

    rng = np.maximum(maxs - mins, 1e-6)

    def _style(ax):
        ax.set_xlim(mins[0], maxs[0]); ax.set_ylim(mins[1], maxs[1]); ax.set_zlim(mins[2], maxs[2])
        ax.set_box_aspect((rng[0], rng[1], rng[2]))   # TRUE data proportions (no forced cube)
        ax.view_init(elev=cfg.elev, azim=cfg.azim)
        if not cfg.axes:
            ax.set_axis_off()
        else:  # keep gridlines + box edges, drop tick NUMBERS and the grey pane fill
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_ticklabels([])
                axis.pane.fill = False
                axis.pane.set_edgecolor((0.55, 0.55, 0.55, 1.0))
            ax.grid(True)

    if cfg.overlay:
        cmap = plt.get_cmap(cfg.cmap)
        ops = np.linspace(cfg.op_lo, 1.0, len(cfg.frames))
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d"); ax.set_proj_type("ortho")
        for k, f in enumerate(cfg.frames):
            P = traj[f]
            ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=cfg.psize, linewidths=0, depthshade=False,
                       color=cmap(k / max(len(cfg.frames) - 1, 1)), alpha=float(ops[k]))
        _style(ax)
        fig.tight_layout(pad=0.2)
        name = f"{cfg.prefix}_overlay.png"
        fig.savefig(os.path.join(ASSETS, name), dpi=150, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"[gradu-orig] OVERLAY {cfg.frames} (time-cmap {cfg.cmap}, alpha {cfg.op_lo}->1.0) "
              f"-> {ASSETS}/{name}")
        return

    for f in cfg.frames:
        fig = plt.figure(figsize=(6, 5))
        ax = fig.add_subplot(111, projection="3d")
        ax.set_proj_type("ortho")   # no perspective -> edges stay parallel across side-by-side frames
        ax.scatter(traj[f][:, 0], traj[f][:, 1], traj[f][:, 2], c=cval(f), s=cfg.psize,
                   cmap=cfg.cmap, vmin=0, vmax=vmax, depthshade=False)
        _style(ax)
        fig.tight_layout(pad=0.2)
        name = f"{cfg.prefix}_f{f}.png"
        fig.savefig(os.path.join(ASSETS, name), dpi=150, transparent=True, bbox_inches="tight")
        plt.close(fig)
        print(f"[gradu-orig] frame {f} -> {ASSETS}/{name}")
    print(f"[gradu-orig] cmap {cfg.cmap}, color=disp, vmax(q98)={vmax:.4f}, axes={cfg.axes}")


if __name__ == "__main__":
    run(tyro.cli(Config))

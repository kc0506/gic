# coding=utf-8
"""Forward-only previews of NON-UNIFORM v0 GT candidates (field-v0 next stage).

Round 2 (user feedback: round-1 variants all looked like uniform y):
  - ANALYTIC per-particle fields (no grid): res-4 has only 4 nodes along z
    (spacing = 1/3 of the long axis), too coarse for narrow profiles. Visual
    selection first; discretization (likely z-refined anisotropic grid) is a
    fit-stage decision.
  - every variant ALSO gets a vs_uniform.gif overlay (blue = uniform -0.5y,
    red = variant) -- the fair visual test of "does non-uniformity show".
  - --no_anchor: same cord but fully FREE (cheap probe of the 'free object'
    idea; on a free body a v0 gradient = visible rotation/arching).

Variants (zt = 0 at the anchor end of z, 1 at the free tip):
  ramp_y     v_y = -0.5*zt                          whip: tip fastest
  mid_kick   v_y = -0.5*exp(-((zt-.5)/.12)^2)       middle kicked, tail at rest
  true_bend  v_y = +0.5*exp(-((zt-.45)/.15)^2)
                   -0.5*exp(-((zt-1.)/.18)^2)       middle vs tail OPPOSITE
  twist_xy   v = 0.5*(cos(pi*zt), sin(pi*zt), 0)    direction rotates along z

Usage (gic env, gic root):
  python preview_v0_fields.py --scene_cache ...            # anchored sweep
  python preview_v0_fields.py --scene_cache ... --no_anchor --out_root .../v0_preview_free
"""

from roundtrip_ours_scene import AnchoredEstimator, load_our_scene, save_overlay_gif
from roundtrip_sim2sim import rollout_collect_surfaces, set_params

import json
import math
import os
import time
from argparse import ArgumentParser, Namespace

import taichi as ti
import torch
import torch.nn as nn

from simulator import Estimator
from v0_field_ours import variant_field

VARIANTS = ("ramp_y", "mid_kick", "true_bend", "twist_xy")


class AnalyticV0(nn.Module):
    """Analytic v0(pos) shim for set_v0_field (preview only; nothing trained).

    Exposes a dummy `grid` param because set_v0_field builds an optimizer on it.
    zt is the particle's normalized position along z in [z_lo, z_hi], flipped
    so zt=0 sits at the anchor end.
    """

    def __init__(self, name: str, z_lo: float, z_hi: float, flip: bool,
                 scale: float = 1.0) -> None:
        super().__init__()
        self.name, self.z_lo, self.z_hi, self.flip = name, z_lo, z_hi, flip
        self.scale = scale
        self.grid = nn.Parameter(torch.zeros(1))  # placeholder for the optimizer

    def forward(self, pos: torch.Tensor) -> torch.Tensor:
        """pos (N,3) -> v0 (N,3)."""
        zt = ((pos[:, 2] - self.z_lo) / (self.z_hi - self.z_lo + 1e-8)).clamp(0.0, 1.0)
        if self.flip:
            zt = 1.0 - zt
        return variant_field(self.name, zt) * self.scale


def save_motion_gif(frames: list, v0_mag: torch.Tensor, path: str,
                    max_pts: int = 1500) -> None:
    """Single-cloud rollout, 3D + xy/xz/yz views; color = |v0| of the particle.

    frames: list of (N,3) cuda tensors; v0_mag: (N,) per-particle |v0|.
    """
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    sel = np.random.default_rng(0).permutation(frames[0].shape[0])[:max_pts]
    pts = np.stack([f.cpu().numpy()[sel] for f in frames])    # (F, M, 3)
    c = v0_mag.cpu().numpy()[sel]                             # (M,)
    mins, maxs = pts.reshape(-1, 3).min(0), pts.reshape(-1, 3).max(0)
    fig = plt.figure(figsize=(14.5, 3.9))
    ax3d = fig.add_subplot(1, 4, 1, projection="3d")
    axes2d = [fig.add_subplot(1, 4, k) for k in (2, 3, 4)]
    planes = (("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2))

    def update(f: int):
        ax3d.cla()
        ax3d.scatter(pts[f][:, 0], pts[f][:, 2], pts[f][:, 1], s=1, c=c, cmap="plasma")
        ax3d.set_xlim(mins[0], maxs[0]); ax3d.set_ylim(mins[2], maxs[2]); ax3d.set_zlim(mins[1], maxs[1])
        ax3d.set_xlabel("x"); ax3d.set_ylabel("z"); ax3d.set_zlabel("y")
        for ax, (na, nb, a, b) in zip(axes2d, planes):
            ax.cla()
            ax.scatter(pts[f][:, a], pts[f][:, b], s=1, c=c, cmap="plasma")
            ax.set_xlim(mins[a], maxs[a]); ax.set_ylim(mins[b], maxs[b])
            ax.set_xlabel(na); ax.set_ylabel(nb)
            ax.set_aspect("equal")
        fig.suptitle(f"frame {f} — color: |v0|", fontsize=10)

    anim = FuncAnimation(fig, update, frames=len(frames))
    anim.save(path, writer=PillowWriter(fps=8))
    plt.close(fig)


def save_field_quiver(xyz_free: torch.Tensor, v: torch.Tensor, path: str,
                      n_arrows: int = 300) -> None:
    """GT field at free particles: 3 plane views, color = |v0|, quiver arrows."""
    import numpy as np
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = xyz_free.cpu().numpy()
    vv = v.cpu().numpy()
    mag = np.linalg.norm(vv, axis=-1)
    order = np.argsort(mag)                                  # max-on-top
    sel = np.random.default_rng(0).permutation(p.shape[0])[:n_arrows]
    planes = (("x", "y", 0, 1), ("x", "z", 0, 2), ("y", "z", 1, 2))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4))
    for ax, (na, nb, a, b) in zip(axes, planes):
        sc = ax.scatter(p[order, a], p[order, b], c=mag[order], s=2, cmap="plasma")
        ax.quiver(p[sel, a], p[sel, b], vv[sel, a], vv[sel, b], color="k",
                  width=0.0025, alpha=0.6, scale=6.0)
        ax.set_xlabel(na); ax.set_ylabel(nb)
        ax.set_aspect("equal")
    fig.colorbar(sc, ax=axes, label="|v0|", shrink=0.85)
    fig.suptitle("GT v0 field at free particles", fontsize=10)
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    start = time.time()
    parser = ArgumentParser(description="forward-only non-uniform v0 GT previews")
    parser.add_argument("--config", default="config/ours/telephone.json", type=str)
    parser.add_argument("--scene_cache", required=True, type=str)
    parser.add_argument("--gt_logE", default=5.0, type=float)
    parser.add_argument("--gt_nu", default=0.3, type=float)
    parser.add_argument("--rot_z_deg", default=67.6, type=float)
    parser.add_argument("--anchor_mass_scale", default=1e4, type=float)
    parser.add_argument("--no_anchor", action="store_true",
                        help="release the freeze: whole object FREE (gravity is 0)")
    parser.add_argument("--mpm_iter_cnt", default=64, type=int)
    parser.add_argument("--n_frames", default=14, type=int)
    parser.add_argument("--ti_mem_frac", default=0.3, type=float)
    parser.add_argument("--v_scale", default=1.0, type=float,
                        help="multiply ALL variant fields (and the uniform reference) "
                             "by this -- exaggerate until differences are visible")
    parser.add_argument("--variants", default=",".join(VARIANTS), type=str,
                        help="comma list; run ONLY these variants")
    parser.add_argument("--out_root", default="output/ours_telephone/v0_preview2", type=str)
    args = parser.parse_args()

    with open(args.config) as f:
        phys_args = Namespace(**json.load(f)["physics"])
    if args.mpm_iter_cnt:
        phys_args.mpm_iter_cnt = args.mpm_iter_cnt
    phys_args.n_frames = args.n_frames

    xyz, anchor_mask = load_our_scene(args.scene_cache)
    t = math.radians(args.rot_z_deg)
    c, s = math.cos(t), math.sin(t)
    x, y = xyz[:, 0] - 0.5, xyz[:, 1] - 0.5
    xyz = xyz.clone()
    xyz[:, 0] = c * x - s * y + 0.5
    xyz[:, 1] = s * x + c * y + 0.5

    # zt orientation comes from the ORIGINAL freeze layout even in --no_anchor
    # mode, so profiles stay comparable between the two sweeps
    anchor_at_high_z = bool(xyz[anchor_mask][:, 2].mean() > xyz[~anchor_mask][:, 2].mean())
    if args.no_anchor:
        anchor_mask = torch.zeros_like(anchor_mask)
        print("[preview] NO-ANCHOR mode: whole object free")
    free = ~anchor_mask
    z_lo, z_hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    print(f"[preview] anchors at {'HIGH' if anchor_at_high_z else 'LOW'} z; "
          f"zt=0 there, zt=1 at free tip")

    ti.init(arch=ti.cuda, debug=False, fast_math=False,
            device_memory_fraction=args.ti_mem_frac)
    dummy_gts = [xyz.clone() for _ in range(args.n_frames)]
    estimator = AnchoredEstimator(
        phys_args, "float32", dummy_gts, surface_index=None, init_vol=xyz,
        dynamic_scene=None, image_scale=1.0, pipeline=None, image_op=None,
    )
    estimator.set_anchor(anchor_mask, args.anchor_mass_scale)
    set_params(estimator, args.gt_logE, args.gt_nu, [0.0, 0.0, 0.0])

    def rollout_variant(name: str) -> tuple:
        """-> (frames: list of (N,3) cuda, v_part (N,3) cpu)."""
        shim = AnalyticV0(name, z_lo, z_hi, flip=anchor_at_high_z, scale=args.v_scale)
        estimator.set_v0_field(shim, xyz, lr=0.0)
        v_part = shim(xyz).detach() * free.float().unsqueeze(1)   # (N,3)
        estimator.max_f = args.n_frames
        return rollout_collect_surfaces(estimator), v_part

    os.makedirs(args.out_root, exist_ok=True)
    uni_frames, _ = rollout_variant("uniform_y")

    summary = {}
    for name in [v.strip() for v in args.variants.split(",") if v.strip()]:
        out_dir = os.path.join(args.out_root, name)
        os.makedirs(out_dir, exist_ok=True)
        frames, v_part = rollout_variant(name)
        disp = (frames[-1] - frames[0]).norm(dim=-1)
        dvu = (frames[-1] - uni_frames[-1]).norm(dim=-1)          # divergence vs uniform
        summary[name] = {
            "free_mean_disp": float(disp[free].mean()),
            "free_max_disp": float(disp[free].max()),
            "vs_uniform_lastframe_mean": float(dvu[free].mean()),
            "vs_uniform_lastframe_max": float(dvu[free].max()),
        }
        print(f"[preview] {name}: {summary[name]}")
        save_field_quiver(xyz[free], v_part[free], os.path.join(out_dir, "gt_field.png"))
        save_motion_gif(frames, v_part.norm(dim=-1), os.path.join(out_dir, "motion.gif"))
        save_overlay_gif(uni_frames, frames, os.path.join(out_dir, "vs_uniform.gif"),
                         labels=("uniform -0.5y", name))
    with open(os.path.join(args.out_root, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[preview] DONE {len(VARIANTS)} variants, wall {time.time() - start:.0f}s")

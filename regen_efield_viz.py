# coding=utf-8
"""Offline-regenerate E-field viz from a run's ckpt.pt (no GPU, no re-train).

Reads {run_dir}/ckpt.pt (best_grid / gt_grid / aabb / res) + the scene cache,
re-evals both grids at the free particles, and writes:
  - E_gt_vs_pred.png : recovered vs GT per-particle log10 E (diagonal = perfect),
                       parametrization-free so it handles multi-valued (circular)
  - E_z_heatmap.png  : (circular only) GT (z, logE) density + recovered points

Use to backfill runs that predate the viz, or after changing viz code. Pure CPU
(reads the cache on CPU, evals grids on CPU) so it never touches a GPU.

Usage (gic env, gic repo root):
  python regen_efield_viz.py --run_dir output/ours_efield/efcirc_uniformv0 --gt_kind circular
"""
import argparse
import os

import torch

from ours.fields import eval_Egrid_at
from ours.geom import rot_xyz
from ours.viz import plot_E_gt_vs_pred, plot_E_z_heatmap

GEN = "/tmp2/b10401006/ev-project/generative-phys"


def load_scene_cpu(cache_path: str) -> tuple:
    """(xyz (N,3) f32 CPU, anchor_mask (N,) bool CPU), origin ghosts removed.

    CPU twin of ours.scene.load_our_scene (which forces .cuda()) so regen needs
    no GPU.
    """
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    disc = cache["disc"]
    xyz = disc["sim_xyzs"]            # (N0, 3) normalized [0,1]^3
    freeze = disc["freeze_mask"]      # (N0,) bool
    keep = ~(xyz == 0).all(dim=1)     # drop kmeans origin ghosts
    return xyz[keep].float(), freeze[keep]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--scene_cache",
                    default=f"{GEN}/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt")
    ap.add_argument("--rot_z_deg", type=float, default=67.6)
    ap.add_argument("--gt_kind", default="circular",
                    help="circular -> z-E heatmap; else (ramp/uniform) -> z profile")
    args = ap.parse_args()

    ck = torch.load(os.path.join(args.run_dir, "ckpt.pt"),
                    map_location="cpu", weights_only=False)
    best_grid, gt_grid, aabb = ck["best_grid"], ck["gt_grid"], ck["aabb"]

    xyz, anchor = load_scene_cpu(args.scene_cache)
    if args.rot_z_deg:
        xyz = rot_xyz(xyz, args.rot_z_deg)
    free = ~anchor
    gt_lp = eval_Egrid_at(gt_grid, aabb, xyz)[free]      # (M,) GT per-particle logE
    pred_lp = eval_Egrid_at(best_grid, aabb, xyz)[free]  # (M,) recovered
    z_lo, z_hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    flip_z = bool(xyz[anchor][:, 2].mean() > xyz[free][:, 2].mean())
    zt = ((xyz[free][:, 2] - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0, 1)
    if flip_z:
        zt = 1.0 - zt

    tag = os.path.basename(args.run_dir.rstrip("/"))
    err = float((pred_lp - gt_lp).abs().mean())
    plot_E_gt_vs_pred(gt_lp, pred_lp,
                      os.path.join(args.run_dir, "E_gt_vs_pred.png"),
                      color=zt, color_label="zt (0=anchor end)",
                      title=f"{tag}: recovered vs GT log10 E (err all {err:.3f})")
    msg = "E_gt_vs_pred.png"
    if args.gt_kind == "circular":
        plot_E_z_heatmap(zt, gt_lp, pred_lp,
                         os.path.join(args.run_dir, "E_z_heatmap.png"),
                         title=f"{tag}: GT (z, logE) density + recovered points")
        msg += " + E_z_heatmap.png"
    print(f"[regen] {tag}: err all {err:.3f}; wrote {msg}")


if __name__ == "__main__":
    main()

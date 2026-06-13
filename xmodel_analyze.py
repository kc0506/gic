# coding=utf-8
"""Cross-model comparison analysis: warp vs gic trajectories, pairwise (no chamfer).

Inputs: warp_traj.npy (14,7555,3), gic_traj.npy (14,7025,3); ghost-filter warp
to align indices. Outputs under reports/20260611_gic_anchor/xmodel/.
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

GEN = "/tmp2/b10401006/ev-project/generative-phys"

_ap = argparse.ArgumentParser()
_ap.add_argument("--label", type=str, default=None,
                 help="run label; derives warp/gic/out paths (None = original tele_E1e5 layout)")
_ap.add_argument("--cache", type=str, default=f"{GEN}/outputs/forward_gen/06_tele_E1e5/scene_cache.pt")
_a = _ap.parse_args()

if _a.label is None:  # original (pre-parameterization) layout
    WARP = f"{GEN}/outputs/explore/xmodel_dump/tele_E1e5/warp_traj.npy"
    GIC = "/tmp2/b10401006/ev-project/gic/output/xmodel/gic_traj.npy"
    OUT = f"{GEN}/reports/20260611_gic_anchor/xmodel"
else:
    WARP = f"{GEN}/outputs/explore/xmodel_dump/{_a.label}/warp_traj.npy"
    GIC = f"/tmp2/b10401006/ev-project/gic/output/xmodel/{_a.label}/gic_traj.npy"
    OUT = f"{GEN}/reports/20260611_gic_anchor/xmodel/{_a.label}"
CACHE = _a.cache
os.makedirs(OUT, exist_ok=True)

cache = torch.load(CACHE, map_location="cpu", weights_only=False)
disc = cache["disc"]
ghost = (disc["sim_xyzs"] == 0).all(dim=1).numpy()
anchors = disc["freeze_mask"].numpy()[~ghost]
free = ~anchors

warp = np.load(WARP)[:, ~ghost]          # (T, N, 3)
gic = np.load(GIC)                        # (T, N, 3)
assert warp.shape == gic.shape, (warp.shape, gic.shape)
T, N, _ = warp.shape

d = np.linalg.norm(warp - gic, axis=-1)              # (T, N) pairwise per-particle
motion = np.linalg.norm(warp - warp[0:1], axis=-1)   # (T, N) warp displacement vs frame0

print(f"frame0 max pairwise d = {d[0].max():.3e} (should be ~0)")
rows = []
for f in range(T):
    rows.append(dict(
        frame=f,
        free_mean=float(d[f, free].mean()), free_median=float(np.median(d[f, free])),
        free_p95=float(np.quantile(d[f, free], 0.95)), free_max=float(d[f, free].max()),
        anchor_mean=float(d[f, anchors].mean()), anchor_max=float(d[f, anchors].max()),
        warp_motion_free_mean=float(motion[f, free].mean()),
    ))
    r = rows[-1]
    print(f"f{f:02d}: free d mean={r['free_mean']:.4f} p95={r['free_p95']:.4f} "
          f"max={r['free_max']:.4f} | anchor mean={r['anchor_mean']:.5f} "
          f"| warp motion mean={r['warp_motion_free_mean']:.4f}")

with open(f"{OUT}/per_frame_stats.json", "w") as fjson:
    json.dump(rows, fjson, indent=2)

# 1. per-frame diff curves (with context scales)
fig, ax = plt.subplots(figsize=(7, 4.5))
fr = np.arange(T)
ax.plot(fr, [r["free_mean"] for r in rows], "-o", ms=3, label="pairwise diff mean (free)")
ax.plot(fr, [r["free_p95"] for r in rows], "-s", ms=3, label="pairwise diff p95 (free)")
ax.plot(fr, [r["free_max"] for r in rows], "-^", ms=3, label="pairwise diff max (free)")
ax.plot(fr, [r["anchor_mean"] for r in rows], "--", label="pairwise diff mean (anchor)")
ax.plot(fr, [r["warp_motion_free_mean"] for r in rows], ":k", label="warp motion mean (signal scale)")
ax.axhline(0.03125, color="gray", lw=0.8, ls="-.", label="dx = 1/32")
ax.set_yscale("log"); ax.set_xlabel("frame"); ax.set_ylabel("distance (normalized units)")
ax.legend(fontsize=8); ax.set_title("warp vs gic, identical params: pairwise divergence")
fig.tight_layout(); fig.savefig(f"{OUT}/diff_curves.png", dpi=120); plt.close(fig)

# 2. histograms (free particles), selected frames
sel = [1, 3, 5, 7, 9, 11, 13]
fig, axes = plt.subplots(1, len(sel), figsize=(3.0 * len(sel), 3.2), sharey=True)
bins = np.logspace(-6, 0, 60)
for ax, f in zip(axes, sel):
    ax.hist(np.clip(d[f, free], 1e-6, None), bins=bins)
    ax.set_xscale("log"); ax.set_title(f"frame {f}", fontsize=9)
    ax.axvline(0.03125, color="gray", ls="-.", lw=0.8)
axes[0].set_ylabel("free particles")
fig.suptitle("pairwise |warp - gic| histograms (gray line = dx)")
fig.tight_layout(); fig.savefig(f"{OUT}/diff_histograms.png", dpi=120); plt.close(fig)

# 3. overlay gif: three axis-aligned views + 3D, blue=warp red=gic
from matplotlib.animation import FuncAnimation, PillowWriter

sub = np.random.default_rng(0).permutation(np.where(free)[0])[:1800]
fig = plt.figure(figsize=(16, 4.5))
a1 = fig.add_subplot(1, 4, 1)
a2 = fig.add_subplot(1, 4, 2)
a3 = fig.add_subplot(1, 4, 3)
a4 = fig.add_subplot(1, 4, 4, projection="3d")
allp = np.concatenate([warp[:, sub], gic[:, sub]]).reshape(-1, 3)
mins, maxs = allp.min(0) - 0.01, allp.max(0) + 0.01


def update(f: int):
    for ax, (i, j), name in ((a1, (2, 1), "z-y"), (a2, (0, 1), "x-y"), (a3, (0, 2), "x-z")):
        ax.cla()
        ax.scatter(warp[f, sub, i], warp[f, sub, j], s=1.5, c="tab:blue", label="warp")
        ax.scatter(gic[f, sub, i], gic[f, sub, j], s=1.5, c="tab:red", alpha=0.55, label="gic")
        ax.set_xlim(mins[i], maxs[i]); ax.set_ylim(mins[j], maxs[j])
        ax.set_title(f"{name}  frame {f}", fontsize=9); ax.legend(fontsize=7, loc="upper right")
    a4.cla()
    a4.scatter(warp[f, sub, 0], warp[f, sub, 2], warp[f, sub, 1], s=1, c="tab:blue", label="warp")
    a4.scatter(gic[f, sub, 0], gic[f, sub, 2], gic[f, sub, 1], s=1, c="tab:red", alpha=0.5, label="gic")
    a4.set_xlim(mins[0], maxs[0]); a4.set_ylim(mins[2], maxs[2]); a4.set_zlim(mins[1], maxs[1])
    a4.set_title(f"3D (x,z,y)  frame {f}", fontsize=9); a4.legend(fontsize=7, loc="upper right")


anim = FuncAnimation(fig, update, frames=T)
anim.save(f"{OUT}/overlay.gif", writer=PillowWriter(fps=4))
plt.close(fig)

# 4. per-frame diff coloring (z-y view), 4 frames
selc = [1, 5, 9, 13]
fig, axes = plt.subplots(1, len(selc), figsize=(3.4 * len(selc), 3.4))
for ax, f in zip(axes, selc):
    sc = ax.scatter(warp[f, free, 2], warp[f, free, 1], s=2, c=d[f, free],
                    cmap="inferno", vmin=0, vmax=max(1e-4, d[:, free].max()))
    ax.set_title(f"frame {f}", fontsize=9)
fig.colorbar(sc, ax=axes, shrink=0.85, label="|warp - gic|")
fig.suptitle("per-frame pairwise diff (z-y view, free particles)")
fig.savefig(f"{OUT}/diff_colored.png", dpi=120); plt.close(fig)

print(f"saved -> {OUT}: diff_curves.png, diff_histograms.png, overlay.gif, diff_colored.png")

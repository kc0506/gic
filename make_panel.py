# coding=utf-8
"""Tile multiple roundtrip_ours_scene runs into one panel.gif overview.

Each ROW = one run: [overlay animation | loss curve | E_traj | nu_traj].
Static columns are rendered once per run; only the overlay column animates.
CPU-only: reads each run's existing overlay.gif + result.json (no re-simulation).

Usage (any env with PIL+matplotlib):
  python make_panel.py --runs output/ours_telephone/tagA output/ours_telephone/tagB \
      --out /path/panel.gif
"""
import argparse
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

ROW_H = 240  # px, all images in a row are resized to this height


def gif_frames(path: str) -> list:
    """Load all frames of a gif as RGB PIL Images."""
    im = Image.open(path)
    frames = []
    try:
        while True:
            frames.append(im.convert("RGB").copy())
            im.seek(im.tell() + 1)
    except EOFError:
        pass
    return frames


def fig_to_img(fig) -> Image.Image:
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3]
    plt.close(fig)
    return Image.fromarray(buf)


# runs that pre-save their static figures (efield_fit) just get them collected
# here, in this fixed order, rather than re-rendered from a schema.
PRESAVED_STATICS = ["loss.png", "profile_1d.png", "E_vs_strain.png", "Eerr.png",
                    "E_proj.png", "E_grid.png", "v0field_proj.png", "v0field_grid.png"]


def render_static_cols(run_dir: str, tag: str) -> list:
    """Render the per-run curve columns from result.json.

    Layout adapts to the run kind: alternating runs get BOTH stage losses with
    round-boundary markers; frozen (flat) nu is omitted; v0 traj is a standing
    column whenever recorded. Runs whose scenario pre-saves figures (efield)
    just collect those PNGs instead of re-rendering.
    """
    r = json.load(open(os.path.join(run_dir, "result.json")))
    scen = r.get("scenario", "")
    # traj grid runs (efield / joint) pre-save all figures; image runs (incl
    # ours_image_joint_v0E) go through the generic curve path below.
    if ("efield" in scen or "joint" in scen) and "image" not in scen:
        from PIL import Image
        return [Image.open(os.path.join(run_dir, f)).convert("RGB")
                for f in PRESAVED_STATICS if os.path.exists(os.path.join(run_dir, f))]
    e_traj = r.get("E_traj")
    nu_traj = r.get("nu_traj")
    gt = r.get("gt", {})
    best = r.get("best", {})
    alt_vb = r.get("alt_vel_bounds")
    alt_pb = r.get("alt_phys_bounds")
    imgs = []

    def round_marks(ax, bounds):
        if bounds:
            for b in bounds[:-1]:
                ax.axvline(b - 0.5, color="k", ls=":", lw=0.6)

    lv, lp = r.get("losses_vel"), r.get("losses_phys")
    loss_cols = []
    if lv and lp:  # alternating: one loss column per stage
        loss_cols = [(lv, alt_vb, "loss vel-stage (v0)"), (lp, alt_pb, "loss phys-stage (E)")]
    else:
        loss_cols = [((lp or lv), alt_pb or alt_vb, "loss")]
    alt_n = r.get("alt_rounds")
    for ci, (losses, bounds, lbl) in enumerate(loss_cols):
        fig, ax = plt.subplots(figsize=(3.0, 2.4), dpi=100)
        ax.plot(losses, lw=1.2)
        if best and lbl != "loss vel-stage (v0)":
            ax.axvline(best.get("iter", 0), color="tab:green", ls=":", lw=1,
                       label="best-loss iter")
            ax.legend(fontsize=5)
        round_marks(ax, bounds)
        ax.set_yscale("log")
        head = (f"{tag}" + (f" — ALT x{alt_n}" if alt_n else "") + "\n") if ci == 0 else ""
        ax.set_title(f"{head}{lbl} (best {min(losses):.2e})", fontsize=7)
        ax.tick_params(labelsize=6)
        fig.tight_layout()
        imgs.append(fig_to_img(fig))

    if e_traj:
        fig, ax = plt.subplots(figsize=(3.0, 2.4), dpi=100)
        ax.plot(e_traj, lw=1.2)
        if "E" in gt:
            ax.axhline(gt["E"], color="k", ls="--", lw=1, label="GT")
        round_marks(ax, alt_pb)
        ax.set_yscale("log")
        err = r.get("rel_err_E")
        ax.set_title(f"E traj (err {err:+.1%})" if err is not None else "E traj", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        fig.tight_layout()
        imgs.append(fig_to_img(fig))

    nu_moves = nu_traj and (max(nu_traj) - min(nu_traj)) > 1e-6
    if nu_moves:  # frozen nu would waste a column
        fig, ax = plt.subplots(figsize=(3.0, 2.4), dpi=100)
        ax.plot(nu_traj, lw=1.2, color="tab:orange")
        if "nu" in gt:
            ax.axhline(gt["nu"], color="k", ls="--", lw=1, label="GT")
        round_marks(ax, alt_pb)
        ax.set_title("nu traj", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        fig.tight_layout()
        imgs.append(fig_to_img(fig))

    v0_traj = r.get("v0_traj")
    if not v0_traj and r.get("alt_round_summary"):
        # legacy alt runs: per-iter v0 lost, round-end snapshots only
        summ = r["alt_round_summary"]
        arr = np.asarray([[0.0, 0.0, 0.0]] + [s["v0"] for s in summ])  # (R+1, 3)
        fig, ax = plt.subplots(figsize=(3.0, 2.4), dpi=100)
        for k, (c, col) in enumerate(zip("xyz", ["tab:blue", "tab:orange", "tab:green"])):
            ax.plot(range(len(arr)), arr[:, k], "-o", ms=3, lw=1.2, color=col,
                    label=f"v0_{c}")
            if "vel" in gt:
                ax.axhline(gt["vel"][k], color=col, ls="--", lw=0.8, alpha=0.6)
        ax.set_xticks(range(len(arr)))
        ax.set_xticklabels(["init"] + [f"r{s['round']}" for s in summ], fontsize=6)
        ax.set_title("v0 @ round end (per-iter traj lost)", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        fig.tight_layout()
        imgs.append(fig_to_img(fig))
    if v0_traj:
        arr = np.asarray(v0_traj)  # (iters, 3)
        fig, ax = plt.subplots(figsize=(3.0, 2.4), dpi=100)
        # alt records v0 in vel phases; joint in the phys loop (no vel bounds)
        round_marks(ax, alt_vb if (lv and lp) else None)
        p05, p95 = r.get("v0_traj_p05"), r.get("v0_traj_p95")
        for k, (c, col) in enumerate(zip("xyz", ["tab:blue", "tab:orange", "tab:green"])):
            ax.plot(arr[:, k], lw=1.2, color=col, label=f"v0_{c}")
            if p05 and p95:  # field runs: p5-p95 spread across free particles
                lo_a, hi_a = np.asarray(p05), np.asarray(p95)
                ax.fill_between(range(len(arr)), lo_a[:, k], hi_a[:, k],
                                color=col, alpha=0.15)
            if "vel" in gt:
                ax.axhline(gt["vel"][k], color=col, ls="--", lw=0.8, alpha=0.6)
        err = r.get("v0_rel_err")
        # v0_rel_err compares MEANS over particles -- an aggregate; for field
        # runs the per-particle number is the field relL2 column, not this
        ax.set_title(f"v0 mean traj (mean-vec err {err:+.1%})" if err is not None
                     else "v0 mean traj", fontsize=7)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)
        fig.tight_layout()
        imgs.append(fig_to_img(fig))

    vf = r.get("v0_field")
    if vf:
        fig, ax = plt.subplots(figsize=(3.0, 2.4), dpi=100)
        xy_tr = vf.get("rel_l2_xy_traj")
        if xy_tr:  # headline = observable subspace; all-axes demoted to grey
            ax.plot(vf["rel_l2_traj"], lw=0.9, color="0.75", label="all axes")
            ax.plot(xy_tr, lw=1.2, color="tab:red", label="xy-only")
            pa = vf.get("per_axis_err_best")
            sub = (f" | xyz err {pa[0]:.2f}/{pa[1]:.2f}/{pa[2]:.2f}" if pa else "")
            ax.set_title(f"field relL2 xy-only (best {min(xy_tr):.1%}){sub}", fontsize=7)
            ax.legend(fontsize=5)
        else:  # legacy runs without xy traj
            ax.plot(vf["rel_l2_traj"], lw=1.2, color="tab:red")
            ax.set_title(f"field relL2 ALL-axes (best {vf['rel_l2_best']:.1%}) "
                         f"— incl. unobservable", fontsize=7)
        ax.set_yscale("log")
        ax.tick_params(labelsize=6)
        fig.tight_layout()
        imgs.append(fig_to_img(fig))
        for extra in ("profile_1d.png", "field_err_hist.png", "field_proj.png",
                      "grid_quiver.png", "grid_hist.png"):
            p = os.path.join(run_dir, extra)
            if os.path.exists(p):
                imgs.append(Image.open(p).convert("RGB"))
    return imgs


def resize_h(im: Image.Image, h: int) -> Image.Image:
    w = round(im.width * h / im.height)
    return im.resize((w, h), Image.LANCZOS)


def build_row(run_dir: str) -> tuple:
    """Return (anim_frames: list[PIL], static_strip: PIL) for one run."""
    tag = os.path.basename(os.path.normpath(run_dir))
    # image runs: the rendered gt|pred|diff is the meaningful animation; fall back
    # to the 3D-points overlay (traj runs).
    gpd = os.path.join(run_dir, "gt_pred_diff.gif")
    ov = gpd if os.path.exists(gpd) else os.path.join(run_dir, "overlay.gif")
    anim = [resize_h(f, ROW_H) for f in gif_frames(ov)] if os.path.exists(ov) else []
    statics = [resize_h(im, ROW_H) for im in render_static_cols(run_dir, tag)]
    sw = sum(im.width for im in statics)
    scat = Image.new("RGB", (sw, ROW_H), "white")
    x = 0
    for im in statics:
        scat.paste(im, (x, 0))
        x += im.width
    return anim, scat


def save_row_gif(anim: list, scat: Image.Image, out: str, fps: int) -> None:
    """Save one run's row as its own panel gif (+ static last-frame png)."""
    aw = anim[0].width if anim else 0
    frames = []
    for f in range(max(len(anim), 1)):
        canvas = Image.new("RGB", (aw + scat.width, ROW_H), "white")
        if anim:
            canvas.paste(anim[f], (0, 0))
        canvas.paste(scat, (aw, 0))
        frames.append(canvas)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    frames[-1].save(out.replace(".gif", "_lastframe.png"))


def save_run_panel(anim: list, statics: list, out: str, fps: int,
                   max_row_w: int = 1100) -> None:
    """Per-run panel, STACKED layout: animation on top, statics word-wrapped
    into rows of height ROW_H below (instead of one unreadably wide row)."""
    rows, cur, cw = [], [], 0
    for im in statics:
        if cur and cw + im.width > max_row_w:
            rows.append(cur)
            cur, cw = [], 0
        cur.append(im)
        cw += im.width
    if cur:
        rows.append(cur)
    static_w = max(sum(im.width for im in r) for r in rows) if rows else 0
    anim_w, anim_h = (anim[0].width, anim[0].height) if anim else (0, 0)
    W = max(static_w, anim_w)
    H = anim_h + ROW_H * len(rows)
    frames = []
    for f in range(max(len(anim), 1)):
        canvas = Image.new("RGB", (W, H), "white")
        if anim:
            canvas.paste(anim[f], ((W - anim_w) // 2, 0))
        y = anim_h
        for r in rows:
            x = 0
            for im in r:
                canvas.paste(im, (x, y))
                x += im.width
            y += ROW_H
        frames.append(canvas)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=int(1000 / fps), loop=0)
    frames[-1].save(out.replace(".gif", "_lastframe.png"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True, help="run dirs (each with result.json + overlay.gif)")
    ap.add_argument("--out", default=None, type=str, help="combined multi-run panel (optional)")
    ap.add_argument("--per_run", action="store_true",
                    help="also write <run_dir>/panel.gif for each run")
    ap.add_argument("--fps", default=4, type=int)
    args = ap.parse_args()
    assert args.out or args.per_run, "need --out and/or --per_run"

    rows = []  # (anim_frames(list of PIL), static_img(PIL))
    for run_dir in args.runs:
        anim, scat = build_row(run_dir)
        rows.append((anim, scat))
        if args.per_run:
            tag = os.path.basename(os.path.normpath(run_dir))
            statics = [resize_h(im, ROW_H) for im in render_static_cols(run_dir, tag)]
            p = os.path.join(run_dir, "panel.gif")
            save_run_panel(anim, statics, p, args.fps)
            print(f"per-run panel -> {p}")
    if args.out is None:
        return

    n_frames = max(len(a) for a, _ in rows if a) if any(a for a, _ in rows) else 1
    anim_w = max((a[0].width for a, _ in rows if a), default=0)
    total_w = anim_w + max(s.width for _, s in rows)
    total_h = ROW_H * len(rows)

    out_frames = []
    for f in range(n_frames):
        canvas = Image.new("RGB", (total_w, total_h), "white")
        for ri, (anim, scat) in enumerate(rows):
            y = ri * ROW_H
            if anim:
                canvas.paste(anim[min(f, len(anim) - 1)], (0, y))
            canvas.paste(scat, (anim_w, y))
        out_frames.append(canvas)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out_frames[0].save(args.out, save_all=True, append_images=out_frames[1:],
                       duration=int(1000 / args.fps), loop=0)
    # also a static png of the last frame for quick glancing
    out_frames[-1].save(args.out.replace(".gif", "_lastframe.png"))
    print(f"panel: {len(rows)} runs x {n_frames} frames -> {args.out}")


if __name__ == "__main__":
    main()

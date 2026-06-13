# coding=utf-8
"""Pure-gic IMAGE-loss roundtrip on our telephone scene (v2 cache, rot68).

Question: can image supervision alone recover global E in our scene class,
and why does gic's production config keep alpha (w_alp=1) but drop RGB
(w_img=0)?  Modes:

  --mode fit       : fix-v0 fit of E(+nu) from image loss (choose w_img/w_alp)
  --mode landscape : 1D loss(logE) curves for rgb-only AND alpha-only losses
  --mode fit_v0field : fix-E, learn the v0 FIELD (V0VoxelField) from image loss
        alone. Runs in the PHYS stage (image pos-grads are only injected there;
        the vel stage would force chamfer against the dummy gt buffers) with the
        phys optimizer swapped to the field grid, so E/nu never move. GT scene =
        uniform gt_vel or --gt_v0_variant profile (same class & res => zero
        representation floor).

Self-consistent roundtrip: GT = gic rollout at GT params rendered through ONE
static camera; pred rendered identically.  gic renders particles AS gaussians
1:1 (d_xyz = particles - canonical), so we build a pseudo-GaussianModel at the
MPM particle positions with attributes (color) transferred from each
particle's nearest original PhysDreamer gaussian -- keeps texture so RGB loss
has structure to match.
"""

# importing roundtrip_sim2sim picks a free GPU before torch/taichi touch CUDA
from roundtrip_ours_scene import (
    AnchoredEstimator,
    load_our_scene,
    set_params,
    train_ours,
)
from roundtrip_sim2sim import rollout_collect_surfaces

import json
import math
import os
import time
from argparse import ArgumentParser, Namespace

import numpy as np
import taichi as ti
import torch

from arguments import OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from simulator import Estimator
from utils.general_utils import inverse_sigmoid

GEN = "/tmp2/b10401006/ev-project/generative-phys"


class SceneShim:
    """Duck-typed stand-in for gic's Scene: just gaussians + camera list."""

    def __init__(self, gaussians: GaussianModel, cams: list) -> None:
        self.gaussians = gaussians
        self._cams = cams

    def getTrainCameras(self, scale: float = 1.0) -> list:
        return self._cams


def rot_z(p: torch.Tensor, deg: float) -> torch.Tensor:
    """Rotate (N,3) about z through (0.5, 0.5)."""
    t = math.radians(deg)
    c, s = math.cos(t), math.sin(t)
    x, y = p[:, 0] - 0.5, p[:, 1] - 0.5
    q = p.clone()
    q[:, 0] = c * x - s * y + 0.5
    q[:, 1] = s * x + c * y + 0.5
    return q


def build_pseudo_gaussians(xyz: torch.Tensor, pv: torch.Tensor,
                           dataset_dir: str, shift: torch.Tensor, scale: float,
                           rot_deg: float) -> GaussianModel:
    """GaussianModel AT the particles; DC color from nearest original gaussian.

    xyz: (N,3) cuda normalized (already rotated); pv: (N,) particle volumes.
    """
    orig = GaussianModel(3)
    orig.load_ply(os.path.join(dataset_dir, "point_cloud.ply"))
    oxyz = (orig._xyz.detach() + shift.cuda()) / scale  # to normalized space
    if rot_deg:
        oxyz = rot_z(oxyz, rot_deg)
    # nearest original gaussian per particle (chunked cdist)
    idx = []
    for i in range(0, xyz.shape[0], 4096):
        d = torch.cdist(xyz[i:i + 4096], oxyz)
        idx.append(d.argmin(dim=1))
    idx = torch.cat(idx)

    g = GaussianModel(0)
    n = xyz.shape[0]
    g._xyz = xyz.detach().clone()
    g._features_dc = orig._features_dc.detach()[idx].clone()        # (N,1,3)
    g._features_rest = torch.zeros((n, 0, 3), device="cuda")
    s = pv.cuda().clamp_min(1e-12) ** (1.0 / 3.0)                   # (N,)
    g._scaling = torch.log(s).unsqueeze(1).repeat(1, 3)
    rot = torch.zeros((n, 4), device="cuda")
    rot[:, 0] = 1.0
    g._rotation = rot
    g._opacity = inverse_sigmoid(0.9 * torch.ones((n, 1), device="cuda"))
    g.active_sh_degree = 0
    return g


def make_pose(view: str = "front") -> tuple:
    """Static camera pose (sim space). w2c rows = (right, down, forward).

    front : at (0.5,-1.4,0.5) looking +y -- y is the OPTICAL axis (depth
            motion ~invisible; +-y near sign-ambiguous in projection).
    side_x: at (-1.4,0.5,0.5) looking +x -- y becomes IN-PLANE (horizontal),
            for image-supervised y-motion identifiability.
    """
    if view == "front":
        R_w2c = np.array([[1.0, 0.0, 0.0],   # cam x = world +x
                          [0.0, 0.0, -1.0],  # cam y (down) = world -z
                          [0.0, 1.0, 0.0]])  # cam z (forward) = world +y
        C = np.array([0.5, -1.4, 0.5])
    elif view == "side_x":
        R_w2c = np.array([[0.0, 1.0, 0.0],   # cam x = world +y
                          [0.0, 0.0, -1.0],  # cam y (down) = world -z
                          [1.0, 0.0, 0.0]])  # cam z (forward) = world +x
        C = np.array([-1.4, 0.5, 0.5])
    elif view == "diag45":
        c45 = 1.0 / math.sqrt(2.0)           # looking along (+x+y)/sqrt2:
        R_w2c = np.array([[c45, -c45, 0.0],  # BOTH x and y are partly in-plane
                          [0.0, 0.0, -1.0],
                          [c45, c45, 0.0]])
        C = np.array([0.5 - 1.9 * c45, 0.5 - 1.9 * c45, 0.5])
    else:
        raise ValueError(view)
    T = -R_w2c @ C
    return R_w2c.T, T  # gic Camera stores R as the transpose convention


def make_camera(fid: int, image: torch.Tensor, alpha: np.ndarray,
                fov: float = 0.30, view: str = "front") -> Camera:
    R, T = make_pose(view)
    return Camera(colmap_id=0, R=R, T=T, FoVx=fov, FoVy=fov,
                  image=image, gt_alpha_mask=alpha, image_name=f"f{fid:03d}",
                  uid=fid, fid=fid)


def render_positions(positions: torch.Tensor, gaussians: GaussianModel,
                     pipe, pose_cam: Camera) -> tuple:
    """Render particle positions -> (image (3,H,W), alpha (1,H,W)), detached."""
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    with torch.no_grad():
        d_xyz = positions - gaussians.get_xyz
        out = render(pose_cam, gaussians, pipe, bg, d_xyz, 0.0, 0.0, False)
    return out["render"].detach(), out["alpha"].detach()


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--config", default="config/ours/telephone.json")
    ap.add_argument("--scene_cache",
                    default=f"{GEN}/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt")
    ap.add_argument("--rot_z_deg", default=67.6, type=float)
    ap.add_argument("--gt_logE", default=5.0, type=float)
    ap.add_argument("--gt_nu", default=0.3, type=float)
    ap.add_argument("--gt_vel", nargs=3, default=[0.0, -0.5, 0.0], type=float)
    ap.add_argument("--init_logE", default=4.0, type=float)
    ap.add_argument("--init_nu", default=0.1, type=float)
    ap.add_argument("--anchor_mass_scale", default=1e4, type=float)
    ap.add_argument("--mpm_iter_cnt", default=64, type=int)
    ap.add_argument("--n_frames", default=8, type=int,
                    help="FIT window (frames the loss sees)")
    ap.add_argument("--gt_frames", default=None, type=int,
                    help="GT rollout/render length (> n_frames => the tail is "
                         "held-out extrapolation, marked orange in gifs)")
    ap.add_argument("--mode", required=True,
                    choices=["fit", "landscape", "fit_v0field", "fit_v0scalar",
                             "fit_joint"])
    ap.add_argument("--warmup_iters", default=40, type=int,
                    help="fit_joint: v0-only warmup iters before the joint phase")
    ap.add_argument("--v0_field_res", default="4x4x16", type=str,
                    help="fit_v0field: grid res, int or 'RXxRYxRZ'")
    ap.add_argument("--gt_v0_variant", default=None, type=str,
                    help="GT = this analytic profile in a same-res grid (else uniform gt_vel)")
    ap.add_argument("--gt_v0_scale", default=1.0, type=float)
    ap.add_argument("--v0_field_lr", default=0.025, type=float)
    ap.add_argument("--v0_field_init_std", default=0.05, type=float)
    ap.add_argument("--v0_field_seed", default=0, type=int)
    ap.add_argument("--v0_field_tv", default=1e-3, type=float)
    ap.add_argument("--param_stop_tol", default=0.02, type=float)
    ap.add_argument("--camera", default="front", choices=["front", "side_x", "diag45"],
                    help="static camera pose; side_x puts y motion IN-PLANE")
    ap.add_argument("--cameras", default=None, type=str,
                    help="comma list of poses for MULTI-VIEW supervision (overrides "
                         "--camera), e.g. 'front,side_x'; estimator averages views")
    ap.add_argument("--w_img", default=1.0, type=float)
    ap.add_argument("--w_alp", default=0.0, type=float)
    ap.add_argument("--iter_cnt", default=None, type=int)
    ap.add_argument("--patience", default=16, type=int)
    ap.add_argument("--min_iters", default=25, type=int)
    ap.add_argument("--fine_lr", default=0.02, type=float)
    ap.add_argument("--grid_n", default=31, type=int)
    ap.add_argument("--logE_range", nargs=2, default=[4.0, 6.0], type=float)
    ap.add_argument("--tag", required=True)
    args = ap.parse_args()
    t0 = time.time()
    out_dir = os.path.join("output/ours_telephone", args.tag)
    os.makedirs(out_dir, exist_ok=True)

    phys_args = Namespace(**json.load(open(args.config))["physics"])
    phys_args.mpm_iter_cnt = args.mpm_iter_cnt
    gt_n = args.gt_frames if args.gt_frames is not None else args.n_frames
    assert gt_n >= args.n_frames, (gt_n, args.n_frames)
    phys_args.n_frames = gt_n
    phys_args.img_loss = True
    phys_args.w_img = args.w_img
    phys_args.w_alp = args.w_alp
    phys_args.w_geo = 0.0
    if args.iter_cnt is not None:
        phys_args.iter_cnt = args.iter_cnt

    # 3DGS pipeline/optimization params (defaults; only lambda_dssim is read)
    _p = ArgumentParser()
    pipe = PipelineParams(_p)
    iop = OptimizationParams(_p)
    _defaults = _p.parse_args([])
    pipe = pipe.extract(_defaults)
    iop = iop.extract(_defaults)

    xyz, anchor_mask = load_our_scene(args.scene_cache)
    if args.rot_z_deg:
        xyz = rot_z(xyz, args.rot_z_deg)
    cache = torch.load(args.scene_cache, map_location="cpu", weights_only=False)
    ghost = (cache["disc"]["sim_xyzs"] == 0).all(dim=1)
    pv = torch.from_numpy(cache["disc"]["points_vol"]).float()[~ghost]
    gaussians = build_pseudo_gaussians(
        xyz, pv, cache["dataset_dir"], cache["disc"]["shift"].reshape(-1),
        float(cache["disc"]["scale"]), args.rot_z_deg)
    print(f"[img] pseudo gaussians: {xyz.shape[0]} at particles, "
          f"colors from {cache['dataset_dir']}/point_cloud.ply")
    torch.cuda.empty_cache()  # release cdist chunks before taichi grabs its pool

    # 0.3 (not 0.5): taichi grabs this fraction UP FRONT; two parallel runs at
    # 0.5 OOM each other on a shared card. Our 7k-particle chunked sims fit easily.
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=0.3)
    dummy_gts = [xyz.clone() for _ in range(gt_n)]
    est = AnchoredEstimator(
        phys_args, "float32", dummy_gts, surface_index=None, init_vol=xyz,
        dynamic_scene=None, image_scale=1.0, pipeline=pipe, image_op=iop,
    )
    est.set_anchor(anchor_mask, args.anchor_mass_scale)
    est.set_pvol(pv)
    est.geo_loss = False

    # ---- GT rollout + rendered GT images ----
    # img branch must be OFF during pure rollouts: scene/views not set yet, and
    # render_forward would be called inside estimator.forward for f>0.
    est.img_loss = False
    set_params(est, args.gt_logE, args.gt_nu, list(args.gt_vel))
    pad = 2.0 * phys_args.voxel_size
    aabb = torch.stack([xyz.min(0).values - pad, xyz.max(0).values + pad])  # (2,3)
    z_lo, z_hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    flip_z = bool(xyz[anchor_mask][:, 2].mean() > xyz[~anchor_mask][:, 2].mean())
    field_res = (tuple(int(p) for p in args.v0_field_res.lower().split("x"))
                 if "x" in args.v0_field_res else (int(args.v0_field_res),) * 3)
    gt_part = None
    if args.gt_v0_variant is not None:
        from v0_field_ours import V0VoxelField, fill_profile_grid
        gt_field = V0VoxelField(aabb.cpu(), res=field_res)
        fill_profile_grid(gt_field, args.gt_v0_variant, args.gt_v0_scale,
                          z_lo, z_hi, flip=flip_z)
        est.set_v0_field(gt_field, xyz, lr=0.0)
        gt_part = (gt_field(xyz).detach()
                   * (~anchor_mask).float().unsqueeze(1)).cpu()       # (N,3)
        print(f"[img] GT v0 = '{args.gt_v0_variant}' x{args.gt_v0_scale} on "
              f"{field_res} grid; free |v0| mean "
              f"{gt_part[(~anchor_mask).cpu()].norm(dim=-1).mean():.3f}")
    gt_roll = rollout_collect_surfaces(est)  # list of (N,3) cuda
    # multi-view: estimator.render_forward already loops self.views[f] (a LIST per
    # frame) and averages the loss, so N cameras at the same fid = multi-view
    # supervision for free. --cameras overrides the single --camera.
    views = ([v.strip() for v in args.cameras.split(",")] if args.cameras
             else [args.camera])
    pose_cams = {v: make_camera(0, torch.zeros(3, 480, 480),
                                np.zeros((1, 480, 480), np.float32), view=v)
                 for v in views}
    pose_cam = pose_cams[views[0]]  # representative pose for pred side-by-side gif
    cams = []
    import imageio
    gt_frames_png = []       # views[0] only, for the pred comparison gif
    gt_frames_multi = []     # all views tiled, for gt_render.gif
    for f, pos in enumerate(gt_roll):
        row = []
        for vi, v in enumerate(views):
            img, alp = render_positions(pos, gaussians, pipe, pose_cams[v])
            cam = make_camera(f, img.cpu(), alp.cpu().numpy(), view=v)
            # gic's Camera multiplies original_image by gt_alpha_mask; our render is
            # ALREADY alpha-composited (color*a), so that double-multiplies the soft
            # edges (a vs a^2) and fakes a large RGB-loss floor. Overwrite with the
            # once-composited image.
            cam.original_image = img.clamp(0.0, 1.0).cuda()
            cams.append(cam)
            png = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            row.append(png)
            if vi == 0:
                gt_frames_png.append(png)
        gt_frames_multi.append(np.concatenate(row, axis=1))
    imageio.mimsave(os.path.join(out_dir, "gt_render.gif"), gt_frames_multi, fps=8)
    print(f"[img] GT rendered: {len(gt_roll)} frames x {len(views)} view(s) "
          f"{views} @480^2 -> gt_render.gif")
    est.set_scene(SceneShim(gaussians, cams))
    est.gts = dummy_gts
    est.load_gts(dummy_gts)  # buffers only; geo_loss stays False
    est.img_loss = True      # scene/views ready; enable the image branch

    if args.mode == "landscape":
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        logEs = np.linspace(*args.logE_range, args.grid_n)
        curves = {}
        for name, wi, wa in (("rgb", 1.0, 0.0), ("alpha", 0.0, 1.0)):
            est.w_img = torch.tensor(wi, device=est.device)
            est.w_alp = torch.tensor(wa, device=est.device)
            L = []
            for le in logEs:
                set_params(est, float(le), args.gt_nu, list(args.gt_vel))
                est.max_f = args.n_frames
                dt = est.simulator.dt_ori[None]
                for idx in range(args.n_frames):
                    if idx == 0:
                        est.initialize()  # also resets image_loss
                        est.simulator.set_dt(dt)
                    est.forward(idx, img_backward=False)
                L.append(float(est.image_loss) if est.succeed() else float("nan"))
                print(f"[img-ls {name}] logE={le:.3f} loss={L[-1]:.6g}")
            curves[name] = np.array(L)
        np.savez(os.path.join(out_dir, "imgloss_E1d.npz"), logEs=logEs, **curves)
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        for name, v in curves.items():
            ax.plot(logEs, v, "-o", ms=3, label=f"{name}-only")
            i = int(np.nanargmin(v))
            ax.axvline(logEs[i], ls=":", lw=0.8, color=ax.lines[-1].get_color())
        ax.axvline(args.gt_logE, color="k", ls="--", lw=1.2, label="GT")
        ax.set_yscale("log")
        ax.set_xlabel("log10 E")
        ax.set_ylabel("image loss")
        ax.set_title(f"image-loss(logE) | {args.tag} | {args.n_frames}f single static cam")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "imgloss_E1d.png"), dpi=130)
        print(f"[img] landscape done in {time.time() - t0:.0f}s -> {out_dir}")
        return

    if args.mode == "fit_joint":
        # ---- joint E + v0 (scalar) from image loss, with v0 warmup ----
        # Mirrors the traj-side hybrid: (1) warmup phase trains v0 alone at the
        # (possibly wrong) init E -- J1 showed cold-start joint lets E wander
        # decades while v0 is still ~0; (2) joint phase steps E (scheduled lr)
        # and v0 together; nu rides along frozen (lr 0).
        import torch.nn as nn
        from roundtrip_ours_scene import save_overlay_gif
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        set_params(est, args.init_logE, args.gt_nu, [0.0, 0.0, 0.0])
        est.set_stage(Estimator.physical_params_stage)
        # phase 1: v0 warmup at init E
        est.optimizer = torch.optim.Adam(
            [{"params": est.init_vel, "lr": args.v0_field_lr, "name": "velocity"}])
        saved_iter_cnt = phys_args.iter_cnt
        phys_args.iter_cnt = args.warmup_iters
        print(f"[img] JOINT phase1 warmup: v0 lr {args.v0_field_lr}, "
              f"E pinned at 1e{args.init_logE}, {args.warmup_iters} iters")
        l1, e1 = train_ours(
            est, phys_args, args.n_frames, out_dir, gts=None,
            patience=args.patience, min_iters=10, ckpt_every=10,
            overlay_every=0, fine_lr=0.0, param_stop_tol=args.param_stop_tol)
        phys_args.iter_cnt = saved_iter_cnt
        # phase 2: joint E + v0 (nu frozen; pop its scheduler or the lr-0 freeze
        # gets overwritten by update_learning_rate -- J1 lesson)
        e_lr = phys_args.params["Youngs modulus"]["init_lr"]
        est.optimizer = torch.optim.Adam([
            {"params": est.E, "lr": e_lr, "name": "Youngs modulus"},
            {"params": est.nu, "lr": 0.0, "name": "Poisson ratio"},
            {"params": est.init_vel, "lr": args.v0_field_lr, "name": "velocity"},
        ])
        est.lr_schedulers.pop("Poisson ratio", None)
        print(f"[img] JOINT phase2: E lr {e_lr} (scheduled) + v0 lr {args.v0_field_lr}")
        l2, e2 = train_ours(
            est, phys_args, args.n_frames, out_dir, gts=None,
            patience=args.patience, min_iters=args.min_iters, ckpt_every=10,
            overlay_every=0, fine_lr=args.fine_lr, param_stop_tol=args.param_stop_tol)

        losses = [float(x) for x in (l1 + l2)]
        E_traj = [10.0 ** args.init_logE] * len(l1) + [d["Youngs modulus"] for d in e2]
        v0_traj = ([[float(x) for x in d["velocity"]] for d in e1]
                   + [[float(x) for x in d["velocity"]] for d in e2])
        # best over the JOINT phase only (warmup loss isn't comparable: E wrong)
        best2 = l2.index(min(l2))
        best_E = e2[best2]["Youngs modulus"]
        v0_best = [float(x) for x in e2[best2]["velocity"]]
        gt_E = 10.0 ** args.gt_logE
        rel_E = (best_E - gt_E) / gt_E
        gt_v = np.array(list(args.gt_vel))
        rel_v = float(np.linalg.norm(np.array(v0_best) - gt_v)
                      / max(np.linalg.norm(gt_v), 1e-12))

        est.img_loss = False
        est.E.data.copy_(torch.log10(torch.tensor(best_E, device=est.device)))
        est.init_vel = nn.Parameter(torch.tensor(v0_best, device=est.device))
        est.max_f = gt_n
        pred_roll = rollout_collect_surfaces(est)
        est.set_stage(Estimator.physical_params_stage)
        sbs = []
        for f, pos in enumerate(pred_roll):
            img, _ = render_positions(pos, gaussians, pipe, pose_cam)
            pred_png = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            frame = np.concatenate([gt_frames_png[f], pred_png], axis=1)
            if f >= args.n_frames:
                frame = frame.copy()
                frame[:6], frame[-6:] = (255, 140, 0), (255, 140, 0)
                frame[:, :6], frame[:, -6:] = (255, 140, 0), (255, 140, 0)
            sbs.append(frame)
        imageio.mimsave(os.path.join(out_dir, "gt_vs_pred_render.gif"), sbs, fps=8)
        save_overlay_gif(gt_roll, pred_roll, os.path.join(out_dir, "overlay.gif"),
                         fit_frames=args.n_frames)

        result = {
            "scenario": "ours_image_loss_joint_v0E",
            "gt": {"E": gt_E, "logE": args.gt_logE, "nu": args.gt_nu,
                   "vel": list(args.gt_vel)},
            "init": {"logE": args.init_logE},
            "w_img": args.w_img, "w_alp": args.w_alp, "n_frames": args.n_frames,
            "camera": (args.cameras or args.camera),
            "warmup_iters": len(l1),
            "losses_phys": losses,
            "alt_phys_bounds": [len(l1), len(l1) + len(l2)],  # warmup|joint marker
            "E_traj": E_traj,
            "v0_traj": v0_traj,
            "best": {"Youngs modulus": best_E, "iter": len(l1) + best2,
                     "loss": float(l2[best2])},
            "rel_err_E": rel_E,
            "v0_estimated": v0_best,
            "v0_rel_err": rel_v,
            "wall_time_s": time.time() - t0,
        }
        with open(os.path.join(out_dir, "result.json"), "w") as fj:
            json.dump(result, fj, indent=2)
        from utils.system_utils import draw_curve
        draw_curve(losses, out_dir, name="loss_phys")
        fig, ax = plt.subplots(figsize=(6, 4))
        arr = np.array(v0_traj)
        for k, c in enumerate("xyz"):
            ax.plot(arr[:, k], color=f"C{k}", label=f"v0_{c}")
            ax.axhline(gt_v[k], color=f"C{k}", ls="--", alpha=0.5)
        ax.axvline(len(l1) - 0.5, color="k", ls=":", lw=1, label="warmup|joint")
        ax.set_xlabel("iter")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "v0_traj.png"))
        print(f"[img] DONE(joint) tag={args.tag} E {best_E:.4g} ({rel_E:+.2%}) "
              f"v0 {[round(x, 3) for x in v0_best]} (rel {rel_v:.2%}), "
              f"wall {time.time() - t0:.0f}s")
        return

    if args.mode == "fit_v0scalar":
        # ---- fix-E, learn SCALAR v0 (3 DOF) from image loss ----
        # The disentangling baseline for fit_v0field: if 3 DOF already fails for
        # a given (direction, camera), the failure is loss identifiability, not
        # field parameterization. Same phys-stage trick, optimizer = init_vel.
        import torch.nn as nn
        from roundtrip_ours_scene import save_overlay_gif
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        set_params(est, args.gt_logE, args.gt_nu, [0.0, 0.0, 0.0])  # rest init
        est.optimizer = torch.optim.Adam(
            [{"params": est.init_vel, "lr": args.v0_field_lr, "name": "velocity"}])
        est.set_stage(Estimator.physical_params_stage)
        print(f"[img] v0-SCALAR from image loss: lr {args.v0_field_lr}, "
              f"camera {args.camera}, gt_vel {list(args.gt_vel)}")
        losses, e_s = train_ours(
            est, phys_args, args.n_frames, out_dir, gts=None,
            patience=args.patience, min_iters=args.min_iters, ckpt_every=10,
            overlay_every=0, fine_lr=0.0, param_stop_tol=args.param_stop_tol)
        best_idx = losses.index(min(losses))
        v0_traj = [[float(x) for x in d["velocity"]] for d in e_s]
        v0_best = v0_traj[best_idx]
        gt_v = np.array(list(args.gt_vel))
        rel = float(np.linalg.norm(np.array(v0_best) - gt_v)
                    / max(np.linalg.norm(gt_v), 1e-12))

        est.img_loss = False
        est.init_vel = nn.Parameter(torch.tensor(v0_best, device=est.device))
        est.max_f = gt_n
        pred_roll = rollout_collect_surfaces(est)
        est.set_stage(Estimator.physical_params_stage)
        sbs = []
        for f, pos in enumerate(pred_roll):
            img, _ = render_positions(pos, gaussians, pipe, pose_cam)
            pred_png = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            frame = np.concatenate([gt_frames_png[f], pred_png], axis=1)
            if f >= args.n_frames:  # held-out extrapolation: orange border
                frame = frame.copy()
                frame[:6], frame[-6:] = (255, 140, 0), (255, 140, 0)
                frame[:, :6], frame[:, -6:] = (255, 140, 0), (255, 140, 0)
            sbs.append(frame)
        imageio.mimsave(os.path.join(out_dir, "gt_vs_pred_render.gif"), sbs, fps=8)
        save_overlay_gif(gt_roll, pred_roll, os.path.join(out_dir, "overlay.gif"),
                         fit_frames=args.n_frames)

        result = {
            "scenario": "ours_image_loss_learn_v0scalar",
            "gt": {"E": 10.0 ** args.gt_logE, "logE": args.gt_logE, "nu": args.gt_nu,
                   "vel": list(args.gt_vel)},
            "w_img": args.w_img, "w_alp": args.w_alp, "n_frames": args.n_frames,
            "camera": (args.cameras or args.camera),
            "losses_phys": [float(l) for l in losses],
            "v0_traj": v0_traj,
            "v0_estimated": v0_best,
            "v0_rel_err": rel,
            "wall_time_s": time.time() - t0,
        }
        with open(os.path.join(out_dir, "result.json"), "w") as fj:
            json.dump(result, fj, indent=2)
        from utils.system_utils import draw_curve
        draw_curve([float(l) for l in losses], out_dir, name="loss_phys")
        fig, ax = plt.subplots(figsize=(6, 4))
        arr = np.array(v0_traj)
        for k, c in enumerate("xyz"):
            ax.plot(arr[:, k], color=f"C{k}", label=f"v0_{c}")
            ax.axhline(gt_v[k], color=f"C{k}", ls="--", alpha=0.5)
        ax.set_xlabel("iter")
        ax.legend()
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "v0_traj.png"))
        print(f"[img] DONE(v0scalar) tag={args.tag} cam={args.camera} "
              f"GT {list(args.gt_vel)} -> {[round(x, 3) for x in v0_best]} "
              f"(rel err {rel:.2%}), wall {time.time() - t0:.0f}s")
        return

    if args.mode == "fit_v0field":
        # ---- fix-E, learn v0 FIELD from image loss ----
        from v0_field_ours import V0VoxelField, eval_grid_at
        from roundtrip_ours_scene import (plot_field_projections, plot_grid_nodes,
                                          plot_profile_1d, save_overlay_gif)
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fit_field = V0VoxelField(aabb.cpu(), res=field_res)
        fit_field.randomize_(args.v0_field_init_std, seed=args.v0_field_seed)
        n_starved = fit_field.freeze_starved_(xyz[~anchor_mask].cpu(), 1.0)
        est.set_v0_field(fit_field, xyz, lr=args.v0_field_lr)  # builds vel_optimizer
        est.optimizer = est.vel_optimizer  # phys stage steps the FIELD; E/nu frozen
        set_params(est, args.gt_logE, args.gt_nu, [0.0, 0.0, 0.0])
        est.set_stage(Estimator.physical_params_stage)
        n_nodes = field_res[0] * field_res[1] * field_res[2]
        print(f"[img] v0-FIELD from image loss: res={field_res} ({3 * n_nodes} DOF), "
              f"starved {n_starved}/{n_nodes}, lr {args.v0_field_lr}, "
              f"tv {args.v0_field_tv}, init std {args.v0_field_init_std}")
        losses, e_s = train_ours(
            est, phys_args, args.n_frames, out_dir, gts=None,
            patience=args.patience, min_iters=args.min_iters, ckpt_every=10,
            overlay_every=0, fine_lr=0.0, param_stop_tol=args.param_stop_tol,
            tv_weight=args.v0_field_tv)

        # metrics vs the GT field (per-particle, free only)
        fm = (~anchor_mask).cpu()
        xyz_c = xyz.cpu()
        if gt_part is not None:
            gt_pf = gt_part[fm].float()                                # (M,3)
        else:
            gt_pf = torch.tensor(list(args.gt_vel), dtype=torch.float32
                                 ).expand(int(fm.sum()), 3)
        gt_scale = max(float(gt_pf.norm(dim=-1).mean()), 1e-12)
        per_iter_v = [eval_grid_at(d["velocity"].float(), fit_field.aabb.cpu(),
                                   xyz_c)[fm] for d in e_s]
        rel_traj = [float((v - gt_pf).norm(dim=-1).mean() / gt_scale)
                    for v in per_iter_v]
        rel_xy_traj = [float((v - gt_pf)[:, :2].norm(dim=-1).mean() / gt_scale)
                       for v in per_iter_v]
        best_idx = losses.index(min(losses))
        v_best = per_iter_v[best_idx]
        per_err = (v_best - gt_pf).norm(dim=-1)
        ax_err_best = (v_best - gt_pf).abs().mean(0)               # (3,)
        rel_xy = rel_xy_traj[best_idx]

        # pred rollout at best field (train_ours restored it), rendered vs GT
        est.img_loss = False
        est.max_f = gt_n
        pred_roll = rollout_collect_surfaces(est)
        est.set_stage(Estimator.physical_params_stage)
        sbs = []
        for f, pos in enumerate(pred_roll):
            img, _ = render_positions(pos, gaussians, pipe, pose_cam)
            pred_png = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
            frame = np.concatenate([gt_frames_png[f], pred_png], axis=1)
            if f >= args.n_frames:  # held-out extrapolation: orange border
                frame = frame.copy()
                frame[:6], frame[-6:] = (255, 140, 0), (255, 140, 0)
                frame[:, :6], frame[:, -6:] = (255, 140, 0), (255, 140, 0)
            sbs.append(frame)
        imageio.mimsave(os.path.join(out_dir, "gt_vs_pred_render.gif"), sbs, fps=8)
        save_overlay_gif(gt_roll, pred_roll, os.path.join(out_dir, "overlay.gif"),
                         fit_frames=args.n_frames)

        result = {
            "scenario": "ours_image_loss_learn_v0field",
            "gt": {"E": 10.0 ** args.gt_logE, "logE": args.gt_logE, "nu": args.gt_nu,
                   "vel": list(args.gt_vel), "v0_variant": args.gt_v0_variant,
                   "v0_scale": args.gt_v0_scale,
                   "v0_mean_vec": gt_pf.mean(0).tolist()},
            "w_img": args.w_img, "w_alp": args.w_alp, "n_frames": args.n_frames,
            "camera": (args.cameras or args.camera),
            "losses_phys": [float(l) for l in losses],
            "v0_traj": [v.mean(0).tolist() for v in per_iter_v],
            "v0_traj_p05": [v.quantile(0.05, dim=0).tolist() for v in per_iter_v],
            "v0_traj_p95": [v.quantile(0.95, dim=0).tolist() for v in per_iter_v],
            "v0_rel_err": float(np.linalg.norm(v_best.mean(0).numpy()
                                               - gt_pf.mean(0).numpy())
                                / max(float(gt_pf.mean(0).norm()), 1e-12)),
            "v0_estimated": v_best.mean(0).tolist(),
            "v0_field": {
                "res": list(field_res), "init_std": args.v0_field_init_std,
                "seed": args.v0_field_seed, "lr": args.v0_field_lr,
                "tv_weight": args.v0_field_tv, "n_starved_frozen": n_starved,
                "per_axis_err_best": ax_err_best.tolist(),
                "rel_l2_xy_best": rel_xy, "rel_l2_xy_traj": rel_xy_traj,
                "rel_l2_best": rel_traj[best_idx], "rel_l2_traj": rel_traj,
                "mean_vec_best": v_best.mean(0).tolist(),
                "per_axis_std_best": v_best.std(0).tolist(),
                "per_particle_err_max": float(per_err.max()) / gt_scale,
                "per_particle_err_p95": float(per_err.quantile(0.95)) / gt_scale,
            },
            "wall_time_s": time.time() - t0,
        }
        with open(os.path.join(out_dir, "result.json"), "w") as fj:
            json.dump(result, fj, indent=2)
        from utils.system_utils import draw_curve
        draw_curve([float(l) for l in losses], out_dir, name="loss_phys")
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rel_traj, color="0.7", lw=1.0,
                label="all axes (incl. UNOBSERVABLE — cross-compare only)")
        ax.plot(rel_xy_traj, color="tab:red", label="xy-only (observable)")
        ax.legend(fontsize=7)
        ax.set_yscale("log")
        ax.set_xlabel("iter")
        ax.set_ylabel("field rel L2 (image-supervised)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "field_err.png"))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.hist((per_err / gt_scale).numpy(), bins=60)
        ax.set_xlabel("per-particle |v0 - gt| / mean|gt| (best iter)")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "field_err_hist.png"))
        plt.close(fig)
        plot_field_projections(xyz_c[fm], v_best, gt_pf,
                               os.path.join(out_dir, "field_proj.png"),
                               aabb=fit_field.aabb.cpu(), res=field_res,
                               xyz_anchor=xyz_c[~fm])
        gt_nodes = (gt_field.grid.data.detach().cpu() if gt_part is not None
                    else torch.tensor(list(args.gt_vel), dtype=torch.float32))
        plot_grid_nodes(e_s[best_idx]["velocity"].float(), fit_field.aabb.cpu(),
                        gt_nodes, xyz_c[fm],
                        os.path.join(out_dir, "grid_quiver.png"),
                        os.path.join(out_dir, "grid_hist.png"))
        if gt_part is not None:
            zt_f = ((xyz_c[fm][:, 2] - z_lo) / (z_hi - z_lo + 1e-8)).clamp(0, 1)
            if flip_z:
                zt_f = 1.0 - zt_f
            plot_profile_1d(zt_f, v_best, gt_pf,
                            os.path.join(out_dir, "profile_1d.png"))
        print(f"[img] DONE(v0field) tag={args.tag} per-axis |err| "
              f"x {ax_err_best[0]:.3f} y {ax_err_best[1]:.3f} z {ax_err_best[2]:.3f} | "
              f"xy relL2 {rel_xy:.2%} (all-axes {rel_traj[best_idx]:.2%} — inflated by "
              f"unobservable axes; judge by visuals), wall {time.time() - t0:.0f}s")
        return

    # ---- mode == fit: fix-v0, learn E(+nu) from image loss ----
    set_params(est, args.init_logE, args.init_nu, list(args.gt_vel))
    est.set_stage(Estimator.physical_params_stage)
    losses, e_s = train_ours(
        est, phys_args, args.n_frames, out_dir, gts=None,
        patience=args.patience, min_iters=args.min_iters,
        ckpt_every=10, overlay_every=0, fine_lr=args.fine_lr)
    min_idx = losses.index(min(losses))
    best = e_s[min_idx]
    gt_E = 10.0 ** args.gt_logE
    rel = abs(best["Youngs modulus"] - gt_E) / gt_E

    # pred rollout at best params, rendered side-by-side vs GT
    est.img_loss = False  # pure rollout again
    set_params(est, math.log10(best["Youngs modulus"]), best["Poisson ratio"],
               list(args.gt_vel))
    est.max_f = args.n_frames
    pred_roll = rollout_collect_surfaces(est)
    est.set_stage(Estimator.physical_params_stage)
    sbs = []
    for f, pos in enumerate(pred_roll):
        img, _ = render_positions(pos, gaussians, pipe, pose_cam)
        pred_png = (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
        sbs.append(np.concatenate([gt_frames_png[f], pred_png], axis=1))
    imageio.mimsave(os.path.join(out_dir, "gt_vs_pred_render.gif"), sbs, fps=8)

    result = {
        "scenario": "ours_telephone_image_loss_roundtrip",
        "gt": {"E": gt_E, "logE": args.gt_logE, "nu": args.gt_nu, "vel": list(args.gt_vel)},
        "init": {"logE": args.init_logE, "nu": args.init_nu},
        "w_img": args.w_img, "w_alp": args.w_alp, "n_frames": args.n_frames,
        "camera": (args.cameras or args.camera),
        "best": {**best, "iter": min_idx, "loss": float(losses[min_idx])},
        "final": e_s[-1],
        "rel_err_E": rel,
        "losses_phys": [float(l) for l in losses],
        "E_traj": [d.get("Youngs modulus") for d in e_s],
        "nu_traj": [d.get("Poisson ratio") for d in e_s],
        "wall_time_s": time.time() - t0,
    }
    with open(os.path.join(out_dir, "result.json"), "w") as fj:
        json.dump(result, fj, indent=2)
    from utils.system_utils import draw_curve
    draw_curve([float(l) for l in losses], out_dir, name="loss_phys")
    print(f"[img] DONE tag={args.tag} w_img={args.w_img} w_alp={args.w_alp} "
          f"GT E={gt_E:.3g} -> best {best['Youngs modulus']:.4g} "
          f"(rel err {rel:.2%}), wall {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

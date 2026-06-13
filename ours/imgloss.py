# coding=utf-8
"""Shared image-loss machinery for the fit_image_* entrypoints.

Extracted verbatim from the old image_fit_ours omni-entrypoint so the three
optimization-axis entrypoints (fit_image_Escalar / _v0 / _joint) share ONE copy
of: the pseudo-GaussianModel build (gic renders particles AS gaussians 1:1, so
we put a gaussian at every MPM particle with colour transferred from the nearest
original PhysDreamer gaussian), the static camera set, the GT rollout+render, and
the AnchoredEstimator image-branch wiring.

Background (RenderCfg.bg_image): None keeps the old black-bg / obj-only masked
loss (de-backgrounded demos). When set, the SAME static bg is composited under
both the GT frames (here) and the pred render (estimator.render_forward reads
est.bg_image) so the loss sees full-RGB frames -- the realistic setting where
only with-bg video exists. 'scene' renders the original gaussians (moving-object
region removed) as the bg; anything else is treated as a 3xHxW image file.
"""
from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Optional

import numpy as np
import torch

from arguments import OptimizationParams, PipelineParams
from gaussian_renderer import render
from scene.cameras import Camera
from scene.gaussian_model import GaussianModel
from utils.general_utils import inverse_sigmoid

from ours.estimator import AnchoredEstimator
from ours.geom import rot_xyz as rot_z
from ours.scene import load_our_scene, rollout_collect_surfaces, set_params

if TYPE_CHECKING:
    from argparse import Namespace

    from ours.config import GTCfg, RenderCfg, SceneCfg

HW = 480  # render resolution (square)


class SceneShim:
    """Duck-typed stand-in for gic's Scene: just gaussians + camera list."""

    def __init__(self, gaussians: GaussianModel, cams: list) -> None:
        self.gaussians = gaussians
        self._cams = cams

    def getTrainCameras(self, scale: float = 1.0) -> list:
        return self._cams


def _load_orig(dataset_dir: str) -> GaussianModel:
    orig = GaussianModel(3)
    orig.load_ply(os.path.join(dataset_dir, "point_cloud.ply"))
    return orig


def _orig_xyz_norm(orig: GaussianModel, shift: torch.Tensor, scale: float,
                   rot_deg: float) -> torch.Tensor:
    """Original gaussian centres mapped into our normalized+rotated sim space."""
    oxyz = (orig._xyz.detach() + shift.cuda()) / scale
    return rot_z(oxyz, rot_deg) if rot_deg else oxyz


def build_pseudo_gaussians(xyz: torch.Tensor, pv: torch.Tensor, orig: GaussianModel,
                           oxyz: torch.Tensor) -> tuple:
    """GaussianModel AT the particles; DC colour from nearest original gaussian.

    xyz: (N,3) cuda normalized (already rotated); pv: (N,) particle volumes;
    oxyz: (M,3) original gaussian centres in the SAME normalized space.
    Returns (gaussians, idx) where idx[(N,)] = nearest original gaussian per
    particle (reused to carve the moving object out of the 'scene' background).
    """
    idx = []
    for i in range(0, xyz.shape[0], 1024):  # 1024x(~6e5) cdist ~2.4GB/chunk (busy-GPU safe)
        d = torch.cdist(xyz[i:i + 1024], oxyz)
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
    return g, idx


def _rotz_quat(quats: torch.Tensor, deg: float) -> torch.Tensor:
    """Left-multiply each [w,x,y,z] quaternion by a rotation of `deg` about +z.

    The similarity that maps raw->sim space rotates positions about z by rot_z_deg
    (ours.geom.rot_xyz); the gaussian ORIENTATIONS must rotate by the same amount
    or every anisotropic splat points 68 deg wrong. qz = (cos(t/2),0,0,sin(t/2)).
    """
    t = math.radians(deg)
    c, s = math.cos(t / 2.0), math.sin(t / 2.0)
    w, x, y, z = quats[:, 0], quats[:, 1], quats[:, 2], quats[:, 3]
    return torch.stack([c * w - s * z, c * x - s * y, c * y + s * x, c * z + s * w], dim=1)


def _build_bg_gaussians(orig: GaussianModel, oxyz: torch.Tensor, scale: float,
                        sim_mask: torch.Tensor, rot_deg: float) -> GaussianModel:
    """Original scene gaussians in normalized+rotated space, OBJECT removed.

    Centres are the pre-rotated oxyz=(raw+shift)/scale; scales divide by the
    scalar `scale`; orientations rotate by rot_deg about z (matches the position
    rotation). sim_mask = the cache's object mask over the ORIGINAL gaussians
    (75k for telephone, not just the ~7k particle nearest-neighbours) -- those are
    zeroed so the static object doesn't ghost behind the simulated one.
    """
    g = GaussianModel(orig.max_sh_degree)
    g._xyz = oxyz.detach().clone()
    g._features_dc = orig._features_dc.detach().clone()
    g._features_rest = orig._features_rest.detach().clone()
    g._scaling = (orig._scaling.detach() - math.log(scale)).clone()
    rot = orig._rotation.detach().clone()
    g._rotation = _rotz_quat(rot, rot_deg) if rot_deg else rot
    op = orig._opacity.detach().clone()
    op[sim_mask] = inverse_sigmoid(torch.tensor(1e-6, device=op.device))
    g._opacity = op
    g.active_sh_degree = orig.active_sh_degree
    return g


def build_fullres_gaussians(orig: GaussianModel, oxyz: torch.Tensor, scale: float,
                            rot_deg: float) -> GaussianModel:
    """The FULL original scene in normalized+rotated sim space, nothing removed.

    Same similarity as the object cache ((raw+shift)/scale, rot_z) applied to ALL
    gaussians with their real anisotropic scale / rotation / SH kept. The object
    subset (sim_mask) is driven by the particles via TopKDrive; the rest render
    static (the realistic bg + non-simulated foreground). Renders the actual phone
    texture instead of isotropic blobs -- this is PhysDreamer's render input.
    """
    g = GaussianModel(orig.max_sh_degree)
    g._xyz = oxyz.detach().clone()
    g._features_dc = orig._features_dc.detach().clone()
    g._features_rest = orig._features_rest.detach().clone()
    g._scaling = (orig._scaling.detach() - math.log(scale)).clone()
    rot = orig._rotation.detach().clone()
    g._rotation = _rotz_quat(rot, rot_deg) if rot_deg else rot
    g._opacity = orig._opacity.detach().clone()
    g.active_sh_degree = orig.active_sh_degree
    return g


def render_drive_frame(gaussians: GaussianModel, drive, particle_xyz: torch.Tensor,
                       pipe, cam: Camera, bg_image: Optional[torch.Tensor] = None) -> tuple:
    """No-grad full-res render at one particle frame (GT frames / pred viz).

    Drives the object gaussians from the particle positions (TopKDrive), renders
    the full scene with anisotropic covariance (compute_cov3D_python). Mirrors what
    estimator.render_forward does on the gradient path so GT and pred use the same
    renderer. bg_image optionally composites over a static bg (else the scene's own
    static gaussians ARE the background).
    """
    pipe.compute_cov3D_python = True
    bg = torch.zeros(3, device="cuda")
    with torch.no_grad():
        d_xyz, rot_full = drive(particle_xyz - drive.p0)
        gaussians._rotation = rot_full
        out = render(cam, gaussians, pipe, bg, d_xyz, 0.0, 0.0, False)
    img, alpha = out["render"].detach(), out["alpha"].detach()
    if bg_image is not None:
        img = img + (1.0 - alpha) * bg_image
    return img, alpha


def make_pose(view: str = "front", pan: float = 0.0) -> tuple:
    """Static camera pose (sim space). w2c rows = (right, down, forward).

    pan>0 slides the camera centre along its right axis (world dir = w2c row 0),
    panning the framing right -- used to push 3DGS artefacts near the object out
    of a narrow fov without changing the look direction."""
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
    if pan:
        C = C + pan * R_w2c[0]  # slide along cam-right (world) axis
    T = -R_w2c @ C
    return R_w2c.T, T  # gic Camera stores R as the transpose convention


def make_camera(fid: int, image: torch.Tensor, alpha: np.ndarray,
                fov: float = 0.30, view: str = "front", pan: float = 0.0) -> Camera:
    R, T = make_pose(view, pan)
    return Camera(colmap_id=0, R=R, T=T, FoVx=fov, FoVy=fov,
                  image=image, gt_alpha_mask=alpha, image_name=f"f{fid:03d}",
                  uid=fid, fid=fid)


def render_positions(positions: torch.Tensor, gaussians: GaussianModel, pipe,
                     pose_cam: Camera, bg_image: Optional[torch.Tensor] = None) -> tuple:
    """Render particle positions -> (image (3,H,W), alpha (1,H,W)), detached.

    bg_image (3,H,W cuda, optional): composite the object over this static bg
    (same operation render_forward does for the pred during training).
    """
    bg = torch.tensor([0.0, 0.0, 0.0], device="cuda")
    with torch.no_grad():
        d_xyz = positions - gaussians.get_xyz
        out = render(pose_cam, gaussians, pipe, bg, d_xyz, 0.0, 0.0, False)
    img, alpha = out["render"].detach(), out["alpha"].detach()
    if bg_image is not None:
        img = img + (1.0 - alpha) * bg_image
    return img, alpha


def _png(img: torch.Tensor) -> np.ndarray:
    return (img.clamp(0, 1).cpu().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)


def gt_pred_diff_gif(gt_pngs: list, pred_pngs: list, path: str, fit_frames: int,
                     fps: int = 8) -> None:
    """3-column gt | pred | (gt-pred) gif; diff is +128-centred (grey=match).

    Image-loss runs can't be overlaid (RGB textures), so we show GT, prediction,
    and a centred difference side by side. Frames beyond fit_frames (held-out
    extrapolation) get an orange border.
    """
    import imageio
    frames = []
    for f, (g, p) in enumerate(zip(gt_pngs, pred_pngs)):
        diff = np.clip(128.0 + g.astype(np.int16) - p.astype(np.int16), 0, 255).astype(np.uint8)
        frame = np.concatenate([g, p, diff], axis=1)
        if f >= fit_frames:
            frame = frame.copy()
            frame[:6], frame[-6:] = (255, 140, 0), (255, 140, 0)
            frame[:, :6], frame[:, -6:] = (255, 140, 0), (255, 140, 0)
        frames.append(frame)
    imageio.mimsave(path, frames, fps=fps)


def build_panel(run_root: str) -> None:
    """Per-run panel.gif fixture (CPU): tiles the gt|pred|diff animation with the
    loss / E / v0 curves from result.json. The auto-fixture every fit_image_* run
    should leave behind, mirroring the traj side (make_panel reads saved data, no
    re-simulation)."""
    import os
    import subprocess
    import sys
    gic_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    subprocess.run([sys.executable, "make_panel.py", "--per_run", "--runs", run_root],
                   cwd=gic_root)


def setup_image_scene(scene: "SceneCfg", gt: "GTCfg", render_cfg: "RenderCfg",
                      phys_args: "Namespace", gt_n: int, *,
                      gt_v0_variant: Optional[str] = None, gt_v0_scale: float = 1.0,
                      v0_field_res: Optional[tuple] = None) -> dict:
    """Build pseudo gaussians + cameras, roll out + render the GT, wire the est.

    Mode-independent setup shared by every fit_image_* entrypoint. The GT is a
    scalar-E GIC self-sim (gt.logE/nu) with a scalar (gt.vel) or field
    (gt_v0_variant) initial velocity. Returns a dict with the estimator (image
    branch enabled, bg_image set if requested), the gaussians, the static
    GT-frame pngs for the first view, and the geometry needed by field metrics.
    """
    import taichi as ti

    phys_args.n_frames = gt_n
    phys_args.img_loss = True
    phys_args.w_img = render_cfg.w_img
    phys_args.w_alp = render_cfg.w_alp
    phys_args.w_geo = 0.0

    # 3DGS pipeline/optimization params (defaults; only lambda_dssim is read)
    from argparse import ArgumentParser
    _p = ArgumentParser()
    pipe = PipelineParams(_p).extract(_p.parse_args([]))
    iop = OptimizationParams(_p).extract(_p.parse_args([]))

    xyz, anchor_mask = load_our_scene(scene.cache)
    if scene.rot_z_deg:
        xyz = rot_z(xyz, scene.rot_z_deg)
    cache = torch.load(scene.cache, map_location="cpu", weights_only=False)
    ghost = (cache["disc"]["sim_xyzs"] == 0).all(dim=1)
    pv = torch.from_numpy(cache["disc"]["points_vol"]).float()[~ghost]
    shift = cache["disc"]["shift"].reshape(-1)
    cscale = float(cache["disc"]["scale"])
    orig = _load_orig(cache["dataset_dir"])
    oxyz = _orig_xyz_norm(orig, shift, cscale, scene.rot_z_deg)
    fullres = getattr(render_cfg, "gaussians", "pseudo") == "fullres"
    drive = None
    if fullres:
        from ours.gauss_drive import TopKDrive
        gaussians = build_fullres_gaussians(orig, oxyz, cscale, scene.rot_z_deg)
        sim_mask = cache["disc"]["sim_mask"].cuda()
        top_k_index = cache["disc"]["top_k_index"].cuda()
        drive = TopKDrive(gaussians.get_xyz.detach(), gaussians.get_rotation.detach(),
                          xyz.detach(), top_k_index, sim_mask)
        print(f"[img] FULL-RES gaussians: {gaussians.get_xyz.shape[0]} real splats, "
              f"object={int(sim_mask.sum())} driven by top_{top_k_index.shape[1]} particles")
    else:
        gaussians, idx = build_pseudo_gaussians(xyz, pv, orig, oxyz)
        print(f"[img] pseudo gaussians: {xyz.shape[0]} blobs at particles, "
              f"colours from {cache['dataset_dir']}/point_cloud.ply")
    torch.cuda.empty_cache()  # release cdist chunks before taichi grabs its pool

    ti.init(arch=ti.cuda, debug=False, fast_math=False,
            device_memory_fraction=scene.ti_mem_frac)
    dummy_gts = [xyz.clone() for _ in range(gt_n)]
    est = AnchoredEstimator(phys_args, "float32", dummy_gts, surface_index=None,
                            init_vol=xyz, dynamic_scene=None, image_scale=1.0,
                            pipeline=pipe, image_op=iop)
    est.set_anchor(anchor_mask, scene.anchor_mass_scale)
    est.set_pvol(pv)
    est.geo_loss = False

    # ---- GT rollout (img branch OFF: scene/views not ready) ----
    est.img_loss = False
    set_params(est, gt.logE, gt.nu, list(gt.vel))
    pad = 2.0 * phys_args.voxel_size
    aabb = torch.stack([xyz.min(0).values - pad, xyz.max(0).values + pad])  # (2,3)
    z_lo, z_hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    flip_z = bool(xyz[anchor_mask][:, 2].mean() > xyz[~anchor_mask][:, 2].mean())
    gt_part, gt_v0_grid = None, None
    if gt_v0_variant is not None:
        from ours.fields import V0VoxelField, fill_profile_grid
        gt_field = V0VoxelField(aabb.cpu(), res=v0_field_res)
        fill_profile_grid(gt_field, gt_v0_variant, gt_v0_scale, z_lo, z_hi, flip=flip_z)
        est.set_v0_field(gt_field, xyz, lr=0.0)
        gt_v0_grid = gt_field.grid.data.detach().cpu()
        gt_part = (gt_field(xyz).detach() * (~anchor_mask).float().unsqueeze(1)).cpu()
        print(f"[img] GT v0 = '{gt_v0_variant}' x{gt_v0_scale} on {v0_field_res} grid; "
              f"free |v0| mean {gt_part[(~anchor_mask).cpu()].norm(dim=-1).mean():.3f}")
    gt_roll = rollout_collect_surfaces(est)

    # ---- static background (pseudo path only; fullres has intrinsic bg) ----
    bg_image = None
    if render_cfg.bg_image is not None and not fullres:
        if render_cfg.bg_image == "scene":
            sim_mask = cache["disc"]["sim_mask"].to(oxyz.device)
            bg_g = _build_bg_gaussians(orig, oxyz, cscale, sim_mask, scene.rot_z_deg)
        else:
            bg_g = None  # external image path, loaded below
        if bg_g is None:
            import imageio
            raw = imageio.imread(render_cfg.bg_image)
            t = torch.from_numpy(raw[..., :3]).float().permute(2, 0, 1) / 255.0
            bg_image = torch.nn.functional.interpolate(
                t.unsqueeze(0), size=(HW, HW), mode="bilinear", align_corners=False
            )[0].cuda()
        print(f"[img] bg mode: '{render_cfg.bg_image}' -> full-RGB loss (w_alp forced 0)")

    # ---- cameras + rendered GT frames ----
    views = ([v.strip() for v in render_cfg.cameras.split(",")] if render_cfg.cameras
             else [render_cfg.camera])
    pose_cams = {v: make_camera(0, torch.zeros(3, HW, HW),
                                np.zeros((1, HW, HW), np.float32),
                                fov=render_cfg.fov, view=v,
                                pan=render_cfg.cam_pan) for v in views}
    # 'scene' bg is rendered per view (depends on camera); cache it here
    bg_per_view = {}
    if not fullres and render_cfg.bg_image == "scene":
        for v in views:
            bimg, _ = render_positions(oxyz, bg_g, pipe, pose_cams[v])  # d_xyz=0: canonical
            bg_per_view[v] = bimg.clamp(0, 1)
    elif bg_image is not None:
        bg_per_view = {v: bg_image for v in views}

    import imageio
    cams, gt_frames_png, gt_frames_multi = [], [], []
    for f, pos in enumerate(gt_roll):
        row = []
        for vi, v in enumerate(views):
            bimg = bg_per_view.get(v)
            if fullres:
                img, alp = render_drive_frame(gaussians, drive, pos, pipe, pose_cams[v])
            else:
                img, alp = render_positions(pos, gaussians, pipe, pose_cams[v], bg_image=bimg)
            cam = make_camera(f, img.cpu(), alp.cpu().numpy(), fov=render_cfg.fov, view=v,
                              pan=render_cfg.cam_pan)
            # our render is ALREADY alpha-composited; overwrite the gic Camera's
            # original_image (it would re-multiply by gt_alpha_mask -> a vs a^2).
            cam.original_image = img.clamp(0.0, 1.0).cuda()
            cams.append(cam)
            png = _png(img)
            row.append(png)
            if vi == 0:
                gt_frames_png.append(png)
        gt_frames_multi.append(np.concatenate(row, axis=1))

    est.set_scene(SceneShim(gaussians, cams))
    est.gts = dummy_gts
    est.load_gts(dummy_gts)  # buffers only; geo_loss stays False
    est.img_loss = True
    if fullres:
        est.gauss_drive = drive            # render_forward drives gaussians from particles
        est.particle_xyz0 = xyz.detach()   # canonical particle positions (sim space)
    elif render_cfg.bg_image == "scene":
        est.bg_image = bg_per_view[views[0]]  # single-view bg for the loss composite
    elif bg_image is not None:
        est.bg_image = bg_image
    print(f"[img] GT rendered: {len(gt_roll)} frames x {len(views)} view(s) {views} "
          f"@{HW}^2; mode={'fullres' if fullres else 'pseudo'}, "
          f"bg={'intrinsic' if fullres else ('on' if render_cfg.bg_image else 'off')}")

    return dict(
        est=est, gaussians=gaussians, pipe=pipe, gt_roll=gt_roll, drive=drive,
        gt_frames_png=gt_frames_png, gt_frames_multi=gt_frames_multi,
        pose_cam=pose_cams[views[0]], pose_cams=pose_cams, views=views,
        bg_per_view=bg_per_view, xyz=xyz, anchor_mask=anchor_mask, pv=pv,
        aabb=aabb, z_lo=z_lo, z_hi=z_hi, flip_z=flip_z, gt_part=gt_part,
        gt_v0_grid=gt_v0_grid,
    )


def render_pred_frames(pred_roll: list, gaussians: GaussianModel, pipe,
                       pose_cam: Camera, view: str = "front",
                       bg_per_view: Optional[dict] = None, drive=None) -> list:
    """Render a predicted rollout to a list of uint8 pngs (first view).

    drive set => full-res PhysDreamer render (object driven by top_k particles);
    else the pseudo-blob render (optionally composited over bg_per_view)."""
    bimg = (bg_per_view or {}).get(view)
    out = []
    for pos in pred_roll:
        if drive is not None:
            img, _ = render_drive_frame(gaussians, drive, pos, pipe, pose_cam)
        else:
            img, _ = render_positions(pos, gaussians, pipe, pose_cam, bg_image=bimg)
        out.append(_png(img))
    return out

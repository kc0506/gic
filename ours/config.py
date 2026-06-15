# coding=utf-8
"""Shared tyro/dataclass sub-configs for the fit_traj_* / fit_image_* entrypoints.

Nested into each entrypoint's Config so the ~20 common knobs (scene cache, GT
params, train early-stop, field options) are declared ONCE here instead of being
re-copied per entrypoint. tyro turns the nesting into `--scene.cache ...` etc.
"""
import json
from argparse import Namespace
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class SceneCfg:
    """Scene cache + geometry setup shared by every traj entrypoint."""
    cache: str
    """path to the reuse_mpm scene cache (.pt)"""
    config: str = "config/ours/telephone.json"
    """physics config json (voxel_size, mpm_iter_cnt, lr schedules, ...)"""
    rot_z_deg: float = -22.4
    """rotate scene about z through (0.5,0.5). UNIFIED at -22.4 (was rot68=67.6, the
    wrong snap: it put the wall normal at +x parallel to the +y optical axis ->
    edge-on wall artifacts). -22.4 = 67.6-90 points the wall normal at -y, facing the
    camera. gic-self-sim roundtrips are rotation-invariant (GT+pred rotate together);
    only alignment to a warp-GT dump baked at 67.6 needs an explicit override."""
    euler: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    """3-axis XYZ-extrinsic Euler rotation (deg) to align a skewed scene's principal
    axes (e.g. carnation = (-3.8,-4.32,21.55)). When ANY component is nonzero this
    OVERRIDES rot_z_deg (fullres path only: rotates positions + gaussian quaternions).
    Settle it visually in a viewer (dump the ply) -- see probe_carnation_cam."""
    anchor_mass_scale: float = 1e4
    inject_pvol: bool = False
    """use the cache's per-particle points_vol instead of (dx/2)^3"""
    ti_mem_frac: float = 0.3
    """taichi device_memory_fraction (0.3 fits busy shared cards)"""
    mpm_iter_cnt: Optional[int] = None
    """override substeps/frame (None = physics config value)"""
    voxel_size: Optional[float] = None
    """override dx (None = physics config value)"""


@dataclass
class GTCfg:
    """Ground-truth params for GIC self-simulation."""
    logE: float = 5.0
    nu: float = 0.3
    vel: Tuple[float, float, float] = (0.0, -0.5, 0.0)


@dataclass
class TrainCfg:
    """train_ours early-stop / lr / checkpoint knobs (mode-independent)."""
    patience: int = 8
    min_iters: int = 15
    overlay_every: int = 10
    ckpt_every: int = 10
    param_stop_tol: float = 0.0
    """vel stage: early stop also needs per-iter max|delta| < tol (0=off)"""
    fine_lr: float = 0.0
    """phys stage two-phase: on first plateau restore best + flat fine E lr (0=off)"""
    iter_cnt: Optional[int] = None
    """override phys-stage iters (None = physics config value)"""
    vel_iter_cnt: Optional[int] = None
    """override vel-stage iters (None = physics config value)"""


@dataclass
class FieldCfg:
    """v0 / E voxel-field options (shared by the field-mode entrypoints)."""
    res: str = "4x4x16"
    """'0'=scalar; '4'=cubic; 'RXxRYxRZ' e.g. 4x4x16 = anisotropic z-refined"""
    init_std: float = 0.05
    lr: Optional[float] = None
    """field grid lr (None = config vel_lr)"""
    seed: int = 0
    tv: float = 1e-3
    """TV smoothness weight on the grid (mask-aware; 0=off)"""
    starve_thresh: float = 1.0
    """fix-to-init + de-train nodes with < this many particle-equivalents of support"""


@dataclass
class FramesCfg:
    """Fit-window overrides (None = use the physics config / full GT length)."""
    n_frames: Optional[int] = None
    """truncate fit to the first N GT frames"""
    vel_frames: Optional[int] = None
    """vel-stage fit window (vel_estimation_frames)"""
    phys_frames: Optional[int] = None
    """phys-stage fit window"""


@dataclass
class RenderCfg:
    """Image-render + loss-target options shared by the fit_image_* entrypoints.

    The optimization axis (E-scalar / v0 / joint) is split across entrypoints;
    this is the orthogonal *render/observation* axis, so it stays a flag set the
    three image entrypoints inherit unchanged. bg_image is the realistic with-bg
    setting: composite BOTH the GT and the pred render over the same static
    background and supervise on full-RGB frames (no clean alpha, so w_alp=0).
    """
    gaussians: str = "fullres"
    """'fullres' (default) = the real PhysDreamer render: full-res object gaussians
    driven by top_k particle interpolation (real anisotropic texture) + static
    bg/foreground; needs the cache's sim_mask + top_k_index. 'pseudo' = isotropic
    blob per MPM particle (lazy fallback / traj-only data: clean gradient, no texture)."""
    camera: str = "front"
    """'front'|'side_x'|'diag45'|'elev45' preset, or 'lookat' to use the free
    (cam_az,cam_el,cam_dist) viewpoint -- rotate the CAMERA around the fixed scene
    (e.g. cam_az=45 to see a face that's edge-on to the front camera)."""
    cam_az: float = 0.0
    """lookat camera azimuth (deg); 0 = front (-y looking +y), 90 = +x side"""
    cam_el: float = 0.0
    """lookat camera elevation (deg); 0 = eye-level, >0 looks down"""
    cam_dist: float = 1.5
    """lookat camera distance from the scene centre (sim units)"""
    cameras: Optional[str] = None
    """CSV of views for multi-view supervision (e.g. 'front,side_x'); None=single"""
    fov: float = 0.14
    """camera FOV (rad); 0.14 frames the telephone tightly + crops peripheral scene
    clutter. Wider (0.30) pulls in surrounding gaussians as noise."""
    cam_pan: float = 0.0
    """slide the camera centre along its right axis (sim units); pans the framing
    right to push 3DGS artefacts near the object out of a narrow fov."""
    w_img: float = 1.0
    """RGB image-loss weight"""
    w_alp: float = 0.0
    """alpha/silhouette-loss weight; 0 in bg mode (no clean alpha GT exists)"""
    bg_image: Optional[str] = None
    """background: None=black bg + obj-only loss (de-backgrounded demos);
    'scene'=render the original PhysDreamer gaussians as a static bg; else a path
    to a 3xHxW image. When set, GT and pred composite over the SAME bg -> full-RGB
    loss (static bg cancels, gradient comes from the moving object region)."""


def make_phys_args(scene: SceneCfg, train: TrainCfg, frames: FramesCfg,
                   init_logE: float, init_nu: float,
                   E_lr: Optional[float] = None, nu_lr: Optional[float] = None) -> Namespace:
    """Read the physics config json + apply the shared-config overrides.

    Single source of truth for the json -> phys_args Namespace conversion that
    every traj entrypoint needs (voxel/mpm/iter/frame overrides + init E/nu + lr).
    """
    pa = Namespace(**json.load(open(scene.config))["physics"])
    pa.init_E, pa.init_nu = init_logE, init_nu
    if scene.voxel_size is not None:
        pa.voxel_size = scene.voxel_size
    if scene.mpm_iter_cnt is not None:
        pa.mpm_iter_cnt = scene.mpm_iter_cnt
    if train.iter_cnt is not None:
        pa.iter_cnt = train.iter_cnt
    if train.vel_iter_cnt is not None:
        pa.vel_iter_cnt = train.vel_iter_cnt
    if frames.vel_frames is not None:
        pa.vel_estimation_frames = frames.vel_frames
    for cli_lr, pname in ((E_lr, "Youngs modulus"), (nu_lr, "Poisson ratio")):
        if cli_lr is not None:
            info = pa.params[pname]
            ratio = cli_lr / info["init_lr"]
            info["init_lr"], info["final_lr"] = cli_lr, info["final_lr"] * ratio
    return pa

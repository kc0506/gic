# coding=utf-8
"""Scene IO, parameter setting, and forward rollout — no GPU side effects at import.

set_params / rollout_collect_surfaces / load_or_build_static_cache were the lib
half of roundtrip_sim2sim.py; load_our_scene was in roundtrip_ours_scene.py.
"""
import os

import torch

from simulator import Estimator
from simulator.estimator import constraint_inv
from train_dynamic import prepare_gt


def load_or_build_static_cache(model, gs_args, pipeline, phys_args, cache_path: str) -> dict:
    """Return {'vol': (N,3) f32, 'vol_surface': (S,) i64, 'gts_recon': list[(Sf,3)], ...} on CPU.

    Runs GIC's prepare_gt (GS + deform model density filling) once and caches
    the static volume + surface index; later runs skip the GS machinery.
    """
    if os.path.exists(cache_path):
        print(f"[roundtrip] loading static cache {cache_path}")
        return torch.load(cache_path, map_location="cpu")
    gts, vol, vol_densities, grid_size, vol_surface, _cam_info = prepare_gt(
        model.extract(gs_args), gs_args.iteration, pipeline.extract(gs_args), phys_args
    )
    cache = {
        "vol": vol.cpu(),                      # (N, 3) initial particle positions
        "vol_densities": vol_densities.cpu(),  # (N,)
        "grid_size": grid_size.cpu(),          # (1,)
        "vol_surface": vol_surface.cpu(),      # (S,) indices into vol
        "gts_recon": [g.cpu() for g in gts],   # list of (Sf, 3) reconstructed surfaces
    }
    torch.save(cache, cache_path)
    print(f"[roundtrip] saved static cache {cache_path}")
    return cache


def rollout_collect_surfaces(estimator: Estimator) -> list:
    """Forward-roll the simulator with current params; return per-frame surface points.

    Returns list of (S, 3) float32 cuda tensors, one per frame (S = sim surface count).
    Replicates train_dynamic.forward()'s CFL-halving retry loop.
    """
    estimator.set_stage(Estimator.physical_params_stage)
    saved_geo_loss = estimator.geo_loss
    estimator.geo_loss = False  # pure rollout: no matching, no loss
    dt = estimator.simulator.dt_ori[None]
    while True:
        surfaces = []
        for idx in range(estimator.max_f):
            if idx == 0:
                estimator.initialize()
                estimator.simulator.set_dt(dt)
            estimator.forward(idx, img_backward=False)
            pts, _color = estimator.get_surface_vertics(idx)  # (S, 3) f32 numpy
            surfaces.append(torch.from_numpy(pts).to(estimator.device))
        if not estimator.succeed():
            dt /= 2
            print(f"[roundtrip] gen: cfl dissatisfied, shrink dt to {dt}")
        else:
            break
    estimator.geo_loss = saved_geo_loss
    return surfaces


def set_params(estimator: Estimator, logE: float, nu: float, vel: list) -> None:
    """Overwrite estimator's learnable params in place (E in log10 space)."""
    dev = estimator.device
    estimator.E.data = torch.tensor(logE, device=dev)
    estimator.nu.data = constraint_inv(torch.tensor(nu, device=dev), estimator.nu_bound)
    estimator.init_vel.data = torch.tensor(vel, device=dev)


def load_our_scene(cache_path: str) -> tuple:
    """Return (xyz (N,3) f32 cuda, anchor_mask (N,) bool cuda) with origin ghosts removed."""
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    disc = cache["disc"]
    xyz = disc["sim_xyzs"]            # (N0, 3) normalized [0,1]^3
    freeze = disc["freeze_mask"]      # (N0,) bool
    ghost = (xyz == 0).all(dim=1)     # kmeans origin ghosts (all frozen, inert)
    keep = ~ghost
    print(f"[ours] cache {cache_path}: {xyz.shape[0]} particles, "
          f"{int(ghost.sum())} origin ghosts removed, "
          f"{int((freeze & keep).sum())} real anchors kept")
    return xyz[keep].float().cuda(), freeze[keep].cuda()


def build_anchored_scene(scene_cache: str, rot_z_deg: float, phys_args,
                         anchor_mass_scale: float, n_frames: int,
                         inject_pvol: bool = True, ti_mem_frac: float = 0.3,
                         init_xyz=None) -> dict:
    """Shared traj-entrypoint setup: load + rotate + ti.init + AnchoredEstimator.

    The common scene-setup half of roundtrip_ours_scene / efield_fit mains. Does
    NOT set stage / geo_loss / F0 / external-traj GT -- those stay per-entrypoint
    (they differ by training mode). ti.init touches CUDA, so the CALLER must have
    run ours.gpu.pick_gpu() first.

    Returns dict: est, xyz (N,3 cuda), anchor_mask, free, aabb (2,3), z_lo, z_hi,
    flip_z (anchor end is the high-z end).
    """
    import taichi as ti

    from ours.estimator import AnchoredEstimator
    from ours.geom import rot_xyz

    xyz, anchor_mask = load_our_scene(scene_cache)
    if init_xyz is not None:                       # deformed-snapshot t0 (F0 sysid)
        assert init_xyz.shape == xyz.shape, (init_xyz.shape, xyz.shape)
        xyz = init_xyz
    if rot_z_deg:
        xyz = rot_xyz(xyz, rot_z_deg)
    free = ~anchor_mask
    pad = 2.0 * phys_args.voxel_size
    aabb = torch.stack([xyz.min(0).values - pad, xyz.max(0).values + pad])  # (2,3)
    z_lo, z_hi = float(xyz[:, 2].min()), float(xyz[:, 2].max())
    flip_z = bool(xyz[anchor_mask][:, 2].mean() > xyz[free][:, 2].mean())

    ti.init(arch=ti.cuda, debug=False, fast_math=False,
            device_memory_fraction=ti_mem_frac)
    dummy = [xyz.clone() for _ in range(n_frames)]
    est = AnchoredEstimator(phys_args, "float32", dummy, surface_index=None,
                            init_vol=xyz, dynamic_scene=None, image_scale=1.0,
                            pipeline=None, image_op=None)
    est.set_anchor(anchor_mask, anchor_mass_scale)
    if inject_pvol:
        cache = torch.load(scene_cache, map_location="cpu", weights_only=False)
        ghost = (cache["disc"]["sim_xyzs"] == 0).all(dim=1)
        pvol = torch.from_numpy(cache["disc"]["points_vol"]).float()[~ghost]
        est.set_pvol(pvol)
    return {"est": est, "xyz": xyz, "anchor_mask": anchor_mask, "free": free,
            "aabb": aabb, "z_lo": z_lo, "z_hi": z_hi, "flip_z": flip_z}

#!/usr/bin/env python
# coding=utf-8
"""fit_image_F0_block: recover the pre-stress field F0 from IMAGE loss on a synthetic
block bend (gic->gic self-sim, pseudo gaussians, no dataset_dir / no FCR).

This is the FIRST time F0 is fit under a rendered-video loss (every prior F0 run used
per-particle trajectory / geometry loss).  It validates the F0 -> image-loss -> F0.grad
-> MLP gradient path end to end before investing in a real scene (telephone).

  MLP(x0) -> w -> F0 = (I - grad_{x0} w)^{-1}      [ours.f0field]
    -> est.set_F0(F0.detach()) -> gic self-sim release -> render pseudo gaussians
    -> image L1+ssim vs GT-bend render -> render_forward backward -> pos_grad_seq
    -> gic BPTT (train_dynamic.backward) -> F.grad[p,0] -> _read_F0_grad = dL/dF0
    -> F0.backward(dL/dF0) -> MLP -> Adam

GT: the bundle's V0_true (warp-derived pre-stress) injected + rolled out IN gic with
gic's own material (material from --config), rendered identically.  Self-consistent, so
the only thing under test is whether the rendered image constrains the field.

Usage (gic env, gic repo root):
  python fit_image_F0_block.py \
    --bundle /tmp2/.../f0_dump_gt/gradu_ybend_E4p5 --gt-logE 4.5 \
    --iters 120 --lr 2e-3 --run-label f0img_bend
"""
from ours.gpu import pick_gpu

pick_gpu()  # pick a free GPU before torch/taichi create a CUDA context

import json
import os
import time
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import tyro

from arguments import OptimizationParams, PipelineParams
from simulator import Estimator
from train_dynamic import backward as gic_backward
from utils.system_utils import draw_curve

from ours.estimator import AnchoredEstimator, forward_bounded
from ours.f0field import UMLP, F0_from_w
from ours.imgloss import (HW, SceneShim, _png, build_synthetic_pseudo_gaussians,
                          build_fullres_gaussians, gt_pred_diff_gif, make_oblique_camera,
                          render_drive_frame, render_positions, _load_orig, _orig_xyz_norm)
from ours.rundir import RunDir
from ours.scene import rollout_collect_surfaces, set_params


@dataclass
class Config:
    bundle: str = "/tmp2/b10401006/ev-project/generative-phys/outputs/explore/f0_dump_gt/gradu_ybend_E4p5"
    config: str = "config/ours/telephone.json"   # physics json (material/lr); block has none
    gt_logE: float = 4.5
    gt_nu: float = 0.3
    K: int = 24
    mpm_iter_cnt: int = 64
    iters: int = 120
    lr: float = 2e-3
    width: int = 64
    elev: float = 60.0     # camera depression (deg); front (elev 0) is BLIND to a +y bend
    azim: float = -30.0    # camera azimuth about +z (deg); both pull y-motion into view
    fov: float = 0.30
    w_img: float = 1.0
    w_alp: float = 0.0
    ti_mem_frac: float = 0.25
    use_cache_anchor: bool = False  # anchor the cache freeze_mask (hanging scenes like telephone) vs free release
    fullres: bool = False  # realistic PhysDreamer render (object driven by top_k) vs pseudo blobs; no external bg
    render_cache: str = "/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt"
    dataset_dir: str = "/tmp2/b10401006/PhysDreamer/data/physics_dreamer/telephone"  # original 3DGS ply
    seed: int = 0          # MLP init seed (reproducible runs / fair view comparisons)
    save_states: bool = True  # per-iter MLP ckpt + trajectory -> any metric reconstructable offline
    run_label: str = ""
    out: Optional[str] = None


def _v0_err(F0: torch.Tensor, V0_true: torch.Tensor) -> float:
    """mean over particles of the per-particle max |V0(F0) - V0_true|.  Gauge-free."""
    ev, Q = torch.linalg.eigh(F0 @ F0.mT)
    V0 = (Q * ev.clamp_min(1e-9).sqrt().unsqueeze(-2)) @ Q.mT
    return float((V0 - V0_true).abs().reshape(F0.shape[0], -1).amax(1).mean())


def run(cfg: Config, rd: RunDir) -> None:
    t0 = time.time()
    K_frames = cfg.K + 1
    dev = "cuda"

    # ---- physics args (borrow telephone.json material; GT and fit share it) ----
    pa = Namespace(**json.load(open(cfg.config))["physics"])
    pa.init_E, pa.init_nu = cfg.gt_logE, cfg.gt_nu
    pa.mpm_iter_cnt = cfg.mpm_iter_cnt
    pa.n_frames = K_frames
    pa.img_loss = True
    pa.w_img, pa.w_alp, pa.w_geo = cfg.w_img, cfg.w_alp, 0.0

    # ---- block geometry from the bundle (geometry only; no dataset_dir needed) ----
    cache = torch.load(f"{cfg.bundle}/scene_cache.pt", map_location="cpu", weights_only=False)
    ghost = (cache["disc"]["sim_xyzs"] == 0).all(dim=1)                 # (N0,) bool
    Xrest = cache["disc"]["sim_xyzs"][~ghost].float()                   # (N,3) rest (colour only)
    pv = torch.as_tensor(cache["disc"]["points_vol"]).float()[~ghost]   # (N,) (numpy or tensor)
    x0 = torch.from_numpy(np.load(f"{cfg.bundle}/init_xyz.npy")).float().to(dev)  # (N,3) observed t0
    V0_true = torch.from_numpy(np.load(f"{cfg.bundle}/f0.npy")).float().to(dev)   # (N,3,3) GT pre-stress
    n = x0.shape[0]
    print(f"[f0img] block N={n}, K={cfg.K}, material={pa.material}, config={cfg.config}")

    # ---- 3DGS pipeline/opt params + synthetic pseudo gaussians (colour by rest pos) ----
    _p = ArgumentParser()
    pipe = PipelineParams(_p).extract(_p.parse_args([]))
    iop = OptimizationParams(_p).extract(_p.parse_args([]))
    drive = None
    if cfg.fullres:
        from ours.gauss_drive import TopKDrive
        rc = torch.load(cfg.render_cache, map_location="cpu", weights_only=False)["disc"]
        cscale = float(rc["scale"]); shift = rc["shift"].reshape(-1)
        sim_mask = rc["sim_mask"].cuda(); top_k_index = rc["top_k_index"].cuda()
        orig = _load_orig(cfg.dataset_dir)
        oxyz = _orig_xyz_norm(orig, shift, cscale, 0.0)               # rot_deg=0: match unrotated bundle particles
        gaussians = build_fullres_gaussians(orig, oxyz, cscale, 0.0)
        drive = TopKDrive(gaussians.get_xyz.detach(), gaussians.get_rotation.detach(),
                          x0.detach(), top_k_index, sim_mask)
        print(f"[f0img] FULL-RES: {gaussians.get_xyz.shape[0]} splats, object={int(sim_mask.sum())} "
              f"driven by top_{top_k_index.shape[1]}")
    else:
        lo, hi = Xrest.min(0).values, Xrest.max(0).values
        rgb = ((Xrest - lo) / (hi - lo).clamp_min(1e-9)).to(dev)       # (N,3) visible texture
        gaussians = build_synthetic_pseudo_gaussians(x0, pv, rgb)
    torch.cuda.empty_cache()

    # ---- estimator (with pipeline for render) ----
    import taichi as ti
    ti.init(arch=ti.cuda, debug=False, fast_math=False, device_memory_fraction=cfg.ti_mem_frac)
    dummy = [x0.clone() for _ in range(K_frames)]
    est = AnchoredEstimator(pa, "float32", dummy, surface_index=None, init_vol=x0,
                            dynamic_scene=None, image_scale=1.0, pipeline=pipe, image_op=iop)
    if cfg.use_cache_anchor and cache["disc"].get("freeze_mask") is not None:
        anchor_mask = cache["disc"]["freeze_mask"][~ghost].bool().to(dev)  # hanging scene (telephone): hold the hang
        print(f"[f0img] cache anchor: {int(anchor_mask.sum())} frozen (hang)")
    else:
        anchor_mask = torch.zeros(n, dtype=torch.bool, device=dev)       # free release (no anchors)
    est.set_anchor(anchor_mask, 1e4)
    est.set_pvol(pv)
    est.geo_loss = False
    est.set_stage(Estimator.physical_params_stage)

    # ---- GT rollout (img OFF) + render -> cameras ----
    est.img_loss = False
    set_params(est, cfg.gt_logE, cfg.gt_nu, [0.0, 0.0, 0.0])
    est.set_F0(V0_true)
    est.max_f = K_frames
    gt_roll = rollout_collect_surfaces(est)
    cams, gt_pngs = [], []
    for f, pos in enumerate(gt_roll):
        cam = make_oblique_camera(f, torch.zeros(3, HW, HW), np.zeros((1, HW, HW), np.float32),
                                  fov=cfg.fov, elev_deg=cfg.elev, azim_deg=cfg.azim)
        img, alp = (render_drive_frame(gaussians, drive, pos, pipe, cam) if cfg.fullres
                    else render_positions(pos, gaussians, pipe, cam))
        cam.original_image = img.clamp(0.0, 1.0).cuda()
        cams.append(cam)
        gt_pngs.append(_png(img))
    # sanity: the object is actually in frame (non-empty render)
    fill = [float((torch.from_numpy(p).float().sum(-1) > 4).float().mean()) for p in gt_pngs]
    print(f"[f0img] GT rendered {K_frames}f @ {HW}^2; fov {cfg.fov}; "
          f"object pixel-fill mid-frame {fill[K_frames // 2]:.1%} (min {min(fill):.1%})")
    import imageio
    imageio.mimsave(rd.path("gt.gif"), gt_pngs, fps=8)

    # ---- wire est for image loss ----
    est.set_scene(SceneShim(gaussians, cams))
    est.gts = dummy
    est.load_gts(dummy)
    est.img_loss = True
    if cfg.fullres:                                  # render_forward drives object gaussians from particles
        est.gauss_drive = drive
        est.particle_xyz0 = x0.detach()

    # ---- recover F0 (MLP), image loss + F0 grad bridge ----
    torch.manual_seed(cfg.seed)
    mlp = UMLP(cfg.width).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=cfg.lr)
    eye = torch.eye(3, device=dev)
    gt_stack = torch.stack(gt_roll)                                     # (K+1,N,3) GT positions
    # persist the reconstruction inputs ONCE: x0 (MLP input) + GT traj. With these +
    # the per-iter MLP ckpt, ANY metric (F0/V0err/traj/render) is reconstructable
    # offline -- no re-run to add a metric (the lesson from the repeated re-runs).
    np.save(rd.path("x0.npy"), x0.cpu().numpy())
    np.save(rd.path("gt_traj.npy"), gt_stack.cpu().numpy())
    states_dir = rd.path("states")
    if cfg.save_states:
        os.makedirs(states_dir, exist_ok=True)
    losses, v0errs, trajs = [], [], []
    for it in range(cfg.iters):
        opt.zero_grad()
        F0, Jw, w = F0_from_w(mlp, x0, eye)
        est.set_F0(F0.detach())
        try:
            forward_bounded(est)            # initialize() writes F0; render_forward backward per frame
        except Exception as e:
            print(f"[f0img] iter {it}: forward failed ({e}); stopping")
            break
        img_loss = float(est.image_loss)
        gic_backward(est)                   # MPM BPTT consuming pos_grad_seq -> F.grad[p,0]
        g = np.zeros((n, 3, 3), dtype=np.float32)
        est._read_F0_grad(g, n)
        F0.backward(gradient=torch.from_numpy(g).to(dev))
        opt.step()

        v0e = _v0_err(F0.detach(), V0_true)
        # trajectory MSE: the position-space objective the image is a proxy for. A cheap
        # no-grad rollout with this iter's F0 (no render). If traj-MSE falls while V0err
        # stays high -> the dynamics are matched by a wrong field (image/traj under-
        # determine F0); if both fall -> genuine recovery.
        est.img_loss = False
        with torch.no_grad():
            pred = torch.stack(rollout_collect_surfaces(est))          # (K+1,N,3)
        est.img_loss = True
        est.set_stage(Estimator.physical_params_stage)
        tmse = float((pred[1:] - gt_stack[1:]).pow(2).sum(-1).mean())
        losses.append(img_loss); v0errs.append(v0e); trajs.append(tmse)
        print(f"[f0img] iter {it:3d}  img_loss={img_loss:.5f}  traj_mse={tmse:.3e}  "
              f"V0err={v0e:.4f}  ({time.time()-t0:.0f}s)")
        np.save(rd.path("losses.npy"), np.array(losses))
        np.save(rd.path("v0err.npy"), np.array(v0errs))
        np.save(rd.path("traj.npy"), np.array(trajs))
        if cfg.save_states:  # compact MLP ckpt (reconstructs F0/field) + rolled traj (fp16)
            torch.save({k: v.cpu() for k, v in mlp.state_dict().items()},
                       os.path.join(states_dir, f"mlp_{it:04d}.pt"))
            np.save(os.path.join(states_dir, f"traj_{it:04d}.npy"),
                    pred.cpu().half().numpy())

    # ---- final pred render + diff gif + validation ----
    est.img_loss = False
    F0, Jw, w = F0_from_w(mlp, x0, eye); F0 = F0.detach()
    est.set_F0(F0)
    est.max_f = K_frames
    pred_roll = rollout_collect_surfaces(est)
    pred_pngs = []
    for f, pos in enumerate(pred_roll):
        img, _ = (render_drive_frame(gaussians, drive, pos, pipe, cams[f]) if cfg.fullres
                  else render_positions(pos, gaussians, pipe, cams[f]))
        pred_pngs.append(_png(img))
    gt_pred_diff_gif(gt_pngs, pred_pngs, rd.path("gt_pred_diff.gif"), K_frames)

    v0_max = float((torch.linalg.eigh(F0 @ F0.mT)[0].clamp_min(1e-9).sqrt().max() - 1).abs())
    result = {"scenario": "ours_image_F0_block", "bundle": cfg.bundle,
              "gt_logE": cfg.gt_logE, "gt_nu": cfg.gt_nu, "K": cfg.K, "iters": len(losses),
              "lr": cfg.lr, "w_img": cfg.w_img, "w_alp": cfg.w_alp, "fov": cfg.fov,
              "final_img_loss": losses[-1] if losses else None,
              "final_V0err_mean": v0errs[-1] if v0errs else None,
              "min_V0err": min(v0errs) if v0errs else None, "wall_s": time.time() - t0}
    with open(rd.path("result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if losses:
        draw_curve(losses, rd.root, name="img_loss")
    torch.save(mlp.state_dict(), rd.path("mlp.pt"))
    print(f"[f0img] DONE img_loss {losses[-1] if losses else 'NA'} | "
          f"V0err {v0errs[-1] if v0errs else 'NA'} (min {min(v0errs) if v0errs else 'NA'}) "
          f"| wall {time.time()-t0:.0f}s -> {rd.root}")


def main() -> None:
    cfg = tyro.cli(Config)
    rd = RunDir.create(__name__, cfg.run_label, cfg.out, config=cfg)
    run(cfg, rd)


if __name__ == "__main__":
    main()

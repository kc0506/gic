# coding=utf-8
"""train_ours: GIC's train loop + early stop + two-phase lr + checkpoints.

Plus param snapshot/checkpoint helpers. The training-mode entrypoints all drive
this; field vs scalar is just which params the estimator's optimizer points at.
"""
import os

import numpy as np
import torch

from simulator import Estimator

from ours.estimator import CFLExhausted, forward_bounded
from ours.fields import eval_Egrid_at, eval_grid_at
from ours.scene import rollout_collect_surfaces
from ours.viz import save_overlay_gif


def _record_params(estimator: Estimator) -> dict:
    """Snapshot current params (same extraction as train_dynamic.train)."""
    d = {}
    for params in estimator.get_optimizer().param_groups:
        name = params["name"]
        p = params["params"][0].detach().cpu()
        if name == "Poisson ratio":
            p = estimator.get_nu().detach().cpu()
        elif name in ["Youngs modulus", "Yield stress", "plastic viscosity",
                      "shear modulus", "bulk modulus"]:
            p = 10 ** p
        d[name] = p if name == "velocity" else p.item()
    return d


def save_ckpt(estimator: Estimator, losses: list, estimated_params: list,
              it: int, path: str) -> None:
    """Checkpoint raw params + optimizer state so a fit can be RESUMED exactly.

    Raw (unconstrained) tensors: E is log10-space, nu is pre-tanh; optimizer
    state matters for Adam moments, hence state_dict.
    """
    field = getattr(estimator, "_v0_field", None)
    torch.save({
        "iter": it,
        "E_raw": estimator.E.data.detach().cpu(),
        "nu_raw": estimator.nu.data.detach().cpu(),
        "init_vel": estimator.init_vel.data.detach().cpu(),
        "v0_field_grid": None if field is None else field.grid.data.detach().cpu(),
        "optimizer": estimator.get_optimizer().state_dict(),
        "stage": int(estimator.stage[None]),
        "losses": [float(l) for l in losses],
        "estimated_params": [
            {k: (v if not torch.is_tensor(v) else v.tolist()) for k, v in d.items()}
            for d in estimated_params],
    }, path)


def train_ours(estimator: Estimator, phys_args, max_f: int, out_dir: str,
               gts: list = None, patience: int = 8, min_iters: int = 15,
               rel_improve: float = 0.02, ckpt_every: int = 10,
               overlay_every: int = 0, fine_lr: float = 0.0,
               param_stop_tol: float = 0.0, tv_weight: float = 0.0) -> tuple:
    """train_dynamic.train + early stop + checkpoints + periodic overlay gifs.

    Early stop: quit when `patience` iters pass without the best loss improving
    by >`rel_improve` relative (never before `min_iters`). Best-so-far params
    are what the caller exports (same min-loss convention as gic's train), so
    stopping can only truncate post-convergence oscillation, not hurt the best.

    fine_lr > 0 (phys stage only) enables a two-phase schedule: the FIRST time
    the plateau condition fires, restore the best-so-far raw params, reset the
    Adam state, and swap every lr scheduler to a FLAT lr (E = fine_lr, others
    scaled by the same ratio vs their init), then keep going; the SECOND
    plateau stops. Rationale (measured, E1e4 8f): coarse lr 0.2 is needed to
    traverse a decade but oscillates +-5-20% around the valley; flat lr/4
    starves (never arrives, +21.5%). Coarse-traverse + fine-refine fixes both.

    param_stop_tol > 0 (vel stage): early stop ALSO requires the velocity params
    to have stopped moving (per-iter max |delta| < tol for `patience` iters).
    Guards against the vyz failure mode: a slowly-crawling component (z moved
    +0.013/iter) gets killed by a pure loss-plateau criterion while still
    converging (Q2 report, corrected lesson 7).
    Returns (losses, estimated_params) exactly like train_dynamic.train.
    """
    from train_dynamic import backward as gic_backward

    stage = int(estimator.stage[None])
    iter_cnt = (phys_args.vel_iter_cnt if stage == Estimator.velocity_stage
                else phys_args.iter_cnt)
    estimator.max_f = max_f
    losses, estimated_params = [], []
    best_loss, last_sig_improve = float("inf"), 0
    last_big_move = 0  # last iter where velocity params moved >= param_stop_tol
    best_raw = None  # (E_raw, nu_raw, vel_raw) at the best loss so far
    dropped = False
    for i in range(iter_cnt):
        estimated_params.append(_record_params(estimator))
        if param_stop_tol > 0 and len(estimated_params) >= 2:
            # movement across ALL optimized params (J1 lesson: watching only
            # velocity let a loss-plateau stop kill joint mid E-rebound).
            # E compared in log10 (0.005 dex ~ 1.2%); velocity/nu in raw units.
            import math as _math
            prev, cur = estimated_params[-2], estimated_params[-1]
            delta = 0.0
            for k, v in cur.items():
                if k not in prev:
                    continue
                if torch.is_tensor(v):
                    delta = max(delta, float((v - prev[k]).abs().max()))
                elif k == "Youngs modulus":
                    delta = max(delta, abs(_math.log10(max(v, 1e-12))
                                           - _math.log10(max(prev[k], 1e-12))))
                else:
                    delta = max(delta, abs(v - prev[k]))
            if delta >= param_stop_tol:
                last_big_move = i
        estimator.zero_grad()
        estimator.loss[None] = 0.0
        try:
            forward_bounded(estimator)
        except CFLExhausted as e:
            estimated_params.pop()
            print(f"[train_ours] STOP at iter {i}: {e}; keeping best-so-far "
                  f"(loss {best_loss:.6f})")
            break
        loss = estimator.loss[None] + estimator.image_loss
        if loss < best_loss:  # params recorded pre-step => stash BEFORE step()
            best_raw = (estimator.E.data.clone(), estimator.nu.data.clone(),
                        estimator.init_vel.data.clone())
        losses.append(loss)
        gic_backward(estimator)
        field = getattr(estimator, "_v0_field", None)
        if field is not None and tv_weight > 0:
            # TV grads ACCUMULATE onto the data grads before step (pure torch
            # path; the starve grad-mask hook applies to these too)
            (tv_weight * field.regularization()).backward()
        estimator.step(i)

        if loss < best_loss * (1.0 - rel_improve):
            last_sig_improve = i
        best_loss = min(best_loss, loss)
        min_idx = losses.index(min(losses))
        best_show = {k: (f"grid{tuple(v.shape)} mean {v.mean(dim=(0, 2, 3, 4)).tolist()}"
                         if torch.is_tensor(v) and v.numel() > 3 else v)
                     for k, v in estimated_params[min_idx].items()}
        print(f"[train_ours] iter {i} loss {loss:.6f} | best {losses[min_idx]:.6f} "
              f"@ {min_idx} | {best_show}")

        if ckpt_every and ((i + 1) % ckpt_every == 0 or i == iter_cnt - 1):
            save_ckpt(estimator, losses, estimated_params, i,
                      os.path.join(out_dir, f"ckpt_latest_stage{stage}.pt"))
        if overlay_every and gts is not None and (i + 1) % overlay_every == 0:
            estimator.max_f = len(gts)                  # full-frame rollout even in
            pred = rollout_collect_surfaces(estimator)  # the 4-frame vel stage
            estimator.set_stage(stage)                  # rollout forces phys stage
            estimator.max_f = max_f
            save_overlay_gif(gts, pred, os.path.join(out_dir, f"overlay_iter{i + 1:03d}.gif"),
                             fit_frames=max_f)

        params_settled = (param_stop_tol == 0 or (i - last_big_move) >= patience)
        if i + 1 >= min_iters and (i - last_sig_improve) >= patience and params_settled:
            can_drop = (fine_lr > 0 and not dropped
                        and stage == Estimator.physical_params_stage)
            if can_drop:
                dropped = True
                estimator.E.data.copy_(best_raw[0])
                estimator.nu.data.copy_(best_raw[1])
                opt = estimator.get_optimizer()
                for p in (estimator.E, estimator.nu):
                    opt.state.pop(p, None)  # stale coarse-phase Adam moments
                scale = fine_lr / phys_args.params["Youngs modulus"]["init_lr"]
                for name, info in phys_args.params.items():
                    if name in estimator.lr_schedulers:
                        flat = info["init_lr"] * scale
                        estimator.lr_schedulers[name] = (lambda lr: (lambda it: lr))(flat)
                        for pg in opt.param_groups:
                            if pg["name"] == name:
                                pg["lr"] = flat
                last_sig_improve = i
                print(f"[train_ours] plateau at iter {i}: restored best "
                      f"(loss {best_loss:.6f}) and dropped to flat fine lr "
                      f"(E lr {fine_lr}); refining until next plateau")
                continue
            print(f"[train_ours] early stop at iter {i}: no >{rel_improve:.0%} best-loss "
                  f"improvement for {patience} iters (best {best_loss:.6f} @ {min_idx})")
            break

    if ckpt_every:
        save_ckpt(estimator, losses, estimated_params, len(losses) - 1,
                  os.path.join(out_dir, f"ckpt_latest_stage{stage}.pt"))
    field = getattr(estimator, "_v0_field", None)
    restore_vel = losses and (stage == Estimator.velocity_stage or field is not None)
    if restore_vel and "velocity" in estimated_params[losses.index(min(losses))]:
        import torch.nn as nn
        best = estimated_params[losses.index(min(losses))]
        if field is not None:  # "velocity" record = (1,3,rz,ry,rx) grid in field mode
            field.grid.data.copy_(best["velocity"].to(estimator.device))
        else:
            estimator.init_vel = nn.Parameter(best["velocity"].to(estimator.device))
    return losses, estimated_params


def train_efield_grid(est, fit_field, gt_lp_free, fm, w, aabb_c, xyz_c, iter_cnt: int,
                      tv_weight: float, fine_lr: float, patience: int, min_iters: int,
                      estop_tol: float, v0_field=None, gt_v0_pf=None) -> dict:
    """E-grid (+optional joint v0-grid) fit loop -- the custom loop the E-field
    work needs INSTEAD of train_ours.

    Unlike train_ours (which restores a scalar best_raw E/nu/vel), this restores
    the best GRID, drops to a flat fine lr on the first plateau, and adds TV. The
    caller must have set up est.optimizer beforehand (E-only: [E-grid]; joint:
    [v0-param, E-grid]). gt_lp_free (M,) GT per-particle log10 E at free particles,
    w (M,) normalized strain weight, fm free mask (cpu bool), aabb_c/xyz_c cpu.
    Returns dict(grid_traj, err_traj, errw_traj, loss_traj, v0_traj, best_grid).
    """
    from train_dynamic import backward as gic_backward

    grid_traj, err_traj, errw_traj, loss_traj, v0_traj = [], [], [], [], []
    v0_rel_traj = []  # per-iter v0 xy-relL2 vs gt_v0_pf (joint diagnostic; needs gt_v0_pf)
    gt_v0_scale = (max(float(gt_v0_pf.norm(dim=-1).mean()), 1e-12)
                   if gt_v0_pf is not None else 1.0)
    best_loss, best_grid, last_improve, last_move, dropped = float("inf"), None, 0, 0, False
    for it in range(iter_cnt):
        prev_grid = fit_field.grid.detach().cpu().clone()
        est.optimizer.zero_grad()
        est.loss[None] = 0.0
        try:
            forward_bounded(est, max_halvings=3)
        except CFLExhausted:
            print(f"[ef] iter {it}: CFL exhausted, keep best"); break
        loss = float(est.loss[None])
        gic_backward(est)
        if tv_weight > 0:
            tv = fit_field.regularization()
            if v0_field is not None:
                tv = tv + v0_field.regularization()
            (tv_weight * tv).backward()
        est.optimizer.step()

        lp = eval_Egrid_at(fit_field.grid.detach().cpu(), aabb_c, xyz_c)[fm]
        e_all = float((lp - gt_lp_free).abs().mean())
        e_w = float((np.abs((lp - gt_lp_free).numpy()) * w).sum())   # strain-weighted
        grid_traj.append(fit_field.grid.detach().cpu().clone())
        err_traj.append(e_all); errw_traj.append(e_w); loss_traj.append(loss)
        if v0_field is not None:
            vrec = eval_grid_at(v0_field.grid.detach().cpu(), aabb_c, xyz_c)[fm]  # (M,3)
            v0_traj.append(vrec.mean(0).numpy().tolist())
            if gt_v0_pf is not None:
                v0_rel_traj.append(float((vrec - gt_v0_pf)[:, :2].norm(dim=-1).mean()
                                         / gt_v0_scale))
        else:
            v0_traj.append(est.init_vel.detach().cpu().numpy().tolist())
        if loss < best_loss:
            best_loss, best_grid = loss, fit_field.grid.detach().cpu().clone()
        if loss < best_loss * 1.02:
            last_improve = it
        move = (fit_field.grid.detach().cpu() - prev_grid).abs().max().item()
        if move >= estop_tol:
            last_move = it
        print(f"[ef] iter {it} loss {loss:.6f} | logE err all {e_all:.3f} "
              f"strain-w {e_w:.3f} | best loss {best_loss:.6f}")
        plateau = it + 1 >= min_iters and (it - last_improve) >= patience
        if plateau and not dropped:
            dropped = True
            fit_field.grid.data.copy_(best_grid.to(est.device))
            est.optimizer.state.pop(fit_field.grid, None)
            for g in est.optimizer.param_groups:
                g["lr"] = fine_lr
            last_improve = it
            print(f"[ef] plateau -> restore best + fine lr {fine_lr}")
            continue
        if plateau and dropped and (it - last_move) >= patience:
            print(f"[ef] early stop at iter {it}"); break

    if best_grid is not None:
        fit_field.grid.data.copy_(best_grid.to(est.device))
    return dict(grid_traj=grid_traj, err_traj=err_traj, errw_traj=errw_traj,
                loss_traj=loss_traj, v0_traj=v0_traj, v0_rel_traj=v0_rel_traj,
                best_grid=best_grid)


def warmup_v0(est, v0_param, warmup_iters: int, v0_lr: float, v0_field=None,
              gt_v0_pf=None, fm=None, aabb_c=None, xyz_c=None) -> tuple:
    """v0-only warmup at the (wrong) init E, RECORDING loss + v0 xy-relL2 per iter.

    The vel-stage diagnostic the old efield_fit dropped (it only printed the final
    mean). The optimization steps are byte-identical to efield_fit's warmup
    (zero_grad/forward/backward/step over v0_param); only the recording is added.
    v0_param = the field grid (double field) or scalar init_vel. Returns
    (loss_traj, v0_rel_traj).
    """
    from train_dynamic import backward as gic_backward

    warm_opt = torch.optim.Adam([{"params": v0_param, "lr": v0_lr}])
    gt_scale = (max(float(gt_v0_pf.norm(dim=-1).mean()), 1e-12)
                if gt_v0_pf is not None else 1.0)
    loss_traj, v0_rel_traj = [], []
    for wi in range(warmup_iters):
        warm_opt.zero_grad()
        est.loss[None] = 0.0
        try:
            forward_bounded(est, max_halvings=3)
        except CFLExhausted:
            print(f"[warmup] iter {wi}: CFL exhausted"); break
        loss = float(est.loss[None])
        gic_backward(est)
        warm_opt.step()
        loss_traj.append(loss)
        if gt_v0_pf is not None:
            if v0_field is not None:
                vrec = eval_grid_at(v0_field.grid.detach().cpu(), aabb_c, xyz_c)[fm]
            else:
                vrec = est.init_vel.detach().cpu().unsqueeze(0).expand(int(fm.sum()), 3)
            v0_rel_traj.append(float((vrec - gt_v0_pf)[:, :2].norm(dim=-1).mean() / gt_scale))
        print(f"[warmup] iter {wi} loss {loss:.6f}")
    return loss_traj, v0_rel_traj

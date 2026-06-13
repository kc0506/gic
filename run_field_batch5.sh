#!/bin/zsh
# batch 5 (2026-06-13): A) iso-16^3 DOF-limit test (uniform + ramp_y);
# B) diag45 camera (ramp_y + bend) -- the "visible => learnable" closing piece.
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt

traj_one() {  # $1=tag $2...=args
  tag=$1; shift
  [ -f "output/ours_telephone/$tag/result.json" ] && { echo "[f5] skip $tag"; return }
  timeout 1500 $PY roundtrip_ours_scene.py --scene_cache $CACHE \
    --inject_pvol --mpm_iter_cnt 64 --rot_z_deg 67.6 \
    --v0_field_tv 1e-3 --param_stop_tol 0.02 \
    --vel_iter_cnt 200 --min_iters 50 --patience 32 --overlay_every 25 \
    --tag $tag "$@" > output/q2_logs/$tag.log 2>&1
  echo "[f5] $tag exit=$?"
  $PY make_panel.py --per_run --runs output/ours_telephone/$tag 2>/dev/null
}
img_one() {  # $1=tag $2...=args
  tag=$1; shift
  [ -f "output/ours_telephone/$tag/result.json" ] && { echo "[f5] skip $tag"; return }
  timeout 3600 $PY image_fit_ours.py --mode fit_v0field --tag $tag \
    --w_img 1.0 --w_alp 0.0 --iter_cnt 200 --min_iters 50 --patience 32 \
    --param_stop_tol 0.02 "$@" > output/q2_logs/$tag.log 2>&1
  echo "[f5] $tag exit=$?"
  $PY make_panel.py --per_run --runs output/ours_telephone/$tag 2>/dev/null
}

# A: iso 16^3 (4096 nodes, 139 supported) -- redundant-DOF robustness
traj_one field_iso16_uniform_y --fix_E_gt --gt_logE 5.0 --init_logE 5.0 \
  --gt_vel 0.0 -0.5 0.0 --v0_field_res 16 --v0_field_init_std 0.05
traj_one field_iso16_rampy_s6 --fix_E_gt --gt_logE 4.0 --init_logE 4.0 \
  --gt_v0_variant ramp_y --gt_v0_scale 6 --v0_field_res 16 \
  --v0_field_init_std 0.15 --v0_field_lr 0.075

# B: diag45 camera (x AND y both partly in-plane)
img_one imgfield_rampy_diag45 --gt_logE 4.0 --gt_v0_variant ramp_y --gt_v0_scale 6 \
  --camera diag45 --v0_field_lr 0.075 --v0_field_init_std 0.15
img_one imgfield_bend_diag45 --gt_logE 4.0 --gt_v0_variant true_bend --gt_v0_scale 6 \
  --camera diag45 --v0_field_lr 0.075 --v0_field_init_std 0.15
echo "[f5] BATCH DONE"

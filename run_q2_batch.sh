#!/bin/zsh
# Q2 batch (2026-06-12): Path A = E1e6 fix-v0 x2; Path B = v0-only x3 (fix_E_gt).
# Two parallel chains (user-approved dual GPU; >=2 idle verified at launch).
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt
DUMP=/tmp2/b10401006/ev-project/generative-phys/outputs/explore/xmodel_dump
mkdir -p output/q2_logs

fixv0_run() {  # $1=traj_label $2=tag $3..5=v0
  [ -f "output/ours_telephone/$2/result.json" ] && { echo "[q2] skip $2"; return }
  timeout 900 $PY roundtrip_ours_scene.py --scene_cache $CACHE \
    --gt_traj $DUMP/$1/warp_traj.npy --gt_traj_normalized \
    --gt_logE 6.0 --init_logE 5.0 --gt_vel $3 $4 $5 --fix_v0_gt --inject_pvol \
    --mpm_iter_cnt 64 --n_frames 8 --patience 16 --min_iters 25 \
    --fine_lr 0.02 --nu_lr 0.0125 --tag $2 > output/q2_logs/$2.log 2>&1
  echo "[q2] $2 exit=$?"
}

v0only_run() {  # $1=traj_label $2=tag $3..5=v0
  [ -f "output/ours_telephone/$2/result.json" ] && { echo "[q2] skip $2"; return }
  timeout 900 $PY roundtrip_ours_scene.py --scene_cache $CACHE \
    --gt_traj $DUMP/$1/warp_traj.npy --gt_traj_normalized \
    --gt_logE 5.0 --init_logE 5.0 --gt_vel $3 $4 $5 --fix_E_gt --inject_pvol \
    --mpm_iter_cnt 64 --vel_iter_cnt 200 --patience 32 --min_iters 50 \
    --tag $2 > output/q2_logs/$2.log 2>&1
  echo "[q2] $2 exit=$?"
}

chain1() {
  fixv0_run  tele_v2_E1e6_y    xsim_v2_gtE6_fixv0_8f_y    0.0 -0.5 0.0
  v0only_run tele_v2_E1e5_y    xsim_v2_E1e5_v0only_y      0.0 -0.5 0.0
  v0only_run tele_v2_E1e5_vyz  xsim_v2_E1e5_v0only_vyz    0.0 -0.354 0.354
}
chain2() {
  sleep 30  # let chain1's GPU picker claim its card first
  fixv0_run  tele_v2_E1e6_vxy  xsim_v2_gtE6_fixv0_8f_vxy  -0.354 -0.354 0.0
  v0only_run tele_v2_E1e5_vxy  xsim_v2_E1e5_v0only_vxy    -0.354 -0.354 0.0
}

chain1 &
chain2 &
wait
echo "[q2] BATCH DONE"
$PY make_panel.py --per_run --runs \
  output/ours_telephone/xsim_v2_gtE6_fixv0_8f_y \
  output/ours_telephone/xsim_v2_gtE6_fixv0_8f_vxy \
  output/ours_telephone/xsim_v2_E1e5_v0only_y \
  output/ours_telephone/xsim_v2_E1e5_v0only_vxy \
  output/ours_telephone/xsim_v2_E1e5_v0only_vyz \
  --out /tmp2/b10401006/ev-project/generative-phys/reports/20260612_gic_q2/panel_q2_batch.gif

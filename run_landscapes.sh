#!/bin/zsh
# Landscape batch (2026-06-12): E1d wide+zoom (E1e4, windows 8/14), Enu (E1e4 8f
# + 2phase optimizer path overlay), v0xy (rot68 diag45 GT, 4f vel-stage window).
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
DUMP=/tmp2/b10401006/ev-project/generative-phys/outputs/explore/xmodel_dump
mkdir -p output/q2_logs

timeout 1200 $PY loss_landscape.py --mode E1d \
  --gt_traj $DUMP/tele_v2_E1e4/warp_traj.npy --gt_traj_normalized --gt_logE 4.0 \
  --windows 8 14 --grid_n 31 --logE_range 3.5 6.5 --tag E1e4_wide \
  > output/q2_logs/ls_E1e4_wide.log 2>&1; echo "[ls] wide exit=$?"

timeout 1200 $PY loss_landscape.py --mode E1d \
  --gt_traj $DUMP/tele_v2_E1e4/warp_traj.npy --gt_traj_normalized --gt_logE 4.0 \
  --windows 8 14 --grid_n 41 --logE_range 3.85 4.15 --tag E1e4_zoom \
  > output/q2_logs/ls_E1e4_zoom.log 2>&1; echo "[ls] zoom exit=$?"

timeout 2400 $PY loss_landscape.py --mode Enu \
  --gt_traj $DUMP/tele_v2_E1e4/warp_traj.npy --gt_traj_normalized --gt_logE 4.0 \
  --n_frames 8 --grid_n 21 --logE_range 3.5 4.5 \
  --overlay_run output/ours_telephone/xsim_v2_gtE4_fixv0_8f_2phase/result.json \
  --tag E1e4_8f > output/q2_logs/ls_Enu.log 2>&1; echo "[ls] Enu exit=$?"

timeout 1800 $PY loss_landscape.py --mode v0xy \
  --gt_traj $DUMP/tele_v2_rot68_d1/warp_traj.npy --gt_traj_normalized --rot_z_deg 67.6 \
  --gt_logE 5.0 --gt_vel 0.354 -0.354 0.0 --n_frames 4 --grid_n 21 \
  --tag rot68_d1_4f > output/q2_logs/ls_v0xy.log 2>&1; echo "[ls] v0xy exit=$?"

echo "[ls] ALL DONE"

#!/bin/zsh
# Landscape batch 2 (2026-06-12, user-specified): z-channel landscapes on rot68.
# 1) v0z1d  2) (v0y,v0z)  3) (logE,v0z) joint hypothesis  4) (logE,nu) under z GT
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
PYW=/tmp2/b10401006/.symlinks/miniforge3/envs/physdreamer/bin/python
GEN=/tmp2/b10401006/ev-project/generative-phys
DUMP=$GEN/outputs/explore/xmodel_dump
mkdir -p output/q2_logs

# GT dumps (warp, rot68, 14f)
cd $GEN
timeout 300 $PYW -m reuse_mpm.explore.xmodel_dump --logE 5.0 --v0 0.0 0.0 0.5 \
  --rot_z_deg 67.6 --num_frames 14 --cache_path outputs/_scene_cache/telephone_ds0.1_g32_k8.pt \
  --label tele_v2_rot68_zp14 >> /tmp2/b10401006/ev-project/gic/output/q2_logs/ls2_dumps.log 2>&1
timeout 300 $PYW -m reuse_mpm.explore.xmodel_dump --logE 5.0 --v0 0.0 -0.354 0.354 \
  --rot_z_deg 67.6 --num_frames 14 --cache_path outputs/_scene_cache/telephone_ds0.1_g32_k8.pt \
  --label tele_v2_rot68_vyz14 >> /tmp2/b10401006/ev-project/gic/output/q2_logs/ls2_dumps.log 2>&1
cd /tmp2/b10401006/ev-project/gic

timeout 1200 $PY loss_landscape.py --mode v0z1d \
  --gt_traj $DUMP/tele_v2_rot68_zp14/warp_traj.npy --gt_traj_normalized --rot_z_deg 67.6 \
  --gt_logE 5.0 --gt_vel 0.0 0.0 0.5 --windows 4 8 --grid_n 41 --vz_range -0.75 0.75 \
  --tag rot68_zp > output/q2_logs/ls2_v0z1d.log 2>&1; echo "[ls2] v0z1d exit=$?"

timeout 1800 $PY loss_landscape.py --mode vyvz \
  --gt_traj $DUMP/tele_v2_rot68_vyz14/warp_traj.npy --gt_traj_normalized --rot_z_deg 67.6 \
  --gt_logE 5.0 --gt_vel 0.0 -0.354 0.354 --n_frames 4 --grid_n 21 \
  --vy_range -0.75 0.75 --vz_range -0.75 0.75 \
  --tag rot68_vyvz_4f > output/q2_logs/ls2_vyvz.log 2>&1; echo "[ls2] vyvz exit=$?"

timeout 2400 $PY loss_landscape.py --mode Evz \
  --gt_traj $DUMP/tele_v2_rot68_zp14/warp_traj.npy --gt_traj_normalized --rot_z_deg 67.6 \
  --gt_logE 5.0 --gt_vel 0.0 0.0 0.5 --n_frames 8 --grid_n 21 \
  --logE_range 4.0 6.0 --vz_range 0.0 0.75 \
  --tag rot68_zp_8f > output/q2_logs/ls2_Evz.log 2>&1; echo "[ls2] Evz exit=$?"

timeout 2400 $PY loss_landscape.py --mode Enu \
  --gt_traj $DUMP/tele_v2_rot68_zp14/warp_traj.npy --gt_traj_normalized --rot_z_deg 67.6 \
  --gt_logE 5.0 --gt_vel 0.0 0.0 0.5 --n_frames 8 --grid_n 21 \
  --logE_range 4.5 5.5 --nu_range 0.02 0.45 \
  --tag rot68_zGT_8f > output/q2_logs/ls2_Enu.log 2>&1; echo "[ls2] Enu-zGT exit=$?"

echo "[ls2] ALL DONE"

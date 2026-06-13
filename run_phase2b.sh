#!/bin/zsh
# Phase 2b (cross-sim: our warp trajectory -> gic fit) + leftover phase 2a combo.
cd /tmp2/b10401006/ev-project/gic
export PYTHONUNBUFFERED=1
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE_2B=/tmp2/b10401006/ev-project/generative-phys/outputs/dataset_gen/04_tel_axisy_rest_T16/scene_cache.pt
TRAJ=/tmp2/b10401006/ev-project/generative-phys/outputs/dataset_gen/04_tel_axisy_rest_T16/sample_0000/mpm_xyz.npy
CACHE_2A=/tmp2/b10401006/ev-project/generative-phys/outputs/forward_gen/06_tele_E1e5/scene_cache.pt
V0Y=1.0976270078546495  # sample_0000 GT v0
FLOOR=45360  # 12.6h: leaves headroom above the 12h watchdog line

guard() {
  if [ -f /tmp2/b10401006/ev-project/QUOTA_STOP ]; then echo "[p2b] ABORT: stop flag"; exit 1; fi
  q=$(ws-status 2>/dev/null | grep -oE 'GPU quota remaining: [0-9]+' | grep -oE '[0-9]+' | head -1)
  echo "[p2b] quota ${q}s before $1"
  if [ -n "$q" ] && [ "$q" -lt $FLOOR ]; then echo "[p2b] ABORT: quota below floor"; exit 1; fi
}

# 1. cross-sim, joint v0+E (gic original recipe) from init 1e6
guard xsim_joint
$PY roundtrip_ours_scene.py --scene_cache $CACHE_2B --gt_traj $TRAJ \
    --gt_logE 5.0 --gt_vel 0.0 $V0Y 0.0 --init_logE 6.0 \
    --iter_cnt 50 --vel_iter_cnt 40 \
    --tag xsim_gtE5_initE6_joint > output/ours_telephone_xsim_joint.log 2>&1
echo "[p2b] xsim_joint exit=$?"

# 2. cross-sim, v0 fixed to GT (isolate E across the sim gap)
guard xsim_fixv0
$PY roundtrip_ours_scene.py --scene_cache $CACHE_2B --gt_traj $TRAJ \
    --gt_logE 5.0 --gt_vel 0.0 $V0Y 0.0 --init_logE 6.0 \
    --iter_cnt 50 --fix_v0_gt \
    --tag xsim_gtE5_initE6_fixv0 > output/ours_telephone_xsim_fixv0.log 2>&1
echo "[p2b] xsim_fixv0 exit=$?"

# 3. leftover phase 2a combo: stiffer GT, joint
guard tele_gtE6
$PY roundtrip_ours_scene.py --scene_cache $CACHE_2A \
    --gt_logE 6.0 --init_logE 5.0 --iter_cnt 50 --vel_iter_cnt 40 \
    --tag tele_gtE6_initE5 > output/ours_telephone_tele_gtE6.log 2>&1
echo "[p2b] tele_gtE6 exit=$?"
echo "[p2b] ALL DONE"

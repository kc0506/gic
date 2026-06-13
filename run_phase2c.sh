#!/bin/zsh
# Phase 2c: per-scene bias test — carnation (pd) + ficus (pg) in the gic harness,
# same protocol as telephone (joint v0+E, GT logE 5, init logE 6, v0=(0,-0.5,0)).
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CARN=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/carnations_ds0.1_g32_k8.pt
FICUS=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/PG_ficus_whitebg-trained_ds0.1_g32_k8.pt
TELE=/tmp2/b10401006/ev-project/generative-phys/outputs/forward_gen/06_tele_E1e5/scene_cache.pt
FLOOR=45360  # 12.6h

guard() {
  if [ -f /tmp2/b10401006/ev-project/QUOTA_STOP ]; then echo "[p2c] ABORT: stop flag"; exit 1; fi
  q=$(ws-status 2>/dev/null | grep -oE 'GPU quota remaining: [0-9]+' | grep -oE '[0-9]+' | head -1)
  echo "[p2c] quota ${q}s before $1"
  if [ -n "$q" ] && [ "$q" -lt $FLOOR ]; then echo "[p2c] ABORT: quota below floor"; exit 1; fi
}

guard carn_joint
$PY roundtrip_ours_scene.py --scene_cache $CARN \
    --gt_logE 5.0 --init_logE 6.0 --iter_cnt 50 --vel_iter_cnt 40 \
    --tag carn_gtE5_initE6_joint > output/ours_telephone_carn_joint.log 2>&1
echo "[p2c] carn_joint exit=$?"

guard ficus_joint
$PY roundtrip_ours_scene.py --scene_cache $FICUS \
    --gt_logE 5.0 --init_logE 6.0 --iter_cnt 50 --vel_iter_cnt 40 \
    --tag ficus_gtE5_initE6_joint > output/ours_telephone_ficus_joint.log 2>&1
echo "[p2c] ficus_joint exit=$?"

# stiffer-GT telephone combo, only if budget still allows and not already done
if [ ! -f output/ours_telephone/tele_gtE6_initE5/result.json ]; then
  guard tele_gtE6
  $PY roundtrip_ours_scene.py --scene_cache $TELE \
      --gt_logE 6.0 --init_logE 5.0 --iter_cnt 50 --vel_iter_cnt 40 \
      --tag tele_gtE6_initE5 > output/ours_telephone_tele_gtE6.log 2>&1
  echo "[p2c] tele_gtE6 exit=$?"
fi
echo "[p2c] ALL DONE"

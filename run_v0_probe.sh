#!/bin/zsh
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/forward_gen/06_tele_E1e5/scene_cache.pt
run() {
  if [ -f /tmp2/b10401006/ev-project/QUOTA_STOP ]; then echo "[v0probe] ABORT"; exit 1; fi
  $PY roundtrip_ours_scene.py --scene_cache $CACHE --fix_E_gt \
      --gt_logE 5.0 --init_logE 5.0 --gt_vel $1 $2 $3 \
      --tag $4 > output/ours_telephone_$4.log 2>&1
  echo "[v0probe] $4 exit=$?"
}
run 0.0 -0.5 0.0 telev0_y05
run 0.0 -1.0 0.0 telev0_y10
run 0.35 -0.35 0.0 telev0_diag
echo "[v0probe] ALL DONE"

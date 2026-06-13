#!/bin/zsh
# Phase 2d: fix-v0 E-landscape per-scene tests (carnation, ficus, telephone-gtE6).
# Single GPU, sequential. User granted 2.5h on 06/11 noon; floor 9.5h.
cd /tmp2/b10401006/ev-project/gic
export PYTHONUNBUFFERED=1
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CARN=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/carnations_ds0.1_g32_k8.pt
FICUS=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/PG_ficus_whitebg-trained_ds0.1_g32_k8.pt
TELE=/tmp2/b10401006/ev-project/generative-phys/outputs/forward_gen/06_tele_E1e5/scene_cache.pt
FLOOR=34200  # 9.5h

run_one() {
  local cache=$1 gt=$2 init=$3 tag=$4
  if [ -f "output/ours_telephone/$tag/result.json" ]; then echo "[p2d] skip $tag"; return; fi
  if [ -f /tmp2/b10401006/ev-project/QUOTA_STOP ]; then echo "[p2d] ABORT: stop flag"; exit 1; fi
  q=$(ws-status 2>/dev/null | grep -oE 'GPU quota remaining: [0-9]+' | grep -oE '[0-9]+' | head -1)
  echo "[p2d] quota ${q}s before $tag"
  if [ -n "$q" ] && [ "$q" -lt $FLOOR ]; then echo "[p2d] ABORT: quota below floor"; exit 1; fi
  $PY roundtrip_ours_scene.py --scene_cache $cache \
      --gt_logE $gt --init_logE $init --iter_cnt 50 --fix_v0_gt \
      --tag $tag > output/ours_telephone_$tag.log 2>&1
  echo "[p2d] $tag exit=$?"
}

run_one $CARN  5.0 6.0 carn_gtE5_initE6_fixv0
run_one $FICUS 5.0 6.0 ficus_gtE5_initE6_fixv0
run_one $TELE  6.0 5.0 tele_gtE6_initE5_fixv0
echo "[p2d] ALL DONE"

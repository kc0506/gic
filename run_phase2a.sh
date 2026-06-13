#!/bin/zsh
# Phase 2a: telephone-in-gic roundtrip, 3 combos, quota-guarded.
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/forward_gen/06_tele_E1e5/scene_cache.pt
FLOOR=63000  # 17.5h: each combo ~0.6h, keeps us above the 16h watchdog line

run_one() {
  local gt=$1 init=$2 tag=$3
  if [ -f "output/ours_telephone/$tag/result.json" ]; then echo "[p2a] skip $tag"; return; fi
  if [ -f /tmp2/b10401006/ev-project/QUOTA_STOP ]; then echo "[p2a] ABORT: stop flag"; exit 1; fi
  q=$(ws-status 2>/dev/null | grep -oE 'GPU quota remaining: [0-9]+' | grep -oE '[0-9]+' | head -1)
  echo "[p2a] quota ${q}s before $tag"
  if [ -n "$q" ] && [ "$q" -lt $FLOOR ]; then echo "[p2a] ABORT: quota below floor"; exit 1; fi
  $PY roundtrip_ours_scene.py --scene_cache $CACHE \
      --gt_logE $gt --init_logE $init --iter_cnt 50 --vel_iter_cnt 40 \
      --tag $tag > output/ours_telephone_$tag.log 2>&1
  echo "[p2a] $tag exit=$?"
}

mkdir -p output/ours_telephone
run_one 5.0 4.0 tele_gtE5_initE4
run_one 5.0 6.0 tele_gtE5_initE6
run_one 6.0 5.0 tele_gtE6_initE5
echo "[p2a] ALL DONE"

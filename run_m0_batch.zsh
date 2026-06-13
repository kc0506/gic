#!/bin/zsh
# M0: shared-E from 2 obs vs single-obs baselines (telephone rot68, traj, E-only).
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
run() {  # $1=tag $2=initE $3...=obs
  tag=$1; ie=$2; shift 2
  [ -f "output/ours_multiobs/$tag/result.json" ] && { echo "[m0] skip $tag"; return }
  timeout 1800 $PY multiobs_fit.py --tag $tag --init_logE $ie --obs $@ \
    > output/q2_logs/$tag.log 2>&1
  echo "[m0] $tag exit=$?"
}
# 2-obs (the amortize test), both init sides
run m0_xpym_initE4 4.0 xp ym
run m0_xpym_initE6 6.0 xp ym
# single-obs baselines (same driver, N=1) for the comparison
run m0_xp_initE4 4.0 xp
run m0_ym_initE4 4.0 ym
run m0_xp_initE6 6.0 xp
run m0_ym_initE6 6.0 ym
echo "[m0] BATCH DONE"

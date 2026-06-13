#!/bin/zsh
# image-v0 disentangling matrix (2026-06-13): direction-identifiability vs field-DOF.
# All uniform GT, E fixed at GT (1e5), RGB-only, 200 iter budget.
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python

run_one() {  # $1=tag $2=mode $3=camera $4..6=gt_vel
  tag=$1; mode=$2; cam=$3; shift 3
  [ -f "output/ours_telephone/$tag/result.json" ] && { echo "[imgd] skip $tag"; return }
  timeout 3600 $PY image_fit_ours.py --mode $mode --tag $tag \
    --gt_logE 5.0 --gt_vel $1 $2 $3 --camera $cam \
    --w_img 1.0 --w_alp 0.0 --iter_cnt 200 --min_iters 50 --patience 32 \
    --v0_field_lr 0.025 --v0_field_init_std 0.05 --param_stop_tol 0.005 \
    > output/q2_logs/$tag.log 2>&1
  echo "[imgd] $tag exit=$?"
  $PY make_panel.py --per_run --runs output/ours_telephone/$tag 2>/dev/null
}

run_one imgscalar_y_front fit_v0scalar front 0.0 -0.5 0.0
run_one imgscalar_x_front fit_v0scalar front 0.5 0.0 0.0
run_one imgfield_x_front  fit_v0field  front 0.5 0.0 0.0
run_one imgfield_y_side   fit_v0field  side_x 0.0 -0.5 0.0
echo "[imgd] BATCH DONE"

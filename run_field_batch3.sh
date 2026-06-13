#!/bin/zsh
# field-v0 batch 3 (2026-06-13): aniso 4x4x16 — uniform regression, then the two
# selected non-uniform GTs (mid_kick@E4x10, true_bend@E4x6). Serial, one GPU.
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt

run_one() {  # $1=tag $2=logE $3=init_std $4=tol $5...=extra args
  tag=$1; logE=$2; std=$3; tol=$4; shift 4
  [ -f "output/ours_telephone/$tag/result.json" ] && { echo "[f3] skip $tag"; return }
  timeout 1500 $PY roundtrip_ours_scene.py --scene_cache $CACHE \
    --gt_logE $logE --init_logE $logE \
    --fix_E_gt --inject_pvol --mpm_iter_cnt 64 --rot_z_deg 67.6 \
    --v0_field_res 4x4x16 --v0_field_init_std $std --v0_field_tv 1e-3 \
    --param_stop_tol $tol --vel_iter_cnt 200 --min_iters 50 --patience 32 \
    --overlay_every 25 --tag $tag "$@" > output/q2_logs/$tag.log 2>&1
  echo "[f3] $tag exit=$?"
  $PY make_panel.py --per_run --runs output/ours_telephone/$tag
}

run_one field_a16_uniform_y 5.0 0.05 0.005 --gt_vel 0.0 -0.5 0.0
run_one field_a16_mid_s10   4.0 0.2  0.02  --gt_v0_variant mid_kick  --gt_v0_scale 10
run_one field_a16_bend_s6   4.0 0.15 0.02  --gt_v0_variant true_bend --gt_v0_scale 6
echo "[f3] BATCH DONE"

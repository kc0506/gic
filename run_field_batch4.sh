#!/bin/zsh
# field-v0 batch 4 (2026-06-13): (1) ramp_y/ramp_x fixed-E (smooth-profile proof);
# (3) wrong-E warmup-viability test (uniform GT @E1e5, fit E pinned +-1 decade).
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt

run_one() {  # $1=tag $2...=extra args
  tag=$1; shift
  [ -f "output/ours_telephone/$tag/result.json" ] && { echo "[f4] skip $tag"; return }
  timeout 1500 $PY roundtrip_ours_scene.py --scene_cache $CACHE \
    --inject_pvol --mpm_iter_cnt 64 --rot_z_deg 67.6 \
    --v0_field_res 4x4x16 --v0_field_tv 1e-3 --param_stop_tol 0.02 \
    --vel_iter_cnt 200 --min_iters 50 --patience 32 --overlay_every 25 \
    --tag $tag "$@" > output/q2_logs/$tag.log 2>&1
  echo "[f4] $tag exit=$?"
  $PY make_panel.py --per_run --runs output/ours_telephone/$tag
}

# item 1: smooth ramps, same regime as bend (E1e4 x6, lr/std scaled to amplitude)
run_one field_a16_rampy_s6 --fix_E_gt --gt_logE 4.0 --init_logE 4.0 \
  --gt_v0_variant ramp_y --gt_v0_scale 6 --v0_field_init_std 0.15 --v0_field_lr 0.075
run_one field_a16_rampx_s6 --fix_E_gt --gt_logE 4.0 --init_logE 4.0 \
  --gt_v0_variant ramp_x --gt_v0_scale 6 --v0_field_init_std 0.15 --v0_field_lr 0.075

# item 3: wrong-E warmup test (uniform y GT @E1e5; fit E pinned wrong)
run_one field_a16_uniE5_fitE4 --fix_E_logE 4.0 --gt_logE 5.0 --init_logE 5.0 \
  --gt_vel 0.0 -0.5 0.0 --v0_field_init_std 0.05
run_one field_a16_uniE5_fitE6 --fix_E_logE 6.0 --gt_logE 5.0 --init_logE 5.0 \
  --gt_vel 0.0 -0.5 0.0 --v0_field_init_std 0.05
echo "[f4] BATCH DONE"

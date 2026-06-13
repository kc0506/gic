#!/bin/zsh
# field-v0 milestone batch 2 (2026-06-12): TV + res8 + new direction, serial on one GPU.
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=3

field_run() {  # $1=tag $2=res $3=tv $4..6=v0
  [ -f "output/ours_telephone/$1/result.json" ] && { echo "[field] skip $1"; return }
  timeout 1200 $PY roundtrip_ours_scene.py --scene_cache $CACHE \
    --gt_logE 5.0 --init_logE 5.0 --gt_vel $4 $5 $6 \
    --fix_E_gt --inject_pvol --mpm_iter_cnt 64 --rot_z_deg 67.6 \
    --v0_field_res $2 --v0_field_init_std 0.05 --v0_field_tv $3 \
    --param_stop_tol 0.005 --vel_iter_cnt 200 --min_iters 50 --patience 32 \
    --overlay_every 25 --tag $1 > output/q2_logs/$1.log 2>&1
  echo "[field] $1 exit=$?"
  $PY make_panel.py --per_run --runs output/ours_telephone/$1
}

field_run field_v0_res4_y_tv    4 1e-3  0.0 -0.5 0.0
field_run field_v0_res8_y_tv    8 1e-3  0.0 -0.5 0.0
field_run field_v0_res4_vxy_tv  4 1e-3  -0.354 -0.354 0.0
echo "[field] BATCH DONE"

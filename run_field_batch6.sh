#!/bin/zsh
# batch 6 (2026-06-13): C0-1 -- Evy landscape + scalar wrong-E reproduction.
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt

# 1) pure-gic GT dump (uniform y @E1e5, rot68, 8 frames)
if [ ! -f output/xmodel/gic_E1e5_y_rot68/gic_traj.npy ]; then
  timeout 600 $PY xmodel_dump_gic.py --label gic_E1e5_y_rot68 --cache $CACHE \
    --logE 5.0 --v0 0.0 -0.5 0.0 --rot_z_deg 67.6 --n_frames 8 \
    > output/q2_logs/gic_dump_E1e5_y_rot68.log 2>&1
  echo "[f6] dump exit=$?"
fi

# 2) Evy landscape at the vel-stage window (4f)
timeout 1800 $PY loss_landscape.py --mode Evy --tag rot68_uniy_4f \
  --gt_traj output/xmodel/gic_E1e5_y_rot68/gic_traj.npy --gt_traj_normalized \
  --rot_z_deg 67.6 --gt_logE 5.0 --n_frames 4 --grid_n 21 \
  > output/q2_logs/landscape_Evy.log 2>&1
echo "[f6] Evy landscape exit=$?"

# 3) scalar wrong-E reproduction (same protocol as the field version)
for fe in 4.0 6.0; do
  tag=scalar_uniE5_fitE${fe%%.*}
  [ -f "output/ours_telephone/$tag/result.json" ] && { echo "[f6] skip $tag"; continue }
  timeout 1500 $PY roundtrip_ours_scene.py --scene_cache $CACHE \
    --fix_E_logE $fe --gt_logE 5.0 --init_logE 5.0 --gt_vel 0.0 -0.5 0.0 \
    --inject_pvol --mpm_iter_cnt 64 --rot_z_deg 67.6 \
    --vel_iter_cnt 200 --min_iters 50 --patience 32 --param_stop_tol 0.005 \
    --overlay_every 25 --tag $tag > output/q2_logs/$tag.log 2>&1
  echo "[f6] $tag exit=$?"
  $PY make_panel.py --per_run --runs output/ours_telephone/$tag 2>/dev/null
done
echo "[f6] BATCH DONE"

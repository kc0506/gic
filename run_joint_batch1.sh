#!/bin/zsh
# J1 (2026-06-13): scalar joint v0+E, alternating vs true-joint, init E +-1 decade.
# GT uniform y @E1e5, nu fixed at GT (0.3), v0 zero-init, vel 4f / phys 8f.
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt

common=(--scene_cache $CACHE --gt_logE 5.0 --gt_nu 0.3 --gt_vel 0.0 -0.5 0.0
  --init_nu 0.3 --inject_pvol --mpm_iter_cnt 64 --rot_z_deg 67.6
  --phys_frames 8 --param_stop_tol 0.005 --overlay_every 25)

run_alt() {  # $1=initE
  tag=j1_alt_initE${1%%.*}
  [ -f "output/ours_telephone/$tag/result.json" ] && { echo "[j1] skip $tag"; return }
  timeout 2400 $PY roundtrip_ours_scene.py $common --init_logE $1 --nu_lr 0 \
    --alt_rounds 4 --alt_vel_iters 40 --alt_phys_iters 40 \
    --patience 32 --min_iters 10 --tag $tag > output/q2_logs/$tag.log 2>&1
  echo "[j1] $tag exit=$?"
  $PY make_panel.py --per_run --runs output/ours_telephone/$tag 2>/dev/null
}
run_joint() {  # $1=initE
  tag=j1_joint_initE${1%%.*}
  [ -f "output/ours_telephone/$tag/result.json" ] && { echo "[j1] skip $tag"; return }
  timeout 2400 $PY roundtrip_ours_scene.py $common --init_logE $1 --joint_v0E \
    --iter_cnt 200 --patience 32 --min_iters 50 --fine_lr 0.02 \
    --tag $tag > output/q2_logs/$tag.log 2>&1
  echo "[j1] $tag exit=$?"
  $PY make_panel.py --per_run --runs output/ours_telephone/$tag 2>/dev/null
}

run_alt 4.0
run_joint 4.0
run_alt 6.0
run_joint 6.0
echo "[j1] BATCH DONE"

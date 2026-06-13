#!/bin/zsh
# Overnight chain (2026-06-13, user-approved A+B):
#   B) image joint (smoke-gated) x2
#   A) carnations: auto excitation probe -> SOTA suite x4
# Quota floor 16h checked before every run (watchdog kills at the same floor).
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
TCACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt
CCACHE=/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/carnations_ds0.1_g32_k8.pt
LOG=output/q2_logs

quota_ok() {
  secs=$(ws-status 2>/dev/null | grep -i 'quota remaining' | grep -oE '[0-9]+' | head -1)
  [ -n "$secs" ] && [ "$secs" -ge 57600 ]
}
guard() { quota_ok || { echo "[ovn] quota floor hit before $1, stopping chain"; exit 0 } }

# ---------- B: image joint ----------
guard B-smoke
$PY image_fit_ours.py --mode fit_joint --tag imgjoint_smoke \
  --gt_logE 5.0 --gt_vel 0.0 -0.5 0.0 --init_logE 4.0 --camera front \
  --warmup_iters 5 --iter_cnt 10 --min_iters 1 --patience 99 \
  > $LOG/imgjoint_smoke.log 2>&1
if [ $? -ne 0 ] || [ ! -f output/ours_telephone/imgjoint_smoke/result.json ]; then
  echo "[ovn] B smoke FAILED -> skipping B full runs"
else
  echo "[ovn] B smoke ok"
  for ie in 4.0 6.0; do
    guard B-full
    tag=imgjoint_initE${ie%%.*}
    timeout 5400 $PY image_fit_ours.py --mode fit_joint --tag $tag \
      --gt_logE 5.0 --gt_vel 0.0 -0.5 0.0 --init_logE $ie --camera front \
      --gt_frames 14 --warmup_iters 40 --iter_cnt 200 --min_iters 50 \
      --patience 32 --param_stop_tol 0.005 > $LOG/$tag.log 2>&1
    echo "[ovn] $tag exit=$?"
    $PY make_panel.py --per_run --runs output/ours_telephone/$tag 2>/dev/null
  done
fi

# ---------- A: carnations ----------
guard A-probe
for d in xp xm yp ym; do
  case $d in
    xp) v=(0.5 0.0 0.0);; xm) v=(-0.5 0.0 0.0);;
    yp) v=(0.0 0.5 0.0);; ym) v=(0.0 -0.5 0.0);;
  esac
  [ -f output/xmodel/carn_probe_$d/gic_traj.npy ] && continue
  $PY xmodel_dump_gic.py --label carn_probe_$d --cache $CCACHE \
    --logE 5.0 --v0 $v --n_frames 8 --rot_z_deg 0 \
    > $LOG/carn_probe_$d.log 2>&1
done
"$PY" - <<'PYEOF' > $LOG/carn_probe_pick.txt 2>&1
import numpy as np, torch, json
cache = torch.load("/tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/carnations_ds0.1_g32_k8.pt",
                   map_location="cpu", weights_only=False)
xyz = cache["disc"]["sim_xyzs"]; ghost = (xyz == 0).all(dim=1)
freeze = cache["disc"]["freeze_mask"][~ghost].numpy()
dirs = {"xp": [0.5,0,0], "xm": [-0.5,0,0], "yp": [0,0.5,0], "ym": [0,-0.5,0]}
best, best_disp = None, -1
for d, v in dirs.items():
    t = np.load(f"output/xmodel/carn_probe_{d}/gic_traj.npy")
    disp = np.linalg.norm(t[-1] - t[0], axis=-1)[~freeze].mean()
    print(d, v, "free mean disp", round(float(disp), 5))
    if disp > best_disp: best, best_disp = d, disp
print("PICK", best, dirs[best], best_disp)
json.dump({"dir": best, "vel": dirs[best], "disp": float(best_disp)},
          open("output/xmodel/carn_probe_pick.json", "w"))
PYEOF
DIR=$($PY -c "import json; print(' '.join(str(x) for x in json.load(open('output/xmodel/carn_probe_pick.json'))['vel']))")
echo "[ovn] carnations excitation = $DIR"

ccommon=(--scene_cache $CCACHE --gt_logE 5.0 --gt_nu 0.3 --init_nu 0.3
  --gt_vel ${=DIR} --inject_pvol --mpm_iter_cnt 64 --rot_z_deg 0
  --param_stop_tol 0.005 --overlay_every 25 --out_root output/ours_carnations)

run_c() {  # $1=tag $2...=args
  tag=$1; shift
  guard $tag
  [ -f "output/ours_carnations/$tag/result.json" ] && { echo "[ovn] skip $tag"; return }
  timeout 2400 $PY roundtrip_ours_scene.py $ccommon --tag $tag "$@" \
    > $LOG/$tag.log 2>&1
  echo "[ovn] $tag exit=$?"
  $PY make_panel.py --per_run --runs output/ours_carnations/$tag 2>/dev/null
}

run_c carn_Efit_fixv0  --fix_v0_gt --init_logE 4.0 --nu_lr 0 --phys_frames 8 \
  --patience 16 --min_iters 25 --fine_lr 0.02
run_c carn_v0scalar    --fix_E_gt --init_logE 5.0 --vel_iter_cnt 200 \
  --patience 32 --min_iters 50
run_c carn_v0field_a16 --fix_E_gt --init_logE 5.0 --vel_iter_cnt 200 \
  --patience 32 --min_iters 50 --v0_field_res 4x4x16 --v0_field_tv 1e-3 \
  --v0_field_init_std 0.05
run_c carn_hybrid      --joint_v0E --init_logE 4.0 --vel_iter_cnt 40 \
  --iter_cnt 200 --phys_frames 8 --patience 32 --min_iters 50 --fine_lr 0.02

# ---------- morning summary ----------
"$PY" - <<'PYEOF' > $LOG/overnight_summary.txt 2>&1
import json, glob, os
print("=== overnight summary ===")
for p in ["output/ours_telephone/imgjoint_initE4", "output/ours_telephone/imgjoint_initE6",
          "output/ours_carnations/carn_Efit_fixv0", "output/ours_carnations/carn_v0scalar",
          "output/ours_carnations/carn_v0field_a16", "output/ours_carnations/carn_hybrid"]:
    f = os.path.join(p, "result.json")
    if not os.path.exists(f):
        print(os.path.basename(p), ": MISSING"); continue
    r = json.load(open(f))
    bits = [os.path.basename(p)]
    if "rel_err_E" in r: bits.append(f"E err {r['rel_err_E']:+.2%}")
    if "v0_rel_err" in r: bits.append(f"v0 err {r['v0_rel_err']:.2%}")
    vf = r.get("v0_field")
    if vf and vf.get("rel_l2_xy_best") is not None:
        bits.append(f"field xy relL2 {vf['rel_l2_xy_best']:.2%}")
    bits.append(f"wall {r.get('wall_time_s', 0):.0f}s")
    print(" | ".join(bits))
PYEOF
echo "[ovn] CHAIN DONE"

#!/bin/zsh
cd /tmp2/b10401006/ev-project/gic
PY=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
LOG=output/q2_logs
# 1) wait for bend/mid (cards 0,1) to finish
while pgrep -f "efdbl_rampE_" >/dev/null 2>&1; do sleep 30; done
echo "[circ] bend/mid done, smoking circular"
# 2) smoke circular E-only (v0 fixed uniform), res16, 8 iters
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 timeout 600 $PY efield_fit.py \
  --tag efcirc_smoke --gt_kind circular --res 16x16x16 --obs ym --gt_ramp 4.5 5.5 \
  --iter_cnt 8 --min_iters 99 > $LOG/efcirc_smoke.log 2>&1
if [ ! -f output/ours_efield/efcirc_smoke/result.json ]; then
  echo "[circ] SMOKE FAILED, aborting"; tail -5 $LOG/efcirc_smoke.log; exit 1
fi
rm -rf output/ours_efield/efcirc_smoke
echo "[circ] smoke ok, launching 2 full jobs"
# 3a) circular E + uniform v0 (FIXED, known) -> card 0 : isolates circular-E learnability
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=0 timeout 3000 $PY efield_fit.py \
  --tag efcirc_uniformv0 --gt_kind circular --res 16x16x16 --obs ym --gt_ramp 4.5 5.5 \
  --init_logE 5.0 --iter_cnt 200 > $LOG/efcirc_uniformv0.log 2>&1 &
# 3b) circular E + ramp v0 (DOUBLE non-uniform) -> card 1 (stacked on neighbour)
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 timeout 3000 $PY efield_fit.py \
  --tag efcirc_rampv0 --gt_kind circular --res 16x16x16 --v0_field --gt_v0_variant ramp_y \
  --gt_v0_scale 1.0 --obs ym --gt_ramp 4.5 5.5 --init_logE 5.0 \
  --warmup_iters 40 --iter_cnt 200 > $LOG/efcirc_rampv0.log 2>&1 &
wait
echo "[circ] both done"

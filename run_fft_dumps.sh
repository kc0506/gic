#!/bin/zsh
# FFT probing dumps (2026-06-12): long forwards for modal analysis.
# warp: z/y excitation x E{1e4,1e5} @128f/30fps; E1e6 @256f/60Hz (same sub-dt).
# gic: z @ E1e5 128f for the warp-vs-gic axial frequency delta.
GEN=/tmp2/b10401006/ev-project/generative-phys
PYW=/tmp2/b10401006/.symlinks/miniforge3/envs/physdreamer/bin/python
PYG=/tmp2/b10401006/.symlinks/miniforge3/envs/gic/bin/python
CACHE=outputs/_scene_cache/telephone_ds0.1_g32_k8.pt
LOG=/tmp2/b10401006/ev-project/gic/output/q2_logs/fft_dumps.log
cd $GEN

for spec in "4.0 0.0 0.0 0.5 fft_z_E4" "5.0 0.0 0.0 0.5 fft_z_E5" \
            "4.0 0.0 -0.5 0.0 fft_y_E4" "5.0 0.0 -0.5 0.0 fft_y_E5"; do
  parts=(${=spec})
  timeout 600 $PYW -m reuse_mpm.explore.xmodel_dump --logE $parts[1] \
    --v0 $parts[2] $parts[3] $parts[4] --rot_z_deg 67.6 --num_frames 128 --streaming \
    --cache_path $CACHE --label $parts[5] >> $LOG 2>&1
  echo "[fft] $parts[5] exit=$?"
done

for spec in "6.0 0.0 0.0 0.5 fft_z_E6" "6.0 0.0 -0.5 0.0 fft_y_E6"; do
  parts=(${=spec})
  timeout 900 $PYW -m reuse_mpm.explore.xmodel_dump --logE $parts[1] \
    --v0 $parts[2] $parts[3] $parts[4] --rot_z_deg 67.6 --num_frames 256 --streaming \
    --delta-t 0.016666666666666666 --substep 32 \
    --cache_path $CACHE --label $parts[5] >> $LOG 2>&1
  echo "[fft] $parts[5] exit=$?"
done

cd /tmp2/b10401006/ev-project/gic
timeout 1200 $PYG xmodel_dump_gic.py --logE 5.0 --v0 0.0 0.0 0.5 --rot_z_deg 67.6 \
  --n_frames 128 --label fft_z_E5_gic \
  --cache /tmp2/b10401006/ev-project/generative-phys/outputs/_scene_cache/telephone_ds0.1_g32_k8.pt \
  >> $LOG 2>&1
echo "[fft] gic z E5 exit=$?"
echo "[fft] ALL DONE"

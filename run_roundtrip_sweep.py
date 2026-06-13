# coding=utf-8
"""Sequential driver for roundtrip_sim2sim.py over a (GT E, init E) matrix.

Checks remaining GPU quota between runs and aborts below the floor.
Run from gic repo root in the gic conda env.
"""

import json
import os
import re
import subprocess
import sys
import time

PYTHON = sys.executable
QUOTA_FLOOR_SECS = 18 * 3600  # stop launching new runs below this (watchdog kills at 16h)
STOP_FLAG = "/tmp2/b10401006/ev-project/QUOTA_STOP"

SCENE_ARGS = [
    "-c", "config/pacnerf/torus.json",
    "-s", "data/pacnerf/torus",
    "-m", "output/pacnerf/torus",
]

# (gt_logE, init_logE); gt_nu=0.3, init_nu=0.1, gt_vel default everywhere
COMBOS: list = []
for gt in (4.0, 5.0, 6.0):
    for off in (-1.0, +1.0):
        COMBOS.append((gt, gt + off))


def remaining_quota_secs() -> int:
    out = subprocess.check_output(["ws-status"], text=True)
    m = re.search(r"GPU quota remaining: (\d+) secs", out)
    assert m, "cannot parse ws-status quota"
    return int(m.group(1))


def main() -> None:
    results = []
    for gt_logE, init_logE in COMBOS:
        tag = f"gtE{gt_logE:g}_initE{init_logE:g}"
        if (gt_logE, init_logE) == (6.0, 5.0):
            tag = "pilot_gtE6_initE5"  # covered by the pilot run (100/100 iters)
        out_json = f"output/pacnerf/torus/roundtrip/{tag}/result.json"
        if os.path.exists(out_json):
            print(f"[sweep] skip {tag} (already done)")
            results.append((tag, json.load(open(out_json))))
            continue
        if os.path.exists(STOP_FLAG):
            print("[sweep] ABORT: QUOTA_STOP flag present (watchdog fired)")
            break
        quota = remaining_quota_secs()
        print(f"[sweep] quota remaining {quota / 3600:.1f}h")
        if quota < QUOTA_FLOOR_SECS:
            print(f"[sweep] ABORT: quota below floor {QUOTA_FLOOR_SECS / 3600:.1f}h")
            break
        t0 = time.time()
        print(f"[sweep] === {tag} ===")
        cmd = [PYTHON, "roundtrip_sim2sim.py", *SCENE_ARGS,
               "--gt_logE", str(gt_logE), "--init_logE", str(init_logE),
               "--gt_nu", "0.3", "--init_nu", "0.1", "--tag", tag,
               "--iter_cnt", "60", "--vel_iter_cnt", "60"]
        log_path = f"output/pacnerf/torus/roundtrip/{tag}.log"
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w") as logf:
            ret = subprocess.run(cmd, stdout=logf, stderr=subprocess.STDOUT).returncode
        print(f"[sweep] {tag} exit={ret} in {(time.time() - t0) / 60:.1f}min")
        if ret == 0 and os.path.exists(out_json):
            results.append((tag, json.load(open(out_json))))

    print("\n[sweep] ===== SUMMARY =====")
    print(f"{'tag':<24}{'GT E':>10}{'best E':>12}{'rel err':>9}{'nu':>7}{'min loss':>10}")
    for tag, r in results:
        print(f"{tag:<24}{r['gt']['E']:>10.3g}{r['best']['Youngs modulus']:>12.4g}"
              f"{r['rel_err_E']:>8.1%}{r['best']['Poisson ratio']:>7.3f}{r['best']['loss']:>10.2e}")


if __name__ == "__main__":
    main()

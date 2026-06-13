# archive/

Retired entrypoints from the field/joint sys-id campaign. Kept (not deleted) so
they can be revived; their `from roundtrip_*` imports may need path fixes if run
from here.

- `profile_joint.py` — multi-instance joint cost profiler. Explicitly "NOT a
  science run"; its profiling question was answered. Revive only to re-measure
  forward+backward cost through a reused simulator.
- `joint_E_F0.py` — (E, alpha) gradient-health test, F0 = V0^alpha ridge probe.
  Retired from the gic side; F0 integration is still open work but lives on the
  reuse_mpm/warp side now. Revive if doing the gic-side F0 alpha experiment again.
- `run_roundtrip_sweep.py` — (GT E, init E) sweep driver for the phase-1
  `roundtrip_sim2sim.py` torus anchor. Phase-1 is done.

To revive: `git mv archive/<file>.py <file>.py` and re-point its imports at `ours/`.

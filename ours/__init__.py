# coding=utf-8
"""Shared library for our telephone/field sys-id experiments on the GIC harness.

Submodules (import-only, no `__main__`):
  gpu        GPU pick + quota helpers (explicit, no import side effects)
  geom       coordinate helpers (rot_xyz)
  fields     V0/E voxel fields, strain proxy, circular/profile grid fill
  scene      scene cache IO, set_params, rollout, static cache
  estimator  AnchoredEstimator (freeze-BC, field injection), bounded forward
  train      train_ours (early stop / two-phase lr / ckpt), param record
  viz        overlay/rollout gifs, field/grid/profile/param plots

Entrypoints (one training mode each) live at the gic repo root and import here.
"""

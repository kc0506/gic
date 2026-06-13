# coding=utf-8
"""DEPRECATED shim: moved to ours/fields.py during the lib/entrypoint cleanup.

Kept so existing `from v0_field_ours import ...` keeps working while callers are
migrated to `from ours.fields import ...`. Remove once no caller references it.
"""
from ours.fields import (  # noqa: F401
    EVoxelField,
    V0VoxelField,
    eval_Egrid_at,
    eval_grid_at,
    fill_profile_grid,
    variant_field,
)

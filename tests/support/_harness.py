"""Shared read-only indexes for analytical and cross-builder checks."""

from functools import lru_cache

import pytest

from sengi import bvh

from ._warp_helpers import BUILDERS, TRIANGLES


parametrize_builders = pytest.mark.parametrize("builder", BUILDERS)


@lru_cache(maxsize=None)
def build_index(builder, mesh):
    """Reuse indexes for query checks over the same collection."""
    source = TRIANGLES.indexed(mesh.vertices, mesh.faces)
    return bvh.build(TRIANGLES, source, builder=builder)


DIST_RTOL = 1e-4
DIST_ATOL = 1e-5
T_RTOL = 1e-4
T_ATOL = 1e-5

"""Layer-1 analytical fixtures.

Hand-computed scenarios with closed-form expected answers. Run before
oracle/brute-force tests as a smoke layer — if these fail, the
framework itself is broken.

Each test parametrizes over the SAH and LBVH builders.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

import jax.numpy as jnp

from sengi import queries
from sengi.metrics.nearest import Euclidean

from tests.support._harness import DIST_ATOL, DIST_RTOL, build_index, parametrize_builders


# =====================================================================
# Single-triangle fixtures with closed-form answers.
# =====================================================================


@pytest.fixture(scope="module")
def single_triangle() -> trimesh.Trimesh:
    """One triangle in z=0: vertices (0,0,0), (1,0,0), (0,1,0)."""
    v = jnp.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=jnp.float32)
    f = jnp.asarray([[0, 1, 2]], dtype=jnp.int32)
    return trimesh.Trimesh(vertices=v, faces=f, process=False)


@parametrize_builders
def test_closest_point_above_centroid_bvh(single_triangle, builder):
    """Query at (1/3, 1/3, 2): closest is (1/3, 1/3, 0), dist = 2."""
    index = build_index(builder, single_triangle)
    x = jnp.asarray([1.0 / 3.0, 1.0 / 3.0, 2.0], dtype=jnp.float32)
    cp, dist, prim_idx, _ = queries.nearest.bind(index.spec, Euclidean())(index.data, x)
    assert int(prim_idx) == 0
    np.testing.assert_allclose(
        np.asarray(cp),
        np.array([1.0 / 3.0, 1.0 / 3.0, 0.0]),
        rtol=DIST_RTOL,
        atol=DIST_ATOL,
    )
    np.testing.assert_allclose(float(dist), 2.0, rtol=DIST_RTOL, atol=DIST_ATOL)


@parametrize_builders
def test_first_hit_axis_ray_bvh(single_triangle, builder):
    """Ray from (0.25, 0.25, 5) heading in -z direction hits the
    triangle at t = 5.
    """
    index = build_index(builder, single_triangle)
    ro = jnp.asarray([0.25, 0.25, 5.0], dtype=jnp.float32)
    rd = jnp.asarray([0.0, 0.0, -1.0], dtype=jnp.float32)
    t_max = jnp.float32(100.0)
    t, prim_idx, hit, _ = queries.rays.bind(index.spec, "first")(
        index.data, ro, rd, t_max
    )
    assert bool(hit)
    assert int(prim_idx) == 0
    np.testing.assert_allclose(float(t), 5.0, rtol=DIST_RTOL, atol=DIST_ATOL)


@parametrize_builders
def test_aabb_overlap_known_bvh(single_triangle, builder):
    """AABB [(-0.1, -0.1, -0.1), (0.6, 0.6, 0.1)] clearly overlaps the
    triangle. AABB [(2, 2, 2), (3, 3, 3)] clearly does not.
    """
    index = build_index(builder, single_triangle)

    lo_in = jnp.asarray([-0.1, -0.1, -0.1], dtype=jnp.float32)
    hi_in = jnp.asarray([0.6, 0.6, 0.1], dtype=jnp.float32)
    overlap = queries.overlap.bind(index.spec)
    assert bool(overlap(index.data, lo_in, hi_in)[0])

    lo_out = jnp.asarray([2.0, 2.0, 2.0], dtype=jnp.float32)
    hi_out = jnp.asarray([3.0, 3.0, 3.0], dtype=jnp.float32)
    assert not bool(overlap(index.data, lo_out, hi_out)[0])

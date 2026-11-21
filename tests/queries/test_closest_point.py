"""Closest point against external and exhaustive references."""

from __future__ import annotations

import numpy as np

from sengi import queries
from sengi.metrics.nearest import Euclidean

from tests.support._oracle import oracle_closest_point
from tests.support._warp_helpers import _oracle_closest
from tests.support._invariants import assert_closest_point_consistent
from tests.support._harness import parametrize_builders, build_index, DIST_RTOL, DIST_ATOL


@parametrize_builders
def test_closest_point_matches_trimesh_and_numpy(tri_fixture, builder):
    index = build_index(builder, tri_fixture.mesh)
    qs = tri_fixture.query_points
    cp_t, d_t, idx_t, _ = queries.nearest.bind(index.spec, Euclidean())(index.data, qs)

    mesh = tri_fixture.mesh
    d_b = [_oracle_closest(mesh.vertices, mesh.faces, q)[1] for q in np.asarray(qs)]
    np.testing.assert_allclose(
        np.asarray(d_t),
        d_b,
        rtol=DIST_RTOL,
        atol=DIST_ATOL,
        err_msg="distances disagree with NumPy oracle",
    )

    qs_np = np.asarray(qs)
    _cp_o, d_o, _face_o = oracle_closest_point(tri_fixture.mesh, qs_np)
    np.testing.assert_allclose(
        np.asarray(d_t),
        d_o,
        rtol=DIST_RTOL,
        atol=DIST_ATOL,
        err_msg="distances disagree with trimesh oracle",
    )

    assert_closest_point_consistent(tri_fixture.mesh, qs, cp_t, d_t, idx_t)

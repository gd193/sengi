"""Silhouette distances and result invariants against exhaustive edges."""

import numpy as np

from sengi import bvh, queries

from tests.support._harness import parametrize_builders, build_index
from tests.support._warp_helpers import _oracle_silhouette


@parametrize_builders
def test_closest_silhouette_matches_oracle_and_invariants(tri_fixture, builder):
    index = build_index(builder, tri_fixture.mesh)
    points = tri_fixture.query_points
    mesh = tri_fixture.mesh
    source = index.spec.primitive.indexed(mesh.vertices, mesh.faces)
    boundary = bvh.prepare_boundary(index, source)
    query = queries.silhouette.bind(index.spec)
    cp, distance, edge, _ = map(np.asarray, query(index.data, boundary, points))
    _, expected = _oracle_silhouette(
        tri_fixture.mesh.vertices, tri_fixture.mesh.faces, points
    )
    np.testing.assert_allclose(distance, expected, rtol=5e-4, atol=5e-5)
    found = np.isfinite(expected)
    assert found.any()
    assert np.all(edge[found] >= 0) and np.all(edge[~found] == -1)
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(points)[found] - cp[found], axis=-1),
        distance[found],
        rtol=5e-4,
        atol=5e-5,
    )

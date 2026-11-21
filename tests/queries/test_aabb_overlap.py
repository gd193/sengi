"""Primitive AABB overlap against an independent NumPy reference."""

from __future__ import annotations

import numpy as np

from sengi import queries

from tests.support._oracle import oracle_aabb_overlap
from tests.support._harness import parametrize_builders, build_index


@parametrize_builders
def test_aabb_overlap_matches_numpy_oracle(tri_fixture, builder):
    index = build_index(builder, tri_fixture.mesh)
    los = tri_fixture.query_aabb_lo
    his = tri_fixture.query_aabb_hi

    out_t = queries.overlap.bind(index.spec)(index.data, los, his)[0]

    out_o = oracle_aabb_overlap(tri_fixture.mesh, np.asarray(los), np.asarray(his))
    np.testing.assert_array_equal(
        np.asarray(out_t), out_o, err_msg="aabb_overlap disagrees with trimesh"
    )

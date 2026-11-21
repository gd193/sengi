"""SAH and LBVH agree on the same query workload."""

import numpy as np
import pytest

from sengi import queries
from sengi.metrics.nearest import Euclidean

from tests.support._harness import build_index
from tests.support._warp_helpers import BUILDERS


@pytest.mark.parametrize("family", ["nearest", "rays", "overlap"])
def test_bvh_builders_agree(tri_fixture_sphere_320, family):
    fixture = tri_fixture_sphere_320
    a, b = (build_index(builder, fixture.mesh) for builder in BUILDERS)
    if family == "nearest":
        query = queries.nearest.bind(a.spec, Euclidean())
        args = (fixture.query_points,)
    elif family == "rays":
        query = queries.rays.bind(a.spec, "first")
        args = (
            fixture.query_ray_origins,
            fixture.query_ray_dirs,
            fixture.query_ray_max_t,
        )
    else:
        query = queries.overlap.bind(a.spec)
        args = (fixture.query_aabb_lo, fixture.query_aabb_hi)

    left, right = (query(index.data, *args) for index in (a, b))
    if family == "nearest":
        np.testing.assert_allclose(left[1], right[1], rtol=1e-4, atol=1e-5)
    elif family == "rays":
        np.testing.assert_array_equal(left[2], right[2])
        mask = np.asarray(left[2])
        np.testing.assert_allclose(left[0][mask], right[0][mask], rtol=1e-4, atol=1e-5)
    else:
        np.testing.assert_array_equal(left[0], right[0])
    np.testing.assert_array_equal(left[-1], right[-1])

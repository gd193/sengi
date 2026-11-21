"""Separate static specialization from the dynamic BVH data passed to JAX."""
import jax
import numpy as np
import pytest

pytest.importorskip("warp")

from sengi import bvh, queries
from sengi.primitives.points import Points


def test_bound_nearest_reuses_specialization_with_different_index_data():
    primitive = Points(2)
    config = bvh.Config(leaf_size_threshold=1)
    first = bvh.build(primitive, primitive.data([[0, 2], [5, 0]]), config=config)
    second = bvh.build(primitive, primitive.data([[0, 4], [3, 0]]),
                       builder="lbvh", config=config)

    nearest = queries.nearest.bind(first.spec)
    assert first.spec == second.spec
    assert queries.nearest.bind(second.spec) is nearest
    assert all(isinstance(leaf, jax.Array) for leaf in jax.tree.leaves(first.data))

    query = jax.jit(nearest)
    point, distance, primitive_id, status = query(first.data, np.zeros(2, np.float32))
    np.testing.assert_array_equal(point, [0, 2])
    assert (distance.item(), primitive_id.item(), status.item()) == (2, 0, 1)

    point, distance, primitive_id, status = query(second.data, np.zeros(2, np.float32))
    np.testing.assert_array_equal(point, [3, 0])
    assert (distance.item(), primitive_id.item(), status.item()) == (3, 1, 1)

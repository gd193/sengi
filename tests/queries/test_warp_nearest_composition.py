"""Static nearest/kNN composition keeps query identity and one fused call."""
from sengi import bvh
from dataclasses import replace

import jax
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from sengi.metrics.nearest import Euclidean
from sengi.primitives.points import Points
from sengi.primitives.items import TriangleItem3
from sengi.primitives.triangles import Triangles
from sengi.queries.knn import bind as bind_knn
from sengi.queries.nearest import bind as bind_nearest
from sengi.queries.candidates import candidate_functions


def test_candidate_storage_policy_is_cached_by_shape_and_output():
    ids = candidate_functions(65)
    points = candidate_functions(65, True)
    assert ids is candidate_functions(65)
    assert points is candidate_functions(65, True)
    assert ids.storage is not points.storage
    assert ids.slot is None and points.slot is not None
    assert ids.storage._fields_[0][0] == "entries"
    assert points.storage._fields_[0][0] == "slots"


def test_nearest_and_knn_output_policies_are_fused_and_cached():
    primitive = Points(2)
    index = bvh.build(primitive, primitive.data(np.array([[1., 0.], [-1., 0.]], np.float32)))
    nearest = bind_nearest(index.spec, Euclidean())
    ids = bind_knn(index.spec, 2, Euclidean())
    points = bind_knn(index.spec, 2, Euclidean(), return_points=True)
    assert nearest.kernel is bind_nearest(index.spec, Euclidean()).kernel
    assert ids.kernel is bind_knn(index.spec, 2, Euclidean()).kernel
    assert ids.kernel is not points.kernel
    assert nearest.describe()["components"] == {
        "node_access": "packed", "scheduling": "grouped",
        "deferred": "cached_bound", "selection": "single", "output": "coordinates",
    }
    assert ids.describe()["components"]["selection"] == "unrolled"
    assert ids.describe()["components"]["output"] == "ids"
    assert points.describe()["components"]["output"] == "coordinates"
    assert nearest.kernel is bind_nearest(index.spec, Euclidean()).kernel

    query = np.zeros((1, 2), np.float32)
    _, distance, item_id, status = jax.jit(nearest)(index.data, query)
    scores, ids_result, count, knn_status = jax.jit(ids)(index.data, query)
    point_scores, coordinates, point_count, point_status = jax.jit(points)(index.data, query)
    np.testing.assert_array_equal(scores[0], [1., 1.])
    np.testing.assert_array_equal(ids_result[0], [0, 1])
    np.testing.assert_array_equal(coordinates[0], [[1., 0.], [-1., 0.]])
    np.testing.assert_array_equal(point_scores, scores)
    np.testing.assert_array_equal(distance, scores[:, 0])
    np.testing.assert_array_equal(item_id, ids_result[:, 0])
    np.testing.assert_array_equal([status[0], knn_status[0], point_status[0]], 1)
    np.testing.assert_array_equal([count[0], point_count[0]], 2)
    for composed in (nearest, ids, points):
        hlo = jax.jit(composed).lower(index.data, query).as_text()
        assert hlo.count("stablehlo.custom_call") == 1


def test_triangle_fused_nearest_respects_replaced_geometry_and_metric():
    primitive = Triangles()
    metric = Euclidean()

    @wp.func
    def first_vertex(query: wp.vec3, item: TriangleItem3):
        return item.a

    @wp.func
    def weighted_vertex(query: wp.vec3, item: TriangleItem3):
        return item.a, 4.0 * wp.length_sq(query - item.a)

    class CustomMetric:
        def bind(self, primitive):
            return replace(metric.bind(primitive), closest=weighted_vertex, score_at=None, point_at=None)

    raw = primitive.indexed([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], [[0, 1, 2]])
    query = np.array([.25, .25, 0.], np.float32)
    # The slot callbacks take precedence until a provider opts out of both.
    modified = replace(primitive, closest_euclidean=first_vertex)
    for provider, policy, point, distance in (
        (modified, metric, query, 0.),
        (replace(modified, score_euclidean_at=None, closest_euclidean_at=None),
         metric, [0., 0., 0.], np.linalg.norm(query)),
        (primitive, CustomMetric(), [0., 0., 0.], 2 * np.linalg.norm(query)),
    ):
        index = bvh.build(provider, raw)
        nearest = bind_nearest(index.spec, policy)
        assert nearest.ops.bind_fused(nearest.ops) is None
        actual = nearest(index.data, query)
        np.testing.assert_array_equal(actual[0], point)
        np.testing.assert_allclose(actual[1:], [distance, 0, 1])

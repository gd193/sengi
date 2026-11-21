"""Node access, traversal scheduling and bound caching vary independently."""
from sengi import bvh
from itertools import product

import jax
import numpy as np
import pytest

wp = pytest.importorskip("warp")
from sengi.metrics.nearest import Euclidean, NearestCallbacks
from sengi.primitives.items import PointItem3
from sengi.primitives.points import Points
from sengi.primitives.segments import Segments
from sengi.queries.knn import bind as bind_knn
from sengi.queries.nearest import bind as bind_nearest
from sengi.bvh import Config  # noqa: E402
from tests.support._oracle import assert_knn_ids, knn_reference  # noqa: E402

OPTIONS = list(product((False, True), repeat=3))


@wp.func
def evaluate(query: wp.vec3, item: PointItem3):
    delta = query - item.a
    return item.a, .25 * delta[0] * delta[0] + 4.0 * delta[1] * delta[1] + delta[2] * delta[2]


@wp.func
def bound(query: wp.vec3, lo: wp.vec3, hi: wp.vec3):
    delta = wp.max(wp.max(lo - query, wp.vec3(0.0)), query - hi)
    return .25 * delta[0] * delta[0] + 4.0 * delta[1] * delta[1] + delta[2] * delta[2]


@pytest.fixture(scope="module")
def points_scene():
    rng = np.random.default_rng(308)
    points = rng.integers(-4, 5, (93, 3)).astype(np.float32)
    queries = rng.integers(-3, 4, (12, 3)).astype(np.float32)
    queries[-1] = np.nan
    caps = np.array([0, 1, 4, 16, np.inf, 0, 2, 8, 32, -1, np.nan, np.inf], np.float32)
    primitive = Points(3)
    index = bvh.build(primitive, primitive.data(points), config=Config(leaf_size_threshold=1))
    scores = np.sum((queries[:, None] - points) ** 2 * [.25, 4., 1.], axis=-1)
    return index, points, queries, caps, scores


@pytest.mark.parametrize("cached,packed,grouped", OPTIONS)
@pytest.mark.parametrize("k", [1, 5, 65])
def test_all_traversal_combinations(points_scene, cached, packed, grouped, k):
    index, points, queries, caps, scores = points_scene
    options = dict(cache_bounds=cached, packed_nodes=packed, group_leaves=grouped)
    if k == 1:
        call = bind_nearest(index.spec, evaluate=evaluate, lower_bound=bound, **options)
    else:
        call = bind_knn(index.spec, k, evaluate=evaluate, lower_bound=bound, **options)
    run = jax.jit(jax.vmap(call, in_axes=(None, 0, 0)))
    result = tuple(np.asarray(v) for v in run(index.data, queries, caps))
    expected = knn_reference(scores, k, caps)
    expected[-1][-3:] = -1
    if k == 1:
        valid = expected[2] > 0
        assert_knn_ids(scores, result[2][:, None], expected[0])
        np.testing.assert_array_equal(result[0][valid], points[result[2][valid]])
        np.testing.assert_array_equal(result[1], expected[0][:, 0])
        np.testing.assert_array_equal(result[3], expected[3])
    else:
        assert_knn_ids(scores, result[1], expected[0])
        for i in (0, 2, 3):
            np.testing.assert_array_equal(result[i], expected[i])
    assert run.lower(index.data, queries, caps).as_text().count("stablehlo.custom_call") == 1


@pytest.mark.parametrize("cached,packed,grouped", OPTIONS)
def test_segment_scheduling_override(cached, packed, grouped):
    primitive = Segments()
    x = np.array([3., -2., 1., 0., 5.], np.float32)
    a = np.column_stack((x, np.zeros((5, 2), np.float32)))
    index = bvh.build(primitive, primitive.data(a, a + [0, 1, 0]), config=Config(leaf_size_threshold=1))
    queries = np.array([[.2, .25, .25], [-1.2, .1, .2]], np.float32)
    scores = np.sqrt((queries[:, None, 0] - x)**2 + queries[:, None, 2]**2)
    call = bind_knn(index.spec, 3, Euclidean(), cache_bounds=cached,
                    packed_nodes=packed, group_leaves=grouped)
    result = jax.jit(call)(index.data, queries)
    expected = knn_reference(scores, 3)
    np.testing.assert_allclose(result[0], expected[0], rtol=2e-6)
    for actual, wanted in zip(result[1:], expected[1:]):
        np.testing.assert_array_equal(actual, wanted)


@pytest.mark.parametrize("factory", [bind_nearest, bind_knn])
def test_options_forward_independently_and_defaults_share_cache(monkeypatch, factory):
    from sengi.queries import knn as _warp_knn
    from sengi.queries import nearest as _warp_nearest

    module = _warp_nearest if factory is bind_nearest else _warp_knn
    original = module.make_walk
    forwarded = []

    def record(*args, **kwargs):
        forwarded.append((kwargs["resume"] is None, kwargs["packed_nodes"], kwargs["group_leaves"]))
        return original(*args, **kwargs)

    monkeypatch.setattr(module, "make_walk", record)
    primitive = Points(3)
    metric = NearestCallbacks(evaluate, bound)
    spec = bvh.IndexSpec(primitive)
    args = (spec, metric) if factory is bind_nearest else (spec, 5, metric)
    kernels = []
    for cached, packed, grouped in OPTIONS:
        kernels.append(factory(*args, block_dim=96, cache_bounds=cached,
                               packed_nodes=packed, group_leaves=grouped).kernel)
    assert forwarded == OPTIONS
    assert len({id(kernel) for kernel in kernels}) == 8
    assert factory(*args) is factory(*args, cache_bounds=True, packed_nodes=True, group_leaves=True)
    assert factory(*args, cache_bounds=False) is factory(
        *args, cache_bounds=False, packed_nodes=True, group_leaves=True)
    spec = bvh.IndexSpec(Segments())
    args = (spec, Euclidean()) if factory is bind_nearest else (spec, 5, Euclidean())
    assert factory(*args) is factory(*args, cache_bounds=True, packed_nodes=True, group_leaves=False)

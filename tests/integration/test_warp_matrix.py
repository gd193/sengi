"""Every supported builtin combination against independent geometry oracles."""
from sengi import bvh
from functools import partial
from dataclasses import replace

import jax
import numpy as np
import pytest
import trimesh
from scipy.optimize import minimize_scalar

wp = pytest.importorskip("warp")

from benchmarks.adapters.sengi import (  # noqa: E402
    METRICS, PRIMITIVES, RAY_MODES, bound_query, kernel_inputs, prepare, prepare_boundary,
)
from benchmarks.adapters._timing import bind_kernel  # noqa: E402
from benchmarks.cases import BOUNDARIES, Case  # noqa: E402
from benchmarks.workloads import BUILTIN_MODES, builtin_geometry, make_inputs  # noqa: E402
from sengi.boundary.robin import prepare_robin  # noqa: E402
from sengi.boundary.sampling import prepare_boundary_sample  # noqa: E402
from tests.support._robin_oracle import brute_robin_radius  # noqa: E402


CASES = tuple(
    Case(query, name, "16", workload="builtin", batch_size=19, metric=metric,
         parameters={"max_hits": 4} if query == "all_hits" else {})
    for group in BUILTIN_MODES
    for name in group["geometry"]
    for query in group["query"]
    for metric in group.get("metric", ("euclidean",))
)


def _case_id(case):
    return f"{case.geometry}/{case.query}/{case.metric}"


def _public_arguments(case, index, boundary, inputs, sampling, robin):
    query = case.query
    points = inputs["queries"] if case.metric == "angular" else inputs.get("points")
    if query in RAY_MODES:
        kwargs = {"min_t": inputs["min_t"], "max_t": inputs["max_t"]}
        if query in ("count_hits", "all_hits"):
            kwargs["exclude_face"] = inputs["excluded"]
        if query == "all_hits":
            kwargs["max_hits"] = case.parameters["max_hits"]
        return (index.data, inputs["origins"], inputs["directions"]), kwargs
    if query == "aabb_overlap":
        return (index.data, inputs["lower"], inputs["upper"]), {}
    if query == "closest_point":
        return (index.data, points, inputs["radii"]), {}
    if query == "classify_point":
        return (boundary, points), {}
    if query == "sample_boundary":
        return (sampling, points, inputs["radii"], inputs["randoms"]), {}
    return (robin if query == "robin_radius" else boundary, points, inputs["radii"]), {}


def _angle(point, origin, direction):
    view = point-origin
    if not np.any(view):
        return np.inf
    cross = np.cross(np.pad(view, (0, 3-len(view))), np.pad(direction, (0, 3-len(direction))))
    return np.arctan2(np.linalg.norm(cross), view @ direction)


def _segments(raw):
    a = np.asarray(raw.a, np.float64)
    return a, np.asarray(raw.b, np.float64) if hasattr(raw, "b") else a


def _segment_point(a, b, origin, direction=None):
    edge = b-a
    if direction is None:
        length2 = np.sum(edge*edge, axis=-1)
        t = np.clip(np.sum((origin-a)*edge, axis=-1)/np.where(length2 > 0, length2, 1.), 0, 1)
        return a+np.expand_dims(t, -1)*edge
    if np.array_equal(a, b):
        return a
    def objective(t):
        return _angle(a+t*edge, origin, direction)
    optimum = minimize_scalar(objective, bounds=(0, 1), method="bounded", options={"xatol": 1e-12})
    return a+min((0., 1., optimum.x), key=objective)*edge


def _candidates(case, raw, query):
    angular = case.metric == "angular"
    origin, direction = query if angular else (query, None)
    if case.geometry == "triangles":
        triangles = np.asarray(raw.vertices, np.float64)[np.asarray(raw.indices)]
        points = trimesh.triangles.closest_point(triangles, np.broadcast_to(origin, (len(triangles), 3)))
    else:
        a, b = _segments(raw)
        points = (np.asarray([_segment_point(start, end, origin, direction) for start, end in zip(a, b, strict=True)])
                  if angular else _segment_point(a, b, origin))
    distances = (np.array([_angle(p, origin, direction) for p in points]) if angular
                 else np.linalg.norm(points-origin, axis=1))
    return points, distances


def _assert_member(case, raw, point, index):
    _, distances = _candidates(replace(case, metric="euclidean"), raw, point.astype(np.float64))
    assert 0 <= index < len(distances) and distances[index] < 8e-6


def _feature_distances(boundary, query, metric):
    origin, direction = query if metric == "angular" else (query, None)
    origin = np.pad(origin, (0, 3-len(origin)))
    if direction is not None:
        direction = np.pad(direction, (0, 3-len(direction)))
    meta = boundary.silhouette
    distances = np.full(len(meta.feature_a), np.inf)
    for i, (a, b, n0, n1) in enumerate(zip(*(np.asarray(x, np.float64) for x in
                              (meta.feature_a, meta.feature_b, meta.feature_n0, meta.feature_n1)), strict=True)):
        if ((origin-a) @ n0)*((origin-a) @ n1) > 0:
            continue
        point = _segment_point(a, b, origin, direction)
        distances[i] = np.linalg.norm(point-origin) if direction is None else _angle(point, origin, direction)
    return distances


def _robin2(raw, boundary, point, cap):
    result = min(cap, _feature_distances(boundary, point, "euclidean").min(initial=np.inf))
    for a, b in zip(*_segments(raw), strict=True):
        edge = b-a
        height = abs((point-a) @ np.array([edge[1], -edge[0]]))/np.linalg.norm(edge)
        def radius(t):
            distance2 = np.sum((a+t*edge-point)**2)
            if distance2 == 0:
                return 0.
            return np.sqrt(distance2)*np.exp(min(height/distance2, 700.))
        optimum = minimize_scalar(radius, bounds=(0, 1), method="bounded")
        result = min(result, radius(0), radius(1), optimum.fun)
    return result


@pytest.mark.parametrize("case,builder", [
    *[pytest.param(case, "sah", id=_case_id(case)) for case in CASES],
    *[pytest.param(Case("closest_point", name, "16", workload="builtin", batch_size=19), "lbvh", id=f"{name}/nearest/lbvh")
      for name in ("points3", "segments2", "triangles")],
])
def test_builtin_query_matrix(case, builder):
    source = builtin_geometry(case.geometry, case.size)
    index, raw = prepare(source, builder=builder)
    boundary = prepare_boundary(index, raw, {case.query})
    inputs = make_inputs(case, source)
    sampling = prepare_boundary_sample(boundary) if case.query == "sample_boundary" else None
    robin = prepare_robin(boundary, 1.) if case.query == "robin_radius" else None
    args, kwargs = _public_arguments(case, index, boundary, inputs, sampling, robin)
    if case.query == "aabb_overlap":
        lower, upper = (array.copy() for array in args[1:])
        lower[-2], lower[-1], upper[-1] = np.nan, 2., 1.
        args = (args[0], lower, upper)
    got = tuple(map(np.asarray, jax.jit(partial(bound_query(case, index.spec), **kwargs))(*args)))
    call, flat, shapes = kernel_inputs(case, index, inputs, boundary=boundary, sampling=sampling, robin=robin)
    device = wp.device_from_jax(jax.devices()[0])
    timed = bind_kernel(call.kernel, flat, case.batch_size, device, call.block_dim, shapes)
    timed()
    direct = tuple(v.numpy() for v in timed.outputs)
    for a, b in zip(direct, got, strict=True):
        if case.query == "aabb_overlap":
            a, b = a[:-2], b[:-2]
        if np.issubdtype(a.dtype, np.floating):
            np.testing.assert_allclose(a, b, atol=5e-6, rtol=3e-5)
        else:
            np.testing.assert_array_equal(a, b)
    primitive = PRIMITIVES[case.geometry]
    if case.query == "closest_point":
        for i, query in enumerate(np.asarray(args[1], np.float64)):
            candidates, distances = _candidates(case, raw, query)
            expected = distances.min()
            found = expected <= float(args[2][i])
            assert got[3][i] == int(found)
            if found:
                index = got[2][i]
                assert 0 <= index < len(candidates)
                np.testing.assert_allclose(got[1][i], expected, atol=5e-6, rtol=3e-5)
                # Angular minima can tie across primitives or along a whole segment.
                a, b = (candidates[index], candidates[index]) if case.geometry == "triangles" else tuple(v[index] for v in _segments(raw))
                np.testing.assert_allclose(got[0][i], _segment_point(a, b, got[0][i]), atol=8e-6, rtol=3e-5)
                score = _angle(got[0][i], *query) if case.metric == "angular" else np.linalg.norm(got[0][i]-query)
                np.testing.assert_allclose(score, expected, atol=5e-6, rtol=3e-5)
            else:
                assert np.isposinf(got[1][i]) and got[2][i] == -1
    elif case.query == "aabb_overlap":
        lo, hi = primitive.bounds(raw)
        lower, upper = map(np.asarray, args[1:])
        hits = np.any(np.all((lo[None] <= upper[:, None]) & (hi[None] >= lower[:, None]), axis=-1), axis=1)
        np.testing.assert_array_equal(got[0], hits)
        valid = np.isfinite(lower).all(axis=1) & np.all(lower <= upper, axis=1)
        np.testing.assert_array_equal(got[1], np.where(valid, hits.astype(int), -1))
    elif case.query in RAY_MODES:
        origin, direction = map(np.asarray, args[1:])
        a, b = (-1-origin)/direction, (1-origin)/direction
        enter, leave = np.min(np.stack((a, b)), axis=0).max(axis=1), np.max(np.stack((a, b)), axis=0).min(axis=1)
        times = np.column_stack((enter, leave))
        lower = np.broadcast_to(np.asarray(kwargs.get("min_t", 0.)), (len(origin),))
        valid = (enter <= leave)[:, None] & (times >= lower[:, None]) & (times <= 10.)
        times = np.sort(np.where(valid, times, np.inf), axis=1)
        count = valid.sum(axis=1)
        np.testing.assert_array_equal(got[-1], (count > 0).astype(int))
        if case.query == "first_hit":
            np.testing.assert_allclose(got[0], np.minimum(times[:, 0], 10.), atol=5e-6)
            np.testing.assert_array_equal(got[2], count > 0)
            np.testing.assert_array_equal(got[1] >= 0, count > 0)
        elif case.query == "all_hits":
            np.testing.assert_array_equal(got[2], count)
            np.testing.assert_allclose(np.sort(got[0], axis=1), np.pad(times, ((0, 0), (0, 2)), constant_values=np.inf), atol=5e-6)
            np.testing.assert_array_equal((got[1] >= 0).sum(axis=1), count)
        else:
            np.testing.assert_array_equal(got[0], count > 0 if case.query == "any_hit" else count)
        if case.query in ("first_hit", "all_hits"):
            for o, d, ts, ids in zip(origin, direction, got[0], got[1], strict=True):
                for t, index in zip(np.atleast_1d(ts), np.atleast_1d(ids), strict=True):
                    if index >= 0:
                        _assert_member(case, raw, o+t*d, index)
                    else:
                        assert index == -1
    elif case.query == "classify_point":
        points = np.asarray(args[1])
        expected = np.where(np.max(np.abs(points), axis=1) < 1, -1., 1.)
        np.testing.assert_array_equal(got, [expected, np.ones(len(points))])
    elif case.query in ("closest_silhouette", "star_radius"):
        queries = np.asarray(args[1], np.float64)
        distances = np.array([_feature_distances(boundary, q, case.metric) for q in queries])
        expected = distances.min(axis=1, initial=np.inf)
        caps = np.asarray(args[2])
        if case.query == "star_radius":
            np.testing.assert_allclose(got[0], np.minimum(expected, caps), atol=5e-6, rtol=3e-5)
            np.testing.assert_array_equal(got[1], 1)
        else:
            np.testing.assert_allclose(got[1], np.where(expected <= caps, expected, np.inf), atol=5e-6, rtol=3e-5)
            np.testing.assert_array_equal(got[3], (expected <= caps).astype(int))
            meta = boundary.silhouette
            a, b = (np.asarray(v)[:, :primitive.dimension] for v in (meta.feature_a, meta.feature_b))
            for i, index in enumerate(got[2]):
                if got[3][i] == 1:
                    assert 0 <= index < len(a)
                    np.testing.assert_allclose(distances[i, index], expected[i], atol=5e-6, rtol=3e-5)
                    np.testing.assert_allclose(got[0][i], _segment_point(a[index], b[index], got[0][i]), atol=8e-6)
                    score = _angle(got[0][i], *queries[i]) if case.metric == "angular" else np.linalg.norm(got[0][i]-queries[i])
                    np.testing.assert_allclose(score, expected[i], atol=5e-6, rtol=3e-5)
                else:
                    assert index == -1
    elif case.query == "robin_radius":
        points, caps = map(np.asarray, args[1:])
        expected = [(_robin2(raw, boundary, q.astype(np.float64), float(cap)) if primitive.dimension == 2
                     else brute_robin_radius(np.asarray(raw.vertices), np.asarray(raw.indices), q.astype(np.float64), 1., float(cap)))
                    for q, cap in zip(points, caps, strict=True)]
        np.testing.assert_allclose(got[0], expected, atol=3e-5, rtol=3e-5)
        np.testing.assert_array_equal(got[1], 1)
    else:
        np.testing.assert_array_equal(got[4], 1)
        np.testing.assert_allclose(got[2], 1/np.asarray(boundary.core.primitive_measure).sum(), rtol=3e-6)
        np.testing.assert_allclose(np.linalg.norm(got[1], axis=1), 1., atol=3e-6)
        for point, index in zip(got[0], got[3], strict=True):
            _assert_member(case, raw, point, index)


def test_unsupported_builtin_combinations_fail_explicitly():
    assert len({_case_id(case) for case in CASES}) == len(CASES)
    for name, primitive in PRIMITIVES.items():
        missing = ([Case("closest_point", name, "16", metric="angular")] if primitive.segment_endpoints is None else [])
        if primitive.intersect is None:
            missing += [Case(q, name, "16") for q in RAY_MODES]
        if name not in BOUNDARIES:
            missing += [Case(q, name, "16") for q in ("closest_silhouette", "star_radius", "robin_radius", "classify_point", "sample_boundary")]
        for case in missing:
            with pytest.raises(TypeError):
                bound_query(case, bvh.IndexSpec(primitive))


@pytest.mark.parametrize("case", CASES, ids=_case_id)
def test_equivalent_static_policies_reuse_kernel(case):
    from sengi.queries.domain import RayOps

    from sengi.queries.nearest import bind as bind_nearest
    from sengi.queries.rays import bind as bind_rays
    from sengi.queries.star_radius import bind as bind_star_radius
    from sengi.queries.silhouette import bind as bind_silhouette
    from sengi.queries.classification import bind as bind_classification
    from sengi.queries.boundary_sample import bind as bind_boundary_sample
    from sengi.queries.robin_radius import bind as bind_robin_radius
    from sengi.queries.overlap import bind as bind_overlap
    from sengi.boundary.data import resolve_boundary_adapter

    class DelegatingMetric:
        def bind(self, primitive):
            return metric.bind(primitive)

    mode, primitive, metric = case.query, PRIMITIVES[case.geometry], METRICS[case.metric]
    default = bound_query(case, bvh.IndexSpec(primitive))
    if mode == "closest_point":
        explicit = bind_nearest(bvh.IndexSpec(primitive), DelegatingMetric())
    elif mode == "closest_silhouette":
        explicit = bind_silhouette(bvh.IndexSpec(primitive), DelegatingMetric())
    elif mode == "star_radius":
        default = default.silhouette
        explicit = bind_star_radius(bvh.IndexSpec(primitive), adapter=resolve_boundary_adapter(primitive)).silhouette
    elif mode == "classify_point":
        explicit = bind_classification(bvh.IndexSpec(primitive), resolve_boundary_adapter(primitive).classification(primitive))
    elif mode == "sample_boundary":
        explicit = bind_boundary_sample(bvh.IndexSpec(primitive), primitive.boundary_adapter.measure(primitive))
    elif mode == "robin_radius":
        explicit = bind_robin_radius(bvh.IndexSpec(primitive), primitive.boundary_adapter.robin(primitive))
    elif mode == "aabb_overlap":
        explicit = bind_overlap(bvh.IndexSpec(replace(primitive, name="same_math")))
    else:
        explicit = bind_rays(bvh.IndexSpec(primitive), RAY_MODES[mode], ops=RayOps(intersect=primitive.intersect))
    assert default.kernel is explicit.kernel

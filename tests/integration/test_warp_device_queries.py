"""Public query functions compose inside caller-owned Warp kernels."""

from sengi import bvh
import jax
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from benchmarks._warp_cases import scene  # noqa: E402
from sengi.bvh.device_tree import DeviceTree
from sengi.metrics.nearest import Euclidean
from sengi.primitives.items import PointItem3
from sengi.primitives.points import Points
from sengi.warp.device import configure_query_kernel
from sengi.queries.boundary_sample import bind as bind_boundary_sample
from sengi.queries.classification import bind as bind_classification
from sengi.queries.knn import bind as bind_knn
from sengi.queries.nearest import bind as bind_nearest
from sengi.queries.overlap import bind as bind_overlap
from sengi.queries.rays import bind as bind_rays
from sengi.queries.robin_radius import bind as bind_robin_radius
from sengi.queries.silhouette import bind as bind_silhouette
from sengi.queries.star_radius import bind as bind_star_radius
from sengi.boundary.sampling import prepare_boundary_sample
from sengi.boundary.robin import prepare_robin
from tests.support._warp_device import device_payload  # noqa: E402


def _device():
    return wp.device_from_jax(jax.devices()[0])


@wp.func
def _weighted(query: wp.vec3, item: wp.vec3):
    delta = query - item
    return item, delta[0] * delta[0] + 2.0 * delta[1] * delta[1] + 4.0 * delta[
        2
    ] * delta[2]


@wp.func
def _weighted_item(query: wp.vec3, item: PointItem3):
    return _weighted(query, item.a)


@wp.func
def _bound(query: wp.vec3, lower: wp.vec3, upper: wp.vec3):
    delta = wp.max(lower - query, wp.max(query - upper, wp.vec3(0.0)))
    return delta[0] * delta[0] + 2.0 * delta[1] * delta[1] + 4.0 * delta[2] * delta[2]


@pytest.mark.parametrize("k", [4, 16, 64, 65])
@pytest.mark.parametrize("coordinates", [False, True])
def test_nearest_and_static_knn_inside_custom_kernel(k, coordinates, tmp_path):
    primitive = Points()
    rng = np.random.default_rng(92)
    index = bvh.build(
        primitive, primitive.data(rng.normal(size=(80, 3)).astype(np.float32))
    )
    tree, geometry = index.data
    queries = np.array(
        [[0.0, 0.0, 0.0], [1.0, 2.0, 3.0], [100.0, 0.0, 0.0], [np.nan, 0.0, 0.0]],
        np.float32,
    )
    cap = 20.0
    nearest = bind_nearest(index.spec, evaluate=_weighted_item, lower_bound=_bound)
    knn = bind_knn(
        index.spec,
        k,
        evaluate=_weighted_item,
        lower_bound=_bound,
        return_points=coordinates,
    )
    other = bind_knn(index.spec, 3, Euclidean())
    nearest_fn, knn_fn, other_fn = nearest.device, knn.device, other.device
    Geometry = primitive.data_type
    Output = wp.vec3 if coordinates else wp.int32

    @wp.func
    def evaluate(
        point: wp.vec3,
        tree: DeviceTree,
        geometry: Geometry,
        values: wp.array2d(dtype=float),
        neighbors: wp.array2d(dtype=Output),
        other_values: wp.array2d(dtype=float),
        other_ids: wp.array2d(dtype=int),
        row: int,
    ):
        closest, distance, item, nearest_status = wp.static(nearest_fn)(
            point, cap, tree, geometry
        )
        count, status = wp.static(knn_fn)(
            point, cap, tree, geometry, values, neighbors, row
        )
        _, _ = wp.static(other_fn)(
            point, cap, tree, geometry, other_values, other_ids, row
        )
        # Consume the scalar result in the same kernel, with no intermediate array.
        return closest * 2.0, distance, item, nearest_status, count, status

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        tree: DeviceTree,
        geometry: Geometry,
        points: wp.array(dtype=wp.vec3),
        values: wp.array2d(dtype=float),
        neighbors: wp.array2d(dtype=Output),
        other_values: wp.array2d(dtype=float),
        other_ids: wp.array2d(dtype=int),
        closest: wp.array(dtype=wp.vec3),
        distances: wp.array(dtype=float),
        ids: wp.array(dtype=int),
        near_status: wp.array(dtype=int),
        counts: wp.array(dtype=int),
        statuses: wp.array(dtype=int),
    ):
        row = wp.tid()
        p, d, i, ns, count, status = evaluate(
            points[row], tree, geometry, values, neighbors, other_values, other_ids, row
        )
        closest[row] = p
        distances[row] = d
        ids[row] = i
        near_status[row] = ns
        counts[row] = count
        statuses[row] = status

    header = tmp_path / "user_query.h"
    header.write_text("#pragma once\nnamespace user_query { constexpr int value=7; }\n")
    own = wp.ModuleBuildOptions(
        extra_cpu_include_dirs=[tmp_path],
        extra_cuda_include_dirs=[tmp_path],
        extra_cpu_preamble='#include "user_query.h"',
        extra_cuda_preamble='#include "user_query.h"',
        extra_build_dependencies=[header],
    )
    wp.set_module_options(
        {"extra_build_options": own, "fast_math": False}, module=kernel.module
    )
    configure_query_kernel(kernel, nearest, knn, other)
    first = kernel.module.get_module_hash()
    configure_query_kernel(kernel, nearest, knn, other)
    assert first == kernel.module.get_module_hash()
    configured = wp.get_module_options(kernel.module)
    assert not configured["fast_math"]
    assert header in configured["extra_build_options"].extra_build_dependencies
    assert (
        '#include "user_query.h"'
        in configured["extra_build_options"].extra_cuda_preamble
    )
    assert configured["max_unroll"] >= min(k, 64)

    device = _device()
    dt, dg = device_payload(DeviceTree, tree, device), device_payload(Geometry, geometry, device)
    points = wp.array(queries, dtype=wp.vec3, device=device)
    outs = [
        wp.empty((4, k), dtype=float, device=device),
        wp.empty((4, k), dtype=Output, device=device),
        wp.empty((4, 3), dtype=float, device=device),
        wp.empty((4, 3), dtype=int, device=device),
        wp.empty(4, dtype=wp.vec3, device=device),
        wp.empty(4, dtype=float, device=device),
        *[wp.empty(4, dtype=int, device=device) for _ in range(4)],
    ]
    wp.launch(kernel, dim=4, inputs=[dt, dg, points], outputs=outs, device=device)
    expected = knn(index.data, queries, cap)
    for actual, reference in zip(
        [outs[0], outs[1], outs[-2], outs[-1]], expected, strict=True
    ):
        np.testing.assert_array_equal(actual.numpy(), reference)
    expected_near = nearest(index.data, queries, cap)
    np.testing.assert_array_equal(outs[4].numpy(), np.asarray(expected_near[0]) * 2)
    for actual, reference in zip(outs[5:8], expected_near[1:], strict=True):
        np.testing.assert_array_equal(actual.numpy(), reference)
    for actual, reference in zip(
        outs[2:4], other(index.data, queries, cap)[:2], strict=True
    ):
        np.testing.assert_array_equal(actual.numpy(), reference)


@pytest.mark.parametrize("mode", ["first", "any", "count", "all"])
def test_ray_device_modes_match_host(mode):
    index = scene("triangles", 64).index
    primitive = index.spec.primitive
    tree, geometry = index.data
    query = bind_rays(index.spec, mode, exclude=True)
    ray = query.device
    Geometry = primitive.data_type
    if mode in ("first", "all"):
        Times = wp.array2d(dtype=float) if mode == "all" else wp.array(dtype=float)
        Faces = wp.array2d(dtype=int) if mode == "all" else wp.array(dtype=int)

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            tree: DeviceTree,
            geometry: Geometry,
            origins: wp.array(dtype=wp.vec3),
            directions: wp.array(dtype=wp.vec3),
            times: Times,
            faces: Faces,
            counts: wp.array(dtype=int),
            statuses: wp.array(dtype=int),
        ):
            i = wp.tid()
            wp.static(ray)(
                origins[i],
                directions[i],
                0.0,
                10.0,
                -1,
                tree,
                geometry,
                times,
                faces,
                counts,
                statuses,
                i,
            )
    else:

        @wp.kernel(module="unique", enable_backward=False)
        def kernel(
            tree: DeviceTree,
            geometry: Geometry,
            origins: wp.array(dtype=wp.vec3),
            directions: wp.array(dtype=wp.vec3),
            counts: wp.array(dtype=int),
            statuses: wp.array(dtype=int),
        ):
            i = wp.tid()
            wp.static(ray)(
                origins[i],
                directions[i],
                0.0,
                10.0,
                -1,
                tree,
                geometry,
                counts,
                statuses,
                i,
            )

    configure_query_kernel(kernel, query)
    origins = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], np.float32
    )
    directions = np.tile([1.0, 0.0, 0.0], (3, 1)).astype(np.float32)
    device = _device()
    inputs = [
        device_payload(DeviceTree, tree, device),
        device_payload(Geometry, geometry, device),
        wp.array(origins, dtype=wp.vec3, device=device),
        wp.array(directions, dtype=wp.vec3, device=device),
    ]
    outputs = [
        wp.empty(3, dtype=int, device=device),
        wp.empty(3, dtype=int, device=device),
    ]
    if mode in ("first", "all"):
        shape = (3, 1) if mode == "all" else 3
        outputs = [
            wp.empty(shape, dtype=float, device=device),
            wp.empty(shape, dtype=int, device=device),
            *outputs,
        ]
    wp.launch(kernel, dim=3, inputs=inputs, outputs=outputs, device=device)
    args = {"max_hits": 1} if mode == "all" else {}
    expected = query(index.data, origins, directions, 10.0, exclude_face=-1, **args)
    for actual, reference in zip(outputs, expected, strict=True):
        np.testing.assert_array_equal(actual.numpy(), reference)


@pytest.mark.parametrize("kind", ["segments2", "triangles"])
def test_boundary_device_functions_compose(kind):
    index, boundary, _ = scene(kind, 64)
    primitive = index.spec.primitive
    tree, geometry = index.data
    classification = bind_classification(index.spec)
    overlap = bind_overlap(index.spec)
    silhouette = bind_silhouette(index.spec)
    star = bind_star_radius(index.spec)
    robin = bind_robin_radius(index.spec)
    sample = bind_boundary_sample(index.spec)
    sample_meta = prepare_boundary_sample(boundary)
    robin_meta = prepare_robin(boundary, 0.5)
    cf, ov, sf, st, rb, sp = [
        q.device for q in [classification, overlap, silhouette, star, robin, sample]
    ]
    Vec, Geometry = primitive.point_type, primitive.data_type
    Core, Feature, Coeff = robin.group_types
    Sampling = sample.group_types[0]

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        tree: DeviceTree,
        geometry: Geometry,
        core: Core,
        feature: Feature,
        coeff: Coeff,
        sampling: Sampling,
        points: wp.array(dtype=Vec),
        scalar: wp.array2d(dtype=float),
        status: wp.array2d(dtype=int),
        vectors: wp.array2d(dtype=Vec),
    ):
        i = wp.tid()
        point = points[i]
        sign, cs = wp.static(cf)(point, tree, geometry, core)
        hit, os = wp.static(ov)(point - Vec(0.25), point + Vec(0.25), tree, geometry)
        closest, distance, edge, ss = wp.static(sf)(point, 3.0, tree, feature)
        radius, ts = wp.static(st)(point, 3.0, tree, feature)
        rradius, rs = wp.static(rb)(point, 3.0, tree, geometry, core, feature, coeff)
        sampled, normal, pdf, face, ps = wp.static(sp)(
            point, 3.0, wp.vec3(0.1, 0.2, 0.3), tree, geometry, sampling
        )
        scalar[i, 0] = sign
        scalar[i, 1] = distance
        scalar[i, 2] = radius
        scalar[i, 3] = rradius
        scalar[i, 4] = pdf
        status[i, 0] = cs
        status[i, 1] = hit
        status[i, 2] = os
        status[i, 3] = edge
        status[i, 4] = ss
        status[i, 5] = ts
        status[i, 6] = rs
        status[i, 7] = face
        status[i, 8] = ps
        vectors[i, 0] = closest
        vectors[i, 1] = sampled
        vectors[i, 2] = normal

    configure_query_kernel(
        kernel, classification, overlap, silhouette, star, robin, sample
    )
    points = np.array(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [np.nan, 0.0, 0.0]], np.float32
    )[:, : primitive.dimension]
    device = _device()
    inputs = [
        device_payload(schema, host, device)
        for schema, host in [
            (DeviceTree, tree),
            (Geometry, geometry),
            (Core, boundary.core),
            (Feature, boundary.silhouette),
            (Coeff, robin_meta.coefficients),
            (Sampling, sample_meta),
        ]
    ]
    inputs.append(wp.array(points, dtype=Vec, device=device))
    outputs = [
        wp.empty((3, 5), dtype=float, device=device),
        wp.empty((3, 9), dtype=int, device=device),
        wp.empty((3, 3), dtype=Vec, device=device),
    ]
    wp.launch(kernel, dim=3, inputs=inputs, outputs=outputs, device=device)
    scalars, statuses, vectors = [x.numpy() for x in outputs]
    expected = [
        classification(boundary, points),
        overlap(index.data, points - 0.25, points + 0.25),
        silhouette(boundary, points, 3.0),
        star(boundary, points, 3.0),
        robin(robin_meta, points, 3.0),
        sample(sample_meta, points, 3.0, np.array([0.1, 0.2, 0.3])),
    ]
    refs_scalar = [
        expected[0][0],
        expected[2][1],
        expected[3][0],
        expected[4][0],
        expected[5][2],
    ]
    refs_status = [
        expected[0][1],
        *expected[1],
        expected[2][2],
        expected[2][3],
        expected[3][1],
        expected[4][1],
        expected[5][3],
        expected[5][4],
    ]
    for i, ref in enumerate(refs_scalar):
        np.testing.assert_array_equal(scalars[:, i], ref)
    for i, ref in enumerate(refs_status):
        np.testing.assert_array_equal(statuses[:, i], ref)
    for i, ref in enumerate([expected[2][0], expected[5][0], expected[5][1]]):
        np.testing.assert_array_equal(vectors[:, i], ref)


def test_configuration_requires_explicit_kernel_and_queries():
    with pytest.raises(TypeError, match="Warp kernel"):
        configure_query_kernel(object(), bind_nearest(bvh.IndexSpec(Points()), Euclidean()))


def test_external_primitive_device_call_uses_public_contract():
    from tests.typing.warp_api.real_extension import (
        ExternalArrays,
        ExternalData,
        ExternalPoints,
        external_evaluate,
        external_bound,
    )
    import jax.numpy as jnp

    primitive = ExternalPoints()
    index = bvh.build(
        primitive,
        ExternalArrays(jnp.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], jnp.float32)),
    )
    tree, geometry = index.data
    query = bind_knn(
        index.spec, 2, evaluate=external_evaluate, lower_bound=external_bound
    )
    knn = query.device

    @wp.kernel(module="unique", enable_backward=False)
    def kernel(
        tree: DeviceTree,
        geometry: ExternalData,
        scores: wp.array2d(dtype=float),
        ids: wp.array2d(dtype=int),
        summary: wp.array(dtype=int),
    ):
        count, status = wp.static(knn)(
            wp.vec3(1.0, 0.0, 0.0), 10.0, tree, geometry, scores, ids, 0
        )
        summary[0] = count
        summary[1] = status

    configure_query_kernel(kernel, query)
    device = _device()
    outputs = [
        wp.empty((1, 2), dtype=float, device=device),
        wp.empty((1, 2), dtype=int, device=device),
        wp.empty(2, dtype=int, device=device),
    ]
    wp.launch(
        kernel,
        dim=1,
        inputs=[
            device_payload(DeviceTree, tree, device),
            device_payload(ExternalData, geometry, device),
        ],
        outputs=outputs,
        device=device,
    )
    np.testing.assert_array_equal(outputs[0].numpy(), [[1.0, 4.0]])
    np.testing.assert_array_equal(outputs[1].numpy(), [[0, 1]])
    np.testing.assert_array_equal(outputs[2].numpy(), [2, 1])

"""Warp query checks against independent NumPy oracles."""

from __future__ import annotations

from dataclasses import replace
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from sengi import bvh, queries as query_api
from sengi.bvh import Config
from sengi.boundary.sampling import prepare_boundary_sample
from sengi.metrics.nearest import Euclidean


from tests.support._warp_helpers import (
    BUILDERS,
    TRIANGLES,
    _mesh,
    _sphere,
    _build,
    _oracle_closest,
    _oracle_ray,
    _oracle_silhouette,
)


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_query_bridge_preserves_cpu_execution(builder):
    accel = _build(builder, [[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    cpu = jax.devices("cpu")[0]
    payload = jax.tree_util.tree_map(lambda x: jax.device_put(x, cpu), accel.data)
    origin = jax.device_put(np.asarray([0.2, 0.2, 1], np.float32), cpu)
    direction = jax.device_put(np.asarray([0, 0, -1], np.float32), cpu)
    cap = jax.device_put(np.float32(2), cpu)
    outputs = jax.jit(query_api.rays.bind(accel.spec, "first"))(
        payload, origin, direction, cap
    )
    assert all(value.device.platform == "cpu" for value in outputs)
    t, face, hit, status = map(np.asarray, outputs)
    assert float(t) == 1 and int(face) == 0 and bool(hit) and int(status) == 1

    cp_outputs = jax.jit(query_api.nearest.bind(accel.spec, Euclidean()))(
        payload, origin
    )
    assert all(value.device.platform == "cpu" for value in cp_outputs)
    cp, distance, face, status = map(np.asarray, cp_outputs)
    np.testing.assert_allclose(cp, [0.2, 0.2, 0.0], rtol=0, atol=2e-6)
    np.testing.assert_allclose(distance, 1.0, rtol=0, atol=2e-6)
    assert int(face) == 0 and int(status) == 1

    source = _mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])[0]
    boundary = jax.device_put(bvh.prepare_boundary(accel, source), cpu)
    contact = jax.device_put(np.zeros(3, np.float32), cpu)
    outputs = jax.jit(query_api.silhouette.bind(accel.spec))(
        boundary, contact, jax.device_put(np.float32(0), cpu)
    )
    assert all(value.device.platform == "cpu" for value in outputs)
    cp, distance, edge, status = map(np.asarray, outputs)
    np.testing.assert_array_equal(cp, [0.0, 0.0, 0.0])
    assert float(distance) == 0 and int(edge) == 0 and int(status) == 1


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_bvh_closest_point_matches_numpy_oracle(builder):
    prims, v, f = _sphere(2)
    accel = bvh.build(TRIANGLES, prims, builder=builder)
    rng = np.random.default_rng(817)
    qs = np.concatenate(
        [
            rng.uniform(-1.5, 1.5, (29, 3)),
            np.array([[1, 0, 0], [-1.2, 0.2, 0.1]], np.float32),
        ]
    )
    got = jax.jit(
        jax.vmap(partial(query_api.nearest.bind(accel.spec, Euclidean()), accel.data))
    )(jnp.asarray(qs))
    cp, dist, ids = map(np.asarray, got[:3])
    for k, q in enumerate(qs):
        c, d, _ = _oracle_closest(v, f, q)
        np.testing.assert_allclose(cp[k], c, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(dist[k], d, atol=3e-5, rtol=3e-5)
        np.testing.assert_allclose(np.linalg.norm(cp[k] - q), dist[k], atol=3e-5)
        assert 0 <= ids[k] < len(f)


@pytest.mark.parametrize("builder", ["sah"])
def test_warp_bvh_closest_point_equal_distance_tie_across_children(builder):
    # Equal distances across children permit either face as the minimizer.
    v = np.asarray(
        [[1, 0, 0], [2, 0, 0], [1, 1, 0], [-2, 0, 0], [-1, 0, 0], [-1, 1, 0]],
        np.float32,
    )
    f = np.asarray([[0, 1, 2], [3, 4, 5]], np.int32)
    accel = bvh.build(
        TRIANGLES, _mesh(v, f)[0], builder=builder, config=Config(leaf_size_threshold=1)
    )
    payload = accel.data

    left = np.asarray(payload.tree.node_left)
    right = np.asarray(payload.tree.node_right)
    leaf_offset = np.asarray(payload.tree.node_leaf_offset)
    leaf_count = np.asarray(payload.tree.node_leaf_count)
    prim_ids = np.asarray(payload.tree.prim_ids)
    assert left[0] >= 0 and right[0] >= 0
    assert leaf_count[left[0]] == leaf_count[right[0]] == 1
    assert prim_ids[leaf_offset[left[0]]] == 1
    assert prim_ids[leaf_offset[right[0]]] == 0

    q = np.asarray([0.0, 0.25, 1.0], np.float32)
    query = jax.jit(query_api.nearest.bind(accel.spec, Euclidean()))
    cp, distance, face_id, status = map(np.asarray, query(payload, jnp.asarray(q)))
    assert int(face_id) in (0, 1) and int(status) == 1
    np.testing.assert_allclose(
        cp, [1.0 - 2.0 * int(face_id), 0.25, 0.0], rtol=0, atol=1e-6
    )
    np.testing.assert_allclose(distance, np.sqrt(2.0), rtol=0, atol=1e-6)


@pytest.mark.parametrize(
    ("vertices", "foot", "offset", "cap"),
    [
        (
            [[227, 171, 198], [1088, -403, 198], [3563, 5175, -3416]],
            [1491.5, 1135, -705.5],
            [2, 3, 6],
            7,
        ),
        (
            [[0, 0, 0], [1800000, 2400000, 0], [-40000, 30000, 0]],
            [600000, 800000, 0],
            [4, -3, 12],
            13,
        ),
    ],
    ids=["face", "edge"],
)
def test_warp_triangle_nonaxis_closed_distance_cap(vertices, foot, offset, cap):
    index = _build("sah", vertices, [[0, 1, 2]])
    foot, cap = np.asarray(foot, np.float32), np.float32(cap)
    query = foot + np.asarray(offset, np.float32)
    queries = np.stack((query, query, foot))
    caps = np.array([np.nextafter(cap, np.float32(0)), cap, 0], np.float32)

    nearest = jax.jit(query_api.nearest.bind(index.spec, Euclidean()))
    points, distance, ids, status = map(np.asarray, nearest(index.data, queries, caps))
    np.testing.assert_array_equal(ids, [-1, 0, 0])
    np.testing.assert_array_equal(status, [0, 1, 1])
    np.testing.assert_array_equal(distance, [np.inf, cap, 0.0])
    np.testing.assert_array_equal(points[1:], np.broadcast_to(foot, (2, 3)))

    knn = jax.jit(query_api.knn.bind(index.spec, 1, Euclidean()))
    distance, ids, count, status = map(np.asarray, knn(index.data, queries, caps))
    np.testing.assert_array_equal(ids[:, 0], [-1, 0, 0])
    np.testing.assert_array_equal(count, [0, 1, 1])
    np.testing.assert_array_equal(status, [0, 1, 1])
    np.testing.assert_array_equal(distance[:, 0], [np.inf, cap, 0.0])


@pytest.mark.parametrize(
    ("third", "query_z", "expected_z"),
    [
        ([0.0, 0.0, 0.0], 0.0, 0.0),
        ([0.0, 0.0, 3.0], 1.0, 1.0),
    ],
    ids=["collapsed-edge", "face"],
)
def test_warp_triangle_far_query_keeps_closest_point_on_triangle(
    third, query_z, expected_z
):
    vertices = [[-1.0, -2.0, 0.0], [1.0, 2.0, 0.0], third]
    accel = _build("sah", vertices, [[0, 1, 2]])
    query = jnp.asarray([2.0**56, -(2.0**55), query_z], jnp.float32)
    point, distance, face_id, status = map(
        np.asarray, query_api.nearest.bind(accel.spec, Euclidean())(accel.data, query)
    )
    np.testing.assert_allclose(point, [0.0, 0.0, expected_z], rtol=0, atol=1e-6)
    np.testing.assert_allclose(distance, np.sqrt(5.0) * 2.0**55, rtol=2e-7, atol=0)
    assert int(face_id) == 0 and int(status) == 1


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_bvh_rays_caps_axes_grazes_and_start_contact(builder):
    # Regular, collinear and point triangles.
    v = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [2, 0, 0],
            [3, 0, 0],
            [4, 0, 0],
            [5, 0, 0],
            [5, 0, 0],
            [5, 0, 0],
        ],
        np.float32,
    )
    f = np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8]], np.int32)
    accel = _build(builder, v, f)
    origins = np.array(
        [[0.2, 0.2, 1], [1, 0.5, 0], [0.5, 0.5, 0], [0.2, 0.2, -1], [8, 0, 1]],
        np.float32,
    )
    dirs = np.array(
        [[0, 0, -1], [-1, 0, 0], [0, 0, 1], [0, 0, 1], [0, 0, 0]], np.float32
    )
    max_t = np.array([1, 2, 0, 0.5, 10], np.float32)
    first = jax.jit(partial(query_api.rays.bind(accel.spec, "first"), accel.data))
    any_hit_query = jax.jit(partial(query_api.rays.bind(accel.spec, "any"), accel.data))
    for min_t in (0.0, 1e-4):
        ts, ids, hit, _ = map(np.asarray, first(origins, dirs, max_t, min_t=min_t))
        for k in range(len(origins)):
            t, idx = _oracle_ray(v, f, origins[k], dirs[k], max_t[k], min_t)
            assert bool(hit[k]) == np.isfinite(t), (
                f"lower={min_t}, ray={k}, builder t/id/hit="
                f"{ts[k]}/{ids[k]}/{hit[k]}, oracle t/id={t}/{idx}"
            )
            if np.isfinite(t):
                np.testing.assert_allclose(ts[k], t, atol=2e-5, rtol=2e-5)
                assert ids[k] == idx
            else:
                assert ids[k] == -1
                # Misses return max_t with ID -1.
                assert ts[k] == max_t[k]

        any_hit, any_status = map(
            np.asarray, any_hit_query(origins, dirs, max_t, min_t=min_t)
        )
        np.testing.assert_array_equal(any_hit, hit)
        valid_ray = (np.linalg.norm(dirs, axis=1) > 1e-15) & (max_t >= min_t)
        np.testing.assert_array_equal(
            any_status, np.where(valid_ray, np.where(hit, 1, 0), -1)
        )


@pytest.mark.parametrize("builder", ["sah"])
@pytest.mark.parametrize("speed", [1.0, 25.0])  # 1/25 rounds down in float32.
def test_warp_bvh_first_hit_equal_t_face_id_tie_across_children(builder, speed):
    # Coincident hits occur in both root subtrees.
    centers = np.asarray(
        [4.0, -4.0, -3.0, -2.0, -1.0, -1.0, 1.0, 2.0, 3.0, 4.0], np.float32
    )
    triangles = np.stack(
        [
            np.column_stack(
                [centers - 8, np.full_like(centers, -2), np.zeros_like(centers)]
            ),
            np.column_stack(
                [centers + 8, np.full_like(centers, -2), np.zeros_like(centers)]
            ),
            np.column_stack(
                [centers, np.full_like(centers, 4), np.zeros_like(centers)]
            ),
        ],
        axis=1,
    )
    vertices = triangles.reshape((-1, 3)).astype(np.float32)
    faces = np.arange(len(vertices), dtype=np.int32).reshape((-1, 3))
    accel = bvh.build(
        TRIANGLES,
        _mesh(vertices, faces)[0],
        builder=builder,
        config=Config(leaf_size_threshold=1),
    )
    payload = accel.data

    assert payload.tree.node_left[0] >= 0 and payload.tree.node_right[0] >= 0

    query = jax.jit(query_api.rays.bind(accel.spec, "first"))
    t, face_id, hit, status = query(
        payload,
        jnp.asarray([0.0, 0.0, 1.0], jnp.float32),
        jnp.asarray([0.0, 0.0, -speed], jnp.float32),
        jnp.float32(1.0),
    )
    assert bool(hit) and int(status) == 1
    np.testing.assert_allclose(t, np.float32(1.0 / speed), rtol=0, atol=0)
    assert 0 <= int(face_id) < len(faces)


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_bvh_ray_preserves_large_int32_source_face_id(builder):
    vertices = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    faces = np.asarray([[0, 1, 2]], np.int32)
    accel = _build(builder, vertices, faces)
    source_id = np.int32(2**24 + 1)
    index = replace(accel.data.tree, prim_ids=jnp.asarray([source_id], dtype=jnp.int32))
    payload = accel.data._replace(tree=index)

    query = jax.jit(query_api.rays.bind(accel.spec, "first"))
    t, face_id, hit, status = query(
        payload,
        jnp.asarray([0.2, 0.2, 1.0], jnp.float32),
        jnp.asarray([0.0, 0.0, -1.0], jnp.float32),
        jnp.float32(1.0),
    )
    assert bool(hit) and int(status) == 1
    np.testing.assert_allclose(t, 1.0, rtol=0, atol=0)
    assert np.asarray(face_id).dtype == np.int32
    assert int(face_id) == int(source_id)


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
@pytest.mark.parametrize(
    "tiny",
    [np.float32(0), np.float32(1e-30), np.nextafter(np.float32(0), np.float32(1))],
)
@pytest.mark.parametrize("side", [-1, 1])
def test_warp_bvh_tiny_direction_on_child_box_boundary(builder, tiny, side):
    # Exercise 0*inf on an internal child's slab boundary.
    triangles = []
    for i in range(9):
        x = side * 8.0 * i
        triangles.append([[x, 0, 1 + i], [x + side, 0, 1 + i], [x, 1, 1 + i]])
    vertices = np.asarray(triangles, np.float32).reshape((-1, 3))
    faces = np.arange(len(vertices), dtype=np.int32).reshape((-1, 3))
    primitives, _, _ = _mesh(vertices, faces)
    accel = bvh.build(
        TRIANGLES, primitives, builder=builder, config=Config(leaf_size_threshold=1)
    )
    assert np.any(np.asarray(accel.data.tree.node_leaf_count) == 0)

    origins = jnp.asarray([[0, 0.25, 0]] * 3, jnp.float32)
    directions = jnp.asarray([[side * tiny, 0, 1]] * 3, jnp.float32)
    lower = jnp.asarray([0, 1, 1.0001], jnp.float32)
    upper = jnp.asarray([jnp.inf, 1, jnp.inf], jnp.float32)
    ts, ids, hits, status = map(
        np.asarray,
        jax.jit(query_api.rays.bind(accel.spec, "first"))(
            accel.data, origins, directions, upper, min_t=lower
        ),
    )
    np.testing.assert_array_equal(hits, [True, True, False])
    np.testing.assert_array_equal(status, [1, 1, 0])
    np.testing.assert_array_equal(ids, [0, 0, -1])
    np.testing.assert_allclose(ts[:2], [1, 1], rtol=0, atol=0)
    assert np.isinf(ts[2])

    any_hit, any_status = map(
        np.asarray,
        jax.jit(query_api.rays.bind(accel.spec, "any"))(
            accel.data, origins, directions, upper, min_t=lower
        ),
    )
    np.testing.assert_array_equal(any_hit, hits)
    np.testing.assert_array_equal(any_status, status)


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_bvh_constant_origin_broadcasts_across_ray_directions(builder):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(builder, v, f)
    origin = jnp.asarray([0.2, 0.2, 1.0], jnp.float32)  # deliberately unbatched
    directions = jnp.asarray(
        [[0.0, 0.0, -1.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], jnp.float32
    )
    caps = jnp.asarray([2.0, 2.0, 2.0])
    got = jax.jit(
        jax.vmap(
            lambda d, cap: query_api.rays.bind(accel.spec, "first")(
                accel.data, origin, d, cap
            )[:3],
            in_axes=(0, 0),
        )
    )(directions, caps)
    ts, ids, hit = map(np.asarray, got)
    np.testing.assert_array_equal(hit, [True, False, False])
    np.testing.assert_array_equal(ids, [0, -1, -1])
    np.testing.assert_allclose(ts, [1.0, 2.0, 2.0], atol=2e-5)


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_bvh_nested_ray_batches_broadcast_outer_min_t(builder):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(builder, v, f)
    origins = jnp.asarray([[0.2, 0.2, 0.0], [0.2, 0.2, 1.0]], jnp.float32)
    directions = jnp.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]], jnp.float32)
    max_ts = jnp.asarray([2.0, 2.0], jnp.float32)
    min_ts = jnp.asarray([0.0, 1e-4], jnp.float32)

    def per_min(min_t):
        return jax.vmap(
            lambda o, d, hi: query_api.rays.bind(accel.spec, "first")(
                accel.data, o, d, hi, min_t=min_t
            )[:3],
            in_axes=(0, 0, 0),
        )(origins, directions, max_ts)

    t, idx, hit = map(np.asarray, jax.jit(jax.vmap(per_min))(min_ts))
    np.testing.assert_array_equal(hit, [[True, True], [False, True]])
    np.testing.assert_allclose(t[0], [0.0, 1.0], atol=2e-5)
    assert idx[0, 0] == 0 and idx[0, 1] == 0 and idx[1, 0] == -1 and idx[1, 1] == 0


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_index_data_reuse_nested_vmap_jit_and_lax_loop(builder):
    source, _, _ = _sphere(1)
    accel = bvh.build(TRIANGLES, source, builder=builder)
    nearest = query_api.nearest.bind(accel.spec, Euclidean())
    batch_nearest = jax.vmap(nearest, in_axes=(None, 0))
    qs = jnp.asarray(np.random.default_rng(7).normal(size=(3, 5, 3)).astype(np.float32))
    qflat = qs.reshape((-1, 3))
    qbank = jnp.concatenate([qflat, jnp.zeros((4, 3), jnp.float32)], axis=0)
    query = jax.jit(jax.vmap(batch_nearest, in_axes=(None, 0)))
    nested = query(accel.data, qs)
    assert np.asarray(nested[0]).shape == (3, 5, 3)

    # Reject vmap rules that tile BVH arrays.
    batch = 37
    probe = jax.make_jaxpr(batch_nearest)(
        accel.data,
        jnp.zeros((batch, 3), jnp.float32),
    )
    source_tree, source_geometry = accel.data.tree, accel.data.geometry
    source_shapes = [tuple(x.shape) for x in (*jax.tree.leaves(source_tree), *source_geometry)]

    def ffi_inputs(jaxpr):
        calls = []
        for eqn in jaxpr.eqns:
            if eqn.primitive.name == "ffi_call":
                calls.append([tuple(var.aval.shape) for var in eqn.invars])
            for value in eqn.params.values():
                nested = getattr(value, "jaxpr", value)
                if hasattr(nested, "eqns"):
                    calls.extend(ffi_inputs(nested))
        return calls

    calls = ffi_inputs(probe.jaxpr)
    assert len(calls) == 1
    assert calls[0][-len(source_shapes) :] == source_shapes

    def loop(payload, points):
        out = jnp.zeros_like(points)
        return jax.lax.fori_loop(
            0,
            points.shape[0],
            lambda i, acc: acc.at[i].set(nearest(payload, points[i])[0]),
            out,
        )

    got = np.asarray(jax.jit(loop)(accel.data, qflat))
    want = np.asarray(nested[0]).reshape((-1, 3))
    np.testing.assert_allclose(got, want, atol=2e-5)
    nearest_batch = jax.jit(batch_nearest)
    for n in (1, 7, 19):
        point, _, _, _ = nearest_batch(accel.data, qbank[:n])
        assert point.shape == (n, 3)


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_silhouette_includes_open_boundary_and_no_candidate(builder):
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(builder, v, f)
    boundary = bvh.prepare_boundary(accel, _mesh(v, f)[0])
    qs = jnp.asarray([[0.5, 0.5, 1], [4, 4, 1]], jnp.float32)
    cp, d, edge_id = map(
        np.asarray,
        jax.jit(
            jax.vmap(
                partial(query_api.silhouette.bind(accel.spec), boundary)
            )
        )(qs)[:3],
    )
    for i, q in enumerate(np.asarray(qs)):
        edges = [(v[0], v[1]), (v[1], v[2]), (v[2], v[0])]
        cps = []
        for a, b in edges:
            t = np.clip(np.dot(q - a, b - a) / np.dot(b - a, b - a), 0, 1)
            cps.append(a + t * (b - a))
        ds = [np.linalg.norm(q - x) for x in cps]
        j = int(np.argmin(ds))
        np.testing.assert_allclose(cp[i], cps[j], atol=2e-5)
        np.testing.assert_allclose(d[i], ds[j], atol=2e-5)
        assert edge_id[i] >= 0


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_closed_mesh_has_no_interior_silhouette(builder):
    v = np.array([[1, 1, 1], [-1, -1, 1], [-1, 1, -1], [1, -1, -1]], np.float32)
    f = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], np.int32)
    accel = _build(builder, v, f)
    boundary = bvh.prepare_boundary(accel, _mesh(v, f)[0])
    cp, dist, edge_id = query_api.silhouette.bind(accel.spec)(
        boundary, jnp.zeros((3,), jnp.float32)
    )[:3]
    assert np.isinf(float(dist))
    assert int(edge_id) == -1
    assert np.all(np.isfinite(np.asarray(cp)))


@pytest.mark.parametrize("builder", ["sah"])
def test_warp_silhouette_cap_and_ties_across_children(builder):
    v = np.asarray(
        [[1, 0, 0], [2, 0, 0], [1, 1, 0], [-2, 0, 0], [-1, 0, 0], [-1, 1, 0]],
        np.float32,
    )
    f = np.asarray([[0, 1, 2], [3, 4, 5]], np.int32)
    accel = bvh.build(
        TRIANGLES, _mesh(v, f)[0], builder=builder, config=Config(leaf_size_threshold=1)
    )
    boundary = bvh.prepare_boundary(accel, _mesh(v, f)[0])
    payload = accel.data
    left = int(payload.tree.node_left[0])
    assert int(payload.tree.prim_ids[payload.tree.node_leaf_offset[left]]) == 1

    points = jnp.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    caps = jnp.asarray([1.0, np.nextafter(np.float32(1), np.float32(0)), 0.0])
    query = jax.jit(query_api.silhouette.bind(accel.spec))
    cp, distance, edge, status = map(np.asarray, query(boundary, points, caps))
    np.testing.assert_array_equal(status, [1, 0, 1])
    assert edge[1] == -1
    np.testing.assert_array_equal(cp[2], points[2])
    assert abs(cp[0, 0]) == 1.0
    np.testing.assert_array_equal(cp[0, 1:], 0.0)
    features = boundary.silhouette
    for row in (0, 2):
        assert 0 <= edge[row] < len(features.feature_a)
        endpoints = np.array([features.feature_a[edge[row]], features.feature_b[edge[row]]])
        assert np.any(np.all(endpoints == cp[row], axis=1))
    np.testing.assert_array_equal(distance[[0, 2]], [1.0, 0.0])
    assert np.isinf(distance[1])


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_star_radius_applies_only_the_requested_cap(builder):
    v = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(builder, v, f)
    boundary = bvh.prepare_boundary(accel, _mesh(v, f)[0])
    q = jnp.broadcast_to(jnp.asarray([1.0, 1.0, 1.0]), (5, 3))
    caps = jnp.asarray([10.0, 0.2, 0.0, -1.0, jnp.nan])
    radius, status = map(
        np.asarray,
        jax.jit(
            jax.vmap(
                partial(query_api.star_radius.bind(accel.spec), boundary)
            )
        )(q, caps),
    )
    np.testing.assert_allclose(radius[:3], [1.0, 0.2, 0.0], atol=2e-5)
    np.testing.assert_array_equal(status, [1, 1, 1, -1, -1])
    assert np.isnan(radius[3:]).all()

    # No silhouette lies within the cap.
    _, bounded_dist, bounded_id, bounded_status = query_api.silhouette.bind(accel.spec)(
        boundary, q[0], max_distance=jnp.float32(0.2)
    )
    assert np.isinf(float(bounded_dist))
    assert int(bounded_id) == -1 and int(bounded_status) == 0

    # Map caps outside point batches; share the tree.
    points = jnp.asarray([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], jnp.float32)
    outer_caps = jnp.asarray([0.2, 10.0], jnp.float32)

    def per_cap(cap):
        return jax.vmap(
            lambda point: query_api.star_radius.bind(accel.spec)(
                boundary, point, cap
            )
        )(points)

    radii, statuses = map(np.asarray, jax.jit(jax.vmap(per_cap))(outer_caps))
    np.testing.assert_allclose(radii, [[0.2, 0.2], [1.0, 1.0]], atol=2e-5)
    assert np.all(statuses == 1)


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_boundary_sampler_area_pdf_and_null_event(builder):
    prims, vertices, faces = _sphere(2)
    accel = bvh.build(TRIANGLES, prims, builder=builder)
    boundary = bvh.prepare_boundary(accel, prims, silhouettes=False)
    sampling = prepare_boundary_sample(boundary)
    rng = np.random.default_rng(136)
    n = 12000
    randoms = rng.random((n, 3), dtype=np.float32)
    points = np.zeros((n, 3), np.float32)
    areas = (
        np.linalg.norm(
            np.cross(
                vertices[faces[:, 1]] - vertices[faces[:, 0]],
                vertices[faces[:, 2]] - vertices[faces[:, 0]],
            ),
            axis=1,
        )
        * 0.5
    )
    total_area = float(areas.sum())
    sampler = jax.jit(
        partial(query_api.boundary_sample.bind(accel.spec), sampling)
    )

    p, normal, pdf, ids, status = map(
        np.asarray,
        sampler(
            jnp.asarray(points),
            jnp.full((n,), jnp.inf, jnp.float32),
            jnp.asarray(randoms),
        ),
    )
    assert np.all(status == 1)
    assert np.all((ids >= 0) & (ids < len(faces)))
    np.testing.assert_allclose(pdf, 1.0 / total_area, rtol=3e-5, atol=3e-6)
    np.testing.assert_allclose(np.linalg.norm(normal, axis=1), 1.0, atol=3e-5)
    expected = areas / total_area
    observed = np.bincount(ids.astype(np.int64), minlength=len(faces)) / n
    # Allow sampling noise across 320 faces.
    assert np.max(np.abs(observed - expected)) < 0.012
    assert np.all(np.isfinite(p))

    # Scalar broadcasting must match repeated point storage.
    small_n = 64
    small_randoms = rng.random((small_n, 3), dtype=np.float32)
    small_radii = np.resize(np.array([np.inf, 0.99, 0.99, 0.0], np.float32), small_n)
    scalar_point_out = jax.jit(
        jax.vmap(
            lambda r, u: query_api.boundary_sample.bind(accel.spec)(
                sampling, jnp.zeros((3,), jnp.float32), r, u
            ),
            in_axes=(0, 0),
        )
    )(jnp.asarray(small_radii), jnp.asarray(small_randoms))
    repeated_point_out = sampler(
        jnp.zeros((small_n, 3), jnp.float32),
        jnp.asarray(small_radii),
        jnp.asarray(small_randoms),
    )
    for scalar_leaf, repeated_leaf in zip(scalar_point_out, repeated_point_out):
        np.testing.assert_allclose(scalar_leaf, repeated_leaf, atol=2e-6, rtol=2e-6)

    # Nulls contribute zero to the area estimator.
    radius = 0.99
    p, normal, pdf, ids, status = map(
        np.asarray,
        sampler(
            jnp.zeros((n, 3), jnp.float32),
            jnp.full((n,), radius, jnp.float32),
            jnp.asarray(randoms),
        ),
    )
    assert np.all(status >= 0)
    accepted = status == 1
    rejected = status == 0
    assert accepted.any() and rejected.any()
    assert np.all(pdf[accepted] > 0) and np.all(ids[accepted] >= 0)
    assert np.all(pdf[rejected] == 0) and np.all(ids[rejected] == -1)
    assert np.all(normal[rejected] == 0.0)
    assert np.all(np.linalg.norm(p[accepted], axis=1) <= radius + 2e-5)
    estimated_area = float(
        np.mean(np.where(accepted, 1.0 / np.maximum(pdf, 1e-30), 0.0))
    )

    # Independent area-uniform Monte Carlo.
    m = 250000
    rr = np.random.default_rng(2025).random((m, 3))
    face_ids = np.searchsorted(np.cumsum(areas / total_area), rr[:, 0], side="right")
    u = np.sqrt(rr[:, 1])
    bary = np.stack([1 - u, u * (1 - rr[:, 2]), u * rr[:, 2]], axis=1)
    samples = np.sum(vertices[faces[face_ids]] * bary[..., None], axis=1)
    oracle_area = total_area * float(np.mean(np.linalg.norm(samples, axis=1) <= radius))
    np.testing.assert_allclose(estimated_area, oracle_area, rtol=0.08, atol=0.015)

    # Infinite radius on an empty tree is a valid null event.
    empty_source = _mesh(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))[0]
    empty = bvh.build(TRIANGLES, empty_source, builder=builder)
    empty_boundary = bvh.prepare_boundary(empty, empty_source, silhouettes=False)
    empty_sampling = prepare_boundary_sample(empty_boundary)
    empty_out = query_api.boundary_sample.bind(empty.spec)(
        empty_sampling,
        jnp.zeros((1, 3), jnp.float32),
        jnp.asarray([jnp.inf]),
        jnp.asarray([[0.2, 0.3, 0.4]], jnp.float32),
    )
    assert int(np.asarray(empty_out[4])[0]) == 0
    assert int(np.asarray(empty_out[3])[0]) == -1
    assert float(np.asarray(empty_out[2])[0]) == 0.0


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
@pytest.mark.parametrize("empty_tree", [False, True])
def test_warp_bvh_empty_mesh_and_query_batches(builder, empty_tree):
    if empty_tree:
        v = np.zeros((0, 3), np.float32)
        f = np.zeros((0, 3), np.int32)
    else:
        v = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
        f = np.asarray([[0, 1, 2]], np.int32)
    accel = _build(builder, v, f)
    # Empty launches preserve output shapes.
    q = jnp.empty((0, 3), jnp.float32)
    caps = jnp.empty((0,), jnp.float32)
    outputs = [
        jax.jit(partial(query_api.nearest.bind(accel.spec, Euclidean()), accel.data))(
            q
        ),
        jax.jit(partial(query_api.rays.bind(accel.spec, "first"), accel.data))(
            q, q, caps
        ),
        jax.jit(partial(query_api.rays.bind(accel.spec, "any"), accel.data))(
            q, q, caps
        ),
        jax.jit(partial(query_api.overlap.bind(accel.spec), accel.data))(q, q),
    ]
    boundary = bvh.prepare_boundary(accel, _mesh(v, f)[0])
    sampling = prepare_boundary_sample(boundary)
    outputs.extend(
        [
            jax.jit(query_api.silhouette.bind(accel.spec))(boundary, q),
            jax.jit(query_api.boundary_sample.bind(accel.spec))(
                sampling, q, caps, q
            ),
        ]
    )
    jax.block_until_ready(outputs)
    assert all(value.shape[0] == 0 for result in outputs for value in result)
    if empty_tree:
        cp, distance, idx, status = query_api.nearest.bind(accel.spec, Euclidean())(
            accel.data, jnp.zeros(3)
        )
        assert np.isinf(float(distance)) and int(idx) == -1 and int(status) == 0
        assert np.all(np.isfinite(np.asarray(cp)))


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
@pytest.mark.parametrize("scale", [1e-4, 1.0, 1e4])
def test_warp_bvh_scale_robust_triangle_cp_and_ray(builder, scale):
    # Check seven orders of scale against analytic answers.
    v = scale * np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(builder, v, f)
    q = jnp.asarray(scale * np.array([0.2, 0.3, 0.4], np.float32))
    cp, dist, idx = query_api.nearest.bind(accel.spec, Euclidean())(accel.data, q)[:3]
    np.testing.assert_allclose(
        cp, scale * np.array([0.2, 0.3, 0]), atol=2e-7 * scale, rtol=2e-5
    )
    np.testing.assert_allclose(dist, 0.4 * scale, atol=2e-7 * scale, rtol=2e-5)
    assert int(idx) == 0
    origin = jnp.asarray(scale * np.array([0.2, 0.3, 1], np.float32))
    direction = jnp.asarray([0.0, 0.0, -1.0], jnp.float32)
    t, prim, hit = query_api.rays.bind(accel.spec, "first")(
        accel.data, origin, direction, jnp.float32(2 * scale)
    )[:3]
    assert bool(hit) and int(prim) == 0
    np.testing.assert_allclose(t, scale, atol=2e-7 * scale, rtol=2e-5)


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
@pytest.mark.parametrize(
    "scale,aspect_ratio",
    [(1e-4, 1.0), (1e-4, 1e-5), (1.0, 0.11), (1.0, 0.09), (1e4, 1e-3), (1e4, 1e-4)],
    ids=[
        "small",
        "small-thin",
        "above-threshold",
        "below-threshold",
        "large-thin",
        "large-thinner",
    ],
)
def test_warp_bvh_closest_point_fastpath_threshold(builder, scale, aspect_ratio):
    vertices = scale * np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.25, aspect_ratio, 0.0]], np.float32
    )
    faces = np.asarray([[0, 1, 2]], np.int32)
    accel = _build(builder, vertices, faces)

    centroid = np.mean(vertices.astype(np.float64), axis=0).astype(np.float32)
    queries = np.asarray(
        [
            centroid + np.asarray([0.0, 0.0, 0.4 * scale], np.float32),
            np.asarray([0.4 * scale, -0.1 * scale, 0.3 * scale], np.float32),
        ],
        np.float32,
    )
    expected_cp = np.asarray(
        [[queries[0, 0], queries[0, 1], 0.0], [queries[1, 0], 0.0, 0.0]], np.float32
    )
    expected_distance = np.asarray(
        [
            abs(float(queries[0, 2])),
            np.hypot(float(queries[1, 1]), float(queries[1, 2])),
        ],
        np.float32,
    )

    cp, distance, face_id = map(
        np.asarray,
        jax.jit(
            jax.vmap(
                partial(query_api.nearest.bind(accel.spec, Euclidean()), accel.data)
            )
        )(jnp.asarray(queries))[:3],
    )
    tolerance = 2e-6 * scale
    np.testing.assert_allclose(cp, expected_cp, rtol=2e-5, atol=tolerance)
    np.testing.assert_allclose(distance, expected_distance, rtol=2e-5, atol=tolerance)
    np.testing.assert_array_equal(face_id, [0, 0])


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_bvh_status_distinguishes_invalid_from_empty(builder):
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], np.float32)
    f = np.array([[0, 1, 2]], np.int32)
    accel = _build(builder, v, f)
    cp_result = query_api.nearest.bind(accel.spec, Euclidean())(
        accel.data, jnp.asarray([jnp.nan, 0.0, 0.0])
    )
    assert int(cp_result[3]) == -1
    overflow = query_api.nearest.bind(accel.spec, Euclidean())(
        accel.data, jnp.asarray([1.0e20, 0.0, 0.0], jnp.float32)
    )
    point, distance, face, status = map(np.asarray, overflow)
    np.testing.assert_array_equal(point, np.zeros(3, np.float32))
    assert np.isinf(distance) and int(face) == -1 and int(status) == -1
    ray_result = query_api.rays.bind(accel.spec, "first")(
        accel.data,
        jnp.zeros((3,), jnp.float32),
        jnp.asarray([jnp.nan, 0.0, -1.0]),
        jnp.float32(2.0),
    )
    assert int(ray_result[3]) == -1

    empty = _build(builder, np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    empty_cp = query_api.nearest.bind(empty.spec, Euclidean())(
        empty.data, jnp.zeros((3,), jnp.float32)
    )
    assert int(empty_cp[3]) == 0
    empty_ray = query_api.rays.bind(empty.spec, "first")(
        empty.data,
        jnp.zeros((3,), jnp.float32),
        jnp.asarray([0.0, 0.0, 1.0]),
        jnp.float32(1.0),
    )
    assert int(empty_ray[3]) == 0 and not bool(empty_ray[2])

    # Unbounded max_t is supported; an infinite lower bound is invalid.
    hit = query_api.rays.bind(accel.spec, "first")(
        accel.data,
        jnp.asarray([0.2, 0.2, 1.0], jnp.float32),
        jnp.asarray([0.0, 0.0, -1.0], jnp.float32),
        jnp.float32(jnp.inf),
    )
    assert bool(hit[2]) and int(hit[1]) == 0 and int(hit[3]) == 1
    np.testing.assert_allclose(hit[0], 1.0, atol=2e-5)
    bad_interval = query_api.rays.bind(accel.spec, "first")(
        accel.data,
        jnp.asarray([0.2, 0.2, 1.0], jnp.float32),
        jnp.asarray([0.0, 0.0, -1.0], jnp.float32),
        jnp.float32(jnp.inf),
        min_t=jnp.float32(jnp.inf),
    )
    assert int(bad_interval[3]) == -1 and not bool(bad_interval[2])


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_bvh_build_config_depth_and_leaf_bounds(builder):
    prims, _, _ = _sphere(2)
    with pytest.raises(ValueError):
        bvh.build(TRIANGLES, prims, builder=builder, config=Config(max_stack_depth=0))
    with pytest.raises(ValueError):
        bvh.build(TRIANGLES, prims, builder=builder, config=Config(max_stack_depth=65))
    with pytest.raises(ValueError):
        bvh.build(
            TRIANGLES, prims, builder=builder, config=Config(leaf_size_threshold=0)
        )
    with pytest.raises(ValueError, match="depth"):
        bvh.build(
            TRIANGLES,
            prims,
            builder=builder,
            config=Config(max_stack_depth=1, leaf_size_threshold=1),
        )
    built = bvh.build(
        TRIANGLES,
        prims,
        builder=builder,
        config=Config(max_stack_depth=64, leaf_size_threshold=1),
    )
    assert built.data is not None


def _triangle_regressions():
    regular = [[0, 0, 0], [2, 0, 0], [0, 1, 0]]
    thin = np.array([[0, 0, 0], [1e4, 0, 0], [1e4, 1e-3, 0]], np.float32)
    target = 0.2 * thin[1] + 0.3 * thin[2]
    oblique = np.array([[0, 0, 0], [0.03, 0.01, 0.02], [-0.01, 0.02, 0.01]], np.float32)
    a, b, c = oblique.astype(np.float64)
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    return [
        pytest.param(
            regular,
            [[0.3, 0.2, 1], [3, 0, 0], [-1, 2, 0]],
            None,
            2e-6,
            0.0,
            id="regular",
        ),
        pytest.param(
            [[0, 0, 0], [2, 0, 0], [1, 0, 0]],
            [[0.75, 1, 0]],
            None,
            2e-6,
            0.0,
            id="collinear",
        ),
        pytest.param([[4, 0, 0]] * 3, [[4.5, 0, 0]], None, 2e-6, 0.0, id="point"),
        pytest.param(
            thin, [target + [0, 0, 2]], target[None], 2e-4, 2e-6, id="long-thin"
        ),
        pytest.param(
            oblique,
            [
                a + 0.25 * (b - a) + 0.35 * (c - a) + 5 * normal,
                0.5 * (a + b) + 5 * normal,
                c + 5 * normal,
            ],
            None,
            2e-5,
            2e-5,
            id="oblique",
        ),
        pytest.param(
            [[10000, 10000, 0], [10001, 10000, 0], [10000, 10001, 0]],
            [[10000.123, 10000.321, 0.4], [10000.25, 10000.5, -0.3]],
            None,
            2e-6,
            0.0,
            id="translated",
        ),
        *(
            pytest.param(
                np.roll([[0, 0, 0], [1, 0, 0], [-1, 0.2, 0]], shift, axis=0),
                [[0, 0.05, 0.4]],
                None,
                2e-6,
                0.0,
                id=f"obtuse-{shift}",
            )
            for shift in range(3)
        ),
    ]


@pytest.mark.parametrize(
    "vertices,queries,expected_cp,atol,rtol", _triangle_regressions()
)
@pytest.mark.parametrize("builder", BUILDERS)
def test_warp_triangle_closest_point_regressions(
    builder, vertices, queries, expected_cp, atol, rtol
):
    vertices, queries = (np.asarray(v, np.float32) for v in (vertices, queries))
    if expected_cp is None:
        expected_cp = np.stack(
            [
                _oracle_closest(
                    vertices.astype(np.float64), [[0, 1, 2]], q.astype(np.float64)
                )[0]
                for q in queries
            ]
        )
    accel = _build(builder, vertices, [[0, 1, 2]])
    cp, distance, face = map(
        np.asarray,
        jax.jit(partial(query_api.nearest.bind(accel.spec, Euclidean()), accel.data))(
            queries
        )[:3],
    )
    np.testing.assert_allclose(cp, expected_cp, atol=atol, rtol=rtol)
    np.testing.assert_allclose(
        distance, np.linalg.norm(queries - expected_cp, axis=-1), atol=atol, rtol=rtol
    )
    np.testing.assert_array_equal(face, 0)


def _concave_prism():
    outline = np.array([[-1, -1], [0, -1], [1, -1], [1, 0], [0, 0], [0, 1], [-1, 1]])
    n = len(outline)
    vertices = np.concatenate(
        [np.column_stack([outline, np.full(n, z)]) for z in (-0.5, 0.5)]
    )
    top = [[0, 1, 4], [0, 4, 5], [0, 5, 6], [1, 2, 3], [1, 3, 4]]
    faces = [f[::-1] for f in top] + [[i + n for i in f] for f in top]
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, j + n], [i, j + n, i + n]])
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


@pytest.mark.parametrize(
    "case,scale",
    [
        ("concave", 1.0),
        ("long", 1.0),
        ("torus", 1.0),
        ("thin", 1e-3),
        ("thin", 1.0),
        ("thin", 1e3),
        ("flat", 1.0),
    ],
)
def test_warp_silhouette_geometry_regressions(case, scale):
    rng = np.random.default_rng(917)
    caps, atol = np.inf, 5e-5
    if case == "concave":
        mesh = _concave_prism()
        points = rng.uniform([-2, -2, -1.5], [2, 2, 1.5], (47, 3))
    elif case == "long":
        mesh = trimesh.creation.box(extents=[1e4, 2.0, 1.0])
        points = [[4999, 3, 2], [-5001, 2, 0.5], [0, 1.1, 0.6], [5000, -2, -1]]
        atol = 4e-4
    elif case == "torus":
        mesh = trimesh.creation.torus(
            major_radius=0.7, minor_radius=0.3, major_sections=24, minor_sections=12
        )
        points = np.random.default_rng(647).uniform(-1.2, 1.2, (35, 3))
        caps = np.resize(np.array([0.15, 0.75, np.inf], np.float32), len(points))
    elif case == "thin":
        mesh = trimesh.creation.box(extents=[4.0, 2.0, 0.125])
        rotation = trimesh.transformations.rotation_matrix(0.37, [0.0, 0.0, 1.0])
        mesh.apply_transform(rotation)
        mesh.vertices *= scale
        points = (
            np.array(
                [
                    [0, 0, 0.0625],
                    [2, 0, 0.0625],
                    [0, 1, 0.0625],
                    [2.0001, 0, 0.0625],
                    [0, 0, 0.062501],
                    [0, 0, 0.062499],
                ]
            )
            @ rotation[:3, :3].T
            * scale
        )
        atol = 2e-6 * scale
    else:
        mesh = trimesh.creation.icosphere(subdivisions=2)
        mesh.vertices *= [1.0, 1.0, 1e-4]
        points = np.random.default_rng(927).uniform(-1.5, 1.5, (71, 3)) * [
            1.0,
            1.0,
            1e-4,
        ]
    primitives, vertices, faces = _mesh(mesh.vertices, mesh.faces)
    accel = bvh.build(
        TRIANGLES, primitives, builder="sah", config=Config(leaf_size_threshold=1)
    )
    boundary = bvh.prepare_boundary(accel, primitives)
    points = np.asarray(points, np.float32)
    cp, distance, edge, status = map(
        np.asarray,
        jax.jit(partial(query_api.silhouette.bind(accel.spec), boundary))(
            points, max_distance=caps
        ),
    )
    expected_cp, expected = _oracle_silhouette(vertices, faces, points)
    found = np.isfinite(expected) & (expected <= caps)
    np.testing.assert_array_equal(status, found.astype(np.int32))
    np.testing.assert_allclose(
        distance, np.where(found, expected, np.inf), atol=atol, rtol=5e-5
    )
    assert np.all(edge[found] >= 0) and np.all(edge[~found] == -1)
    np.testing.assert_allclose(
        np.linalg.norm(cp[found] - points[found], axis=1),
        distance[found],
        atol=atol,
        rtol=5e-5,
    )
    if case in ("concave", "long", "torus"):
        np.testing.assert_allclose(cp[found], expected_cp[found], atol=atol, rtol=5e-5)
    if case == "flat":
        features = jax.tree_util.tree_map(np.asarray, boundary.silhouette)
        for node in np.flatnonzero(accel.data.tree.node_leaf_count):
            sin_alpha, cos_alpha = features.node_cone_sin_cos[node]
            if cos_alpha <= 0:
                continue
            start = features.node_feature_offset[node]
            refs = features.feature_refs[
                start : start + features.node_feature_count[node]
            ]
            normals = np.concatenate(
                [features.feature_n0[refs], features.feature_n1[refs]]
            ).astype(np.float64)
            axis = features.node_cone_axis[node].astype(np.float64)
            angles = np.arctan2(
                np.linalg.norm(np.cross(normals, axis), axis=1), normals @ axis
            )
            assert angles.max() <= np.arctan2(float(sin_alpha), float(cos_alpha)) + 1e-7


@pytest.mark.parametrize("builder", BUILDERS, ids=BUILDERS)
def test_warp_index_shares_indexed_geometry(builder):
    prims, _, faces = _sphere(1)
    index = bvh.build(TRIANGLES, prims, builder=builder)
    payload = index.data
    boundary = bvh.prepare_boundary(index, prims)
    sampling = prepare_boundary_sample(boundary)
    ids = np.asarray(payload.tree.prim_ids)
    assert not np.array_equal(ids, np.arange(len(faces)))
    np.testing.assert_array_equal(payload.geometry.indices, faces[ids])
    assert payload.geometry.vertices is prims.vertices
    assert boundary.core.vertices is payload.geometry.vertices
    assert boundary.core.indices is payload.geometry.indices
    assert sampling.primitive_weight is boundary.core.primitive_measure
    assert boundary.silhouette is not None


def test_warp_preserves_edge_and_sampled_face_ids():
    vertices = [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, -1, 0]]
    primitives, _, _ = _mesh(vertices, [[2, 0, 1], [1, 0, 3]])
    accel = bvh.build(
        TRIANGLES, primitives, builder="sah", config=Config(leaf_size_threshold=1)
    )
    boundary = bvh.prepare_boundary(accel, primitives)
    sampling = prepare_boundary_sample(boundary)
    point, distance, edge_id, status = query_api.silhouette.bind(accel.spec)(
        boundary,
        jnp.asarray([0.5, 0.0, 0.0], jnp.float32),
        max_distance=0.0,
    )
    np.testing.assert_array_equal(point, [0.5, 0.0, 0.0])
    assert float(distance) == 0.0 and int(edge_id) == 1 and int(status) == 1
    # Fixed draws on opposite children return the corresponding source faces.
    draws = jnp.array([[0.2, 0.3, 0.4], [0.8, 0.3, 0.4]])
    np.testing.assert_array_equal(
        query_api.boundary_sample.bind(accel.spec)(
            sampling, jnp.zeros((2, 3)), jnp.inf, draws
        )[3],
        [1, 0],
    )


def test_silhouettes_validate_degeneracy_and_manifold_connectivity():
    source, _, _ = _mesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]])
    index = bvh.build(TRIANGLES, source)
    with pytest.raises(ValueError, match="non-degenerate"):
        bvh.prepare_boundary(index, source)

    vertices = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1]]
    source, _, _ = _mesh(vertices, [[0, 1, 2], [1, 0, 3], [0, 1, 4]])
    index = bvh.build(TRIANGLES, source)
    with pytest.raises(ValueError, match="manifold"):
        bvh.prepare_boundary(index, source)

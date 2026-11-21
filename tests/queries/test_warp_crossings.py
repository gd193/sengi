"""Ray/line crossings against analytic cases and a float64 oracle."""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from sengi import bvh, queries
from sengi.bvh import Config
from tests.support._warp_helpers import TRIANGLES, BUILDERS, _mesh, _build_mesh as _build


def _oracle(vertices, faces, origin, direction, low, high, excluded=-1):
    triangles = np.asarray(vertices, np.float64)[faces]
    a, b, c = triangles.transpose(1, 0, 2)
    e1, e2 = b - a, c - a
    h = np.cross(direction, e2)
    det = np.einsum("ij,ij->i", e1, h)
    valid = det != 0
    inv = np.divide(1.0, det, out=np.zeros_like(det), where=valid)
    s = origin - a
    u = np.einsum("ij,ij->i", s, h) * inv
    q = np.cross(s, e1)
    v = (q @ direction) * inv
    t = np.einsum("ij,ij->i", e2, q) * inv
    # Random oracle rays avoid edges/vertices; analytic tests cover ownership.
    valid &= (u >= 0) & (v >= 0) & (u + v <= 1) & (t >= low) & (t <= high)
    valid &= np.arange(len(faces)) != excluded
    ids = np.flatnonzero(valid)
    return t[ids], ids


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("line", [False, True])
@pytest.mark.parametrize("shape", ["sphere", "torus"])
def test_warp_crossings_matches_oracle(builder, line, shape):
    mesh = (
        trimesh.creation.icosphere(subdivisions=2)
        if shape == "sphere"
        else trimesh.creation.torus(
            major_radius=0.7, minor_radius=0.3, major_sections=24, minor_sections=12
        )
    )
    a = _build(mesh, builder)
    rng = np.random.default_rng(231)
    origins = rng.uniform(-1.2, 1.2, (89, 3)).astype(np.float32)
    directions = rng.normal(size=origins.shape).astype(np.float32)
    low = np.full(len(origins), -np.inf if line else 0, np.float32)
    high = np.full(len(origins), np.inf, np.float32)
    low[:10] = -0.3 if line else 0.1
    high[:10] = 0.5
    exclude = rng.integers(-1, len(mesh.faces), len(origins), dtype=np.int32)
    count, status = jax.jit(
        partial(queries.rays.bind(a.spec, "count", exclude=True), a.data)
    )(origins, directions, high, min_t=low, exclude_face=exclude)
    times, ids, totals, stored_status = jax.jit(
        lambda o, d, lo, hi, ex: queries.rays.bind(a.spec, "all", exclude=True)(
            a.data, o, d, hi, min_t=lo, max_hits=8, exclude_face=ex
        )
    )(origins, directions, low, high, exclude)
    np.testing.assert_array_equal(count, totals)
    np.testing.assert_array_equal(status, stored_status)
    for i in range(len(origins)):
        t, face = _oracle(
            np.asarray(mesh.vertices, np.float32),
            mesh.faces,
            origins[i],
            directions[i],
            low[i],
            high[i],
            exclude[i],
        )
        n = len(t)
        assert int(count[i]) == n
        assert int(status[i]) == int(n > 0)
        order = np.argsort(np.asarray(ids[i, :n]))
        np.testing.assert_array_equal(np.asarray(ids[i, :n])[order], face)
        np.testing.assert_allclose(
            np.asarray(times[i, :n])[order], t, atol=3e-6, rtol=3e-5
        )
        np.testing.assert_array_equal(ids[i, n:], -1)
        assert np.isposinf(times[i, n:]).all()


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("leaf", [1, 4, 32])
def test_warp_crossings_shared_diagonals_winding_and_signed_bounds(builder, leaf):
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    mesh.faces[::2] = mesh.faces[::2, ::-1]
    a = _build(mesh, builder, leaf)
    origins = jnp.array([[-2, 0, 0], [0, 0, 0], [2, 0, 0]], jnp.float32)
    for axis in range(3):
        o = jnp.roll(origins, axis, axis=1)
        d = jnp.broadcast_to(jnp.eye(3)[axis], o.shape)
        np.testing.assert_array_equal(
            queries.rays.bind(a.spec, "count", exclude=True)(a.data, o, d)[0], [2, 1, 0]
        )
        np.testing.assert_array_equal(
            queries.rays.bind(a.spec, "count", exclude=True)(
                a.data, o, d, min_t=-jnp.inf
            )[0],
            [2, 2, 2],
        )
        np.testing.assert_array_equal(
            queries.rays.bind(a.spec, "count", exclude=True)(a.data, o, -d)[0],
            [0, 1, 2],
        )
    times, ids, n, status = queries.rays.bind(a.spec, "all", exclude=True)(
        a.data, origins[0], jnp.array([1.0, 0, 0]), 3.0, min_t=1.0, max_hits=2
    )
    np.testing.assert_array_equal(np.sort(times), [1, 3])
    assert int(n) == 2 and int(status) == 1
    assert (
        int(
            queries.rays.bind(a.spec, "count", exclude=True)(
                a.data, origins[0], jnp.array([1.0, 0, 0]), 1.0, min_t=1.0
            )[0]
        )
        == 1
    )
    assert (
        int(
            queries.rays.bind(a.spec, "count", exclude=True)(
                a.data,
                origins[0],
                jnp.array([1.0, 0, 0]),
                exclude_face=ids[np.argmin(times)],
            )[0]
        )
        == 1
    )
    origin_face = ids[np.argmin(times)]
    assert (
        int(
            queries.rays.bind(a.spec, "count", exclude=True)(
                a.data,
                jnp.array([-1.0, 0, 0]),
                jnp.array([1.0, 0, 0]),
                exclude_face=origin_face,
            )[0]
        )
        == 1
    )


def test_warp_crossings_overflow_reports_full_count():
    meshes = [trimesh.creation.box(extents=[s, s, s]) for s in (2, 4, 6, 8, 10)]
    a = _build(trimesh.util.concatenate(meshes))
    o, d = jnp.zeros(3), jnp.array([1.0, 0, 0])
    times, faces, count, status = queries.rays.bind(a.spec, "all", exclude=True)(
        a.data, o, d, min_t=-jnp.inf, max_hits=3
    )
    assert int(count) == 10 and int(status) == -2
    assert np.isfinite(times).all() and np.all(np.asarray(faces) >= 0)
    assert (
        int(
            queries.rays.bind(a.spec, "count", exclude=True)(
                a.data, o, d, min_t=-jnp.inf
            )[0]
        )
        == 10
    )
    times, _, count, status = queries.rays.bind(a.spec, "all", exclude=True)(
        a.data, o, d, min_t=-jnp.inf, max_hits=10
    )
    assert int(count) == 10 and int(status) == 1
    np.testing.assert_array_equal(np.sort(times), [-5, -4, -3, -2, -1, 1, 2, 3, 4, 5])


@pytest.mark.parametrize("builder", BUILDERS)
def test_warp_crossings_batching_cpu_and_dynamic_geometry(builder):
    a = _build(trimesh.creation.box(extents=[2, 2, 2]), builder)
    b = _build(trimesh.creation.box(extents=[4, 4, 4]), builder)
    o = jnp.array([[[0, 0, 0], [3, 0, 0]], [[1.5, 0, 0], [-3, 0, 0]]], jnp.float32)
    d = jnp.broadcast_to(jnp.array([1.0, 0, 0]), o.shape)
    count = jax.jit(
        jax.vmap(
            jax.vmap(
                queries.rays.bind(a.spec, "count", exclude=True), in_axes=(None, 0, 0)
            ),
            in_axes=(None, 0, 0),
        )
    )
    np.testing.assert_array_equal(count(a.data, o, d)[0], [[1, 0], [0, 2]])
    np.testing.assert_array_equal(count(b.data, o, d)[0], [[1, 0], [1, 2]])
    all_hits = jax.jit(
        jax.vmap(
            jax.vmap(
                lambda payload, x, v: queries.rays.bind(a.spec, "all", exclude=True)(
                    payload, x, v, max_hits=3
                ),
                in_axes=(None, 0, 0),
            ),
            in_axes=(None, 0, 0),
        )
    )
    result = all_hits(a.data, o, d)
    assert result[0].shape == (2, 2, 3)
    np.testing.assert_array_equal(result[2], [[1, 0], [0, 2]])
    cpu = jax.devices("cpu")[0]
    payload, oc, dc = (jax.device_put(x, cpu) for x in (a.data, o, d))
    cpu_result = all_hits(payload, oc, dc)
    assert cpu_result[0].device.platform == "cpu"
    for expected, actual in zip(result, cpu_result, strict=True):
        np.testing.assert_allclose(actual, expected)
    assert queries.rays.bind(a.spec, "count", exclude=True)(
        a.data, jnp.empty((2, 0, 3)), jnp.empty((2, 0, 3))
    )[0].shape == (2, 0)
    assert queries.rays.bind(a.spec, "all", exclude=True)(
        a.data, jnp.empty((2, 0, 3)), jnp.empty((2, 0, 3)), max_hits=4
    )[0].shape == (2, 0, 4)


def test_warp_crossings_invalid_inputs_and_empty_geometry():
    a = _build(trimesh.creation.box())
    points = jnp.array([[jnp.nan, 0, 0], [jnp.inf, 0, 0], [0, 0, 0]], jnp.float32)
    directions = jnp.array([[1, 0, 0], [1, 0, 0], [0, 0, 0]], jnp.float32)
    n, status = queries.rays.bind(a.spec, "count", exclude=True)(
        a.data, points, directions
    )
    np.testing.assert_array_equal(n, -1)
    np.testing.assert_array_equal(status, -1)
    o, d = jnp.zeros(3), jnp.array([1.0, 0, 0])
    for lo, hi, ex in [
        (1, 0, -1),
        (jnp.nan, 1, -1),
        (0, jnp.nan, -1),
        (jnp.inf, jnp.inf, -1),
        (-jnp.inf, -jnp.inf, -1),
        (0, 1, -2),
        (0, 1, 12),
        (0, 1, jnp.uint32(2**32 - 1)),
    ]:
        ts, ids, n, status = queries.rays.bind(a.spec, "all", exclude=True)(
            a.data, o, d, hi, min_t=lo, exclude_face=ex, max_hits=2
        )
        assert int(n) == -1 and int(status) == -1
        assert np.isposinf(ts).all()
        np.testing.assert_array_equal(ids, -1)
    for capacity in (0, -1, 2**31):
        with pytest.raises(ValueError, match="capacity"):
            queries.rays.bind(a.spec, "all", exclude=True)(
                a.data, o, d, max_hits=capacity
            )[:3]
    with pytest.raises(TypeError):
        queries.rays.bind(a.spec, "all", exclude=True)(a.data, o, d, max_hits=1.5)[:3]
    with pytest.raises(TypeError, match="integer"):
        queries.rays.bind(a.spec, "count", exclude=True)(
            a.data, o, d, exclude_face=1.5
        )[0]
    with pytest.raises(ValueError, match="batch shapes"):
        queries.rays.bind(a.spec, "count", exclude=True)(a.data, jnp.zeros((2, 3)), d)[
            0
        ]
    empty = bvh.build(
        TRIANGLES, _mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int32))[0], builder="sah"
    )
    times, ids, n, status = queries.rays.bind(empty.spec, "all", exclude=True)(
        empty.data, o, d, max_hits=2
    )
    assert int(n) == 0 and int(status) == 0
    assert np.isposinf(times).all()
    np.testing.assert_array_equal(ids, -1)


@pytest.mark.parametrize("scale,shift", [(1.0e-3, 0), (1.0, 10000), (1.0e3, -10000)])
def test_warp_crossings_scale_translation_and_coplanar_faces(scale, shift):
    mesh = trimesh.creation.box(extents=[2 * scale] * 3)
    mesh.apply_translation([shift] * 3)
    a = _build(mesh)
    o, d = jnp.full(3, shift, jnp.float32), jnp.array([scale, 0, 0], jnp.float32)
    times, _, n, status = queries.rays.bind(a.spec, "all", exclude=True)(
        a.data, o, d, min_t=-jnp.inf, max_hits=2
    )
    np.testing.assert_allclose(np.sort(times), [-1, 1], rtol=3e-5, atol=3e-6)
    assert int(n) == 2 and int(status) == 1
    vertices = np.array([[0, 0, 0], [2, 0, 0], [0, 2, 0]], np.float32)
    a = bvh.build(TRIANGLES, _mesh(vertices, [[0, 1, 2]])[0], builder="sah")
    assert (
        int(
            queries.rays.bind(a.spec, "count", exclude=True)(
                a.data, jnp.array([0.2, 0.2, 0]), jnp.array([1.0, 0, 0]), min_t=-jnp.inf
            )[0]
        )
        == 0
    )


def test_warp_crossings_degenerate_faces_and_many_hits():
    n = 129
    vertices = np.array(
        [[[i, -1, -1], [i, 1, -1], [i, 0, 1]] for i in range(1, n + 1)], np.float32
    )
    faces = np.arange(3 * n, dtype=np.int32).reshape(-1, 3)
    vertices = vertices.reshape(-1, 3)
    a = bvh.build(
        TRIANGLES,
        _mesh(vertices, faces)[0],
        builder="sah",
        config=Config(leaf_size_threshold=4),
    )
    o, d = jnp.zeros(3), jnp.array([1.0, 0, 0])
    assert int(queries.rays.bind(a.spec, "count", exclude=True)(a.data, o, d)[0]) == n
    times, _, count, status = queries.rays.bind(a.spec, "all", exclude=True)(
        a.data, o, d, max_hits=n
    )
    assert int(count) == n and int(status) == 1
    np.testing.assert_array_equal(np.sort(times), np.arange(1, n + 1))
    a = bvh.build(
        TRIANGLES,
        _mesh([[0, 0, 0], [1, 0, 0], [2, 0, 0]], [[0, 1, 2]])[0],
        builder="sah",
    )
    assert (
        int(
            queries.rays.bind(a.spec, "count", exclude=True)(
                a.data, o, d, min_t=-jnp.inf
            )[0]
        )
        == 0
    )


@pytest.mark.parametrize("magnitude", [1.0e-12, 1.0e12])
def test_warp_crossings_direction_scaling(magnitude):
    a = _build(trimesh.creation.box(extents=[2, 2, 2]))
    o, d = jnp.zeros(3), jnp.array([magnitude, 0, 0], jnp.float32)
    ts, _, n, status = queries.rays.bind(a.spec, "all", exclude=True)(
        a.data, o, d, min_t=-jnp.inf, max_hits=2
    )
    assert int(n) == 2 and int(status) == 1
    np.testing.assert_allclose(np.sort(ts), [-1 / magnitude, 1 / magnitude], rtol=3e-5)


@pytest.mark.parametrize("dtype", [jnp.int8, jnp.uint8, jnp.uint32])
def test_warp_crossings_integer_face_ids_broadcast(dtype):
    a = _build(trimesh.creation.box(extents=[2, 2, 2]))
    o, d = jnp.array([-2.0, 0, 0]), jnp.array([1.0, 0, 0])
    times, faces, _ = queries.rays.bind(a.spec, "all", exclude=True)(
        a.data, o, d, max_hits=2
    )[:3]
    excluded = jnp.full((2, 1), faces[jnp.argmin(times)], dtype=dtype)
    origins = jnp.broadcast_to(o, (2, 3, 3))
    directions = jnp.broadcast_to(d, origins.shape)
    np.testing.assert_array_equal(
        queries.rays.bind(a.spec, "count", exclude=True)(
            a.data, origins, directions, exclude_face=excluded
        )[0],
        1,
    )


def test_warp_crossings_exact_endpoint_nonaligned_rays():
    a = bvh.build(
        TRIANGLES,
        _mesh([[0, -1, -1], [0, 1, -1], [0, 0, 1]], [[0, 1, 2]])[0],
        builder="sah",
    )
    directions = np.random.default_rng(2).uniform(0.1, 10, (1000, 3)).astype(np.float32)
    directions[::2] *= -1
    for device in (jax.devices()[0], jax.devices("cpu")[0]):
        payload, d = (jax.device_put(v, device) for v in (a.data, directions))
        for endpoint in (-1.0, 1.0):
            o = -endpoint * d
            result = jax.jit(
                lambda tree, x, v: queries.rays.bind(a.spec, "all", exclude=True)(
                    tree, x, v, endpoint, min_t=endpoint, max_hits=1
                )
            )(payload, o, d)
            np.testing.assert_array_equal(result[0], endpoint)
            np.testing.assert_array_equal(result[2:], 1)

"""Robin radius contracts and exhaustive geometry checks."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from functools import partial

from sengi import bvh, queries
from sengi.bvh import Config
from sengi.boundary.robin import prepare_robin
from sengi.metrics.nearest import Euclidean

from tests.support._robin_oracle import brute_robin_radius
from tests.support._warp_helpers import TRIANGLES, _mesh


def _tree(vertices, faces, leaf_size=1):
    source = _mesh(vertices, faces)[0]
    index = bvh.build(TRIANGLES, source, config=Config(leaf_size_threshold=leaf_size))
    return index, bvh.prepare_boundary(index, source)


def test_robin_mu_zero_is_existing_silhouette_radius():
    # A single open triangle has boundary silhouette edges at every view.
    vertices = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    faces = [[0, 1, 2]]
    tree, boundary = _tree(vertices, faces)
    robin = prepare_robin(boundary, 0.0)
    points = jnp.asarray([[0.2, 0.3, 1.0], [0.6, 0.4, 2.0]], dtype=jnp.float32)
    cap = jnp.asarray([10.0, 0.5], dtype=jnp.float32)
    expected = queries.star_radius.bind(tree.spec)(boundary, points, cap)[0]
    actual, status = jax.jit(
        jax.vmap(partial(queries.robin_radius.bind(tree.spec), robin))
    )(points, cap)
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)
    np.testing.assert_array_equal(status, [1, 1])


def test_robin_infinite_coefficient_is_capped_closest_surface_distance():
    vertices = [[0, 0, 0], [2, 0, 0], [0, 2, 0]]
    tree, boundary = _tree(vertices, [[0, 1, 2]])
    robin = prepare_robin(boundary, np.inf)
    points = jnp.asarray(
        [[0.2, 0.3, 1.0], [3.0, 3.0, 2.0], [0.2, 0.3, 0.0]], dtype=jnp.float32
    )
    caps = jnp.asarray([10.0, 10.0, 3.0], dtype=jnp.float32)
    radius = queries.robin_radius.bind(tree.spec)
    actual, _ = jax.jit(jax.vmap(partial(radius, robin)))(points, caps)
    _, outside_distance, _, _ = queries.nearest.bind(tree.spec, Euclidean())(
        tree.data, points[1]
    )
    expected = np.minimum(
        np.asarray(caps), np.asarray([1.0, float(outside_distance), 0.0])
    )
    np.testing.assert_allclose(actual, expected, atol=2e-5, rtol=2e-5)


def test_robin_finite_coefficient_matches_triangle_bound_and_monotonicity():
    # The center has no visible silhouette; Robin still restricts its radius.
    vertices = [
        [-1, -1, -1],
        [1, -1, -1],
        [1, 1, -1],
        [-1, 1, -1],
        [-1, -1, 1],
        [1, -1, 1],
        [1, 1, 1],
        [-1, 1, 1],
    ]
    faces = [
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [1, 2, 6],
        [1, 6, 5],
        [2, 3, 7],
        [2, 7, 6],
        [3, 0, 4],
        [3, 4, 7],
    ]
    tree, boundary = _tree(vertices, faces)
    point = jnp.asarray([0.0, 0.0, 0.0], dtype=jnp.float32)
    cap = jnp.asarray(10.0, dtype=jnp.float32)

    coefficients = [0.5, 1.0, 2.0, 4.0, 8.0]
    radii = []
    for mu in coefficients:
        robin = prepare_robin(boundary, mu)
        got = float(
            queries.robin_radius.bind(tree.spec)(robin, point, cap)[0]
        )
        radii.append(got)
        expected = brute_robin_radius(vertices, faces, np.zeros(3), mu, 10.0)
        np.testing.assert_allclose(got, expected, atol=2e-5, rtol=2e-5)
    assert np.all(np.diff(radii) <= 2e-5)


def test_robin_interval_uses_upper_bound_and_updates_share_geometry():
    tree, boundary = _tree([[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    point = jnp.asarray([0.0, 0.0, 1.0], dtype=jnp.float32)
    cap = jnp.asarray(10.0, dtype=jnp.float32)
    lower_zero = prepare_robin(boundary, 0.0, maximum=2.0)
    lower_positive = prepare_robin(boundary, 1.0, maximum=2.0)
    np.testing.assert_allclose(
        queries.robin_radius.bind(tree.spec)(lower_zero, point, cap)[0],
        queries.robin_radius.bind(tree.spec)(lower_positive, point, cap)[0],
        atol=2e-5,
        rtol=2e-5,
    )
    updated = prepare_robin(boundary, 1.5, maximum=3.0)
    assert updated.core is lower_positive.core is boundary.core
    assert updated.silhouette is lower_positive.silhouette is boundary.silhouette
    np.testing.assert_array_equal(
        lower_positive.coefficients.primitive_bounds, [[1.0, 2.0]]
    )
    np.testing.assert_array_equal(updated.coefficients.primitive_bounds, [[1.5, 3.0]])


def test_robin_nested_vmap_over_shared_geometry():
    tree, boundary = _tree([[0, 0, 0], [2, 0, 0], [0, 2, 0]], [[0, 1, 2]])
    robin = prepare_robin(boundary, np.inf)
    points = jnp.asarray(
        [[[0.2, 0.3, 1.0], [0.3, 0.2, 2.0]], [[0.5, 0.5, 1.0], [0.1, 0.1, 3.0]]],
        dtype=jnp.float32,
    )
    caps = jnp.asarray([[2.0, 1.0], [4.0, 2.0]], dtype=jnp.float32)
    radius = queries.robin_radius.bind(tree.spec)
    query = jax.jit(jax.vmap(jax.vmap(partial(radius, robin))))
    actual, _ = query(points, caps)
    np.testing.assert_allclose(
        actual, np.minimum(caps, points[..., 2]), atol=2e-5, rtol=2e-5
    )


def test_robin_face_coefficients_follow_original_face_order():
    # Face order opposes BVH order; the distant face must not change the bound.
    vertices = [[-3, -1, 0], [-1, -1, 0], [-2, 1, 0], [1, -1, 0], [3, -1, 0], [2, 1, 0]]
    faces = [[3, 4, 5], [0, 1, 2]]
    tree, boundary = _tree(vertices, faces, leaf_size=1)
    np.testing.assert_array_equal(tree.data.tree.prim_ids, [1, 0])
    p = jnp.asarray([-2.0, 0.0, 1.0], dtype=jnp.float32)
    cap = jnp.asarray(10.0, dtype=jnp.float32)
    radius = queries.robin_radius.bind(tree.spec)
    a, b = [
        radius(prepare_robin(boundary, mu), p, cap)[0]
        for mu in ([8.0, 0.5], [0.1, 0.5])
    ]
    np.testing.assert_allclose(a, b, atol=2e-5, rtol=2e-5)


def test_robin_invalid_query_status_and_host_validation():
    tree, boundary = _tree([[0, 0, 0], [2, 0, 0], [0, 2, 0]], [[0, 1, 2]])
    robin = prepare_robin(boundary, 0.5)
    points = jnp.asarray(
        [[jnp.nan, 0, 0], [0, jnp.inf, 0], [0, 0, 1], [0, 0, 1]], dtype=jnp.float32
    )
    caps = jnp.asarray([1.0, 1.0, jnp.nan, -1.0], dtype=jnp.float32)
    radius, status = jax.jit(
        jax.vmap(partial(queries.robin_radius.bind(tree.spec), robin))
    )(points, caps)
    np.testing.assert_array_equal(status, [-1, -1, -1, -1])
    assert np.isnan(np.asarray(radius)).all()

    for bad in (-1.0, np.nan, [-1.0], [0.0, 1.0], [np.inf, np.nan], 1e40, 1j):
        with pytest.raises((ValueError, TypeError)):
            prepare_robin(boundary, bad)
    with pytest.raises((ValueError, TypeError)):
        prepare_robin(boundary, 0.5, maximum=-0.1)


def test_robin_empty_geometry_zero_cap_and_dynamic_cap():
    tree, boundary = _tree(np.zeros((0, 3), np.float32), np.zeros((0, 3), np.int32))
    robin = prepare_robin(boundary, 1.0)
    points = jnp.asarray([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]], dtype=jnp.float32)
    query = jax.jit(
        jax.vmap(partial(queries.robin_radius.bind(tree.spec), robin))
    )
    for caps in ([0.0, 2.0], [1.0, 0.0], [1.0e-30, jnp.inf]):
        caps = jnp.asarray(caps, jnp.float32)
        radius, status = query(points, caps)
        np.testing.assert_array_equal(radius, caps)
        np.testing.assert_array_equal(status, [1, 1])


@pytest.mark.parametrize("shape", ["sphere", "torus"])
@pytest.mark.parametrize(
    "coefficient", [None, 1.0, 10.0], ids=["varying", "mu1", "mu10"]
)
def test_robin_tree_pruning_matches_exhaustive_float64_oracle(shape, coefficient):
    mesh = (
        trimesh.creation.icosphere(subdivisions=1)
        if shape == "sphere"
        else trimesh.creation.torus(
            major_radius=0.7, minor_radius=0.3, major_sections=12, minor_sections=8
        )
    )
    vertices, faces = (
        np.asarray(mesh.vertices, np.float32),
        np.asarray(mesh.faces, np.int32),
    )
    rng = np.random.default_rng(1053)
    coefficients = np.power(10.0, rng.uniform(-2.0, 2.0, len(faces))).astype(np.float32)
    if coefficient is not None:
        coefficients.fill(coefficient)
    points = rng.uniform(-1.4, 1.4, (36, 3)).astype(np.float32)
    caps = np.resize(np.asarray([0.125, 1.0, 5.0, 100.0], np.float32), len(points))
    tree, boundary = _tree(vertices, faces, leaf_size=2)
    robin = prepare_robin(boundary, coefficients * 0.1, coefficients)
    radius, status = jax.jit(
        partial(queries.robin_radius.bind(tree.spec), robin)
    )(points, caps)
    expected = [
        brute_robin_radius(vertices, faces, p, coefficients, cap)
        for p, cap in zip(points, caps, strict=True)
    ]
    np.testing.assert_array_equal(status, np.ones(len(points), np.int32))
    np.testing.assert_allclose(radius, expected, atol=3e-5, rtol=1e-4)


def test_robin_cpu_execution_and_dynamic_coefficients():
    tree, boundary = _tree([[-10, -10, 0], [10, -10, 0], [0, 10, 0]], [[0, 1, 2]])
    cpu = jax.devices("cpu")[0]
    point = jax.device_put(np.asarray([0.0, 0.0, 1.0], np.float32), cpu)
    cap = jax.device_put(np.float32(10), cpu)
    query = jax.jit(queries.robin_radius.bind(tree.spec))
    outputs = []
    for coefficient in (1.0, 4.0):
        metadata = jax.device_put(
            prepare_robin(boundary, coefficient), cpu
        )
        assert all(
            x.device.platform == "cpu"
            for x in jax.tree_util.tree_leaves(metadata)
        )
        radius, status = query(metadata, point, cap)
        assert radius.device.platform == status.device.platform == "cpu"
        assert int(status) == 1
        outputs.append(float(radius))
    assert outputs[1] < outputs[0]


def test_robin_coefficient_interval_rounding():
    tree, boundary = _tree([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 2]])
    with pytest.raises(ValueError, match="minimum exceeds"):
        prepare_robin(boundary, 1.00000004, 1.00000003)
    value = 0.1
    configured = prepare_robin(boundary, value)
    lower, upper = np.asarray(configured.coefficients.primitive_bounds)[0].astype(
        np.float64
    )
    assert lower <= value <= upper


@pytest.mark.parametrize("scale", [1e-3, 1.0, 1e3])
def test_robin_radius_scales_with_geometry_and_inverse_coefficient(scale):
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    tree, boundary = _tree(np.asarray(mesh.vertices) * scale, mesh.faces)
    metadata = prepare_robin(boundary, 1.0 / scale)
    radius, status = queries.robin_radius.bind(tree.spec)(
        metadata, jnp.zeros(3), 10.0 * scale
    )
    assert int(status) == 1
    np.testing.assert_allclose(radius, 1.5 * np.sqrt(3.0) * scale, rtol=2e-5)


def test_robin_coplanar_queries_and_empty_batch():
    tree, boundary = _tree([[0, 0, 0], [2, 0, 0], [0, 2, 0]], [[0, 1, 2]])
    robin = prepare_robin(boundary, 0.1)
    points = jnp.asarray([[0.2, 0.3, 0.0], [3.0, 3.0, 0.0]], dtype=jnp.float32)
    radius, status = jax.jit(
        partial(queries.robin_radius.bind(tree.spec), robin)
    )(points, 10.0)
    np.testing.assert_array_equal(status, [1, 1])
    np.testing.assert_allclose(radius, [0.0, np.sqrt(8.0)], atol=2e-5, rtol=2e-5)
    empty_radius, empty_status = jax.jit(
        partial(queries.robin_radius.bind(tree.spec), robin)
    )(jnp.zeros((0, 3), jnp.float32), jnp.zeros(0, jnp.float32))
    assert empty_radius.shape == empty_status.shape == (0,)


def test_robin_infinite_cap_preserves_unrestricted_radius():
    mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
    tree, boundary = _tree(mesh.vertices, mesh.faces)
    for coefficient in (0.0, 0.01):
        metadata = prepare_robin(boundary, coefficient)
        radius, status = queries.robin_radius.bind(tree.spec)(
            metadata, jnp.zeros(3), jnp.inf
        )
        assert int(status) == 1 and np.isposinf(float(radius))


def test_robin_radius_respects_triangle_distance_interval():
    mesh = trimesh.creation.icosphere(subdivisions=5)
    tree, boundary = _tree(mesh.vertices, mesh.faces, leaf_size=4)
    robin = prepare_robin(boundary, 1.0)
    points = jnp.asarray(
        [
            [-0.21108818, 0.40054315, -0.81500834],
            [0.77663988, -0.34156108, -0.27615789],
            [0.74352419, -0.45358282, 0.13813464],
            [-0.83153093, -0.51310819, 0.03087507],
            [0.23223017, -0.28848827, -0.75982475],
        ],
        dtype=jnp.float32,
    )
    # Exhaustive float64 bounds on the same float32 mesh.
    expected = [1.695200758, 2.0, 2.0, 0.905456576, 2.0]
    radius, status = jax.jit(
        partial(queries.robin_radius.bind(tree.spec), robin)
    )(points, 2.0)
    np.testing.assert_array_equal(status, [1, 1, 1, 1, 1])
    np.testing.assert_allclose(radius, expected, atol=3e-5, rtol=3e-5)

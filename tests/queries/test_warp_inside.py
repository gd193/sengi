"""Classification against solid angles and analytic closed meshes."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import trimesh

from sengi import bvh, queries
from tests.support._warp_helpers import BUILDERS, TRIANGLES, _mesh, _build_mesh


def _build(mesh, builder="sah", leaf=4):
    index = _build_mesh(mesh, builder, leaf)
    source = _mesh(mesh.vertices, mesh.faces)[0]
    return index, bvh.prepare_boundary(index, source, silhouettes=False)


def _winding(vertices, faces, points):
    triangles = np.asarray(vertices, np.float64)[faces]
    result = []
    for p in points:
        a, b, c = (triangles - p).transpose(1, 0, 2)
        la, lb, lc = (np.linalg.norm(v, axis=1) for v in (a, b, c))
        numerator = np.einsum("ij,ij->i", a, np.cross(b, c))
        denominator = (
            la * lb * lc
            + np.einsum("ij,ij->i", a, b) * lc
            + np.einsum("ij,ij->i", b, c) * la
            + np.einsum("ij,ij->i", c, a) * lb
        )
        result.append(
            -1 if abs(np.sum(2 * np.arctan2(numerator, denominator))) > 2 * np.pi else 1
        )
    return result


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("shape", ["sphere", "torus"])
def test_warp_inside_matches_solid_angles(builder, shape):
    mesh = (
        trimesh.creation.icosphere(subdivisions=2)
        if shape == "sphere"
        else trimesh.creation.torus(
            major_radius=0.7, minor_radius=0.3, major_sections=24, minor_sections=12
        )
    )
    a, boundary = _build(mesh, builder)
    points = np.random.default_rng(407).uniform(-1.2, 1.2, (181, 3)).astype(np.float32)
    points = np.concatenate([points, [[0, 0, 0], [0.7, 0, 0]]]).astype(np.float32)
    sign, status = jax.jit(queries.classification.bind(a.spec))(
        boundary, points
    )
    np.testing.assert_array_equal(
        sign, _winding(np.asarray(mesh.vertices, np.float32), mesh.faces, points)
    )
    np.testing.assert_array_equal(status, 1)


@pytest.mark.parametrize("builder", BUILDERS)
@pytest.mark.parametrize("leaf", [1, 4, 32])
def test_warp_inside_shared_edges_vertices_and_boundary(builder, leaf):
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    # All these +X rays cross triangulation diagonals, edges or vertices.
    points = np.array(
        [
            [0, 0, 0],
            [-2, 0, 0],
            [2, 0, 0],
            [-2, 1, 1],
            [-2, -1, -1],
            [-2, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0.5, 0.25],
            [1, 1, 0],
            [-1, -1, -1],
        ],
        np.float32,
    )
    expected = [-1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
    # Independent face reversal must preserve parity and edge ownership.
    rng = np.random.default_rng(88)
    for reverse in (False, True):
        if reverse:
            mask = rng.random(len(mesh.faces)) < 0.5
            mesh.faces[mask] = mesh.faces[mask, ::-1]
            rng.shuffle(mesh.faces)
        a, boundary = _build(mesh, builder, leaf)
        signs, statuses = jax.jit(queries.classification.bind(a.spec))(
            boundary, points
        )
        np.testing.assert_array_equal(signs, expected)
        np.testing.assert_array_equal(statuses, 1)


@pytest.mark.parametrize("scale,shift", [(1e-3, 0), (1, 10000), (1e3, -10000)])
def test_warp_inside_scales_and_nextafter_contacts(scale, shift):
    mesh = trimesh.creation.box(extents=[2, 2, 2])
    mesh.vertices = mesh.vertices * scale + shift
    a, boundary = _build(mesh)
    lo, hi = np.asarray(mesh.bounds, np.float32)
    points = np.tile(np.float32(shift), (7, 3))
    points[1] = hi
    points[2, 0] = np.nextafter(hi[0], np.float32(np.inf))
    points[3, 0] = np.nextafter(hi[0], np.float32(-np.inf))
    points[4, 1] = np.nextafter(lo[1], np.float32(-np.inf))
    points[5, 1] = np.nextafter(lo[1], np.float32(np.inf))
    points[6, 2] = hi[2]
    np.testing.assert_array_equal(
        queries.classification.bind(a.spec)(boundary, points)[0],
        [-1, 0, 1, -1, 1, -1, 0],
    )


def test_warp_inside_disconnected_and_nested_shells():
    outer = trimesh.creation.box(extents=[4, 4, 4])
    inner = trimesh.creation.box(extents=[2, 2, 2])
    other = trimesh.creation.box(extents=[2, 2, 2])
    other.apply_translation([5, 0, 0])
    a, boundary = _build(trimesh.util.concatenate([outer, inner, other]))
    np.testing.assert_array_equal(
        queries.classification.bind(a.spec)(
            boundary,
            jnp.array([[0, 0, 0], [1.5, 0, 0], [3, 0, 0], [5, 0, 0], [7, 0, 0]]),
        )[0],
        [1, -1, 1, -1, 1],
    )


@pytest.mark.parametrize("damage", ["open", "nonmanifold", "degenerate"])
def test_warp_inside_invalid_meshes_fail(damage):
    mesh = trimesh.creation.box()
    if damage == "open":
        mesh.update_faces(np.arange(len(mesh.faces) - 1))
    elif damage == "nonmanifold":
        mesh.faces = np.concatenate([mesh.faces, mesh.faces[:1]])
    else:
        mesh.vertices[mesh.faces[0, 1]] = mesh.vertices[mesh.faces[0, 0]]
    a, boundary = _build(mesh)
    sign, status = queries.classification.bind(a.spec)(
        boundary, jnp.array([[0, 0, 0], [10, 10, 10]])
    )
    assert np.isnan(sign).all()
    np.testing.assert_array_equal(status, -1)


def test_warp_inside_invalid_points_empty_mesh_and_batch():
    a, boundary = _build(trimesh.creation.box())
    points = jnp.array([[jnp.nan, 0, 0], [0, jnp.inf, 0], [0, 0, -jnp.inf]])
    sign, status = queries.classification.bind(a.spec)(boundary, points)
    assert np.isnan(sign).all()
    np.testing.assert_array_equal(status, -1)
    source = _mesh(np.zeros((0, 3)), np.zeros((0, 3), np.int32))[0]
    empty = bvh.build(TRIANGLES, source)
    empty_boundary = bvh.prepare_boundary(empty, source, silhouettes=False)
    sign, status = queries.classification.bind(empty.spec)(
        empty_boundary, jnp.zeros((3, 3))
    )
    np.testing.assert_array_equal(sign, 1)
    np.testing.assert_array_equal(status, 0)
    assert queries.classification.bind(a.spec)(boundary, jnp.empty((2, 0, 3)))[
        0
    ].shape == (2, 0)
    with pytest.raises(ValueError, match="query shape"):
        queries.classification.bind(a.spec)(boundary, jnp.zeros(2))[0]


@pytest.mark.parametrize("builder", BUILDERS)
def test_warp_inside_cpu_nested_vmap_and_dynamic_geometry(builder):
    a, boundary = _build(trimesh.creation.box(extents=[2, 2, 2]), builder)
    b, b_boundary = _build(trimesh.creation.box(extents=[4, 4, 4]), builder)
    points = jnp.array([[[0, 0, 0], [1.5, 0, 0]], [[0, 1, 0], [3, 3, 3]]], jnp.float32)
    classify = queries.classification.bind(a.spec)
    query = jax.jit(
        jax.vmap(jax.vmap(classify, in_axes=(None, 0)), in_axes=(None, 0))
    )
    np.testing.assert_array_equal(query(boundary, points)[0], [[-1, 1], [0, 1]])
    np.testing.assert_array_equal(
        query(b_boundary, points)[0], [[-1, -1], [-1, 1]]
    )
    cpu = jax.devices("cpu")[0]
    metadata, cpu_points = jax.device_put((boundary, points), cpu)
    output, _ = jax.jit(classify)(metadata, cpu_points)
    assert output.device.platform == "cpu"
    np.testing.assert_array_equal(output, [[-1, 1], [0, 1]])


def test_warp_inside_vertex_rays_and_rotated_contacts():
    mesh = trimesh.creation.icosphere(subdivisions=2)
    a, boundary = _build(mesh)
    vertices = np.asarray(mesh.vertices, np.float32)
    points = vertices - np.array([3.0, 0.0, 0.0], np.float32)
    np.testing.assert_array_equal(
        queries.classification.bind(a.spec)(boundary, points)[0], 1
    )
    np.testing.assert_array_equal(
        queries.classification.bind(a.spec)(boundary, vertices)[0], 0
    )
    # Dyadic coordinates give exactly representable slanted face contacts.
    vertices = np.array([[0, 0, 0], [2, 0, 1], [0, 2, 1], [0, 0, 2]], np.float32)
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
    source = _mesh(vertices, faces)[0]
    a = bvh.build(TRIANGLES, source)
    boundary = bvh.prepare_boundary(a, source, silhouettes=False)
    points = np.array(
        [[0.5, 0.5, 0.5], [1, 1, 1], [0.25, 0.25, 1], [1, 1, 0.5]], np.float32
    )
    np.testing.assert_array_equal(
        queries.classification.bind(a.spec)(boundary, points)[0], [0, 0, -1, 1]
    )


@pytest.mark.parametrize("aspect", [1.0e-4, 1.0, 1.0e4])
def test_warp_inside_thin_rotated_mesh_near_faces(aspect):
    mesh = trimesh.creation.icosphere(subdivisions=1)
    mesh.vertices *= [1.0, aspect, 0.7]
    rotation = trimesh.transformations.rotation_matrix(0.83, [0.2, 0.7, 0.4])
    mesh.apply_transform(rotation)
    a, boundary = _build(mesh)
    vertices = np.asarray(mesh.vertices, np.float32)
    triangles = vertices[mesh.faces].astype(np.float64)
    centers = triangles.mean(axis=1)
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    normals /= np.linalg.norm(normals, axis=1)[:, None]
    delta = 1.0e-5 * min(1.0, aspect)
    points = np.concatenate(
        [centers + delta * normals, centers - delta * normals]
    ).astype(np.float32)
    np.testing.assert_array_equal(
        queries.classification.bind(a.spec)(boundary, points)[0],
        _winding(vertices, mesh.faces, points),
    )

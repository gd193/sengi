"""An indexed convex quad supplies its own primitive and boundary operations."""
from sengi import bvh
from sengi.queries import boundary_sample, classification, region
from dataclasses import dataclass
from pathlib import Path
import shutil
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from sengi.boundary.data import BoundaryAdapter
from sengi.boundary.sampling import MeasureOps, prepare_boundary_sample
from sengi.bvh.device_tree import DeviceTree
from sengi.metrics.nearest import Euclidean
from sengi.queries.knn import bind as bind_knn
from sengi.warp.native import NativeHeaders
from sengi.queries.nearest import bind as bind_nearest
from sengi.queries.rays import bind as bind_rays
from sengi.bvh import Config  # noqa: E402
from tests.support._warp_device import device_payload, run_kernel  # noqa: E402


@wp.struct
class QuadData:
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array(dtype=wp.vec4i)


Quad = wp.types.matrix(shape=(4, 3), dtype=wp.float32)


@wp.func
def load_quad(data: QuadData, slot: int):
    i = data.indices[slot]
    return wp.matrix_from_rows(data.vertices[i[0]], data.vertices[i[1]],
                               data.vertices[i[2]], data.vertices[i[3]])


@wp.func
def quad_bounds(q: Quad):
    return (wp.min(wp.min(q[0], q[1]), wp.min(q[2], q[3])),
            wp.max(wp.max(q[0], q[1]), wp.max(q[2], q[3])))


@wp.func
def closest_on_quad(query: wp.vec3, q: Quad):
    normal = wp.normalize(wp.cross(q[1] - q[0], q[3] - q[0]))
    inside = True
    best = q[0]
    for i in range(4):
        edge = q[(i + 1) % 4] - q[i]
        delta = query - q[i]
        inside = inside and wp.dot(wp.cross(edge, delta), normal) >= 0.
        candidate = q[i] + edge * wp.clamp(wp.dot(delta, edge) / wp.length_sq(edge), 0., 1.)
        if wp.length_sq(query - candidate) < wp.length_sq(query - best):
            best = candidate
    return query - wp.dot(query - q[0], normal) * normal if inside else best


class QuadArrays(NamedTuple):
    vertices: jax.Array
    indices: jax.Array


@dataclass(frozen=True)
class PlanarQuads:
    dimension = 3
    point_type = wp.vec3
    data_type = QuadData
    item_type = Quad
    load = load_quad
    bounds_of_item = quad_bounds
    closest_euclidean = closest_on_quad

    def bounds(self, data):
        corners = np.asarray(data.vertices)[np.asarray(data.indices)]
        return corners.min(axis=1), corners.max(axis=1)

    def pack(self, data, ids):
        return QuadArrays(data.vertices, data.indices[ids])


def test_external_indexed_quads_nearest():
    primitive = PlanarQuads()
    face = np.array([[0, 0, 0], [3, 0, 0], [2, 2, 0], [0, 2, 0]], np.float32)
    vertices = np.vstack((face + [10, 0, 0], face))
    raw = QuadArrays(jnp.asarray(vertices, jnp.float32), jnp.arange(8, dtype=jnp.int32).reshape(2, 4))
    built_index = bvh.build(primitive, raw, config=Config(leaf_size_threshold=1), builder="lbvh")
    tree, data = built_index.data
    np.testing.assert_array_equal(tree.prim_ids, [1, 0])
    with pytest.raises(TypeError, match="region intersection policy"):
        region.bind(built_index.spec)
    with pytest.raises(TypeError, match="intersection"):
        bind_rays(built_index.spec)

    # Interior at its cap, slanted edge, corner and the same interior just outside its cap.
    queries = np.array([[[11, 1, 2], [3, 2, 0]], [[-1, -1, 1], [11, 1, 2]]], np.float32)
    caps = np.array([[2, 1], [2, np.nextafter(np.float32(2), np.float32(0))]], np.float32)
    nearest = bind_nearest(built_index.spec, Euclidean())

    def scalar(q, cap):
        return nearest(built_index.data, q, cap)

    point, distance, index, status = jax.jit(jax.vmap(jax.vmap(scalar)))(queries, caps)
    np.testing.assert_allclose(point, [[[11, 1, 0], [2.2, 1.6, 0]], [[0, 0, 0], [0, 0, 0]]])
    np.testing.assert_allclose(distance, [[2, np.sqrt(.8)], [np.sqrt(3), np.inf]])
    np.testing.assert_array_equal(index, [[0, 1], [1, -1]])
    np.testing.assert_array_equal(status, [[1, 1], [1, 0]])

    closest = nearest.device

    @wp.kernel(module="unique", enable_backward=False)
    def caller(tree: DeviceTree, geometry: QuadData, ids: wp.array(dtype=int)):
        point, distance, index, status = wp.static(closest)(wp.vec3(11., 1., 2.), 2., tree, geometry)
        ids[0] = index

    device = wp.device_from_jax(jax.devices()[0])
    args = [device_payload(DeviceTree, tree, device), device_payload(QuadData, data, device)]
    ids, = run_kernel(caller, 1, args, [(1, int)], queries=(nearest,), device=device)
    np.testing.assert_array_equal(ids, [0])


@wp.func
def uniform_parallelogram(quad: Quad, uv: wp.vec2):
    u, v = quad[1] - quad[0], quad[3] - quad[0]
    normal = wp.cross(u, v)
    return quad[0] + uv[0]*u + uv[1]*v, wp.normalize(normal), 1.0/wp.length(normal)


def test_external_quad_boundary_sampling():
    class Core(NamedTuple):
        primitive_measure: jax.Array

    def prepare(primitive, data):
        a, b, _, d = jnp.moveaxis(data.vertices[data.indices], 1, 0)
        return Core(jnp.linalg.norm(jnp.cross(b - a, d - a), axis=1))

    adapter = BoundaryAdapter(prepare, lambda core, src, tree, packed: Core(core.primitive_measure[tree.prim_ids]),
                              measure=lambda primitive: MeasureOps(uniform_parallelogram))
    source = QuadArrays(jnp.array([[0., 0., 0.], [2., 0., 0.], [3., 2., 0.], [1., 2., 0.]]),
                        jnp.array([[0, 1, 2, 3]], jnp.int32))
    index = bvh.build(PlanarQuads(), source)
    boundary = bvh.prepare_boundary(index, source, adapter=adapter, silhouettes=False)
    with pytest.raises(TypeError, match="classification"):
        classification.bind(index.spec, adapter=adapter)
    sample = boundary_sample.bind(index.spec, adapter=adapter)
    point, normal, pdf, face, status = jax.jit(sample)(
        prepare_boundary_sample(boundary), jnp.zeros(3), 10., jnp.array([0., .25, .75]))
    np.testing.assert_allclose(point, [1.25, 1.5, 0.])
    np.testing.assert_array_equal(normal, [0., 0., 1.])
    assert (float(pdf), int(face), int(status)) == (.25, 0, 1)


@wp.func_native("return external_quad::closest(query, item);")
def closest_native(query: wp.vec3, item: Quad) -> wp.vec3: ...


@dataclass(frozen=True)
class NativeQuads(PlanarQuads):
    native: NativeHeaders
    closest_euclidean = closest_native


@pytest.mark.parametrize("device_name", ["cpu", "cuda:0"])
def test_external_native_quads_and_helper_edit(tmp_path, device_name):
    if device_name == "cuda:0" and not wp.is_cuda_available():
        pytest.skip("CUDA device required")
    fixtures = Path(__file__).with_name("native")
    entry = tmp_path / "quad.h"
    helper = tmp_path / "quad_helper.h"
    shutil.copy(fixtures / entry.name, entry)
    shutil.copy(fixtures / helper.name, helper)
    primitive = NativeQuads(NativeHeaders((tmp_path,), (entry.name,), (helper,)))
    device = wp.get_device(device_name)
    jax_device = jax.devices("cuda" if device.is_cuda else "cpu")[0]
    with jax.default_device(jax_device):
        vertices = jnp.array([[0, 0, 0], [3, 0, 0], [2, 2, 0], [0, 2, 0]], dtype=jnp.float32)
        raw = QuadArrays(vertices, jnp.arange(4, dtype=jnp.int32).reshape(1, 4))
        index = bvh.build(primitive, raw, builder="lbvh")
    tree, data = index.data
    args = [device_payload(DeviceTree, tree, device), device_payload(QuadData, data, device)]
    nearest = bind_nearest(index.spec)
    knn = bind_knn(index.spec, 1, Euclidean(), return_points=True)
    assert bind_nearest(index.spec) is nearest
    closest, neighbors = nearest.device, knn.device

    @wp.kernel(module="unique", enable_backward=False)
    def caller(tree: DeviceTree, geometry: QuadData, out_nearest: wp.array(dtype=wp.vec3),
               out_scores: wp.array2d(dtype=float), out_points: wp.array2d(dtype=wp.vec3)):
        point, distance, index, status = wp.static(closest)(wp.vec3(1., 1., 2.), 10., tree, geometry)
        out_nearest[0] = point
        wp.static(neighbors)(wp.vec3(1., 1., 2.), 10., tree, geometry, out_scores, out_points, 0)

    def run_caller(queries):
        return run_kernel(caller, 1, args, [(1, wp.vec3), ((1, 1), float), ((1, 1), wp.vec3)],
                          queries=queries, device=device)

    def run_jax(nearest, knn):
        with jax.default_device(jax_device):
            point = jnp.array([[1., 1., 2.]], dtype=jnp.float32)
            found, distance, ids, status = jax.jit(nearest)(index.data, point)
            scores, neighbors, count, knn_status = jax.jit(knn)(index.data, point)
        np.testing.assert_array_equal(ids, [0])
        np.testing.assert_array_equal(status, [1])
        np.testing.assert_array_equal(count, [1])
        np.testing.assert_array_equal(knn_status, [1])
        np.testing.assert_allclose(distance, scores[:, 0])
        return found, neighbors

    initial_point = [[1., 1., 0.]]
    changed_point = [[1.25, 1., 0.]]
    first, _, first_knn = run_caller((nearest, knn))
    np.testing.assert_allclose(first, initial_point)
    np.testing.assert_allclose(first_knn[:, 0], initial_point)
    found, neighbors_jax = run_jax(nearest, knn)
    np.testing.assert_allclose(found, initial_point)
    np.testing.assert_allclose(neighbors_jax[:, 0], initial_point)
    caller_hash = caller.module.get_module_hash()
    nearest_hash = nearest.kernel.module.get_module_hash()
    helper.write_text(helper.read_text().replace("0.0f", "0.25f"))

    # Reconfigure the same caller and rebind generated queries with the edited helper.
    second, _, second_knn = run_caller((nearest, knn))
    assert caller.module.get_module_hash() != caller_hash
    np.testing.assert_allclose(second, changed_point)
    np.testing.assert_allclose(second_knn[:, 0], changed_point)
    updated_nearest = bind_nearest(index.spec)
    updated_knn = bind_knn(index.spec, 1, Euclidean(), return_points=True)
    assert updated_nearest.kernel.module.get_module_hash() != nearest_hash
    found, neighbors_jax = run_jax(updated_nearest, updated_knn)
    np.testing.assert_allclose(found, changed_point)
    np.testing.assert_allclose(neighbors_jax[:, 0], changed_point)

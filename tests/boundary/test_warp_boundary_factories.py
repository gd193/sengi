"""Indexed boundary features and metric-independent visibility."""
from sengi import bvh
from dataclasses import replace
from typing import Any

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from benchmarks.adapters.sengi import prepare, prepare_boundary  # noqa: E402
from benchmarks.workloads import builtin_geometry  # noqa: E402

from sengi.boundary.data import SilhouetteMeta, resolve_boundary_adapter
from sengi.primitives.segments import Segments
from sengi.primitives.triangles import Triangles
from sengi.queries.silhouette import bind as bind_silhouette  # noqa: E402
from sengi.bvh.device_tree import DeviceTree  # noqa: E402
from sengi.bvh import Config  # noqa: E402


@wp.func
def _invisible(viewpoint: wp.vec3, tree: DeviceTree, meta: SilhouetteMeta, node: int):
    return False


@wp.func
def _invisible_feature(query: wp.vec2, item: Any):
    return False


@pytest.mark.parametrize("reject", ["node", "feature"])
def test_boundary_custom_visibility_and_optional_metadata(reject):
    index, raw = prepare(builtin_geometry("segments2", 16))
    boundary = prepare_boundary(index, raw, {"closest_silhouette"})
    options = {"visibility": _invisible} if reject == "node" else {"visibility": None, "predicate": _invisible_feature}
    adapter = replace(resolve_boundary_adapter(index.spec.primitive), **options)
    result = bind_silhouette(index.spec, adapter=adapter)(boundary, np.array([3, 0], np.float32))
    assert int(result[2]) == -1 and int(result[3]) == 0
    with pytest.raises(ValueError, match="metadata"):
        bind_silhouette(index.spec)(replace(boundary, silhouette=None), np.zeros(2))


@pytest.mark.parametrize("dimension", [2, 3])
def test_boundary_feature_enclosure_and_leaf_order(dimension):
    primitive = Segments(2) if dimension == 2 else Triangles()
    vertices = np.array([[1e8, 0, 0], [-1, 0, 0], [1e8, 1, 0], [1e8, 0, 1]], np.float32)
    indices = np.array([[0, 1], [1, 2]], np.int32) if dimension == 2 else np.array([[0, 1, 2], [0, 3, 1]], np.int32)
    data = primitive.indexed(vertices[:, :dimension], indices)
    index = bvh.build(primitive, data, config=Config(leaf_size_threshold=1))
    tree, geometry = index.data
    boundary = bvh.prepare_boundary(index, data)
    assert boundary.core.indices is geometry.indices
    if dimension == 3:
        assert boundary.core.vertices is geometry.vertices
    np.testing.assert_array_equal(boundary.core.indices, indices[np.asarray(tree.prim_ids)])
    meta = boundary.silhouette
    for node in np.flatnonzero(np.asarray(tree.node_left) < 0):
        start, count = int(meta.node_feature_offset[node]), int(meta.node_feature_count[node])
        refs = np.asarray(meta.feature_refs[start:start+count])
        points = np.concatenate([np.asarray(meta.feature_a)[refs], np.asarray(meta.feature_b)[refs]])
        assert np.all(points >= np.asarray(tree.node_lower[node]))
        assert np.all(points <= np.asarray(tree.node_upper[node]))
    collapsed = (primitive.indexed([[0., 0], [0., 0]], [[0, 1]]) if dimension == 2 else
                 primitive.indexed([[1e8, 1e8, 0], [1, 1, 0], [0, 0, 0]], [[0, 1, 2]]))
    with np.errstate(divide="raise", invalid="raise"):
        with pytest.raises(ValueError, match="non-degenerate"):
            index = bvh.build(primitive, collapsed)
            bvh.prepare_boundary(index, collapsed)
        index = bvh.build(primitive, collapsed)
        invalid = bvh.prepare_boundary(index, collapsed, silhouettes=False)
    np.testing.assert_array_equal(invalid.core.primitive_measure, 0.)
    np.testing.assert_array_equal(invalid.core.closed, 0)
    assert np.isfinite(invalid.core.primitive_normal).all()
    if dimension == 2:
        source = primitive.indexed([[0., 0], [1e-30, 0]], [[0, 1]])
        index = bvh.build(primitive, source)
        tiny = bvh.prepare_boundary(index, source)
        np.testing.assert_allclose(tiny.core.primitive_measure, [1e-30], rtol=1e-6, atol=0)
        np.testing.assert_array_equal(tiny.core.primitive_normal, [[0., -1., 0.]])


def test_boundary_preparation_rejects_a_crossed_source_with_the_same_shape():
    primitive = Triangles()
    vertices = np.array([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]], np.float32)
    faces = [[0, 1, 2]]
    source = primitive.indexed(vertices, faces)
    index = bvh.build(primitive, source)

    for wrong in (primitive.indexed(vertices + 1, faces),
                  primitive.indexed(vertices, [[0, 2, 1]]),
                  primitive.indexed(vertices, faces * 2)):
        with pytest.raises(ValueError, match="source geometry does not match"):
            bvh.prepare_boundary(index, wrong)

    boundary = bvh.prepare_boundary(index, primitive.indexed(vertices.copy(), faces))
    assert boundary.data is index.data

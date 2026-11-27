"""SAH and Morton construction feed the same binary Warp traversal."""
from sengi import bvh
import numpy as np
import pytest

pytest.importorskip("warp")

from sengi.primitives.points import Points  # noqa: E402
from sengi.bvh import Config  # noqa: E402


def test_lbvh_duplicate_morton_codes_and_leaf_sizes():
    primitive = Points(2)
    # Repeated codes span more than one leaf and share the tree with other codes.
    vertices = np.repeat([[1., 2.], [-3., 2.], [10., 2.]], [11, 5, 7], axis=0)
    source = primitive.data(vertices)
    tree, _ = bvh.build(primitive, source, config=Config(leaf_size_threshold=2), builder="lbvh").data
    np.testing.assert_array_equal(np.sort(tree.prim_ids), np.arange(len(vertices)))
    assert int(tree.node_leaf_count.max()) == 2
    np.testing.assert_array_equal(tree.node_lower[0], [-3., 2., 0.])
    np.testing.assert_array_equal(tree.node_upper[0], [10., 2., 0.])


def test_lbvh_retains_float32_morton_quantization_with_overflow_fallback():
    primitive = Points(3)
    config = Config(leaf_size_threshold=1)
    source = primitive.data([[1e-29, 0., 0.], [-1e-29, 0., 0.], [0., 0., 0.]])
    tree, _ = bvh.build(primitive, source, config=config, builder="lbvh").data
    # Historically the minimum normalization span put these in the same bucket.
    np.testing.assert_array_equal(tree.prim_ids, [0, 1, 2])
    extreme = np.finfo(np.float32).max
    source = primitive.data([[extreme, 0., 0.], [-extreme, 0., 0.], [0., 0., 0.]])
    tree, _ = bvh.build(primitive, source, config=config, builder="lbvh").data
    np.testing.assert_array_equal(tree.prim_ids, [1, 2, 0])
    np.testing.assert_array_equal(tree.node_lower[0], [-extreme, 0., 0.])
    np.testing.assert_array_equal(tree.node_upper[0], [extreme, 0., 0.])


@pytest.mark.parametrize("builder", ["lbvh", "sah"])
def test_empty_and_singleton(builder):
    primitive = Points(2)
    for count in (0, 1):
        tree, _ = bvh.build(primitive, primitive.data(np.zeros((count, 2))), builder=builder).data
        assert tree.node_lower.shape == (count, 3)
        assert tree.packed_nodes.shape == (count, 2, 4)
        np.testing.assert_array_equal(tree.prim_ids, np.arange(count))
        np.testing.assert_array_equal(tree.node_leaf_count, np.ones(count))


def test_unknown_builder():
    primitive = Points(2)
    with pytest.raises(ValueError, match="builder"):
        bvh.build(primitive, primitive.data([[0., 0.]]), builder="unknown").data


@pytest.mark.parametrize("builder", ["lbvh", "sah"])
def test_packed_node_records(builder):
    primitive = Points(3)
    source = primitive.data(np.arange(30, dtype=np.float32).reshape(10, 3))
    tree, _ = bvh.build(primitive, source, config=Config(leaf_size_threshold=2), builder=builder).data
    records = np.asarray(tree.packed_nodes)
    np.testing.assert_array_equal(records[:, 0, :3], tree.node_lower)
    np.testing.assert_array_equal(records[:, 1, :3], tree.node_upper)

    words = records[:, :, 3].view(np.uint32)
    leaves = np.asarray(tree.node_leaf_count) > 0
    np.testing.assert_array_equal(words[~leaves, 0], tree.node_left[~leaves])
    np.testing.assert_array_equal(words[~leaves, 1], tree.node_right[~leaves])
    start = np.asarray(tree.node_leaf_offset)[leaves].astype(np.uint32)
    stop = start + np.asarray(tree.node_leaf_count)[leaves].astype(np.uint32)
    np.testing.assert_array_equal(words[leaves, 0], start | np.uint32(0x80000000))
    np.testing.assert_array_equal(words[leaves, 1], stop | np.uint32(0x80000000))

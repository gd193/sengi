"""Validated-tree postorder reductions."""
from __future__ import annotations

import numpy as np

from sengi.bvh.tree import _validate_tree


def reduce_nodes(primitive, left, right, offset, count, postorder, combine):
    """Reduce leaf-order values on an already validated binary tree.

    Each leaf slice is reduced before its children-first parent. Internal
    reductions receive left then right, preserving float32 summation order.
    Trailing primitive dimensions and empty-tree output shape are retained.
    """
    result = np.empty((len(left),) + primitive.shape[1:], np.float32)
    for node in postorder:
        if count[node]:
            result[node] = combine(primitive[offset[node]:offset[node] + count[node]])
        else:
            result[node] = combine(result[[left[node], right[node]]])
    return result


def _reduce_nodes(tree, primitive, combine):
    if not tree.node_left.shape[0]:
        return np.empty((0,)+primitive.shape[1:], np.float32)
    arrays, _, order = _validate_tree(tree, len(primitive))
    left, right, offset, count = (arrays[name] for name in
        ("node_left", "node_right", "node_leaf_offset", "node_leaf_count"))
    return reduce_nodes(primitive, left, right, offset, count, order, combine)

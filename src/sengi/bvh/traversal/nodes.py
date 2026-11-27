"""Immutable host bundle for indexed or packed node access."""
from dataclasses import dataclass
from functools import lru_cache

import warp as wp

from sengi.bvh.device_tree import DeviceTree
from sengi.bvh.traversal.abi import node_is_leaf, node_lower, node_upper, pack_node, load_node_bound
from sengi.warp.arrays import load_array


@dataclass(frozen=True)
class NodeFunctions:
    read: wp.Function
    child_indices: wp.Function
    is_leaf: wp.Function
    leaf_range: wp.Function
    metadata: wp.Function


@lru_cache(maxsize=None)
def node_functions(packed_nodes: bool) -> NodeFunctions:
    if packed_nodes:
        @wp.func
        def read(tree: DeviceTree, node: int):
            lo = load_node_bound(tree.packed_nodes, node, 0)
            hi = load_node_bound(tree.packed_nodes, node, 1)
            return wp.vec3(lo[0], lo[1], lo[2]), wp.vec3(hi[0], hi[1], hi[2]), pack_node(lo, hi)

        @wp.func
        def child_indices(tree: DeviceTree, node: int, current: wp.uint64):
            return node_lower(current), node_upper(current)

        @wp.func
        def is_leaf(tree: DeviceTree, node: int, current: wp.uint64):
            return node_is_leaf(current)

        @wp.func
        def leaf_range(tree: DeviceTree, node: int, current: wp.uint64):
            return node_lower(current), node_upper(current)

        @wp.func
        def metadata(tree: DeviceTree, node: int):
            return pack_node(tree.packed_nodes[node, 0], tree.packed_nodes[node, 1])
    else:
        @wp.func
        def read(tree: DeviceTree, node: int):
            return load_array(tree.node_lower, node), load_array(tree.node_upper, node), wp.uint64(0)

        @wp.func
        def child_indices(tree: DeviceTree, node: int, current: wp.uint64):
            return load_array(tree.node_left, node), load_array(tree.node_right, node)

        @wp.func
        def is_leaf(tree: DeviceTree, node: int, current: wp.uint64):
            return load_array(tree.node_left, node) < 0

        @wp.func
        def leaf_range(tree: DeviceTree, node: int, current: wp.uint64):
            begin = load_array(tree.node_leaf_offset, node)
            return begin, begin + load_array(tree.node_leaf_count, node)

        @wp.func
        def metadata(tree: DeviceTree, node: int):
            return wp.uint64(0)

    return NodeFunctions(read, child_indices, is_leaf, leaf_range, metadata)

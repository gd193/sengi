"""BVH payload — flat SoA arrays describing a binary AABB tree.

The host builders return this JAX-array pytree. Warp construction packs
it into its query payload without changing the host tree's ordering.

Layout (N_nodes total, N_leaves <= N_nodes, N_prim_refs >= n_primitives):
  node_lower:        (N_nodes, 3)   per-node AABB lower bound
  node_upper:        (N_nodes, 3)   per-node AABB upper bound
  node_left:         (N_nodes,)     left child index, -1 if leaf
  node_right:        (N_nodes,)     right child index, -1 if leaf
  node_leaf_offset:  (N_nodes,)     start in `prim_ids`, valid if leaf
  node_leaf_count:   (N_nodes,)     count in `prim_ids`, 0 if internal
  prim_ids:          (N_prim_refs,) primitive indices in leaf order
  root:              ()             root node index (typically 0)

Warp query construction stores packed tree and geometry in `IndexData`.
"""
from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class BVHPayload(NamedTuple):
    node_lower: jnp.ndarray
    node_upper: jnp.ndarray
    node_left: jnp.ndarray
    node_right: jnp.ndarray
    node_leaf_offset: jnp.ndarray
    node_leaf_count: jnp.ndarray
    prim_ids: jnp.ndarray
    root: jnp.ndarray


_PACKED_PAYLOAD_LIMIT = 1 << 31


def _int32_array(name, value):
    raw = np.asarray(value)
    if not np.issubdtype(raw.dtype, np.integer):
        raise ValueError(f"tree {name} must contain integers")
    if raw.size and (np.any(raw < np.iinfo(np.int32).min)
                     or np.any(raw > np.iinfo(np.int32).max)):
        raise ValueError(f"tree {name} exceeds int32 indexing")
    return raw.astype(np.int32, copy=False)


def _validate_tree(tree, n_refs):
    """Return index arrays, depths and postorder; root must be 0."""
    arrays = {
        name: _int32_array(name, getattr(tree, name))
        for name in ("node_left", "node_right", "node_leaf_offset", "node_leaf_count")
    }
    n_nodes = arrays["node_left"].shape[0] if arrays["node_left"].ndim == 1 else -1
    if n_nodes < 1 or any(value.shape != (n_nodes,) for value in arrays.values()):
        raise ValueError("tree index and leaf arrays must have matching shape (N,)")
    root_array = _int32_array("root", getattr(tree, "root", 0))
    if root_array.size != 1 or int(root_array.reshape(())) != 0:
        raise ValueError("Warp BVH tree root must be node 0")
    if n_nodes >= _PACKED_PAYLOAD_LIMIT:
        raise ValueError("Warp BVH node indices exceed the 31-bit packed payload")

    left = arrays["node_left"]
    right = arrays["node_right"]
    offset = arrays["node_leaf_offset"]
    count = arrays["node_leaf_count"]
    refs_covered = np.zeros((n_refs,), dtype=bool)
    for node in range(n_nodes):
        if count[node] < 0:
            raise ValueError("tree leaf counts must be nonnegative")
        if count[node] > 0:
            begin = int(offset[node])
            end = begin + int(count[node])
            if begin < 0 or end > n_refs or end >= _PACKED_PAYLOAD_LIMIT:
                raise ValueError("tree leaf range is outside packed primitive references")
            if left[node] >= 0 or right[node] >= 0:
                raise ValueError("tree leaf nodes cannot have child indices")
            if np.any(refs_covered[begin:end]):
                raise ValueError("tree leaf primitive ranges overlap")
            refs_covered[begin:end] = True
        else:
            if offset[node] < 0:
                raise ValueError("tree leaf offsets must be nonnegative")
            if left[node] < 0 or right[node] < 0:
                raise ValueError("internal tree nodes must have two children")
            if left[node] == right[node]:
                raise ValueError("internal tree children must be distinct")
            if left[node] >= n_nodes or right[node] >= n_nodes:
                raise ValueError("tree child index is outside node arrays")
    if n_refs and not np.all(refs_covered):
        raise ValueError("tree leaf ranges do not cover packed primitive references")

    # Validate topology before accumulating areas.
    state = np.zeros((n_nodes,), dtype=np.uint8)
    depths = np.zeros((n_nodes,), dtype=np.int32)
    postorder = []
    stack = [(0, False, 0)]
    while stack:
        node, exiting, depth = stack.pop()
        if exiting:
            state[node] = 2
            postorder.append(node)
            continue
        if state[node] != 0:
            raise ValueError("tree topology contains a cycle or shared child")
        state[node] = 1
        depths[node] = depth
        stack.append((node, True, depth))
        if count[node] == 0:
            stack.append((int(right[node]), False, depth + 1))
            stack.append((int(left[node]), False, depth + 1))
    if np.any(state != 2):
        raise ValueError("tree contains nodes unreachable from root 0")
    return arrays, depths, postorder


def _pack_ray_nodes(node_lower, node_upper, left, right, offset, count):
    """float4 xyz=bounds; w holds uint32 bits, never numeric floats.
    Bit 31 marks leaves; low bits hold left/begin or right/end (exclusive)."""
    end = offset.astype(np.int64) + count.astype(np.int64)
    ray_lower = np.zeros((len(left), 4), dtype=np.float32)
    ray_upper = np.zeros_like(ray_lower)
    ray_lower[:, :3] = node_lower
    ray_upper[:, :3] = node_upper
    lower_bits = left.astype(np.uint32, copy=True)
    upper_bits = right.astype(np.uint32, copy=True)
    leaves = count > 0
    lower_bits[leaves] = offset[leaves].astype(np.uint32) | np.uint32(0x80000000)
    upper_bits[leaves] = end[leaves].astype(np.uint32) | np.uint32(0x80000000)
    ray_lower[:, 3] = lower_bits.view(np.float32)
    ray_upper[:, 3] = upper_bits.view(np.float32)
    return ray_lower, ray_upper

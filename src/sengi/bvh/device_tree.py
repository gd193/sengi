"""Primitive-independent BVH arrays."""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import warp as wp

from sengi.bvh.config import Config
from sengi.bvh.index import IndexSpec
from sengi.bvh import lbvh, sah
from sengi.bvh.tree import MAX_TREE_DEPTH, _int32_array, _pack_nodes, _validate_tree
from sengi.bvh.bounds import _validated_bounds
from sengi.warp.contracts import warp_struct


STACK_CAPACITY = MAX_TREE_DEPTH


@warp_struct
class DeviceTree:
    node_lower: wp.array(dtype=wp.vec3)
    node_upper: wp.array(dtype=wp.vec3)
    node_left: wp.array(dtype=wp.int32)
    node_right: wp.array(dtype=wp.int32)
    node_leaf_offset: wp.array(dtype=wp.int32)
    node_leaf_count: wp.array(dtype=wp.int32)
    prim_ids: wp.array(dtype=wp.int32)
    packed_nodes: wp.array2d(dtype=wp.vec4)


@dataclass(frozen=True)
class _IndexSchema:
    """Static primitive and host geometry schema."""

    spec: IndexSpec[Any]
    geometry_type: type


@partial(jax.tree_util.register_dataclass,
         data_fields=tuple(DeviceTree.vars), meta_fields=("_schema",))
@dataclass(frozen=True, eq=False)
class TreeData:
    node_lower: jax.Array
    node_upper: jax.Array
    node_left: jax.Array
    node_right: jax.Array
    node_leaf_offset: jax.Array
    node_leaf_count: jax.Array
    prim_ids: jax.Array
    packed_nodes: jax.Array
    _schema: _IndexSchema

    def validate_geometry(self, geometry: object, primitive: object) -> None:
        """Reject geometry incompatible with a query's static specialization."""
        schema = self._schema
        if schema.spec.primitive != primitive or type(geometry) is not schema.geometry_type:
            raise TypeError("index data is incompatible with the bound primitive or geometry schema")


def build_device_tree(primitive, data, config=None, *, builder="sah"):
    """Build a binary SAH or Morton LBVH and pack geometry in leaf order."""
    if builder == "sah":
        build = partial(sah._build_payload, dimension=primitive.dimension)
    elif builder == "lbvh":
        build = lbvh._build_payload
    else:
        raise ValueError("builder must be 'sah' or 'lbvh'")
    config = Config() if config is None else config
    if primitive.dimension not in (2, 3):
        raise ValueError("BVH bounds support 2D or 3D primitives")
    lo, hi = _validated_bounds(*primitive.bounds(data), dimension=primitive.dimension)
    if not len(lo):
        empty = jnp.empty((0,), jnp.int32)
        bounds = jnp.empty((0, 3), jnp.float32)
        packed = jnp.empty((0, 2, 4), jnp.float32)
        geometry = primitive.pack(data, empty)
        tree = TreeData(bounds, bounds, empty, empty, empty, empty, empty, packed,
                        _IndexSchema(IndexSpec(primitive), type(geometry)))
        return tree, geometry
    if len(lo) >= 2**31:
        raise ValueError("primitive IDs must fit int32")
    if primitive.dimension == 2:
        lo, hi = (np.pad(v, ((0, 0), (0, 1))) for v in (lo, hi))
    built = build(lo, hi, min(config.leaf_size_threshold, len(lo)))
    arrays, _ = _validate_tree(built, len(lo))

    ids = _int32_array("prim_ids", built.prim_ids)
    if (ids.shape != (len(lo),) or np.any(ids < 0) or np.any(ids >= len(lo))
            or np.unique(ids).size != len(lo)):
        raise ValueError("tree primitive IDs must be a permutation of the input IDs")
    node_lo, node_hi = np.asarray(built.node_lower), np.asarray(built.node_upper)
    count = arrays["node_leaf_count"]
    offset = arrays["node_leaf_offset"]
    left, right = arrays["node_left"], arrays["node_right"]
    if (node_lo.shape != (len(count), 3) or node_hi.shape != node_lo.shape
            or node_lo.dtype != np.float32 or node_hi.dtype != np.float32
            or not np.isfinite(node_lo).all() or not np.isfinite(node_hi).all()
            or np.any(node_lo > node_hi)):
        raise ValueError("tree bounds must be finite, ordered float32 arrays of shape (nodes, 3)")

    leaves = np.flatnonzero(count)
    leaf_order = leaves[np.argsort(offset[leaves])]
    slot_leaves = np.repeat(leaf_order, count[leaf_order])
    if np.any(node_lo[slot_leaves] > lo[ids]) or np.any(node_hi[slot_leaves] < hi[ids]):
        raise ValueError("tree leaf bounds must enclose their primitives")
    internal = count == 0
    for child in (left[internal], right[internal]):
        if np.any(node_lo[internal] > node_lo[child]) or np.any(node_hi[internal] < node_hi[child]):
            raise ValueError("tree internal bounds must enclose their children")

    return pack_device_tree(primitive, data, node_lo, node_hi, left, right, offset, count, ids)


def pack_device_tree(primitive, data, node_lower, node_upper, node_left, node_right,
                     node_leaf_offset, node_leaf_count, prim_ids):
    """Upload a validated tree and pack its source geometry in leaf order."""
    packed = _pack_nodes(node_lower, node_upper, node_left, node_right,
                             node_leaf_offset, node_leaf_count)
    device_ids = jnp.asarray(prim_ids)
    geometry = primitive.pack(data, device_ids)
    tree = TreeData(*(jnp.asarray(a) for a in (node_lower, node_upper, node_left, node_right,
                                               node_leaf_offset, node_leaf_count, device_ids, packed)),
                    _schema=_IndexSchema(IndexSpec(primitive), type(geometry)))
    return tree, geometry

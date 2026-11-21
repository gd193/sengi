"""Boundary sampling metadata and measure preparation."""
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

import jax
from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
import warp as wp

from sengi.warp.native import NativeHeaders
from sengi.bvh.index import IndexData
from sengi.boundary.metadata import _reduce_nodes
from sengi.warp.contracts import warp_struct

ItemT = TypeVar("ItemT")
P = TypeVar("P")
H = TypeVar("H")
CH = TypeVar("CH")


@warp_struct
class SamplingMeta:
    primitive_weight: wp.array(dtype=wp.float32)
    node_weight: wp.array(dtype=wp.float32)


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class SamplingData(Generic[H]):
    data: IndexData[H]
    primitive_weight: ArrayLike
    node_weight: ArrayLike


@dataclass(frozen=True)
class MeasureOps(Generic[ItemT, P, CH]):
    """sample(item, uniforms2) returns point, normal and conditional density."""
    sample: Any
    weights: Any = None
    native: NativeHeaders | None = None


def prepare_boundary_sample(boundary, measure=None):
    """Use the same static measure policy when preparing and querying."""
    tree = boundary.data.tree
    core = boundary.core
    if measure is not None and measure.weights is not None:
        raw = measure.weights(core)
        if np.iscomplexobj(raw):
            raise ValueError("sampling weights must be real")
        raw = np.asarray(raw, np.float64)
        if (raw.shape != tree.prim_ids.shape or not np.isfinite(raw).all() or np.any(raw < 0)
                or np.any(raw > np.finfo(np.float32).max)):
            raise ValueError("sampling weights must have one finite float32 value per leaf primitive")
        primitive_weight = jnp.asarray(raw, jnp.float32, device=tree.node_lower.sharding)
    else:
        primitive_weight = core.primitive_measure
    values = np.asarray(primitive_weight)
    if values.shape != tree.prim_ids.shape or not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError("primitive measure must be finite, nonnegative and match tree slots")
    nodes = _reduce_nodes(tree, values, lambda a: a.sum(axis=0, dtype=np.float32))
    if not np.isfinite(nodes).all():
        raise ValueError("subtree sampling weight exceeds float32 range")
    return SamplingData(boundary.data, primitive_weight,
                        jnp.asarray(nodes, device=tree.node_lower.sharding))

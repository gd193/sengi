"""Boundary sampling metadata and measure preparation."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, NamedTuple, TypeVar

from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
import warp as wp

from sengi.boundary.metadata import _reduce_nodes
from sengi.warp.contracts import warp_struct

ItemT = TypeVar("ItemT")
P = TypeVar("P")
CH = TypeVar("CH")


@warp_struct
class SamplingMeta:
    primitive_weight: wp.array(dtype=wp.float32)
    node_weight: wp.array(dtype=wp.float32)


class SamplingData(NamedTuple):
    primitive_weight: ArrayLike
    node_weight: ArrayLike


@dataclass(frozen=True)
class MeasureOps(Generic[ItemT, P, CH]):
    """sample(item, uniforms2) returns point, normal and conditional density."""
    sample: Any
    weights: Any = None


def prepare_boundary_sample(tree, boundary, measure=None):
    """Use the same static measure policy when preparing and querying."""
    core = boundary.core if hasattr(boundary, "core") else boundary
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
    return SamplingData(primitive_weight, jnp.asarray(nodes, device=tree.node_lower.sharding))


@lru_cache(maxsize=None)
def _uniform_measure(primitive):
    Vec, Item = primitive.point_type, Any
    dimension = primitive.dimension
    if dimension == 2:
        @wp.func
        def sample(item: Item, uniforms: wp.vec2):
            edge = item.b-item.a
            length = wp.length(edge)
            normal = Vec(edge[1], -edge[0])/length
            return item.a+uniforms[0]*edge, normal, 1.0/length
    else:
        @wp.func
        def sample(item: Item, uniforms: wp.vec2):
            a, b, c = wp.vec3d(item.a), wp.vec3d(item.b), wp.vec3d(item.c)
            ab, bc, ca = b - a, c - b, a - c
            normal = wp.cross(ab, bc)
            ca2 = wp.dot(ca, ca)
            if ca2 < wp.dot(ab, ab) and ca2 < wp.dot(bc, bc):
                normal = wp.cross(bc, ca)
            twice_area = wp.length(normal)
            root, fraction = wp.sqrt(wp.float64(uniforms[0])), wp.float64(uniforms[1])
            one = wp.float64(1.0)
            point = (one - root) * a + root * ((one - fraction) * b + fraction * c)
            return Vec(point), Vec(normal / twice_area), wp.float32(wp.float64(2.0) / twice_area)
    return MeasureOps(sample)

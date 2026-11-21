"""Robin boundary coefficients and metadata preparation."""
from typing import Generic, NamedTuple, TypeVar

from jax.typing import ArrayLike
import jax.numpy as jnp
import numpy as np
import warp as wp

from sengi.bvh.index import IndexData
from sengi.boundary.metadata import _reduce_nodes
from sengi.warp.contracts import warp_struct

H = TypeVar("H")
CH = TypeVar("CH")
FH = TypeVar("FH")


@warp_struct
class CoefficientMeta:
    primitive_bounds: wp.array(dtype=wp.vec2)
    node_bounds: wp.array(dtype=wp.vec2)


class CoefficientData(NamedTuple):
    primitive_bounds: ArrayLike
    node_bounds: ArrayLike


class RobinData(NamedTuple, Generic[H, CH, FH]):
    data: IndexData[H]
    core: CH
    silhouette: FH
    coefficients: CoefficientData


def prepare_robin_coefficients(tree, minimum, maximum=None):
    values = []
    for raw in (minimum, minimum if maximum is None else maximum):
        if np.iscomplexobj(raw):
            raise ValueError("Robin coefficients must be real")
        raw = np.asarray(raw, np.float64)
        if np.isnan(raw).any() or np.any(raw < 0) or np.any(np.isfinite(raw) & (raw > np.finfo(np.float32).max)):
            raise ValueError("Robin coefficients must be nonnegative float32 or infinity")
        values.append(np.broadcast_to(raw, tree.prim_ids.shape))
    lower, upper = values
    if np.any(lower > upper):
        raise ValueError("Robin coefficient minimum exceeds maximum")
    lo, hi = lower.astype(np.float32), upper.astype(np.float32)
    lo = np.where(lo.astype(np.float64) > lower, np.nextafter(lo, np.float32(-np.inf)), lo)
    hi = np.where(hi.astype(np.float64) < upper, np.nextafter(hi, np.float32(np.inf)), hi)
    primitive = np.stack((lo, hi), axis=-1)[np.asarray(tree.prim_ids)]
    nodes = _reduce_nodes(tree, primitive, lambda a: np.array([a[:, 0].min(), a[:, 1].max()], np.float32))
    return CoefficientData(*(jnp.asarray(a, device=tree.node_lower.sharding) for a in (primitive, nodes)))


def prepare_robin(boundary, minimum, maximum=None):
    if boundary.silhouette is None:
        raise ValueError("Robin radius requires silhouette metadata")
    return RobinData(boundary.data, boundary.core, boundary.silhouette,
                     prepare_robin_coefficients(boundary.data.tree, minimum, maximum))

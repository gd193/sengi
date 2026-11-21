"""Star radius from the nearest visible boundary feature."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Generic, TypeVar

import jax.numpy as jnp
import warp as wp

from sengi.bvh.device_tree import DeviceTree, STACK_CAPACITY
from sengi.queries.silhouette import bind as bind_silhouette


H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
CH = TypeVar("CH")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")
FI = TypeVar("FI")


@dataclass(frozen=True)
class StarRadiusQuery(Generic[H, D, ItemT, P, CH, CD, FH, FD, FI]):
    silhouette: object
    device: object

    @property
    def _device_kernel(self):
        return self.silhouette.kernel

    def __call__(self, data, boundary, points, cap):
        _, distance, _, status = self.silhouette(data, boundary, points, cap)
        cap = jnp.asarray(cap, jnp.float32)
        valid = (status >= 0) & (cap >= 0.0) & ~jnp.isnan(cap)
        return (jnp.where(valid, jnp.minimum(cap, distance), jnp.nan),
                jnp.where(valid, 1, -1).astype(jnp.int32))


def bind(spec, *, adapter=None, stack_capacity=STACK_CAPACITY, block_dim=64):
    return _make_star_radius(bind_silhouette(spec, adapter=adapter,
        stack_capacity=stack_capacity, block_dim=block_dim))


@lru_cache(maxsize=None)
def _make_star_radius(silhouette):
    closest = silhouette.device
    Point = silhouette.query.ops.query_type
    Meta = silhouette.query.primitive.data_type

    @wp.func
    def device(point: Point, cap: float, tree: DeviceTree, meta: Meta):
        closest_point, distance, index, status = wp.static(closest)(point, cap, tree, meta)
        if status < 0 or not (cap >= 0.0):
            return wp.nan, -1
        return wp.min(cap, distance), 1

    return StarRadiusQuery(silhouette, device)

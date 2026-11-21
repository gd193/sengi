"""Primitive-AABB overlap signatures; box shapes are checked at runtime."""
from typing import Generic, TypeVar

import jax
from jax.typing import ArrayLike

from sengi.warp.launch import BoundQuery
from sengi.warp.contracts import PrimitiveSpec, WarpFunction
from sengi.bvh.index import IndexData, IndexSpec

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")


class OverlapQuery(BoundQuery, Generic[HostDataT]):
    device: WarpFunction[..., tuple[int, int]]
    def __call__(self, data: IndexData[HostDataT], lower: ArrayLike,
                 upper: ArrayLike) -> tuple[jax.Array, jax.Array]: ...


def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]], *,
                 stack_capacity: int = ..., block_dim: int = ...
                 ) -> OverlapQuery[HostDataT]: ...

"""Typed exact closed-box query signatures."""
from dataclasses import dataclass
from typing import Generic, Literal, TypeVar

import jax
import warp as wp
from jax.typing import ArrayLike

from sengi.warp.native import NativeHeaders
from sengi.warp.contracts import PrimitiveSpec, WarpFunction
from sengi.warp.launch import BoundQuery
from sengi.bvh.device_tree import DeviceTree
from sengi.bvh.index import IndexData, IndexSpec

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")

@dataclass(frozen=True)
class RegionIntersectionPolicy(Generic[DeviceDataT]):
    intersects: WarpFunction[[DeviceDataT, int, wp.vec3, wp.vec3], bool]
    native: NativeHeaders | None = None

@dataclass(frozen=True)
class RegionQuery(BoundQuery, Generic[HostDataT, DeviceDataT, PointT]):
    mode: str
    device: WarpFunction[[PointT, PointT, DeviceTree, DeviceDataT], tuple[int, int]]
    def __call__(self, data: IndexData[HostDataT], lower: ArrayLike,
                 upper: ArrayLike) -> tuple[jax.Array, jax.Array]: ...


def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                policy: RegionIntersectionPolicy[DeviceDataT] | None = ..., *, mode: Literal["any", "count"] = ...,
                count_limit: int | None = ..., stack_capacity: int = ...,
                block_dim: int = ...) -> RegionQuery[HostDataT, DeviceDataT, PointT]: ...

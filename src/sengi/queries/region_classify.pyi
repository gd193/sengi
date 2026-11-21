"""Typed exact closed-box query signatures."""
from dataclasses import dataclass
from typing import Any, Generic, TypeVar, overload

import jax
from jax.typing import ArrayLike

from sengi.warp.contracts import PrimitiveSpec, WarpFunction
from sengi.warp.launch import BoundQuery
from sengi.queries.domain import ClassificationOps
from sengi.bvh.device_tree import DeviceTree
from sengi.boundary.data import BoundaryData
from sengi.bvh.index import IndexSpec

from sengi.queries.region import RegionIntersectionPolicy

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
MetaT = TypeVar("MetaT")
HostMetaT = TypeVar("HostMetaT")


@dataclass(frozen=True)
class RegionClassifyQuery(BoundQuery, Generic[HostDataT, DeviceDataT, PointT, HostMetaT, MetaT]):
    device: WarpFunction[[PointT, PointT, DeviceTree, DeviceDataT, MetaT], tuple[int, int]]
    def __call__(self, boundary: BoundaryData[HostDataT, HostMetaT, Any],
                 lower: ArrayLike, upper: ArrayLike) -> tuple[jax.Array, jax.Array]: ...


@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                         region_policy: RegionIntersectionPolicy[DeviceDataT],
                         classify_policy: ClassificationOps[PointT, DeviceDataT, MetaT], *,
                         metadata_host_type: type[HostMetaT],
                         stack_capacity: int = ..., block_dim: int = ...
                         ) -> RegionClassifyQuery[HostDataT, DeviceDataT, PointT, HostMetaT, MetaT]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                         region_policy: RegionIntersectionPolicy[DeviceDataT],
                         classify_policy: ClassificationOps[PointT, DeviceDataT, MetaT], *,
                         metadata_host_type: None = ...,
                         stack_capacity: int = ..., block_dim: int = ...
                         ) -> RegionClassifyQuery[HostDataT, DeviceDataT, PointT, MetaT, MetaT]: ...

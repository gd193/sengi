"""Structural primitive specifications."""
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, TypeVar

import jax
import numpy as np
import warp as wp

from sengi.queries.domain import Ray
from sengi.warp.contracts import FusedNearestOps, PrimitiveSpec, WarpFunction, WarpType

if TYPE_CHECKING:
    from sengi.boundary.data import BoundaryAdapter


HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")


@dataclass(frozen=True)
class Primitive(Generic[HostDataT, DeviceDataT, ItemT, PointT]):
    """Immutable structural primitive with explicit host and device operations.

    Dynamic Warp struct and host payload classes remain runtime objects; the
    four type parameters connect extensions' payloads and device callbacks.
    """
    dimension: int
    point_type: WarpType[PointT]
    data_type: WarpType[DeviceDataT]
    load: WarpFunction[[DeviceDataT, int], ItemT]
    bounds_of_item: WarpFunction[[ItemT], tuple[wp.vec3, wp.vec3]]
    host_bounds: Callable[[HostDataT], tuple[np.ndarray, np.ndarray]]
    host_pack: Callable[[HostDataT, np.ndarray | jax.Array], HostDataT]
    item_type: WarpType[ItemT] | None = None
    closest_euclidean: WarpFunction[[PointT, ItemT], PointT] | None = None
    intersect: WarpFunction[[Ray, float, float, DeviceDataT, int, bool, bool], wp.vec2] | None = None
    segment_endpoints: WarpFunction[[ItemT], tuple[PointT, PointT]] | None = None
    normal_sides: WarpFunction[[ItemT], tuple[PointT, wp.vec3, wp.vec3]] | None = None
    point_bounds: bool = False
    name: str = "primitive"
    payload_type: Any = None
    prepare_data: Callable[..., HostDataT] | None = None
    prepare_indexed: Callable[[Any, Any], HostDataT] | None = None
    boundary_adapter: "BoundaryAdapter[HostDataT, DeviceDataT, ItemT, PointT, Any, Any, Any, Any, Any] | None" = None

    score_euclidean_at: WarpFunction[[PointT, DeviceDataT, int], wp.float64] | None = None
    closest_euclidean_at: WarpFunction[[PointT, DeviceDataT, int], PointT] | None = None
    group_leaves: bool = False
    cache_nearest_bounds: bool = True
    bind_fused_euclidean: Callable[
        [PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
        FusedNearestOps[PointT, DeviceDataT, ItemT, PointT] | None,
    ] | None = None

    def data(self, *arrays: Any) -> HostDataT:
        if self.prepare_data is None:
            raise TypeError("primitive has no array data constructor")
        return self.prepare_data(*arrays)

    def indexed(self, vertices: Any, indices: Any) -> HostDataT:
        if self.prepare_indexed is None:
            raise TypeError("primitive has no indexed data constructor")
        return self.prepare_indexed(vertices, indices)

    def bounds(self, data: HostDataT) -> tuple[np.ndarray, np.ndarray]:
        return self.host_bounds(data)

    def pack(self, data: HostDataT, ids: np.ndarray | jax.Array) -> HostDataT:
        return self.host_pack(data, ids)

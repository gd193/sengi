"""Static query specialization and dynamic data for a binary BVH."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Literal, NamedTuple, TypeVar, cast, overload

from .config import Config

if TYPE_CHECKING:
    from sengi.boundary.data import BoundaryAdapter, BoundaryData, BoundaryPrimitive
    from sengi.bvh.device_tree import TreeData
    from sengi.warp.contracts import PrimitiveSpec

PrimitiveT = TypeVar("PrimitiveT", covariant=True)
GeometryT = TypeVar("GeometryT")
DeviceT = TypeVar("DeviceT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
CoreHostT = TypeVar("CoreHostT")
CoreDeviceT = TypeVar("CoreDeviceT")
FeatureHostT = TypeVar("FeatureHostT")
FeatureDeviceT = TypeVar("FeatureDeviceT")
FeatureItemT = TypeVar("FeatureItemT")


@dataclass(frozen=True)
class BinaryLayout:
    """Root-zero binary hierarchy with geometry packed in leaf order."""


@dataclass(frozen=True)
class IndexSpec(Generic[PrimitiveT]):
    """Static primitive and tree layout used to specialize a query."""

    primitive: PrimitiveT
    layout: BinaryLayout = BinaryLayout()

    def __post_init__(self):
        if type(self.layout) is not BinaryLayout:
            raise ValueError("BVH queries require BinaryLayout")


class IndexData(NamedTuple, Generic[GeometryT]):
    """Array pytree passed into a bound query, including under JAX transforms."""

    tree: TreeData
    geometry: GeometryT


@dataclass(frozen=True)
class Index(Generic[PrimitiveT, GeometryT]):
    spec: IndexSpec[PrimitiveT]
    data: IndexData[GeometryT]


@overload
def build(
    primitive: BoundaryPrimitive[GeometryT, DeviceT, ItemT, PointT,
                                 CoreHostT, CoreDeviceT, FeatureHostT, FeatureDeviceT, FeatureItemT],
    source: GeometryT,
    *,
    builder: Literal["sah", "lbvh"] = "sah",
    config: Config | None = None,
) -> Index[BoundaryPrimitive[GeometryT, DeviceT, ItemT, PointT,
                                 CoreHostT, CoreDeviceT, FeatureHostT, FeatureDeviceT, FeatureItemT], GeometryT]: ...


@overload
def build(
    primitive: PrimitiveSpec[GeometryT, DeviceT, ItemT, PointT],
    source: GeometryT,
    *,
    builder: Literal["sah", "lbvh"] = "sah",
    config: Config | None = None,
) -> Index[PrimitiveSpec[GeometryT, DeviceT, ItemT, PointT], GeometryT]: ...


def build(
    primitive: PrimitiveSpec[GeometryT, DeviceT, ItemT, PointT],
    source: GeometryT,
    *,
    builder: Literal["sah", "lbvh"] = "sah",
    config: Config | None = None,
) -> Index[PrimitiveSpec[GeometryT, DeviceT, ItemT, PointT], GeometryT]:
    """Build an index over a primitive's source geometry."""
    from sengi.bvh.device_tree import build_device_tree

    tree, geometry = build_device_tree(primitive, source, config, builder=builder)
    return Index(IndexSpec(primitive), IndexData(tree, geometry))


@overload
def prepare_boundary(
    index: Index[PrimitiveSpec[GeometryT, DeviceT, ItemT, PointT], GeometryT],
    source: GeometryT,
    *,
    adapter: BoundaryAdapter[GeometryT, DeviceT, ItemT, PointT,
                             CoreHostT, CoreDeviceT, FeatureHostT, FeatureDeviceT, FeatureItemT],
    silhouettes: bool = True,
) -> BoundaryData[CoreHostT, FeatureHostT]: ...


@overload
def prepare_boundary(
    index: Index[BoundaryPrimitive[GeometryT, DeviceT, ItemT, PointT,
                                 CoreHostT, CoreDeviceT, FeatureHostT, FeatureDeviceT, FeatureItemT], GeometryT],
    source: GeometryT,
    *,
    adapter: None = None,
    silhouettes: bool = True,
) -> BoundaryData[CoreHostT, FeatureHostT]: ...


def prepare_boundary(
    index: Index[PrimitiveSpec[Any, Any, Any, Any], Any],
    source: Any,
    *,
    adapter: BoundaryAdapter[Any, Any, Any, Any, Any, Any, Any, Any, Any] | None = None,
    silhouettes: bool = True,
) -> BoundaryData[Any, Any]:
    """Prepare metadata for a BVH using the same source used to build it."""
    from sengi.boundary.data import prepare_boundary as prepare

    primitive = index.spec.primitive
    tree, geometry = index.data
    if adapter is not None:
        return prepare(primitive, source, tree, geometry, adapter=adapter, silhouettes=silhouettes)
    boundary_primitive = cast("BoundaryPrimitive[Any, Any, Any, Any, Any, Any, Any, Any, Any]", primitive)
    return prepare(boundary_primitive, source, tree, geometry, silhouettes=silhouettes)

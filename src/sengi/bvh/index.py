"""Static query specialization and dynamic data for a binary BVH."""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Generic, Literal, NamedTuple, TypeVar, cast, overload

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

    @property
    def spec(self) -> IndexSpec[Any]:
        """Static specialization for this packed index data."""
        return self.tree._schema.spec


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
    data = IndexData(tree, geometry)
    return Index(data.spec, data)


def _validate_source(index: Index[PrimitiveSpec[Any, Any, Any, Any], Any], source: Any) -> None:
    """Check that source geometry and connectivity reproduce the packed index."""
    import jax
    import numpy as np

    primitive = index.spec.primitive
    tree, geometry = index.data
    tree.validate_geometry(geometry, primitive)
    # JAX exposes the tree definition through its extension module.
    flatten = cast(Callable[[Any], tuple[list[Any], object]],
                   jax.tree_util.tree_flatten)  # pyright: ignore[reportUnknownMemberType]
    expected_leaves, expected_schema = flatten(geometry)
    try:
        source_lower, source_upper = primitive.bounds(source)
        if len(source_lower) != len(tree.prim_ids) or len(source_upper) != len(tree.prim_ids):
            raise ValueError("source primitive count differs")
        packed = primitive.pack(source, tree.prim_ids)
    except (AttributeError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("source geometry does not match the packed index") from exc
    leaves, schema = flatten(packed)
    if schema != expected_schema or any(
        actual is not expected and not np.array_equal(actual, expected)
        for actual, expected in zip(leaves, expected_leaves, strict=True)
    ):
        raise ValueError("source geometry does not match the packed index")


@overload
def prepare_boundary(
    index: Index[PrimitiveSpec[GeometryT, DeviceT, ItemT, PointT], GeometryT],
    source: GeometryT,
    *,
    adapter: BoundaryAdapter[GeometryT, DeviceT, ItemT, PointT,
                             CoreHostT, CoreDeviceT, FeatureHostT, FeatureDeviceT, FeatureItemT],
    silhouettes: bool = True,
) -> BoundaryData[GeometryT, CoreHostT, FeatureHostT]: ...


@overload
def prepare_boundary(
    index: Index[BoundaryPrimitive[GeometryT, DeviceT, ItemT, PointT,
                                 CoreHostT, CoreDeviceT, FeatureHostT, FeatureDeviceT, FeatureItemT], GeometryT],
    source: GeometryT,
    *,
    adapter: None = None,
    silhouettes: bool = True,
) -> BoundaryData[GeometryT, CoreHostT, FeatureHostT]: ...


def prepare_boundary(
    index: Index[PrimitiveSpec[Any, Any, Any, Any], Any],
    source: Any,
    *,
    adapter: BoundaryAdapter[Any, Any, Any, Any, Any, Any, Any, Any, Any] | None = None,
    silhouettes: bool = True,
) -> BoundaryData[Any, Any, Any]:
    """Prepare metadata for a BVH using the same source used to build it."""
    from sengi.boundary.data import BoundaryData, resolve_boundary_adapter

    _validate_source(index, source)
    primitive = index.spec.primitive
    tree, geometry = index.data
    boundary_primitive = cast("BoundaryPrimitive[Any, Any, Any, Any, Any, Any, Any, Any, Any]", primitive)
    resolved = (resolve_boundary_adapter(boundary_primitive) if adapter is None
                else resolve_boundary_adapter(primitive, adapter))
    core = resolved.prepare(primitive, source)
    meta = None
    if silhouettes:
        if resolved.build_silhouette is None or resolved.feature_primitive is None:
            raise TypeError("boundary adapter has no silhouette construction")
        meta = resolved.build_silhouette(core, tree, geometry)
    core = resolved.pack_core(core, source, tree, geometry)
    return BoundaryData(index.data, core, meta)

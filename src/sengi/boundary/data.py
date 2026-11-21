"""Indexed boundary adapters and optional silhouette metadata."""
from dataclasses import dataclass
from typing import Callable, Generic, NamedTuple, Protocol, TypeVar

import jax
from jax.typing import ArrayLike
import warp as wp

from sengi.warp.contracts import PrimitiveSpec, warp_struct
from sengi.bvh.index import IndexData
from sengi.bvh.device_tree import DeviceTree, TreeData


@warp_struct
class BoundaryCore:
    vertices: wp.array(dtype=wp.vec3)
    indices: wp.array2d(dtype=wp.int32)
    closed: wp.array(dtype=wp.int32)
    primitive_normal: wp.array(dtype=wp.vec3)
    primitive_measure: wp.array(dtype=wp.float32)


@warp_struct
class SilhouetteMeta:
    feature_a: wp.array(dtype=wp.vec3)
    feature_b: wp.array(dtype=wp.vec3)
    feature_n0: wp.array(dtype=wp.vec3)
    feature_n1: wp.array(dtype=wp.vec3)
    feature_refs: wp.array(dtype=wp.int32)
    node_feature_offset: wp.array(dtype=wp.int32)
    node_feature_count: wp.array(dtype=wp.int32)
    node_cone_axis: wp.array(dtype=wp.vec3)
    node_cone_sin_cos: wp.array(dtype=wp.vec2)
    node_cone_radius: wp.array(dtype=wp.float32)


class BoundaryCoreData(NamedTuple):
    vertices: ArrayLike
    indices: ArrayLike
    closed: ArrayLike
    primitive_normal: ArrayLike
    primitive_measure: ArrayLike


class SilhouetteData(NamedTuple):
    feature_a: ArrayLike
    feature_b: ArrayLike
    feature_n0: ArrayLike
    feature_n1: ArrayLike
    feature_refs: ArrayLike
    node_feature_offset: ArrayLike
    node_feature_count: ArrayLike
    node_cone_axis: ArrayLike
    node_cone_sin_cos: ArrayLike
    node_cone_radius: ArrayLike


H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
CH = TypeVar("CH")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")
FI = TypeVar("FI")


@jax.tree_util.register_dataclass
@dataclass(frozen=True, eq=False)
class BoundaryData(Generic[H, CH, FH]):
    """Packed index and boundary metadata prepared together."""

    data: IndexData[H]
    core: CH
    silhouette: FH | None = None


@dataclass(frozen=True)
class BoundaryAdapter(Generic[H, D, ItemT, P, CH, CD, FH, FD, FI]):
    """Prepare source-order core, then explicitly pack it into tree leaf order.

    build_silhouette returns ready-to-use host metadata, retaining its schema.
    """

    prepare: Callable[[PrimitiveSpec[H, D, ItemT, P], H], CH]
    pack_core: Callable[[CH, H, TreeData, H], CH]
    feature_primitive: object = None
    build_silhouette: Callable[[CH, TreeData, H], FH] | None = None
    visibility: object = None
    classification: object = None
    predicate: object = None
    measure: object = None
    robin: object = None


class BoundaryPrimitive(PrimitiveSpec[H, D, ItemT, P], Protocol[H, D, ItemT, P, CH, CD, FH, FD, FI]):
    """A declared adapter retains the primitive and metadata type connections."""

    @property
    def boundary_adapter(self) -> BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI]: ...


@wp.func_native("return bvh::cone_may_have_silhouette(viewpoint, node, tree.node_lower, tree.node_upper, meta.node_cone_sin_cos, meta.node_cone_axis, meta.node_cone_radius);")
def cone_visibility(viewpoint: wp.vec3, tree: DeviceTree, meta: SilhouetteMeta, node: int) -> bool: ...


def resolve_boundary_adapter(primitive, adapter=None):
    if adapter is not None:
        return adapter
    declared = getattr(primitive, "boundary_adapter", None)
    if declared is not None:
        return declared
    raise TypeError("boundary queries require an explicit BoundaryAdapter")


def _adapter_policy(primitive, name, adapter=None):
    policy = getattr(resolve_boundary_adapter(primitive, adapter), name)
    if policy is None:
        raise TypeError(f"boundary adapter requires an explicit {name} policy")
    return policy(primitive)


def _check_functions(*functions):
    if any(function is not None and not isinstance(function, wp.Function) for function in functions):
        raise TypeError("boundary device callbacks must be decorated Warp Functions")

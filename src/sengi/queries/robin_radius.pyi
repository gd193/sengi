"""Typed Robin radius queries and policies."""
from dataclasses import dataclass
from typing import Generic, Protocol, TypeVar, overload

import jax
import warp as wp
from jax.typing import ArrayLike

from sengi.warp.native import NativeHeaders
from sengi.boundary.data import BoundaryCoreData, SilhouetteData, BoundaryCore, SilhouetteMeta
from sengi.boundary.robin import CoefficientMeta, RobinData
from sengi.bvh.device_tree import DeviceTree
from sengi.bvh.index import IndexData, IndexSpec
from sengi.metrics.nearest import Euclidean, NearestPolicy
from sengi.warp.contracts import PrimitiveSpec, WarpFunction, WarpType
from sengi.warp.launch import BoundQuery


H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
PointContra = TypeVar("PointContra", contravariant=True)
Q = TypeVar("Q")
CH = TypeVar("CH")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")


@dataclass(frozen=True)
class RobinOps(Generic[Q, D, ItemT, CD, FD]):
    query_type: WarpType[Q]
    query_shape: tuple[int, ...]
    valid: WarpFunction[[Q], bool]
    node_bound: WarpFunction[[Q, wp.vec3, wp.vec3, wp.vec2], float] | None
    primitive_bound: WarpFunction[[Q, ItemT, float], float] | None
    feature_bound: WarpFunction[[Q, wp.vec3, wp.vec3, wp.vec3, wp.vec3], float] | None
    node_bound_with_data: WarpFunction[[Q, DeviceTree, D, CD, FD, CoefficientMeta, int], float] | None = ...
    feature_bound_with_data: WarpFunction[[Q, DeviceTree, D, CD, FD, CoefficientMeta, int, int], float] | None = ...
    core_type: WarpType[CD] | None = ...
    feature_type: WarpType[FD] | None = ...
    to_score: WarpFunction[[float], float] | None = ...
    to_distance: WarpFunction[[float], float] | None = ...
    leaf_bound_with_data: WarpFunction[[Q, D, CD, CoefficientMeta, int, int, float], float] | None = ...
    node_visit: WarpFunction[[Q, DeviceTree, D, CD, FD, CoefficientMeta, int, wp.vec3, wp.vec3, float], tuple[float, float, bool, bool]] | None = ...
    native: NativeHeaders | None = None

class RobinPolicy(Protocol[D, ItemT, P, Q, CD, FD]):
    def bind(self, primitive: PrimitiveSpec[H, D, ItemT, P], metric: NearestPolicy[D, ItemT, P, Q]) -> RobinOps[Q, D, ItemT, CD, FD]: ...

class RobinMetric(Protocol[D, ItemT, PointContra, Q, CD, FD]):
    def bind_robin(self, primitive: PrimitiveSpec[H, D, ItemT, PointContra]) -> RobinOps[Q, D, ItemT, CD, FD]: ...


class RobinQuery(BoundQuery, Generic[H, D, ItemT, Q, CH, CD, FH, FD]):
    device: WarpFunction[[Q, float, DeviceTree, D, CD, FD, CoefficientMeta], tuple[float, int]]
    ops: RobinOps[Q, D, ItemT, CD, FD]
    def __call__(self, data: IndexData[H], metadata: RobinData[CH, FH],
                 queries: ArrayLike, cap: ArrayLike = ...
                 ) -> tuple[jax.Array, jax.Array]: ...


@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]],
    policy: None = ..., *, metric: RobinMetric[D, ItemT, P, Q, CD, FD],
    core_host_type: type[CH], feature_host_type: type[FH],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, CH, CD, FH, FD]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]],
    policy: None = ..., *, metric: RobinMetric[D, ItemT, P, Q, CD, FD],
    core_host_type: type[BoundaryCoreData] = ..., feature_host_type: type[SilhouetteData] = ...,
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, BoundaryCoreData, CD, SilhouetteData, FD]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]],
    policy: None, metric: RobinMetric[D, ItemT, P, Q, CD, FD], *,
    core_host_type: type[CH], feature_host_type: type[FH],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, CH, CD, FH, FD]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]],
    policy: None, metric: RobinMetric[D, ItemT, P, Q, CD, FD], *,
    core_host_type: type[BoundaryCoreData] = ..., feature_host_type: type[SilhouetteData] = ...,
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, BoundaryCoreData, CD, SilhouetteData, FD]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]],
    policy: RobinOps[Q, D, ItemT, CD, FD] | RobinPolicy[D, ItemT, P, Q, CD, FD],
    metric: NearestPolicy[D, ItemT, P, Q] | None = ..., *, core_host_type: type[CH],
    feature_host_type: type[FH], stack_capacity: int = ...,
    block_dim: int = ...) -> RobinQuery[H, D, ItemT, Q, CH, CD, FH, FD]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]],
    policy: RobinOps[Q, D, ItemT, CD, FD] | RobinPolicy[D, ItemT, P, Q, CD, FD],
    metric: NearestPolicy[D, ItemT, P, Q] | None = ..., *,
    core_host_type: type[BoundaryCoreData] = ..., feature_host_type: type[SilhouetteData] = ...,
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, Q, BoundaryCoreData, CD, SilhouetteData, FD]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]],
    policy: None = ..., metric: Euclidean | None = ..., *,
    core_host_type: type[BoundaryCoreData] = ..., feature_host_type: type[SilhouetteData] = ...,
    stack_capacity: int = ..., block_dim: int = ...
    ) -> RobinQuery[H, D, ItemT, P, BoundaryCoreData, BoundaryCore, SilhouetteData, SilhouetteMeta]: ...

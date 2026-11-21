"""Host specialization signatures; JAX arrays do not encode vector dimensions."""
from sengi.bvh.device_tree import DeviceTree
from typing import Generic, TypeVar, overload
import jax
from jax.typing import ArrayLike
import warp as wp
from sengi.warp.launch import BoundQuery
from sengi.metrics.nearest import NearestOps, NearestPolicy, NearestPredicate
from sengi.warp.contracts import PrimitiveSpec, WarpFunction
from sengi.queries.results import ResultPolicy, IdPoint, IdPointResult
from sengi.bvh.index import IndexData, IndexSpec

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")
ResultT = TypeVar("ResultT")

class NearestQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT], tuple[PointT, float, int, int]]
    ops: NearestOps[QueryT, ItemT, PointT]
    def __call__(self, data: IndexData[HostDataT], queries: ArrayLike,
                 max_distance: ArrayLike = ...) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...

class SelectedNearestQuery(BoundQuery,
                           Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT], tuple[ResultT, int]]
    result_policy: ResultPolicy[QueryT, ItemT, PointT, ResultT]
    ops: NearestOps[QueryT, ItemT, PointT]

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
         metric: None = ..., *,
         predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
         stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool | None = ...,
         packed_nodes: bool = ..., group_leaves: bool | None = ..., result_policy: None = ...
         ) -> NearestQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                 result_policy: ResultPolicy[QueryT, ItemT, PointT, ResultT],
                 predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool | None = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 metric: None = ..., *,
                 evaluate: None = ...,
                 lower_bound: None = ...,
                 result_policy: ResultPolicy[PointT, ItemT, PointT, ResultT],
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool | None = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, ResultT]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 metric: None = ..., *,
                 evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                 lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                 result_policy: ResultPolicy[PointT, ItemT, PointT, ResultT],
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool | None = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, ResultT]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                 result_policy: IdPoint,
                 predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool | None = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT, IdPointResult[PointT]]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 metric: None = ..., *,
                 evaluate: None = ...,
                 lower_bound: None = ...,
                 result_policy: IdPoint,
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool | None = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, IdPointResult[PointT]]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 metric: None = ..., *,
                 evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                 lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                 result_policy: IdPoint,
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool | None = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> SelectedNearestQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, IdPointResult[PointT]]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                 predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                 result_policy: None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool | None = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> NearestQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 metric: None = ..., *,
                 evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                 lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 result_policy: None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool | None = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ...
                 ) -> NearestQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT]: ...

def prepare_nearest(ops: NearestOps[QueryT, ItemT, PointT], queries: ArrayLike,
                    max_distance: ArrayLike) -> tuple[jax.Array, jax.Array, tuple[int, ...]]: ...

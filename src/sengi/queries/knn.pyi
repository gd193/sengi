"""Host specialization signatures; JAX arrays do not encode vector dimensions."""
from sengi.bvh.device_tree import DeviceTree
from typing import Generic, Literal, TypeVar, overload
import jax
from jax.typing import ArrayLike
import warp as wp
from sengi.warp.launch import BoundQuery
from sengi.metrics.nearest import NearestOps, NearestPolicy, NearestPredicate
from sengi.warp.contracts import PrimitiveSpec, WarpFunction, DeviceArray2D
from sengi.queries.results import ResultPolicy, IdPoint, IdPointResult
from sengi.bvh.index import IndexData, IndexSpec

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
QueryT = TypeVar("QueryT")
ResultT = TypeVar("ResultT")

class KnnQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT, DeviceArray2D[float], DeviceArray2D[int] | DeviceArray2D[PointT], int], tuple[int, int]]
    k: int
    return_points: bool
    ops: NearestOps[QueryT, ItemT, PointT]
    def __call__(self, data: IndexData[HostDataT], queries: ArrayLike,
                 max_distance: ArrayLike = ...) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...

class SelectedKnnQuery(BoundQuery,
                       Generic[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]):
    device: WarpFunction[[QueryT, float, DeviceTree, DeviceDataT, DeviceArray2D[float], DeviceArray2D[ResultT], int], tuple[int, int]]
    result_policy: ResultPolicy[QueryT, ItemT, PointT, ResultT]
    ops: NearestOps[QueryT, ItemT, PointT]
    k: int

def validate_batch_size(rows: int, k: int) -> None: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
             k: int, metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
             result_policy: ResultPolicy[QueryT, ItemT, PointT, ResultT],
             predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT, ResultT]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
             k: int, metric: None = ..., *,
             evaluate: None = ...,
             lower_bound: None = ...,
             result_policy: ResultPolicy[PointT, ItemT, PointT, ResultT],
             predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, ResultT]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
             k: int, metric: None = ..., *,
             evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
             lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
             result_policy: ResultPolicy[PointT, ItemT, PointT, ResultT],
             predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, ResultT]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
             k: int, metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
             result_policy: IdPoint,
             predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT, IdPointResult[PointT]]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
             k: int, metric: None = ..., *,
             evaluate: None = ...,
             lower_bound: None = ...,
             result_policy: IdPoint,
             predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, IdPointResult[PointT]]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
             k: int, metric: None = ..., *,
             evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
             lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
             result_policy: IdPoint,
             predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
             stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
             packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: Literal[False] = ...
             ) -> SelectedKnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT, IdPointResult[PointT]]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 k: int, metric: NearestPolicy[DeviceDataT, ItemT, PointT, QueryT], *,
                 predicate: WarpFunction[[QueryT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, QueryT] | None = ...,
                 result_policy: None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: bool = ...
                 ) -> KnnQuery[HostDataT, DeviceDataT, ItemT, PointT, QueryT]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 k: int, metric: None = ..., *,
                 evaluate: None = ...,
                 lower_bound: None = ...,
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 result_policy: None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: bool = ...
                 ) -> KnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]],
                 k: int, metric: None = ..., *,
                 evaluate: WarpFunction[[PointT, ItemT], tuple[PointT, float]],
                 lower_bound: WarpFunction[[PointT, wp.vec3, wp.vec3], float],
                 predicate: WarpFunction[[PointT, ItemT], bool] | NearestPredicate[DeviceDataT, ItemT, PointT, PointT] | None = ...,
                 result_policy: None = ...,
                 stack_capacity: int = ..., block_dim: int = ..., cache_bounds: bool = ...,
                 packed_nodes: bool = ..., group_leaves: bool | None = ..., return_points: bool = ...
                 ) -> KnnQuery[HostDataT, DeviceDataT, ItemT, PointT, PointT]: ...

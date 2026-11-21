"""Static classification callback and host/device metadata connections."""
from sengi.bvh.device_tree import DeviceTree
from typing import Any, Generic, TypeVar, overload
import jax
from jax.typing import ArrayLike
from sengi.boundary.data import BoundaryAdapter, BoundaryData, BoundaryPrimitive
from sengi.queries.domain import ClassificationOps
from sengi.warp.launch import BoundQuery
from sengi.warp.contracts import PrimitiveSpec, WarpFunction
from sengi.bvh.index import IndexSpec

H = TypeVar("H")
D = TypeVar("D")
I = TypeVar("I")
P = TypeVar("P")
M = TypeVar("M")
HM = TypeVar("HM")

class ClassifyQuery(BoundQuery, Generic[H, D, I, P, HM, M]):
    device: WarpFunction[[P, DeviceTree, D, M], tuple[float, int]]
    policy: ClassificationOps[P, D, M]
    def __call__(self, boundary: BoundaryData[H, HM, Any], points: ArrayLike) -> tuple[jax.Array, jax.Array]: ...

@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, I, P]], policy: ClassificationOps[P, D, M], *, metadata_host_type: type[HM], adapter: None = ..., stack_capacity: int = ..., block_dim: int = ...) -> ClassifyQuery[H, D, I, P, HM, M]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, I, P]], policy: ClassificationOps[P, D, M], *, metadata_host_type: None = ..., adapter: None = ..., stack_capacity: int = ..., block_dim: int = ...) -> ClassifyQuery[H, D, I, P, M, M]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, I, P]], policy: None = ..., *, adapter: BoundaryAdapter[H, D, I, P, HM, M, Any, Any, Any], metadata_host_type: None = ..., stack_capacity: int = ..., block_dim: int = ...) -> ClassifyQuery[H, D, I, P, HM, M]: ...
@overload
def bind(spec: IndexSpec[BoundaryPrimitive[H, D, I, P, HM, M, Any, Any, Any]], policy: None = ..., *, adapter: None = ..., metadata_host_type: None = ..., stack_capacity: int = ..., block_dim: int = ...) -> ClassifyQuery[H, D, I, P, HM, M]: ...

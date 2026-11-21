from sengi.bvh.device_tree import DeviceTree
from dataclasses import dataclass
from typing import Generic, TypeVar, overload
import jax
from jax.typing import ArrayLike
from sengi.warp.contracts import PrimitiveSpec, WarpFunction
from sengi.boundary.data import BoundaryAdapter, BoundaryData, BoundaryPrimitive
from sengi.metrics.nearest import NearestPolicy
from sengi.queries.nearest import NearestQuery
from sengi.bvh.index import IndexSpec
H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
Q = TypeVar("Q")
CH = TypeVar("CH")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")
FI = TypeVar("FI")

@dataclass(frozen=True)
class SilhouetteQuery(Generic[H, D, ItemT, P, Q, CH, CD, FH, FD, FI]):
    device: WarpFunction[[Q, float, DeviceTree, FD], tuple[P, float, int, int]]
    @property
    def _device_kernel(self) -> object: ...
    query: NearestQuery[FH, FD, FI, P, Q]
    @property
    def kernel(self) -> object: ...
    def __call__(self, boundary: BoundaryData[H, CH, FH],
                 queries: ArrayLike, max_distance: ArrayLike = ...
                 ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]: ...


@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]], metric: None = ..., *,
    adapter: BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> SilhouetteQuery[H, D, ItemT, P, P, CH, CD, FH, FD, FI]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]], metric: NearestPolicy[FD, FI, P, Q], *,
    adapter: BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> SilhouetteQuery[H, D, ItemT, P, Q, CH, CD, FH, FD, FI]: ...
@overload
def bind(spec: IndexSpec[BoundaryPrimitive[H, D, ItemT, P, CH, CD, FH, FD, FI]], metric: None = ..., *,
    adapter: None = ..., stack_capacity: int = ..., block_dim: int = ...
    ) -> SilhouetteQuery[H, D, ItemT, P, P, CH, CD, FH, FD, FI]: ...
@overload
def bind(spec: IndexSpec[BoundaryPrimitive[H, D, ItemT, P, CH, CD, FH, FD, FI]], metric: NearestPolicy[FD, FI, P, Q], *,
    adapter: None = ..., stack_capacity: int = ..., block_dim: int = ...
    ) -> SilhouetteQuery[H, D, ItemT, P, Q, CH, CD, FH, FD, FI]: ...

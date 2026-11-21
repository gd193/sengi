from sengi.bvh.device_tree import DeviceTree
from dataclasses import dataclass
from typing import Generic, TypeVar, overload
import jax
from jax.typing import ArrayLike
from sengi.warp.contracts import PrimitiveSpec, WarpFunction
from sengi.boundary.data import BoundaryAdapter, BoundaryData, BoundaryPrimitive
from sengi.queries.silhouette import SilhouetteQuery
from sengi.bvh.index import IndexData, IndexSpec
H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
CH = TypeVar("CH")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")
FI = TypeVar("FI")

@dataclass(frozen=True)
class StarRadiusQuery(Generic[H, D, ItemT, P, CH, CD, FH, FD, FI]):
    device: WarpFunction[[P, float, DeviceTree, FD], tuple[float, int]]
    @property
    def _device_kernel(self) -> object: ...
    silhouette: SilhouetteQuery[H, D, ItemT, P, P, CH, CD, FH, FD, FI]
    def __call__(self, data: IndexData[H], boundary: BoundaryData[CH, FH] | FH,
                 points: ArrayLike, cap: ArrayLike) -> tuple[jax.Array, jax.Array]: ...


@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]], *,
    adapter: BoundaryAdapter[H, D, ItemT, P, CH, CD, FH, FD, FI],
    stack_capacity: int = ..., block_dim: int = ...
    ) -> StarRadiusQuery[H, D, ItemT, P, CH, CD, FH, FD, FI]: ...
@overload
def bind(spec: IndexSpec[BoundaryPrimitive[H, D, ItemT, P, CH, CD, FH, FD, FI]], *,
    adapter: None = ..., stack_capacity: int = ..., block_dim: int = ...
    ) -> StarRadiusQuery[H, D, ItemT, P, CH, CD, FH, FD, FI]: ...

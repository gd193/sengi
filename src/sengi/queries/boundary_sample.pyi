"""Typed boundary sampling query."""
from dataclasses import dataclass
from typing import Generic, TypeVar, overload

import jax
import warp as wp
from jax.typing import ArrayLike

from sengi.warp.native import NativeHeaders
from sengi.boundary.sampling import MeasureOps, SamplingData, SamplingMeta
from sengi.bvh.device_tree import DeviceTree
from sengi.bvh.index import IndexData, IndexSpec
from sengi.warp.contracts import PrimitiveSpec, WarpFunction, WarpType
from sengi.warp.launch import BoundQuery

H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
Q = TypeVar("Q")
CH = TypeVar("CH")


@dataclass(frozen=True)
class RegionOps(Generic[Q, P]):
    query_type: WarpType[Q]
    query_shape: tuple[int, ...]
    valid: WarpFunction[[Q, float], bool]
    overlaps: WarpFunction[[Q, float, wp.vec3, wp.vec3], bool]
    contains: WarpFunction[[Q, float, P], bool]
    native: NativeHeaders | None = None


class BoundarySampleQuery(BoundQuery, Generic[H, D, ItemT, P, Q]):
    device: WarpFunction[[Q, float, wp.vec3, DeviceTree, D, SamplingMeta], tuple[P, P, float, int, int]]
    region: RegionOps[Q, P]
    def __call__(self, data: IndexData[H], metadata: SamplingData,
                 queries: ArrayLike, radii: ArrayLike, randoms: ArrayLike
                 ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]: ...


@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]], *, region: RegionOps[Q, P],
    block_dim: int = ...) -> BoundarySampleQuery[H, D, ItemT, P, Q]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]],
    measure: MeasureOps[ItemT, P, CH] | None = ..., region: None = ...,
    *, block_dim: int = ...) -> BoundarySampleQuery[H, D, ItemT, P, P]: ...
@overload
def bind(spec: IndexSpec[PrimitiveSpec[H, D, ItemT, P]],
    measure: MeasureOps[ItemT, P, CH] | None, region: RegionOps[Q, P],
    *, block_dim: int = ...) -> BoundarySampleQuery[H, D, ItemT, P, Q]: ...

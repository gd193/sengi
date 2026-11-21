"""Typed boundary sampling metadata and measure preparation."""
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

import warp as wp
from jax.typing import ArrayLike

from sengi.warp.native import NativeHeaders
from sengi.boundary.data import BoundaryData
from sengi.bvh.index import IndexData
from sengi.warp.contracts import DeviceArray, WarpFunction

ItemT = TypeVar("ItemT")
P = TypeVar("P")
H = TypeVar("H")
CH = TypeVar("CH")
FH = TypeVar("FH")


class SamplingMeta:
    primitive_weight: DeviceArray[float]
    node_weight: DeviceArray[float]


@dataclass(frozen=True, eq=False)
class SamplingData(Generic[H]):
    data: IndexData[H]
    primitive_weight: ArrayLike
    node_weight: ArrayLike


@dataclass(frozen=True)
class MeasureOps(Generic[ItemT, P, CH]):
    sample: WarpFunction[[ItemT, wp.vec2], tuple[P, P, float]]
    weights: Callable[[CH], ArrayLike] | None = ...
    native: NativeHeaders | None = None


def prepare_boundary_sample(boundary: BoundaryData[H, CH, FH],
    measure: MeasureOps[ItemT, P, CH] | None = ...) -> SamplingData[H]: ...

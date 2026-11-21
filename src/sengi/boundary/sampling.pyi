"""Typed boundary sampling metadata and measure preparation."""
from dataclasses import dataclass
from typing import Callable, Generic, NamedTuple, TypeVar

import warp as wp
from jax.typing import ArrayLike

from sengi.boundary.data import BoundaryData
from sengi.bvh.device_tree import TreeData
from sengi.warp.contracts import DeviceArray, WarpFunction

ItemT = TypeVar("ItemT")
P = TypeVar("P")
CH = TypeVar("CH")
FH = TypeVar("FH")


class SamplingMeta:
    primitive_weight: DeviceArray[float]
    node_weight: DeviceArray[float]


class SamplingData(NamedTuple):
    primitive_weight: ArrayLike
    node_weight: ArrayLike


@dataclass(frozen=True)
class MeasureOps(Generic[ItemT, P, CH]):
    sample: WarpFunction[[ItemT, wp.vec2], tuple[P, P, float]]
    weights: Callable[[CH], ArrayLike] | None = ...


def prepare_boundary_sample(tree: TreeData, boundary: BoundaryData[CH, FH] | CH,
    measure: MeasureOps[ItemT, P, CH] | None = ...) -> SamplingData: ...

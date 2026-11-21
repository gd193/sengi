"""Typed Robin boundary coefficients and metadata preparation."""
from typing import Generic, NamedTuple, TypeVar

import warp as wp
from jax.typing import ArrayLike

from sengi.boundary.data import BoundaryData
from sengi.bvh.device_tree import TreeData
from sengi.bvh.index import IndexData
from sengi.warp.contracts import DeviceArray

H = TypeVar("H")
CH = TypeVar("CH")
FH = TypeVar("FH")


class CoefficientMeta:
    primitive_bounds: DeviceArray[wp.vec2]
    node_bounds: DeviceArray[wp.vec2]


class CoefficientData(NamedTuple):
    primitive_bounds: ArrayLike
    node_bounds: ArrayLike
class RobinData(NamedTuple, Generic[H, CH, FH]):
    data: IndexData[H]
    core: CH
    silhouette: FH
    coefficients: CoefficientData


def prepare_robin_coefficients(tree: TreeData, minimum: ArrayLike, maximum: ArrayLike | None = ...) -> CoefficientData: ...
def prepare_robin(boundary: BoundaryData[H, CH, FH], minimum: ArrayLike,
                  maximum: ArrayLike | None = ...) -> RobinData[H, CH, FH]: ...

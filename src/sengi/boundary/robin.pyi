"""Typed Robin boundary coefficients and metadata preparation."""
from typing import Generic, NamedTuple, TypeVar

import warp as wp
from jax.typing import ArrayLike

from sengi.boundary.data import BoundaryData
from sengi.bvh.device_tree import TreeData
from sengi.warp.contracts import DeviceArray

CH = TypeVar("CH")
FH = TypeVar("FH")


class CoefficientMeta:
    primitive_bounds: DeviceArray[wp.vec2]
    node_bounds: DeviceArray[wp.vec2]


class CoefficientData(NamedTuple):
    primitive_bounds: ArrayLike
    node_bounds: ArrayLike
class RobinData(NamedTuple, Generic[CH, FH]):
    core: CH
    silhouette: FH
    coefficients: CoefficientData


def prepare_robin_coefficients(tree: TreeData, minimum: ArrayLike, maximum: ArrayLike | None = ...) -> CoefficientData: ...
def prepare_robin(tree: TreeData, boundary: BoundaryData[CH, FH], minimum: ArrayLike,
                  maximum: ArrayLike | None = ...) -> RobinData[CH, FH]: ...

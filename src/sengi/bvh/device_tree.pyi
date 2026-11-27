"""Binary tree array schemas and construction API."""
from dataclasses import dataclass
from typing import Any, Literal, TypeVar

import jax
import numpy as np
from numpy.typing import NDArray
import warp as wp

from sengi.bvh.config import Config
from sengi.bvh.index import IndexSpec
from sengi.warp.contracts import DeviceArray, DeviceArray2D, PrimitiveSpec

H = TypeVar("H")
D = TypeVar("D")
I = TypeVar("I")
P = TypeVar("P")

STACK_CAPACITY: int


class DeviceTree:
    node_lower: DeviceArray[wp.vec3]
    node_upper: DeviceArray[wp.vec3]
    node_left: DeviceArray[int]
    node_right: DeviceArray[int]
    node_leaf_offset: DeviceArray[int]
    node_leaf_count: DeviceArray[int]
    prim_ids: DeviceArray[int]
    packed_nodes: DeviceArray2D[wp.vec4]


@dataclass(frozen=True)
class _IndexSchema:
    spec: IndexSpec[Any]
    geometry_type: type


@dataclass(frozen=True, eq=False)
class TreeData:
    node_lower: jax.Array
    node_upper: jax.Array
    node_left: jax.Array
    node_right: jax.Array
    node_leaf_offset: jax.Array
    node_leaf_count: jax.Array
    prim_ids: jax.Array
    packed_nodes: jax.Array
    _schema: _IndexSchema

    def validate_geometry(self, geometry: object, primitive: object) -> None: ...

def build_device_tree(primitive: PrimitiveSpec[H, D, I, P], data: H,
               config: Config | None = ..., *,
               builder: Literal["sah", "lbvh"] = ...) -> tuple[TreeData, H]: ...


def pack_device_tree(primitive: PrimitiveSpec[H, D, I, P], data: H,
                     node_lower: NDArray[np.float32], node_upper: NDArray[np.float32],
                     node_left: NDArray[np.int32], node_right: NDArray[np.int32],
                     node_leaf_offset: NDArray[np.int32], node_leaf_count: NDArray[np.int32],
                     prim_ids: NDArray[np.int32]) -> tuple[TreeData, H]: ...

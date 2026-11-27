"""Static crossing and classification operations."""
from dataclasses import dataclass
from typing import Generic, TypeVar, cast

from sengi.warp.native import NativeHeaders
from sengi.warp.contracts import WarpFunction, WarpType, warp_func, warp_struct

import warp as wp


@warp_struct
class Ray:
    origin: wp.vec3
    direction: wp.vec3
    inverse: wp.vec3
    fast: bool
    dominant_axis: int
    frame_origin: wp.vec3
    frame_direction: wp.vec3
    shear: wp.vec2


@warp_func
def prepare_ray(origin: wp.vec3, direction: wp.vec3, dimension: int) -> Ray:
    ray = Ray()
    ray.origin = origin
    ray.direction = direction
    components = direction
    if dimension == 2:
        components[2] = 1.0
    ray.fast = wp.min(wp.abs(components)) >= 1.e-20
    if ray.fast:
        ray.inverse = wp.cw_div(wp.vec3(1.0), components)
    ax, ay, az = wp.abs(direction[0]), wp.abs(direction[1]), wp.abs(direction[2])
    # Fixed component accesses let CUDA keep the vectors in registers. Strict
    # comparisons choose the first axis when magnitudes tie.
    if az > (ay if ay > ax else ax):
        ray.dominant_axis = 2
        ray.frame_origin = origin
        ray.frame_direction = direction
    elif ay > ax:
        ray.dominant_axis = 1
        ray.frame_origin = wp.vec3(origin[2], origin[0], origin[1])
        ray.frame_direction = wp.vec3(direction[2], direction[0], direction[1])
    else:
        ray.dominant_axis = 0
        ray.frame_origin = wp.vec3(origin[1], origin[2], origin[0])
        ray.frame_direction = wp.vec3(direction[1], direction[2], direction[0])
    frame = ray.frame_direction
    ray.shear = wp.vec2(frame[0] / frame[2], frame[1] / frame[2])
    return ray


DeviceDataT = TypeVar("DeviceDataT")
StateT = TypeVar("StateT")
PointT = TypeVar("PointT")
MetaT = TypeVar("MetaT")


@dataclass(frozen=True)
class RayOps(Generic[DeviceDataT, StateT]):
    """Single hit or next_hit -> (state, t, flag): 1 hit, 0 done, -1 failure."""
    intersect: WarpFunction[[Ray, float, float, DeviceDataT, int, bool, bool], wp.vec2] | None = None
    state_type: WarpType[StateT] = cast(WarpType[StateT], wp.int32)
    next_hit: WarpFunction[[Ray, float, float, DeviceDataT, int, StateT, bool, bool], tuple[StateT, float, int]] | None = None

    native: NativeHeaders | None = None


@dataclass(frozen=True)
class ClassificationOps(Generic[PointT, DeviceDataT, MetaT]):
    """event(point, data, slot, id, meta): 0 miss, 1 parity, 2 contact, -1 failure."""
    metadata_type: WarpType[MetaT]
    valid: WarpFunction[[MetaT], bool]
    event: WarpFunction[[PointT, DeviceDataT, int, int, MetaT], int]

    native: NativeHeaders | None = None

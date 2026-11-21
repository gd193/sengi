"""Shared Warp callbacks for custom metrics and third-party ball primitives."""
from dataclasses import dataclass
from typing import Any, NamedTuple

import warp as wp


def weighted_callbacks(dimension, weights):
    Vec = wp.vec2 if dimension == 2 else wp.vec3
    weights = Vec(*weights)

    @wp.func(module="unique")
    def evaluate(q: Vec, item: Any):
        d = q-item.a
        score = float(0.0)
        for axis in range(wp.static(dimension)):
            score += d[axis]*d[axis]*wp.static(weights)[axis]
        return item.a, score

    @wp.func(module="unique")
    def lower_bound(q: Vec, lo: wp.vec3, hi: wp.vec3):
        score = float(0.0)
        for axis in range(wp.static(dimension)):
            d = wp.max(wp.max(lo[axis]-q[axis], 0.0), q[axis]-hi[axis])
            score += d*d*wp.static(weights)[axis]
        return score

    return evaluate, lower_bound


@wp.struct
class BallData:
    centers: wp.array(dtype=wp.vec3)
    radii: wp.array(dtype=wp.float32)


@wp.struct
class Ball:
    center: wp.vec3
    radius: float


@wp.func
def ball_load(data: BallData, slot: int):
    item = Ball()
    item.center = data.centers[slot]
    item.radius = data.radii[slot]
    return item


@wp.func
def ball_bounds(item: Ball):
    return item.center-wp.vec3(item.radius), item.center+wp.vec3(item.radius)


@wp.func
def ball_evaluate(query: wp.vec3, item: Ball):
    delta = query-item.center
    length = wp.length(delta)
    point = query
    if length > item.radius:
        point = item.center+delta*(item.radius/length)
    return point, wp.length_sq(query-point)


@wp.func
def ball_lower_bound(query: wp.vec3, lo: wp.vec3, hi: wp.vec3):
    delta = wp.max(wp.max(lo-query, wp.vec3(0.0)), query-hi)
    return wp.length_sq(delta)


class BallArrays(NamedTuple):
    centers: Any
    radii: Any


@dataclass(frozen=True)
class Balls:
    dimension: int = 3
    point_type: Any = wp.vec3
    data_type: Any = BallData
    load: Any = ball_load
    bounds_of_item: Any = ball_bounds

    def bounds(self, data):
        return data.centers-data.radii[:, None], data.centers+data.radii[:, None]

    def pack(self, data, ids):
        return BallArrays(data.centers[ids], data.radii[ids])

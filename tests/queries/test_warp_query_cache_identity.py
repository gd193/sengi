"""Resolved query defaults share device specializations without merging callbacks."""
from sengi import bvh
from dataclasses import replace
from typing import Any

import pytest

wp = pytest.importorskip("warp")

from sengi.boundary.data import resolve_boundary_adapter  # noqa: E402
from sengi.queries.boundary_sample import _ball_region
from sengi.queries.boundary_sample import bind as bind_boundary_sample
from sengi.queries.robin_radius import bind as bind_robin_radius  # noqa: E402
from sengi.metrics.nearest import Euclidean
from sengi.queries.domain import RayOps
from sengi.primitives.segments import Segments
from sengi.queries.rays import bind as bind_rays
from sengi.queries.silhouette import bind as bind_silhouette
from sengi.queries.star_radius import bind as bind_star_radius
from sengi.queries.domain import Ray  # noqa: E402


@pytest.mark.parametrize("mode", ["first", "any", "count", "all"])
def test_ray_resolved_defaults_and_options(mode):
    primitive = Segments(2)
    query = bind_rays(bvh.IndexSpec(primitive), mode)
    assert query is bind_rays(spec=bvh.IndexSpec(primitive), mode=mode, ops=RayOps(intersect=primitive.intersect),
                                   exclude=False, stack_capacity=64, block_dim=128)
    assert query is not bind_rays(bvh.IndexSpec(primitive), mode, block_dim=64)
    assert bind_rays(bvh.IndexSpec(primitive), mode, block_dim=64).block_dim == 64
    assert query is not bind_rays(bvh.IndexSpec(primitive), mode, exclude=True)


def test_ray_callback_identity():
    primitive = Segments(2)

    def callback(value):
        @wp.func(module="unique")
        def intersect(ray: Ray, minimum: float, maximum: float, data: Any,
                      slot: int, half_open: bool, store_times: bool):
            return wp.vec2(wp.static(value), 1.0)
        return intersect

    first, second = callback(1.0), callback(2.0)
    query = bind_rays(bvh.IndexSpec(primitive), ops=RayOps(intersect=first))
    assert query is bind_rays(bvh.IndexSpec(primitive), ops=RayOps(intersect=first))
    assert query is not bind_rays(bvh.IndexSpec(primitive), ops=RayOps(intersect=callback(1.0)))
    assert query.kernel is not bind_rays(bvh.IndexSpec(primitive), ops=RayOps(intersect=second)).kernel


def test_boundary_resolved_defaults_and_host_markers():
    primitive = Segments(2)
    adapter = resolve_boundary_adapter(primitive)
    metric = Euclidean()
    robin = bind_robin_radius(bvh.IndexSpec(primitive))
    assert robin is bind_robin_radius(bvh.IndexSpec(primitive), metric=metric)
    assert robin is bind_robin_radius(bvh.IndexSpec(primitive), adapter.robin(primitive), metric,
                                      core_host_type=object, feature_host_type=tuple)
    assert robin is not bind_robin_radius(bvh.IndexSpec(primitive), block_dim=128)
    class BoundPolicy:
        def bind(self, primitive, metric):
            return robin.ops

    assert robin is bind_robin_radius(bvh.IndexSpec(primitive), BoundPolicy())
    assert robin.kernel is not bind_robin_radius(bvh.IndexSpec(primitive), replace(robin.ops, to_score=_score,
                                                                to_distance=_score)).kernel
    sample = bind_boundary_sample(bvh.IndexSpec(primitive))
    assert sample is bind_boundary_sample(bvh.IndexSpec(primitive), adapter.measure(primitive), _ball_region(primitive), block_dim=256)


@wp.func
def _score(value: float):
    return value


def test_silhouette_resolved_defaults():
    primitive = Segments(2)
    adapter = resolve_boundary_adapter(primitive)
    query = bind_silhouette(bvh.IndexSpec(primitive))
    assert query is bind_silhouette(bvh.IndexSpec(primitive), Euclidean(), adapter=adapter,
                                       stack_capacity=64, block_dim=64)
    assert query is not bind_silhouette(bvh.IndexSpec(primitive), block_dim=128)
    assert bind_star_radius(bvh.IndexSpec(primitive)) is bind_star_radius(bvh.IndexSpec(primitive), adapter=adapter)

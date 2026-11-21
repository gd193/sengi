"""Boundary sampling with static measure and region policies."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, TypeVar

import jax.numpy as jnp
import warp as wp

from sengi.warp.native import NativeHeaders
from sengi.boundary.data import _adapter_policy, _check_functions
from sengi.boundary.sampling import SamplingMeta
from sengi.bvh.device_tree import DeviceTree
from sengi.queries.nearest import prepare_nearest
from sengi.warp.arrays import load_array
from sengi.warp.launch import BoundQuery, bind_query, cache_query

H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
Q = TypeVar("Q")


@dataclass(frozen=True)
class RegionOps(Generic[Q, P]):
    """overlaps is conservative; contains tests the sampled point itself."""
    query_type: Any
    query_shape: tuple
    valid: Any
    overlaps: Any
    contains: Any
    native: NativeHeaders | None = None


@lru_cache(maxsize=None)
def _ball_region(primitive):
    Vec, dimension = primitive.point_type, primitive.dimension

    @wp.func
    def valid(query: Vec, radius: float):
        return wp.isfinite(query) and not (radius != radius) and radius >= 0.0

    @wp.func
    def overlaps(query: Vec, radius: float, lower: wp.vec3, upper: wp.vec3):
        distance2 = float(0.0)
        for j in range(wp.static(dimension)):
            delta = wp.max(wp.max(lower[j]-query[j], 0.0), query[j]-upper[j])
            distance2 += delta*delta
        return distance2 <= radius*radius

    @wp.func
    def contains(query: Vec, radius: float, point: Vec):
        delta = point-query
        return wp.dot(delta, delta) <= radius*radius

    return RegionOps(Vec, (dimension,), valid, overlaps, contains)


@dataclass(frozen=True)
class BoundarySampleQuery(BoundQuery, Generic[H, D, ItemT, P, Q]):
    region: object

    def __call__(self, data, metadata, queries, radii, randoms):
        tree, geometry = data
        queries, caps, shape = prepare_nearest(self.region, queries, radii)
        randoms = jnp.asarray(randoms, jnp.float32)
        if randoms.shape[-1:] != (3,):
            raise ValueError("randoms must have trailing shape (3,)")
        randoms = jnp.broadcast_to(randoms, shape+(3,)).reshape(-1, 3)
        return self.launch(tree, geometry, shape, queries, caps, randoms, metadata=(metadata,))


def bind(spec, measure=None, region=None, *, block_dim=256):
    primitive = spec.primitive
    measure = _adapter_policy(primitive, "measure") if measure is None else measure
    region = _ball_region(primitive) if region is None else region
    return _make_boundary_sample(primitive, measure, region, block_dim)


@cache_query
def _make_boundary_sample(primitive, measure, region, block_dim):
    if not 1 <= block_dim <= 1024:
        raise ValueError("invalid block dimension")
    _check_functions(measure.sample, region.valid, region.overlaps, region.contains)
    Vec, Query, Geometry = primitive.point_type, region.query_type, primitive.data_type

    @wp.func
    def sample(query: Query, radius: float, randoms: wp.vec3, tree: DeviceTree,
               geometry: Geometry, meta: SamplingMeta):
        point, normal, density, index, status = Vec(0.0), Vec(0.0), float(0.0), int(-1), int(0)
        if (not wp.static(region.valid)(query, radius) or not wp.isfinite(randoms)
                or wp.min(randoms) < 0.0 or wp.max(randoms) >= 1.0):
            return point, normal, density, index, -1
        if tree.node_left.shape[0] == 0 or meta.node_weight[0] <= 0.0:
            return point, normal, density, index, status
        if not wp.static(region.overlaps)(query, radius, tree.node_lower[0], tree.node_upper[0]):
            return point, normal, density, index, status
        node, u, probability = int(0), float(randoms[0]), float(1.0)
        while tree.node_leaf_count[node] == 0:
            left, right = tree.node_left[node]+0, tree.node_right[node]+0
            a, b = float(0.0), float(0.0)
            if wp.static(region.overlaps)(query, radius, tree.node_lower[left], tree.node_upper[left]):
                a = meta.node_weight[left]
            if wp.static(region.overlaps)(query, radius, tree.node_lower[right], tree.node_upper[right]):
                b = meta.node_weight[right]
            if a+b <= 0.0:
                return point, normal, density, index, status
            threshold = a/(a+b)
            if b <= 0.0 or (a > 0.0 and u < threshold):
                probability *= threshold
                u /= threshold
                node = left
            else:
                probability *= 1.0-threshold
                u = (u-threshold)/(1.0-threshold)
                node = right
            u = wp.clamp(u, 0.0, 0.99999994)
        offset, count = tree.node_leaf_offset[node], tree.node_leaf_count[node]
        total, selected = float(0.0), int(-1)
        for j in range(count):
            slot = offset+j
            item = wp.static(primitive.load)(geometry, slot)
            lower, upper = wp.static(primitive.bounds_of_item)(item)
            if meta.primitive_weight[slot] > 0.0 and wp.static(region.overlaps)(query, radius, lower, upper):
                total += meta.primitive_weight[slot]
                selected = slot
        if total <= 0.0:
            return point, normal, density, index, status
        threshold, accumulated = u*total, float(0.0)
        for j in range(count):
            slot = offset+j
            item = wp.static(primitive.load)(geometry, slot)
            lower, upper = wp.static(primitive.bounds_of_item)(item)
            if meta.primitive_weight[slot] > 0.0 and wp.static(region.overlaps)(query, radius, lower, upper):
                accumulated += meta.primitive_weight[slot]
                if threshold < accumulated:
                    selected = slot
                    break
        candidate, candidate_normal, local_pdf = wp.static(measure.sample)(
            wp.static(primitive.load)(geometry, selected), wp.vec2(randoms[1], randoms[2]))
        if not wp.isfinite(candidate) or not wp.isfinite(candidate_normal) or not wp.isfinite(local_pdf) or local_pdf < 0.0:
            return point, normal, density, index, -1
        if local_pdf == 0.0:
            return point, normal, density, index, status
        if not wp.static(region.contains)(query, radius, candidate):
            return point, normal, density, index, status
        density = probability*(meta.primitive_weight[selected]/total)*local_pdf
        if density <= 0.0:
            return point, normal, float(0.0), index, -1
        return candidate, candidate_normal, density, tree.prim_ids[selected], 1

    inputs = [("points", wp.array(dtype=Query)), ("radii", wp.array(dtype=wp.float32)),
              ("randoms", wp.array(dtype=wp.vec3))]
    outputs = [("out_point", wp.array(dtype=Vec)), ("out_normal", wp.array(dtype=Vec)),
               ("out_pdf", wp.array(dtype=wp.float32)), ("out_id", wp.array(dtype=wp.int32)),
               ("out_status", wp.array(dtype=wp.int32))]
    body = ["p, n, pdf, index, status = wp.static(sample)(load_array(points, i), load_array(radii, i), load_array(randoms, i), tree, geometry, meta)",
            "store_array(out_point, i, p)", "store_array(out_normal, i, n)", "store_array(out_pdf, i, pdf)", "store_array(out_id, i, index)", "store_array(out_status, i, status)"]
    return bind_query(BoundarySampleQuery, "boundary_sample_kernel", primitive, inputs, outputs, body,
        {"sample": sample}, block_dim, device=sample, groups=(("meta", SamplingMeta),), region=region,
        native_providers=(measure, region),
        components=(("selection", "subtree_weight"), ("region", type(region).__name__),
                    ("measure", type(measure).__name__)))

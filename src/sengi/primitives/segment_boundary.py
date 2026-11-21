"""Indexed segment boundary operations."""
from functools import lru_cache
from typing import Any

import numpy as np
import warp as wp

from sengi.boundary.data import BoundaryAdapter, BoundaryCore, SilhouetteMeta, cone_visibility
from sengi.boundary.indexed import closed, feature_primitive, indexed_core, indexed_facets, indexed_silhouette, pack_core, robin_bounds
from sengi.boundary.robin import CoefficientMeta
from sengi.boundary.sampling import MeasureOps
from sengi.bvh.device_tree import DeviceTree
from sengi.metrics.nearest import Silhouette
from sengi.queries.domain import ClassificationOps
from sengi.queries.robin_radius import RobinOps
from sengi.warp.native import package_headers


def prepare(primitive, data):
    facets = indexed_facets(data, len(data.a), 2, 2).astype(np.float64)
    edge = facets[:, 1] - facets[:, 0]
    cross = np.column_stack((edge[:, 1], -edge[:, 0]))
    length = np.linalg.norm(cross, axis=1)
    normal = cross / np.where(length[:, None] > 0, length[:, None], 1.)
    return indexed_core(np.pad(data.vertices, ((0, 0), (0, 1))), data.indices,
                        np.pad(normal, ((0, 0), (0, 1))), length, np.asarray(data.indices).reshape(-1, 1))


def vertex_keys(face):
    return ((v,) for v in face)


def silhouette(core, tree, geometry):
    return indexed_silhouette(core, tree, vertex_keys)


@lru_cache(maxsize=None)
def classification(primitive):
    Geometry = primitive.data_type

    @wp.func_native("return bvh::classify_segment(point, meta, slot);")
    def event(point: wp.vec2, geometry: Geometry, slot: int, original: int, meta: BoundaryCore) -> int: ...

    headers = package_headers("primitives", "segment_classification.h",
                              dependencies=("primitives/native/determinant.h",))
    return ClassificationOps(BoundaryCore, closed, event, native=headers)


@lru_cache(maxsize=None)
def measure(primitive):
    @wp.func
    def sample(item: Any, uniforms: wp.vec2):
        edge = item.b-item.a
        length = wp.length(edge)
        normal = wp.vec2(edge[1], -edge[0])/length
        return item.a+uniforms[0]*edge, normal, 1.0/length
    return MeasureOps(sample)


_ROBIN_HEADERS = package_headers(
    "boundary", "silhouette.h",
    dependencies=("primitives/native/segment_distance.h",),
)


@lru_cache(maxsize=None)
def robin(primitive):
    Geometry = primitive.data_type
    nearest, valid, node_bound, feature_bound = robin_bounds(primitive)

    @wp.func
    def bound_with_data(query: wp.vec2, tree: DeviceTree, geometry: Geometry, core: BoundaryCore,
                        feature: SilhouetteMeta, meta: CoefficientMeta, node: int):
        lower, upper = tree.node_lower[node], tree.node_upper[node]
        box2 = node_bound(query, lower, upper, meta.node_bounds[node])
        point = wp.vec3(query[0], query[1], 0.0)
        coefficient = meta.node_bounds[node][1]
        if coefficient == 0.0:
            return wp.where(cone_visibility(point, tree, feature, node), box2, wp.inf)
        angles = feature.node_cone_sin_cos[node]
        if angles[0] < 0.0 or angles[1] <= 0.0 or wp.isinf(coefficient):
            return box2
        view = .5*(lower+upper)-point
        axis = feature.node_cone_axis[node]
        projection = wp.dot(axis, view)
        extent = .5*(upper-lower)
        gap = wp.abs(projection)-wp.dot(wp.abs(axis), extent)
        if gap <= 0.0:
            return box2
        radius = feature.node_cone_radius[node]
        h = gap*angles[1]-angles[0]*(wp.length(view-projection*axis)+radius)-4e-6*(wp.length(view)+radius)
        if not (h > 0.0 and wp.isfinite(h)):
            return box2
        far = wp.max(wp.abs(lower-point), wp.abs(upper-point))
        distance2 = wp.clamp(2.0*(h/coefficient), box2, wp.dot(far, far))
        bound = distance2*wp.exp(2.0*h/(coefficient*distance2))*0.999996
        return wp.max(box2, bound)

    @wp.func
    def primitive_bound(query: wp.vec2, item: Any, coefficient: float):
        if coefficient == 0.0:
            return wp.inf
        closest, near2 = wp.static(nearest.closest)(query, item)
        if not wp.isfinite(near2):
            return wp.nan
        if wp.isinf(coefficient) or near2 == 0.0:
            return near2
        edge = item.b-item.a
        normal = wp.vec2(edge[1], -edge[0])/wp.length(edge)
        height = wp.abs(wp.dot(query-item.a, normal))
        if height == 0.0:
            return near2
        a, b = query-item.a, query-item.b
        far2 = wp.max(wp.dot(a, a), wp.dot(b, b))
        distance2 = wp.clamp(2.0*(height/coefficient), near2, far2)
        bound = distance2*wp.exp(2.0*height/(coefficient*distance2))*0.999996
        return wp.max(near2, bound)

    return RobinOps(wp.vec2, (2,), valid, node_bound, primitive_bound, feature_bound,
                    node_bound_with_data=bound_with_data,
                    to_score=nearest.to_score, to_distance=nearest.to_distance, native=_ROBIN_HEADERS)


@lru_cache(maxsize=None)
def boundary_adapter():
    return BoundaryAdapter(prepare, pack_core, feature_primitive(wp.vec2, edge_feature=False),
                           silhouette, cone_visibility, classification, Silhouette(), measure, robin)

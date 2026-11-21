"""Indexed triangle boundary operations."""
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


def triangle_metadata(facets):
    """Measure indexed triangles in float64 before packing boundary metadata."""
    a, b, c = np.asarray(facets, np.float64).transpose(1, 0, 2)
    ab, bc, ca = b - a, c - b, a - c
    # Include the shortest edge to preserve normals when two vertices are close.
    ab2, bc2, ca2 = (np.einsum("ij,ij->i", edge, edge) for edge in (ab, bc, ca))
    use_ca = (ca2 < ab2) & (ca2 < bc2)
    cross = np.cross(ab, bc)
    cross[use_ca] = np.cross(bc[use_ca], ca[use_ca])
    lengths = np.linalg.norm(cross, axis=1)
    area = 0.5 * lengths
    normal = cross / np.where(lengths[:, None] > 0, lengths[:, None], 1.)
    return normal, area, lengths


def triangle_boundary_features(indices):
    """Undirected triangle edges, including repeated edges of degeneracies."""
    return np.sort(indices[:, [[0, 1], [1, 2], [2, 0]]].reshape(-1, 2), axis=1)


def prepare(primitive, data):
    normal, area, _ = triangle_metadata(indexed_facets(data, len(data.a), 3, 3))
    return indexed_core(data.vertices, data.indices, normal, area, triangle_boundary_features(np.asarray(data.indices)))


def edge_keys(face):
    return (tuple(sorted((face[i], face[(i+1) % 3]))) for i in range(3))


def silhouette(core, tree, geometry):
    return indexed_silhouette(core, tree, edge_keys)


@lru_cache(maxsize=None)
def classification(primitive):
    Geometry = primitive.data_type

    @wp.func_native("return bvh::classify_triangle(point, meta, slot);")
    def event(point: wp.vec3, geometry: Geometry, slot: int, original: int, meta: BoundaryCore) -> int: ...

    headers = package_headers("primitives", "triangle_classification.h",
                              dependencies=("primitives/native/triangle_crossing.h", "primitives/native/determinant.h",))
    return ClassificationOps(BoundaryCore, closed, event, native=headers)


@lru_cache(maxsize=None)
def measure(primitive):
    @wp.func
    def sample(item: Any, uniforms: wp.vec2):
        a, b, c = wp.vec3d(item.a), wp.vec3d(item.b), wp.vec3d(item.c)
        ab, bc, ca = b - a, c - b, a - c
        normal = wp.cross(ab, bc)
        ca2 = wp.dot(ca, ca)
        if ca2 < wp.dot(ab, ab) and ca2 < wp.dot(bc, bc):
            normal = wp.cross(bc, ca)
        twice_area = wp.length(normal)
        root, fraction = wp.sqrt(wp.float64(uniforms[0])), wp.float64(uniforms[1])
        one = wp.float64(1.0)
        point = (one - root) * a + root * ((one - fraction) * b + fraction * c)
        return wp.vec3(point), wp.vec3(normal / twice_area), wp.float32(wp.float64(2.0) / twice_area)
    return MeasureOps(sample)


_ROBIN_HEADERS = package_headers(
    "primitives", "triangle_robin.h", "silhouette.h",
    include_owners=("boundary",),
    dependencies=("primitives/native/triangle_distance.h", "primitives/native/determinant.h",
                  "metrics/native/box_distance.h", "metrics/native/rounding.h",
                  "primitives/native/segment_distance.h"),
)


@lru_cache(maxsize=None)
def robin(primitive):
    Geometry = primitive.data_type
    nearest, valid, node_bound, feature_bound = robin_bounds(primitive)

    @wp.func_native("return bvh::robin_leaf(query, geometry, core, coefficients, offset, count, best);")
    def leaf_with_data(query: wp.vec3, geometry: Geometry, core: BoundaryCore,
                       coefficients: CoefficientMeta, offset: int, count: int, best: float) -> float: ...

    @wp.func_native("bvh::robin_node(query, feature, meta, node, lower, upper, best, score, bound, skip, accepted);")
    def visit_values(query: wp.vec3, tree: DeviceTree, geometry: Geometry, core: BoundaryCore,
                     feature: SilhouetteMeta, meta: CoefficientMeta, node: int,
                     lower: wp.vec3, upper: wp.vec3, best: float, score: wp.ref[float], bound: wp.ref[float],
                     skip: wp.ref[bool], accepted: wp.ref[bool]): ...

    @wp.func
    def visit(query: wp.vec3, tree: DeviceTree, geometry: Geometry, core: BoundaryCore,
              feature: SilhouetteMeta, meta: CoefficientMeta, node: int,
              lower: wp.vec3, upper: wp.vec3, best: float):
        score, bound, skip, accepted = float(0.0), float(best), bool(False), bool(False)
        visit_values(query, tree, geometry, core, feature, meta, node, lower, upper, best, score, bound, skip, accepted)
        return score, bound, skip, accepted
    return RobinOps(wp.vec3, (3,), valid, node_bound, None, feature_bound,
                    leaf_bound_with_data=leaf_with_data, node_visit=visit,
                    to_score=nearest.to_score, to_distance=nearest.to_distance, native=_ROBIN_HEADERS)


@lru_cache(maxsize=None)
def boundary_adapter():
    return BoundaryAdapter(prepare, pack_core, feature_primitive(wp.vec3, edge_feature=True),
                           silhouette, cone_visibility, classification, Silhouette(), measure, robin)

"""Shared operations for indexed boundaries with vertex or edge silhouettes."""
from dataclasses import dataclass
from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import warp as wp

from sengi.boundary.data import BoundaryCore, BoundaryCoreData, SilhouetteData, SilhouetteMeta
from sengi.warp.arrays import load_array
from sengi.warp.native import NativeHeaders, package_headers


def indexed_facets(data, count, dimension, arity):
    vertices, indices = np.asarray(data.vertices), np.asarray(data.indices)
    if count and not len(indices):
        raise ValueError("boundary queries require source geometry with indexed connectivity")
    if vertices.shape[1:] != (dimension,) or indices.shape != (count, arity):
        raise ValueError("indexed boundary arrays have incompatible shapes")
    return vertices[indices]


def indexed_core(vertices, indices, normal, measure, features):
    if (not np.isfinite(normal).all() or not np.isfinite(measure).all()
            or np.any(measure > np.finfo(np.float32).max)):
        raise ValueError("boundary normals and measures must be finite float32")
    _, counts = np.unique(features, axis=0, return_counts=True)
    closed = int(np.all(counts == 2) and np.all(measure > 0))
    return BoundaryCoreData(vertices, indices, np.array([closed], np.int32),
                            normal.astype(np.float32), measure.astype(np.float32))


@wp.func
def closed(meta: BoundaryCore):
    if meta.closed.shape[0] != 1:
        return False
    return meta.closed[0] != 0


def indexed_silhouette(core, tree, feature_keys):
    vertices, indices, normals = (np.asarray(v) for v in
        (core.vertices, core.indices, core.primitive_normal))
    if np.any(np.asarray(core.primitive_measure) <= 0):
        raise ValueError("silhouettes require non-degenerate boundary elements")
    adjacent = {}
    for primitive, face in enumerate(indices):
        for key in feature_keys(face):
            adjacent.setdefault(key, []).append(primitive)
    features, owners = [], {}
    for key, incident in adjacent.items():
        if len(incident) > 2:
            raise ValueError("silhouettes require manifold boundary connectivity")
        index = len(features)
        n0 = normals[incident[0]]
        n1 = normals[incident[1]] if len(incident) == 2 else -n0
        features.append((vertices[key[0]], vertices[key[-1]], n0, n1))
        for primitive in incident:
            owners.setdefault(primitive, []).append(index)
    values = np.asarray(features, np.float32).reshape(-1, 4, 3)
    left, right, offset, count, ids = map(np.asarray,
        (tree.node_left, tree.node_right, tree.node_leaf_offset, tree.node_leaf_count, tree.prim_ids))
    n = len(left)
    node_offset, node_count = np.zeros(n, np.int32), np.zeros(n, np.int32)
    axis, angles = np.zeros((n, 3), np.float32), np.tile(np.array([-1., 0.], np.float32), (n, 1))
    radii = np.zeros(n, np.float32)
    references, contained, post = [], [set() for _ in range(n)], []
    pending = [(0, False)] if n else []
    while pending:
        node, done = pending.pop()
        if done:
            post.append(node)
        else:
            pending.append((node, True))
            if left[node] >= 0:
                pending.extend(((int(left[node]), False), (int(right[node]), False)))
    lo, hi = np.asarray(tree.node_lower, np.float64), np.asarray(tree.node_upper, np.float64)
    for node in post:
        if left[node] < 0:
            contained[node] = {feature for primitive in ids[offset[node]:offset[node]+count[node]]
                               for feature in owners.get(int(primitive), ())}
            node_offset[node], node_count[node] = len(references), len(contained[node])
            references.extend(sorted(contained[node]))
        else:
            contained[node] = contained[left[node]] | contained[right[node]]
        if not contained[node]:
            continue
        ns = values[sorted(contained[node]), 2:].reshape(-1, 3).astype(np.float64)
        average = ns.sum(axis=0)
        length = np.linalg.norm(average)
        angles[node] = [1., 0.]
        if length > 1e-12:
            direction = (average/length).astype(np.float32)
            alpha = np.arctan2(np.linalg.norm(np.cross(ns, direction), axis=1), ns @ direction).max()
            alpha = np.nextafter(np.float32(alpha), np.float32(np.inf))
            axis[node] = direction
            if alpha < np.pi/2:
                angles[node] = (np.nextafter(np.float32(np.sin(alpha)), np.float32(np.inf)),
                                np.nextafter(np.float32(np.cos(alpha)), np.float32(-np.inf)))
        radii[node] = np.nextafter(np.float32(.5*np.linalg.norm(hi[node]-lo[node])), np.float32(np.inf))
    arrays = (*(values[:, i] for i in range(4)), np.asarray(references, np.int32),
              node_offset, node_count, axis, angles, radii)
    return SilhouetteData(*(jnp.asarray(v) for v in arrays))


_SILHOUETTE_HEADERS = package_headers(
    "boundary", "silhouette.h",
    dependencies=("primitives/native/segment_distance.h",),
)


@dataclass(frozen=True)
class _FeaturePrimitive:
    dimension: int
    point_type: object
    data_type: object
    load: object
    bounds_of_item: object
    closest_euclidean: object
    segment_endpoints: object
    normal_sides: object
    native: NativeHeaders = _SILHOUETTE_HEADERS


@lru_cache(maxsize=None)
def feature_primitive(Vec, edge_feature):
    from sengi.warp.contracts import make_struct
    dimension = len(Vec())
    Feature = make_struct(f"SilhouetteFeature{dimension}", {"index": wp.int32, "data": SilhouetteMeta})

    @wp.func
    def load(data: SilhouetteMeta, slot: int):
        value = Feature()
        value.index = load_array(data.feature_refs, slot)
        value.data = data
        return value

    @wp.func
    def coordinates(p: wp.vec3):
        if wp.static(dimension == 2):
            return wp.vec2(p[0], p[1])
        else:
            return p

    @wp.func
    def endpoints(item: Feature):
        a = coordinates(load_array(item.data.feature_a, item.index))
        if wp.static(edge_feature):
            return a, coordinates(load_array(item.data.feature_b, item.index))
        else:
            return a, a

    @wp.func
    def bounds(item: Feature):
        a = load_array(item.data.feature_a, item.index)
        b = a
        if wp.static(edge_feature):
            b = load_array(item.data.feature_b, item.index)
        return wp.min(a, b), wp.max(a, b)

    if edge_feature:
        @wp.func_native("return bvh::closest_segment(query, a, b);")
        def project(query: Vec, a: Vec, b: Vec) -> Vec: ...

        @wp.func
        def closest(query: Vec, item: Feature):
            a, b = endpoints(item)
            return project(query, a, b)
    else:
        @wp.func
        def closest(query: Vec, item: Feature):
            point, _ = endpoints(item)
            return point

    @wp.func
    def sides(item: Feature):
        return (load_array(item.data.feature_a, item.index), load_array(item.data.feature_n0, item.index),
                load_array(item.data.feature_n1, item.index))

    return _FeaturePrimitive(dimension, Vec, SilhouetteMeta, load, bounds, closest, endpoints, sides)


def pack_core(core, data, tree, geometry):
    ids = tree.prim_ids
    # Reuse packed connectivity only when the adapter retained the source array.
    shared = (core.indices is getattr(data, "indices", None)
              and hasattr(geometry, "indices"))
    indices = geometry.indices if shared else jnp.asarray(core.indices)[ids]
    vertices = geometry.vertices if core.vertices is data.vertices else jnp.asarray(core.vertices)
    return BoundaryCoreData(vertices, indices, jnp.asarray(core.closed),
                            jnp.asarray(core.primitive_normal)[ids],
                            jnp.asarray(core.primitive_measure)[ids])



@lru_cache(maxsize=None)
def robin_bounds(primitive):
    from sengi.metrics.nearest import Euclidean
    Vec, dimension = primitive.point_type, primitive.dimension
    nearest = Euclidean().bind(primitive)

    @wp.func
    def valid(query: Vec):
        return wp.isfinite(query)

    @wp.func
    def node_bound(query: Vec, lower: wp.vec3, upper: wp.vec3, coefficients: wp.vec2):
        return wp.static(nearest.node_bound)(query, lower, upper)

    @wp.func_native("return bvh::closest_segment(query, a, b);")
    def project(query: wp.vec3, a: wp.vec3, b: wp.vec3) -> wp.vec3: ...

    @wp.func
    def feature_bound(query: Vec, a: wp.vec3, b: wp.vec3, n0: wp.vec3, n1: wp.vec3):
        point = wp.vec3(0.0)
        for j in range(wp.static(dimension)):
            point[j] = query[j]
        view = point-a
        side0, side1 = wp.dot(view, n0), wp.dot(view, n1)
        if (side0 < 0.0 and side1 < 0.0) or (side0 > 0.0 and side1 > 0.0):
            return wp.inf
        delta = point - project(point, a, b)
        return wp.dot(delta, delta)

    return nearest, valid, node_bound, feature_bound

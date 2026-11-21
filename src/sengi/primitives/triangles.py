"""Triangle primitive specification and distance operations."""
from functools import lru_cache
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from sengi.warp.native import package_headers
from sengi.primitives.crossings import native_triangle_crossing
from sengi.primitives.data import TrianglesArrays3, TrianglesData3
from sengi.primitives.device import item_functions
from sengi.primitives.items import TriangleItem3
from sengi.primitives.packing import _indexed_data
from sengi.primitives.spec import IndexedBoundaryPrimitive
from sengi.queries.domain import Ray
from sengi.warp.contracts import FusedNearestOps, as_warp_function

if TYPE_CHECKING:
    from sengi.boundary.data import BoundaryCore, BoundaryCoreData, SilhouetteData, SilhouetteMeta
    from sengi.primitives.items import SilhouetteItem3


_NATIVE = package_headers(
    "primitives", "triangle_nearest.h", "triangle_crossing.h",
    dependencies=("primitives/native/triangle_distance.h", "primitives/native/determinant.h",
                  "metrics/native/box_distance.h", "metrics/native/rounding.h"),
)


def _triangle_bounds(data):
    facets = np.asarray(data.vertices)[np.asarray(data.indices)]
    return facets.min(axis=1), facets.max(axis=1)


def _triangle_pack(data, ids):
    return type(data)(*(a[ids] for a in data[:3]), data.vertices, data.indices[ids],
                      data.b[ids], data.c[ids], data.frame[ids])


@lru_cache(maxsize=None)
def Triangles() -> "IndexedBoundaryPrimitive[TrianglesArrays3, TrianglesData3, TriangleItem3, wp.vec3, BoundaryCoreData, BoundaryCore, SilhouetteData, SilhouetteMeta, SilhouetteItem3]":
    from sengi.primitives.triangle_boundary import boundary_adapter as triangle_boundary_adapter

    dimension = 3
    Vec, Data, Payload, Item = wp.vec3, TrianglesData3, TrianglesArrays3, TriangleItem3
    from sengi.queries.region import RegionIntersectionPolicy
    from sengi.primitives.region import triangle3_intersects

    functions = item_functions("triangles", dimension, Data, Item)
    load, item_bounds = functions.load, functions.bounds
    host_bounds, host_pack = _triangle_bounds, _triangle_pack

    @wp.func_native("return bvh::triangle_score(query, data, slot);")
    def score_at(query: Vec, data: Data, slot: int) -> wp.float64: ...

    @wp.func_native("return bvh::triangle_closest(query, data, slot);")
    def closest_at(query: Vec, data: Data, slot: int) -> Vec: ...

    @wp.func_native("return bvh::closest_triangle(query, item.a, item.b, item.c);")
    def closest(query: Vec, item: Item) -> Vec: ...

    @wp.func_native("return bvh::triangle_nearest_at(query, data, slot, limit);")
    def nearest_at(query: Vec, data: Data, slot: int, limit: float) -> wp.vec4: ...

    @wp.func_native("return bvh::triangle_nearest_candidate(query, item.a, item.b, item.c, limit);")
    def nearest_item(query: Vec, item: Item, limit: float) -> wp.vec4: ...

    @wp.func
    def candidate_at(query: Vec, data: Data, slot: int, limit: float):
        evaluated = nearest_at(query, data, slot, limit)
        return wp.vec3(evaluated[0], evaluated[1], evaluated[2]), evaluated[3]

    @wp.func
    def candidate_item(query: Vec, item: Item, limit: float):
        evaluated = nearest_item(query, item, limit)
        return wp.vec3(evaluated[0], evaluated[1], evaluated[2]), evaluated[3]

    @wp.func_native("return bvh::conservative_box_score<true>(query, lower, upper);")
    def nearest_bound(query: Vec, lower: wp.vec3, upper: wp.vec3) -> float: ...

    fused = FusedNearestOps(as_warp_function(candidate_at),
                            as_warp_function(candidate_item),
                            as_warp_function(nearest_bound))

    def bind_fused_euclidean(primitive):
        # The slot evaluator reads the original triangle layout directly.
        # Copied descriptors may supply a different geometry operation.
        if (primitive.dimension == dimension and primitive.point_type is Vec
                and primitive.data_type is Data and primitive.item_type is Item
                and primitive.load is load and primitive.bounds_of_item is item_bounds
                and primitive.closest_euclidean is closest
                and primitive.score_euclidean_at is score_at
                and primitive.closest_euclidean_at is closest_at
                and primitive.host_bounds is host_bounds
                and primitive.host_pack is host_pack):
            return fused
        return None

    @wp.func
    def intersect(ray: Ray, minimum: float, maximum: float,
                  data: Data, slot: int, half_open: bool, store_times: bool):
        return native_triangle_crossing(ray, minimum, maximum, data.a[slot],
                                        data.b[slot], data.c[slot], half_open, store_times)

    prepare_indexed = _indexed_data(Payload, dimension, 3)

    return IndexedBoundaryPrimitive(
        dimension=dimension, point_type=Vec, data_type=Data, item_type=Item,
        region_policy=RegionIntersectionPolicy(as_warp_function(triangle3_intersects)),
        load=load, bounds_of_item=item_bounds, host_bounds=host_bounds, host_pack=host_pack,
        closest_euclidean=as_warp_function(closest),
        score_euclidean_at=as_warp_function(score_at),
        closest_euclidean_at=as_warp_function(closest_at),
        intersect=as_warp_function(intersect),
        group_leaves=True, cache_nearest_bounds=False, name="triangles", payload_type=Payload,
        prepare_indexed=prepare_indexed,
        boundary_adapter=triangle_boundary_adapter(), bind_fused_euclidean=bind_fused_euclidean, native=_NATIVE,
    )

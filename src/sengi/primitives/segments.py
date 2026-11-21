"""Segment primitive specifications."""
from functools import lru_cache
from typing import TYPE_CHECKING, Any, Literal, overload

import warp as wp

from sengi.warp.native import package_headers
from sengi.primitives.data import SegmentsArrays2, SegmentsArrays3, SegmentsData2, SegmentsData3
from sengi.primitives.device import item_functions
from sengi.primitives.items import SegmentItem2, SegmentItem3
from sengi.primitives.packing import _array_bounds, _array_data, _indexed_data, _segment_pack
from sengi.primitives.spec import ArrayBoundaryPrimitive, ArrayPrimitive
from sengi.queries.domain import Ray
from sengi.warp.contracts import as_warp_function

if TYPE_CHECKING:
    from sengi.boundary.data import BoundaryCore, BoundaryCoreData, SilhouetteData, SilhouetteMeta
    from sengi.primitives.items import SilhouetteItem3


@overload
def Segments(dimension: Literal[2]) -> "ArrayBoundaryPrimitive[SegmentsArrays2, SegmentsData2, SegmentItem2, wp.vec2, BoundaryCoreData, BoundaryCore, SilhouetteData, SilhouetteMeta, SilhouetteItem3]": ...
@overload
def Segments(dimension: Literal[3] = 3) -> ArrayPrimitive[SegmentsArrays3, SegmentsData3, SegmentItem3, wp.vec3]: ...
@overload
def Segments(dimension: int) -> ArrayPrimitive[Any, Any, Any, Any]: ...
def Segments(dimension: int = 3) -> ArrayPrimitive[Any, Any, Any, Any]:
    return _segments(dimension)


@lru_cache(maxsize=None)
def _segments(dimension: int) -> ArrayPrimitive[Any, Any, Any, Any]:
    if dimension not in (2, 3):
        raise ValueError("segments support 2D or 3D")
    Vec = wp.vec2 if dimension == 2 else wp.vec3
    Data, Payload, Item = (
        (SegmentsData2, SegmentsArrays2, SegmentItem2) if dimension == 2
        else (SegmentsData3, SegmentsArrays3, SegmentItem3)
    )
    from sengi.queries.region import RegionIntersectionPolicy
    from sengi.primitives.region import segment2_intersects, segment3_intersects

    functions = item_functions("segments", dimension, Data, Item)

    @wp.func
    def endpoints(item: Item):
        return item.a, item.b

    @wp.func_native("return bvh::closest_segment(query, item.a, item.b);")
    def closest(query: Vec, item: Item) -> Vec: ...

    intersect = boundary_adapter = None
    if dimension == 2:
        from sengi.primitives.segment_boundary import boundary_adapter as segment_boundary_adapter

        @wp.func_native("return bvh::segment_intersection(ray, minimum, maximum, data, slot, half_open, store_times);")
        def intersect(ray: Ray, minimum: float, maximum: float,
                      data: Data, slot: int, half_open: bool, store_times: bool) -> wp.vec2: ...

        boundary_adapter = segment_boundary_adapter()

    primitive_type = ArrayBoundaryPrimitive if boundary_adapter is not None else ArrayPrimitive
    boundary = {} if boundary_adapter is None else {"boundary_adapter": boundary_adapter}
    return primitive_type(
        dimension=dimension, point_type=Vec, data_type=Data, item_type=Item,
        region_policy=RegionIntersectionPolicy(as_warp_function(segment2_intersects if dimension == 2 else segment3_intersects)),
        load=functions.load, bounds_of_item=functions.bounds,
        host_bounds=_array_bounds, host_pack=_segment_pack,
        closest_euclidean=as_warp_function(closest), segment_endpoints=as_warp_function(endpoints),
        intersect=as_warp_function(intersect) if intersect is not None else None,
        name="segments", payload_type=Payload,
        prepare_data=_array_data(Payload, dimension, 2, "segments", segment=True),
        prepare_indexed=_indexed_data(Payload, dimension, 2),
        native=package_headers("primitives", "segment_distance.h",
                               *(("segment_crossing.h",) if dimension == 2 else ())),
        **boundary,
    )

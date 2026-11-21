"""Segment primitive specifications."""
from functools import lru_cache
from typing import Any, Literal, overload

import warp as wp

from sengi.primitives.data import SegmentsArrays2, SegmentsArrays3, SegmentsData2, SegmentsData3
from sengi.primitives.device import item_functions
from sengi.primitives.items import SegmentItem2, SegmentItem3
from sengi.primitives.packing import _array_bounds, _array_data, _indexed_data, _segment_pack
from sengi.primitives.spec import Primitive
from sengi.queries.domain import Ray
from sengi.warp.native import package_headers
from sengi.warp.contracts import as_warp_function


@overload
def Segments(dimension: Literal[2]) -> Primitive[SegmentsArrays2, SegmentsData2, SegmentItem2, wp.vec2]: ...
@overload
def Segments(dimension: Literal[3] = 3) -> Primitive[SegmentsArrays3, SegmentsData3, SegmentItem3, wp.vec3]: ...
@overload
def Segments(dimension: int) -> Primitive[Any, Any, Any, Any]: ...
def Segments(dimension: int = 3) -> Primitive[Any, Any, Any, Any]:
    return _segments(dimension)


@lru_cache(maxsize=None)
def _segments(dimension: int) -> Primitive[Any, Any, Any, Any]:
    if dimension not in (2, 3):
        raise ValueError("segments support 2D or 3D")
    Vec = wp.vec2 if dimension == 2 else wp.vec3
    Data, Payload, Item = (
        (SegmentsData2, SegmentsArrays2, SegmentItem2) if dimension == 2
        else (SegmentsData3, SegmentsArrays3, SegmentItem3)
    )
    functions = item_functions("segments", dimension, Data, Item)

    @wp.func
    def endpoints(item: Item):
        return item.a, item.b

    @wp.func_native("return bvh::closest_segment(query, item.a, item.b);")
    def closest(query: Vec, item: Item) -> Vec: ...

    intersect = boundary_adapter = None
    if dimension == 2:
        from sengi.boundary.data import _builtin_adapter

        @wp.func_native("return bvh::segment_intersection(ray, minimum, maximum, data, slot, half_open, store_times);")
        def intersect(ray: Ray, minimum: float, maximum: float,
                      data: Data, slot: int, half_open: bool, store_times: bool) -> wp.vec2: ...

        boundary_adapter = _builtin_adapter(2)

    return Primitive(
        dimension=dimension, point_type=Vec, data_type=Data, item_type=Item,
        load=functions.load, bounds_of_item=functions.bounds,
        host_bounds=_array_bounds, host_pack=_segment_pack,
        closest_euclidean=as_warp_function(closest), segment_endpoints=as_warp_function(endpoints),
        intersect=as_warp_function(intersect) if intersect is not None else None,
        name="segments", payload_type=Payload,
        prepare_data=_array_data(Payload, dimension, 2, "segments", segment=True),
        prepare_indexed=_indexed_data(Payload, dimension, 2),
        boundary_adapter=boundary_adapter,
        native=package_headers("primitives", "segment_distance.h",
                               *(("segment_crossing.h",) if dimension == 2 else ())),
    )

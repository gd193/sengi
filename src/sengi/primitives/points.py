"""Point primitive specifications."""
from functools import lru_cache
from typing import Any, Literal, overload

import warp as wp

from sengi.primitives.data import PointsArrays2, PointsArrays3, PointsData2, PointsData3
from sengi.primitives.device import item_functions
from sengi.primitives.items import PointItem2, PointItem3
from sengi.primitives.packing import _array_bounds, _array_data, _array_pack
from sengi.primitives.spec import ArrayPrimitive
from sengi.warp.contracts import as_warp_function


@overload
def Points(dimension: Literal[2]) -> ArrayPrimitive[PointsArrays2, PointsData2, PointItem2, wp.vec2]: ...
@overload
def Points(dimension: Literal[3] = 3) -> ArrayPrimitive[PointsArrays3, PointsData3, PointItem3, wp.vec3]: ...
@overload
def Points(dimension: int) -> ArrayPrimitive[Any, Any, Any, Any]: ...
def Points(dimension: int = 3) -> ArrayPrimitive[Any, Any, Any, Any]:
    return _points(dimension)


@lru_cache(maxsize=None)
def _points(dimension: int) -> ArrayPrimitive[Any, Any, Any, Any]:
    if dimension not in (2, 3):
        raise ValueError("points support 2D or 3D")
    Vec = wp.vec2 if dimension == 2 else wp.vec3
    Data, Payload, Item = (
        (PointsData2, PointsArrays2, PointItem2) if dimension == 2
        else (PointsData3, PointsArrays3, PointItem3)
    )
    functions = item_functions("points", dimension, Data, Item)

    @wp.func
    def endpoints(item: Any):
        return item.a, item.a

    @wp.func
    def closest(query: Vec, item: Any):
        return item.a

    return ArrayPrimitive(
        dimension=dimension, point_type=Vec, data_type=Data, item_type=Item,
        load=functions.load, bounds_of_item=functions.bounds,
        host_bounds=_array_bounds, host_pack=_array_pack,
        closest_euclidean=as_warp_function(closest), segment_endpoints=as_warp_function(endpoints),
        point_bounds=True, name="points", payload_type=Payload,
        prepare_data=_array_data(Payload, dimension, 1, "points"),
    )

"""Silhouette-edge primitive specification."""
from functools import lru_cache

import warp as wp

from sengi.warp.native import package_headers
from sengi.primitives.data import SilhouetteEdgesArrays3, SilhouetteEdgesData3
from sengi.primitives.device import item_functions
from sengi.primitives.items import SilhouetteItem3
from sengi.primitives.packing import _array_bounds, _array_data, _array_pack
from sengi.primitives.spec import Primitive
from sengi.warp.contracts import as_warp_function


@lru_cache(maxsize=None)
def SilhouetteEdges() -> Primitive[SilhouetteEdgesArrays3, SilhouetteEdgesData3, SilhouetteItem3, wp.vec3]:
    Data, Payload, Item = SilhouetteEdgesData3, SilhouetteEdgesArrays3, SilhouetteItem3
    functions = item_functions("silhouette_edges", 3, Data, Item)

    @wp.func_native("return bvh::closest_segment(query, item.a, item.b);")
    def closest(query: wp.vec3, item: Item) -> wp.vec3: ...

    @wp.func
    def endpoints(item: Item):
        return item.a, item.b

    @wp.func
    def normal_sides(item: Item):
        return item.a, item.n0, item.n1

    return Primitive(
        dimension=3, point_type=wp.vec3, data_type=Data, item_type=Item,
        load=functions.load, bounds_of_item=functions.bounds,
        host_bounds=_array_bounds, host_pack=_array_pack,
        closest_euclidean=as_warp_function(closest), segment_endpoints=as_warp_function(endpoints),
        normal_sides=as_warp_function(normal_sides), name="silhouette_edges", payload_type=Payload,
        prepare_data=_array_data(Payload, 3, 4, "silhouette_edges"),
        native=package_headers("primitives", "segment_distance.h"),
    )

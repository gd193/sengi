"""A third-party extension uses only public contracts and concrete signatures."""
from sengi import bvh
from typing import Callable, assert_type
import warp as wp
from sengi.metrics.nearest import NearestCallbacks, NearestOps, NearestPolicy
from sengi.primitives.items import PointItem3
from sengi.primitives.points import Points
from sengi.warp.contracts import PrimitiveSpec, WarpFunction, as_warp_function, warp_func
from sengi.queries.nearest import bind as bind_nearest

from tests.typing.warp_api.real_extension import (
    ExternalArrays as Host, ExternalData as Device, ExternalItem as Item,
    builtin_evaluate, external_bound,
)
class OtherItem: ...

@warp_func
def evaluate(query: wp.vec3, item: Item) -> tuple[wp.vec3, float]:
    return query, 0.0

@warp_func
def lower_bound(query: wp.vec3, lower: wp.vec3, upper: wp.vec3) -> float:
    return 0.0

callbacks: NearestCallbacks[Item, wp.vec3] = NearestCallbacks(evaluate, lower_bound)
typed_evaluate: WarpFunction[[wp.vec3, Item], tuple[wp.vec3, float]] = evaluate

def extension(primitive: PrimitiveSpec[Host, Device, Item, wp.vec3]) -> None:
    assert_type(callbacks.bind(primitive), NearestOps[wp.vec3, Item, wp.vec3])
    policy: NearestPolicy[Device, Item, wp.vec3, wp.vec3] = callbacks
    bind_nearest(bvh.IndexSpec(primitive), policy)
    bind_nearest(bvh.IndexSpec(primitive), evaluate=evaluate, lower_bound=lower_bound)

# Existing Warp functions require an explicit, reviewable signature boundary.
def existing(function: Callable[[wp.vec3, Item], tuple[wp.vec3, float]]) -> WarpFunction[[wp.vec3, Item], tuple[wp.vec3, float]]:
    return as_warp_function(function)

builtin_callbacks: NearestCallbacks[PointItem3, wp.vec3] = NearestCallbacks(builtin_evaluate, external_bound)
assert_type(builtin_callbacks.bind(Points(3)), NearestOps[wp.vec3, PointItem3, wp.vec3])
bind_nearest(bvh.IndexSpec(Points(3)), builtin_callbacks)
bind_nearest(bvh.IndexSpec(Points(3)), evaluate=builtin_evaluate, lower_bound=external_bound)

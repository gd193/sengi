"""Concrete kNN mismatches must be rejected by the public signatures."""
from sengi import bvh
from sengi.bvh.device_tree import TreeData
import jax
import warp as wp
from sengi.metrics.nearest import Euclidean, NearestCallbacks, NearestPolicy
from sengi.primitives.points import Points
from sengi.warp.contracts import PrimitiveSpec, warp_func
from sengi.queries.knn import bind as bind_knn
from sengi.queries.nearest import bind as bind_nearest
from tests.typing.warp_api.valid import Device, Host, Item, OtherItem, evaluate, lower_bound
from tests.typing.warp_api.real_extension import external_evaluate, external_bound

@warp_func
def wrong_item(query: wp.vec3, item: OtherItem) -> tuple[wp.vec3, float]:
    return query, 0.0

@warp_func
def wrong_predicate(query: wp.vec3, item: OtherItem) -> bool:
    return True

@warp_func
def wrong_state(query: float, item: Item) -> bool:
    return True


def mismatches(primitive: PrimitiveSpec[Host, Device, Item, wp.vec3],
               policy: NearestPolicy[Device, OtherItem, wp.vec3, wp.vec3],
               tree: TreeData, queries: jax.Array) -> None:
    bind_knn(bvh.IndexSpec(primitive), 3, policy)  # expect: reportArgumentType
    bind_knn(bvh.IndexSpec(primitive), 3, evaluate=wrong_item, lower_bound=lower_bound)  # expect: reportArgumentType
    bind_knn(bvh.IndexSpec(primitive), 3, NearestCallbacks(evaluate, lower_bound), predicate=wrong_predicate)  # expect: reportArgumentType
    bind_knn(bvh.IndexSpec(primitive), 3, NearestCallbacks(evaluate, lower_bound), predicate=wrong_state)  # expect: reportArgumentType
    query = bind_knn(bvh.IndexSpec(primitive), 3, evaluate=evaluate, lower_bound=lower_bound)
    query(bvh.IndexData(tree, object()), queries)  # expect: reportArgumentType

bind_knn(bvh.IndexSpec(Points(3)), 3, evaluate=external_evaluate, lower_bound=external_bound)  # expect: reportArgumentType
bind_knn(bvh.IndexSpec(Points(3)), 3, NearestCallbacks(external_evaluate, external_bound))  # expect: reportArgumentType
bind_knn(bvh.IndexSpec(Points(3)), 3, Euclidean(), packed_nodes="packed")  # expect: reportArgumentType
bind_nearest(bvh.IndexSpec(Points(3)), Euclidean(), group_leaves="flat")  # expect: reportArgumentType # expect: reportCallIssue

bind_knn(bvh.IndexSpec(Points(3)), 3, Euclidean(), return_points="points")  # expect: reportArgumentType

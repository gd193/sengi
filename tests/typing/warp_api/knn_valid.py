"""kNN preserves primitive, callback, policy and host payload types."""
from sengi.bvh.device_tree import TreeData
from sengi import bvh
from typing import assert_type
import jax
import warp as wp
from sengi.metrics.nearest import Euclidean, NearestCallbacks, NearestPolicy
from sengi.primitives.points import Points
from sengi.warp.contracts import PrimitiveSpec, warp_func
from sengi.queries.knn import bind as bind_knn
from sengi.queries.nearest import bind as bind_nearest
from tests.typing.warp_api.real_extension import (
    ExternalArrays, ExternalData, ExternalItem, ExternalPoints,
    external_bound, external_evaluate,
)

@warp_func
def accepts(query: wp.vec3, item: ExternalItem) -> bool:
    return True


def extension(primitive: PrimitiveSpec[ExternalArrays, ExternalData, ExternalItem, wp.vec3],
              tree: TreeData, data: ExternalArrays, queries: jax.Array) -> None:
    callbacks: NearestCallbacks[ExternalItem, wp.vec3] = NearestCallbacks(external_evaluate, external_bound)
    policy: NearestPolicy[ExternalData, ExternalItem, wp.vec3, wp.vec3] = callbacks
    query = bind_knn(bvh.IndexSpec(primitive), 3, policy, predicate=accepts)
    assert_type(query(bvh.IndexData(tree, data), queries), tuple[jax.Array, jax.Array, jax.Array, jax.Array])
    points_query = bind_knn(bvh.IndexSpec(primitive), 3, policy, predicate=accepts, return_points=True)
    assert_type(points_query.return_points, bool)
    assert_type(points_query(bvh.IndexData(tree, data), queries), tuple[jax.Array, jax.Array, jax.Array, jax.Array])
    bind_knn(bvh.IndexSpec(primitive), 3, evaluate=external_evaluate, lower_bound=external_bound, predicate=accepts)
    bind_knn(bvh.IndexSpec(primitive), 3, evaluate=external_evaluate, lower_bound=external_bound, return_points=True)

bind_knn(bvh.IndexSpec(ExternalPoints()), 3, NearestCallbacks(external_evaluate, external_bound))
bind_knn(bvh.IndexSpec(Points(3)), 3, Euclidean())
bind_knn(bvh.IndexSpec(Points(3)), 3, Euclidean(), cache_bounds=False, packed_nodes=True, group_leaves=False)
bind_nearest(bvh.IndexSpec(Points(3)), Euclidean(), cache_bounds=True, packed_nodes=False, group_leaves=True)

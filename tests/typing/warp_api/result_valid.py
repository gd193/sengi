"""Custom selected structs preserve callback, primitive and output types."""
from sengi import bvh
from typing import assert_type
import warp as wp
from sengi.warp.contracts import DeviceArray2D, warp_func, warp_struct
from sengi.bvh.device_tree import DeviceTree
from sengi.metrics.nearest import Euclidean
from sengi.queries.results import IdPoint, IdPointResult, ResultPolicy
from sengi.primitives.points import Points
from sengi.primitives.data import PointsData3
from sengi.queries.knn import bind as bind_knn
from sengi.queries.nearest import bind as bind_nearest
from tests.typing.warp_api.real_extension import (
    ExternalPoints, ExternalItem, ExternalData, external_evaluate, external_bound,
)


@warp_struct
class Selected:
    original_id: int
    point: wp.vec3
    score: float


@warp_func
def build(query: wp.vec3, item: ExternalItem, index: int, point: wp.vec3, value: float) -> Selected:
    selected = Selected()
    selected.original_id = index
    selected.point = point
    selected.score = value
    return selected


policy = ResultPolicy(Selected, build)
nearest = bind_nearest(bvh.IndexSpec(ExternalPoints()), evaluate=external_evaluate,
                       lower_bound=external_bound, result_policy=policy)
knn = bind_knn(bvh.IndexSpec(ExternalPoints()), 4, evaluate=external_evaluate,
              lower_bound=external_bound, result_policy=policy)


def signatures(tree: DeviceTree, data: ExternalData, points: PointsData3, query: wp.vec3,
               scores: DeviceArray2D[float], selected: DeviceArray2D[Selected],
               id_points: DeviceArray2D[IdPointResult[wp.vec3]]) -> None:
    assert_type(nearest.device(query, 2.0, tree, data), tuple[Selected, int])
    assert_type(knn.device(query, 2.0, tree, data, scores, selected, 0), tuple[int, int])
    plain = bind_nearest(bvh.IndexSpec(Points()), Euclidean(), result_policy=IdPoint())
    assert_type(plain.device(query, 2.0, tree, points), tuple[IdPointResult[wp.vec3], int])
    both = bind_knn(bvh.IndexSpec(Points()), 4, Euclidean(), result_policy=IdPoint())
    assert_type(both.device(query, 2.0, tree, points, scores, id_points, 0), tuple[int, int])

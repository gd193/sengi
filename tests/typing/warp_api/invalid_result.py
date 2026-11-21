"""Custom result queries are device-only and have typed output buffers."""
from sengi import bvh
import warp as wp
from sengi.warp.contracts import DeviceArray2D
from sengi.bvh.device_tree import DeviceTree
from sengi.metrics.nearest import Euclidean
from sengi.primitives.points import Points
from sengi.queries.nearest import bind as bind_nearest
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalData
from tests.typing.warp_api.result_valid import nearest, knn, policy


def invalid(tree: DeviceTree, data: ExternalData, point: wp.vec3,
            index_data: bvh.IndexData[ExternalArrays],
            scores: DeviceArray2D[float], ids: DeviceArray2D[int]) -> None:
    nearest(index_data, point)  # expect: reportCallIssue
    knn(index_data, point)  # expect: reportCallIssue
    knn.device(point, 1.0, tree, data, scores, ids, 0)  # expect: reportArgumentType
    nearest.device(wp.vec2(), 1.0, tree, data)  # expect: reportArgumentType
    bind_nearest(bvh.IndexSpec(Points()), Euclidean(), result_policy=policy)  # expect: reportCallIssue  # expect: reportArgumentType

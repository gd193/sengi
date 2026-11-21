"""Reject mode, context, state, primitive and geometry mismatches."""
from sengi import bvh
import warp as wp
from sengi.bvh.device_tree import DeviceTree
from sengi.metrics.nearest import Euclidean
from sengi.primitives.points import Points
from sengi.primitives.data import PointsData3
from sengi.primitives.triangles import Triangles
from sengi.warp.contracts import WarpRef, warp_struct
from sengi.queries.radius import bind as bind_radius
from tests.typing.warp_api.range_valid import Context, SumState, accumulator, reduction, traversal


@warp_struct
class WrongState:
    limit: float
    failed: bool
    stop: bool


def invalid(tree: DeviceTree, geometry: PointsData3, point: wp.vec3,
            state: WarpRef[SumState]) -> None:
    state.total = "wrong"  # expect: reportAttributeAccessIssue
    traversal.device(tree, point, state)  # expect: reportArgumentType
    traversal.device(tree, Context(), WrongState())  # expect: reportArgumentType
    reduction.device(wp.vec2(), 1.0, tree, geometry)  # expect: reportArgumentType
    reduction.device(point, 1.0, tree, point)  # expect: reportArgumentType
    bind_radius(bvh.IndexSpec(Points()), Euclidean(), mode="all")  # expect: reportArgumentType # expect: reportCallIssue
    bind_radius(bvh.IndexSpec(Points()), Euclidean(), mode="accumulate")  # expect: reportArgumentType # expect: reportCallIssue
    bind_radius(bvh.IndexSpec(Triangles()), Euclidean(), mode="accumulate", accumulator=accumulator)  # expect: reportCallIssue  # expect: reportArgumentType

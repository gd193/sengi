"""Reject region callbacks for other primitives and swapped metadata."""
from sengi import bvh
from sengi.boundary.data import BoundaryData
import warp as wp
from sengi.bvh.device_tree import DeviceTree, TreeData
from sengi.primitives.points import Points
from sengi.primitives.data import PointsData3
from sengi.queries.region import bind as bind_region
from sengi.primitives.triangles import Triangles
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalData
from tests.typing.warp_api.query_extension import ExternalMetadata
from tests.typing.warp_api.region_valid import external, classified


def invalid(tree: DeviceTree, points: PointsData3, data: ExternalData,
            host_tree: TreeData, host: ExternalArrays, metadata: ExternalMetadata, p: wp.vec3) -> None:
    wrong = Triangles().region_policy
    assert wrong is not None
    bind_region(bvh.IndexSpec(Points()), wrong)  # expect: reportArgumentType
    external.device(p, p, tree, points)  # expect: reportArgumentType
    external.device(wp.vec2(), p, tree, data)  # expect: reportArgumentType
    classified.device(p, p, tree, data, points)  # expect: reportArgumentType
    classified(BoundaryData(bvh.IndexData(host_tree, host), metadata), 0.0, 0.0)  # expect: reportArgumentType

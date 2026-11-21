"""Region factories preserve primitive and host/device metadata types."""
from sengi import bvh
from typing import assert_type
import warp as wp
from sengi.queries.domain import ClassificationOps
from sengi.bvh.device_tree import DeviceTree
from sengi.primitives.points import Points
from sengi.primitives.data import PointsData3
from sengi.queries.region import RegionIntersectionPolicy, bind as bind_region
from sengi.queries.region_classify import bind as bind_region_classify
from sengi.warp.contracts import warp_func
from tests.typing.warp_api.real_extension import ExternalPoints, ExternalData
from tests.typing.warp_api.native_math import within_axis
from tests.typing.warp_api.query_extension import (
    ExternalMetadata, MetadataArrays, metadata_valid, classification_event,
)


@warp_func
def intersect(data: ExternalData, slot: int, lower: wp.vec3, upper: wp.vec3) -> bool:
    point = data.positions[slot]
    return (within_axis(point, lower, upper, 0) and
            within_axis(point, lower, upper, 1) and
            within_axis(point, lower, upper, 2))


policy = RegionIntersectionPolicy(intersect)
classification = ClassificationOps(ExternalMetadata, metadata_valid, classification_event)
external = bind_region(bvh.IndexSpec(ExternalPoints()), policy, mode="count")
classified = bind_region_classify(bvh.IndexSpec(ExternalPoints()), policy, classification,
                                 metadata_host_type=MetadataArrays)


def signatures(tree: DeviceTree, points: PointsData3, data: ExternalData,
               metadata: ExternalMetadata, lower: wp.vec3, upper: wp.vec3) -> None:
    query = bind_region(bvh.IndexSpec(Points()), mode="count")
    assert_type(query.device(lower, upper, tree, points), tuple[int, int])
    assert_type(external.device(lower, upper, tree, data), tuple[int, int])
    assert_type(classified.device(lower, upper, tree, data, metadata), tuple[int, int])

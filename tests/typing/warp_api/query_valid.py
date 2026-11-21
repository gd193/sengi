"""All query families retain concrete third-party callback and payload types."""
from sengi.bvh.device_tree import TreeData
from sengi import bvh
import warp as wp
from sengi.queries.domain import ClassificationOps, RayOps
from sengi.boundary.sampling import MeasureOps, SamplingData
from sengi.queries.boundary_sample import RegionOps
from sengi.queries.boundary_sample import bind as bind_boundary_sample
from sengi.queries.classification import bind as bind_classification
from sengi.queries.overlap import bind as bind_overlap
from sengi.queries.rays import bind as bind_rays
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalData, ExternalItem, ExternalPoints
from tests.typing.warp_api.query_extension import (
    Cursor, ExternalMetadata, ExternalRobinMetric, MetadataArrays, classification_event, intersect, metadata_valid,
    next_hit, region_contains, region_overlaps, region_valid, sample,
)
from sengi.boundary.data import BoundaryCore, BoundaryData, SilhouetteMeta
from sengi.queries.robin_radius import RobinOps
from sengi.queries.robin_radius import bind as bind_robin_radius
from tests.typing.warp_api.query_extension import (
    radius_to_score, robin_feature, robin_node, robin_primitive,
    robin_valid, score_to_radius,
)
import jax
from sengi.queries.silhouette import bind as bind_silhouette
from sengi.queries.star_radius import bind as bind_star_radius
from sengi.metrics.nearest import NearestCallbacks
from tests.typing.warp_api.query_extension import external_adapter, FeatureArrays
from tests.typing.warp_api.real_extension import external_bound, external_evaluate
from sengi.primitives.points import Points
from sengi.primitives.data import PointsData3
from tests.typing.warp_api.query_extension import builtin_intersect
from sengi.boundary.data import BoundaryCoreData, SilhouetteData
from sengi.boundary.robin import RobinData

single: RayOps[ExternalData, int] = RayOps(intersect=intersect)
multiple: RayOps[ExternalData, Cursor] = RayOps(state_type=Cursor, next_hit=next_hit)
classification: ClassificationOps[wp.vec3, ExternalData, ExternalMetadata] = ClassificationOps(
    ExternalMetadata, metadata_valid, classification_event)
measure: MeasureOps[ExternalItem, wp.vec3, ExternalArrays] = MeasureOps(sample)
region: RegionOps[float, wp.vec3] = RegionOps(float, (), region_valid, region_overlaps, region_contains)
primitive = ExternalPoints()
ray = bind_rays(bvh.IndexSpec(primitive), ops=single)
stateful_ray = bind_rays(bvh.IndexSpec(primitive), "all", ops=multiple)
classify = bind_classification(bvh.IndexSpec(primitive), classification, metadata_host_type=MetadataArrays)
sampler = bind_boundary_sample(bvh.IndexSpec(primitive), measure, region)
overlap = bind_overlap(bvh.IndexSpec(primitive))

robin: RobinOps[wp.vec3, ExternalData, ExternalItem, BoundaryCore, SilhouetteMeta] = RobinOps(
    wp.vec3, (3,), robin_valid, robin_node, robin_primitive, robin_feature,
    core_type=BoundaryCore, feature_type=SilhouetteMeta,
    to_score=radius_to_score, to_distance=score_to_radius)
robin_query = bind_robin_radius(bvh.IndexSpec(primitive), robin)
robin_from_metric = bind_robin_radius(bvh.IndexSpec(primitive), metric=ExternalRobinMetric())


def invoke(tree: TreeData, geometry: ExternalArrays, arrays: jax.Array,
           sampling: SamplingData[ExternalArrays], metadata: MetadataArrays) -> None:
    classify(BoundaryData(bvh.IndexData(tree, geometry), metadata), arrays)
    ray(bvh.IndexData(tree, geometry), arrays, arrays)
    stateful_ray(bvh.IndexData(tree, geometry), arrays, arrays, max_hits=4)
    sampler(sampling, arrays, 1.0, arrays)
    overlap(bvh.IndexData(tree, geometry), arrays, arrays)

silhouette_query = bind_silhouette(bvh.IndexSpec(primitive), NearestCallbacks(external_evaluate, external_bound), adapter=external_adapter)
star_query = bind_star_radius(bvh.IndexSpec(primitive), adapter=external_adapter)

def invoke_features(tree: TreeData, geometry: ExternalArrays, core: BoundaryCoreData,
                    features: FeatureArrays, arrays: jax.Array) -> None:
    silhouette_query(BoundaryData(bvh.IndexData(tree, geometry), core, features), arrays)
    star_query(BoundaryData(bvh.IndexData(tree, geometry), core, features), arrays, 10.0)

builtin_ray_ops: RayOps[PointsData3, int] = RayOps(intersect=builtin_intersect)
builtin_ray = bind_rays(bvh.IndexSpec(Points(3)), ops=builtin_ray_ops)


def invoke_robin(metadata: RobinData[ExternalArrays, BoundaryCoreData, SilhouetteData],
                 arrays: jax.Array) -> None:
    robin_query(metadata, arrays)
    robin_from_metric(metadata, arrays)


def construct_boundary(geometry: ExternalArrays) -> tuple[BoundaryCoreData, FeatureArrays | None]:
    index = bvh.build(primitive, geometry)
    boundary = bvh.prepare_boundary(index, geometry, adapter=external_adapter)
    return boundary.core, boundary.silhouette

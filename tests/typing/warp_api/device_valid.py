"""Device calls keep query, geometry, metadata and buffer types."""
from sengi import bvh
from typing import assert_type
import warp as wp
from sengi.warp.contracts import DeviceArray, DeviceArray2D
from sengi.bvh.device_tree import DeviceTree
from sengi.metrics.nearest import Euclidean
from sengi.primitives.points import Points
from sengi.primitives.data import PointsData3, TrianglesData3
from sengi.primitives.triangles import Triangles
from sengi.boundary.data import BoundaryCore, SilhouetteMeta
from sengi.boundary.sampling import SamplingMeta
from sengi.boundary.robin import CoefficientMeta
from sengi.queries.nearest import bind as bind_nearest
from sengi.queries.knn import bind as bind_knn
from sengi.queries.rays import bind as bind_rays
from sengi.queries.classification import bind as bind_classification
from sengi.queries.boundary_sample import bind as bind_boundary_sample
from sengi.queries.silhouette import bind as bind_silhouette
from sengi.queries.star_radius import bind as bind_star_radius
from sengi.queries.robin_radius import bind as bind_robin_radius


def point_queries(tree: DeviceTree, geometry: PointsData3, point: wp.vec3,
                  values: DeviceArray2D[float], ids: DeviceArray2D[int],
                  points: DeviceArray2D[wp.vec3]) -> None:
    nearest = bind_nearest(bvh.IndexSpec(Points()), Euclidean())
    assert_type(nearest.device(point, 10., tree, geometry), tuple[wp.vec3, float, int, int])
    knn = bind_knn(bvh.IndexSpec(Points()), 4, Euclidean())
    assert_type(knn.device(point, 10., tree, geometry, values, ids, 0), tuple[int, int])
    knn_points = bind_knn(bvh.IndexSpec(Points()), 4, Euclidean(), return_points=True)
    assert_type(knn_points.device(point, 10., tree, geometry, values, points, 0), tuple[int, int])


def boundaries(tree: DeviceTree, geometry: TrianglesData3, point: wp.vec3,
               core: BoundaryCore, feature: SilhouetteMeta, sampling: SamplingMeta, coefficients: CoefficientMeta,
               times: DeviceArray[float], ids: DeviceArray[int], counts: DeviceArray[int], statuses: DeviceArray[int],
               all_times: DeviceArray2D[float], all_ids: DeviceArray2D[int]) -> None:
    first = bind_rays(bvh.IndexSpec(Triangles()), 'first')
    first.device(point, point, 0., 10., -1, tree, geometry, times, ids, counts, statuses, 0)
    bind_rays(bvh.IndexSpec(Triangles()), 'any').device(point, point, 0., 10., -1, tree, geometry, counts, statuses, 0)
    bind_rays(bvh.IndexSpec(Triangles()), 'count').device(point, point, 0., 10., -1, tree, geometry, counts, statuses, 0)
    bind_rays(bvh.IndexSpec(Triangles()), 'all').device(point, point, 0., 10., -1, tree, geometry, all_times, all_ids, counts, statuses, 0)
    assert_type(bind_classification(bvh.IndexSpec(Triangles())).device(point, tree, geometry, core), tuple[float, int])
    assert_type(bind_silhouette(bvh.IndexSpec(Triangles())).device(point, 10., tree, feature), tuple[wp.vec3, float, int, int])
    assert_type(bind_star_radius(bvh.IndexSpec(Triangles())).device(point, 10., tree, feature), tuple[float, int])
    assert_type(bind_robin_radius(bvh.IndexSpec(Triangles())).device(point, 10., tree, geometry, core, feature, coefficients), tuple[float, int])
    assert_type(bind_boundary_sample(bvh.IndexSpec(Triangles())).device(point, 10., point, tree, geometry, sampling), tuple[wp.vec3, wp.vec3, float, int, int])

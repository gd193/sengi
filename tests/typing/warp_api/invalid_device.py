"""Reject host payloads, incorrect geometry and wrong device result buffers."""
from sengi import bvh
import warp as wp
from sengi.warp.contracts import DeviceArray, DeviceArray2D
from sengi.bvh.device_tree import DeviceTree
from sengi.metrics.nearest import Euclidean
from sengi.primitives.points import Points
from sengi.primitives.data import PointsData3, TrianglesData3, PointsArrays3, TrianglesArrays3
from sengi.primitives.triangles import Triangles
from sengi.boundary.data import BoundaryCore, SilhouetteMeta, BoundaryCoreData, SilhouetteData
from sengi.queries.nearest import bind as bind_nearest
from sengi.queries.knn import bind as bind_knn
from sengi.queries.rays import bind as bind_rays
from sengi.queries.classification import bind as bind_classification
from sengi.queries.star_radius import bind as bind_star_radius
from sengi.queries.star_radius import StarRadiusQuery
from sengi.primitives.items import TriangleItem3, SilhouetteItem3


def invalid(tree: DeviceTree, geometry: PointsData3, triangles: TrianglesData3,
            host: PointsArrays3, point: wp.vec3, core: BoundaryCore, feature: SilhouetteMeta,
            values: DeviceArray2D[float], ids: DeviceArray2D[int], scalar: DeviceArray[int]) -> None:
    bind_nearest(bvh.IndexSpec(Points()), Euclidean()).device(point, 10., tree, host)  # expect: reportArgumentType
    bind_nearest(bvh.IndexSpec(Points()), Euclidean()).device(point, 10., tree, triangles)  # expect: reportArgumentType
    bind_knn(bvh.IndexSpec(Points()), 4, Euclidean()).device(point, 10., tree, triangles, values, ids, 0)  # expect: reportArgumentType
    bind_knn(bvh.IndexSpec(Points()), 4, Euclidean()).device(point, 10., tree, geometry, values, scalar, 0)  # expect: reportArgumentType
    bind_rays(bvh.IndexSpec(Triangles()), 'all').device(point, point, 0., 10., -1, tree, triangles, values, scalar, scalar, scalar, 0)  # expect: reportArgumentType
    bind_rays(bvh.IndexSpec(Triangles()), 'any').device(point, point, 0., 10., -1, tree, geometry, scalar, scalar, 0)  # expect: reportArgumentType
    bind_classification(bvh.IndexSpec(Triangles())).device(point, tree, triangles, feature)  # expect: reportArgumentType
    star: StarRadiusQuery[TrianglesArrays3, TrianglesData3, TriangleItem3, wp.vec3, BoundaryCoreData, BoundaryCore, SilhouetteData, SilhouetteMeta, SilhouetteItem3] = bind_star_radius(bvh.IndexSpec(Triangles()))
    star.device(point, 10., tree, core)  # expect: reportArgumentType

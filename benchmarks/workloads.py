"""Seeded geometry and query inputs shared by every adapter."""

from __future__ import annotations

import numpy as np
import trimesh

from benchmarks.assets import Geometry
from benchmarks.cases import ALL_PRIMITIVES, BOUNDARIES, CATALOG, geometry_support

EXTRA_FIXTURES = {
    "circle": "segments2",
    "star": "segments2",
    "cluster2": "points2",
    "cluster3": "points3",
}
BUILTINS = (*ALL_PRIMITIVES, *EXTRA_FIXTURES)
POINT_WORKLOADS = ("uniform", "clustered", "surface")
WORKLOADS = (
    "gaussian",
    "shell",
    "interior",
    "center",
    "near_surface",
    "builtin",
    *POINT_WORKLOADS,
)
TRIANGLE_BASELINE = ("sphere", "torus", "bunny", "armadillo")
RAY_MODES = ("first_hit", "any_hit", "count_hits", "all_hits")
TRIANGLE_MODES = tuple(
    name for name, query in CATALOG.items() if "triangles" in query.primitives
)

BUILTIN_MODES = (
    {"geometry": ALL_PRIMITIVES, "query": ("closest_point", "aabb_overlap")},
    {
        "geometry": ("points2", "points3", "segments2", "segments3", "edges"),
        "query": ("closest_point",),
        "metric": ("angular",),
    },
    {"geometry": BOUNDARIES, "query": RAY_MODES},
    {
        "geometry": BOUNDARIES,
        "query": ("closest_silhouette",),
        "metric": ("euclidean", "angular"),
    },
    {
        "geometry": BOUNDARIES,
        "query": ("star_radius", "robin_radius", "classify_point", "sample_boundary"),
    },
)
LOCAL_QUERIES = (
    "aabb_overlap",
    "radius_any",
    "radius_count",
    "radius_accumulate",
    "region_any",
    "region_count",
    "sample_boundary",
)
SUITES = {
    "smoke": (
        {
            "geometry": ("sphere",),
            "size": ("small",),
            "query": ("region_count", "sample_boundary"),
            "workload": ("near_surface",),
            "batch_size": (512,),
        },
        {
            "geometry": ("points3",),
            "size": ("1024",),
            "query": ("closest_point", "knn"),
            "batch_size": (512,),
        },
    ),
    "triangles": (
        {
            "geometry": TRIANGLE_BASELINE,
            "size": ("*",),
            "query": tuple(q for q in TRIANGLE_MODES if q not in LOCAL_QUERIES),
            "batch_size": (8192,),
        },
        {
            "geometry": TRIANGLE_BASELINE,
            "size": ("*",),
            "query": LOCAL_QUERIES,
            "workload": ("near_surface",),
            "batch_size": (8192,),
        },
    ),
    "boundary": (
        {
            "geometry": TRIANGLE_BASELINE,
            "size": ("*",),
            "workload": ("near_surface",),
            "batch_size": (8192,),
            "query": (
                "closest_silhouette",
                "star_radius",
                "sample_boundary",
                "robin_radius",
            ),
        },
    ),
    "builtins": (
        *(
            dict(spec, size=("1024",), workload=("builtin",), batch_size=(8192,))
            for spec in BUILTIN_MODES
        ),
        {
            "geometry": ("triangles",),
            "size": ("192",),
            "workload": ("builtin",),
            "query": ("radius_accumulate", "nearest_selected"),
            "batch_size": (512,),
        },
        {
            "geometry": ("open_torus",),
            "size": ("small",),
            "query": ("closest_silhouette", "star_radius"),
        },
    ),
    "reference_queries": (
        {
            "geometry": ("segments2", "circle", "star"),
            "size": ("1024",),
            "workload": ("builtin",),
            "batch_size": (8192,),
            "query": ("closest_point", "first_hit", "any_hit", "closest_silhouette"),
        },
        {
            "geometry": ("points2", "points3", "cluster2", "cluster3"),
            "size": ("1024",),
            "workload": ("builtin",),
            "query": ("closest_point",),
            "batch_size": (8192,),
        },
    ),
    "point_knn": tuple(
        {
            "geometry": ("points2", "points3"),
            "size": ("1024", "65536", "1048576"),
            "workload": POINT_WORKLOADS,
            "query": ("knn",),
            "batch_size": (1024, 32768),
            "seed": (619,),
            "parameters": {"k": k},
        }
        for k in (1, 4, 8, 16, 32, 64, 128)
    ),
    "saturation": (
        {
            "geometry": ("sphere",),
            "size": ("large",),
            "query": ("region_count", "closest_point", "first_hit"),
            "batch_size": (512, 8192, 32768, 131072),
        },
    ),
}
SUITES["full"] = tuple(block for preset in SUITES.values() for block in preset)


def geometry_info(name, size):
    if name not in BUILTINS or not str(size).isdecimal() or int(size) < 1:
        raise ValueError(f"unknown builtin geometry {name}/{size}")
    if name in ("circle", "star") and int(size) < 3:
        raise ValueError(f"{name} requires at least three vertices")
    primitive = EXTRA_FIXTURES.get(name, name)
    return primitive, primitive in BOUNDARIES


def _sample_points(dimension, count, kind, seed, *, queries=False):
    rng = np.random.default_rng([seed, 2] if queries else seed)
    if kind == "clustered":
        centers = np.random.default_rng(seed + 1).uniform(-1, 1, (8, dimension))
        points = centers[rng.integers(8, size=count)] + rng.normal(
            0, 0.035, (count, dimension)
        )
    elif kind == "surface":
        points = rng.normal(size=(count, dimension))
        points /= np.linalg.norm(points, axis=1, keepdims=True)
    else:
        points = rng.uniform(-1, 1, (count, dimension))
    return points.astype(np.float32)


def builtin_geometry(name, size, seed=73, workload="gaussian"):
    primitive, closed = geometry_info(name, size)
    size = int(size)
    rng = np.random.default_rng(73)
    if name == "triangles":
        mesh = trimesh.creation.box(extents=[2.0, 2.0, 2.0])
        while len(mesh.faces) < size:
            mesh = mesh.subdivide()
        arrays = {
            "vertices": np.asarray(mesh.vertices, dtype=np.float32),
            "faces": np.asarray(mesh.faces, dtype=np.int32),
        }
    elif primitive == "segments2":
        if name == "segments2":
            corners = np.array(
                [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], np.float32
            )
            steps = max(1, size // 4)
            t = np.arange(steps) / steps
            vertices = np.concatenate(
                [
                    a + t[:, None] * (b - a)
                    for a, b in zip(corners, np.roll(corners, -1, axis=0))
                ]
            ).astype(np.float32)
        else:
            angle = np.linspace(0.0, 2 * np.pi, size, endpoint=False)
            vertices = np.column_stack((np.cos(angle), np.sin(angle))).astype(
                np.float32
            )
            if name == "star":
                vertices *= (1 + 0.35 * np.cos(7 * angle))[:, None]
        first = np.arange(len(vertices), dtype=np.int32)
        indices = np.column_stack((first, np.roll(first, -1)))
        arrays = {
            "vertices": vertices,
            "indices": indices,
            "starts": vertices[indices[:, 0]],
            "ends": vertices[indices[:, 1]],
        }
    else:
        dimension = 2 if primitive == "points2" else 3
        if primitive.startswith("points"):
            if workload in POINT_WORKLOADS:
                starts = _sample_points(dimension, size, workload, seed)
            elif name.startswith("cluster"):
                cluster_rng = np.random.default_rng(619)
                centers = cluster_rng.normal(size=(7, dimension))
                jitter = cluster_rng.normal(size=(size, dimension)) * np.geomspace(
                    0.005, 0.3, dimension
                )
                starts = (centers[np.arange(size) % 7] + jitter).astype(np.float32)
            else:
                starts = rng.normal(size=(size, dimension)).astype(np.float32)
            arrays = {"points": starts}
        else:
            starts = rng.normal(size=(size, dimension)).astype(np.float32)
            ends = (starts + 0.5 * rng.normal(size=starts.shape)).astype(np.float32)
            arrays = {"starts": starts, "ends": ends}
            if name == "edges":
                arrays.update(
                    normal0=rng.normal(size=starts.shape).astype(np.float32),
                    normal1=rng.normal(size=starts.shape).astype(np.float32),
                )
    return Geometry(name, str(size), primitive, arrays, closed)


def near_surface_points(vertices, faces, n, radius, seed=404):
    """Sample by triangle area, with signed offsets along the sampled normals."""
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    )
    area = np.linalg.norm(normals, axis=1).astype(np.float64)
    rng = np.random.default_rng(seed + n)
    ids = rng.choice(len(faces), size=n, p=area / area.sum())
    a, b, c = triangles[ids].transpose(1, 0, 2)
    u, v = rng.uniform(size=(2, n, 1))
    t = np.sqrt(u)
    surface = (1 - t) * a + t * ((1 - v) * b + v * c)
    offset = rng.uniform(-2 * radius, 2 * radius, size=(n, 1))
    return (surface + offset * normals[ids] / area[ids, None]).astype(np.float32)


def _triangle_queries(n, seed, workload):
    rng = np.random.default_rng(seed + n)
    points = rng.normal(size=(n, 3)).astype(np.float32) * 1.5
    origins = rng.normal(size=(n, 3)).astype(np.float32) * 2.0
    directions = rng.normal(size=(n, 3)).astype(np.float32) - origins
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-30)
    if workload in ("shell", "interior", "center"):
        point_rng = np.random.default_rng(seed + n + 100000)
        unit = point_rng.normal(size=(n, 3)).astype(np.float32)
        unit /= np.maximum(np.linalg.norm(unit, axis=1, keepdims=True), 1e-30)
        if workload == "shell":
            points = (unit * point_rng.uniform(0.95, 1.05, size=(n, 1))).astype(
                np.float32
            )
        elif workload == "interior":
            radius = 0.95 * np.cbrt(point_rng.random((n, 1)))
            points = (unit * radius).astype(np.float32)
        else:
            points = np.zeros_like(points)
    return points, origins, directions


def make_inputs(case, geometry):
    """Materialize the named contiguous arrays consumed by a query."""

    if reason := geometry_support(case, geometry.primitive, geometry.closed):
        raise ValueError(reason)

    n, seed, workload, mode = case.batch_size, case.seed, case.workload, case.query
    params = case.parameters
    primitive = geometry.primitive
    arrays = geometry.arrays
    dimension = 2 if primitive.endswith("2") else 3
    builtin = workload == "builtin" or primitive != "triangles"
    default_radius = 0.75 if builtin else 0.5
    radius = params.get(
        "radius", 0.05 if workload == "near_surface" else default_radius
    )
    rng = np.random.default_rng(seed + n)
    if workload in POINT_WORKLOADS:
        points = _sample_points(dimension, n, workload, seed, queries=True)
        origins = points
        if case.metric == "angular":
            directions = rng.normal(size=points.shape).astype(np.float32)
            directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    elif builtin:
        points, directions = rng.normal(size=(2, n, dimension)).astype(np.float32)
        points *= 1.5
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        origins = points
    else:
        points, origins, directions = _triangle_queries(n, seed, workload)
        if workload == "near_surface":
            points = near_surface_points(
                arrays["vertices"], arrays["faces"], n, radius, seed
            )

    crossing = mode in ("count_hits", "all_hits")
    min_t = np.full(
        n, -np.inf if crossing and params.get("line", False) else 0.0, dtype=np.float32
    )
    if crossing and params.get("mixed_line", workload == "builtin"):
        min_t[::2] = -np.inf
    cap = radius
    if mode in ("closest_point", "nearest_selected"):
        cap = params.get("cap", radius if builtin else np.inf)
    elif mode == "knn":
        cap = params.get(
            "cap_squared" if primitive.startswith("points") else "cap", np.inf
        )
    elif mode == "sample_boundary" and workload == "builtin":
        cap = np.inf

    inputs = {}
    for name in CATALOG[mode].inputs:
        if name == "points":
            array = points
        elif name == "origins":
            array = origins
        elif name == "directions":
            array = directions
        elif name == "min_t":
            array = min_t
        elif name == "max_t":
            limit = 10.0 if not crossing or workload == "builtin" else np.inf
            array = np.full(n, params.get("max_t", limit), dtype=np.float32)
        elif name == "excluded":
            array = np.full(n, -1, dtype=np.int32)
        elif name == "radii":
            array = np.full(n, cap, dtype=np.float32)
        elif name == "lower":
            array = points - radius
        elif name == "upper":
            array = points + radius
        elif name == "randoms":
            array = (
                rng.random((n, 3), dtype=np.float32)
                if workload == "builtin"
                else np.random.default_rng(seed + 404 + n).random(
                    (n, 3), dtype=np.float32
                )
            )
        else:
            raise ValueError(f"no workload input generator for {name!r}")
        inputs[name] = np.ascontiguousarray(array)
    if case.metric == "angular":
        inputs["queries"] = np.stack((points, directions), axis=1)
    return inputs

"""Independent Warp mesh index and native query bindings."""

import hashlib
from pathlib import Path
from time import perf_counter

import numpy as np
import warp as wp

from ._timing import bind_kernel, describe, dtypes, measure


@wp.func
def finite3(v: wp.vec3) -> bool:
    return (
        v[0] == v[0]
        and v[1] == v[1]
        and v[2] == v[2]
        and wp.abs(v[0]) < wp.float32(3.4028234e38)
        and wp.abs(v[1]) < wp.float32(3.4028234e38)
        and wp.abs(v[2]) < wp.float32(3.4028234e38)
    )


@wp.func
def _valid_ray(o: wp.vec3, d: wp.vec3, lo: float, hi: float) -> bool:
    return (
        finite3(o)
        and finite3(d)
        and not wp.isnan(lo)
        and not wp.isnan(hi)
        and not wp.isinf(lo)
        and lo >= 0.0
        and hi >= lo
        and wp.dot(d, d) > 1.0e-30
    )


@wp.func
def _native_ray_result(
    o: wp.vec3, d: wp.vec3, lo: float, hi: float, mesh_id: wp.uint64
) -> tuple[float, int, int, int]:
    t = hi
    face = wp.int32(-1)
    hit = wp.int32(0)
    status = wp.int32(0)
    if not _valid_ray(o, d, lo, hi):
        status = -1
    else:
        q = wp.mesh_query_ray(mesh_id, o + lo * d, d, hi - lo)
        if q.result:
            t = q.t + lo
            face = q.face
            hit = 1
            status = 1
    return t, face, hit, status


@wp.kernel
def _native_first_hit(
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    min_t: wp.array(dtype=float),
    max_t: wp.array(dtype=float),
    mesh_id: wp.uint64,
    out_t: wp.array(dtype=float),
    out_id: wp.array(dtype=int),
    out_hit: wp.array(dtype=int),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    t, face, hit, status = _native_ray_result(
        origins[i], directions[i], min_t[i], max_t[i], mesh_id
    )
    out_t[i] = t
    out_id[i] = face
    out_hit[i] = hit
    out_status[i] = status


@wp.func_native(r"""
    return (wp::mesh_query_ray_count_intersections(mesh_id, point, wp::vec3(1.0f, 0.0f, 0.0f)) & 1)
        ? -1.0f : 1.0f;
""")
def _native_parity(mesh_id: wp.uint64, point: wp.vec3) -> float: ...


@wp.kernel
def _native_classify_point(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    out_sign: wp.array(dtype=float),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    out_sign[i] = wp.nan
    out_status[i] = -1
    if finite3(points[i]):
        out_sign[i] = _native_parity(mesh_id, points[i])
        out_status[i] = 1


@wp.func_native(r"""
    int count = wp::mesh_query_ray_count_intersections(mesh_id, origin, direction);
    if (line) count += wp::mesh_query_ray_count_intersections(mesh_id, origin, -direction);
    return count;
""")
def _native_count(
    mesh_id: wp.uint64, origin: wp.vec3, direction: wp.vec3, line: bool
) -> int: ...


@wp.kernel
def _native_count_hits(
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    min_t: wp.array(dtype=float),
    mesh_id: wp.uint64,
    out_count: wp.array(dtype=int),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    count = _native_count(mesh_id, origins[i], directions[i], min_t[i] < 0.0)
    out_count[i] = count
    out_status[i] = wp.int32(count > 0)


@wp.func
def _mesh_projection(p: wp.vec3, mesh_id: wp.uint64, search_cap: float):
    if not finite3(p):
        return wp.vec3(0.0), wp.inf, -1, -1
    q = wp.mesh_query_point_no_sign(mesh_id, p, search_cap)
    if q.result:
        point = wp.mesh_eval_position(mesh_id, q.face, q.u, q.v)
        return point, wp.length(point - p), q.face, 1
    return wp.vec3(0.0), wp.inf, -1, 0


@wp.func
def _capped_projection(p: wp.vec3, mesh_id: wp.uint64, cap: float):
    if not (cap >= 0.0):
        return wp.vec3(0.0), wp.inf, -1, -1
    # Warp tests the squared search cap strictly. Widen it by a few FP32 ulps
    # and keep a positive squared limit at zero, then enforce the closed cap.
    search_cap = cap + wp.max(cap * 2.384185791015625e-7, 2.168404344971009e-19)
    point, distance, face, status = _mesh_projection(p, mesh_id, search_cap)
    if status == 1 and distance > cap:
        return wp.vec3(0.0), wp.inf, -1, 0
    return point, distance, face, status


@wp.kernel
def _native_closest_point(
    points: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    out_point: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=float),
    out_id: wp.array(dtype=int),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    point, distance, face, status = _mesh_projection(points[i], mesh_id, wp.inf)
    out_point[i] = point
    out_distance[i] = distance
    out_id[i] = face
    out_status[i] = status


@wp.kernel
def _native_closest_capped(
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    mesh_id: wp.uint64,
    out_point: wp.array(dtype=wp.vec3),
    out_distance: wp.array(dtype=float),
    out_id: wp.array(dtype=int),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    point, distance, face, status = _capped_projection(points[i], mesh_id, radii[i])
    out_point[i] = point
    out_distance[i] = distance
    out_id[i] = face
    out_status[i] = status


@wp.struct
class Selected:
    point: wp.vec3
    distance: float
    id: int


@wp.kernel
def _native_selected(
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    mesh_id: wp.uint64,
    out_selected: wp.array(dtype=Selected),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    point, distance, face, status = _capped_projection(points[i], mesh_id, radii[i])
    out_selected[i] = Selected(point, distance, face)
    out_status[i] = status


@wp.kernel
def _native_any_hit(
    origins: wp.array(dtype=wp.vec3),
    directions: wp.array(dtype=wp.vec3),
    min_t: wp.array(dtype=float),
    max_t: wp.array(dtype=float),
    mesh_id: wp.uint64,
    out_hit: wp.array(dtype=int),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    o, d, lo, hi = origins[i], directions[i], min_t[i], max_t[i]
    out_hit[i] = 0
    out_status[i] = 0
    if not _valid_ray(o, d, lo, hi):
        out_status[i] = -1
    elif wp.mesh_query_ray_anyhit(mesh_id, o + lo * d, d, hi - lo):
        out_hit[i] = 1
        out_status[i] = 1


@wp.kernel
def _native_aabb_overlap(
    lower: wp.array(dtype=wp.vec3),
    upper: wp.array(dtype=wp.vec3),
    mesh_id: wp.uint64,
    out_hit: wp.array(dtype=int),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    query = wp.mesh_query_aabb(mesh_id, lower[i], upper[i])
    face = wp.int32(0)
    hit = int(wp.mesh_query_next(query, face))
    out_hit[i] = hit
    out_status[i] = hit


@wp.kernel
def _native_radius_any(
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    mesh_id: wp.uint64,
    out_hit: wp.array(dtype=int),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    query = wp.mesh_query_sphere(mesh_id, points[i], radii[i])
    face = wp.int32(0)
    hit = int(wp.mesh_query_next(query, face))
    out_hit[i] = hit
    out_status[i] = hit


@wp.kernel
def _native_radius_count(
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    mesh_id: wp.uint64,
    out_count: wp.array(dtype=int),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    query = wp.mesh_query_sphere(mesh_id, points[i], radii[i])
    face = wp.int32(0)
    count = wp.int32(0)
    while wp.mesh_query_next(query, face):
        count += 1
    out_count[i] = count
    out_status[i] = int(count > 0)


@wp.kernel
def _native_radius_accumulate(
    points: wp.array(dtype=wp.vec3),
    radii: wp.array(dtype=float),
    mesh_id: wp.uint64,
    out_value: wp.array(dtype=float),
    out_status: wp.array(dtype=int),
):
    i = wp.tid()
    query = wp.mesh_query_sphere(mesh_id, points[i], radii[i])
    face = wp.int32(0)
    total = wp.float32(0.0)
    while wp.mesh_query_next(query, face):
        total += float(face % 13 + 1)
    out_value[i] = total
    out_status[i] = int(total > 0.0)


# Native sphere iteration applies the exact triangle distance test.
_BINDINGS = {
    "first_hit": (
        _native_first_hit,
        ("origins", "directions", "min_t", "max_t"),
        ("distance", "id", "hit", "status"),
        128,
    ),
    "any_hit": (
        _native_any_hit,
        ("origins", "directions", "min_t", "max_t"),
        ("value", "status"),
        128,
    ),
    "count_hits": (
        _native_count_hits,
        ("origins", "directions", "min_t"),
        ("value", "status"),
        128,
    ),
    "closest_point": (
        _native_closest_point,
        ("points",),
        ("point", "distance", "id", "status"),
        64,
    ),
    "nearest_selected": (
        _native_selected,
        ("points", "radii"),
        ("point", "distance", "id", "status"),
        64,
    ),
    "classify_point": (_native_classify_point, ("points",), ("value", "status"), 128),
    "aabb_overlap": (
        _native_aabb_overlap,
        ("lower", "upper"),
        ("value", "status"),
        256,
    ),
    "radius_any": (_native_radius_any, ("points", "radii"), ("value", "status"), 64),
    "radius_count": (
        _native_radius_count,
        ("points", "radii"),
        ("value", "status"),
        64,
    ),
    "radius_accumulate": (
        _native_radius_accumulate,
        ("points", "radii"),
        ("value", "status"),
        64,
    ),
}


def support(case, primitive, closed):
    if primitive != "triangles" or case.query not in _BINDINGS:
        return "Warp Mesh has no equivalent binding for this query and primitive"
    if case.metric != "euclidean":
        return "Warp Mesh queries use Euclidean geometry"
    if case.query == "classify_point" and not closed:
        return "parity classification requires a closed mesh"
    if case.query == "count_hits" and (
        case.parameters.get("mixed_line")
        or case.workload == "builtin"
        or "max_t" in case.parameters
    ):
        return "Warp intersection counting uses unbounded rays or full lines"
    return None


def execute(geometry, jobs, options):
    if not jobs:
        return
    device = wp.get_device(options.device)
    start = perf_counter()
    vertices = wp.array(geometry.arrays["vertices"], dtype=wp.vec3, device=device)
    faces = wp.array(
        geometry.arrays["faces"].reshape(-1), dtype=wp.int32, device=device
    )
    wp.synchronize_device(device)
    transfer_ms = (perf_counter() - start) * 1000
    start = perf_counter()
    mesh = wp.Mesh(
        points=vertices,
        indices=faces,
        bvh_constructor=options.builder,
        bvh_leaf_size=options.leaf_size,
    )
    wp.synchronize_device(device)
    build_ms = (perf_counter() - start) * 1000
    header = Path(wp.__file__).parent / "native" / "mesh.h"
    mesh_header_sha256 = hashlib.sha256(header.read_bytes()).hexdigest()
    for job in jobs:
        kernel, input_names, output_names, block = _BINDINGS[job.case.query]
        if (
            job.case.query == "closest_point"
            and not np.isposinf(job.inputs["radii"]).all()
        ):
            kernel, input_names = _native_closest_capped, ("points", "radii")
        if job.case.query == "count_hits" and (
            np.any(job.inputs["excluded"] != -1)
            or not np.isposinf(job.inputs["max_t"]).all()
            or not np.isin(job.inputs["min_t"], [0, -np.inf]).all()
        ):
            raise ValueError(
                "Warp intersection counting requires unbounded rays or lines with no exclusions"
            )
        inputs = [*(job.inputs[key] for key in input_names), mesh.id]
        launch = bind_kernel(
            kernel, inputs, job.case.batch_size, device, options.block_dim or block
        )
        arrays, samples = measure(launch, options)
        if job.case.query == "nearest_selected":
            selected, status = arrays
            arrays = selected["point"], selected["distance"], selected["id"], status
        outputs = dict(zip(output_names, arrays, strict=True))
        precision = (
            "float32 with float64 predicates"
            if job.case.query
            in {"first_hit", "any_hit", "count_hits", "classify_point"}
            else "float32"
        )
        metadata = dict(
            describe(launch, options),
            build_ms=build_ms,
            build_scope=("CPU SAH build including device transfers" if options.builder == "sah"
                         else "GPU LBVH build from resident geometry"),
            geometry_transfer_ms=transfer_ms,
            builder=options.builder,
            leaf_size=options.leaf_size,
            mesh_header_sha256=mesh_header_sha256,
            precision={
                "input": dtypes(job.inputs),
                "output": dtypes(outputs),
                "internal": precision,
            },
        )
        if job.case.query == "radius_accumulate":
            metadata["composition"] = "native sphere iteration summing face_id % 13 + 1"
        if job.case.query == "nearest_selected":
            metadata["composition"] = (
                "native nearest mesh query emitting a selected result struct"
            )
        if job.case.query in {"closest_point", "nearest_selected"}:
            metadata["projection"] = (
                "Warp Mesh closest-point projection excludes sliver triangles"
            )
        yield job, outputs, samples, metadata

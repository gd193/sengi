"""Sengi query bindings over indexes and Warp arrays resident on the target device."""

from dataclasses import replace
from functools import lru_cache
from time import perf_counter
from typing import Any

import jax
import numpy as np
import warp as wp

from benchmarks.cases import CATALOG
from sengi import bvh
from sengi.boundary.robin import prepare_robin
from sengi.boundary.sampling import prepare_boundary_sample
from sengi.boundary.silhouette_edges import SilhouetteEdges
from sengi.bvh import Config
from sengi.bvh.device_tree import DeviceTree
from sengi.bvh.traversal.policy import make_traversal_state
from sengi.metrics.nearest import Angular, Euclidean
from sengi.primitives.items import TriangleItem3
from sengi.primitives.points import Points
from sengi.primitives.segments import Segments
from sengi.primitives.triangles import Triangles
from sengi.queries import (
    boundary_sample,
    classification,
    knn,
    nearest,
    overlap,
    radius,
    rays,
    region,
    region_classify,
    robin_radius,
    silhouette,
    star_radius,
)
from sengi.queries.results import ResultPolicy
from sengi.warp.device import configure_query_kernel
from sengi.warp.launch import BoundQuery, bind_query, struct_arrays

from ._timing import bind_kernel, describe, dtypes, measure

PRIMITIVES = {
    "points2": Points(2),
    "points3": Points(3),
    "segments2": Segments(2),
    "segments3": Segments(3),
    "triangles": Triangles(),
    "edges": SilhouetteEdges(),
}
METRICS = {"euclidean": Euclidean(), "angular": Angular()}
RAY_MODES = {
    "first_hit": "first",
    "any_hit": "any",
    "count_hits": "count",
    "all_hits": "all",
}
FEATURE_MODES = frozenset(("closest_silhouette", "star_radius", "robin_radius"))
BOUNDARY_MODES = FEATURE_MODES | {"classify_point", "classify_box", "sample_boundary"}
_BLOCK_DIMS = {
    "closest_point": 64,
    "nearest_selected": 64,
    "knn": 64,
    "radius_any": 64,
    "radius_count": 64,
    "radius_accumulate": 64,
    "aabb_overlap": 256,
    "region_any": 256,
    "region_count": 256,
    "classify_box": 128,
    "classify_point": 128,
    "closest_silhouette": 64,
    "star_radius": 64,
    "robin_radius": 64,
    "sample_boundary": 256,
    "first_hit": 128,
    "any_hit": 128,
    "count_hits": 128,
    "all_hits": 128,
}


def prepare(geometry, *, builder="sah", leaf_size=4):
    primitive = PRIMITIVES[geometry.primitive]
    arrays = geometry.arrays
    if geometry.primitive == "triangles":
        source = primitive.indexed(arrays["vertices"], arrays["faces"])
    elif "indices" in arrays:
        source = primitive.indexed(arrays["vertices"], arrays["indices"])
    elif geometry.primitive.startswith("points"):
        source = primitive.data(arrays["points"])
    elif geometry.primitive == "edges":
        source = primitive.data(
            arrays["starts"], arrays["ends"], arrays["normal0"], arrays["normal1"]
        )
    else:
        source = primitive.data(arrays["starts"], arrays["ends"])
    index = bvh.build(
        primitive, source, config=Config(leaf_size_threshold=leaf_size), builder=builder
    )
    return index, source


def prepare_boundary(index, source, modes):
    if modes & BOUNDARY_MODES:
        return bvh.prepare_boundary(
            index, source, silhouettes=bool(modes & FEATURE_MODES)
        )
    return None


def support(case, primitive, closed):
    query = case.query
    if query not in _BLOCK_DIMS or primitive not in PRIMITIVES:
        return "no Sengi binding for this query and primitive"
    spec = PRIMITIVES[primitive]
    if case.metric == "angular":
        if query not in {"closest_point", "closest_silhouette"}:
            return "the benchmark has no angular binding for this query"
        if query == "closest_point" and spec.segment_endpoints is None:
            return "angular nearest requires point or segment geometry"
    if query in RAY_MODES and spec.intersect is None:
        return "primitive has no ray intersection"
    if query in BOUNDARY_MODES and primitive not in ("segments2", "triangles"):
        return "primitive has no indexed boundary operations"
    if query in ("classify_point", "classify_box") and not closed:
        return "classification requires a closed boundary"
    if query.startswith("region_") and spec.region_policy is None:
        return "primitive has no exact region intersection"
    if query == "nearest_selected" and spec.dimension != 3:
        return "the representative selected result uses a 3D point"
    if query == "radius_accumulate" and primitive != "triangles":
        return "the representative weighted accumulator uses triangle face IDs"
    if query == "knn":
        try:
            knn.validate_batch_size(case.batch_size, case.parameters.get("k", 4))
        except ValueError as exc:
            return str(exc)
    return None


@wp.struct
class Selected:
    point: wp.vec3
    distance: float
    id: int


@wp.func
def _selected(query: wp.vec3, item: Any, index: int, point: wp.vec3, distance: float):
    return Selected(point, distance, index)


WeightState = make_traversal_state(
    "BenchmarkWeightState", {"matched": wp.bool, "total": wp.float32}
)


@wp.func
def _weight_init(query: wp.vec3, cap: float):
    return WeightState()


@wp.func
def _weight_update(
    query: wp.vec3,
    item: TriangleItem3,
    point: wp.vec3,
    score: float,
    index: int,
    state: wp.ref[WeightState],
):
    state.total += float(index % 13 + 1)


@wp.func
def _weight_finish(state: WeightState):
    return state.total


def bound_query(case, spec, block_dim=None):
    query, params = case.query, case.parameters
    block = block_dim or _BLOCK_DIMS[query]
    metric = METRICS[case.metric]
    if query == "closest_point":
        return nearest.bind(spec, metric, block_dim=block)
    if query == "nearest_selected":
        return nearest.bind(
            spec,
            metric,
            block_dim=block,
            result_policy=ResultPolicy(Selected, _selected),
        )
    if query == "knn":
        kwargs = {}
        if spec.primitive.point_bounds:
            ops = Euclidean().bind(spec.primitive)
            kwargs = {"evaluate": ops.closest, "lower_bound": ops.node_bound}
            metric = None
        for key in ("cache_bounds", "packed_nodes", "group_leaves"):
            if key in params:
                kwargs[key] = params[key]
        return knn.bind(spec, params.get("k", 4), metric, block_dim=block, **kwargs)
    if query in RAY_MODES:
        return rays.bind(spec, RAY_MODES[query], block_dim=block)
    if query == "radius_accumulate":
        accumulator = radius.RadiusAccumulator(
            WeightState, _weight_init, _weight_update, _weight_finish
        )
        return radius.bind(
            spec, metric, mode="accumulate", accumulator=accumulator, block_dim=block
        )
    if query in ("radius_any", "radius_count"):
        return radius.bind(spec, metric, mode=query.split("_")[1], block_dim=block)
    if query in ("region_any", "region_count"):
        return region.bind(
            spec,
            mode=query.split("_")[1],
            block_dim=block,
            count_limit=params.get("count_limit"),
        )
    if query == "closest_silhouette":
        return silhouette.bind(spec, metric, block_dim=block)
    binder = {
        "aabb_overlap": overlap.bind,
        "classify_point": classification.bind,
        "classify_box": region_classify.bind,
        "star_radius": star_radius.bind,
        "sample_boundary": boundary_sample.bind,
        "robin_radius": robin_radius.bind,
    }[query]
    return binder(spec, block_dim=block)


@lru_cache
def _composed_kernel(call, primitive, query, block):
    fn = call.device
    geometry = call.silhouette.query.primitive if query == "star_radius" else primitive
    point = primitive.point_type
    inputs = [("points", wp.array(dtype=point)), ("caps", wp.array(dtype=float))]
    outputs = [("values", wp.array(dtype=float)), ("statuses", wp.array(dtype=int))]
    body = [
        "value, status = wp.static(fn)(points[i], caps[i], tree, geometry)",
        "values[i] = value",
        "statuses[i] = status",
    ]
    bound = bind_query(
        BoundQuery,
        f"benchmark_{query}",
        geometry,
        inputs,
        outputs,
        body,
        {"fn": fn},
        block,
        device=fn,
    )
    configure_query_kernel(bound.kernel, call)
    return bound


def kernel_inputs(case, index, inputs, *, block_dim=None, boundary=None, sampling=None, robin=None):
    query = case.query
    block = block_dim or _BLOCK_DIMS[query]
    call = bound_query(case, index.spec, block)
    tree, geometry = index.data
    metadata, shapes = (), None
    values = [
        inputs["queries" if key == "points" and case.metric == "angular" else key]
        for key in CATALOG[query].inputs
    ]
    if query in ("all_hits", "knn"):
        n = len(values[0])
        if query == "all_hits":
            width = case.parameters.get(
                "max_hits", 4 if case.workload == "builtin" else 16
            )
        else:
            width = case.parameters.get("k", 4)
        shapes = ((n, width), (n, width), (n,), (n,))
    if query == "closest_silhouette":
        call = call.query
        geometry = boundary.silhouette
        tree = replace(
            tree,
            node_leaf_offset=geometry.node_feature_offset,
            node_leaf_count=geometry.node_feature_count,
            prim_ids=geometry.feature_refs,
        )
    elif query in ("star_radius", "radius_accumulate"):
        call = _composed_kernel(call, index.spec.primitive, query, block)
        if query == "star_radius":
            geometry = boundary.silhouette
    elif query in ("classify_point", "classify_box"):
        metadata = (boundary.core,)
    elif query == "sample_boundary":
        metadata = (sampling,)
    elif query == "robin_radius":
        metadata = (robin.core, robin.silhouette, robin.coefficients)
    flat = [
        *values,
        *struct_arrays(tree, DeviceTree),
        *struct_arrays(geometry, call.primitive.data_type),
        *(
            array
            for group, schema in zip(metadata, call.group_types, strict=True)
            for array in struct_arrays(group, schema)
        ),
    ]
    return call, flat, shapes


def _query_precision(case, primitive):
    query = case.query
    if case.metric == "angular":
        return "float32 with float64 angle evaluation"
    if query in ("region_count", "region_any"):
        return (
            "float32"
            if primitive.startswith("points")
            else "float32 with float64 primitive intersection"
        )
    if query in RAY_MODES or query in ("classify_point", "classify_box"):
        return "float32 with float64 predicates"
    if primitive == "triangles":
        if query == "sample_boundary":
            return "float32 with float64 triangle sampling"
        if query == "robin_radius":
            return "float32 with float64 fallback"
        if query in ("knn", "radius_any", "radius_count", "radius_accumulate"):
            return "float32 with float64 triangle scoring"
    return "float32"


def execute(geometry, jobs, options):
    if not jobs:
        return
    device = wp.get_device(options.device)
    jax_device = wp.device_to_jax(device)
    with wp.ScopedDevice(device), jax.default_device(jax_device):
        start = perf_counter()
        index, source = prepare(
            geometry, builder=options.builder, leaf_size=options.leaf_size
        )
        jax.block_until_ready(index.data)
        build_ms = (perf_counter() - start) * 1000
        modes = {job.case.query for job in jobs}
        start = perf_counter()
        boundary = prepare_boundary(index, source, modes)
        sampling = (
            prepare_boundary_sample(boundary) if "sample_boundary" in modes else None
        )
        robins = {
            mu: prepare_robin(boundary, mu)
            for mu in {
                job.case.parameters.get("robin_coefficient", 1.0)
                for job in jobs
                if job.case.query == "robin_radius"
            }
        }
        jax.block_until_ready((boundary, sampling, robins))
        boundary_ms = (perf_counter() - start) * 1000 if boundary is not None else None
        for job in jobs:
            case = job.case
            robin = robins.get(case.parameters.get("robin_coefficient", 1.0))
            call, inputs, shapes = kernel_inputs(
                case, index, job.inputs, block_dim=options.block_dim, boundary=boundary, sampling=sampling, robin=robin
            )
            launch = bind_kernel(
                call.kernel, inputs, case.batch_size, device, call.block_dim, shapes
            )
            arrays, samples = measure(launch, options)
            if case.query == "nearest_selected":
                selected, status = arrays
                arrays = (
                    selected["point"],
                    np.where(status == 1, selected["distance"], np.inf),
                    np.where(status == 1, selected["id"], -1),
                    status,
                )
            outputs = dict(zip(CATALOG[case.query].outputs, arrays, strict=True))
            metadata = dict(
                describe(launch, options),
                build_ms=build_ms,
                build_scope="index from host geometry, including transfers and any cold JAX compilation",
                builder=options.builder,
                jax_device_id=jax_device.id,
                leaf_size=options.leaf_size,
                precision={
                    "input": dtypes(job.inputs),
                    "output": dtypes(outputs),
                    "internal": _query_precision(case, geometry.primitive),
                },
            )
            if case.query in BOUNDARY_MODES and boundary_ms is not None:
                metadata["boundary_ms"] = boundary_ms
            if case.query == "radius_accumulate":
                metadata["composition"] = "radius accumulator summing face_id % 13 + 1"
            yield job, outputs, samples, metadata

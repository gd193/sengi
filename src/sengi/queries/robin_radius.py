"""Robin radius traversal and conservative callback policies."""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Generic, Protocol, TypeVar

import jax.numpy as jnp
import warp as wp

from sengi.warp.native import NativeHeaders, package_headers
from sengi.boundary.data import _adapter_policy, _check_functions
from sengi.boundary.robin import CoefficientMeta
from sengi.bvh.device_tree import DeviceTree, STACK_CAPACITY
from sengi.bvh.traversal.scheduling import make_walk
from sengi.metrics.nearest import Euclidean
from sengi.queries.nearest import prepare_nearest
from sengi.warp.arrays import load_array
from sengi.warp.contracts import make_struct
from sengi.warp.launch import BoundQuery, bind_query, cache_query


H = TypeVar("H")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
PointContra = TypeVar("PointContra", contravariant=True)
CH = TypeVar("CH")
Q = TypeVar("Q")
D = TypeVar("D")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")


@dataclass(frozen=True)
class RobinOps(Generic[Q, D, ItemT, CD, FD]):
    """Node bounds include features; node_visit returns (score, limit, skip_features, accepted). Scores default to radii."""
    query_type: Any
    query_shape: tuple
    valid: Any
    node_bound: Any
    primitive_bound: Any
    feature_bound: Any
    node_bound_with_data: Any = None
    feature_bound_with_data: Any = None
    core_type: Any = None
    feature_type: Any = None
    to_score: Any = None
    to_distance: Any = None
    leaf_bound_with_data: Any = None
    node_visit: Any = None
    native: NativeHeaders | None = None


@dataclass(frozen=True)
class _RobinHandlers:
    """Device callbacks selected from the policy's optional capabilities."""
    bounds: wp.Function
    feature: wp.Function
    leaf: wp.Function
    resume: wp.Function


class RobinPolicy(Protocol[D, ItemT, P, Q, CD, FD]):
    """Static Robin policy binding, shared by external and builtin callers."""

    def bind(self, primitive, metric): ...


class RobinMetric(Protocol[D, ItemT, PointContra, Q, CD, FD]):
    """Optional static metric capability for conservative Robin policies."""

    def bind_robin(self, primitive): ...


_ROBIN_HEADERS = package_headers(
    "queries", "robin.h", "silhouette.h",
    include_owners=("boundary",),
    dependencies=("primitives/native/triangle_distance.h", "primitives/native/determinant.h",
                  "metrics/native/box_distance.h", "metrics/native/rounding.h",
                  "primitives/native/segment_distance.h"),
)
_ROBIN_2D_HEADERS = package_headers(
    "boundary", "silhouette.h",
    dependencies=("primitives/native/segment_distance.h",),
)


@lru_cache(maxsize=None)
def _euclidean_robin(primitive):
    from sengi.boundary.data import BoundaryCore, SilhouetteMeta, cone_visibility

    Vec, dimension = primitive.point_type, primitive.dimension
    Geometry = primitive.data_type
    nearest = Euclidean().bind(primitive)
    leaf_with_data = visit = bound_with_data = primitive_bound = None

    @wp.func
    def valid(query: Vec):
        return wp.isfinite(query)

    @wp.func
    def node_bound(query: Vec, lower: wp.vec3, upper: wp.vec3, coefficients: wp.vec2):
        return wp.static(nearest.node_bound)(query, lower, upper)

    if dimension == 3:
        @wp.func_native("return bvh::robin_leaf(query, geometry, core, coefficients, offset, count, best);")
        def leaf_with_data(query: Vec, geometry: Geometry, core: BoundaryCore,
                           coefficients: CoefficientMeta, offset: int, count: int, best: float) -> float: ...

        @wp.func_native("bvh::robin_node(query, feature, meta, node, lower, upper, best, score, bound, skip, accepted);")
        def visit_values(query: Vec, tree: DeviceTree, geometry: Geometry, core: BoundaryCore,
                         feature: SilhouetteMeta, meta: CoefficientMeta, node: int,
                         lower: wp.vec3, upper: wp.vec3, best: float, score: wp.ref[float], bound: wp.ref[float],
                         skip: wp.ref[bool], accepted: wp.ref[bool]): ...

        @wp.func
        def visit(query: Vec, tree: DeviceTree, geometry: Geometry, core: BoundaryCore,
                  feature: SilhouetteMeta, meta: CoefficientMeta, node: int,
                  lower: wp.vec3, upper: wp.vec3, best: float):
            score, bound, skip, accepted = float(0.0), float(best), bool(False), bool(False)
            visit_values(query, tree, geometry, core, feature, meta, node, lower, upper, best, score, bound, skip, accepted)
            return score, bound, skip, accepted
    else:
        @wp.func
        def bound_with_data(query: Vec, tree: DeviceTree, geometry: Geometry, core: BoundaryCore,
                            feature: SilhouetteMeta, meta: CoefficientMeta, node: int):
            lower, upper = tree.node_lower[node], tree.node_upper[node]
            box2 = node_bound(query, lower, upper, meta.node_bounds[node])
            point = wp.vec3(query[0], query[1], 0.0)
            coefficient = meta.node_bounds[node][1]
            if coefficient == 0.0:
                return wp.where(cone_visibility(point, tree, feature, node), box2, wp.inf)
            angles = feature.node_cone_sin_cos[node]
            if angles[0] < 0.0 or angles[1] <= 0.0 or wp.isinf(coefficient):
                return box2
            view = .5*(lower+upper)-point
            axis = feature.node_cone_axis[node]
            projection = wp.dot(axis, view)
            extent = .5*(upper-lower)
            gap = wp.abs(projection)-wp.dot(wp.abs(axis), extent)
            if gap <= 0.0:
                return box2
            radius = feature.node_cone_radius[node]
            h = gap*angles[1]-angles[0]*(wp.length(view-projection*axis)+radius)-4e-6*(wp.length(view)+radius)
            if not (h > 0.0 and wp.isfinite(h)):
                return box2
            far = wp.max(wp.abs(lower-point), wp.abs(upper-point))
            distance2 = wp.clamp(2.0*(h/coefficient), box2, wp.dot(far, far))
            bound = distance2*wp.exp(2.0*h/(coefficient*distance2))*0.999996
            return wp.max(box2, bound)

        @wp.func
        def primitive_bound(query: Vec, item: Any, coefficient: float):
            if coefficient == 0.0:
                return wp.inf
            closest, near2 = wp.static(nearest.closest)(query, item)
            if not wp.isfinite(near2):
                return wp.nan
            if wp.isinf(coefficient) or near2 == 0.0:
                return near2
            edge = item.b-item.a
            normal = Vec(edge[1], -edge[0])/wp.length(edge)
            height = wp.abs(wp.dot(query-item.a, normal))
            if height == 0.0:
                return near2
            a, b = query-item.a, query-item.b
            far2 = wp.max(wp.dot(a, a), wp.dot(b, b))
            distance2 = wp.clamp(2.0*(height/coefficient), near2, far2)
            bound = distance2*wp.exp(2.0*height/(coefficient*distance2))*0.999996
            return wp.max(near2, bound)

    @wp.func
    def feature_bound(query: Vec, a: wp.vec3, b: wp.vec3, n0: wp.vec3, n1: wp.vec3):
        point = wp.vec3(0.0)
        for j in range(wp.static(dimension)):
            point[j] = query[j]
        view, edge = point-a, b-a
        side0, side1 = wp.dot(view, n0), wp.dot(view, n1)
        if (side0 < 0.0 and side1 < 0.0) or (side0 > 0.0 and side1 > 0.0):
            return wp.inf
        t, denominator = float(0.0), wp.dot(edge, edge)
        if denominator > 0.0:
            t = wp.clamp(wp.dot(view, edge)/denominator, 0.0, 1.0)
        delta = point-a-t*edge
        return wp.dot(delta, delta)

    return RobinOps(Vec, (dimension,), valid, node_bound, primitive_bound, feature_bound, bound_with_data,
                    to_score=nearest.to_score, to_distance=nearest.to_distance,
                    leaf_bound_with_data=leaf_with_data, node_visit=visit,
                    native=_ROBIN_HEADERS if dimension == 3 else _ROBIN_2D_HEADERS)


@dataclass(frozen=True)
class RobinQuery(BoundQuery, Generic[H, D, ItemT, Q, CH, CD, FH, FD]):
    ops: object
    handlers: _RobinHandlers

    def __call__(self, metadata, queries, cap=jnp.inf):
        tree, geometry = metadata.data
        queries, caps, shape = prepare_nearest(self.ops, queries, cap)
        return self.launch(tree, geometry, shape, queries, caps,
                           metadata=(metadata.core, metadata.silhouette, metadata.coefficients))


def bind(spec, policy=None, metric=None, *, adapter=None, core_host_type=None,
                      feature_host_type=None, stack_capacity=STACK_CAPACITY, block_dim=64):
    primitive = spec.primitive
    # Resolve defaults and bind policies before caching; host schema markers are typing only.
    metric = Euclidean() if metric is None else metric
    if adapter is not None and (policy is not None or not isinstance(metric, Euclidean)):
        raise TypeError("adapter requires the default Euclidean Robin policy")
    if policy is None and (adapter is not None or type(metric) is Euclidean):
        ops = _adapter_policy(primitive, "robin", adapter)
    elif policy is None:
        bind_robin = getattr(metric, "bind_robin", None)
        if bind_robin is None:
            raise TypeError("this metric requires explicit conservative RobinOps or bind_robin")
        ops = bind_robin(primitive)
    else:
        ops = policy
    if hasattr(ops, "bind"):
        ops = ops.bind(primitive, metric)
    return _make_robin_radius(primitive, ops, stack_capacity, block_dim)


@cache_query
def _make_robin_radius(primitive, ops, stack_capacity, block_dim):
    from sengi.boundary.data import BoundaryCore, SilhouetteMeta

    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    required = (ops.valid, ops.leaf_bound_with_data or ops.primitive_bound,
                ops.node_visit or ops.node_bound_with_data or ops.node_bound,
                ops.feature_bound_with_data or ops.feature_bound)
    if not all(callable(operation) for operation in required):
        raise TypeError("RobinOps requires validity, primitive, node and feature callbacks")
    _check_functions(ops.valid, ops.node_bound, ops.primitive_bound, ops.feature_bound,
                     ops.node_bound_with_data, ops.feature_bound_with_data, ops.to_score,
                     ops.to_distance, ops.leaf_bound_with_data, ops.node_visit)
    if (ops.to_score is None) != (ops.to_distance is None):
        raise TypeError("RobinOps score conversions must be supplied together")
    Core, Feature = ops.core_type or BoundaryCore, ops.feature_type or SilhouetteMeta
    for name in ("feature_refs", "node_feature_offset", "node_feature_count"):
        field = Feature.vars.get(name)
        if field is None or not isinstance(field.type, wp.array) or field.type.dtype is not wp.int32 or field.type.ndim != 1:
            raise TypeError(f"Robin feature metadata requires int32 array {name}")
    if ops.feature_bound_with_data is None and any(name not in Feature.vars for name in
            ("feature_a", "feature_b", "feature_n0", "feature_n1")):
        raise TypeError("custom feature metadata requires feature_bound_with_data")
    Query, Geometry = ops.query_type, primitive.data_type
    Context = make_struct("RobinContext", {"query": Query, "geometry": Geometry, "core": Core,
                                       "feature": Feature, "coefficients": CoefficientMeta})
    State = make_struct("RobinState", {"limit": wp.float32, "failed": wp.bool, "stop": wp.bool})

    if ops.node_visit is not None:
        @wp.func
        def bounds(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
            score, bound, skip, eligible = wp.static(ops.node_visit)(ctx.query, tree, ctx.geometry,
                ctx.core, ctx.feature, ctx.coefficients, node, lower, upper, state.limit)
            state.failed = wp.where(bound >= 0.0, state.failed, True)
            state.limit = wp.min(state.limit, bound)
            return score, eligible, skip
    elif ops.node_bound is not None and ops.node_bound_with_data is not None:
        @wp.func
        def bounds(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
            score = wp.static(ops.node_bound)(ctx.query, lower, upper, load_array(ctx.coefficients.node_bounds, node))
            if score <= state.limit:
                score = wp.static(ops.node_bound_with_data)(ctx.query, tree, ctx.geometry,
                    ctx.core, ctx.feature, ctx.coefficients, node)
            return score, score < wp.inf, False
    elif ops.node_bound_with_data is not None:
        @wp.func
        def bounds(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
            score = wp.static(ops.node_bound_with_data)(ctx.query, tree, ctx.geometry,
                ctx.core, ctx.feature, ctx.coefficients, node)
            return score, score < wp.inf, False
    else:
        @wp.func
        def bounds(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
            score = wp.static(ops.node_bound)(ctx.query, lower, upper, load_array(ctx.coefficients.node_bounds, node))
            return score, score < wp.inf, False

    if ops.feature_bound_with_data is not None:
        @wp.func
        def feature_candidate(ctx: Context, tree: DeviceTree, node: int, index: int):
            return wp.static(ops.feature_bound_with_data)(ctx.query, tree, ctx.geometry,
                ctx.core, ctx.feature, ctx.coefficients, node, index)
    else:
        @wp.func
        def feature_candidate(ctx: Context, tree: DeviceTree, node: int, index: int):
            return wp.static(ops.feature_bound)(ctx.query,
                load_array(ctx.feature.feature_a, index), load_array(ctx.feature.feature_b, index),
                load_array(ctx.feature.feature_n0, index), load_array(ctx.feature.feature_n1, index))

    if ops.leaf_bound_with_data is not None:
        @wp.func
        def visit_leaf(ctx: Context, begin: int, end: int, state: wp.ref[State]):
            candidate = wp.static(ops.leaf_bound_with_data)(ctx.query, ctx.geometry, ctx.core,
                ctx.coefficients, begin, end-begin, state.limit)
            state.failed = wp.where(candidate >= 0.0, state.failed, True)
            state.limit = wp.min(state.limit, candidate)
    else:
        @wp.func
        def visit_leaf(ctx: Context, begin: int, end: int, state: wp.ref[State]):
            for slot in range(begin, end):
                candidate = wp.static(ops.primitive_bound)(ctx.query, wp.static(primitive.load)(ctx.geometry, slot),
                    load_array(ctx.coefficients.primitive_bounds, slot)[1])
                state.failed = wp.where(candidate >= 0.0, state.failed, True)
                state.limit = wp.min(state.limit, candidate)

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, node: int, begin: int, end: int, skip: bool, state: wp.ref[State]):
        if not skip:
            offset = load_array(ctx.feature.node_feature_offset, node)
            for j in range(load_array(ctx.feature.node_feature_count, node)):
                index = load_array(ctx.feature.feature_refs, offset+j)
                candidate = wp.static(feature_candidate)(ctx, tree, node, index)
                state.failed = wp.where(candidate >= 0.0, state.failed, True)
                state.limit = wp.min(state.limit, candidate)
        wp.static(visit_leaf)(ctx, begin, end, state)
        state.stop = state.limit == 0.0
        return

    if ops.node_bound is not None:
        @wp.func
        def resume(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3):
            return wp.static(ops.node_bound)(ctx.query, lower, upper, load_array(ctx.coefficients.node_bounds, node))
    else:
        @wp.func
        def resume(ctx: Context, tree: DeviceTree, node: int, lower: wp.vec3, upper: wp.vec3):
            return -wp.inf

    handlers = _RobinHandlers(bounds, feature_candidate, visit_leaf, resume)
    walk = make_walk(Context, State, handlers.bounds, visit, stack_capacity=stack_capacity,
                     resume=handlers.resume)

    @wp.func
    def radius(query: Query, cap: float, tree: DeviceTree, geometry: Geometry,
               core: Core, feature: Feature, meta: CoefficientMeta):
        if not wp.static(ops.valid)(query) or not (cap >= 0.0):
            return wp.nan, -1
        if cap == 0.0 or tree.node_left.shape[0] == 0:
            return cap, 1
        ctx = Context()
        ctx.query = query
        ctx.geometry = geometry
        ctx.core = core
        ctx.feature = feature
        ctx.coefficients = meta
        state = State()
        state.limit = cap
        if wp.static(ops.to_score is not None):
            state.limit = wp.static(ops.to_score)(cap)
        if state.limit != state.limit:
            return wp.nan, -1
        wp.static(walk)(tree, ctx, state)
        if state.failed:
            return wp.nan, -1
        value = state.limit
        if wp.static(ops.to_distance is not None):
            value = wp.static(ops.to_distance)(value)
        if not (value >= 0.0):
            return wp.nan, -1
        return wp.min(cap, value), 1

    inputs = [("queries", wp.array(dtype=Query)), ("caps", wp.array(dtype=wp.float32))]
    outputs = [("out_radius", wp.array(dtype=wp.float32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["value, status = wp.static(radius)(load_array(queries, i), load_array(caps, i), tree, geometry, core, feature, meta)",
            "store_array(out_radius, i, value)", "store_array(out_status, i, status)"]
    return bind_query(RobinQuery, "robin_radius_kernel", primitive, inputs, outputs, body, {"radius": radius},
        block_dim, device=radius, ops=ops, handlers=handlers, native_providers=(ops,),
        groups=(("core", Core), ("feature", Feature), ("meta", CoefficientMeta)),
        components=(("node", "visit" if ops.node_visit is not None else
                     "coarse_and_data" if ops.node_bound is not None and ops.node_bound_with_data is not None else
                     "data" if ops.node_bound_with_data is not None else "coarse"),
                    ("feature", "data" if ops.feature_bound_with_data is not None else "geometry"),
                    ("leaf", "data" if ops.leaf_bound_with_data is not None else "primitive")))

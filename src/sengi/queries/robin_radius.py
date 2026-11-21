"""Robin radius traversal and conservative callback policies."""
from dataclasses import dataclass
from typing import Any, Generic, Protocol, TypeVar

import jax.numpy as jnp
import warp as wp

from sengi.warp.native import NativeHeaders
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

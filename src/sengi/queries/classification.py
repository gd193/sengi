"""Even-odd classification with half-open +X ray intersections."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import warp as wp

from sengi.warp.launch import BoundQuery, bind_query, cache_query
from sengi.jax.ffi import _flatten_vec_queries
from sengi.warp.contracts import make_struct
from sengi.warp.arrays import load_array
from sengi.bvh.traversal.scheduling import make_walk
from sengi.bvh.device_tree import DeviceTree, STACK_CAPACITY


HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
MetaT = TypeVar("MetaT")
HostMetaT = TypeVar("HostMetaT")


@dataclass(frozen=True)
class ClassifyQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, HostMetaT, MetaT]):
    policy: object
    metadata_host_type: type | None

    def __call__(self, boundary, points):
        if self.metadata_host_type is not None and not isinstance(boundary.core, self.metadata_host_type):
            raise TypeError(f"boundary core must be an instance of {self.metadata_host_type.__name__}")
        tree, geometry = boundary.data
        queries, shape = _flatten_vec_queries(points, self.primitive.dimension)
        return self.launch(tree, geometry, shape, queries, metadata=(boundary.core,))


def bind(spec, policy=None, *, adapter=None, metadata_host_type=None, stack_capacity=STACK_CAPACITY, block_dim=128):
    if policy is not None and adapter is not None:
        raise TypeError("supply a classification policy or an adapter")
    primitive = spec.primitive
    return _make_classify(primitive, policy, adapter=adapter, metadata_host_type=metadata_host_type,
                          stack_capacity=stack_capacity, block_dim=block_dim)


@cache_query
def _make_classify(primitive, policy=None, *, adapter=None, metadata_host_type=None,
                   stack_capacity=STACK_CAPACITY, block_dim=128):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    if policy is None:
        from sengi.boundary.data import resolve_boundary_adapter
        adapter = resolve_boundary_adapter(primitive, adapter)
        if adapter.classification is None:
            raise TypeError("boundary adapter has no classification operation")
        policy = adapter.classification(primitive)
    Vec, Geometry, Meta = primitive.point_type, primitive.data_type, policy.metadata_type
    Context = make_struct("ClassifyContext", {"point": Vec, "geometry": Geometry, "meta": Meta})
    State = make_struct("ClassifyState", {"limit": float, "failed": bool, "stop": bool,
                                     "parity": int, "contact": bool})
    dimension = primitive.dimension

    @wp.func
    def bounds(ctx: Context, lo: wp.vec3, hi: wp.vec3, state: wp.ref[State]):
        eligible = ctx.point[0] <= hi[0]
        for axis in range(1, wp.static(dimension)):
            eligible = eligible and ctx.point[axis] >= lo[axis] and ctx.point[axis] <= hi[axis]
        return 0.0, eligible, False

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, begin: int, end: int, state: wp.ref[State]):
        for slot in range(begin, end):
            event = wp.static(policy.event)(ctx.point, ctx.geometry, slot, load_array(tree.prim_ids, slot), ctx.meta)
            if event < 0 or event > 2:
                state.failed = True
                return
            if event == 2:
                state.contact = True
                state.stop = True
                return
            state.parity = state.parity ^ event
        return

    walk = make_walk(Context, State, bounds, visit, stack_capacity=stack_capacity, near_first=False, packed=True)

    @wp.func
    def classify(point: Vec, tree: DeviceTree, geometry: Geometry, meta: Meta):
        if (not wp.isfinite(point) or wp.max(wp.abs(point)) >= 3.4028234e38
                or not wp.static(policy.valid)(meta)):
            return wp.nan, -1
        if tree.node_left.shape[0] == 0:
            return 1.0, 0
        ctx = Context(point, geometry, meta)
        state = State()
        state.limit = 0.0
        wp.static(walk)(tree, ctx, state)
        if state.failed:
            return wp.nan, -1
        return wp.where(state.contact, 0.0, wp.where(state.parity != 0, -1.0, 1.0)), 1

    inputs = [("queries", wp.array(dtype=Vec))]
    outputs = [("out_sign", wp.array(dtype=wp.float32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["sign, status = wp.static(classify)(load_array(queries, i), tree, geometry, meta)",
            "store_array(out_sign, i, sign)", "store_array(out_status, i, status)"]
    return bind_query(ClassifyQuery, "classify_kernel", primitive, inputs, outputs, body,
                      {"classify": classify}, block_dim, device=classify, groups=(("meta", Meta),), policy=policy,
                      metadata_host_type=metadata_host_type,
                      native_providers=(policy,), components=(("event", "policy"),))

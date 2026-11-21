"""Exact primitive versus closed-box queries with static intersection policies."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import warp as wp

from sengi.warp.native import NativeHeaders
from sengi.warp.contracts import WarpFunction, make_struct
from sengi.jax.ffi import _flatten_vec_queries
from sengi.warp.launch import BoundQuery, bind_query, cache_query
from sengi.primitives.geometry import boxes_overlap
from sengi.bvh.traversal.policy import make_traversal
from sengi.bvh.device_tree import DeviceTree, STACK_CAPACITY

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
PointT = TypeVar("PointT")


@dataclass(frozen=True)
class RegionIntersectionPolicy(Generic[DeviceDataT]):
    """Static callback intersects(geometry, packed_slot, box_lower, box_upper)."""
    intersects: WarpFunction[[DeviceDataT, int, wp.vec3, wp.vec3], bool]
    native: NativeHeaders | None = None

    def __post_init__(self):
        if not isinstance(self.intersects, wp.Function):
            raise TypeError("intersection callback must be a decorated Warp function")


@dataclass(frozen=True)
class RegionQuery(BoundQuery, Generic[HostDataT, DeviceDataT, PointT]):
    mode: str
    def __call__(self, data, lower, upper):
        tree, geometry = data
        lower, shape = _flatten_vec_queries(lower, self.primitive.dimension)
        upper, other = _flatten_vec_queries(upper, self.primitive.dimension)
        if shape != other:
            raise ValueError("box lower and upper batch shapes must match")
        result, status = self.launch(tree, geometry, shape, lower, upper)
        if self.mode == "any":
            result = result.astype(bool)
        return result, status


def _resolve_policy(primitive, policy):
    if policy is None:
        policy = getattr(primitive, "region_policy", None)
        if policy is None:
            raise TypeError("exact region queries require a region intersection policy")
    return policy


def bind(spec, policy: RegionIntersectionPolicy | None = None, *, mode="any", count_limit=None,
                stack_capacity=STACK_CAPACITY, block_dim=256):
    """Count exact hits or test for any exact hit; a limit returns a saturated count."""
    primitive = spec.primitive
    if mode not in ("any", "count"):
        raise ValueError("mode must be 'any' or 'count'")
    if count_limit is not None and (mode != "count" or type(count_limit) is not int or not 1 <= count_limit <= 2147483647):
        raise ValueError("count_limit requires count mode and a positive static int32 integer")
    policy = _resolve_policy(primitive, policy)
    return _make_region(primitive, policy, mode, count_limit, stack_capacity, block_dim)


@cache_query
def _make_region(primitive, policy, mode, count_limit, stack_capacity, block_dim):
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    Vec, Geometry = primitive.point_type, primitive.data_type
    Context = make_struct("RegionContext", {"lower": wp.vec3, "upper": wp.vec3, "geometry": Geometry})
    State = make_struct("RegionState", {"limit": float, "failed": bool, "stop": bool, "count": int})

    @wp.func
    def bounds(ctx: Context, tree: DeviceTree, node: int, lo: wp.vec3, hi: wp.vec3, state: wp.ref[State]):
        return 0.0, boxes_overlap(ctx.lower, ctx.upper, lo, hi), False

    @wp.func
    def visit(ctx: Context, tree: DeviceTree, node: int, begin: int, end: int, tag: bool, state: wp.ref[State]):
        for slot in range(begin, end):
            if wp.static(policy.intersects)(ctx.geometry, slot, ctx.lower, ctx.upper):
                if state.count == 2147483647:
                    state.failed = True
                    return
                state.count += 1
                if wp.static(mode == "any"):
                    state.stop = True
                    return
                if wp.static(count_limit is not None):
                    if state.count >= wp.static(count_limit if count_limit is not None else 0):
                        state.stop = True
                        return
        return

    walk = make_traversal(Context, State, bounds, visit, stack_capacity=stack_capacity,
                          near_first=False, packed_nodes=True)

    @wp.func
    def region(lower: Vec, upper: Vec, tree: DeviceTree, geometry: Geometry):
        if not wp.isfinite(lower) or not wp.isfinite(upper) or wp.max(lower - upper) > 0.0:
            return 0, -1
        lo = wp.vec3(lower[0], lower[1], 0.0)
        hi = wp.vec3(upper[0], upper[1], 0.0)
        if wp.static(primitive.dimension == 3):
            lo = wp.vec3(lower[0], lower[1], lower[2])
            hi = wp.vec3(upper[0], upper[1], upper[2])
        state = State()
        wp.static(walk.device)(tree, Context(lo, hi, geometry), state)
        if state.failed:
            return 0, -1
        return state.count, 1

    inputs = [("lower", wp.array(dtype=Vec)), ("upper", wp.array(dtype=Vec))]
    outputs = [("out_result", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["result, status = wp.static(region)(load_array(lower, i), load_array(upper, i), tree, geometry)",
            "store_array(out_result, i, result)", "store_array(out_status, i, status)"]
    return bind_query(RegionQuery, "region_kernel", primitive, inputs, outputs, body,
                      {"region": region}, block_dim, device=region, mode=mode,
                      native_providers=(policy,), components=(("intersection", "exact_policy"), ("result", mode)))

"""Ray queries using the shared BVH walker."""
from dataclasses import dataclass
from operator import index
from typing import Any, Generic, TypeVar

import jax.numpy as jnp
import warp as wp

from sengi.warp.native import package_headers
from sengi.queries.domain import Ray, RayOps, prepare_ray
from sengi.warp.launch import BoundQuery, bind_query, cache_query
from sengi.jax.ffi import _ray_args
from sengi.warp.contracts import make_struct
from sengi.warp.arrays import load_array, store_array
from sengi.bvh.traversal.scheduling import make_walk
from sengi.bvh.device_tree import DeviceTree, STACK_CAPACITY


HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
ItemT = TypeVar("ItemT")
PointT = TypeVar("PointT")
ModeT = TypeVar("ModeT", bound=str)


@dataclass(frozen=True)
class RayQuery(BoundQuery, Generic[HostDataT, DeviceDataT, ItemT, PointT, ModeT]):
    mode: str
    exclusion: bool

    def __call__(self, data, origin, direction, max_t=jnp.inf, *,
                 min_t=0.0, exclude_face=None, max_hits=None):
        tree, geometry = data
        if exclude_face is not None and not self.exclusion:
            raise ValueError("construct this ray query with exclude=True")
        if self.exclusion and exclude_face is None:
            exclude_face = -1
        args, shape = _ray_args(origin, direction, min_t, max_t,
                                self.primitive.dimension, exclude_face)
        shapes = None
        if self.mode == "all":
            max_hits = index(max_hits)
            if not 1 <= max_hits < 2**31:
                raise ValueError("max_hits must be a positive static int32 capacity")
            shapes = ((max_hits,), (max_hits,), (), ())
        elif max_hits is not None:
            raise ValueError("max_hits requires an all-hit query")
        result = self.launch(tree, geometry, shape, *args, output_shapes=shapes)
        if self.mode == "first":
            return result[0], result[1], result[2].astype(jnp.bool_), result[3]
        if self.mode == "any":
            return result[0].astype(jnp.bool_), result[1]
        return result


_RAY_HEADERS = package_headers("queries", "rays.h")


@wp.func_native("return bvh::ray_box(ray, lower, upper, minimum, maximum, dimension);")
def ray_box(ray: Ray, lower: wp.vec3, upper: wp.vec3,
            minimum: float, maximum: float, dimension: int) -> wp.vec2: ...


@wp.func_native("return bvh::first_ray_inverse(ray);")
def first_ray_inverse(ray: Ray) -> wp.vec3: ...


@wp.func_native("return bvh::first_ray_box(ray, inverse, lower, upper, minimum, maximum);")
def first_ray_box(ray: Ray, inverse: wp.vec3, lower: wp.vec3, upper: wp.vec3,
                  minimum: float, maximum: float) -> wp.vec2: ...


@wp.func_native("return bvh::valid_ray(origin, direction, minimum, maximum, excluded, count, line, exclusion);")
def valid_ray(origin: Any, direction: Any, minimum: float, maximum: float,
              excluded: int, count: int, line: bool, exclusion: bool):
    return bool(False)  # Generic native return-type inference.


def bind(spec, mode="first", *, ops=None, exclude=False,
                   stack_capacity=STACK_CAPACITY, block_dim=128):
    primitive = spec.primitive
    ops = RayOps(intersect=getattr(primitive, "intersect", None)) if ops is None else ops
    return _make_ray_query(primitive, mode, ops, exclude, stack_capacity, block_dim)


@cache_query
def _make_ray_query(primitive, mode, ops, exclude, stack_capacity, block_dim):
    if mode not in ("first", "any", "count", "all"):
        raise ValueError("ray mode must be first, any, count, or all")
    if not 1 <= stack_capacity <= STACK_CAPACITY or not 1 <= block_dim <= 1024:
        raise ValueError("invalid stack capacity or block dimension")
    if ops.intersect is None and ops.next_hit is None:
        raise TypeError("primitive has no ray intersection operation")
    multiple = ops.next_hit is not None and (mode in ("count", "all") or ops.intersect is None)
    Iterator = ops.state_type
    exclusion, half_open = exclude or mode in ("count", "all"), mode in ("count", "all")
    store_times = mode in ("first", "all")
    Vec, Geometry = primitive.point_type, primitive.data_type
    fields = {"ray": Ray, "minimum": wp.float32, "excluded": wp.int32, "geometry": Geometry, "qi": wp.int32}
    forward_3d = mode == "first" and primitive.dimension == 3
    if forward_3d:
        fields["inverse"] = wp.vec3
    if mode == "all":
        fields.update(times=wp.array2d(dtype=wp.float32), faces=wp.array2d(dtype=wp.int32))
    Context = make_struct("RayContext", fields)
    state_fields = {"limit": wp.float32, "failed": wp.bool, "stop": wp.bool,
                    "count": wp.int32, "index": wp.int32}
    State = make_struct("RayState", state_fields)

    @wp.func
    def bounds(ctx: Context, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
        hit = ray_box(ctx.ray, lower, upper, ctx.minimum, state.limit, wp.static(primitive.dimension))
        return hit[0], hit[1] > 0.0, False

    # Resolve the accumulation rule and intersection capability in Python.  Each
    # generated visit function has one intersection algorithm and one emitter.
    if mode == "any":
        @wp.func
        def emit(ctx: Context, state: wp.ref[State], hit: wp.vec2, face: int):
            state.count = 1
            state.stop = True
    elif mode == "first":
        @wp.func
        def emit(ctx: Context, state: wp.ref[State], hit: wp.vec2, face: int):
            if hit[0] < state.limit or state.index < 0:
                state.limit = hit[0]
                state.index = face
    elif mode == "all":
        @wp.func
        def emit(ctx: Context, state: wp.ref[State], hit: wp.vec2, face: int):
            if state.count < ctx.times.shape[1]:
                store_array(ctx.times, wp.int64(ctx.qi)*wp.int64(ctx.times.shape[1])+wp.int64(state.count), hit[0])
                store_array(ctx.faces, wp.int64(ctx.qi)*wp.int64(ctx.faces.shape[1])+wp.int64(state.count), face)
            state.count += 1
    else:
        @wp.func
        def emit(ctx: Context, state: wp.ref[State], hit: wp.vec2, face: int):
            state.count += 1

    if multiple:
        @wp.func
        def visit(ctx: Context, tree: DeviceTree, begin: int, end: int, state: wp.ref[State]):
            for slot in range(begin, end):
                face = load_array(tree.prim_ids, slot)
                if wp.static(exclusion):
                    if face == ctx.excluded:
                        continue
                if wp.static(wp.types.type_is_scalar(Iterator)):
                    iterator = Iterator(0)
                else:
                    iterator = Iterator()
                while True:
                    iterator, t, flag = wp.static(ops.next_hit)(ctx.ray, ctx.minimum, state.limit,
                        ctx.geometry, slot, iterator, wp.static(half_open), wp.static(store_times))
                    if flag == 0:
                        break
                    if flag < 0:
                        state.failed = True
                        return
                    wp.static(emit)(ctx, state, wp.vec2(t, wp.float32(flag)), face)
                    if state.stop:
                        return
            return
    else:
        @wp.func
        def visit(ctx: Context, tree: DeviceTree, begin: int, end: int, state: wp.ref[State]):
            for slot in range(begin, end):
                face = load_array(tree.prim_ids, slot)
                if wp.static(exclusion):
                    if face == ctx.excluded:
                        continue
                hit = wp.static(ops.intersect)(ctx.ray, ctx.minimum, state.limit, ctx.geometry,
                                               slot, wp.static(half_open), wp.static(store_times))
                if hit[1] < 0.0:
                    state.failed = True
                    return
                if hit[1] > 0.0:
                    wp.static(emit)(ctx, state, hit, face)
                    if state.stop:
                        return
            return

    group_leaves = mode == "first" and getattr(primitive, "group_leaves", False)

    def make_ray_walk(box_test):
        return make_walk(Context, State, box_test, visit, stack_capacity=stack_capacity,
                         near_first=mode in ("first", "any"), packed=True,
                         test_root=half_open, group_leaves=group_leaves)

    walk = make_ray_walk(bounds)
    if forward_3d:
        @wp.func
        def first_bounds(ctx: Context, lower: wp.vec3, upper: wp.vec3, state: wp.ref[State]):
            hit = first_ray_box(ctx.ray, ctx.inverse, lower, upper, ctx.minimum, state.limit)
            return hit[0], hit[1] > 0.0, False

        first_walk = make_ray_walk(first_bounds)

    @wp.func
    def ray_query(qi: int, origin: Vec, direction: Vec, minimum: float, maximum: float,
                  excluded: int, tree: DeviceTree, geometry: Geometry,
                  times: Any, faces: Any, out_count: Any, out_status: Any):
        store_array(out_count, qi, 0)
        store_array(out_status, qi, 0)
        if wp.static(mode == "first"):
            store_array(times, qi, maximum)
            store_array(faces, qi, -1)
        if wp.static(mode == "all"):
            for j in range(times.shape[1]):
                store_array(times, wp.int64(qi)*wp.int64(times.shape[1])+wp.int64(j), wp.inf)
                store_array(faces, wp.int64(qi)*wp.int64(faces.shape[1])+wp.int64(j), -1)
        valid = valid_ray(origin, direction, minimum, maximum, excluded,
                          tree.prim_ids.shape[0], wp.static(half_open), wp.static(exclusion))
        state = State()
        state.limit = maximum
        state.index = -1
        if valid:
            ctx = Context()
            o, d = wp.vec3(origin[0], origin[1], 0.0), wp.vec3(direction[0], direction[1], 0.0)
            if wp.static(primitive.dimension == 3):
                o[2] = origin[2]
                d[2] = direction[2]
            ctx.ray = prepare_ray(o, d, wp.static(primitive.dimension))
            if wp.static(forward_3d):
                ctx.inverse = first_ray_inverse(ctx.ray)
            ctx.minimum = minimum
            ctx.excluded = excluded
            ctx.geometry = geometry
            ctx.qi = qi
            if wp.static(mode == "all"):
                ctx.times = times
                ctx.faces = faces
            if wp.static(forward_3d):
                # A tiny nonzero direction needs division-based slabs for the entire ray.
                if wp.isnan(ctx.inverse[0]):
                    wp.static(walk)(tree, ctx, state)
                else:
                    wp.static(first_walk)(tree, ctx, state)
            else:
                wp.static(walk)(tree, ctx, state)
        else:
            state.failed = True
        if state.failed:
            store_array(out_status, qi, -1)
            if wp.static(half_open):
                store_array(out_count, qi, -1)
            if wp.static(mode == "all"):
                for j in range(times.shape[1]):
                    store_array(times, wp.int64(qi)*wp.int64(times.shape[1])+wp.int64(j), wp.inf)
                    store_array(faces, wp.int64(qi)*wp.int64(faces.shape[1])+wp.int64(j), -1)
        else:
            if wp.static(mode == "first"):
                if state.index >= 0:
                    store_array(times, qi, state.limit)
                    store_array(faces, qi, state.index)
                    state.count = 1
            store_array(out_count, qi, state.count)
            store_array(out_status, qi, wp.int32(state.count > 0))
            if wp.static(mode == "all"):
                if state.count > times.shape[1]:
                    store_array(out_status, qi, -2)

    if store_times:
        Times = wp.array2d(dtype=wp.float32) if mode == "all" else wp.array(dtype=wp.float32)
        Faces = wp.array2d(dtype=wp.int32) if mode == "all" else wp.array(dtype=wp.int32)

        @wp.func
        def device(origin: Vec, direction: Vec, minimum: float, maximum: float,
                   excluded: int, tree: DeviceTree, geometry: Geometry,
                   times: Times, faces: Faces, counts: wp.array(dtype=wp.int32),
                   statuses: wp.array(dtype=wp.int32), row: int):
            wp.static(ray_query)(row, origin, direction, minimum, maximum, excluded,
                                 tree, geometry, times, faces, counts, statuses)
    else:
        @wp.func
        def device(origin: Vec, direction: Vec, minimum: float, maximum: float,
                   excluded: int, tree: DeviceTree, geometry: Geometry,
                   counts: wp.array(dtype=wp.int32), statuses: wp.array(dtype=wp.int32), row: int):
            wp.static(ray_query)(row, origin, direction, minimum, maximum, excluded,
                                 tree, geometry, counts, statuses, counts, statuses)

    outputs = [("out_count", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    times, faces = "out_count", "out_status"
    if store_times:
        array = wp.array2d if mode == "all" else wp.array
        outputs = [("out_t", array(dtype=wp.float32)), ("out_face", array(dtype=wp.int32)), *outputs]
        times, faces = "out_t", "out_face"
    inputs = [("origins", wp.array(dtype=Vec)), ("directions", wp.array(dtype=Vec)),
              ("minimum", wp.array(dtype=wp.float32)), ("maximum", wp.array(dtype=wp.float32))]
    if exclusion:
        inputs.append(("excluded", wp.array(dtype=wp.int32)))
    excluded = "load_array(excluded, i)" if exclusion else "-1"
    body = [f"wp.static(ray_query)(i, load_array(origins, i), load_array(directions, i), load_array(minimum, i), load_array(maximum, i), {excluded}, "
            f"tree, geometry, {times}, {faces}, out_count, out_status)"]
    kernel_options = {}
    if (group_leaves and forward_3d and block_dim == 128 and not exclusion
            and ops.intersect is getattr(primitive, "intersect", None) and not multiple):
        kernel_options = {"enable_cuda_smem_spilling": True}
    return bind_query(RayQuery, "ray_kernel", primitive, inputs, outputs, body,
                      {"ray_query": ray_query}, block_dim, device=device, mode=mode, exclusion=exclusion,
                      kernel_options=kernel_options, native_providers=(ops,), native_requirements=(_RAY_HEADERS._build_options(),),
                      components=(("mode", mode),
                                  ("hit_evaluation", "stateful" if multiple else "single"),
                                  ("output", "times_and_faces" if store_times else "count_and_status")))

"""Bounded closest silhouette with box-support cone pruning."""
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import Generic, TypeVar

import jax.numpy as jnp
import warp as wp

from sengi.bvh.device_tree import DeviceTree, STACK_CAPACITY, _IndexSchema
from sengi.bvh.index import IndexData, IndexSpec
from sengi.warp.launch import cache_query


H = TypeVar("H")
D = TypeVar("D")
ItemT = TypeVar("ItemT")
P = TypeVar("P")
Q = TypeVar("Q")
CH = TypeVar("CH")
CD = TypeVar("CD")
FH = TypeVar("FH")
FD = TypeVar("FD")
FI = TypeVar("FI")


@dataclass(frozen=True)
class _SilhouetteMetric:
    metric: object
    visibility: object

    @lru_cache(maxsize=None)
    def bind(self, primitive):
        from sengi.bvh.device_tree import DeviceTree
        ops = self.metric.bind(primitive)
        if self.visibility is not None and ops.viewpoint is None:
            raise TypeError("silhouette metrics must supply a viewpoint")
        Query, Meta = ops.query_type, primitive.data_type
        visibility = self.visibility

        @wp.func
        def visible(query: Query, tree: DeviceTree, meta: Meta, node: int):
            if wp.static(ops.node_filter is not None):
                if not wp.static(ops.node_filter)(query, tree, meta, node):
                    return False
            if wp.static(visibility is not None):
                return wp.static(visibility)(wp.static(ops.viewpoint)(query), tree, meta, node)
            return True

        return replace(ops, node_filter=visible,
                       cull_items=False)


@dataclass(frozen=True)
class SilhouetteQuery(Generic[H, D, ItemT, P, Q, CH, CD, FH, FD, FI]):
    primitive: object
    query: object
    device: object

    @property
    def _device_kernel(self):
        return self.query.kernel

    @property
    def kernel(self):
        return self.query.kernel

    def __call__(self, boundary, queries, max_distance=jnp.inf):
        tree, geometry = boundary.data
        tree.validate_geometry(geometry, self.primitive)
        meta = boundary.silhouette
        if meta is None:
            raise ValueError("build boundary silhouette metadata before querying")
        feature_tree = replace(tree, node_leaf_offset=meta.node_feature_offset,
            node_leaf_count=meta.node_feature_count, prim_ids=meta.feature_refs,
            _schema=_IndexSchema(IndexSpec(self.query.primitive), type(meta)))
        return self.query(IndexData(feature_tree, meta), queries, max_distance)


def bind(spec, metric=None, *, adapter=None, stack_capacity=STACK_CAPACITY, block_dim=64):
    primitive = spec.primitive
    from sengi.boundary.data import resolve_boundary_adapter
    from sengi.metrics.nearest import Euclidean

    metric = Euclidean() if metric is None else metric
    adapter = resolve_boundary_adapter(primitive, adapter)
    return _make_silhouette(primitive, metric, adapter, stack_capacity, block_dim)


@cache_query
def _make_silhouette(primitive, metric, adapter, stack_capacity, block_dim):
    from sengi.queries.nearest import bind as bind_nearest

    if adapter.visibility is not None and not isinstance(adapter.visibility, wp.Function):
        raise TypeError("silhouette visibility must be a decorated Warp Function")
    feature = adapter.feature_primitive
    if feature is None:
        raise TypeError("boundary adapter has no silhouette feature operations")
    query = bind_nearest(IndexSpec(feature), _SilhouetteMetric(metric, adapter.visibility), predicate=adapter.predicate,
                         stack_capacity=stack_capacity, block_dim=block_dim, cache_bounds=False,
                         packed_nodes=False, group_leaves=False)
    nearest = query.device
    Query, Meta = query.ops.query_type, feature.data_type

    @wp.func
    def device(point: Query, cap: float, tree: DeviceTree, meta: Meta):
        feature_tree = DeviceTree(tree.node_lower, tree.node_upper, tree.node_left, tree.node_right,
                                 meta.node_feature_offset, meta.node_feature_count, meta.feature_refs,
                                 tree.node_ray_lower, tree.node_ray_upper)
        return wp.static(nearest)(point, cap, feature_tree, meta)

    return SilhouetteQuery(primitive, query, device)

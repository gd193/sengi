"""Closed-box classification from exact region and point queries."""

from dataclasses import dataclass
from typing import Generic, TypeVar

import warp as wp

from sengi.jax.ffi import _flatten_vec_queries
from sengi.warp.launch import BoundQuery, bind_query, cache_query
from sengi.bvh.device_tree import DeviceTree, STACK_CAPACITY
from sengi.queries.classification import bind as bind_classification
from sengi.bvh.index import IndexSpec

from sengi.queries.region import RegionIntersectionPolicy, _resolve_policy, bind as bind_region

HostDataT = TypeVar("HostDataT")
DeviceDataT = TypeVar("DeviceDataT")
PointT = TypeVar("PointT")
HostMetaT = TypeVar("HostMetaT")
MetaT = TypeVar("MetaT")



@dataclass(frozen=True)
class RegionClassifyQuery(BoundQuery, Generic[HostDataT, DeviceDataT, PointT, HostMetaT, MetaT]):
    metadata_host_type: type | None

    def __call__(self, boundary, lower, upper):
        if self.metadata_host_type is not None and not isinstance(boundary.core, self.metadata_host_type):
            raise TypeError(f"boundary core must be an instance of {self.metadata_host_type.__name__}")
        tree, geometry = boundary.data
        lower, shape = _flatten_vec_queries(lower, self.primitive.dimension)
        upper, other = _flatten_vec_queries(upper, self.primitive.dimension)
        if shape != other:
            raise ValueError("box lower and upper batch shapes must match")
        return self.launch(tree, geometry, shape, lower, upper,
                           metadata=(boundary.core,))


def bind(spec, region_policy: RegionIntersectionPolicy | None = None, classify_policy=None,
                         *, adapter=None, metadata_host_type=None, stack_capacity=STACK_CAPACITY, block_dim=128):
    """Classify a closed box: -1 outside, 0 surface contact, +1 inside."""
    primitive = spec.primitive
    region_policy = _resolve_policy(primitive, region_policy)
    if classify_policy is not None and adapter is not None:
        raise TypeError("supply a classification policy or an adapter")
    if classify_policy is None:
        from sengi.boundary.data import _adapter_policy
        classify_policy = _adapter_policy(primitive, "classification", adapter)
    return _make_region_classify(primitive, region_policy, classify_policy, metadata_host_type=metadata_host_type,
                                 stack_capacity=stack_capacity, block_dim=block_dim)


@cache_query
def _make_region_classify(primitive, region_policy, classify_policy, *,
                          metadata_host_type, stack_capacity, block_dim):
    spec = IndexSpec(primitive)
    exact = bind_region(spec, region_policy, mode="any", stack_capacity=stack_capacity,
                        block_dim=block_dim)
    point = bind_classification(spec, classify_policy, stack_capacity=stack_capacity,
                          block_dim=block_dim)
    Vec, Geometry, Meta = primitive.point_type, primitive.data_type, classify_policy.metadata_type

    @wp.func
    def classify_region(lower: Vec, upper: Vec, tree: DeviceTree, geometry: Geometry, meta: Meta):
        if not wp.static(classify_policy.valid)(meta):
            return 0, -1
        count, status = wp.static(exact.device)(lower, upper, tree, geometry)
        if status < 0:
            return 0, -1
        if count != 0:
            return 0, 1
        sign, point_status = wp.static(point.device)(lower, tree, geometry, meta)
        if point_status < 0:
            return 0, -1
        if sign == 0.0:
            return 0, 1
        return wp.where(sign < 0.0, 1, -1), 1

    inputs = [("lower", wp.array(dtype=Vec)), ("upper", wp.array(dtype=Vec))]
    outputs = [("out_class", wp.array(dtype=wp.int32)), ("out_status", wp.array(dtype=wp.int32))]
    body = ["result, status = wp.static(classify_region)(load_array(lower, i), load_array(upper, i), tree, geometry, meta)",
            "store_array(out_class, i, result)", "store_array(out_status, i, status)"]
    return bind_query(RegionClassifyQuery, "region_classify_kernel", primitive, inputs,
                      outputs, body, {"classify_region": classify_region}, block_dim,
                      device=classify_region, groups=(("meta", Meta),),
                      metadata_host_type=metadata_host_type,
                      native_providers=(region_policy, classify_policy),
                      components=(("intersection", "exact_policy"), ("classification", "policy")))

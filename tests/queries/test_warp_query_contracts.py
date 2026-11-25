"""Public typed policies compile as real Warp extensions."""
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from sengi import bvh
from sengi.queries.domain import ClassificationOps, RayOps
from sengi.queries.boundary_sample import RegionOps
from sengi.boundary.data import BoundaryData
from sengi.boundary.sampling import MeasureOps, SamplingData
from sengi.queries.boundary_sample import bind as bind_boundary_sample
from sengi.queries.classification import bind as bind_classification
from sengi.queries.rays import bind as bind_rays
from tests.typing.warp_api.real_extension import ExternalArrays, ExternalPoints
from tests.typing.warp_api.query_extension import (
    Cursor, ExternalMetadata, MetadataArrays, classification_event, intersect,
    metadata_valid, next_hit, region_contains, region_overlaps, region_valid, sample,
)


def _tree():
    primitive = ExternalPoints()
    return bvh.build(primitive, ExternalArrays(jnp.array([[0., 0., 0.]])))


def _fused(query, *args, **kwargs):
    compiled = jax.jit(query, static_argnames=tuple(kwargs))
    assert compiled.lower(*args, **kwargs).as_text().count("stablehlo.custom_call") == 1
    return compiled(*args, **kwargs)


@pytest.mark.parametrize("mode", ["first", "any", "count", "all"])
@pytest.mark.parametrize("stateful", [False, True])
def test_external_ray_policy_modes_compile_and_fuse(mode, stateful):
    index = _tree()
    ops = RayOps(state_type=Cursor, next_hit=next_hit) if stateful else RayOps(intersect=intersect)
    query = bind_rays(index.spec, mode, ops=ops)
    result = _fused(query, index.data, jnp.array([[-1., 0., 0.]]),
                    jnp.array([[1., 0., 0.]]), **({"max_hits": 3} if mode == "all" else {}))
    count = 2 if stateful else 1
    if mode == "first":
        np.testing.assert_allclose(result[0], [1.])
        np.testing.assert_array_equal(result[1], [0])
        np.testing.assert_array_equal(result[2], [True])
    elif mode == "any":
        np.testing.assert_array_equal(result[0], [True])
    elif mode == "count":
        np.testing.assert_array_equal(result[0], [count])
    else:
        np.testing.assert_allclose(result[0][0, :count], [1., 2.] if stateful else [1.])
        np.testing.assert_array_equal(result[1][0, :count], 0)
        np.testing.assert_array_equal(result[2], [count])
    np.testing.assert_array_equal(result[-1], [1])


def test_external_classification_policy_compiles_and_fuses():
    index = _tree()
    policy = ClassificationOps(ExternalMetadata, metadata_valid, classification_event)
    query = bind_classification(index.spec, policy, metadata_host_type=MetadataArrays)
    assert query is bind_classification(index.spec, policy, metadata_host_type=MetadataArrays)
    assert query is not bind_classification(index.spec, policy)
    metadata = BoundaryData(index.data, MetadataArrays(jnp.array([1], dtype=jnp.int32)))
    sign, status = _fused(query, metadata, jnp.array([[-1., 0., 0.], [1., 0., 0.]]))
    np.testing.assert_array_equal(sign, [-1., 1.])
    np.testing.assert_array_equal(status, [1, 1])
    sign, status = jax.jit(query)(BoundaryData(index.data, MetadataArrays(jnp.array([0], dtype=jnp.int32))), jnp.zeros((1, 3)))
    assert np.isnan(sign).all()
    np.testing.assert_array_equal(status, [-1])
    with pytest.raises(TypeError, match="boundary core.*MetadataArrays"):
        query(BoundaryData(index.data, metadata.core._asdict()), jnp.zeros((1, 3)))


def test_external_sample_region_policies_compile_and_fuse():
    import warp as wp
    index = _tree()
    measure = MeasureOps(sample)
    region = RegionOps(wp.float32, (), region_valid, region_overlaps, region_contains)
    query = bind_boundary_sample(index.spec, measure, region)
    metadata = SamplingData(index.data, jnp.ones((1,)), jnp.ones(index.data.tree.node_left.shape))
    point, normal, pdf, primitive_id, status = _fused(query, metadata,
        jnp.array([1., -1.]), 0., jnp.full((2, 3), .5))
    np.testing.assert_array_equal(point[0], [0., 0., 0.])
    np.testing.assert_array_equal(normal[0], [0., 1., 0.])
    np.testing.assert_array_equal(pdf, [1., 0.])
    np.testing.assert_array_equal(primitive_id, [0, -1])
    np.testing.assert_array_equal(status, [1, -1])


def test_external_robin_policy_compiles_and_fuses():
    from dataclasses import replace
    import warp as wp
    from sengi.boundary.data import BoundaryCore, SilhouetteMeta
    from sengi.boundary.robin import RobinData
    from sengi.queries.robin_radius import RobinOps
    from sengi.queries.robin_radius import bind as bind_robin_radius
    from sengi.boundary.robin import prepare_robin_coefficients
    from tests.typing.warp_api.query_extension import (
        ExternalRobinMetric, external_adapter, radius_to_score, robin_feature, robin_node, robin_primitive,
        robin_valid, score_to_radius,
    )
    index = _tree()
    policy = RobinOps(wp.vec3, (3,), robin_valid, robin_node, robin_primitive, robin_feature,
        core_type=BoundaryCore, feature_type=SilhouetteMeta,
        to_score=radius_to_score, to_distance=score_to_radius)
    # No boundary features: only the external point contributes a radius.
    # These are the actual metadata schemas expected by the fused kernel.
    def empty_payload(schema):
        fields = {}
        for name, field in schema.vars.items():
            dtype = field.type.dtype
            components = {wp.vec2: 2, wp.vec3: 3}.get(dtype, 0)
            shape = (index.data.tree.node_left.shape[0], components) if components else (index.data.tree.node_left.shape[0],) * field.type.ndim
            fields[name] = jnp.zeros(shape,
                                    dtype=jnp.int32 if dtype is wp.int32 else jnp.float32)
        return namedtuple(schema.key + "Arrays", fields)(**fields)
    metadata = RobinData(index.data, empty_payload(BoundaryCore), empty_payload(SilhouetteMeta),
        prepare_robin_coefficients(index.data.tree, jnp.inf))
    adapter = replace(external_adapter, robin=lambda _: policy)
    query = bind_robin_radius(index.spec, adapter=adapter)
    assert query is bind_robin_radius(index.spec, policy)
    radius, status = _fused(query, metadata,
                           jnp.array([[0., 2., 0.]]), 10.)
    np.testing.assert_allclose(radius, [2.])
    np.testing.assert_array_equal(status, [1])
    metric_result = _fused(bind_robin_radius(index.spec, metric=ExternalRobinMetric()),
                           metadata, jnp.array([[0., 2., 0.]]), 10.)
    np.testing.assert_array_equal(metric_result[0], radius)
    np.testing.assert_array_equal(metric_result[1], status)


def test_external_silhouette_and_star_adapter_compile():
    from sengi.metrics.nearest import NearestCallbacks
    from sengi.queries.silhouette import bind as bind_silhouette
    from sengi.queries.star_radius import bind as bind_star_radius
    from tests.typing.warp_api.query_extension import external_adapter
    from tests.typing.warp_api.real_extension import external_bound, external_evaluate
    primitive = ExternalPoints()
    source = ExternalArrays(jnp.array([[0., 0., 0.]]))
    index = bvh.build(primitive, source)
    boundary = bvh.prepare_boundary(index, source, adapter=external_adapter)
    custom = bind_silhouette(index.spec, NearestCallbacks(external_evaluate, external_bound),
                             adapter=external_adapter)
    point, score, primitive_id, status = _fused(custom, boundary, jnp.array([[0., 2., 0.]]))
    np.testing.assert_array_equal(point, [[0., 0., 0.]])
    np.testing.assert_allclose(score, [4.])
    np.testing.assert_array_equal(primitive_id, [0])
    np.testing.assert_array_equal(status, [1])
    # Star composes a silhouette kernel with ordinary host JAX operations.
    star = jax.jit(bind_star_radius(index.spec, adapter=external_adapter))
    radius, status = star(boundary, jnp.array([[0., 2., 0.]]), 10.)
    np.testing.assert_allclose(radius, [2.])
    np.testing.assert_array_equal(status, [1])


def test_builtin_payload_uses_same_external_ray_contract():
    from sengi.primitives.points import Points
    from tests.typing.warp_api.query_extension import builtin_intersect
    primitive = Points(3)
    index = bvh.build(primitive, primitive.data([[2., 0., 0.]]))
    result = _fused(bind_rays(index.spec, ops=RayOps(intersect=builtin_intersect)),
                    index.data, jnp.zeros((1, 3)), jnp.array([[1., 0., 0.]]))
    np.testing.assert_allclose(result[0], [2.])
    np.testing.assert_array_equal(result[1], [0])
    np.testing.assert_array_equal(result[2], [True])
    np.testing.assert_array_equal(result[3], [1])


def test_external_overlap_preserves_primitive_aabb_semantics_and_fuses():
    from sengi.queries.overlap import bind as bind_overlap
    index = _tree()
    hit, status = _fused(bind_overlap(index.spec), index.data,
        jnp.array([[-1., -1., -1.], [2., 2., 2.]]),
        jnp.array([[1., 1., 1.], [3., 3., 3.]]))
    np.testing.assert_array_equal(hit, [True, False])
    np.testing.assert_array_equal(status, [1, 0])

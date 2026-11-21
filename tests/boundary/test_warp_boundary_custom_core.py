"""Boundary construction preserves adapter-owned metadata and leaf packing."""
from sengi import bvh
from dataclasses import dataclass, replace
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest

wp = pytest.importorskip("warp")

from sengi.boundary.data import BoundaryAdapter, resolve_boundary_adapter
from sengi.queries.domain import ClassificationOps  # noqa: E402
from sengi.boundary.sampling import MeasureOps, prepare_boundary_sample
from sengi.queries.boundary_sample import bind as bind_boundary_sample  # noqa: E402
from sengi.queries.classification import bind as bind_classification  # noqa: E402
from sengi.primitives.segments import Segments
from sengi.warp.contracts import warp_struct  # noqa: E402
from sengi.bvh import Config  # noqa: E402


class CustomCore(NamedTuple):
    enabled: object
    required_extra: object


@warp_struct
class CustomDeviceCore:
    enabled: wp.array(dtype=wp.int32)
    required_extra: wp.array(dtype=wp.int32)


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CustomFeatures:
    """A supported typed schema need not be a positional iterable."""
    extra: object


@wp.func
def _valid(meta: CustomDeviceCore):
    return meta.enabled[0] == 1


@wp.func
def _event(point: wp.vec2, geometry: Any, slot: int, original: int, meta: CustomDeviceCore):
    if meta.required_extra[slot] != original + 10:
        return -1
    return wp.int32(meta.required_extra[slot] == 11)


def _classification(primitive):
    return ClassificationOps(CustomDeviceCore, _valid, _event)


def test_custom_build_packs_extra_arrays_and_classifies():
    primitive = Segments(2)
    vertices = np.array([[x, 0.] for x in [8., 9., -4., -3., 2., 3.]], np.float32)
    data = primitive.indexed(vertices, np.array([[0, 1], [2, 3], [4, 5]], np.int32))
    source = CustomCore(np.array([1], np.int32), np.arange(10, 13, dtype=np.int32))
    features = CustomFeatures(jnp.array([7.], jnp.float32))
    calls = []

    def prepare(primitive, original):
        assert original is data
        return source

    def pack(core, original, tree, geometry):
        assert core is source and original is data
        calls.append(np.asarray(tree.prim_ids))
        return CustomCore(jnp.asarray(core.enabled), jnp.asarray(core.required_extra)[tree.prim_ids])

    def build_features(core, tree, geometry):
        assert core is source
        return features

    builtin_measure = resolve_boundary_adapter(primitive).measure(primitive)
    measure = MeasureOps(builtin_measure.sample, weights=lambda core: core.required_extra)
    adapter = BoundaryAdapter(prepare, pack,
        feature_primitive=resolve_boundary_adapter(primitive).feature_primitive,
        build_silhouette=build_features, classification=_classification, measure=lambda _: measure)
    index = bvh.build(primitive, data, config=Config(leaf_size_threshold=1))
    tree = index.data.tree
    boundary = bvh.prepare_boundary(index, data, adapter=adapter)
    ids = np.asarray(tree.prim_ids)
    assert not np.array_equal(ids, np.arange(3))
    assert len(calls) == 1
    assert isinstance(boundary.core, CustomCore)
    assert boundary.silhouette is features
    np.testing.assert_array_equal(boundary.core.required_extra, np.arange(10, 13)[ids])
    query = jax.jit(bind_classification(index.spec, adapter=adapter))
    sign, status = query(boundary, jnp.array([[-10., 0.]], jnp.float32))
    np.testing.assert_array_equal(sign, [-1.])
    np.testing.assert_array_equal(status, [1])
    sampling = prepare_boundary_sample(boundary, measure)
    np.testing.assert_array_equal(sampling.primitive_weight, boundary.core.required_extra)
    sampler = jax.jit(bind_boundary_sample(index.spec, adapter=adapter))
    _, _, pdf, face, sample_status = sampler(sampling,
        jnp.array([[0., 0.]], jnp.float32), 100., jnp.array([[.25, .5, .5]], jnp.float32))
    np.testing.assert_allclose(pdf, (np.asarray(face) + 10.) / 33.)
    np.testing.assert_array_equal(sample_status, [1])
    # A separate build must also use the callback when features are disabled.
    index = bvh.build(primitive, data)
    without_features = bvh.prepare_boundary(index, data, adapter=adapter, silhouettes=False)
    assert isinstance(without_features.core, CustomCore)
    assert without_features.silhouette is None


def test_builtin_distinct_connectivity_is_not_reused():
    primitive = Segments(2)
    data = primitive.indexed([[4., 0.], [5., 0.], [-2., 0.], [-1., 0.]], [[0, 1], [2, 3]])
    builtin = resolve_boundary_adapter(primitive)

    def prepare(primitive, data):
        core = builtin.prepare(primitive, data)
        return core._replace(indices=np.array(core.indices, copy=True))

    adapter = replace(builtin, prepare=prepare)
    index = bvh.build(primitive, data, config=Config(leaf_size_threshold=1))
    tree = index.data.tree
    boundary = bvh.prepare_boundary(index, data, adapter=adapter, silhouettes=False)
    assert boundary.core.indices is not index.data.geometry.indices
    np.testing.assert_array_equal(boundary.core.indices, np.asarray(data.indices)[tree.prim_ids])

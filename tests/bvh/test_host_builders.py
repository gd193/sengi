"""Host construction works without the optional Warp runtime."""
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from sengi import bvh


def test_integer_bounds_enclose_input_after_float32_conversion():
    source = np.array([[2**53 + 1, -(2**53 + 1), 2**53]], dtype=np.int64)
    payload = bvh.build_sah(source, source)
    # Compare as integers so the test itself cannot round away the discrepancy.
    lower = np.asarray(payload.node_lower).astype(np.int64)
    upper = np.asarray(payload.node_upper).astype(np.int64)
    assert np.all(lower <= source) and np.all(source <= upper)
    assert lower[0, 2] == upper[0, 2] == source[0, 2]


@pytest.mark.parametrize("dtype", [np.int64, np.uint64])
def test_integer_bounds_at_dtype_max(dtype):
    maximum = int(np.iinfo(dtype).max)
    source = np.full((1, 3), maximum, dtype=dtype)
    payload = bvh.build_sah(source, source)
    assert int(payload.node_upper[0, 0]) == maximum + 1
    assert int(payload.node_lower[0, 0]) < maximum
    with pytest.raises(ValueError, match="ordered"):
        bvh.build_sah(source.astype(np.float64), source)


@pytest.mark.parametrize("value", [2**89 + 1, "1"])
def test_bounds_reject_implicit_numeric_coercion(value):
    source = [[value, 0, 0]]
    with pytest.raises(ValueError, match="real numeric dtype"):
        bvh.build_sah(source, source)


def test_leaf_size_is_positive_and_keyword_only():
    with pytest.raises(TypeError):
        bvh.Config(8)
    with pytest.raises(ValueError, match="leaf size"):
        bvh.Config(leaf_size_threshold=0)


@pytest.mark.parametrize("builder", ["sah", "lbvh"])
@pytest.mark.parametrize("indexed", [False, True], ids=["bounds", "index"])
def test_binary_depth_limit(builder, indexed):
    config = bvh.Config(leaf_size_threshold=1)
    if builder == "sah":
        # Increasing radii on alternating axes make each SAH split peel one point.
        radii = np.exp2(4 * (np.arange(22) - 12))
        deep = (radii[:, None, None] * np.eye(3)).reshape(-1, 3)
    else:
        # One branch per Morton bit, then duplicate origins split by index.
        bits = (np.exp2(np.arange(21))[:, None, None] * np.eye(3)).reshape(-1, 3)
        deep = np.vstack((bits, [[2**21 - 1] * 3], np.zeros((3, 3))))
    deep = deep.astype(np.float32)
    shallow = deep[:-1]  # Exactly 64 edges. The final point requires a 65th.

    error = rf"{builder.upper()} reached depth 65.*at most 64"
    if indexed:
        pytest.importorskip("warp")
        from sengi.primitives.points import Points
        from sengi.queries.nearest import bind

        primitive = Points(3)
        index = bvh.build(primitive, primitive.data(shallow), builder=builder, config=config)
        # The innermost point defers one sibling at every edge of the deep branch.
        point = shallow[2] if builder == "sah" else shallow[-1]
        _, distance, _, status = bind(index.spec)(index.data, point)
        assert int(status) == 1 and float(distance) == 0
        *_, status = bind(index.spec, stack_capacity=63)(index.data, point)
        assert int(status) == -1
        with pytest.raises(ValueError, match=error):
            bvh.build(primitive, primitive.data(deep), builder=builder, config=config)
    else:
        build = bvh.build_sah if builder == "sah" else bvh.build_lbvh
        payload = build(shallow, shallow, config)
        assert np.count_nonzero(payload.node_leaf_count) == len(shallow)
        with pytest.raises(ValueError, match=error):
            build(deep, deep, config)
        if builder == "sah":
            # Coincident points make the deepest split unprofitable at depth 64.
            deep[1] = deep[2]
            payload = build(deep, deep, config)
            assert int(payload.node_leaf_count.max()) == 2


def test_builders_work_when_warp_is_unavailable():
    # Isolate imports and simulate a missing Warp installation.
    code = r'''
import importlib.abc
import sys

class BlockWarpImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "warp" or fullname.startswith("warp."):
            raise ModuleNotFoundError("Warp is blocked for this test")
        return None

sys.meta_path.insert(0, BlockWarpImports())

import jax
import jax.numpy as jnp
import numpy as np
from sengi.bvh import Config, build_lbvh, build_sah
from sengi.kd import build_left_balanced, build_sah as build_kd_sah
from sengi.octree import build_morton_explicit

assert jax.devices()[0].platform == "cpu"
lower = jnp.asarray([[0., 0., 0.]], dtype=jnp.float32)
upper = jnp.asarray([[1., 1., 0.]], dtype=jnp.float32)
for build in (build_lbvh, build_sah, build_left_balanced,
              build_kd_sah, build_morton_explicit):
    with np.testing.assert_raises_regex(ValueError, "at least one primitive"):
        build(lower[:0], upper[:0])
    payload = build(lower, upper)
    assert int(payload.root) == 0
    np.testing.assert_array_equal(payload.prim_ids, [0])
    np.testing.assert_array_equal(payload.node_lower, [[0., 0., 0.]])
    np.testing.assert_array_equal(payload.node_upper, [[1., 1., 0.]])
unit = np.eye(3, dtype=np.float32)
assert build_sah(unit, unit, Config(leaf_size_threshold=1)).node_leaf_count.max() == 1
assert "warp" not in sys.modules
'''
    env = dict(os.environ, JAX_PLATFORMS="cpu",
               XLA_PYTHON_CLIENT_PREALLOCATE="false")
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=90,
    )

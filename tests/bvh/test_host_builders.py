"""Host construction works without the optional Warp runtime."""
import os
from pathlib import Path
import subprocess
import sys


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
from sengi.bvh import build_lbvh, build_sah
from sengi.kd import build_left_balanced, build_sah as build_kd_sah
from sengi.octree import build_morton_explicit

assert jax.devices()[0].platform == "cpu"
lower = jnp.asarray([[0., 0., 0.]], dtype=jnp.float32)
upper = jnp.asarray([[1., 1., 0.]], dtype=jnp.float32)
for build in (build_lbvh, build_sah, build_left_balanced,
              build_kd_sah, build_morton_explicit):
    payload = build(lower, upper)
    assert int(payload.root) == 0
    np.testing.assert_array_equal(payload.prim_ids, [0])
    np.testing.assert_array_equal(payload.node_lower, [[0., 0., 0.]])
    np.testing.assert_array_equal(payload.node_upper, [[1., 1., 0.]])
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

"""Regression checks for persistent benchmark SAH indexes."""

import jax
import numpy as np
import pytest

from benchmarks.adapters import sah_cache, sengi
from benchmarks.workloads import builtin_geometry


@pytest.mark.parametrize("kind", ["triangles", "points2", "points3"])
def test_sah_cache_round_trip(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(sah_cache, "CACHE_DIR", tmp_path)
    geometry = builtin_geometry(kind, 64)

    def prepare(**options):
        metadata = {}
        index, _ = sengi.prepare(geometry, metadata=metadata, **options)
        return index, metadata

    cold, first = prepare()
    assert first["sah_cache"] == "miss"
    assert "build_ms" in first
    with monkeypatch.context() as patch:
        def unexpected_build(*args, **kwargs):
            raise AssertionError("cache hit invoked the SAH builder")

        patch.setattr(sengi.bvh, "build", unexpected_build)
        warm, second = prepare()
    assert second["sah_cache"] == "hit"
    assert "index_load_ms" in second and "build_ms" not in second
    assert warm.spec == cold.spec
    for actual, expected in zip(jax.tree.leaves(warm.data), jax.tree.leaves(cold.data), strict=True):
        np.testing.assert_array_equal(actual, expected)

    archive, = tmp_path.glob("*.npz")
    archive.write_bytes(b"interrupted write")
    assert prepare()[1]["sah_cache"] == "miss"
    assert prepare()[1]["sah_cache"] == "hit"
    assert prepare(leaf_size=16)[1]["sah_cache"] == "miss"
    field = "faces" if kind == "triangles" else "points"
    geometry.arrays[field] = geometry.arrays[field][::-1].copy()
    assert prepare()[1]["sah_cache"] == "miss"
    assert "sah_cache" not in prepare(builder="lbvh")[1]
    assert len(list(tmp_path.glob("*.npz"))) == 3

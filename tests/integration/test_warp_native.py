"""Native build requirements follow their owning query components."""
from importlib.resources import files
from pathlib import Path

import numpy as np
import pytest

wp = pytest.importorskip("warp")

from sengi import bvh
from sengi.bvh.device_tree import DeviceTree
from sengi.primitives.data import PointsData3
from sengi.primitives.items import PointItem3
from sengi.primitives.points import Points
from sengi.primitives.triangles import Triangles
from sengi.queries.classification import bind as bind_classification
from sengi.queries.domain import ClassificationOps
from sengi.queries.nearest import bind as bind_nearest
from sengi.queries.results import ResultPolicy
from sengi.warp.native import NativeHeaders
from tests.support._warp_device import device_payload, run_kernel


def _requirements(query):
    return wp.get_module_options(query.kernel.module)["extra_build_options"]


def test_builtin_native_requirements_follow_owners():
    root = files("sengi")
    assert set(map(Path, _requirements(bind_nearest(bvh.IndexSpec(Points(2)))).extra_build_dependencies)) == {
        root / "bvh/traversal/native/traversal.h",
    }
    triangle_headers = set(map(Path, _requirements(bind_nearest(bvh.IndexSpec(Triangles()))).extra_build_dependencies))
    assert root / "primitives/native/triangle_nearest.h" in triangle_headers
    assert root / "primitives/native/triangle_distance.h" in triangle_headers
    assert root / "primitives/native/triangle_robin.h" not in triangle_headers


def test_native_result_and_boundary_policies_compose(tmp_path):
    def provider(name):
        root = tmp_path / name
        root.mkdir()
        helper = root / "helper.h"
        helper.write_text(f"namespace {name} {{ CUDA_CALLABLE inline int value() {{ return 1; }} }}\n")
        (root / "entry.h").write_text('#pragma once\n#include "helper.h"\n')
        return NativeHeaders((root,), ("entry.h",), (helper,)), helper

    result_headers, result_helper = provider("result_provider")
    boundary_headers, boundary_helper = provider("boundary_provider")

    @wp.struct
    class Result:
        mark: int

    @wp.struct
    class Meta:
        enabled: wp.array(dtype=bool)

    @wp.func_native("return result_provider::value();")
    def mark() -> int: ...

    @wp.func_native("return boundary_provider::value();")
    def event(point: wp.vec3, data: PointsData3, slot: int, original: int, meta: Meta) -> int: ...

    @wp.func_native("return caller_tag::apply(value);")
    def tag(value: int) -> int: ...

    @wp.func
    def build(query: wp.vec3, item: PointItem3, original: int, point: wp.vec3,
              distance: float) -> Result:
        return Result(mark())

    @wp.func
    def valid(meta: Meta):
        return meta.enabled[0]

    result = ResultPolicy(Result, build, native=result_headers)
    classify = ClassificationOps(Meta, valid, event, native=boundary_headers)
    primitive = Points(3)
    index = bvh.build(primitive, primitive.data([[0., 0., 0.]]))
    tree, data = index.data
    selected, classified = bind_nearest(index.spec, result_policy=result), bind_classification(index.spec, classify)
    near, boundary = selected.device, classified.device

    @wp.kernel(module="unique", enable_backward=False)
    def caller(tree: DeviceTree, data: PointsData3, meta: Meta, values: wp.array(dtype=int)):
        point = wp.vec3(0.)
        result, _ = wp.static(near)(point, 1., tree, data)
        sign, _ = wp.static(boundary)(point, tree, data, meta)
        values[0] = tag(result.mark)
        values[1] = int(sign)

    result_entry = result_helper.with_name("entry.h")
    preamble = (f'#if 0\n#include "{result_entry.as_posix()}"\n#endif\n'
                "namespace caller_tag {\nCUDA_CALLABLE inline int apply(int x) {\n    return x + 3;\n}\n}\n")
    wp.set_module_options({"extra_build_options": wp.ModuleBuildOptions(extra_cpu_preamble=preamble)},
                          module=caller.module)
    meta = Meta()
    meta.enabled = wp.array([True], dtype=bool, device="cpu")
    args = [device_payload(DeviceTree, tree, "cpu"), device_payload(PointsData3, data, "cpu"), meta]

    def run(selected, classified):
        return run_kernel(caller, 1, args, [(2, int)], queries=(selected, classified), device="cpu")[0]

    np.testing.assert_array_equal(run(selected, classified), [4, -1])
    configured = wp.get_module_options(caller.module)["extra_build_options"]
    assert configured.extra_cpu_preamble.startswith(preamble)
    assert set(configured.extra_build_dependencies) >= {result_helper, boundary_helper}
    hashes = tuple(query.kernel.module.get_module_hash() for query in (selected, classified))

    result_helper.write_text(result_helper.read_text().replace("return 1", "return 2"))
    boundary_helper.write_text(boundary_helper.read_text().replace("return 1", "return 2"))
    updated_selected = bind_nearest(index.spec, result_policy=result)
    updated_classified = bind_classification(index.spec, classify)
    assert updated_selected.kernel.module.get_module_hash() != hashes[0]
    assert updated_classified.kernel.module.get_module_hash() != hashes[1]
    np.testing.assert_array_equal(run(updated_selected, updated_classified), [5, 0])

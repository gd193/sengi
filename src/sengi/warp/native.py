"""Native headers and Warp kernel build options."""
from pathlib import Path

import warp as wp

from sengi.bvh.device_tree import STACK_CAPACITY


_PACKAGE = Path(__file__).resolve().parents[1]
_NATIVE_DIRS = [
    _PACKAGE / owner / "native"
    for owner in ("bvh/traversal", "primitives", "metrics", "boundary", "queries")
]
_NATIVE_HEADERS = [
    "traversal.h",
    "box_distance.h",
    "angular.h",
    "triangle_nearest.h",
    "segment_crossing.h",
    "classification.h",
    "rays.h",
    "silhouette.h",
    "robin.h",
]
_NATIVE_DEPENDENCIES = [header for directory in _NATIVE_DIRS for header in sorted(directory.glob("*.h"))]
_PREAMBLE = "".join(f'#include "{header}"\n' for header in _NATIVE_HEADERS) + "\n".join(
    f"namespace bvh {{ using Stack_{bits}_{n} = Stack<uint{bits}_t,{n}>; }}"
    for bits in (32, 64) for n in range(1, STACK_CAPACITY + 1)
)
QUERY_BUILD_OPTIONS = wp.ModuleBuildOptions(
    extra_cpu_include_dirs=_NATIVE_DIRS, extra_cuda_include_dirs=_NATIVE_DIRS,
    extra_cpu_preamble=_PREAMBLE, extra_cuda_preamble=_PREAMBLE,
    extra_build_dependencies=_NATIVE_DEPENDENCIES,
)



"""Declarations for headers used by native Warp callbacks."""
from dataclasses import dataclass, field
from pathlib import Path

import warp as wp

from sengi.bvh.device_tree import STACK_CAPACITY


@dataclass(frozen=True)
class NativeHeaders:
    """CPU and CUDA headers owned by a primitive or static query policy.

    Header names are relative to the absolute include directories. Declare
    transitive package-local files in dependencies so Warp hashes their contents.
    """
    include_dirs: tuple[Path, ...]
    headers: tuple[str, ...]
    dependencies: tuple[Path, ...] = ()
    _entries: tuple[Path, ...] = field(init=False, repr=False)

    def __post_init__(self):
        directories = tuple(Path(path) for path in self.include_dirs)
        headers = tuple(self.headers)
        dependencies = tuple(Path(path) for path in self.dependencies)
        for directory in directories:
            if not directory.is_absolute() or not directory.is_dir():
                raise ValueError(f"native include directory must be absolute and exist: {directory}")
        for header in headers:
            if not header or Path(header).is_absolute() or ".." in Path(header).parts:
                raise ValueError(f"native entry header must be relative to an include directory: {header!r}")
        for dependency in dependencies:
            if not dependency.is_absolute() or not dependency.is_file():
                raise ValueError(f"native dependency must be an absolute existing file: {dependency}")
        object.__setattr__(self, "include_dirs", directories)
        object.__setattr__(self, "headers", headers)
        object.__setattr__(self, "dependencies", dependencies)
        object.__setattr__(self, "_entries", self._entry_paths())

    def _entry_paths(self) -> tuple[Path, ...]:
        entries = []
        for name in self.headers:
            header = next((path for directory in self.include_dirs
                           if (path := directory / name).is_file()), None)
            if header is None:
                raise ValueError(f"native entry header was not found: {name}")
            if any(character in str(header) for character in '\"\n\r'):
                raise ValueError(f"native entry header cannot be quoted in C++: {header!s}")
            entries.append(header)
        return tuple(entries)

    def _build_options(self) -> wp.ModuleBuildOptions:
        return native_options(self.include_dirs,
                              "".join(f'#include "{header.as_posix()}"\n' for header in self._entries),
                              (*self._entries, *self.dependencies))


def native_options(include_dirs=(), preamble="", dependencies=()):
    return wp.ModuleBuildOptions(
        extra_cpu_include_dirs=include_dirs, extra_cuda_include_dirs=include_dirs,
        extra_cpu_preamble=preamble, extra_cuda_preamble=preamble,
        extra_build_dependencies=dependencies,
    )


_PACKAGE = Path(__file__).resolve().parents[1]


def package_headers(owner, *headers, dependencies=(), include_owners=()):
    directories = tuple(_PACKAGE / name / "native" for name in (owner, *include_owners))
    return NativeHeaders(directories, headers, tuple(_PACKAGE / path for path in dependencies))


TRAVERSAL_HEADERS = package_headers("bvh/traversal", "traversal.h")
_STACK_OPTIONS = native_options(preamble="\n".join(
    f"namespace bvh {{ using Stack_{bits}_{n} = Stack<uint{bits}_t,{n}>; }}"
    for bits in (32, 64) for n in range(1, STACK_CAPACITY + 1)
))


def _missing_lines(existing, addition):
    seen = set(existing.splitlines())
    return "\n".join(line for line in addition.splitlines() if line and line not in seen)


def merge_native_options(build, *requirements):
    """Add Sengi's generated include and ABI lines once, in dependency order."""
    for required in requirements:
        build = build.merged(wp.ModuleBuildOptions(
            extra_cpu_include_dirs=required.extra_cpu_include_dirs,
            extra_cuda_include_dirs=required.extra_cuda_include_dirs,
            extra_cpu_preamble=_missing_lines(build.extra_cpu_preamble, required.extra_cpu_preamble),
            extra_cuda_preamble=_missing_lines(build.extra_cuda_preamble, required.extra_cuda_preamble),
            extra_build_dependencies=required.extra_build_dependencies,
        ))
    return build


def query_build_options(*providers, requirements=()):
    build = TRAVERSAL_HEADERS._build_options().merged(_STACK_OPTIONS)
    for provider in providers:
        declaration = getattr(provider, "native", None)
        if declaration is not None:
            if not isinstance(declaration, NativeHeaders):
                raise TypeError("native must be a NativeHeaders declaration")
            build = merge_native_options(build, declaration._build_options())
    return merge_native_options(build, *requirements)

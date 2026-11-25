"""Build the selected native benchmark adapters against supplied library checkouts."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from benchmarks.adapters import native
from benchmarks.results import file_hash

ROOT = Path(__file__).resolve().parent
ADAPTERS = ("cubql", "fcpw", "optix", "cudakdtree")


def _revision(path):
    def git(*args):
        return subprocess.check_output(["git", "-C", path, *args])

    return {
        "path": str(path),
        "commit": git("rev-parse", "HEAD").decode().strip(),
        "dirty": bool(git("status", "--porcelain")),
        "diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
    }


def _commands(args, selected):
    cuda = args.cuda
    cuda_target = cuda / "targets/x86_64-linux"

    def include(*paths):
        return ["-I" + str(path) for path in paths]

    def link(path):
        return ["-L" + str(path), "-Wl,-rpath," + str(path)]

    host = [
        args.cxx,
        "-std=c++20",
        "-O3",
        "-DNDEBUG",
        *include(ROOT),
        *link(cuda / "lib"),
    ]
    nvcc = [
        cuda / "bin/nvcc",
        "-ccbin",
        args.cuda_host,
        "-std=c++17",
        "-O3",
        *include(ROOT),
    ]
    cuda_device = [
        *nvcc,
        "-arch=sm_" + args.arch,
        "-lineinfo",
        "-Xlinker",
        "-rpath",
        "-Xlinker",
        str(cuda / "lib"),
    ]
    if "optix" in selected:
        yield (
            "optix_programs.ptx",
            [
                *nvcc,
                "--relocatable-device-code=true",
                "--ptx",
                "-arch=compute_" + args.arch,
                *include(args.optix / "include"),
                ROOT / "optix/optix_programs.cu",
            ],
        )
        yield (
            "optix_mesh",
            [
                *host,
                *include(cuda_target / "include", args.optix / "include"),
                ROOT / "optix/optix_mesh.cpp",
                *link(cuda_target / "lib"),
                "-lcudart",
                "-lcupti",
                "-L" + str(cuda_target / "lib/stubs"),
                "-lcuda",
                "-ldl",
            ],
        )
    if "cubql" in selected:
        for source, dim, name in [
            ("cubql_mesh.cu", 3, "cubql_mesh"),
            *(("cubql_points.cu", d, f"cubql_points{d}") for d in args.dimensions),
        ]:
            yield (
                name,
                [
                    *cuda_device,
                    *include(args.cubql),
                    "-DCUBQL_HAVE_CUDA=1",
                    "-DCUBQL_INSTANTIATE_T=float",
                    f"-DCUBQL_INSTANTIATE_D={dim}",
                    f"-DBENCH_DIM={dim}",
                    ROOT / "cubql" / source,
                    args.cubql / "cuBQL/builder/cuda/instantiate_builders.cu",
                ],
            )
    if "cudakdtree" in selected:
        for dim in args.dimensions:
            for traversal in ("cct", "stack"):
                for mode in ("fixed", "heap"):
                    name = f"cudakdtree_{mode}" + (
                        "_stack" if traversal == "stack" else ""
                    )
                    yield (
                        f"{name}_d{dim}",
                        [
                            *cuda_device,
                            *include(args.cudakdtree),
                            f"-DBENCH_DIM={dim}",
                            f"-DBENCH_HEAP={int(mode == 'heap')}",
                            f"-DBENCH_STACK={int(traversal == 'stack')}",
                            ROOT / "cudakdtree/knn_cudakdtree.cu",
                        ],
                    )
    if "fcpw" in selected:
        eigen = args.fcpw / "deps/eigen"
        fcpw_host = [*host, *include(args.fcpw / "include", eigen)]
        slang = args.fcpw_build / "_deps/slang-src"
        rhi = args.fcpw_build / "deps/slang-rhi"
        yield (
            "fcpw_queries",
            [
                *fcpw_host,
                "-DFCPW_USE_GPU",
                '-DSLANG_USER_CONFIG="slang-user-config.h"',
                *include(
                    args.fcpw / "deps/slang-rhi/include",
                    args.fcpw / "deps/slang-rhi/src",
                    rhi / "include",
                    slang / "include",
                    cuda_target / "include",
                ),
                ROOT / "fcpw/fcpw_queries.cpp",
                ROOT / "fcpw/fcpw_queue.cpp",
                rhi / "libslang-rhi.a",
                slang / "lib/libslang-compiler.so",
                rhi / "libslang-rhi-resources.a",
                *link(slang / "lib"),
                "-L" + str(cuda_target / "lib/stubs"),
                "-lcuda",
                "-ldl",
            ],
        )


def build(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks build", description=__doc__
    )
    parser.add_argument(
        "--adapter",
        required=True,
        help=f"comma-separated names from {', '.join(ADAPTERS)}",
    )
    for name in ("cubql", "cudakdtree", "fcpw", "fcpw-build", "optix"):
        parser.add_argument("--" + name, type=Path)
    parser.add_argument(
        "--cuda", type=Path, required=True, help="Conda CUDA toolkit prefix"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cxx")
    parser.add_argument("--cuda-host")
    parser.add_argument("--arch", default="86")
    parser.add_argument(
        "--dimensions", type=int, nargs="+", choices=(2, 3), default=(2, 3)
    )
    args = parser.parse_args(argv)
    selected = set(args.adapter.split(","))
    unknown = selected - set(ADAPTERS)
    if unknown:
        parser.error(f"unknown build adapters: {', '.join(sorted(unknown))}")
    dependencies = {
        "cubql": ("cubql",),
        "cudakdtree": ("cudakdtree",),
        "optix": ("optix",),
        "fcpw": ("fcpw", "fcpw_build"),
    }
    for key in sorted(set().union(*(dependencies[name] for name in selected))):
        if getattr(args, key) is None:
            parser.error(
                f"--{key.replace('_', '-')} is required for the selected adapters"
            )
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    args.cxx = args.cxx or str(args.cuda / "bin/g++")
    args.cuda_host = args.cuda_host or shutil.which("clang++") or args.cxx
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / ".build.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with tempfile.TemporaryDirectory(
            prefix=".build-", dir=args.output
        ) as directory:
            _compile(args, selected, Path(directory))


def _compile(args, selected, stage):
    manifest = args.output / "build.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else {}
    owners = {
        native.binary_name(adapter, primitive): "cudakdtree"
        if adapter in native.KD_VARIANTS
        else adapter
        for adapter, supported in native.SUPPORT.items()
        for primitive in supported
    }
    owners["optix_programs.ptx"] = "optix"

    def dependencies(name):
        owner = owners[name]
        return (owner, "eigen", "slang-rhi") if owner == "fcpw" else (owner,)

    artifacts = {}
    for name, record in records.get("artifacts", {}).items():
        if name in owners:
            record["references"] = {
                key: revision
                for key, revision in record["references"].items()
                if key in dependencies(name)
            }
            artifacts[name] = record
    records["artifacts"] = artifacts
    revisions = {key: _revision(getattr(args, key)) for key in selected}
    if "fcpw" in selected:
        for name in ("eigen", "slang-rhi"):
            path = args.fcpw / "deps" / name
            if (path / ".git").exists():
                revisions[name] = _revision(path)
    runtime = {}
    auxiliary = {}
    if "fcpw" in selected:
        runtime = {
            str(path): file_hash(path)
            for path in sorted((args.fcpw / "include/fcpw/gpu").rglob("*.slang"))
        }
        runtime.update(
            {
                str(path): file_hash(path)
                for path in sorted(
                    (args.fcpw_build / "_deps/slang-src/lib").glob("*.so*")
                )
                if path.is_file()
            }
        )
        shader = ROOT / "fcpw/fcpw_boundary.slang"
        published = stage / shader.name
        shutil.copyfile(shader, published)
        auxiliary[shader.name] = file_hash(published)
    compilers = {
        compiler: subprocess.check_output([compiler, "--version"], text=True).strip()
        for compiler in sorted({args.cxx, args.cuda_host, str(args.cuda / "bin/nvcc")})
        if shutil.which(compiler)
    }

    def source_hashes():
        return {
            str(p.relative_to(ROOT)): file_hash(p)
            for p in sorted(ROOT.rglob("*"))
            if p.suffix in {".cu", ".cpp", ".h", ".slang"}
        }

    sources = source_hashes()
    for name, flags in _commands(args, selected):
        binary = stage / name
        command = list(map(str, [*flags, "-o", binary]))
        print(shlex.join(command), flush=True)
        subprocess.run(command, check=True)
        artifacts[name] = {
            "sha256": file_hash(binary),
            "command": command,
            "references": {
                key: revisions[key] for key in dependencies(name) if key in revisions
            },
            "cuda_root": str(args.cuda),
            "compilers": compilers,
            "sources": sources,
        }
        if name == "fcpw_queries":
            artifacts[name].update(runtime_sha256=runtime, auxiliary=auxiliary)
    if sources != source_hashes():
        raise RuntimeError("native sources changed during compilation; rebuild")
    for path in stage.iterdir():
        path.replace(args.output / path.name)
    temporary = stage / "build.json"
    temporary.write_text(json.dumps(records, indent=2) + "\n")
    temporary.replace(manifest)

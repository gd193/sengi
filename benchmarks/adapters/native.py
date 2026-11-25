"""Independent C++/CUDA/Slang implementations behind the named fixture protocol."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from functools import lru_cache
from pathlib import Path

import numpy as np

from benchmarks.results import file_hash

KD_VARIANTS = (
    "cudakdtree_fixed",
    "cudakdtree_heap",
    "cudakdtree_fixed_stack",
    "cudakdtree_heap_stack",
)
NAMES = ("cubql", "fcpw", "optix", *KD_VARIANTS)
POINT_K = (1, 4, 8, 16, 32, 64, 128)
RAYS = ("first_hit", "any_hit", "count_hits", "all_hits")
FCPW_GPU_QUERIES = (
    "first_hit",
    "any_hit",
    "closest_point",
    "closest_silhouette",
    "star_radius",
    "sample_boundary",
)
SUPPORT = {
    "cubql": {
        "triangles": (
            *RAYS,
            "closest_point",
            "classify_point",
            "aabb_overlap",
            "knn",
            "radius_any",
            "radius_count",
            "region_any",
            "region_count",
            "classify_box",
        ),
        "points2": ("closest_point", "knn"),
        "points3": ("closest_point", "knn"),
    },
    "fcpw": {
        "segments2": FCPW_GPU_QUERIES,
        "triangles": FCPW_GPU_QUERIES,
    },
    "optix": {"triangles": (*RAYS, "classify_point")},
    **{name: {"points2": ("knn",), "points3": ("knn",)} for name in KD_VARIANTS},
}


def _width(case):
    if case.query == "knn":
        return case.parameters.get("k", 4)
    return case.parameters.get("max_hits", 4 if case.workload == "builtin" else 16)


def support(name, case, primitive, closed):
    mode = case.query
    params = case.parameters
    if mode not in SUPPORT[name].get(primitive, ()):
        return f"{name} has no {primitive} implementation of {mode}"
    if case.metric == "angular":
        return f"{name} implements the Euclidean metric only"
    if mode == "knn":
        if primitive == "triangles" and _width(case) != 4:
            return "cuBQL triangle KNN requires k=4"
        if primitive.startswith("points") and _width(case) not in POINT_K:
            return f"{name} point KNN supports k in {POINT_K}"
    if mode == "region_count" and params.get("count_limit") is not None:
        return "cuBQL region_count benchmarks a complete count"
    if (
        mode in ("count_hits", "all_hits")
        and name != "cubql"
        and (params.get("line") or params.get("mixed_line", case.workload == "builtin"))
    ):
        return f"{name} supports forward crossings only"
    return None


def binary_name(name, primitive):
    if name == "cubql":
        return (
            "cubql_mesh" if primitive == "triangles" else f"cubql_points{primitive[-1]}"
        )
    if name in KD_VARIANTS:
        return f"{name}_d{primitive[-1]}"
    return {
        "fcpw": "fcpw_queries",
        "optix": "optix_mesh",
    }[name]


@lru_cache
def _read_artifacts(path, mtime):
    manifest = json.loads(path.read_text())
    if not isinstance(manifest, dict) or not isinstance(
        manifest.get("artifacts"), dict
    ):
        raise ValueError(f"invalid native build manifest: {path}")  # noqa: TRY004 -- malformed file contents
    return manifest["artifacts"]


def _artifact(name, primitive, binaries):
    binary = Path(binaries).resolve() / binary_name(name, primitive)
    path = binary.with_name("build.json")
    if not os.access(binary, os.X_OK) or not path.is_file():
        raise FileNotFoundError(f"build {binary.name} with python -m benchmarks build")
    return binary, _artifact_record(binary)


def _artifact_record(file):
    path = file.with_name("build.json")
    record = _read_artifacts(path, path.stat().st_mtime_ns).get(file.name)
    if record is None:
        raise FileNotFoundError(f"{file.name} is absent from {path}")
    return record


@lru_cache
def _digest_at(path, changed, size):
    return file_hash(path)


def _runtime_digest(path):
    stat = path.stat()
    return _digest_at(path, stat.st_ctime_ns, stat.st_size)


def fingerprints(binaries, pairs):
    """Hash only executables, build records and dependencies used by the plan."""
    root = Path(binaries).resolve()
    paths = set()
    records = {}
    for adapter, primitive in sorted(pairs):
        try:
            binary, record = _artifact(adapter, primitive, root)
        except (FileNotFoundError, ValueError, KeyError):
            continue
        records[binary.name] = record
        paths.add(binary)
        paths.update(map(Path, record.get("runtime_sha256", {})))
        paths.update(root / name for name in record.get("auxiliary", {}))
        if adapter == "optix":
            paths.add(root / "optix_programs.ptx")
    hashes = {}
    for path in sorted(paths):
        try:
            hashes[str(path)] = _runtime_digest(path)
        except FileNotFoundError:
            hashes[str(path)] = None
    return {"records": records, "files": hashes}


def available(name, case, primitive, options):
    if name != "optix" and options.block_dim is not None and options.block_dim > 1024:
        return f"{name} requires a CUDA block size at most 1024"
    try:
        binary, record = _artifact(name, primitive, options.binaries)
        if name == "optix" and not binary.with_name("optix_programs.ptx").is_file():
            return "OptiX PTX is missing. Rebuild OptiX."
        for filename in record.get("auxiliary", {}):
            if not binary.with_name(filename).is_file():
                return f"native shader artifact is missing: {filename}"
        for path in record.get("runtime_sha256", {}):
            if not Path(path).is_file():
                return f"native runtime dependency is missing: {path}"
    except (OSError, ValueError, KeyError) as exc:
        return str(exc)
    return None


def _environment(record, device):
    env = os.environ.copy()
    cuda = Path(record["cuda_root"])
    env["CUDA_PATH"] = str(cuda / "targets/x86_64-linux")
    paths = [str(cuda / "lib")]
    if env.get("LD_LIBRARY_PATH"):
        paths.append(env["LD_LIBRARY_PATH"])
    env["LD_LIBRARY_PATH"] = os.pathsep.join(paths)
    import warp as wp

    uuid = wp.get_device(device).uuid
    env["CUDA_VISIBLE_DEVICES"] = uuid if uuid.startswith("GPU-") else "GPU-" + uuid
    return env


def _write_fixture(directory, geometry, jobs, options, block):
    (directory / "out").mkdir()
    (directory / "arrays").mkdir()
    seen = set()
    lines = [
        "\t".join(
            map(
                str,
                (
                    "sengi-native-1",
                    block,
                    options.warmups,
                    options.launches,
                    options.repeats,
                    options.reference_leaf_size,
                ),
            )
        )
    ]

    def array_row(record, key, array):
        value = np.ascontiguousarray(array, dtype=array.dtype.newbyteorder("<"))
        fingerprint = hashlib.sha256(f"{value.dtype.str}:{value.shape}\n".encode())
        fingerprint.update(memoryview(value).cast("B"))
        identity = fingerprint.hexdigest()
        relative = f"arrays/{identity}.bin"
        if identity not in seen:
            value.tofile(directory / relative)
            seen.add(identity)
        shape = ",".join(map(str, value.shape))
        lines.append(f"{record}\t{key}\t{relative}\t{value.dtype.str}\t{shape}")

    for key, array in geometry.arrays.items():
        array_row("geometry", key, array)
    for i, job in enumerate(jobs):
        case = job.case
        lines.append(f"case\t{i}\t{case.query}\t{case.batch_size}")
        for key, array in job.inputs.items():
            array_row("input", key, array)
        params = dict(case.parameters)
        if case.query in ("all_hits", "knn"):
            params["max_hits" if case.query == "all_hits" else "k"] = _width(case)
        if geometry.primitive.startswith("points"):
            radii = job.inputs["radii"]
            cap = radii[0]
            if not np.all(radii == cap):
                raise ValueError("native point queries require a single cap per batch")
            squared = float(cap) if case.query == "knn" else float(cap) ** 2
            params["cap_squared"] = (
                squared if squared <= float(np.finfo(np.float32).max) else np.inf
            )
        for key, value in params.items():
            lines.append(
                f"parameter\t{key}\t{int(value) if isinstance(value, bool) else value}"
            )
    (directory / "manifest.tsv").write_text("\n".join(lines) + "\n")


def _decode(name, geometry, job, prefix):
    mode = job.case.query
    n = job.case.batch_size
    dimension = 2 if geometry.primitive.endswith("2") else 3
    one = np.ones(n, dtype=np.int32)

    def load(suffix, dtype="<f4", width=None):
        values = np.fromfile(prefix.with_suffix(suffix), dtype=dtype)
        return values.reshape(n) if width is None else values.reshape(n, width)

    if name == "fcpw":
        hit = load(
            ".bin",
            dtype=[
                ("point", "3f4"),
                ("normal", "3f4"),
                ("uv", "2f4"),
                ("distance", "f4"),
                ("id", "i4"),
            ],
        )
        ids = hit["id"]
        found = ids >= 0
        distance = hit["distance"]
        if mode == "any_hit":
            value = found.astype(np.int32)
            return {"value": value, "status": value}
        if mode == "star_radius":
            return {"distance": distance, "status": one}
        if mode == "sample_boundary":
            return {
                "point": hit["point"][:, :dimension],
                "normal": hit["normal"][:, :dimension],
                "pdf": distance,
                "id": ids,
                "status": found.astype(np.int32),
            }
    if mode in ("closest_point", "closest_silhouette", "first_hit"):
        if name != "fcpw":
            ids = load(".ids", "<i4")
            found = ids >= 0
            distance = load(".dist")
        result = {
            "id": ids,
            "distance": np.where(
                found, distance, job.inputs["max_t"] if mode == "first_hit" else np.inf
            ),
            "status": found.astype(np.int32),
        }
        if mode == "first_hit":
            result["hit"] = found
        else:
            points = (
                hit["point"][:, :dimension]
                if name == "fcpw"
                else load(".points", width=dimension)
            )
            result["point"] = np.where(found[:, None], points, 0)
        return result
    if mode in ("knn", "all_hits"):
        width = _width(job.case)
        squared = mode == "knn" and geometry.primitive.startswith("points")
        ids = load(".ids", "<i4", width)
        distance = load(".scores" if squared else ".dist", width=width)
        if squared:
            count = load(".count", "<i4")
            status = load(".status", "<i4")
        elif mode == "all_hits":
            raw = load(".value", "<i4")
            count = raw
            status = np.where(raw > width, -2, raw > 0).astype(np.int32)
        else:
            count = np.count_nonzero(ids >= 0, axis=1).astype(np.int32)
            status = (count > 0).astype(np.int32)
        return {"distance": distance, "id": ids, "count": count, "status": status}
    value = load(".value", "<i4")
    status = (
        one
        if mode in ("region_any", "region_count", "classify_point", "classify_box")
        else (value > 0).astype(np.int32)
    )
    return {"value": value, "status": status}


def execute(name, geometry, jobs, options):
    for job in jobs:
        excluded = job.inputs.get("excluded")
        if excluded is not None and name != "cubql" and np.any(excluded != -1):
            raise ValueError(f"{name} has no primitive exclusion binding")
        min_t = job.inputs.get("min_t")
        if min_t is not None and name == "fcpw" and np.any(min_t != 0):
            raise ValueError("FCPW rays require a zero lower bound")
    binary, record = _artifact(name, geometry.primitive, options.binaries)
    owner = "cudakdtree" if name in KD_VARIANTS else name
    _verify_artifact(binary, record, owner)
    if name == "optix":
        shader = binary.with_name("optix_programs.ptx")
        ptx_record = _artifact_record(shader)
        _verify_artifact(shader, ptx_record, owner)
    shader_hashes = record.get("auxiliary", {})
    for filename, expected in shader_hashes.items():
        if file_hash(binary.with_name(filename)) != expected:
            raise ValueError(f"native shader artifact changed since build: {filename}")
    runtime = record.get("runtime_sha256", {})
    for path, expected in runtime.items():
        if _runtime_digest(Path(path)) != expected:
            raise ValueError(f"native runtime dependency changed since build: {path}")
    block = (
        128
        if name == "optix"
        else options.block_dim or (256 if name == "fcpw" else 128)
    )
    metadata = {
        "device": options.device,
        "build_scope": {
            "fcpw": "CPU BVH build before GPU upload",
            "cubql": "GPU BVH build from resident bounds",
            "optix": "GPU acceleration structure build from resident geometry",
        }.get(name, "GPU k-d tree build"),
        "binary_sha256": record["sha256"],
        "runtime_sha256": runtime,
        "reference_revisions": record["references"],
        "compilers": record.get("compilers", {}),
        "block_dim": None if name == "optix" else block,
        "launches": options.launches,
        "warmups": max(1, options.warmups) if name == "fcpw" else options.warmups,
        "repeats": options.repeats,
        "timing_scope": "resident kernel",
        "instrumentation": "CUPTI activity" if name == "optix" else "CUDA graph events",
        "precision": {
            "input": "float32",
            "output": "float32 / int32",
            "internal": "float32"
            if name == "cubql" or name in KD_VARIANTS or name == "fcpw"
            else "unknown",
        },
        "builder": (
            "SAH"
            if name == "fcpw" or (name == "cubql" and geometry.primitive != "points2")
            else "spatial median"
            if name == "cubql"
            else "native default"
        ),
    }
    if name == "optix":
        shader_hashes = {shader.name: ptx_record["sha256"]}
    if shader_hashes:
        metadata["shader_sha256"] = shader_hashes
    if name == "cubql":
        metadata["leaf_size"] = options.reference_leaf_size
    with tempfile.TemporaryDirectory(prefix="sengi-native-") as tmp:
        directory = Path(tmp)
        _write_fixture(directory, geometry, jobs, options, block)
        command = [str(binary), str(directory)]
        if name == "fcpw":
            command += [
                record["references"]["fcpw"]["path"],
                "2" if geometry.primitive == "segments2" else "3",
            ]
        process = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=_environment(record, options.device),
        )
        if process.returncode:
            raise RuntimeError(
                f"{binary.name} exited with code {process.returncode}:\n"
                f"{process.stdout}{process.stderr}"
            )
        for i, job in enumerate(jobs):
            prefix = directory / "out" / str(i)
            samples = np.loadtxt(prefix.with_suffix(".ms"), ndmin=1)
            if samples.shape != (options.repeats,) or not np.all(
                np.isfinite(samples) & (samples > 0)
            ):
                raise ValueError(
                    f"invalid timings from {binary.name} for {job.case.query}"
                )
            stats = prefix.with_suffix(".json")
            properties = json.loads(stats.read_text()) if stats.is_file() else {}
            phases = prefix.with_suffix(".prepare.json")
            if phases.is_file():
                properties.update(json.loads(phases.read_text()))
            yield (
                job,
                _decode(name, geometry, job, prefix),
                samples.tolist(),
                dict(metadata, **properties),
            )


def _verify_artifact(file, record, owner):
    if file_hash(file) != record["sha256"]:
        raise ValueError(f"native artifact changed since build: {file}")
    root = Path(__file__).resolve().parents[1] / "references"
    sources = {p for p in (root / owner).rglob("*") if p.suffix in {".cu", ".cpp", ".h", ".slang"}}
    sources.update(root / key for key in record["sources"] if key.startswith(owner + "/"))
    sources.add(root / "fixture.h")
    if owner in {"cubql", "cudakdtree"}:
        sources.add(root / "cuda_fixture.h")
    for source in sources:
        key = str(source.relative_to(root))
        if not source.is_file() or record["sources"].get(key) != _runtime_digest(source):
            raise ValueError(f"{key} changed since building {file.name}; rebuild {owner}")

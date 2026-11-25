"""Plan compatible cases before building indexes, then publish executions one at a time."""

from __future__ import annotations

import fcntl
import importlib.metadata
import importlib.util
import itertools
import json
import platform
import subprocess
import tomllib
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from benchmarks import assets, cases, workloads
from benchmarks.adapters import native, sengi, warp
from benchmarks.execution import Job, Timing, require_cuda_device, selected_suite
from benchmarks.queries import validation
from benchmarks.results import (
    RUN_SCHEMA,
    digest,
    file_hash,
    load_arrays,
    store_arrays,
    write_json,
    write_report,
)

ROOT = Path(__file__).resolve().parents[1]
ADAPTERS = ("sengi", "warp", *native.NAMES)


def support(name, case, primitive, closed):
    reason = cases.geometry_support(case, primitive, closed)
    if reason:
        return reason
    if name in ("sengi", "warp"):
        return (sengi if name == "sengi" else warp).support(case, primitive, closed)
    return native.support(name, case, primitive, closed)


def plan(args):
    assets_by_name = assets.discover_assets(args.assets)
    preset = selected_suite(args)
    selected = args.adapter or ADAPTERS
    if unknown := set(selected).difference(ADAPTERS):
        raise ValueError(f"unknown adapter(s): {sorted(unknown)}")
    if args.query and (unknown := set(args.query).difference(cases.CATALOG)):
        raise ValueError(f"unknown queries: {sorted(unknown)}")

    def axis(requested, preset_values):
        if not requested:
            return preset_values
        if args.suite and "*" not in preset_values and "*" not in requested:
            return [value for value in requested if value in preset_values]
        return requested if "*" not in requested else preset_values

    jobs = {}
    for selectors in workloads.SUITES[preset]:
        queries = selectors["query"]
        if args.query:
            queries = [query for query in args.query if query in queries]
        if not queries:
            continue
        if args.mesh and "*" in args.mesh:
            if all(name in workloads.BUILTINS for name in selectors["geometry"]):
                continue
            geometries = tuple(assets_by_name)
        else:
            geometries = axis(args.mesh, selectors["geometry"])
        for geometry in geometries:
            asset = assets_by_name.get(geometry)
            if asset is None and geometry not in workloads.BUILTINS:
                raise ValueError(f"unknown geometry {geometry}")
            sizes = axis(args.size, selectors.get("size", ("*",)))
            if "*" in sizes:
                sizes = tuple(asset.variants) if asset else ("1024",)
            for size in sizes:
                size = str(size)
                if asset and size not in asset.variants:
                    if args.mesh and args.size:
                        raise ValueError(f"{geometry} has no tier {size}")
                    continue
                primitive, closed = (
                    ("triangles", asset.closed)
                    if asset
                    else workloads.geometry_info(geometry, size)
                )
                default_workload = selectors.get("workload", ("gaussian",))
                if args.mesh and not args.suite and primitive != "triangles":
                    default_workload = ("gaussian",)
                axes = (
                    queries,
                    args.workload or default_workload,
                    args.count or selectors.get("batch_size", (512,)),
                    args.seed or selectors.get("seed", (404,)),
                    axis(args.metric, selectors.get("metric", ("euclidean",))),
                )
                for query, workload, n, seed, metric in itertools.product(*axes):
                    parameters = dict(selectors.get("parameters", {}))
                    parameters.update(args.params)
                    case = cases.Case(
                        query=query,
                        geometry=geometry,
                        size=size,
                        workload=workload,
                        batch_size=n,
                        seed=seed,
                        metric=metric,
                        parameters=parameters,
                    )
                    case_id = digest(asdict(case))
                    jobs[case_id] = (
                        case,
                        primitive,
                        closed,
                        {
                            name: support(name, case, primitive, closed)
                            for name in selected
                        },
                    )
    if not jobs:
        raise ValueError("no cases matched the selection")
    return jobs


def hardware(device):
    import warp as wp

    gpu = wp.get_device(device)
    return {
        "host": platform.node(),
        "gpu": gpu.name,
        "uuid": gpu.uuid,
        "cuda_toolkit": wp.get_cuda_toolkit_version(),
        "driver": wp.get_cuda_driver_version(),
    }


def provenance(options, selected):
    package = Path(importlib.util.find_spec("sengi").origin).parent
    sources = {}
    for prefix, base in (("benchmarks", ROOT / "benchmarks"), ("sengi", package)):
        for path in sorted(base.rglob("*")):
            if path.suffix in {".py", ".h", ".cpp", ".cu", ".slang"}:
                sources[prefix + "/" + str(path.relative_to(base))] = file_hash(path)
    return {
        "commit": subprocess.check_output(
            ["git", "-C", ROOT, "rev-parse", "HEAD"], text=True
        ).strip(),
        "dirty": bool(
            subprocess.check_output(["git", "-C", ROOT, "status", "--porcelain"])
        ),
        "sources": sources,
        "binaries": native.fingerprints(options.binaries, selected),
        "versions": {
            "python": platform.python_version(),
            **{
                p: importlib.metadata.version(p)
                for p in ("numpy", "jax", "jaxlib", "warp-lang", "trimesh")
            },
            "sengi": tomllib.loads((ROOT / "pyproject.toml").read_text())["project"][
                "version"
            ],
        },
    }


def publish(root, item, **data):
    write_json(root / "results" / (item["id"] + ".json"), data)


def check_answers(job, geometry, outputs, expected):
    try:
        return validation.validate(job.case, geometry, job.inputs, outputs, expected)
    except (
        ValueError,
        TypeError,
        KeyError,
        IndexError,
        ArithmeticError,
        RuntimeError,
    ) as exc:
        return "failed", {"error": f"validation: {type(exc).__name__}: {exc}"}


def observations(detail, times):
    observation = {
        key: detail[key] for key in ("hit_rate", "cap_frequency") if key in detail
    }
    if "sampling" in detail:
        sampling = detail["sampling"]
        observation.update(
            accepted_samples_per_s=float(
                sampling["accepted"] * 1000 / np.median(times)
            ),
            null_fraction=sampling["null_fraction"],
        )
    return observation


def comparison_peer(anchor, cid, name):
    peer, outputs = anchor.get(cid, (None, None))
    if name in native.KD_VARIANTS and peer in native.KD_VARIANTS:
        return None, None
    return peer, outputs


def save_measurement(
    root,
    item,
    job,
    geometry,
    geo_hash,
    input_hash,
    outputs,
    samples,
    metadata,
    options,
    anchor,
):
    name = item["adapter"]
    require_cuda_device(metadata.get("device"))
    times = np.asarray(samples, dtype=float)
    if (
        times.ndim != 1
        or len(times) != options.repeats
        or not np.all(np.isfinite(times) & (times > 0))
    ):
        raise ValueError(f"{name} produced invalid timing samples")
    output_key = store_arrays(root, outputs)
    peer, expected = comparison_peer(anchor, job.key, name)
    status, detail = check_answers(job, geometry, outputs, expected)
    publish(
        root,
        item,
        status=status,
        validation=detail,
        compared_to=peer,
        observation=observations(detail, times),
        samples_ms=times.tolist(),
        metadata=metadata,
        outputs=output_key,
        geometry_sha256=geo_hash,
        input_sha256=input_hash,
    )
    return status


def execute_group(root, group, geometry, options, resume):
    geo_hash = store_arrays(root, geometry.arrays)
    batches = {}
    knn_inputs = {}
    for item, case in group:
        if item["case_id"] in batches:
            continue
        if case.query == "knn":
            spec = asdict(case)
            spec["parameters"].pop("k", None)
            input_id = digest(spec)
            if input_id not in knn_inputs:
                arrays = workloads.make_inputs(case, geometry)
                knn_inputs[input_id] = arrays, store_arrays(root, arrays)
            arrays, key = knn_inputs[input_id]
        else:
            arrays = workloads.make_inputs(case, geometry)
            key = store_arrays(root, arrays)
        batches[item["case_id"]] = (Job(case, arrays, item["case_id"]), key)

    by_adapter = defaultdict(list)
    anchor = {}
    for item, case in group:
        by_adapter[item["adapter"]].append(item)
    for name in sorted(by_adapter, key=lambda x: (x != "sengi", x)):
        pending = {}
        for item in by_adapter[name]:
            cid = item["case_id"]
            p = root / "results" / (item["id"] + ".json")
            if resume and p.is_file():
                old = json.loads(p.read_text())
                if (
                    old.get("geometry_sha256") == geo_hash
                    and old.get("input_sha256") == batches[cid][1]
                    and "outputs" in old
                    and old["status"] != "failed"
                ):
                    answers = load_arrays(root, old["outputs"])
                    peer, expected = comparison_peer(anchor, cid, name)
                    status, detail = check_answers(
                        batches[cid][0], geometry, answers, expected
                    )
                    updates = {
                        "status": status,
                        "validation": detail,
                        "compared_to": peer,
                        "observation": observations(detail, old["samples_ms"]),
                    }
                    if any(old.get(k) != v for k, v in updates.items()):
                        publish(root, item, **(old | updates))
                    if cid not in anchor and status == "unchecked":
                        anchor[cid] = (name, answers)
                    continue
            pending[cid] = item
        if not pending:
            continue
        jobs = [batches[k][0] for k in pending]
        try:
            adapter = (
                (sengi if name == "sengi" else warp)
                if name in {"sengi", "warp"}
                else None
            )
            stream = (
                adapter.execute(geometry, jobs, options)
                if adapter
                else native.execute(name, geometry, jobs, options)
            )
            for job, outputs, samples, metadata in stream:
                item = pending[job.key]
                status = save_measurement(
                    root,
                    item,
                    job,
                    geometry,
                    geo_hash,
                    batches[job.key][1],
                    outputs,
                    samples,
                    metadata,
                    options,
                    anchor,
                )
                if job.key not in anchor and status == "unchecked":
                    anchor[job.key] = (name, outputs)
                del pending[job.key]
            if pending:
                raise RuntimeError(
                    f"adapter returned no answer for {len(pending)} selected cases"
                )
        except Exception as exc:  # noqa: BLE001 -- record adapter failures and continue the matrix
            for item in pending.values():
                publish(
                    root, item, status="failed", reason=f"{type(exc).__name__}: {exc}"
                )
            print(f"{name}: {type(exc).__name__}: {exc}", flush=True)


def run(args, jobs):
    root = args.output.resolve()
    root.mkdir(parents=True, exist_ok=True)
    options = Timing(
        args.builder,
        args.leaf_size,
        args.device,
        args.launches,
        args.repeats,
        args.warmups,
        args.block_dim,
        args.binaries.resolve(),
        args.reference_leaf_size,
    )
    hw = hardware(options.device)
    settings = {**asdict(options), "binaries": str(options.binaries)}
    native_pairs = set()
    groups = defaultdict(list)
    rows = []
    for cid, (case, primitive, closed, adapters) in jobs.items():
        for name, reason in adapters.items():
            item = {
                "case": asdict(case),
                "case_id": cid,
                "adapter": name,
                "builder": options.builder if name == "sengi" else None,
            }
            item["id"] = digest(
                [cid, name, {k: v for k, v in settings.items() if k != "binaries"}, hw]
            )
            if not reason and name in native.NAMES:
                native_pairs.add((name, primitive))
                missing = native.available(name, case, primitive, options)
            else:
                missing = None
            item.update(
                planned_status="unsupported"
                if reason
                else "missing"
                if missing
                else "ready",
                planned_reason=reason or missing or "",
            )
            rows.append(item)
            point_fixture = (
                case.geometry in workloads.BUILTINS
                and case.workload in workloads.POINT_WORKLOADS
            )
            groups[
                case.geometry,
                case.size,
                case.seed if point_fixture else 73,
                case.workload if point_fixture else "gaussian",
            ].append((item, case))
    manifest = {
        "schema": RUN_SCHEMA,
        "settings": settings,
        "hardware": hw,
        "provenance": provenance(options, native_pairs),
        "plan": rows,
    }
    with (root / ".lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        saved = root / "run.json"
        if saved.exists():
            if not args.resume or digest(json.loads(saved.read_text())) != digest(
                manifest
            ):
                raise ValueError(
                    "output already contains a run; resume requires identical settings, plan, environment and sources"
                )
        else:
            write_json(saved, manifest)
        try:
            for key, group in groups.items():
                active = []
                for item, case in group:
                    if item["planned_status"] == "ready":
                        active.append((item, case))
                    else:
                        publish(
                            root,
                            item,
                            status=item["planned_status"],
                            reason=item["planned_reason"],
                        )
                if not active:
                    continue
                name, size, seed, workload = key
                print(f"{name}/{size}: {len(active)} executions", flush=True)
                try:
                    geom = (
                        workloads.builtin_geometry(
                            name, size, seed=seed, workload=workload
                        )
                        if name in workloads.BUILTINS
                        else assets.load_geometry(
                            name, size, root=args.assets, cache=args.cache
                        )
                    )
                except (OSError, RuntimeError, ValueError) as exc:
                    for item, _ in active:
                        publish(root, item, status="failed", reason=str(exc))
                    continue
                execute_group(root, active, geom, options, args.resume)
        except (KeyboardInterrupt, SystemExit):
            write_report([root], root / "summary.csv")
            raise
        rows = write_report([root], root / "summary.csv")
        counts = Counter(r["status"] for r in rows)
        print(dict(counts), root / "summary.csv", flush=True)
        explicit_unsupported = (
            all(
                len(values or ()) == 1
                for values in (args.mesh, args.query, args.adapter)
            )
            and counts["unsupported"]
        )
        return int(
            bool(explicit_unsupported)
            or any(counts[x] for x in ("failed", "mismatch", "pending"))
            or (bool(args.adapter) and bool(counts["missing"]))
            or not any("median_ms" in r for r in rows)
        )

"""Atomic run records and comparisons of like-for-like measurements."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import statistics
from collections import defaultdict
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from benchmarks.execution import require_cuda_device
from benchmarks.queries.validation import sample_area_is_resolved


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, allow_nan=False).encode()
    ).hexdigest()


def array_digest(arrays):
    checksum = hashlib.sha256()
    for name, value in sorted(arrays.items()):
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject:
            raise ValueError(f"{name} contains Python objects")
        checksum.update(json.dumps((name, array.dtype.str, array.shape)).encode())
        checksum.update(memoryview(array).cast("B") if array.size else b"")
    return checksum.hexdigest()


def atomic_file(path, write):
    path.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=path.parent, delete=False) as stream:
        try:
            write(stream)
            stream.close()
            os.replace(stream.name, path)
        finally:
            Path(stream.name).unlink(missing_ok=True)


def write_json(path, value):
    atomic_file(
        path,
        lambda f: f.write(
            json.dumps(value, sort_keys=True, allow_nan=False, indent=2).encode()
        ),
    )


def store_arrays(root, arrays):
    key = array_digest(arrays)
    path = root / "arrays" / (key + ".npz")
    if not path.exists():
        atomic_file(path, lambda f: np.savez(f, **arrays))
    return key


def load_arrays(root, key):
    with np.load(root / "arrays" / (key + ".npz"), allow_pickle=False) as archive:
        result = dict(archive)
    if array_digest(result) != key:
        raise ValueError(f"corrupt result arrays: {key}")
    return result


RUN_SCHEMA = 3


def load_rows(root):
    manifest = json.loads((root / "run.json").read_text())
    if manifest.get("schema") != RUN_SCHEMA:
        raise ValueError("unsupported benchmark record schema; rerun the benchmark")
    for job in manifest["plan"]:
        path = root / "results" / (job["id"] + ".json")
        record = (
            json.loads(path.read_text()) if path.exists() else {"status": "pending"}
        )
        if "samples_ms" in record:
            require_cuda_device(record["metadata"].get("device"))
        yield {**job, **record, "hardware": manifest["hardware"]}


def report_rows(root, *, fp32_only=False):
    records = list(load_rows(root))
    if fp32_only:
        records = [
            r
            for r in records
            if r.get("metadata", {}).get("precision", {}).get("internal") == "float32"
        ]
    pairs = {(r["case_id"], r["adapter"]): r for r in records}
    agreements = defaultdict(list)
    disagreements = defaultdict(list)
    disputed = set()
    for r in records:
        key = (r["case_id"], r["adapter"])
        peer = (r["case_id"], r.get("compared_to"))
        if r.get("compared_to") and peer not in pairs:
            r = pairs[key] = dict(r, status="unchecked", compared_to=None, validation={})
        elif r["status"] == "sample support" and not sample_area_is_resolved(r["validation"]):
            r = pairs[key] = dict(r, status="unchecked", reason="sample area uncertainty exceeds 10%")
        if r["status"] == "mismatch":
            disputed.add(r["case_id"])
        if peer in pairs and (r["status"] in GOOD or r["status"] == "mismatch"):
            peers = agreements if r["status"] in GOOD else disagreements
            peers[key].append(peer[1])
            peers[peer].append(key[1])
    for key, row in pairs.items():
        case = row["case"]
        yield_row = {
            **case,
            "adapter": row["adapter"],
            "builder": row.get("builder"),
            "status": row["status"],
            "case_id": row["case_id"],
            "id": row["id"],
            "hardware": row["hardware"],
            "disputed": row["case_id"] in disputed,
            "validation": row.get("validation", {}),
            "reason": row.get("reason", ""),
        }
        if "samples_ms" in row:
            metadata = row["metadata"]
            for field in (
                "instrumentation",
                "build_ms",
                "build_scope",
                "geometry_transfer_ms",
                "launches",
                "warmups",
                "repeats",
                "boundary_ms",
                "block_dim",
                "leaf_size",
            ):
                if field in metadata:
                    yield_row[field] = metadata[field]
            ms = statistics.median(row["samples_ms"])
            yield_row.update(
                median_ms=ms,
                mad_ms=statistics.median(
                    abs(sample - ms) for sample in row["samples_ms"]
                ),
                queries_per_s=case["batch_size"] * 1000 / ms,
                timing_scope=row["metadata"].get("timing_scope"),
                precision=row["metadata"].get("precision"),
                input_sha256=row["input_sha256"],
                geometry_sha256=row["geometry_sha256"],
                device=row["metadata"].get("device"),
                actual_builder=row["metadata"].get("builder"),
                observation=row.get("observation", {}),
                compared_to=row.get("compared_to"),
                verified=bool(agreements[key]),
                agreeing_adapters=sorted(agreements[key]),
                disagreeing_adapters=sorted(disagreements[key]),
            )
            anchor = pairs.get((row["case_id"], "sengi"), {})
            if (
                row.get("compared_to") == "sengi"
                and row["status"] in GOOD
                and anchor.get("status") in GOOD | {"unchecked"}
            ):
                methods_match = all(
                    metadata.get(k) == anchor["metadata"].get(k)
                    for k in ("device", "timing_scope", "instrumentation")
                )
                if not methods_match:
                    yield_row["ratio_note"] = "timing methods or devices differ"
                else:
                    other = anchor["metadata"]
                    differences = [
                        key
                        for key in ("builder", "leaf_size")
                        if str(metadata.get(key)).lower() != str(other.get(key)).lower()
                    ]
                    if (
                        metadata["precision"]["internal"]
                        != other["precision"]["internal"]
                    ):
                        differences.append("precision")
                    if differences:
                        yield_row["ratio_note"] = "different " + ", ".join(differences)
                    yield_row["sengi_over_reference"] = (
                        statistics.median(anchor["samples_ms"]) / ms
                    )
                yield_row["comparison_devices"] = (
                    f"{anchor['metadata']['device']} / {metadata['device']}"
                )
        yield yield_row


GOOD = {"pass", "sample support"}


def write_report(roots, path, *, fp32_only=False):
    rows = list(report_rows(roots[0], fp32_only=fp32_only))
    if len(roots) == 2:
        before = {(r["case_id"], r["adapter"]): r for r in rows}
        rows = list(report_rows(roots[1], fp32_only=fp32_only))
        for row in rows:
            previous = before.pop((row["case_id"], row["adapter"]), None)
            if previous is None:
                row["comparison"] = "added"
                continue
            fields = (
                "hardware",
                "launches",
                "warmups",
                "repeats",
                "timing_scope",
                "precision",
                "instrumentation",
                "input_sha256",
                "geometry_sha256",
                "device",
                "actual_builder",
                "block_dim",
                "leaf_size",
            )
            changed = [k for k in fields if row.get(k) != previous.get(k)]
            if "median_ms" not in row or "median_ms" not in previous:
                row["comparison"] = (
                    row["status"]
                    if row["status"] == previous["status"]
                    else f"{previous['status']} to {row['status']}"
                )
            elif changed:
                row["comparison"] = "changed " + ", ".join(changed)
            elif not (row.get("verified") and previous.get("verified")):
                row["comparison"] = "unverified"
            else:
                row["candidate_over_baseline"] = (
                    row["median_ms"] / previous["median_ms"]
                )
                row["comparison"] = "matched"
        rows.extend({**old, "comparison": "removed"} for old in before.values())
    for row in rows:
        for k, v in row.items():
            if isinstance(v, (dict, list)):
                row[k] = json.dumps(v, sort_keys=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows

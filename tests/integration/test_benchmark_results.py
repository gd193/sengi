"""Exercise resumable publication and the comparison gate through the benchmark CLI."""

import json

import pytest
import warp as wp

from benchmarks.cli import main
from benchmarks.results import load_rows, write_report


@pytest.mark.skipif(not wp.is_cuda_available(), reason="requires a CUDA GPU")
def test_resume_replaces_missing_record_and_preserves_completed_timing(tmp_path):
    run = tmp_path / "gpu"
    args = [
        "run",
        "--mesh",
        "sphere",
        "--size",
        "small",
        "--count",
        "8",
        "--query",
        "first_hit,any_hit",
        "--adapter",
        "sengi,warp",
        "--launches",
        "1",
        "--repeats",
        "1",
        "--warmups",
        "1",
        "--output",
        str(run),
    ]
    assert main(args) == 0
    first, second, *_ = load_rows(run)
    assert first["status"] == "unchecked" and second["status"] == "pass"
    path = run / "results" / (second["id"] + ".json")
    path.unlink()
    assert list(load_rows(run))[1]["status"] == "pending"
    assert main([*args, "--resume"]) == 0
    restored = list(load_rows(run))
    assert restored[0] == first
    assert restored[1]["outputs"] == second["outputs"]

    rows = write_report([run, run], tmp_path / "comparison.csv")
    assert [row["candidate_over_baseline"] for row in rows] == [1] * 4
    candidate = tmp_path / "candidate"
    assert main([*args, "--reference-leaf-size", "8", "--output", str(candidate)]) == 0
    rows = write_report([run, candidate], tmp_path / "comparison.csv")
    assert [row["comparison"] for row in rows] == ["matched"] * 4
    record = json.loads(path.read_text())
    record["status"] = "mismatch"
    path.write_text(json.dumps(record))
    rows = write_report([run, run], tmp_path / "comparison.csv")
    assert "candidate_over_baseline" not in rows[1]
    with pytest.raises(SystemExit) as rejected:
        main([*args, "--resume", "--leaf-size", "8"])
    assert rejected.value.code == 2
    record["metadata"]["device"] = "cpu"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match="CUDA GPU"):
        write_report([run], tmp_path / "invalid.csv")


def test_suite_filters_apply_to_each_preset_block(capsys):
    assert (
        main(
            [
                "list",
                "--coverage",
                "--suite",
                "saturation",
                "--query",
                "closest_point",
                "--adapter",
                "sengi",
            ]
        )
        == 0
    )
    lines = capsys.readouterr().out.splitlines()
    assert len(lines) == 4 and all("gaussian" in line for line in lines)


def test_run_rejects_cpu_device(tmp_path):
    with pytest.raises(SystemExit) as rejected:
        main(
            ["run", "--adapter", "sengi", "--device", "cpu", "--output", str(tmp_path)]
        )
    assert rejected.value.code == 2
    assert not (tmp_path / "run.json").exists()


def test_full_suite_contains_the_distinct_cases_from_every_preset():
    from argparse import Namespace

    from benchmarks import runner, workloads

    args = Namespace(
        suite="full",
        assets=None,
        adapter=["sengi"],
        mesh=None,
        size=None,
        query=None,
        workload=None,
        count=None,
        seed=None,
        metric=None,
        params={},
    )
    full = runner.plan(args)
    expected = set()
    for name in workloads.SUITES.keys() - {"full"}:
        args.suite = name
        expected.update(runner.plan(args))
    assert full.keys() == expected
    args.suite = None
    assert runner.plan(args).keys() == expected
    for mesh in workloads.TRIANGLE_BASELINE:
        queries = {
            case.query
            for case, *_ in full.values()
            if case.geometry == mesh and case.size == "1.3m" and case.batch_size == 8192
        }
        assert queries == set(workloads.TRIANGLE_MODES)


def test_targeted_queries_use_the_same_workloads_as_the_triangle_suite(capsys):
    args = [
        "list",
        "--coverage",
        "--mesh",
        "bunny",
        "--size",
        "small",
        "--query",
        "first_hit,region_count",
        "--adapter",
        "sengi",
    ]
    assert main(args) == 0
    targeted = capsys.readouterr().out
    assert main([*args, "--suite", "triangles"]) == 0
    assert capsys.readouterr().out == targeted
    assert "first_hit euclidean gaussian" in targeted
    assert "region_count euclidean near_surface" in targeted
    assert len(targeted.splitlines()) == 2


def test_disagreement_preserves_passing_pairs_and_fp32_filter_requires_visible_peer(
    monkeypatch,
):
    from benchmarks import results

    def row(adapter, status, precision, peer=None):
        return {
            "adapter": adapter,
            "status": status,
            "compared_to": peer,
            "case_id": "ray",
            "id": adapter,
            "case": {"batch_size": 1},
            "hardware": {},
            "samples_ms": [2],
            "input_sha256": "i",
            "geometry_sha256": "g",
            "metadata": {"device": "cuda:0", "precision": {"internal": precision}},
        }

    records = [
        row("sengi", "unchecked", "mixed"),
        row("warp", "pass", "float32", "sengi"),
        row("cubql", "mismatch", "float32", "sengi"),
    ]
    monkeypatch.setattr(results, "load_rows", lambda _: records)
    anchor, passed, mismatch = results.report_rows(None)
    assert all(r["disputed"] for r in (anchor, passed, mismatch))
    assert anchor["verified"] and passed["verified"] and not mismatch["verified"]
    assert passed["sengi_over_reference"] == 1 and "precision" in passed["ratio_note"]
    assert anchor["disagreeing_adapters"] == ["cubql"]
    assert mismatch["disagreeing_adapters"] == ["sengi"]
    filtered = list(results.report_rows(None, fp32_only=True))
    assert len(filtered) == 2
    assert all(r["status"] == "unchecked" and not r["verified"] and not r["disputed"] for r in filtered)


def test_reference_only_anchor_and_resume_observations(tmp_path, monkeypatch):
    import numpy as np

    from benchmarks import runner
    from benchmarks.assets import Geometry
    from benchmarks.cases import Case
    from benchmarks.execution import Timing

    geom = Geometry("points3", "1", "points3", {"points": np.zeros((1, 3), np.float32)})
    case = Case("closest_point", geom.name, geom.size, batch_size=1)
    group = [
        ({"adapter": a, "id": a, "case_id": "case"}, case) for a in ("warp", "cubql")
    ]

    def original(geometry, jobs, options):
        for job in jobs:
            yield (
                job,
                {
                    "distance": np.array([np.inf]),
                    "id": np.array([-1]),
                    "status": np.array([0]),
                    "point": np.zeros((1, 3)),
                },
                [1.0],
                {"device": "cuda:0"},
            )

    monkeypatch.setattr(runner.warp, "execute", original)
    monkeypatch.setattr(runner.native, "execute", lambda name, *args: original(*args))
    options = Timing(repeats=1)
    runner.execute_group(tmp_path, group, geom, options, False)
    path = tmp_path / "results" / "warp.json"
    saved = json.loads(path.read_text())
    assert saved["compared_to"] == "cubql" and saved["status"] == "pass"
    saved["observation"]["hit_rate"] = 1.0
    path.write_text(json.dumps(saved))
    runner.execute_group(tmp_path, group, geom, options, True)
    assert json.loads(path.read_text())["observation"]["hit_rate"] == 0
    mtime = path.stat().st_mtime_ns
    runner.execute_group(tmp_path, group, geom, options, True)
    assert path.stat().st_mtime_ns == mtime


def test_two_run_report_preserves_unmeasured_status(tmp_path, monkeypatch):
    from benchmarks import results

    record = {
        "adapter": "warp",
        "case_id": "unsupported-query",
        "status": "unsupported",
    }
    monkeypatch.setattr(
        results, "report_rows", lambda root, **kwargs: iter([record.copy()])
    )
    compared = results.write_report(["before", "after"], tmp_path / "comparison.csv")
    assert compared[0]["comparison"] == "unsupported"
    assert "candidate_over_baseline" not in compared[0]

"""Benchmark runs persist point-KNN failures and validated answers."""

import csv

import numpy as np
import pytest

from benchmarks.adapters import sengi
from benchmarks.cli import main
from benchmarks.queries import validation


@pytest.mark.parametrize("failure_stage", ["prepare", "execute", "validate", None])
def test_point_knn_run_persists_failures(tmp_path, monkeypatch, failure_stage):
    monkeypatch.setattr("benchmarks.runner.hardware", lambda device: {"gpu": "fixture"})
    output = tmp_path / "run"

    def execute(geometry, jobs, options):
        if failure_stage == "prepare":
            raise RuntimeError("deliberate prepare failure")
        for job in jobs:
            if failure_stage == "execute":
                raise RuntimeError("deliberate execute failure")
            result = {
                "distance": np.zeros((1, 1), np.float32),
                "id": np.zeros((1, 1), np.int32),
                "count": np.ones(1, np.int32),
                "status": np.ones(1, np.int32),
            }
            yield job, result, [0.1], {"device": options.device}

    monkeypatch.setattr(sengi, "execute", execute)

    if failure_stage == "validate":

        def fail_validation(*args):
            raise RuntimeError("deliberate validate failure")

        monkeypatch.setattr(validation, "validate", fail_validation)

    status = main(
        [
            "run",
            "--adapter",
            "sengi",
            "--mesh",
            "points2",
            "--size",
            "1",
            "--query",
            "knn",
            "--workload",
            "uniform",
            "--count",
            "1",
            "--param",
            "k=1",
            "--repeats",
            "1",
            "--output",
            str(output),
        ]
    )

    with (output / "summary.csv").open() as report:
        (row,) = csv.DictReader(report)
    assert row["query"] == "knn"
    assert row["geometry"] == "points2"
    if failure_stage is None:
        assert status == 0
        assert row["status"] == "unchecked"
        assert float(row["median_ms"]) == 0.1
    else:
        assert status == 1
        assert row["status"] == "failed"
        (result,) = (output / "results").glob("*.json")
        assert f"deliberate {failure_stage} failure" in result.read_text()

"""List, run and compare Sengi query benchmarks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from time import monotonic

from benchmarks.execution import Timing, selected_suite
from benchmarks.results import write_report


def words(text):
    return [item for item in text.split(",") if item]


def positive(text):
    n = int(text)
    if n < 1:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return n


def parameter(text):
    try:
        name, value = text.split("=", 1)
        return name, json.loads(value)
    except (ValueError, TypeError) as exc:
        raise argparse.ArgumentTypeError("use name=JSON-value") from exc


def main(argv=None, *, deadline=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv[:1] == ["build"]:
        from benchmarks.references.build import build

        return build(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("build", help="compile external benchmark adapters")
    query = commands.add_parser(
        "list", help="list suites, geometry and queries, or show the planned matrix"
    )
    query.add_argument("--coverage", action="store_true")
    run = commands.add_parser("run", help="validate and time the selected cases")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--resume", action="store_true")
    run.add_argument("--cache", type=Path)
    run.add_argument("--binaries", type=Path, default=Path("output/benchmarks/bin"))
    run.add_argument("--builder", choices=["sah", "lbvh"], default="lbvh")
    run.add_argument("--leaf-size", type=positive, default=16)
    run.add_argument(
        "--reference-leaf-size", type=positive, default=4, help="cuBQL leaf size"
    )
    run.add_argument("--device", default="cuda:0")
    run.add_argument("--block-dim", type=positive)
    run.add_argument("--launches", type=positive, default=Timing.launches)
    run.add_argument("--repeats", type=positive, default=Timing.repeats)
    run.add_argument("--warmups", type=positive, default=Timing.warmups)
    for cmd in (query, run):
        cmd.add_argument("--suite")
        cmd.add_argument("--assets", type=Path)
        for flag in ("mesh", "size", "query", "adapter", "workload", "metric"):
            cmd.add_argument("--" + flag, type=words, help="comma-separated selections")
        cmd.add_argument("--count", type=lambda v: [positive(n) for n in words(v)])
        cmd.add_argument("--seed", type=lambda v: [int(n) for n in words(v)])
        cmd.add_argument(
            "--param", type=parameter, action="append", default=[], metavar="NAME=VALUE"
        )
    report = commands.add_parser(
        "report", help="write one run summary or compare a baseline and candidate"
    )
    report.add_argument(
        "--fp32-only",
        action="store_true",
        help="only include measurements with audited float32 internal arithmetic",
    )
    report.add_argument("runs", nargs="+", type=Path)
    report.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if (
        args.command == "run"
        and deadline is not None
        and selected_suite(args) == "full"
    ):
        # Reserve ten seconds for the worker to publish partial results and clean up.
        remaining = deadline - monotonic() - 10
        if remaining <= 0:
            parser.exit(124, "Full suite deadline reached during startup.\n")
        os.execvp(
            "timeout",
            [
                "timeout",
                "--signal=TERM",
                "--kill-after=10s",
                "--verbose",
                f"{remaining:.3f}s",
                sys.executable,
                "-m",
                "benchmarks.cli",
                *argv,
            ],
        )
    try:
        if args.command == "report":
            if len(args.runs) > 2:
                parser.error("report takes one or two run directories")
            path = args.output or args.runs[-1] / (
                "summary.csv" if len(args.runs) == 1 else "comparison.csv"
            )
            if args.fp32_only and args.output is None:
                path = path.with_stem(path.stem + "-fp32")
            records = write_report(args.runs, path, fp32_only=args.fp32_only)
            print(f"{len(records)} executions: {path}")
            return 0
        os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
        from benchmarks import assets, cases, runner, workloads

        if args.suite and args.suite not in workloads.SUITES:
            parser.error(
                f"unknown suite {args.suite}; available: {', '.join(workloads.SUITES)}"
            )
        args.params = dict(args.param)
        if args.command == "list" and not args.coverage:
            for name, asset in assets.discover_assets(args.assets).items():
                print(name, ", ".join(asset.variants))
            print("Built-in geometry:", ", ".join(workloads.BUILTINS))
            print("Queries:", ", ".join(cases.CATALOG))
            print("Suites:", ", ".join(workloads.SUITES))
            print("Adapters:", ", ".join(runner.ADAPTERS))
            return 0
        jobs = runner.plan(args)
        if args.command == "list":
            for case, _, _, support in jobs.values():
                print(
                    case.geometry,
                    case.size,
                    case.query,
                    case.metric,
                    case.workload,
                    case.batch_size,
                    " | ".join(
                        name + (f" (unsupported: {reason})" if reason else "")
                        for name, reason in support.items()
                    ),
                )
            return 0
        return runner.run(args, jobs)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")


if __name__ == "__main__":
    import signal

    def terminate(signum, frame):
        print(
            "Benchmark interrupted by SIGTERM. Completed results are on disk.",
            file=sys.stderr,
            flush=True,
        )
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    raise SystemExit(main())

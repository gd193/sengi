"""Run the benchmark CLI with a wall-clock budget for the full suite."""

from time import monotonic

if __name__ == "__main__":
    deadline = monotonic() + 1200
    from benchmarks.cli import main

    raise SystemExit(main(deadline=deadline))

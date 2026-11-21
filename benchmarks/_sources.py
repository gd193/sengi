"""Installed source inputs to benchmark fingerprints."""
from importlib.resources import files


def package_sources():
    root = files("sengi")
    return sorted(path for path in root.rglob("*") if path.suffix in {".py", ".h"})

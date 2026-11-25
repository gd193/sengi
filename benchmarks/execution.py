"""Data shared by the benchmark runner and implementation adapters."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from benchmarks.cases import Case


@dataclass(frozen=True)
class Job:
    case: Case
    inputs: dict[str, np.ndarray]
    key: str


def selected_suite(args):
    return args.suite or (
        "triangles" if args.mesh or args.query or args.size else "full"
    )


def require_cuda_device(device):
    if not isinstance(device, str) or not re.fullmatch(r"cuda(?::[0-9]+)?", device):
        raise ValueError("benchmarks require a CUDA GPU (--device cuda:N)")


@dataclass(frozen=True)
class Timing:
    builder: str = "lbvh"
    leaf_size: int = 16
    device: str = "cuda:0"
    launches: int = 1
    repeats: int = 3
    warmups: int = 1
    block_dim: int | None = None
    binaries: Path = Path("output/benchmarks/bin")
    reference_leaf_size: int = 4

    def __post_init__(self):
        require_cuda_device(self.device)
        if self.device == "cuda":
            import warp as wp

            object.__setattr__(self, "device", str(wp.get_device(self.device)))

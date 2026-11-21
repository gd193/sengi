"""Query families specialized from a BVH index specification."""
from importlib import import_module
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from . import (
        boundary_sample,
        classification,
        knn,
        nearest,
        overlap,
        radius,
        rays,
        region,
        region_classify,
        robin_radius,
        silhouette,
        star_radius,
    )

__all__ = [
    "boundary_sample",
    "classification",
    "knn",
    "nearest",
    "overlap",
    "radius",
    "rays",
    "region",
    "region_classify",
    "robin_radius",
    "silhouette",
    "star_radius",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"{__name__!r} has no attribute {name!r}")
    return import_module(f".{name}", __name__)

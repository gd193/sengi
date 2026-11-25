"""Case semantics, catalog and validation independent of implementation adapters."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

ALL_PRIMITIVES = ("points2", "points3", "segments2", "segments3", "triangles", "edges")
BOUNDARIES = ("segments2", "triangles")
EXACT_PRIMITIVES = ALL_PRIMITIVES[:-1]
NEAREST_INPUTS = ("points", "radii")
NEAREST_OUTPUTS = ("point", "distance", "id", "status")
RAY_INPUTS = ("origins", "directions", "min_t", "max_t")
BOX_INPUTS = ("lower", "upper")
VALUE_OUTPUTS = ("value", "status")
LIST_OUTPUTS = ("distance", "id", "count", "status")


@dataclass(frozen=True)
class Query:
    semantics: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    primitives: tuple[str, ...]
    closed: bool = False
    angular: tuple[str, ...] = ()
    parameters: tuple[str, ...] = ()


CATALOG = {
    "closest_point": Query(
        "Closest point and original primitive ID",
        NEAREST_INPUTS,
        NEAREST_OUTPUTS,
        ALL_PRIMITIVES,
        angular=("points2", "points3", "segments2", "segments3", "edges"),
        parameters=("cap",),
    ),
    "nearest_selected": Query(
        "Closest point emitted through a custom ID/point result policy",
        NEAREST_INPUTS,
        NEAREST_OUTPUTS,
        ("triangles",),
        parameters=("cap",),
    ),
    "knn": Query(
        "K distinct nearest primitives ordered by score; point scores are squared distances",
        NEAREST_INPUTS,
        LIST_OUTPUTS,
        ("points2", "points3", "triangles"),
        parameters=(
            "k",
            "cap",
            "cap_squared",
            "cache_bounds",
            "packed_nodes",
            "group_leaves",
        ),
    ),
    "aabb_overlap": Query(
        "Any primitive bounding box overlaps the closed query box",
        BOX_INPUTS,
        VALUE_OUTPUTS,
        ALL_PRIMITIVES,
    ),
    "region_any": Query(
        "Any primitive intersects the closed query box",
        BOX_INPUTS,
        VALUE_OUTPUTS,
        EXACT_PRIMITIVES,
    ),
    "region_count": Query(
        "Number of primitives intersecting the closed query box",
        BOX_INPUTS,
        VALUE_OUTPUTS,
        EXACT_PRIMITIVES,
        parameters=("count_limit",),
    ),
    "radius_any": Query(
        "Any primitive intersects the closed ball",
        NEAREST_INPUTS,
        VALUE_OUTPUTS,
        ALL_PRIMITIVES,
    ),
    "radius_count": Query(
        "Number of primitives intersecting the closed ball",
        NEAREST_INPUTS,
        VALUE_OUTPUTS,
        ALL_PRIMITIVES,
    ),
    "radius_accumulate": Query(
        "Sum 1 + (original primitive ID modulo 13) over primitives intersecting the closed ball",
        NEAREST_INPUTS,
        VALUE_OUTPUTS,
        ("triangles",),
    ),
    "first_hit": Query(
        "First ray hit within the supplied parameter interval",
        RAY_INPUTS,
        ("distance", "id", "hit", "status"),
        BOUNDARIES,
        parameters=("max_t",),
    ),
    "any_hit": Query(
        "Any ray hit within the supplied parameter interval",
        RAY_INPUTS,
        VALUE_OUTPUTS,
        BOUNDARIES,
        parameters=("max_t",),
    ),
    "count_hits": Query(
        "Number of half-open crossings within a ray or line parameter interval",
        (*RAY_INPUTS, "excluded"),
        VALUE_OUTPUTS,
        BOUNDARIES,
        parameters=("line", "mixed_line", "max_t"),
    ),
    "all_hits": Query(
        "Half-open crossings within a ray or line interval, capped by the result capacity",
        (*RAY_INPUTS, "excluded"),
        LIST_OUTPUTS,
        BOUNDARIES,
        parameters=("line", "mixed_line", "max_t", "max_hits"),
    ),
    "classify_point": Query(
        "Closed-boundary point classification: inside -1, contact 0, outside +1",
        ("points",),
        VALUE_OUTPUTS,
        BOUNDARIES,
        closed=True,
    ),
    "classify_box": Query(
        "Closed-boundary box classification: inside +1, contact 0, outside -1",
        BOX_INPUTS,
        VALUE_OUTPUTS,
        BOUNDARIES,
        closed=True,
    ),
    "closest_silhouette": Query(
        "Distance to closest eligible silhouette feature and its ID",
        NEAREST_INPUTS,
        NEAREST_OUTPUTS,
        BOUNDARIES,
        angular=BOUNDARIES,
    ),
    "star_radius": Query(
        "Distance to the closest eligible silhouette, limited to the supplied cap",
        NEAREST_INPUTS,
        ("distance", "status"),
        BOUNDARIES,
    ),
    "robin_radius": Query(
        "Capped radius of a star under the supplied uniform Robin coefficient",
        NEAREST_INPUTS,
        ("distance", "status"),
        BOUNDARIES,
        parameters=("robin_coefficient",),
    ),
    "sample_boundary": Query(
        "Draw a boundary point with area density inside a closed ball, or reject",
        ("points", "radii", "randoms"),
        ("point", "normal", "pdf", "id", "status"),
        BOUNDARIES,
    ),
}


@dataclass(frozen=True)
class Case:
    query: str
    geometry: str
    size: str
    workload: str = "gaussian"
    batch_size: int = 512
    seed: int = 404
    metric: str = "euclidean"
    parameters: dict = field(default_factory=dict)

    def __post_init__(self):
        from benchmarks.workloads import WORKLOADS

        if self.query not in CATALOG:
            raise ValueError(f"unknown query: {self.query}")
        if self.workload not in WORKLOADS:
            raise ValueError(f"unknown workload: {self.workload}")
        if not 1 <= self.batch_size <= np.iinfo(np.int32).max or self.seed < 0:
            raise ValueError(
                "batch size must be positive and fit int32, and seed must be nonnegative"
            )
        if self.metric not in ("euclidean", "angular"):
            raise ValueError(f"unknown metric: {self.metric}")
        for name, val in self.parameters.items():
            if name not in (*CATALOG[self.query].parameters, "radius"):
                raise ValueError(f"{self.query} does not accept parameter {name}")
            if name in {"k", "max_hits", "count_limit"} and (
                type(val) is not int or not 1 <= val <= np.iinfo(np.int32).max
            ):
                raise ValueError(f"{name} must be a positive int32")
            if (
                name
                in {
                    "line",
                    "mixed_line",
                    "cache_bounds",
                    "packed_nodes",
                    "group_leaves",
                }
                and type(val) is not bool
            ):
                raise ValueError(f"{name} must be a boolean")
            if name in {
                "radius",
                "cap",
                "cap_squared",
                "max_t",
                "robin_coefficient",
            } and (
                type(val) not in (int, float)
                or not 0 <= val <= float(np.finfo(np.float32).max)
            ):
                raise ValueError(f"invalid nonnegative {name}: {val}")
        if self.query in ("knn", "all_hits"):
            width = (
                self.parameters.get("k", 4)
                if self.query == "knn"
                else self.parameters.get(
                    "max_hits", 4 if self.workload == "builtin" else 16
                )
            )
            if self.batch_size * width > np.iinfo(np.int32).max:
                raise ValueError("batch size times result capacity must fit int32")


def geometry_support(case, primitive, closed):
    from benchmarks.workloads import POINT_WORKLOADS

    query = CATALOG[case.query]
    if primitive not in query.primitives:
        return f"{case.query} requires one of {', '.join(query.primitives)}"
    if query.closed and not closed:
        return f"{case.query} requires closed boundary geometry"
    if case.metric == "angular" and primitive not in query.angular:
        return f"{case.query} has no angular metric for {primitive}"
    if (
        case.workload in {"near_surface", "shell", "interior", "center"}
        and primitive != "triangles"
    ):
        return f"{case.workload} requires indexed triangles"
    if case.workload in POINT_WORKLOADS and case.geometry not in ("points2", "points3"):
        return f"{case.workload} requires a points2 or points3 fixture"
    if case.query == "knn":
        unused = "cap" if primitive.startswith("points") else "cap_squared"
        if unused in case.parameters:
            return f"{primitive} KNN does not use {unused}"
    return None

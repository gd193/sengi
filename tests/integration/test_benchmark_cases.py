import json

import numpy as np

from benchmarks.assets import Geometry
from benchmarks.cases import Case
from benchmarks.queries.validation import validate


def triangle():
    return Geometry(
        "fixture",
        "tiny",
        "triangles",
        {
            "vertices": np.array(
                [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32
            ),
            "faces": np.array([[0, 1, 2]], dtype=np.int32),
        },
    )


def test_region_count_requires_a_paired_result():
    case = Case("region_count", "fixture", "tiny", batch_size=2)
    inputs = {
        "lower": np.array([[0.75, 0.75, -0.1], [1.0, 0.0, 0.0]], dtype=np.float32),
        "upper": np.array([[1.0, 1.0, 0.1], [1.0, 0.0, 0.0]], dtype=np.float32),
    }
    actual = {"value": np.array([0, 1]), "status": np.array([1, 1])}
    assert validate(case, triangle(), inputs, actual)[0] == "unchecked"
    other = dict(actual, value=np.array([1, 1]))
    status, details = validate(case, triangle(), inputs, actual, other)
    assert status == "mismatch"
    assert details["query_indices"] == [0]


def test_bad_records_remain_serializable():
    case = Case("robin_radius", "fixture", "tiny", batch_size=1)
    inputs = {"radii": np.array([0.5])}
    actual = {"distance": np.array([np.inf]), "status": np.array([-1])}
    status, details = validate(case, triangle(), inputs, actual)
    assert status == "mismatch"
    assert details["actual"]["distance"] == ["inf"]
    json.dumps(details, allow_nan=False)
    actual["distance"] = np.array([[0.5]])
    assert validate(case, triangle(), inputs, actual)[1]["wrong_batch_fields"] == [
        "distance"
    ]


def test_sample_support_compares_integral_and_checks_radius():
    case = Case("sample_boundary", "fixture", "tiny", batch_size=2)
    inputs = {
        "radii": np.array([0.5, 0.5]),
        "points": np.array([[0.2, 0.2, 0.1], [1.0, 1.0, 0.0]]),
    }
    actual = {
        "point": np.array([[0.2, 0.2, 0.0], [0.0, 0.0, 0.0]]),
        "normal": np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]),
        "pdf": np.array([2.0, 0.0]),
        "id": np.array([0, -1]),
        "status": np.array([1, 0]),
    }
    expected = dict(actual, point=np.array([[0.25, 0.25, 0.0], [0.0, 0.0, 0.0]]))
    status, details = validate(case, triangle(), inputs, actual, expected)
    assert status == "unchecked"
    assert details["sampling"]["null_fraction"] == 0.5
    assert details["sampling"]["pdf_weighted_integral"] == 0.25
    assert details["area_difference"] == 0.0
    actual["point"][0, 2] = 1.0
    assert validate(case, triangle(), inputs, actual, expected)[0] == "mismatch"


def test_nearest_compares_distance_and_allows_tied_winners():
    geometry = Geometry(
        "fixture",
        "tiny",
        "points2",
        {"points": np.array([[-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]])},
    )
    case = Case("closest_point", "fixture", "tiny", batch_size=1)
    inputs = {"radii": np.array([5.0]), "points": np.array([[0.0, 0.0]])}
    actual = {
        "point": np.array([[1.0, 0.0]]),
        "distance": np.array([1.0]),
        "id": np.array([1]),
        "status": np.array([1]),
    }
    expected = dict(actual, point=np.array([[-1.0, 0.0]]), id=np.array([0]))
    assert validate(case, geometry, inputs, actual, expected)[0] == "pass"
    wrong = dict(
        actual, point=np.array([[2.0, 0.0]]), distance=np.array([2.0]), id=np.array([2])
    )
    assert validate(case, geometry, inputs, wrong, expected)[0] == "mismatch"


def test_sample_support_flags_systematic_pdf_error():
    n = 32
    case = Case("sample_boundary", "fixture", "tiny", batch_size=n)
    positions = np.tile([0.2, 0.2, 0.0], (n, 1))
    inputs = {"radii": np.full(n, 0.5), "points": positions}
    actual = {
        "point": positions,
        "normal": np.tile([0.0, 0.0, 1.0], (n, 1)),
        "pdf": np.full(n, 2.0),
        "id": np.zeros(n, dtype=int),
        "status": np.ones(n, dtype=int),
    }
    assert validate(case, triangle(), inputs, actual, actual)[0] == "sample support"
    noisy_pdf = np.tile([1 / 3, 1.0], n // 2)
    stable = dict(actual, pdf=np.full(n, 0.5))
    noisy = dict(actual, pdf=noisy_pdf)
    status, details = validate(case, triangle(), inputs, noisy, stable)
    assert status == "unchecked" and details["area_tolerance"] > 0.2
    expected = dict(actual, pdf=np.full(n, 1.0))
    status, details = validate(case, triangle(), inputs, actual, expected)
    assert status == "mismatch"
    assert details["area_check"] == "mismatch"
    rejected = dict(actual, pdf=np.zeros(n), id=np.full(n, -1), status=np.zeros(n))
    assert validate(case, triangle(), inputs, rejected, expected)[0] == "mismatch"
    assert validate(case, triangle(), inputs, rejected, rejected)[0] == "unchecked"


def test_polygon_silhouette_accepts_distinct_native_numbering():
    from benchmarks.workloads import builtin_geometry

    geometry = builtin_geometry("segments2", 4)
    case = Case("closest_silhouette", "segments2", "4", batch_size=1)
    inputs = {"radii": np.array([2.0]), "points": np.array([[0.0, 0.0]])}
    expected = {
        "point": np.array([[1.0, 1.0]]),
        "distance": np.array([np.sqrt(2)]),
        "id": np.array([0]),
        "status": np.array([1]),
    }
    actual = dict(expected, id=np.array([15]))
    assert validate(case, geometry, inputs, actual, expected)[0] == "pass"


def test_workload_types_are_checked_before_generation():
    import pytest

    from benchmarks.workloads import builtin_geometry, make_inputs

    geometry = builtin_geometry("points3", 16)
    case = Case("knn", "points3", "16", workload="near_surface")
    with pytest.raises(ValueError, match="near_surface requires indexed triangles"):
        make_inputs(case, geometry)


def test_all_hits_overflow_preserves_total_and_allows_different_stored_hits():
    vertices = triangle().arrays["vertices"]
    mesh = Geometry(
        "two-faces",
        "tiny",
        "triangles",
        {
            "vertices": np.vstack((vertices, vertices + [0, 0, 1])),
            "faces": np.arange(6).reshape(2, 3),
        },
    )
    case = Case(
        "all_hits", mesh.name, mesh.size, batch_size=1, parameters={"max_hits": 1}
    )
    inputs = {
        "origins": np.array([[0.25, 0.25, -1.0]]),
        "directions": np.array([[0.0, 0.0, 1.0]]),
        "min_t": np.array([0.0]),
        "max_t": np.array([10.0]),
    }
    first = {
        "count": np.array([2]),
        "status": np.array([-2]),
        "id": np.array([[0]]),
        "distance": np.array([[1.0]]),
    }
    last = {**first, "id": np.array([[1]]), "distance": np.array([[2.0]])}
    assert validate(case, mesh, inputs, last, first)[0] == "pass"
    assert (
        validate(case, mesh, inputs, {**last, "count": np.array([1])}, first)[0]
        == "mismatch"
    )


def test_complete_all_hits_compares_crossing_distances_and_multiplicity():
    mesh = Geometry(
        "shared-edge",
        "tiny",
        "triangles",
        {"vertices": np.eye(4, 3), "faces": np.array([[0, 1, 2], [1, 0, 3]])},
    )
    case = Case(
        "all_hits", mesh.name, mesh.size, batch_size=1, parameters={"max_hits": 2}
    )
    inputs = {"min_t": np.array([0.0]), "max_t": np.array([5.0])}
    hit = {
        "count": np.array([1]),
        "status": np.array([1]),
        "id": np.array([[0, -1]]),
        "distance": np.array([[2.0, np.inf]]),
    }
    adjacent = dict(hit, id=np.array([[1, -1]]))
    assert validate(case, mesh, inputs, adjacent, hit)[0] == "pass"
    assert (
        validate(
            case, mesh, inputs, dict(adjacent, distance=np.array([[3.0, np.inf]])), hit
        )[0]
        == "mismatch"
    )
    duplicate = dict(
        hit, count=np.array([2]), id=np.array([[0, 0]]), distance=np.array([[2.0, 2.0]])
    )
    assert validate(case, mesh, inputs, duplicate, hit)[0] == "mismatch"


def test_winner_misses_require_the_documented_distance():
    case = Case("first_hit", "fixture", "tiny", batch_size=1)
    inputs = {"min_t": np.array([0.0]), "max_t": np.array([10.0])}
    miss = {
        "hit": np.array([0]),
        "status": np.array([0]),
        "id": np.array([-1]),
        "distance": inputs["max_t"],
    }
    assert validate(case, triangle(), inputs, miss)[0] == "unchecked"
    for bad in (0.0, np.nan, np.inf):
        assert (
            validate(case, triangle(), inputs, dict(miss, distance=np.array([bad])))[0]
            == "mismatch"
        )


def test_near_surface_radius_fixture_samples_both_sides_and_empty_space():
    from benchmarks.workloads import make_inputs

    mesh = triangle()
    case = Case(
        "radius_any", mesh.name, mesh.size, workload="near_surface", batch_size=1024
    )
    inputs = make_inputs(case, mesh)
    radii = inputs["radii"]
    offsets = inputs["points"][:, 2]
    assert np.all(radii == np.float32(0.05))
    assert np.any(offsets < -radii) and np.any(offsets > radii)
    assert np.any(np.abs(offsets) < radii / 2)
    assert np.all(np.abs(offsets) <= 2 * radii)


def test_triangle_precision_distinguishes_fused_nearest_from_radius_scoring():
    from benchmarks.adapters.sengi import _query_precision

    for query in ("closest_point", "nearest_selected"):
        assert (
            _query_precision(Case(query, "sphere", "large"), "triangles") == "float32"
        )
    assert "float64" in _query_precision(
        Case("radius_count", "sphere", "large"), "triangles"
    )

"""Check output contracts and compare results from GPU query implementations."""

import numpy as np

from benchmarks.cases import CATALOG

FLOAT_TOL = 3e-5
WINNERS = {"closest_point", "nearest_selected", "closest_silhouette", "first_hit"}


def _close(a, b):
    return np.isclose(a, b, atol=FLOAT_TOL, rtol=FLOAT_TOL)


def _primitive_count(geometry):
    if geometry.primitive == "triangles":
        return len(geometry.arrays["faces"])
    if geometry.primitive.startswith("points"):
        return len(geometry.arrays["points"])
    return len(geometry.arrays["starts"])


def _sample_support(out, geometry, inputs, details, prefix):
    ids, pdf = out["id"], out["pdf"]
    accepted = out["status"] == 1
    valid = (ids >= 0) & (ids < _primitive_count(geometry))
    ok = (accepted == valid) & (accepted | ((ids == -1) & (pdf == 0)))
    ok[valid] &= np.isfinite(pdf[valid]) & (pdf[valid] > 0)
    ok[valid] &= _close(np.linalg.norm(out["normal"][valid], axis=1), 1)
    ok[valid] &= (
        np.linalg.norm(out["point"][valid] - inputs["points"][valid], axis=1)
        <= inputs["radii"][valid] + FLOAT_TOL
    )
    area = np.divide(
        1.0,
        pdf.astype(np.float64),
        where=accepted & (pdf > 0),
        out=np.zeros(len(pdf), dtype=np.float64),
    )
    details[prefix] = {
        "attempts": len(pdf),
        "accepted": int(accepted.sum()),
        "null_fraction": float((~accepted).mean()),
        "pdf_weighted_integral": float(area.mean()),
        "standard_error": float(area.std(ddof=1) / np.sqrt(len(area)))
        if len(area) > 1
        else None,
    }
    return ok, area


def sample_area_is_resolved(details):
    tolerance = FLOAT_TOL + 6 * details["area_difference_se"]
    smaller_area = min(details[key]["pdf_weighted_integral"] for key in ("sampling", "reference_sampling"))
    return tolerance <= 0.1 * smaller_area


def _check_winners(case, geometry, inputs, out, ok):
    ids = out["id"]
    found = out["status"] == 1
    valid = ids >= 0
    if case.query != "closest_silhouette":
        valid &= ids < _primitive_count(geometry)
    ok &= (found == valid) & (found | (ids == -1))
    misses = out["distance"][~found]
    if case.query == "first_hit":
        ok[~found] &= misses == inputs["max_t"][~found]
    else:
        ok[~found] &= np.isposinf(misses)
    distance = out["distance"][found]
    if case.query == "first_hit":
        ok[found] &= (distance >= inputs["min_t"][found] - FLOAT_TOL) & (
            distance <= inputs["max_t"][found] + FLOAT_TOL
        )
        ok &= out["hit"].astype(bool) == found
    else:
        ok[found] &= np.isfinite(distance) & (
            distance <= inputs["radii"][found] + FLOAT_TOL
        )
        if case.metric != "angular":
            reported = np.linalg.norm(
                out["point"][found] - inputs["points"][found], axis=1
            )
            ok[found] &= _close(distance, reported)


def _check_lists(case, geometry, inputs, out, ok):
    ids, distance, count = (out[key] for key in ("id", "distance", "count"))
    slots = np.arange(ids.shape[1])[None, :] < count[:, None]
    valid = (ids >= 0) & (ids < _primitive_count(geometry)) & np.isfinite(distance)
    ok &= np.where(slots, valid, ids == -1).all(axis=1) & (count >= 0)
    overflow = count > ids.shape[1]
    sorted_ids = np.sort(ids, axis=1)
    ok &= ((sorted_ids[:, 1:] < 0) | (np.diff(sorted_ids, axis=1) != 0)).all(axis=1)
    if case.query == "all_hits":
        ok &= out["status"] == np.where(overflow, -2, count > 0)
        in_range = (distance >= inputs["min_t"][:, None] - FLOAT_TOL) & (
            distance <= inputs["max_t"][:, None] + FLOAT_TOL
        )
    else:
        ok &= ~overflow & ((out["status"] == 1) == (count > 0))
        ok &= (distance[:, :-1] <= distance[:, 1:]).all(axis=1)
        in_range = (distance >= 0) & (distance <= inputs["radii"][:, None] + FLOAT_TOL)
    ok &= (~slots | in_range).all(axis=1)


def _compare(case, actual, expected):
    n = len(actual["status"])
    ok = np.ones(n, dtype=bool)
    for key in CATALOG[case.query].outputs:
        if key in {"id", "point"} and (case.query in WINNERS or case.query == "knn"):
            continue
        a, b = actual[key], expected[key]
        if case.query == "all_hits" and key in {"id", "distance"}:
            if key == "id":
                continue  # A shared-edge crossing may name either incident face.
            complete = (actual["status"] != -2) & (expected["status"] != -2)
            distances = [
                np.sort(
                    np.where(out["id"][complete] >= 0, out[key][complete], np.inf),
                    axis=1,
                )
                for out in (actual, expected)
            ]
            ok[complete] &= _close(*distances).all(axis=1)
        elif key == "distance" and case.query in WINNERS:
            found = (actual["status"] == 1) & (expected["status"] == 1)
            ok[found] &= _close(a[found], b[found])
        elif key == "distance" and case.query == "knn":
            slots = np.arange(a.shape[1])[None, :] < actual["count"][:, None]
            ok &= (_close(a, b) | ~slots).all(axis=1)
        else:
            if key == "value" and case.query == "radius_accumulate":
                # Integer face weights sum exactly in FP32 through 2**24.
                exact = (
                    (np.maximum(a, b) < 2**24) & (a == np.floor(a)) & (b == np.floor(b))
                )
                cells = np.where(exact, a == b, _close(a, b))
            else:
                cells = _close(a, b) if key == "distance" else a == b
            ok &= cells.reshape(n, -1).all(axis=1)
    return ok


def _example_values(arrays, indices):
    result = {}
    for key, array in arrays.items():
        values = np.asarray(array)[indices]
        if np.issubdtype(values.dtype, np.floating) and not np.isfinite(values).all():
            safe = values.astype(object)
            for marker, mask in (
                ("inf", np.isposinf(values)),
                ("-inf", np.isneginf(values)),
                ("nan", np.isnan(values)),
            ):
                safe[mask] = marker
            values = safe
        result[key] = values.tolist()
    return result


def validate(case, geometry, inputs, actual, expected=None):
    """Return validation status and reproducible mismatch details."""
    mode, n = case.query, case.batch_size
    details = {"seed": case.seed}
    missing = set(CATALOG[mode].outputs) - actual.keys()
    if missing:
        return "mismatch", dict(details, missing_fields=sorted(missing))
    dim = inputs.get("points", inputs.get("origins", np.empty((n, 0)))).shape[-1]
    dimensions = {"point": dim, "normal": dim}
    if mode in {"all_hits", "knn"}:
        width = (
            case.parameters.get("k", 4)
            if mode == "knn"
            else case.parameters.get(
                "max_hits", 4 if case.workload == "builtin" else 16
            )
        )
        dimensions.update(distance=width, id=width)
    invalid = [
        key
        for key in CATALOG[mode].outputs
        if np.shape(actual[key])
        != ((n, dimensions[key]) if key in dimensions else (n,))
    ]
    if invalid:
        return "mismatch", dict(details, wrong_batch_fields=invalid)

    ok = (actual["status"] == 0) | (actual["status"] == 1)
    if mode == "all_hits":
        ok |= actual["status"] == -2
        details["truncated_rows"] = int(np.count_nonzero(actual["status"] == -2))
    status = "unchecked"
    if mode == "sample_boundary":
        rows, integral = _sample_support(actual, geometry, inputs, details, "sampling")
        ok &= rows
        if expected is not None:
            rows, reference = _sample_support(
                expected, geometry, inputs, details, "reference_sampling"
            )
            ok &= rows
            difference = integral - reference
            se = float(difference.std(ddof=1) / np.sqrt(n)) if n > 1 else None
            mean = float(difference.mean())
            details.update(area_difference=mean, area_difference_se=se)
            counts = (
                details["sampling"]["accepted"],
                details["reference_sampling"]["accepted"],
            )
            tolerance = FLOAT_TOL + 6 * se if se is not None else None
            details["area_tolerance"] = tolerance
            if max(counts) >= 16 and abs(mean) > tolerance:
                status = "mismatch"
                details["area_check"] = "mismatch"
            elif min(counts) >= 16 and not sample_area_is_resolved(details):
                details["area_check"] = "uncertainty exceeds 10% of the smaller area estimate"
            elif min(counts) >= 16:
                status = "sample support"
                details["area_check"] = "consistent"
            else:
                details["area_check"] = "insufficient accepted samples"
    elif mode in WINNERS:
        _check_winners(case, geometry, inputs, actual, ok)
        details["hit_rate"] = float((actual["status"] == 1).mean())
    elif mode in {"knn", "all_hits"}:
        _check_lists(case, geometry, inputs, actual, ok)
        details["hit_rate"] = float((actual["count"] > 0).mean())
    elif mode in {"star_radius", "robin_radius"}:
        distance = actual["distance"]
        ok &= (
            np.isfinite(distance)
            & (distance >= 0)
            & (distance <= inputs["radii"] + FLOAT_TOL)
        )
        details["cap_frequency"] = float(_close(distance, inputs["radii"]).mean())
    elif "value" in actual and mode not in {"classify_point", "classify_box"}:
        details["hit_rate"] = float((actual["value"] > 0).mean())

    if expected is not None and mode != "sample_boundary":
        ok &= _compare(case, actual, expected)
        status = "pass"
    if not ok.all():
        bad = np.flatnonzero(~ok)
        examples = bad[:16]
        details.update(
            failed_rows=len(bad),
            query_indices=examples.tolist(),
            actual=_example_values(actual, examples),
        )
        if expected is not None:
            details["expected"] = _example_values(expected, examples)
        return "mismatch", details
    return status, details

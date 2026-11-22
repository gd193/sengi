"""Normalize primitive bounds for host spatial tree construction."""
import numpy as np


def _round_outward(values, *, upper, dtype=np.float32):
    if values.dtype == dtype:
        return values
    rounded = values.astype(dtype)
    if values.dtype.kind in "iu":
        exact_limit = 1 << (np.finfo(dtype).nmant + 1)
        if int(values.min(initial=0)) >= -exact_limit and int(values.max(initial=0)) <= exact_limit:
            return rounded
    if values.dtype.kind in "iu" and values.dtype.itemsize == 8:
        # The only rounded integer outside the source dtype is max + 1.
        # Compare the rest back in the source dtype, preserving every bit.
        above_max = rounded >= float(int(np.iinfo(values.dtype).max) + 1)
        with np.errstate(invalid="ignore"):
            integer = rounded.astype(values.dtype)
        inward = ((integer < values) & ~above_max if upper
                  else (integer > values) | above_max)
    else:
        inward = rounded < values if upper else rounded > values
    if values.dtype.kind in "iu" and rounded.dtype == np.float32:
        # An inexact integer rounds to a finite, nonzero float. Adjacent
        # FP32 values differ by one bit-pattern step, reversed for negatives.
        bits = rounded.view(np.int32)
        step = inward.astype(np.int32)
        increment = (bits >= 0) if upper else (bits < 0)
        bits += np.where(increment, step, -step)
        return rounded
    direction = np.inf if upper else -np.inf
    return np.nextafter(rounded, direction, out=rounded, where=inward)


def _validated_bounds(lower, upper, *, dimension=3):
    """Return finite float32 bounds rounded outwards."""
    lo, hi = np.asarray(lower), np.asarray(upper)
    if lo.dtype.kind not in "biuf" or hi.dtype.kind not in "biuf":
        raise ValueError("primitive bounds must have a real numeric dtype")
    if lo.ndim != 2 or lo.shape[1] != dimension or hi.shape != lo.shape:
        raise ValueError("primitive bounds must have matching shape (N, dimension)")
    # Compare endpoints exactly when NumPy would promote a 64-bit integer
    # to a float. Round an integer lower bound up (upper bound down) for
    # comparison against a floating endpoint on that grid.
    common = np.result_type(lo, hi)
    if common.kind != "f" or all(v.dtype.kind not in "iu" or v.dtype.itemsize < 8 for v in (lo, hi)):
        unordered = lo > hi
    elif lo.dtype.kind in "iu" and hi.dtype.kind in "iu":
        if lo.dtype.kind == "i":
            unordered = (lo >= 0) & (lo.astype(np.uint64) > hi)
        else:
            unordered = (hi < 0) | (lo > hi.astype(np.uint64))
    elif lo.dtype.kind in "iu":
        unordered = _round_outward(lo, upper=True, dtype=common) > hi
    else:
        unordered = lo > _round_outward(hi, upper=False, dtype=common)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(unordered):
        raise ValueError("primitive bounds must be finite and ordered")
    if lo.dtype == hi.dtype == np.float32:
        return lo, hi
    with np.errstate(over="ignore", under="ignore"):
        lo, hi = _round_outward(lo, upper=False), _round_outward(hi, upper=True)
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("primitive bounds exceed finite float32 range")
    return lo, hi

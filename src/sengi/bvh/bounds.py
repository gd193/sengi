"""Normalize primitive bounds for host spatial tree construction."""
import numpy as np


def _validated_bounds(lower, upper, *, dimension=3):
    """Return finite float32 bounds rounded outwards."""
    raw = (lower, upper)
    if any(np.iscomplexobj(v) for v in raw):
        raise ValueError("primitive bounds must be real")
    lo, hi = (np.asarray(v, np.float64) for v in raw)
    if lo.ndim != 2 or lo.shape[1] != dimension or hi.shape != lo.shape:
        raise ValueError("primitive bounds must have matching shape (N, dimension)")
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or np.any(lo > hi):
        raise ValueError("primitive bounds must be finite and ordered")
    with np.errstate(over="ignore", under="ignore"):
        low32, high32 = lo.astype(np.float32), hi.astype(np.float32)
        np.nextafter(low32, np.float32(-np.inf), out=low32, where=low32 > lo)
        np.nextafter(high32, np.float32(np.inf), out=high32, where=high32 < hi)
    lo, hi = low32, high32
    if not np.isfinite(lo).all() or not np.isfinite(hi).all():
        raise ValueError("primitive bounds exceed finite float32 range")
    return lo, hi

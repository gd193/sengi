"""Host validation and packing for primitive arrays."""
import jax.numpy as jnp
import numpy as np

from sengi.primitives.triangle_frames import pack_triangle_frames


def _array_data(payload, dimension, count, name, *, segment=False):
    def prepare(*arrays):
        if len(arrays) != count:
            raise ValueError(f"{name} expects {count} arrays")
        arrays = tuple(jnp.asarray(a, jnp.float32) for a in arrays)
        if any(a.ndim != 2 or a.shape != arrays[0].shape or a.shape[1] != dimension for a in arrays):
            raise ValueError(f"primitive arrays must have matching shape (N, {dimension})")
        if any(not np.isfinite(np.asarray(a)).all() for a in arrays):
            raise ValueError("primitive data must be finite float32")
        if segment:
            arrays += (jnp.empty((0, dimension), jnp.float32), jnp.empty((0, 2), jnp.int32))
        return payload(*arrays)
    return prepare


def _indexed_data(payload, dimension, width):
    def prepare(vertices, indices):
        source_vertices = vertices
        vertices, indices = np.asarray(vertices, np.float32), np.asarray(indices)
        if vertices.ndim != 2 or vertices.shape[1:] != (dimension,):
            raise ValueError(f"vertices must have shape (V, {dimension})")
        if indices.ndim != 2 or indices.shape[1:] != (width,) or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"indices must contain integer tuples of length {width}")
        if indices.size and (np.any(indices < 0) or np.any(indices >= len(vertices))):
            raise ValueError("primitive index outside vertices")
        items = vertices[indices]
        arrays = ((items[:, 0], items[:, 1]-items[:, 0], items[:, 2]-items[:, 0])
                  if width == 3 else (items[:, 0], items[:, 1]))
        if any(not np.isfinite(a).all() for a in arrays):
            raise ValueError("primitive data must be finite float32")
        fields = (*arrays, jnp.asarray(source_vertices, dtype=jnp.float32), indices.astype(np.int32))
        if width == 3:
            fields += (items[:, 1], items[:, 2], pack_triangle_frames(items))
        return payload(*(jnp.asarray(field) for field in fields))
    return prepare


def _array_bounds(data):
    arrays = [np.asarray(a, np.float32) for a in data[:2]]
    return np.minimum.reduce(arrays), np.maximum.reduce(arrays)


def _array_pack(data, ids):
    return type(data)(*(a[ids] for a in data))


def _segment_pack(data, ids):
    indices = data.indices[ids] if data.indices.shape[0] else data.indices
    return type(data)(data.a[ids], data.b[ids], data.vertices, indices)

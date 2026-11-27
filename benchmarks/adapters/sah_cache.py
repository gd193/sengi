"""Persist validated Sengi mesh and point hierarchies between benchmark processes."""

from functools import lru_cache
from pathlib import Path
from zipfile import BadZipFile

import numpy as np
from numpy.lib.npyio import NpzFile

from benchmarks.assets import ASSET_ROOT
from benchmarks.results import array_digest, atomic_file, digest, file_hash
from sengi import bvh
from sengi.bvh import bounds, device_tree, sah, tree
from sengi.primitives import packing, points, triangles

CACHE_DIR = ASSET_ROOT / "sah"
_TREE_FIELDS = ("node_lower", "node_upper", "node_left", "node_right",
                "node_leaf_offset", "node_leaf_count", "prim_ids")


@lru_cache(maxsize=1)
def _builder_version(primitive_name):
    geometry_modules = (triangles,) if primitive_name == "triangles" else (points, packing)
    modules = (sah, bounds, tree, device_tree, *geometry_modules)
    return digest([file_hash(Path(module.__file__)) for module in modules])


def cached_index(primitive, source, arrays, config):
    kind = "mesh" if primitive.name == "triangles" else f"points{primitive.dimension}"
    key = digest((
        f"sah-{kind}-v1", _builder_version(primitive.name), config.leaf_size_threshold,
        array_digest(arrays),
    ))
    path = CACHE_DIR / f"{key}.npz"
    try:
        with path.open("rb") as stream, NpzFile(stream, allow_pickle=False) as archive:
            nodes = {name: archive[name] for name in _TREE_FIELDS}
            if archive["checksum"].item() != digest((key, array_digest(nodes))):
                raise ValueError("SAH index checksum mismatch")
    except (FileNotFoundError, BadZipFile, EOFError, KeyError, ValueError):
        index = bvh.build(primitive, source, config=config, builder="sah")
        nodes = {name: np.asarray(getattr(index.data.tree, name)) for name in _TREE_FIELDS}
        checksum = digest((key, array_digest(nodes)))
        atomic_file(path, lambda stream: np.savez(stream, checksum=checksum, **nodes))
        return index, "miss"

    data = bvh.IndexData(*device_tree.pack_device_tree(primitive, source, **nodes))
    return bvh.Index(data.spec, data), "hit"

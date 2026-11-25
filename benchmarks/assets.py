"""Discover manifests and verify indexed geometry without changing primitive IDs."""

from __future__ import annotations

import hashlib
import json
import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import trimesh

from benchmarks.results import array_digest, file_hash

ASSET_ROOT = Path(__file__).resolve().parents[1] / "asset"


@dataclass(frozen=True)
class MeshAsset:
    name: str
    path: Path
    variants: dict
    closed: bool
    oriented: bool
    provenance: str


@dataclass(frozen=True)
class Geometry:
    name: str
    size: str
    primitive: str
    arrays: dict[str, np.ndarray]
    closed: bool = False


def discover_assets(root=None):
    assets = {}
    root = Path(root) if root is not None else ASSET_ROOT
    for path in sorted(root.glob("*/mesh.toml")):
        manifest = tomllib.loads(path.read_text())
        name = path.parent.name
        variants = manifest["variants"]
        if not variants:
            raise ValueError(f"{path}: at least one geometry variant is required")
        for size, entry in variants.items():
            if ("file" in entry) == ("recipe" in entry):
                raise ValueError(
                    f"{path}: {size} must specify exactly one file or recipe"
                )
        assets[name] = MeshAsset(
            name,
            path,
            variants,
            manifest["closed"],
            manifest.get("oriented", False),
            manifest["provenance"],
        )
    return assets


def _check_file(path, entry):
    mesh = None
    if "recipe" in entry:
        expected = entry["geometry_sha256"]
        mesh = trimesh.load_mesh(path, process=False)
        actual = array_digest(
            {
                "vertices": np.asarray(mesh.vertices, dtype="<f4"),
                "faces": np.asarray(mesh.faces, dtype="<i4"),
            }
        )
    else:
        expected = entry["sha256"]
        actual = file_hash(path)
    if actual != expected:
        raise ValueError(f"{path}: expected SHA256 {expected}, got {actual}")
    return actual, mesh


def _materialize(assets, name, size, cache, parents=()):
    if name not in assets or size not in assets[name].variants:
        raise ValueError(f"unknown mesh variant {name}/{size}")
    if (name, size) in parents:
        raise ValueError(f"cyclic mesh preparation recipe at {name}/{size}")
    asset = assets[name]
    entry = asset.variants[size]
    if "file" in entry:
        path = asset.path.parent / entry["file"]
        return (path, *_check_file(path, entry))

    recipe = entry["recipe"]
    kind = recipe["kind"]
    options = {k: v for k, v in recipe.items() if k not in {"kind", "source", "mesh"}}
    source, source_hash, source_mesh = (None, None, None)
    if "source" in recipe:
        source, source_hash, source_mesh = _materialize(
            assets,
            recipe.get("mesh", name),
            recipe["source"],
            cache,
            (*parents, (name, size)),
        )
    key = hashlib.sha256(
        json.dumps(
            (recipe, source_hash, entry["geometry_sha256"]),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    path = cache / key / "mesh.ply"
    if path.is_file():
        return (path, *_check_file(path, entry))

    if kind == "icosphere":
        mesh = trimesh.creation.icosphere(**options)
    elif kind == "torus":
        mesh = trimesh.creation.torus(**options)
    elif kind == "subdivide":
        mesh = (
            source_mesh
            if source_mesh is not None
            else trimesh.load_mesh(source, process=False)
        ).subdivide()
    elif kind == "clip_centroids":
        mesh = (
            source_mesh
            if source_mesh is not None
            else trimesh.load_mesh(source, process=False)
        )
        mesh.update_faces(mesh.triangles_center[:, options["axis"]] < options["upper"])
        mesh.remove_unreferenced_vertices()
    else:
        raise ValueError(f"{asset.path}: unknown preparation recipe {kind}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as staging:
        staged = Path(staging) / path.name
        mesh.export(staged, file_type="ply", encoding="binary_little_endian")
        digest, mesh = _check_file(staged, entry)
        staged.replace(path)
    return path, digest, mesh


def _cache_path(cache):
    if cache is not None:
        return Path(cache)
    if directory := os.environ.get("SENGI_ASSET_CACHE"):
        return Path(directory)
    return ASSET_ROOT.parent / "output" / "asset-cache"


def load_geometry(name, size, root=None, cache=None):
    size = str(size)
    assets = discover_assets(root)
    path, _, mesh = _materialize(assets, name, size, _cache_path(cache))
    asset = assets[name]
    if mesh is None:
        mesh = trimesh.load_mesh(path, process=False)
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or not np.isfinite(vertices).all()
        or faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or faces.min() < 0
        or faces.max() >= len(vertices)
        or faces.max() > np.iinfo(np.int32).max
    ):
        raise ValueError(f"{path}: invalid indexed triangle mesh")
    # Twice-area / longest edge squared is independent of scale and subdivision.
    for chunk in np.array_split(faces, 1 + len(faces) // 65536):
        a, b, c = vertices[chunk].astype(np.float64).transpose(1, 0, 2)
        longest2 = np.maximum.reduce(
            [np.sum(edge * edge, axis=1) for edge in (b - a, c - b, a - c)]
        )
        area2 = np.linalg.norm(np.cross(b - a, c - a), axis=1)
        if np.any(area2 <= 0.0005 * longest2):
            raise ValueError(f"{path}: degenerate or sliver triangle")
    count = asset.variants[size].get("faces", len(faces))
    if len(faces) != count:
        raise ValueError(f"{path}: {len(faces)} triangles, expected {count}")
    if asset.closed and not mesh.is_watertight:
        raise ValueError(f"{path}: declared closed but boundary is not watertight")
    if asset.oriented and (
        not mesh.is_winding_consistent or (asset.closed and mesh.volume <= 0)
    ):
        raise ValueError(
            f"{path}: declared oriented but winding is inconsistent or inward"
        )
    return Geometry(
        name,
        size,
        "triangles",
        {"vertices": vertices, "faces": faces.astype(np.int32)},
        closed=asset.closed,
    )

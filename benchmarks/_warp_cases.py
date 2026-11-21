"""Supported builtin combinations and shared correctness/timing fixtures."""
from sengi import bvh
from functools import lru_cache
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np
import trimesh

from sengi.bvh import Config
from sengi.metrics.nearest import Angular, Euclidean
from sengi.primitives.points import Points
from sengi.primitives.segments import Segments
from sengi.boundary.silhouette_edges import SilhouetteEdges
from sengi.primitives.triangles import Triangles
from sengi.queries.nearest import bind as bind_nearest
from sengi.queries.overlap import bind as bind_overlap
from sengi.queries.rays import bind as bind_rays
from sengi.queries.silhouette import bind as bind_silhouette
from sengi.queries.star_radius import bind as bind_star_radius
from sengi.queries.classification import bind as bind_classification
from sengi.queries.boundary_sample import bind as bind_boundary_sample
from sengi.queries.robin_radius import bind as bind_robin_radius
from sengi.boundary.sampling import prepare_boundary_sample
from sengi.boundary.robin import prepare_robin

PRIMITIVES = dict(points2=Points(2), points3=Points(3), segments2=Segments(2),
                  segments3=Segments(3), triangles=Triangles(), edges=SilhouetteEdges())
METRICS = {"euclidean": Euclidean(), "angular": Angular()}
RAYS = ("first", "any", "count", "all")
BOUNDARIES = ("segments2", "triangles")


class QueryCase(NamedTuple):
    primitive: str
    query: str
    metric: str = "none"

    @property
    def id(self):
        return "/".join(self)


CASES = tuple(
    [QueryCase(p, "nearest", m) for p, primitive in PRIMITIVES.items() for m in METRICS
     if m == "euclidean" or primitive.segment_endpoints is not None]
    + [QueryCase(p, "overlap") for p in PRIMITIVES]
    + [QueryCase(p, q) for p, primitive in PRIMITIVES.items() if primitive.intersect is not None for q in RAYS]
    + [QueryCase(p, "silhouette", m) for p in BOUNDARIES for m in METRICS]
    + [QueryCase(p, q, m) for p in BOUNDARIES for q, m in
       (("star", "euclidean"), ("robin", "euclidean"), ("classify", "none"), ("sample", "none"))]
)


def query_factory(case, spec):
    metric = METRICS.get(case.metric)
    if case.query == "nearest":
        return bind_nearest(spec, metric)
    if case.query == "silhouette":
        return bind_silhouette(spec, metric)
    if case.query in RAYS:
        return bind_rays(spec, case.query)
    return {"overlap": bind_overlap, "star": bind_star_radius, "robin": bind_robin_radius,
            "classify": bind_classification, "sample": bind_boundary_sample}[case.query](spec)


class Scene(NamedTuple):
    index: bvh.Index
    boundary: object
    raw: object


@lru_cache(maxsize=None)
def scene(name, size=16, leaf_size=4):
    primitive = PRIMITIVES[name]
    rng = np.random.default_rng(73)
    if name == "triangles":
        mesh = trimesh.creation.box(extents=[2., 2., 2.])
        while len(mesh.faces) < size:
            mesh = mesh.subdivide()
        raw = primitive.indexed(mesh.vertices, mesh.faces)
    elif name == "segments2":
        corners = np.array([[-1., -1], [1., -1], [1., 1], [-1., 1]], np.float32)
        t = np.arange(max(1, size//4))/max(1, size//4)
        vertices = np.concatenate([a+t[:, None]*(b-a) for a, b in zip(corners, np.roll(corners, -1, axis=0))])
        raw = primitive.indexed(vertices, np.column_stack((np.arange(len(vertices)), np.roll(np.arange(len(vertices)), -1))))
    else:
        a = rng.normal(size=(size, primitive.dimension)).astype(np.float32)
        if name.startswith("points"):
            raw = primitive.data(a)
        else:
            b = a+.5*rng.normal(size=a.shape)
            raw = (primitive.data(a, b, rng.normal(size=a.shape), rng.normal(size=a.shape))
                   if name == "edges" else primitive.data(a, b))
    index = bvh.build(primitive, raw, config=Config(leaf_size_threshold=leaf_size))
    boundary = bvh.prepare_boundary(index, raw) if name in BOUNDARIES else None
    return Scene(index, boundary, raw)


def arguments(case, geometry, n, radius=.75):
    dim = PRIMITIVES[case.primitive].dimension
    rng = np.random.default_rng(404+n)
    points, directions = rng.normal(size=(2, n, dim)).astype(np.float32)
    points *= 1.5
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    queries = np.stack((points, directions), axis=1) if case.metric == "angular" else points
    points, directions, queries = map(jnp.asarray, (points, directions, queries))
    caps = jnp.full((n,), radius, jnp.float32)
    index, boundary, _ = geometry
    if case.query in RAYS:
        kwargs = dict(max_t=jnp.full((n,), 10., jnp.float32))
        if case.query in ("count", "all"):
            kwargs.update(min_t=jnp.where(jnp.arange(n) % 2 == 0, -jnp.inf, 0.))
        if case.query == "all":
            kwargs["max_hits"] = 4
        return (index.data, points, directions), kwargs
    if case.query == "nearest":
        return (index.data, queries, caps), {}
    if case.query == "overlap":
        return (index.data, points-radius, points+radius), {}
    if case.query == "classify":
        return (index.data, boundary, points), {}
    if case.query in ("silhouette", "star"):
        return (index.data, boundary, queries, caps), {}
    if case.query == "robin":
        return (index.data, prepare_robin(index.data.tree, boundary, 1.), points, caps), {}
    randoms = jnp.asarray(rng.random((n, 3)), jnp.float32)
    return (index.data, prepare_boundary_sample(index.data.tree, boundary), points,
            jnp.full_like(caps, jnp.inf), randoms), {}

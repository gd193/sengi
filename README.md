# Sengi

Sengi builds spatial trees from JAX arrays and runs primitive-generic geometry
queries with Warp. Host tree builders are available in `sengi.bvh`, `sengi.kd`,
and `sengi.octree`. Pass lower and upper primitive bounds as arrays of shape
`(N, 3)` to these builders. Host builders require at least one primitive.
The query index built by `bvh.build` also accepts empty geometry. Warp is
optional for host-only construction.

For Warp queries, construct a primitive and build a BVH once. Bind each query
to the index specification, then pass the index data and query inputs when
calling it.

```python
import jax.numpy as jnp

from sengi import bvh
from sengi.primitives import Points
from sengi.queries import nearest
from sengi.metrics.nearest import Euclidean

points = Points()
source = points.data(jnp.asarray([[0., 0., 0.], [1., 1., 1.]]))
index = bvh.build(points, source)
closest = nearest.bind(index.spec, Euclidean())
point, distance, primitive_id, status = closest(index.data, jnp.asarray([0.25, 0., 0.]))
```

The `sengi.primitives.segments` and `sengi.primitives.triangles` modules provide
the segment and triangle specifications. Query families in `sengi.queries`
include nearest, kNN, ray intersections, and boundary queries. Use
`bvh.prepare_boundary(index, source)` to prepare boundary metadata when a query
requires it.
See the [BVH query guide](docs/warp_bvh.md) for boundary preparation, custom
metrics and calls from Warp kernels.

## Source layout

```text
src/sengi/
├── primitives/      Primitive bounds, packing and per-item geometry operations
├── bvh/             SAH and Morton builders, binary index and device tree
│   └── traversal/   Binary node access, visit order and stack
├── kd/              Host k-d tree builders
├── octree/          Host octree builder
├── queries/         Query families and their bind functions
├── metrics/         Distance scores and conservative bounds
├── boundary/        Boundary topology, silhouettes and sampling data
├── jax/             JAX array and FFI integration
└── warp/            Warp compilation, array access and kernel launch
```

Native headers live in the owning module's `native/` directory. For example,
triangle projection lives in `primitives/native/triangle_distance.h`, Robin
radius evaluation in `queries/native/robin.h`, and the binary traversal stack
in `bvh/traversal/native/traversal.h`. SAH and LBVH produce the same binary
layout, so a bound query can run on either tree.

## Conda installation

Build and install with conda tooling from the repository root.

```sh
mamba create -n spatial-build -c conda-forge rattler-build
mamba run -n spatial-build rattler-build build --recipe recipe/recipe.yaml -c conda-forge --output-dir "$PWD/output"
mamba create -n spatial-dev -c "file://$PWD/output" -c conda-forge python=3.14 sengi=0.3.0 jax=0.10.2 warp-lang=1.18.0 pytest scipy trimesh
mamba run -n spatial-dev python -m pytest tests
```

For source development, set `PYTHONPATH="$PWD/src"` when running tests in that
environment. Omit it to exercise the installed package and bundled native
headers.

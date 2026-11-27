# Binary BVH queries

Sengi uses two objects to represent an index. `index.spec` contains the static
primitive specification and the binary layout. `index.data` is a JAX pytree
containing topology and leaf-ordered geometry. Bind a query once, then pass
the data into the bound query. The same specialization works with data built
by SAH and LBVH when both use the same primitive specification.

```python
import jax
import jax.numpy as jnp

from sengi import bvh, queries
from sengi.metrics.nearest import Euclidean
from sengi.primitives.points import Points

primitive = Points(3)
source = primitive.data(jnp.asarray([[0., 0., 0.], [2., 0., 0.]]))
sah = bvh.build(primitive, source)
lbvh = bvh.build(primitive, source, builder="lbvh")
nearest = jax.jit(queries.nearest.bind(sah.spec, Euclidean()))

points = jnp.asarray([[1.5, 0., 0.]])
point, distance, original_id, status = nearest(sah.data, points)
point, distance, original_id, status = nearest(lbvh.data, points)
```

The default leaf-size threshold is 4. Set it with `bvh.Config(leaf_size_threshold=...)`. Select
`builder="lbvh"` to partition in Morton order. Both builders pack geometry
into leaf order for traversal. Both support depths up to 64 edges from the root.
If a split would require a deeper tree, construction raises `ValueError`.
A larger leaf threshold or a different builder can produce a shallower tree.
Packed traversal reads one 32-byte record per node. The dynamic JAX leaf
`tree.packed_nodes` has shape `(nodes, 2, 4)` and dtype `float32`. Its two
rows hold the lower and upper XYZ bounds. Each fourth lane stores raw topology
bits for a child index or leaf range endpoint.
Query result IDs refer to the original input.
Batch points as `(..., D)`, where `D` is the primitive dimension. Caps are
scalar or match the query batch. Dynamic inputs, including the index arrays,
can cross JIT and vmap boundaries.

## Query families

| Binding | JAX inputs | Output |
| --- | --- | --- |
| `queries.nearest.bind(spec, metric)` | index data, points, optional distance cap | point, value, ID, status |
| `queries.knn.bind(spec, k, metric)` | index data, points, optional distance cap | values, IDs or points, count, status |
| `queries.radius.bind(spec, metric, mode="count")` | index data, points, radius | count, status |
| `queries.rays.bind(spec, "first")` | index data, origin, direction, optional maximum and minimum parameter | time, ID, hit flag, status |
| `queries.rays.bind(spec, "any")` | index data, origin, direction, optional maximum and minimum parameter | hit flag, status |
| `queries.rays.bind(spec, "count")` | index data, origin, direction, optional maximum and minimum parameter | count, status |
| `queries.rays.bind(spec, "all")` | index data, origin, direction, optional maximum and minimum parameter, static `max_hits` | times, IDs, full count, status |
| `queries.overlap.bind(spec)` | index data, box lower and upper corners | primitive-AABB overlap, status |
| `queries.region.bind(spec)` | index data, box lower and upper corners | exact geometric overlap, status |
| `queries.region_classify.bind(spec)` | boundary metadata, box lower and upper corners | box classification, status |
| `queries.classification.bind(spec)` | boundary metadata, points | sign, status |
| `queries.silhouette.bind(spec)` | boundary metadata, points, optional distance cap | point, distance, feature ID, status |
| `queries.star_radius.bind(spec)` | boundary metadata, points, cap | radius, status |
| `queries.boundary_sample.bind(spec)` | sampling metadata, points, radii, triples of uniform draws | point, normal, PDF, ID, status |
| `queries.robin_radius.bind(spec)` | Robin metadata, points, optional cap | radius, status |

Ordinary candidate queries use status `1` for a match, `0` for a miss and
`-1` for invalid inputs or traversal failure. Ray intervals and distance caps
are closed. A first-hit ray may return any face at the earliest hit time. Use `exclude=True` when binding ray queries that take an
`exclude_face`. An all-hit query returns the full count even when capacity is
insufficient, with status `-2` for truncation. kNN specializes on a positive
static `k` and pads with infinite values and ID `-1`. Nearest, kNN and radius
use Euclidean distance by default. Nearest and kNN rank by internal metric score before conversion to the
reported distance. Any equally scoring primitive can win. kNN can return any
tied subset in any order within that score. Custom metrics can define the units of
the score and cap.
Exact region queries bind the primitive’s `region_policy`. Pass a
`RegionIntersectionPolicy` explicitly to override it. Closed-box classification
combines that operation with the boundary provider’s point classification.

## Indexed boundaries

A boundary provider supplies the primitive's classification, silhouette features,
sampling measure and Robin bounds. A query binds the operation supplied by the
provider. Two primitives with 3D coordinates can therefore use different
geometry and metadata in the same query. For example, an indexed quad can
sample its surface using its own area measure. `PrimitiveSpec` requires the
build and item-access operations. Each query checks its additional operations
when bound. A sampling-only boundary adapter is valid. Binding classification
to it raises `TypeError`. With an indexed triangle provider, classification
binds successfully and reports status `-1` for an open mesh.

Pass the same indexed source used to build the BVH to
`bvh.prepare_boundary`. The prepared boundary holds the packed index data alongside its adjacency and core attributes. Pass it directly to boundary queries.
The default also prepares silhouette features. Use `silhouettes=False`
for classification or sampling without silhouette features. For a primitive
without a declared boundary adapter, pass `adapter=` to prepare the boundary
and to bind queries that use adapter-supplied operations.

```python
import jax.numpy as jnp

from sengi import bvh, queries
from sengi.boundary.robin import prepare_robin
from sengi.boundary.sampling import prepare_boundary_sample
from sengi.primitives.triangles import Triangles

primitive = Triangles()
source = primitive.indexed(
    [[0., 0., 0.], [1., 0., 0.], [0., 1., 0.], [0., 0., 1.]],
    [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]],
)
index = bvh.build(primitive, source)
boundary = bvh.prepare_boundary(index, source)
points = jnp.array([[0.1, 0.1, 0.1]])

sign, status = queries.classification.bind(index.spec)(boundary, points)
sampling = prepare_boundary_sample(boundary)
sample = queries.boundary_sample.bind(index.spec)
point, normal, pdf, face, status = sample(
    sampling, points, jnp.inf, jnp.array([[.2, .3, .4]]),
)
robin = prepare_robin(boundary, minimum=1.0)
radius, status = queries.robin_radius.bind(index.spec)(robin, points, 0.5)
```

Classification uses even-odd parity for closed indexed surfaces. The sign
is negative inside, positive outside and zero on the boundary. If a
boundary sample falls outside a restricted query region, the query reports
a null sample with zero PDF. Robin coefficients can be scalars or arrays in
source primitive order. Supply an optional maximum for interval bounds.
Prepare new coefficient metadata when coefficients change and reuse the same
index data and bound query.

## Extending queries

A custom primitive implements the typed
`PrimitiveSpec[HostDataT, DeviceDataT, ItemT, PointT]` protocol from
`sengi.warp.contracts`. It supplies host bounds and a pack operation, the
device data type, and device callbacks for loading items and accessing item
bounds. The immutable `Primitive` descriptor from `sengi.primitives.spec`
accepts these operations directly. Add a geometric callback only when a
query needs it. Use the concrete item types in `sengi.primitives.items` for
extensions to built-in primitives. Triangle items expose their original
vertices through `a`, `b` and `c`.

### Native provider headers

For callbacks implemented with `@wp.func_native`, ship the C++ headers in your
wheel and attach a `NativeHeaders` declaration to your primitive's `native`
attribute or the `native=` field of its `Primitive` descriptor. Use an ordinary
filesystem installation so Warp can read the headers directly.

```python
from pathlib import Path

from sengi.warp.native import NativeHeaders

include = Path(__file__).resolve().parent / "native"
quad_headers = NativeHeaders(
    include_dirs=(include,),
    headers=("quad.h",),
    dependencies=(include / "quad_helper.h",),
)
```

List entry headers in include order and declare every transitive package-local
file under `dependencies`. Use quoted, header-relative includes for adjacent
helpers or distinct package-qualified names for shared include directories.
Sengi passes the same declaration to Warp's CPU and
CUDA compilers and tracks both entry and dependency contents. Use your own C++
namespace and make the headers safe to include more than once. A metric can
declare its headers in the `NearestOps` returned by its `bind` method or in
`NearestCallbacks`. Static query operation descriptors such as `RayOps` and
`ClassificationOps` also accept `native=`.

Bind a new query and retrace its JAX function after changing a header. For a
caller-owned Warp kernel, call `configure_query_kernel(kernel, *queries)` again
before launching to refresh the compiled module. Both paths retain dependencies
shared by several composed queries only once.

### Query policies

A nearest metric implements evaluation and a conservative AABB lower bound.
Use the same units for both. The lower bound must never exceed the score of
any candidate in its subtree. The `NearestCallbacks` policy packages the
callbacks and can be passed to nearest, kNN and radius bindings.

```python
import warp as wp

from sengi import bvh, queries
from sengi.metrics.nearest import NearestCallbacks
from sengi.primitives.items import PointItem3
from sengi.primitives.points import Points
from sengi.warp.contracts import warp_func

@warp_func
def evaluate(point: wp.vec3, item: PointItem3) -> tuple[wp.vec3, float]:
    return item.a, wp.length_sq(point - item.a)

@warp_func
def lower_bound(point: wp.vec3, lo: wp.vec3, hi: wp.vec3) -> float:
    delta = wp.max(wp.max(lo - point, wp.vec3(0.0)), point - hi)
    return wp.length_sq(delta)

primitive = Points(3)
index = bvh.build(primitive, primitive.data([[0., 0., 0.], [2., 0., 0.]]))
query = queries.nearest.bind(index.spec, NearestCallbacks(evaluate, lower_bound))
point, squared_distance, original_id, status = query(index.data, [[1.5, 0., 0.]])
```

Use `warp_struct` on device payload types and `warp_func` with concrete
device annotations to retain callback signatures in Python type checking.
See the [typed extension example](../tests/typing/warp_api/real_extension.py)
and [typing checks](../tests/typing/check_warp_api.py). For boundary
extensions, pair a primitive with a `BoundaryAdapter` from
`sengi.boundary.data` to prepare core and optional feature metadata.

### Calling from Warp kernels

A bound query exposes `query.device`. A nearest call in a kernel takes
`(point, cap, tree, geometry)` and returns the same four result fields as
the JAX call. Use the query's `device` callable in `wp.static(...)` and call
`configure_query_kernel(kernel, query)` from `sengi.warp.device` before
launch to configure native dependencies. Upload arrays into the device
structs before launching the kernel. For kNN in a caller-owned Warp kernel, call
`sengi.queries.knn.validate_batch_size(rows, k)` on the host before launch.
JAX checks this automatically at trace time, including rows combined by `vmap`.
The row count must fit a signed 32-bit integer and `rows * k` may be at most
`2**31`, so the last zero-based offset fits. Allocate dense `(rows, k)`
value and neighbor arrays and pass them and the row index to the device
call. Device examples are exercised in the
[Warp kernel tests](../tests/integration/test_warp_device_queries.py).

For CPU validation set `JAX_PLATFORMS=cpu` before starting Python. GPU
validation can select a device through `CUDA_VISIBLE_DEVICES`. Set
`XLA_PYTHON_CLIENT_PREALLOCATE=false` to disable JAX memory preallocation
when sharing the GPU with Warp or other benchmarks.

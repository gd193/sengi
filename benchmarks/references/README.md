# Native benchmark references

Native references use independent indexes over the suite's geometry and share
the input arrays generated for each case. They are compiled against external
checkouts and selected with `python -m benchmarks run --adapter ...`.

| Adapter | Geometry and queries |
| --- | --- |
| `cubql` | Triangle rays, nearest, 4-nearest, parity, region and radius tests. 2D/3D point nearest and KNN. |
| `fcpw` | GPU closest point, first hit, occlusion and silhouette on triangle meshes and indexed 2D segments. Star radius and boundary sampling in both dimensions. |
| `optix` | Hardware triangle ray crossings, first hit, occlusion and point parity. |
| `cudakdtree_fixed`, `cudakdtree_heap`, `cudakdtree_fixed_stack`, `cudakdtree_heap_stack` | 2D/3D point KNN with CCT or stack traversal. |

GPU timings use CUDA graph events, with CUPTI activity timing for OptiX.
Construction,
transfers, shader compilation and result decoding are outside the timing
interval. Point KNN reports squared distances. Heap candidates are sorted and
remapped to original primitive IDs inside the measured cudaKDTree kernel.
The FCPW sampler rejects proposals outside the ball inside the measured shader.

## Build and run

Use a Conda CUDA toolkit with a supported C++ compiler and (for OptiX) CUPTI
headers and libraries. Select the architecture with `--arch`. The native build
manifest records the command, compiler versions, upstream revisions and content
hashes for each executable. Use a work-drive output directory.

The triangle baseline used cuBQL `20f9db19`, FCPW `e498ca81`, Slang-RHI
`12f1bffb`, and OptiX 9.1 `f1f6dd80`. The previous
point-KNN comparison used cuBQL `d2e62a39` and cudaKDTree `37ce4c37`.
For a build that includes triangles and points, use a cuBQL revision that
includes the triangle box and radius queries.

Initialize FCPW's Eigen and Slang-RHI submodules. Build Slang-RHI with CUDA support before building the GPU adapter.

```sh
cmake -S "$FCPW" -B "$ART/fcpw-build" -DFCPW_USE_ENOKI=OFF \
    -DFCPW_ENABLE_GPU_SUPPORT=ON -DSLANG_RHI_ENABLE_OPTIX=OFF \
    -DSLANG_RHI_ENABLE_VULKAN=OFF -DSLANG_RHI_ENABLE_CPU=OFF
cmake --build "$ART/fcpw-build" --target slang-rhi -j8

python -m benchmarks build --adapter cubql,fcpw,optix,cudakdtree \
    --cuda "$CUDA_ROOT" --cuda-host "$CUDA_HOST_CXX" --cubql "$CUBQL" \
    --cudakdtree "$CUDA_KD_TREE" --fcpw "$FCPW" --fcpw-build "$ART/fcpw-build" \
    --optix "$OPTIX" --output "$ART/bin"
python -m benchmarks run --suite triangles --binaries "$ART/bin" \
    --count 512 --param radius=0.05 --output "$ART/triangles"
python -m benchmarks run --suite point_knn --binaries "$ART/bin" \
    --mesh points2,points3 --size 1024 --count 1024 --output "$ART/knn"
python -m benchmarks report "$ART/triangles"
```

cuBQL builds 2D points with a spatial median and uses SAH for 3D points and
meshes. Its leaf threshold defaults to 4 and can be set with
`--reference-leaf-size`.
`--launches` sets the number of GPU launches per sample. The FCPW queue shim accesses private Slang-RHI CUDA queue and heap
state. Recheck Slang-RHI compatibility if the revision changes. On the CUDA
13.2 lab toolkit, use Clang as the nvcc host compiler for cuBQL.

## Native fixture exchange

The runner writes a temporary directory with `manifest.tsv`, `arrays/`, and
`out/`. The manifest starts with `sengi-native-1`, followed by the block
size, warmups, launches per repeat, repeats and leaf threshold. Subsequent
tab-separated records have these fields.

- `geometry NAME PATH DTYPE SHAPE` describes each geometry array.
- `case KEY MODE BATCH_SIZE` opens one selected case and sets its output prefix
  to `out/KEY`.
- `input NAME PATH DTYPE SHAPE` describes an array for the open case.
- `parameter NAME VALUE` supplies a scalar mode parameter.

Array paths are relative to the fixture directory. Shapes are comma-separated
dimensions. Arrays use contiguous row-major little-endian element ordering.
Identical arrays are stored once per exchange. The executable builds its
index from the geometry records, runs the requested cases against that index
and writes per-repeat milliseconds and binary answer fields at the output prefix.
Build and feature preparation timings, where measured, are emitted separately. The adapter decodes the corresponding named answer arrays, including
native interaction records, into the query catalog's output fields.

The scalar and vector field names are `.value`, `.ids`, `.dist` and
`.points`. Integer fields use little-endian int32 and floating fields use
little-endian float32. Ray and nearest list fields use the requested capacity
as their second dimension. The point KNN paths write squared distances in
`.scores` with scalar `.count` and `.status` fields. The FCPW GPU executable
writes `.bin` interaction records with float32 point (three), normal (three),
UV (two) and distance (one), followed by an int32 primitive ID. Each file has
one row per case query. The adapter checks the expected output shape before
publishing the normalized arrays.

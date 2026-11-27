# Comparing Sengi queries

Run from the repository root in the project's Conda environment with `PYTHONPATH=src:.` to measure the checkout. Every execution uses a named query, one geometry variant and one seeded batch. Each adapter builds its own index, runs on that batch and publishes its timings and answers.

```sh
python -m benchmarks run --suite full --cache /work/cache --binaries /work/bin --output /work/full
python -m benchmarks list
python -m benchmarks list --suite triangles --mesh bunny --size 1.3m --coverage
python -m benchmarks run --suite smoke --output output/benchmarks/smoke
python -m benchmarks run --suite triangles --builder sah --binaries /work/bin --output /work/triangles
python -m benchmarks run --suite boundary --binaries /work/bin --output /work/boundary
python -m benchmarks report /work/baseline /work/candidate
```

The default unfiltered run selects `full`. It combines the seven presets below in one process, measures identical cases once, and shares each geometry build across its queries for each adapter. It retains every mesh tier, the 8,192-query triangle batches, the full point kNN sweep, and the saturation batches up to 131,072. The full suite has a 20-minute end-to-end budget on a lab RTX 3090 with prepared assets and reference binaries. For the full suite the Linux CLI sends SIGTERM to the process group at 19m 50s and allows up to ten seconds for cleanup and a partial report. Targeted runs can be timed separately. Use a fresh directory on local storage for wall measurements. The default timing schedule uses one warmup, one launch per timed sample, and three samples. Graph-based adapters also replay the graph once before timing.

The smoke run samples queries near a sphere and checks nearest neighbors on points. It can use Sengi on its own. Build the independent libraries before timing them. A full run includes reasons for skipped adapter/query pairs. A missing explicitly selected adapter, a failed execution, or a numerical mismatch produces a nonzero exit code. A report can be generated for a partial run. Reports require the GPU-only run format because older records could certify results using host-computed answers. [Building and selecting the reference libraries](../benchmarks/references/README.md)

## Choosing the comparison

| Preset | Coverage |
| --- | --- |
| `smoke` | Small triangle and point cases for checking the setup |
| `triangles` | All 19 cataloged triangle modes across the sphere, torus, bunny and armadillo at six matched sizes from 320 to 1,310,720 triangles |
| `boundary` | Near-surface silhouette, star radius, boundary sampling and Robin radius across the same 24 meshes |
| `builtins` | Primitive and metric bindings, open-boundary queries, selected nearest results and weighted radius accumulation |
| `reference_queries` | Point and planar segment fixtures for independent nearest, silhouette and ray comparisons |
| `point_knn` | Two and three dimensions, three distributions, seven K values, different point counts and batch sizes |
| `saturation` | Fixed geometry at 512, 8,192, 32,768 and 131,072 queries to distinguish latency and throughput |

Use `--mesh`, `--size`, `--query`, `--workload`, `--count` and `--seed` for targeted runs. Each takes a comma-separated selection. Select a library with `--adapter`. Coverage shows which libraries implement the query with the requested metric and parameters. For example, a broad-phase triangle-box overlap and an exact triangle-box intersection count measure different operations.

```sh
python -m benchmarks run --mesh bunny --size small,1.3m --query region_count \
  --adapter sengi,cubql --count 512,8192 --param radius=0.05 \
  --builder sah --leaf-size 4 --reference-leaf-size 4 \
  --binaries /work/bin --output /work/sah-boxes
```

The near-surface workload uses a default radius of 0.05 and offsets sampled surface points uniformly along the face normal from −2 to +2 radii. Local region, radius, overlap and sampling queries draw points near the surface in both the triangle preset and targeted mesh runs. For the archived Gaussian mesh comparison use `--workload gaussian --count 512 --seed 404 --param radius=0.05 --builder lbvh --leaf-size 16 --reference-leaf-size 4`. The triangle and near-surface presets otherwise use 8,192 queries per batch. Point-cloud workloads use separate seeded streams for geometry and queries. Clustered workloads share the cluster centers between the two streams. The Sengi and Warp adapters default to SAH with a leaf-size threshold of 4. Use `--builder lbvh` to measure the tradeoff between build time and query time. Mesh, size, query and metric selections narrow an explicit preset. Count, workload and seed override its batch settings. Give each experiment its own directory. Set `SENGI_ASSET_CACHE` or pass `--cache` to share generated meshes across runs. [Mesh tiers and provenance](../asset/README.md)

Sengi stores validated SAH triangle and point-cloud trees in `asset/sah/` on the first run. Later runs with the same ordered geometry arrays, leaf size and builder fingerprint load the tree and repack the geometry on the selected GPU. Cache entries are disposable and ignored by Git. On lab machines, `asset/sah` can be a symlink to shared work storage.

## Reading results

- `run.json` contains the planned matrix, machine and device, timing settings, package versions, source fingerprints and binary hashes. Each finished execution has an atomically published result under `results/`. Geometry, inputs and answers use content-addressed NPZ files under `arrays/`. Run the same command with `--resume` to finish a partial run with matching settings and provenance.
- `summary.csv` reports median batch milliseconds, median absolute deviation (MAD) and queries per second. The individual record contains every raw repeat, the timing method, build or preparation times where measured, and validation details. GPU measurements cover warmed resident queries. Only CUDA devices are accepted. The build scope column describes the work included by each library. For cached Sengi SAH indexes, `sah_cache` records a hit or miss. A miss reports `build_ms` including the cache write. A hit reports `index_load_ms` including geometry preparation and device transfers. Its boundary time covers joint preparation for the requested boundary queries. Automatic within-run ratios require matching device, instrumentation and timing scope. OptiX records CUPTI kernel durations while graph events include gaps between kernels. `sengi_over_reference` above one means the reference completed the batch sooner.
- Run reports include unsupported, missing, failed, mismatched and unchecked cells. The first adapter with a valid result anchors each case, with Sengi preferred. The compared_to column identifies the peer. A query without a GPU reference is timed and marked unchecked. Host-side validation checks returned fields and compares GPU outputs. `verified` includes a Sengi row validated through its paired reference. The agreeing_adapters and disagreeing_adapters columns name the visible peers on both sides of each comparison. A disagreement marks the case disputed. Each agreeing pair retains its verification and eligible timing ratio. Builder, leaf-size and precision differences are flagged alongside those ratios. Sampling stays unchecked when either adapter accepts fewer than 16 draws or the comparison tolerance (six paired standard errors plus the absolute tolerance) exceeds 10% of the smaller area estimate. A statistically significant disagreement still counts as a mismatch. Two-run reports pair the same case and adapter, identify changed settings, and show speed ratios for verified matches with the same fixture hashes and execution settings.
- Input, output and internal precision are recorded separately. Sengi’s fused triangle nearest-point queries use FP32. Triangle radius and kNN queries use FP64 for scoring, and the angular metric evaluates angles in FP64. An unaudited implementation has internal precision `unknown`. Use `python -m benchmarks report /work/triangles --fp32-only` to select rows with confirmed FP32 internal arithmetic. Verification in this view requires both members of the pair to pass that filter.
- Counts and classifications compare exactly. Returned IDs are checked for valid ranges where IDs name source primitives. Paired GPU results are compared for distances, ray intervals, crossing counts and the sorted crossing distances of complete hit lists. Equal-distance winners may have different IDs. Sampling records support, PDF, null fraction, accepted samples per second and uncertainty in its area estimate. The result includes example rows for disagreements and the hashes needed to replay their inputs.

## Extending the suite

**A mesh** needs one `asset/<name>/mesh.toml` and a file or preparation recipe for each available tier. Use its directory name with `--mesh`, or pass `--mesh '*'` to sweep all discovered mesh manifests. The loader checks hashes, primitive counts, indices and declared topology. The fixed four-family triangle preset stays reproducible as new assets appear. [Manifest example](../asset/README.md)

**A query** needs a catalog entry specifying the primitive requirements, parameters, named inputs and outputs. Add its workload fields and validation rule, bind it in the Sengi adapter and add an implementation to each applicable reference. Include a small fixture with a known answer. Select the mode by name immediately or add it to a preset. The weighted radius case exercises this path through an application-provided accumulation policy. [Catalog](../benchmarks/cases.py) · [Workloads and presets](../benchmarks/workloads.py) · [Answer validation](../benchmarks/queries/validation.py) · [Adapters](../benchmarks/adapters/)

**A native reference** declares its supported cases, build command, named fixture inputs and output decoder. Keep library-specific kernels and shaders with that library under `benchmarks/references/`. The shared harness handles timing and fixture exchange. All adapters emit the same case answer fields so reporting and correctness rules stay independent of library choice.

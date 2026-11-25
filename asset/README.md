# Shared triangle meshes

Each baseline shape has six mesh tiers. Sphere and torus ship three PLY files each, and bunny and armadillo ship five each. The loader generates the remaining tiers. A given size has the same
triangle count for every shape, so shape and mesh size can be varied independently.
Coordinates are float32 and face indices are zero-based int32. Face order in the
file defines the original primitive IDs.

| Size | Triangles per mesh |
| --- | ---: |
| small | 320 |
| medium | 5,120 |
| large | 20,480 |
| 82k | 81,920 |
| 328k | 327,680 |
| 1.3m | 1,310,720 |

| Directory | Geometry |
| --- | --- |
| [sphere](sphere/) | Convex unit icosphere at subdivisions 2, 4, 5, 6, 7, and 8. |
| [torus](torus/) | Ring with major radius 0.7 and minor radius 0.3. Tessellations range from 20 × 8 to 1,280 × 512 sections. |
| [bunny](bunny/) | Stanford Bunny with the five scan openings capped for closed-boundary queries. |
| [armadillo](armadillo/) | Stanford Armadillo with thin limbs and narrow gaps. |

All 24 meshes are connected, watertight, and outward-oriented. The same
vertex and face arrays can be passed to any triangle query, including inside/outside
classification, nearest silhouette, and boundary sampling. Load with
`trimesh.load_mesh(path, process=False)` to preserve the stored indexing.

## Sources and preparation

Sphere and torus were generated with Trimesh 4.12.1. The two scans are
derived from the original reconstructions provided by the Stanford University
Computer Graphics Laboratory. Stanford permits free redistribution with
acknowledgment and requires permission for commercial use. [Stanford 3D
Scanning Repository and terms](https://graphics.stanford.edu/data/3Dscanrep/)

- **Bunny** uses `reconstruction/bun_zipper.ply` from the [Stanford Bunny archive](https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz).
- **Armadillo** uses the [Stanford Armadillo reconstruction](https://graphics.stanford.edu/pub/3Dscanrep/armadillo/Armadillo.ply.gz).

For each scan, we centered its original bounding box at the origin and scaled
its longest side to two units before simplification. For the bunny, we closed
each boundary loop with triangles meeting at its mean vertex position. We ran
libigl 2.6.2 QSlim with intersection blocking from the full mesh to large, then medium,
then small. All sizes use the original transform. The stored bunny fixtures
therefore differ from the open scan used in early nearest-only profiles.

## Million-triangle fixtures

The denser Armadillo scans come from simplifying the original 345,944-face
reconstruction. For the 81,920-face bunny we took the outer surface of the
capped scan, subdivided it, and simplified it with QSlim. We smoothed the few
vertices incident to residual intersections or collapsed faces after rounding. We then improved the 81,920-face bunny by relaxing vertices incident to thin faces 35% toward their neighbor mean and rounding to the same binary grid after each of 100 iterations. Each iteration selected faces with twice-area below 2×10⁻⁶ or twice-area divided by longest edge squared below 0.015. Face connectivity and ordering were preserved and the repaired mesh was checked for self-intersections. The bunny at 81,920 and the
Armadillo at 327,680 are stored on a binary coordinate grid with spacing
2⁻²⁰. Their subsequent midpoint subdivisions are exactly representable in
float32. Those subdivisions increase the workload while preserving the
piecewise-planar scanned surface. Sphere and torus are sampled analytically
at each resolution.

Each mesh has a `mesh.toml` declaring its indexed variants, provenance, and
boundary properties. A stored variant names a file relative to that manifest.
A generated variant specifies an analytical mesh, midpoint subdivision of a
pinned base, or centroid clipping of a pinned base. Face order in the binary
PLY defines primitive IDs. The loader verifies the file's SHA-256, index range,
triangle count, dimensionless triangle quality (twice area divided by longest edge squared must exceed 0.0005), and any declared watertightness and orientation.

Generated variants pin the SHA-256 of the ordered float32 vertices and int32 face array, including their shapes and dtypes. PLY header changes preserve this identity. Stored files pin the file SHA-256. The recipe, source hash and expected geometry hash identify each generated cache entry. Set `SENGI_ASSET_CACHE` or pass `--cache` to use a shared work drive. The default is `output/asset-cache` in the checkout.

Add a mesh by creating a directory under `asset/` with a manifest and geometry.
Variants need only describe the tiers that exist. For example, a single-tier
asset can use this manifest, with the actual triangle count and SHA-256
substituted for the example values.

```toml
provenance = "Description and source of the geometry"
closed = true
oriented = true

[variants.small]
file = "small.ply"
faces = 320
sha256 = "<SHA-256 of small.ply>"
```

Use `python -m benchmarks list` to discover assets and
`python -m benchmarks list --mesh bunny --coverage` to see eligible queries. The optional `open_torus`
asset shares the torus tiers, with faces clipped by their centroids. Its
manifest excludes closed-boundary classification.

```sh
python -m benchmarks run --suite smoke --output runs/smoke
python -m benchmarks run --mesh bunny --size 1.3m --query region_count \
  --adapter sengi,cubql --output runs/bunny-boxes
python -m benchmarks report runs/bunny-boxes
```

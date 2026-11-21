# Sengi

JAX spatial tree construction and primitive-generic Warp geometry queries.
Import host builders from `sengi.bvh`, `sengi.kd`, or
`sengi.octree`. Warp query factories live in `sengi.bvh.warp`.

Build and install with conda tooling from the repository root:

```sh
mamba create -n spatial-build -c conda-forge rattler-build
mamba run -n spatial-build rattler-build build --recipe recipe/recipe.yaml -c conda-forge --output-dir "$PWD/output"
mamba create -n spatial-dev -c "file://$PWD/output" -c conda-forge python=3.14 sengi=0.1.0 jax=0.10.2 warp-lang=1.18.0 pytest
mamba run -n spatial-dev python -m pytest tests
```

For source development, set `PYTHONPATH="$PWD/src"` when running tests in that
environment. Omit it to exercise the installed artifact, including the native
headers loaded by Warp. Warp is optional for host-only tree construction.

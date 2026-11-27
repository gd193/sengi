# Compiler settings

Prefer compiler defaults for numeric tuning parameters. Hard-coded launch
bounds and other numeric compiler settings can overfit to a particular GPU
model, architecture, or compiler version. Experiments and profiling may use
these overrides without prior approval. Using one outside that scope requires
explicit permission. Record the approved value, the approver’s name, and the
approval date alongside the setting. Boolean compiler options such as
`enable_cuda_smem_spilling` are allowed without this approval.

# Speed benchmarks

The benchmark suite compares GPU acceleration structures. Every timed query and
every reference query must execute on a GPU. CPU reference implementations,
brute-force CPU oracles, CPU timing modes, and CPU fallbacks are forbidden,
including during validation. Compare outputs from independent GPU adapters.
When no GPU reference covers a case, report its timing as unchecked.

Host code may prepare fixtures, build indexes for GPU libraries, compare returned
outputs, and produce reports. It must never recompute a query answer on the CPU.
Apply this rule to benchmark implementation and review, including delegated work.

The full GPU speed suite must finish within 20 minutes end to end on a lab RTX
3090 with prepared assets and reference binaries. Preserve all mesh tiers,
including every 1,310,720-triangle mesh. Measure the complete run through
`python -m benchmarks run --suite full`. A timeout is a failed budget check.

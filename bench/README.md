# Benchmarks

Two different things live here:

- `micro/` and `p2p_fetch_probe.cu` — microbenchmarks and probes used while tuning kernels. They are developer
  tools, not a product benchmark.
- `results/` — historical result write-ups. They have moved to [`docs/benchmarks/results/`](../docs/benchmarks/results/)
  so they sit with the rest of the documentation.

The user-facing benchmark is `./maya.sh --bench` (decode: three greedy 256-token answers; prefill: an 8K-token
prompt). Report a machine with `--report` and open a speed report issue; rows are collected in
[`docs/benchmarks/SPEED-REPORTS.md`](../docs/benchmarks/SPEED-REPORTS.md).

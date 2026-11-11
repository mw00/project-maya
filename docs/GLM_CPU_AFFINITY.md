# Choosing GLM shared-pool CPU affinity on Linux

The engine's automatic CPU sizing and unpinned shared pool remain the defaults.
`tools/glm_pool_cpus.py` prepares an optional CPU list for a thread count **you
choose for your machine and workload**. It does not find an optimal count or
change a config, CPU affinity, or running service.

## Choose the count and generate a list

Run the tool inside the same container, cpuset, `taskset` or `numactl` restrictions
that will apply to the engine. Replace `YOUR_THREAD_COUNT` with a positive integer:

```sh
python tools/glm_pool_cpus.py --threads YOUR_THREAD_COUNT
```

There is no default count or machine-size preset. Choose a count that leaves
capacity for the engine's service threads, disk readers, other programs and your
CPU-time quota. The allowed logical-CPU mask is a capacity ceiling, **not** a
performance recommendation or a measurement of the container's CPU-time quota.
More threads, SMT or use of another NUMA node can make a workload slower.

Stdout is a JSON fragment containing exactly two entries under `env`:
`MAYA_GLM_CPU_LANE` (your chosen count) and `MAYA_GLM_POOL_CPUS` (the generated CPU
list). Merge those entries into a separate copy of your config's existing `env`
object; preserve its other entries. Explanations go to stderr. The tool does not
start Maya or write either config. Direct engine users can set the corresponding
`STRATA_GLM_*` environment variables; the Maya launcher handles the aliases.

The tool intersects `sched_getaffinity(0)` with online CPUs, reads their NUMA-node
membership and `thread_siblings_list`, then takes one allowed logical CPU per
physical core before taking additional SMT siblings. Within each SMT round it
cycles through nodes in numeric order, skipping nodes whose capacity is exhausted.
Unequal node capacities and restricted masks therefore need not produce equal
node counts. Core grouping follows the [Linux sysfs topology](https://www.kernel.org/doc/html/latest/admin-guide/cputopology.html)
reported to the process; in a VM this may differ from the host's physical topology.

Non-contiguous CPU/node IDs, one or multiple nodes, partial sibling sets and
different SMT widths are supported. Missing/inconsistent topology, too many
requested threads, no usable CPUs, or allowed CPU IDs at or above 1024 are errors.
The last restriction matches the engine's current fixed Linux affinity mask.
Regenerate after changing CPU online state, topology or the engine's restrictions.

## Which engine path uses it

These settings apply to the **global shared CPU pool**, normally used by a GLM
split across three or more GPUs. A two-GPU split can explicitly select that
existing path with `MAYA_GLM_CPU_SHARED=1`; the tool does not change the pool mode.

When an explicit global `MAYA_GLM_CPU_LANE` reaches this shared-pool path, its
count is bounded by the calling thread's actual Linux affinity mask rather than
the previous manual limit of 64. Unknown affinity and non-Linux builds retain
the 64 limit. The automatic sizing branch is unchanged. Per-device
`MAYA_GLM_CPU_LANE<n>`, independent pools, and the socket/node-split pool paths
retain their existing sizing and placement. `MAYA_GLM_LANE_CPUS<n>` is a separate
per-device control. Check the startup log for the path and effective count;
older binaries still clamp the manual count to 64.

A count of N constructs N-1 background workers; the caller also executes work.
The generated list contains N entries to match the requested count, but only its
first N-1 entries pin background workers. **The last entry is unused: it does not
reserve or pin a caller.** Service/calling threads keep their existing affinity.
The tool does not set `MAYA_GLM_SVC_CPUS<n>`, migrate memory pages or change the
worker spin time. This is not the DeepSeek Engram page-placement setting.

## Compare on your own machine

Keep your normal configuration as a baseline. To compare affinity at a chosen
count, make two test configs with the same explicit `MAYA_GLM_CPU_LANE`: one with
both aliases of `GLM_POOL_CPUS` absent, the other with the generated list. Keep the
model, expert profile, context, KV, prefill chunk, speculation, power and other
settings the same. Comparing different thread counts is a separate configuration
comparison, not a pure affinity test.

Run the official `--bench` entrypoint with each config and repeat the baseline
after the candidate. It measures serialized requests; use the separate agent
benchmark if concurrency is the workload you need. Inspect the effective
`cpu_threads` count and each worker's `Cpus_allowed_list` under
`/proc/<native-pid>/task/<tid>/status`: the current engine does not report failed
pinning calls. Compare output identity as well as timings; CPU/GPU expert routing
and automatic lane retiming can change both.

Retain a choice only on evidence from your hardware and workload. Removing both
aliases of the two overrides from config/environment returns to automatic sizing
and default shared-pool placement at the next normal start.

[One dual-Xeon observation](benchmarks/glm-pool-affinity-20261111.md) contains the
128/144-thread experiments. Those counts are historical test inputs, **not
defaults or recommended values for other machines**.

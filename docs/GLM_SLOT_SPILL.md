# Opt-in RAM and disk conversation slots

Concurrent agents often interleave more conversations than the engine has active contexts. An agent may pause for a tool or another task and then return to the same token prefix. With the legacy count cap, that conversation can already have been discarded even when host memory and disk have room. This setting lets the conversation remain in RAM, then spill to disk under separate byte budgets, reducing repeated prefill when it returns.

This is a capacity and reuse feature. It does not add decoding workers, change the scheduler, share arbitrary agent memory, or make unrelated prefixes reusable. It does not change weights, routing, sampling, or the token-prefix/image compatibility check.

## Configuration

`MAYA_GLM_SLOT_SPILL` is **off by default**. Set it to `1` to opt in; leave it unset or set it to `0` for the existing count-capped behavior.

| Setting | Behavior when spill is enabled |
|---|---|
| `MAYA_GLM_SLOT_RAM_GB` | RAM state budget in GiB; 8 when unset |
| `MAYA_GLM_SLOT_GB` | Existing disk state budget in GiB; 16 when unset |
| `MAYA_GLM_SLOT_DIR` | Existing slot directory; choose a private, writable directory appropriate to the deployment |
| `MAYA_GLM_SLOTS=0` | Disables slots, including spill |

The deployment chooses both budgets; no machine's RAM capacity or deployment-specific cache allocation is embedded in the implementation. These are retained-state budgets, not preallocated memory or a limit on total process RSS. Serialization can temporarily need another state buffer; token metadata and other engine allocations also need headroom. The existing free-disk guard remains in place. The separate DSA prompt-cache path still takes precedence where it disables conversation slots.

RAM pressure spills the least recently used RAM state. Disk pressure evicts the least recently used disk state, including a state just spilled from RAM. Moving a state is not a use and preserves its last-use timestamp across restart. A state larger than the disk cap, including a zero cap, is rejected before writing or evicting another file. If a state cannot be retained, its next request reads that prefix again.

Spill mode retains the folder across runs and flushes RAM states on normal CLI/serve exit. A crash or forced kill does not guarantee persistence. When shutdown has more state than the disk budget, older states can be dropped. This is a bounded cache, not durable archival storage. Miss logs identify no candidate, divergent tokens, an insufficient prompt length, or changed images without printing prompt text.

## Current standalone branch: host verification

Base: upstream `7e68a30f2f208056195352129188661fb6b94445`, without the separate development branch's NUMA/shared-pool and dynamic-DFlash overlays.

On 2026-11-15, Windows/MSVC 14.50 completed **17/17 host scenarios** and compiled the complete modified `generate.cpp` translation unit. The latter is a host object build, not a CUDA build or final engine link. The source and harness hashes, each scenario's actual output/assertions, and exit codes are in [the host receipt](testing/glm-slot-spill-host.json); only temporary filesystem paths were normalized.

The harness extracts the actual slot structure and management lambdas, plus the actual CLI success tail, from `generate.cpp`. Only the model save/load surface uses deterministic byte buffers. Coverage includes:

- RAM pressure and disk LRU, including cold spill versus a recently used file;
- oversized/zero budgets, a discarded snapshot's retained flag, and failed writes;
- normal-return flushing, CLI exit, restart indexing and exact saved-byte recovery;
- mixed RAM/disk last-use order across restart;
- the runtime miss reasons and refusal to load invalid prefixes;
- legacy disk/RAM-only count caps, an unset opt-in, and configurable fallback budgets.

Run it with a C++20 compiler, with no model or GPU:

```sh
python tools/test_glm_slot_spill.py --receipt glm-slot-spill.json
```

On Windows, run from an MSVC developer shell. On POSIX, the default compiler is `c++`. `--cxx` or `CXX` selects an executable; no local Visual Studio path is hardcoded. The normal GitHub CPU-only workflow also invokes the harness. Fixture cleanup is confined to its fresh temporary directory.

## Historical evidence reused, not rerun

The following measurements belong to the earlier v1.1.63 development/deployment build, not this standalone PR head. They used a GLM-5.3-Flash Maya-L pack, four RTX 4090s at 255W each, about 251.3GiB host RAM, 144 CPU lanes, four active contexts, a 1M context limit, 5120-token prefill chunks and layer split `12,24,35`. That branch included separate NUMA/shared-pool and dynamic-DFlash changes. [The historical receipt](testing/glm-slot-spill-prior.json) records exact source/binary identities, explicit test budgets, outputs, and version differences.

| Window | Evidence | Boundary |
|---|---|---|
| Pre-fix host regression | 6/15 passed; nine expected failures | Negative control for the original spill prototype; not upstream `main` |
| Repaired host regression | 15/15 passed | Deterministic host state bytes, not real GPU/model state |
| Repaired deployment | Linux/CUDA incremental link succeeded; runtime executable hash matched | Earlier deployment binary, not a build of this standalone PR |
| Repaired deployment smoke | Seven requests all returned `OK`; returning to the first 1470-token prompt restored 1469 tokens from RAM; editing its beginning restored 0 | Five different short conversations, return, then negative control; natural production traffic interleaved |
| Earlier capacity observation | After nine different 9236-token conversations, returning to the first gave 4.480s / 0 cached tokens with legacy slots, versus 0.202s / 9235 cached tokens with RAM spill | Pre-fix prototype, a single synthetic return; not an isolated benchmark or a production average |

The five-conversation version of the historical capacity check did not exhaust four active contexts plus four legacy slots. It therefore did not answer the eviction question. Large client delays in other historical rows included production queuing and are not prefill comparisons. These observations motivate retaining more conversations; they do not establish an average speedup for a real multi-agent workload.

The standalone branch deliberately reuses the existing 16GiB disk fallback instead of the prototype's 32GiB fallback. Explicit environment budgets still override it. The historical smoke supplied explicit budgets, so its allocation is not a PR default.

## What remains unverified

No connection to the model host, model load, inference request, benchmark, or production change was performed to prepare this PR. There is no new `tools/agent_bench.py` comparison, decode/prefill throughput claim, full quality suite, exact PR-head token-parity measurement, or PR-head CUDA link. No local WSL was started.

The historical model smoke hit RAM and did not force disk overflow, restart and real-model disk restoration. Full-budget memory pressure, crash recovery, and production-wide multi-agent hit rate/latency remain unmeasured. Host byte tests cover policy and persistence ordering, not those model-level boundaries. The existing deployment was left running while the standalone PR was prepared.

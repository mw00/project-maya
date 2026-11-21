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

## Standalone branch: real-model disk restoration

On 2026-11-15, the product source at `082464a8b295304d56c796f0b5984057be3eb0b8` completed a Linux/CUDA final link and a separate native GLM-5.3-Flash Maya-L IQ3_S check. This used four RTX 4090s, a 4096-token context, one active sequence, 1024-token prefill chunks, the upstream shared CPU pool (76 threads observed), and no MTP, DFlash or vision. It did not use the development branch's local overlays. Source, executable, tokenizer and driver hashes are in [the model receipt](testing/glm-slot-spill-model.json); the [frozen token inputs](testing/glm-slot-spill-model-inputs.json) are synthetic.

The isolated slot directory started empty. Its RAM cap was deliberately **0.001 GiB** and its disk cap **2 GiB**, so a real 161,738,820-byte state had to spill. These are fixture settings, not deployment defaults. Two model processes handled exactly four requests, with no separate warm-up request:

| Request | Input tokens | Reused tokens | Result |
|---|---:|---:|---|
| Process 1: A, cold | 1228 | 0 | 10 output tokens, normal stop |
| Process 1: B, different prefix | 1228 | 0 | A's `.bin/.meta` present before shutdown |
| Process 1: A, again | 1228 | 1227 | Native log confirms disk restore; all 10 output token IDs equal the cold result |
| Process 2: A, after normal exit/restart | 1228 | 1227 | Startup indexes two retained files; disk restore and all 10 output token IDs still match |

Both processes exited normally with code 0. The A output hash, including its stop token, is `a74c111b571bac5749c085badc7bcef748a04c0f45da51c056b9823182da72d7`. An earlier executable-start attempt failed because `libcudart.so.13` was absent from the test process's library search path; it loaded no model and sent no requests. That failure is retained in the receipt. Supplying the existing runtime libraries and checking the exact native command with `--help` preceded the one corrected attempt.

The [executed Linux fixture](../tools/check_glm_slot_spill_model.py) records inputs before sending, preserves native transcripts, and stops on the first mismatch. It is explicitly scoped to four idle GPUs with at least 20 GiB free each, the split `12,24,35`, and this host's 160 GiB expert-RAM ceiling with 32 GiB headroom. It never stops a service; reserve the test resources separately. `--preflight-only` checks imports, frozen inputs, shared libraries and argument parsing without loading a model, and reports occupied GPUs; an actual run requires them to be free. Use a fresh output directory each time:

```sh
# Supply LD_LIBRARY_PATH if the built executable needs a private CUDA runtime directory.
python tools/check_glm_slot_spill_model.py \
  --repo . --exe /path/to/pr-build/maya \
  --pack /path/to/pack --tokenizer /path/to/pack/tokenizer \
  --output /path/to/fresh-results --gpus 0,1,2,3
```

This verifies file-backed state recovery across processes, not cold-SSD latency or power-loss durability: the OS page cache was not dropped. It is one short synthetic pair and one restart, with tiny budgets. It does not replace a multi-agent benchmark, a full quality suite, long-context checks or other decoding modes.

## Independent two-CMP example (same PR source)

Example and reused test evidence contributed by [mercant1982](https://github.com/mercant1982); [original gist](https://gist.github.com/mercant1982/1dca5f922c56b5c8226541d40ddb2f53).

The independent two-GPU check used this PR head
`fe3ba47d9596224dfdf89a0595185ffc9236827e`, two CMP 170HX cards with
64 GiB VRAM each, and the GLM-5.3-Flash Maya-L IQ3_S pack. The layer split
and RAM ceilings below are that tested configuration, not defaults for
other GPUs or models. Use an isolated build, idle test GPUs, a fresh private
slot directory, and a shell without inherited `MAYA_*` / `STRATA_*` overrides.
Set `MAYA_EXE` and `MAYA_PACK` to that build and pack, and `MAYA_RESULTS` to
a fresh writable directory. Supply `LD_LIBRARY_PATH` if needed by the build.

```sh
env CUDA_VISIBLE_DEVICES=0,1 \
  MAYA_GLM_SLOTS=1 MAYA_GLM_SLOT_SPILL=1 \
  MAYA_GLM_SLOT_RAM_GB=0.001 MAYA_GLM_SLOT_GB=8 \
  MAYA_GLM_SLOT_MIN=1024 MAYA_GLM_SLOT_DIR="$MAYA_RESULTS/slots" \
  MAYA_GLM_SLOT_KEEP=1 MAYA_GLM_CKPTS=0 MAYA_GLM_CKPT_EVERY=0 \
  MAYA_GLM_PREFIX_CUT=0 MAYA_GLM_MULTI_CKPTS=0 MAYA_GLM_NO_MTP=1 \
  MAYA_GLM_RAM_GB=160 MAYA_GLM_RAM_HEADROOM_GB=32 MAYA_GLM_CPU_LANE=6 \
  MAYA_GLM_USAGE="$MAYA_RESULTS/usage.bin" \
  "$MAYA_EXE" --glm-pack "$MAYA_PACK" --max-context 40960 --parallel 1 \
  --prefill 1024 --layer-split 23 --glm-ram-resident \
  --seed 1729 --greedy --temperature 0 --serve
```

`--serve` here is the native stdin/stdout protocol, not an HTTP endpoint.
The [complete C1 driver, inputs, token IDs and native logs](https://gist.github.com/mercant1982/1dca5f922c56b5c8226541d40ddb2f53)
are in `worker.py`, `protocol_2gpu.py`, `bench-fixtures.json` and
`C1-README.md`. The worker defines the eight-request study; do not invoke
the protocol library's older four-request `main` as that study. Preserve the
same fresh slot directory across the two native processes and exit each
normally with `QUIT`.

Actual prompt sizes were 3,276/3,280 and 26,191/26,185 tokens, with at most
16 output tokens per request. Each length used A cold, B cold, A from disk,
then A after restart: eight requests in two processes. All four restored A
outputs matched their cold controls exactly, including finish reasons.

The same evidence also records a separate `--parallel 8`, 8192-context,
1-GiB disk-capacity study and matched cache-off controls. Output differences
occurred in both conditions. C1 parity does not establish C8 output
equivalence, a cache-specific cause, general answer quality, throughput,
cold-SSD latency or power-loss durability. No new model run is implied by
this documentation example.

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

The initial PR preparation used host checks only. The later isolated check above adds a CUDA link, real-model disk spill/restart and exact output-token equality for the stated input and configuration. There is still no new `tools/agent_bench.py` comparison, decode/prefill throughput claim, or full quality suite. No local WSL was started.

The historical model smoke hit RAM; the later isolated check supplies the separately identified disk evidence. Full-budget memory pressure, crash recovery, long-context and speculative-decoding parity, and production-wide multi-agent hit rate/latency remain unmeasured. Host byte tests cover policy and persistence ordering, not those remaining model-level boundaries.

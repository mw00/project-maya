# GLM CUDA+HIP helper: Strix Halo / RTX 5090

Measured 2026-11-12 on upstream v1.1.60
(`8b0a449c599fbb370a1dcc0979ae7c84d4310529`) plus this contribution.
Earlier throughput is retained in Git history; HELPER_COST_BENCHMARK.md records
historical v1.1.55 cost-admission measurements.

## Hardware, method and results

I ran Maya’s unchanged built-in benchmark twice per mode on a Ryzen AI Max+ 395 / Radeon 8060S, 128 GB unified RAM with 96 GiB BIOS UMA, and an RTX 5090 32 GB over PCIe 4.0 x4 OCuLink. Ubuntu 26.04, CUDA 13.1 / HIP 7.1; GLM-5.3-Flash Maya-M IQ2_S, context 9216, INT8 KV and automatic CUDA memory sizing. Each run starts from the same expert profile; greedy decoding, thinking off, three answers up to 256 tokens each.

| CUDA+HIP configuration | Decode tok/s | 2K prefill tok/s | 8K prefill tok/s |
|---|---:|---:|---:|
| No speculation, static prefill | 44.95 | 330.5 | 953.0 |
| No speculation, adaptive prefill | 45.25 | 383.0 | 973.0 |
| Built-in MTP, 2 fixed drafts | 64.35 | 323.5 | 920.5 |
| DFlash2, up to 4 drafts | 68.60 | 324.5 | 943.5 |

Static prefill uses a 64-row per-expert cap and a 0.25 helper admission ceiling; adaptive prefill permits up to 128 rows after calibration. MTP and DFlash use static prefill. Sampled device-wide CUDA peaks were 31566–31590 MiB (30.83–30.85 GiB).

These are two-run means for the current CUDA+HIP configuration, not a fresh CUDA+CPU comparison or a general speed guarantee. The existing model pack was retained.
| Run | Decode | 2K prefill | 8K prefill | CUDA slots | Sampled CUDA MiB |
|---|---:|---:|---:|---:|---:|
| none-1 | 44.5 | 330.0 | 948.0 | 2782 | 31566 |
| adaptive-1 | 45.0 | 383.0 | 972.0 | 2782 | 31566 |
| mtp2-1 | 64.3 | 324.0 | 920.0 | 2705 | 31590 |
| dflash4-1 | 68.5 | 324.0 | 944.0 | 2667 | 31588 |
| dflash4-2 | 68.7 | 325.0 | 943.0 | 2667 | 31588 |
| mtp2-2 | 64.4 | 323.0 | 921.0 | 2705 | 31590 |
| adaptive-2 | 45.5 | 383.0 | 974.0 | 2782 | 31566 |
| none-2 | 45.4 | 331.0 | 958.0 | 2782 | 31566 |

Order: static/adaptive/MTP2/DFlash4/DFlash4/MTP2/adaptive/static. Native warmups,
no profiling/tracing. Memory samples every 500 ms; sampled peaks, not exact maxima.
The Ubuntu desktop prefers AMD. Both runtime libraries were observed in inference;
Maya's native report lists CUDA and host-visible RAM, not the external HIP budget.
Physical unified RAM is counted once. All eight runs exited successfully.

## Reproduce the settings


Build this branch's CUDA engine plus the
[separate executor modules](../../docs/EXPERT_PLUGIN.md). Use the same pinned
llama.cpp/ggml dependency as the installer:
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. The measured engine used Release,
`STRATA_ENABLE_CUDA=ON`, `CMAKE_CUDA_ARCHITECTURES=120`, and `STRATA_BUILD_TESTS=OFF`.

Use a Maya CUDA configuration with one primary GPU, the matching executable,
source directory and tokenizer. Common engine arguments:

```text
--glm-pack /absolute/path/to/Maya-M/pack
--max-context 9216 --expert-cache auto --prefill auto
--kv-streaming on --suffix-draft 0
--expert-profile /absolute/path/to/a-fresh-copy-of-profile.bin
```

Omit `--kv` to use v1.1.60's INT8 default. Run
`python maya.py --bench --config /absolute/path/to/config.json` in the normal
Maya Python environment. Preserve `maya-bench.txt` and `maya-bench.log` after each
run; copy the same starting profile for each run. The measured starting profile's
SHA256 is `53b03a13482b6fedc599d149be2d69f88c38d9edc4611a740da46314ab1dc5a7`.
A different profile changes initial residency and may change performance.

The CUDA+HIP configuration uses:

```json
{
  "HIP_VISIBLE_DEVICES": "0",
  "MAYA_GLM_SPLIT": "0",
  "MAYA_GLM_RAM_GB": "0",
  "MAYA_GLM_NO_REUSE": "1",
  "MAYA_GLM_CPU_LANE": "0",
  "MAYA_GLM_PREFILL_CPU": "0",
  "MAYA_GLM_DISK_READERS": "2",
  "MAYA_GLM_EXPERT_PLUGIN": "/absolute/path/to/modules/libmaya_expert_hip.so",
  "MAYA_GLM_EXPERT_PLUGIN_DEVICE": "0",
  "MAYA_GLM_EXPERT_PLUGIN_MIB": "90112",
  "MAYA_GLM_EXPERT_PLUGIN_READ": "1",
  "MAYA_GLM_HELPER_CHECK": "1",
  "MAYA_GLM_EXPERT_PLUGIN_PREFILL_SHARE": "0.25",
  "MAYA_GLM_EXPERT_PLUGIN_PRESTAGE": "32",
  "MAYA_GLM_EXPERT_PLUGIN_PREFILL_MAX_ROWS": "64",
  "MAYA_GLM_PREFILL_PRED_T": "0",
  "MAYA_GLM_HELPER_ZEROCOPY": "1"
}
```

The helper ordinal belongs to the HIP runtime. The example memory budget requires
sufficient actual UMA allocation. Share 0.25 caps admitted rows; it is not a
compute-time fraction. These are the measured settings, not universal defaults.
Stock retains native conversation reuse; the hybrid measurements disabled it.

Select exactly one speculative mode (unset flags that are not listed):

| Mode | Environment additions | Extra engine arguments |
|---|---|---|
| None | `MAYA_GLM_NO_MTP=1`, `MAYA_GLM_NO_SPEC=1` | none |
| MTP2 | `MAYA_GLM_MTP_DECODE=1`, `MAYA_GLM_MTP_DRAFT=2`, `MAYA_GLM_MTP_FIXED=2` | none |
| DFlash4 | `MAYA_GLM_NO_MTP=1` | `--dflash /absolute/path/to/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf --spec 5` |

`--spec 5` is the current token plus at most four draft tokens; confidence filtering
may accept fewer proposals for verification. The DFlash drafter is on CUDA;
target verification uses the selected engine's expert execution path.


Adaptive prefill adds `MAYA_GLM_HELPER_COST=1` and sets
`MAYA_GLM_EXPERT_PLUGIN_PREFILL_MAX_ROWS=128`, retaining the 0.25 share ceiling.
Static and both speculative modes leave cost admission off and use a 64-row cap.

## Artifact checksums


No model weights are included in this contribution. Maya-M GGUF shard SHA256:

- `GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf`: `3f740d1b1022b4a0ddad50cd861bb071dd415bde94afa1292d0476641186c638`
- `GLM-5.3-Flash-Maya-M-IQ2_S-00002-of-00003.gguf`: `285951d2afa0cd98285b03d0dc4aa68d83daf1a6f4a2594fe40b0cdd27ade485`
- `GLM-5.3-Flash-Maya-M-IQ2_S-00003-of-00003.gguf`: `ebf1ce713f71207747e10eeebed87597969d8b5e1d9dd817420d2c2f7ca51e0e`

DFlash: `Anbeeld/GLM-5.3-Flash-DFlash2-GGUF` at
`f3569acb7f9cee95e66d24442d75536eb00ab842`, Q4_K_M, 697017248 bytes,
SHA256 `f4158ffb805b26c4fe439669c2009908a80ea89e1576ecea4d03eaded18f017c`.
It has its own CC BY-NC-ND 4.0 license; no drafter weights are redistributed here.


## Validation and limits

- Ubuntu server: 946 tests (55 skipped), tools: 959 (41 skipped), HIP mappings and
  Release helper dispatch/reader/cost plus upstream CPU fixtures pass.
- Fresh engine and separate CUDA/HIP 7.1 modules build. Source, static libraries,
  modules, actual runtime libraries, original GGUF and existing pack verified.
- All 108 real-weight expert arrays exactly match previous validated modules;
  row/single, mapped/copy, weighted routes, generation and concurrent-call checks pass.
- At 1,025 tokens with FP16 KV and a fixed 22 GiB pool, static token/batch/repeat
  endpoints and adaptive sequential endpoint match v1.1.56 bitwise. Adaptive
  batch/repeat KL 4.20993e-6 / 2.28158e-6, maximum absolute differences 3.40570 / 3.86947, same top token.
  Within-run sequential/prefill and repeated-prefill checks pass. Throughput uses
  native INT8 KV and automatic sizing instead of these numerical-probe controls.
- Unsupported plugin/native-helper-overlap startup is rejected before allocation.
  Native gfx1151/model/budget preflight and explicit architecture mismatch rejection pass.

This is cross-version regression evidence, not stock-versus-hybrid numerical
parity or broad quality qualification. The existing pack was retained, so upstream
packing changes were not exercised by repacking. Other hardware, Windows, EXL3,
DeepSeek, other shapes, long-context quality and the maintainers' full NVIDIA/V100
matrix remain unqualified. Primary/helper hot-expert exchange remains deferred.
Static and optional cost-based prefill admission are included in this contribution.

Measured engine SHA256: `b19833dc4c414aaf148cf15c1baf0ca4d50c180230ec85428c5076361fb58754`.
HIP executor SHA256: `12b58a0ee4b5d8ee918efd21f516055611339e7eb0e366fc2917b598a230acd2`.

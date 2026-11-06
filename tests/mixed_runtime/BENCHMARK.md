# GLM CUDA+HIP helper: Strix Halo / RTX 5090

Measured 2026-11-06 on upstream v1.1.55 (`c4effee67cb95fb4303720919b075fddf04b4e69`).
The stock engine is pristine upstream; the hybrid engine includes this contribution.
Latest main was checked before and after both benchmark sessions.

## Hardware and method

- Ryzen AI Max+ 395 / Radeon 8060S (gfx1151), 128 GB physical unified RAM;
  RTX 5090 32 GB (sm_120) over OCuLink, measured PCIe 4.0 x4 under load.
- Ubuntu 26.04, Linux 7.0.0-42-generic, NVIDIA 595.99.02, CUDA 13.1,
  HIP 7.1.52801. A Release CUDA engine; separately compiled CUDA and HIP executors.
- GLM-5.3-Flash-Maya-M IQ2_S, three GGUF shards (116003563168 bytes),
  native pack with BF16 compatibility projections. Same tokenizer and starting
  expert profile in every run. Verified model hashes are listed below.
- Native `maya.py --bench`: three greedy answers, thinking off, up to 256 tokens
  each, native warm-ups and 2048/8192-token prefill; context 9216, native INT8 KV.
  Two runs per mode, in none / MTP2 / DFlash4 / DFlash4 / MTP2 / none order.
  Prefill input token hashes match between source trees, including the warm-up.
- Stock CUDA+CPU: 4 GiB UMA reservation, 119 GiB host RAM, native automatic RAM
  sizing and CPU scheduling. Hybrid: 96 GiB UMA reservation, about 30 GiB host RAM,
  a 90112 MiB HIP executor budget, CPU expert computation disabled. Count physical
  unified memory once; the UMA and host budgets share the same physical RAM.

This is a practical configuration comparison with different memory reservations
and scheduler settings, not an isolated code-only A/B. Short greedy benchmarks
are not quality or long-context qualification. CPU/GPU arithmetic can change
output tokens; stock-versus-hybrid generated-output equality was not established.

## Throughput

Mean native reported tokens/s over two runs:

| Mode | Stock decode | Hybrid decode | Stock 2K | Hybrid 2K | Stock 8K | Hybrid 8K |
|---|---:|---:|---:|---:|---:|---:|
| No speculation | 30.65 | 45.10 | 213.5 | 332.0 | 771.5 | 969.0 |
| Built-in MTP, fixed 2 drafts | 32.95 | 64.85 | 209.0 | 321.0 | 752.5 | 932.5 |
| DFlash2, up to 4 drafts | 36.80 | 69.05 | 213.5 | 321.5 | 768.5 | 954.0 |

Individual native results, decode / 2K / 8K tokens/s:

| Mode | Stock run 1 | Stock run 2 | Hybrid run 1 | Hybrid run 2 |
|---|---|---|---|---|
| No speculation | 30.6 / 214 / 768 | 30.7 / 213 / 775 | 45.1 / 333 / 969 | 45.1 / 331 / 969 |
| Built-in MTP, fixed 2 drafts | 32.7 / 209 / 752 | 33.2 / 209 / 753 | 64.8 / 321 / 933 | 64.9 / 321 / 932 |
| DFlash2, up to 4 drafts | 36.8 / 213 / 766 | 36.8 / 214 / 771 | 68.9 / 322 / 953 | 69.2 / 321 / 955 |

All 12 runs exited successfully. Stock CUDA peaks were 31751 / 31769 / 31758 MiB
(no-spec / MTP2 / DFlash4); hybrid peaks were 31569 / 31567 / 31575 MiB. These are
device-wide samples every 500 ms, including desktop usage. Native CUDA pool sizing
was automatic in both configurations. Stock used roughly 84–90 GiB pinned expert
RAM, 16 CPU threads and native DFlash verification lanes of 16 and 12 threads.
Measured decode logs show no stock disk reads. Hybrid expert residency was
76.25 / 79.92 / 77.18 GiB on HIP, within the configured helper budget.

Measured-answer acceptance across both runs, excluding warm-up:
stock MTP 952/1148 (82.93%), hybrid MTP 954/1146 (83.25%);
stock DFlash 941/985 (95.53%), hybrid DFlash 968/1012 (95.65%).
DFlash uses its native confidence filter; these are accepted proposals, not an
unconditional drafter-accuracy or model-quality score.

## Reproduce the settings

Build the upstream CUDA engine normally, or this branch's CUDA engine plus the
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

Omit `--kv` to use v1.1.55's INT8 default. Run
`python maya.py --bench --config /absolute/path/to/config.json` in the normal
Maya Python environment. Preserve `maya-bench.txt` and `maya-bench.log` after each
run; copy the same starting profile for each run. The measured starting profile's
SHA256 is `53b03a13482b6fedc599d149be2d69f88c38d9edc4611a740da46314ab1dc5a7`.
A different profile changes initial residency and may change performance.

Stock configuration has only the speculation-selection environment below.
Hybrid adds:

```json
{
  "HIP_VISIBLE_DEVICES": "0",
  "STRATA_GLM_SPLIT": "0",
  "STRATA_GLM_RAM_GB": "0",
  "STRATA_GLM_NO_REUSE": "1",
  "STRATA_GLM_CPU_LANE": "0",
  "STRATA_GLM_PREFILL_CPU": "0",
  "STRATA_GLM_DISK_READERS": "2",
  "STRATA_GLM_EXPERT_PLUGIN": "/absolute/path/to/modules/libmaya_expert_hip.so",
  "STRATA_GLM_EXPERT_PLUGIN_DEVICE": "0",
  "STRATA_GLM_EXPERT_PLUGIN_MIB": "90112",
  "STRATA_GLM_EXPERT_PLUGIN_READ": "1",
  "STRATA_GLM_HELPER_CHECK": "1",
  "STRATA_GLM_EXPERT_PLUGIN_PREFILL_SHARE": "0.25",
  "STRATA_GLM_EXPERT_PLUGIN_PRESTAGE": "32",
  "STRATA_GLM_EXPERT_PLUGIN_PREFILL_MAX_ROWS": "64",
  "STRATA_GLM_PREFILL_PRED_T": "0",
  "STRATA_GLM_HELPER_ZEROCOPY": "1"
}
```

The helper ordinal belongs to the HIP runtime. The example memory budget requires
sufficient actual UMA allocation. Share 0.25 caps admitted rows; it is not a
compute-time fraction. These are the measured settings, not universal defaults.
Stock retains native conversation reuse; the hybrid measurements disabled it.

Select exactly one speculative mode (unset flags that are not listed):

| Mode | Environment additions | Extra engine arguments |
|---|---|---|
| None | `STRATA_GLM_NO_MTP=1`, `STRATA_GLM_NO_SPEC=1` | none |
| MTP2 | `STRATA_GLM_MTP_DECODE=1`, `STRATA_GLM_MTP_DRAFT=2`, `STRATA_GLM_MTP_FIXED=2` | none |
| DFlash4 | `STRATA_GLM_NO_MTP=1` | `--dflash /absolute/path/to/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf --spec 5` |

`--spec 5` is the current token plus at most four draft tokens; confidence filtering
may accept fewer proposals for verification. The DFlash drafter is on CUDA;
target verification uses the selected engine's expert execution path.

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

- Server suite: 738 tests, 33 skipped; tools suite: 730 tests, 37 skipped.
  HIP API mapping and the Release reader-lifetime regression pass.
- Fresh CUDA engine and separate CUDA/HIP executor builds; source, static libraries,
  modules and actual loaded runtime identities verified. Stock process maps show
  CUDA with no HIP plugin. Hybrid maps show both runtime libraries and the executor.
- Real-weight expert fixtures: 144 arrays exactly match the previous validated
  executor modules after rebasing to v1.1.55. Coverage includes mapped/copy and
  row/single execution, concurrent callers and stale-generation/failure checks.
- Previous modified v1.1.53 versus modified v1.1.55: three full-vocabulary FP16
  endpoints match exactly at 1025 tokens with the helper, and three match at
  17 tokens with the plugin disabled. This is a cross-version regression, not a
  pristine-upstream-versus-modified numerical comparison.
- Maya-M and UD-IQ3_XXS INT8 short smoke probes pass. Observed token/batch KL was
  0.01109 and 0.04174 respectively, repeat KL zero; the harness's finite-output /
  KL < 0.1 smoke threshold is not broad quality qualification.

The adapter supports the documented native-GGUF GLM expert shape on Linux only.
DeepSeek, EXL3, Windows, other model shapes, long-context quality, other hardware,
and the maintainers' full NVIDIA/V100 validation matrix remain unqualified.
Primary/helper hot-expert exchange and adaptive prefill balancing are outside this
contribution. The external executor is opt-in; shared reader-lifetime and error
handling changes also affect the existing path and are covered by the host fixture.

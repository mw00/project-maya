# DeepSeek V4.1 on four RTX 4090s: bounded configuration selection

Measured on **2026-11-02 (Asia/Shanghai)**, upstream v1.1.46 commit `3589d6f12a3f77e813b184b51a586f8fc3de2185`, with two local CPU startup patches. This is **not a stock v1.1.46 score**. This documentation-only contribution records the selected configuration, not the highest individual answer.

## Official benchmark result

- Decode: **85.3 tokens/s**, the official arithmetic mean of its eligible answers.
- Prefill: **642 tokens/s at 2,048 tokens**, **1585 tokens/s at 8,192 tokens**.
- Selected arm **B**: chunk **5120**, INDEX_TC **off**, DSpark on, context **262,144**, serving capacity **8**. Requests were serial; this is not eight-client throughput.
- The unmodified v1.1.46 `maya.bench` function was called once, with the frozen upstream documentation corpus and tokenizer: one 64-token warm-up, three greedy 256-token answers (thinking off), one prefill warm-up and the 2K/8K prefill measurements. Exactly **7 native GEN commands**, each guarded before sending. The wrapper isolates output paths and records native token IDs; it does not change topics or timing calculations.
- [Official output](maya-bench.txt) includes all three answers, both prefill lengths and the engine summary. [Machine-readable results](results.json) retain the individual native replies and screening samples.

## Machine and model

Four RTX 4090 24 GB cards, PCIe 4.0 x16 maximum link width each, power limit 285 W each; NVIDIA driver 595.99.02. Two Xeon Platinum 8380 CPUs (80 physical cores total, 160 logical CPUs), 251.3 GiB host-visible RAM, 245 GiB container limit, no CPU quota. Ubuntu 24.04.4 LTS, kernel 6.8.0-100-generic, CUDA 13 build targeting sm89. CPU expert pool uses 80 threads. The existing CUDA container has `SYS_NICE` with its default seccomp policy, allowing the NUMA placement operation. Model storage is an ext4/LVM volume; the OS exposes Dell PERC/BOSS logical devices, so no specific underlying SSD model or media-speed claim is made. No CPU governor or GPU power settings were changed for this run. The raw benchmark timestamp uses the container's UTC clock; the date above is Asia/Shanghai.

Model label: `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS` (mixed GGUF CPU expert types IQ1_M/IQ2_XXS/IQ2_XS), Engram tables pinned in RAM. This report does not infer a Maya tier from that filename. KV format FP4; vision disabled. Requested `--tp 2` resolves to four singleton groups (`tp_n=1`), with layers 0–8 / 9–19 / 20–29 / 30–39. It is not a true TP2 result.

## Configuration screening (separate from the official benchmark)

Nine requests total, no screening warm-ups or retries: A/B each 8K, 32K, 128K in that order; C only 128K; D only 8K, 32K. Each generated 256 tokens. All used the same binary, CPU fixes, tokenizer and matching frozen input at each length. Prompt cache reuse was zero in every eligible sample. Automatic expert-slot differences are recorded, not hidden by restarting until counts match. **C's 128K request is its first, whereas B's is its third; expert-cache history is not matched. Therefore C/B, especially Decode, cannot isolate the INDEX_TC effect.** This bounded screen does not supersede a controlled same-history experiment.

| Arm | Chunk | INDEX_TC | Expert profile |
| --- | ---: | --- | --- |
| A | 8192 | off | frozen baseline |
| B | 5120 | off | same baseline |
| C | 5120 | on | same baseline |
| D | 5120 | off | previously prepared frequency-based trunk, current baseline MTP tail unchanged |

| Arm | Input tokens | Prefill tok/s | Decode tok/s | First effective output, s | Draft accepted/proposed | GPU expert slots |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| A | 8,192 | 1604.0 | 57.4 | 5.164 | 91/134 | 7338 |
| A | 32,768 | 3677.0 | 66.2 | 9.009 | 88/107 | 7338 |
| A | 131,072 | 4862.9 | 73.7 | 27.309 | 94/107 | 7338 |
| B | 8,192 | 2258.7 | 58.3 | 3.686 | 86/106 | 7657 |
| B | 32,768 | 4314.4 | 71.4 | 7.698 | 96/117 | 7657 |
| B | 131,072 | 5019.5 | 68.6 | 26.428 | 85/108 | 7657 |
| C | 131,072 | 5621.3 | 57.7 | 23.656 | 84/113 | 7657 |
| D | 8,192 | 2286.7 | 58.1 | 3.637 | 94/110 | 7646 |
| D | 32,768 | 4189.2 | 66.8 | 7.920 | 90/115 | 7646 |

| Comparison | Input | Prefill change | Decode change | Same output text SHA256 |
| --- | --- | ---: | ---: | --- |
| B/A | 8k_r1 | +40.82% | +1.57% | False |
| B/A | 32k_r1 | +17.33% | +7.85% | False |
| B/A | 128k_r1 | +3.22% | -6.92% | False |
| C/B | 128k_r1 | +11.99% | -15.89% | False |
| D/B | 8k_r1 | +1.24% | -0.34% | False |
| D/B | 32k_r1 | -2.90% | -6.44% | False |

Selection: **B is a prefill-oriented choice: 8K/32K prefill +40.82%/+17.33%, with 128K prefill +3.22% but decode -6.92% in one 256-token answer. The advisory all-cells 5% non-regression rule suggests A and was NOT passed; B is retained as an explicit latency trade-off, not a universal winner. D showed no useful gain and was rejected; INDEX_TC stays off. No repeated screening.** These are single paired inputs, not population estimates or statistical significance claims. There is no A/A drift control. Automatic capacities, CPU/GPU routing, output sequences and actual NUMA locality can vary; the raw startup locality counts are retained in `results.json`. These are configuration-level observations, not isolated kernel causal estimates. Successful marker retrieval is only a narrow functional check. No broad quality-equivalence claim is made. Percentages from different comparisons are not added. The 128K screening scores are not substituted into the README's 8K column.

## Local CPU changes and reproduction boundary

Both CPU changes are off by default in the attached [reproduction patch](cpu-startup-reproduction.patch), which is supplied as an artifact, not applied to the upstream source by this PR:

1. `DS4_ENGRAM_RAM_NUMA=1`: distribute the existing shared Engram file pages across NUMA nodes before allocating CPU expert arenas, with strict placement checks. File bytes are unchanged; no full-table shadow copy or global cache flush.
2. `DS4_CPU_PREPARE_ROWS=1`: prepare the original per-format one/multi-token guarded CPU row checks during loading, including the later IQ1_S native-multi fallback. Numerical checks and fallback decisions are retained.

Binary SHA256: `6a0da4cbf3098e99b8865c05d0d681012ff87799b8bdce3de5aa4c9c56d028c2`. Patch SHA256: `bc64a0d3bfda2466ce0e623a295b8c00ac0db3261b6b2097f8ff9d6273ba8aa7`. The patch applies to the stated upstream commit; the actual build used CUDA 13, Release/O3, C++20, native experts enabled, sm89. Apply it only in a disposable source checkout before rebuilding `maya-ds4`.

The earlier CPU deployment verified physical placement and narrowly targeted row behavior. It did **not** establish an isolated end-to-end CPU speedup. Shared file-page placement survives process exits, so restarting stock is not a valid physical A2 reset. This report does not relabel those placement changes as throughput gains.

Normalized engine options (paths are placeholders):

```sh
DS4_ENGRAM_RAM=1 DS4_REQUIRE_ALL_RAM=0 \
  STRATA_DS4_SLOT_KEEP=1 STRATA_DS4_SLOT_CAP=16 STRATA_DS4_SLOT_RAM_GB=24 \
  STRATA_DS4_ENGRAM_RAM_NUMA=1 STRATA_DS4_CPU_PREPARE_ROWS=1 \
  STRATA_DS4_SLOT_DIR=/isolated/slots /patched/maya-ds4 --serve \
  --pack /model/pack --gpus 0,1,2,3 --max-context 262144 --tp 2 \
  --prefill-chunk 5120 --reserve-mib 1536 --mmap-experts auto \
  --dspark on --expert-profile /isolated/seed.bin --profile-readonly \
  --profile-out /isolated/run.bin --vision off --parallel 8 --layer-split auto
```

INDEX_TC aliases were unset/off for the selected run. Use `./maya.sh --bench` with an equivalent isolated runtime and the stated source revision; all serving processes must be stopped first. Exact route replay additionally requires the measured profile seed, identified in `results.json`; seeds are not distributed here. Hardware, learned profiles and automatic placement can change the result.

No GPU kernel, TP2 implementation, quantization or speculative policy change is proposed by this documentation PR. No broad hardware matrix or extra performance reruns were performed: **9 screening + 7 official generations**, followed by service/collector recovery without another generation.

# GLM-5.3-Flash Maya-M + Maya-L on 4x RTX 4090: standard `--bench` on a downstream v1.1.74 build (115.9 / 81.4 tok/s)

2026-11-24. Host: 2x Xeon Platinum 8380 (80C/160T, two NUMA nodes), 251 GB RAM, 4x RTX 4090 24 GB at a 300 W
power limit each, Ubuntu 24.04. Engine: v1.1.74 with the R-20 host-side patch set (64-thread shared CPU-expert
pool, `ROW_HALVES=1`, AVX-512 single-thread expert kernel, targeted retime; binary SHA-256
`44546c9f3176ca67325f5e472d0c3f2cc52a0b0bc3e3338ac514fe27c145f27b`). Same layered config for both models:
`--layer-split 12,24,35`, `--prefill 4096`, `--parallel 4`, experts RAM-resident (160 GB budget), DFlash2
drafting (`--spec 8`), KV streaming on.

Method: the unmodified official `maya.py --bench` (three decode questions + 2K / 8K prefill, greedy, thinking
off). One run per model, back to back, from the same binary and host tuning; no retries and no parameter sweep
between them.

| official measurement | Maya-M (IQ2_S, 116 GB) | Maya-L (IQ3_S) |
| --- | ---: | ---: |
| Decode: Python CSV function | 113.5 tok/s | 81.4 tok/s |
| Decode: photosynthesis | 110.0 tok/s | 76.4 tok/s |
| Decode: HTML canvas animation | 124.1 tok/s | 86.4 tok/s |
| **Decode arithmetic mean** | **115.9 tok/s** | **81.4 tok/s** |
| Prefill, 2048 tokens | **1100 tok/s** | **806 tok/s** |
| Prefill, 8192 tokens | **2185 tok/s** | **1678 tok/s** |

Same-host history for Maya-L through the same official benchmark (earlier v1.1.x configs, 255 W era):

| version / config | decode mean | 2K prefill | 8K prefill |
| --- | ---: | ---: | ---: |
| v1.1.57, 96 CPU threads | 60.6 tok/s | 654 tok/s | 1458 tok/s |
| v1.1.58, 144-thread NUMA + DFlash2 | 69.5 tok/s | 647 tok/s | 1385 tok/s |
| **v1.1.74 R-20 @ 300 W (this run)** | **81.4 tok/s (+17.1% vs v1.1.58)** | **806 tok/s (+24.6%)** | **1678 tok/s (+21.2%)** |

The 115.9 and 81.4 tokens/s are short, serial measurements, one run per pack. The earlier v1.1.57
entry used 96 workers and v1.1.58 used 144. Engine version, CPU/NUMA tuning, prefill chunk and GPU
power also differ across historical entries. The same-binary M/L comparison above holds the settings
fixed. The individual R-20 changes and answer quality were not isolated by this benchmark.

A separate sustained-load check on this chassis alternated cold 32K prefill and single-stream decode.
The 300 W-per-GPU run reached our 82 C software guard on GPU0 after about five minutes. An already-warm
300 W repeat reached it in about one minute. A five-minute warm 255 W run completed, peaking at 78 C.
The short M/L results describe the benchmark window. The [community report](https://github.com/mw00/project-maya/discussions/1124#discussioncomment-19835844)
contains the related deployment measurements.

Evidence (SHA-256 of the raw `maya-bench.txt` the engine wrote, kept with our operator receipts):

- Maya-M: `1784073f3f0b29736cb3e3926fe0294dd0384746f1aaa406eff25c8beaaf6804`
- Maya-L: `6d507a3ea58d292a4ca861ad3d35ca7eb939064c3bd1745731ae61f8e4c7adc5`

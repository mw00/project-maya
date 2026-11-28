# Inkling 2 Mini Q3/Q4/Q8 on RTX 5060 Ti 16 GB, Windows

2026-11-23, Asia/Shanghai. On this host and build, UD-Q4_K_XL is fastest at
46.0 tok/s, followed by UD-Q3_K_XL at 34.2 and UD-Q8_K_XL at 12.6.
All three original GGUFs load and complete the benchmark after the local
quant-format support fix. These measurements include each quant's automatic
execution path; they do not isolate the cost of its quantization kernels.

## Host and build

- RTX 5060 Ti 16 GB, driver 610.88; PCIe Gen 5 x8 observed under load.
- Core i7-12700, 12 cores / 20 threads, AVX2; 127.7 GiB reported RAM.
- Windows 11 build 26200; CUDA 12.8.93; MSVC toolset 14.42.34433; Python 3.12.7.
- Release build for sm_120. Source base `fd601f2988aaacc0d3e39feec41a765912810a9c`
  (v1.1.74) plus the separately published
  [measurement patch](https://github.com/Thymos2027/maya/commit/3e5876424b0f51b0b4358cc80cdf2876103f8cdc).
  The patch's source matches the build fingerprint below; this is a modified build,
  not a claim that stock v1.1.74 loads all three files.
- Engine SHA-256: `c649efac6242daa9d2a19ef0831ef22c605e88422be9e01adf3e98144602be75`.
- Build source fingerprint: `22a28636046c86d7fd3d001a45c79346e855b8387d7a067cce563271621d66ed`.

The unmodified `maya.bench` / `maya.report` harness runs from source
`2e475762615bc5d14b62def8fbfbc7cffaa7a005` against that separate fixed executable.
The harness commit is a local merge; its `maya.py`, frontend, tokenizer and chat
template match the engine base byte for byte. Its server module additionally has
opt-in request tracing, outside the directly invoked native benchmark path.
The hashes above identify the measured engine; the frozen native requests below
preserve the exact inputs without needing that local checkout.

## Related work and contribution

This report adds measurements for one RTX 5060 Ti 16 GB on Windows with the
original Q3/Q4/Q8 files, three fresh runs each, resource samples, and execution
path evidence. It does not claim a new quantization algorithm or first Windows support.

- [#1159](https://github.com/mw00/project-maya/issues/1159) already reports Windows
  Inkling Mini on **two** RTX 5060 Ti 16 GB cards, Q4, including two-GPU bugs.
- [#1133](https://github.com/mw00/project-maya/pull/1133), open when checked on
  2026-11-23, already adds Q3-related block readers and the expert row-width fix,
  plus broader offload work, with measurements on an RTX 3060 Ti 8 GB. Our
  measurement patch overlaps those readers and the alignment correction.
  These numbers measure our linked patch, not that PR's implementation or tuning.
- [#1163](https://github.com/mw00/project-maya/issues/1163) and
  [#1116](https://github.com/mw00/project-maya/issues/1116) provide additional
  AMD Q4 reports; #1116 uses a modified checkpoint, unlike the original files here.

The Q8 arm additionally exercises BF16 matrices and experts in the target model.
Our repository searches found no matching one-card Q3/Q4/Q8 comparison, but are
not proof that none exists. The useful addition is the independent dataset and
documented path differences. This report proposes no engine changes for merging.

## Method and results

One GPU, one conversation, 32,768-token context, FP8 E4M3 KV cache,
`--chunk 1024 --threads 8 --drafting off`. Temperature 0, thinking off with
the answer text segment opened as the official benchmark does. No drafter or
vision encoder is loaded. The Q4 CPU lane automatically uses 10 threads;
the engine's `--threads 8` argument does not set that separate pool.

Each quant gets three fresh processes. Each process runs a decode warm-up of
up to 64 tokens, three answers of up to 256 tokens (CSV Python function,
photosynthesis explanation, bouncing-ball HTML), a 2K prefill warm-up, then
measured 2K and 8K prefills. Photosynthesis ends naturally after 151 / 153 / 149
tokens for Q3 / Q4 / Q8; the other two measured answers reach 256 tokens.
Each round's decode figure is the arithmetic mean of the three answer rates,
as reported by `maya.bench`; the aggregate is the median of those three round means.
Every round starts with the same expert profile and a separate identical copy
of the existing Q4 offload calibration. Run order is Q3, Q4, Q8, three rounds each.

| Original mixed-format GGUF | Round means, tok/s | Median decode, tok/s | 2K prefill, tok/s | 8K prefill, tok/s |
| --- | --- | ---: | ---: | ---: |
| UD-Q3_K_XL | 34.3 / 34.2 / 33.4 | 34.2 | 966 | 978 |
| UD-Q4_K_XL | 46.0 / 45.1 / 47.2 | 46.0 | 1168 | 1196 |
| UD-Q8_K_XL | 12.6 / 12.8 / 12.5 | 12.6 | 940 | 964 |

Prefill columns are also three-round medians. The full 32K cache is reserved,
but the longest measured prompt is 8K; these results do not test a 32K prompt.

| Quant | Peak sampled process working set, GiB | Peak sampled private commit, GiB | Peak sampled total device use, MiB |
| --- | ---: | ---: | ---: |
| Q3 | 13.711 | 15.666 | 15721 |
| Q4 | 17.228 | 19.399 | 15522 |
| Q8 | 21.705 | 24.025 | 15672 |

RAM includes mapped model files and other process allocations. It is not the
minimum RAM requirement or the size of the expert offload tier. GPU readings
cover the whole device, including the Windows desktop; monitoring samples at
roughly one-second intervals and can miss short peaks.

## Paths observed in the engine logs

- **Q3:** all 3,072 experts fit in VRAM (11.4 GiB of expert blocks). The log says
  `resident path off: a dense matrix is not kept as blocks`, and `INFO ... path=cache`.
  It uses the general cache path despite the fully resident experts.
- **Q4:** 2,371 experts cached in VRAM (10.7 GiB); approximately 4.0 GiB of expert
  blocks pinned in host memory. `decode mode: lane parity`, with AVX2 CPU expert
  rows and GPU fetches on cache misses. `INFO ... path=offload`.
- **Q8:** 2,144 experts cached in VRAM (9.7 GiB); approximately 8.6 GiB of expert
  blocks pinned in host memory. The BF16 experts cannot use the existing exact
  CPU lane: `lane parity: off - an expert format the exact lane does not have`.
  It uses GPU fetches, `INFO ... path=offload`.

These observed path differences are plausible contributors to the speed gap;
no ablation here attributes a percentage of the gap to any one of them.
The quant names describe mixed tensor formats, not uniformly 3-, 4- or 8-bit weights.

## Validation and limits

The build, sampling checks, existing `ik2_kernels_test` and `ik2_off_test`, and
the added `ik2_quant_formats_test` pass. The new test compares IQ2_XS, IQ2_S,
IQ3_S, IQ4_NL, Q3_K and Q4_K decoding against the pinned ggml CPU decoders,
checks BF16 conversion, and exercises dense and expert products on the GPU.

All three real models answer the arithmetic smoke question with `42` and stop
normally. Original and fixed Q4 return identical token IDs for that short
question: `[68872, 200010, 200006]`. All 63 benchmark requests finish with `DONE`;
each quant repeats its generated token stream across its three rounds.
This does not establish general output parity or relative model quality.

An earlier, unmodified v1.1.74 Q4 run on the same machine measured round means
47.6 / 47.8 / 45.8 tok/s, median 47.6. The fixed build's median is about 3.4%
lower. The sessions were not interleaved A/B tests, so this difference cannot
be attributed to the patch or dismissed as noise from these measurements alone.

## Files and reproduction

[summary.json](summary.json) records model sizes and SHA-256 values, build
identity, per-round measurements, token-stream hashes and resource observations.
The Q3 and Q8 files were verified after download from ModelScope at revision
`9fb85380e4af45c3599213139ec551d217eccc13`; the existing Q4 file was also size/SHA-256
verified. The original files were used without re-quantization.

Each of [Q3](Q3), [Q4](Q4), [Q8](Q8) contains all three official benchmark texts,
structured request timings, and engine startup / `STAT` / `DONE` excerpts.
Startup and statistics are on protocol stdout, so the official stderr-only
`maya-bench.log` is empty; the `round-*-engine.txt` files preserve those lines.
Full local raw artifacts are retained under run ID
`ik2-quant-fix-20261123-093932`, including configs, request IDs, protocol output,
per-second telemetry and official system reports. The full prompt corpus is the
harness checkout's README, docs and benchmark reports, as `maya.bench` selects it.
[requests.jsonl](requests.jsonl) freezes all seven native `GEN` commands in order;
every measured process used these same commands. [calibration.txt](calibration.txt)
is the initial offload calibration copied for each round. The two artifact
checksums normalize CRLF to LF, so Git line-ending conversion does not change them.

To repeat, check out the linked measurement patch commit and build for sm_120 with
CUDA 12.8, native experts and tests enabled; run the three GPU tests.
Set each model config's `exe` to that built
`maya-ik2`, set the model and tokenizer paths, and use the context, flags and
environment above (`IK2_KV=fp8`, `CUDA_VISIBLE_DEVICES=0`,
`IK2_WARM=data/expert-profile-ik2mini.bin`, a separate copy of `calibration.txt` as
`IK2_CALIB` per round). For exact input replay, launch `maya-ik2 --serve` with
those flags and environment, wait for `READY`, send each saved `command`, wait
for its `DONE` before sending the next, then send `QUIT`. Parse the same native
`DONE` timing fields as `maya.bench` does. The JSONL records contain native
protocol commands, not HTTP requests.

For a new standard benchmark run, use
`python maya.py --bench --config <config.json>` followed by
`python maya.py --report --config <config.json>`, saving outputs before the next
round. That harness selects the checkout's current docs as its prefill corpus;
adding this report changes those inputs, so it will not replay the frozen
prefill tokens unless the original corpus is used.

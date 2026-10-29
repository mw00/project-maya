# DeepSeek prefill transfers: only the experts a chunk picks

CUDA: on for chunks of up to `DS4_SKIP_UNPICKED_ROWS` rows (1536) - an agent's turn, a short prompt. HIP: off unless
`DS4_SKIP_UNPICKED=1`, as before. `DS4_SKIP_UNPICKED=1` takes every chunk, `0` none, on either. This affects the normal
chunked prefill streamer, not `DS4_FLOW_TP` or decode. `DS4_FLOW_STATS=1` reports copied/skipped expert counts.

Why by size on CUDA: a chunk that skips waits for each layer's routing before it copies that layer's experts, where
otherwise the copies run ahead across layers. A short chunk picks few of the CPU lane's experts, so its streams are that
much shorter; a long one picks most, and the waits cost more than the few skipped save. An RTX 3090 + eight RTX 5060 Ti
(PCIe 3.0 x8 each), EXL3 3.5 bpw, two thirds of the experts on the CPU lane, an agent's turns at 118-134K tokens of
context (one chunk each):

| New tokens | Every expert streamed | Only the picked ones |
| --- | --- | --- |
| 530 | 4.6-4.9 s | 3.6-3.8 s |
| 1,040 | 5.0-5.2 s | 4.4-4.9 s |
| 2,050 | 6.0-6.3 s | 6.7 s |
| 117,670 (29 chunks of 4,096) | 1,515 tok/s | 1,282 tok/s |

Parity on that setup: a 2,048-token prompt in one chunk (`--no-expert-cache --window 2048 --dump-logits`), every row's
logits (2,048 x 129,280) the same bit for bit with it and without it.

Wait for each layer/chunk routing result and skip zero-row experts without compacting virtual slots or changing batches, kernels or accumulation order. Transfers still overlap computation within each layer. With explicit RAM copies, stage only the TP member's gate/up rows into existing blob offsets; keep full down matrices for strided DMA. File-backed reads retain their existing layout. Deferring cross-layer prefetch may hurt workloads with many active experts.

## Measured on the earlier base

Base: `abc09c74ba692d808062dfa92d2b7c081831ac03`. Ryzen 9950X, 256 GB RAM, two Radeon AI PRO R9700 32 GB cards (PCIe 5 x8 upstream each), one 16 GB RX 9070-family card behind PCIe 4 x4, ROCm 7.14.1. DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL, context 262144, TP2 plus singleton, `DS4_SPLIT=36,4`, `--reserve-mib 2048 --mmap-experts off --dspark none`. Both arms use identical hardware/split. `DS4_FLOW_TP` and `DS4_PROF` off.

Fresh synthetic API prompts, one output token, temperature zero, reasoning none, every response `cache_n=0`:

| Prompt tokens | Baseline seconds | Enabled seconds | Speedup |
| --- | --- | --- | --- |
| 1043 A | 42.293 | 17.988 | 2.35x |
| 1043 B | 40.671 | 16.698 | 2.44x |
| 2067 C | 42.167 | 19.558 | 2.16x |

Individual measurements, not confidence intervals or a general performance claim. Routing diversity and cache warmness matter. Other quants, NUMA hardware and CUDA have not been measured.

## Correctness

Original and enabled binaries on that base consumed identical 512 input IDs and separate copies of a frozen initial expert profile, identical weights and settings. CLI: `--ids <comma-separated IDs> --window 512 --n 0 --dump-logits <output.f32>`. All 512 x 129280 float32 logits were bit-identical, maximum absolute difference zero. An independent original repeat was also bit-identical. Copy a frozen profile for each arm; do not reuse the profile modified by the preceding run.

The current-main port still needs a full-model parity/performance rerun before merge. The earlier numbers do not establish parity or performance on the newer upstream engine.

CPU regression coverage: random byte payloads, TP widths 1/2/4, RAM parts 1/2, gate/up and full-down slices, bounds and destination guards. Run:

```sh
g++ -std=c++20 -O2 -DNDEBUG -Wall -Wextra -fsanitize=address,undefined tests/ds4_ram_slice_test.cpp -o /tmp/ds4_ram_slice_test
/tmp/ds4_ram_slice_test
```

Also registered as `ds4_ram_slice_test` with `STRATA_BUILD_TESTS=ON`. Byte-slicing checks do not replace full-model parity.

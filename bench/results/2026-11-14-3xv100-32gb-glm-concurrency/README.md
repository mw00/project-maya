# GLM-5.3-Flash Maya-S v2 on 3x V100 32 GB, fully resident: plain `--parallel 4` is the sweet spot (98 tok/s in all, 1.61x), the rows family halves it, and prefill does not scale with streams

2026-11-14. Host: 2x Xeon E5-2696 v3 (36 cores / 72 threads, two NUMA nodes), 3x Tesla V100-PCIE-32GB (TCC,
PCIe 3.0 x16) + a GTX 1060 6 GB display card, 128 GB DDR3L-1600 ECC REG, Windows 10 Enterprise LTSC 2021.
One NVMe holds the weights. Maya-S-v2-IQ2_XXS (`--glm-pack`, the pack's dense.bin + expert blobs), engine
v1.1.63 (source build, sm_70, CUDA 12.4). Layer split (auto): [0, 17) CUDA0, [17, 32) CUDA1, [32, 45) CUDA2.
Everything is in VRAM at run time (98% resident by design; `glm stat` vram hit 99.5%, disk 0.00 reads/tok) —
the opposite memory regime of the 2026-11-02 write-up, whose 16 GB cards tiered experts to RAM.

Run config: `--glm-pack <pack> --max-context 262144 --prefill auto --parallel N`, env
`STRATA_GLM_PCIE_SHARE=0.25`, `STRATA_GLM_CPU_LANE=18`, `STRATA_GLM_CPU_CAL=<table>`; `--kv int8` is not set
(the engine's own default serves it), served through `serve/server.py --engine maya --port 8080`.

## Solo bands (1 stream)

Client: one `/v1/chat/completions` per round, 3 rounds per band, server-side timings from the engine's own
`glm prefill:` / `glm stat:` lines.

| band | value |
| --- | ---: |
| short decode (~44-token prompt, 320 tokens out) | 59.5-63.3 tok/s (median 60.9) |
| long prefill (12.1-12.2k-token prompt) | 1045-1053 tok/s (median 1049) |
| long decode (200 tokens out) | 46.6 tok/s |
| drafts accepted | 80-87% per request |

## Concurrent decode (N streams, decode-bound)

Client: N concurrent `/v1/chat/completions` requests, `temperature: 0`, prompt "Count from 1 to 1200,
separated by commas" (~15 tokens — prefill noise), max_tokens 2048, one request per stream in flight;
40-60 s window, then every in-flight request runs to its cap; aggregate = completion tokens / elapsed.
Every request hit the 2048 cap, so the whole window is decode.

| arm | streams | aggregate tok/s | a stream |
| --- | ---: | ---: | ---: |
| solo | 1 | 60.9 | 60.9 |
| **plain `--parallel 4`** | **4** | **98.0** (elapsed 83.6 s, 4 x 2048) | **24.5** |
| plain `--parallel 8` | 8 | 90.6 (elapsed 181 s, 8 x 2048) | 11.3 |
| `MAYA_GLM_SPLIT_ROWS=4` on `--parallel 4` | 4 | 51.0 | 12.8 |

`--parallel 4` is the sweet spot (1.61x solo); 8 streams fall back (the pipeline's hand-offs and the one
CPU lane spread over more conversations than they can pace). `MAYA_GLM_SPLIT_ROWS=4` halves the aggregate
on this shape. The 2026-11-02 write-up already saw the rows family slower on its 16 GB cards
(SPLIT_ROWS=4: 8.3-8.6 vs 20-21 there); this is the same conclusion on the opposite memory regime —
fully resident, no RAM tier, no small card in the split — so the rows engine's wins stay confined to
shapes where experts would otherwise stream over PCIe. `MAYA_GLM_ROWS_ATTN` / `MAYA_GLM_MULTI_MTP` /
`MAYA_GLM_PREFILL_INTERLEAVE` all build on `MAYA_GLM_SPLIT_ROWS` and were not measured further for that
reason.

## Prefill does not scale with streams

Four concurrent 12.1-12.2k-token prompts (client as above but prompt = 3400 pseudo-words + "Summarise...",
max_tokens 48):

| | a request's prefill | request wall | aggregate |
| --- | ---: | ---: | ---: |
| 1 stream | 1045-1053 tok/s | ~13 s | 1049 tok/s |
| 4 streams | 1040-1045 tok/s each | **51 s each** | **~1043 tok/s (no gain)** |

Each concurrent request's own prefill runs at full solo speed, but the requests' prefills are serialized —
three of the four wait ~37 s for their turn (wall 51 s = 11.6 s prefill + the queue + ~2 s of decode). The
engine's per-card `glm prefill:` chunk lines confirm one request's chunks own the pipeline at a time. So on
this shape the concurrency payoff is decode-only; a queue of large prompts prefills at the same aggregate
rate no matter how many clients ask at once.

## Client and files

The measuring client and the raw JSON (per-stream tokens, per-request rows) ship with this folder's
companion scripts; the run configs are `maya-maya-s-v2-iq2_xxs-p4.json` / `-p8.json` (the base config plus
`--parallel N`). Greedy client (temperature 0) for the concurrency table; the solo bands used the config's
sampling (temperature 1.0, top_p 0.95). Nothing lossy is turned on.

# Qwen3.8-Flash-Next on Strata's current engine (opt-in)

Maya's `maya` engine, which runs Qwen3.8-Flash-Next, is a fork of Strata's engine taken at **engine 0.1.27**
(2026-09-29). Strata's engine has since reached **0.1.67** (2026-11-24) with, among others, the sm_86 exact speed
switches, the profile-ranked expert-cache fill, the device-side hit graph, prompt chunks up to 32K on NVIDIA and the
DFlash2 drafter. Both engines still speak the same serve protocol (READY / GEN / T / PP / DONE / STOP / QUIT), so Maya's
server, web app and APIs drive the newer engine as they are.

`tools/strata_qwen.py` builds Strata's engine at a pinned commit with one small patch for Maya's Monitor, and writes a
Maya config that uses it. Nothing else changes; the existing config and `build/maya` stay as they were.

```sh
# NVIDIA, Linux; an existing Qwen3.8-Flash-Next config from ./maya.sh --setup
.venv/bin/python tools/strata_qwen.py --from maya-qwen-ud-iq2_xs.json --out maya-qwen-strata.json
./maya.sh --config maya-qwen-strata.json

# optional: draft with a DFlash2 pack instead of the MTP layer (faster on the PC below; Strata's own notes find it
# slower than the MTP on 12 GB NVIDIA cards, so it is not the default)
.venv/bin/python tools/strata_qwen.py --from maya-qwen-ud-iq2_xs.json --out maya-qwen-strata.json \
    --dflash2 /path/to/qwen3.8-flash-next-dflash2.q4_0-selq4.pack
```

The tool fetches `engine/` of https://github.com/Niko1221/Strata at commit `d46f5f26` (MIT, as Maya) into
`third_party/strata`, checks that the checkout is that commit, applies `tools/strata_qwen/maya-metrics.patch` and builds
with Strata's release flags for the CUDA archs in `build/MAYA-BUILD.json` (or `--arch 86,89`). The engine lands in
`build/strata-qwen/strata` with a `BUILD.json` beside it (about 3 minutes on 12 cores). `--no-build` only writes the
config. A Maya update leaves it alone; if a rebuild clears `build/`, run the tool again. `tools/test_strata_qwen.py`
checks the config writing without a GPU.

## What the patch adds

Strata's engine writes the INFO keys of its own monitor; Maya's Monitor reads three more kinds of lines, which the patch
adds in the formats Maya's own engines write them (76 lines in `src/program/generate.cpp`, nothing else touched):

- `INFO engine_kind=qwen experts= vram_slots= vram_gb= ram_slots= ram_gb= mtp= dflash= vision_lend=0`: the "Where the
  experts live" card and the Speculation line (MTP / DFlash2 / Lookup).
- `VRAM dev=0 ... weights= kv= kv_ram= act= drafter= window= pool= borrow= slots= ctx=`: the "What fills the GPU" map.
  The parts are measured: the free VRAM at each of the engine's own load steps (its `mem_mark`, which now always records
  until the serve start), the expert pool from the cache itself. Written on one GPU; a layer split writes no VRAM line
  (the Monitor then shows the plain bar, as before).
- `STAT tok_s= ms_tok= vram_hit= ram_fetch= disk= promo= vram_used= vram_slots= ...` every 8 tokens of the serial decode
  loop, as `maya`'s Qwen path writes it.

## Measured

One PC: RTX 3090 24 GB (PCIe 4.0 x16, 300 W), Ryzen 9 5900X, 94 GB DDR4, Ubuntu 22.04, CUDA 13.0, driver 595.91.
Qwen3.8-Flash-Next GSQ-RCO IQ3_S (the same two GGUF files and pack in every row), 262,144 context, int8 KV,
`--kv-resident 32768`, `--spec 4 --spec-min-p 0.5`, `--prefill auto`, the expert profile, vision on. Strata's
`strata bench run --suite standard-v1` (thinking on, temperature 1.0, 1 warm-up, 3 repetitions per case, 12 sanity
checks), sent to Maya's server. Medians; decode is the engine's own timing.

| Engine and drafter | Decode short | Decode 4K | Decode 16K | Prompt 4K | Prompt 16K | Sanity |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `build/maya` (engine 0.1.27 fork), MTP, 2026-11-24 | 73.6 | 75.9 | 73.7 | 1,892 | 2,273 | 12/12 |
| Strata 0.1.67 in Strata's own server, MTP, 2026-11-24 | 96.3 | 85.6 | 93.6 | 2,143 | 2,877 | 12/12 |
| Strata 0.1.67 in Maya's server, MTP (2 runs) | 102.8, 107.3 | 94.2, 96.7 | 100.7, 99.8 | 2,205, 2,203 | 2,875, 2,858 | 12/12 |
| Strata 0.1.67 + this patch in Maya's server, DFlash2 (4 runs) | 106.8, 111.0, 102.7, 107.1 | 101.9, 100.8, 97.1, 104.6 | 106.3, 97.9, 99.2, 96.1 | 2,254, 2,245, 2,260, 2,262 | 2,881, 2,905, 2,889, 2,886 | 12/12 |

The run-to-run spread is about 5% (sampling at temperature 1.0 changes the drafts' acceptance). In Maya's server the
newer engine gets a bigger expert cache than in Strata's own (8,586 against 6,892 slots) because Strata's launcher here
also ran `--batch 2` and kept its vision encoder on the GPU. Not changed by this: `--kv-resident 20480`,
`--lookup-chain 4` and `STRATA_L3_PF=1` were all within that spread on this PC.

## Limits

- NVIDIA on Linux only (the tool builds the CUDA engine; AMD and Windows keep `build/maya`).
- Pictures: the engine does not lend Maya's on-demand vision encoder its expert pool (`vision_lend=0`), so the server
  runs the encoder on the CPU, as it does wherever the GPU cannot spare its memory. Text speed is unaffected; a picture
  takes longer to read.
- The VRAM map is written for one GPU.
- The patch is against one Strata commit. A newer one may need the patch refreshed; `--commit` picks another.

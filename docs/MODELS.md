# The models

Maya runs five models, each on an engine built for it. This page has each one's files, what they need and what is
special about it. Back to the [README](../README.md).

- [How the quants are named](#how-the-quants-are-named)
- [GLM-5.3-Flash](#glm-53-flash)
- [GLM-5.3 (744B)](#glm-53-744b)
- [DeepSeek-V4.1-Flash-1017](#deepseek-v41-flash-1017)
- [MiniMax-M3.1-Flash](#minimax-m31-flash)
- [Hy4-Flash](#hy4-flash)
- [Inkling-2-Mini (experimental preview)](#inkling-2-mini-experimental-preview)
- [Inkling-2-Small (by hand, for now)](#inkling-2-small-by-hand-for-now)
- [Community quants of GLM-5.3-Flash](#community-quants-of-glm-53-flash)

## How the quants are named

Every Maya quant is named after its model first, then its tier - XS, S, M or L, by how many bits its experts keep - so
"GLM-5.3-Flash · Maya-S" and "MiniMax M3.1 · Maya-S" never get mixed up. The setup asks which one to download; to pick one
yourself, give its setup id: `./setup.sh --model glm-maya-s24`. The names used before (`--model Maya-S24`,
`--ds4-quant maya-xs`, ...) keep working.

## GLM-5.3-Flash

**[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** (zai-org, MIT license; 321 B parameters, about 18 B
active per token, a context of up to 1 M tokens), on Maya's first engine (`maya`).

Project Maya's own quants of GLM-5.3-Flash, made from Z.ai's FP8 release with statistics from the FP8 model. Each keeps
the model's MTP block, which drafts tokens ahead.

| Model | Setup id | Size | Made for | Against the FP8 model |
| --- | --- | ---: | --- | --- |
| **GLM-5.3-Flash · Maya-S** (default) | `glm-maya-s` | 96.5 GB | PCs with a smaller memory pool (RAM and VRAM together) | 97.9% of its zero-shot accuracy; the same next token 83% of the time |
| **GLM-5.3-Flash · Maya-S24** | `glm-maya-s24` | 94.7 GB | cards of 24 GB or less: 4-bit attention leaves ~1.5 GB more of the card for experts, decode +14% | 97.7%; 83% |
| **GLM-5.3-Flash · Maya-XS** | `glm-maya-xs` | 73.4 GB | PCs with 64 GB of RAM and a 24 GB card, at 128K context (int8 KV cache, thinking budget 8192) | - |
| **GLM-5.3-Flash · Maya-M** | `glm-maya-m` | 116 GB | a bigger memory pool; more bits where they count, calibrated toward tool calls and front-end code | 97.9%; 86% |
| **GLM-5.3-Flash · Maya-M-Derisked** | `glm-maya-m-derisked` | 116 GB | Maya-M with a weight edit by [Blackfrost_AI](https://x.com/Blackfrost_AI) that reduces blanket refusals ([its repo](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF)); experimental | not measured yet |
| **GLM-5.3-Flash · Maya-L** | `glm-maya-l` | 156.3 GB | the biggest memory pools | **99.2%**; 90% |

The setup recommends one for your PC. How they're made and measured: [MAYA-S](../bench/results/MAYA-S.md),
[MAYA-S24](../bench/results/MAYA-S24.md), [MAYA-M](../bench/results/MAYA-M.md), [MAYA-L](../bench/results/MAYA-L.md). On
[Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF).

On AMD, GLM-5.3-Flash reads pictures too: setup builds its image encoder for the GPU.

### GLM-5.3-Flash's options

- **Several requests at once:** `./maya.sh --parallel 4` (Windows: `START-MAYA.bat --parallel 4`; up to 8) keeps it in
  the config, and the dashboard shows how many answers are being written. Each extra request keeps its own context in
  VRAM, so a few fewer experts fit there ([how](SETTINGS.md#several-requests-at-once)). They add up: on one GPU a
  token of each goes through the model together, so the dense weights are read once for all of them. In tokens per
  second for all of them together, against the same questions one after another: x1.41 for two and x1.35 for four on a
  V100 that streams most of Maya-S's experts from RAM, x1.67 for two and x2.02 for four on a Gorgon Halo (Radeon
  8065S, 160 GB for the GPU) that holds every expert in GPU memory. The more of the model VRAM holds, the more it
  gains.
- **Long contexts:** new setups keep each attention layer's cache in RAM with a window of it in VRAM
  (`--kv-streaming on`; `./maya.sh --setup` adds it to an existing config), so the VRAM holds more experts - at 512K
  on an RTX 3090, prompts 4-9x faster and decode +5%; 1M context fits one 3090 ([how](SETTINGS.md)).
- **Faster answers with a DFlash2 drafter:** `./maya.sh --setup --draft dflash2` (NVIDIA RTX 30 / A100 and newer,
  and one AMD Ryzen AI Max APU) downloads a small drafter that drafts up to 7 tokens a round; the model checks every drafted token, so it only writes tokens it would pick itself
  - Maya-L +44% on nine GPUs and +12% on two (against the MTP block), +19% on one RTX 3090, +9% on a Radeon 8060S
  with unsloth's UD-IQ2_XXS ([how](SETTINGS.md#drafting-with-dflash2)). How much
  it gains depends on the PC: on one GPU that reads much of the model from RAM it is often slower. So a
  [speed calibration](SETTINGS.md#the-speed-calibration) measures it on your PC and keeps it only where it is faster.
  The drafter's license is non-commercial (CC BY-NC-ND 4.0).

## GLM-5.3 (744B)

**[GLM-5.3](https://huggingface.co/zai-org/GLM-5.3)** (zai-org; 744 B parameters, about 40 B active per token) runs on the same engine as
GLM-5.3-Flash, with its own sparse-attention indexer. Text only, up to 1M context (the setup offers what fits).

| Model | Setup id | Size | Made for |
| --- | --- | ---: | --- |
| **GLM-5.3 (744B) · Maya-XS** ([Maya's own GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Maya-GGUF)) | `glm-5.3-maya-xs` | 170.0 GB | smaller memory pools (~192 GB of VRAM and RAM together keeps every expert in memory); as close to the original as unsloth's UD-IQ1_S at 78% of its size |
| **GLM-5.3 (744B) · UD-IQ2_M** ([unsloth's GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF)) | `GLM-5.3-UD-IQ2_M` | 238.6 GB | the biggest memory pools (~256 GB of VRAM and RAM together keeps every expert in memory); closer to the original |

NVIDIA from the V100 on (CUDA, Linux and Windows) and AMD (HIP, the Strix and Gorgon Halo included); not on Mac.
Everything about it - setup, context, defaults, APIs: [GLM53.md](GLM53.md).

## DeepSeek-V4.1-Flash-1017

The setup also offers **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**
(DeepSeek, MIT license; 748 B parameters) on an engine of its own (`maya-ds4`): `./setup.sh --setup` and pick it, or
`./setup.sh --model DeepSeek-V4.1-Flash-1017`.

The server uses the pinned official prompt encoder, with numeric reasoning budgets and ID-matched parallel tool
results. See [the prompt format notes](deepseek-prompt-format.md) for API details and the standalone Jinja limits.

| Files | Size | Runs on |
| --- | ---: | --- |
| **[DeepSeek V4.1 · Maya-S](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)** (`deepseek-maya-s`), Project Maya's own quant (default): the experts in 2-3 bits, 92.6% the same next token as the original | 340 GB, with the Engram tables (Q6_K) | NVIDIA from Volta (V100) on, and AMD; Linux and Windows |
| **[DeepSeek V4.1 · Maya-XS](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)** (`deepseek-maya-xs`): the experts in ~1.6 bits, with its own 2-bit drafter - the whole model in a 128 GB PC's memory (the Engram tables stay on the SSD) | 277 GB, with the Engram tables (Q6_K) | the same |
| unsloth's [UD-IQ1_M GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (`--ds4-quant gguf-iq1_m`): for PCs with about 150 GB of VRAM + RAM | 301 GB, with Maya's n-gram tables | the same |
| unsloth's [UD-IQ3_S GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (`--ds4-quant gguf-iq3_s`): close to Q8's quality, for PCs with about 270 GB of VRAM + RAM; shares its n-gram tables with Maya's quants | 424 GB, with the n-gram tables | the same |
| unsloth's [UD-Q8_K_XL GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (`--ds4-quant gguf-q8`; DeepSeek's own formats): the closest to the original, for PCs with about 310 GB of VRAM + RAM | 467 GB, with Maya's n-gram tables | the same |
| turboderp's [EXL3 2.25 bpw](https://huggingface.co/turboderp/DeepSeek-V4.1-Flash-1017-exl3/tree/2.25bpw) + the same Engram tables (`--ds4-quant exl3-2.25`: the smallest EXL3, ~160 GB of VRAM + RAM; its vision tower is DeepSeek's own, shared with the GGUFs) | 362 GB | the same |
| LibertAIDAI's [EXL3 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) + DeepSeek's Engram tables | 414 GB | NVIDIA from Volta (V100) on; pictures from RTX 30 / A100 on; and AMD |
| LibertAIDAI's [EXL3 3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) + the same Engram tables (`--ds4-quant exl3-3.5`: closer to the original, ~280 GB of VRAM + RAM) | 461 GB | the same |
| orcarouter's [Abliterated GGUFs](https://huggingface.co/orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF), the model with its refusals removed: Mix-128G (`--ds4-quant abliterated-128g`, ~114 GB of VRAM + RAM) or Mix-256G (`abliterated-256g`, ~236 GB); `abliterated` picks the one your PC holds. Their n-gram table is left out and Maya's Q6_K tables are read instead (once for both). A gated repo: accept its terms, then `HF_TOKEN` or `hf auth login` | 275 / 397 GB, with the 161 GB of n-gram tables shared with Maya's DeepSeek quants (downloaded once) | NVIDIA from Volta (V100) on (AMD not tried yet) |

- **Any amount of memory:** the most-used experts go to the GPUs, the others are read in place from the model's files -
  from RAM where they fit, from the SSD otherwise. More VRAM and RAM is faster; keep the files on an NVMe SSD. The
  setup recommends a 256K context where your VRAM and RAM hold the model, 64K otherwise.
- **The DSpark drafter:** with the GGUFs the setup also fetches the model's own drafter (8 GB from Maya's repo:
  DeepSeek's FP4 experts, its dense weights in 8-bit) and measures whether drafting makes your PC faster; it keeps it
  only where it does.
- **Pictures:** with the GGUFs the setup fetches DeepSeek's vision tower (1 GB of its checkpoint). It waits in RAM
  and takes no VRAM between pictures: while a picture is read it borrows about 320 MB of the first GPU's memory, then
  gives it back to the model. `--no-vision` leaves it out.
- **One GPU or several:** the layers are split across the GPUs (pairs of equal GPUs also split each layer where the
  files' formats allow), its thinking levels (Off, Low, High, Max) in the chat, and its own numbers in the monitor.
- **Strix Halo / Gorgon Halo on Windows:** Windows counts the GPU's share of the memory against its page file, so the
  setup asks for a page file big enough to keep every expert on the GPU and says the size
  ([AMD_MAYA.md](AMD_MAYA.md)).
- **Other EXL3 quants from Hugging Face:** `tools/ds4/pack.py` packs them too. A head kept as BF16 (`head.weight`)
  is read as it is. Engram tables a quant stored in its own INT4 layout (`U8` + `embed.scale`, as in the quants derived
  from Diffbot's mixed 3 bpw) are not read: the tables are DeepSeek's own and the quant leaves them unchanged, so put
  DeepSeek's FP8 shards 47 / 48 of deepseek-ai/DeepSeek-V4.1-Flash-1017 in their place. Shards that write their
  tensors grouped by type instead of expert by expert stop the pack with that reason: rewrite them in expert order
  in place with `tools/ds4/reorder.py --model <model folder>` - the same tensors and bytes, each tensor's SHA-256
  checked ([type-ordered shards, and Diffbot's 3 bpw](ds4-exl3-packs.md)).

### DeepSeek's options

All of these are optional; the details and every setting are in [SETTINGS.md](SETTINGS.md).

- **Several requests at once:** `./maya.sh --parallel 4` (Windows: `START-MAYA.bat --parallel 4`; up to 8), as for
  GLM-5.3-Flash: each conversation keeps its own state, and a token of each goes through the model together. In tokens
  per second for all of them together: x1.57 for two and x1.26 for four on a V100 with most of Maya-XS's experts in
  RAM, against the same questions one after another with the DSpark drafter
  ([how](SETTINGS.md#deepseek-v41-flash-several-requests-at-once)).
- **Conversations kept:** the DeepSeek-V4.1-Flash-1017 engine holds one conversation: a prompt that continues it
  reuses its state, anything else starts over. With `MAYA_DS4_SLOT_KEEP=1`, a conversation another request takes the
  model from - a chat, or an agent's parent and workers - is set aside, and a request that continues it reads only its
  new tokens: in RAM, or across a restart from the folder. Off by default. How many are kept in RAM, the folder, its
  size cap and how long a slot lives are settings too
  ([conversations kept](SETTINGS.md#deepseek-conversations-kept-an-agents-parent-and-workers-on-one-model)).
- **Loading the experts:** the engine reads the pack's expert regions into VRAM and RAM at start. Once a region's
  bytes are copied, the load drops its pages from the file cache (`posix_fadvise`), so the same bytes are not held
  twice and a pack whose copies would not fit beside their cache loads at the disk's speed
  (`MAYA_DS4_LOAD_DROP_CACHE`, on; `0` keeps them: a restart then starts warm, and a pack that fits loads a few
  seconds faster without the calls). `MAYA_DS4_LOAD_THREADS` (12) and `MAYA_DS4_LOAD_INTERLEAVE` (off) set the load's
  threads and file order; [loading the experts](SETTINGS.md#deepseek-loading-the-experts) has the numbers for a GGUF and an EXL3
  pack, and the huge-page note for a load that stalls part-way.
- **Engram prompt read-ahead:** on one GPU, an optional read-ahead fills the ordinary row cache before each consuming
  layer. It groups sorted rows into larger zone reads, removes duplicate reads, and sizes look-ahead from the cache
  budget instead of letting a large prompt overwrite rows before their use. The model's row bytes and decode path are
  unchanged; it is off by default ([settings and caveats](SETTINGS.md#deepseek-engram-prompt-read-ahead)).
- **Large-RAM Linux hosts:** optional
  [locked Engram tables and strict CPU-expert RAM placement](SETTINGS.md#deepseek-explicit-ram-residency). Both are
  off by default; they need enough physical RAM.
- **Expert-copy startup on Linux (GGUF):** optional [direct reads](SETTINGS.md#deepseek-expert-copy-io) for the
  experts copied into RAM (off by default).
- **Prompt experts on AMD:** `MAYA_DS4_EXPERT_MMQ=1` uses the existing int8 MMQ path for routed IQ experts. It is off
  by default and adds q8 activation rounding ([routed prompt experts on AMD](SETTINGS.md#deepseek-routed-prompt-experts-on-amd)).

## MiniMax-M3.1-Flash

The setup also offers **[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash)** (MiniMax; 428 B
parameters, 23 B active per token, up to 1 M tokens; the [MiniMax Community License](../LICENSE.minimax-m31): free for
non-commercial use, commercial use has conditions) on an engine of its own (`maya-mm3`, contributed by
[@needmorevram](https://github.com/needmorevram), #477): `./setup.sh --model MiniMax-M3.1-Flash`.

| Files | Size | Runs on |
| --- | ---: | --- |
| **[MiniMax M3.1 · Maya-XS](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF)** (`minimax-maya-xs`), Project Maya's own quant: the experts in ~1.6 bits - the whole model in a 128 GB PC's memory | 93 GB | NVIDIA V100 / RTX 20 and newer, and AMD; Linux and Windows; macOS on Apple Silicon (the [Metal engine](METAL.md#minimax-m31-flash), experimental) |
| **[MiniMax M3.1 · Maya-S](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF)** (`minimax-maya-s`): the experts in ~2.1 bits, closer to the original; 87.6% the same next token as the original (the average over its categories) | 122 GB | the same |
| Mia-AiLab's [EXL3 3.5 bpw](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw), with MiniMax's DSpark drafter and vision tower | 197 GB | the same |

With Maya's quants the setup fetches the vision tower and the DSpark drafter from the EXL3 repo (about 4 GB) for
pictures and drafting, on NVIDIA and AMD alike.

- **Set up for your PC:** the experts that do not fit in VRAM are computed on the CPU from RAM; the context (up to the
  model's 1M) is recommended by your memory; tensor parallel where the GPUs form groups; and the DSpark drafter only
  where it measures faster on your PC.
- **MiniMax's own settings:** the model card's sampling (temperature 1.0, top-p 0.95) and its thinking levels - Auto
  (the default: it decides whether to think), the five efforts Low to Max, and Off - with MiniMax's own prompt format.
- **Pictures:** on from the start, the model's vision tower running in the engine; `--no-vision` leaves it out.
- **Several requests at once:** `./maya.sh --parallel 4` (up to 8) keeps it in the config. Each conversation has its
  own KV cache in VRAM, so KV streaming and the DSpark drafter are off while it is above 1, and the setup recommends a
  context that leaves the caches room ([how](SETTINGS.md#several-requests-at-once)).
- **Tool calls, the OpenAI- and Anthropic-compatible API and the dashboard** work as for the other models, with its
  own numbers in the monitor.

Its engine settings: [SETTINGS.md](SETTINGS.md#minimax-m31-flash).

## Hy4-Flash

The setup also offers **[Hy4-Flash](https://huggingface.co/tencent/Hy4-flash)** (Tencent's Hy4-flash; 220 B
parameters: a 155.5 B language backbone with 512 experts a layer, 63.5 B of n-gram "Engram" tables, a vision encoder
and a drafter; the [Tencent Hy License](../LICENSE.hy4-flash): based on Apache-2.0, with extra commercial terms for large
model-as-a-service providers) on an engine of its own (`maya-hy4`; base NVIDIA engine contributed by
[@needmorevram](https://github.com/needmorevram), #1040): `./maya.sh --setup` and pick "Hy4-Flash", or
`./maya.sh --setup --model hy4-flash-maya-xs` (also `--hy4-quant maya-xs`; Windows: `START-MAYA.bat`).

| Files | Size | Runs on |
| --- | ---: | --- |
| **[Hy4-Flash · Maya-XS](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF)** (`hy4-flash-maya-xs`), Project Maya's own quant: the routed experts in 2 bits - for PCs with about 56-64 GB of RAM + VRAM | 44.8 GB, and the n-gram table (63.4 GB, on the SSD); about 110 GB of disk in all | NVIDIA from the V100 on (CUDA); Linux and Windows |
| **[Hy4-Flash · Maya-S](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF)** (`hy4-flash-maya-s`): the routed experts in 2 bits on layers 1-23 and 3 bits on layers 24-41 - for PCs with about 68-80 GB of RAM + VRAM | 53.0 GB, and the same n-gram table; about 116 GB of disk in all | NVIDIA from the V100 on, cards from 12 GB; Linux and Windows |
| **[Hy4-Flash · Maya-M](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF)** (`hy4-flash-maya-m`): the routed experts in 3 bits (4 bits on 7 layers) - for PCs with about 80-96 GB of RAM + VRAM | 68.4 GB, and the same n-gram table; about 132 GB of disk in all | NVIDIA from the V100 on, cards from 12 GB; Linux and Windows |
| **[Hy4-Flash · Maya-L](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF)** (`hy4-flash-maya-l`): the routed experts in 4 bits (5 bits on 5 layers) - for PCs with about 104-128 GB of RAM + VRAM | 89.5 GB, and the same n-gram table; about 153 GB of disk in all | NVIDIA from the V100 on, cards from 12 GB; Linux and Windows |

The setup recommends a size by the PC's RAM + VRAM. Quality figures are on
the [model card](https://huggingface.co/peasantsmith/Hy4-Flash-Maya-GGUF).

- **One n-gram table for every size:** Tencent's FP8 table, value for value (63.4 GB). It is downloaded once, stays
  on the SSD and is read page by page while the model answers: it does not need to fit in memory and is never loaded
  into RAM whole. Keep it on an SSD / NVMe, never a hard disk.
- **AMD (Linux):** `./maya.sh --setup --backend hip --model hy4-flash-maya-xs`. Run by us on a Gorgon Halo (Radeon 8065S, `gfx1151`, ROCm 7.14) with the Radeon alone: Maya-XS and Maya-L with every expert in the GPU's memory, pictures, tool calls and both thinking levels. On a Strix Halo / Gorgon Halo the setup says whether the size sits in the GPU's memory whole ([MEMORY.md](MEMORY.md): 64 GB - Maya-XS, 96 GB - Maya-M, 128 GB - Maya-L). Radeon cards (`gfx1100`): the engine's AMD build was contributed by [@Cass67](https://github.com/Cass67), who ran Tencent's own Q4_K_M on four RX 7900 XT; we have not run such cards ourselves. With an AMD GPU on Windows it is experimental - the setup offers it and says so; we have not run it there ([AMD_MAYA.md](AMD_MAYA.md#hy4-flash-under-windows-experimental)).
- **Memory:** RAM + VRAM of about 56-64 GB for Maya-XS (a rule of thumb: the model file plus 4-8 GB for the context).
  With less it still runs, slower ([MEMORY.md](MEMORY.md)).
- **VRAM:** everything but the routed experts has to fit in VRAM; the experts that do not fit are computed on the CPU
  from RAM. Maya-XS needs about 8.7 / 9.0 / 9.8 GiB of free VRAM at 4K / 8K / 32K context. More VRAM puts more
  experts on the GPU, which is faster.
- **Cards of 8 GB:** for a card under about 10 GB the setup picks an "8 GB" layout of Maya-XS by itself - a second
  model file with smaller attention weights (43.6 GB; slightly lower quality) that needs about 6.5 / 6.8 GiB of free
  VRAM at 4K / 8K context, and about 7.0 / 7.1 GiB at 16K / 32K, where the engine keeps the context's cache in 8
  bits by itself (the setup says so). A card with less than about 7 GB cannot run it.
- **Set up for your PC:** the setup checks the PC, picks the layout and the context that fit the card (4K to 1M,
  each offered where it fits the card's VRAM; run end to end up to 32K so far), downloads the model file and
  the n-gram table from Maya's repo, builds the engine's index of them (the pack) and the engine, and writes
  `maya-hy4-flash-maya-xs.json`. `--model-dir <folder>` uses files you already have (checked by size and sha256).
- **Tencent's own settings:** temperature 0.9, top-p 1.0, no top-k, with the release's own chat template (inside the
  GGUF). The model thinks at High (its default) or not at all (Off), so those are the two levels offered - a
  request for another level runs at High; the thinking budget is 32768 tokens.
- **Tool calls, the OpenAI- and Anthropic-compatible API and the dashboard** work as for the other models: its entry
  on the Models page (load, unload, delete; context, GPUs, tensor parallel, prompt chunk, CPU threads, VRAM kept
  free), and in the Monitor the
  experts on the GPU against those computed on the CPU and the n-gram table's read time. The loop guard is on, as for
  every model.
- **Pictures:** the model's own vision tower runs in the engine (contributed by
  [@needmorevram](https://github.com/needmorevram), #1051). The setup downloads Maya's vision file with the model
  (0.8 GB, one for every size; `--no-vision` skips it). Its weights stay in the file; a picture takes up to 0.8 GB of
  VRAM while it is read, lent by the engine's prompt buffers (idle while a picture is read): no VRAM is kept for
  pictures, so every card that runs the model reads them at full size, 8 GB cards included, and the VRAM goes to the
  model. Only where the tower does not load are pictures
  off: the setup says so, text works as before, a picture sent is refused in plain words and a picture in a tool
  result becomes a short note. Pictures up to 2048 x 2048 pixels (larger ones are scaled down), a square one 4,160 tokens of the context. Pictures only, no videos. Images on the Models page
  turns them on or off.
- **Drafting:** Maya's sizes draft with Tencent's MTP block, the same block Tencent's own GGUF files carry. It is
  one extra file for every size (7.5 GB, in Maya's repo; the setup downloads it once and links it into each size's
  folder). The block guesses the next tokens and the model checks every one, so Maya only writes tokens the model
  itself would pick. It takes about 4 GB of VRAM on the main GPU, so the setup turns it on where that fits beside
  the model's own parts (cards from about 16 GB at a 32K context) and leaves it off, saying so, on smaller cards.
  `--drafting off` sets a size up without it; Drafting on the Models page switches it. Because the block takes VRAM
  that would hold experts, other experts sit in VRAM than without it, and an answer can differ from a run without
  the block.
- **In llama.cpp:** the files also load there with its Hy4-flash pull request
  ([ggml-org/llama.cpp #32517](https://github.com/ggml-org/llama.cpp/pull/32517), not merged yet).

Not available for Hy4-Flash yet (the dashboard offers no switch for them):

- **Several requests at once:** no `--parallel`; one request at a time for now.
- **AMD and Mac:** NVIDIA only.

### Tencent's own Q4_K_M (AngelSlim), with drafting

The same setup also offers Tencent's own Q4_K_M from
[AngelSlim's GGUF](https://huggingface.co/AngelSlim/Hy4-flash-GGUF) (the download's license file,
`License_AngelSlim_model_and_dataset.txt`, and Tencent's model card give the terms), listed after Maya's sizes -
contributed with the engine by [@needmorevram](https://github.com/needmorevram) (#1040), and run by him on Linux with
NVIDIA RTX 30 / A100 and newer cards.

**Set up:** `./maya.sh --setup --model hy4-flash-q4_k_m` (or pick Hy4-Flash in the setup's model list, then
"Q4_K_M") checks the PC, builds `maya-hy4`, downloads the two files and the vision tower's (AngelSlim's
`mmproj-Hy4-flash-bf16.gguf`, 1 GB; pinned, sha256-checked; `--model-dir <folder>` uses the ones you have), writes the
pack and `maya-hy4-flash.json`; `./maya.sh` starts it. `--drafting off` sets it up without MTP drafting, `--no-vision`
without pictures.

| Files | Size | Runs on |
| --- | ---: | --- |
| AngelSlim's **Q4_K_M** (`Hy4-flash-Q4_K_M-00001-of-00002.gguf` and the n-gram table): Q4_K experts (Q6_K downs on 19 layers), Q8_0 elsewhere | 157 GiB | NVIDIA; run on RTX 30 / A100 and newer, Linux |

- **Tensor parallel:** where two or more GPUs are of one kind, every layer runs on all of them at once (about twice
  the layer split's decode speed on eight RTX 5060 Ti); the fastest GPU holds the head and the drafter and computes a
  share of the experts.
- **Where the experts live:** all of them in VRAM where they fit (about 108 GB); where they do not, each layer keeps
  a share of its experts in VRAM (its most used, with an expert profile) and the CPU computes the others from the
  model's file.
- **Context:** up to 256K tested (the setup offers 32K to 256K, and 512K / 1M on cards of 20 / 40 GB and more). With tensor parallel every GPU of the group holds the
  whole cache, so where 16-bit rows do not fit (256K on 16 GB cards) the engine keeps them in 8 bits.
- **Drafting:** the model's own MTP block proposes up to three tokens a step and the model checks them - nearly twice
  the decode speed on code, the same answers.
- **Thinking and tools:** the GGUF's own template; the model thinks at High (its default) or not at all (Off), so those
  are the two levels offered - a request for another level runs at High. Chat, thinking and tool calls through Maya's
  OpenAI-compatible server.
- **Pictures:** the model's own vision tower runs in the engine on the main GPU (its weights stay in RAM; a picture
  takes up to 0.8 GB there while it is read, lent by the prompt buffers - none kept for pictures). Pictures up to 2048 x 2048 pixels (larger ones are
  scaled down), a square one 4,160 tokens of the context; 1024 x 768 is read in about 0.25 s, the largest in about
  5 s on an RTX 3090. Pictures only, no videos. Images on the Models page turns them off.
- **The pack:** `python tools/hy4/pack_gguf.py --model <folder>` indexes the GGUF set and the vision tower's file
  beside it (no weights copied).

Other GGUF sets of the model are not offered by the setup; `tools/hy4/pack_gguf.py` indexes one by hand
([SETTINGS.md](SETTINGS.md#hy4-flash)).

Its engine settings: [SETTINGS.md](SETTINGS.md#hy4-flash).

## Inkling-2-Mini (experimental preview)

**[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini)** (thinkingmachines; 26 B, a looped model:
layers 6-17 run three times, 8 of 128 experts a token; Apache 2.0) runs on its own engine, `maya-ik2`, from unsloth's
GGUF as it is. An **experimental preview**: `maya.py --setup --model inkling-2-mini`, or the Models page's list.

| Files | Size | Runs on |
| --- | ---: | --- |
| [unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (NVFP4 experts in layers 0-19, Q8_0 elsewhere) | 17.6 GB | NVIDIA V100 / RTX 20 and newer, or AMD RDNA 3 / 3.5 / 4 (Radeon RX 7000 / 9000, Ryzen AI Max); Linux; NVIDIA with 8 GB of VRAM or more (experts that do not fit stay in RAM: about 13 GB of RAM with an 8 GB card), AMD or a Halo / APU with 24 GB (every expert in memory); Windows is not supported yet |

- **Where it runs:** the whole model sits on one GPU. The setup says so in one line and stops on a Mac, Windows, two
  GPUs or more, NVIDIA cards older than V100 / RTX 20, AMD GPUs older than RDNA 3 and GPUs under 24 GB.
- **AMD (`--backend hip`):** run by Maya's team on a Radeon 8065S (Gorgon Halo, Linux); Radeon RX 7000 / 9000 cards
  build the same engine but have not been run here yet - tell us how it goes.
- **Context:** up to 256K (the setup offers 32K to 256K and recommends the largest that keeps every expert in VRAM on
  the card); the cache in FP8 by default (F16 on the Models page).
- **Drafting:** not offered by the setup in this preview (the engine's `--draft` with the model's drafter is
  by hand: docs/INKLING2.md).
- **Thinking and tools:** thinkingmachines' own template; its effort levels Off, Low, Medium, High (its default) and
  Max. Tool calls through Maya's OpenAI-compatible server.
- **Pictures:** on by default - the setup fetches the model's vision tower (unsloth's `mmproj-BF16.gguf`, 0.6 GB, from the
  same repo; `--no-vision` skips it) and builds Maya's image encoder; the encoder starts only when a picture arrives
  (on the GPU where its free VRAM holds it, else on the CPU), so the context's fit is the same. Images on the Models
  page turns them off and on.
- One request at a time for now.

## Inkling-2-Small (by hand, for now)

**[Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small)** (thinkingmachines; 594 B, ~17 B active,
Apache 2.0) runs on its own engine, `maya-ik2` (#974), from unsloth's GGUFs as they are - not from the setup yet: a
tokenizer folder and a run config by hand, then `serve/server.py` ([INKLING2.md](INKLING2.md) section 6).

| Files | Size | Runs on |
| --- | --- | --- |
| [unsloth's UD-Q4_K_XL](https://huggingface.co/unsloth/Inkling-2-Small-GGUF) (NVFP4 experts, layers 0-39) | 330 GiB; the experts ~323 GiB of pinned RAM | NVIDIA, Ampere on (CUDA), Linux; one or two GPUs. Run on H200 (35 tok/s) and RTX PRO 6000 (22; two: 42) |
| unsloth's UD-Q3_K_XL (IQ3_XXS / IQ4_XS experts) | 266 GiB; ~260 GiB of pinned RAM | the same; H200: 44.6 tok/s |

Text only, one request at a time. Its engine settings: [SETTINGS.md](SETTINGS.md#inkling-2-small).

## Community quants of GLM-5.3-Flash

These also run on Maya:

| Model | Size | How | Notes |
| --- | ---: | --- | --- |
| [EXL3 3.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw) by turboderp | 125.3 GB | in the setup: `--model EXL3-3.05bpw` | NVIDIA from V100, and AMD |
| [EXL3 4.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/4.05bpw) by turboderp | 165.2 GB | in the setup: `--model EXL3-4.05bpw` | NVIDIA from V100, and AMD |

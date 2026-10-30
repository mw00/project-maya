# The models

Maya runs three models, each on an engine built for it. This page has each one's files, what they need and what is
special about it. Back to the [README](../README.md).

- [How the quants are named](#how-the-quants-are-named)
- [GLM-5.3-Flash](#glm-53-flash)
- [DeepSeek-V4.1-Flash-1017](#deepseek-v41-flash-1017)
- [MiniMax-M3.1-Flash](#minimax-m31-flash)
- [Community quants of GLM-5.3-Flash](#community-quants-of-glm-53-flash)

## How the quants are named

Every Maya quant is named after its model first, then its tier - XS, S, M or L, by how many bits its experts keep - so
"GLM-5.3 · Maya-S" and "MiniMax M3.1 · Maya-S" never get mixed up. The setup asks which one to download; to pick one
yourself, give its setup id: `./setup.sh --model glm-maya-s24`. The names used before (`--model Maya-S24`,
`--ds4-quant maya-xs`, ...) keep working.

## GLM-5.3-Flash

**[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** (zai-org, MIT license; 321 B parameters, about 18 B
active per token, a context of up to 1 M tokens), on Maya's first engine (`maya`).

Project Maya's own quants of GLM-5.3-Flash, made from Z.ai's FP8 release with statistics from the FP8 model. Each keeps
the model's MTP block, which drafts tokens ahead.

| Model | Setup id | Size | Made for | Against the FP8 model |
| --- | --- | ---: | --- | --- |
| **GLM-5.3 · Maya-S** (default) | `glm-maya-s` | 96.5 GB | PCs with a smaller memory pool (RAM and VRAM together) | 97.9% of its zero-shot accuracy; the same next token 83% of the time |
| **GLM-5.3 · Maya-S24** | `glm-maya-s24` | 94.7 GB | cards of 24 GB or less: 4-bit attention leaves ~1.5 GB more of the card for experts, decode +14% | 97.7%; 83% |
| **GLM-5.3 · Maya-M** | `glm-maya-m` | 116 GB | a bigger memory pool; more bits where they count, calibrated toward tool calls and front-end code | 97.9%; 86% |
| **GLM-5.3 · Maya-M-Derisked** | `glm-maya-m-derisked` | 116 GB | Maya-M with a weight edit by [Blackfrost_AI](https://x.com/Blackfrost_AI) that reduces blanket refusals ([its repo](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF)); experimental | not measured yet |
| **GLM-5.3 · Maya-L** | `glm-maya-l` | 156.3 GB | the biggest memory pools | **99.2%**; 90% |

The setup recommends one for your PC. How they're made and measured: [MAYA-S](../bench/results/MAYA-S.md),
[MAYA-S24](../bench/results/MAYA-S24.md), [MAYA-M](../bench/results/MAYA-M.md), [MAYA-L](../bench/results/MAYA-L.md). On
[Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF).

On AMD, GLM-5.3-Flash reads text only.

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
- **Faster answers with a DFlash2 drafter:** `./maya.sh --setup --draft dflash2` (NVIDIA RTX 30 / A100 and newer)
  downloads a small drafter that drafts up to 7 tokens a round and gives the same answers - Maya-L +44% on nine GPUs
  and +12% on two (against the MTP block), +19% on one RTX 3090 ([how](SETTINGS.md#drafting-with-dflash2)). How much
  it gains depends on the PC: on one GPU that reads much of the model from RAM it is often slower. So a
  [speed calibration](SETTINGS.md#the-speed-calibration) measures it on your PC and keeps it only where it is faster.
  The drafter's license is non-commercial (CC BY-NC-ND 4.0).

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
| unsloth's [UD-Q8_K_XL GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (DeepSeek's own formats) | 508 GB | the same |
| LibertAIDAI's [EXL3 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) + DeepSeek's Engram tables | 414 GB | NVIDIA from Volta (V100) on; pictures from RTX 30 / A100 on; AMD experimental |
| LibertAIDAI's [EXL3 3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) + the same Engram tables (`--ds4-quant exl3-3.5`: closer to the original, ~280 GB of VRAM + RAM) | 461 GB | the same |
| orcarouter's [Abliterated GGUFs](https://huggingface.co/orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF), the model with its refusals removed: Mix-128G (`--ds4-quant abliterated-128g`, ~114 GB of VRAM + RAM) or Mix-256G (`abliterated-256g`, ~236 GB); `abliterated` picks the one your PC holds. Their n-gram table is left out and Maya's Q6_K tables are read instead (once for both). A gated repo: accept its terms, then `HF_TOKEN` or `hf auth login` | 275 / 397 GB, with the 161 GB of n-gram tables shared with Maya's DeepSeek quants (downloaded once) | NVIDIA from Volta (V100) on (AMD not tried yet) |
| ISTA-DASLab's [GSQ-RCO IQ2_XXS GGUF](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF): each tensor in the type a search picked, 2 bits on average, with IQ4_NL Engram tables; ISTA reports it matches the original on its benchmarks. Not offered by the setup: [download, pack and start it by hand](deepseek-ista-gguf.md) | 249 GB, with the Engram tables (IQ4_NL) | NVIDIA, on Linux (AMD and Windows not tried) |

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
| **[MiniMax M3.1 · Maya-XS](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF)** (`minimax-maya-xs`), Project Maya's own quant: the experts in ~1.6 bits - the whole model in a 128 GB PC's memory | 93 GB | NVIDIA V100 / RTX 20 and newer, and AMD; Linux and Windows |
| **[MiniMax M3.1 · Maya-S](https://huggingface.co/peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF)** (`minimax-maya-s`): the experts in ~2.1 bits, closer to the original; 87.6% the same next token as the original (the average over its categories) | 122 GB | the same |
| Mia-AiLab's [EXL3 3.5 bpw](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw), with MiniMax's DSpark drafter and vision tower | 197 GB | NVIDIA V100 / RTX 20 and newer, Linux and Windows |

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

## Community quants of GLM-5.3-Flash

These also run on Maya:

| Model | Size | How | Notes |
| --- | ---: | --- | --- |
| [EXL3 3.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw) by turboderp | 125.3 GB | in the setup: `--model EXL3-3.05bpw` | experimental; NVIDIA, and AMD |
| [EXL3 4.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/4.05bpw) by turboderp | 165.2 GB | in the setup: `--model EXL3-4.05bpw` | experimental; NVIDIA, and AMD |

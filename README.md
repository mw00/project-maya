<h1 align="center">Project Maya</h1>

<p align="center"><b>GLM-5.3-Flash (321 B) and DeepSeek-V4.1-Flash-1017 (748 B) on your own GPU</b><br>
private and fast · one NVIDIA GPU or up to 16 · AMD and Windows (experimental) · chat, pictures, OpenAI- and
Anthropic-compatible API · several requests at once</p>

<p align="center">
<a href="https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fpeasantsmith%2FGLM-5.3-Flash-Maya-GGUF%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=GLM-5.3-Flash%20Maya%20downloads" alt="GLM-5.3-Flash Maya downloads"></a>
<a href="https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fpeasantsmith%2FGLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=Maya-M-Derisked%20downloads" alt="Maya-M-Derisked downloads"></a>
<a href="https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF"><img src="https://img.shields.io/badge/dynamic/json?url=https%3A%2F%2Fhuggingface.co%2Fapi%2Fmodels%2Fpeasantsmith%2FDeepSeek-V4.1-Flash-1017-Maya-GGUF%3Fexpand%3DdownloadsAllTime&amp;query=%24.downloadsAllTime&amp;label=DeepSeek-V4.1-Flash%20Maya%20downloads" alt="DeepSeek-V4.1-Flash Maya downloads"></a>
</p>

<p align="center">Created and led by <b>Peasant Smith</b> (<a href="https://github.com/mw00">@mw00</a>), co-maintained
with <b><a href="https://github.com/needmorevram">@needmorevram</a></b></p>

<p align="center"><a href="https://buymeacoffee.com/peasantsmith"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a><br>
<sub>Project Maya is free. If it runs well on your PC, a coffee keeps the work on it going.</sub></p>

Models this size usually need a datacenter. Maya runs **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**
(zai-org, MIT license; 321 B parameters, about 18 B active per token, a context of up to 1 M tokens) and
**[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)** (DeepSeek, MIT license;
748 B parameters) on the hardware you already have. Each model has an engine built for it. It keeps the experts your
conversation uses most on the GPU, the next ones in RAM and the rest on your NVMe SSD, and moves them as you chat. It
measures your GPUs, CPU, RAM and SSD and tunes itself to them. Nothing leaves your machine.

- **Fast:** up to 124 tokens/s on two H100s, 118 on four RTX 4090s, 81 on one RTX PRO 6000, 30+ on a single RTX 5090. Real numbers from
  real users below.
- **Close to the original:** Maya's own quants of GLM-5.3-Flash keep 97.7-99.2% of the FP8 model's zero-shot
  accuracy; Maya-S of DeepSeek-V4.1-Flash-1017 picks the same next token as the original 92.6% of the time.
- **Plugs into your tools:** OpenAI- and Anthropic-compatible API with tool calls, ready for coding agents.
- **Several requests at once:** up to 8 answered together - for agents that run several sessions, or several people
  on one PC.
- **Everything in the box:** a chat and a live monitor in the browser, pictures, thinking levels, a one-command setup.

## The models

Project Maya's own quants of GLM-5.3-Flash, made from Z.ai's FP8 release with statistics from the FP8 model. Each keeps
the model's MTP block, which drafts tokens ahead.

| Model | Size | Made for | Against the FP8 model |
| --- | ---: | --- | --- |
| **Maya-S** (default) | 96.5 GB | PCs with a smaller memory pool (RAM and VRAM together) | 97.9% of its zero-shot accuracy; the same next token 83% of the time |
| **Maya-S24** | 94.7 GB | cards of 24 GB or less: 4-bit attention leaves ~1.5 GB more of the card for experts, decode +14% | 97.7%; 83% |
| **Maya-M** | 116 GB | a bigger memory pool; more bits where they count, calibrated toward tool calls and front-end code | 97.9%; 86% |
| **Maya-M-Derisked** | 116 GB | Maya-M with a weight edit by [Blackfrost_AI](https://x.com/Blackfrost_AI) that reduces blanket refusals ([its repo](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF)); experimental | not measured yet |
| **Maya-L** | 156.3 GB | the biggest memory pools | **99.2%**; 90% |

The setup asks which one to download and recommends one for your PC. How they're made and measured: [MAYA-S](bench/results/MAYA-S.md),
[MAYA-S24](bench/results/MAYA-S24.md), [MAYA-M](bench/results/MAYA-M.md), [MAYA-L](bench/results/MAYA-L.md). On
[Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF).

### DeepSeek-V4.1-Flash-1017 (new)

The setup also offers **[DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)**
(DeepSeek, MIT license; 748 B parameters) on an engine of its own (`maya-ds4`): `./setup.sh --setup` and pick it, or
`./setup.sh --model DeepSeek-V4.1-Flash-1017`.

The server uses the pinned official prompt encoder, with numeric reasoning budgets and ID-matched parallel tool
results. See [the prompt format notes](docs/deepseek-prompt-format.md) for API details and the standalone Jinja limits.

| Files | Size | Runs on |
| --- | ---: | --- |
| **[Maya-S](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF)**, Project Maya's own quant (default): the experts in 2-3 bits, 92.6% the same next token as the original | 340 GB, with the Engram tables (Q6_K) | NVIDIA from Volta (V100) on, and AMD; Linux and Windows |
| unsloth's [UD-Q8_K_XL GGUF](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (DeepSeek's own formats) | 508 GB | the same |
| unsloth's UD-Q4_K_XL GGUF (dense weights Q8_0: less VRAM) | 501 GB | the same |
| LibertAIDAI's [EXL3 2.9 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) + DeepSeek's Engram tables | 414 GB | NVIDIA RTX 30 / A100 and newer |
| LibertAIDAI's [EXL3 3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) + the same Engram tables (`--ds4-quant exl3-3.5`: closer to the original, ~280 GB of VRAM + RAM) | 461 GB | the same |

- **Any amount of memory:** the most-used experts go to the GPUs, the others are read in place from the model's files -
  from RAM where they fit, from the SSD otherwise. More VRAM and RAM is faster; keep the files on an NVMe SSD.
- **The DSpark drafter:** with the GGUFs the setup also fetches the model's own drafter (8 GB from Maya's repo:
  DeepSeek's FP4 experts, its dense weights in 8-bit) and measures whether drafting makes your PC faster; it keeps it
  only where it does.
- **Pictures:** with the GGUFs the setup fetches DeepSeek's vision tower (1 GB of its checkpoint); it runs on the first
  GPU in about 320 MB. `--no-vision` leaves it out.
- **One GPU or several:** the layers are split across the GPUs (pairs of equal GPUs also split each layer where the
  files' formats allow), its thinking levels (Off, Low, High, Max) in the chat, and its own numbers in the monitor.
- **Strix Halo / Gorgon Halo on Windows:** Windows counts the GPU's share of the memory against its page file, so the
  setup asks for a page file big enough to keep every expert on the GPU and says the size
  ([docs/AMD_MAYA.md](docs/AMD_MAYA.md)).

<details>
<summary>Community quants that also run on Maya</summary>

| Model | Size | How | Notes |
| --- | ---: | --- | --- |
| [EXL3 3.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw) by turboderp | 125.3 GB | in the setup: `--model EXL3-3.05bpw` | NVIDIA only, experimental |
| [EXL3 4.05 bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/4.05bpw) by turboderp | 165.2 GB | in the setup: `--model EXL3-4.05bpw` | NVIDIA only, experimental |

</details>

## How fast is it?

Real machines, real numbers, most of them measured with `./maya.sh --bench`. A token is about ¾ of a word.

| Machine | Model | Decode (writing the answer) | Prefill (reading your prompt) | By |
| --- | --- | ---: | ---: | --- |
| **2x H100 80 GB** (PCIe 5.0), Xeon Platinum 8488C, 504 GB RAM | Maya-L / Maya-M-Derisked, 1M context | **122.4 / 123.7 tokens/s** | 2099 / 1986 tokens/s | @Molyleaf (v1.1.11, #258, #259) |
| **4x RTX 4090 24 GB** (PCIe 4.0 x16), Xeon w7-3445, 62 GB RAM | Maya-S24 | **118 tokens/s** | 4129 tokens/s | @0xPreDa (v1.0.31, #89) |
| **RTX PRO 6000 Blackwell 96 GB**, Ryzen 7 9800X3D, 61 GB RAM | Maya-M, 256K context | **81.0 tokens/s** | 3102 tokens/s | @yundddd (v1.1.7, #172) |
| **1x H100 80 GB** (PCIe 5.0 x16), Xeon Platinum 8488C, 504 GB RAM | Maya-M-Derisked, 1M context | 61.4 tokens/s | 2014 tokens/s | @Molyleaf (v1.1.15, #260) |
| **2x NVIDIA CMP 170HX 64 GB** (PCIe Gen2 x4 each), 2x Xeon E5-2690 v4, 91 GB RAM | Maya-M, every expert in VRAM | 61.3 tokens/s | 514 tokens/s | @ZackO2o (v1.0.6, #22) |
| **2x RTX 5880 Ada 45 GB** (PCIe 4.0 x16), Xeon Gold 5420+, 393 GB RAM | Maya-L, 256K context | 54.5 tokens/s | 1401 tokens/s | @vjsyong (v1.1.4, #149) |
| **2x Tesla V100 32 GB** (PCIe 3.0 x16 each), 2x Xeon Gold 6138, 183 GB RAM | Maya-S24 | 54.5 tokens/s | 542 tokens/s | maintainer (v1.1.13, #226) |
| **2x NVIDIA CMP 170HX 40 GB** (PCIe Gen2 x16 each), Xeon w3-2423, 128 GB DDR5 | Maya-S, 128K context | 54.3 tokens/s | 831 tokens/s | @nvsofts (v1.1.15, #263) |
| **2x RTX 3090 24 GB** (PCIe 4.0 x16, power-capped at 225 W), Threadripper 3970X, 128 GB RAM | Maya-S24 | 53.0 tokens/s | 885 tokens/s | @briepala (v1.1.2, #135) |
| **4x Tesla V100 32 GB** (PCIe 3.0 x16), 24 CPU cores, a 100 GiB RAM limit | Maya-L, 32K context | 48.3 tokens/s (after `--calibrate`) | 499 tokens/s | @areslp (v1.1.3, [#130](https://github.com/mw00/project-maya/issues/130)) |
| **4x RTX 3090 24 GB** (PCIe 4.0 x8 behind one switch), Xeon Platinum 8260M, 376 GB RAM | Maya-L, 1M context | 44.2 tokens/s (tuned) | 698 tokens/s | @foureight84 (v1.1.17, #302) |
| **2x RTX 3090 24 GB** (PCIe 4.0 x16, power-capped at 250 W), EPYC 7452, 256 GB RAM | Maya-M / Maya-L, 128K context | 44.0 / 30.2 tokens/s | 860 / 801 tokens/s | @orochifj (v1.1.4, #165) |
| **2x RTX 4090 24 GB** (PCIe 3.0 + 4.0 x16), 2x Xeon Silver 4309Y, 188 GB RAM | Maya-S24 | 40.5 tokens/s | 1271 tokens/s | @fuad00 (v1.1.0, #122) |
| **2x Tesla V100 32 GB** (PCIe 3), Xeon E5-2690 v4, 30 GB RAM | Maya-S | up to 40 tokens/s | up to 670 tokens/s | maintainer |
| **4x Tesla T10 15 GB** (PCIe 3.0 x16), Xeon E5-2680 v4, 63 GB RAM | Maya-S24, 100K context | 38.4 tokens/s | 393 tokens/s | @zeallag (v1.1.16, #291) |
| **2x Tesla V100 32 GB** (PCIe 3.0 x8 each), Ryzen 9 7945HX3D, 96 GB RAM | Maya-M / Maya-L | 33.8 / 13.3 tokens/s | 490 / 372 tokens/s | @nkhaooat (v1.1.11, #208) |
| **RTX 3090 24 GB + RTX 3060 12 GB** (PCIe 4.0 x16), EPYC 7702P (8-channel DDR4), 141 GB RAM (a VM) | Maya-M, 128K context | 33.7 tokens/s | 682 tokens/s | @erikash (v1.1.4, #157) |
| **RTX 5090 32 GB**, Ryzen 9 7950X, 96 GB DDR5 | Maya-M | 30+ tokens/s | ~2000 tokens/s | @siganos (#43) |
| **AMD Radeon 8065S** (Ryzen AI Max+ PRO 495, 192 GB LPDDR5X, 160 GB of it for the GPU), Windows 11 | Maya-S, 32K context | up to 29.6 tokens/s (23.2 after an 8K-token prompt) | 278 tokens/s | maintainer (v1.1.10) |
| **2x RTX 3090 24 GB** (PCIe 4.0 x16, NVLink), Threadripper PRO 3955WX, 247 GB RAM | EXL3-3.05bpw, 128K context | 29.2 tokens/s | - | @catonooka (v1.1.9, #187) |
| **2x Tesla P40 24 GB** (Pascal, PCIe 3.0 x16), EPYC 7302P (a VM), 184 GB RAM | Maya-S24 / Maya-L | 28.0 / 16.9 tokens/s | 202 / 194 tokens/s | @Jw88888 (v1.1.11 + #214, #212) |
| **1x RTX 3090 24 GB** (PCIe 4.0 x16), EPYC 7702P (8-channel DDR4), 141 GB RAM (a VM) | Maya-M, 128K context | 27.0 tokens/s | 1069 tokens/s | @erikash (v1.1.4, #157) |
| **1x Tesla V100 32 GB** (PCIe 3), 2x Xeon Gold 6138, 183 GB RAM | Maya-S24 | 26.8 tokens/s | 541 tokens/s | maintainer (v1.1.7, #171) |
| **RTX 4080 SUPER 32 GB** (a 32 GB modded card), Threadripper 3970X, 157 GB RAM, SATA SSD | Maya-S24, 262K context | 26.7 tokens/s | 1552 tokens/s | @Micro-ATP (v1.1.11, #200) |
| **3x Tesla V100 16 GB** (PCIe 3.0 x16), 2x Xeon E5-2690 v4, 62 GB RAM | Maya-S, 128K context | 26.6 tokens/s | 497 tokens/s | @saatvik-viniak (v1.1.11, #211) |
| **4x RTX 3080 20 GB**, 64 GB DDR4 | Maya-M, 512K context | 26.3 tokens/s (a day of use) | - | @CoolJohn-lab (#104) |
| **1x RTX 3090 24 GB** (power-capped at 225 W), Threadripper 3970X, 128 GB RAM | Maya-S24 | 26.0 tokens/s | 913 tokens/s | @briepala (v1.1.2, #135) |
| **RTX PRO 4000 Blackwell 24 GB**, Core i9-13900K, 128 GB DDR5 | Maya-S24 | 24.7 tokens/s | 1146 tokens/s | @morrisfamily (v1.1.7, #174) |
| **RTX 5090 32 GB** (PCIe 5.0 x16), Core i9-13900K, 64 GB RAM, Windows 11 | Maya-S, 128K context | 23.0 tokens/s | 2173 tokens/s | @gianbing (v1.1.3, #139) |
| **4x Tesla T4 16 GB**, Xeon E5-2660 v2, 220 GB RAM | Maya-S | 22.6 tokens/s | 416 tokens/s | @ksanislo (v1.0.16, #36) |
| **Radeon AI PRO R9700 32 GB** (gfx1201), Core i5-14400F, 128 GB RAM | Maya-S | 21.4 tokens/s | - | @potatopote-debug (v1.1.12, #213) |
| **1x NVIDIA CMP 170HX 64 GB** (PCIe Gen2 x16), Core i7-9700K, 32 GB RAM | Maya-S24, 32K context | 20.6 tokens/s | 797 tokens/s | @Kanorin-chan (v1.1.8, #281) |
| **1x Tesla V100 32 GB** (PCIe 3), Core i5-12600T, 64 GB RAM | Maya-S | up to 19 tokens/s | up to 620 tokens/s | maintainer |
| **AMD Radeon 8060S** (Ryzen AI Max+ 395, 128 GB, 96 GB of it for the GPU), Windows 11 | Maya-S | 18.3 tokens/s | 320 tokens/s | @LIN-LAB-AI (v1.0.30, #102) |
| **RTX 3090 Ti 24 GB** (PCIe 3.0 x16, a VM), EPYC 7302P, 240 GB RAM | Maya-L, 32K context | 18.2 tokens/s (tuned, #177) | 791 tokens/s | @Jw88888 (v1.1.7, #177) |
| **RTX 4090 24 GB**, Ryzen 7 9700X, 64 GB RAM | Maya-S24, 32K context | 16.6 tokens/s | 1320 tokens/s | @MagikEh (#274) |
| **NVIDIA DGX Spark** (GB10, 121 GB unified memory) | Maya-S, 128K context | 16.5 tokens/s | ~730 tokens/s (6K prompt) | @ericoh88 (v1.1.15, #282) |
| **RTX 4090 24 GB**, Ryzen 9 9950X3D, 192 GB DDR5, Windows 11 | Maya-L, 128K context | 16.4 tokens/s | 1371 tokens/s | @npc97 (v1.0.24, #63) |
| **RTX 5090 Laptop 24 GB**, Core Ultra 9 275HX, 64 GB RAM, Windows 11 (a laptop) | Maya-S24 | 14.5 tokens/s | 994 tokens/s | @klvnblst (v1.0.30, #67) |

Prefill is an 8K-token prompt. The speed holds as a conversation grows, and the first answers after a start are the
slowest while the caches fill with what you use. Share yours: `--bench`, then `--report`, in a
[GitHub issue](https://github.com/mw00/project-maya/issues).

**Several requests at once** (`--parallel`) add up: on one GPU a token of each goes through the model together, so
the dense weights are read once for all of them. In tokens per second for all of them together, against the same
questions one after another: x1.41 for two and x1.35 for four on a V100 that streams most of Maya-S's experts from
RAM, x1.67 for two and x2.02 for four on a Gorgon Halo (Radeon 8065S, 160 GB for the GPU) that holds every expert in
GPU memory. The more of the model
VRAM holds, the more it gains ([how](docs/SETTINGS.md#several-requests-at-once)).

## What you need

| | |
| --- | --- |
| **GPU** | NVIDIA, compute capability 7.0 or newer (V100, RTX 20 and newer): one, or up to 16 that share the model. More VRAM is faster. Pascal (Tesla P40 / P100, GTX 10) is experimental, not validated yet. **AMD (experimental):** RX 7900 XT / XTX, Radeon AI PRO R9700 / RX 9070, Strix Halo and Gorgon Halo (Radeon 8060S / 8065S); RX 6800 / 6900 and Radeon PRO V620 / W6800 not validated yet; GLM-5.3-Flash reads text only there, DeepSeek-V4.1-Flash-1017 pictures too - [docs/AMD_MAYA.md](docs/AMD_MAYA.md) |
| **RAM** | 32 GB runs it. More RAM keeps more experts close and is faster |
| **Disk** | **A fast NVMe SSD** with room for the model (~100 GB for Maya-S, ~160 GB for Maya-L, ~350 GB for DeepSeek-V4.1-Flash-1017's Maya-S, ~520 GB for unsloth's GGUF) and its picture encoder (1.1 GB). Not a hard disk |
| **System** | Linux (x86-64; a CPU with AVX2 is best), the NVIDIA driver, CUDA 12.x (CUDA 13 works for RTX 20 and newer, not for the V100 or Pascal), g++, Python 3.10+. Windows 10/11 (experimental): [docs/WINDOWS.md](docs/WINDOWS.md). Not WSL2. AMD: ROCm 7 |

The setup checks all of this, prints the exact command for anything missing, and installs nothing system-wide.

## Install

```sh
git clone https://github.com/mw00/project-maya.git && cd project-maya
./setup.sh
```

Answer a few questions (Enter takes the recommended choice). The setup shows every download before it starts it,
builds everything once (20-40 minutes plus the download) and **starts the model**. Then open
`http://127.0.0.1:8080`. Ctrl+C stops it, and `./setup.sh` starts it again right away.

- **Windows:** `START-MAYA.bat` instead of `./setup.sh` - see [docs/WINDOWS.md](docs/WINDOWS.md).
- **AMD:** [docs/AMD_MAYA.md](docs/AMD_MAYA.md).
- **Updating:** the dashboard's **About > Update**, or `git pull` then `./setup.sh`. The model isn't downloaded again.
- **Every option** (GPUs, context, a model folder you already have, remote access, ...): [docs/SETUP.md](docs/SETUP.md).

> **The start takes a few minutes:** the engine pins most of the free RAM (all but about 6 GB) for its expert tier,
> so other programs get little RAM while Maya runs.

## Using it

- **In the browser:** chat (kept in your browser, per-chat instructions, edit and regenerate, HTML previews), a live
  **Monitor** of the model and your hardware (down to what fills each GPU's memory), and a context size and port you
  can change in Settings.
- **Your apps and coding agents:** an OpenAI-compatible API at `http://127.0.0.1:8080/v1`, and Anthropic's at
  `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`). Tool calls are
  streamed as the model writes them, and a long prompt's reading progress too when a request asks for it
  (`"return_progress": true`). A web page served from another address can call it too once you allow its
  address (`"cors_origins"`, [docs/SETUP.md](docs/SETUP.md#web-pages-on-another-address)).
- **Responses API:** `/v1/responses` supports text/image conversations, streaming,
  function tool calls, schema-validated output, background tasks with cancellation/replay,
  and optional shared in-memory history (`store:true`, `previous_response_id`).
  Responses are not stored by default. Options and limits: [docs/RESPONSES.md](docs/RESPONSES.md).
- **Thinking:** Off, Low, High (the default) or Max, in Settings or per request.
- **Pictures:** attach one in the chat, or send image parts through either API.
- **From another device:** `./maya.sh --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **Tuning:** `./maya.sh --calibrate` tunes the engine to your hardware in 10-15 minutes. Every setting:
  [docs/SETTINGS.md](docs/SETTINGS.md).
- **DeepSeek on large-RAM Linux hosts:** optional [locked Engram tables and strict CPU-expert RAM placement](docs/SETTINGS.md#deepseek-explicit-ram-residency). Both are off by default; they need enough physical RAM.
- **Several requests at once** (GLM-5.3-Flash): `./maya.sh --parallel 4` (Windows: `START-MAYA.bat --parallel 4`; up to 8) keeps
  it in the config, and the dashboard shows how many answers are being written. Each extra request keeps its own
  context in VRAM, so a few fewer experts fit there ([how](docs/SETTINGS.md#several-requests-at-once)).
- **Long contexts** (GLM-5.3-Flash): new setups keep each attention layer's cache in RAM with a window of it in VRAM
  (`--kv-streaming on`; `./maya.sh --setup` adds it to an existing config), so the VRAM holds more experts - at 512K
  on an RTX 3090, prompts 4-9x faster and decode +5%; 1M context fits one 3090
  ([how](docs/SETTINGS.md)).
- **Faster answers with a DFlash2 drafter** (GLM-5.3-Flash): `./maya.sh --setup --draft dflash2` (NVIDIA RTX 30 / A100 and newer)
  downloads a small drafter that drafts up to 7 tokens a round and gives the same answers - Maya-L +44% on nine GPUs
  and +12% on two (against the MTP block), +19% on one RTX 3090 ([how](docs/SETTINGS.md#drafting-with-dflash2)). How
  much it gains depends on the PC: on one GPU that reads much of the model from RAM it is often slower. So a
  [speed calibration](docs/SETTINGS.md#the-speed-calibration) measures it on your PC and keeps it only where it is
  faster. The drafter's license is non-commercial (CC BY-NC-ND 4.0).

## Something went wrong?

Run **`./maya.sh --report`** (Windows: `START-MAYA.bat --report`) right after a slow answer or an error, and attach the
`maya-report.txt` it writes to a [GitHub issue](https://github.com/mw00/project-maya/issues). It holds your hardware,
your setup and the engine's speed lines; nothing is sent anywhere by itself. The engine's own log is
`maya-<model>.log` in the Maya folder.

- **"nvcc ... cannot build for these GPUs":** Volta needs CUDA 12.x, Blackwell 12.8 or newer. The setup takes the
  newest installed toolkit that fits.
- **`cuda_profiler_api.h` or `nvtx3/nvToolsExt.h` is missing:** Maya's CUDA build needs the profiler API
  and NVTX headers as well as `nvcc` and the runtime. Install those development components for your toolkit.
  With a split CUDA 13 toolkit, NVIDIA's official header wheels can also stay in a project-local folder:

  ```sh
  python3 -m pip install --target .cuda-headers \
    nvidia-cuda-profiler-api==13.3.27 nvidia-nvtx==13.3.29
  CPATH="$PWD/.cuda-headers/nvidia/cu13/include${CPATH:+:$CPATH}" ./maya.sh --setup
  ```

  Select wheel versions that match your toolkit. This include layout was checked with CUDA 13.3,
  `nvidia-cuda-profiler-api==13.3.27` and `nvidia-nvtx==13.3.29`; other CUDA major versions can use a
  different layout. Use the same `CPATH` when rebuilding after an update. No shared toolkit files need changing.
- **"unsupported GNU version":** `./maya.sh --setup --host-compiler g++-12` (install `g++-12` first).
- **Slow, with the SSD busy all the time:** not enough free RAM for the experts. Close programs or add RAM.
- **Windows: answers suddenly several times slower:** the card is too full and Windows moves GPU memory to RAM. Leave
  more of it free: `START-MAYA.bat --setup --vram-reserve-mib 2048`
  ([the expert pool's size](docs/SETTINGS.md#the-expert-pools-size)).
- **Windows on a Strix Halo / Gorgon Halo: other programs run out of memory, or the setup warns about the page
  file:** set the page file to the size the setup prints ([docs/AMD_MAYA.md](docs/AMD_MAYA.md)).
- **The download stopped:** run `./maya.sh` again; it continues where it stopped.
- **Port 8080 is in use:** Maya is already running, or `--port 8081` starts another one. Settings > Port moves Maya to
  another port for good (it starts again there).

## Support

Project Maya is free and open source. If it's useful to you, you can support it here - thank you!

<p align="center"><a href="https://buymeacoffee.com/peasantsmith"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a></p>

Speed reports, bug reports and pull requests are welcome: [CONTRIBUTING.md](CONTRIBUTING.md) says what helps most.

## Credits and license

- **People:** Project Maya is created and led by Peasant Smith ([@mw00](https://github.com/mw00),
  [X](https://x.com/PeasantSmith)), who owns the project, its name and the Maya models.
  [@needmorevram](https://github.com/needmorevram) ([X](https://x.com/needmorevram)) is a co-maintainer and
  contributed major features: DFlash2 drafting, EXL3 support, KV streaming,
  conversation checkpoints, the layer-split search, the expert profile and the CPU expert kernels, and for
  `maya-ds4` the 3.5-bit EXL3 option, tensor parallel over 4 / 8 GPUs and the threaded expert-cache swaps (#276). Thanks also to everyone who sent code and speed reports
  ([contributors](https://github.com/mw00/project-maya/graphs/contributors)).
- **Built on [Strata](https://github.com/Niko1221/Strata)** (MIT License, Copyright (c) 2026 Niko1221 and the Strata
  contributors): Maya's engine started from Strata's and was rewritten for GLM-5.3-Flash (the expert tiers across VRAM,
  RAM and SSD, the layer split, MTP decoding), and its server and dashboard grew from Strata's. **And on
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT License, Copyright (c) 2023-2026 The ggml
  authors): the quantization formats, the CPU dot products and the prompt path's MMQ kernels, built from a pinned
  commit (`third_party/ggml/LICENSE`).
- **EXL3 support** follows [exllamav3](https://github.com/turboderp-org/exllamav3)'s format, codebooks, CPU expert
  kernel and tensor-core decoding of a tile (MIT License, Copyright (c) 2025 Turboderp; `third_party/exllamav3/LICENSE`). **The DFlash2 drafter's
  forward** follows [beellama.cpp](https://github.com/Anbeeld/beellama.cpp)'s (MIT License, Copyright (c) 2023-2026
  The ggml authors, Anbeeld; `third_party/beellama.cpp/LICENSE`).
- The models: [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by zai-org (MIT); Maya's quants and the
  image encoder file are made from zai-org's released weights and keep its license.
  [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) by DeepSeek (MIT); Maya's
  quant, its Engram tables and the drafter are made from DeepSeek's released weights and keep its license; the vision
  tower comes from DeepSeek's checkpoint, the GGUF option from
  [unsloth's GGUFs](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF), the EXL3 options are
  LibertAIDAI's ([2.9](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw) and
  [3.5 bpw](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw)). The optional DFlash2 drafter
  ([GLM-5.3-Flash-DFlash2-GGUF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF) by Anbeeld) is CC BY-NC-ND
  4.0, non-commercial use only, and is downloaded only when you choose it.
- The dashboard's font: Outfit (SIL Open Font License 1.1, `serve/web/fonts/OFL.txt`).
- Maya is open source under the [MIT License](LICENSE); the notices of Strata, ggml, exllamav3 and beellama.cpp stay
  with every copy.

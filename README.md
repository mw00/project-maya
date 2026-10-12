<h1 align="center">Project Maya</h1>

<p align="center"><b>A 321-billion-parameter AI model on your own GPU</b><br>
GLM-5.3-Flash, private and fast · one NVIDIA GPU or up to 16 · AMD and Windows (experimental) · chat, pictures,
OpenAI- and Anthropic-compatible API · several requests at once</p>

<p align="center"><a href="https://buymeacoffee.com/peasantsmith">☕ Support Project Maya - buy me a coffee</a></p>

Models this size usually need a datacenter. Maya runs **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)**
(zai-org, MIT license) on the hardware you already have: 321 B parameters, about 18 B active per token, and a context
of up to 1 M tokens. Its engine is built for this one model. It keeps the experts your conversation uses most on the
GPU, the next ones in RAM and the rest on your NVMe SSD, and moves them as you chat. It measures your GPUs, CPU, RAM
and SSD and tunes itself to them. Nothing leaves your machine.

- **Fast:** up to 118 tokens/s on four RTX 4090s, 30+ on a single RTX 5090. Real numbers from real users below.
- **Close to the original:** Maya's own quants keep 97.7-99.2% of the FP8 model's zero-shot accuracy.
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

## How fast is it?

Real machines, real numbers, most of them measured with `./maya.sh --bench`. A token is about ¾ of a word.

| Machine | Model | Decode (writing the answer) | Prefill (reading your prompt) | By |
| --- | --- | ---: | ---: | --- |
| **4x RTX 4090 24 GB** (PCIe 4.0 x16), Xeon w7-3445, 62 GB RAM | Maya-S24 | **118 tokens/s** | 4129 tokens/s | @0xPreDa (v1.0.31, #89) |
| **2x NVIDIA CMP 170HX 64 GB** (PCIe Gen2 x4 each), 2x Xeon E5-2690 v4, 91 GB RAM | Maya-M, every expert in VRAM | 61.3 tokens/s | 514 tokens/s | @ZackO2o (v1.0.6, #22) |
| **2x RTX 4090 24 GB** (PCIe 3.0 + 4.0 x16), 2x Xeon Silver 4309Y, 188 GB RAM | Maya-S24 | 40.5 tokens/s | 1271 tokens/s | @fuad00 (v1.1.0, #122) |
| **2x Tesla V100 32 GB** (PCIe 3), Xeon E5-2690 v4, 30 GB RAM | Maya-S | up to 40 tokens/s | up to 670 tokens/s | maintainer |
| **RTX 5090 32 GB**, Ryzen 9 7950X, 96 GB DDR5 | Maya-M | 30+ tokens/s | ~2000 tokens/s | @siganos (#43) |
| **4x RTX 3080 20 GB**, 64 GB DDR4 | Maya-M, 512K context | 26.3 tokens/s (a day of use) | - | @CoolJohn-lab (#104) |
| **4x Tesla T4 16 GB**, Xeon E5-2660 v2, 220 GB RAM | Maya-S | 22.6 tokens/s | 416 tokens/s | @ksanislo (v1.0.16, #36) |
| **1x Tesla V100 32 GB** (PCIe 3), Core i5-12600T, 64 GB RAM | Maya-S | up to 19 tokens/s | up to 620 tokens/s | maintainer |
| **AMD Radeon 8060S** (Ryzen AI Max+ 395, 128 GB, 96 GB of it for the GPU), Windows 11 | Maya-S | 18.3 tokens/s | 320 tokens/s | @LIN-LAB-AI (v1.0.30, #102) |
| **RTX 4090 24 GB**, Ryzen 9 9950X3D, 192 GB DDR5, Windows 11 | Maya-L, 128K context | 16.4 tokens/s | 1371 tokens/s | @npc97 (v1.0.24, #63) |
| **RTX 5090 Laptop 24 GB**, Core Ultra 9 275HX, 64 GB RAM, Windows 11 (a laptop) | Maya-S24 | 14.5 tokens/s | 994 tokens/s | @klvnblst (v1.0.30, #67) |

Prefill is an 8K-token prompt. The speed holds as a conversation grows, and the first answers after a start are the
slowest while the caches fill with what you use. Share yours: `--bench`, then `--report`, in a
[GitHub issue](https://github.com/mw00/project-maya/issues).

**Several requests at once** (`--parallel`) add up: on one GPU a token of each goes through the model together, so
the dense weights are read once for all of them. In tokens per second for all of them together, against the same
questions one after another: x1.41 for two and x1.35 for four on a V100 that streams most of Maya-S's experts from
RAM, x1.67 for two and x2.02 for four on a Strix Halo that holds every expert in GPU memory. The more of the model
VRAM holds, the more it gains ([how](docs/SETTINGS.md#several-requests-at-once)).

## What you need

| | |
| --- | --- |
| **GPU** | NVIDIA, compute capability 7.0 or newer (V100, RTX 20 and newer): one, or up to 16 that share the model. More VRAM is faster. **AMD (experimental):** RX 7900 XT / XTX, Radeon AI PRO R9700 / RX 9070, Strix Halo and Gorgon Halo (Radeon 8060S / 8065S), text only - [docs/AMD_MAYA.md](docs/AMD_MAYA.md) |
| **RAM** | 32 GB runs it. More RAM keeps more experts close and is faster |
| **Disk** | **A fast NVMe SSD** with room for the model (~100 GB for Maya-S, ~160 GB for Maya-L) and its picture encoder (1.1 GB). Not a hard disk |
| **System** | Linux (x86-64; a CPU with AVX2 is best), the NVIDIA driver, CUDA 12.x (CUDA 13 works for RTX 20 and newer, not for the V100), g++, Python 3.10+. Windows 10/11 (experimental): [docs/WINDOWS.md](docs/WINDOWS.md). Not WSL2. AMD: ROCm 7 |

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
  **Monitor** of the model and your hardware, and a context size you can change in Settings.
- **Your apps and coding agents:** an OpenAI-compatible API at `http://127.0.0.1:8080/v1`, and Anthropic's at
  `http://127.0.0.1:8080/v1/messages` (Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`). Tool calls are
  streamed as the model writes them. A web page served from another address can call it too once you allow its
  address (`"cors_origins"`, [docs/SETUP.md](docs/SETUP.md#web-pages-on-another-address)).
- **Thinking:** Off, Low, High (the default) or Max, in Settings or per request.
- **Pictures:** attach one in the chat, or send image parts through either API.
- **From another device:** `./maya.sh --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **Tuning:** `./maya.sh --calibrate` tunes the engine to your hardware in 10-15 minutes. Every setting:
  [docs/SETTINGS.md](docs/SETTINGS.md).
- **Several requests at once:** `./maya.sh --parallel 4` (Windows: `START-MAYA.bat --parallel 4`; up to 8) keeps
  it in the config, and the dashboard shows how many answers are being written. Each extra request keeps its own
  context in VRAM, so a few fewer experts fit there ([how](docs/SETTINGS.md#several-requests-at-once)).
- **Faster answers with a DFlash2 drafter:** `./maya.sh --setup --draft dflash2` (NVIDIA RTX 30 / A100 and newer)
  downloads a small drafter that drafts up to 7 tokens a round and gives the same answers - Maya-L +44% on nine GPUs
  and +12% on two (against the MTP block), +19% on one RTX 3090 ([how](docs/SETTINGS.md#drafting-with-dflash2)). The
  drafter's license is non-commercial (CC BY-NC-ND 4.0).

## Something went wrong?

Run **`./maya.sh --report`** (Windows: `START-MAYA.bat --report`) right after a slow answer or an error, and attach the
`maya-report.txt` it writes to a [GitHub issue](https://github.com/mw00/project-maya/issues). It holds your hardware,
your setup and the engine's speed lines; nothing is sent anywhere by itself. The engine's own log is
`maya-<model>.log` in the Maya folder.

- **"nvcc ... cannot build for these GPUs":** Volta needs CUDA 12.x, Blackwell 12.8 or newer. The setup takes the
  newest installed toolkit that fits.
- **"unsupported GNU version":** `./maya.sh --setup --host-compiler g++-12` (install `g++-12` first).
- **Slow, with the SSD busy all the time:** not enough free RAM for the experts. Close programs or add RAM.
- **The download stopped:** run `./maya.sh` again; it continues where it stopped.
- **Port 8080 is in use:** Maya is already running, or `--port 8081` starts another one.

## Support

Project Maya is free and open source. If it's useful to you, you can support it at
**[buymeacoffee.com/peasantsmith](https://buymeacoffee.com/peasantsmith)**. Thank you!
Speed reports, bug reports and pull requests are welcome: [CONTRIBUTING.md](CONTRIBUTING.md) says what helps most.

## Credits and license

- **Built on [Strata](https://github.com/Niko1221/Strata)** (MIT License, Copyright (c) 2026 Niko1221 and the Strata
  contributors): Maya's engine started from Strata's and was rewritten for GLM-5.3-Flash (the expert tiers across VRAM,
  RAM and SSD, the layer split, MTP decoding), and its server and dashboard grew from Strata's. **And on
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT License, Copyright (c) 2023-2026 The ggml
  authors): the quantization formats, the CPU dot products and the prompt path's MMQ kernels, built from a pinned
  commit (`third_party/ggml/LICENSE`).
- **EXL3 support** follows [exllamav3](https://github.com/turboderp-org/exllamav3)'s format, codebooks and CPU expert
  kernel (MIT License, Copyright (c) 2025 Turboderp; `third_party/exllamav3/LICENSE`). **The DFlash2 drafter's
  forward** follows [beellama.cpp](https://github.com/Anbeeld/beellama.cpp)'s (MIT License, Copyright (c) 2023-2026
  The ggml authors, Anbeeld; `third_party/beellama.cpp/LICENSE`).
- The model: [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by zai-org (MIT); Maya's quants and the
  image encoder file are made from zai-org's released weights and keep its license. The optional DFlash2 drafter
  ([GLM-5.3-Flash-DFlash2-GGUF](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF) by Anbeeld) is CC BY-NC-ND
  4.0, non-commercial use only, and is downloaded only when you choose it.
- The dashboard's font: Outfit (SIL Open Font License 1.1, `serve/web/fonts/OFL.txt`).
- Maya is open source under the [MIT License](LICENSE); the notices of Strata, ggml, exllamav3 and beellama.cpp stay
  with every copy.

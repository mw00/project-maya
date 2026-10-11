<h1 align="center">Project Maya</h1>

<p align="center"><b>Run GLM-5.3-Flash - a 321-billion-parameter AI model - on your own GPU(s)</b><br>
One NVIDIA GPU or several (up to 16), AMD (experimental) · Linux, Windows (experimental) · chat in the browser, pictures, OpenAI- and
Anthropic-compatible API</p>

<p align="center"><a href="https://buymeacoffee.com/peasantsmith">☕ Support Project Maya - buy me a coffee</a></p>

Maya runs **[GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash)** (zai-org, MIT license): a mixture-of-experts
model with 321 B parameters, of which about 18 B work on each token, and a context of up to 1 M tokens. Models this
size normally need a server with hundreds of GB of GPU memory. Maya's engine keeps the most-used experts on your
GPU(s), the next ones in RAM and the rest on your NVMe SSD, and moves them as the conversation needs them. Nothing
leaves your machine.

## The models

Project Maya's own quants of GLM-5.3-Flash, made from Z.ai's FP8 release - the precision the model is served at - with
statistics from the FP8 model and error-feedback rounding of the experts. Each keeps the model's MTP block, which
drafts tokens ahead.

| Model | Size | Made for | Against the FP8 model |
| --- | ---: | --- | --- |
| **Maya-S** (default) | 96.5 GB | PCs with a smaller memory pool (RAM and VRAM together) | 97.9% of its zero-shot accuracy; the same next token 83% of the time |
| **Maya-S24** | 94.7 GB | cards of 24 GB or less (the setup recommends it there): 4-bit attention leaves ~1.5 GB more of the card for experts, decode +14% | 97.7%; 83% |
| **Maya-M** | 116 GB | a bigger memory pool; more bits where they count, calibrated toward tool calls and front-end code | 97.9%; 86% (KL divergence 23% below Maya-S's) |
| **Maya-M-Derisked** | 116 GB | Maya-M with a weight edit by [Blackfrost_AI](https://x.com/Blackfrost_AI) that reduces blanket refusals ([its repo](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF)); experimental | not measured yet |
| **Maya-L** | 156.3 GB | the biggest memory pools | **99.2%**; 90% |

The setup asks which one to download; `./setup.sh --setup --model Maya-M` (Windows: `START-MAYA.bat --setup --model
Maya-M`) switches later. How they are made and measured (zero-shot tasks, KL divergence, long answers):
[MAYA-S](bench/results/MAYA-S.md), [MAYA-S24](bench/results/MAYA-S24.md), [MAYA-M](bench/results/MAYA-M.md),
[MAYA-L](bench/results/MAYA-L.md). The models are [on Hugging Face](https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-GGUF).

## How fast is it?

A token is about ¾ of a word. `./maya.sh --bench` measures your machine the same way.

| Machine | Model | Decode (writing the answer) | Prefill (reading your prompt) | By |
| --- | --- | ---: | ---: | --- |
| **4x RTX 4090 24 GB** (PCIe 4.0 x16), Xeon w7-3445, 62 GB RAM | Maya-S24 | **118 tokens/s** | 4129 tokens/s | @0xPreDa (v1.0.31, #89) |
| **2x NVIDIA CMP 170HX 64 GB** (PCIe Gen2 x4 each), 2x Xeon E5-2690 v4, 91 GB RAM | Maya-M, every expert in VRAM | 61.3 tokens/s | 514 tokens/s | @ZackO2o (v1.0.6, #22) |
| **2x Tesla V100 32 GB** (PCIe 3), Xeon E5-2690 v4, 30 GB RAM | Maya-S | up to 40 tokens/s | up to 670 tokens/s | maintainer |
| **RTX 5090 32 GB**, Ryzen 9 7950X, 96 GB DDR5 | Maya-M | 30+ tokens/s | ~2000 tokens/s | @siganos (#43) |
| **4x Tesla T4 16 GB**, Xeon E5-2660 v2, 220 GB RAM | Maya-S | 22.6 tokens/s | 416 tokens/s | @ksanislo (v1.0.16, #36) |
| **1x Tesla V100 32 GB** (PCIe 3), Core i5-12600T, 64 GB RAM | Maya-S | up to 19 tokens/s | up to 620 tokens/s | maintainer |
| **RTX 4090 24 GB**, Ryzen 9 9950X3D, 192 GB DDR5, Windows 11 | Maya-L, 128K context | 16.4 tokens/s | 1371 tokens/s | @npc97 (v1.0.24, #63) |
| **RTX 5090 Laptop 24 GB**, Core Ultra 9 275HX, 64 GB RAM, Windows 11 (a laptop) | Maya-S24 | 14.5 tokens/s | 994 tokens/s | @klvnblst (v1.0.30, #67) |

Prefill is an 8K-token prompt. The speed holds as a conversation grows (the attention's selection step is linear in
the context), and the first answers after a start are the slowest while the expert caches fill with what you use.
Every machine is different: the engine measures the GPUs, RAM, CPU and SSD it finds and adapts to them. Send your
numbers: `--bench`, then `--report`, in a [GitHub issue](https://github.com/mw00/project-maya/issues).

## What you need

| | |
| --- | --- |
| **GPU** | NVIDIA, compute capability 7.0 or newer (V100, RTX 20 and newer): one, or up to 16 that share the model. More VRAM is faster. **AMD (experimental):** RX 7900 XT / XTX, Radeon AI PRO R9700 / RX 9070 (one or two) and Strix Halo / Gorgon Halo, Radeon 8060S / 8065S, text only - [docs/AMD_MAYA.md](docs/AMD_MAYA.md) |
| **RAM** | 32 GB runs it (the 2x V100 machine above has 30 GB). More RAM keeps more experts close and is faster |
| **Disk** | **A fast NVMe SSD** with room for the model (~100 GB for Maya-S, ~120 GB for Maya-M, ~160 GB for Maya-L) and its pictures encoder (1.1 GB): the engine reads from it while it answers. Not a hard disk |
| **System** | Linux (x86-64; a CPU with AVX2 is best), the NVIDIA driver, CUDA toolkit 12.x (CUDA 13 works for RTX 20 and newer, not for the V100), g++, Python 3.10+. Windows 10/11: experimental, with Visual Studio 2022 Build Tools ([docs/WINDOWS.md](docs/WINDOWS.md)). Not WSL2. AMD: ROCm 7 |

The setup checks all of this and prints the exact command for anything missing. It installs nothing system-wide.

## Install

```sh
git clone https://github.com/mw00/project-maya.git && cd project-maya
./setup.sh
```

The setup runs on a screen of its own: answer a few questions with the arrow keys and Enter, or press Enter each time
for the recommended choice - which GPUs, how much context, which model, pictures. It shows every download before it
starts it, builds everything (20-40 minutes plus the download, once) and **starts the model**. Then open the dashboard
at `http://127.0.0.1:8080`. Ctrl+C stops it; `./setup.sh` again starts it right away.

- **Windows:** `START-MAYA.bat` instead of `./setup.sh` - [docs/WINDOWS.md](docs/WINDOWS.md) lists what to install
  first.
- **AMD:** `./maya.sh --backend hip --gpu 0 --check` first, then [docs/AMD_MAYA.md](docs/AMD_MAYA.md).
- **Updating:** the dashboard's **About > Update** (it says when a new version is out), or `git pull` then
  `./setup.sh`. The engine recompiles what changed; the model is not downloaded again.
- **More:** what the first run does, every option (`--gpus`, `--context`, `--gguf-dir` for GGUF files you already
  have, `--models-dir`, `--host` and `--api-key`, ...): [docs/SETUP.md](docs/SETUP.md).

> **The start takes a few minutes**: the engine pins most of the free RAM (all but about 6 GB) for its expert tier
> and warms its caches. Other programs get little RAM while Maya runs.

## Using it

- **In the browser:** `http://127.0.0.1:8080` - **Chat** (your chats kept in this browser, instructions per chat,
  edit and regenerate, a preview for HTML pages), a live **Monitor** of the model, its expert caches and your
  GPU/CPU/RAM (with **Copy report**), and **About**. The chat's **Settings** change the context size: the model
  reloads with it in a minute or two.
- **Your apps and coding agents:** an OpenAI-compatible provider at `http://127.0.0.1:8080/v1` (any model name; any
  API key unless you set one), or Anthropic's API at `http://127.0.0.1:8080/v1/messages` (Claude Code:
  `ANTHROPIC_BASE_URL=http://127.0.0.1:8080`). Tool calls work through both, streamed as the model writes them.
- **Thinking:** GLM's own levels - Off, Low, High (the default) or Max - in the chat's Settings, or the request's
  reasoning effort (`none`, `low`, `high`, `max`). A reasoning block is capped at 32K tokens
  (`"thinking_budget"` in the config; 0 = no cap).
- **Pictures:** attach one in the chat, or send `image_url` parts (OpenAI) / `image` blocks (Anthropic). The image
  encoder runs only while a new picture is read, in GPU memory the model lends it.
- **From another device:** `./maya.sh --setup --host 0.0.0.0 --api-key <secret>`. Always set a key.
- **Several requests:** one at a time by default, the others in turn. `STRATA_FAIR_SLICE_S` lets a short request in
  during a long answer, and on two GPUs or more `STRATA_GLM_SEQS` answers several at once
  ([docs/SETTINGS.md](docs/SETTINGS.md)).

## Tuning

The engine sizes itself for your PC. **`./maya.sh --calibrate`** (also offered at the end of the setup) tunes its CPU
lane on your hardware in about 10-15 minutes and keeps a setting only when it is more than 3% faster. Every engine
setting, and what it does: [docs/SETTINGS.md](docs/SETTINGS.md).

## Something went wrong?

Run **`./maya.sh --report`** (Windows: `START-MAYA.bat --report`) right after a slow answer or an error, and attach
the `maya-report.txt` it writes to a [GitHub issue](https://github.com/mw00/project-maya/issues): your GPUs, CPU, RAM
and disks, your setup and the engine's speed lines. Nothing is sent anywhere by itself; your home folder shows as `~`
and no API key is included. For a speed report, also run **`./maya.sh --bench`** with Maya stopped.

- **"nvcc ... cannot build for these GPUs"** - Volta needs CUDA 12.x; Blackwell needs 12.8 or newer. Several
  toolkits can be installed side by side; the setup takes the newest that fits.
- **"unsupported GNU version"** while compiling - `./maya.sh --setup --host-compiler g++-12` (install `g++-12` first).
- **Slow, and the SSD is busy all the time** - not enough free RAM for the experts: close programs, or add RAM.
- **The download stopped** - run `./maya.sh` again: it continues where it stopped.
- **Port 8080 is in use** - Maya (or another server) is already running; `--port 8081` starts another one.
- **A model downloaded before 2026-10-09** names its architecture `glm5next`: Maya reads it, and
  `python tools/gguf_fix_arch.py <the first .gguf> --in-place` renames it to the standard `glm5-next`.
- The engine's log is `maya-<model>.log` in the Maya folder.

## Support

Project Maya is free and open source. If it is useful to you, you can support its development:
**[buymeacoffee.com/peasantsmith](https://buymeacoffee.com/peasantsmith)**. Thank you!

Speed reports, bug reports and pull requests are welcome: [CONTRIBUTING.md](CONTRIBUTING.md) says what helps most.

## Credits and license

- **Built on [Strata](https://github.com/Niko1221/Strata)** (MIT License, Copyright (c) 2026 Niko1221 and the Strata
  contributors): Maya's engine started from Strata's and was rewritten for GLM-5.3-Flash (the expert tiers across VRAM,
  RAM and SSD, the layer split, MTP decoding), and its server and dashboard grew from Strata's. **And on
  [ggml / llama.cpp](https://github.com/ggml-org/llama.cpp)** (MIT License, Copyright (c) 2023-2026 The ggml
  authors): the quantization formats, the CPU dot products and the prompt path's MMQ kernels, built from a pinned
  commit (`third_party/ggml/LICENSE`).
- The model: [GLM-5.3-Flash](https://huggingface.co/zai-org/GLM-5.3-Flash) by zai-org (MIT); Maya's quants and the
  image encoder file are made from zai-org's released weights and keep its license.
- The dashboard's font: Outfit (SIL Open Font License 1.1, `serve/web/fonts/OFL.txt`).
- Maya is open source under the [MIT License](LICENSE); the notices of Strata and ggml stay with every copy.

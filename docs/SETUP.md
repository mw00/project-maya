# Setting up Maya

The setup in detail: what its first run does, starting again and updating, and its options. Back to the
[README](../README.md).

## What the first run does

It first asks which model to set up:

- **GLM-5.3-Flash** - Project Maya's GGUF quants (or turboderp's EXL3 quants), from one GPU up; the steps below;
- **DeepSeek-V4.1-Flash-1017** (experimental) - [LibertAIDAI's EXL3 quant at 2.9 bits](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
  (211 GB) and its two Engram tables from [DeepSeek's repo](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  (203 GB), on its own engine (`build/strata-ds4`): Linux, NVIDIA RTX 30 / A100 or newer. It is fast with about 230 GB
  of VRAM and RAM together; with less, the setup warns and the engine reads the experts that do not fit from the
  model's files on the SSD (`--mmap-experts auto`), much slower - one RTX 3090 with 64 GB of RAM: prompts about 50
  tokens/s, answers 4-5 tokens/s (faster with `--dspark none`). The same steps with these differences: the
  contexts are 32K-256K (64K recommended); step 5 shows both
  sources and checks every file against its pinned size (and, once downloaded, its sha256: `tools/ds4/download.json`);
  `--model-dir DIR` uses files you already have; the pack is `DIR/maya-ds4/` (`tools/ds4/pack.py`, about a minute);
  pictures need nothing more - the model's own vision tower runs in the engine; the config is `maya-ds4.json`, with
  tensor parallel across pairs of equal GPUs (`--tp 2`); there is no tuning step (the engine balances its own split).

GLM-5.3-Flash's first run takes 20-40 minutes plus the download:

1. checks the PC (GPUs, driver, CUDA toolkit, compiler, RAM, CPU);
2. asks which GPUs to use (all of them by default, up to 16), how much context (32K recommended), which model to
   download: Maya-S (recommended; Maya-S24 on cards of 24 GB or less), Maya-M, Maya-M-Derisked or Maya-L, and how it
   drafts: the model's MTP block (the default) or a [DFlash2 drafter](SETTINGS.md#drafting-with-dflash2) (NVIDIA RTX
   30 / A100 and newer);
3. installs its Python packages into `.venv` and gets llama.cpp's source at a pinned commit (it lists both and asks);
4. compiles the engine for your GPU(s) (10-30 minutes, once);
5. **the model**: it shows the source, the size (Maya-S: 96.5 GB) and the exact `curl` commands, and downloads only
   when you answer `y`; every file is checked against its published sha256. You can run the commands yourself
   instead, or use files you already have: `./maya.sh --gguf-dir DIR` - other quants than these four are
   experimental: they run, but Maya is not measured with them. With DFlash2 it also downloads the drafter (0.7 GB,
   checked against its sha256) next to the models;
6. builds the *pack* - the engine's index of the model files, about 1 GB, written into the model folder. A Maya
   model's files that were already there are checked against their sha256 first (once each); the pack records the
   files, and a start refuses it if they change afterwards (`--repack` builds it again);
7. **pictures**: compiles the image encoder (10-20 minutes, once) and fetches its files (1.1 GB, shown and asked
   first); `--no-vision` skips it;
8. writes `maya-<model>.json` and `run-maya-<model>.sh`; with DFlash2 it runs the
   [speed calibration](SETTINGS.md#the-speed-calibration) (5-20 minutes: DFlash2 stays only if it is faster on your
   PC); then the settings [tuned for this PC](SETTINGS.md) earlier, or it offers the tuning (about 10-15 minutes;
   `./maya.sh --calibrate` any time) - and starts the dashboard.

> **The start takes a few minutes**: the engine pins most of the free RAM (all but about 6 GB) for its expert tier
> and warms its caches. Other programs get little RAM while Maya runs.

## Next time, and updating

**Next time**, just run `./setup.sh` again (or the model's `./run-maya-<model>.sh`): it starts right away, nothing is
downloaded twice, on the same screen - its dashboard shows the Monitor's numbers (the speed, where the experts live, the
GPUs, the requests), Tab its log. Ctrl+C stops it.

**Updating:** the dashboard's **About** says when a new version is out, and its **Update** button does it: Maya
downloads the new code (git), compiles only the engine files that changed and loads the same model again - the model
is not downloaded again, and your settings stay. By hand: `git pull`, then `./setup.sh`.
(The check asks GitHub for the latest release, at most every six hours; `MAYA_UPDATE_CHECK=0` turns it off.)

## Options

| Option | |
| --- | --- |
| `--setup` | set up again (other GPUs, context, model folder) |
| `--check` | only check the PC |
| `--gguf-dir DIR` | use GLM-5.3-Flash GGUF files you already have: their folder, or the `.gguf` file (the first one of a split model) when the folder holds several models. The folder must be writable: the pack goes inside it |
| `--models-dir DIR` | where downloaded models go, each in a folder named after it - `DIR/Maya-L/` holds that model's GGUF files (default `../Maya-data/models`); put it on the NVMe |
| `--download-model` | download the model without asking (the commands and size are still printed) |
| `--model NAME` | which model, without asking: `DeepSeek-V4.1-Flash-1017`, or a GLM-5.3-Flash download: `Maya-S-v2-IQ2_XXS` (Maya-S), `Maya-S24`, `Maya-M`, `Maya-M-Derisked`, `Maya-L`, or `EXL3-3.05bpw` / `EXL3-4.05bpw` - an experimental EXL3 quant at 3.05 or 4.05 bits per weight ([3.05bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw), 125.3 GB; [4.05bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/4.05bpw), 165.2 GB; both with the MTP block) that the engine reads in place from its safetensors: NVIDIA only, 3.05bpw tested on Linux, neither measured against the FP8 model |
| `--model-dir DIR` | use DeepSeek-V4.1-Flash-1017's files you already have: the folder with the EXL3 quant's files and the Engram shards `model-00047-of-00048.safetensors` / `model-00048-of-00048.safetensors` of DeepSeek's repo (it must be writable: the pack goes inside it) |
| `--no-vision` | text only: no image encoder (DeepSeek: its vision tower off) |
| `--draft mtp` / `--draft dflash2` | how the model drafts tokens ahead: its MTP block (the default), or a DFlash2 drafter - downloaded (0.7 GB) into `DIR/GLM-5.3-Flash-DFlash2/` and passed as `--dflash` (Maya-L: +44% on nine GPUs, +12% on two, +19% on one GPU; NVIDIA RTX 30 / A100 and newer - older cards keep the MTP block unless you pass this; the drafter's license is CC BY-NC-ND 4.0, non-commercial use only). The [speed calibration](SETTINGS.md#the-speed-calibration) then keeps it only if it is faster on your PC |
| `--gpu N` / `--gpus 0,1,2,3` | one GPU, or several (up to 16) that split the model's layers |
| `--context N` | context length in tokens: 8192, 32768 (default), 65536, 131072; DeepSeek: 32768, 65536 (default), 131072, 262144 |
| `--vram-reserve-mib N` | VRAM in MiB the expert pool leaves free on each GPU (default 700), kept in the config - more where the desktop or other programs on the GPU need room ([the expert pool's size](SETTINGS.md#the-expert-pools-size)) |
| `--port N`, `--host 0.0.0.0 --api-key KEY` | another port; reachable from your network (always set a key; several: `KEY1,KEY2`) |
| `--env KEY=VALUE` | an engine setting kept in the config (see [Tuning](SETTINGS.md)) |
| `--host-compiler g++-12` | when your g++ is newer than your CUDA accepts ("unsupported GNU version") |
| `--rebuild`, `--repack` | compile the engine / build the pack again |
| `--calibrate` | tune the engine's CPU lane for this PC (see [Tuning](SETTINGS.md)), then start; with `--no-start` only tune |
| `--speed-calibration` | measure again whether DFlash2 or the MTP block decodes faster on this PC ([how](SETTINGS.md#the-speed-calibration); 5-20 minutes, with Maya stopped) and keep the faster, then start; with `--no-start` only measure |
| `--no-speed-calibration` | with `--draft dflash2`: keep DFlash2 without the speed calibration |
| `--yes` | the recommended answers (the model download still needs `--download-model`) |
| `--plain` | the setup in plain text with typed answers, and Maya in the terminal, not on their own screen (a pipe gets this too, and `--yes` a plain setup); the setup screen's whole output is in `maya-setup.log` |

## Web pages on another address

A browser stops a web page served from another address (a local chat client on `http://127.0.0.1:8090`, say) from
calling Maya's API unless Maya allows that page. List the pages in the config, `maya-<model>.json`:

```json
"cors_origins": ["http://127.0.0.1:8090"]
```

Exact addresses (scheme, host and port, no path); `["*"]` lets every page in - set an `api_key` with it, or any page
you open could use your model. It opens `/v1/*` and nothing else: the settings and the tools stay for Maya's own page.
Without it nothing changes. A new `--setup` writes the config again: add the line back after it.

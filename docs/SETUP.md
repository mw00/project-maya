# Setting up Maya

The setup in detail: what its first run does, starting again and updating, and its options. Back to the
[README](../README.md).

## What the first run does

It first asks which model to set up:

- **GLM-5.3-Flash** - Project Maya's GGUF quants (or turboderp's EXL3 quants), from one GPU up; the steps below;
- **DeepSeek-V4.1-Flash-1017** (experimental) - [LibertAIDAI's EXL3 quant at 2.9 bits](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw)
  (211 GB) or [at 3.5 bits](https://huggingface.co/LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-3.5bpw) (258 GB, about
  280 GB of VRAM and RAM; `--ds4-quant exl3-3.5`, its own folder; a second quant hard-links the Engram tables from the
  first instead of downloading them) and its two Engram tables from [DeepSeek's repo](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017)
  (203 GB), on its own engine (`build/maya-ds4`): Linux, NVIDIA V100 or newer (pictures: RTX 30 / A100 or newer); AMD experimental. It is fast with about 230 GB
  of VRAM and RAM together; with less, the setup warns and the engine reads the experts that do not fit from the
  model's files on the SSD (`--mmap-experts auto`), much slower - one RTX 3090 with 64 GB of RAM: prompts about 50
  tokens/s, answers 4-5 tokens/s (faster with `--dspark none`). The same steps with these differences: the
  contexts are 32K-1M (256K recommended where the PC's VRAM + RAM hold the model's files, else 64K; 512K and 1M
  take VRAM from the expert cache, and a prompt that long reads
  slowly - the indexer's work grows with the square of its length); step 5 shows both
  sources and checks every file against its pinned size (and, once downloaded, its sha256: `tools/ds4/download.json`, `download-3.5bpw.json`);
  `--model-dir DIR` uses files you already have; the pack is `DIR/maya-ds4/` (`tools/ds4/pack.py`, about a minute);
  pictures need nothing more - the model's own vision tower runs in the engine; the config is `maya-ds4.json`, with
  tensor parallel across pairs of equal GPUs (`--tp 2`). The engine balances its own layer split; `./maya.sh --calibrate`
  (not offered by the setup) measures each `--tp` that groups your GPUs differently and fewer CPU threads, an engine
  start each (about an hour on 9 GPUs), and keeps the fastest answers whose long prompts still read at 60% of the
  config's own `--tp`'s speed; a setting the engine does not start with is left out (`tools/calibrate_ds4_tp.py`; the
  result goes into `maya-ds4.json` and a setup again keeps it).

  Or from [unsloth's GGUFs](https://huggingface.co/unsloth/DeepSeek-V4.1-Flash-1017-GGUF) (`--ds4-quant`; the setup asks):
  **UD-Q8_K_XL** (508 GB, the default: dense weights BF16, routed experts MXFP4, the Engram tables FP8 - DeepSeek's own
  formats) or **UD-Q4_K_XL** (501 GB: dense weights Q8_0, less VRAM). The same engine reads them in place (the pack is
  `tools/ds4/pack_gguf.py`'s), from NVIDIA Volta (V100) on: the most-used experts in VRAM, the others read from the
  GGUF through the OS page cache - from RAM where they fit, from the SSD otherwise - so it runs whatever the memory
  (more VRAM and RAM is faster; keep the files on an NVMe SSD; `--mmap-experts off` copies as many of them into RAM as
  fit instead, on huge pages split across the CPU sockets' memory - measured no faster than the page cache). Prompts
  go through in chunks of 8192 tokens, also where tensor parallel falls back to one GPU per layer group (the log says
  `tensor parallel 2 -> 1`, as with the Maya quants) - before v1.1.17 those ran every prompt in windows of 16 tokens,
  an order of magnitude slower on long prompts (a community measurement, 7x RTX 3090, Maya-S read in place: 32K tokens
  99 tokens/s in windows, 1010 in chunks). The GGUFs have no DSpark drafter and no vision tower.
- **MiniMax-M3.1-Flash** (experimental; `--model MiniMax-M3.1-Flash`) - [Mia-AiLab's EXL3 quant at 3.5 bits](https://huggingface.co/Mia-AiLab/MiniMax-M3.1-Flash-EXL3-EAQ-v1-3.5-bpw)
  (197 GB with MiniMax's DSpark drafter; every file pinned and checked, `tools/mm3/download.json`) on its own engine
  (`build/maya-mm3`): NVIDIA from V100 on, Linux and Windows; AMD experimental (as GLM's EXL3 quants there: the
  kernels pass their tests on AMD, the whole model has not run on one yet) - on AMD, Maya's GGUF quants
  (`--model minimax-maya-xs` / `minimax-maya-s`: [SETTINGS.md](SETTINGS.md#minimax-m31-flash)) are the tested way.
  About 250 GB of VRAM
  and RAM hold all of it; with less the setup warns. The same steps with these differences: the contexts are 32K-1M,
  the recommended one the largest (up to 256K) whose KV cache fits beside the model in this PC's VRAM + RAM, its RAM
  said, with warnings past 256K and when the cache would take the experts' RAM; the pack is `DIR/maya-mm3/`
  (`tools/mm3/pack.py`, Maya's own copy of MiniMax's chat template inside it); pictures are on - the model's vision
  tower runs in the engine (`--no-vision`: off); the config is `maya-mm3.json`, with tensor parallel in groups of 4 or 2
  where the GPUs make whole groups, KV streaming on, MiniMax's sampling (temperature 1.0, top-p 0.95) and thinking level
  Auto; then the DSpark drafter is measured on this PC (`--drafting` chooses without measuring), as DeepSeek's. Its
  settings: [SETTINGS.md](SETTINGS.md#minimax-m31-flash).

  Or orcarouter's [abliterated model](https://huggingface.co/orcarouter/DeepSeek-V4.1-Flash-1017-abliterated-GGUF)
  (DeepSeek-V4.1-Flash-1017 with its refusal direction taken out of the weights), in two versions:
  **Mix-128G** (`--ds4-quant abliterated-128g`; experts ~1.6 bits, ~114 GB of VRAM + RAM) and **Mix-256G**
  (`--ds4-quant abliterated-256g`; experts 2.5-4.25 bits by layer - IQ2_S / IQ3_XXS / IQ3_S / MXFP4 - ~236 GB of
  VRAM + RAM), in `<models>/DeepSeek-V4.1-Flash-1017-abliterated-GGUF/<version>`; `--ds4-quant abliterated` picks the
  one your PC holds. Their n-gram table (shard 2, not abliterated) is not downloaded: both read Maya's Q6_K tables
  (161 GB, in `Engram-Q6_K` beside them - linked from a Maya DeepSeek quant on the same drive, else downloaded once).
  Its repo is gated: accept its terms on Hugging Face, then set `HF_TOKEN` or run `hf auth login` - the download
  sends that token (curl reads it from a file only you can read). It is served as `deepseek-v4.1-flash-abliterated`,
  with DeepSeek's DSpark drafter and vision tower as for unsloth's GGUFs (the drafter only proposes tokens - the model
  checks each one).

GLM-5.3-Flash's first run takes 20-40 minutes plus the download:

1. checks the PC (GPUs, driver, CUDA toolkit, compiler, RAM, CPU);
2. asks which GPUs to use (all of them by default, up to 16), how much context (32K recommended), which model to
   download: GLM-5.3 · Maya-S (recommended; Maya-S24 on cards of 24 GB or less), Maya-M, Maya-M-Derisked or Maya-L, and how it
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
GPUs, the requests), Tab its log. Ctrl+C stops it - and so does any other way of ending `maya.py`: `kill`, a
service manager, closing the terminal or the window, even a crash: the server and the engine end with it (the
engine keeps its conversation where there is time for that).

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
| `--model NAME` | which model and files, without asking. Maya's quants by their setup id - the model first, then the tier (XS, S, M, L by the experts' bits): `glm-maya-s`, `glm-maya-s24`, `glm-maya-m`, `glm-maya-m-derisked`, `glm-maya-l`, `deepseek-maya-xs`, `deepseek-maya-s`, `deepseek-maya-m`, `minimax-maya-xs`, `minimax-maya-s` (or the full name, e.g. `MiniMax-M3.1-Flash-Maya-XS`). Or a model, its files asked: `DeepSeek-V4.1-Flash-1017`, `MiniMax-M3.1-Flash`. The names used before still work: `Maya-S-v2-IQ2_XXS` (GLM-5.3 · Maya-S), `Maya-S24`, `Maya-M`, `Maya-M-Derisked`, `Maya-L`, and `EXL3-3.05bpw` / `EXL3-4.05bpw` / `EXL3-sc_4.10bpw` - an experimental EXL3 quant of GLM-5.3-Flash at 3.05 or 4.05 bits per weight, or the self-calibrated one at 4.10 with 8-bit attention and shared experts ([3.05bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/3.05bpw), 125.3 GB; [4.05bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/4.05bpw), 165.2 GB; [sc_4.10bpw](https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/tree/sc_4.10bpw), 167.8 GB; all with the MTP block) that the engine reads in place from its safetensors: NVIDIA, and AMD (experimental); 3.05bpw tested on Linux, none measured against the FP8 model |
| `--model-dir DIR` | use DeepSeek-V4.1-Flash-1017's files you already have: the folder with a GGUF's shards, or with the EXL3 quant's files and the Engram shards `model-00047-of-00048.safetensors` / `model-00048-of-00048.safetensors` of DeepSeek's repo (it must be writable: the pack goes inside it) |
| `--ds4-quant Q` | DeepSeek-V4.1-Flash-1017's files: `maya-s`, `maya-xs` (Maya's quants, the same as `--model deepseek-maya-s` / `deepseek-maya-xs`; Maya-S the default), `gguf-q8` (unsloth's UD-Q8_K_XL GGUF), `gguf-q4` (its UD-Q4_K_XL: dense weights Q8_0), `exl3` (LibertAIDAI's EXL3 quant at 2.9 bits), `exl3-3.5` (its 3.5-bit quant: closer to the original, about 280 GB of VRAM and RAM), `abliterated-128g` / `abliterated-256g` (orcarouter's abliterated model as its Mix-128G / Mix-256G GGUF, with Maya's Q6_K n-gram tables; `abliterated`: the one this PC holds; a gated repo: `HF_TOKEN` or `hf auth login`); the EXL3 quants need NVIDIA V100 or newer (pictures: RTX 30 / A100 or newer) and are offered only there, and on AMD as experimental (as GLM's EXL3 quants there; [docs/EXL3_HIP.md](EXL3_HIP.md)) |
| `--mm3-quant Q` | MiniMax-M3.1-Flash's files: `exl3` (Mia-AiLab's EXL3 quant at 3.5 bits, NVIDIA), `maya-xs` or `maya-s` (Maya's GGUF quants, the same as `--model minimax-maya-xs` / `minimax-maya-s`; `gguf-xs` / `gguf-s` before) |
| `--no-vision` | text only: no image encoder (DeepSeek, MiniMax: their vision tower off) |
| `--draft mtp` / `--draft dflash2` | how the model drafts tokens ahead: its MTP block (the default), or a DFlash2 drafter - downloaded (0.7 GB) into `DIR/GLM-5.3-Flash-DFlash2/` and passed as `--dflash` (Maya-L: +44% on nine GPUs, +12% on two, +19% on one GPU; NVIDIA RTX 30 / A100 and newer - older cards keep the MTP block unless you pass this; the drafter's license is CC BY-NC-ND 4.0, non-commercial use only). The [speed calibration](SETTINGS.md#the-speed-calibration) then keeps it only if it is faster on your PC |
| `--gpu N` / `--gpus 0,1,2,3` | one GPU, or several (up to 16) that split the model's layers |
| `--context N` | context length in tokens: 8192, 32768 (default), 65536, 131072; DeepSeek: 32768, 65536, 131072, 262144, 524288, 1048576 (the default 262144 where VRAM + RAM hold the model, else 65536); MiniMax: the same, the default by this PC's memory |
| `--vram-reserve-mib N` | VRAM in MiB the expert pool leaves free on each GPU (default 700), kept in the config - more where the desktop or other programs on the GPU need room ([the expert pool's size](SETTINGS.md#the-expert-pools-size)) |
| `--port N`, `--host 0.0.0.0 --api-key KEY` | another port; reachable from your network (always set a key; several: `KEY1,KEY2`) |
| `--env KEY=VALUE` | an engine setting kept in the config (see [Tuning](SETTINGS.md)) |
| `--host-compiler g++-12` | when your g++ is newer than your CUDA accepts ("unsupported GNU version") |
| `--rebuild`, `--repack` | compile the engine / build the pack again |
| `--calibrate` | tune the engine for this PC (see [Tuning](SETTINGS.md)): GLM's CPU lane, DeepSeek's `--tp` and CPU threads, MiniMax's [prefill calibration](SETTINGS.md#minimax-m31-flash) (the chunk and the matrix products' library a prompt is read with); then start; with `--no-start` only tune |
| `--speed-calibration` | measure again whether the drafter (DFlash2, or DeepSeek's or MiniMax's DSpark) decodes faster on this PC ([how](SETTINGS.md#the-speed-calibration); 5-30 minutes, with Maya stopped) and keep the faster, then start; with `--no-start` only measure |
| `--drafting measure\|on\|off` | the drafter without the question: `measure` = the speed calibration (the default with `--yes`), `on` = draft with it, `off` = no drafting - both skip the measuring |
| `--no-speed-calibration` | the same as `--drafting on` |
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

## Giving the GPU back when idle

To share the PC with other GPU programs (another model server, games), let Maya unload the model after a while without
requests. In the config, `maya-<model>.json`:

```json
"idle_unload_s": 900,
"env": {"STRATA_GLM_SLOT_KEEP": "1"}
```

(`--idle-unload 900` on the server's command line does the same.) After that many seconds with no request running or
waiting, the engine ends: its GPU memory and its pinned RAM tier are given back (an RTX 3090 with Maya-S24: 22.9 GB of
VRAM and 79 GB of RAM). The dashboard and `/health` say `unloaded`, and `/v1/models` still lists the model with the
status `unloaded`. The next request loads the model again - about a minute before its first token - and the experts
in VRAM warm up again over the next answers. With `STRATA_GLM_SLOT_KEEP=1` the conversation in the model is written to
the disk on the way out and read back, so the next turn of the same chat reads only its new tokens (1,630 of 1,690
prompt tokens reused after an unload). `0` (the default): never. A new `--setup` writes the config again: add the
lines back after it.

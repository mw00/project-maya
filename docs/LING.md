# Running Ling-3.2-air

What works today, and what does not yet. Ling-3.2-air runs on its own engine, `maya-ling` (`src/ling`), which
alesha-pro contributed (PR #1224).

## 1. The model

[inclusionAI/Ling-3.2-air](https://huggingface.co/inclusionAI/Ling-3.2-air) (MIT), architecture `bailingmoe3`:

| | |
| --- | --- |
| Layers | 42: 35 with a constant-size recurrent state (KDA), 7 with a latent KV cache (MLA) |
| Experts | 40 of the layers route: 512 experts a layer, 8 a token |
| Draft block | the model's own MTP block (block 42), where the file carries it |
| Pictures | a vision tower, as a separate `mmproj` GGUF |
| Context | up to 262,144 tokens |

## 2. Where it runs

| | |
| --- | --- |
| NVIDIA, Linux | V100 / RTX 20 and newer (compute capability 7.0 on), one GPU or several |
| NVIDIA, Windows | the same cards (`START-MAYA.bat --setup --model ling-3.2-air-maya-s`); the experts read from the SSD go through Windows' file cache (no unbuffered reads there) |
| AMD, Linux | one RDNA 3 / 3.5 / 4 GPU (gfx11 / gfx12): a Radeon RX 7000 / 9000 or PRO card of 8 GB or more, or a Ryzen AI Max APU (Strix / Gorgon Halo), whose memory is the RAM |
| CPU | AVX2 (the experts that are not in VRAM are computed on the CPU) |
| AMD on Windows, Mac; several AMD GPUs; older AMD GPUs | not yet: the setup says so in one line and stops |

On AMD the engine is the same program built with ROCm (HIP), as the other models' AMD engines are. Run by Maya's
team on a Gorgon Halo (Radeon 8065S, gfx1151, Ubuntu, ROCm 7.14); Radeon RX 7000 / 9000 cards build the same engine
but have not been run here yet. On an APU the GPU's share is the PC's memory less max(8 GB, 10%) for the system
(and no more than Linux lets the GPU take: `ttm.pages_limit`, see [AMD_MAYA.md](AMD_MAYA.md)).

## 3. Setting it up

```
./maya.sh --setup --model ling-3.2-air-maya-s
./maya.sh --setup --backend hip --model ling-3.2-air-maya-s     # an AMD GPU
```

| Size | Setup id | File |
| --- | --- | --- |
| Ling-3.2-air · Maya-S | `ling-3.2-air-maya-s` | `Ling-3.2-air-Maya-S-IQ2_XXS.gguf`, 36.9 GB |
| Ling-3.2-air · Maya-M | `ling-3.2-air-maya-m` | `Ling-3.2-air-Maya-M-IQ2_S.gguf`, 46.0 GB |
| Ling-3.2-air · Maya-L | `ling-3.2-air-maya-l` | `Ling-3.2-air-Maya-L-IQ3_S-00001-of-00002.gguf` and `-00002-of-00002`, 59.2 GB |

The file is pinned in `tools/ling/download.json` by size and sha256, and checked after the download and when a copy
is already in the folder (`--model-dir <folder>` names a folder that has it).

Any other GGUF of the model (one file, or a split set by its first shard):

```
./maya.sh --setup --model ling-3.2-air --model-dir <folder with the .gguf>
```

A Maya size gets its own config (`maya-ling-3.2-air-maya-s.json`), a file of your own `maya-ling-3.2-air.json`.
Running the setup again keeps what you set by hand in the config (engine flags such as `--cpu-threads` or
`--reserve-gb`, the port, the API key, your variables) and leaves a `.bak` of the config before.

## 4. What the setup chooses, and what the engine measures

Nothing here is a fixed number for one machine: the setup reads the PC, and the engine measures on it.

- **The size**: the one you name; with none named, the largest downloadable size whose file and cache fit this PC's
  VRAM and RAM without reading experts from the SSD.
- **The context**: recommended from the cards' VRAM (the full 262,144 stays on offer). The cache is small for this
  model: 8,064 bytes a token in FP16, 4,704 in 8-bit rows.
- **The fit**: one planner (`tools/ling_fit.py`) for the setup and the Models page: what goes into VRAM on each
  GPU, what is computed from RAM, what is read from the SSD. "Runs slowly" and "does not fit" are said before
  anything is downloaded; `--force-fit` goes on anyway.
- **Several GPUs**: the layers are split across them, the cards ordered by the engine (a card driving a display
  last). Tensor parallel is not there yet.
- **Expert placement**: at its first start the engine measures how to split the experts between GPU, CPU and SSD on
  this PC and keeps the result in its calibration file, beside the model (`maya-ling/ling-calibration.txt`).
  `./maya.sh --calibrate` clears it, and the next start measures again.
- **Drafting**: `--drafting auto` by default. Where the file has the model's draft block, the engine measures once
  whether drafting makes answers faster on this PC and keeps the faster. `--drafting off` turns it off;
  `--drafting on` turns it on without measuring. A file without the draft block runs without it. The Models page
  offers On and Off (they write `--drafting on` / `--drafting off`); a setup's `auto` shows there as On and stays
  `auto` until one of them is picked.
- **The cache format**: FP16 rows by default. `--kv q8` in the config's arguments halves the cache (8-bit rows); the
  engine has no measured choice between the two yet.

## 5. Defaults, template, thinking, tools

- **Sampling**: the model's own `generation_config.json` - temperature 1.0, top-p 0.95, top-k 20; no min-p and no
  penalties. Written into the config at every setup; a value you change on the Settings page is followed.
- **Chat template**: inclusionAI's `chat_template.jinja`, the official text (`tools/ling/chat_template.jinja` adds a
  licence line above it). A `chat_template.jinja` you put in the model's folder is used instead.
- **Thinking**: the template has one switch, on by default. Maya offers Off and High; Off is really off (the
  template's "detailed thinking off").
- **Tool calls**: in the template's own format (`<tools>` signatures in the system turn, `<tool_call>` blocks in
  the answer), read back by the server into the API's tool-call form.
- **Loop guard**: on by default, as for every model (`"loop_guard": false` in the config turns it off).

## 6. Pictures

Pictures work: the setup downloads the model's vision file (`Ling-3.2-air-mmproj-BF16.gguf`, 0.9 GB, from Maya's
repo, pinned by size and sha256) beside the model and turns pictures on. An `mmproj` GGUF of Ling-3.2-air already
beside the model is used as it is. No VRAM is kept for it: the engine borrows room only while a picture is read.
`--no-vision` leaves pictures off (nothing is downloaded).

## 7. Restore points on disk

Off by default. `--restore-dir <folder>` in the config's arguments turns them on (`--restore-dir-max-mb`, default
2048, caps the folder; the oldest files go first). The files hold the conversation's tokens in plain form.

## 8. Not there yet

- One answer at a time: no `--parallel`.
- No tensor parallel.
- AMD on Windows, Mac, and several AMD GPUs at once.
- Windows: one node's CPU pinning only (a PC with several NUMA nodes runs as one); `--image` on the engine's own
  command line takes PPM files only (the server's pictures are not affected).
- The same answer on one GPU and on several holds with `--repeatable` only; a normal run can differ in the last
  digits between layouts.

## 9. Benchmark

`./maya.sh --bench` knows the model: it reads the config the setup wrote and prints one line with the prompt read
and the answer speed for this PC.

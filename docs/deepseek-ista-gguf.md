# DeepSeek-V4.1 on ISTA-DASLab's GSQ-RCO GGUF

[ISTA-DASLab's GSQ-RCO GGUFs](https://huggingface.co/ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF) of
DeepSeek-V4.1-Flash-1017 run on Maya's DeepSeek engine (`maya-ds4`) like its other GGUFs. The setup does not offer
them, so the steps below do by hand what it does for Maya's quants: download, pack, write a config. The repo has three
builds (IQ1_M, IQ2_XXS, IQ2_XS) and one Engram file for all of them; **IQ2_XXS** is the one described here (138.24 GB
of backbone, 2 bits on average, each tensor in the type ISTA's search picked) and the only one run with Maya. ISTA
reports that it matches the original model on AIME25, GPQA-Diamond and LiveCodeBench v6 at reasoning effort 50 (its
own measurements).

Linux with NVIDIA GPUs is what it was run on; AMD and Windows have not been tried with these files.

## What you need

- Maya set up once for DeepSeek with any of its quants (`./setup.sh --setup`, pick DeepSeek): that builds the engine
  (`build/maya-ds4`) and `.venv`, and writes the `maya-ds4.json` that step 4 starts from.
- A writable folder on an NVMe SSD with about 260 GB free: the backbone (138.24 GB), the Engram tables (110.6 GB), the
  drafter (8 GB) and the vision tower (1 GB). The commands below call it `/path/to/models`.
- Memory: the backbone's experts go to the GPUs, then to RAM, as with any GGUF ([the model's notes](MODELS.md#deepseek-v41-flash-1017)),
  so about 140 GB of VRAM and RAM together hold it, plus the drafter and the context. With less, the rest is read from
  the files, slower. The Engram tables (110.6 GB) are read row by row through the page cache and are not counted:
  free RAM keeps more of them warm; `DS4_ENGRAM_RAM=1` locks all of them in RAM instead
  ([explicit RAM residency](SETTINGS.md#deepseek-explicit-ram-residency): `IQ4_NL` is supported).

## 1. Download

```sh
hf download ISTA-DASLab/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF \
  --revision 653a9874aa38c356233c5f3cc8c96b19a43b57eb \
  --include "*IQ2_XXS*.gguf" --include "*Engram*.gguf" \
  --local-dir /path/to/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
```

That is four shards and the Engram file, in one folder (repeat `--include` for each pattern: with some versions of
`hf`, a second pattern after one `--include` is read as a file name). Their sizes and sha256 at that revision:

| File | Bytes | sha256 |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00004.gguf` | 39075725728 | `2dfe9711eac1f7814aa59154d11e8f27739f8b96875dc1e667eac9f8c756e1fe` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00004.gguf` | 39388449504 | `cec1b2c3da429dc768b8bb9bc4dc4403d81f74d34db660a6a34ee3e555bf2c70` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00004.gguf` | 39246891616 | `9d9c2140fcb134a0b9a74dedd0e5167e68077af6d08ce7f856bce112f0a35dd3` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00004.gguf` | 20527837312 | `9b790a2297ed3cdede379fc984f4a52bb95f3b23724c797af11d1fa6a69db059` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 110595290816 | `eb19f30d368d393784e1fc54fdfaffc427ebb2110bf3762fdf5a8fbb9a2d740a` |

The revision is pinned because the repo changed its files' layout on October 27, 2026: the current files are six
shards in llama.cpp's own DeepSeek-V4.1 layout (b11870 and newer), with the Engram file linked as shard 6. That layout
packs too (the Engram gate comes folded into one tensor, which the engine reads), but it has not been served with
Maya yet; the four-shard files above are the ones that were. The repo's `chat_template.jinja`,
`patch_chat_template.py` and `mmproj-*.gguf` are not used: the pack carries Maya's own chat template and the server
renders prompts with DeepSeek's pinned encoder ([prompt format](deepseek-prompt-format.md)), and the pictures come
from DeepSeek's own vision tower (step 2).

## 2. The drafter and the vision tower

Both are DeepSeek's, shared by every GGUF of this model, and sit beside the model's folder as `dspark/` and `vision/`
(a PC where the setup already installed a Maya DeepSeek GGUF has them in `DeepSeek-V4.1-Flash-1017-GGUF/`: skip
this step and pass them to the pack with `--drafter` and `--vision`). The revisions are the ones
`tools/ds4/download_gguf.json` pins (`maya` and `vision`):

```sh
# the DSpark drafter, 8 GB (DeepSeek's FP4 experts, its dense weights in Q8_0): /path/to/models/dspark/
hf download peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF dspark/dspark.bin dspark/dspark.json \
  --revision c273fcbdda76a1db074c2f414cb70edbfca2f2ef --local-dir /path/to/models

# the vision tower, 1 GB of DeepSeek's checkpoint, then one file of it: /path/to/models/vision/
hf download deepseek-ai/DeepSeek-V4.1-Flash-1017 model-00001-of-00048.safetensors config.json \
  --revision 1bae0e1088993b9b82de08f59de5be5a34e42170 --local-dir /path/to/models/vision
.venv/bin/python tools/ds4/vision_pack.py --dir /path/to/models/vision
```

`vision_pack.py` writes `vision.bin` and `vision.json` (it reads 30 KB of the checkpoint's shard 2 over HTTP, never
the whole file); the shard is not needed after that.

## 3. Pack

```sh
.venv/bin/python tools/ds4/pack_gguf.py --model /path/to/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
```

This writes `maya-ds4/` inside the model's folder: an index of the files, no weights copied, a few seconds. It finds
the Engram file in the same folder, the drafter at `../dspark/dspark.json` and the tower at `../vision`. Its last line
names what went in; it ends `..., the DSpark drafter, the vision tower -> .../maya-ds4`, and without those two the
tool did not find their folders. The options:

- `--drafter /path/dspark.json`, `--vision /path/to/vision-folder`: where they are, if not beside the model's folder;
- `--engram DIR`: a folder holding the Engram file alone, if it is not in the model's folder;
- `--shards FILE ...`: the shard names of one build, if the folder holds more than one (IQ1_M, IQ2_XS);
- `--out DIR`: the pack's folder (default `<model>/maya-ds4`).

## 4. Config and start

Copy the `maya-ds4.json` your DeepSeek setup wrote to `maya-ds4-ista.json` in the same folder (the dashboard lists
every `maya-*.json`), keep its `exe`, `cwd`, `lib_dirs` and `port`, and set these:

```json
"args": ["--pack", "/path/to/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/maya-ds4",
         "--max-context", "262144", "--tp", "2", "--dspark", "on", "--vision", "on"],
"tokenizer": "/path/to/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF/maya-ds4/tokenizer",
"model_name": "deepseek-v4.1-flash",
"gpu": [0, 1],
"env": {"MAYA_DS4_TP_GGUF_UNEVEN": "1", "DS4_FLOW_TP": "1"},
"installer": {"family": "deepseek", "ds4_quant": "ista-iq2xxs", "models_dir": "/path/to/models"}
```

```sh
./maya.sh --config maya-ds4-ista.json
```

- `--max-context`: 262144 is what the setup recommends where VRAM and RAM hold the model; 512K and 1M take VRAM from
  the expert cache ([SETUP.md](SETUP.md)).
- `--tp 2` and the two `env` settings: tensor parallel over a pair of equal GPUs. A GGUF's experts split only in whole
  256-value blocks, which these settings allow ([tensor parallel 2](SETTINGS.md#deepseek-tensor-parallel-2-on-mayas-own-quants)).
  On one GPU write `"--tp", "1", "--threads", "on"` and leave `env` out, as the setup does.
- `--dspark on`: drafting with the drafter; `none` leaves it out. Whether drafting pays depends on the PC
  ([the speed calibration](SETTINGS.md#the-speed-calibration)).
- `--vision on` needs the tower in the pack (step 2); without it write `--vision off`.
- `installer.ds4_quant`: `ista-iq2xxs` names these files, so the dashboard shows "DeepSeek V4.1 · ISTA GSQ-RCO
  IQ2_XXS" (a DeepSeek config without a key is shown as the EXL3 quant).

Check that `/health` reports `images: true`, then send a picture. Other settings of the DeepSeek engine
([SETTINGS.md](SETTINGS.md)) apply as they do to Maya's quants, such as
[conversations kept](SETTINGS.md#deepseek-conversations-kept-an-agents-parent-and-workers-on-one-model).

## Things to know

- **The pack records where the model is.** `ds4.txt` holds the model folder's absolute path (symlinks resolved), and
  the engine opens every file as that path plus the name `tensors.txt` lists (`../dspark/dspark.bin`,
  `../vision/vision.bin`). Moving or renaming a folder, or starting the engine where the models are mounted somewhere
  else (a container), fails with "cannot open": run step 3 again from the path the engine sees, or edit the
  `model_dir` line.
- **The pack records each tensor's offset in its shard.** Change a shard and pack again. ISTA's
  `patch_chat_template.py` rewrites the first shard in place, and Maya does not read the GGUF's own template, so leave
  it alone.
- **The setup does not know these files.** A later `./setup.sh` writes `maya-ds4.json` and leaves
  `maya-ds4-ista.json` as it is.

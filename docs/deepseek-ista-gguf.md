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
  --revision 65d53bc8be65123e255ff02f3ded958b6dbf54fd \
  --include "*IQ2_XXS*.gguf" --include "*Engram*.gguf" \
  --local-dir /path/to/models/DeepSeek-V4.1-Flash-1017-GSQ-RCO-GGUF
```

That is five shards and the Engram file, in one folder (repeat `--include` for each pattern: with some versions of
`hf`, a second pattern after one `--include` is read as a file name). Their sizes and sha256 at that revision:

| File | Bytes | sha256 |
| --- | ---: | --- |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00001-of-00006.gguf` | 5780704 | `f356f198561f56df36f7ab92d7238a926646d76de992c34f15bcb939a5f529b9` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00002-of-00006.gguf` | 39535781184 | `25af3785ccf925f54137e1ead4c481a8952315dc0b1af6519a99407f67bce8f3` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00003-of-00006.gguf` | 39736712000 | `8168d0998fc494def2e1952d83c131425a108660aae9d29190f5a8670ab62b48` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00004-of-00006.gguf` | 39036908384 | `9eb9b4de1968f150dfc207eb31c27be9cd9b5666a9a6246bde8a22b184b4bee0` |
| `DeepSeek-V4.1-Flash-1017-GSQ-RCO-IQ2_XXS-00005-of-00006.gguf` | 19923784832 | `7e35f6b4e6fc46f6c4b152b593d84d683c5ece695ae6ad734f556198c041bec1` |
| `DeepSeek-V4.1-Flash-Engram-IQ4_NL.gguf` | 110595290816 | `3dac4c54893ad7aa89103c0028500eadc78e74112cef3a2930a97e67dc9fb4f3` |

This is llama.cpp's own DeepSeek-V4.1 layout (b11870 and newer), which the repo moved to on October 27, 2026: the
first shard holds only the metadata, the Engram file is the split's sixth file (the pack reads it as such; no link
named `-00006-of-00006` is needed), the Engram gate comes folded into one F32 tensor, and the image tokens' router
bias is added. The four-shard files of the earlier revision (`653a9874`) hold the same weights and pack the same
way; their Engram file has a different header, so the two revisions' files do not mix in one folder. The repo's
`chat_template.jinja`, `convert_ds41_upstream.py`, `exp_probs_b_vl.f32` and `mmproj-*.gguf` are not used: the pack
carries Maya's own chat template and the server renders prompts with DeepSeek's pinned encoder
([prompt format](deepseek-prompt-format.md)), the downloaded shards already hold what the converter adds, and the
pictures come from DeepSeek's own vision tower (step 2).

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
- **The pack records each tensor's offset in its shard.** Change a shard and pack again. Maya does not
  read the GGUF's own chat template, so there is no need to patch the first shard.
- **The setup does not know these files.** A later `./setup.sh` writes `maya-ds4.json` and leaves
  `maya-ds4-ista.json` as it is.

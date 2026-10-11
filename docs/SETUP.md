# Setting up Maya

The setup in detail: what its first run does, starting again and updating, and its options. Back to the
[README](../README.md).

## What the first run does

It takes 20-40 minutes plus the download:

1. checks the PC (GPUs, driver, CUDA toolkit, compiler, RAM, CPU);
2. asks which GPUs to use (all of them by default, up to 16), how much context (32K recommended) and which model to
   download: Maya-S (recommended; Maya-S24 on cards of 24 GB or less), Maya-M, Maya-M-Derisked or Maya-L;
3. installs its Python packages into `.venv` and gets llama.cpp's source at a pinned commit (it lists both and asks);
4. compiles the engine for your GPU(s) (10-30 minutes, once);
5. **the model**: it shows the source, the size (Maya-S: 96.5 GB) and the exact `curl` commands, and downloads only
   when you answer `y`; every file is checked against its published sha256. You can run the commands yourself
   instead, or use files you already have: `./maya.sh --gguf-dir DIR` - other quants than these four are
   experimental: they run, but Maya is not measured with them;
6. builds the *pack* - the engine's index of the model files, about 1 GB, written into the model folder;
7. **pictures**: compiles the image encoder (10-20 minutes, once) and fetches its files (1.1 GB, shown and asked
   first); `--no-vision` skips it;
8. writes `maya-<model>.json` and `run-maya-<model>.sh` - with the settings [tuned for this PC](SETTINGS.md) earlier, or
   it offers the tuning (about 10-15 minutes; `./maya.sh --calibrate` any time) - and starts the dashboard.

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
| `--no-vision` | text only: no image encoder |
| `--gpu N` / `--gpus 0,1,2,3` | one GPU, or several (up to 16) that split the model's layers |
| `--context N` | context length in tokens: 8192, 32768 (default), 65536, 131072 |
| `--port N`, `--host 0.0.0.0 --api-key KEY` | another port; reachable from your network (always set a key; several: `KEY1,KEY2`) |
| `--env KEY=VALUE` | an engine setting kept in the config (see [Tuning](SETTINGS.md)) |
| `--host-compiler g++-12` | when your g++ is newer than your CUDA accepts ("unsupported GNU version") |
| `--rebuild`, `--repack` | compile the engine / build the pack again |
| `--calibrate` | tune the engine's CPU lane for this PC (see [Tuning](SETTINGS.md)), then start; with `--no-start` only tune |
| `--yes` | the recommended answers (the model download still needs `--download-model`) |
| `--plain` | the setup in plain text with typed answers, and Maya in the terminal, not on their own screen (a pipe gets this too, and `--yes` a plain setup); the setup screen's whole output is in `maya-setup.log` |

#!/usr/bin/env python3
"""Project Maya - set up and start GLM-5.3 (744B), GLM-5.3-Flash, DeepSeek-V4.1-Flash-1017, MiniMax-M3.1-Flash or Qwen-3.8-Flash-Next
(CYBER-FROST) on your own GPU(s).  Qwen-3.8-Flash-Next (CYBER-FROST): CUDA from V100 on and HIP, Linux and Windows, pictures too.
Hy4-Flash: CUDA from V100 on, Linux and Windows, pictures too (Tencent's own Q4_K_M with its MTP drafting: run on
RTX 30 / A100 and newer, Linux).
GLM-5.3-Flash: CUDA on Linux and Windows; HIP: gfx1100/gfx1201/gfx1151 (gfx1030: experimental, not
validated), Linux and Windows, text only; Metal: experimental, macOS on Apple Silicon, GLM-5.3-Flash only, text only,
one conversation at a time.  DeepSeek-V4.1-Flash-1017 and MiniMax-M3.1-Flash: CUDA from V100 on, Linux
and Windows; on HIP too (their GGUFs and their EXL3 quants).
The quants are named model first, then the tier by the experts' bits (XS, S, M, L): --model glm-maya-s,
deepseek-maya-xs, minimax-maya-xs, glm-5.3-maya-xs ... (tools/maya_quants.py; GLM-5.3's: its download's name,
G53_MODELS); every name used before still works.

    ./maya.sh                 the first run sets everything up and starts the dashboard; later runs just start it
    ./maya.sh --setup         set up again (other GPUs, another context length, another model folder)
    ./maya.sh --check         only check this PC
    ./maya.sh --backend hip --gpu 0 --check    check an RX 7900 XT / XTX with system ROCm 7

On Windows START-MAYA.bat takes the same options.  Both make the private Python environment (.venv) and run this
file.  Its helpers (setup.py) do the PC checks, pip, llama.cpp's source and resumable downloads.

In a terminal the setup runs on a screen of its own (tools/setup_tui.py): the steps, what runs now with a progress
bar, and the questions as menus - then Maya runs on the same screen, and every later start too: its dashboard (the
Monitor's numbers from the engine's GET /metrics), its server's output, Ctrl+C stops it.  Its library, Textual, comes
with Maya (third_party/wheels) and goes into .venv from there - nothing is downloaded.  --plain or a pipe: plain text,
and Maya in the terminal as before; --yes: the setup in plain text.

What the first run does (each step is skipped when it is already done) - first it asks which model: GLM-5.3-Flash
(the steps below) or DeepSeek-V4.1-Flash-1017 (the same steps for its own engine, build/maya-ds4: LibertAIDAI's EXL3
quant at 2.9 bits, 211 GB, and its two Engram tables from DeepSeek's base repo, 203 GB; its pack is tools/ds4/pack.py's;
its vision tower runs in the engine - a GGUF's comes from DeepSeek's own checkpoint, 0.97 GB, shown and asked first;
no tuning step - it balances its own split):

  1. checks the PC: NVIDIA GPU(s) of compute capability 7.0+ (6.x, Pascal: experimental), driver, CUDA toolkit (nvcc),
     the C++ compiler (g++; on Windows Visual Studio's Build Tools - 2019/2022 always, 2026 with CUDA 13.2+), CMake,
     RAM, CPU; HIP checks AMD gfx1100/gfx1201/gfx1151 and ROCm 7 instead
     (on Windows AMD's ROCm SDK wheels in .venv, offered when they are not there)
  2. asks: which GPUs (one, or several that split the layers), how much context, which model to download (GLM-5.3-Flash ·
     Maya-S, Maya-S24, Maya-M, Maya-M-Derisked or Maya-L; or turboderp's EXL3 quant at 3.05 or 4.05 bits or the
     self-calibrated one at 4.10: NVIDIA from V100, and AMD)
  3. Python packages into .venv, llama.cpp's source at the pinned commit (it lists them and asks first)
  4. compiles the engine (`build/maya`, or `build-hip/maya`) for your GPU(s): 10-30 minutes, once
  5. the model: GGUF files you already have (--gguf-dir), or a download it shows you first - the exact commands
     and the size - and starts only after you answer y (or pass --download-model)
  6. builds the pack (the engine's index of the model files) inside the model folder
  7. images: compiles the vision encoder (`build-vision/bin/maya-vision`) and fetches the model's vision files
     (1.1 GB, shown and asked first like the model; --no-vision skips it)
  8. writes maya-<model>.json and run-maya-<model>.sh (.bat on Windows) - with the settings tuned for this PC
     earlier, or it offers the tuning (./maya.sh --calibrate does it any time) - and starts the dashboard on
     http://127.0.0.1:8080

The tuning (tools/calibrate_glm.py): decode speed measured with a few splits of the RAM-tier
experts between the CPU and the PCIe link, with fewer and more CPU threads, and - when part of the model is read from
the SSD - with smaller disk reads, in one engine run (~10-15 minutes, the model loads first); a setting is kept when
it is more than 3% faster than the engine's own choice.  The result goes into the config's "env" (MAYA_GLM_PCIE_SHARE,
MAYA_GLM_CPU_LANE, MAYA_GLM_READ_CHUNKS) and into ~/.config/project-maya/calibration.json for this PC, model and
context, so a setup again keeps it.  DeepSeek-V4.1-Flash-1017's (tools/calibrate_ds4_tp.py, ./maya.sh --calibrate only):
decode speed with each --tp that groups the GPUs differently and with fewer CPU threads, an engine start each (~25-40
minutes on 9 GPUs); a setting is kept when it is more than 5% faster and still reads a long prompt at 60% of the
config's own speed.  The result goes into the config's args (--tp, --cpu-threads) and into calibration.json.

Nothing is installed system-wide: a missing tool is reported with the command that installs it.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import shlex
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "tools"))
import setup as S  # noqa: E402  Strata's installer: PC checks, pip, llama.cpp, downloads (nothing runs on import)
from setup import ask, fail, ok, run, say, step, warn  # noqa: E402
import maya_import as MI  # noqa: E402  --import-model: model files downloaded some other way, found by size + sha256
import maya_names as N  # noqa: E402  the settings' names users see (MAYA_*) and the older ones the code reads
import maya_quants as Q  # noqa: E402  the quants' names users see (MiniMax M3.1 · Maya-XS) and the older ones as aliases
import calibrate_glm as CAL  # noqa: E402  the CPU lane's tuning for this PC (./maya.sh --calibrate)
import calibrate_ds4_tp as CAL4  # noqa: E402  DeepSeek's: --tp and the CPU threads (./maya.sh --calibrate)
import calibrate_draft as CD  # noqa: E402  the speed calibration: DFlash2 or the MTP block on this PC
import calibrate_ds4 as CDS  # noqa: E402  DeepSeek's: the DSpark drafter or none, on this PC
import calibrate_mm3_prefill as CPF  # noqa: E402  MiniMax's: how this PC reads prompts fastest (chunk, GEMMs)
from serve import glm_dsa_product as G53  # noqa: E402  GLM-5.3 (744B, glm-dsa)
from serve import glm_link as GL  # noqa: E402  GLM on an AMD APU + an NVIDIA card ("mixed_gpus")
from strata import maya_sync  # noqa: E402  (tools/strata/) a glm-dsa pack's expert profile and drafting

WIN = S.WIN
ME = "START-MAYA.bat" if WIN else "./maya.sh"      # how this is started, for the messages
UPDATE_EXIT = 75                                   # the server's exit after the dashboard updated Maya (serve/update.py)
RESTART_EXIT = 76                                  # ... after it saved a new port: start again on the saved one
SWITCH_FILE = ".maya-switch.json"                  # ... or the Models page's note: the config to start (serve/models.py)
ROOT = S.ROOT
BUILD = ROOT / "build"
EXE = BUILD / ("maya.exe" if WIN else "maya")
STAMP = BUILD / "MAYA-BUILD.json"                  # what the engine in build/ was compiled from and for
VBUILD = ROOT / "build-vision"
VEXE = VBUILD / "bin" / ("maya-vision.exe" if WIN else "maya-vision")  # the image encoder (llama.cpp's mtmd)
VSTAMP = VBUILD / "MAYA-BUILD.json"
MIN_CC = 60                                        # Pascal and newer (the GLM path; see arch_setting) ...
VALIDATED_CC = 70                                  # ... Volta (V100) and newer validated, Pascal experimental (#212)
SMALL_GPU_SHARE = 0.25                             # choose_gpus: a card under this share of the biggest one's VRAM is left out
MAX_GPUS = 16                                     # the layer split's parts at most (glm_model.hpp, kMaxParts)
HIP_ARCHS = ("gfx1100", "gfx1201", "gfx1151")        # Maya also supports Strix Halo (unified memory)
HIP_ARCHS_UNVALIDATED = ("gfx1030",)               # RDNA 2 (RX 6800 / 6900, Radeon PRO V620 / W6800), experimental: not
                                                   # validated on a card; no matrix cores, the portable kernels; compiled
                                                   # only when such a GPU is chosen
HIP_ARCHS_WAVE64 = ("gfx908", "gfx906", "gfx942", "gfx950")  # CDNA 1 (MI100), Vega 20 (gfx906), CDNA 3 (MI300X / MI325X),
                                                   # CDNA 4 (MI350X / MI355X): wave64, run through the hip_compat layer's
                                                   # logical-32-lane half-wavefront warp mapping; experimental
PY_PACKAGES = list(S.PY_PACKAGES)                  # (pillow: pictures in formats other than JPEG/PNG/BMP/GIF)
CONTEXTS = [8192, 32768, 65536, 131072]
DEFAULT_CONTEXT = 32768
# KV streaming, on in every config (--kv-streaming on): the attention layers' latent cache lives in RAM and 32768
# positions of each layer in VRAM (--kv-resident N, beside it, sets another window) - from a context above that (a
# shorter one is not streamed); 12 KB of RAM a context token, 1.6 GB at 128K.  "--kv-streaming", "off" in the config's
# args turns it off; a setup again keeps what the config says
KV_STREAMING = "on"
MODEL_NAME = "glm-5.3-flash"
SAMPLING, EFFORT = Q.MODEL_DEFAULTS["glm"]        # GLM's own sampling (its generation_config) and level (Max) -
#                                                    the dashboard's and the API's defaults for requests that set none
HF = "https://huggingface.co/{repo}/resolve/{revision}/{path}"
# The models the installer can download: Project Maya's own quants, made from Z.ai's FP8 release (their model card
# has the measurements against it).
# Another glm5-next GGUF can be used with --gguf-dir (experimental).  "folder": the files' folder in the repo ("" =
# its top).  "file": the names on Hugging Face, with the quant label its file list groups (and adds up) them by -
# one label per model; "was": the names they had there before (v1.0.18 and earlier: no label; v1.0.19: Maya-S24 as
# IQ2_XXS, which Hugging Face added to Maya-S's) - a download under them is used as it is.  "sha256" per file name (the
# Hugging Face one): every download is verified; a list holds every version published (v1.0.28: the first shards' header
# names the architecture "glm5-next", as llama.cpp does - the data is the same, so a download of either is good).  "vision": the image encoder's files (the mmproj,
# made from the official vision tower, and the tokenizer it reads its markers with) - from "repo" / "revision" when
# it names them, else the model's own repo.
MODELS = {
    "Maya-S-v2-IQ2_XXS": {
        "about": "Maya-S, Project Maya's compact quant: error-feedback-rounded IQ2_XXS gate/up experts, "
                 "IQ2_S/IQ3_XXS down projections, Q6_K attention, the MTP draft block, made from Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-S-v2-IQ2_XXS",
        "file": "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-{i:05d}-of-{n:05d}.gguf", "shards": 3, "download_gb": 96.5,
        "sha256": {
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf":
                ["e56ef50efd71d81ddf6249e08bd2ef5af96ed348c2b7a004491d46168d0ecaa7",    # "glm5-next" (v1.0.28)
                 "a507f2b7b25e04624ee55c631d3b281caea7cfffc747e5247970cd5a7ea91b3f"],   # "glm5next", before
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00002-of-00003.gguf":
                "2d65d88a69f8dc124c8d24bd33b33161ddab218918b4253ad4f500aeede84df5",
            "GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00003-of-00003.gguf":
                "a6a981c4fee7a53d78bdbd97d8f465d48ddf439cf938ff90617f395e0351b5a6"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-S24": {
        "about": "Maya-S24, Maya-S with Q4_K attention and shared experts: about 1.5 GB less that stays on the GPU, so "
                 "24 GB cards hold more experts (decode about 14% faster there, 11% on 32 GB), close to Maya-S in "
                 "quality",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-S24",
        "file": "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-{i:05d}-of-{n:05d}.gguf",
        "was": ["GLM-5.3-Flash-Maya-S24-IQ2_XXS-{i:05d}-of-{n:05d}.gguf",
                "GLM-5.3-Flash-Maya-S24-{i:05d}-of-{n:05d}.gguf"], "shards": 3, "download_gb": 94.7,
        "sha256": {
            "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00001-of-00003.gguf":
                ["8ebd5c74bcf65ece2a8699b7b6fd2b8d177f350ed252cee252fdd99b86a0138c",
                 "3dc347757686c1435eae36c4872f5c151191cc5063bc188ce699b97700aae076"],
            "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00002-of-00003.gguf":
                "4bd445da3a0128a9c5b32228c924e0a622aa4a132c7a8dc9a2c207a4beea8e34",
            "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00003-of-00003.gguf":
                "a6fd4e88007ac49b431c7d02be34e43ab7a09224d53a0c52782327613cb41917"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-XS-IQ1_S": {
        "about": "Maya-XS, Project Maya's smallest GLM quant: about 73 GB, for PCs with 64 GB of RAM and a 24 GB "
                 "card at 128K context (KV cache in int8; thinking budget 8192)",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "6728cf78a5b198a6a6f84b7fb88956a17ec56521",
        "folder": "Maya-XS", "file": "GLM-5.3-Flash-Maya-XS-IQ1_S-{i:05d}-of-{n:05d}.gguf", "shards": 2,
        "download_gb": 73.4, "kv": "int8", "context": 131072, "min_ram_gb": 64, "min_vram_gb": 24,
        "sha256": {
            "GLM-5.3-Flash-Maya-XS-IQ1_S-00001-of-00002.gguf": "711b427e55fa6da1cbe28933eb00174424f6201f0210b5e4a216d5e33b6222c7",
            "GLM-5.3-Flash-Maya-XS-IQ1_S-00002-of-00002.gguf": "a9841bc37abef5e3e5bc84e49e753b67f7e249098e8e9bc3dd17415104112885"},
        "vision": {
            "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main",
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-M": {
        "about": "Maya-M, Project Maya's larger quant: error-feedback-rounded IQ2_S gate/up experts, IQ3_XXS down "
                 "projections (IQ3_S in the most sensitive layers), Q6_K attention, the MTP draft block, made from "
                 "Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-M",
        "file": "GLM-5.3-Flash-Maya-M-IQ2_S-{i:05d}-of-{n:05d}.gguf",
        "was": ["GLM-5.3-Flash-Maya-M-{i:05d}-of-{n:05d}.gguf"], "shards": 3, "download_gb": 116.0,
        "sha256": {
            "GLM-5.3-Flash-Maya-M-IQ2_S-00001-of-00003.gguf":
                ["3f740d1b1022b4a0ddad50cd861bb071dd415bde94afa1292d0476641186c638",
                 "3ac0f066ec45af3432d59b33de49bdfb29156432b627240b35769c7d02cc6c02"],
            "GLM-5.3-Flash-Maya-M-IQ2_S-00002-of-00003.gguf":
                "285951d2afa0cd98285b03d0dc4aa68d83daf1a6f4a2594fe40b0cdd27ade485",
            "GLM-5.3-Flash-Maya-M-IQ2_S-00003-of-00003.gguf":
                "ebf1ce713f71207747e10eeebed87597969d8b5e1d9dd817420d2c2f7ca51e0e"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    # Maya-M with Blackfrost_AI's weight edit, in a repo of its own (the same label as Maya-M's: Hugging Face groups by
    # it per repo); the vision files are Maya's, from the main repo
    "Maya-M-Derisked": {
        "about": "Maya-M-Derisked, Maya-M with a directional weight edit by Blackfrost_AI that reduces blanket "
                 "refusals: the same IQ2_S quant, files and size, some of its weights changed (experimental)",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF", "revision": "main", "folder": "",
        "file": "GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-{i:05d}-of-{n:05d}.gguf", "shards": 3, "download_gb": 116.0,
        "sha256": {
            "GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-00001-of-00003.gguf":
                "da6a9755946f85a68aed34e80e86a7cf942f4ca637e605be81481979c871591e",
            "GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-00002-of-00003.gguf":
                "afc0705798bb293495168889a63186193fcd84e271aea3cde53c4814e9c96449",
            "GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-00003-of-00003.gguf":
                "e397c79669b84e8f54b7c946def0ea94f59771eecd7575766a3f70abfdd12157"},
        "vision": {
            "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main",
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    "Maya-L": {
        "about": "Maya-L, Project Maya's largest quant: error-feedback-rounded IQ3_S gate/up experts, IQ4_XS down "
                 "projections (Q5_K in the most sensitive layers), Q6_K attention, the MTP draft block, made from "
                 "Z.ai's FP8 release",
        "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "Maya-L",
        "file": "GLM-5.3-Flash-Maya-L-IQ3_S-{i:05d}-of-{n:05d}.gguf",
        "was": ["GLM-5.3-Flash-Maya-L-{i:05d}-of-{n:05d}.gguf"], "shards": 4, "download_gb": 156.3,
        "sha256": {
            "GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf":
                ["c41aeaf3a3e0022150b7c355e7f4df6c92ba68c4794e3e827e02e56b54886b9e",
                 "5fc82a6c9af4c6898e8d45cf32964f9a7b10640caf51e82be735d2d50e9f3d45"],
            "GLM-5.3-Flash-Maya-L-IQ3_S-00002-of-00004.gguf":
                "351d59366afb7be7dfc3fb7af91281f7ccf00760b17dc45f82090325a7b34f64",
            "GLM-5.3-Flash-Maya-L-IQ3_S-00003-of-00004.gguf":
                "3685962fdfeaf130b74f98d3ce703ee33d818aa9991725932cb305b7c48db8af",
            "GLM-5.3-Flash-Maya-L-IQ3_S-00004-of-00004.gguf":
                "1af62cd72de6460c85369d988abbb790d6ec3a97507df51faf9bf32aa2b600d0"},
        "vision": {
            "folder": "vision", "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf",
            "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    # turboderp's EXL3 (exllamav3) quants, read in place from their safetensors: the pack (tools/exl3_pack.py) puts a
    # small GGUF of the unquantized tensors beside them.  "files": every file of the branch with its size (what is
    # downloaded, and how a download is known whole); "revision": the branch's commit the hashes belong to.
    "EXL3-3.05bpw": {
        "about": "turboderp's EXL3 quant at 3.05 bits per weight (exllamav3's trellis format, which the engine reads "
                 "in place from its safetensors), with the MTP draft block - NVIDIA from V100, and AMD",
        "format": "exl3", "repo": "turboderp/GLM-5.3-Flash-exl3", "branch": "3.05bpw",
        "revision": "332ab457b709b7ba30dd9a448be5de03b80a7ac9", "folder": "", "download_gb": 125.3,
        "files": {
            "config.json": 86509, "LICENSE": 1070, "README.md": 7238, "chat_template.jinja": 8617,
            "generation_config.json": 194, "kpool_aux.safetensors": 11548768,
            "model-00001-of-00015.safetensors": 8589636152, "model-00002-of-00015.safetensors": 8588356452,
            "model-00003-of-00015.safetensors": 8590179904, "model-00004-of-00015.safetensors": 8588365508,
            "model-00005-of-00015.safetensors": 8588365508, "model-00006-of-00015.safetensors": 8588365500,
            "model-00007-of-00015.safetensors": 8590186912, "model-00008-of-00015.safetensors": 8588365508,
            "model-00009-of-00015.safetensors": 8588365508, "model-00010-of-00015.safetensors": 8588365500,
            "model-00011-of-00015.safetensors": 8590186912, "model-00012-of-00015.safetensors": 8588365508,
            "model-00013-of-00015.safetensors": 8588365508, "model-00014-of-00015.safetensors": 8590694168,
            "model-00015-of-00015.safetensors": 2059162376, "model.safetensors.index.json": 15960981,
            "mtp.safetensors": 2862645889, "processor_config.json": 909, "quantization_config.json": 47905678,
            "tokenizer.json": 20217442, "tokenizer_config.json": 761},
        "sha256": {
            "config.json":
                "0bef79e79b28fa9c51d57c49bd53adc59005c916832879ad3f0862e284a7bc06",
            "LICENSE":
                "30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1",
            "README.md":
                "4f46db37e6a7cbe3a494d6d7f616a96f4fffe6d1de68be361199a715c8d928fc",
            "chat_template.jinja":
                "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3",
            "generation_config.json":
                "230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f",
            "kpool_aux.safetensors":
                "39053cd051c75e81b7e608818111a4b4c3c57da540599089419fe11b51ca02d3",
            "model-00001-of-00015.safetensors":
                "6b6b96190a5114a55d058e3f4ae8a4350ac334dfcf804f61c39d5c6e40340ba4",
            "model-00002-of-00015.safetensors":
                "29fda4953918a156758169afa120929fdd35407860eddf5cad0cb747f65475ed",
            "model-00003-of-00015.safetensors":
                "9844130df84df3138706a0a708775c8d010a7ed370539880e327cac6bec7f2b7",
            "model-00004-of-00015.safetensors":
                "4f2a6fb68fb6cceab231d9c4a54b96174b4ec2b1bbf5cf042a6e4be37c739e29",
            "model-00005-of-00015.safetensors":
                "73cd6d0b5ba7f6d0ce23982729c6342d38114dd89d8a884a2fa08133ef52b47a",
            "model-00006-of-00015.safetensors":
                "160fcfc258b61c7aa7cbc27c4d6a33daee8bfe6dbb0e7dd5e73ce901789f391e",
            "model-00007-of-00015.safetensors":
                "a94f480c4256b7e5311b5384574471caff4f4fc0def8c3e1c57c411222cb1ce4",
            "model-00008-of-00015.safetensors":
                "e513ba853f8b8cb7dac8f17ff9f281787a63cc67daea15fc288591f93eec62f5",
            "model-00009-of-00015.safetensors":
                "825928573f534c09dc8e157356359bc525e883933777fa0d09d7a30a3f13d8b0",
            "model-00010-of-00015.safetensors":
                "f82170395c010c1b40b3695bcca439ac54445b268936ee71eb48c28f347d0d0d",
            "model-00011-of-00015.safetensors":
                "240ece46708e5c41aaf365bb2659ed7da2134bc12d9fa9b5393555c103228c6f",
            "model-00012-of-00015.safetensors":
                "2738042b40d05e144e54ed3c1013831d7dc156b694a1d68a49d55b25369ac8d9",
            "model-00013-of-00015.safetensors":
                "cf097956efdf8c307a6e560786445f1715c7bba9b2bddfb300f7346b1e1eb3ba",
            "model-00014-of-00015.safetensors":
                "1add7e6d8f4866e5aeecf77ac9ca2aacce3f071455fa9c167fa9bc678551c923",
            "model-00015-of-00015.safetensors":
                "d14a0087710b04a680f2d2a8769fa26f4e53dd3c1d5ce5391798ba25da1947f4",
            "model.safetensors.index.json":
                "559538d5006f757c809ef0a545062c2f79abb0140667399d590af0da96f26735",
            "mtp.safetensors":
                "7274f06d75842edfa258830c836fd790a19324513134e05cd1e931929c4c384d",
            "processor_config.json":
                "aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed",
            "quantization_config.json":
                "308d9a4b28407e2ada887f512bc8793c608586a73d9e0108ca258220ac714280",
            "tokenizer.json":
                "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
            "tokenizer_config.json":
                "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"},
        # the pictures: the vision tower and its tokenizer are the same in every GLM-5.3-Flash quant
        "vision": {
            "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "vision",
            "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf", "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    # the same at 4.05 bits: 4-bit routed experts (3-bit at 3.05), mostly 6-bit dense weights; the indexer's kpool
    # tensors are in its shards (3.05bpw has them in kpool_aux.safetensors)
    "EXL3-4.05bpw": {
        "about": "turboderp's EXL3 quant at 4.05 bits per weight (exllamav3's trellis format, which the engine reads "
                 "in place from its safetensors), with the MTP draft block - NVIDIA from V100, and AMD",
        "format": "exl3", "repo": "turboderp/GLM-5.3-Flash-exl3", "branch": "4.05bpw",
        "revision": "2a30229e67012798ba9f0cd832bb78abf4c363d5", "folder": "", "download_gb": 165.2,
        "files": {
            "config.json": 86509, "LICENSE": 1070, "README.md": 7238, "chat_template.jinja": 8617,
            "generation_config.json": 194, "model-00001-of-00019.safetensors": 8587818716,
            "model-00002-of-00019.safetensors": 8587936824, "model-00003-of-00019.safetensors": 8590375904,
            "model-00004-of-00019.safetensors": 8590340084, "model-00005-of-00019.safetensors": 8590383216,
            "model-00006-of-00019.safetensors": 8587986708, "model-00007-of-00019.safetensors": 8590342372,
            "model-00008-of-00019.safetensors": 8590383800, "model-00009-of-00019.safetensors": 8587944528,
            "model-00010-of-00019.safetensors": 8590383896, "model-00011-of-00019.safetensors": 8590342356,
            "model-00012-of-00019.safetensors": 8590383392, "model-00013-of-00019.safetensors": 8587986540,
            "model-00014-of-00019.safetensors": 8590342372, "model-00015-of-00019.safetensors": 8590383888,
            "model-00016-of-00019.safetensors": 8590341500, "model-00017-of-00019.safetensors": 8587986956,
            "model-00018-of-00019.safetensors": 8590342372, "model-00019-of-00019.safetensors": 6661824340,
            "model.safetensors.index.json": 15963650, "mtp.safetensors": 3793519833, "processor_config.json": 909,
            "quantization_config.json": 47905678, "tokenizer.json": 20217442, "tokenizer_config.json": 761},
        "sha256": {
            "config.json":
                "df80c17a68120aeae4c1eca8a9aa67866603484f0487cd5458c08dfd3c45156d",
            "LICENSE":
                "30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1",
            "README.md":
                "4f46db37e6a7cbe3a494d6d7f616a96f4fffe6d1de68be361199a715c8d928fc",
            "chat_template.jinja":
                "41cff9af7b3a86c96751b107a8444f245fbda0bd5320b636a5bb1f7f4ba1a5c3",
            "generation_config.json":
                "230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f",
            "model-00001-of-00019.safetensors":
                "1c7a87b0fb108521d0cf3e495ca63c37732a06744131e8933ca12d2df0d8806e",
            "model-00002-of-00019.safetensors":
                "a67cd8e559ac013fc0d1dc58b66bd105e54bee0a31147b295089b2537646d4dc",
            "model-00003-of-00019.safetensors":
                "d069cea25c6082a70b2dbc601725687f8d164a5dd047894acb58476b4175c1af",
            "model-00004-of-00019.safetensors":
                "d222d3137bec069a165219542d49d02defef2df69763464e93463a176e2ba935",
            "model-00005-of-00019.safetensors":
                "6fdd4557f29e1b77248b6bbd6b35c691ef4b1dc63b7305c9f5cb0080e1c82836",
            "model-00006-of-00019.safetensors":
                "3e9c1f39a631dedd5cce7614947d1f3eab71c93dbb4c79bd2eec0ab413236055",
            "model-00007-of-00019.safetensors":
                "62eca4decce10159107948a320083d8e9dcf60f25e058877a1662b80f1412220",
            "model-00008-of-00019.safetensors":
                "e8f8762fa1c6a698fc74d50760b82ec72d6043fb9ef6e05dac4032999f57d7c5",
            "model-00009-of-00019.safetensors":
                "f9d2e2df7fc7a65c33514a584e5a228d5eb4bec72fc516d9c649c94b06e3d00e",
            "model-00010-of-00019.safetensors":
                "5eb19bda8461e44323898172492f9f5c948f5c8ad00816b0e004feaf8b11ccfa",
            "model-00011-of-00019.safetensors":
                "f2e687a3d410af6bd48f90d5770fac1054ad5c337d3cfa2bd56a5600387ba36a",
            "model-00012-of-00019.safetensors":
                "caee2cdd11e781195a3bbb70b53da0c08c4add3485e200a4a28119d238201523",
            "model-00013-of-00019.safetensors":
                "094d5b9341505d25852993e0eef6d137ba75c4179bec2204bd067af264da5dd1",
            "model-00014-of-00019.safetensors":
                "765f21794314da81bccdf6a1c9b3fff3f86711d34d83f7551667455a9775dc1a",
            "model-00015-of-00019.safetensors":
                "f08d208c8ad27b3bb9c75c42c90e33a3e8acd722ccf5f4f74df1638084d46067",
            "model-00016-of-00019.safetensors":
                "e4c0bbf93639fd96eddb9a38505c4de99d650f82d376c0e303a7c3fd38544c3d",
            "model-00017-of-00019.safetensors":
                "8e954f3e1660889a50974f46e66ad75a4e3bcadb0398a91646d58a09d9c2f54c",
            "model-00018-of-00019.safetensors":
                "bad6f1e51e7bd25a4da4fe55efe4f91512bf4e82085db2baa5cce1adea027c58",
            "model-00019-of-00019.safetensors":
                "de2031315b49f89e3d091e92bead8b239ea46e304236f34b5734a68b0b07d547",
            "model.safetensors.index.json":
                "ee1f2dbea800dea0b4225c38193f7ef41180ed42dd51f80a0cee58daf44cf606",
            "mtp.safetensors":
                "1bd09bd4eff4adc568e4ac3e92efb436c2d3aa5f0dceab4c72005d796066822d",
            "processor_config.json":
                "aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed",
            "quantization_config.json":
                "d18c35bf6184a67a3b612b15c216bdfdf9e81505c0e02b646152925b13fafef3",
            "tokenizer.json":
                "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
            "tokenizer_config.json":
                "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"},
        "vision": {
            "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "vision",
            "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf", "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
    # turboderp's self-calibrated quant at 4.10 bits (exllamav3 1.6): 4-bit routed experts (5-bit in the first MoE
    # layer), 8-bit attention and shared experts, 6-bit dense layers and output; the MTP block is in its shards (layer
    # 45 of the index, no mtp.safetensors)
    "EXL3-sc_4.10bpw": {
        "about": "turboderp's self-calibrated EXL3 quant at 4.10 bits per weight (exllamav3's trellis format, which the "
                 "engine reads in place from its safetensors): 4-bit routed experts like EXL3-4.05bpw, 8-bit "
                 "attention and shared experts, with the MTP draft block - NVIDIA from V100, and AMD",
        "format": "exl3", "repo": "turboderp/GLM-5.3-Flash-exl3", "branch": "sc_4.10bpw",
        "revision": "c6b8ac3f6069173343e253e1cba90536ed2c72dc", "folder": "", "download_gb": 167.8,
        "files": {
            "config.json": 7866, "LICENSE": 1070, "README.md": 8033, "chat_template.jinja": 10950,
            "generation_config.json": 194, "model-00001-of-00023.safetensors": 6756990246,
            "model-00002-of-00023.safetensors": 7614064514, "model-00003-of-00023.safetensors": 7615704591,
            "model-00004-of-00023.safetensors": 7614064514, "model-00005-of-00023.safetensors": 7615711593,
            "model-00006-of-00023.safetensors": 7614071504, "model-00007-of-00023.safetensors": 7615711593,
            "model-00008-of-00023.safetensors": 7614071504, "model-00009-of-00023.safetensors": 7615711593,
            "model-00010-of-00023.safetensors": 7614071504, "model-00011-of-00023.safetensors": 7615711593,
            "model-00012-of-00023.safetensors": 7614071504, "model-00013-of-00023.safetensors": 7615711593,
            "model-00014-of-00023.safetensors": 7614071504, "model-00015-of-00023.safetensors": 7615711593,
            "model-00016-of-00023.safetensors": 7614071504, "model-00017-of-00023.safetensors": 7615711593,
            "model-00018-of-00023.safetensors": 7614071504, "model-00019-of-00023.safetensors": 7615711593,
            "model-00020-of-00023.safetensors": 7614071504, "model-00021-of-00023.safetensors": 7615711593,
            "model-00022-of-00023.safetensors": 8567785824, "model-00023-of-00023.safetensors": 107056658,
            "model.safetensors.index.json": 16343098, "processor_config.json": 909,
            "quantization_config.json": 47833416, "tokenizer.json": 20217442, "tokenizer_config.json": 761},
        "sha256": {
            "config.json":
                "8e27397f93723062062d2a379d9c2a30e338906f5ad838183459ad44242a35aa",
            "LICENSE":
                "30b85b6b9659f2e78aa259f8faf5d920a68dee7c9ced3fa6dba1f19f2bc4fca1",
            "README.md":
                "5c48ce1a84c5901ab96d602333e9201fa385a73e9c298fab3467183b8b95a549",
            "chat_template.jinja":
                "0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5",
            "generation_config.json":
                "230c30609ecbbb9e6583bedde8e7bdda0c6eb8fe5fad0eaeb3d1b293d751cb4f",
            "model-00001-of-00023.safetensors":
                "d53ec79f27568969924ae5675e47da913f15d2641f5713a5a1821855951b5508",
            "model-00002-of-00023.safetensors":
                "41bec1004e051021fe70b169ebbcca54d2cddb4cb74a256f8fcc4100e7c4e5ac",
            "model-00003-of-00023.safetensors":
                "bb608460adf5844fc022fd26611d6a349875a1a12d57fa215fc4eb105b065e8b",
            "model-00004-of-00023.safetensors":
                "a96a54b70bb9e12212435c1a546927b2fb6499a8681c7f9f8f3801bc2851addb",
            "model-00005-of-00023.safetensors":
                "4d237d5f8fb96662948c863b4887221f8fa84cd313431b0094fd51f78b3a04d2",
            "model-00006-of-00023.safetensors":
                "d2c9f343e5e085c28ec46e8e41362d3944cd6359ec203c2f27eb217e0aac2867",
            "model-00007-of-00023.safetensors":
                "340bf0d4c5947397eb4486bcbcf37c36519d3da9ac6448145aaae78b6f54046f",
            "model-00008-of-00023.safetensors":
                "fc74dfd79a539ec720f046c76858ab76a0e53f5600f5d9313b095e06935e67cd",
            "model-00009-of-00023.safetensors":
                "caf28ef4a95c8bdf4e6ad7d06c570ab729837f635cb2734d55bf51a92627003c",
            "model-00010-of-00023.safetensors":
                "2e0e1e6bd27c54d53115e70e9eb84d396538db4b645cc14b53cc5ace1809707d",
            "model-00011-of-00023.safetensors":
                "19f264dbf6da1876a0b9eea32a1d468e9e938130af6c69a0f2e076ab9736c09d",
            "model-00012-of-00023.safetensors":
                "1306d044b597fed6ea88146c7d08e5b0da55ec736cbd2b4311d319cca9a78dcb",
            "model-00013-of-00023.safetensors":
                "cfda30da128c7f1787a3cf7c60abda6bab15a1d3c693698d4a8fd59f7f1a3966",
            "model-00014-of-00023.safetensors":
                "2fe60640bdf58982d1f39a0e7bfc93a37c54cf8249f31b9feb39098aa99bddb1",
            "model-00015-of-00023.safetensors":
                "eeb92d7822dd2bdd66b8541251d060b10b1c8c26b98455ab727a208f7dfccb76",
            "model-00016-of-00023.safetensors":
                "7ef771a0105c0e33df8335f75336a21f7b5836ab88c068153b5ec8eff38ae105",
            "model-00017-of-00023.safetensors":
                "bcddb8f5ba687ccec306153884a3201b6c15ba01e0a077e797adf3d8dd0e67f3",
            "model-00018-of-00023.safetensors":
                "1df5395e73a10e2bf93427c7bb5c726e2c3a81bc611b0a6c61ef2d9b25d51090",
            "model-00019-of-00023.safetensors":
                "e151f871e76e03ec729f11d41d56f56e85b6308f590853047b1871c7a480ca70",
            "model-00020-of-00023.safetensors":
                "eee4dcc5b46b2ecc206d1836a3b881ad1649dd8f3772451df90f0e4552d73566",
            "model-00021-of-00023.safetensors":
                "7fa6316ec90aa7ce3273fd48a4e7f0826af3b8b28643079cce1239db88a3fdbf",
            "model-00022-of-00023.safetensors":
                "0cd44a37d30d7651ab95474ff97c9b7368b61ff349041b42dc61edddd2608d14",
            "model-00023-of-00023.safetensors":
                "7cce8ab6fd899ccbe40db8cf9952647fe8e50fcdaab2787cac0a5f9cbeaf0d58",
            "model.safetensors.index.json":
                "aaae0dd2761dc33c3d56ddc54a5943877096b25739a963fb2ab83cc931a09f13",
            "processor_config.json":
                "aae38374c94b08cc9b0547c6e64f05b951bd9735cea571c6988f5ed552bed3ed",
            "quantization_config.json":
                "82dc427917ec37c3aebb25950f1a16378d7a2a7175fb37b5cf464a4add1e1b1e",
            "tokenizer.json":
                "19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d",
            "tokenizer_config.json":
                "98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc"},
        "vision": {
            "repo": "peasantsmith/GLM-5.3-Flash-Maya-GGUF", "revision": "main", "folder": "vision",
            "mmproj": "mmproj-GLM-5.3-Flash-F16.gguf", "vocab": "GLM-5.3-Flash-vocab.gguf", "download_gb": 1.14,
            "sha256": {"mmproj-GLM-5.3-Flash-F16.gguf":
                           "3627575df16bd152db0f3fd7e488d270b33f3a9e6c7fa3b1b8ac381faafde882",
                       "GLM-5.3-Flash-vocab.gguf":
                           "8f53cb1bd2e631c14ef413e3284735d9e53f3c508d07a6f609e705b487105912"}}},
}
# Downloads the setup's model list no longer offers (the owner, v1.1.34): they stay in MODELS, so --model EXL3-sc_4.10bpw,
# an install's update and a config that names them work as before; the list shows one only on a PC that has it set up
# (its default there: a setup again, --yes included, keeps the model).
MODELS_BY_HAND = ("EXL3-sc_4.10bpw",)
# GLM-5.3 (744B, glm-dsa) downloads, in MODELS' form: offered on every backend but Metal, and kept out of MODELS itself - its list, --model's help and the dashboard's catalog stay GLM-5.3-Flash's
# (glm_downloads() adds them).  Maya's own quant first: GLM-5.3-Maya-XS (v1.1.58; the experts in ~1.6 bits, one repo
# per model: peasantsmith/GLM-5.3-Maya-GGUF, folder Maya-XS), then the owner's pick "for now" (2026-11-06): unsloth's
# UD-IQ2_M.  Each pinned to the revision checked - each shard's size and sha256 from the Hugging Face API, equal to
# the sha256 of the copies made from it.  "memory_gb": the VRAM + RAM that keep every expert in memory (G53.MEMORY_GB
# when an entry names none: UD-IQ2_M's), "dense_gb": the dense weights the GPUs hold (G53.DENSE_GB likewise) - Maya-XS:
# its weights (169.98 GB by the GGUF's header) and UD-IQ2_M's headroom (the same model's cache and working memory), the
# dense part as measured on an AMD APU (16.98 GB; its header's non-expert tensors, 18.4 GB, less what UD-IQ2_M's
# measurement leaves out of its own 15.7)
G53_MODELS = {
    "GLM-5.3-Maya-XS": {
        "about": "Maya-XS, Project Maya's own GLM-5.3 quant (744B, 40B active a token; the routed experts in ~1.6 "
                 "bits), text only, drafting with its own NextN block",
        "repo": "peasantsmith/GLM-5.3-Maya-GGUF", "revision": "2fb34fe0d9398f33782b6ece7dca1bb798f8b168",
        "folder": "Maya-XS", "file": "GLM-5.3-Maya-XS-{i:05d}-of-{n:05d}.gguf", "shards": 4, "download_gb": 170.0,
        "memory_gb": 176, "dense_gb": 17.0, "maya": True,
        "bytes": {"GLM-5.3-Maya-XS-00001-of-00004.gguf": 47680896576,
                  "GLM-5.3-Maya-XS-00002-of-00004.gguf": 47406972992,
                  "GLM-5.3-Maya-XS-00003-of-00004.gguf": 47620593024,
                  "GLM-5.3-Maya-XS-00004-of-00004.gguf": 27281804864},
        "sha256": {
            "GLM-5.3-Maya-XS-00001-of-00004.gguf": "3c9b12e7d36c675e961185cd947841f650745abcb746b4c5042c2e97212cf75c",
            "GLM-5.3-Maya-XS-00002-of-00004.gguf": "3be4fd84c71938ab39a6834c8e70aba411e766b80fb68cd48c5f95d8b6bb16d8",
            "GLM-5.3-Maya-XS-00003-of-00004.gguf": "26ac27adf0d5d441aa10daf3cfb7eb7745447be26a8bce4cbdc7794f4c093fad",
            "GLM-5.3-Maya-XS-00004-of-00004.gguf": "ef968744e05d5f062e203dc22146700db00e35c80655cc4880cfeee047500c26"}},
    "GLM-5.3-UD-IQ2_M": {
        "about": "Z.ai's GLM-5.3 (744B, 40B active a token) in unsloth's UD-IQ2_M GGUF (its dynamic 2-bit quant), "
                 "text only, drafting with its own NextN block",
        "repo": "unsloth/GLM-5.3-GGUF", "revision": "346b3591c7f28d1a23716f97a065ecf12ec14771", "folder": "UD-IQ2_M",
        "file": "GLM-5.3-UD-IQ2_M-{i:05d}-of-{n:05d}.gguf", "shards": 6, "download_gb": 238.6,
        "bytes": {"GLM-5.3-UD-IQ2_M-00001-of-00006.gguf": 9428677,
                  "GLM-5.3-UD-IQ2_M-00002-of-00006.gguf": 49223976800,
                  "GLM-5.3-UD-IQ2_M-00003-of-00006.gguf": 49143176640,
                  "GLM-5.3-UD-IQ2_M-00004-of-00006.gguf": 49143176640,
                  "GLM-5.3-UD-IQ2_M-00005-of-00006.gguf": 49143176640,
                  "GLM-5.3-UD-IQ2_M-00006-of-00006.gguf": 41914650304},
        "sha256": {
            "GLM-5.3-UD-IQ2_M-00001-of-00006.gguf": "88ec1b9e924bbfe2f5e210bcedb5146aefca41567cdc021c344b253cb733b21d",
            "GLM-5.3-UD-IQ2_M-00002-of-00006.gguf": "fc036c1c96f30d0b8fa334339cd1f9c68b1a3ab7fbc9440ee92df2a65b2a5e11",
            "GLM-5.3-UD-IQ2_M-00003-of-00006.gguf": "684b069f8e7a6b28ba04dc2a4d63757db4270bdad95ac931cd013394c638b4da",
            "GLM-5.3-UD-IQ2_M-00004-of-00006.gguf": "10c735d4af5d923adff14edb9bfbef63e5dce6187720c5db08a45f630638eccb",
            "GLM-5.3-UD-IQ2_M-00005-of-00006.gguf": "69ee92423d65605f3b7b07744492d42fd85c0fd898127cb23a44f213823f4cff",
            "GLM-5.3-UD-IQ2_M-00006-of-00006.gguf": "b8f96f7183eba431e047c52adab9091b3e465819c83c8c42ed94697b63507708"}},
}


def glm_downloads() -> dict:
    """The GLM downloads the setup takes: MODELS and G53_MODELS."""
    return {**MODELS, **G53_MODELS}
# The DFlash2 drafter the setup offers instead of drafting with the model's MTP block (--dflash): a small block drafter
# made for GLM-5.3-Flash (any of the quants above), pinned to the revision it was measured with and verified.
DFLASH = {"repo": "Anbeeld/GLM-5.3-Flash-DFlash2-GGUF", "revision": "f3569acb7f9cee95e66d24442d75536eb00ab842",
          "file": "GLM-5.3-Flash-DFlash2-Q4_K_M.gguf", "bytes": 697017248,
          "sha256": "f4158ffb805b26c4fe439669c2009908a80ea89e1576ecea4d03eaded18f017c",
          "license": "CC BY-NC-ND 4.0"}   # (the repo's: non-commercial use only, shown wherever it is offered)
DRAFTS = ("mtp", "dflash2")                        # the setup's drafting choices; MTP the default
# DeepSeek-V4.1-Flash-1017: its own engine (src/ds4, build/maya-ds4) reads LibertAIDAI's EXL3 quant at 2.9 bits in
# place from its safetensors; its two Engram tables come from DeepSeek's base repo.  Every file is pinned and checked
# (tools/ds4/download.json).  Linux and NVIDIA RTX 30 / A100 or newer only: its vision tower's BF16 GEMMs, the CPU
# lane's NUMA memory.  Tensor parallel across pairs of equal GPUs (--tp 2; ./maya.sh --calibrate measures the others),
# the rest of the experts in RAM.
DS4 = "DeepSeek-V4.1-Flash-1017"
DS4_EXE = BUILD / ("maya-ds4.exe" if WIN else "maya-ds4")
DS4_MANIFEST = ROOT / "tools" / "ds4" / "download.json"
DS4_PACK_TOOLS = ("tools/ds4/pack.py", "tools/ds4/engram_hash.py", "tools/ds4/st.py", "tools/ds4/chat_template.jinja")
DS4_MIN_CC = 70                                    # (V100 on: the experts' kernels before sm_80 since v1.1.20)
# ... or unsloth's GGUFs (tools/ds4/download_gguf.json, packed by tools/ds4/pack_gguf.py): dense weights BF16 (UD-Q8_K_XL)
# or Q8_0 (UD-Q4_K_XL), routed experts MXFP4, read in place - the experts VRAM does not hold come from RAM or, through
# the OS page cache, from the SSD, so it runs whatever the memory; Volta or newer.  The DSpark drafter and the vision
# tower, which the GGUFs leave out, come from DeepSeek's own checkpoint (download_gguf.json's "drafter" and "vision").
# Maya's own quants of it (download_gguf.json's "maya": Maya-S, the experts in 2-3 bits; Maya-M once published) are
# each one GGUF split with everything in it, the two Engram tables in Q6_K (made from DeepSeek's released weights).
# Files of an earlier layout are used as they are when a folder holds them whole (ds4_earlier): v1.1.16's Maya-S (4
# shards + unsloth's MXFP8 shard of the Engram tables in <...-GGUF>/Engram), unsloth's GGUFs before their re-shard.
DS4_GGUF_MANIFEST = ROOT / "tools" / "ds4" / "download_gguf.json"
DS4_GGUF_PACK_TOOLS = ("tools/ds4/pack_gguf.py", "tools/ds4/chat_template.jinja")
DS4_QUANTS = {"gguf-q8": "UD-Q8_K_XL", "gguf-q4": "UD-Q4_K_XL", "maya-s": "Maya-S", "maya-m": "Maya-M", "maya-xs": "Maya-XS",
              "exl3-2.25": "EXL3 2.25bpw", "exl3": "EXL3 2.9bpw", "exl3-3.5": "EXL3 3.5bpw", "abliterated-128g": "Mix-128G",
              "abliterated-256g": "Mix-256G", "gguf-iq1_s": "UD-IQ1_S", "gguf-iq1_m": "UD-IQ1_M",
              "gguf-iq2_xxs": "UD-IQ2_XXS", "gguf-q2_k_xl": "UD-Q2_K_XL", "gguf-iq3_xxs": "UD-IQ3_XXS",
              "gguf-iq3_s": "UD-IQ3_S", "gguf-q3_k_xl": "UD-Q3_K_XL"}
DS4_MAYA = ("maya-s", "maya-m", "maya-xs")
# unsloth's UD-Q4_K_XL is no longer in the setup's list (the owner, v1.1.34): --ds4-quant gguf-q4 and an install of it
# work as before; the list shows it only on a PC that has it set up
DS4_BY_HAND = ("gguf-q4",)
# unsloth's smaller UD quants (its re-upload of 2026-10-28, each pinned to that revision in download_gguf.json): the
# experts 1.6-3.5 bits by layer, the dense weights Q5_K - BF16, the Engram tables in the last shard (IQ4_NL, Q5_K,
# Q6_K or MXFP8; read from the SSD as every quant's).  --ds4-quant names one; as UD-Q4_K_XL, the list shows one only on
# a PC that has it set up - but for DS4_UD_LISTED, which the list always shows
DS4_UD = ("gguf-iq1_s", "gguf-iq1_m", "gguf-iq2_xxs", "gguf-q2_k_xl", "gguf-iq3_xxs", "gguf-iq3_s", "gguf-q3_k_xl")
DS4_UD_LISTED = ("gguf-iq1_m", "gguf-iq3_s")   # (UD-IQ3_S: close to Q8's quality; listed between Maya-S and the Q8, the owner, v1.1.43; UD-IQ1_M before UD-IQ3_S, v1.1.44)
# ... and orcarouter's abliterated model (its refusals removed from the weights) as GGUFs (download_gguf.json's
# "abliterated": Mix-128G and Mix-256G, the experts in other low-bit types layer by layer; a gated repo, so its download
# sends the user's Hugging Face token), in <models>/DeepSeek-V4.1-Flash-1017-abliterated-GGUF/<version>, served under a
# name of its own.  Each version's shard 2 holds only the n-gram (Engram) table - DeepSeek's own, quantized - so it is
# left out and Maya's Q6_K tables are read instead (ds4_abl_engram_step: one copy for both versions).  --ds4-quant
# abliterated: the version that fits this PC's VRAM + RAM (ds4_abl_recommended)
DS4_ABL = ("abliterated-128g", "abliterated-256g")
DS4_ABL_ALIAS = "abliterated"
DS4_ABL_MODEL_NAME = "deepseek-v4.1-flash-abliterated"
DS4_DEFAULT_QUANT = "gguf-q8"
DS4_GGUF_MIN_CC = 70
# (the model's own reach, natively: 1M positions - its config's max_position_embeddings, nothing scaled; 512K and 1M
# cost VRAM from the expert cache, about 1.1 KB a position on each GPU of a tensor-parallel group, and a full prompt
# there reads slowly: the indexer's work grows with the square of its length)
DS4_CONTEXTS = [32768, 65536, 131072, 262144, 524288, 1048576]
# the context the setup recommends: 256K where the PC's VRAM + RAM hold the model's files and the context's caches
# (ds4_recommended_context, as MiniMax's), else - a PC that reads experts from the SSD, or RAM not readable - 64K
DS4_DEFAULT_CONTEXT = 65536
DS4_RECOMMENDED_CONTEXT = 262144
DS4_KV_BYTES = 1126                                 # a position's caches on each GPU (~1.1 KB: FP4 KV, index keys)
DS4_MODEL_NAME = "deepseek-v4.1-flash"
DS4_SAMPLING, DS4_EFFORT = Q.MODEL_DEFAULTS["deepseek"]   # DeepSeek's own (its card; encoding.py's default: an
#                                                           effort of 75)
DS4_MEMORY_GB = 230                                 # VRAM + RAM that hold it all (211 GB of weights, the pinned
                                                    # tables); with less the engine reads experts from the SSD
DS4_35 = DS4 + "-3.5bpw"                            # the same model in LibertAIDAI's EXL3 quant at 3.5 bits
DS4_225 = DS4 + "-EXL3-2.25bpw"                     # ... and in turboderp's at 2.25 bits (its repo's 2.25bpw branch)
# The EXL3 quants: each one's maker ("by"), manifest, the GB of its weights, the VRAM + RAM that hold all of it (the
# engine's own use measured with every expert resident, the context aside) and its folder (<models>/<dir>).
# LibertAIDAI's card: perplexity over the unquantized model's on held-out code / prose +1.8% / +0.4% at 2.9 bits, +0.8%
# / +0.1% at 3.5.  turboderp's 2.25: its vision tower is EXL3 too (6 bits), which the engine does not read - "tower":
# DeepSeek's BF16 one (the vision folder a GGUF install makes, ds4_exl3_vision_step) is packed in its place
DS4_EXL3 = {"exl3": {"bits": "2.9", "by": "LibertAIDAI", "manifest": DS4_MANIFEST, "weights_gb": 211,
                     "memory_gb": DS4_MEMORY_GB, "dir": DS4},
            "exl3-3.5": {"bits": "3.5", "by": "LibertAIDAI", "manifest": ROOT / "tools" / "ds4" / "download-3.5bpw.json",
                         "weights_gb": 258, "memory_gb": 280, "dir": DS4_35},
            "exl3-2.25": {"bits": "2.25", "by": "turboderp", "manifest": ROOT / "tools" / "ds4" / "download-2.25bpw.json",
                          "weights_gb": 159, "memory_gb": 160, "dir": DS4_225, "tower": "deepseek"}}
# MiniMax-M3.1-Flash: its own engine (src/mm3, build/maya-mm3) reads Mia-AiLab's EXL3 quant at 3.5 bits (EAQ v1) in place
# from its safetensors; every file is pinned and checked (tools/mm3/download.json, its DSpark draft a download of its
# own).  NVIDIA: the layers split across the GPUs, the most-routed experts in VRAM (data/expert-profile-mm3.bin), the
# others computed on the CPU from RAM.  Its chat template is Maya's own (tools/mm3/chat_template.jinja: MiniMax's plus Off).
MM3 = "MiniMax-M3.1-Flash"
MM3_EXE = BUILD / ("maya-mm3.exe" if WIN else "maya-mm3")

MM3_MANIFEST = ROOT / "tools" / "mm3" / "download.json"
MM3_PACK_TOOLS = ("tools/mm3/pack.py", "tools/mm3/st.py", "tools/strata_tokenizer.py", "tools/mm3/chat_template.jinja")
MM3_DIR = "MiniMax-M3.1-Flash-EXL3-3.5bpw"
# Where its EXL3 quant is offered: everywhere DeepSeek's EXL3 runs (NVIDIA from V100, cc 7.0 = DS4_MIN_CC; Linux and
# Windows) once maya-mm3 builds and runs there - MM3_ALL_PLATFORMS, the one switch.  Until then: Linux, RTX 30 / A100 or
# newer (validated on sm_86 and sm_120); MAYA_MM3_MIN_CC=70 takes a V100 for testing.  AMD (HIP): too, as GLM's EXL3
# quants there - maya-mm3's EXL3 kernels take the shared exl3.cu's HIP paths.
MM3_ALL_PLATFORMS = True
MM3_MIN_CC = 70                                    # (with MM3_ALL_PLATFORMS: DeepSeek's EXL3 reach)
MM3_MIN_CC_NOW = 80                                # (until then)
MM3_TEST_CC_ENV = "MAYA_MM3_MIN_CC"                # (testing only: 70 lets the setup take a V100 - see mm3_min_cc)
# Context: the model's own reach is 1M positions (config.json's max_position_embeddings).  With KV streaming (the
# setup's default) every position's K / V sits in pinned RAM - per token, by the cache's format (maya-mm3 --kv-cache;
# 60 layers x 4 KV heads x 128 dims x K and V, plus the scales) - and a ring of the newest 32K positions in VRAM; the
# setup recommends the largest size whose cache fits beside the model in this PC's VRAM + RAM (mm3_recommended_context).
MM3_CONTEXTS = [32768, 65536, 131072, 262144, 524288, 1048576]
MM3_DEFAULT_CONTEXT = 65536                        # (the smallest the setup recommends with the memory for it)
MM3_TESTED_CONTEXT = 262144                        # (past it: not run end to end yet - said, as the dashboard does)
# pinned RAM a context token, by --kv-cache format: half for K, half for V (the engine's row sizes: 60 x 4 rows of 256
# bytes BF16, 128 + 4 INT8 / FP8, 64 + 8 FP4); "K,V" mixes add K's half and V's half
MM3_KV_BYTES = {"bf16": 122880, "int8": 63360, "fp8": 63360, "fp4": 34560}
MM3_KV_FORMAT = "int8"                             # the format the engine uses when the config names none
# VRAM a context takes with KV streaming: the ring of each layer's newest K / V (MM3_KV_RING positions, any context
# past it) and every position's indexer key (60 layers x 128 dims in BF16) - the share of it that grows with the
# context is bounded where it would push experts out of VRAM (mm3_context_limit: as Qwen's, 12% of the VRAM, 6% on a
# PC that reads experts from the SSD)
MM3_KV_RING = 32768
MM3_INDEX_BYTES = 60 * 128 * 2
# Windows on an APU (Strix Halo): the GPU's carve-out holds the model, and the pinned KV cache is GPU-visible RAM, which
# Windows caps at half of the RAM it has (its "shared GPU memory") - the cache takes at most three quarters of that, the
# rest left to the engine's other host buffers (#663: a 31.7 GB Radeon 8065S PC halved its decode with 256K's 15.5 GB)
MM3_WIN_APU_KV_SHARE = 0.5 * 0.75
# --parallel above 1: no KV streaming, every conversation's whole cache in VRAM (~61 KB a position with INT8) - the setup
# recommends the context at which the caches take at most this share of the GPUs' memory, the rest left to the dense
# weights and the expert pool (the engine shortens a context that does not fit, with a NOTE)
MM3_PARALLEL_VRAM_SHARE = 1 / 3
MM3_MODEL_NAME = "minimax-m3.1-flash"
# MiniMax's model card and generation_config.json (54e4e69): temperature 1.0, top-p 0.95 - no top-k, no min-p
MM3_SAMPLING = Q.MODEL_DEFAULTS["minimax"][0]
# thinking level for requests that name none: "auto" - the card names no default effort, so none is set and the
# template's own default applies (adaptive: the model decides whether to think); the card's five efforts low .. max
MM3_EFFORT = Q.MODEL_DEFAULTS["minimax"][1]
MM3_MEMORY_GB = 250                                # VRAM + RAM: 179 GiB of weights, the experts not in VRAM pinned
MM3_PROFILE = ROOT / "data" / "expert-profile-mm3.bin"
# ... or Project Maya's own GGUF quants of it (tools/mm3/download_gguf.json, packed by tools/mm3/pack_gguf.py): the routed
# experts in ~1.6 bits (Maya-XS, ~93 GB) or ~2.1 bits (Maya-S, ~122 GB) on maya-ds4's low-bit kernels - NVIDIA from V100
# on and AMD (HIP), Linux and Windows; pictures and DSpark drafting from the companion files (MM3_COMPANIONS, below).
# Listed (and offered) once published: until then --model minimax-maya-xs / minimax-maya-s (--mm3-quant gguf-xs /
# gguf-s before) with --model-dir takes files already there.  Named as every Maya quant (tools/maya_quants.py): the
# model first, in the repo peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF (Maya-XS/, Maya-S/), each in <models folder>/<its
# full name> on this PC, its config maya-minimax-maya-xs.json; the manifest's "was": the names its files had before,
# used as they are.
MM3_GGUF_MANIFEST = ROOT / "tools" / "mm3" / "download_gguf.json"
MM3_QUANTS = {"exl3": "EXL3 3.5 bits", "gguf-xs": "MiniMax-M3.1-Flash-Maya-XS", "gguf-s": "MiniMax-M3.1-Flash-Maya-S"}
MM3_GGUF_PACK_TOOLS = ("tools/mm3/pack_gguf.py", "tools/ds4/pack_gguf.py", "tools/mm3/pack.py", "tools/mm3/st.py",
                       "tools/mm3/chat_template.jinja")
# the GGUFs' pictures and drafting: the EXL3 repo's vision tower (model-vision-00001-of-00001.safetensors, 1.7 GB, with
# config.json / preprocessor_config.json) and DSpark draft (dspark/, 2.3 GB) - download.json's own pinned files, the same
# for any quant - in <models folder>/MiniMax-M3.1-Flash-companions (the repo's layout; a whole EXL3 download beside the
# GGUF serves as well), indexed by pack_gguf.py.  NVIDIA and AMD: the tower's and the draft's matrix products run on
# cuBLAS / hipBLAS.
MM3_COMPANIONS = "MiniMax-M3.1-Flash-companions"
MM3_VISION_FILES = ("config.json", "preprocessor_config.json", "model-vision-00001-of-00001.safetensors")
MM3_GGUF_MIN_CC = 70                               # (maya-ds4's GGUF kernels' reach: V100 on)
# CYBER-FROST-3.8-PS (v1.1.32): Blackfrost-AI's fine-tune of Qwen/Qwen3.8-Flash-Next (qwen4exp: 512 experts, 10 a token;
# GatedDeltaNet and sparse-attention layers; hyper-connections; an n-gram table; an MTP block) in peasantsmith's one
# GGUF (tools/qwen/download.json: pinned, sha256-checked).  It runs on the `maya` engine's own Qwen path from the file
# in place: the experts in VRAM, the next-most-used in RAM as far as this PC's RAM goes, the rest read from the SSD; the
# 52 GB n-gram table read from the SSD row by row (never in RAM).  Pictures: the base model's vision tower (the fine-tune
# kept it byte for byte) as llama.cpp's mmproj, read by maya-vision when a picture arrives (download.json's "vision").  Its chat
# template is the GGUF's (thinking always on, reasoning effort low / medium / xhigh; Qwen's XML tool calls); sampling
# its card's (general.sampling: temperature 1.0, top-p 0.95, top-k 20).  The MTP drafter is built from the GGUF's
# NextN block (tools/mtp_from_gguf.py) and kept where it measures faster on this PC.
QWEN = "CYBER-FROST-3.8-PS"                        # (its id: the folder, the installer record, --model; never shown)
QWEN_SHOWN = Q.SHOWN["qwen"]                       # (its name on screen: "Qwen-3.8-Flash-Next (CYBER-FROST)")
QWEN_MANIFEST = ROOT / "tools" / "qwen" / "download.json"
QWEN_PACK_TOOLS = ("tools/iq_pack.py", "tools/strata_tokenizer.py", "tools/mtp_from_gguf.py", "tools/mtp_rt.py")
QWEN_MODEL_NAME = "cyber-frost-3.8-ps"
QWEN_SAMPLING, QWEN_EFFORT = Q.MODEL_DEFAULTS["qwen"]   # (its card's; its template's default level: Qwen's "xhigh")
QWEN_CONTEXTS = [32768, 65536, 131072, 262144]     # (the model's native reach: 256K)
QWEN_DEFAULT_CONTEXT = 65536
QWEN_MIN_CC = 70                                   # (V100 on: the Qwen path's kernels are validated from Volta)
QWEN_PROFILE = ROOT / "data" / "expert-profile-qwen.bin"
# KV bytes a context token: 12 attention layers x K and V (2 heads x 256) in INT8 with an FP16 scale a 64 values, and the
# indexer's pooled keys.  INT8 is the setup's KV format (--kv int8: the model card's own llama.cpp line runs q8_0 K / V;
# the attention's time the same as FP16's, half its VRAM left to experts)
QWEN_KV = "int8"
QWEN_KV_BYTES = 12 * (2 * 2 * 256 + 2 * 2 * 256 // 64 * 2 + 128)
# Qwen3.8-Flash-Next itself in unsloth's UD-IQ2_XS GGUF (v1.1.54; the only one of unsloth's UD quants listed): the same
# Qwen path, three shards (1 the metadata, 2 the weights, 3 the n-gram table alone) and its MTP head as a separate GGUF
# (tools/qwen/download_unsloth.json: pinned, sha256-checked; shard 3 from Maya's own repo, byte for byte unsloth's).  The
# n-gram table is memory-mapped (--ple-io mmap: the page cache, never loaded into RAM by the engine).
QWEN_UD = "ud-iq2_xs"                              # (its key: --qwen-quant, the installer record's qwen_quant)
QWEN_UD_DIR = "Qwen3.8-Flash-Next-UD-IQ2_XS"       # (its folder in the models folder, the installer's "quant")
QWEN_UD_MANIFEST = ROOT / "tools" / "qwen" / "download_unsloth.json"
QWEN_UD_MODEL_NAME = "qwen3.8-flash-next"
QWEN_QUANTS = ("q5_k_m", QWEN_UD)
# Hy4-Flash (v1.1.68): Tencent's tencent/Hy4-flash (hy_v4_flash, 220B with its n-gram tables: 42 layers and an MTP
# block, MLA attention with a sliding window and 12 indexed layers, 512 experts, 16 a token, 4 hyper-connection streams,
# Engram) on its own engine maya-hy4 (src/hy4; NVIDIA: CUDA, Linux and Windows, V100 on; AMD: HIP, Linux - a
# unified-memory APU (Strix Halo / Gorgon Halo) or Radeon cards - and Windows, experimental (not run by us yet); no Mac
# engine yet),
# which reads GGUF sets in place (packed by tools/hy4/pack_gguf.py).  tools/hy4/download.json is the one table of its
# files: Project Maya's own quants ("quants": which sizes are uploaded) and Tencent's own Q4_K_M from AngelSlim
# ("others": pinned, sha256-checked - the one set with the MTP block, so the one that drafts: --mtp; run on RTX 30 /
# A100 and newer on Linux, tensor parallel over the equal GPUs with the fastest holding the head and the drafts).
# What is not a routed expert sits in VRAM (a Maya quant's "plan": hy4_fit works out before any download whether it
# fits, and for which context); the engine fills the VRAM left with the most-routed experts and computes the others on
# its CPU lane straight from the memory-mapped file (RAM as the page cache).  The n-gram table (63 GB, one file for
# every set, from Maya's repo: downloaded once, hard-linked) is read from the SSD page by page, never loaded into RAM.
# Pictures: the model's vision tower (the mmproj file beside the set - Maya's own Q8_0 one for every Maya size,
# AngelSlim's BF16 for the Q4_K_M; not with --no-vision) runs in the engine a picture at a time: its weights stay in the
# file, and a picture's room (HY4_VISION_GIB while it is read) is lent by the engine's prompt buffers, idle then -
# no VRAM is kept for pictures, so every card that runs the model reads them at full size (an 8 GB card's buffers at
# its 256-token chunk: 1.0 GB).  It thinks at one level (high) or not at all (no_think); its sampling is its
# generation_config.json's (maya_quants.MODEL_DEFAULTS["hy4"]); one request at a time.  Each Maya size in <models
# folder>/<its full name> (a layout of it: <name>-8GB), its config maya-hy4-flash-maya-xs.json; the Q4_K_M in <models
# folder>/Hy4-flash, its config maya-hy4-flash.json.
HY4 = Q.MODEL["hy4"]                               # "Hy4-Flash" (its id: the installer record, --model)
HY4_SHOWN = Q.SHOWN["hy4"]
HY4_EXE = BUILD / ("maya-hy4.exe" if WIN else "maya-hy4")
HY4_MANIFEST = ROOT / "tools" / "hy4" / "download.json"
HY4_PACK_TOOLS = ("tools/hy4/pack_gguf.py", "tools/ds4/pack_gguf.py", "tools/strata_tokenizer.py")
HY4_PACK = "maya-hy4"                              # (the pack's folder in the model's: hy4.txt, tensors.txt, tokenizer/)
HY4_QUANTS = {k: Q.full_name("hy4", k) for f, k, _, _ in Q.QUANTS if f == "hy4"}   # "maya-xs": "Hy4-Flash-Maya-XS" ...
HY4_Q4 = "q4_k_m"                                  # Tencent's own Q4_K_M (AngelSlim): the table's "others"
HY4_MIN_CC = 70                                    # (built and run on V100 sm_70, RTX 30 and RTX 50)
HY4_Q4_TESTED_CC = 80                              # (the Q4_K_M with its drafts has run on RTX 30 and RTX 50 cards: Ampere on)
# (a Maya quant: every context up to the model's own reach, 1M, that fits the cards' VRAM - hy4_fit; the cache has no
#  RAM path in this engine yet, so a context is VRAM alone)
HY4_CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
HY4_TESTED_CONTEXT = 32768                         # (a Maya quant past it: not run end to end yet - said; the setup recommends up to it)
# (the Q4_K_M: run end to end up to 256K - above what F16 rows fit, the engine's --kv auto keeps a tensor-parallel
#  group's cache in 8-bit rows; the model's own reach: 1M - HY4_Q4_LONG, offered where the smallest GPU is large
#  enough for the cache: hy4_q4_contexts)
HY4_Q4_CONTEXTS = [32768, 65536, 131072, 262144]
HY4_Q4_LONG = [524288, 1048576]
HY4_Q4_LONG_SHARE = 0.25                           # (the cache in 8-bit rows: at most this share of the smallest GPU's VRAM)
# (Maya's sizes draft with Tencent's MTP block from a file of its own - download.json's "mtp": the engine rounds its BF16
#  matrices to Q8_0, HY4_MTP_VRAM of the file's bytes, always in VRAM, and keeps a prompt chunk of rows beside it)
HY4_MTP_VRAM = 0.56
# (... and where the block does not fit, with the size's DFly2 drafter - its "drafter": its matrices as they are in the
#  file, beside them its lists and rows, under HY4_DFLY2_EXTRA bytes)
HY4_DFLY2_EXTRA = 32 << 20
# (where both fit, which drafts is measured on the PC - speed_calibration_hy4: a short A/B, the faster kept unless
#  within HY4_DRAFT_TIE, where this rule stays; until it runs, the rule: the faster for the size on the V100 measured
#  (32 and 16 GB, server answers of code and chat, greedy and sampled): XS DFly2 (+10-20% over the block), S the block
#  (+5-35% on chat), L a tie (DFly2 ahead on greedy code, the block on sampled chat: the block kept); M not measured)
HY4_DRAFT_RULE = {"maya-xs": "dfly2", "maya-s": "mtp", "maya-m": "mtp", "maya-l": "mtp"}
HY4_DRAFT_TIE = 0.02
HY4_DRAFT_PROBE = {"mixed": True, "tokens": 512, "answer": 128, "warm": 32}
HY4_VISION_GIB = 0.81                              # (the engine's room for its largest picture, 2048 x 2048: 828 MB of the main GPU while it is read)
# (lent by the prompt buffers - src/hy4/hy4_model.cu's vis_encode: no VRAM of its own; the 8 GB layout's buffers at its
#  256-token chunk come to 1013 MB on an RTX 3050 at 4K-32K, so the fit counts nothing for pictures)
HY4_DEFAULT_CONTEXT = 65536
# its cache a context token: the 13 layers that keep every position (the 12 indexed ones and the MTP block) - the latent
# (512 + 64 rope, F16) and the indexer's pooled keys (128 F32 every 4 positions); the sliding-window layers keep a ring
# of fixed size.  With tensor parallel every GPU of the group holds all of it
HY4_KV_BYTES = 13 * (2 * (512 + 64) + 128 * 4 // 4)
# (a latent row in the engine's 8-bit rows: the 512 values int8 in blocks of 32 with an F16 scale, the rope part F16)
HY4_KV_Q8_ROW = 512 + 512 // 32 * 2 + 64 * 2
HY4_MODEL_NAME = "hy4-flash"
HY4_SAMPLING, HY4_EFFORT = Q.MODEL_DEFAULTS["hy4"]  # (Tencent's card: temperature 0.9, top-p 1.0; its template's High)
HY4_PROFILE = ROOT / "data" / "expert-profile-hy4.bin"
HY4_STANDARD = "standard"                          # the layout a size's own "files" and "plan" are
# VRAM in use on a card now that counts against the model (the desktop's; more is a program that can be closed - as
# GLM_FIT_USED_CAP_MIB), what a card that may drive a desktop is taken to hold when nvidia-smi gives no reading, and
# what a context the manifest does not list (worked out by hy4_need_gib, not measured on a card) may be off by
HY4_FIT_USED_CAP_GIB = 1.0
HY4_FIT_DISPLAY_GIB = 0.4
HY4_FIT_MARGIN_GIB = 0.25
# what a CUDA start takes of a card beside the free VRAM the engine plans with (cudaMemGetInfo) - its context and the
# driver's share - which nvidia-smi's free figure still holds under Linux, where the sums above were measured (0.57 GiB
# on a V100 32 GB: a context 308 MiB, the driver 274); under Windows CUDA's own figure has it out already (the display
# driver keeps it in its budget: a context shows as 68 MiB on an RTX 3050, whose engine draws its DFly2 line 0.48-0.53
# GiB above CUDA's figure) - hy4_engine_room adds this much back, or what this PC shows of it where that is more: the
# first guess only, until the engine's first start on the PC measures the figure for its card (HY4_ROOM_STORE)
HY4_FIT_CUDA_OWN_GIB = 0.5
# (Windows: CUDA's own share of each card in the setup's terms, as the engine measured it at its first start on this PC
#  - hy4_room_measure - by PC, card and driver; the setup and the Models page count it from then on)
HY4_ROOM_STORE = ROOT / "data" / "vram-room.json"
HY4_KV_RESIDENT = 32768                            # (KV streaming: the global layers' newest positions kept in VRAM)
# a small-card layout's prompt chunk doubled where it fits: the prompt scratch grows by about 0.9 GiB from 256 to 512
# tokens (an 8 GB card at 4K: measured) - counted as 0.95, conservatively
HY4_CHUNK_UP_GIB = 0.95
# Inkling-2-Mini (an EXPERIMENTAL PREVIEW, PR #1095): thinkingmachines' looped 26B on its own engine maya-ik2 (src/ik2,
# docs/INKLING2.md section 8) - NVIDIA only, Linux only, V100 / RTX 20 and newer with 8 GB or more (the engine's
# kernels per architecture: Volta's mma.m8n8k4, Turing's and Ampere's mma.sync); the experts the card does not hold
# stay in RAM (about 13 GB of RAM with an 8 GB card; tools/ik2_fit.py); AMD: RDNA 3 / 3.5 / 4 or a Halo / APU with 24 GB.
# Its file: unsloth's UD-Q4_K_XL (Unsloth's looped layout, which the engine reads) - tools/ik2/download.json, pinned
# by size and sha256; the setup offers no drafter (the engine's --draft is by hand, docs/INKLING2.md).  Pictures: the
# model's vision tower as unsloth's mmproj-BF16.gguf (download.json's "vision", beside the GGUF; not with --no-vision),
# read by maya-vision (tools/vision/inkling2v_patch.py) only while a request's pictures are encoded - on the GPU where
# its free VRAM holds the encoder, else on the CPU; never kept in VRAM, and the context's fit is the same.  What fits on
# the card (the context): tools/ik2_fit.py, shared with the Models page.  Its defaults: maya_quants.MODEL_DEFAULTS
# ["ik2-mini"] (its author's); thinkingmachines' chat template (tools/ik2/chat_template.jinja) in the tokenizer
# folder.  One request at a time.
import ik2_fit as IK2F  # noqa: E402  (tools/) what fits on the card
IK2 = Q.MODEL["ik2"]                               # "Inkling-2-Mini" (its id: the installer record, --model)
IK2_SHOWN = Q.SHOWN["ik2"]
IK2_EXE = BUILD / ("maya-ik2.exe" if WIN else "maya-ik2")
IK2_MANIFEST = ROOT / "tools" / "ik2" / "download.json"
IK2_TEMPLATE = ROOT / "tools" / "ik2" / "chat_template.jinja"
IK2_TOKENIZER_TOOLS = ("tools/strata_tokenizer.py", "tools/ik2/chat_template.jinja")
IK2_DIR = "maya-ik2"                               # (in the model's folder: tokenizer/, sources.json)
IK2_MODEL_NAME = "inkling-2-mini"
IK2_MIN_CC = IK2F.MIN_CC
IK2_PROFILE = ROOT / "data" / "expert-profile-ik2mini.bin"   # (the offload path's starting VRAM tier: maya-ik2's IK2_WARM)
FAMILIES =("glm", "deepseek", "minimax", "qwen", "hy4", "ik2")
SHARD_RE = re.compile(r"-(\d{5})-of-(\d{5})\.gguf$")


def exe_for_target(target: str) -> Path:
    """The engine binary for a build target name ("maya" or "maya-ds4"; EXE's and DS4_EXE's own stems) - resolved
    fresh on every call since select_build_backend() reassigns EXE/DS4_EXE once the backend (cuda/hip) is known."""
    if target == "maya-glm-helper":                    # (GLM's AMD helper program for the two-GPU mode: mixed_builds)
        return BUILD / ("maya-glm-helper.exe" if WIN else "maya-glm-helper")
    return {"maya": EXE, "maya-ds4": DS4_EXE, "maya-mm3": MM3_EXE, "maya-hy4": HY4_EXE, "maya-ik2": IK2_EXE}[target]
GLM_ARCHS = ("glm5-next", "glm5next")             # llama.cpp's spelling and unsloth's


def glm_archs() -> tuple:
    """The GGUF architectures the GLM setup takes: GLM-5.3-Flash's, and GLM-5.3's (glm-dsa, 744B, same engine)."""
    return GLM_ARCHS + (G53.ARCH,)


# the expert profile setup passes to every GLM config (tools/make_profile.py --glm):
# its ranking of every (layer, expert) pair decides each layer's VRAM slots and which experts start there
PROFILE = ROOT / "data" / "expert-profile-glm.bin"
# DeepSeek's starting expert profile (tools/ds4/profile.py): a pack without a profile of its own starts from it - the
# engine reads it to place the most-routed experts in VRAM and read the next ones ahead, then adds this PC's routing to
# it and saves it in the pack (<pack>/expert-profile.bin); its counts are scaled to a few thousand tokens' worth, so
# what this PC routes soon outweighs it
DS4_PROFILE = ROOT / "data" / "expert-profile-ds4.bin"
PACK_FILES = ("index.txt", "native_experts.txt", "dense.bin", "tokenizer/vocab.json", "tokenizer/merges.txt",
              "tokenizer/token_type.json", "tokenizer/chat_template.jinja")


# ------------------------------------------------------------------------------------------------ small helpers
def hf_url(repo: str, revision: str, folder: str, file: str) -> str:
    """A file's download URL on Hugging Face (folder "" = the repo's top)."""
    return HF.format(repo=repo, revision=revision, path=f"{folder}/{file}" if folder else file)


def read_json(path: Path) -> dict:
    try:
        return N.env_in(json.loads(path.read_text(encoding="utf-8-sig")))   # (a config's settings: see maya_names)
    except (OSError, ValueError):
        return {}


def configs() -> list:
    """The installed models' configs, the most recently used first - not the dashboard's settings the server keeps
    beside each (<config>.shared-settings.json), which a restart took for a config of the CUDA backend."""
    return sorted((p for p in ROOT.glob("maya-*.json") if not p.name.endswith(".shared-settings.json")),
                  key=lambda p: p.stat().st_mtime, reverse=True)


def zfs_arc_gb(path: str = "/proc/spl/kstat/zfs/arcstats") -> float:
    """What ZFS's ARC holds above its floor (c_min), in GiB: it gives that back under pressure, as the page cache does,
    but MemAvailable leaves it out (#89); 0 without ZFS."""
    vals = {}
    try:
        with open(path) as f:
            for line in f:
                p = line.split()
                if len(p) == 3 and p[0] in ("size", "c_min") and p[2].isdigit():
                    vals[p[0]] = int(p[2])
    except OSError:
        return 0.0
    if "size" not in vals or "c_min" not in vals:
        return 0.0
    return max(0, vals["size"] - vals["c_min"]) / 2**30


def mem_gb() -> tuple:
    """(total, available) RAM in GiB - the engine sizes its RAM tier from MemAvailable and ZFS's reclaimable ARC (on
    Windows ullAvailPhys)."""
    if WIN:
        m = S._memory_status()
        return m.ullTotalPhys / 2**30, m.ullAvailPhys / 2**30
    if sys.platform == "darwin":             # free + inactive + speculative pages: what a new program can take
        vm, total = S.out(["vm_stat"]), S.out(["sysctl", "-n", "hw.memsize"]).strip()
        page = re.search(r"page size of (\d+) bytes", vm)
        pages = sum(int(n) for n in re.findall(r"^Pages (?:free|inactive|speculative):\s+(\d+)", vm, re.M))
        return int(total or 0) / 2**30, pages * int(page.group(1) if page else 0) / 2**30
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024 / 2**30
    except (OSError, ValueError):
        pass
    avail = info.get("MemAvailable", 0.0)
    return info.get("MemTotal", 0.0), avail + zfs_arc_gb() if avail > 0 else avail


def apple_silicon() -> bool:
    """macOS on an M-series Mac: the machine of the Metal backend."""
    return platform.system() == "Darwin" and platform.machine() == "arm64"


def metal_gpus() -> list:
    """The Mac's one GPU, which shares its unified memory with the CPU.
    shortcut: 85% of the RAM stands for what Metal lets a program use (MTLDevice.recommendedMaxWorkingSetSize), the
    real figure is not asked - ask the engine for it if the estimate misleads."""
    return [{"index": 0, "name": S.out(["sysctl", "-n", "machdep.cpu.brand_string"]).strip() or "Apple GPU",
             "vram_gb": 0.85 * mem_gb()[0], "vendor": "apple"}]


def existing(path: Path) -> Path:
    """`path`, or its nearest folder that exists (where it would be created)."""
    while not path.exists() and path.parent != path:
        path = path.parent
    return path


def rotational(path: Path) -> bool:
    """True when `path` is on a spinning disk (best effort: findmnt + lsblk; on Windows the drive's MediaType)."""
    if WIN:
        drive = existing(path).drive[:1]
        return drive.isalpha() and S.out(
            ["powershell", "-NoProfile", "-Command", "Get-PhysicalDisk | Where-Object DeviceId -eq "
             f"(Get-Partition -DriveLetter {drive}).DiskNumber | Select-Object -ExpandProperty MediaType"]).strip() == "HDD"
    src = S.out(["findmnt", "-no", "SOURCE", "--target", str(existing(path))]).strip().split("[")[0]
    rota = S.out(["lsblk", "-ndo", "ROTA", src]).split() if src.startswith("/dev/") else []
    return bool(rota) and rota[0] == "1"


def tool_version(exe: str) -> tuple:
    m = re.search(r"(\d+)\.(\d+)", S.out([exe, "--version"]))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)


def venv_tool(name: str):
    p = Path(sys.executable).parent / (name + (".exe" if WIN else ""))
    return str(p) if p.exists() else None


def pick_cmake():
    """CMake 3.24 or newer: the one pip put into .venv first, then the system's."""
    for c in (venv_tool("cmake"), shutil.which("cmake")):
        if c and tool_version(c) >= (3, 24):
            return c
    return None


def choose(question: str, intro: list, options: list, default: int, yes: bool, outro: list = ()) -> int:
    """The answer's index among `options`, each (label, note or None, about or None): a menu on the setup's screen
    (tools/setup_tui.py; `about` shows under the highlighted one), else the numbered list and the number typed."""
    if S.UI is not None and not yes:
        pick = S.UI.choose(question, intro, options, default, outro)
        if pick is not None:
            return pick
    say()
    for line in intro:
        say("  " + line)
    for i, (label, note, _) in enumerate(options, 1):
        say(f"  {i}) {label}" + (f"   ({note})" if note else ""))
    for line in outro:
        say("     " + line)
    return int(ask(question, [str(i) for i in range(1, len(options) + 1)], str(default + 1), yes)) - 1


def gpu_label(g) -> str:
    if g.get("vendor") == "amd":
        return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB, {g['arch']})"
    if g.get("unified_memory"):
        return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB shared with the system)"
    return f"GPU {g['index']} ({g['name']}, {g['vram_gb']:.0f} GB)"


# v1.1.16 named the programs after Maya: the ones an earlier version compiled are renamed in place (nothing compiled
# again), their build stamps and the configs' paths follow (rename_programs, fix_config_names)
OLD_NAMES = {"strata": "maya", "strata-ds4": "maya-ds4", "strata-vision": "maya-vision"}


def rename_programs() -> None:
    ext = ".exe" if WIN else ""
    pairs = [(ROOT / d / (old + ext), ROOT / d / (new + ext)) for d in ("build", "build-hip")
             for old, new in (("strata", "maya"), ("strata-ds4", "maya-ds4"))]
    pairs.append((VBUILD / "bin" / ("strata-vision" + ext), VBUILD / "bin" / ("maya-vision" + ext)))
    for old, new in pairs:
        if old.exists() and not new.exists():
            try:
                os.replace(old, new)
            except OSError as e:   # (in use: a server still running; the next start renames it)
                warn(f"could not rename {old} to {new.name} ({e})")
    for d in ("build", "build-hip"):
        stamp = ROOT / d / "MAYA-BUILD.json"
        meta = read_json(stamp)
        targets = meta.get("targets") or (["strata"] if meta else [])
        if any(t in OLD_NAMES for t in targets):
            meta["targets"] = sorted({OLD_NAMES.get(t, t) for t in targets})
            stamp.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def fix_config_names(cfg: dict) -> bool:
    """A config written before v1.1.16 names the programs by their old names: True when it was changed.  A path moves to
    the new name only when the program is there under it (rename_programs renames Maya's own builds; a program
    elsewhere keeps its path)."""
    def renamed(p: str) -> str:
        q = Path(p)
        stem, ext = (q.name[:-4], ".exe") if q.name.endswith(".exe") else (q.name, "")
        if stem not in OLD_NAMES:
            return p
        new = q.with_name(OLD_NAMES[stem] + ext)
        return str(new) if (new if new.is_absolute() else ROOT / new).exists() else p
    done = False
    if cfg.get("exe") and renamed(cfg["exe"]) != cfg["exe"]:
        cfg["exe"], done = renamed(cfg["exe"]), True
    v = cfg.get("vision")
    if isinstance(v, dict) and v.get("exe") and renamed(v["exe"]) != v["exe"]:
        v["exe"], done = renamed(v["exe"]), True
    return done


def select_build_backend(backend: str) -> None:
    """Separate build folders and stamps prevent reusing a binary from the other backend."""
    global BUILD, EXE, STAMP, DS4_EXE, MM3_EXE, HY4_EXE, IK2_EXE
    BUILD = ROOT / {"hip": "build-hip", "metal": "build-metal"}.get(backend, "build")
    EXE = BUILD / ("maya-metal" if backend == "metal" else "maya.exe" if WIN else "maya")
    DS4_EXE = BUILD / ("maya-ds4.exe" if WIN else "maya-ds4")
    MM3_EXE = BUILD / ("maya-mm3.exe" if WIN else "maya-mm3")
    HY4_EXE = BUILD / ("maya-hy4.exe" if WIN else "maya-hy4")
    IK2_EXE = BUILD / ("maya-ik2.exe" if WIN else "maya-ik2")
    STAMP = BUILD / "MAYA-BUILD.json"


def hip_unified_memory(g) -> bool:
    # gfx1151 is Strix Halo / Gorgon Halo, even when KFD/sysfs has no product name or reports
    # only a tiny firmware VRAM carve-out. This is independent of vram_gb.
    return g.get("arch") == "gfx1151" or bool(g.get("integrated")) or bool(
        re.search(r"Ryzen AI Max|Radeon(?:\s*\(TM\))?\s+80[4-6]\dS", g.get("name", ""), re.I))


def apu_margin_gb(ram: float) -> float:
    """The memory both engines keep free for Windows / Linux and the other programs on a unified-memory APU,
    max(8 GB, 10% of the RAM)."""
    return max(8.0, 0.1 * ram)


def hip_reserve_mib(gpus) -> int:
    """The VRAM (MiB) an engine on these AMD GPUs leaves free on each: the desktop runs on them and allocates after
    the engine has filled its expert pool - 3072, but 1024 on a Linux APU (its pool is sized from the RAM it shares;
    Windows draws its desktop from the APU's carve-out too)."""
    if any(hip_unified_memory(g) for g in gpus) and not WIN:
        return 1024
    return 3072


def hip_clang(root: Path) -> Path | None:
    """ROCm's clang++: <root>/llvm/bin (a system ROCm on Linux), <root>/lib/llvm/bin (TheRock's wheels),
    <root>/lib/llvm/<version>/bin (a distribution's LLVM slot, e.g. Gentoo's), <root>/lib64/rocm/llvm/bin (a
    distribution's own ROCm packages under /usr, e.g. Fedora's), or <root>/bin (AMD's HIP SDK for Windows)."""
    exe = "clang++.exe" if WIN else "clang++"
    slots = sorted((root / "lib" / "llvm").glob(f"*/bin/{exe}"),
                   key=lambda p: [int(x) for x in re.findall(r"\d+", p.parent.parent.name)],
                   reverse=True) if not WIN else []
    # Debian/Ubuntu's own ROCm packages: hipcc drives the distribution's LLVM, /usr/lib/llvm-<N>/bin
    slots += sorted((root / "lib").glob(f"llvm-[0-9]*/bin/{exe}"),
                    key=lambda p: [int(x) for x in re.findall(r"\d+", p.parent.parent.name)],
                    reverse=True) if not WIN else []
    return next((p for p in (root / "llvm" / "bin" / exe, root / "lib" / "llvm" / "bin" / exe,
                             root / "lib64" / "rocm" / "llvm" / "bin" / exe, *slots, root / "bin" / exe)
                 if p.exists()), None)


def rocm_sdk() -> str | None:
    """TheRock's rocm-sdk (AMD's ROCm wheels for Windows): ROCM_VENV's, this .venv's, else the one on PATH."""
    venv = os.environ.get("ROCM_VENV")
    for c in (venv and str(Path(venv) / "Scripts" / "rocm-sdk.exe"), venv_tool("rocm-sdk"), shutil.which("rocm-sdk")):
        if c and Path(c).exists():
            return c
    return None


def rocm_devel_links(site: Path, repair: bool = True) -> tuple:
    """TheRock's device-library links (rocBLAS's / hipBLASLt's kernels for each GPU) in an expanded ROCm SDK, checked
    and repaired: `rocm-sdk init` hard-links each file of a rocm-sdk-device-<arch> wheel's .devel_links manifest into
    _rocm_sdk_devel, but tests its target as <dest folder>/../../.. unnormalized, and under a long install folder that
    path passes Windows' 260 characters: the link is skipped without a word and rocBLAS crashes on its first call
    (0xC0000005, #574). Here each target is normalized first, and an absent link is made from it (extended-length
    paths, so the long-path setting does not matter). -> (links made, [relpaths still absent]); (0, []) without an
    expanded SDK in `site`."""
    devel, made, absent = site / "_rocm_sdk_devel", 0, []
    if not devel.is_dir():
        return 0, []
    long = (lambda p: "\\\\?\\" + str(p)) if WIN else str
    for manifest in sorted((site / "_rocm_sdk_libraries" / ".devel_links").glob("*.json")):
        try:
            links = json.loads(manifest.read_text(encoding="utf-8")).get("links") or []
        except (OSError, ValueError):
            continue
        for link in links:
            rel, target = link.get("relpath"), link.get("target")
            if not rel or not target:
                continue
            dest = devel / rel
            if os.path.isfile(long(dest)):
                continue
            source = os.path.normpath(os.path.join(str(dest.parent), target))
            if repair and os.path.isfile(long(source)):
                try:
                    os.makedirs(long(dest.parent), exist_ok=True)
                    os.link(long(source), long(dest))
                    made += 1
                    continue
                except OSError:
                    pass
            absent.append(rel)
    return made, absent


def rocm_sdk_site(sdk: str) -> Path:
    """The site-packages of the environment a rocm-sdk(.exe) belongs to (<env>/Scripts/rocm-sdk.exe on Windows)."""
    env = Path(sdk).resolve().parent.parent
    return env / "Lib" / "site-packages" if WIN else next(iter(sorted(env.glob("lib/python*/site-packages"))), env)


def windows_rocm() -> tuple:
    """(ROCm's root, the folder of its DLLs) on Windows, or (None, None): ROCM_PATH, else TheRock's wheels, else AMD's
    HIP SDK for Windows (HIP_PATH, which its installer sets, else the newest in C:\\Program Files\\AMD\\ROCm)."""
    if os.environ.get("ROCM_PATH"):
        root = Path(os.environ["ROCM_PATH"]).resolve()
        return root, root / "bin"
    sdk = rocm_sdk()
    if sdk is not None:
        # the first `rocm-sdk init` unpacks the compiler, the headers and the libraries (about 3 GB, minutes) and links
        # the device wheels in; later ones only check. Before any `rocm-sdk path`: that would unpack it too, unseen and
        # within S.out's minute, and start over on every run (as llama.cpp's Windows CI does: init, then path)
        say("  AMD's ROCm SDK: rocm-sdk init (the first time it unpacks about 3 GB) ...")
        run([sdk, "init"], check=False)
        made, absent = rocm_devel_links(rocm_sdk_site(sdk))
        if made:
            say(f"  AMD's ROCm SDK: linked {made} GPU library file(s) rocm-sdk init had left out (a long folder path)")
        if absent:
            fail(f"AMD's ROCm SDK is missing {len(absent)} GPU library file(s) after rocm-sdk init (e.g. "
                 f"{absent[0]}): rocBLAS would crash on its first call",
                 f"install Maya in a shorter folder (e.g. C:\\Maya) and run {ME} there")
        path = lambda what: (S.out([sdk, "path", what]).strip().splitlines() or [""])[-1].strip()
        root = path("--root")
        if root:
            return Path(root), Path(path("--bin") or Path(root) / "bin")
    base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "AMD" / "ROCm"
    sdks = sorted(base.glob("[0-9]*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)], reverse=True)
    for root in ([Path(os.environ["HIP_PATH"])] if os.environ.get("HIP_PATH") else []) + sdks:
        if hip_clang(root) is not None:
            return root.resolve(), root.resolve() / "bin"
    return None, None


def rocm_devices_missing(archs) -> list:
    """Of these architectures, those whose library kernels (TheRock's rocm-sdk-device-<arch> wheel) are not in this
    .venv; [] for a ROCm SDK elsewhere (ROCM_PATH, ROCM_VENV, PATH: not checked)."""
    if os.environ.get("ROCM_PATH") or rocm_sdk() is None or rocm_sdk() != venv_tool("rocm-sdk"):
        return []
    from importlib import invalidate_caches, metadata
    invalidate_caches()
    missing = []
    for arch in dict.fromkeys(archs):
        try:
            metadata.version(f"rocm-sdk-device-{arch}")
        except metadata.PackageNotFoundError:
            missing.append(arch)
    return missing


def hip_info(bindir) -> str | None:
    """ROCm's hipInfo (on Windows: lists the GPUs as HIP numbers them, with their architecture)."""
    p = Path(bindir) / "hipInfo.exe" if bindir else None
    return str(p) if p and p.exists() else shutil.which("hipInfo")


def installed_hip_info() -> str | None:
    """hipInfo of a ROCm already installed (ROCM_PATH, HIP_PATH, the newest in C:\\Program Files\\AMD\\ROCm, PATH),
    only to name the GPUs - not windows_rocm(), whose rocm-sdk init can unpack gigabytes."""
    base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "AMD" / "ROCm"
    roots = [Path(os.environ[k]) for k in ("ROCM_PATH", "HIP_PATH") if os.environ.get(k)] + sorted(
        base.glob("[0-9]*"), key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)], reverse=True)
    return next((str(r / "bin" / "hipInfo.exe") for r in roots if (r / "bin" / "hipInfo.exe").exists()),
                shutil.which("hipInfo"))


def first_run_backend() -> str:
    """The backend of a first run without --backend (nothing set up yet): Metal on a Mac with Apple Silicon, HIP on a
    PC with an AMD GPU Maya's HIP build takes (check_hip_pc's list, the experimental ones too) and no NVIDIA one, else
    CUDA.  On Windows the AMD GPUs are named by an installed ROCm's hipInfo first: Windows' adapter table knows the
    common cards only, and an AMD-only PC whose card it did not know was sent to the CUDA checks, which stop at
    nvidia-smi (#805)."""
    if apple_silicon():
        return "metal"
    if S.gpus():
        return "cuda"
    amd = S.amd_gpus(installed_hip_info()) if WIN else S.amd_gpus()
    takes = HIP_ARCHS + HIP_ARCHS_UNVALIDATED + HIP_ARCHS_WAVE64
    return "hip" if any(g["arch"] in takes for g in amd) else "cuda"


THEROCK_INDEX = "https://repo.amd.com/rocm/whl-multi-arch/"   # AMD's ROCm for Windows: TheRock's pip wheels
THEROCK_VERSION = "7.14.1"                                     # 7.14: the first with Gorgon Halo (Ryzen AI Max 400)


def rocm_pip(archs) -> list:
    """pip's command that puts AMD's ROCm SDK for these GPUs into this Python (.venv): the compiler, the libraries and
    their kernels for each architecture."""
    devices = ",".join(f"device-{x}" for x in sorted(set(archs)))
    return [sys.executable, "-m", "pip", "install", "--disable-pip-version-check", "--index-url", THEROCK_INDEX,
            f"rocm[libraries,devel,{devices}]=={THEROCK_VERSION}"]


def install_rocm(a, chosen) -> None:
    """Windows: AMD's ROCm SDK with the kernels for these GPUs into Maya's .venv (asked first; nothing goes outside
    this folder), else the command."""
    archs = sorted({g["arch"] for g in chosen})
    cmd = rocm_pip(archs)
    what = f"AMD's ROCm SDK for Windows with the kernels for {', '.join(archs)}"
    if getattr(a, "check", False) or sys.prefix == sys.base_prefix:
        fail(f"{what} is not installed", f"install it into Maya's Python environment: {cmdline(cmd)}"
             f" - then run {ME} again (or set ROCM_PATH / ROCM_VENV to a ROCm 7 that has one)")
    if ask(f"Install {what} ({THEROCK_VERSION}) into .venv now (pip, from repo.amd.com)?",
           ["y", "n"], "y", getattr(a, "yes", False)) != "y":
        fail("Maya's HIP engine is compiled with AMD's ROCm SDK", f"install it with: {cmdline(cmd)}")
    run(cmd)
    if windows_rocm()[0] is None:
        fail("pip installed AMD's ROCm SDK, but rocm-sdk does not answer", f"run {cmdline(cmd)} again and check its "
             "messages")


VS2026_HIP_HINT = ("with Visual Studio 2026's Build Tools alone, ROCm 7.2's compiler can stop on its C++ library's math "
                   "headers ('__device__ function 'isgreater' cannot overload __host__ __device__ function', #544): "
                   "install the VS 2022 Build Tools beside it (winget install -e --id "
                   "Microsoft.VisualStudio.2022.BuildTools; the setup takes 2022 first), delete the build-hip folder "
                   "and run the setup again")


def vs_2026(vcvars) -> bool:
    """Whether a vcvars64.bat is Visual Studio 2026's or newer (its folder is named by its major version, 18; 2019 and
    2022 by their year)."""
    from pathlib import PureWindowsPath   # (a Windows path, read the same on any OS: the tests run on Linux too)
    return vcvars is not None and any(p.isdigit() and 18 <= int(p) < 1000 for p in PureWindowsPath(str(vcvars)).parts)


def check_hip_pc(a, installed=False, text_only=True) -> dict:
    """a.gpus_before: the GPUs this model's config ran on - a setup again without --gpu / --gpus keeps them."""
    if not WIN and (not sys.platform.startswith("linux") or S.is_wsl()):
        fail("Maya's HIP backend runs on native Linux or Windows")
    vcvars = S.find_vcvars(cuda=False) if WIN else None
    if WIN and vcvars is None:      # (before ROCm's download: ROCm's clang needs its headers)
        fail("Visual Studio Build Tools (2022 or 2026) are needed: ROCm's compiler uses their headers and libraries",
             "winget install -e --id Microsoft.VisualStudio.2022.BuildTools --override \"--wait --passive --add "
             "Microsoft.VisualStudio.Workload.VCTools --includeRecommended\" - or "
             "https://visualstudio.microsoft.com/visual-cpp-build-tools/ with 'Desktop development with C++'")
    if vs_2026(vcvars):
        warn("only Visual Studio 2026's Build Tools were found - " + VS2026_HIP_HINT)
    root, bindir = windows_rocm() if WIN else (None, None)
    hipinfo = hip_info(bindir) if WIN else None
    found = S.amd_gpus(hipinfo) if WIN else S.amd_gpus()
    usable = [g for g in found if g["arch"] in HIP_ARCHS + HIP_ARCHS_UNVALIDATED + HIP_ARCHS_WAVE64]
    for g in found:
        say(f"    {gpu_label(g)} - " + ("can be used" if g in usable else "not supported by Maya's HIP build"))
    if not usable:
        fail("no supported AMD GPU found", "this port targets RX 7900 XT / XTX (gfx1100), RX 9070 / AI PRO R9700 "
             "(gfx1201), and Strix Halo / Gorgon Halo, Radeon 8050S / 8060S / 8065S (gfx1151); experimental, not "
             "validated: RX 6800 / 6900, Radeon PRO V620 / W6800 (gfx1030)")
    if WIN and hipinfo and all(g.get("source") == "registry" for g in found):
        warn(f"ROCm's hipInfo sees no AMD GPU (the list above is Windows'): the engine cannot start until it does - "
             f"update AMD's driver (AMD Software: Adrenalin Edition, https://www.amd.com/en/support), restart, and "
             f"check with {hipinfo}")
    elif WIN and len(found) > 1 and any(g.get("source") == "registry" for g in found):
        warn("these GPU numbers are Windows' order (ROCm's hipInfo was not found): HIP's own can differ")
    kept = None if (a.gpus or a.gpu is not None) else gpus_before(getattr(a, "gpus_before", None), usable)
    if kept:                                           # (a setup again: the GPUs its config ran on, said)
        chosen = [next(g for g in usable if g["index"] == i) for i in kept]
        ok("GPUs: " + ", ".join(str(i) for i in kept) + " - as set up before (--gpu / --gpus picks others)")
    elif a.gpus:
        # two or more cards split the layers (each caches the experts of its own portion); the larger card goes first, as it
        # takes the bigger first portion - the order measured on an R9700 + RX 7900 XT
        try:
            want = [int(x) for x in str(a.gpus).split(",") if x.strip()]
        except ValueError:
            fail(f"--gpus takes GPU numbers as --check shows them, e.g. --gpus 0,1, not {a.gpus!r}")
        if not 2 <= len(want) <= MAX_GPUS or len(set(want)) != len(want):
            fail("Maya's HIP port runs on one GPU or splits the model across two or more: --gpu 0, or --gpus 0,1,2,3")
        picked = [next((g for g in usable if g["index"] == i), None) for i in want]
        for i, g in zip(want, picked):
            if g is None:
                fail(f"GPU {i} is not a supported AMD card")
        # A mixed pair puts the discrete GPU first and the unified-memory APU last.
        chosen = sorted(picked, key=lambda g: (hip_unified_memory(g), -g["vram_gb"]))
    else:
        one = next((g for g in usable if g["index"] == a.gpu), None) if a.gpu is not None else max(
            usable, key=lambda g: g["vram_gb"])
        if one is None:
            fail(f"GPU {a.gpu} is not a supported AMD card")
        chosen = [one]
    if WIN:
        # no ROCm SDK, or Maya's without the library kernels (rocm-sdk-device-<arch>) for a chosen GPU: offered once,
        # then checked again with ROCm there (its hipInfo numbers the GPUs as HIP does)
        missing = sorted({g["arch"] for g in chosen}) if root is None else rocm_devices_missing(g["arch"] for g in chosen)
        if missing and not installed and (root is None or not getattr(a, "check", False)):   # (--check: says so below)
            install_rocm(a, chosen)
            return check_hip_pc(a, installed=True)
        if root is None:
            fail("AMD's ROCm SDK for Windows was not found after its install", f"run {ME} again")
        if missing:
            warn(f"the ROCm SDK still has no library kernels for {', '.join(missing)} (rocm-sdk-device-...): hipBLAS "
                 f"fails on that GPU - {cmdline(rocm_pip(g['arch'] for g in chosen))}")
        if hip_clang(root) is None or not (list(root.glob("lib/hipblas*.lib")) or list(bindir.glob("hipblas*.dll"))):
            fail(f"ROCm's HIP compiler and hipBLAS are not in {root}",
                 f"install AMD's ROCm SDK into Maya's .venv: {cmdline(rocm_pip(g['arch'] for g in chosen))} - or set "
                 "ROCM_PATH to the root of a ROCm 7 for Windows")
        hipcc = next((p for p in (bindir / "hipcc.exe", root / "bin" / "hipcc.exe") if p.exists()), None)
        if hipcc is not None and tool_version(str(hipcc)) < (7, 0):
            fail("Maya's HIP port requires ROCm 7 or newer")
    else:
        root = Path(os.environ.get("ROCM_PATH") or "/opt/rocm")
        if not os.environ.get("ROCM_PATH") and not root.exists():
            # versioned installs without the /opt/rocm link (e.g. /opt/rocm-7.2.2): the newest one
            versions = sorted(Path("/opt").glob("rocm-[0-9]*"),
                              key=lambda p: [int(x) for x in re.findall(r"\d+", p.name)])
            if versions:
                root = versions[-1]
            elif (Path("/usr/bin/hipcc").exists() and hip_clang(Path("/usr")) is not None):
                root = Path("/usr")   # a distribution's own ROCm packages (Fedora's: /usr/lib64/rocm/llvm, lib64)
        root = root.resolve()
        clang = hip_clang(root)
        if (clang is None or clang.parent == root / "bin"
                or not any(list((root / d).glob("libhipblas.so*"))
                           for d in ("lib", "lib64", *(q.relative_to(root) for q in (root / "lib").glob("*-linux-gnu"))))):
            fail("ROCm's HIP compiler and hipBLAS are required", "install ROCm 7, or set ROCM_PATH to its root")
        if tool_version(str(root / "bin/hipcc")) < (7, 0):
            if os.environ.get("MAYA_HIP_ALLOW_ROCM6") != "1":
                fail("Maya's HIP port requires ROCm 7 or newer",
                     "or set MAYA_HIP_ALLOW_ROCM6=1 to try a ROCm 6.x (untested, e.g. a distribution's packages)")
            warn("ROCm older than 7 (MAYA_HIP_ALLOW_ROCM6=1): untested, the build or the engine may fail")
        if not shutil.which("c++"):
            fail("a C++ compiler is needed", "Ubuntu/Debian: sudo apt install build-essential")
    cpu, avx2, avx512 = S.cpu_info()
    if not avx2:
        # as on CUDA (#32): the CPU expert lane falls back to ggml's own kernels (src/kernels/cpu/expert_layout.cpp)
        warn(f"the CPU ({cpu}) has no AVX2: the engine runs, its CPU expert lane on ggml's own kernels (slower). "
             "MAYA_GLM_CPU_LANE=0 turns it off")
    total, avail = mem_gb()
    split = f"{len(chosen)} GPUs (layer split)" if len(chosen) > 1 else "one GPU"
    ok(f"using {' + '.join(gpu_label(g) for g in chosen)}; HIP, {split}" +
       (", text only" if text_only else ""))
    ok(f"ROCm: {root}; CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'})")
    ok(f"RAM: {total:.0f} GB, {avail:.0f} GB available now")
    if any(hip_unified_memory(g) for g in chosen) and WIN:
        # Windows gives the APU's GPU a fixed carve-out, which it does not count as RAM: the engine sizes the pool
        # from it as on a discrete card (src/core/glm_fast_path.cu, fast_setup)
        vram = sum(g["vram_gb"] for g in chosen)
        ok(f"APU: {vram:.0f} GB of the memory is the GPU's (Windows' share for it): the engine fills it with experts "
           "and keeps the next ones in RAM")
        if vram < total:
            warn(f"the GPU has {vram:.0f} GB and Windows {total:.0f} GB: set Variable Graphics Memory higher (AMD "
                 "Software: Performance > Tuning, or the iGPU memory size in the BIOS), e.g. 96 GB on a 128 GB PC, "
                 "and restart - the experts on the GPU are the fastest")
        # Windows charges every allocation on the APU's GPU to its commit (RAM + page file), though the carve-out is
        # not RAM it counts (a Radeon 8065S: the free commit fell by each one): the engine fills the GPU's share, the
        # RAM holds Windows and the other programs, so the page file needs the GPU's share and the margin both
        # engines keep free for the others, max(8 GB, 10% of the RAM) - with less, experts stay off the GPU or the
        # other programs run out of memory while the model runs (docs/AMD_MAYA.md, "The page file on a Windows APU")
        pf = S.page_file_gb()
        need_pf = math.ceil(vram + apu_margin_gb(total))
        if pf is not None and pf < need_pf:
            warn(f"Windows' page file is {pf:.0f} GB: set it to at least {need_pf} GB ({need_pf * 1024} MB, with that "
                 f"much free disk). Windows charges all of the GPU's memory ({vram:.0f} GB, as HIP reports it) to its "
                 "commit (RAM + page file), so with a smaller one fewer experts fit on the GPU (slower answers) or "
                 "other programs run out of memory while the model runs - "
                 "System > About > Advanced system settings > Performance > Settings > Advanced > Virtual memory > "
                 "Change: a custom size (initial and maximum), then restart Windows")
        elif pf is not None:
            ok(f"page file {pf:.0f} GB: Windows' commit holds the GPU's {vram:.0f} GB beside the other programs")
    elif any(hip_unified_memory(g) for g in chosen):
        ok("APU / unified memory: the engine sizes the GPU expert pool from available system RAM")
    unvalidated = sorted({g["arch"] for g in chosen if g["arch"] in HIP_ARCHS_UNVALIDATED})
    if unvalidated:
        warn(f"{', '.join(unvalidated)} (RDNA 2) is experimental: it builds and has the portable kernels, but Maya has "
             "not been validated on such a card yet - please report how it goes in an issue "
             "(https://github.com/mw00/project-maya/issues)")
    select_build_backend("hip")
    return {"backend": "hip", "gpus": chosen, "archs": sorted({g["arch"] for g in chosen}), "rocm": str(root),
            "amd_found": usable,                       # (the build's archs: every installed model's GPUs - hip_archs_in_use)
            **({"rocm_bin": str(bindir)} if WIN else {})}


def check_metal_pc(a) -> dict:
    """The Metal backend's PC check: macOS on Apple Silicon (its GPU always does Metal), the Xcode command line tools'
    C++ compiler, CMake, RAM and free disk.  Nothing asks nvidia-smi."""
    if not apple_silicon():
        fail("Maya's Metal backend runs on macOS on Apple Silicon (an M-series Mac)",
             "other PCs: --backend cuda (NVIDIA) or --backend hip (AMD)")
    if a.gpus or a.gpu is not None:
        warn("--gpu / --gpus do not apply on a Mac: the one Apple GPU is used")
    problems = []
    gpu = metal_gpus()[0]
    total, avail = mem_gb()
    ok(f"using {gpu['name']} ({total:.0f} GB unified memory, about {gpu['vram_gb']:.0f} GB of it for the GPU); "
       "experimental Metal: text only, one conversation at a time")
    if total < 100:
        warn(f"the model is about 95-100 GB (Maya-S) and this Mac has {total:.0f} GB of unified memory: the routed experts "
             "stream from the SSD while it answers, so decode speed is limited by the SSD (an internal NVMe drive)")
    ok(f"RAM: {total:.0f} GB, {avail:.0f} GB available now")
    if not S.out(["xcode-select", "-p"]).strip() or not S.out(["xcrun", "--find", "clang++"]).strip():
        problems.append(("no C++ compiler (the Xcode command line tools)", "xcode-select --install"))
    else:
        ok("C++ compiler: " + (S.out(["xcrun", "clang++", "--version"]).splitlines() or ["clang++"])[0])
    sc = pick_cmake() or shutil.which("cmake")
    scv = tool_version(sc) if sc else (0, 0)
    if scv >= (3, 24):
        ok(f"CMake {scv[0]}.{scv[1]}")
    else:
        ok((f"CMake {scv[0]}.{scv[1]} is older than 3.24" if sc else "no CMake on PATH") +
           ": a current one goes into .venv in step 3 (pip)")
    ok(f"Python {sys.version.split()[0]} ({sys.prefix})")
    free = shutil.disk_usage(ROOT).free / 1e9
    if free < 5:
        problems.append((f"only {free:.0f} GB free disk in {ROOT}: the engine's build needs a few GB", "free some space"))
    else:
        ok(f"free disk: {free:.0f} GB here")
    if problems:
        say()
        for what, how in problems:
            say(f"  [X] {what}")
            say(f"      {how}")
        fail("something Maya needs is missing (above)", f"install it and run {ME} again - the script installs nothing system-wide")
    select_build_backend("metal")
    return {"backend": "metal", "gpus": [gpu]}


def nvcc_range(archs) -> tuple:
    """The CUDA toolkit versions that can build for these GPUs: (lowest, first too new or None)."""
    lo = (12, 8) if max(archs) >= 100 else (12, 0)  # Blackwell needs 12.8
    hi = (13, 0) if min(archs) < 75 else None       # CUDA 13 no longer compiles for Volta (sm_70) or Pascal (sm_6x)
    return lo, hi


def find_nvcc(archs, given=None) -> tuple:
    """(nvcc, version) for these GPUs: --nvcc, else the newest toolkit that can build for them (Strata's find_nvcc
    takes the newest of all, but CUDA 13 cannot build for Volta), else the newest found (reported as unfit), else
    (None, None)."""
    if given:
        return given, S.nvcc_version(given)
    return pick_nvcc([(c, S.nvcc_version(c)) for c in dict.fromkeys(c for c in nvcc_candidates() if c and Path(c).exists())],
                     archs)


def nvcc_candidates() -> list:
    """Every place a CUDA toolkit's nvcc may be (some do not exist): PATH, CUDA_PATH, the default install folders."""
    exe = "nvcc.exe" if WIN else "nvcc"
    cands = [shutil.which("nvcc"), os.environ.get("CUDA_PATH") and str(Path(os.environ["CUDA_PATH"]) / "bin" / exe)]
    if WIN:
        base = Path(os.environ.get("ProgramFiles", r"C:\Program Files")) / "NVIDIA GPU Computing Toolkit" / "CUDA"
        return cands + [str(p / "bin" / exe) for p in sorted(base.glob("v*"))]
    cands += [str(p / "bin" / exe) for p in sorted(Path("/usr/local").glob("cuda*"))]
    return cands + [str(p / "bin" / exe) for p in sorted(Path("/opt").glob("cuda*"))] + ["/usr/bin/nvcc"]


def pick_nvcc(found, archs) -> tuple:
    """Of [(nvcc, version)], the newest that can build for `archs`, else the newest, else (None, None)."""
    found = sorted([f for f in found if f[1]], key=lambda f: f[1], reverse=True)
    lo, hi = nvcc_range(archs)
    fit = [f for f in found if f[1] >= lo and (hi is None or f[1] < hi)]
    return (fit or found or [(None, None)])[0]


def toolkit_hint(archs) -> str:
    if WIN:
        return ("NVIDIA's CUDA Toolkit 12.8 for Windows (the driver can stay as it is): "
                "https://developer.nvidia.com/cuda-12-8-0-download-archive" +
                (" - Volta (V100) and Pascal need 12.x, CUDA 13 dropped them" if min(archs) < 75 else ""))
    if min(archs) < 75:
        return ("Volta (V100) and Pascal need CUDA 12.x - Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit "
                "(CUDA 12.0), "
                "or NVIDIA's cuda-toolkit-12-8 package: https://developer.nvidia.com/cuda-12-8-0-download-archive")
    if max(archs) >= 100:
        return "NVIDIA's cuda-toolkit-12-8 (or newer) package: https://developer.nvidia.com/cuda-downloads"
    return ("Ubuntu 24.04: sudo apt-get install -y nvidia-cuda-toolkit (CUDA 12.0), or NVIDIA's packages: "
            "https://developer.nvidia.com/cuda-downloads")


def cuda_lib_dirs(nvcc: str) -> list:
    """The toolkit's library folders for the engine's LD_LIBRARY_PATH (none for a distribution's /usr/bin/nvcc); on
    Windows its DLL folders (bin, and bin\\x64 in CUDA 13) for the engine's PATH."""
    if WIN:
        b = Path(nvcc).parent
        return [str(p) for p in (b, b / "x64") if p.is_dir()]
    base = Path(nvcc).resolve().parent.parent
    if base == Path("/usr"):
        return []
    return [str(p) for p in (base / "lib64", base / "targets" / "x86_64-linux" / "lib") if p.is_dir()]


# ------------------------------------------------------------------------------------------------ 1. the PC
def gpus_before(before, usable: list) -> list | None:
    """The GPU numbers a config of this model ran on (its "gpu"), when every one of them is still here and usable -
    a setup again offers them first (#1066: it took all the GPUs again, adding one the user had left out)."""
    ids = before if isinstance(before, list) else [before] if isinstance(before, int) else []
    ids = [i for i in ids if isinstance(i, int)]
    byid = {g["index"]: g for g in usable}
    if not ids or len(set(ids)) != len(ids) or any(i not in byid for i in ids):
        return None
    return ids


def ik2_gpus_asked(a) -> int:
    """How many GPUs --gpu / --gpus name (0: none named) - Inkling-2-Mini's preview refuses more than one."""
    if getattr(a, "gpus", None):
        return len({x.strip() for x in str(a.gpus).split(",") if x.strip()})
    return 1 if getattr(a, "gpu", None) is not None else 0


def choose_gpus(a, found, min_cc: int = MIN_CC, family: str = "", before=None) -> list:
    """The GPUs the model runs on: --gpu / --gpus, else asked when several can share it (all of them recommended).
    `before`: the GPUs this model's config ran on - a setup again keeps them (the default answer; --gpus picks
    others)."""
    byid = {g["index"]: g for g in found}
    usable = [g for g in found if int(g["arch"]) >= min_cc]
    kept = None if (a.gpus or a.gpu is not None) else \
        gpus_before(getattr(a, "gpus_before", None) if before is None else before, usable)
    if a.gpus or a.gpu is not None:
        try:
            want = [int(x) for x in str(a.gpus).split(",") if x.strip()] if a.gpus else [a.gpu]
        except ValueError:
            fail(f"--gpus takes GPU numbers as nvidia-smi numbers them, e.g. --gpus 0,1, not {a.gpus!r}")
        if not 1 <= len(want) <= MAX_GPUS or len(set(want)) != len(want):
            fail(f"Maya runs on one GPU or splits the model's layers across up to {MAX_GPUS}: --gpu 0, or "
                 "--gpus 0,1,2,3 (each number once)")
        for i in want:
            if i not in byid or int(byid[i]["arch"]) < min_cc:
                fail(f"GPU {i} cannot be used" + ("" if i in byid else " (not found)"),
                     "use one of: " + ", ".join(str(g["index"]) for g in usable))
        return [byid[i] for i in want]
    if not usable:
        fail("none of the GPUs can run this model",
             f"it needs an NVIDIA GPU of compute capability {min_cc / 10:.1f} or newer (" +
             ("V100 and newer; Pascal - Tesla P40 / P100, GTX 10 - is experimental)" if min_cc < 80 else
              "RTX 30 / A100 and newer)"))
    if family == "ik2":   # (Inkling-2-Mini, a preview: one GPU - the one its config ran on, else the largest)
        pick = [byid[kept[0]]] if kept and len(kept) == 1 else \
            [max(usable, key=lambda g: (round(g["vram_gb"]), -g["index"]))]
        if len(usable) > 1:
            ok(f"GPU: {gpu_label(pick[0])} - {IK2_SHOWN} runs on one GPU for now (--gpu N picks another)")
        return pick
    best = sorted(usable, key=lambda g: (-round(g["vram_gb"]), g["index"]))
    # A card with under a quarter of the biggest one's VRAM (a 2 GB display card beside 32 GB ones) still gets whole
    # layers of the split and runs out of memory at the pack; the automatic pick leaves it out (--gpus still takes it).
    small = [g for g in best[1:] if g["vram_gb"] < SMALL_GPU_SHARE * best[0]["vram_gb"]]
    for g in small:
        if not kept or g["index"] not in kept:
            say(f"    {gpu_label(g)} is left out: too little VRAM beside the others for a part of the split "
                f"(--gpus takes it anyway)")
    best = [g for g in best if g not in small]
    kept_gs = [byid[i] for i in kept] if kept else None

    def label(gs):
        if len(gs) > 2:
            return (f"all {len(gs)}" if gs == best[:MAX_GPUS] else f"{len(gs)}") + " together: GPUs " + \
                ", ".join(str(g["index"]) for g in gs) + f" ({sum(g['vram_gb'] for g in gs):.0f} GB of VRAM)"
        if len(gs) == 2:
            return f"{gpu_label(gs[0])} + {gpu_label(gs[1])} together"
        return f"{gpu_label(gs[0])} only"
    if len(best) == 1:
        if kept_gs and kept_gs != best:                # (chosen before with --gpus: still the answer)
            ok("GPUs: " + label(kept_gs) + " - as set up before (--gpus picks others)")
            return kept_gs
        return best
    opts = ([best[:MAX_GPUS]] if len(best) > 2 else []) + [best[:2], best[:1]]
    rec, default = 0, 0                                # (the recommended answer, and the one Enter / --yes takes)
    if kept_gs:
        at = next((i for i, gs in enumerate(opts) if [g["index"] for g in gs] == kept), None)
        if at is None:                                 # (another set: offered first, the recommended one after it)
            opts.insert(0, kept_gs)
            at, rec = 0, 1
        default = at
    labels = [label(gs) + (" - as set up before" if kept_gs and i == default else "") for i, gs in enumerate(opts)]
    intro = (["MiniMax-M3.1-Flash runs best on all of them: the GPUs share the layers (or split them in groups where",
              "that measures faster), and the more VRAM, the fewer experts the CPU computes."]
             if family == "minimax" else
             [f"{HY4} runs best on all of them: the largest group of equal GPUs shares every layer (tensor parallel),",
              "the fastest GPU holds the head and the drafts, and the more VRAM, the fewer experts the CPU computes."]
             if family == "hy4" else
             ["DeepSeek-V4.1-Flash runs best on all of them: pairs of equal GPUs split each of their layers (tensor",
              "parallel), the pairs take the layers in turn, and the more VRAM, the fewer experts the CPU computes."]
             if min_cc >= DS4_MIN_CC else
             ["Maya can run on one GPU, or split the model's layers across several: each then caches the experts "
              "of its", "own layers, so together they hold more of them (V100s: ~50 tok/s benchmark on two, ~24 "
              "on one)."])
    pick = choose("Which GPUs?", intro,
                  [(lab, "recommended" if i == rec else None, None) for i, lab in enumerate(labels)], default,
                  a.yes or a.check,
                  ["(--gpus picks any others, e.g. --gpus 0,2,5; a model with a draft block drafts tokens on two GPUs "
                   "or more)"] if len(best) > 2 and S.UI is None else [])   # (the plain setup's; not on the screen)
    return opts[pick]


def oomd_active() -> bool:
    """Linux: systemd-oomd runs (stock Ubuntu 22.04 and newer; it ends a session's whole scope when its memory pressure
    stays over 50% for 20 s - a long load's page-cache reclaim can do that, #497)."""
    if WIN or not sys.platform.startswith("linux"):
        return False
    return S.out(["systemctl", "is-active", "systemd-oomd"]).strip() == "active"


def check_pc(a, family: str = "glm", gguf: bool = False, ds4_quant: str = "exl3") -> dict:
    """Step 1.  a.gpus_before: the GPUs the config of this model's setup before ran on (the default answer)."""
    step(1, "checking this PC")
    ds4 = family == "deepseek"
    mm3 = family == "minimax"

    if mm3 and WIN and not MM3_ALL_PLATFORMS:
        fail(f"{MM3}'s engine (maya-mm3) runs on Linux only for now (its Windows build is on the way)",
             f"set it up on a Linux PC with an NVIDIA GPU ({mm3_gpus()})")
    ik2 = family == "ik2"
    if ik2:   # (Inkling-2-Mini, an experimental preview: NVIDIA, Linux, one GPU - refused in one line before anything)
        why = IK2F.unsupported(a.backend, WIN, ik2_gpus_asked(a), [])
        if why:
            fail(why)
    hy4 = family == "hy4"
    if hy4 and a.backend == "hip" and WIN:
        # (maya-hy4's AMD build: built and run on Linux - a Gorgon Halo's Radeon 8065S, and RX 7900 XT cards by its
        #  contributor; under Windows it builds as the other engines' HIP builds do, with Windows' own waits and memory
        #  rules, but no one has run it there yet: experimental, said once, and the setup goes on)
        warn(f"{HY4} on an AMD GPU under Windows is experimental: Maya's team has not tested it there (it runs on "
             "Linux) - if it does not start or its answers look wrong, docs/AMD_MAYA.md says how to report it and "
             "what to use instead")
    if a.backend == "hip":
        return check_hip_pc(a, text_only=not (mm3 or hy4 or ik2))   # (Inkling 2: pictures on AMD too)
    if a.backend == "metal":
        if family != "glm":
            fail(f"Maya's setup runs GLM-5.3-Flash only on a Mac (not {ds4 and DS4 or mm3 and MM3 or hy4 and HY4 or QWEN_SHOWN})",
                 (f"{MM3} has an experimental Metal engine (maya-metal-mm3) that you build and start by hand: "
                  "see docs/METAL.md; " if mm3 else "")
                 + "other models: an NVIDIA or AMD GPU (--backend cuda / --backend hip)")
        return check_metal_pc(a)
    if ds4 and not gguf and (S.ARM or S.PPC):
        # the EXL3 quant's expert kernels are x86 (AVX2/AVX-512): neither this ARM nor this POWER CPU has them
        fail(f"{DS4}'s EXL3 quants need the x86 expert kernels (AVX2/AVX-512), which this {platform.machine()} CPU "
             "does not have",
             f"set up one of Maya's GGUF quants instead: {ME} --setup --model deepseek-maya-s (or deepseek-maya-xs)")
    if not (WIN or sys.platform.startswith("linux")):
        fail("Project Maya runs on Linux and Windows (and on Apple Silicon: --backend metal)",
             "the engine needs an NVIDIA GPU and CUDA")
    min_cc = ((DS4_GGUF_MIN_CC if gguf else DS4_MIN_CC) if ds4 else (MM3_GGUF_MIN_CC if gguf else mm3_min_cc()) if mm3
              else QWEN_MIN_CC if family == "qwen" else HY4_MIN_CC if hy4 else IK2_MIN_CC if ik2 else MIN_CC)
    if S.is_wsl():
        warn("this is WSL2, which Maya does not support: WSL2's driver pins only about 1 GB of "
             "RAM for the GPU, and Maya's RAM tier pins tens of GB. Use a native Linux install.")
    found = S.gpus()
    if not found:
        fail("no NVIDIA GPU found (nvidia-smi did not answer)",
             ("install the NVIDIA driver (https://www.nvidia.com/drivers)" if WIN else
              "install the NVIDIA driver (Ubuntu: sudo ubuntu-drivers install)") + f", restart, and run {ME} again")
    if ik2:                                            # (no card new or large enough: the one line, before the list)
        why = IK2F.unsupported("cuda", False, 0, found, mem_gb()[0])
        if why:
            fail(why)
    say("  NVIDIA GPUs:")
    for g in found:
        say(f"    {gpu_label(g)} - " + ("can be used" if int(g["arch"]) >= max(min_cc, VALIDATED_CC) else
                                       "can be used (Pascal: experimental)" if int(g["arch"]) >= min_cc else
                                       f"too old (this model needs compute capability {min_cc / 10:.1f} or newer)"))
    chosen = choose_gpus(a, found, min_cc, family)
    if ik2:                                            # (the card picked: 8 GB or more)
        why = IK2F.unsupported("cuda", False, len(chosen), chosen, mem_gb()[0])
        if why:
            fail(why)
    archs = sorted({int(g["arch"]) for g in chosen})
    vram = sum(g["vram_gb"] for g in chosen)
    ok("using " + " + ".join(gpu_label(g) for g in chosen) +
       ("" if len(chosen) < 2 else
        ": the equal GPUs among them share every layer (tensor parallel), else the layers are split across them"
        if family == "hy4" else f": the model's layers are split across the {len(chosen)}"))
    if ik2 and any(g.get("unified_memory") for g in chosen):
        warn(f"{IK2_SHOWN} has not been run yet on a GPU whose memory is the system's RAM: untested here")
    elif any(g.get("unified_memory") for g in chosen):
        say("  the GPU's memory is the system's RAM: the engine fills it with the most-used experts and leaves the "
            "rest to the RAM tier and the SSD, so the free RAM at the start is what the pool can take")
    if vram < 23 and not ds4 and not mm3 and family not in ("qwen", "hy4", "ik2"):
        # one small card: can any GLM-5.3-Flash quant start on it at all (glm_fit_verdict)?  Where none can, say so
        # here - the model's choice (step 2) then stops before any download
        try:
            small = {"backend": "cuda", "gpus": chosen}
            fits = [glm_fit_verdict(q, small, used=u) for u in [gpu_used_mib() if glm_fit_card(small, {}) else {}]
                    for q in GLM_FIT if q in MODELS]
        except Exception:  # noqa: BLE001 - the card does not read: the usual line
            fits = []
        slow = fits and all(v and v["verdict"] in ("no", "slow") for v in fits)
        if slow and any(v["verdict"] == "slow" for v in fits) and getattr(a, "glm_part", None) != "glm53":
            say(f"  {vram:.0f} GB of VRAM on one card is too little to hold GLM-5.3-Flash's experts: it starts, with "
                "the experts in RAM and computed on the CPU - much slower than on a card of 12 GB or more")
        elif slow and getattr(a, "glm_part", None) != "glm53":
            least = min(fits, key=lambda v: v["at"]["need"])["at"]
            warn(f"{vram:.0f} GB of VRAM on one card is too little for GLM-5.3-Flash: its smallest quant needs about "
                 f"{least['need'] / 1024:.1f} GB on the card before any expert is cached, and this card has "
                 f"{least['have'] / 1024:.1f} GB free now (step 2 says what is short for the quant you pick)")
        else:
            say(f"  {vram:.0f} GB of VRAM in total: the engine fills it with the most-used experts and serves the rest "
                "from RAM and the SSD - give it a try (more VRAM is faster; tell us your speed)")
    if min(archs) < VALIDATED_CC:
        warn("Pascal (compute capability 6.x) is experimental: the engine builds for it with its portable kernels "
             "(the prompt attention on the F32 cores) and caches global loads in L1 there, but Maya has not been "
             "validated on such a card yet - tell us how it goes in an issue")
    problems = []                                      # (what is wrong, how to fix it): all of them at once

    nvcc, nv = find_nvcc(archs, a.nvcc)
    lo, hi = nvcc_range(archs)
    want = f"CUDA {lo[0]}.{lo[1]} or newer" + (
        f" but older than {hi[0]}.0 (CUDA 13 dropped Volta and Pascal)" if hi else "")
    if nvcc is None:
        problems.append((f"the CUDA toolkit (nvcc) is not installed; these GPUs need {want}", toolkit_hint(archs)))
    elif nv is None:
        problems.append((f"{nvcc} does not say its version (nvcc --version)", "pass the toolkit's nvcc with --nvcc"))
    elif nv < lo or (hi is not None and nv >= hi):
        problems.append((f"nvcc {nv[0]}.{nv[1]} ({nvcc}) cannot build for these GPUs; they need {want}",
                         toolkit_hint(archs) + " (toolkits can be installed side by side: the newest that fits is "
                                                "used, or pass one with --nvcc)"))
    else:
        ok(f"CUDA toolkit {nv[0]}.{nv[1]}: {nvcc}")

    drv = min(S.driver_major(g) for g in chosen)
    need_drv = 580 if nv and nv >= (13, 0) else 525
    if drv < need_drv:
        problems.append((f"the NVIDIA driver {chosen[0]['driver']} is too old: {need_drv} or newer is needed",
                         ("" if WIN else "Ubuntu: sudo ubuntu-drivers install, then restart; or ") +
                         "https://www.nvidia.com/drivers"))
    else:
        ok(f"NVIDIA driver {chosen[0]['driver']}")

    if WIN:
        vcvars = S.find_vcvars(nvcc=nvcc)
        if vcvars is None:
            problems.append(("the C++ compiler (Visual Studio Build Tools, 'Desktop development with C++') is "
                             "not installed", "https://visualstudio.microsoft.com/visual-cpp-build-tools/ - in the "
                             "installer tick 'Desktop development with C++' (CUDA 12.x needs 2019 or 2022; "
                             + ("CUDA 12.x does not take 2026 - install 2022 alongside it, or use CUDA 13.2+"
                                if nv and nv < (13, 2) else "2026 works with CUDA 13.2 and newer") + ")"))
        else:
            ok(f"C++ compiler: Visual Studio Build Tools ({vcvars})")
    elif a.host_compiler:
        if shutil.which(a.host_compiler) is None:
            problems.append((f"--host-compiler {a.host_compiler} is not installed",
                             f"Ubuntu/Debian: sudo apt-get install -y {Path(a.host_compiler).name}"))
        else:
            ok(f"C++ compiler for nvcc: {a.host_compiler}")
    elif shutil.which("g++") is None:
        problems.append(("the C++ compiler (g++) is not installed", "Ubuntu/Debian: sudo apt-get install -y build-essential"))
    else:
        ok("C++ compiler: " + (S.out(["g++", "--version"]).splitlines() or ["g++"])[0])

    sc = pick_cmake() or shutil.which("cmake")
    scv = tool_version(sc) if sc else (0, 0)
    if scv >= (3, 24):
        ok(f"CMake {scv[0]}.{scv[1]}")
    else:
        ok((f"CMake {scv[0]}.{scv[1]} is older than 3.24" if sc else "no CMake on PATH") +
           ": a current one goes into .venv in step 3 (pip)")
    ok(f"Python {sys.version.split()[0]} ({sys.prefix})")

    total, avail = mem_gb()
    msg = f"RAM: {total:.0f} GB, {avail:.0f} GB available now"
    if ds4:   # (below DS4_MEMORY_GB the engine reads the experts RAM cannot hold from the SSD: --mmap-experts auto)
        if gguf:   # (a GGUF's: VRAM first, the rest read in place - RAM through the page cache, then the SSD;
            # --mmap-experts off copies them into RAM)
            ok(msg + f"; with {vram:.0f} GB of VRAM the most-used experts sit in VRAM, the others in RAM and on the "
                     "SSD (more RAM is faster)")
        elif vram + total < DS4_EXL3.get(ds4_quant, DS4_EXL3["exl3"])["memory_gb"]:
            q = DS4_EXL3.get(ds4_quant, DS4_EXL3["exl3"])
            warn(f"{msg}; with {vram:.0f} GB of VRAM that is below the ~{q['memory_gb']} GB that hold all of {DS4}'s "
                 f"{q['weights_gb']} GB of weights at {q['bits']} bits, so the experts that do not fit are read from the "
                 "model's files on the SSD, much slower (2.9 bits, one RTX 3090 and 64 GB measured: prompts ~50 "
                 "tokens/s, answers ~4-5 tokens/s). More GPUs (--gpus) or RAM make it faster")
        else:
            ok(msg + f"; with {vram:.0f} GB of VRAM that holds {DS4} at {DS4_EXL3.get(ds4_quant, DS4_EXL3['exl3'])['bits']} bits")
        if len(chosen) % 2 or len({round(g["vram_gb"]) for g in chosen}) > 1:
            ok("tensor parallel pairs equal GPUs: a GPU without an equal neighbour runs its layers alone")
    elif hy4 or ik2:   # (Hy4: its routed experts that are not in VRAM read in place; Inkling 2: all in VRAM)
        ok(msg)
    else:
        ok(msg + ("" if total >= 60 else " - it runs with 32 GB; more RAM keeps more experts close and is faster"))
    pf = S.page_file_gb()
    need_pf = int(vram + 8.999)   # the card's memory and the pinned RAM tier are both charged to RAM + page file
    if pf is not None and pf < need_pf:
        warn(f"Windows' page file is {pf:.0f} GB; set it to at least {need_pf} GB. Under Windows every allocation on "
             f"the graphics card ({vram:.0f} GB here) and the engine's pinned RAM tier are both charged to RAM + page "
             f"file, so with {pf:.0f} GB the RAM tier stays about {need_pf - pf:.0f} GB short of your free RAM and "
             "those experts are read from the SSD (issue #20: 2.4-4.1 -> 8.4-11 tokens/s on an RTX 5090). \"System "
             f"managed\" can stay small (8 GB on a 128 GB PC): set a custom size of {need_pf} GB or more - System > "
             "About > Advanced system settings > Performance > Advanced > Virtual memory - and restart")
    if oomd_active() and (ds4 or mm3 or total >= 128):
        warn("systemd-oomd is running: while a large RAM tier loads (the page cache reclaimed for minutes), it can "
             "end the terminal's whole session - Maya, the setup's own tuning - with nothing in dmesg (issue #497: "
             "~200 GB of DeepSeek experts on stock Ubuntu 22.04, 'systemd-oomd killed ... processes' in the journal). "
             "Stop it for the run (sudo systemctl stop systemd-oomd) or exempt your session: docs/SETUP.md, "
             "'Linux: systemd-oomd'")
    cpu, avx2, avx512 = S.cpu_info()
    if S.ARM:
        ok(f"CPU: {cpu} (ARM64, {os.cpu_count()} threads; the expert kernels use ggml-cpu's NEON paths)")
    elif not avx2:
        warn(f"the CPU ({cpu}) has no AVX2: the engine runs, its CPU expert lane on ggml's own kernels (slower; "
             "on 2x Xeon E5-2660 v2 it still added ~15% to 4-GPU decode). MAYA_GLM_CPU_LANE=0 turns it off")
    else:
        ok(f"CPU: {cpu} ({'AVX-512' if avx512 else 'AVX2'}, {os.cpu_count()} threads)")

    if problems:
        say()
        for what, how in problems:
            say(f"  [X]  {what}")
            say(f"       {how}")
        fail("something Maya needs is missing (above)",
             f"install it and run {ME} again - this script installs nothing system-wide")
    select_build_backend("cuda")
    return {"backend": "cuda", "gpus": chosen, "archs": archs, "nvcc": nvcc,
            "vcvars": str(vcvars) if WIN else None}


# ------------------------------------------------------------------------------------------------ 2. choices
def context_label(n: int) -> str:
    """A context size as the menus show it: 32768 -> "32K", 1048576 -> "1M"."""
    return f"{n // 1048576}M" if n % 1048576 == 0 else f"{n // 1024}K"


def choose_context(a, prev_ctx, contexts=CONTEXTS, recommended=DEFAULT_CONTEXT) -> int:
    if a.context:
        return a.context
    default = prev_ctx if prev_ctx in contexts else recommended
    pick = choose("Context?", ["Context length = how much text the model sees at once (the chat, files, tool output). "
                               "A longer one takes", "VRAM from the expert cache, so answers get a little slower:"],
                  [(f"{context_label(c)} tokens", "recommended" if c == recommended else None, None) for c in contexts],
                  contexts.index(default), a.yes)
    return contexts[pick]


def recommended_context(choice, pc) -> int:
    """The context the setup recommends for the model chosen: its own (GLM Maya-XS: 128K) on a PC with the RAM and the
    card it names (64 GB + 24 GB), else the usual one."""
    m = MODELS.get(choice[1], {}) if choice and choice[0] == "download" else {}
    if not m.get("context") or pc.get("backend") == "metal":
        return DEFAULT_CONTEXT
    try:
        vram = max((g.get("vram_gb") or 0 for g in pc.get("gpus") or []), default=0)
        fits = mem_gb()[0] >= m.get("min_ram_gb", 0) * 0.95 and vram >= m.get("min_vram_gb", 0) * 0.95
    except Exception:  # noqa: BLE001 - no reading of the PC: the usual context
        fits = False
    return m["context"] if fits and m["context"] in CONTEXTS else DEFAULT_CONTEXT


def shard_names(m: dict, pattern: str | None = None) -> list:
    """A download's files under their names on Hugging Face: a GGUF's shards, or an EXL3 model's every file."""
    if "files" in m and pattern is None:
        return list(m["files"])
    return [(pattern or m["file"]).format(i=i, n=m["shards"]) for i in range(1, m["shards"] + 1)]


def is_exl3(quant: str) -> bool:
    return MODELS.get(quant, {}).get("format") == "exl3"


def old_names(m: dict) -> list:
    """The names a model's files had on Hugging Face before ("was", newest first), each as its list of shards."""
    return [shard_names(m, p) for p in m.get("was") or []]


def local_shards(m: dict, d: Path) -> list:
    """The model's files in d: under an earlier name when any of its files is there (a download started or finished
    under it - its run config points at them), else under the name on Hugging Face."""
    for names in old_names(m):
        if any((d / n).exists() for n in names):
            return [d / n for n in names]
    return [d / n for n in shard_names(m)]


def hf_name(m: dict, local: Path) -> str:
    """A local file's name on Hugging Face (a file under an earlier name downloads from, and checks against, it)."""
    for names in old_names(m):
        if local.name in names:
            return shard_names(m)[names.index(local.name)]
    return local.name


def download_dir(models: Path, quant: str) -> Path:
    """A download's folder: <models folder>/<quant> (--models-dir).  Files already there are used where they are:
    in that folder, in <models folder>/glm-5.3-flash-<quant> (an earlier version's downloads), or straight in the
    models folder - under their names on Hugging Face or an earlier one."""
    m = glm_downloads()[quant]
    d = models / quant
    for c in (d, models / f"glm-5.3-flash-{quant}".lower(), models):
        for names in [shard_names(m)] + old_names(m):
            if all((c / n).exists() for n in names):
                return c
    return d


def choose_draft(a, pc: dict, inst: dict) -> str:
    """"mtp" (the model's own MTP block; the default) or "dflash2" (a DFlash2 drafter, downloaded in step 5): --draft,
    else asked - the earlier setup's answer is the default.  AMD (HIP) setups keep MTP, except one Ryzen AI Max APU
    (hip_unified_memory: Strix / Gorgon Halo), where DFlash2 is measured (a Radeon 8060S with every expert in its
    memory: glm_dflash_test passes; --bench decode +9% against the MTP block, the speed calibration +5%).  So do NVIDIA
    GPUs before sm_80 (RTX 30 / A100): on Tesla V100s DFlash2 decoded slower than the MTP block (one GPU and two).  A
    DFlash2 answer is then checked on this PC by the speed calibration (speed_calibration): it stays only where it is
    faster.  On Metal both are exact and neither is measured: MTP is the default, the choice is yours."""
    if getattr(a, "draft", None):
        return a.draft
    gpus = pc.get("gpus") or []
    one_apu = pc.get("backend") == "hip" and len(gpus) == 1 and hip_unified_memory(gpus[0])
    if pc.get("backend") == "hip" and not one_apu:
        return "mtp"
    if pc.get("backend") != "hip" and pc.get("archs") and min(int(x) for x in pc["archs"]) < 80:
        return "mtp"
    default = inst.get("draft") if inst.get("draft") in DRAFTS else "mtp"
    metal = pc.get("backend") == "metal"
    mtp_where = ("" if metal else
                 " It drafts on this one GPU (an APU that holds the model in its memory)." if one_apu else
                 " On two GPUs or more; one GPU decodes without drafting.")
    rows = [("the model's MTP block", "default",
             "The model's own draft block guesses the next token, which the model then checks." + mtp_where),
            (f"a DFlash2 drafter (download {DFLASH['bytes'] / 1e9:.1f} GB from Hugging Face)", None,
             "A small drafter made for GLM-5.3-Flash drafts up to 7 tokens a round, which the model then checks"
             + (", so every token is the model's own pick. Not measured on a Mac: which one is faster is yours to try. " if metal else
             ": every token is the model's own pick, faster. Maya-L: +44% on nine GPUs and +12% on two against the MTP block, +19% on one RTX 3090. A speed "
             "calibration then measures it against the MTP block on this PC (5-20 minutes) and keeps it only if it "
             "is faster here. ") + f"The drafter's license is {DFLASH['license']}: non-commercial use only.")]
    pick = choose("Drafting?", ["How the model drafts tokens ahead (the answers are the same either way):"], rows,
                  DRAFTS.index(default), a.yes)
    return DRAFTS[pick]


def config_family(cfg: dict) -> str:
    """"minimax" for a MiniMax-M3.1-Flash config (maya-mm3), "deepseek" for a DeepSeek-V4.1-Flash config (maya-ds4's
    --pack), else "glm" (maya_quants.family_of: the server reads configs by it too)."""
    return Q.family_of(cfg)


# each model's name on the setup's screen, under its logo (Qwen's: the model, whichever of its files - never its id)
FAMILY_TITLE = {**Q.MODEL, "qwen": "Qwen-3.8-Flash-Next"}


def setup_title(cfg: dict) -> str:
    """The setup screen's header name for a config: its family's, except a glm-dsa config (family "glm"), which
    is GLM-5.3's own name, not GLM-5.3-Flash."""
    if G53.config_is_glm_dsa(cfg):
        return G53.SHOWN
    return FAMILY_TITLE.get(config_family(cfg), "")


# the models whose engine answers several requests at once (--parallel): GLM-5.3-Flash's, DeepSeek-V4.1-Flash's
# (maya-ds4 --parallel: a conversation's state each, their tokens decoded together), MiniMax-M3.1-Flash's (maya-mm3
# --parallel: a KV cache each in VRAM - no KV streaming, no DSpark)
PARALLEL_FAMILIES = ("glm", "deepseek", "minimax")


def check_parallel(a, family: str) -> None:
    if getattr(a, "parallel", None) is not None and family not in PARALLEL_FAMILIES:
        model = DS4 if family == "deepseek" else MM3 if family == "minimax" else HY4 if family == "hy4" else \
            IK2_SHOWN if family == "ik2" else QWEN_SHOWN
        fail(f"--parallel is supported by GLM-5.3-Flash, DeepSeek-V4.1-Flash and MiniMax-M3.1-Flash; {model} serves "
             "one request at a time",
             "remove --parallel; saved conversations do not enable simultaneous decoding")


def g53_offered() -> list:
    """GLM-5.3 (744B)'s downloads the setup offers (the ones GLM's download list holds)."""
    return [q for q in G53_MODELS if q in glm_downloads()]


def choose_family(a, prev: dict) -> str:
    """"glm" or "deepseek": what the options say (--model, --gguf-dir, --draft), else asked - the model set up before
    is the default."""
    if a.model:
        return "deepseek" if a.model == DS4 else "minimax" if a.model == MM3 else "qwen" if a.model == QWEN else \
            "hy4" if a.model == HY4 else "ik2" if a.model == IK2 else "glm"
    if getattr(a, "qwen_quant", None):                 # (--qwen-quant alone: Qwen's files named)
        return "qwen"
    if getattr(a, "hy4_quant", None):                  # (--hy4-quant alone: Hy4-Flash's)
        return "hy4"
    metal = getattr(a, "backend", None) == "metal"
    if a.gguf_dir or a.draft or (metal and not a.model_dir):     # (Metal runs GLM only: no question)
        return "glm"
    if a.model_dir:                                    # (an EXL3 folder of GLM or MiniMax; else DeepSeek's files)
        return exl3_folder_family(a.model_dir) or "deepseek"
    default = config_family(prev) if prev else "glm"
    try:
        have = sum(g["vram_gb"] for g in S.gpus() if g.get("vram_gb")) + mem_gb()[0]
    except Exception:  # noqa: BLE001 - no NVIDIA tools: no figure for this PC
        have = None
    here = f" (this PC: {have:.0f} GB)" if have else ""
    rows = [("GLM-5.3-Flash", None,
             "Z.ai's GLM-5.3-Flash in Project Maya's GGUF quants (GLM-5.3-Flash-Maya-S to -Maya-L, 95-156 GB) or "
             "turboderp's EXL3 quants: from one GPU up, the experts that do not fit in VRAM from RAM and the SSD; "
             "NVIDIA and AMD; pictures with its vision encoder."),
            (G53.LABEL, None,
             f"Z.ai's full {G53.SHOWN} (744B) on the engine GLM-5.3-Flash runs on"
             + (": " + ", ".join(f"{Q.glm53_quant(q)} ({G53_MODELS[q]['download_gb']:.0f} GB)" for q in g53_offered())
                if g53_offered() else "") +
             "; or its GGUF files you already have (--gguf-dir); NVIDIA and AMD, the experts that do not fit in VRAM "
             "from RAM and the SSD; text only."),
            (DS4, None,
             "DeepSeek's DeepSeek-V4.1-Flash-1017 (748B) on its own engine: unsloth's GGUF (UD-Q8_K_XL, 508 GB; "
             "NVIDIA from V100 on or AMD, Linux or Windows), orcarouter's abliterated one (Mix-128G or "
             "Mix-256G, its refusals removed), turboderp's EXL3 quant at 2.25 bits or LibertAIDAI's at 2.9 or "
             "3.5 bits with DeepSeek's two Engram tables (362 / 414 / 461 GB; NVIDIA V100 or newer, DSpark drafting, "
             "pictures from RTX 30 / A100 on); "
             f"fast with about {DS4_MEMORY_GB} GB of VRAM and RAM together{here}, with less the experts that do not "
             "fit are read from the SSD; tensor parallel across pairs of equal GPUs, a 256K context."),
            (MM3, None,
             "MiniMax's MiniMax-M3.1-Flash (428B, 23B active per token) on its own engine: Mia-AiLab's EXL3 quant at "
             f"3.5 bits (197 GB); NVIDIA {mm3_gpus()}{' on Linux and Windows (Windows: new)' if MM3_ALL_PLATFORMS else ' on Linux'}; about "
             f"{MM3_MEMORY_GB} GB of VRAM and RAM together{here}; the layers split across the GPUs, the experts that "
             "do not fit computed on the CPU; tool calls, MiniMax's thinking levels, pictures, DSpark drafting and "
             "up to a 1M context" + ("; or Maya's own GGUF quants (MiniMax-M3.1-Flash-Maya-XS ~93 GB, -Maya-S ~122 "
                                     "GB: NVIDIA from V100 on and AMD)"
                                     if any(map(mm3_gguf_published, ("gguf-xs", "gguf-s"))) else "") + "."),
            ("Qwen-3.8-Flash-Next", None,
             "Qwen-3.8-Flash-Next: 180B, 10 of 512 experts a token; Blackfrost-AI's cybersecurity fine-tune of Qwen3.8-Flash-Next "
             "in peasantsmith's GGUF, on Maya's own Qwen path; NVIDIA from V100 on or AMD, Linux or Windows; all its "
             f"experts in about {qwen_entry()['memory_gb']:.0f} GB of VRAM and RAM together{here}, with less the others "
             "read from the SSD; tool calls, its three reasoning efforts, MTP drafting, up to a 256K context; text only. "
             f"Or the base model itself in unsloth's UD-IQ2_XS GGUF ({qwen_entry(QWEN_UD)['download_gb']:.0f} GB).")]
    keys = ("glm", "glm53") + FAMILIES[1:]             # (the rows: GLM-5.3 (744B) is its own row of the "glm" family)
    # Hy4-Flash, the last row: Maya's own sizes that are uploaded (hy4_published), then Tencent's own Q4_K_M;
    # --model / --hy4-quant name it either way (AMD under Windows: experimental - check_pc says so)
    hy4s = [k for k in HY4_QUANTS if hy4_published(k)]
    q4 = hy4_entry() if hy4_published(HY4_Q4) else None
    if default == "hy4" or hy4s or q4:
        rows.append((HY4, None,
                     "Tencent's Hy4-flash (220B with its n-gram tables; 512 experts, 16 a token) on its own engine"
                     + (", in Maya's own GGUF quants: " +
                        ", ".join(f"{HY4_QUANTS[k]} ({hy4_entry(k)['download_gb']:.0f} GB)" for k in hy4s)
                        if hy4s else "") +
                     "; NVIDIA from V100 on, Linux or Windows, or AMD on Linux (Strix Halo / Gorgon Halo, Radeon "
                     "cards) - under Windows experimental; what is not a routed expert in VRAM, the experts that do "
                     "not fit computed on the CPU, the n-gram table read from the SSD; tool calls, thinking High or "
                     "Off; pictures (its vision tower, where the VRAM has room for one beside the model)." +
                     (f" Or Tencent's own Q4_K_M (AngelSlim; {q4['download_gb']:.0f} GB, about {q4['memory_gb']:.0f} GB "
                      f"of VRAM and RAM together{here}): with MTP drafting, tensor parallel across equal GPUs (the "
                      "fastest one holds the head and the drafts), up to a 256K context - run on NVIDIA RTX 30 / A100 "
                      "and newer, Linux." if q4 else "")))
    else:
        keys = tuple(k for k in keys if k != "hy4")
    # Inkling-2-Mini, after it: an experimental preview (PR #1095) - Linux, one NVIDIA GPU of 8 GB or more or an AMD one of 24 GB
    rows.append((f"{IK2_SHOWN} (experimental)", None, ik2_about()))
    was53 = default == "glm" and bool(prev) and G53.config_is_glm_dsa(prev)
    pick = keys[choose("Model?", ["Which model to set up:"], rows, keys.index("glm53" if was53 else default), a.yes)]
    if pick in ("glm", "glm53"):
        a.glm_part = "glm53" if pick == "glm53" else "flash"   # (choose_model lists that model's files only)
        return "glm"
    return pick


# The quant the setup's lists recommend (fit_recommendation): RAM the engines leave the OS (MAYA_GLM_RAM_HEADROOM_GB's
# default), and the least share of a quant's memory (with its cache) that should sit in GPU memory - below it the
# decode reads most experts from RAM every token, and a smaller quant of the same model answers faster
FIT_RAM_HEADROOM_GB = 6.0
FIT_MIN_VRAM_SHARE = 0.2


def fit_memory(gpus: list, ram: float) -> tuple:
    """(the memory a model's files and caches may take without reading experts from the SSD, of it GPU memory) on a
    PC with these GPUs (their "vram_gb") and this RAM (GiB, mem_gb's total): VRAM + RAM less FIT_RAM_HEADROOM_GB.  A
    unified-memory APU (apu_kind: Strix / Gorgon Halo) counts its one memory once - on Linux the GPU's memory is drawn
    from the RAM (its VRAM figure, a carve-out or the RAM it maps, is part of it: the larger of the two), on Windows
    the carve-out sits beside the RAM Windows counts (the sum) - and all of it as GPU memory (the GPU reads the RAM at
    the same speed); a Mac's GPU (metal_gpus: a share of its RAM) as Linux's APU."""
    vram = sum(float(g.get("vram_gb") or 0) for g in gpus or [])
    apu = "linux" if any(g.get("vendor") == "apple" for g in gpus or []) else apu_kind(gpus)   # (a Mac: as Linux's)
    pool = (max(vram, ram) if apu == "linux" else vram + ram) - FIT_RAM_HEADROOM_GB
    return max(pool, 0.0), (max(pool, 0.0) if apu else vram)


def fit_recommendation(sizes: dict, gpus: list, ram: float, kv_gb: float = 0.0, top: str | None = None) -> str:
    """The quant to recommend among `sizes` ({quant: the VRAM + RAM it takes, GB, the cache aside}, quants of one model)
    on this PC (fit_memory): no larger than `top` (the family's usual pick; None: any) - the largest whose memory with
    the default context's cache (kv_gb) fits without reading experts from the SSD and has at least FIT_MIN_VRAM_SHARE
    of it in GPU memory (no VRAM figure: the largest that fits), else the smallest that fits; none fits: the smallest
    of all (the SSD streams the rest).  A PC whose RAM does not read: `top` (or the largest)."""
    pool, gpu = fit_memory(gpus, ram)
    ladder = sorted((k for k in sizes if top is None or sizes[k] <= sizes[top]), key=lambda k: -sizes[k])
    if ram <= 0 or pool <= 0:
        return top or ladder[0]
    fit = [k for k in ladder if sizes[k] + kv_gb <= pool]
    if not fit:
        return min(sizes, key=sizes.get)
    if gpu <= 0:                                   # (no VRAM figure: by the memory alone)
        return fit[0]
    return next((k for k in fit if gpu >= FIT_MIN_VRAM_SHARE * (sizes[k] + kv_gb)), fit[-1])


def setup_gpus(a) -> list:
    """The GPUs a setup's lists size their recommendation by: AMD's on --backend hip, else NVIDIA's; none when the
    tools do not read."""
    try:
        return list((S.amd_gpus() if getattr(a, "backend", None) == "hip" else S.gpus()) or [])
    except Exception:  # noqa: BLE001 - no GPU tools
        return []


# Can a GLM-5.3-Flash GGUF quant start on this PC's one NVIDIA card?  Worked out before the download (glm_fit): the
# engine refuses to start when, after everything that must sit on the card, the expert pool has no room for a route's
# experts in every layer ("no room for the expert pool": src/core/glm_fast_path.cu) - an 8 GB card that also drives
# the desktop found that out after 73 GB (2026-11-18).  Per quant, from its GGUF header on Hugging Face (the tensor
# table of every shard, peasantsmith/GLM-5.3-Flash-Maya-GGUF at MODELS' revision; block 45 is the MTP block):
#   "dense_mib": the weights the card always holds.  Maya-XS: the engine's own start line on an RTX 3050 ("dense weights
#     5.24" GiB); Maya-S24 has the same non-expert tensors, byte for byte (5759 MiB), so the same figure.  The others
#     ("computed"): their non-expert tensors less the token embedding (7445 - 496 MiB), times the 1.02 by which the
#     same sum falls short of the engine's figure on Maya-XS (5263 against 5366 MiB)
#   "slot_mib": one routed expert (gate + up + down) of each of the 45 layers, the pool's unit - Maya-XS 217.9 MiB by
#     the header, 218.3 by the engine's message (1,747 MiB for 8)
GLM_FIT = {
    "Maya-XS-IQ1_S": {"dense_mib": 5366, "slot_mib": 217.9},
    "Maya-S24": {"dense_mib": 5366, "slot_mib": 284.9},
    "Maya-S-v2-IQ2_XXS": {"dense_mib": 7088, "slot_mib": 284.9, "computed": True},
    "Maya-M": {"dense_mib": 7088, "slot_mib": 346.9, "computed": True},
    "Maya-M-Derisked": {"dense_mib": 7088, "slot_mib": 346.9, "computed": True},   # (Maya-M's tensors, edited)
    "Maya-L": {"dense_mib": 7088, "slot_mib": 477.2, "computed": True},
}
GLM_FIT_EXPERTS = 8                                # a route's experts (the header's expert_used_count): the pool's least
# the context's state on the card, from the engine's start line on that card with the INT8 cache (0.28 GiB at 8K,
# 0.45 at 32K): a fixed part and a part a position - with FP16 latents docs/GLM5-FLASH.md's 11.3 KB a position.  KV
# streaming keeps KV_RESIDENT positions on the card, so a longer context is counted as that one (never more)
GLM_FIT_STATE_MIB = 229
GLM_FIT_KV_BYTES = {"int8": 7427, "fp16": 11571}
GLM_FIT_KV_RESIDENT = 32768
# the engine's working memory on the card - the CUDA context, the MTP block, scratch, the prompt path, activations:
# that start line's "other" and "activations" (1.13 GiB) less the ~0.5 GiB the Windows desktop held (620-700 MiB over
# three starts; drafting off and --prefill 0 did not lower it)
GLM_FIT_WORK_MIB = 650
GLM_FIT_RESERVE_MIB = 300                          # the least the engine's automatic reserve goes to (pool_reserve)
# what the estimate can be off by: within it the setup warns and goes on (more for a quant whose dense figure is
# computed, not read from an engine start)
GLM_FIT_MARGIN_MIB = 300
GLM_FIT_MARGIN_COMPUTED_MIB = 500
# VRAM in use on the card now that counts against a model: the desktop's.  More than this is taken for a program
# that can be closed (another engine: busy_gpus' 1 GiB) - said, never a reason to refuse
GLM_FIT_USED_CAP_MIB = 1024


def gpu_used_mib() -> dict:
    """{GPU index: MiB of VRAM in use now} by nvidia-smi (the desktop and every program on the card); {} when it does
    not answer."""
    try:
        txt = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    used = {}
    for line in txt.splitlines():
        i, _, m = line.partition(",")
        try:
            used[int(i)] = int(float(m))
        except ValueError:
            pass
    return used


# (run by cuda_room_bytes in a process of its own - the setup itself never holds a CUDA context: each NVIDIA device's
#  free VRAM as CUDA's driver says it with one context held - the engine's own reading at its start, cudaMemGetInfo -
#  then with a second context beside it, and its total; {PCI bus id: [free, free with two, total]} in bytes)
CUDA_ROOM_PROBE = r"""
import ctypes, json, sys
cu = ctypes.WinDLL("nvcuda.dll") if sys.platform == "win32" else ctypes.CDLL("libcuda.so.1")
out = {}
n = ctypes.c_int(0)
if cu.cuInit(0) == 0 and cu.cuDeviceGetCount(ctypes.byref(n)) == 0:
    for i in range(n.value):
        dev, bus, p, c = ctypes.c_int(0), ctypes.create_string_buffer(32), ctypes.c_void_p(), ctypes.c_void_p()
        if cu.cuDeviceGet(ctypes.byref(dev), i) or cu.cuDeviceGetPCIBusId(bus, 32, dev) or \
                cu.cuDevicePrimaryCtxRetain(ctypes.byref(p), dev):
            continue
        one, two, tot = ctypes.c_size_t(0), ctypes.c_size_t(0), ctypes.c_size_t(0)
        ok = cu.cuCtxSetCurrent(p) == 0 and cu.cuMemGetInfo_v2(ctypes.byref(one), ctypes.byref(tot)) == 0
        if ok and cu.cuCtxCreate_v2(ctypes.byref(c), 0, dev) == 0:
            ok = cu.cuMemGetInfo_v2(ctypes.byref(two), ctypes.byref(tot)) == 0
            cu.cuCtxDestroy_v2(c)
        else:
            ok = False
        cu.cuCtxSetCurrent(None)
        cu.cuDevicePrimaryCtxRelease_v2(dev)
        if ok:
            out[bus.value.decode("ascii", "replace")] = [one.value, two.value, tot.value]
print(json.dumps(out))
"""


def cuda_room_bytes() -> dict:
    """{PCI id (setup.py's _pci_key form): (free, free with a second context, total) in bytes} - what CUDA's driver
    says of each NVIDIA device's VRAM with a context held (CUDA_ROOM_PROBE, in a process of its own); {} where it
    cannot be asked (no driver, a failing call, a timeout)."""
    try:
        txt = subprocess.run([sys.executable, "-I", "-c", CUDA_ROOM_PROBE], capture_output=True, text=True,
                             timeout=60).stdout
        got = json.loads(txt.strip().splitlines()[-1]) if txt.strip() else {}
        return {S._pci_key(k): tuple(int(x) for x in v) for k, v in got.items() if len(v) == 3}
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, IndexError):
        return {}


def glm_fit_state_mib(ctx: int, kv: str | None = None) -> float:
    """The VRAM a context's state takes on the leading card, MiB (GLM_FIT_STATE_MIB and GLM_FIT_KV_BYTES)."""
    per = GLM_FIT_KV_BYTES["int8" if kv == "int8" else "fp16"]
    return GLM_FIT_STATE_MIB + min(int(ctx), GLM_FIT_KV_RESIDENT) * per / 2**20


def glm_fit(quant: str, total_mib: float, used_mib: float, ctx: int = DEFAULT_CONTEXT,
            reserve_mib: int = GLM_FIT_RESERVE_MIB) -> dict | None:
    """What GLM-5.3-Flash quant `quant` needs on one card of `total_mib` with `used_mib` already in use, at a context
    of `ctx`, before it can start - each term in MiB: "dense", "state", "work" (GLM_FIT_WORK_MIB), "pool" (a route's
    experts in every layer: GLM_FIT_EXPERTS slots), "reserve" (the VRAM kept free; the engine's automatic reserve goes
    down to 300 MiB, its default is 700), their sum "need", the card's "have" (total less in use) and "short" (need
    less have: above 0 the engine has no room for its expert pool).  None: a quant with no figures (GLM_FIT)."""
    f = GLM_FIT.get(quant)
    if not f:
        return None
    t = {"dense": float(f["dense_mib"]), "state": glm_fit_state_mib(ctx, MODELS.get(quant, {}).get("kv")),
         "work": float(GLM_FIT_WORK_MIB), "pool": GLM_FIT_EXPERTS * f["slot_mib"], "reserve": float(reserve_mib)}
    t["need"] = sum(t.values())
    t["have"] = float(total_mib) - float(used_mib)
    t["short"] = t["need"] - t["have"]
    t.update(total=float(total_mib), used=float(used_mib), ctx=int(ctx),
             margin=GLM_FIT_MARGIN_COMPUTED_MIB if f.get("computed") else GLM_FIT_MARGIN_MIB)
    return t


def glm_fit_card(pc: dict | None, used: dict | None = None) -> tuple | None:
    """(the card, its VRAM in MiB, the MiB in use on it now) where glm_fit's figures hold: a setup on ONE NVIDIA card
    with its own VRAM.  None anywhere else - two GPUs or more (the layers are split, each card holds a part), AMD
    (HIP: its figures are not measured), a unified-memory GPU, Metal, a card whose VRAM does not read."""
    gpus = (pc or {}).get("gpus") or []
    if (pc or {}).get("backend", "cuda") != "cuda" or len(gpus) != 1:
        return None
    g = gpus[0]
    if g.get("vendor") in ("amd", "apple") or g.get("unified_memory") or not g.get("vram_gb") or "index" not in g:
        return None
    used = gpu_used_mib() if used is None else used
    return g, float(g["vram_gb"]) * 1024.0, float(used.get(g["index"], 0))


def glm_fit_verdict(quant: str, pc: dict | None, contexts: list | None = None, used: dict | None = None) -> dict | None:
    """Whether GLM-5.3-Flash quant `quant` can start on this PC's one card (glm_fit_card), None where that is not
    known.  "verdict": "no" - short by more than the margin even at the smallest context the setup offers, and
    without an expert pool too; "slow" - short there, but the engine starts without an expert pool (the experts stay
    in RAM and are computed on the CPU: much slower); "tight" - within the margin there; "fits".  "contexts": the offered contexts that fit (within the margin), "at": glm_fit at
    the smallest context, "card": the GPU, "in_use": all the VRAM in use now (of it at most GLM_FIT_USED_CAP_MIB
    counts: more is a program that can be closed)."""
    card = glm_fit_card(pc, used)
    if card is None or quant not in GLM_FIT:
        return None
    g, total, in_use = card
    counted = min(in_use, GLM_FIT_USED_CAP_MIB)
    contexts = sorted(contexts or CONTEXTS)
    fits = {c: glm_fit(quant, total, counted, c) for c in contexts}
    at = fits[contexts[0]]
    verdict = ("no" if at["short"] - at["pool"] > at["margin"] else "slow" if at["short"] > at["margin"] else
               "tight" if at["short"] > -at["margin"] else "fits")
    return {"verdict": verdict, "at": at, "card": g, "in_use": in_use, "quant": quant,
            "contexts": [c for c in contexts if fits[c]["short"] <= fits[c]["margin"]],
            "by_context": fits}


def glm_fit_card_name(g: dict) -> str:
    return f"{g.get('name') or 'card'} ({float(g['vram_gb']):.0f} GB)"


def glm_fit_message(v: dict, others: list = ()) -> tuple:
    """(what is short, what to do) for a quant that cannot start (glm_fit_verdict "no"): the terms, the card's room,
    and the GLM-5.3-Flash quants that do fit this card (`others`)."""
    t, G = v["at"], 1024.0
    name = Q.shown("glm", v["quant"])
    used = (f"{t['total'] / G:.1f} less the {t['used'] / G:.1f} GB the desktop and other programs use on it now"
            if t["used"] >= 50 else "nothing else is on it")
    what = (f"{name} cannot start on this PC's {glm_fit_card_name(v['card'])}: before any expert is cached it needs "
            f"about {t['need'] / G:.1f} GB of VRAM at the smallest context ({context_label(t['ctx'])}) - dense weights "
            f"{t['dense'] / G:.1f}, the context's state {t['state'] / G:.1f}, the engine's working memory "
            f"{t['work'] / G:.1f}, the smallest expert pool {t['pool'] / G:.1f}, kept free {t['reserve'] / G:.1f} - "
            f"and the card has {t['have'] / G:.1f} GB ({used}): about {t['short'] / G:.1f} GB short")
    alone = t["short"] - t["used"]
    how = ("with nothing else on the card it would be " + ("a tight fit" if alone > -t["margin"] else "enough")
           + " (the desktop on another GPU or the CPU's graphics); " if t["used"] >= 50 and alone <= t["margin"] else "")
    how += (("these GLM-5.3-Flash quants fit this card: " + ", ".join(Q.shown("glm", q) for q in others)) if others
            else "no GLM-5.3-Flash quant fits this card") + \
           "; the other models' memory needs: docs/MEMORY.md. Nothing was downloaded. To try anyway: --force-fit"
    return what, how


def glm_fit_gate(a, quant: str, pc: dict | None, used: dict | None = None) -> dict | None:
    """The check before the download: a quant that cannot start on this card ends the setup here with what is short
    (--force-fit: a warning, and on); one that starts only without an expert pool ("slow") and a tight one are said.
    The verdict, or None where the check does not apply."""
    v = glm_fit_verdict(quant, pc, used=used)
    if v is None:
        return None
    if v["in_use"] > GLM_FIT_USED_CAP_MIB:
        warn(f"{v['in_use'] / 1024:.1f} GB of this card's VRAM is in use now (another program, or a model that is "
             "running): the engine gets what is free when it starts")
    others = [] if v["verdict"] not in ("no", "slow") else [
        q for q in GLM_FIT if q != quant and q in MODELS and q not in MODELS_BY_HAND
        and (glm_fit_verdict(q, pc, used=used) or {}).get("verdict") in ("fits", "tight")]
    if v["verdict"] == "no":
        what, how = glm_fit_message(v, others)
        if getattr(a, "force_fit", False):
            warn(what + " - going on (--force-fit)")
            return v
        fail(what, how)
    if v["verdict"] == "slow":
        t = v["at"]
        warn(f"{Q.shown('glm', quant)} will run slowly on this PC's {glm_fit_card_name(v['card'])}: the card has "
             f"about {t['have'] / 1024:.1f} GB free and holding the model's experts takes about "
             f"{t['need'] / 1024:.1f} GB, so Maya starts without an expert pool - the experts stay in RAM and are "
             "computed on the CPU. The answers are the same; writing and reading prompts are much slower than on a "
             "card that holds the experts"
             + ("; these GLM-5.3-Flash quants fit this card: " + ", ".join(Q.shown("glm", q) for q in others)
                if others else ""))
    if v["verdict"] == "tight":
        t = v["at"]
        warn(f"{Q.shown('glm', quant)} is a tight fit on this PC's {glm_fit_card_name(v['card'])}: by the setup's "
             f"estimate it needs about {t['need'] / 1024:.1f} GB of the card's {t['have'] / 1024:.1f} GB free at a "
             f"{context_label(t['ctx'])} context, and the estimate can be off by {t['margin']} MiB. If the engine says "
             "it has no room for the expert pool: close programs that use the card, or a lower --vram-reserve-mib")
    return v


def glm_fit_context(a, v: dict | None, ctx_wanted: int | None) -> list | None:
    """The contexts to offer for a checked quant (glm_fit_gate's verdict): all of them where all fit or nothing is
    known (None), else the ones that fit, said.  A --context that does not fit ends the setup (--force-fit: on)."""
    if v is None or v["verdict"] in ("no", "slow") or len(v["contexts"]) == len(v["by_context"]):
        return None
    fit = v["contexts"] or [min(v["by_context"])]
    big = v["by_context"][max(v["by_context"])]
    line = (f"{Q.shown('glm', v['quant'])} starts on this card with a context of up to {context_label(max(fit))}: a "
            f"longer one needs about {big['short'] / 1024:.1f} GB more VRAM than the card has free")
    if ctx_wanted and ctx_wanted > max(fit):
        if not getattr(a, "force_fit", False):
            fail(line, f"--context {max(fit)} (or leave --context out); to try anyway: --force-fit")
        warn(line + " - going on (--force-fit)")
        return None
    warn(line)
    return fit


def choose_model(a, models: Path, inst: dict, pc: dict | None = None) -> tuple:
    """("download", one of MODELS) or ("local", the model's first .gguf file): --gguf-dir, --model, else asked among
    the downloads - the earlier download is the default.  ("exl3", the folder): --model-dir naming an EXL3 folder of
    GLM-5.3-Flash.  `pc` (check_pc's): the rows of quants that cannot start on its one card say so (glm_fit_verdict),
    and the recommendation is one that can."""
    metal = getattr(a, "backend", None) == "metal"
    if getattr(a, "model_dir", None) and exl3_folder_family(a.model_dir) == "glm":
        if metal:
            fail("an EXL3 quant runs on NVIDIA GPUs only", f"a GGUF quant runs on Metal: {ME} --setup --model {next(iter(MODELS))}")
        return "exl3", Path(a.model_dir).expanduser().resolve()
    if a.gguf_dir:
        first, why = local_first(Path(a.gguf_dir))
        if first is None:
            fail(why, "--gguf-dir takes the folder that holds the GLM-5.3-Flash .gguf files, or the (first) .gguf file")
        return "local", first
    if a.model:
        if metal and is_exl3(a.model):
            fail(f"{a.model} runs on NVIDIA GPUs only", f"a GGUF quant runs on Metal: {ME} --setup --model {next(iter(MODELS))}")
        if metal and a.model in G53_MODELS:            # (GLM-5.3: before its download, not after it)
            fail(f"{G53.SHOWN} runs on NVIDIA and AMD GPUs (not on Metal yet)", f"{ME} --setup picks another model")
        return "download", a.model
    opts = [q for q in MODELS if (q not in MODELS_BY_HAND or q == inst.get("quant"))
            and not (metal and is_exl3(q))]   # (a chosen-by-hand download: only where it is installed; Metal: GGUFs)
    # GLM-5.3 (744B): its downloads after GLM-5.3-Flash's - not on Metal
    g53 = [q for q in G53_MODELS if q in glm_downloads()] if not metal else []
    # cards of 24 GB or less: Maya-S24 (its 4-bit attention leaves ~1.5 GB more of the card for experts - decode about
    # 14% faster there), else Maya-S
    try:
        cards = [g for g in S.gpus() if g.get("vram_gb")]
    except Exception:  # noqa: BLE001 - no NVIDIA tools (AMD): no card-based recommendation
        cards = []
    if not cards:   # AMD: the same recommendation, from the cards ROCm or Windows lists
        try:
            cards = [g for g in S.amd_gpus() if g.get("vram_gb")]
        except Exception:  # noqa: BLE001 - no card-based recommendation
            cards = []
    top = "Maya-S24" if cards and all(g["vram_gb"] <= 24.5 for g in cards) and "Maya-S24" in MODELS else opts[0]
    # by this PC's VRAM + RAM (fit_recommendation): no larger than that one, Maya-S24 only where it is that one (its
    # 4-bit attention is for small cards); the cache at the default context sized as GLM-5.3's (G53.KV_BYTES: its
    # INT8 latent, at least Flash's)
    try:
        ram = mem_gb()[0]
        gpus = metal_gpus() if metal else setup_gpus(a)
    except Exception:  # noqa: BLE001 - the PC does not read: the usual pick
        ram, gpus = 0.0, []
    sizes = {q: MODELS[q]["download_gb"] for q in opts if q in (top, "Maya-XS-IQ1_S") or
             (q != "Maya-S24" and not is_exl3(q))}
    rec = fit_recommendation(sizes, gpus, ram, DEFAULT_CONTEXT * G53.KV_BYTES / 2**30, top)
    # the quants that cannot start on this PC's one card (glm_fit_verdict): marked, and never the recommendation
    # where a smaller one can
    try:
        used = gpu_used_mib() if glm_fit_card(pc, {}) else {}
        nofit = {q: v for q in opts for v in [glm_fit_verdict(q, pc, used=used)]
                 if v and v["verdict"] in ("no", "slow")}
    except Exception:  # noqa: BLE001 - the card does not read: no marks
        nofit = {}
    if rec in nofit:
        rec = next((q for q in sorted(sizes, key=lambda q: -sizes[q]) if q not in nofit and sizes[q] <= sizes[rec]),
                   rec)
    default =inst["quant"] if inst.get("quant") in MODELS or inst.get("quant") in g53 else rec
    part = getattr(a, "glm_part", None)                # (the model menu's row: "flash" or "glm53"; None: not asked)
    if part == "glm53":                                # (each row of the model menu lists its own model's files)
        opts = []
    elif part == "flash":
        g53 = []
    rows = []
    for q in opts:
        m, d = MODELS[q], download_dir(models, q)
        have = all(p.exists() for p in local_shards(m, d))
        rows.append((f"{Q.shown('glm', q)}: " + (f"downloaded, in {d}" if have else
                                                 f"download {m['download_gb']:.1f} GB from Hugging Face"),
                     (f"does not fit this PC's {nofit[q]['card']['vram_gb']:.0f} GB card"
                      if nofit[q]["verdict"] == "no" else
                      f"runs slowly on this PC's {nofit[q]['card']['vram_gb']:.0f} GB card (experts on the CPU)")
                     if q in nofit else
                     ("recommended for cards of 24 GB or less" if q == "Maya-S24" else "recommended") if q == rec
                     else None, m["about"]))
    # GLM-5.3 (744B)'s own recommendation where it lists more than one (fit_recommendation by g53_figures' memory),
    # at most Maya's own quant (the owner, 2026-11-11): a larger third-party quant is listed, never recommended
    rec53 = fit_recommendation({q: g53_figures(q)[0] for q in g53}, gpus, ram,
                               DEFAULT_CONTEXT * G53.KV_BYTES / 2**30,
                               "GLM-5.3-Maya-XS" if "GLM-5.3-Maya-XS" in g53 else None) if len(g53) > 1 else None
    for q in g53:                                      # GLM-5.3 (744B, glm-dsa): its downloads (hidden)
        m, d = G53_MODELS[q], download_dir(models, q)
        have = all(p.exists() for p in local_shards(m, d))
        rows.append((f"{G53.LABEL} · {Q.glm53_quant(q)}, text only: " +
                     (f"downloaded, in {d}" if have else f"download {m['download_gb']:.1f} GB from Hugging Face"),
                     f"recommended for {G53.SHOWN}" if q == rec53 else None,
                     f"{m['about']}; {g53_memory_note(quant=q)}"))
    if part != "flash":
        rows.append((f"{G53.SHOWN} (744B, text only): GGUF files you already have", None,   # GLM-5.3 (744B, glm-dsa)
                     "Z.ai's GLM-5.3 on the same engine; set up from its GGUF folder with --gguf-dir <folder>"))
    if default not in opts + g53:                      # (the other model's file was the default: this model's own)
        default = (rec53 or g53[0]) if part == "glm53" and g53 else rec
    name = G53.LABEL if part == "glm53" else "GLM-5.3-Flash"
    pick = choose(f"Which {name}?", [f"The model to download ({name} GGUF files you already have: "
                                     "--gguf-dir <file or folder>):"], rows,
                  (opts + g53).index(default) if default in opts + g53 else 0, a.yes)
    if pick >= len(opts) + len(g53):
        fail(f"{G53.SHOWN} is set up from its GGUF files", f"{ME} --setup --gguf-dir <the folder of its .gguf files>")
    return "download", (opts + g53)[pick]


def g53_choice(choice: tuple | None) -> bool:
    """choose_model's answer is GLM-5.3 (744B, glm-dsa), known before its files are read: one of its downloads, or
    GGUF files whose header says glm-dsa."""
    if not choice:
        return False
    if choice[0] == "download":
        return choice[1] in G53_MODELS
    return choice[0] == "local" and Path(choice[1]).suffix == ".gguf" and gguf_head(Path(choice[1]))[0] == G53.ARCH


def g53_figures(quant: str | None) -> tuple:
    """(memory_gb, dense_gb) of a GLM-5.3 download (G53_MODELS' own figures), G53's (UD-IQ2_M's) for any other files."""
    m = G53_MODELS.get(quant or "", {})
    return m.get("memory_gb", G53.MEMORY_GB), m.get("dense_gb", G53.DENSE_GB)


def g53_contexts(pc: dict, parallel: int = 1, kv_bytes: int | None = None, reserve_mib: int | None = None,
                 dense_gb: float | None = None) -> list:
    """The contexts the setup offers for GLM-5.3 (744B).  Its engine keeps the whole cache in VRAM for now
    (G53.KV_STREAMING False), so on discrete GPUs only the contexts whose cache - one a conversation (`parallel`: the
    config's --parallel), `kv_bytes` a position (G53.KV_BYTES: the INT8 latent) - with the dense weights (`dense_gb`:
    the quant's, g53_figures; G53.DENSE_GB) and on each GPU its working memory and the VRAM the engine leaves free
    (`reserve_mib`, G53.RESERVE_MIB: CUDA's default) leaves a tenth of the VRAM to the experts (G53's estimates): a
    larger one would not start.  g53_cache_plan reads them from what the setup writes.  An APU that shares its RAM
    (hip_unified_memory: its VRAM figure is no limit), a PC whose VRAM does not read, and KV streaming once it is
    served: every one.  The smallest is always offered."""
    gpus = pc.get("gpus") or []
    vram = sum(g.get("vram_gb") or 0 for g in gpus)
    if G53.KV_STREAMING or not gpus or not vram or any(hip_unified_memory(g) for g in gpus):
        return list(CONTEXTS)
    reserve = G53.RESERVE_MIB if reserve_mib is None else reserve_mib
    dense = G53.DENSE_GB if dense_gb is None else dense_gb
    room = 0.9 * vram - dense - (G53.GPU_WORK_GB + reserve / 1024) * len(gpus)
    fit = [c for c in CONTEXTS if max(1, parallel) * c * (kv_bytes or G53.KV_BYTES) / 2**30 <= room]
    if len(fit) < len(CONTEXTS):
        (ok if fit else warn)(
            f"{G53.SHOWN} keeps its whole attention cache in VRAM for now: " +
            (f"up to a {context_label(fit[-1])} context on these GPUs ({vram:.0f} GB of VRAM" +
             (f", {parallel} answers at once" if parallel > 1 else "") + ")" if fit else
             f"its dense weights and the smallest context need more VRAM than these GPUs have ({vram:.0f} GB) - "
             "it may not start"))
    return fit or CONTEXTS[:1]


def apu_memory(vram: float, ram: float, linux: bool) -> float:
    """A unified-memory APU's one memory, as fit_memory counts it: on Linux the GPU's memory is drawn from the RAM
    (the larger of the two figures), on Windows the carve-out sits beside the RAM Windows counts (the sum)."""
    return max(vram, ram) if linux else vram + ram


# apu_context's extra room on top of the model, its cache and apu_margin_gb: what `need` leaves out (the engine's work
# buffers, the drafting block's own cache) plus a margin.  Measured on a 192 GB Gorgon Halo (Windows) with
# GLM-5.3-Maya-XS: at 64K, 55 experts (0.43 GB) were still read from the SSD; with this the setup recommends 32K there.
APU_CONTEXT_SAFETY_GB = 4.0


def apu_context(vram: float, ram: float, need: float, kv_bytes: int, parallel: int = 1,
                contexts=CONTEXTS, linux: bool = False) -> int | None:
    """On a unified-memory APU (hip_unified_memory: the GPU's memory and the RAM are one pool; Windows charges the GPU's
    carve-out to its commit): the largest context whose cache (`kv_bytes` a position, one a conversation) still leaves
    the whole model (`need`: VRAM + RAM that keep every expert in memory, g53_figures) in memory - the APU's memory
    (apu_memory: `linux` counts it once) less what the engines keep free (apu_margin_gb) - so no expert is read from
    the SSD.  None when the memory does not read or not even the smallest context leaves the model in memory."""
    if ram <= 0 or need <= 0:
        return None
    room = apu_memory(vram, ram, linux) - apu_margin_gb(ram) - need - APU_CONTEXT_SAFETY_GB
    fits = [c for c in contexts if max(1, parallel) * c * kv_bytes / 2**30 <= room]
    return max(fits) if fits else None


def g53_apu_context(pc: dict, plan: dict, quant: str | None, contexts) -> tuple:
    """(the context to recommend, its one line) for GLM-5.3 on a unified-memory APU (apu_context), else (None, None):
    the memory measured as everywhere else (check_pc's GPUs, mem_gb)."""
    gpus = pc.get("gpus") or []
    if not any(hip_unified_memory(g) for g in gpus):
        return None, None
    try:
        ram = mem_gb()[0]
    except Exception:  # noqa: BLE001 - no reading of the memory: the usual context
        return None, None
    vram, need = sum(g.get("vram_gb") or 0 for g in gpus), g53_figures(quant)[0]
    linux = apu_kind(gpus) == "linux"
    rec = apu_context(vram, ram, need, plan["kv_bytes"], plan["parallel"], contexts, linux=linux)
    pool = apu_memory(vram, ram, linux) - apu_margin_gb(ram)
    if rec is None:
        return None, (f"unified memory: about {pool:.0f} GB for the model here, below the ~{need:.0f} GB that keep all "
                      f"of it in memory - the engine reads some experts from the SSD at any context")
    kv = max(1, plan["parallel"]) * rec * plan["kv_bytes"] / 2**30
    return rec, (f"unified memory: {context_label(rec)} recommended - the largest context whose cache (~{kv:.1f} GB) "
                 f"keeps all of the model (~{need:.0f} GB) in this PC's ~{pool:.0f} GB; a longer one reads experts "
                 "from the SSD")


def g53_cache_plan(a, pc: dict, choice: tuple) -> dict:
    """g53_contexts' "parallel", "kv_bytes", "reserve_mib" and "dense_gb" as the GLM-5.3 config this setup writes will
    have them (write_config): --parallel given now, else the one its config before has (keep_args keeps it - the Models
    page sets it there), else 1; the cache's formats an --env given now asks for (MAYA_GLM_KV_INT8=0: an FP16 latent,
    MAYA_GLM_IDX_FP16=1: FP16 index keys); the reserve: --vram-reserve-mib given now or kept (it wins over the
    environment in the engine), else an --env MAYA_GLM_RESERVE_MB, else the HIP setup's (hip_reserve_mib), else
    CUDA's default; the dense weights: the quant's (g53_figures)."""
    quant = choice[1] if choice[0] == "download" else quant_of(Path(choice[1]))
    old = read_json(glm_config_path(pc, quant)).get("args") or []
    env = parse_env(getattr(a, "env", None) or [])

    def num(v):
        try:
            return int(v)
        except (TypeError, ValueError):
            return None
    given = getattr(a, "vram_reserve_mib", None)
    reserve = num(given if given is not None else flag_value(old, "--vram-reserve-mib"))
    if reserve is None:
        reserve = num(N.lookup(env, "MAYA_GLM_RESERVE_MB"))
    if reserve is None:
        reserve = hip_reserve_mib(pc.get("gpus") or []) if pc.get("backend") == "hip" else G53.RESERVE_MIB
    return {"parallel": max(1, num(getattr(a, "parallel", None) or flag_value(old, "--parallel")) or 1),
            "kv_bytes": G53.kv_bytes_of(G53.GEOMETRY, str(N.lookup(env, "MAYA_GLM_KV_INT8", "1")).strip() != "0",
                                        str(N.lookup(env, "MAYA_GLM_IDX_FP16", "")).strip() == "1"),
            "reserve_mib": max(0, reserve), "dense_gb": g53_figures(quant)[1]}


def g53_drafting_default(pc: dict, pack, ctx: int, env: dict) -> dict | None:
    """GLM-5.3 on one AMD APU (hip_unified_memory): one-GPU drafting's setting until the speed calibration has
    measured it on this PC - {"on", "outside"} from G53.drafting_estimate (on when nearly every routed expert fits in
    GPU memory).  None for any other PC or pack, when the pack's sizes do not read, or when the user set
    MAYA_GLM_MTP_DECODE (--env: theirs stays, nothing measured)."""
    gpus = pc.get("gpus") or []
    if pc.get("backend") != "hip" or len(gpus) != 1 or not hip_unified_memory(gpus[0]) or \
            not G53.pack_is_glm_dsa(pack) or N.lookup(env, "MAYA_GLM_MTP_DECODE") is not None:
        return None
    sizes = G53.pack_bytes(pack)
    if sizes is None:
        return None
    try:
        reserve = int(N.lookup(env, "MAYA_GLM_RESERVE_MB") or hip_reserve_mib(gpus)) / 1024
        headroom = float(N.lookup(env, "MAYA_GLM_RAM_HEADROOM_GB") or 16)
    except ValueError:
        reserve, headroom = hip_reserve_mib(gpus) / 1024, 16.0
    per = G53.kv_bytes_per_token(pack, str(N.lookup(env, "MAYA_GLM_KV_INT8", "1")).strip() != "0",
                                 str(N.lookup(env, "MAYA_GLM_IDX_FP16", "")).strip() == "1") or G53.KV_BYTES
    gpu = float(gpus[0].get("vram_gb") or 0)
    if not WIN:          # a Linux APU's experts sit in the RAM it shares (its VRAM figure is only the carve-out)
        gpu = max(gpu, mem_gb()[0] - headroom)
    on, out = G53.drafting_estimate(gpu, sizes, ctx * per / 2**30, reserve)
    return {"on": on, "outside": out}


def g53_one_gpu_drafting(cfg: dict) -> bool:
    """A GLM-5.3 config whose one-GPU drafting the speed calibration decides: the setup wrote its default
    (installer.one_gpu_drafting, g53_drafting_default), one conversation at a time, no DFlash2."""
    args = cfg.get("args") or []
    return bool((cfg.get("installer") or {}).get("one_gpu_drafting")) and "--dflash" not in args[:-1] and \
        parallel_of(cfg) <= 1 and cfg.get("backend") == "hip"


def g53_memory_gb(pc: dict) -> float | None:
    """VRAM (the GPUs check_pc chose) and RAM together, GB; None when it does not read."""
    try:
        return sum(g.get("vram_gb") or 0 for g in pc.get("gpus") or []) + mem_gb()[0]
    except Exception:  # noqa: BLE001 - no figure for this PC
        return None


def g53_memory_note(pc: dict | None = None, quant: str | None = None) -> str:
    """GLM-5.3's memory guidance, neutral (no speeds): from about the quant's figure (g53_figures: G53.MEMORY_GB for
    UD-IQ2_M and any other files) of VRAM and RAM together every expert stays in memory, with less the engine reads
    the rest from the SSD while it answers - with this PC's figure when `pc` (check_pc's) is given and reads."""
    have = g53_memory_gb(pc) if pc else None
    here = f" (this PC: {have:.0f} GB)" if have else ""
    return (f"every expert stays in memory from about {g53_figures(quant)[0]} GB of VRAM and RAM together{here}; with "
            "less, the engine reads the rest from the SSD while it answers (keep the model on an NVMe SSD)")


# ------------------------------------------------------------------------------------------------ 3. tools
def tools_step(a, packages=PY_PACKAGES) -> Path:
    """The Python packages (into .venv) and llama.cpp's source at the pinned commit - listed, and asked, first."""
    step(3, "Python packages and llama.cpp's source")
    stamp = Path(sys.prefix) / ".strata-pip.json"          # S.pip_install's record of what it installed
    have = read_json(stamp) if stamp.exists() else []
    need = [p for p in packages if p not in (have if isinstance(have, list) else [])]
    llama = Path(a.llama_dir).expanduser().resolve() if a.llama_dir else ROOT / "third_party" / "llama.cpp"
    llama_ok = (llama / "ggml" / "CMakeLists.txt").exists() and (llama / "gguf-py").is_dir()
    if a.llama_dir and not llama_ok:
        fail(f"{llama} is not a llama.cpp checkout (its ggml/ and gguf-py/ folders are missing)")
    downloads = []
    if need:
        downloads.append("Python packages from PyPI into .venv: " + ", ".join(need) + " (roughly 50-100 MB)")
    if not llama_ok:
        downloads.append(f"llama.cpp's source at the pinned commit {S.LLAMA_CPP_COMMIT[:10]} from GitHub, "
                         f"{S.LLAMA_CPP_ZIP} (tens of MB): ggml for the engine build, gguf-py for the pack builder")
    if downloads:
        say("  This step downloads:")
        for d in downloads:
            say("    - " + d)
        if ask("  Download them now?", ["y", "n"], "y", a.yes) != "y":
            fail("nothing was downloaded", f"run {ME} again when you are ready (or pass --llama-dir with a "
                                           f"llama.cpp checkout at commit {S.LLAMA_CPP_COMMIT[:10]})")
    S.pip_install(list(packages), ", ".join(packages))
    if not llama_ok:
        llama = S.get_llama_cpp()
    ok(f"llama.cpp {S.LLAMA_CPP_COMMIT[:10]}: {llama}")
    return llama


# ------------------------------------------------------------------------------------------------ 4. the engine
def cached_generator():
    """The CMake generator build/ was configured with before (another one cannot be used there), or None."""
    m = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (BUILD / "CMakeCache.txt").read_text(errors="replace"), re.M) \
        if (BUILD / "CMakeCache.txt").exists() else None
    return m.group(1).strip() if m else None


def arch_setting(archs) -> str:
    """CMAKE_CUDA_ARCHITECTURES for these GPUs: the compute capabilities of the cards choose_gpus() took, by name -
    never "native", which also builds for every other card in the PC (a Quadro P400 display card beside V100s: sm_61,
    which maya-mm3's kernels cannot be compiled for).  CMakeLists.txt takes sm_60 and newer (Pascal experimental)."""
    return ";".join(str(x) for x in archs)


def toolkit_root_defs(nvcc) -> list:
    """CUDAToolkit_ROOT for the toolkit whose nvcc builds the engine.  Without it CMake can take cudart and cuBLAS from
    another toolkit: with --nvcc /opt/cuda-13.0/bin/nvcc on Ubuntu 24.04 that also has the distribution's CUDA 12.0
    (nvidia-cuda-toolkit), the engine was compiled with the 13.0 headers but linked libcudart.so.12 from
    /usr/lib/x86_64-linux-gnu.  The distribution's own nvcc (/usr/bin) keeps CMake's search as before (Strata #1359)."""
    root = Path(nvcc).resolve().parent.parent
    return [] if root == Path("/usr") else [f"-DCUDAToolkit_ROOT={root}"]


def compile_engine(archs, gpu_ids, nvcc, host_compiler, llama: Path, src: str, soft=False,
                   target: str = "maya", also=()) -> dict | None:
    """cmake configure + build of the `maya` target (DeepSeek: `maya-ds4`) in build/ - and of the targets in `also`
    (--rebuild: every engine the PC uses) in the same build; the exact commands are printed as they run.  soft: an
    update's rebuild (refresh_engine) - a failure stops with the reason and says that the engine from before is not
    started (#805)."""
    def stop(what):
        msg = f"the engine build stopped while {what} (the reason is above)"
        hint = ("common causes: 'unsupported Microsoft Visual Studio version' - CUDA 12.x needs Visual Studio 2019 or "
                "2022 (its Build Tools are enough; only CUDA 13.2 and newer take 2026);" if WIN else
                "common causes: 'unsupported GNU version' - install an older g++ your CUDA accepts (e.g. g++-13) and run "
                "again (it is found by itself), or pass --host-compiler g++-13;") + (
                "\n       'Unsupported gpu architecture compute_70' (or compute_61) - CUDA 13 cannot build for Volta "
                "or Pascal, install CUDA 12.x;\n       the compiler killed (out of memory) - close programs and run it "
                "again "
                "(it continues)")
        if soft:
            # (not the engine compiled before: it is from the older source, and this version's server and config
            #  start it with settings it may not know or silently ignore - #805, on NVIDIA as on AMD)
            fail("Maya's NVIDIA engine changed with this update, and " + msg + ": the engine from before does not "
                 "match this version, so it is not started",
                 hint + "\n       fix the reason, then start Maya again (it compiles again)")
        fail(msg, hint)

    cmake = pick_cmake()
    if cmake is None:
        return stop(f"looking for CMake 3.24+ (pip installs one into .venv: run {ME} --setup)")
    gen = cached_generator()
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and gen in (None, "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    cuda_archs = arch_setting(archs)
    conf += ["-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
             "-DSTRATA_ENABLE_CUDA=ON", f"-DCMAKE_CUDA_ARCHITECTURES={cuda_archs}",
             "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
             f"-DCMAKE_CUDA_COMPILER={nvcc}", *toolkit_root_defs(nvcc), f"-DSTRATA_GGML_DIR={llama}"]
    if S.ARM and S.arm_cpu_arch():
        conf.append(f"-DGGML_CPU_ARM_ARCH={S.arm_cpu_arch()}")
    if host_compiler:
        conf.append(f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which(host_compiler) or host_compiler}")
    # (CMake's compiler checks see only the chosen cards, numbered as nvidia-smi numbers them)
    env = dict(os.environ, CUDA_DEVICE_ORDER="PCI_BUS_ID", CUDA_VISIBLE_DEVICES=",".join(str(i) for i in gpu_ids))
    # nvcc takes 2-4 GB per job on the big kernels: half the threads, and no more jobs than RAM / 4 GB
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(mem_gb()[0] // 4) or 2))
    targets = [target] + [t for t in also if t != target]
    build = [cmake, "--build", str(BUILD), "--target", *targets, "-j", str(jobs)]
    say("  Compiling the engine" + ("s " + ", ".join(targets) if len(targets) > 1 else "") + " for " +
        ", ".join(f"sm_{x}" for x in archs) + " (10-30 minutes the first time, a few minutes after an update) ...")
    stopped = cmake_steps(conf, build, env, "build-maya.bat", nvcc=nvcc)
    if stopped:
        return stop(stopped)
    # the targets compiled from this source for these GPUs (a GLM-5.3-Flash and a DeepSeek setup share build/)
    old = read_json(STAMP)
    same = old.get("src") == src and old.get("archs") == archs and old.get("nvcc") == nvcc
    meta = {"src": src, "archs": archs, "gpu_ids": list(gpu_ids), "nvcc": nvcc, "host_compiler": host_compiler,
            "llama": str(llama), "lib_dirs": cuda_lib_dirs(nvcc), "date": time.strftime("%Y-%m-%d %H:%M"),
            "targets": sorted(set(targets) | (set(old.get("targets") or ["maya"]) if same else set()))}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(("engines compiled: " if len(targets) > 1 else "engine compiled: ") +
       ", ".join(str(exe_for_target(t)) for t in targets))
    return meta


def cmdline(cmd) -> str:
    """One command as a line of a Windows .bat."""
    return " ".join(f'"{x}"' if re.search(r'[\s;&|<>^()]', str(x)) else str(x) for x in cmd)


def shell_join(cmd) -> str:
    """One command as it is typed here: a POSIX shell, or Windows' cmd."""
    return cmdline(cmd) if WIN else shlex.join(cmd)


def path_unquoted(value: str) -> str:
    """A Windows PATH with the double quotes taken out of its entries (an entry that is quoted because it holds a ';'
    is left out: it cannot be written without them)."""
    entries, cur, quoted = [], [], False
    for ch in value:
        if ch == '"':
            quoted = not quoted
        elif ch == ";" and not quoted:
            entries.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    entries.append("".join(cur))
    return ";".join(e for e in entries if ";" not in e)


def vs_env(env=None) -> dict:
    """The environment for a .bat that calls Visual Studio's vcvars64.bat: PATH's entries without quotes.  A quoted
    entry with a parenthesis in it - "C:\\Program Files (x86)\\NVIDIA Corporation\\PhysX\\Common", as some installers
    write it - stops vcvars64.bat at once with '\\NVIDIA was unexpected at this time.' and the build or the toolkit
    probe with it (issue #251); Windows finds the same programs without the quotes."""
    env = dict(os.environ if env is None else env)
    for k in [k for k in env if k.upper() == "PATH"]:
        if '"' in env[k]:
            env[k] = path_unquoted(env[k])
    return env


def cmake_steps(conf, build, env, bat_name: str, vcvars=None, nvcc=None) -> str | None:
    """CMake's configure, then its build (once more when that stops: it continues where it stopped).  None when both
    worked, else what stopped.  On Windows both run in a .bat that first calls Visual Studio's vcvars64.bat - the
    compiler's environment - the way Strata's setup.py builds there (cmake_build).  `nvcc` says which toolkit the
    CUDA build uses: its release decides whether a VS 2026-only PC is usable (S.find_vcvars)."""
    if not WIN:
        if run(conf, env=env, check=False).returncode != 0:
            return "configuring"
        if run(build, env=env, check=False).returncode != 0:
            say("  (the build stopped - trying it once more: it continues where it stopped)")
            if run(build, env=env, check=False).returncode != 0:
                return "compiling"
        return None
    vcvars = vcvars or S.find_vcvars(nvcc=nvcc)
    if vcvars is None:
        return "looking for Visual Studio's C++ compiler (the Build Tools with 'Desktop development with C++')"
    bat = ROOT / bat_name
    bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{cmdline(conf)} || exit /b 3\r\n{cmdline(build)} && exit /b 0'
                   '\r\necho   (the build stopped - trying it once more: it continues where it stopped)\r\n'
                   f'{cmdline(build)} || exit /b 4\r\n', encoding="utf-8")
    say(f"  > {bat}   (Visual Studio's environment, then:)")
    say("  > " + cmdline(conf))
    say("  > " + cmdline(build))
    rc = run(["cmd", "/c", str(bat)], env=vs_env(env), check=False, echo=False).returncode
    return None if rc == 0 else "compiling" if rc == 4 else "configuring"


PROBE_CU ="""#include <cmath>
#include <string>
#include <vector>
__global__ void k(float* x) { x[0] = rsqrtf(x[0]) + sinf(x[1]) + expf(x[2]) + __expf(x[3]) + sqrtf(x[4]); }
int main() { std::vector<std::string> v(1); return (int) std::cos(0.0) - 1 + (int) v.size() - 1; }
"""


def toolchain_for(archs, nvcc_given: str | None, hc_given: str | None) -> tuple:
    """(nvcc, host compiler or None for the default) that compile the engine's kind of CUDA here.  Every installed
    toolkit that can build for these GPUs (newest first; --nvcc alone if given) with the default g++, then each
    installed g++-N (newest first; --host-compiler alone if given), on a small file the way the engine compiles:
    C++20 and the math functions the kernels use.  A toolkit refuses a g++ newer than it supports, and a newer
    glibc's own rsqrtf/sinpi declarations clash with older CUDA headers - nothing but trying tells which pair works
    on a given system.  On Windows the compiler is Visual Studio's (in its vcvars64.bat environment): only the
    toolkits are tried."""
    import tempfile

    if nvcc_given:
        nvccs = [nvcc_given]
    else:
        seen, found = set(), []
        for c in nvcc_candidates():
            if c and Path(c).exists() and os.path.realpath(c) not in seen:
                seen.add(os.path.realpath(c))
                found.append((c, S.nvcc_version(c)))
        lo, hi = nvcc_range(archs)
        nvccs = [c for c, v in sorted((f for f in found if f[1]), key=lambda f: f[1], reverse=True)
                 if v >= lo and (hi is None or v < hi)]
    hcs = [None] if WIN else [hc_given] if hc_given else [None] + sorted(
        {q.name for d in ("/usr/bin", "/usr/local/bin") for q in Path(d).glob("g++-[0-9]*")
         if re.fullmatch(r"g\+\+-\d+", q.name)}, key=lambda n: -int(n[4:]))
    say("  Finding a CUDA toolkit and C++ compiler that build the engine here ...")
    with tempfile.TemporaryDirectory() as t:
        src = Path(t) / "probe.cu"
        src.write_text(PROBE_CU, encoding="utf-8")
        for nv in nvccs:
            vcvars = S.find_vcvars(nvcc=nv) if WIN else None   # which VS a toolkit takes is in its release (S.find_vcvars)
            for c in hcs:
                cmd = [nv, "-std=c++20", "-c", str(src), "-o", str(Path(t) / "probe.o")] + \
                      (["-ccbin", shutil.which(c) or c] if c else [])
                if vcvars:                             # cl.exe is on PATH only inside Visual Studio's environment
                    bat = Path(t) / "probe.bat"
                    bat.write_text(f'@echo off\r\ncall "{vcvars}" >nul\r\n{cmdline(cmd)}\r\n', encoding="utf-8")
                    cmd = ["cmd", "/c", str(bat)]
                if subprocess.run(cmd, capture_output=True, text=True,
                                  env=vs_env() if vcvars else None).returncode == 0:
                    ok(f"{nv} with " + ("Visual Studio's C++ compiler" if WIN else c or "the default g++"))
                    return nv, c
    warn("no installed CUDA toolkit and C++ compiler pair compiled a test file: the build will show why (" +
         toolkit_hint(archs) + ")")
    return (nvccs[0] if nvccs else nvcc_given), hc_given


def hip_toolchain(pc: dict) -> tuple:
    """(ROCm root, its clang++, its bin folder, the build's environment) for a HIP build - the engine's and, on
    Windows, the vision encoder's (#345)."""
    root = Path(pc["rocm"])
    clang = hip_clang(root) or root / "llvm/bin/clang++"
    bindir = Path(pc.get("rocm_bin") or root / "bin")
    env = dict(os.environ, HIP_PLATFORM="amd", HIP_COMPILER="clang", HIP_RUNTIME="rocclr",
               ROCM_PATH=str(root), HIP_PATH=str(root))
    env["PATH"] = os.pathsep.join([str(bindir), str(clang.parent), env.get("PATH", "")])
    if not WIN:
        libs = [str(root / d) for d in ("lib", "lib64") if (root / d).is_dir() and root != Path("/usr")]
        env["LD_LIBRARY_PATH"] = os.pathsep.join(libs + [env.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
    # Keep compiler-cache writes inside this project, including when run in a workspace sandbox.
    env.setdefault("CCACHE_DIR", str(BUILD / ".ccache"))
    if WIN:
        env["HIP_DEVICE_LIB_PATH"] = str(clang.parent.parent / "amdgcn" / "bitcode")
    elif (clang.parent.parent.parent == root / "lib" and re.fullmatch(r"llvm-\d+", clang.parent.parent.name)
          and not (root / "amdgcn" / "bitcode").is_dir() and "HIP_DEVICE_LIB_PATH" not in env):
        # Debian/Ubuntu's rocm-device-libs-<N>: the bitcode sits in the LLVM's clang resource folder
        # (/usr/lib/llvm-<N>/lib/clang/<N>/amdgcn/bitcode), not under ROCM_PATH, where clang looks with ROCM_PATH set.
        # Only for that layout's compiler (<root>/lib/llvm-<N>/bin): every other ROCm builds as before
        found = sorted(clang.parent.parent.glob("lib/clang/*/amdgcn/bitcode"),
                       key=lambda p: [int(x) for x in re.findall(r"\d+", p.parent.parent.name)])
        if found:
            env["HIP_DEVICE_LIB_PATH"] = str(found[-1])
    return root, clang, bindir, env


def copy_hip_dlls(bindir: Path, dest: Path) -> None:
    # Windows takes a program's DLLs from its own folder, then System32 - where AMD's driver puts its own HIP
    # runtime - and PATH only after: the runtime the program was built with goes next to it (as llama.cpp's
    # Windows builds do; with the driver's, cudaMemGetInfo failed there)
    for pattern in ("amdhip64*.dll", "amd_comgr*.dll", "rocm_kpack*.dll"):
        for dll in bindir.glob(pattern):
            try:
                shutil.copy2(dll, dest)
            except OSError as e:                   # (in use: a program still running keeps the one it has)
                warn(f"could not copy {dll.name} next to {dest} ({e})")


def compile_engine_hip(pc: dict, llama: Path, src: str, soft=False, target: str = "maya", also=()) -> dict | None:
    """The HIP build of `target` - and of the targets in `also` (--rebuild: every engine the PC uses) - in build-hip/,
    for this setup's AMD GPUs and those every installed model of build-hip/ runs on (hip_archs_in_use: one binary
    each for all of them, as the CUDA build); where this ROCm cannot compile for one of the others, this model's."""
    cmake = pick_cmake()
    if not cmake:
        fail("CMake 3.24 or newer is needed")
    root, clang, bindir, env = hip_toolchain(pc)
    own = sorted(str(x) for x in pc["archs"])
    archs = hip_archs_in_use(own, pc.get("amd_found"))
    if archs != own and not hip_builds(root, env, clang, archs):
        warn("another installed model runs on " + ", ".join(x for x in archs if x not in own) + f", which the ROCm in "
             f"{root} does not compile for: the engine is compiled for this model's GPUs only")
        archs = own
    elif archs != own:
        ok("also compiled for " + ", ".join(x for x in archs if x not in own) + " (the GPUs another installed model "
           "runs on)")
    pc = dict(pc, archs=archs)
    conf = [cmake]
    if WIN:
        # as tools\hip\build_maya_windows.bat, which built it on Windows (#54): ROCm's clang for the host code too, in
        # Visual Studio's environment (cmake_steps), with Ninja, the paths with forward slashes; AVX2 ggml
        ninja = venv_tool("ninja") or shutil.which("ninja")
        if not ninja:
            fail("Ninja is needed to compile the HIP engine on Windows", f"run {ME} --setup again (pip puts it into .venv)")
        bitcode = Path(env["HIP_DEVICE_LIB_PATH"])
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={Path(ninja).as_posix()}",
                 f"-DCMAKE_C_COMPILER={(clang.parent / 'clang.exe').as_posix()}",
                 f"-DCMAKE_CXX_COMPILER={clang.as_posix()}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root.as_posix()}",
                 "-DSTRATA_PORTABLE=ON"]
        if " " not in str(root) + str(bitcode):        # (with a space in them, clang takes HIP_PATH and the above)
            conf.append(f"-DCMAKE_HIP_FLAGS=--rocm-path={root.as_posix()} --rocm-device-lib-path={bitcode.as_posix()}")
    conf += ["-S", str(ROOT), "-B", str(BUILD), "-DCMAKE_BUILD_TYPE=Release",
             "-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_PREFILL_MMQ=ON",
             "-DSTRATA_NATIVE_EXPERTS=ON", "-DSTRATA_BUILD_TESTS=OFF",
             f"-DCMAKE_HIP_ARCHITECTURES={';'.join(pc['archs'])}",
             f"-DCMAKE_HIP_COMPILER={clang.as_posix()}", f"-DCMAKE_PREFIX_PATH={root.as_posix()}",
             f"-DSTRATA_GGML_DIR={llama.as_posix()}"]
    if not WIN and root == Path("/usr"):   # a distribution's ROCm: CMake's HIP check looks for hip-lang under this root
        conf.append(f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root.as_posix()}")
    jobs = max(1, min(4, (os.cpu_count() or 4) // 2))
    targets = [target] + [t for t in also if t != target]
    say("  Compiling Maya" + (" (" + ", ".join(targets) + ")" if len(targets) > 1 else "") + " for AMD " +
        ", ".join(pc["archs"]) + " ...")
    vcvars = S.find_vcvars(cuda=False) if WIN else None
    if cmake_steps(conf, [cmake, "--build", str(BUILD), "--target", *targets, "-j", str(jobs)], env,
                   "build-maya-hip.bat", vcvars=vcvars):
        if soft:
            # (not the engine compiled before: it is from the older source, and this version's server and config
            #  start it with settings it may not know or silently ignore - #805's start failed that way)
            fail("Maya's AMD engine changed with this update, and compiling it failed: the engine from before does "
                 "not match this version, so it is not started",
                 "the compiler's reason is above - fix it, then start Maya again (it compiles again)" +
                 ("; build-maya-hip.bat > build-hip.txt 2>&1 runs the same build alone and keeps its output"
                  if WIN else "") +
                 (" - " + VS2026_HIP_HINT if vs_2026(vcvars) else ""))
        fail("the HIP engine build failed", "see the compiler output above" +
             (" - " + VS2026_HIP_HINT if vs_2026(vcvars) else ""))
    if WIN:
        copy_hip_dlls(bindir, EXE.parent)              # (lib_dirs keeps the libraries)
    old = read_json(STAMP)   # (the targets compiled from this source: a GLM-5.3-Flash and a DeepSeek setup share it)
    same = old.get("backend") == "hip" and old.get("src") == src and old.get("archs") == pc["archs"]
    meta = {"backend": "hip", "src": src, "archs": pc["archs"], "rocm": str(root),
            "gpu_ids": [g["index"] for g in pc["gpus"]], "llama": str(llama),
            "lib_dirs": [str(bindir if WIN else root / "lib")], "date": time.strftime("%Y-%m-%d %H:%M"),
            "targets": sorted(set(targets) | (set(old.get("targets") or ["maya"]) if same else set()))}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(("engines compiled: " if len(targets) > 1 else "engine compiled: ") +
       ", ".join(str(exe_for_target(t)) for t in targets))
    return meta


def compile_engine_metal(llama: Path, src: str, soft=False) -> dict | None:
    """`maya-metal` (GLM-5.3-Flash on ggml's Metal backend) in build-metal/; the commands are printed as they run.
    soft: an update's rebuild (refresh_engine) - a failure stops with the reason and says that the engine from before
    is not started (#805)."""
    cmake = pick_cmake()
    if cmake is None:
        fail("CMake 3.24 or newer is needed", f"pip installs one into .venv: run {ME} --setup")
    gen = cached_generator()
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and gen in (None, "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    conf += ["-S", str(ROOT), "-B", str(BUILD), "-DMAYA_ENABLE_METAL=ON", "-DMAYA_METAL_TESTS=OFF",
             f"-DMAYA_GGML_DIR={llama}", "-DCMAKE_BUILD_TYPE=Release"]
    say("  Compiling Maya for Apple Silicon (Metal) ... (a few minutes first time)")
    if cmake_steps(conf, [cmake, "--build", str(BUILD), "--target", "maya-metal", "-j", str(os.cpu_count() or 4)],
                   dict(os.environ), ""):
        if soft:
            # (the engine from before would run under this version's server and config: not started - #805)
            fail("Maya's Metal engine changed with this update, and compiling it failed: the engine from before does "
                 "not match this version, so it is not started",
                 "the compiler's reason is above (no compiler: xcode-select --install) - fix it, then start Maya "
                 "again (it compiles again)")
        fail("the Metal engine build failed", "see the compiler output above (no compiler: xcode-select --install)")
    meta = {"backend": "metal", "src": src, "llama": str(llama), "version": maya_version(),
            "date": time.strftime("%Y-%m-%d %H:%M"), "targets": ["maya-metal"]}
    STAMP.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    ok(f"engine compiled: {EXE}")
    return meta


ENGINE_TARGETS = ("maya", "maya-ds4", "maya-mm3", "maya-hy4", "maya-ik2")


def target_of_exe(exe) -> str | None:
    """The build target whose binary in this build folder `exe` is (maya, maya-ds4, maya-mm3, maya-hy4), or None (an
    engine of another backend's folder, or of none)."""
    if not exe:
        return None
    p = Path(str(exe)).expanduser().resolve()
    return next((t for t in ENGINE_TARGETS if p == exe_for_target(t).resolve()), None)


def engines_in_use(target: str) -> list:
    """What --rebuild compiles: `target` (the model set up now) first, then every other engine of this build folder
    the PC uses - an installed config's engine, a target the stamp lists, an engine binary compiled here before (the
    stamp keeps only the targets compiled from its source, so after an update it may list fewer).  A setup that
    compiles only its own model's engine left the others as they were (#1066: maya-hy4 and maya-ds4 stayed old)."""
    found = {t for t in (read_json(STAMP).get("targets") or []) if t in ENGINE_TARGETS + ("maya-glm-helper",)}
    found |= {t for t in map(lambda p: target_of_exe(read_json(p).get("exe")), configs()) if t}
    found |= {t for t in ENGINE_TARGETS + ("maya-glm-helper",) if exe_for_target(t).exists()}
    return [target] + sorted(found - {target})


def config_gpu_archs(cfg: dict, found: list | None = None) -> set:
    """The compute capabilities of the NVIDIA cards a config runs on (its "gpu" numbers, as nvidia-smi numbers them;
    a card that is gone, or too old for any engine, is left out)."""
    try:
        cards = {int(g["index"]): int(g["arch"]) for g in (S.gpus() if found is None else found) or []}
    except Exception:  # noqa: BLE001 - no nvidia-smi, or it failed: none
        cards = {}
    ids = cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [cfg["gpu"]] if isinstance(cfg.get("gpu"), int) else []
    return {cards[i] for i in ids if isinstance(i, int) and i in cards and cards[i] >= MIN_CC}


def archs_in_use(archs) -> list:
    """The CUDA archs build/ is compiled for: `archs` (the GPUs chosen now) and those of the cards every installed
    config of this build folder runs on - the engines share one CMake cache, so a setup (or --rebuild) for one model
    on one card would otherwise compile the next engine without another model's cards (an RTX 5090 + 4070 SUPER: GLM
    on the 5090, Hy4-Flash on both)."""
    want = {int(x) for x in archs}
    cfgs = [c for c in map(read_json, configs()) if target_of_exe(c.get("exe"))]
    if cfgs:
        try:
            found = S.gpus() or []
        except Exception:  # noqa: BLE001
            found = []
        for c in cfgs:
            want |= config_gpu_archs(c, found)
    return sorted(want)


def config_amd_archs(cfg: dict, found: list) -> set:
    """The gfx archs of the AMD GPUs a config of the HIP build runs on (its "gpu" numbers, as HIP numbers them -
    setup.py's amd_gpus list `found`; a GPU that is gone, or one Maya's HIP build does not take, is left out)."""
    takes = HIP_ARCHS + HIP_ARCHS_UNVALIDATED + HIP_ARCHS_WAVE64
    cards = {int(g["index"]): str(g["arch"]) for g in found or [] if str(g.get("arch")) in takes and "index" in g}
    ids = cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [cfg["gpu"]] if isinstance(cfg.get("gpu"), int) else []
    return {cards[i] for i in ids if isinstance(i, int) and i in cards}


def hip_archs_in_use(archs, found: list | None = None) -> list:
    """The AMD archs build-hip/ is compiled for: `archs` (the GPUs chosen now) and those of the GPUs every installed
    config of this build folder runs on - the engines share one CMake cache and one binary each, so a setup (or
    --rebuild, or a start's update build) for one model on one GPU would otherwise compile the next engine without
    another model's GPUs (an RX 7900 XTX + RX 9070 XT: GLM on the 7900, Hy4-Flash on both) - as archs_in_use does
    for CUDA.  found: the PC's AMD GPUs (check_hip_pc's "amd_found"; None: read now)."""
    want = {str(x) for x in archs}
    cfgs = [c for c in map(read_json, configs()) if target_of_exe(c.get("exe"))]
    if cfgs:
        if found is None:
            try:
                found = S.amd_gpus(installed_hip_info()) if WIN else S.amd_gpus()
            except Exception:  # noqa: BLE001 - no GPU list: this setup's archs
                found = []
        for c in cfgs:
            want |= config_amd_archs(c, found)
    return sorted(want)


def hip_builds(root: Path, env: dict, clang: Path, archs) -> bool:
    """Whether this ROCm compiles for all of these GPUs: its device libraries hold each one's ISA bitcode
    (oclc_isa_version_<n>.bc - gfx1151: 1151); where no such folder is found, assumed (as nvcc_builds)."""
    dirs = [Path(env["HIP_DEVICE_LIB_PATH"])] if env.get("HIP_DEVICE_LIB_PATH") else []
    dirs += [root / "amdgcn" / "bitcode", *root.glob("lib/llvm/lib/clang/*/lib/amdgcn/bitcode"),
             *clang.parent.parent.glob("lib/clang/*/amdgcn/bitcode"), *clang.parent.parent.glob("amdgcn/bitcode")]
    have = next((d for d in dirs if d.is_dir() and any(d.glob("oclc_isa_version_*.bc"))), None)
    return have is None or all((have / f"oclc_isa_version_{str(x)[3:]}.bc").exists() for x in archs)


def nvcc_builds(nvcc, archs) -> bool:
    """Whether this toolkit's release can compile for all of these GPUs (nvcc_range; an unknown release: assumed)."""
    v = S.nvcc_version(nvcc) if nvcc else None
    if not v:
        return True
    lo, hi = nvcc_range(archs)
    return v >= lo and (hi is None or v < hi)


def build_step(a, pc, llama: Path, target: str = "maya") -> dict:
    step(4, "the engine")
    src = S.source_hash(S.ENGINE_SOURCES)              # src/, include/, third_party/ggml, CMakeLists.txt
    meta = read_json(STAMP)
    exe = exe_for_target(target)
    if pc.get("backend") == "metal":
        if exe.exists() and not a.rebuild and meta.get("backend") == "metal" and meta.get("src") == src:
            ok(f"Metal engine already compiled: {exe}")
            return meta
        return compile_engine_metal(llama, src)
    # --rebuild: every engine this PC uses, with the same compiler and GPUs (not only this model's - #1066)
    every = engines_in_use(target) if a.rebuild else [target]
    if len(every) > 1:
        say("  --rebuild: compiling every engine this PC uses (" + ", ".join(every) + "), not only this model's")
    if pc.get("backend") == "hip":
        # (compiled for this model's GPUs and the other installed models' already: nothing to do - hip_archs_in_use)
        if (exe.exists() and not a.rebuild and meta.get("backend") == "hip" and meta.get("src") == src
                and set(hip_archs_in_use(pc["archs"], pc.get("amd_found"))) <= set(meta.get("archs") or [])
                and meta.get("rocm") == pc["rocm"]
                and target in (meta.get("targets") or ["maya"])):
            ok(f"HIP engine already compiled: {exe}")
            return meta
        return compile_engine_hip(pc, llama, src, target=target, also=every[1:])
    archs = archs_in_use(pc["archs"])                  # (the other installed models' cards too)
    if (exe.exists() and not a.rebuild and meta.get("src") == src and set(archs) <= set(meta.get("archs", []))
            and target in (meta.get("targets") or ["maya"])
            and (not a.nvcc or a.nvcc == meta.get("nvcc"))
            and (not a.host_compiler or a.host_compiler == meta.get("host_compiler"))):
        ok(f"engine already compiled for " + ", ".join(f"sm_{x}" for x in meta["archs"]) + f": {exe}")
        return meta
    nvcc, hc = toolchain_for(archs, a.nvcc, a.host_compiler)
    if archs != sorted(int(x) for x in pc["archs"]) and not nvcc_builds(nvcc, archs):
        # (no toolkit here compiles for this model's cards and another model's together: this model's, as before)
        warn("another installed model runs on " + ", ".join(f"sm_{x}" for x in archs if x not in pc["archs"]) +
             f", which {nvcc} does not compile for beside " + ", ".join(f"sm_{x}" for x in pc["archs"]) +
             ": the engine is compiled for this model's GPUs only")
        archs = sorted(int(x) for x in pc["archs"])
        nvcc, hc = toolchain_for(archs, a.nvcc, a.host_compiler)
    elif archs != sorted(int(x) for x in pc["archs"]):
        ok("also compiled for " + ", ".join(f"sm_{x}" for x in archs if x not in pc["archs"]) +
           " (the GPUs another installed model runs on)")
    pc["nvcc"] = nvcc or pc["nvcc"]
    return compile_engine(archs, [g["index"] for g in pc["gpus"]], pc["nvcc"], hc, llama, src, target=target,
                          also=every[1:])


def mixed_builds(a, pc, llama: Path, target: str, quiet: bool = False) -> None:
    """Linux, an AMD GPU (HIP) and an NVIDIA card beside it: the second engine the two-GPU mode ("mixed_gpus") needs -
    the CUDA build of this model's engine in build-cuda/ (for the NVIDIA card, with the toolkit the CUDA setup finds)
    and, for GLM, the AMD GPU's helper program (build-hip/maya-glm-helper).  A build that stops does not stop the setup
    or the start: the model then runs on the AMD GPU alone.  Elsewhere (Windows, no NVIDIA card, a CUDA or Metal
    setup): nothing, not a line.  quiet (a start): a line only when something is compiled."""
    global BUILD, STAMP, EXE, DS4_EXE, MM3_EXE, HY4_EXE
    if WIN or pc.get("backend") != "hip" or target not in ("maya", "maya-ds4"):
        return
    try:
        cards = [g for g in (S.gpus() or []) if int(g.get("arch") or 0) >= MIN_CC]
    except Exception:  # noqa: BLE001 - no nvidia-smi, or it failed: none
        cards = []
    if not cards:
        return
    src = S.source_hash(S.ENGINE_SOURCES)
    archs = sorted({int(g["arch"]) for g in cards})
    hip_meta = read_json(STAMP)
    helper = exe_for_target("maya-glm-helper")
    need_helper = target == "maya" and not (helper.exists() and hip_meta.get("src") == src and
                                            "maya-glm-helper" in (hip_meta.get("targets") or []))
    cuda_dir = ROOT / "build-cuda"                     # (serve/glm_link.py and serve/mixed_gpus.py look there)
    cuda_meta = read_json(cuda_dir / "MAYA-BUILD.json")
    cuda_exe = cuda_dir / Path(exe_for_target(target)).name
    need_cuda = not (cuda_exe.exists() and cuda_meta.get("src") == src and set(archs) <= set(cuda_meta.get("archs", []))
                     and target in (cuda_meta.get("targets") or ["maya"]))
    if not (need_helper or need_cuda):
        if not quiet:
            ok(f"the NVIDIA card's engine already compiled: {cuda_exe}")
        return
    say("  This PC also has an NVIDIA card (" + ", ".join(gpu_label(g) for g in cards) + "): compiling its engine too, "
        "so the model can run on both GPUs (experimental; the first start measures which is faster)")
    if need_helper:                                    # GLM: the AMD GPU's helper program, beside build-hip/maya
        try:
            compile_engine_hip(pc, llama, src, target="maya-glm-helper")
        except SystemExit:
            warn("the AMD GPU's helper program did not compile (the reason is above): the model runs on the AMD GPU "
                 "alone")
            return
    if not need_cuda:
        return
    old = (BUILD, STAMP, EXE, DS4_EXE, MM3_EXE, HY4_EXE)
    BUILD, STAMP = cuda_dir, cuda_dir / "MAYA-BUILD.json"
    EXE, DS4_EXE, MM3_EXE, HY4_EXE = BUILD / "maya", BUILD / "maya-ds4", BUILD / "maya-mm3", BUILD / "maya-hy4"
    try:
        nvcc, hc = toolchain_for(archs, (a.nvcc if a else None) or cuda_meta.get("nvcc"),
                                 (a.host_compiler if a else None) or cuda_meta.get("host_compiler"))
        if not nvcc:
            warn("no CUDA toolkit found for the NVIDIA card (" + toolkit_hint(archs) + "): the model runs on the AMD "
                 "GPU alone")
            return
        BUILD.mkdir(parents=True, exist_ok=True)
        compile_engine(archs, [g["index"] for g in cards], nvcc, hc, llama, src, target=target)
    except SystemExit:
        warn("the NVIDIA card's engine did not compile (the reason is above): the model runs on the AMD GPU alone")
    finally:
        BUILD, STAMP, EXE, DS4_EXE, MM3_EXE, HY4_EXE = old


def refresh_mixed_builds(cfg: dict) -> None:
    """At a start: the two-GPU mode's second engine (mixed_builds) compiled again when the source changed (a git
    pull) - only where the setup compiled it before and the config does not say "mixed_gpus": false."""
    if WIN or cfg.get("backend") != "hip" or cfg.get("mixed_gpus") is False or mixed_family(cfg) is None or \
            not (ROOT / "build-cuda" / "MAYA-BUILD.json").exists() or cfg.get("mixed_exe"):
        return
    meta = read_json(STAMP)
    if not meta.get("archs") or not meta.get("rocm"):
        return
    pc = {"backend": "hip", "rocm": meta["rocm"], "archs": meta["archs"],
          "gpus": [{"index": i} for i in (cfg.get("gpu") or [0])]}
    mixed_builds(None, pc, Path(meta.get("llama") or ROOT / "third_party/llama.cpp"),
                 "maya-ds4" if mixed_family(cfg) == "deepseek" else "maya", quiet=True)


def refresh_engine(cfg: dict) -> None:
    """At a start: the engine's source changed since it was compiled (a git pull) - compile what changed.  Also when
    the stamp has this source but not this target: another model's start compiled only its own engine after the pull
    (the targets share one stamp), so this one is still the old build."""
    meta = read_json(STAMP)
    exe = Path(cfg["exe"]).resolve()
    target = "maya-ds4" if exe == DS4_EXE.resolve() else "maya-mm3" if exe == MM3_EXE.resolve() else \
        "maya-hy4" if exe == HY4_EXE.resolve() else "maya-ik2" if exe == IK2_EXE.resolve() else "maya"
    if not meta or exe not in (EXE.resolve(), DS4_EXE.resolve(), MM3_EXE.resolve(), HY4_EXE.resolve(), IK2_EXE.resolve()):
        return
    src = S.source_hash(S.ENGINE_SOURCES)
    if meta.get("src") == src and target in (meta.get("targets") or ["maya"]):
        return
    if cfg.get("backend") == "hip":
        root = Path(meta.get("rocm") or "/opt/rocm")
        llama = Path(meta.get("llama") or ROOT / "third_party/llama.cpp")
        if hip_clang(root) is None or not (llama / "ggml/CMakeLists.txt").exists():
            # (the engine from before would run under this version's server and config: not started - #805)
            fail(f"Maya's AMD engine changed with this update, but the HIP compiler ({root}) or llama.cpp's source "
                 f"({llama}) it was compiled with is gone: the engine from before does not match this version, so it "
                 "is not started",
                 f"{ME} --setup --rebuild finds the compiler and compiles the engine again")
        pc = {"rocm": str(root), "archs": meta["archs"],
              "gpus": [{"index": i} for i in cfg.get("gpu", [0])]}
        if WIN and meta.get("lib_dirs"):
            pc["rocm_bin"] = meta["lib_dirs"][0]
        compile_engine_hip(pc, llama, src, soft=True, target=target)
        return
    if cfg.get("backend") == "metal":
        llama = Path(meta.get("llama") or ROOT / "third_party/llama.cpp")
        if not (llama / "ggml/CMakeLists.txt").exists():
            # (the engine from before would run under this version's server and config: not started - #805)
            fail(f"Maya's Metal engine changed with this update, but llama.cpp's source ({llama}) it was compiled "
                 "with is gone: the engine from before does not match this version, so it is not started",
                 f"{ME} --setup --rebuild fetches it and compiles the engine again")
        compile_engine_metal(llama, src, soft=True)
        return
    say("  The engine's source changed since it was compiled (an update): compiling what changed ...")
    nvcc, llama = meta.get("nvcc"), Path(meta.get("llama") or ROOT / "third_party" / "llama.cpp")
    if not nvcc or not Path(nvcc).exists() or not (llama / "ggml" / "CMakeLists.txt").exists():
        # (the engine from before would run under this version's server and config: not started - #805)
        fail(f"Maya's NVIDIA engine changed with this update, but the CUDA compiler ({nvcc}) or llama.cpp's source "
             f"({llama}) it was compiled with is gone: the engine from before does not match this version, so it is "
             "not started",
             f"{ME} --setup --rebuild finds the compiler and compiles the engine again")
    gpu_ids = meta.get("gpu_ids") or (cfg.get("gpu") if isinstance(cfg.get("gpu"), list) else [0])
    # (the cards this config runs on too: the last setup may have been another model's, on other GPUs - #1066)
    archs, extra = meta["archs"], config_gpu_archs(cfg) - {int(x) for x in meta["archs"]}
    if extra and nvcc_builds(nvcc, sorted({int(x) for x in meta["archs"]} | extra)):
        archs = sorted({int(x) for x in meta["archs"]} | extra)
    compile_engine(archs, gpu_ids, nvcc, meta.get("host_compiler"), llama, src, soft=True, target=target)


# ------------------------------------------------------------------------------------------------ 5. the model
def shard_set(first: Path) -> list:
    """All files of a split GGUF from its first one (<name>-00001-of-0000N.gguf), or just the file."""
    m = SHARD_RE.search(first.name)
    if not m:
        return [first]
    n, stem = int(m.group(2)), first.name[:m.start()]
    return [first.with_name(f"{stem}-{i:05d}-of-{n:05d}.gguf") for i in range(1, n + 1)]


def gguf_head(path: Path) -> tuple:
    """(general.architecture, tensor count) from the start of a GGUF header: only the key/values before the
    architecture are read (gguf-py writes it first), so a folder of big files is scanned quickly.  ("", 0) when it
    is not a GGUF file or the key does not come early."""
    import struct
    from gguf_reader import GGUF_MAGIC, GGUF_META
    try:
        with open(path, "rb") as f:
            def blob():
                (n,) = struct.unpack("<Q", f.read(8))
                if n > 1 << 20:
                    raise ValueError("not a GGUF string")
                return f.read(n)
            magic, _, n_tensors, n_kv = struct.unpack("<IIQQ", f.read(24))
            if magic != GGUF_MAGIC:
                return "", 0
            for _ in range(min(n_kv, 32)):
                key = blob()
                kind, size = GGUF_META[struct.unpack("<I", f.read(4))[0]]
                if kind == "string":
                    val = blob()
                    if key == b"general.architecture":
                        return val.decode("utf-8", "replace"), n_tensors
                elif kind == "array":
                    _, esize = GGUF_META[struct.unpack("<I", f.read(4))[0]]
                    (count,) = struct.unpack("<Q", f.read(8))
                    if esize is None:                  # an array of strings (the vocabulary) - the key was not early
                        break
                    f.seek(esize * count, 1)
                else:
                    f.seek(size, 1)
    except (OSError, ValueError, KeyError, struct.error):
        pass
    return "", 0


def glm_files(d: Path) -> list:
    """The GLM-5.3-Flash models in the folder `d`, as their first files (-00001-of- for a split one).  A single file
    without tensors (a vocabulary, like the vision folder's) is no model; unsloth's split models keep only metadata
    in their first file."""
    try:
        files = sorted(d.glob("*.gguf"))
    except OSError:
        return []
    found = []
    for f in files:
        m = SHARD_RE.search(f.name)
        if m and int(m.group(1)) != 1:
            continue
        arch, n = gguf_head(f)
        if arch in glm_archs() and (n > 0 or m):
            found.append(f)
    return found


def local_first(p: Path) -> tuple:
    """(the model's first .gguf file, None) from one of its files or the folder that holds it, or (None, why not)."""
    p = p.expanduser().resolve()
    if p.is_file():
        first = shard_set(p)[0]
        arch = gguf_head(first)[0] if first.exists() else ""
        if first.exists() and arch not in glm_archs():
            return None, f"{first.name} is " + (f"a {arch!r} model, not GLM-5.3-Flash ({GLM_ARCHS[0]})" if arch else
                                                "not a GGUF file")
        return first, None
    found = glm_files(p) if p.is_dir() else []
    if not found:
        return None, (f"no GLM-5.3-Flash GGUF file in {p}" if p.is_dir() else f"{p} does not exist")
    if len(found) > 1:
        return None, f"{p} holds {len(found)} GLM-5.3-Flash models: name the file (" + \
                     ", ".join(f.name for f in found) + ")"
    return found[0], None


def incomplete(path: Path):
    """Why this GGUF file is not whole, or None when it is: as long as its own tensor directory says (the same test
    as Strata's check_shards, without stopping).  Reads the header only."""
    if not path.exists():
        return "missing"
    from gguf_reader import GGUFFile
    try:
        g = GGUFFile(path)
    except Exception as e:                             # noqa: BLE001 - a partial header raises anything
        return f"not a whole GGUF file ({e})"
    # a shard of metadata alone (unsloth's first: the tokenizer, no tensors) is whole at its header's end - its writer
    # leaves out the alignment padding no data follows (#83), and the engine reads it so
    need = (g.data_start + max(t.offset + (t.expected_bytes() or 0) for t in g.tensors) if g.tensors
            else g.header_end)
    have = path.stat().st_size
    return None if have >= need else f"short: {have:,} of {need:,} bytes"


def file_incomplete(m: dict, path: Path):
    """Why a download's file is not whole, or None when it is: a GGUF by its own tensor directory (incomplete), an
    EXL3 model's file by its published size."""
    if "files" not in m:
        return incomplete(path)
    if not path.exists():
        return "missing"
    have, want = path.stat().st_size, m["files"][hf_name(m, path)]
    return None if have == want else f"{have:,} of {want:,} bytes"


def quant_of(first: Path) -> str:
    for q, mm in MODELS.items():                       # a Maya download, under its name on Hugging Face or its old one
        if "files" in mm:                              # (an EXL3 download is no .gguf file)
            continue
        if first.name in [names[0] for names in [shard_names(mm)] + old_names(mm)]:
            return q
    m = re.match(r"GLM-5\.3-Flash-(.+?)(-\d{5}-of-\d{5})?\.gguf$", first.name, re.I)
    if m is None:                                      # GLM-5.3 (glm-dsa): "GLM-5.3-UD-IQ2_M", its own config name
        d = re.match(r"(GLM-5\.3-(?!Flash-).+?)(-\d{5}-of-\d{5})?\.gguf$", first.name, re.I)
        if d and gguf_head(first)[0] == G53.ARCH:
            return d.group(1)
    return m.group(1) if m else first.parent.name


def sha256_ok(path: Path, want) -> bool:
    """`want`: the published sha256, or a list of them - a file published again with only its header changed (v1.0.28:
    the architecture spelled as llama.cpp spells it) keeps the earlier download valid."""
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(64 << 20), b""):
            h.update(b)
    return h.hexdigest() in [w.lower() for w in ([want] if isinstance(want, str) else want)]


VERIFIED = ".maya-verified.json"                   # in a model folder: the files whose sha256 matched, by size and time


def sha256_checked(path: Path, want) -> bool:
    """sha256_ok, read once: a file that matched is remembered (VERIFIED) with its size and modification time, and not
    read again until either changes."""
    book = read_json(path.parent / VERIFIED)
    st = path.stat()
    seen = book.get(path.name) or {}
    if seen.get("size") == st.st_size and seen.get("mtime_ns") == st.st_mtime_ns:
        return True
    say(f"  checking {path.name}'s sha256 ...")
    if not sha256_ok(path, want):
        return False
    book[path.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
    try:
        (path.parent / VERIFIED).write_text(json.dumps(book, indent=1), encoding="utf-8")
    except OSError:
        pass
    return True


def import_catalog(models: Path) -> list:
    """Every published model file Maya's setup would put under the models folder, for --import-model: {"label": the
    model's folder, "rel": the file's place, "size": its published size (None for a GGUF the table lists by sha256
    only), "sha": the accepted sha256 values}, with the layout the setup downloads into."""
    import contextlib
    import io
    cat = []

    def add(label: str, base: str, names: dict, sizes: dict | None = None):
        for name, sha in names.items():
            cat.append({"label": label, "rel": f"{base}/{name}", "size": (sizes or {}).get(name),
                        "sha": [x.lower() for x in ([sha] if isinstance(sha, str) else sha)]})

    def entry(label: str, base: str, m: dict):
        if m.get("files"):
            add(label, base, m["sha256"], m["files"])
        else:                                          # (a GLM GGUF: shards, the sha256 per name)
            add(label, base, {n: m["sha256"][n] for n in shard_names(m) if n in m["sha256"]})

    for quant, m in MODELS.items():
        entry(quant, quant, m)
        v = m.get("vision")
        if v:
            add(quant, f"{quant}/{v['folder']}", v["sha256"])
    cat.append({"label": "GLM-5.3-Flash-DFlash2", "rel": f"GLM-5.3-Flash-DFlash2/{DFLASH['file']}",
                "size": DFLASH["bytes"], "sha": [DFLASH["sha256"].lower()]})
    quiet = contextlib.redirect_stdout(io.StringIO())

    def tolerant(make, *args):                         # (a manifest that is not published yet: fail() exits)
        try:
            with quiet:
                return make(*args)
        except (SystemExit, OSError, KeyError, ValueError):
            return None
    nsa = argparse.Namespace(model_dir=None)
    for quant in DS4_QUANTS:
        m = tolerant(ds4_gguf_model, quant)
        if m and m.get("files"):
            d = ds4_dir(nsa, models, {}, quant)
            entry(d.relative_to(models).as_posix(), d.relative_to(models).as_posix(), m)
    for spec in DS4_EXL3.values():
        files = (read_json(spec["manifest"]).get("files") or {})
        base = spec["dir"]
        if files:
            add(base, base, {k: v[1] for k, v in files.items()}, {k: v[0] for k, v in files.items()})
    vision = tolerant(ds4_vision_model)
    if vision:
        entry(f"{DS4}-GGUF/vision", f"{DS4}-GGUF/vision", vision)
    mm = tolerant(mm3_models)
    if mm:
        entry(MM3_DIR, MM3_DIR, mm[0])
        entry(f"{MM3_DIR}/dspark", f"{MM3_DIR}/{mm[1]['folder']}", mm[1])
    for quant, shown in MM3_QUANTS.items():
        m = tolerant(mm3_gguf_entry, quant) if quant != "exl3" else None
        if m and m.get("files") and all(m["sha256"].values()):
            entry(shown, shown, m)
    for quant in HY4_QUANTS:                           # (the sizes that are uploaded: the trunk and the n-gram table)
        m = tolerant(hy4_entry, quant) if hy4_published(quant) else None
        if m:
            entry(m["dir"], m["dir"], m)
    q = tolerant(qwen_entry)
    if q:
        entry(QWEN, QWEN, q)
    q = tolerant(qwen_entry, QWEN_UD)
    if q:
        entry(QWEN_UD_DIR, QWEN_UD_DIR, q)
    h = tolerant(hy4_entry) if hy4_published(HY4_Q4) else None   # (Tencent's own Q4_K_M: its trunk, the same table)
    if h:
        entry(h["dir"], h["dir"], h)
    return cat


def import_model(a, folder: str) -> int:
    """--import-model <folder>: find the published model files in a folder by their size and sha256 (whatever the
    downloader named them, in whatever subfolders, also a Hugging Face cache's blobs/snapshots), and move them (same
    drive) or copy them (another drive: --copy) into the models folder's layout; then print what was found, what
    is missing and what did not match.  Nothing that did not match is touched; nothing is deleted."""
    src = Path(folder).expanduser().resolve()
    if not src.is_dir():
        fail(f"{folder} is not a folder", f"{ME} --import-model <the folder with the downloaded files>")
    have = configs()
    inst = (read_json(have[0]).get("installer") or {}) if have else {}
    models = (Path(a.models_dir).expanduser().resolve() if a.models_dir else
              Path(inst["models_dir"]) if inst.get("models_dir") else
              Path(inst["data_dir"]) / "models" if inst.get("data_dir") else ROOT.parent / "Maya-data" / "models")
    cat = import_catalog(models)
    say(f"  Looking in {src} for Maya's model files (by size and sha256, not by name); they go to {models}")
    got = MI.find(src, cat, progress=say)
    placed, shared = MI.choose(got["found"])
    if not placed and not shared:
        warn("none of the files there is one of Maya's published model files")
    for path, _, size, hits in placed:
        say(f"  found  {path.name}  ({size / 1e9:.2f} GB)  ->  {hits[0]['rel']}" +
            (f" (+{len(hits) - 1} more places)" if len(hits) > 1 else ""))
    for path, labels in shared:
        say(f"  shared {path.name}: used by several models ({', '.join(labels[:3])}, ...) and none of them is in this "
            "folder - left where it is (the setup downloads it with the model)")
    have_lines, missing_lines = MI.report_lines(cat, placed, models)
    for label, absent in missing_lines:
        if absent:
            say(f"  missing for {label}: " + ", ".join(absent[:6]) + (f" and {len(absent) - 6} more" if len(absent) > 6 else ""))
        else:
            say(f"  {label}: every file is here")
    skipped = [(p, s) for p, s in got["unmatched"] if s >= (1 << 20)]
    if skipped:
        say(f"  {len(skipped)} file(s) there did not match any published file and are left alone"
            + (" (a download that is not finished, another version, or another model):" if len(skipped) <= 20 else
               " - the first 20:"))
        for p, s in skipped[:20]:
            say(f"      {p}  ({s / 1e9:.2f} GB)" if s >= 1e8 else f"      {p}  ({s / 1e6:.1f} MB)")
    if not placed:
        return 1
    anchor = next(p for p in [models, *models.parents] if p.exists())
    apart = [p for p, r, _, _ in placed if not MI.same_drive(r, anchor) or os.path.abspath(p) != str(r)]
    copy = bool(a.copy)
    if apart and not copy:
        say()
        say(f"  {len(apart)} file(s) are on another drive than {models} (or are links into a cache): they can only be "
            "copied - the original stays where it is, so the space is used twice.")
        copy = ask("  Copy them? (y/n)", ["y", "n"], "n", a.yes) == "y"
    elif not (a.yes or a.copy):
        say()
        if ask(f"  Move {len(placed)} file(s) into {models}? (y/n)", ["y", "n"], "y", False) != "y":
            say("  Nothing was moved.")
            return 0
    moved = copied = 0
    skipped_now = []
    for path, real, size, hits in placed:
        r = MI.place(path, real, size, hits, models, copy, say)
        moved, copied = moved + r["moved"], copied + r["copied"]
        skipped_now += r["skipped"]
    for line in skipped_now:
        warn(line)
    ok(f"{moved} file(s) moved, {copied} copied into {models}; they are recorded as verified (no second check)")
    say(f"  Now run {ME} --setup and pick the model (or {ME} --setup --model <name>): the files are used where they are.")
    return 0


def file_url(m: dict, name: str) -> str:
    """A download's file on Hugging Face: from the model's repo, or the one its "sources" names for it (DeepSeek's
    Engram tables: DeepSeek's base repo).  "hub_names": a file whose name on the Hub is not its name here (Hy4-Flash's
    8 GB-card trunk: .gguf_file there)."""
    repo, revision = (m.get("sources") or {}).get(name) or (m["repo"], m["revision"])
    folder = (m.get("source_folders") or {}).get(name, m["folder"]) if name in (m.get("sources") or {}) else m["folder"]
    return hf_url(repo, revision, folder, (m.get("hub_names") or {}).get(name, name))


def hf_token() -> str | None:
    """The user's Hugging Face token, for a gated repo's files (download_gguf.json's "gated"): HF_TOKEN (or
    HUGGING_FACE_HUB_TOKEN), else the one `hf auth login` saved (<HF_HOME, or ~/.cache/huggingface>/token)."""
    for k in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        if os.environ.get(k, "").strip():
            return os.environ[k].strip()
    try:
        t = (Path(os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface") / "token").read_text(
            encoding="utf-8").strip()
    except OSError:
        return None
    return t or None


# curl's part of every model download: resume a file where it stopped, and give up on a connection that stalls (under
# 1 KB/s for a minute) so it is run again instead of hanging
CURL_RESUME = ["-C", "-", "--speed-limit", "1024", "--speed-time", "60", "--connect-timeout", "30"]
DL_JOBS = 4         # files fetched at once (Hugging Face sends several streams faster than one)
DL_STUCK = 30       # runs of curl in a row that add nothing to a file before the setup gives up on it


def curl_drops_auth(curl: str) -> bool:
    """Whether this curl leaves a custom Authorization header out of a redirect to another host (7.58.0 and later -
    a gated repo's files redirect to its CDN, which must not get the token); False when its version cannot be read."""
    try:
        out = subprocess.run([curl, "--version"], capture_output=True, text=True, timeout=20).stdout
        v = tuple(int(x) for x in re.match(r"curl (\d+)\.(\d+)", out).groups())
    except (OSError, subprocess.SubprocessError, AttributeError, ValueError):
        return False
    return v >= (7, 58)


def fetch_files(m: dict, shards: list, missing: list, urls: dict) -> None:
    """The missing files with curl, DL_JOBS at a time, each resumed where it stopped and run again whenever its
    connection drops or stalls - a download stopped half way (a dropped line, Ctrl+C, a restart) loses nothing: the
    next run continues every file from its size on disk.  One line says how far the model is.  A gated repo's ("gated"):
    the token in an Authorization header, read by curl from a file only the user can read (not on its command line)."""
    want = {s: (m.get("files") or {}).get(hf_name(m, s)) for s in shards}
    auth = None
    token = hf_token() if m.get("gated") else None
    queue, running = list(missing), {}
    try:   # (from here: whatever happens, the finally removes the token's file)
        if token:
            import tempfile
            auth = Path(tempfile.mkdtemp(prefix="maya-hf-")) / "header"   # (mkdtemp's folder: the user's alone)
            fd = os.open(auth, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(f"Authorization: Bearer {token}\n")
        total = sum(want.values()) if all(want.values()) else m.get("download_gb", 0) * 1e9
        size = lambda s: s.stat().st_size if s.exists() else 0   # noqa: E731
        stuck, after, start_size = {s: 0 for s in missing}, {s: 0.0 for s in missing}, {}
        t0, base, last = time.time(), sum(size(s) for s in shards), 0.0
        while queue or running:
            now = time.time()
            for s in [q for q in queue if after[q] <= now][:max(0, DL_JOBS - len(running))]:
                queue.remove(s)
                start_size[s] = size(s)
                cmd = ["curl", "-sS", "-L", "--fail"] + (["-H", f"@{auth}"] if auth else []) + CURL_RESUME + \
                    ["-o", str(s), urls[s]]
                running[s] = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                              stderr=subprocess.PIPE)
            time.sleep(1)
            for s, p in list(running.items()):
                rc = p.poll()
                if rc is None:
                    continue
                del running[s]
                err = (p.communicate()[1] or b"").decode(errors="replace").strip().splitlines()
                have = size(s)
                if want[s] is not None and have > want[s]:   # (more than the file: its tail is wrong - start it over)
                    s.unlink()
                    have = 0
                if want[s] is None and rc == 0 or want[s] is not None and have == want[s]:
                    continue
                stuck[s] = 0 if have > start_size[s] else stuck[s] + 1
                if stuck[s] >= DL_STUCK:
                    S.progress()
                    fail(f"the download of {s.name} keeps stopping without progress "
                         f"({err[-1] if err else f'curl exit code {rc}'})",
                         f"check the connection, then run {ME} --download-model again: it continues where it stopped")
                after[s] = time.time() + min(60, 5 * stuck[s])   # (again at once after progress, slower when stuck)
                queue.append(s)
            if time.time() - last > 2 or not (queue or running):
                last = time.time()
                have = sum(size(s) for s in shards)
                rate = (have - base) / max(1e-3, last - t0)
                eta = (total - have) / rate if rate > 0 and total > have else 0
                left = f" · {int(eta // 3600)} h {int(eta % 3600 // 60):02d} min left" if eta else ""
                share = f" ({100 * have / total:.0f}%)" if total else ""
                S.progress(f"{have / 1e9:.1f} / {total / 1e9:.1f} GB{share} · {rate / 1e6:.0f} MB/s{left} · "
                         f"{len(running)} files at once", have / total if total else None)
    finally:
        for p in running.values():   # (stopped half way: no curl keeps writing behind the setup's back)
            p.terminate()
        if auth is not None:
            auth.unlink(missing_ok=True)
            try:
                auth.parent.rmdir()
            except OSError:
                pass
        S.progress()


def disk_of(path: Path):
    """An id of the disk path is on (its st_dev)."""
    return os.stat(path).st_dev


def linked_downloads(m: dict, d: Path, missing: list) -> dict:
    """The part of a download that lands on another disk than d's: a shard that is a link (a symlink, on Windows too)
    to a file on that disk is written there, not into d.  {the folder it lands in: (GB free on that disk, GB still to
    download into it)}; only for the GGUFs whose shard sizes the manifest lists (m["files"])."""
    out: dict = {}
    if "files" not in m:
        return out
    try:
        home = disk_of(existing(d))
    except OSError:
        return out
    for s in missing:
        if not s.is_symlink():
            continue
        try:
            target = Path(os.path.realpath(s))
            where = existing(target.parent)
            if disk_of(where) == home:
                continue
            have = target.stat().st_size if target.exists() else 0
            need = max(0, m["files"].get(hf_name(m, s), 0) - have) / 1e9
            free = shutil.disk_usage(where).free / 1e9
        except (OSError, KeyError):
            continue
        prev = out.get(str(where), (free, 0.0))
        out[str(where)] = (free, prev[1] + need)
    return out


def offer_download(a, quant: str, d: Path, shards: list, m: dict | None = None) -> bool:
    """Shows what would be downloaded - the source, the size, the exact commands - and downloads only after a yes
    (or --download-model).  False: nothing was downloaded."""
    m = m or glm_downloads()[quant]
    missing = [s for s in shards if file_incomplete(m, s)]
    urls = {s: file_url(m, hf_name(m, s)) for s in shards}
    on_disk = sum(s.stat().st_size for s in shards if s.exists()) / 1e9
    remaining = max(0.0, m["download_gb"] - on_disk)
    free = shutil.disk_usage(existing(d)).free / 1e9
    curl = shutil.which("curl")
    if curl and m.get("gated") and not curl_drops_auth(curl):   # (an old curl would send the token on to the CDN)
        curl = None
    cmds = [["mkdir"] + ([] if WIN else ["-p"]) + [str(d)]] + \
        [["curl", "-L", "--fail", "--retry", "5"] + CURL_RESUME + ["-o", str(s), urls[s]] for s in missing]
    say(f"  {quant}: {m['about']}.")
    say(f"  Source: https://huggingface.co/{m['repo']}" + (f" (folder {m['folder']}/)" if m["folder"] else "") +
        (f" (branch {m['branch']})" if m.get("branch") else "") + "; the files' own license applies.")
    for repo in dict.fromkeys(r for r, _ in (m.get("sources") or {}).values()):
        names = [n for n, (r, _) in m["sources"].items() if r == repo]
        say(f"  And from https://huggingface.co/{repo}: " + ", ".join(names))
    token = hf_token() if m.get("gated") else None
    if m.get("gated"):
        say(f"  The repo is gated: accept its terms at https://huggingface.co/{m['repo']} with your account; the download "
            "sends your Hugging Face token (" + ("found" if token else "none found here") + ").")
    say(f"  The model is {m['download_gb']:.1f} GB in {len(shards)} file{'s' if len(shards) != 1 else ''}; "
        f"{len(missing)} still to download, about "
        f"{remaining:.0f} GB, into")
    say(f"    {d}   ({free:.0f} GB free there)")
    also = float(m.get("also_gb") or 0)   # (what the setup downloads after it, beside it: the space for both)
    if also:
        say(f"  Then {m.get('also_what', 'more files')}: {also:.0f} GB more.")
    linked = linked_downloads(m, d, missing)   # (a shard that is a link to another disk lands on that disk)
    here = remaining - sum(need for _, need in linked.values())
    for where, (gb_free, need) in linked.items():
        say(f"    {need:.0f} GB of it, linked from there, into {where}   ({gb_free:.0f} GB free)")
    if free < here + also + 3:
        fail(f"not enough free space in {d}: about {here + also + 3:.0f} GB are needed (the model" +
             (f", {m.get('also_what', 'more files')}" if also else "") + " + its pack)",
             f"free some space, or put the model on another drive: {ME} --setup --models-dir " +
             (r"D:\Maya-models" if WIN else "/path/on/nvme"))
    for where, (gb_free, need) in linked.items():
        if gb_free < need + 3:
            fail(f"not enough free space in {where}: about {need + 3:.0f} GB of the model are linked into it from {d}",
                 "free some space there, or link the files to another drive")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: the engine reads experts from these files while it answers - use an "
             "NVMe SSD (--models-dir)")
    say("  The exact commands (resumable: running them again continues an interrupted download):")
    bearer = '-H "Authorization: Bearer %HF_TOKEN%" ' if WIN else '-H "Authorization: Bearer $HF_TOKEN" '
    for c in cmds:   # (a gated repo's: the token as the shell expands it, never the token itself)
        line = shell_join(c)
        say("    " + ("curl " + bearer + line[5:] if m.get("gated") and c[0] == "curl" else line))
    if "files" in m:                                   # (--gguf-dir takes GGUF files only)
        say(f"  You can also run them yourself (or download the files any other way into that folder), then run {ME}")
        say("  again.")
    else:
        say("  You can also run them yourself (or download the files any other way into that folder, or pass")
        say(f"  --gguf-dir <folder with the files>), then run {ME} again.")
    say(f"  Files you downloaded with another program, under any names: {ME} --import-model <their folder> finds them")
    say("  by size and sha256 and moves them into place.")
    if not curl:
        warn("curl is not installed" + ("" if WIN else " (sudo apt-get install -y curl)") + ": if you say yes, the "
             "same URLs are downloaded with Python instead (also resumable)")
    if m.get("gated") and not token and missing:
        fail(f"{m['repo']} is gated and no Hugging Face token was found",
             f"accept its terms at https://huggingface.co/{m['repo']}, then set HF_TOKEN (a read token from "
             f"https://huggingface.co/settings/tokens) or run `hf auth login`, and run {ME} again")
    if not a.download_model and ask(f"  Download about {remaining:.0f} GB now with these commands? (y/n)",
                                    ["y", "n"], "n", False) != "y":
        say()
        say(f"Nothing was downloaded. When the files are in place, run {ME} again "
            f"({ME} --download-model downloads them without asking).")
        return False
    d.mkdir(parents=True, exist_ok=True)
    if curl:
        fetch_files(m, shards, missing, urls)
    for s in missing:
        if not curl:
            S.download(urls[s], s, headers={"Authorization": f"Bearer {token}"} if token else None)
        why = file_incomplete(m, s)
        if why:
            fail(f"{s.name} after the download: {why}", f"delete it and run {ME} --download-model again")
        want = m["sha256"].get(hf_name(m, s))
        if want and not sha256_checked(s, want):
            s.unlink(missing_ok=True)                  # complete by size but damaged: kept, it would pass as done
            fail(f"{s.name}: the sha256 does not match the published one (the file was removed)",
                 f"run {ME} --download-model again")
        ok(f"{s.name} downloaded")
    return True


def model_step(a, models: Path, choice: tuple):
    """(model folder, shards, quant name), or None when the files are not there and were not downloaded."""
    kind, what = choice
    step(5, f"the model ({G53.SHOWN if kind == 'download' and what in G53_MODELS else 'GLM-5.3-Flash'})")
    if kind == "exl3":                                 # an EXL3 folder of your own (--model-dir): complete by its own index
        d, quant = what, what.name
        shards = exl3_folder_files(d, ("config.json", "tokenizer.json", "tokenizer_config.json"))
        warn(f"{quant}: experimental - only {', '.join(MODELS)} has been measured with Maya (README.md)")
    elif kind == "local":
        shards, quant, d = shard_set(what), quant_of(what), what.parent
        for s in shards:
            why = incomplete(s)
            if why:
                fail(f"{s.name}: {why}", f"finish copying or downloading it, then run {ME} again")
        if quant not in glm_downloads():
            warn(f"{quant}: experimental - only {', '.join(MODELS)} has been measured with Maya (README.md)")
    else:
        quant = what
        m = glm_downloads()[quant]
        d = download_dir(models, quant)
        shards = local_shards(m, d)
        if any(file_incomplete(m, s) for s in shards) and not offer_download(a, quant, d, shards):
            return None
    if kind == "exl3" or (is_exl3(quant) and kind != "local"):
        conf = read_json(d / "config.json")
        method = (conf.get("quantization_config") or {}).get("quant_method")
        if conf.get("model_type") != "glm5_next" or method != "exl3":
            fail(f"{d / 'config.json'}: a {conf.get('model_type')!r} model quantized as {method!r}, not GLM-5.3-Flash "
                 "in EXL3 (glm5_next, exl3)")
    else:
        from gguf_reader import GGUFFile
        arch = str(GGUFFile(shards[0]).metadata.get("general.architecture", ""))
        if arch not in glm_archs():
            fail(f"{shards[0].name} is a {arch!r} model, not GLM-5.3-Flash (glm5-next)")
    gb = sum(s.stat().st_size for s in shards) / 1e9
    ok(f"{quant}: {len(shards)} file(s), {gb:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: expect slow answers (the engine reads experts from it) - an NVMe SSD "
             "is strongly recommended")
    return d, shards, quant


def kind_is_glm_dsa(choice: tuple, shards: list) -> bool:
    """The model set up is GLM-5.3 (glm-dsa): GGUF files of your own, or one of its downloads (G53_MODELS)."""
    if choice and choice[0] == "download" and choice[1] not in G53_MODELS:
        return False                                   # (GLM-5.3-Flash's downloads: no header read)
    return bool(choice and choice[0] in ("local", "download") and shards and shards[0].suffix == ".gguf"
                and gguf_head(shards[0])[0] == G53.ARCH)


def drafter_step(models: Path) -> Path:
    """The DFlash2 drafter's file in the models folder (GLM-5.3-Flash-DFlash2/): downloaded and its sha256 checked
    unless it is there already."""
    dst = models / "GLM-5.3-Flash-DFlash2" / DFLASH["file"]
    size = dst.stat().st_size if dst.exists() else 0
    if size == DFLASH["bytes"] and sha256_ok(dst, DFLASH["sha256"]):
        ok(f"the DFlash2 drafter: {dst}")
        return dst
    if size >= DFLASH["bytes"]:                    # complete but not this file: a fresh download, not a resumed one
        dst.unlink()
    say(f"  the DFlash2 drafter: {DFLASH['bytes'] / 1e9:.1f} GB from Hugging Face ({DFLASH['repo']}; license "
        f"{DFLASH['license']}: non-commercial use only)")
    fetch(hf_url(DFLASH["repo"], DFLASH["revision"], "", DFLASH["file"]), dst, DFLASH["sha256"])
    ok(f"the DFlash2 drafter: {dst}")
    return dst


# ------------------------------------------------------------------------------------------------ 6. the pack
def pack_source(pack: Path):
    """The first .gguf file a pack indexes (native_experts.txt's header names it), or None (no pack there)."""
    try:
        with open(pack / "native_experts.txt", encoding="utf-8") as f:
            m = re.search(r"absolute offsets in ([^,)]+)", f.readline())
    except OSError:
        return None
    return m.group(1) if m else None


PACK_SOURCES = "sources.json"                      # in a pack: its model files' sizes and times when it was built


def pack_stale(pack: Path, shards: list):
    """Why a pack no longer matches its model files, or None.  The pack holds where every tensor sits in them: a file
    that changed after the pack was built leaves those places wrong (#155: a pack built while a download manager was
    still filling files it had created at full size indexed regions with no data yet, and the engine answered garbage,
    fast).  A pack from before the record (PACK_SOURCES): a model file written after the pack's index counts."""
    rec = read_json(pack / PACK_SOURCES).get("files")
    if rec:
        old = {r.get("name"): r for r in rec}
        for s in shards:
            r, st = old.get(s.name) or {}, s.stat()
            if r.get("size") != st.st_size or r.get("mtime_ns") != st.st_mtime_ns:
                return f"{s.name} changed after the pack was built"
        return None
    try:
        built = (pack / "native_experts.txt").stat().st_mtime
    except OSError:
        return None
    late = [s.name for s in shards if s.stat().st_mtime > built + 60]
    return f"{late[0]} was written after the pack was built" if late else None


def headwise_wanted() -> bool:
    """GLM-5.3 packs keep the GGUF's Q8_0 attn_k_b / attn_v_b native (tools/strata/glm_dsa_pack.py's default);
    MAYA_PACK_NATIVE32=0 upcasts them to BF16 as before."""
    return str(N.lookup(os.environ, "MAYA_PACK_NATIVE32", "")).strip() != "0"


def headwise_stale(pack: Path):
    """Why a GLM-5.3 pack is built again for its attention weights, or None: one built before Q8_0 attn_k_b /
    attn_v_b stayed native (it upcast them to BF16), or the other way round with MAYA_PACK_NATIVE32=0."""
    if not G53.pack_is_glm_dsa(pack):
        return None
    have, want = G53.pack_headwise_q8(pack), headwise_wanted()
    if have is None or have == want:
        return None
    return ("it keeps the attention's Q8_0 weights native now (the model's own weights, about 1 GB less to read a "
            "token)" if want else "MAYA_PACK_NATIVE32=0 upcasts the attention's Q8_0 weights to BF16")


def stamp_pack(pack: Path, shards: list) -> None:
    """The model files' sizes and times into the pack (PACK_SOURCES), for pack_stale."""
    files = [{"name": s.name, "size": s.stat().st_size, "mtime_ns": s.stat().st_mtime_ns} for s in shards]
    try:
        (pack / PACK_SOURCES).write_text(json.dumps({"files": files}, indent=1), encoding="utf-8")
    except OSError:
        pass


def pack_index_files(pack: Path) -> list | None:
    """The model files a DS4 pack's tensors.txt reads (its 'file N <path>' lines, relative to the model folder), or
    None when it names none (an older pack: the caller globs the model folder as before).  A quant that takes its
    Engram tables (or drafter, vision) from a set beside the model folder names them THERE - the pack is built from
    those files, and a same-named file an earlier layout left in the model folder is not one of them."""
    try:
        lines = (pack / "tensors.txt").read_text(encoding="utf-8").splitlines()
    except OSError:
        return None
    out = []
    for ln in lines:
        m = re.match(r"file \d+ (.+)", ln)
        if m:
            p = Path(m.group(1).strip())
            out.append(p if p.is_absolute() else pack.parent / p)
    return out or None


def check_sha256(shards: list, quant: str | None) -> None:
    """The files a pack is built from (or Metal reads): a Maya download's against its published sha256 (once a file - a
    download is checked as it lands), so a file still being written or damaged stops here, not as garbage answers (#155)."""
    m = glm_downloads().get(quant or "") or {}
    for s in shards:
        want = (m.get("sha256") or {}).get(hf_name(m, s)) if m else None
        if want and not sha256_checked(s, want):
            fail(f"{s.name}: the sha256 does not match the published one - still being downloaded, or damaged",
                 f"let the download finish, or delete the file and run {ME} --download-model")


def metal_tokenizer_step(a, d: Path, shards: list, quant: str) -> Path:
    """Step 6 on Metal: the engine reads the GGUF files itself (no pack), the server needs the tokenizer and the chat
    template, exported from the first shard into <model folder>/metal/tokenizer (tools/strata_tokenizer.py)."""
    step(6, "the tokenizer (Metal reads the GGUF files itself: no pack)")
    check_sha256(shards, quant)
    out = d / "metal"
    vocab = out / "tokenizer" / "vocab.json"
    if not a.repack and vocab.exists() and vocab.stat().st_mtime >= shards[0].stat().st_mtime:
        ok(f"tokenizer already exported: {vocab.parent}")
        return out
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the tokenizer is written next to the GGUF files", "pass --gguf-dir with a writable folder")
    r = run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]), "--out", str(out)],
            check=False)
    if r.returncode != 0 or not vocab.exists():
        fail(f"the tokenizer could not be exported from {shards[0].name}", "the reason is in the messages above (the "
             "first file of a GLM-5.3-Flash GGUF carries its tokenizer)")
    ok(f"tokenizer: {vocab.parent}")
    return out


def pack_step(a, d: Path, shards: list, llama: Path, quant: str | None = None) -> Path:
    step(6, "the pack (the engine's index of the model files)")
    if quant is not None and is_exl3(quant):
        return exl3_pack_step(a, d)
    if shards and shards[0].suffix == ".safetensors":  # an EXL3 folder of your own (--model-dir)
        pack = exl3_pack_step(a, d)
        body, label = own_template(d, ROOT / "serve" / "glm5_chat_template.jinja")
        put_template(pack, body)
        ok(f"chat template: {label.replace('the base model', 'GLM-5')}")
        return pack
    # the engine finds the GGUF files at <pack>/.. (src/core/glm_model.cu, load_pack): the pack lives in their folder,
    # in pack/ - or pack-<model>/ when pack/ indexes another model of the same folder
    pack = d / "pack"
    if pack_source(pack) not in (None, shards[0].name):
        pack = d / f"pack-{quant_of(shards[0])}".lower()
    if not a.repack and pack_source(pack) == shards[0].name and all((pack / f).exists() for f in PACK_FILES):
        why = pack_stale(pack, shards) or headwise_stale(pack)
        if why is None:
            if not (pack / PACK_SOURCES).exists():
                stamp_pack(pack, shards)
            ok(f"pack already built: {pack}")
            base_template(pack)
            return pack
        warn(f"the pack is built again: {why}")
    check_sha256(shards, quant)
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack is written next to the GGUF files (the engine finds them at <pack>/..)",
             "link the .gguf files into a writable folder (" + ("mklink /H DIR\\<file> <file>, for each file" if WIN
                                                                else "mkdir -p DIR && ln -s /path/to/*.gguf DIR/") +
             ") and pass "
             "--gguf-dir DIR")
    say("  Writing the index of every tensor, the small float weights (about 1 GB) and the tokenizer; the experts")
    say("  stay in the GGUF files (a few minutes) ...")
    env = dict(os.environ, MAYA_GGUF_PY=str(llama / "gguf-py"))
    # --compat-bf16: the GLM pack stores the projections its kernels read as BF16 (router, indexer, absorbed MLA,
    # KDA gates) in BF16 even where the GGUF quantized them or keeps them F32 (tools/iq_pack.py GLM5NEXT_BF16)
    run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
         "--compat-bf16"], env=env)
    missing = [f for f in PACK_FILES if not (pack / f).exists()]
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    stamp_pack(pack, shards)
    ok(f"pack: {pack}")
    base_template(pack)
    return pack


def base_template(pack: Path) -> None:
    """A GLM pack whose GGUF carried no chat template gets GLM-5's: the server would take its generic one otherwise (a
    GGUF with a template of its own keeps it)."""
    base = ROOT / "serve" / "glm5_chat_template.jinja"
    if G53.pack_is_glm_dsa(pack):                      # GLM-5.3: Z.ai's own template, never Flash's (image slots)
        base = G53.TEMPLATE
    if not (pack / "tokenizer" / "chat_template.jinja").is_file() and base.is_file():
        put_template(pack, base.read_bytes())
        ok("chat template: GLM-5's (the GGUF carries none)")


EXL3_GGUF = "maya-exl3.gguf"                       # what tools/exl3_pack.py writes beside an EXL3 model's files


# ---- an EXL3 folder of the user's own (--model-dir): GLM-5.3-Flash's or MiniMax-M3.1-Flash's, any bits, checked from its
# own index and config, never from the pinned download lists
def exl3_folder_family(path) -> str | None:
    """"glm" or "minimax" for a folder holding an EXL3 quant of GLM-5.3-Flash or MiniMax-M3.1-Flash (its config.json:
    model_type and quantization_config), else None."""
    try:
        conf = json.loads((Path(path).expanduser() / "config.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        return None
    if not isinstance(conf, dict) or (conf.get("quantization_config") or {}).get("quant_method") != "exl3":
        return None
    return {"glm5_next": "glm", "minimax_m3_vl": "minimax"}.get(conf.get("model_type"))


def safetensors_short(path: Path) -> str | None:
    """Why a .safetensors file is not whole, or None: the last tensor its header names must end inside the file."""
    if not path.is_file():
        return "missing"
    try:
        with path.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n)) if n < 1 << 30 else None
        have = path.stat().st_size
    except (OSError, struct.error, ValueError):
        head = None
    if not isinstance(head, dict):
        return "not a whole .safetensors file"
    need = 8 + n + max((v["data_offsets"][1] for k, v in head.items() if k != "__metadata__"), default=0)
    return None if have >= need else f"short: {have:,} of {need:,} bytes"


def exl3_folder_files(d: Path, needs: tuple = ("config.json",)) -> list:
    """The .safetensors files of an EXL3 folder of the user's own - the ones its model.safetensors.index.json names and any
    other beside them - each checked whole by its own header, and the small files the pack needs; the setup stops with
    what is missing."""
    index = d / "model.safetensors.index.json"
    try:
        names = sorted(set(json.loads(index.read_text(encoding="utf-8-sig"))["weight_map"].values()))
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        fail(f"{index} is missing or is no safetensors index", f"{d} is no whole EXL3 download: fetch the repo's files "
                                                               "into it again")
    files = [d / n for n in names] + [p for p in sorted(d.glob("*.safetensors")) if p.name not in names]
    for f in files:
        if (why := safetensors_short(f)):
            fail(f"{f}: {why}", f"finish copying or downloading it, then run {ME} again")
    for n in needs:
        if not (d / n).is_file():
            fail(f"{d / n} is missing", f"fetch the repo's {n} into {d}, then run {ME} again")
    return files


def put_template(pack: Path, text: bytes | None) -> bool:
    """`text` as the chat template the server reads from the pack's tokenizer folder (nothing to put: None).  True when
    the file changed."""
    dst = pack / "tokenizer" / "chat_template.jinja"
    if text is None or (dst.is_file() and dst.read_bytes() == text):
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(text)
    return True


def own_template(d: Path, base: Path | None = None) -> tuple:
    """(the chat template, what it is) of a model folder of the user's own: its chat_template.jinja, or the chat_template
    its tokenizer_config.json carries, when it ships one; else the base model's `base` (None: the pack's own copy of it)."""
    if (d / "chat_template.jinja").is_file():
        return (d / "chat_template.jinja").read_bytes(), "the folder's own (chat_template.jinja)"
    tpl = read_json(d / "tokenizer_config.json").get("chat_template") if (d / "tokenizer_config.json").is_file() else None
    if isinstance(tpl, str) and tpl.strip():
        return tpl.encode("utf-8"), "the folder's own (tokenizer_config.json)"
    return (base.read_bytes() if base is not None and base.is_file() else None), "the base model's (the folder ships none)"


def exl3_pack_step(a, d: Path) -> Path:
    """An EXL3 model's pack (tools/exl3_pack.py): the engine reads the EXL3 weights in place from the safetensors; the
    pack names them, and a small GGUF beside them holds the unquantized tensors and the tokenizer."""
    pack = d / "pack"
    if not a.repack and pack_source(pack) == EXL3_GGUF and (d / EXL3_GGUF).exists() and \
            all((pack / f).exists() for f in PACK_FILES + ("exl3.txt",)):
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack and {EXL3_GGUF} are written next to the model's files")
    say(f"  Writing {EXL3_GGUF} (the unquantized tensors and the tokenizer, about 2 GB), the index of every tensor,")
    say("  the small float weights (about 1 GB) and the tokenizer; the EXL3 weights stay in the safetensors (a few")
    say("  minutes) ...")
    run([sys.executable, str(ROOT / "tools" / "exl3_pack.py"), "--model", str(d), "--out", str(pack)])
    missing = [f for f in PACK_FILES + ("exl3.txt",) if not (pack / f).exists()] + \
        ([] if (d / EXL3_GGUF).exists() else [f"../{EXL3_GGUF}"])
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    ok(f"pack: {pack}")
    return pack


# ------------------------------------------------------------------------------------------------ 7. images
def fetch(url: str, dst: Path, want: str | None) -> None:
    """One file, resumable (curl -C -, else Python), then its sha256 when one is published."""
    dst.parent.mkdir(parents=True, exist_ok=True)
    if shutil.which("curl"):
        if run(["curl", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(dst), url], check=False).returncode != 0:
            fail(f"the download of {dst.name} stopped (the reason is above)", f"run {ME} --setup again: it "
                                                                              "continues where it stopped")
    else:
        S.download(url, dst)
    if want and not sha256_ok(dst, want):
        dst.unlink(missing_ok=True)
        fail(f"{dst.name}: the sha256 does not match the published one (the file was removed)",
             f"run {ME} --setup again to download it again")


# ------------------------------------------------------------------ refusal ablation: the model's direction file
# <Model>-refusal-direction.gguf in the refusal-directions/ folder of each model's own Maya repo (the repo's top before
# v1.1.60, still read; fitted by tools/ablate_fit.py, a few KB):
# downloaded with the model into <models>/refusal-directions/ and recorded as the config's "ablate" (loaded, off by
# default - the Models page's switch).  A repo that ships none yet, no connection, or a file that is not a direction
# for this model: the setup goes on without it and the switch stays off ("not available").
ABLATE_DIRECTIONS = {"glm": ("GLM-5.3-Flash", "peasantsmith/GLM-5.3-Flash-Maya-GGUF"),
                     "glm-dsa": ("GLM-5.3", "peasantsmith/GLM-5.3-Maya-GGUF"),
                     "deepseek": ("DeepSeek-V4.1-Flash", "peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF"),
                     "minimax": ("MiniMax-M3.1-Flash", "peasantsmith/MiniMax-M3.1-Flash-Maya-GGUF")}
ABLATE_DIR = "refusal-directions"
ABLATE_MAX_BYTES = 1 << 20


def ablate_direction_check(path: Path, model: str) -> str:
    """"" when path is a direction file for model (a control vector GGUF with one 1-D F32 direction.1 and, when it
    says, maya.ablate.model = model), else why not."""
    try:
        if path.stat().st_size > ABLATE_MAX_BYTES:
            return "too big for a direction file"
        import gguf_reader as GR
        g = GR.GGUFFile(path)
    except Exception as e:  # noqa: BLE001  (a broken or partial file)
        return f"not a GGUF ({e})"
    if g.metadata.get("general.architecture") != "controlvector":
        return "not a control vector"
    t = [x for x in g.tensors if x.name == "direction.1"]
    if len(t) != 1 or len(t[0].shape) != 1 or t[0].type_name != "F32":
        return "no one-dimensional F32 direction.1"
    if g.metadata.get("maya.ablate.model") not in (None, model):
        return f"fitted for {g.metadata.get('maya.ablate.model')}, not {model}"
    return ""


def fetch_ablate_direction(key: str, models: Path) -> Path | None:
    """The model's direction file from its own repo into <models>/refusal-directions/ (again each setup: a refit
    replaces it), or the copy from an earlier setup when the download fails; None when there is neither."""
    model, repo = ABLATE_DIRECTIONS[key]
    name = model + "-refusal-direction.gguf"
    dst = models / ABLATE_DIR / name
    part = dst.with_name(name + ".part")
    why = ""
    # the file sits in the repo's refusal-directions/ folder (at the repo's top the Hub took it for the model and showed
    # its few thousand values as the parameter count); a repo that still has it at the top is read there
    for url in (hf_url(repo, "main", ABLATE_DIR, name), hf_url(repo, "main", "", name)):
        got = _fetch_ablate_file(url, dst, part, model, repo, name)
        if got is True:
            return dst
        why = got
        if why != "the repo ships none yet":
            break
    if dst.is_file() and not ablate_direction_check(dst, model):
        ok(f"refusal ablation: the {name} from an earlier setup ({why})")
        return dst
    # (no line: until a model's repo has its direction file, the setup says nothing about it)
    return None


def _fetch_ablate_file(url: str, dst: Path, part: Path, model: str, repo: str, name: str):
    """One try of fetch_ablate_direction's download: True (checked, in place) or why it failed."""
    why = ""
    try:
        dst.parent.mkdir(parents=True, exist_ok=True)
        part.unlink(missing_ok=True)
        if shutil.which("curl"):
            r = subprocess.run(["curl", "-sS", "-L", "--fail", "--connect-timeout", "15", "--max-time", "60",
                                "--max-filesize", str(ABLATE_MAX_BYTES), "-o", str(part), url],
                               stdin=subprocess.DEVNULL, capture_output=True, text=True)
            why = "" if r.returncode == 0 else ("the repo ships none yet" if r.returncode == 22 else
                                                (r.stderr.strip().splitlines() or [f"curl exit code {r.returncode}"])[-1])
        else:
            import urllib.error
            import urllib.request
            try:
                with urllib.request.urlopen(url, timeout=60) as resp:
                    part.write_bytes(resp.read(ABLATE_MAX_BYTES + 1))
            except urllib.error.HTTPError as e:
                why = "the repo ships none yet" if e.code in (401, 403, 404) else f"HTTP {e.code}"
            except OSError as e:
                why = str(e)
        if not why:
            why = ablate_direction_check(part, model)
            if not why:
                os.replace(part, dst)
                ok(f"refusal ablation: {name} from https://huggingface.co/{repo} (off until you switch it on: the "
                   "Models page or a request's \"ablate\")")
                return True
    except OSError as e:
        why = str(e)
    finally:
        part.unlink(missing_ok=True)
    return why


def had_ablation(cfg: dict) -> bool:
    """Whether a config loads a refusal direction (its "ablate" record, or --ablate in its args)."""
    return isinstance(cfg.get("ablate"), dict) or "--ablate" in [str(x) for x in cfg.get("args") or []]


def ablate_step(cfg_path: Path, key: str, models: Path, a=None) -> None:
    """The config's refusal direction: the downloaded file recorded as cfg["ablate"] (its model and quant, a default
    set on the Models page kept) with --ablate FILE --ablate-default off in the engine's args.  A file named by hand
    (outside the setup's folder) stays as it is; without a file the three flags go.
    A setup again over a config without a direction leaves it without (#1066: the user's config, not this model's
    default for a new one) - unless that config's own setup found none to download (installer "ablate": "none": tried
    again) or --ablation on asks for it; --ablation off takes it out.  The choice is recorded in installer "ablate"."""
    from serve import models as SM
    cfg = read_json(cfg_path)
    before = config_before(cfg_path)                   # (None: a new config)
    want = getattr(a, "ablation", None)
    inst = cfg.setdefault("installer", {})

    def without(why: str, mark: str) -> None:
        cfg.pop("ablate", None)
        cfg["args"] = SM.ablate_args(cfg, list(cfg.get("args") or []))
        inst["ablate"] = mark
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        ok(why)
    if want == "off":
        return without("refusal ablation: left out (--ablation off; --ablation on loads it again)", "off")
    if want is None and before is not None and not had_ablation(before) and \
            (before.get("installer") or {}).get("ablate") != "none":
        return without("refusal ablation: the config before had none, so it stays without "
                       f"({ME} --setup --ablation on loads the direction, off by default)", "off")
    # (the config before's record: the Models page's default and scale, a file set by hand - the writers start anew)
    old = cfg.get("ablate") if isinstance(cfg.get("ablate"), dict) else \
        before.get("ablate") if before is not None and isinstance(before.get("ablate"), dict) else {}
    mine = models / ABLATE_DIR
    hand = old.get("file") and Path(str(old["file"])).expanduser().is_file() and \
        Path(str(old["file"])).expanduser().resolve().parent != mine.resolve()
    if hand:
        if cfg.get("ablate") != old:                   # (the config before's: written back into the new one)
            cfg["ablate"] = old
            cfg["args"] = SM.ablate_args(cfg, list(cfg.get("args") or []))
            inst["ablate"] = "loaded"
            cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        ok(f"refusal ablation: the direction file set by hand stays ({old['file']})")
        return
    p = fetch_ablate_direction(key, models)
    kept = {k: v for k, v in old.items() if k in ("default", "scale")}   # (what the Models page set)
    if p is None:
        cfg.pop("ablate", None)
    else:
        import gguf_reader as GR
        md = GR.GGUFFile(p).metadata
        cfg["ablate"] = {**kept, "file": str(p), "model": md.get("maya.ablate.model") or ABLATE_DIRECTIONS[key][0],
                         **({"fitted_on": md["maya.ablate.fitted_on"]} if md.get("maya.ablate.fitted_on") else {})}
    inst["ablate"] = "none" if p is None else "loaded"   # (none: no direction to download - a setup again tries)
    cfg["args"] = SM.ablate_args(cfg, list(cfg.get("args") or []))
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")


def compile_vision(pc, meta, llama: Path, src: str) -> bool:
    """maya-vision: llama.cpp's mtmd at the pinned commit with GLM-5.3-Flash's and Inkling 2's vision towers added
    (tools/vision/glm5next_patch.py, inkling2v_patch.py, idempotent), for the same GPUs and toolkit as the engine."""
    cmake = pick_cmake()
    if cmake is None:
        return False
    run([sys.executable, str(ROOT / "tools" / "vision" / "glm5next_patch.py"), str(llama)])
    run([sys.executable, str(ROOT / "tools" / "vision" / "inkling2v_patch.py"), str(llama)])
    gen = re.search(r"^CMAKE_GENERATOR:INTERNAL=(.*)$", (VBUILD / "CMakeCache.txt").read_text(errors="replace"), re.M) \
        if (VBUILD / "CMakeCache.txt").exists() else None
    ninja = venv_tool("ninja") or shutil.which("ninja")
    conf = [cmake]
    if ninja and (gen is None or gen.group(1).strip() == "Ninja"):
        conf += ["-G", "Ninja", f"-DCMAKE_MAKE_PROGRAM={ninja}"]
    metal = pc.get("backend") == "metal"
    archs = meta.get("archs") or pc.get("archs") or []
    conf += ["-S", str(ROOT / "tools" / "vision"), "-B", str(VBUILD), "-DCMAKE_BUILD_TYPE=Release",
             f"-DLLAMA_DIR={llama}"]
    hip = pc.get("backend") == "hip"
    env = vcvars = bindir = None
    if hip:
        conf += ["-DSTRATA_VISION_HIP=ON",
                 "-DSTRATA_VISION_HIP_TARGETS=" + ";".join(str(x) for x in archs),
                 "-DAMDGPU_TARGETS=" + ";".join(str(x) for x in archs),
                 "-DCMAKE_HIP_ARCHITECTURES=" + ";".join(str(x) for x in archs)]
        if WIN:
            # ggml's HIP backend compiles its kernels as C++ on Windows (CMake has no HIP language there): ROCm's
            # clang for C and C++, in Visual Studio's environment, as the engine (compile_engine_hip)
            root, clang, bindir, env = hip_toolchain(pc)
            conf += [f"-DCMAKE_C_COMPILER={(clang.parent / 'clang.exe').as_posix()}",
                     f"-DCMAKE_CXX_COMPILER={clang.as_posix()}", f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root.as_posix()}"]
            vcvars = S.find_vcvars(cuda=False)
        else:
            conf.append("-DCMAKE_EXE_LINKER_FLAGS=-no-pie")
        if S.ARM and S.arm_cpu_arch():
            conf.append(f"-DGGML_CPU_ARM_ARCH={S.arm_cpu_arch()}")
    elif not metal:                                    # (macOS: ggml builds its Metal backend by default, no CUDA)
        conf += ["-DSTRATA_VISION_CUDA=ON",
                 "-DCMAKE_CUDA_ARCHITECTURES=" + ";".join(str(x) for x in archs),
                 f"-DCMAKE_CUDA_COMPILER={meta.get('nvcc') or pc['nvcc']}",
                 *toolkit_root_defs(meta.get('nvcc') or pc['nvcc'])]
        if S.ARM and S.arm_cpu_arch():
            conf.append(f"-DGGML_CPU_ARM_ARCH={S.arm_cpu_arch()}")
    if meta.get("host_compiler") and not metal and not hip:
        conf.append(f"-DCMAKE_CUDA_HOST_COMPILER={shutil.which(meta['host_compiler']) or meta['host_compiler']}")
    jobs = max(2, min((os.cpu_count() or 4) // 2, int(mem_gb()[0] // 4) or 2))
    say("  Compiling the vision encoder (llama.cpp's image library with ggml's " +
        ("Metal backend: a minute or two" if metal else ("HIP" if hip else "CUDA") + " kernels: 10-20 minutes") +
        ", once) ...")
    if cmake_steps(conf, [cmake, "--build", str(VBUILD), "--target", "maya-vision", "-j", str(jobs)],
                   dict(os.environ) if metal else env, "" if metal else "build-maya-vision.bat", vcvars=vcvars,
                   nvcc=None if metal or hip else meta.get("nvcc") or pc["nvcc"]):
        return False
    if bindir is not None and VEXE.exists():
        copy_hip_dlls(bindir, VEXE.parent)
    VSTAMP.write_text(json.dumps({"src": src, "archs": archs, "llama": str(llama),
                                  "backend": pc.get("backend"),
                                  "date": time.strftime("%Y-%m-%d %H:%M")}, indent=1), encoding="utf-8")
    return VEXE.exists()


def vision_step(a, pc, meta, llama: Path, d: Path, quant: str) -> dict | None:
    """The config's "vision" entry - the image encoder and its files - or None (images off).  The server starts the
    encoder only while a request's new pictures are encoded, in GPU memory the model lends it, and measures on the
    first start how much that is on this GPU (serve/server.py, vision_footprint).  On a Mac (Metal: one memory, no
    lending) it stays resident beside the model."""
    step(7, "images (the vision encoder)")
    # a GGUF of your own reads pictures with the same files: the vision tower and the tokenizer are the same in
    # every GLM-5.3-Flash quant
    spec = MODELS.get(quant) or MODELS[next(iter(MODELS))]
    m = spec.get("vision")
    if a.no_vision:
        ok("skipped (--no-vision): the model reads text only")
        return None
    if m is None:
        warn(f"no vision files are published for {quant}: the model reads text only")
        return None
    vd = d / m["folder"]
    files = {k: vd / m[k] for k in ("mmproj", "vocab")}
    missing = [p for p in files.values() if not p.exists() or (m["sha256"].get(p.name) and p.stat().st_size == 0)]
    if missing:
        repo, rev = m.get("repo", spec["repo"]), m.get("revision", spec["revision"])
        src_url = {p: hf_url(repo, rev, m["folder"], p.name) for p in missing}
        say(f"  The vision encoder's files ({m['download_gb']:.2f} GB) from https://huggingface.co/"
            f"{repo} (folder {m['folder']}/), into {vd}:")
        for p in missing:
            say("    " + shell_join(["curl", "-L", "--fail", "-C", "-", "-o", str(p), src_url[p]]))
        if not (a.download_model or a.yes) and ask("  Download them now? (y/n)", ["y", "n"], "y", False) != "y":
            warn(f"not downloaded: the model reads text only ({ME} --setup asks again)")
            return None
        for p in missing:
            fetch(src_url[p], p, m["sha256"].get(p.name))
            ok(f"{p.name} downloaded")
    src = S.source_hash(("tools/vision",))
    vmeta = read_json(VSTAMP)
    if not (VEXE.exists() and not a.rebuild and vmeta.get("src") == src
            and set(meta.get("archs") or pc.get("archs") or []) <= set(vmeta.get("archs", []))):
        if not compile_vision(pc, meta, llama, src):
            warn("the vision encoder did not compile (the reason is above): the model reads text only - "
                 f"{ME} --setup --rebuild tries again")
            return None
    ok(f"vision encoder: {VEXE}")
    return {"exe": str(VEXE), "mmproj": str(files["mmproj"]), "model": str(files["vocab"]), "gpu": True}


# ------------------------------------------------------------------------------------------------ 8. config + start
def parse_env(items) -> dict:
    env = {}
    for it in items:
        k, sep, v = it.partition("=")
        if not sep or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k):
            fail(f"--env takes KEY=VALUE, e.g. --env MAYA_GLM_RAM_HEADROOM_GB=4, not {it!r}")
        env[N.read_as(k)] = v                          # (in memory under the name the code reads; MAYA_ on disk)
    return env


def write_run_script(cfg_path: Path, port: int) -> Path:
    """run-<config>.sh (.bat): this config's start through maya.py - on its screen in a terminal (its loading and
    dashboard), else (--plain, or no terminal: a service) the server's text as it prints."""
    cmd = [sys.executable, str(HERE / "maya.py"), "--config", str(cfg_path), "--port", str(port)]
    if WIN:
        script = ROOT / f"run-{cfg_path.stem}.bat"
        script.write_text(f'@echo off\r\nrem starts the Project Maya dashboard and API (written by maya.py; {ME} does '
                          f'the same)\r\ncd /d "{ROOT}" || exit /b 1\r\n{cmdline(cmd)} %*\r\n', encoding="utf-8")
        return script
    script = ROOT / f"run-{cfg_path.stem}.sh"
    script.write_text("#!/bin/sh\n# starts the Project Maya dashboard and API (written by maya.py; ./maya.sh does the "
                      "same)\ncd " + shlex.quote(str(ROOT)) + " || exit 1\nexec " + shlex.join(cmd) + ' "$@"\n',
                      encoding="utf-8")
    script.chmod(0o755)
    return script


def _number(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool)


# the config's `sampling` keys and the values the server takes (serve/server.py, sampling_defaults_from_config: a value
# outside these refuses to start the server, so a model's own value outside them is left out)
SAMPLING_OK = {
    "temperature": lambda v: _number(v) and v >= 0,
    "top_p": lambda v: _number(v) and 0 < v <= 1,
    "min_p": lambda v: _number(v) and 0 <= v <= 1,
    "top_k": lambda v: _number(v) and v == int(v) and 1 <= v <= 64,
    "presence_penalty": lambda v: _number(v) and v >= 0,
    "frequency_penalty": lambda v: _number(v) and v >= 0,
    "repetition_penalty": lambda v: _number(v) and v > 0,
    "penalty_last_n": lambda v: _number(v) and v == int(v) and v >= -1,
}
# llama.cpp's general.sampling.<key> of a GGUF -> the config's key
GGUF_SAMPLING = {"temp": "temperature", "top_p": "top_p", "top_k": "top_k", "min_p": "min_p",
                 "penalty_repeat": "repetition_penalty", "penalty_last_n": "penalty_last_n"}


def own_sampling(found: dict, base: dict) -> dict:
    """The sampling defaults of a model of the user's own: the ones the model ships (`found`, each in the range the
    server takes; a float32's 0.949999988 as 0.95), else the base model's `base` - never another model's."""
    mine = {}
    for key, v in (found or {}).items():
        if key in SAMPLING_OK and SAMPLING_OK[key](v):
            mine[key] = int(v) if key in ("top_k", "penalty_last_n") else round(float(v), 6)
    return mine or dict(base)


def folder_sampling(folder, base: dict) -> dict:
    """An EXL3 folder's own sampling: its generation_config.json (HF's keys; `do_sample: false` names none), else
    `base`."""
    try:
        gen = json.loads((Path(folder) / "generation_config.json").read_text(encoding="utf-8-sig"))
    except (OSError, ValueError):
        gen = {}
    if not isinstance(gen, dict) or gen.get("do_sample") is False:
        gen = {}
    return own_sampling(gen, base)


def gguf_sampling(first, base: dict) -> dict:
    """A GGUF's own sampling: the general.sampling.* keys it carries, else `base`."""
    try:
        from gguf_reader import GGUFFile
        meta = GGUFFile(Path(first)).metadata
    except Exception:  # noqa: BLE001 - a header that does not read: the base model's
        meta = {}
    return own_sampling({v: meta["general.sampling." + k] for k, v in GGUF_SAMPLING.items()
                         if "general.sampling." + k in meta}, base)


def glm_sampling(choice: tuple | None, quant: str) -> dict:
    """The `sampling` of a GLM config: GLM's for Maya's own and listed downloads (as before); for a GGUF Maya does not
    list (--gguf-dir) or an EXL3 folder (--model-dir) the model's own, else GLM's."""
    if choice and choice[0] == "exl3":
        return folder_sampling(choice[1], SAMPLING)
    if choice and choice[0] == "local" and quant not in MODELS:
        return gguf_sampling(choice[1], SAMPLING)
    return dict(Q.QUANT_SAMPLING.get(("glm", Q.tier("glm", quant)), SAMPLING))   # (Maya-XS: its own)


def g53_defaults(choice: tuple | None, quant: str) -> tuple:
    """(sampling, reasoning_effort) of a GLM-5.3 (744B) config: its own (maya_quants' "glm53" - zai-org/GLM-5.3's card)
    for a listed download, a quant of its own where QUANT_SAMPLING names one; GGUF files not listed: the sampling they
    carry (unsloth's: the card's), else its own - never GLM-5.3-Flash's tuned quant values."""
    sampling, effort = Q.MODEL_DEFAULTS[Q.GLM53]
    if choice and choice[0] == "local" and quant not in G53_MODELS:
        return gguf_sampling(choice[1], sampling), effort
    return dict(Q.QUANT_SAMPLING.get((Q.GLM53, Q.tier("glm", quant)), sampling)), effort


def glm_config_path(pc: dict, quant: str) -> Path:
    """The config write_config writes for `quant` on this PC's backend (and keeps settings from, when it is there)."""
    return ROOT / f"maya-{quant.lower()}{({'hip': '-hip', 'metal': '-metal'}.get(pc.get('backend'), ''))}.json"


# the environment settings the GLM setup itself writes (or leaves out) by the PC and GPUs chosen - a setup again works
# them out anew instead of keeping the config before's
GLM_SETUP_ENV = ("MAYA_GLM_RESERVE_MB", "MAYA_GLM_RAM_HEADROOM_GB", "MAYA_GLM_SPLIT", "MAYA_GLM_MTP_DECODE",
                 "MAYA_GLM_PARALLEL_LOAD", "MAYA_GLM_PREFILL_SUB", "MAYA_GLM_PREFILL_MB", "MAYA_HIPBLASLT_TUNING")

# the server's own settings, set in a config by hand (or with --port / --host / --api-key at a setup): every model's
# setup again keeps them from the config before (#1089: an update's setup turned "model_management": false back on)
# - where it listens, its API key, the pages it lets in, the MCP servers, the model management, the web page (an
# API-only server), the idle unload, the loop guard, the output limit fitted to the context
SERVER_KEYS = ("host", "port", "api_key", "cors_origins", "mcp_servers", "model_management", "web_ui",
               "idle_unload_s", "loop_guard", "fit_max_tokens")


def keep_server_settings(cfg: dict, old: dict, a, extra: tuple = ()) -> list:
    """The config before's SERVER_KEYS (and a family's own `extra` keys) into the new config `cfg`, except a --port /
    --host / --api-key given now (it wins): the keys that differ from what this setup wrote."""
    given = {"port": getattr(a, "port", None), "host": getattr(a, "host", None), "api_key": getattr(a, "api_key", None)}
    kept = []
    for k in SERVER_KEYS + tuple(extra):
        if (old or {}).get(k) is None or given.get(k):
            continue
        if cfg.get(k) != old[k]:
            kept.append(k)
        cfg[k] = old[k]
    return kept


def write_config(a, pc, meta, pack: Path, quant: str, ctx: int, models: Path, vision: dict | None,
                 choice: tuple | None = None, dflash: Path | None = None, gguf: Path | None = None) -> Path:
    step(8, "the configuration and the start script")
    port = a.port or 8080
    metal = pc.get("backend") == "metal"
    # GLM-5.3 (744B, glm-dsa): its own name and defaults; KV streaming only once its engine serves it - until then
    # --kv-streaming off, so that every context the setup offers starts (G53.KV_STREAMING)
    dsa = G53.pack_is_glm_dsa(pack)
    sampling, effort = g53_defaults(choice, quant) if dsa else (glm_sampling(choice, quant), EFFORT)
    # --expert-cache auto: the expert pool takes the VRAM left free less the reserve (--vram-reserve-mib, 700 MiB) and
    # is checked again once written; a number set by hand caps it at that many experts (a setup again keeps it)
    # --prefill auto (as Strata's setup writes it): the engine picks its prompt chunk - the largest its expert pool
    # can lend, up to 8192; a number (`32768`) sets the chunk itself
    cfg = {"exe": str(EXE), "args": ["--glm-pack", str(pack), "--max-context", str(ctx), "--expert-cache", "auto",
                                     "--prefill", "auto", "--kv-streaming",
                                     "off" if dsa and not G53.KV_STREAMING else KV_STREAMING],
           "cwd": str(ROOT),
           "tokenizer": str(pack / "tokenizer"),
           "model_name": G53.MODEL_NAME if dsa else MODEL_NAME, "gpu": [g["index"] for g in pc["gpus"]],
           "sampling": sampling, "reasoning_effort": effort,
           "lib_dirs": meta.get("lib_dirs") or [], "port": port}
    if MODELS.get(quant, {}).get("kv") and not metal:   # a quant with its own KV format (GLM Maya-XS: int8)
        cfg["args"] += ["--kv", MODELS[quant]["kv"]]
    if metal:                      # ggml reads the GGUF files itself: no pack, no expert cache / prefill / KV streaming
        # the learned expert profile (read at start when it is there, rewritten after each request) lives with the
        # tokenizer, in <model folder>/metal/; a setup again keeps it (keep_args)
        cfg["args"] = ["--gguf", str(gguf), "--max-context", str(ctx),
                       "--expert-profile-save", str(pack / "expert-profile.bin")]
        cfg["backend"] = "metal"
    if dflash is not None:                         # DFlash2 drafts instead of the MTP block (the engine loads no MTP)
        cfg["args"] += ["--dflash", str(dflash)]
    env = parse_env(a.env)
    drafting = None
    if pc.get("backend") == "hip":
        cfg["backend"] = "hip"
        # Leave space for the Linux desktop. The engine sizes its prompt chunk from
        # the available prompt-memory budget; forcing 256 here severely slows HIP.
        hip_env = {"MAYA_GLM_RESERVE_MB": str(hip_reserve_mib(pc["gpus"])), "MAYA_GLM_RAM_HEADROOM_GB": "16"}
        gpus = pc["gpus"]
        if len(gpus) == 1:
            hip_env["MAYA_GLM_SPLIT"] = "0"   # two or more cards: the engine picks the split (and drafts with MTP)
        # GLM-5.3 on one APU: one-GPU drafting until the speed calibration measures it (g53_drafting_default)
        drafting = None if dflash is not None else g53_drafting_default(pc, pack, ctx, {**hip_env, **env})
        if drafting is not None:
            hip_env["MAYA_GLM_MTP_DECODE"] = "1" if drafting["on"] else "0"
        # Larger prompt sub-batches feed the matrix cores much better (7900 XT: ~250 -> ~410 tok/s); they need
        # a bigger prompt budget, borrowed from the expert pool only while a prompt runs.
        if any(hip_unified_memory(g) for g in gpus):
            if len(gpus) > 1:
                # Pin the first part's RAM tier before the APU measures the shared memory left.
                hip_env["MAYA_GLM_PARALLEL_LOAD"] = "0"
            if WIN and G53.pack_is_glm_dsa(pack):
                # GLM-5.3 on a Windows APU: the engine already holds its RAM tier within Windows' shared GPU memory
                # (glm_fast_path's apu_shared_room cap), so 8 GB of headroom is enough - a Radeon 8065S (32 GB of
                # system RAM) gets a 10.4 GB tier instead of ~3 GB, decode +10% with the same tokens, 7.8 GB still free
                hip_env["MAYA_GLM_RAM_HEADROOM_GB"] = "8"
            total_ram, _ = mem_gb()
            if WIN:     # Windows counts the GPU's carve-out apart from its RAM, and its desktop runs on that GPU
                total_ram += sum(g.get("vram_gb", 0) for g in gpus)
            hip_env.update({"MAYA_GLM_PREFILL_SUB": "1024",
                            "MAYA_GLM_PREFILL_MB": "6144" if total_ram >= 96 else "4096"})
        elif min(g.get("vram_gb", 0) for g in gpus) >= 19.5:   # (an RX 7900 XT reports 19.98)
            hip_env.update({"MAYA_GLM_PREFILL_SUB": "1024", "MAYA_GLM_PREFILL_MB": "4096"})
        # Pass every available version per architecture: the engine's existing header guard selects the
        # table matching each card and its installed hipBLASLt, or keeps plain hipBLAS when none matches.
        # A version can have more than one table (<version>.txt, then <version>-b.txt, ...: builds of one hipBLASLt
        # version that number their solutions differently); the engine takes, per shape, the first one its build
        # accepts, so each version's own file comes first.
        tables = []
        # GLM-5.3 (glm-dsa) also takes its own prompt shapes' rows (<arch>-glm53-hipblaslt-<version>.txt: N / K of 6144,
        # which no GLM-5.3-Flash projection has) - a Flash config's tables stay what they were
        families = ("glm", "glm53") if G53.pack_is_glm_dsa(pack) else ("glm",)
        for arch in dict.fromkeys(g.get("arch", "") for g in gpus):
            for fam in families:
                found = sorted((ROOT / "tools" / "hip").glob(f"{arch}-{fam}-hipblaslt-*.txt"),
                               key=lambda p: (p.stem.split("-")[3], p.stem))
                tables.extend(str(path) for path in found)
        if tables:
            # the OS's path-list separator: a ':' would also split a Windows drive letter (C:\...)
            hip_env["MAYA_HIPBLASLT_TUNING"] = os.pathsep.join(tables)
        env = {**hip_env, **env}
    cfg_path = glm_config_path(pc, quant)
    old_cfg = read_json(cfg_path)
    # the config before's own variables (set by hand, on the Models page or with --env then) stay, as the DeepSeek,
    # MiniMax, Hy4-Flash and Qwen setups keep them - a GLM setup again dropped them (#1066); the ones this setup works
    # out (the AMD settings, written or left out by the GPUs chosen now) and the tuning's are its own, and an --env
    # given now wins
    own = set(GLM_SETUP_ENV) | set(CAL.KEYS)
    forced, env = env, {**{k: v for k, v in (old_cfg.get("env") or {}).items() if N.shown(str(k)) not in own}, **env}
    if env:
        cfg["env"] = env
    # the server's own settings from the config before (the address it listens on, its API key, ...): kept, as the
    # DeepSeek, MiniMax and Qwen setups keep them - a GLM setup run again dropped them; --host / --api-key given now win
    keep_server_settings(cfg, old_cfg, a, ("mixed_gpus", "mixed_exe", "mixed_gpu"))
    if getattr(a, "mixed_gpus", None):
        set_mixed_gpus(cfg, a.mixed_gpus)
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if vision:
        cfg["vision"] = vision
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    local = choice[1] if choice and choice[0] == "local" else None   # (choose_model's answer)
    if dsa:                                            # GLM-5.3: the pack's own profile, else its shipped one, if it fits
        if (prof := maya_sync.profile_for(pack, PROFILE, G53.PROFILE)) is not None:
            cfg["args"] += ["--expert-profile", str(prof)]
    elif PROFILE.exists() and profile_fits(pack):
        cfg["args"] += ["--expert-profile", str(PROFILE)]
    if cfg_path.exists():
        old = old_cfg.get("args") or []
        # (GLM-5.3 without KV streaming: an older config's --kv-streaming on / --kv-resident is not kept - with them a
        #  context above the window does not start)
        kv_kept = () if dsa and not G53.KV_STREAMING else ("--kv-streaming", "--kv-resident")
        kept = keep_args(old, cfg["args"],
                         ("--expert-cache", "--vram-reserve-mib", "--vram-reserve-later-mib", "--prefill",
                          "--parallel", "--expert-profile-save", "--expert-profile-save-every") + kv_kept)
        kept += keep_profile(old, cfg["args"])
        if kept:
            ok("kept from the config before: " + ", ".join(kept))
    if getattr(a, "parallel", None) and not metal:      # (Metal: one conversation)
        set_arg(cfg["args"], "--parallel", a.parallel)
    if getattr(a, "vram_reserve_mib", None) is not None and not metal:   # given now: written, and a later setup keeps it
        set_arg(cfg["args"], "--vram-reserve-mib", a.vram_reserve_mib)
        ok(f"VRAM kept free beside the expert pool: {a.vram_reserve_mib} MiB (--vram-reserve-mib)")
    # (GLM-5.3-Flash's measurements: not said for GLM-5.3, whose prompt chunks are not measured that way yet)
    for line in ([] if dsa else prefill_tips(cfg["args"], mem_gb()[0], len(pc["gpus"]))):
        ok(line)
    gguf_dir = local.parent if local else Path(a.gguf_dir).expanduser().resolve() if a.gguf_dir else None
    cfg["installer"] = {"models_dir": str(models), "quant": quant, "gguf": str(local) if local else None,
                        "gguf_dir": str(gguf_dir) if gguf_dir else None, "draft": "dflash2" if dflash else "mtp",
                        **({"model_dir": str(choice[1])} if choice and choice[0] == "exl3" else {}),
                        **({"one_gpu_drafting": drafting} if drafting is not None else {}),
                        "written": time.strftime("%Y-%m-%d %H:%M")}
    if drafting is not None:
        ok(f"one-GPU drafting (the MTP block): {'on' if drafting['on'] else 'off'} until the speed calibration "
           f"measures it here (about {drafting['outside'] * 100:.0f}% of the experts would stay outside GPU memory)")
    if cfg_path.exists():                              # a setup again: the budget the user set stays (GLM's 8192 else)
        before = read_json(cfg_path)
        if before.get("thinking_budget") is not None:
            cfg["thinking_budget"] = before["thinking_budget"]
            if (before.get("installer") or {}).get("thinking_budget_set"):
                cfg["installer"]["thinking_budget_set"] = True
    if Q.thinking_budget_for(cfg) is not None:         # (Maya-XS: 8192; every other quant: nothing written, 32768)
        cfg["thinking_budget"] = Q.thinking_budget_for(cfg)
    cal = saved_calibration(cfg, any_context=True)     # tuned on this PC for this model (and context) before
    if cal is not None:
        tuned = CAL.apply(cfg.get("env") or {}, cal.get("settings") or {})
        tuned.update(forced)                           # (an --env given now wins)
        if tuned:
            cfg["env"] = tuned
        if cal.get("gpu_env"):                         # (where it was measured: stale_tuning at a start compares)
            cfg["installer"]["tuned_on"] = cal["gpu_env"]
        ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else "") +
           (": " + ", ".join(f"{k}={v}" for k, v in cal["settings"].items()) if cal.get("settings") else ""))
        if cal.get("context"):
            ok(f"(tuned at context {cal['context']}: {ME} --calibrate tunes this context too)")
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, cfg['port'])}")
    return cfg_path


PREFILL_BIG_RAM_GB = 96        # Strata's benchmarks: 32768-token chunks +21-35% at 96 GB, ~3x slower at 32 GB
PREFILL_RISK_RAM_GB = 64       # below this a chunk set above 8192 is warned about


def prefill_tips(args: list, ram: float, gpus: int = 2) -> list:
    """Strata's --prefill recommendations for a split (one GPU's auto already takes up to 32768): text only, nothing in
    the config changes."""
    prefill = args[args.index("--prefill") + 1] if "--prefill" in args[:-1] else None
    if gpus < 2:
        return []
    if prefill is not None and prefill.isdigit() and int(prefill) > 8192 and ram < PREFILL_RISK_RAM_GB:
        return [f"warning: --prefill {prefill} on {ram:.0f} GB of RAM: on a split, 32768-token chunks read prompts "
                "slower than --prefill auto with little RAM (two V100s, 30 GB: 551 against 709 tok/s); they paid off "
                "(+21-35%) with 96 GB in community benchmarks"]
    if prefill == "auto" and ram >= PREFILL_BIG_RAM_GB:
        return [f"tip: with {ram:.0f} GB of RAM, --prefill 32768 in the config's args read prompts 21-35% faster in "
                "community benchmarks; not set, nothing changes"]
    return []


def set_arg(args: list, flag: str, value) -> None:
    """`flag value` in the engine arguments: its value replaced, or the pair added (`args` is changed in place)."""
    if flag in args[:-1]:
        args[args.index(flag) + 1] = str(value)
    else:
        args += [flag, str(value)]


def set_vram_reserve(args: list, family: str, mib: int, backend: str | None = None) -> str | None:
    """--vram-reserve-mib N (#1026) as the family's engine takes it, written into its arguments (`args` changed in
    place): GLM-5.3-Flash's and Qwen's `maya` --vram-reserve-mib N (MiB), DeepSeek's maya-ds4 --reserve-mib N (MiB),
    MiniMax's maya-mm3 --reserve-gb (GiB) and Hy4-Flash's maya-hy4 --reserve-gb (GB of 10^9 bytes).  The flag and
    value written, for the setup's line; None where the engine has no such setting (Inkling, the Metal engine)."""
    if family in ("glm", "qwen") and backend != "metal":
        flag, value = "--vram-reserve-mib", str(mib)
    elif family == "deepseek":
        flag, value = "--reserve-mib", str(mib)
    elif family == "minimax":
        flag, value = "--reserve-gb", f"{mib / 1024:.3f}".rstrip("0").rstrip(".")
    elif family == "hy4":
        flag, value = "--reserve-gb", f"{mib * 1048576 / 1e9:.3f}".rstrip("0").rstrip(".")
    else:
        return None
    set_arg(args, flag, value)
    return f"{flag} {value}"


def keep_args(old: list, new: list, flags: tuple) -> list:
    """The engine arguments `flags` as the old config had them (edited by hand, or written by an earlier setup):
    their values replace the new ones, or are added.  `new` is changed in place; what was kept is returned."""
    kept = []
    for flag in flags:
        if flag not in old[:-1]:
            continue
        v = old[old.index(flag) + 1]
        if flag in new[:-1]:
            if new[new.index(flag) + 1] == v:
                continue
            new[new.index(flag) + 1] = v
        else:
            new += [flag, v]
        kept.append(f"{flag} {v}")
    return kept


def flag_value(args: list, flag: str):
    """The value after `flag` in an argument list, or None."""
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def tp_keepable(old: dict, pc: dict) -> bool:
    """Whether the config before's --tp (tensor-parallel group size) may stay: it was written for the same GPUs as
    chosen now (its "gpu" list), and it still fits them.  Else a setup works --tp out again from the GPUs (#436)."""
    try:
        now = [int(g["index"]) for g in pc.get("gpus") or []]
        before = [int(i) for i in old.get("gpu") or []]
        tp = int(flag_value(old.get("args") or [], "--tp") or 1)
    except (TypeError, ValueError, KeyError):
        return False
    return bool(now) and before == now and 1 <= tp <= len(now)


def pack_expert_count(pack) -> int | None:
    """The routed experts a layer of a GLM pack's model has: the `n_expert N` in the header of its native_experts.txt
    (both packers write it; the engine reads the same line), or None when the pack does not say."""
    try:
        with open(Path(pack) / "native_experts.txt", encoding="utf-8") as f:
            head = f.readline()
    except OSError:
        return None
    m = re.search(r"\(n_expert (\d+)", head)
    return int(m.group(1)) if m else None


def profile_experts(path) -> int | None:
    """The experts a layer an expert profile file ranks (its header: STRP, version, layers, experts), or None."""
    try:
        with open(path, "rb") as f:
            magic, _, _, experts = struct.unpack("<4sIII", f.read(16))
    except (OSError, struct.error):
        return None
    return experts if magic == b"STRP" else None


def profile_fits(pack) -> bool:
    """Whether the shipped expert profile is for this pack's model: it ranks the full model's experts (288 a layer), and
    the engine refuses one of another size ("a profile for a different artifact") - a pruned model (RCO-152E keeps
    152) starts without a profile and learns this PC's routing.  True where either size is unknown."""
    have, ranked = pack_expert_count(pack), profile_experts(PROFILE)
    return have is None or ranked is None or have == ranked


def add_profile(cfg: dict) -> bool:
    """A GLM config with no --expert-profile (written before setup passed one) gets the shipped profile, as a setup
    would write it: an update restarts the installed config without setting up again.  Not a pruned model's (its
    experts are not the profile's).  True when `cfg` changed."""
    args = cfg.get("args")
    if (not isinstance(args, list) or not any(f in args[:-1] for f in ("--glm-pack", "--gguf"))
            or "--expert-profile" in args or not PROFILE.exists()):
        return False
    pack = flag_value(args, "--glm-pack")              # (a Metal config has no pack: --gguf names the model file)
    if pack is not None and G53.pack_is_glm_dsa(pack):   # GLM-5.3: its own profile only (never Flash's)
        if (prof := maya_sync.profile_for(Path(pack), PROFILE, G53.PROFILE)) is None:
            return False
        args += ["--expert-profile", str(prof)]
        return True
    if pack is not None and not profile_fits(pack):
        return False
    args += ["--expert-profile", str(PROFILE)]
    return True


def ds4_seed_profile(cfg: dict) -> Path | None:
    """A DeepSeek config whose pack has no expert profile yet (a fresh setup, or one from before the shipped profile)
    and that names no other one (--expert-profile): the shipped DS4_PROFILE copied into the pack, where the engine
    reads it and saves what it learns.  Only a profile of the pack's geometry (its layers x routed experts).  The
    profile written, or None."""
    args = cfg.get("args") or []
    pack = flag_value(args, "--pack")
    if pack is None or "--expert-profile" in args[:-1] or not DS4_PROFILE.is_file():
        return None
    dst = Path(pack) / "expert-profile.bin"
    if dst.exists():
        return None
    geo = {}
    try:
        for line in (Path(pack) / "ds4.txt").read_text(encoding="utf-8").splitlines():
            k, _, v = line.partition(" ")
            geo[k] = v.strip()
        want = int(geo["num_hidden_layers"]) * int(geo["n_routed_experts"]) * 4
    except (OSError, KeyError, ValueError):
        return None
    if DS4_PROFILE.stat().st_size != want:
        return None
    try:
        shutil.copyfile(DS4_PROFILE, dst)
    except OSError:
        return None
    return dst


def keep_profile(old: list, new: list) -> list:
    """A learned expert profile wired in by hand: the old config's --expert-profile replaces the
    shipped one only when it is another file that still exists (a moved or deleted one falls back to the shipped
    profile).  `new` is changed in place; what was kept is returned."""
    mine, shipped = flag_value(old, "--expert-profile"), flag_value(new, "--expert-profile")
    if not mine or mine == shipped or Path(mine).name == PROFILE.name or not Path(mine).exists():
        return []                                         # (the shipped one, or another copy's own)
    if shipped is not None:
        new[new.index("--expert-profile") + 1] = mine
    else:
        new += ["--expert-profile", mine]
    return ["--expert-profile " + mine]


# ------------------------------------------------------------------------------------------------ calibration
CAL_STORE = Path(os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config") / "project-maya" / "calibration.json"


def hardware_key(cfg: dict) -> str:
    """What a calibration is valid for: these GPUs, this CPU and RAM, the model (its quant), the context, text or
    images (the context's KV cache and the image encoder take VRAM from the expert pool)."""
    found = {g["index"]: g for g in S.gpus() or []}
    sel = [found.get(i) or {} for i in cfg.get("gpu") or []]
    a = cfg.get("args") or []
    ctx = a[a.index("--max-context") + 1] if "--max-context" in a[:-1] else "?"
    inst = cfg.get("installer") or {}
    quant = inst.get("quant") or cfg.get("model_name", "?")
    if inst.get("ds4_quant"):   # (DeepSeek's quants: each its own tuning)
        quant = f"{quant}/{inst['ds4_quant']}"
    return "|".join([" + ".join(g.get("name", "?") for g in sel) or "?",
                     f"{sum(g.get('vram_gb', 0) for g in sel):.0f}GB", S.cpu_info()[0], f"{S.ram_gb():.0f}GB", quant,
                     str(ctx), "images" if cfg.get("vision") else "text"] + (["dflash2"] if "--dflash" in a else []))


def load_calibrations() -> dict:
    try:
        return json.loads(CAL_STORE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def saved_calibration(cfg: dict, any_context=False) -> dict | None:
    """The settings an earlier tuning found for this PC, model and context, if any (CUDA only, as the tuning).
    any_context: else the ones tuned at the nearest other context, with that context in "context" - the CPU lane's
    split depends on the PC far more than on the context, and a setup for another context used to drop them (#57: the
    engine's own settings, half the speed on a two-socket DDR3 PC)."""
    if cfg.get("backend") in ("hip", "metal"):
        return None
    store, key = load_calibrations(), hardware_key(cfg)
    now = None

    def current(v):                                    # (one measured under another driver or P2P layout: stale, #419)
        nonlocal now
        if not isinstance(v, dict) or not v.get("gpu_env"):
            return True
        now = gpu_env(cfg) if now is None else now
        return not gpu_env_changes(v["gpu_env"], now)
    if key in store or not any_context:
        v = store.get(key)
        return v if v is None or current(v) else None
    want = key.split("|")
    best, best_d = None, None
    for k, v in store.items():
        have = k.split("|")
        if len(have) != len(want) or have[:5] != want[:5] or have[6:] != want[6:]:
            continue
        try:
            d = abs(math.log(int(have[5]) / int(want[5])))
        except (ValueError, ZeroDivisionError):
            continue
        if isinstance(v, dict) and (best_d is None or d < best_d) and current(v):
            best, best_d = dict(v, context=int(have[5])), d
    return best


def p2p_pairs(indices: list) -> dict:
    """Which pairs of these GPUs (nvidia-smi's numbers) read each other's memory directly: {"0-1": "OK", "0-2": "NS",
    ...} from `nvidia-smi topo -p2p r` ({} when it says nothing)."""
    txt = re.sub(r"\x1b\[[0-9;]*m", "", S.out(["nvidia-smi", "topo", "-p2p", "r"]) or "")   # (its header is underlined)
    cols, rows = [], {}
    for line in txt.splitlines():
        t = line.split()
        if t and all(re.fullmatch(r"GPU\d+", x) for x in t):
            cols = [int(x[3:]) for x in t]
        elif t and cols and re.fullmatch(r"GPU\d+", t[0]) and len(t) > len(cols):
            rows[int(t[0][3:])] = dict(zip(cols, t[1:len(cols) + 1]))
    sel = sorted(set(indices))
    return {f"{i}-{j}": rows[i][j] for n, i in enumerate(sel) for j in sel[n + 1:] if j in rows.get(i, {})}


def gpu_env(cfg: dict) -> dict:
    """What a measured setting depends on beyond the hardware key (#419): the NVIDIA driver, the config's GPUs' PCI
    places and, on a split, which pairs of them read each other's memory directly - a P2P driver changes the last
    without any change to the config, and the CPU lane's split tuned before it halved the decode.  {} where nvidia-smi
    says nothing (AMD, no GPU): nothing to compare."""
    if cfg.get("backend") == "hip":
        return {}
    found = {g["index"]: g for g in S.gpus() or []}
    sel = [found[i] for i in cfg.get("gpu") or [] if i in found]
    if not sel:
        return {}
    out = {"driver": sel[0].get("driver") or "?", "pci": ",".join(g.get("pci") or "?" for g in sel)}
    if len(sel) > 1:
        out["p2p"] = p2p_pairs([g["index"] for g in sel])
    return out


def gpu_env_changes(was: dict, now: dict) -> list:
    """What differs between a measurement's gpu_env and this PC's: ["driver 595.58 -> 615.71", ...].  A part either
    side does not know (an older record, nvidia-smi silent once) is no change."""
    out = []
    for k, what in (("driver", "driver"), ("pci", "GPU slots"), ("p2p", "P2P")):
        a, b = (was or {}).get(k), (now or {}).get(k)
        if not a or not b or a == "?" or b == "?" or a == b:
            continue
        if k == "p2p":
            a, b = (" ".join(f"{p}:{s}" for p, s in sorted(x.items())) for x in (a, b))
            if a == b:
                continue
        out.append(f"{what} {a} -> {b}")
    return out


def stale_tuning(cfg: dict) -> str | None:
    """The CPU lane's tuned settings in the config's env (CAL.KEYS: setup's calibration, or one kept by hand) left out
    when the driver, the GPUs' slots or their P2P layout changed since they were measured (#419: a P2P driver installed
    later made a split's MAYA_GLM_PCIE_SHARE=0.25 pin the wrong experts - 25 tok/s instead of 40), so the engine
    measures again at its start.  `cfg` is changed in place; the message, or None (nothing changed).  Settings without
    a record of where they were measured (an older config) get this PC's now, so the next change is seen."""
    env = cfg.get("env") or {}
    keys = [k for k in CAL.KEYS if k in env]
    if not keys:
        return None
    now = gpu_env(cfg)
    if not now:
        return None
    inst = cfg.setdefault("installer", {})
    was = inst.get("tuned_on")
    if not was:
        inst["tuned_on"] = now
        return ""
    diff = gpu_env_changes(was, now)
    if not diff:
        return None
    dropped = ", ".join(f"{N.shown(k)}={env[k]}" for k in keys)
    rest = {k: v for k, v in env.items() if k not in keys}
    if rest:
        cfg["env"] = rest
    else:
        cfg.pop("env", None)
    inst.pop("tuned_on", None)
    return (f"{dropped} was measured on this PC before a change ({'; '.join(diff)}): left out, so the engine "
            f"measures it again at its start ({ME} --calibrate tunes it again)")


def busy_gpus(cfg: dict) -> list:
    """The config's GPUs with more than 1 GiB of VRAM in use (another engine on them): [(index, MiB)]."""
    used = gpu_used_mib()
    return [(i, used[i]) for i in cfg.get("gpu") or [] if used.get(i, 0) > 1024]


# The most a GPU's memory in use can be and still be a display's and the desktop's (#794): 1.5 GiB.  More, and a
# measurement asked for by name still refuses, as it does for a Maya engine on the GPU
DISPLAY_MIB = 1536

# Maya's engines as the GPU's process list names them (and the names they had before the rename)
ENGINE_PROCESSES = ("maya", "maya-ds4", "maya-mm3", "maya-hy4", "maya-ik2", "maya-glm-helper", "strata", "strata-ds4")


def maya_engines_on(cfg: dict) -> list:
    """The Maya engines holding memory on the config's GPUs: [(index, process name, pid)] (nvidia-smi's process list;
    [] where it has none, as on AMD)."""
    def query(what):
        try:
            return subprocess.run(["nvidia-smi", f"--query-{what}", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
    index = {}
    for line in query("gpu=index,uuid").splitlines():
        i, _, uuid = (s.strip() for s in line.partition(","))
        if i.isdigit():
            index[uuid] = int(i)
    mine, out = set(cfg.get("gpu") or []), []
    for line in query("compute-apps=gpu_uuid,pid,process_name").splitlines():
        parts = [s.strip() for s in line.split(",", 2)]
        if len(parts) < 3:
            continue
        exe = re.split(r"[\\/]", parts[2])[-1].lower()
        exe = exe[:-4] if exe.endswith(".exe") else exe
        if exe in ENGINE_PROCESSES and index.get(parts[0]) in mine:
            out.append((index[parts[0]], exe, parts[1]))
    return out


def busy_skip(cfg: dict, asked: bool, name: str, gpus: str, tail: str) -> bool:
    """A measurement's look at the config's GPUs before it loads the model: True = skip it (the warning says why).
    A GPU that drives a display already holds some memory, and read as another model loaded (#794): a measurement
    asked for by name (--speed-calibration, --calibrate) runs anyway when what is in use can be a display's (at most
    DISPLAY_MIB a GPU, and no Maya engine on it), with a warning that says how much; another Maya model loaded there,
    or more memory in use, still refuses.  The automatic ones (a setup's or a start's own) skip on a busy GPU, as
    before."""
    busy = busy_gpus(cfg)
    if not busy:
        return False
    used = ", ".join(f"GPU {i} has {m} MiB in use" for i, m in busy)
    if asked:
        engines = maya_engines_on(cfg)
        if engines:
            warn(f"{name} needs {gpus} to itself: {used} - another Maya model is loaded there (" +
                 ", ".join(f"{exe} on GPU {i}, PID {pid}" for i, exe, pid in engines) + "): stop it, then ask again")
            return True
        if any(m > DISPLAY_MIB for _, m in busy):
            warn(f"{name} needs {gpus} to itself: {used} - more than a display uses (at most {DISPLAY_MIB} MiB a GPU): "
                 "stop what holds it (a model server still running?), then ask again")
            return True
        warn(f"{name} runs although {used} - no more than a display and the desktop use, so it measures anyway, as "
             "asked (the speeds can read a little lower than on a GPU with nothing else on it)")
        return False
    warn(f"{name} needs {gpus} to itself: {used}{tail}")
    return True


def calibrate_config(cfg_path: Path) -> bool:
    """Tune the CPU lane on this PC (tools/calibrate_glm.py): the result goes into the config's env and into
    CAL_STORE for this PC, model and context (a setup again applies it)."""
    cfg = read_json(cfg_path)
    if config_family(cfg) == "deepseek":
        return calibrate_ds4_config(cfg_path)
    if config_family(cfg) == "minimax":   # (maya-mm3 sizes its CPU lane by this PC's cores and its pools by its memory:
        #  what it reads prompts with is measured - the prefill calibration)
        return prefill_calibration(cfg_path, "asked for")
    if config_family(cfg) == "qwen":      # (the Qwen path sizes its CPU pool and its prompt chunks itself; what can be
        #  measured is its drafter)
        return bool(qwen_draft_calibration(cfg_path) is not None)
    if config_family(cfg) == "hy4":       # (maya-hy4 sizes its CPU lane and its expert pool by this PC at every start)
        ok(f"{HY4}: nothing to tune - its engine sizes its CPU lane and its expert pool by this PC at every start")
        return True
    if config_family(cfg) == "ik2":       # (maya-ik2: every expert in VRAM)
        ok(f"{IK2_SHOWN}: nothing to tune - every expert sits in VRAM")
        return True
    if cfg.get("backend") == "hip":
        warn("the tuning measures the CUDA engine's CPU lane (NVIDIA GPUs): the HIP port keeps its own settings")
        return False
    if cfg.get("backend") == "metal":
        warn("the tuning measures the CUDA engine's CPU lane: it does not apply to the Metal engine")
        return False
    if busy_skip(cfg, True, "the tuning", "the GPU(s)",          # (always asked for: --calibrate, or a yes at the setup)
                 " (a model server still running?) - stop it, then run ./maya.sh --calibrate"):
        return False
    say()
    say("  Tuning Maya for this PC: the output speed is measured with a few splits of the work between the CPU and")
    if G53.config_is_glm_dsa(cfg):                     # (GLM-5.3, 744B: a longer load, slower answers)
        say(f"  the PCIe link, and with fewer and more CPU threads. On {G53.SHOWN} it takes about {G53.CALIBRATE_MINUTES}")
        say("  minutes (the model loads first); the PC is busy meanwhile.")
    else:
        say("  the PCIe link, and with fewer and more CPU threads. It takes about 10-15 minutes (the model loads first);")
        say("  the PC is busy meanwhile.")
    log = cfg.get("log")
    since = os.path.getsize(log) if log and os.path.isfile(log) else 0
    try:
        res = CAL.run(cfg, say=say)
    except Exception as e:                             # never stops a setup: the engine's own choices stay
        warn(f"the tuning did not finish ({e}): the engine's own settings stay")
        why = CAL.engine_error(log, since)
        if why:
            say(f"       the engine said: {why}")
        return False
    env = CAL.apply(cfg.get("env") or {}, res["settings"])
    if env:
        cfg["env"] = env
    else:
        cfg.pop("env", None)
    ge = gpu_env(cfg)                                  # (where it was measured: a driver or P2P change re-measures, #419)
    if ge:
        cfg.setdefault("installer", {})["tuned_on"] = ge
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    store = load_calibrations()
    store[hardware_key(cfg)] = {"settings": res["settings"], "tok_s": res["report"].get("tok_s"),
                                "date": time.strftime("%Y-%m-%d"), **({"gpu_env": ge} if ge else {})}
    try:
        CAL_STORE.parent.mkdir(parents=True, exist_ok=True)
        CAL_STORE.write_text(json.dumps(store, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {CAL_STORE} ({e}): {cfg_path.name} has the settings, a setup again does not")
    speed = f" ({res['report']['tok_s']} tok/s)" if res["report"].get("tok_s") else ""
    if res["settings"]:
        ok("tuned for this PC: " + ", ".join(f"{k}={v}" for k, v in res["settings"].items()) + speed)
    else:
        ok("tuned for this PC: the engine's own settings are already the fastest here" + speed)
    return True


def ds4_tuned(chosen: dict) -> str:
    """A DeepSeek calibration's settings as the engine arguments they are ("--tp 8, the engine's own CPU threads")."""
    return ", ".join(f"{k} {v}" if v is not None else f"the engine's own {k[2:].replace('-', ' ')}"
                     for k, v in chosen.items())


def calibrate_ds4_config(cfg_path: Path) -> bool:
    """Tune maya-ds4 on this PC (tools/calibrate_ds4_tp.py): --tp and --cpu-threads go into the config's args and
    into CAL_STORE for this PC, quant and context (a setup again applies them where the config before had none)."""
    cfg = read_json(cfg_path)
    if busy_skip(cfg, True, "the tuning", "the GPU(s)",          # (always asked for: --calibrate, or a yes at the setup)
                 " (a model server still running?) - stop it, then run ./maya.sh --calibrate"):
        return False
    found = {g["index"]: g for g in S.gpus() or []}
    vram = [found[i]["vram_gb"] for i in cfg.get("gpu") or [] if i in found]
    n_tp = len(CAL4.tp_candidates(vram)) or 1
    say()
    say(f"  Tuning {DS4} for this PC: the output speed with each tensor parallel grouping of the GPUs that differs")
    say(f"  ({n_tp} here) and with fewer CPU threads. Every setting is a start of the model of its own, so it takes")
    say(f"  about {6 * (n_tp + 1)}-{10 * (n_tp + 3)} minutes; the PC is busy meanwhile.")   # (9 GPUs: 7-8 starts, 6-12 min each)
    log = cfg.get("log")
    since = os.path.getsize(log) if log and os.path.isfile(log) else 0
    try:
        res = CAL4.run(cfg, vram, say=say)
    except Exception as e:                             # the config stays as it was
        warn(f"the tuning did not finish ({e}): the config's settings stay")
        why = CAL.engine_error(log, since)
        if why:
            say(f"       the engine said: {why}")
        return False
    cfg["args"] = CAL4.apply(cfg.get("args") or [], res["args"])
    cfg.setdefault("installer", {})["calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"), **res["report"]}
    cfg_path.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    store = load_calibrations()
    ge = gpu_env(cfg)                                  # (a driver or P2P change: not offered again by a setup, #419)
    store[hardware_key(cfg)] = {"args": res["args"], "tok_s": res["report"].get("tok_s"),
                                "prompt_tok_s": res["report"].get("prompt_tok_s"), "date": time.strftime("%Y-%m-%d"),
                                **({"gpu_env": ge} if ge else {})}
    try:
        CAL_STORE.parent.mkdir(parents=True, exist_ok=True)
        CAL_STORE.write_text(json.dumps(store, indent=1), encoding="utf-8")
    except OSError as e:
        warn(f"could not save {CAL_STORE} ({e}): {cfg_path.name} has the settings, a setup again does not")
    r = res["report"]
    speed = f" ({r['tok_s']} tok/s answering, the config's own {r['own_tok_s']})" if r.get("tok_s") else ""
    groups = r.get("effective_tp")
    effective = "/".join(map(str, groups)) if groups is not None else "unknown"
    ok(f"tuned for this PC: {ds4_tuned(res['args'])}; effective TP per GPU: {effective}{speed}")
    return True


def drafter_file(cfg: dict) -> Path | None:
    """The DFlash2 drafter this config drafts with (--dflash), else the one a setup downloaded into its models folder
    when it is there; None: no drafter on this PC."""
    args = cfg.get("args") or []
    if "--dflash" in args[:-1]:
        return Path(args[args.index("--dflash") + 1])
    models = (cfg.get("installer") or {}).get("models_dir")
    f = Path(models) / "GLM-5.3-Flash-DFlash2" / DFLASH["file"] if models else None
    return f if f is not None and f.is_file() else None


def parallel_of(cfg: dict) -> int:
    args = cfg.get("args") or []
    v = args[args.index("--parallel") + 1] if "--parallel" in args[:-1] else "1"
    return int(v) if str(v).isdigit() else 1


def unattended_start(a) -> str | None:
    """Why this start has nobody to wait for it, else None: --yes, no terminal on stdin (a service, a pipe), or a
    systemd unit (INVOCATION_ID / NOTIFY_SOCKET).  An unattended start never runs a calibration by itself: it loads the
    model two more times, and a service manager that kills the start after its timeout would repeat that forever (#690)."""
    if getattr(a, "yes", False):
        return "--yes"
    if os.environ.get("INVOCATION_ID") or os.environ.get("NOTIFY_SOCKET"):
        return "started by systemd"
    try:
        if not sys.stdin.isatty():
            return "no terminal"
    except (AttributeError, ValueError, OSError):      # (stdin closed or replaced)
        return "no terminal"
    return None


def skip_start_calibrations(cfg_path: Path, cfg: dict, why: str) -> None:
    """The one line an unattended start prints instead of measuring: what is skipped, and the command to run later."""
    todo = []
    if needs_prefill_calibration(cfg):
        todo.append(("prefill calibration", "--calibrate"))
    if needs_speed_calibration(cfg):
        todo.append(("speed calibration", "--speed-calibration"))
    elif needs_link_calibration(cfg):
        todo.append(("both-GPU speed check", "--speed-calibration"))
    if todo:
        say("  Calibration skipped on this start (" + why + "): the model starts with its default settings. Run it once "
            "when you can wait for it: " + "; ".join(f"{ME} --config {cfg_path} {flag} --no-start ({name})"
                                                      for name, flag in todo))


HY4_Q4X_ENV = "HY4_Q4X"                            # maya-hy4's 4-bit expert copies: 2 / unset measured, 1 on, 0 off


def q4x_config_migrate(cfg_path: Path, cfg: dict) -> dict:
    """A Hy4-Flash config whose env still carries the HY4_Q4X an earlier start's CPU lane check wrote (1.1.69 and
    before, installer.q4x_calibration "on" / "off" - often measured on cold experts, so starts flipped): that value
    dropped.  maya-hy4 now measures its 4-bit copies once its lane's experts are warm and remembers the verdict itself
    (the data folder's hy4-cpu-lane.txt: a line a PC, model and lane-kernel build), so a start never loads the model
    an extra time for it.  A value set by hand (no installer.q4x_calibration, or another value) stays."""
    inst, env = cfg.get("installer") or {}, cfg.get("env") or {}
    res = (inst.get("q4x_calibration") or {}).get("result")
    if config_family(cfg) != "hy4" or res not in ("on", "off") or \
            str(env.get(HY4_Q4X_ENV)) != ("1" if res == "on" else "0"):
        return cfg
    env.pop(HY4_Q4X_ENV, None)
    inst.pop("q4x_calibration", None)
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    return read_json(cfg_path)


def q4x_verdict(log_text: str) -> str | None:
    """maya-hy4's own verdict on the 4-bit copies in its log (HY4_Q4X=2): "on", "off", "unmeasured" (copies made, the
    answers too short for the timed turns), or None - no CPU lane of IQ2_XXS experts, nothing to decide."""
    if "the 4-bit copies kept" in log_text:
        return "on"
    if ("the 4-bit copies dropped" in log_text or "the file's rows kept" in log_text
            or "no 4-bit copies of its experts" in log_text):
        return "off"
    return "unmeasured" if "4-bit copies" in log_text else None


Q4X_ANSWERS = 12                                   # (--speed-calibration's CPU lane check: answers at most for a verdict)
Q4X_BUILD_WAIT_S = 120                             # (... and the wait for the copies, made between answers)
ROUTE_PROMPTS = 12                                 # (... then long prompts at most for the prompt route's rates)


def route_state(log_text: str) -> str | None:
    """maya-hy4's prompt route in its log: "measuring" (its prompt chunks still take the GPU stream and the CPU lane in
    turns), "measured", "pinned" / "remembered", or None - every routed expert in VRAM, no route to choose."""
    if "prompt route: the CPU lane" in log_text:
        return "measured"
    if "prompt route: pinned" in log_text:
        return "pinned"
    if "prompt route: by the rates measured" in log_text:
        return "remembered"
    return "measuring" if "prompt route: measured on this PC" in log_text else None


def hy4_first_start_said(cfg: dict) -> list:
    """The start's lines on what Hy4-Flash's first start on a PC measures once before it is ready, and roughly how
    long it takes (times as the engine's first starts took them: a GPU holding the whole model, GPUs with the CPU
    lane; the server then says the time left at this PC's own pace)."""
    if hy4_engine_drafting(cfg):
        return ["The first start on this PC measures it once before it is ready: which drafting answers fastest here",
                "(each choice loaded and timed on a few short answers - roughly one to two minutes), and where part of",
                "the model runs on the CPU, its CPU lane and prompt route (about one minute more).  Later starts skip",
                "that and are ready in well under a minute."]
    return ["The first start on this PC measures it once before it is ready (where part of the model runs on the CPU:",
            "its CPU lane and prompt route - about one minute more); later starts skip that."]


def hy4_engine_drafting(cfg: dict) -> bool:
    """A Hy4-Flash config whose engine picks its drafting itself (--drafting auto / on): no drafting, the MTP block or
    the DFly2 drafter, whichever decodes fastest here - each loaded once with its own VRAM plan at the first start on
    this PC, before it is ready, and remembered in the data folder (hy4-drafting.txt)."""
    args = [str(x) for x in cfg.get("args") or []]
    return config_family(cfg) == "hy4" and "--drafting" in args[:-1] and \
        args[args.index("--drafting") + 1] in ("auto", "on")


def draft_forget() -> None:
    """--speed-calibration: maya-hy4's drafting measured on this PC (data/hy4-drafting.txt) is dropped, so the engine
    measures it again - in the CPU lane check's run below, else at the next start (before it is ready)."""
    path = ROOT / "data" / "hy4-drafting.txt"
    try:
        path.unlink()
        ok("drafting (none, the MTP block or the DFly2 drafter, whichever decodes fastest here) is measured again")
    except FileNotFoundError:
        pass
    except OSError as e:
        warn(f"{path} could not be removed ({e}): delete it by hand to measure the drafting again")


def ik2_draft_forget() -> None:
    """--speed-calibration: maya-ik2's drafting measured on this PC (data/ik2-drafting.txt, --drafting auto) is
    dropped, so its next start measures it again before it is ready."""
    path = ROOT / "data" / "ik2-drafting.txt"
    try:
        path.unlink()
    except FileNotFoundError:
        pass
    except OSError as e:
        warn(f"{path} could not be removed ({e}): delete it by hand to measure the drafting again")


def route_forget() -> None:
    """--speed-calibration: maya-hy4's prompt route rates measured on this PC are dropped (data/hy4-prompt-route.txt),
    so the engine measures them again - in the CPU lane check's run below, else at the next start (before it is
    ready, on this install's docs: serve/server.py's hy4_measure_ids)."""
    path = ROOT / "data" / "hy4-prompt-route.txt"
    try:
        path.unlink()
        ok("the prompt route's rates (GPU stream or CPU lane for a prompt's experts outside VRAM) are measured again")
    except FileNotFoundError:
        pass
    except OSError as e:
        warn(f"{path} could not be removed ({e}): delete it by hand to measure the prompt route again")


def hot_forget(cfg: dict) -> None:
    """--speed-calibration: maya-hy4's HY4_HOT scores for this config's model (data/hy4-expert-hot-<model>.bin: which
    experts a start places in VRAM, kept as they are between calibrations so that every start begins the same) are
    dropped - the next run learns them again from its answers (the CPU lane check's run below, else the next start)."""
    sys.path.insert(0, str(ROOT))
    import serve.server as SV
    path = ROOT / "data" / SV.hy4_hot_file(cfg).name
    try:
        path.unlink()
        ok("the experts a start places in VRAM (HY4_HOT's scores) are learned again from the next answers")
    except FileNotFoundError:
        pass
    except OSError as e:
        warn(f"{path} could not be removed ({e}): delete it by hand to learn the experts in VRAM again")


def route_prompts(tdir: Path, n: int) -> list[str]:
    """--speed-calibration's prompts for the prompt route's rates: long and each its own (no prefix the engine reuses),
    from this install's docs - a few prompt chunks each."""
    text = ""
    for name in ("docs/SETTINGS.md", "README.md", "CHANGELOG.md"):
        p = ROOT / name
        if p.exists():
            text += p.read_text(encoding="utf-8", errors="replace")
    chat = bench_chat("hy4", tdir)
    step = 6000
    return [chat(f"Summarise this in three sentences:\n\n{text[i * step:(i + 1) * step]}") for i in range(n)
            if len(text) >= (i + 1) * step]


def q4x_calibration(cfg_path: Path, why: str = "") -> bool | None:
    """maya-hy4's 4-bit copies of the CPU lane's most routed experts, measured again on this PC (--speed-calibration):
    the engine's own check (HY4_Q4X=2 with HY4_Q4X_MEASURE=1 - a probe at its start, then, once the copies are made and
    the lane's experts warm, the answers' lane calls timed in turns with and without them until two windows agree; the
    same tokens either way, the copies giving the file's bits) over a warm-up answer and as many more as it takes (up
    to Q4X_ANSWERS).  The engine remembers the verdict itself (the data folder's hy4-cpu-lane.txt), so the next starts
    do not measure; installer.q4x_calibration keeps the result for the record (the config's env is left as it is).
    True / False: on / off; None: nothing to decide here (no CPU lane, no IQ2_XXS experts) or not measured (said)."""
    cfg = read_json(cfg_path)
    tdir = Path(cfg.get("tokenizer") or "")
    if not cfg.get("exe") or not cfg.get("tokenizer") or not (tdir / "chat_template.jinja").exists():
        warn("the CPU lane check needs the config's engine and tokenizer (with its chat template): its 4-bit expert "
             f"copies are measured at each start instead ({ME} --setup writes them again)")
        return None
    if busy_skip(cfg, why == "asked for", "the CPU lane check", "the GPU(s)",
                 f" (a model server still running?) - it runs at the next start, or: {ME} --speed-calibration"):
        return None
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "tools"))
    import serve.server as SV
    import strata_tokenizer as ST
    ids = ST.Tokenizer.from_dir(tdir).encode(bench_chat("hy4", tdir)(BENCH_TOPICS[0]), parse_special=True)
    env = SV.child_env(cfg)
    env[HY4_Q4X_ENV] = "2"
    env["HY4_Q4X_MEASURE"] = "1"                       # (measured again: the verdict remembered before is not used)
    env["HY4_ROUTE_MEASURE"] = "1"                     # (the prompt route's rates too, on long prompts after the answers)
    say()
    say(f"  Checking the CPU lane's 4-bit expert copies on this PC{f' ({why})' if why else ''}: a few answers, the model")
    say("  loads once (a few minutes) ...")
    log_path = ROOT / "maya-q4x-calibration.log"
    done = False

    def said() -> str:
        return log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    with open(log_path, "w", encoding="utf-8") as log:
        try:
            p = subprocess.Popen([cfg["exe"], "--serve"] + SV.engine_args(cfg), stdin=subprocess.PIPE,
                                 stdout=subprocess.PIPE, stderr=log, text=True, env=env,
                                 cwd=cfg.get("cwd") or str(ROOT), bufsize=1)
        except OSError as e:
            warn(f"the CPU lane check could not start the engine ({e}): its copies are measured at each start instead")
            return None

        def gen(n_new: int, prompt: list[int] | None = None) -> bool:
            p.stdin.write(f"GEN {n_new} temperature=0 " + ",".join(map(str, prompt or ids)) + "\n")
            p.stdin.flush()
            for line in p.stdout:
                if line.startswith(("DONE", "ERR")):
                    return line.startswith("DONE")
            return False

        try:
            for line in p.stdout:
                if line.startswith("READY"):
                    # (the warm-up answer gives the routing counts the copies are made by - between answers, so the
                    #  engine is left idle until it times them - then answers until its verdict: the one-token calls
                    #  timed in turns with and without them, window by window, until two agree)
                    done = gen(64)
                    for _ in range(Q4X_BUILD_WAIT_S // 5):
                        if not done or q4x_verdict(said()) != "unmeasured" or "timing the answers" in said():
                            break
                        time.sleep(5)
                    for _ in range(Q4X_ANSWERS):
                        if not done or q4x_verdict(said()) != "unmeasured":
                            break
                        done = gen(256)
                        time.sleep(2)                  # (the verdict's line reaches the log)
                    # (the prompt route's rates: long prompts, each read in chunks whose CPU lane layers take the GPU
                    #  stream and the lane in turns, until two windows of times agree)
                    tok = ST.Tokenizer.from_dir(tdir)
                    for text in route_prompts(tdir, ROUTE_PROMPTS):
                        if not done or route_state(said()) != "measuring":
                            break
                        done = gen(1, tok.encode(text, parse_special=True))
                        time.sleep(1)
                    break
            p.stdin.write("QUIT\n")
            p.stdin.flush()
            p.wait(timeout=120)
        except (OSError, ValueError, subprocess.TimeoutExpired):
            p.kill()
    verdict = q4x_verdict(log_path.read_text(encoding="utf-8", errors="replace"))
    if m := re.search(r"INFO drafting: (.+?) - the fastest on this PC \(measured (now|[0-9-]+)", said()):
        ok(f"drafting: {m.group(1)} - whichever of no drafting, the MTP block and the DFly2 drafter decodes fastest on "
           "this PC, each with its own VRAM plan (measured once, remembered: the next starts use it)")
    rs = route_state(said())
    if rs == "measured":
        ok("the prompt route's rates (GPU stream or CPU lane for a prompt's experts outside VRAM) measured on this PC: "
           "the same prompt now takes the same route at every start")
    elif rs == "measuring":
        warn("the prompt route's rates did not settle in this check: the engine measures them at the next start, "
             "before it is ready, and remembers them")
    if not done or verdict == "unmeasured":
        warn(f"the CPU lane check did not finish ({log_path.name} says why): the engine measures its copies at the "
             f"next start instead, once its experts are warm, and remembers it ({ME} --speed-calibration tries again)")
        cfg.setdefault("installer", {})["q4x_calibration"] = {"result": "unmeasured", "date": time.strftime("%Y-%m-%d")}
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        return None
    # (the verdict is the engine's to remember - per PC, model and lane kernels; the config only records it)
    cfg.setdefault("installer", {})["q4x_calibration"] = {"result": verdict or "n/a", "date": time.strftime("%Y-%m-%d")}
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    if verdict is None:
        ok("CPU lane check: nothing to decide on this PC (no CPU lane of IQ2_XXS experts)")
        return None
    ok(f"the CPU lane's 4-bit expert copies: {'ON - faster' if verdict == 'on' else 'off - not faster'} on this PC "
       "(the same answers either way)")
    return verdict == "on"


def needs_speed_calibration(cfg: dict) -> bool:
    """A config that drafts with DFlash2 but has no speed calibration yet (set up before there was one, or its
    calibration could not run then - the GPUs were busy): its next start measures first.  Not with --parallel above 1:
    the engine then decodes without the drafter."""
    args = cfg.get("args") or []
    if cfg.get("backend") == "metal":                  # (CUDA's measurement: on a Mac MTP or DFlash2 is the user's choice)
        return False
    # (CYBER-FROST: measured at its setup, or --speed-calibration; Hy4-flash: Tencent's Q4_K_M drafts with its MTP block
    #  or not, chosen - its drafts doubled code answers on the GPUs it ran on; a Maya size where the MTP block and the
    #  DFly2 drafter both fit: which of the two, measured once - hy4_both_drafters)
    if config_family(cfg) == "hy4":                    # (--drafting auto / on: the engine measures at its first start)
        return hy4_both_drafters(cfg) and not hy4_engine_drafting(cfg) and \
            (cfg.get("installer") or {}).get("drafting") == "on" and \
            "speed_calibration" not in (cfg.get("installer") or {})
    if config_family(cfg) in ("qwen", "ik2"):         # (Inkling 2: --drafting auto, measured by its engine)
        return False
    if config_family(cfg) in ("deepseek", "minimax"):  # (their DSpark drafter: measured once too)
        return CDS.has_drafter(cfg) and "speed_calibration" not in (cfg.get("installer") or {}) and             not (config_family(cfg) == "minimax" and parallel_of(cfg) > 1)   # (maya-mm3 --parallel: no drafting)
    if g53_one_gpu_drafting(cfg):                      # (GLM-5.3 on one APU: drafting on or off, measured once)
        return "speed_calibration" not in (cfg.get("installer") or {})
    return "--dflash" in args[:-1] and parallel_of(cfg) <= 1 and \
        "speed_calibration" not in (cfg.get("installer") or {})


def speed_calibration(cfg_path: Path, why: str = "") -> bool:
    """The speed calibration (tools/calibrate_draft.py): this config's sustained decode speed with the DFlash2 drafter
    and without it (the MTP block; one GPU: no drafting), measured on this PC.  The config keeps the faster - DFlash2
    only when it wins by more than CD.MIN_GAIN - and the result (installer.speed_calibration), so it runs once: a setup
    that chooses DFlash2, or --speed-calibration, measures again.  False: not measured (the reason is shown; nothing
    changed, and a failure is recorded so that a start does not try again by itself)."""
    cfg = read_json(cfg_path)
    if cfg.get("backend") == "metal":
        warn("no speed calibration on Metal: MTP and DFlash2 both write only tokens the model checks; which is faster here is your choice "
             f"({ME} --setup --draft mtp|dflash2)")
        return False
    if config_family(cfg) == "qwen":                   # (CYBER-FROST: its MTP drafter or none)
        return bool(qwen_draft_calibration(cfg_path) is not None)
    if config_family(cfg) == "ik2":                    # (Inkling 2: --drafting auto measures at the engine's start)
        args = cfg.get("args") or []
        if "--drafting" in args[:-1] and args[args.index("--drafting") + 1] == "auto":
            ik2_draft_forget()
            ok(f"{IK2_SHOWN}: its drafting is measured again at the next start (the engine times no drafting, the "
               "prompt lookup and the drafter once, before its first answer)")
            return True
        ok(f"{IK2_SHOWN}: no speed calibration - drafting is set by hand (--drafting on|off); {ME} --setup "
           "--drafting measure lets the engine measure it")
        return False
    if config_family(cfg) == "hy4":
        route_forget()                                 # (the prompt route's rates: measured again too)
        hot_forget(cfg)                                # (... and the experts a start places in VRAM: learned again)
        if hy4_engine_drafting(cfg):                   # (--drafting auto / on: the engine's start measures it)
            draft_forget()
            inst, env = cfg.setdefault("installer", {}), cfg.get("env") or {}
            if inst.pop("q4x_calibration", None) is not None and HY4_Q4X_ENV in env:
                env.pop(HY4_Q4X_ENV, None)
                cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
            return q4x_calibration(cfg_path, why) is not None
        if hy4_both_drafters(cfg):                     # (a Maya size with both drafters: which is faster here)
            return speed_calibration_hy4(cfg_path, cfg, why)
        if Q.config_key(cfg) == HY4_Q4:                # (Tencent's own Q4_K_M: the one set with the MTP block)
            warn(f"no speed calibration for {HY4} yet: it drafts with its MTP block unless set up with --drafting off "
                 f"({ME} --setup --model hy4-flash-q4_k_m --drafting on|off)")
        else:
            ok(f"{HY4}: no speed calibration here - a Maya size measures which drafter is faster where the MTP block "
               f"and the DFly2 drafter both fit; here the setup found room for one of them or none ({ME} --setup "
               "--drafting on|off|mtp|dfly2)")
            inst, env = cfg.setdefault("installer", {}), cfg.get("env") or {}
            if HY4_Q4X_ENV in env and "q4x_calibration" not in inst:   # (set by hand: the user's choice stays)
                ok(f"the CPU lane's 4-bit expert copies: {HY4_Q4X_ENV}={env[HY4_Q4X_ENV]} in the config's env - kept, "
                   "not measured")
                return False
            if inst.pop("q4x_calibration", None) is not None:   # (measured again, as asked)
                env.pop(HY4_Q4X_ENV, None)
                cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
            return q4x_calibration(cfg_path, why) is not None
        return False
    if config_family(cfg) in ("deepseek", "minimax"):
        return speed_calibration_ds4(cfg_path, cfg, why)
    if g53_one_gpu_drafting(cfg):
        return speed_calibration_g53(cfg_path, cfg, why)
    drafter = drafter_file(cfg)
    if drafter is None or not drafter.is_file():
        warn("the speed calibration compares the DFlash2 drafter with the MTP block, and the drafter is not on this "
             f"PC: {ME} --setup --draft dflash2 downloads it")
        return False
    if parallel_of(cfg) > 1:
        warn("with --parallel above 1 the engine decodes without the DFlash2 drafter: no speed calibration")
        return False
    one = len(cfg.get("gpu") or [0]) < 2
    other = "no drafting" if one else "the MTP block"
    if busy_skip(cfg, why == "asked for", "the speed calibration", "the GPU(s)",
                 f" (a model server still running?) - it runs at the next start, or: {ME} --speed-calibration"):
        return False
    without = CD.without_dflash(cfg)
    cal = saved_calibration(without)                   # the CPU lane tuned on this PC without DFlash2: that side as so
    if cal is not None and cal.get("settings") is not None:
        env = CAL.apply(without.get("env") or {}, cal["settings"])
        if env:
            without["env"] = env
        else:
            without.pop("env", None)
    with_d = CD.with_dflash(cfg, drafter)
    say()
    say("  Speed calibration: DFlash2 or " + other + "?" + (f" ({why})" if why else ""))
    say("  DFlash2 makes some PCs faster and others slower: it depends on how much of the model fits in the GPUs'")
    say("  memory, and on the CPU and the RAM. So Maya measures both here, the way you will use it. It loads the")
    say(f"  model twice ({other}, then DFlash2); each time it reads a long document, writes about {CD.MIN_TOKENS} "
        "tokens")
    say("  of answers after a warm-up, and the sustained speeds are compared. About 5-20 minutes; the PC is busy.")
    log = cfg.get("log")
    since = os.path.getsize(log) if log and os.path.isfile(log) else 0
    try:
        res = CD.run(without, with_d, say=say, other_label=other)
    except Exception as e:                             # never stops a setup or a start: the config stays as it was
        now = "DFlash2" if "--dflash" in (cfg.get("args") or [])[:-1] else other
        warn(f"the speed calibration did not finish ({e}): nothing changed - Maya keeps {now}; "
             f"{ME} --speed-calibration tries again")
        why_not = CAL.engine_error(log, since)
        if why_not:
            say(f"       the engine said: {why_not}")
        cfg.setdefault("installer", {})["speed_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                                                                "failed": str(e)[:300]}
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        return False
    o, d = res["other"]["tok_s"], res["dflash2"]["tok_s"]
    keep = res["keep"] == "dflash2"
    out = with_d if keep else without
    inst = out.setdefault("installer", {})
    inst["draft"] = "dflash2" if keep else "mtp"
    inst["speed_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"), "version": maya_version(),
                                 "dflash2_tok_s": d, ("no_drafting" if one else "mtp") + "_tok_s": o,
                                 "tokens": [res["other"]["tokens"], res["dflash2"]["tokens"]], "kept": inst["draft"]}
    cfg_path.write_text(json.dumps(N.env_out(out), indent=1), encoding="utf-8")
    say()
    if keep:
        ok(f"speed calibration: DFlash2 is faster on this PC - {d:.1f} against {o:.1f} tokens/s with {other} "
           f"(+{res['gain'] * 100:.0f}%): Maya drafts with DFlash2")
    else:
        ok(f"speed calibration: DFlash2 is not faster on this PC - {d:.1f} against {o:.1f} tokens/s with {other}: "
           "Maya " + ("decodes without drafting" if one else "drafts with the MTP block"))
        say(f"       (saved in {cfg_path.name}; to draft with DFlash2 anyway: {ME} --setup --draft dflash2 "
            "--no-speed-calibration)")
    return True


def hy4_both_drafters(cfg: dict) -> bool:
    """A Maya size of Hy4-Flash set up where its MTP block and its DFly2 drafter both fit (installer.draft_choice
    "auto": both in the pack, one GPU): which of the two drafts is the speed calibration's to measure."""
    inst = cfg.get("installer") or {}
    return config_family(cfg) == "hy4" and Q.config_key(cfg) != HY4_Q4 and inst.get("draft_choice") == "auto" and \
        bool(inst.get("draft_block")) and bool(inst.get("dfly2")) and len(cfg.get("gpu") or [0]) < 2


def hy4_with_drafter(cfg: dict, drafter: str) -> dict:
    """A copy of a Hy4-Flash config that drafts with `drafter` ("mtp": --mtp, "dfly2": --dfly2) - the other's flag
    out."""
    out = json.loads(json.dumps(cfg))
    args = [x for x in (out.get("args") or []) if x not in ("--mtp", "--dfly2")]
    out["args"] = args + ["--dfly2" if drafter == "dfly2" else "--mtp"]
    out.setdefault("installer", {})["drafter"] = drafter
    return out


def speed_calibration_hy4(cfg_path: Path, cfg: dict, why: str = "") -> bool:
    """A Maya size of Hy4-Flash where both drafters fit: its sustained decode speed with the MTP block and with the
    DFly2 drafter (both write only tokens the model checks: the same answers), a short probe each as
    tools/calibrate_draft.py measures (HY4_DRAFT_PROBE: code, an explanation, plain text, thinking).  The faster is
    kept; within HY4_DRAFT_TIE the size's rule (HY4_DRAFT_RULE) stays.  False: not measured (nothing changed)."""
    if busy_skip(cfg, why == "asked for", "the speed calibration", "the GPU",
                 f" (a model server still running?) - it runs at the next start, or: {ME} --speed-calibration"):
        return False
    quant = Q.config_key(cfg)
    rule = HY4_DRAFT_RULE.get(quant, "mtp")
    mtp, dfly = hy4_with_drafter(cfg, "mtp"), hy4_with_drafter(cfg, "dfly2")
    say()
    say("  Speed calibration: the MTP block or the DFly2 drafter?" + (f" ({why})" if why else ""))
    say("  Both fit on this GPU and the model checks every drafted token; which drafts faster depends on how much of")
    say(f"  the model the GPU holds beside it. Maya loads the model twice (the MTP block, then DFly2); each time it "
        f"writes about {HY4_DRAFT_PROBE['tokens']} tokens of")
    say("  short answers (code, an explanation, plain text, thinking) after a warm-up, and the speeds are compared.")
    say("  The PC is busy meanwhile.")
    log = cfg.get("log")
    since = os.path.getsize(log) if log and os.path.isfile(log) else 0
    now = (cfg.get("installer") or {}).get("drafter") or ("dfly2" if "--dfly2" in (cfg.get("args") or []) else "mtp")
    try:
        res = CD.run(mtp, dfly, say=say, other_label="the MTP block", draft_label="DFly2", size=HY4_DRAFT_PROBE)
    except Exception as e:                             # never stops a setup or a start: the config stays as it was
        warn(f"the speed calibration did not finish ({e}): nothing changed - Maya keeps "
             f"{'DFly2' if now == 'dfly2' else 'the MTP block'}; {ME} --speed-calibration tries again")
        why_not = CAL.engine_error(log, since)
        if why_not:
            say(f"       the engine said: {why_not}")
        cfg.setdefault("installer", {})["speed_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                                                                "failed": str(e)[:300]}
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        return False
    o, d = res["other"]["tok_s"], res["dflash2"]["tok_s"]
    tie = abs(d - o) <= HY4_DRAFT_TIE * max(o, d)
    keep = rule if tie else "dfly2" if d > o else "mtp"
    out = mtp if keep == "mtp" else dfly
    out.setdefault("installer", {})["speed_calibration"] = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "version": maya_version(), "mtp_tok_s": o, "dfly2_tok_s": d,
        "tokens": [res["other"]["tokens"], res["dflash2"]["tokens"]], "kept": keep}
    cfg_path.write_text(json.dumps(N.env_out(out), indent=1), encoding="utf-8")
    say()
    ok(f"speed calibration: {d:.1f} tokens/s with DFly2 against {o:.1f} with the MTP block - Maya drafts with " +
       ("DFly2" if keep == "dfly2" else "the MTP block") + (" (a tie: this size's default)" if tie else ""))
    say(f"       (saved in {cfg_path.name}; {ME} --speed-calibration measures again, --setup --drafting mtp|dfly2 "
        "picks one by hand)")
    return True


G53_DRAFT_TIE = 0.02     # one-GPU drafting: within this of each other the setup's default (its estimate) stays
# its probe: four short answers a side as in real use - code, an explanation, plain text, thinking (calibrate_draft's
# MIXED; decode speed only - two loads plus a couple of minutes)
G53_DRAFT_PROBE = {"mixed": True, "tokens": 512, "answer": 128, "warm": 32}


def g53_with_drafting(cfg: dict, on: bool) -> dict:
    """A copy of a GLM-5.3 config with one-GPU drafting on or off (its MAYA_GLM_MTP_DECODE)."""
    out = json.loads(json.dumps(cfg))
    out["env"] = {**(out.get("env") or {}), "MAYA_GLM_MTP_DECODE": "1" if on else "0"}
    return out


def speed_calibration_g53(cfg_path: Path, cfg: dict, why: str = "") -> bool:
    """GLM-5.3 on one APU: its sustained decode speed with one-GPU drafting (the MTP block drafts, one verify pass
    checks: the same tokens) and without, measured here as tools/calibrate_draft.py measures DFlash2.  The faster is
    kept; within G53_DRAFT_TIE the setup's own default (installer.one_gpu_drafting) stays.  False: not measured."""
    if busy_skip(cfg, why == "asked for", "the speed calibration", "the GPU",
                 f" (a model server still running?) - it runs at the next start, or: {ME} --speed-calibration"):
        return False
    fallback = bool(((cfg.get("installer") or {}).get("one_gpu_drafting") or {}).get("on"))
    off, on = g53_with_drafting(cfg, False), g53_with_drafting(cfg, True)
    say()
    say("  Speed calibration: drafting with the MTP block on this GPU, or not?" + (f" ({why})" if why else ""))
    say("  The model checks every drafted token; whether drafting is faster depends on how much of the model the GPU holds.")
    say(f"  Maya loads the model twice (no drafting, then drafting); each time it writes about "
        f"{G53_DRAFT_PROBE['tokens']} tokens of")
    say("  short answers (code, an explanation, plain text, thinking) after a warm-up, and the speeds are compared.")
    say("  The PC is busy meanwhile.")
    log = cfg.get("log")
    since = os.path.getsize(log) if log and os.path.isfile(log) else 0
    try:
        res = CD.run(off, on, say=say, other_label="no drafting", draft_label="drafting", size=G53_DRAFT_PROBE)
    except Exception as e:                             # never stops a setup or a start: the config stays as it was
        warn(f"the speed calibration did not finish ({e}): nothing changed - Maya keeps drafting "
             f"{'on' if fallback else 'off'}; {ME} --speed-calibration tries again")
        why_not = CAL.engine_error(log, since)
        if why_not:
            say(f"       the engine said: {why_not}")
        cfg.setdefault("installer", {})["speed_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                                                                "failed": str(e)[:300]}
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        return False
    o, d = res["other"]["tok_s"], res["dflash2"]["tok_s"]
    keep = True if d > o * (1 + G53_DRAFT_TIE) else False if o > d * (1 + G53_DRAFT_TIE) else fallback
    out = on if keep else off
    out.setdefault("installer", {})["speed_calibration"] = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "version": maya_version(), "drafting_tok_s": d,
        "no_drafting_tok_s": o, "tokens": [res["other"]["tokens"], res["dflash2"]["tokens"]],
        "kept": "mtp" if keep else "none"}
    cfg_path.write_text(json.dumps(N.env_out(out), indent=1), encoding="utf-8")
    say()
    ok(f"speed calibration: {d:.1f} tokens/s drafting against {o:.1f} without - Maya " +
       ("drafts with the MTP block on this GPU" if keep else "decodes without drafting") +
       (" (a tie: the default for this PC)" if abs(d - o) <= G53_DRAFT_TIE * max(o, d) else ""))
    say(f"       (saved in {cfg_path.name}; {ME} --speed-calibration measures again)")
    return True


LINK_TIE = 0.03          # both GPUs (GLM link mode) only when their decode beats the AMD GPU alone by more than this
# ... DeepSeek: when reading and writing, weighed alike (tools/calibrate_draft.py's run_phases: documents read and fixed
# answers written), beat it by more than this - and the writing alone is not slower (on by itself only where the
# decode gains too).  The files DeepSeek's engine learns into its pack at every run: put back before each side
# starts and after the check, so both sides start alike and the check's documents lead no later start
DS4_LEARNED = ("expert-profile.bin", "lane-times.txt")
# DeepSeek on both GPUs turns itself on after a winning speed check (False: the check measures and says the result,
# the config stays on the AMD GPU alone - --mixed-gpus on still turns both on).  On again since the helper layout's
# answers are right (the AMD GPU reads the card's sums) and both its reading and its writing measured faster
DS4_BOTH_AUTO = True
# GLM-5.3 (744B, glm-dsa) on both GPUs only when asked (--mixed-gpus on): its helper maps the largest shared RAM tier,
# and the AMD driver at times stops the helper's queues for seconds to minutes while it re-validates that range after
# the kernel moved pages - so the first start does not measure it and it starts on the AMD GPU alone.  GLM-5.3-Flash
# and DeepSeek measure as before
G53_BOTH_AUTO = False


def ds4_learned_kept(cfg: dict):
    """-> put(): the pack's DS4_LEARNED files as they are now (one that is not there: not there again)."""
    args = cfg.get("args") or []
    pack = Path(args[args.index("--pack") + 1]) if "--pack" in args[:-1] else None
    kept = {}
    for name in DS4_LEARNED if pack is not None else ():
        try:
            kept[name] = (pack / name).read_bytes()
        except OSError:
            kept[name] = None

    def put():
        for name, data in kept.items():
            try:
                if data is None:
                    (pack / name).unlink(missing_ok=True)
                else:
                    (pack / name).write_bytes(data)
            except OSError:
                pass
    return put


def mixed_close(eng) -> None:
    """The speed check's engine ended, the NVIDIA card's engine beside it too (serve/mixed_gpus.py's close), and
    their memory back before the next one loads."""
    try:
        eng.close()
    except Exception:
        CAL.close(eng)


def mixed_family(cfg: dict) -> str | None:
    """Which two-GPU layout a config takes ("mixed_gpus"): "glm" (serve/glm_link.py: the NVIDIA card leads), "deepseek"
    (serve/mixed_gpus.py: the AMD GPU leads, the card computes experts for it), None: neither (other models)."""
    if GL.is_glm(cfg):
        return "glm"
    return "deepseek" if config_family(cfg) == "deepseek" else None


def mixed_why_not(cfg: dict) -> str | None:
    """Why this config cannot run on both GPUs here (None: it can), for either engine."""
    fam = mixed_family(cfg)
    if fam == "glm":
        return GL.why_not(cfg)
    if fam == "deepseek":
        from serve import mixed_gpus as MG
        return MG.why_not(cfg)
    return "a DeepSeek or GLM config only"


def mixed_start_engine(c: dict):
    """An engine for the both-GPU speed check: DeepSeek's two engines when the config runs on both
    (serve/mixed_gpus.py), else one (GLM's link config is already the NVIDIA lead's: glm_link.link_config)."""
    from serve import mixed_gpus as MG
    from serve.server import StrataEngine, child_env, engine_args, wait_gpu_release
    m = MG.config_of(c)
    if m is None:
        return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=child_env(c))
    return MG.engine_class(StrataEngine, wait_gpu_release)(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"),
                                                         env=child_env(c), mixed=m)


def set_mixed_gpus(cfg: dict, value: str | None) -> None:
    """--mixed-gpus into the config's one key "mixed_gpus", in the layout of its model (mixed_family)."""
    if value is None:
        return
    if mixed_family(cfg) == "glm":
        if value not in ("on", "off"):
            say(f"  (GLM has one two-GPU layout - the NVIDIA card leads: --mixed-gpus {value} is taken as on)")
            value = "on"
        set_glm_mixed_gpus(cfg, value)
    else:
        set_ds4_mixed_gpus(cfg, value)


def set_glm_mixed_gpus(cfg: dict, value: str | None) -> None:
    """--mixed-gpus on|off into a GLM config ("mixed_gpus": true / false; serve/glm_link.py reads it at each start)."""
    if value is None:
        return
    if value == "off":
        cfg["mixed_gpus"] = False
        ok("both GPUs: off (--mixed-gpus off: the model runs on the AMD GPU alone; --mixed-gpus on turns it on again)")
        return
    why = GL.why_not(cfg)
    cfg["mixed_gpus"] = True
    ok("both GPUs: on (--mixed-gpus on: the NVIDIA card leads, the AMD GPU computes the other experts; answers can be "
       "worded slightly differently from the AMD GPU alone; --mixed-gpus off turns it off)")
    if why:
        warn(f"both GPUs: not possible on this PC yet ({why}) - until then the model runs on the AMD GPU alone")


def needs_link_calibration(cfg: dict) -> bool:
    """A GLM or DeepSeek config on an AMD GPU with an NVIDIA card and the builds beside it, never measured both ways
    and with no "mixed_gpus" of the user's: its next attended start measures (link_calibration).  Not GLM-5.3 (744B)
    while G53_BOTH_AUTO is off: it starts on the AMD GPU alone, --mixed-gpus on turns both on."""
    if not G53_BOTH_AUTO and mixed_family(cfg) == "glm" and G53.config_is_glm_dsa(cfg):
        return False
    return "mixed_gpus" not in cfg and "mixed_gpus_calibration" not in (cfg.get("installer") or {}) and \
        mixed_family(cfg) is not None and mixed_why_not(cfg) is None


def link_calibration(cfg_path: Path, why: str = "") -> bool:
    """On both GPUs (GLM: link mode; DeepSeek: the helper layout, or "mixed_mode") or on the AMD GPU alone, measured
    on this PC.  GLM: the decode speed of each, as the drafting calibration measures (tools/calibrate_draft.py, the
    same short probe).  DeepSeek: documents read and fixed answers written (its run_phases - the card helps both, and
    sampled short answers differed more from run to run than the two sides do), reading and writing weighed alike.
    "mixed_gpus" becomes true only when both GPUs win by more than LINK_TIE.  The numbers are kept
    (installer.mixed_gpus_calibration), so it runs once; --speed-calibration measures again.  False: not measured
    (nothing changed)."""
    cfg = read_json(cfg_path)
    reason = mixed_why_not(cfg)
    if reason:
        warn(f"both GPUs: not measured ({reason})")
        return False
    if busy_skip(cfg, why == "asked for", "the both-GPU measurement", "the GPU",
                 f" - {ME} --speed-calibration tries again"):
        return False
    glm = mixed_family(cfg) == "glm"
    alone = {**json.loads(json.dumps(cfg)), "mixed_gpus": False}
    both = GL.link_config(cfg) if glm else {**json.loads(json.dumps(cfg)), "mixed_gpus": True}
    log_path, log_at = None, 0
    if glm:
        # the drafting length: the adaptive search runs in this measurement and its pick is kept (link starts draft a
        # fixed length, so their answers repeat run to run - serve/glm_link.py)
        both["env"] = {k: v for k, v in (both.get("env") or {}).items() if k != "MAYA_GLM_MTP_FIXED"}
        both["env"]["MAYA_GLM_MTP_ADAPT"] = "1"
        log_path = Path(both["log"]) if both.get("log") else None
        log_at = log_path.stat().st_size if log_path is not None and log_path.is_file() else 0
    say()
    say("  Speed check: this PC has an NVIDIA card beside the AMD GPU - is the model faster on both?" +
        (f" ({why})" if why else ""))
    if glm:
        say("  Maya loads the model twice (the AMD GPU alone, then both GPUs) and compares the decode speed.")
    else:
        say("  Maya loads the model twice (the AMD GPU alone, then both GPUs); each time it reads two documents and")
        say("  writes a fixed answer to each, and both speeds - reading and writing - are compared.")
    put_back = (lambda: None) if glm else ds4_learned_kept(cfg)
    try:
        if glm:
            res = CD.run(alone, both, say=say, start_engine=mixed_start_engine, other_label="the AMD GPU alone",
                         draft_label="both GPUs", size=G53_DRAFT_PROBE)
        else:
            try:
                res = CD.run_phases(alone, both, say=say, start_engine=mixed_start_engine, between=put_back,
                                    close=mixed_close)
            finally:
                put_back()
    except Exception as e:                             # never stops a start: the config stays as it was
        warn(f"the both-GPU measurement did not finish ({e}): the model runs on the AMD GPU alone; "
             f"{ME} --speed-calibration tries again")
        cfg.setdefault("installer", {})["mixed_gpus_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                                                                     "failed": str(e)[:300]}
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        return False
    if not glm:
        o, d = res["other"]["write_tok_s"], res["both"]["write_tok_s"]
        ro, rd = res["other"]["read_tok_s"], res["both"]["read_tok_s"]
        faster = res["gain"] > LINK_TIE and d >= o
        cfg["mixed_gpus"] = faster and DS4_BOTH_AUTO
        cfg.setdefault("installer", {})["mixed_gpus_calibration"] = {
            "date": time.strftime("%Y-%m-%d %H:%M"), "version": maya_version(), "both_tok_s": d, "alone_tok_s": o,
            "both_read_tok_s": rd, "alone_read_tok_s": ro, "doc_tokens": res["other"]["doc_tokens"],
            "gain": res["gain"], "faster": faster}
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        say()
        ok(f"speed check: on both GPUs it reads {rd:.0f} tokens/s and writes {d:.1f}, on the AMD GPU alone {ro:.0f} "
           f"and {o:.1f} ({100.0 * res['gain']:+.0f}% over both) - " +
           ("Maya runs on both GPUs (--mixed-gpus off turns it off)" if cfg["mixed_gpus"] else
            "both GPUs are available, off until their answers are verified: Maya stays on the AMD GPU alone "
            "(--mixed-gpus on turns both on)" if faster else
            "Maya stays on the AMD GPU alone (--mixed-gpus on turns both on)"))
        return True
    o, d = res["other"]["tok_s"], res["dflash2"]["tok_s"]
    cfg["mixed_gpus"] = d > o * (1 + LINK_TIE)
    cfg.setdefault("installer", {})["mixed_gpus_calibration"] = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "version": maya_version(), "both_tok_s": d, "alone_tok_s": o}
    try:                                             # (the both-GPU run's part of the log: after its settings line)
        with open(log_path, "rb") as f:
            f.seek(log_at)
            text = f.read().decode("utf-8", "replace")
        pick = GL.draft_pick(text.rsplit("MAYA_GLM_EXPERT_HELPER=link", 1)[-1]) if "MAYA_GLM_EXPERT_HELPER=link" in text else None
    except (OSError, TypeError):
        pick = None
    if pick is not None:
        cfg["installer"]["mixed_gpus_calibration"]["mtp_draft"] = pick
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    say()
    ok(f"speed check: {d:.1f} tokens/s on both GPUs against {o:.1f} on the AMD GPU alone - " +
       ("Maya runs on both GPUs (--mixed-gpus off turns it off)" if cfg["mixed_gpus"] else
        "Maya stays on the AMD GPU alone (--mixed-gpus on turns both on)"))
    return True


def needs_prefill_calibration(cfg: dict) -> bool:
    """A MiniMax config whose prompt settings were never measured on this PC (set up before there was the prefill
    calibration): its next start measures them once (a failed one is not tried again by itself: --calibrate)."""
    return config_family(cfg) == "minimax" and "prefill_calibration" not in (cfg.get("installer") or {})


def prefill_calibration(cfg_path: Path, why: str = "") -> bool:
    """MiniMax's prefill calibration (tools/calibrate_mm3_prefill.py): the chunk a prompt is read in and the dense GEMMs'
    backend, measured on this PC by the engine itself and kept by it for this PC (the pack's prefill-tune.txt, read at
    every start); the result also in the config (installer.prefill_calibration).  False: not measured (the reason is
    shown; the engine then reads prompts with its defaults)."""
    cfg = read_json(cfg_path)
    if busy_skip(cfg, why == "asked for", "the prefill calibration", "the GPU(s)",
                 f" (a model server still running?) - {ME} --calibrate measures it later"):
        return False
    say()
    say(f"  Prefill calibration: how this PC reads prompts fastest" + (f" ({why})" if why else ""))
    say("  The engine reads a ~4K-token text with a few settings - the chunk of tokens a step takes, and the matrix")
    say("  products' library - and keeps the fastest for this PC. One start of the model (two where most experts stay")
    say("  in RAM: a larger chunk's decode is checked against the default's); a few minutes.")
    log = cfg.get("log")
    since = os.path.getsize(log) if log and os.path.isfile(log) else 0
    try:
        res = CPF.run(cfg, say=say)
    except Exception as e:                             # never stops a setup or a start: the engine's defaults stay
        warn(f"the prefill calibration did not finish ({e}): the engine reads prompts with its defaults; "
             f"{ME} --calibrate tries again")
        why_not = CAL.engine_error(log, since)
        if why_not:
            say(f"       the engine said: {why_not}")
        cfg.setdefault("installer", {})["prefill_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                                                                  "failed": str(e)[:300]}
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        return False
    cfg.setdefault("installer", {})["prefill_calibration"] = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "version": maya_version(), "chunk": res.get("chunk"),
        "gemm": res.get("gemm"), "tok_s": res.get("tok_s"), "default_tok_s": res.get("default_tok_s"),
        "tokens": res.get("tokens"), "file": res.get("file")}
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    t, d = res.get("tok_s") or 0, res.get("default_tok_s") or 0
    gemm = {"hipblaslt": "hipBLASLt", "blas": "the BLAS's"}.get(res.get("gemm"), res.get("gemm"))
    gain = f" (+{(t / d - 1) * 100:.0f}% over the defaults' {d:.0f})" if d > 0 and t > d * 1.005 else \
        " (the engine's defaults are the fastest here)"
    ok(f"prefill calibration: chunks of {res.get('chunk')} tokens, {gemm} matrix products - {t:.0f} tokens/s "
       f"reading a {res.get('tokens')}-token prompt{gain}; kept for this PC")
    return True


def speed_calibration_ds4(cfg_path: Path, cfg: dict, why: str = "") -> bool:
    """DeepSeek's and MiniMax's speed calibration (tools/calibrate_ds4.py): the sustained decode speed with the DSpark
    drafter and without it (--dspark none / off), measured on this PC's GPUs; the config keeps the faster - the drafter
    only when it wins by more than CDS.MIN_GAIN - and the result (installer.speed_calibration)."""
    if config_family(cfg) == "minimax" and parallel_of(cfg) > 1:
        warn("with --parallel above 1 maya-mm3 decodes without the DSpark drafter: no speed calibration")
        return False
    if not CDS.has_drafter(cfg):
        warn(f"the speed calibration compares the model's DSpark drafter with no drafting, and this pack has no "
             f"drafter: {ME} --setup downloads it")
        return False
    if busy_skip(cfg, why == "asked for", "the speed calibration", "the GPU(s)",
                 f" (a model server still running?) - it runs at the next start, or: {ME} --speed-calibration"):
        return False
    say()
    say("  Speed calibration: the DSpark drafter or none?" + (f" ({why})" if why else ""))
    say("  Drafting makes some PCs faster and others slower: it depends on how much of the model fits in the GPUs'")
    say("  memory, and on the CPU, the RAM and the SSD. So Maya measures both here, the way you will use it. It loads")
    say(f"  the model twice; each time it warms up, then writes about {CDS.MIN_TOKENS} tokens of chat, code and tool")
    say("  calls alike (drafting pays most on code), and the combined speeds are compared. About 10-30 minutes; the")
    say("  PC is busy.")
    log = cfg.get("log")
    since = os.path.getsize(log) if log and os.path.isfile(log) else 0
    try:
        res = CDS.run(cfg, say=say)
    except Exception as e:                             # never stops a setup or a start: the config stays as it was
        warn(f"the speed calibration did not finish ({e}): nothing changed; {ME} --speed-calibration tries again")
        why_not = CAL.engine_error(log, since)
        if why_not:
            say(f"       the engine said: {why_not}")
        cfg.setdefault("installer", {})["speed_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                                                                "failed": str(e)[:300]}
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        return False
    o, d = res["none"]["tok_s"], res["dspark"]["tok_s"]
    keep = res["keep"] == "dspark"
    out = CDS.with_dspark(cfg, "on" if keep else "none")
    out.setdefault("installer", {})["speed_calibration"] = {
        "date": time.strftime("%Y-%m-%d %H:%M"), "version": maya_version(), "dspark_tok_s": d, "no_drafting_tok_s": o,
        "drafts_accepted": [res["dspark"]["drafts_accepted"], res["dspark"]["drafts_offered"]],
        "tokens": [res["none"]["tokens"], res["dspark"]["tokens"]], "kept": "dspark" if keep else "none",
        "kinds": {k: [res["none"]["kinds"][k]["tok_s"], res["dspark"]["kinds"][k]["tok_s"]]
                  for k in res["dspark"].get("kinds") or {}}}
    cfg_path.write_text(json.dumps(N.env_out(out), indent=1), encoding="utf-8")
    say()
    each = ", ".join(f"{name} {res['dspark']['kinds'][k]['tok_s']:.1f} against {res['none']['kinds'][k]['tok_s']:.1f}"
                     for k, name, _, _ in CDS.KINDS if k in (res["dspark"].get("kinds") or {}))
    if keep:
        ok(f"speed calibration: the DSpark drafter is faster on this PC - {d:.1f} against {o:.1f} tokens/s "
           f"(+{res['gain'] * 100:.0f}%, chat, code and tool calls combined): Maya drafts with it")
    else:
        ok(f"speed calibration: the DSpark drafter is not faster on this PC - {d:.1f} against {o:.1f} tokens/s "
           "(chat, code and tool calls combined): Maya decodes without drafting (and its memory holds more of the "
           "model)")
    if each:
        say(f"       (each kind with the drafter against without: {each} tokens/s)")
    return True


def set_fast_prefill(cfg: dict, value: str | None) -> None:
    """--fast-prefill on|off into a DeepSeek config: "fast_prefill": true (serve/server.py's child_env gives the engine
    DS4_YOCO_CONE=1, the long prompt's skip as a per-layer cone), or left out (off, the default).  None: unchanged."""
    if value == "on":
        cfg["fast_prefill"] = True
        ok("fast prompt reading: on (--fast-prefill; long prompts read faster, answers can shift slightly)")
    elif value == "off":
        cfg.pop("fast_prefill", None)
        ok("fast prompt reading: off (--fast-prefill off; the default)")


def mixed_gpus_value(v: str) -> str:
    """--mixed-gpus: on, off, helper, split, or a layer count for the NVIDIA card (argparse's type)."""
    if v in ("on", "off", "helper", "split") or (v.isdigit() and int(v) > 0):
        return v
    raise argparse.ArgumentTypeError("on, off, helper, split or a number of layers")


def nvidia_beside() -> bool:
    """An NVIDIA card answers nvidia-smi (a setup on an AMD GPU then mentions --mixed-gpus; nothing is turned on)."""
    try:
        return "GPU 0" in (S.out(["nvidia-smi", "-L"]) or "")
    except Exception:  # noqa: BLE001 - no nvidia-smi: none
        return False


def set_ds4_mixed_gpus(cfg: dict, value: str | None) -> None:
    """--mixed-gpus on|off|<layers> into a DeepSeek config on an AMD GPU: "mixed_gpus": true (or the layer count) or
    false; left out: on by itself where an NVIDIA card and a CUDA build are found (Linux).  serve/mixed_gpus.py then
    starts a CUDA engine on the NVIDIA card beside it for the first layers (sized from its free VRAM at each start) and
    watches both.  None: unchanged."""
    if value is None:
        return
    if value == "off":
        cfg["mixed_gpus"] = False
        ok("two GPUs: off (--mixed-gpus off: the model runs on the AMD GPU alone; --mixed-gpus on turns it on again)")
        return
    if cfg.get("backend") != "hip":
        warn("--mixed-gpus is for a DeepSeek config on an AMD GPU with an NVIDIA card beside it: not this config")
        return
    if value in ("helper", "split"):   # (the way the two GPUs share the model: serve/mixed_gpus.py's mixed_mode)
        cfg["mixed_mode"] = value
        ok("two GPUs: " + ("the model on the AMD GPU, the NVIDIA card computes its most-routed experts for it"
                           if value == "helper" else "the NVIDIA card runs the first layers") + " (mixed_mode)")
        value = "on"
    if value != "on":                  # (a layer count: the split layout)
        cfg["mixed_mode"] = "split"
    cfg["mixed_gpus"] = True if value == "on" else int(value)
    exe = Path(cfg.get("mixed_exe") or Path(cfg["exe"]).resolve().parents[1] / "build-cuda" / Path(cfg["exe"]).name)
    if str(cfg.get("mixed_mode") or "helper") == "split":
        ok("two GPUs: on (--mixed-gpus; the NVIDIA card runs the first layers - " +
           ("as many as its free VRAM holds" if value == "on" else f"{value} of them") +
           "; answers can be worded slightly differently from one GPU alone; a request with pictures, "
           "log-probabilities or refusal ablation runs on the AMD GPU alone; --mixed-gpus off turns it off)")
    else:
        ok("two GPUs: on (--mixed-gpus; the model on the AMD GPU, the NVIDIA card computes its most-used experts; "
           "answers can be worded slightly differently from one GPU alone; --mixed-gpus off turns it off)")
    if not exe.is_file():
        warn(f"two GPUs: no CUDA build of maya-ds4 at {exe} - {ME} --setup builds it (or set mixed_exe in the "
             "config); until then the model runs on the AMD GPU alone")


def measure_asked(a) -> str:
    """A setup's speed calibration asked for by name (--speed-calibration, or --drafting measure): "asked for" - it
    then runs on a GPU that already holds some memory (a display, #794) - else "" (the setup's own: it skips there)."""
    return "asked for" if getattr(a, "speed_calibration", False) or getattr(a, "drafting", None) == "measure" else ""


def drafting_choice(a, drafter: str, minutes: str) -> str:
    """How a setup settles its drafter: "measure" (the speed calibration), or the user's own choice without
    measuring - "on" (draft with it) or "off".  --drafting, or --no-speed-calibration (= on), else asked; --yes
    measures."""
    if getattr(a, "drafting", None):
        return "on" if a.drafting in ("mtp", "dfly2") else a.drafting   # (Hy4-Flash's drafter by name: elsewhere on)
    if getattr(a, "no_speed_calibration", False):
        return "on"
    say()
    say(f"  {drafter}: Maya can measure on this PC whether drafting makes answers faster ({minutes} minutes, the")
    say("  model loaded twice), or you choose now and skip the measuring.")
    r = ask("  m = measure, y = draft with it, n = no drafting (m/y/n)", ["m", "y", "n"], "m", a.yes)
    return {"m": "measure", "y": "on", "n": "off"}[r]


def keep_drafting(cfg_path: Path, on: bool) -> None:
    """The user's choice instead of the speed calibration: the config drafts with its drafter (on) or not, and
    installer.speed_calibration records the choice - so a start does not measure by itself (--speed-calibration
    measures any time)."""
    cfg = read_json(cfg_path)
    if g53_one_gpu_drafting(cfg):                          # (GLM-5.3 on one APU: one-GPU drafting on or off)
        out = g53_with_drafting(cfg, on)
        out.setdefault("installer", {})["speed_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                                                                "version": maya_version(), "skipped": "chosen",
                                                                "kept": "mtp" if on else "none"}
        cfg_path.write_text(json.dumps(N.env_out(out), indent=1), encoding="utf-8")
        ok("speed calibration skipped: Maya " + ("drafts with the MTP block on this GPU" if on else
                                                 "decodes without drafting") + ", as chosen")
        say(f"       ({ME} --speed-calibration measures which is faster on this PC any time)")
        return
    ds4 = config_family(cfg) in ("deepseek", "minimax")   # (DSpark: DeepSeek's and MiniMax's drafter)
    if config_family(cfg) == "qwen":                       # (CYBER-FROST: the MTP drafter in its pack, or none)
        args = cfg.get("args") or []
        if "--mtp-max-t" in args[:-1]:
            i = args.index("--mtp-max-t")
            del args[i:i + 2]
        if "--mtp" not in args[:-1]:   # (the server needs the drafter loaded either way)
            args += ["--mtp", str(Path(args[args.index("--pack") + 1]) / "mtp")]
        if not on:
            args += ["--mtp-max-t", "1"]
        cfg.setdefault("installer", {})["mtp_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                                                               "skipped": "chosen", "kept": "mtp" if on else "none"}
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        ok(f"speed calibration skipped: Maya {'drafts with the MTP drafter' if on else 'decodes without drafting'}, "
           "as chosen")
        return
    if ds4:
        out, kept = CDS.with_dspark(cfg, "on" if on else "none"), ("dspark" if on else "none")
    else:
        out = cfg if on else CD.without_dflash(cfg)
        kept = out.setdefault("installer", {})["draft"] = "dflash2" if on else "mtp"
    out.setdefault("installer", {})["speed_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"),
                                                            "version": maya_version(), "skipped": "chosen",
                                                            "kept": kept}
    cfg_path.write_text(json.dumps(N.env_out(out), indent=1), encoding="utf-8")
    name = "the DSpark drafter" if ds4 else "DFlash2"
    if on:
        ok(f"speed calibration skipped: Maya drafts with {name}, as chosen")
    elif ds4:
        ok("speed calibration skipped: Maya decodes without drafting, as chosen")
    else:
        one = len(cfg.get("gpu") or [0]) < 2
        ok("speed calibration skipped: Maya " + ("decodes without drafting" if one else "drafts with the MTP block") +
           ", as chosen")
    say(f"       ({ME} --speed-calibration measures which is faster on this PC any time)")


def server_command(cfg_path: Path, a) -> tuple:
    """The dashboard's server for an installed config: (its command, its environment, what the setup's screen shows of
    it).  The engine is compiled again first when its source changed (an update)."""
    cfg = read_json(cfg_path)
    check_parallel(a, config_family(cfg))
    rename_programs()
    renamed = fix_config_names(cfg)                    # (v1.1.16's names: the programs, and the settings as MAYA_*)
    try:
        old_env = any(str(k).startswith(N.OLD) for k in
                      json.loads(cfg_path.read_text(encoding="utf-8-sig")).get("env") or {})
    except (OSError, ValueError, AttributeError):
        old_env = False
    if renamed or old_env:
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    if renamed:
        ok(f"{cfg_path.name}: the programs are named after Maya now ({Path(cfg['exe']).name})")
    if getattr(a, "parallel", None):                   # --parallel given at a start: kept in the config
        set_arg(cfg.setdefault("args", []), "--parallel", a.parallel)
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    select_build_backend(cfg.get("backend", "cuda"))
    args = cfg.get("args") or []
    ds4 = config_family(cfg) == "deepseek"
    metal = cfg.get("backend") == "metal"
    ik2 = config_family(cfg) == "ik2"                  # (maya-ik2 reads its GGUF itself: --model, no pack)
    flag = ("--gguf" if metal else "--glm-pack") if config_family(cfg) == "glm" else "--model" if ik2 else "--pack"   # (maya-ds4 and maya-mm3: --pack)
    pack = Path(args[args.index(flag) + 1]) if flag in args[:-1] else None       # (Metal: the first GGUF file)
    for p, what in ((Path(cfg.get("exe", "")), "the engine"),
                    (pack, "the model's GGUF file" if metal or ik2 else "the pack"),
                    (Path(cfg.get("tokenizer", "")) / "vocab.json", "the tokenizer")):
        if p is None or not p.exists():
            fail(f"{cfg_path.name}: {what} is missing ({p})", f"run {ME} --setup to repair it")
    # a model file changed after its pack was built (pack_stale, #155): the pack's places in it are wrong
    if (ds4 or config_family(cfg) == "hy4") and pack is not None and (pack / PACK_SOURCES).exists():
        rec = {r.get("name") for r in read_json(pack / PACK_SOURCES).get("files") or []}
        files = pack_index_files(pack)                     # the files the pack's index itself reads: a shared Engram
        if files is None:                                  # set (or drafter, vision) lives beside the model folder, so
            files = ds4_shards(pack.parent)                # match those, not a same-named file an earlier layout
        why = pack_stale(pack, [s for s in files if s.exists() and (s.name in rec or not rec)])   # left there
        if why is not None:
            fail(f"{cfg_path.name}: {why}, so its index of where the tensors sit is wrong",
                 f"run {ME} --setup --repack to build it again (a minute)")
    if metal:                                          # (no pack: the engine reads every shard of the model itself)
        for s in shard_set(pack):
            if not s.exists():
                fail(f"{cfg_path.name}: the model's file {s.name} is missing ({s.parent})", f"run {ME} --setup to repair it")
    src = None if metal or ik2 else pack_source(pack)
    if src is not None and src != EXL3_GGUF and all(s.exists() for s in shard_set(pack.parent / src)):
        why = pack_stale(pack, shard_set(pack.parent / src))
        if why is not None:
            fail(f"{cfg_path.name}: {why}, so its index of where the tensors sit is wrong",
                 f"run {ME} --setup --repack to build it again (a few minutes)")
        if not (pack / PACK_SOURCES).exists():
            stamp_pack(pack, shard_set(pack.parent / src))
    if add_profile(cfg):                               # a config written before the expert profile (an update)
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    stale = stale_tuning(cfg)                          # tuned before a driver / P2P change: measured again (#419)
    if stale is not None:
        cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
        if stale:
            warn(f"{cfg_path.name}: {stale}")
        ok(f"{cfg_path.name}: starts from the shipped expert profile now (--expert-profile {PROFILE})")
    if ds4 and ds4_seed_profile(cfg):                  # (DeepSeek's: a pack that has not learned one yet)
        ok(f"{cfg_path.name}: starts from DeepSeek's shipped expert profile ({DS4_PROFILE.name}, copied into the pack)")
    cfg_path.touch()                                   # the most recently used model
    refresh_engine(cfg)
    refresh_mixed_builds(cfg)                          # (the two-GPU mode's second engine, if the setup built it)
    unattended = unattended_start(a)
    if unattended:                                     # nobody to wait: no calibration (it loads the model two more times)
        skip_start_calibrations(cfg_path, cfg, unattended)
    if not unattended and needs_prefill_calibration(cfg):   # MiniMax set up before the prefill calibration: measured once
        prefill_calibration(cfg_path, "once, for a setup from before it")
        cfg = read_json(cfg_path)
        args = cfg.get("args") or []
    if not unattended and needs_speed_calibration(cfg):     # a drafter set up without its speed calibration: measured once
        drafter = {"deepseek": "DeepSeek's DSpark drafter", "minimax": "MiniMax's DSpark drafter",
                   "hy4": "the MTP block or the DFly2 drafter"}.get(
            config_family(cfg), "DFlash2")
        if speed_calibration(cfg_path, f"once, for a setup that drafts with {drafter}"):
            cfg = read_json(cfg_path)
            args = cfg.get("args") or []
    if not unattended and needs_link_calibration(cfg):      # an NVIDIA card beside the AMD APU: both GPUs, measured once
        if link_calibration(cfg_path, "once, for this PC"):
            cfg = read_json(cfg_path)
            args = cfg.get("args") or []
    # Hy4-Flash's 4-bit expert copies: measured by the engine once its experts are warm, remembered per PC (no start of
    # its own for it any more); a verdict an earlier start's check wrote into the config is dropped for the engine's
    cfg = q4x_config_migrate(cfg_path, cfg)
    args = cfg.get("args") or []
    port = a.port or cfg.get("port") or 8080
    host = a.host or cfg.get("host") or "127.0.0.1"
    key = a.api_key or cfg.get("api_key")
    cmd = [sys.executable, str(ROOT / "serve" / "server.py"), "--engine", "maya", "--config", str(cfg_path),
           "--port", str(port)]
    if a.host:
        cmd += ["--host", a.host]
    if a.api_key:
        cmd += ["--api-key", a.api_key]
    if a.gpus or a.gpu is not None:                    # this start only, on these cards
        cmd += ["--gpu", str(a.gpus or a.gpu)]
    from serve.server import web_ui_setting            # ("web_ui": false: an API-only server, no page to open - #1089)
    web = web_ui_setting(cfg)
    if web and (WIN or sys.platform == "darwin" or os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")):
        cmd.append("--open")                           # a desktop: the browser opens when the model is ready
    here = "127.0.0.1" if host in ("0.0.0.0", "::", "") else host
    ctx = args[args.index("--max-context") + 1] if "--max-context" in args[:-1] else None
    gpus = str(a.gpus or a.gpu if a.gpus or a.gpu is not None else cfg.get("gpu", 0)).strip("[]").split(",")
    info = {"model": cfg.get("model_name", MODEL_NAME), "quant": (cfg.get("installer") or {}).get("quant"),
            "label": Q.config_label(cfg),                # ("MiniMax M3.1 · Maya-XS": what the screen's card shows)
            "context": ctx, "dashboard": f"http://{here}:{port}/", "api": f"http://{here}:{port}/v1",
            "log": cfg.get("log"), "key": (key or "").split(",")[0],      # (its dashboard reads /metrics with it)
            "gpus": [int(g) for g in gpus if g.strip().isdigit()]}         # (its load: one part each, in this order)
    env = dict(os.environ, MAYA_RESTART_ON_UPDATE="1")  # (the dashboard may update Maya)
    if S.UI is not None:                               # (the setup's screen shows the same in its card, and reads
        env["PYTHONUNBUFFERED"] = "1"                  # the server's output from a pipe)
        found = S.amd_gpus() if cfg.get("backend") == "hip" else metal_gpus() if metal else S.gpus()
        info["vram"] = {g["index"]: g["vram_gb"] for g in found}   # (its load: a GPU not started yet, by its VRAM)
        return cmd, env, info
    say()
    say("  " + "-" * 100)
    say(f"  Starting {info['label'] or cfg.get('model_name', MODEL_NAME)} ({cfg_path.name}).")
    say(f"  Dashboard:        http://{here}:{port}/" if web else "  Dashboard:        off (\"web_ui\": false: the API only)")
    say(f"  API (OpenAI):     http://{here}:{port}/v1        (any model name; the API key: "
        + ("the one you set)" if key else "any)"))
    say(f"  API (Anthropic):  http://{here}:{port}/v1/messages")
    if host not in ("127.0.0.1", "localhost", "::1"):
        say("  Other devices:    the server prints this machine's addresses when it is ready" +
            ("" if key else " - WARNING: no API key, anyone on your network can use it (--api-key KEY)"))
    if config_family(cfg) == "minimax":
        say("  Loading takes a few minutes: the engine places the experts in VRAM by how often they are used and keeps")
        say("  the rest in pinned RAM for its CPU lane - the first answers are the slowest.")
    elif config_family(cfg) == "hy4":
        say("  Loading takes a minute or a few: the engine places what must sit in VRAM and as many of the most-routed")
        say("  experts as fit beside it; the others are computed on the CPU from the model file (RAM as the page cache),")
        say("  the n-gram table is read from the SSD - the first answers are the slowest.")
        for line in hy4_first_start_said(cfg):
            say("  " + line)
    elif config_family(cfg) == "ik2":
        say(f"  {IK2_SHOWN} is an experimental preview: loading takes a few seconds to a minute (every expert is")
        say("  copied into VRAM); one answer at a time" + (", pictures read by the image encoder when one arrives."
                                                          if cfg.get("vision") else ", text only."))
    elif config_family(cfg) == "qwen":
        say("  Loading takes a few minutes: the engine places the most-used experts in VRAM, the next ones in RAM as far")
        say("  as the free RAM goes (MAYA_EXPERT_RAM_GB caps it) and reads the rest from the SSD when they are routed.")

    elif ds4:
        say("  Loading takes a few minutes: the engine places the experts in VRAM by how often they are used, copies the")
        say("  rest into RAM for its CPU lane and pins them - the first answers are the slowest.")
    elif metal:
        say("  Loading takes a few minutes: the engine reads the GGUF files; with less unified memory than the model the")
        say("  experts stream from the SSD while it answers (decode speed is SSD-bound) - the first answers are the slowest.")
    else:
        say("  Loading takes a few minutes: the engine pins up to all but about 6 GB of the free RAM for its expert tier")
        say("  (MAYA_GLM_RAM_HEADROOM_GB changes the 6) and warms its caches - the first answers are the slowest.")
    say("  Ctrl+C (or closing this terminal) stops it. Engine log: " + str(cfg.get("log", "")))
    say("  " + "-" * 100)
    return cmd, env, info


def start(cfg_path: Path, a) -> int:
    """Maya's server for an installed config: on a screen of its own in a terminal (tools/setup_tui.py: its
    dashboard - the Monitor's numbers from the engine - and its log), else in the terminal as it prints."""
    if setup_screen(a, asks=False):
        import setup_tui
        rc = setup_tui.run(lambda: serve_on_screen(cfg_path, a), maya_version(), None, ME, steps=False)
        return after_server(rc, a)
    cmd, env, _ = server_command(cfg_path, a)
    return after_server(run_server(cmd, env=env), a)


STOP_WAIT_S = 150   # a stop's longest wait for the server: its request in progress (40 s) and the engine's QUIT (50 s)


def run_server(cmd, env=None, wait_s: float = STOP_WAIT_S) -> int:
    """Maya's server in this terminal until it ends: its exit code.  It ends with this process, however this one
    ends: its stdin is a pipe from here (--parent-pipe), so when this process ends - Ctrl+C, SIGTERM or SIGHUP (a
    service manager, `kill`, the terminal closed: KeyboardInterrupt here, stop_signals), a crash, even SIGKILL - the
    pipe closes and the server stops as on Ctrl+C: the request in progress answers, the engine gets QUIT and keeps its
    conversation.  On Windows a job object also ends the server and everything it started at once when this process
    ends without the time for that (setup.end_with_me: the console closed, taskkill /F).  A stop here closes the pipe
    and waits for the server (up to `wait_s` seconds, then it is ended)."""
    p = subprocess.Popen(list(cmd) + ["--parent-pipe"], env=env, stdin=subprocess.PIPE)
    S.end_with_me(p)
    deadline = None
    while True:
        try:
            return p.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            if deadline is not None and time.monotonic() > deadline:
                say(f"  (the server did not end within {wait_s:.0f} s of the stop: ended)")
                p.kill()
                return p.wait()
        except KeyboardInterrupt:                      # (the server gets Ctrl+C itself; a signal here: the pipe says it)
            if deadline is None:
                deadline = time.monotonic() + wait_s
                try:
                    p.stdin.close()
                except OSError:
                    pass


def stop_signals() -> None:
    """SIGTERM and SIGHUP (and Windows' SIGBREAK: Ctrl+Break, the console closed) stop maya.py as Ctrl+C does - a
    KeyboardInterrupt - so a service manager, `kill <pid>` or a closed terminal stop the setup, or the server it runs
    (run_server), the same way.  A signal that is ignored (nohup) stays ignored."""
    import signal

    def stop(signum, frame):
        raise KeyboardInterrupt
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        sig = getattr(signal, name, None)
        try:
            if sig is not None and signal.getsignal(sig) is not signal.SIG_IGN:
                signal.signal(sig, stop)
        except (OSError, RuntimeError, ValueError):     # (not the main thread: an embedding caller)
            pass


def serve_on_screen(cfg_path: Path, a) -> int:
    """Maya's server on the setup's screen (setup.UI): its exit code; one it did not choose stops there, with the
    last line its engine logged in this start (mostly the reason: "strata generate: pack: out of memory")."""
    S.model_named(setup_title(read_json(cfg_path)))
    cmd, env, info = server_command(cfg_path, a)
    log = Path(info["log"]) if info.get("log") else None
    seen = log.stat().st_size if log is not None and log.is_file() else 0
    rc = S.UI.serve(cmd, env, info)
    if rc not in (0, UPDATE_EXIT, RESTART_EXIT):
        last = last_line(log, seen)
        fail(f"Maya stopped: its server ended with exit code {rc}",
             (f"its engine's last line: {last} (the whole log: {log})" if last else "the reason is in its output")
             + f"; {ME} starts it again")
    return rc


def last_line(path: Path | None, since: int) -> str:
    """The last line written to `path` after byte `since` ("": none, or no such file)."""
    try:
        with open(path, "rb") as f:
            f.seek(max(since, f.seek(0, os.SEEK_END) - 8192))
            lines = [s.strip() for s in f.read().decode(errors="replace").splitlines() if s.strip()]
    except (OSError, TypeError):
        return ""
    return lines[-1][:200] if lines else ""


def after_server(rc: int, a) -> int:
    """The server ended: its exit code - or About > Updates moved this folder to a new release, and the new version
    starts: its maya.py compiles what changed in the engine and loads the same model (the most recently used, so no
    question)."""
    if rc not in (UPDATE_EXIT, RESTART_EXIT):
        return rc
    again = [sys.executable, str(HERE / "maya.py"), "--yes"]
    # (a new port from the dashboard: not this start's --port - the config's, which it saved, applies)
    port = None if rc == RESTART_EXIT else a.port
    backend, config = a.backend, getattr(a, "config", None)
    # the Models page: another model to start (a switch), or this one with its new engine settings - the server's
    # note says which config, on the port it listened on (so the page finds it again)
    note = read_json(ROOT / SWITCH_FILE) if rc == RESTART_EXIT else {}
    try:
        (ROOT / SWITCH_FILE).unlink()
    except OSError:
        pass
    why = None
    if note.get("config") and Path(note["config"]).is_file() and time.time() - float(note.get("time") or 0) < 600:
        config, why = Path(note["config"]), note.get("why")
        backend = read_json(config).get("backend", "cuda")
        port = note.get("port") or port
    for flag, v in (("--backend", backend), ("--port", port), ("--host", a.host), ("--api-key", a.api_key),
                    ("--gpu", a.gpu), ("--gpus", a.gpus), ("--config", config)):
        if v is not None:
            again += [flag, str(v)]
    say()
    say("  Updated from the dashboard: starting the new version ..." if rc != RESTART_EXIT else
        f"  Switching to {config.name} from the dashboard ..." if why == "switch" else
        f"  {config.name}'s engine settings changed from the dashboard: starting Maya again with them ..."
        if why == "settings" else "  The port changed from the dashboard: starting Maya again on it ...")
    sys.stdout.flush()
    if WIN:
        return subprocess.call(again)
    os.execv(sys.executable, again)
    return 0


# ------------------------------------------------------------------------------------------------ the report
# the engine log's lines that tell where the time goes: how the model was split across VRAM / RAM / the SSD, the
# prompt path's chunks, and the per-token breakdown ("glm stat": VRAM hits, RAM fetches, disk reads, CPU lane)
REPORT_LINES = re.compile(r"glm fast:|glm prefill: (?:CUDA|HIP)|glm split|glm stat|glm slots|glm vram|ds4 serve: prompt read|glm prefill: \d|ERR|error|failed|"
                          r"out of memory|"
                          # maya-ds4's placement and speed lines (issue #416), not its loading progress
                          r"^ds4: (?!copying experts |experts \d+ / |loading \d)|^ds4 (?:slots|cache):|^maya-ds4:|"
                          r"^(?:prompt|decode): \d+ tokens in |^decode per token:|^decode (?:reads|left out)|^dspark|"
                          # maya-mm3's: where the model went (its --serve start) and each answer's prompt / decode speed
                          r"^maya-mm3:|^maya mm3 serve: (?:prompt|DSpark) |^maya-hy4:|"
                          # the Monitor's GPU readings: a reading that failed, a reader that stopped and why
                          r"^\[maya\] (?:a GPU reading failed|the GPU readings)",
                          re.I)
STAT_DECODE = re.compile(r"glm stat: decode ([\d.]+) ms/tok")


def speed_lines(text) -> list:
    """The log lines REPORT_LINES picks.  A request that only read a prompt (one token out) has no decode to report:
    its "glm stat" line keeps only the prompt part, not a meaningless decode speed (96,000 tokens/s, issue #8)."""
    picked = []
    for x in text:
        if not REPORT_LINES.search(x.strip()) or "warming the expert tiers" in x:
            continue
        m = STAT_DECODE.search(x)   # no decode: under 1 ms a token, or no expert touched (one prompt token out)
        if m and (float(m.group(1)) < 1.0 or "vram hit 0.00%" in x) and "| prompt " in x:
            x = "glm stat (prompt only): prompt " + x.split("| prompt ", 1)[1]
        picked.append(x.strip())
    return picked


def gpu_readings_report() -> str:
    """--report: what the dashboard's Monitor reads on each NVIDIA GPU (serve/telemetry.py's NVML reader, a process of
    its own): each card's readings given or not, any that failed with its error - never an error of the report's own."""
    try:
        from serve import telemetry as TM
        return TM.readings_report()
    except Exception as e:  # noqa: BLE001 - the report goes on without this part
        return f"not read ({type(e).__name__}: {e})"


def hip_gpu_report(rocm: Path, bindir=None) -> str:
    """Static AMD details, with KFD's HIP indices/architectures even when no SMI tool is installed.

    Keep the SMI JSON in its own device order: its DRM indices need not match KFD's HIP indices.
    This also retains driver/VRAM fields across the different versions of the two tools.
    On Windows: ROCm's hipInfo (HIP's indices), else Windows' display adapters.
    """
    import platform
    if WIN:
        exe = hip_info(bindir or rocm / "bin")
        gpus = S.amd_gpus(exe)
        lines = [("hipInfo" if exe else "Windows' display adapters (no hipInfo)") + " (HIP GPU indices):"]
        lines += [f"{gpu_label(g)}; driver {g.get('driver', '?')}" for g in gpus] or ["no AMD GPUs found"]
        lines.append(f"Windows {platform.version()}")
        return "\n".join(lines)
    gpus = S.amd_gpus()
    lines = ["KFD topology (HIP GPU indices):"]
    lines += [f"{gpu_label(g)}; driver {g.get('driver', 'amdgpu')}" for g in gpus]
    if not gpus:
        lines.append("no AMD GPUs found in KFD topology")
    try:
        driver = Path("/sys/module/amdgpu/version").read_text().strip()
    except OSError:
        driver = ""
    lines.append(f"amdgpu driver: {driver or 'in-tree'}, kernel {platform.release()}")
    for name, args in (("rocm-smi", ["--showproductname", "--showmeminfo", "vram", "--showdriverversion", "--json"]),
                       ("amd-smi", ["static", "--json"])):
        exe = shutil.which(name)
        if not exe and (rocm / "bin" / name).is_file():
            exe = str(rocm / "bin" / name)
        if not exe:
            continue
        try:
            data = json.loads(S.out([exe, *args]))
        except ValueError:
            continue
        if isinstance(data, (dict, list)) and data:
            lines += [f"\n{name} (tool GPU indices):", json.dumps(data, indent=2)]
            break
    else:
        lines.append("rocm-smi / amd-smi unavailable or did not answer; using KFD topology")
    return "\n".join(lines)


def rocm_report(rocm: Path) -> str:
    """ROCm's install version file first, then hipcc; reporting never requires the compiler."""
    version = ""
    for name in ("version", "version-dev"):
        try:
            version = (rocm / ".info" / name).read_text().strip()
        except OSError:
            continue
        if version:
            break
    if not version:
        hipcc = next((p for p in (rocm / "bin/hipcc", rocm / "bin/hipcc.exe") if p.is_file()), None)
        if hipcc is not None:
            version = S.out([str(hipcc), "--version"]).strip()
    return f"path {rocm}\nversion {version or 'unknown (ROCm version file / hipcc unavailable)'}"


def report(version: str, backend: str | None = None, cfg_path: Path | None = None) -> int:
    """--report: maya-report.txt in the Maya folder - this PC, the installed setup and the engine log's speed lines,
    for a bug or speed report.  Nothing is sent anywhere; the home folder is written as ~ and API keys are left out."""
    home = str(Path.home())
    lines = []
    have = [cfg_path] if cfg_path is not None else configs()
    cfg = {}
    for p in have:
        candidate = read_json(p)
        if backend is None or candidate.get("backend", "cuda") == backend:
            cfg = candidate
            break
    backend = backend or cfg.get("backend", "cuda")
    select_build_backend(backend)
    stamp = read_json(STAMP)

    def add(title, text=""):
        lines.append(f"## {title}")
        lines.extend(str(text).rstrip().replace(home, "~").splitlines() or ["(nothing)"])
        lines.append("")

    git = S.out(["git", "-C", str(ROOT), "log", "-1", "--format=%h %cd", "--date=short"]).strip()
    add("Maya", f"version {version}, commit {git or '?'}, Python {sys.version.split()[0]}, "
                f"{sys.platform}{' (WSL)' if S.is_wsl() else ''}")
    import platform
    add("System", f"{platform.platform()}")
    if backend == "hip":
        rocm = Path((cfg.get("env") or {}).get("ROCM_PATH") or os.environ.get("ROCM_PATH")
                    or stamp.get("rocm") or "/opt/rocm").expanduser().resolve()
        add("GPUs", hip_gpu_report(rocm, (stamp.get("lib_dirs") or [None])[0] if WIN else None))
        add("ROCm", rocm_report(rocm))
    elif backend == "metal":
        add("GPUs", S.out(["system_profiler", "SPDisplaysDataType"]) or "system_profiler did not answer")
    else:
        add("GPUs", S.out(["nvidia-smi", "--query-gpu=index,name,memory.total,memory.used,driver_version,"
                                         "pcie.link.gen.max,pcie.link.width.current,pcie.link.width.max,power.limit,"
                                         "temperature.gpu", "--format=csv"]) or "nvidia-smi did not answer")
        add("GPU readings (the Monitor's NVML reader, read now)", gpu_readings_report())
    cpu, avx2, avx512 = S.cpu_info()
    total, avail = mem_gb()
    pf = S.page_file_gb()
    add("CPU and RAM", f"{cpu}, {os.cpu_count()} threads, {'AVX-512' if avx512 else 'AVX2' if avx2 else 'no AVX2'}\n"
                       f"RAM {total:.1f} GB, {avail:.1f} GB available now" +
                       (f", page file {pf:.0f} GB" if pf is not None else ""))
    if WIN:
        disks = S.out(["powershell", "-NoProfile", "-Command", "Get-PhysicalDisk | Format-Table -AutoSize "
                                                              "FriendlyName,MediaType,BusType,Size | Out-String -Width 200"])
    else:
        disks = S.out(["lsblk", "-d", "-o", "NAME,MODEL,ROTA,TRAN,SIZE"])
    add("Disks", disks)
    keys = ("backend", "archs", "rocm", "date") if backend == "hip" else ("backend", "version", "date") if backend == "metal" else ("archs", "nvcc", "host_compiler", "date")
    add("Engine build", json.dumps({k: stamp.get(k) for k in keys})
        if stamp else "not compiled yet")
    if not have:
        add("Setup", f"no maya-*.json yet: {ME} has not finished a setup here")
    for c in have:
        cfg = read_json(c)
        args = cfg.get("args") or []
        pack = next((Path(args[args.index(k) + 1]) for k in ("--glm-pack", "--pack") if k in args[:-1]), None)
        ctx = args[args.index("--max-context") + 1] if "--max-context" in args[:-1] else "?"
        d = pack.parent if pack else None
        if config_family(cfg) == "deepseek":   # (maya-ds4 runs its vision tower itself: on unless --vision off, #416)
            images = "off" if "--vision" in args[:-1] and args[args.index("--vision") + 1] == "off" else "on"
        elif config_family(cfg) == "minimax":   # (maya-mm3's too: --vision on when the pack has it, left out without)
            images = "on" if flag_value(args, "--vision") == "on" else "off"
        elif config_family(cfg) == "hy4":       # (maya-hy4's: on whenever its pack has the tower, unless --vision off -
            # the tower in RAM, on the GPU a picture at a time: the config has no "vision" entry of its own)
            images = "off (--vision off)" if flag_value(args, "--vision") == "off" else \
                "on (read a picture at a time)" if pack is not None and hy4_pack_has_vision(pack) else \
                "off (no vision tower in the pack)" if pack is not None and (pack / "hy4.txt").exists() else \
                "? (the pack is not here to look in)"
        elif config_family(cfg) == "ik2":       # (maya-ik2: pictures through maya-vision's Inkling 2 tower, read a
            #  request's pictures at a time - on the GPU where its free VRAM holds the encoder, else on the CPU)
            images = "on (read a request's pictures at a time)" if cfg.get("vision") else \
                "off (Images on the Models page)" if cfg.get("vision_off") else "off (set up without the image encoder)"
        else:
            images = "on" if cfg.get("vision") else "off"
        where = ""
        if d is not None and d.exists():
            gb = sum(p.stat().st_size for p in d.glob("*.gguf")) / 1e9
            where = (f"\nmodel folder {d}: {gb:.1f} GB of GGUF, {shutil.disk_usage(d).free / 1e9:.0f} GB free, "
                     f"{'a spinning hard disk' if rotational(d) else 'not a hard disk'}")
        add(f"Setup {c.name}", f"model {Q.config_label(cfg) or '?'} ({Q.config_key(cfg) or '?'}), context {ctx}, "
                               f"GPUs {cfg.get('gpu')}, images {images}, "
                               f"settings {json.dumps(cfg.get('env') or {})}" + where)
        log = Path(cfg.get("log") or c.with_suffix(".log"))
        if log.exists():
            text = log.read_text(encoding="utf-8", errors="replace").splitlines()
            picked = speed_lines(text)
            add(f"Engine log {log.name}: the speed and memory lines (last 80 of {len(picked)})", "\n".join(picked[-80:]))
            add(f"Engine log {log.name}: the last 25 lines", "\n".join(text[-25:]))
        else:
            add(f"Engine log {log.name}", "not written yet (start Maya once and ask it something)")
    bench_txt = ROOT / "maya-bench.txt"
    if bench_txt.exists():
        add("Benchmark (maya-bench.txt, from --bench)", bench_txt.read_text(encoding="utf-8", errors="replace"))
    out = ROOT / "maya-report.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say()
    ok(f"written: {out}")
    say("  Attach this file to your report (GitHub issue, X, or wherever you are asking). It has this PC's hardware,")
    say("  your Maya setup and the engine's speed lines - no API key, and your home folder is shown as ~.")
    say("  Best: run it right after a slow answer, so the log has that answer's numbers.")
    return 0


# ------------------------------------------------------------------------------------------------ the benchmark
# the same three questions on every machine (greedy, thinking off), after one warm-up answer, so speeds compare
BENCH_TOPICS = [
    "Write a Python function that parses a CSV file into a list of dictionaries, with type hints and a docstring.",
    "Explain how photosynthesis works, step by step, for a high-school student.",
    "Write a complete HTML page with a canvas that draws a bouncing ball animation in JavaScript.",
]


def bench_exe(cfg: dict) -> Path:
    """Which engine binary a HIP config's --bench should relaunch - config_family(cfg)'s target, through the same
    exe_for_target every build step uses, so this can't drift out of sync with either."""
    return exe_for_target({"deepseek": "maya-ds4", "minimax": "maya-mm3", "hy4": "maya-hy4",
                           "ik2": "maya-ik2"}.get(config_family(cfg), "maya"))


BENCH_MIN_ANSWER = 64   # an answer shorter than this (the model stopped early) is listed, not in the decode mean (#549)


def bench_chat(family: str, tokenizer_dir: Path):
    """--bench's prompt for a question, in the model family's own chat format, thinking off: GLM's markers, DeepSeek's
    (#416), MiniMax's, Hy4's and Inkling 2's through their chat templates as the server renders them (#549: MiniMax
    got GLM's markers)."""
    if family == "deepseek":
        return "<｜begin▁of▁sentence｜><｜User｜>{}<｜Assistant｜></think>".format
    if family == "minimax":
        from serve.frontend import ChatTemplate
        path = Path(tokenizer_dir) / "chat_template.jinja"
        tpl = ChatTemplate(path if path.exists() else ROOT / "tools" / "mm3" / "chat_template.jinja")
        return lambda q: tpl.render([{"role": "user", "content": q}], enable_thinking=False)
    if family == "hy4":        # (the GGUF's own template, in the pack: its thinking off)
        from serve.frontend import ChatTemplate
        path = Path(tokenizer_dir) / "chat_template.jinja"
        if not path.exists():
            fail(f"the model's chat template is missing ({path})", f"run {ME} --setup --repack to build the pack again")
        tpl = ChatTemplate(path)
        return lambda q: tpl.render([{"role": "user", "content": q}], enable_thinking=False)
    if family == "ik2":        # (Inkling 2: its template at the lowest effort, the answer's text segment opened -
        from serve.frontend import ChatTemplate   #  the server's thinking off)
        path = Path(tokenizer_dir) / "chat_template.jinja"
        tpl = ChatTemplate(path if path.exists() else ROOT / "tools" / "ik2" / "chat_template.jinja")
        return lambda q: tpl.render([{"role": "user", "content": q}], reasoning_effort="none",
                                    enable_thinking=False) + "<|message_model|><|content_text|>"
    if family == "qwen":       # (Qwen's turns, the think block closed: the answer itself)
        return "<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n".format
    return "[gMASK]<sop><|user|>\n{}<|assistant|>\n</think>".format


def bench_gen_line(n_new: int, ids, multi: bool) -> str:
    """The GEN line --bench sends: an engine that runs several conversations (READY ... multi=<n>, a --parallel config)
    takes a request only with its id (#420, #509: 'GEN needs rid=<id>')."""
    return f"GEN {n_new} temperature=0{' rid=0' if multi else ''} " + ",".join(map(str, ids)) + "\n"


def bench_reply(line: str, multi: bool):
    """A DONE / ERR line's fields as a one-conversation engine writes them (several conversations: DONE <rid> ...,
    the id dropped); None for another request's line or one that is neither."""
    f = line.split()
    if not f or f[0] not in ("DONE", "ERR"):
        return None
    if multi and len(f) > 1 and f[1].isdigit():
        if f[1] != "0":
            return None
        f = f[:1] + f[2:]
    return f


def bench_decode_mean(results) -> tuple[float, int]:
    """The decode mean of --bench's answers and how many it covers: the answers of BENCH_MIN_ANSWER tokens or more (a
    short one - an answer the model ended early - says little about the speed, #549); all of them when none is."""
    dec = [r for r in results if r[0] == "decode"]
    full = [r for r in dec if r[3] >= BENCH_MIN_ANSWER] or dec
    return (sum(r[2] for r in full) / len(full) if full else 0.0), len(full)


def bench(cfg_path: Path, version: str) -> int:
    """--bench: a standard speed test of the installed model on this PC - decode (writing an answer) on three
    questions and prefill (reading a prompt) at 2k and 8k tokens of this folder's docs - through the engine alone, the
    way the dashboard starts it.  Writes maya-bench.txt (--report includes it).  A few minutes."""
    import urllib.request
    cfg = read_json(cfg_path)
    select_build_backend(cfg.get("backend", "cuda"))
    if cfg.get("backend") == "hip":
        cfg["exe"] = str(bench_exe(cfg))
    port = cfg.get("port") or 8080
    try:
        urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2)
        fail(f"Maya is running (port {port}): the benchmark needs the GPU memory it holds",
             "stop it (Ctrl+C in its window), then run --bench again")
    except OSError:
        pass
    sys.path.insert(0, str(ROOT))
    import serve.server as SV                      # the server's own engine command and environment
    import strata_tokenizer as ST
    args = cfg.get("args") or []
    pack = Path(args[args.index("--glm-pack") + 1]) if "--glm-pack" in args[:-1] else None
    tp = Path(cfg.get("tokenizer") or (pack / "tokenizer" if pack else ""))
    if not (tp / "vocab.json").exists():
        fail(f"{cfg_path.name}: the tokenizer is missing ({tp})", f"run {ME} --setup to repair it")
    tok = ST.Tokenizer.from_dir(tp)
    ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args[:-1] else 32768
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) +
                       sorted((ROOT / "bench" / "results").glob("*.md")) if p.exists())
    doc = tok.encode(text)
    lens = [n for n in (2048, 8192) if n + 64 <= ctx and n <= len(doc)]
    log_path = ROOT / "maya-bench.log"
    step(1, f"benchmark: {cfg_path.name} (loading the model first - a minute or a few)")
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as log:
        p = subprocess.Popen([cfg["exe"], "--serve"] + SV.engine_args(cfg), stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, text=True, env=SV.child_env(cfg),
                             cwd=cfg.get("cwd") or str(ROOT), bufsize=1)

        multi = False   # (READY ... multi=<n>: a --parallel config's engine - its requests carry an id)

        def gen(ids, n_new):
            p.stdin.write(bench_gen_line(n_new, ids, multi))
            p.stdin.flush()
            while True:
                line = p.stdout.readline()
                if not line:
                    fail("the engine stopped during the benchmark", f"its log: {log_path}")
                f = bench_reply(line, multi)
                if f is not None:
                    return f

        while True:
            line = p.stdout.readline()
            if not line:
                fail("the engine stopped while loading", f"its log: {log_path}")
            if line.startswith("READY"):
                multi = any(w.startswith("multi=") for w in line.split())
                break
        ok(f"loaded in {time.time() - t0:.0f} s")
        fmt = bench_chat(config_family(cfg), tp)   # (each family's own chat format, thinking off - #416, #549)
        chat = lambda q: tok.encode(fmt(q), parse_special=True)   # noqa: E731
        gen(chat("Say hello in five languages."), 64)          # warm-up: the first answer after a start is slower
        results = []
        for q in BENCH_TOPICS:
            f = gen(chat(q), 256)
            if f[0] == "DONE" and float(f[4]) > 0:
                results.append(("decode", q, int(f[1]) / float(f[4]) * 1000.0, int(f[1])))
                say(f"  decode  {results[-1][2]:6.1f} tokens/s   {q[:60]} ({f[1]} tokens)")
            elif f[0] == "ERR":
                warn("the engine refused a benchmark request: " + " ".join(f[1:]))
        if lens:                                            # warm-up: the first prompt after a start is slower
            gen(doc[-lens[0]:], 1)                          # (cold caches, issue #8); its opening is not reused below
        for i, n in enumerate(lens):
            f = gen(doc[i * 997:i * 997 + n], 1)            # different openings: nothing reused between them
            if f[0] == "DONE" and float(f[3]) > 0:
                results.append(("prefill", f"{n} tokens", n / float(f[3]) * 1000.0))
                say(f"  prefill {results[-1][2]:6.0f} tokens/s   a {n}-token prompt")
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        try:
            p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
    stats = speed_lines(log_path.read_text(encoding="utf-8", errors="replace").splitlines())
    mean, n_mean = bench_decode_mean(results)
    found = S.amd_gpus() if cfg.get("backend") == "hip" else S.gpus()
    if cfg.get("backend") == "hip" and SV.gpu_list(cfg):
        by_index = {g["index"]: g for g in found}
        found = [by_index[i] for i in SV.gpu_list(cfg) if i in by_index]
    gpus = ", ".join(f"{g['name']} {g['vram_gb']:.0f} GB" for g in found) or (
        "the Apple GPU (Metal)" if cfg.get("backend") == "metal" else "?")
    total, _ = mem_gb()
    lines = [f"Project Maya v{version} benchmark, {time.strftime('%Y-%m-%d %H:%M')}",
             f"GPUs: {gpus}; RAM {total:.0f} GB; CPU {S.cpu_info()[0]}",
             f"setup: {cfg_path.name}, context {ctx}, GPUs {cfg.get('gpu')}",
             f"decode (writing the answer): mean {mean:.1f} tokens/s over {n_mean} answers of up to 256 tokens "
             f"(greedy, thinking off)"]
    lines += [f"  {r[2]:6.1f} tokens/s  {r[1]}" + (f" ({r[3]} tokens: not in the mean)" if r[3] < BENCH_MIN_ANSWER and
                                                    n_mean < sum(x[0] == "decode" for x in results) else "")
              for r in results if r[0] == "decode"]
    lines += [f"prefill (reading the prompt): {r[2]:.0f} tokens/s, {r[1]}" for r in results if r[0] == "prefill"]
    lines += ["", "engine lines:"] + [x.replace(str(Path.home()), "~") for x in stats[-40:]]
    out = ROOT / "maya-bench.txt"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    say()
    for x in lines[:4 + len(results)]:
        say("  " + x)
    ok(f"written: {out} - attach it (with {ME} --report) to a speed report")
    return 0


# ------------------------------------------------------------------------------------------------ the setup
TUI_WHEELS = ROOT / "third_party" / "wheels"       # the setup's screen: Textual (MIT) and what it needs, pure Python


def setup_screen(a, asks: bool = True) -> bool:
    """Whether the setup - or Maya, `asks` False: nothing to answer - runs on a screen of its own (tools/setup_tui.py):
    in a terminal, not with --plain (nor a setup with --yes: nobody there to answer).  Its library comes with Maya
    (TUI_WHEELS) and goes into .venv from there the first time: nothing is downloaded.  False: plain text, as before."""
    if getattr(a, "plain", False) or (asks and getattr(a, "yes", False)) or os.environ.get("TERM") == "dumb" or \
            not (sys.stdin.isatty() and sys.stdout.isatty()):
        return False
    wheel = max(TUI_WHEELS.glob("textual-*.whl"), default=None)
    if wheel is None:
        return False
    want = wheel.name.split("-")[1]
    from importlib import invalidate_caches, metadata
    try:
        if metadata.version("textual") == want:
            return True
    except metadata.PackageNotFoundError:
        pass
    if sys.prefix == sys.base_prefix:                  # not Maya's .venv: nothing goes into a system Python
        return False
    say("  Preparing the setup's screen (Textual, from third_party/wheels: nothing is downloaded) ...")
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check", "--no-index",
                        "--find-links", str(TUI_WHEELS), f"textual=={want}"], capture_output=True, text=True)
    if r.returncode != 0:
        why = (r.stderr or r.stdout).strip().splitlines()
        warn("the setup's screen could not be installed, so the setup runs in plain text" +
             (f" ({why[-1]})" if why else ""))
        return False
    invalidate_caches()
    return True


# the installed configs as they were when this setup run began ({resolved path: text}): ablate_step reads the config
# before from it, and set_up keeps a copy of each config the run changed next to it (keep_config_backups)
SETUP_BEFORE: dict = {}
# what a setup run may change in a config's installer record without a copy kept: when it was written, the record of
# the refusal-direction choice (the choice itself shows in the args)
CONFIG_VOLATILE = ("written", "ablate")


def args_map(args) -> dict:
    """An engine argument list as {flag: value, or True for a flag without one} (order and paths left as they are)."""
    out, args = {}, [str(x) for x in args or []]
    for i, x in enumerate(args):
        if x.startswith("--"):
            out[x] = args[i + 1] if i + 1 < len(args) and not args[i + 1].startswith("--") else True
    return out


def config_changes(old: dict, new: dict) -> list:
    """What a setup run changed in a config, in words: GPUs, engine arguments, environment, other settings (its
    installer record's date left out) - [] when nothing that matters changed."""
    lines = []
    if old.get("gpu") != new.get("gpu"):
        lines.append(f"GPUs {old.get('gpu')} -> {new.get('gpu')}")
    oa, na = args_map(old.get("args")), args_map(new.get("args"))
    short = lambda v: "" if v is True else f" {v}" if len(str(v)) <= 40 else " ..." + str(v)[-30:]   # noqa: E731
    lines += [f"{f} added" for f in na if f not in oa]
    lines += [f"{f} removed" for f in oa if f not in na]
    lines += [f"{f}{short(oa[f])} ->{short(na[f])}" for f in na if f in oa and na[f] != oa[f]]
    oe, ne = old.get("env") or {}, new.get("env") or {}
    lines += [f"env {k} added" for k in ne if k not in oe] + [f"env {k} removed" for k in oe if k not in ne]
    lines += [f"env {k}={oe[k]} -> {ne[k]}" for k in ne if k in oe and str(ne[k]) != str(oe[k])]
    for k in sorted(set(old) | set(new)):
        if k in ("gpu", "args", "env"):
            continue
        ov, nv = old.get(k), new.get(k)
        if k == "installer":
            ov = {x: y for x, y in (ov or {}).items() if x not in CONFIG_VOLATILE}
            nv = {x: y for x, y in (nv or {}).items() if x not in CONFIG_VOLATILE}
        if ov != nv:
            lines.append(f"{k} " + ("added" if ov is None else "removed" if nv is None else "changed"))
    return lines


def keep_config_backups(before: dict) -> list:
    """After a setup run: each installed config it changed gets the text it had before kept next to it, as
    <config>.json.bak-<date> (not a config the start or the Models page lists), and the changes are said.  The
    backups made."""
    made = []
    stamp = time.strftime("%Y%m%d-%H%M%S")
    for p, text in before.items():
        path = Path(p)
        try:
            now = path.read_text(encoding="utf-8-sig")
        except OSError:
            continue
        if now == text:
            continue
        try:
            changes = config_changes(json.loads(text), json.loads(now))
        except ValueError:
            changes = ["rewritten"]
        if not changes:
            continue
        bak = path.with_name(f"{path.name}.bak-{stamp}")
        try:
            bak.write_text(text, encoding="utf-8")
        except OSError as e:
            warn(f"{path.name}: changed by this setup, but the copy of the config before could not be written ({e})")
            continue
        made.append(bak)
        ok(f"{path.name}: changed by this setup (" + "; ".join(changes[:8]) +
           (f"; {len(changes) - 8} more" if len(changes) > 8 else "") + f") - the config before is kept as {bak.name}")
    return made


def config_before(cfg_path: Path) -> dict | None:
    """The config at cfg_path as it was when this setup run began, or None (a new one, or not a setup run)."""
    text = SETUP_BEFORE.get(str(Path(cfg_path).resolve()))
    try:
        return json.loads(text) if text is not None else None
    except ValueError:
        return None


def set_up(a, prev: dict) -> Path | None:
    """The setup (_set_up): the installed configs read first, and each one the run changed keeps a copy of what it had
    before next to it, said on the screen (#1066: a setup again rewrote a config - --ablate added, a GPU added)."""
    global SETUP_BEFORE
    before = {}
    for p in configs():
        try:
            before[str(p.resolve())] = p.read_text(encoding="utf-8-sig")
        except OSError:
            pass
    SETUP_BEFORE = before
    try:
        return _set_up(a, prev)
    finally:
        SETUP_BEFORE = {}
        keep_config_backups(before)


def _set_up(a, prev: dict) -> Path | None:
    """Steps 1-8, then the tuning: the config written, or None when the model was not downloaded.  `prev`: the
    config of the setup before (its answers are the defaults).  The model comes first: GLM-5.3-Flash or DeepSeek-V4.1-
    Flash-1017 (set_up_ds4), each then with the answers of its own setup before."""
    family = choose_family(a, prev)
    S.model_named(FAMILY_TITLE[family])
    check_parallel(a, family)
    if config_family(prev) != family:
        prev = next((c for c in map(read_json, configs()) if config_family(c) == family and
                     c.get("backend", "cuda") == a.backend), {})
    if family == "deepseek":
        return set_up_ds4(a, prev)
    if family == "minimax":
        return set_up_mm3(a, prev)
    if family == "qwen":
        return set_up_qwen(a, prev)
    if family == "hy4":
        return set_up_hy4(a, prev)
    if family == "ik2":
        return set_up_ik2(a, prev)
    inst = prev.get("installer") or {}
    prev_args = prev.get("args") or []
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) if "--max-context" in prev_args[:-1] else None
    a.gpus_before = prev.get("gpu")                    # (the GPU question's default: this config's GPUs)
    pc = check_pc(a)                                   # 1
    metal = pc["backend"] == "metal"
    step(2, "your choices")                            # 2
    # the models folder: --models-dir, else the one set up before (a config from before the option has its data
    # folder, whose models/ held the downloads), else Maya-data/models next to this folder
    models = (Path(a.models_dir).expanduser().resolve() if a.models_dir else
              Path(inst["models_dir"]) if inst.get("models_dir") else
              Path(inst["data_dir"]) / "models" if inst.get("data_dir") else ROOT.parent / "Maya-data" / "models")
    choice = choose_model(a, models, inst, pc)
    # a GLM-5.3-Flash quant that cannot start on this PC's one card: said here, before the tools, the build and the
    # download (glm_fit_gate; --force-fit goes on)
    fit = glm_fit_gate(a, choice[1], pc) if choice[0] == "download" and choice[1] in GLM_FIT else None
    if choice[0] == "download" and choice[1] in G53_MODELS:   # (GLM-5.3: "GLM-5.3 (744B) · UD-IQ2_M")
        ok(f"model: {G53.LABEL} · {Q.glm53_quant(choice[1])}, in {download_dir(models, choice[1])}")
    elif choice[0] == "download":
        ok(f"model: {Q.shown('glm', choice[1])}, in {download_dir(models, choice[1])}")
    else:
        ok(f"model: {choice[1]} (yours)")
    if g53_choice(choice):                             # GLM-5.3 (744B): the contexts its cache fits on these GPUs
        plan = g53_cache_plan(a, pc, choice)           # (as the config will run: write_config's)
        contexts = g53_contexts(pc, **plan)
        # (a unified-memory APU: the largest context that keeps the whole model in memory - --context still wins)
        rec, line = g53_apu_context(pc, plan, choice[1] if choice[0] == "download" else quant_of(Path(choice[1])),
                                    contexts)
        if line and not getattr(a, "context", None):
            (ok if rec else warn)(line)
        ctx = choose_context(a, prev_ctx, contexts, rec or (DEFAULT_CONTEXT if DEFAULT_CONTEXT in contexts else contexts[-1]))
        if rec and ctx > rec:                          # (--context, or the earlier setup's answer: kept, said)
            warn(f"a {context_label(ctx)} context's cache pushes experts to the SSD on this PC; "
                 f"{context_label(rec)} keeps the whole model in memory")
    else:
        rec_ctx = recommended_context(choice, pc)
        fit_ctx = glm_fit_context(a, fit, getattr(a, "context", None))   # (the contexts that start on this card)
        if fit_ctx:
            ctx = choose_context(a, prev_ctx, fit_ctx, rec_ctx if rec_ctx in fit_ctx else fit_ctx[-1])
        else:
            ctx = choose_context(a, prev_ctx, recommended=rec_ctx)
    ok(f"context: {ctx} tokens")
    draft = choose_draft(a, pc, inst)
    ok("drafting: " + ("a DFlash2 drafter (downloaded with the model)" if draft == "dflash2" else "the MTP block"))
    llama = tools_step(a)                              # 3
    meta = build_step(a, pc, llama)                    # 4
    mixed_builds(a, pc, llama, "maya")                 # (an NVIDIA card beside an AMD GPU: its engine too)
    got = model_step(a, models, choice)                # 5
    if got is None:
        return None
    d, shards, quant = got
    dsa = kind_is_glm_dsa(choice, shards)              # GLM-5.3 (glm-dsa): text only, NextN drafting
    if dsa and metal:
        fail(f"{G53.SHOWN} runs on NVIDIA and AMD GPUs (not on Metal yet)", f"{ME} --setup picks another model")
    if dsa:
        S.model_named(G53.SHOWN)                       # (the header: GLM-5.3, not GLM-5.3-Flash)
        draft = maya_sync.draft_for(draft, G53.ARCH, warn)
        have = g53_memory_gb(pc)                       # (neutral memory guidance, no speeds)
        (ok if have is not None and have >= g53_figures(quant)[0] else warn)(
            f"{G53.SHOWN}: {g53_memory_note(pc, quant)}")
    dflash = drafter_step(models) if draft == "dflash2" else None
    pack = metal_tokenizer_step(a, d, shards, quant) if metal else pack_step(a, d, shards, llama, quant)       # 6
    if dsa:
        step(7, "images (the vision encoder)")
        ok(f"{G53.SHOWN}: text only (the model has no vision tower)")
        vision = None
    else:
        vision = vision_step(a, pc, meta, llama, d, quant)  # 7
    cfg_path = write_config(a, pc, meta, pack, quant, ctx, models, vision, choice, dflash, shards[0] if metal else None)   # 8
    if not metal:                                      # (the Mac engine does not project a direction)
        ablate_step(cfg_path, "glm-dsa" if dsa else "glm", models, a)
    if dflash is not None and metal:                   # (the speed calibration is CUDA's: on a Mac the choice is yours)
        ok("DFlash2 drafts on this Mac, not measured against the MTP block (the model checks both): "
           f"{ME} --setup --draft mtp goes back")
    elif dflash is not None:                           # DFlash2: kept only if it makes THIS PC faster (measured)
        how = drafting_choice(a, "DFlash2", "5-20")
        if how != "measure":                           # (or as the user chose)
            keep_drafting(cfg_path, how == "on")
        else:
            if S.UI is not None:                       # (the screen shows it in the tuning's tab)
                S.UI.step(9, "speed calibration: DFlash2 or the MTP block")
            speed_calibration(cfg_path, measure_asked(a))
    elif g53_one_gpu_drafting(read_json(cfg_path)):    # GLM-5.3 on one APU: drafting kept only where it is faster
        fallback = bool(read_json(cfg_path)["installer"]["one_gpu_drafting"].get("on"))
        how = "on" if fallback else "off"
        if not getattr(a, "no_speed_calibration", False):   # (skipped: the setup's estimate for this PC stays)
            how = drafting_choice(a, "Drafting with the MTP block on this GPU", "5-10")
        if how != "measure":
            keep_drafting(cfg_path, how == "on")
        else:
            if S.UI is not None:
                S.UI.step(9, "speed calibration: drafting or not")
            speed_calibration(cfg_path, measure_asked(a))
    # the tuning: asked for (--calibrate), or offered when nothing was tuned for this PC and model yet and someone
    # answers (--yes and --no-start setups are not held up by it)
    tuned = None
    cfg = read_json(cfg_path)
    offer = (cfg.get("backend") not in ("hip", "metal") and saved_calibration(cfg, any_context=True) is None
             and not a.yes and not a.no_start)
    if metal and a.calibrate:
        warn("--calibrate does not apply to Metal: there is no CPU lane to tune")
    if (a.calibrate and not metal) or (offer and ask(
            "Tune Maya for this PC now? It measures the split of the work between the CPU and the PCIe link and the "
            f"CPU threads (about {G53.CALIBRATE_MINUTES if dsa else '10-15'} minutes; later: ./maya.sh --calibrate)",
            ["y", "n"], "y", a.yes) == "y"):
        if S.UI is not None:                           # (the screen lists the tuning as a step of its own)
            S.UI.step(9, "tuning Maya for this PC")
        tuned = calibrate_config(cfg_path)
    if tuned is False:
        say()
        warn("this PC is NOT tuned (the reason is above): ./maya.sh --calibrate tries again")
    return cfg_path


# ------------------------------------------------------------------------------------------------ DeepSeek-V4.1-Flash
def ds4_maya_published(quant: str) -> bool:
    """Whether Maya's quant of DeepSeek has its files in download_gguf.json (they are listed once uploaded)."""
    q = ((read_json(DS4_GGUF_MANIFEST).get("maya") or {}).get("quants") or {}).get(DS4_QUANTS[quant]) or {}
    return bool(q.get("files"))


def ds4_gguf_model(quant: str) -> dict:
    """A GGUF of DeepSeek as a MODELS entry (offer_download's form): its shards, sizes, sha256, repo folder - unsloth's,
    Maya's (the Engram tables in its own shards) or orcarouter's abliterated one ("gated": its download sends a token) -
    and "earlier": the layouts its files had before, each in the same form (ds4_earlier)."""
    man = read_json(DS4_GGUF_MANIFEST)
    gated = False
    if quant in DS4_ABL:
        ab = man.get("abliterated") or {}
        q = (ab.get("quants") or {}).get(DS4_QUANTS[quant]) or {}
        if not q.get("files"):
            fail(f"{DS4_GGUF_MANIFEST} is missing or damaged", "restore it (git checkout tools/ds4/download_gguf.json)")
        about, repo, revision = (f"{ab.get('name', DS4 + '-abliterated')} {DS4_QUANTS[quant]}, orcarouter's abliterated "
                                 f"model as GGUFs ({q.get('about', 'the experts in 2-4 bits')}; its shard 2, the n-gram "
                                 "table alone, left out: Maya's Q6_K tables are read instead)", ab["repo"],
                                 ab["revision"])
        gated = bool(ab.get("gated"))
        q = dict(q, files={k: v for k, v in q["files"].items() if k != q.get("table")})
    elif quant in DS4_MAYA:
        my = man.get("maya") or {}
        q = (my.get("quants") or {}).get(DS4_QUANTS[quant]) or {}
        if not q.get("files"):
            fail(f"{DS4} {DS4_QUANTS[quant]} is not published yet",
                 f"set up unsloth's GGUF meanwhile: {ME} --setup --model {DS4} --ds4-quant gguf-q8")
        about, repo, revision = (f"{DS4} {DS4_QUANTS[quant]}, Maya's quant ({q.get('about', 'the experts in 2-3 bits')})",
                                 my["repo"], my["revision"])
    else:
        q = (man.get("quants") or {}).get(DS4_QUANTS[quant]) or {}
        if not q.get("files"):
            fail(f"{DS4_GGUF_MANIFEST} is missing or damaged", "restore it (git checkout tools/ds4/download_gguf.json)")
        about, repo, revision = (f"{DS4} as unsloth's {DS4_QUANTS[quant]} GGUF ("
                                 + (q.get("about") or f"dense weights {'BF16' if quant == 'gguf-q8' else 'Q8_0'}, "
                                                      "experts MXFP4, the Engram tables MXFP8") + ")",
                                 man["repo"], q.get("revision") or man["revision"])   # (DS4_UD: their own revision)

    def form(files: dict, rev: str) -> dict:
        files = {Path(k).name: v for k, v in files.items()}
        return {"about": about, "format": "ds4-gguf", "repo": repo, "revision": rev, "folder": DS4_QUANTS[quant],
                "files": {k: v[0] for k, v in files.items()}, "sha256": {k: v[1] for k, v in files.items()},
                "download_gb": sum(v[0] for v in files.values()) / 1e9, **({"gated": True} if gated else {})}
    m = form(q["files"], revision)
    if quant in DS4_ABL and q.get("table"):
        m["table"] = Path(q["table"]).name   # (listed in the manifest, never fetched)
    m["earlier"] = [dict(form(e["files"], e["revision"]), engram=[form(s, e["revision"]) for s in e.get("engram") or []],
                         upgrade=bool(e.get("upgrade")))
                    for e in q.get("earlier") or []]
    m["tables"] = [Path(k).name for k in q.get("tables") or {}]   # (shared table shards: kept by an upgrade)
    return m


def ds4_earlier(m: dict, d: Path) -> dict | None:
    """An earlier layout of m's files (ds4_gguf_model's "earlier") that d holds whole, used as it is - nothing
    downloaded again: unsloth's GGUFs before they were re-sharded on 2026-10-19, or v1.1.16's Maya-S (its 4 shards, the
    Engram tables apart: unsloth's MXFP8 shard of them in <...-GGUF>/Engram, either of its layouts; as "engram" the
    folder then).  None: m's own files are whole, or no earlier layout is."""
    if not any(file_incomplete(m, d / n) for n in m["files"]):
        return None
    for e in m.get("earlier") or []:
        if e.get("upgrade") or any(file_incomplete(e, d / n) for n in e["files"]):
            continue                                   # (an "upgrade" layout is replaced: ds4_upgrade_from)
        if not e["engram"]:
            return dict(e, engram=None, layout="earlier")
        eg = d.parent / "Engram"
        if any(not any(file_incomplete(s, eg / n) for n in s["files"]) for s in e["engram"]):
            return dict(e, engram=eg, layout="earlier")
    return None


def ds4_upgrade_from(m: dict, d: Path) -> dict | None:
    """The earlier layout of m's files that d holds whole and the manifest marks "upgrade": the setup then downloads m's
    own files instead (keeping the shards both layouts share: m["tables"]) and drops the old ones last
    (ds4_upgrade_drop).  None: m's own files are whole, or no such layout is there."""
    if not any(file_incomplete(m, d / n) for n in m["files"]):
        return None
    for e in m.get("earlier") or []:
        if e.get("upgrade") and not any(file_incomplete(e, d / n) for n in e["files"]):
            return e
    return None


def ds4_upgrade_wanted(a, m: dict, d: Path) -> bool:
    """Ask before an upgrade's download: the newer version, or the one on disk as it is (--yes keeps it: no large
    download nobody asked for)."""
    gb = sum(v for n, v in m["files"].items() if file_incomplete(m, d / n)) / 1e9
    rows = [(f"Download the newer {m['folder']}", f"{gb:.0f} GB", "Closer to the original model and faster; the n-gram "
             "table files you have are kept, and the old model files are deleted once the new ones are complete."),
            (f"Keep my current {m['folder']}", "no download", "Starts with the version already on this PC. Run the "
             "setup again later to update.")]
    return choose(f"A newer {m['folder']} is available", [f"A newer version of {m['folder']} is out."], rows,
                  1, a.yes) == 0


def ds4_upgrade_note(m: dict, d: Path) -> None:
    """The one line that says an upgrade is under way and how much it downloads."""
    gb = sum(v for n, v in m["files"].items() if file_incomplete(m, d / n)) / 1e9
    say(f"  Updating {m['folder']} to its newer version: {gb:.0f} GB to download "
        f"({len(m['files'])} model files; the n-gram table files you have are kept).")


def ds4_table_dupes(pack: Path, d: Path) -> list:
    """A second copy of a file the pack reads from beside the model folder (a shared Engram set: Engram-Q6_K) left in
    the model folder d under the same name - Maya-XS v1's table shards after the update to v2 (#777): a hard link to
    the shared copy takes its place, so the disk holds one copy and any path to either name still works.  Only when
    both were checked against the same published sha256 (VERIFIED, size and time unchanged) - else left as they are.
    The names relinked."""
    done = []
    for f in pack_index_files(pack) or []:
        if f.parent.resolve() == d.resolve() or not f.is_file():
            continue
        old = d / f.name
        try:
            if not old.is_file() or os.path.samefile(old, f) or old.stat().st_size != f.stat().st_size:
                continue
        except OSError:
            continue

        def verified(p: Path) -> bool:
            seen, st = read_json(p.parent / VERIFIED).get(p.name) or {}, p.stat()
            return seen.get("size") == st.st_size and seen.get("mtime_ns") == st.st_mtime_ns

        if not (verified(old) and verified(f)):
            continue
        tmp = old.with_name(old.name + ".link")
        try:
            os.link(f, tmp)
            os.replace(tmp, old)
            done.append(old.name)
            book, st = read_json(d / VERIFIED), old.stat()   # (the same checked file now: not read again)
            book[old.name] = {"size": st.st_size, "mtime_ns": st.st_mtime_ns}
            (d / VERIFIED).write_text(json.dumps(book, indent=1), encoding="utf-8")
        except OSError as x:
            tmp.unlink(missing_ok=True)
            warn(f"{old}: a second copy of {f} ({x}) - you can delete it")
    if done:
        ok(f"{len(done)} second cop{'ies' if len(done) != 1 else 'y'} of the n-gram tables in {d.name} now share "
           f"{f.parent.name}'s ({sum((d / n).stat().st_size for n in done) / 1e9:.0f} GB back)")
    return done


def ds4_upgrade_drop(old: dict, m: dict, d: Path) -> list:
    """After the upgrade: delete the old layout's files that m does not use (not m's own, not m["tables"]), but only
    when every one of m's files is whole - an interrupted download leaves the old model working.  The deleted names."""
    if any(file_incomplete(m, d / n) for n in m["files"]):
        return []
    gone = []
    for n in old["files"]:
        if n in m["files"] or n in (m.get("tables") or []):
            continue
        try:
            (d / n).unlink()
            gone.append(n)
        except OSError as x:
            warn(f"{d / n}: could not delete the old file ({x}) - you can delete it")
    if gone:
        ok(f"the earlier version's {len(gone)} model file{'s' if len(gone) != 1 else ''} deleted")
    return gone


def ds4_unused(m: dict, d: Path) -> None:
    """A note for the files of an earlier layout in d (and its Engram tables beside it), now that m's own are used: the
    user may delete them."""
    for where, names in ((d, [n for e in m.get("earlier") or [] for n in e["files"]]),
                         (d.parent / "Engram", [n for e in m.get("earlier") or [] for s in e["engram"] for n in s["files"]])):
        names = [n for n in dict.fromkeys(names) if (where / n).exists() and n not in m["files"]
                 and n not in (m.get("tables") or [])]
        if names:
            warn(f"{where}: {len(names)} file{'s' if len(names) != 1 else ''} of an earlier layout of the model, not "
                 f"used - you can delete {'them' if len(names) != 1 else 'it'} ({', '.join(names[:2])}"
                 f"{', ...' if len(names) > 2 else ''})")


def ds4_drafter_model() -> dict:
    """DeepSeek's DSpark drafter (the GGUFs leave it out) as offer_download's form: the three shards of DeepSeek's own
    checkpoint that hold only its blocks (download_gguf.json's "drafter")."""
    man = (read_json(DS4_GGUF_MANIFEST).get("drafter") or {})
    if not man.get("files"):
        fail(f"{DS4_GGUF_MANIFEST} is missing or damaged", "restore it (git checkout tools/ds4/download_gguf.json)")
    return {"about": "DeepSeek's DSpark drafter (it drafts a few tokens at a time for the model to check: faster "
                     "answers on PCs where that pays - measured after the setup), from DeepSeek's own checkpoint",
            "format": "ds4-drafter", "repo": man["repo"], "revision": man["revision"], "folder": "",
            "files": {k: v[0] for k, v in man["files"].items()}, "sha256": {k: v[1] for k, v in man["files"].items()},
            "download_gb": sum(v[0] for v in man["files"].values()) / 1e9}


def ds4_maya_drafter_model() -> dict | None:
    """The DSpark drafter Maya ships (download_gguf.json's "maya" / "drafter": DeepSeek's FP4 experts as released, its
    FP8 dense weights in Q8_0 - the formats the engine reads, nothing to convert) as offer_download's form, or None
    while it is not pinned there."""
    my = read_json(DS4_GGUF_MANIFEST).get("maya") or {}
    dr = my.get("drafter") or {}
    if not dr.get("files") or not my.get("revision"):
        return None
    files = {Path(k).name: v for k, v in dr["files"].items()}
    return {"about": "the model's DSpark drafter (it drafts a few tokens at a time for the model to check: faster "
                     "answers on PCs where that pays - measured after the setup), from Maya's repo: DeepSeek's FP4 "
                     "experts, its dense weights in 8-bit", "format": "ds4-drafter", "repo": my["repo"],
            "revision": my["revision"], "folder": "dspark",
            "files": {k: v[0] for k, v in files.items()}, "sha256": {k: v[1] for k, v in files.items()},
            "download_gb": sum(v[0] for v in files.values()) / 1e9}


def ds4_quant_drafter(quant: str) -> dict | None:
    """A Maya quant's own DSpark drafter (download_gguf.json's maya / quants / <quant> / "drafter": e.g. Maya-XS's, its
    experts in 2 bits) as offer_download's form plus "json" (its .json's name in dspark/), or None: the shared one."""
    if quant not in DS4_MAYA:
        return None
    my = read_json(DS4_GGUF_MANIFEST).get("maya") or {}
    dr = ((my.get("quants") or {}).get(DS4_QUANTS[quant]) or {}).get("drafter") or {}
    if not dr.get("files") or not dr.get("json"):
        return None
    files = {Path(k).name: v for k, v in dr["files"].items()}
    return {"about": f"{Q.shown('deepseek', quant)}'s DSpark drafter ({dr.get('about', 'its own formats')})",
            "format": "ds4-drafter", "repo": my["repo"], "revision": my["revision"], "folder": "dspark",
            "files": {k: v[0] for k, v in files.items()}, "sha256": {k: v[1] for k, v in files.items()},
            "download_gb": sum(v[0] for v in files.values()) / 1e9, "json": dr["json"]}


def ds4_drafter_step(a, d: Path, quant: str | None = None) -> bool:
    """The DSpark drafter in d (beside the GGUFs' folders, shared by both): downloaded when it is not there, then made
    the GGUF path's formats (tools/ds4/drafter_pack.py: dspark.bin) - its three shards are removed after that.  A quant
    with its own drafter (ds4_quant_drafter) gets that one instead.  False: not there (not downloaded): the model runs
    without it."""
    own = ds4_quant_drafter(quant) if quant else None
    if own is not None:
        want = [d / n for n in own["files"]]
        if any(file_incomplete(own, f) for f in want) and not offer_download(a, "the DSpark drafter", d, want, own):
            warn("the DSpark drafter was not downloaded: the model runs without it")
            return False
        ok(f"the DSpark drafter: {d / own['json']}")
        return True
    if (d / "dspark.bin").is_file() and (d / "dspark.json").is_file():
        ok(f"the DSpark drafter: {d / 'dspark.bin'}")
        return True
    mm = ds4_maya_drafter_model()
    if mm is not None:                                  # (Maya's: ready to use, nothing to convert)
        want = [d / n for n in mm["files"]]
        if any(file_incomplete(mm, f) for f in want) and not offer_download(a, "the DSpark drafter", d, want, mm):
            warn("the DSpark drafter was not downloaded: the model runs without it")
            return False
        ok(f"the DSpark drafter: {d / 'dspark.bin'}")
        return True
    m = ds4_drafter_model()
    shards = [d / n for n in m["files"]]
    if any(file_incomplete(m, f) for f in shards) and not offer_download(a, "the DSpark drafter", d, shards, m):
        warn("the DSpark drafter was not downloaded: the model runs without it")
        return False
    say("  Converting the drafter to the GGUF's formats (exactly the same values; under a minute) ...")
    run([sys.executable, str(ROOT / "tools" / "ds4" / "drafter_pack.py"), "--dir", str(d)])
    if not (d / "dspark.bin").is_file():
        fail("the drafter's conversion did not write dspark.bin", "the reason is in the messages above")
    for f in shards:                                   # (the setup's own download, now in dspark.bin)
        f.unlink(missing_ok=True)
    ok(f"the DSpark drafter: {d / 'dspark.bin'}")
    return True


def ds4_vision_model(mmproj: bool = False) -> dict:
    """DeepSeek's vision tower (the GGUFs leave it out; every GGUF of the model shares it) as offer_download's form:
    shard 1 of DeepSeek's own checkpoint (only the tower and its aligner) and config.json (download_gguf.json's
    "vision"; its "marks" - 30 KB of shard 2 - are read by tools/ds4/vision_pack.py).  mmproj (unsloth's UD quants):
    unsloth's mmproj-BF16.gguf from the quant's own repo instead ("vision" / "mmproj": the same weights, vision_pack.py
    --mmproj makes the same vision.bin from it, byte for byte)."""
    man = (read_json(DS4_GGUF_MANIFEST).get("vision") or {})
    if not man.get("files") or not man.get("marks"):
        fail(f"{DS4_GGUF_MANIFEST} is missing or damaged", "restore it (git checkout tools/ds4/download_gguf.json)")
    mp = man.get("mmproj") or {}
    if mmproj and mp.get("files"):
        return {"about": "DeepSeek's vision tower (it lets the model read pictures), unsloth's mmproj of it",
                "format": "ds4-vision", "repo": mp["repo"], "revision": mp["revision"], "folder": "", "mmproj": True,
                "files": {k: v[0] for k, v in mp["files"].items()}, "sha256": {k: v[1] for k, v in mp["files"].items()},
                "download_gb": sum(v[0] for v in mp["files"].values()) / 1e9}
    return {"about": "DeepSeek's vision tower (it lets the model read pictures), from DeepSeek's own checkpoint",
            "format": "ds4-vision", "repo": man["repo"], "revision": man["revision"], "folder": "",
            "files": {k: v[0] for k, v in man["files"].items()}, "sha256": {k: v[1] for k, v in man["files"].items()},
            "download_gb": sum(v[0] for v in man["files"].values()) / 1e9}


def ds4_vision_dir(d: Path, models: Path | None = None) -> Path:
    """The shared vision folder; a model directly in --models-dir keeps it inside that folder."""
    return (d if models is not None and d == models else d.parent) / "vision"


def ds4_pack_has_vision(pack: Path) -> bool:
    """The pack holds a vision tower (its geometry is in ds4.txt: pack.py's from the EXL3 model, pack_gguf.py's from
    vision_pack.py's files)."""
    try:
        return any(line.startswith("vision.num_hidden_layers ")
                   for line in (pack / "ds4.txt").read_text(encoding="utf-8", errors="replace").splitlines())
    except OSError:
        return False


def ds4_exl3_vision_dir(models: Path) -> Path:
    """The vision folder an EXL3 quant whose own tower the engine does not read (DS4_EXL3's "tower") packs DeepSeek's
    BF16 tower from: one a GGUF install made already (Maya's, unsloth's, orcarouter's - the same vision.bin, shared),
    else Maya's GGUF folder's, where the setup makes it."""
    homes = [models / f"{DS4}-Maya-GGUF" / "vision", models / f"{DS4}-GGUF" / "vision",
             models / f"{DS4}-abliterated-GGUF" / "vision"]
    return next((v for v in homes if (v / "vision.bin").is_file() and (v / "vision.json").is_file()), homes[0])


def ds4_pack_tower(pack: Path) -> bool:
    """The EXL3 pack reads its vision tower from a vision folder's vision.bin (pack.py --vision): tensors.txt's file
    lines, at its top, name it."""
    try:
        with (pack / "tensors.txt").open(encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("file "):
                    return False
                if line.rstrip().endswith("/vision.bin"):
                    return True
    except OSError:
        pass
    return False


def ds4_vision_blocker(pc: dict) -> str | None:
    """Why the vision tower cannot run on this PC, or None: it needs ~1 GB of RAM (its weights, streamed from there)
    and, while a picture is read, ~320 MB of the first GPU's memory (lent, none between pictures) - every GPU the GGUF
    path runs on has the BF16 matrix products it uses."""
    total = mem_gb()[0]
    if 0 < total < 16:
        return f"{total:.0f} GB of RAM is too little beside the model"
    return None


def ds4_vision_step(a, pc: dict, d: Path, mmproj: bool = False, v: Path | None = None) -> bool:
    """Step 7 for a GGUF: the vision tower in ds4_vision_dir(d) - downloaded when it is not there (shard 1 of
    DeepSeek's checkpoint, 0.97 GB, and its config.json), then made the GGUF path's file (tools/ds4/vision_pack.py:
    vision.bin, with the picture markers read from shard 2 by a 30 KB range request); the shard is removed after
    that.  mmproj (unsloth's UD quants): unsloth's mmproj-BF16.gguf (0.97 GB) instead, made into the same vision.bin.
    False: no vision tower (skipped, cannot run here, or not downloaded): the model reads text only.  v: the folder
    (default ds4_vision_dir(d); an EXL3 quant whose own tower the engine does not read: ds4_exl3_vision_dir's)."""
    step(7, "images (the vision tower)")
    if a.no_vision:
        ok("skipped (--no-vision): the model reads text only")
        return False
    why = ds4_vision_blocker(pc)
    if why:
        warn(f"{why}: the model reads text only - the vision tower's weights (~1 GB) are held in pinned RAM while the "
             f"model runs; with 16 GB of RAM or more {ME} --setup turns pictures on")
        return False
    v = v or ds4_vision_dir(d)
    if (v / "vision.bin").is_file() and (v / "vision.json").is_file():
        ok(f"the vision tower: {v / 'vision.bin'} (its weights in RAM; no VRAM between pictures - about 320 MB of the "
           "first GPU's while one is read; the server resizes the pictures)")
        return True
    if not os.access(existing(v), os.W_OK):
        warn(f"{existing(v)} is not writable: no vision tower, the model reads text only")
        return False
    m = ds4_vision_model(mmproj)
    files = [v / n for n in m["files"]]
    if any(file_incomplete(m, f) for f in files) and not offer_download(a, "the vision tower", v, files, m):
        warn("the vision tower was not downloaded: the model reads text only")
        return False
    for f in files:                                    # (offer_download checked the ones it fetched; these too)
        if not sha256_checked(f, m["sha256"][f.name]):
            f.unlink(missing_ok=True)                  # else the next run finds it complete and fails here again
            fail(f"{f}: the sha256 does not match the published one (the file was removed)", "run the setup again")
    if m.get("mmproj"):
        say("  Copying the mmproj's tower into one file (exactly the same values; a few seconds) ...")
    else:
        say("  Reading the picture markers (30 KB of DeepSeek's shard 2, a range request) and copying the tower into one "
            "file (exactly the same values; a few seconds) ...")
    r = run([sys.executable, str(ROOT / "tools" / "ds4" / "vision_pack.py"), "--dir", str(v)]
            + (["--mmproj", str(files[0])] if m.get("mmproj") else []), check=False)
    if r.returncode != 0 or not (v / "vision.bin").is_file():
        warn("the vision tower was not made (the reason is above): the model reads text only - the setup tries again "
             "next time, without downloading the shard again")
        return False
    for f in files:                                    # (the setup's own download, now in vision.bin)
        f.unlink(missing_ok=True)
    ok(f"the vision tower: {v / 'vision.bin'} (its weights in RAM; no VRAM between pictures - about 320 MB of the first "
       "GPU's while one is read; the server resizes the pictures)")
    return True


def choose_ds4_quant(a, prev: dict) -> str:
    """--ds4-quant, else the one set up before, else asked (the GGUF at Q8 recommended)."""
    if getattr(a, "ds4_quant", None):
        return ds4_abl_recommended(getattr(a, "backend", None) == "hip") if a.ds4_quant == DS4_ABL_ALIAS else a.ds4_quant
    hip = getattr(a, "backend", None) == "hip"
    maya = [k for k in DS4_MAYA if ds4_maya_published(k)]
    # the EXL3 quants where they run: NVIDIA V100 or newer (compute capability 7.0+), and AMD (as GLM's EXL3 quants
    # there) - a PC whose NVIDIA GPUs are all older is not offered them (nvidia-smi silent: offered, the
    # PC check says)
    try:
        ccs = [] if hip else [int(g["arch"]) for g in S.gpus()]
    except Exception:  # noqa: BLE001 - no NVIDIA tools
        ccs = []
    exl3_ok = hip or not ccs or max(ccs) >= DS4_MIN_CC
    before = (prev.get("installer") or {}).get("ds4_quant")
    keys = maya + list(DS4_ABL) + list(DS4_UD_LISTED) + [
        k for k in DS4_QUANTS if k not in DS4_MAYA + DS4_ABL + DS4_UD_LISTED and (exl3_ok or k not in DS4_EXL3)
                                   and (k not in DS4_BY_HAND + DS4_UD or k == before)]   # (a chosen-by-hand one: only where it is installed)
    abl = ds4_abl_recommended(hip)
    # Maya's quants by this PC's VRAM + RAM (fit_recommendation): no larger than Maya-S (the first listed), the cache
    # at the default context on each GPU (ds4_kv_gb)
    gpus = setup_gpus(a)
    try:
        ram = mem_gb()[0]
    except Exception:  # noqa: BLE001 - the RAM does not read: the usual pick
        ram = 0.0
    rec = fit_recommendation({k: ds4_memory_gb(k) for k in maya}, gpus, ram,
                             ds4_kv_gb(DS4_DEFAULT_CONTEXT, len(gpus)), maya[0]) if maya else keys[0]
    default = before if before in keys else rec if maya else DS4_DEFAULT_QUANT
    gb = {k: ds4_gguf_model(k)["download_gb"] for k in maya + [k for k in DS4_ABL + DS4_UD if k in keys]}
    for k in DS4_ABL + ("maya-xs",):   # (as the other entries: the whole download, Maya's n-gram tables with it)
        if k in gb:
            gb[k] += ds4_abl_engram_gb()
    gb.update({"gguf-q8": 508, "gguf-q4": 501, "exl3": 414})
    for k in ("gguf-q8", "gguf-q4") + DS4_UD:   # (unsloth's: its table shard left out, Maya's Q6_K tables instead)
        t = ds4_own_table(k)
        if t and k in keys:
            mk = ds4_gguf_model(k)
            gb[k] = mk["download_gb"] - mk["files"].get(t, 0) / 1e9 + ds4_abl_engram_gb()
    for k in ("exl3-3.5", "exl3-2.25"):   # (each one's whole download, from its own manifest)
        if k in keys:
            gb[k] = ds4_model(k)["download_gb"]
    about = {"maya-s": "Maya's quant: experts in 2-3 bits, the most of it in memory",
             "maya-m": "Maya's quant: experts in 3.4 bits, closer to the original",
             "maya-xs": "Maya's smallest quant: experts in ~1.6 bits, the whole model in a 128 GB PC's memory",
             "gguf-q8": ("unsloth's GGUF, the closest to the original model (DeepSeek's own formats: dense weights BF16, "
                         "experts MXFP4): KLD 0.047, 96.4% the same top token as the full-precision original (our scorer, "
                         "the same scale as the UD-IQ3_S and Maya-S numbers); about 306 GB of VRAM + RAM hold it, plus the "
                         "203 GB of n-gram tables (MXFP8) read from the SSD - not shareable with Maya's Q6_K tables"),
             "gguf-q4": "unsloth's GGUF: dense weights Q8_0 (less VRAM), experts MXFP4",
             "exl3": "LibertAIDAI's quant, with the DSpark drafter and the vision tower",
             "exl3-3.5": f"closer to the original, slower: ~{DS4_EXL3['exl3-3.5']['memory_gb']} GB of VRAM + RAM "
                         "hold it",
             "exl3-2.25": f"turboderp's quant, the smallest EXL3: ~{DS4_EXL3['exl3-2.25']['memory_gb']} GB of VRAM + "
                          "RAM hold it; with the DSpark drafter, and DeepSeek's own vision tower (0.97 GB, shared with "
                          "the GGUFs)",
             "abliterated-128g": "orcarouter's model with its refusals removed: experts ~1.6 bits, ~114 GB of VRAM + "
                                 "RAM hold it; its 161 GB of n-gram tables are Maya's, shared with Maya's DeepSeek "
                                 "quants (downloaded once); a gated repo: your Hugging Face token",
             "abliterated-256g": "orcarouter's model with its refusals removed: experts 2.5-4.25 bits by layer (IQ2_S / "
                                 "IQ3_XXS / IQ3_S / MXFP4), ~236 GB of VRAM + RAM hold it; its 161 GB of n-gram tables are "
                                 "Maya's, shared with Maya's DeepSeek quants (downloaded once); a gated repo: your "
                                 "Hugging Face token"}
    for k in DS4_UD:   # (download_gguf.json's own words: the experts' and dense weights' types, the tables')
        q = ((read_json(DS4_GGUF_MANIFEST).get("quants") or {}).get(DS4_QUANTS[k]) or {})
        about[k] = f"unsloth's GGUF: {q.get('about', 'experts in 1.6-3.5 bits')}; ~{q.get('memory_gb', '?')} GB of VRAM + RAM hold it"
    about["gguf-iq1_m"] = ("unsloth's UD-IQ1_M: KLD 0.198 against the original, 90.3% the same top token (our scorer; "
                           "the same scale as the other KLD numbers here); about 140 GB of VRAM + RAM plus 110 GB of "
                           "n-gram tables (IQ4_NL, not shareable with Maya's Q6_K tables) read from the SSD; for PCs "
                           "with about 150 GB of VRAM + RAM")
    about["gguf-iq3_s"] = ("unsloth's UD-IQ3_S: close to Q8's quality (KLD 0.061 vs Q8's 0.047; Maya-S 0.119), about 262 GB of "
                           "VRAM + RAM plus the 161 GB n-gram tables read from the SSD; for PCs with about 270 GB or more of "
                           "VRAM + RAM - below that it reads experts from the disk and gets slow")
    label = {k: (Q.shown("deepseek", k, gb.get(k)), about[k]) for k in DS4_QUANTS}
    pick = choose(f"Which files of {DS4}?", ["The GGUFs run on any NVIDIA GPU from Volta (V100) on and on AMD, whatever "
                                             "the memory: the experts that do not", "fit in VRAM are read from RAM and "
                                             "the SSD. The EXL3 quants need a V100 or newer, or AMD, and "
                                             f"~{DS4_EXL3['exl3-2.25']['memory_gb']} / ~230 / ~280 GB of VRAM + RAM:"],
                  [(label[k][0], "recommended" if k == rec else
                    "abliterated: fits this PC" if k == abl else None, label[k][1]) for k in keys],
                  keys.index(default), a.yes or a.check)
    return keys[pick]


def ds4_memory_gb(quant: str) -> float:
    """The VRAM + RAM that hold DeepSeek in these files, the context aside: the EXL3 quants' (DS4_EXL3); a Maya
    quant's "memory_gb" (download_gguf.json: its files but the Engram tables, which stay on the SSD, and its drafter);
    else the GGUF's files (unsloth's); DS4_MEMORY_GB when the manifest lists none."""
    if quant in DS4_EXL3:
        return float(DS4_EXL3[quant]["memory_gb"])
    man = read_json(DS4_GGUF_MANIFEST)
    q = ((man.get("maya") if quant in DS4_MAYA else man.get("abliterated") if quant in DS4_ABL else man) or {}).get(
        "quants", {}).get(DS4_QUANTS.get(quant)) or {}
    if q.get("memory_gb"):
        return float(q["memory_gb"])
    return sum(v[0] for v in (q.get("files") or {}).values()) / 1e9 or float(DS4_MEMORY_GB)


def ds4_abl_recommended(hip: bool = False) -> str:
    """The orcarouter version this PC's VRAM + RAM hold (as MiniMax's XS / S): the larger that fits, else the smaller."""
    total, _ = mem_gb()
    try:
        vram = sum(g["vram_gb"] for g in (S.amd_gpus() if hip else S.gpus()))
    except Exception:  # noqa: BLE001 - no GPU tools
        vram = 0.0
    fit = [k for k in DS4_ABL if vram + total >= ds4_memory_gb(k)]
    return max(fit, key=ds4_memory_gb) if fit else min(DS4_ABL, key=ds4_memory_gb)


def ds4_tier_files(my: dict, tier: str) -> dict:
    """A Maya quant's files in the manifest, with its Q6_K n-gram table shards ("tables": the pair Maya-XS shares with
    Maya-S, not among its own model files)."""
    q = (my.get("quants") or {}).get(tier) or {}
    return {**(q.get("files") or {}), **(q.get("tables") or {})}


def ds4_table_homes(models: Path, d: Path | None = None) -> list:
    """Folders where a whole copy of Maya's Q6_K tables may be on this PC: each Maya quant's own folder (as published),
    and the Engram-Q6_K folders the setups make (Maya's, orcarouter's, unsloth's)."""
    man = read_json(DS4_GGUF_MANIFEST)
    eg = (man.get("abliterated") or {}).get("engram") or {}
    f = eg.get("folder", "Engram-Q6_K")
    homes = [models / f"{DS4}-Maya-GGUF" / t for t in (eg.get("from") or {})]
    homes += [models / g / f for g in (f"{DS4}-Maya-GGUF", f"{DS4}-abliterated-GGUF", f"{DS4}-GGUF")]
    return homes


def ds4_trim_files(m: dict, names: list) -> dict:
    """m without the files in names (their sizes and sha256 too, the download smaller by them)."""
    files = {k: v for k, v in m["files"].items() if k not in names}
    return dict(m, files=files, sha256={k: v for k, v in m["sha256"].items() if k in files},
                download_gb=sum(files.values()) / 1e9)


def ds4_own_table(quant: str) -> str | None:
    """The shard of a unsloth GGUF that holds only the n-gram tables (download_gguf.json's "table"): left out, Maya's
    Q6_K tables are read instead (every DeepSeek quant takes them from Maya's repo)."""
    q = ((read_json(DS4_GGUF_MANIFEST).get("quants") or {}).get(DS4_QUANTS.get(quant, "")) or {})
    return Path(q["table"]).name if q.get("table") else None


def ds4_abl_engram_gb() -> float:
    """GB of Maya's Q6_K n-gram tables an orcarouter version reads (the pair the setup downloads when none is there)."""
    man = read_json(DS4_GGUF_MANIFEST)
    eg = (man.get("abliterated") or {}).get("engram") or {}
    tier = eg.get("download", "Maya-XS")
    files = ds4_tier_files(man.get("maya") or {}, tier)
    return sum(files[n][0] for n in (eg.get("from") or {}).get(tier, []) if n in files) / 1e9


def ds4_abl_engram_there(d: Path, models: Path) -> bool:
    """Whether a copy of Maya's Q6_K n-gram tables is there for a quant in d - in its Engram-Q6_K folder, or anywhere a
    Maya setup keeps them in <models> on the same drive (linked, not downloaded) - by the files' sizes."""
    man = read_json(DS4_GGUF_MANIFEST)
    eg, my = (man.get("abliterated") or {}).get("engram") or {}, man.get("maya") or {}
    e = d.parent / eg.get("folder", "Engram-Q6_K")
    try:
        dev = os.stat(existing(d)).st_dev
    except OSError:
        dev = None
    for tier, names in (eg.get("from") or {}).items():
        files = {Path(n).name: ds4_tier_files(my, tier)[n][0] for n in names}
        for where in [e] + ds4_table_homes(models):
            try:
                if all((where / f).stat().st_size == n for f, n in files.items()) and \
                        (where == e or os.stat(where).st_dev == dev):
                    return True
            except OSError:
                continue
    return False


def ds4_abl_engram_step(a, d: Path, models: Path) -> Path | None:
    """Maya's Q6_K Engram tables for a quant whose own table shard is left out (orcarouter's, unsloth's, Maya-XS's: all
    take them from Maya's repo): in
    <d's parent>/Engram-Q6_K, shared by both versions - there already (either Maya quant's pair of shards, whole and as
    published), else hard-linked from a Maya quant set up in <models> on the same drive (no second copy), else downloaded
    from Maya's repo (Maya-XS's pair).  The folder, or None when nothing was downloaded."""
    man = read_json(DS4_GGUF_MANIFEST)
    eg, my = (man.get("abliterated") or {}).get("engram") or {}, man.get("maya") or {}
    if not eg.get("from"):
        fail(f"{DS4_GGUF_MANIFEST} is missing or damaged", "restore it (git checkout tools/ds4/download_gguf.json)")
    e = d.parent / eg.get("folder", "Engram-Q6_K")
    pairs = {t: {Path(n).name: ds4_tier_files(my, t)[n] for n in names} for t, names in eg["from"].items()}

    def whole(where: Path, files: dict) -> bool:
        return all((where / f).is_file() and (where / f).stat().st_size == v[0] for f, v in files.items())

    def checked(files: dict) -> bool:
        return all(sha256_checked(e / f, v[1]) for f, v in files.items())

    for tier, files in pairs.items():
        if whole(e, files):
            if not checked(files):
                fail(f"{e}: the Engram tables do not match their published sha256", "delete them and run the setup again")
            ok(f"the n-gram (Engram) tables: Maya's Q6_K, {e} ({sum(v[0] for v in files.values()) / 1e9:.1f} GB)")
            return e
    for tier, files in pairs.items():   # (a Maya quant of DeepSeek beside it: its own pair, linked)
        src = next((h for h in ds4_table_homes(models) if h != e and whole(h, files)), None)
        if src is None:
            continue
        try:
            e.mkdir(parents=True, exist_ok=True)
            for f in files:
                if not (e / f).exists():
                    os.link(src / f, e / f)
        except OSError as x:
            warn(f"the Engram tables of {src} could not be linked into {e} ({x}): downloaded instead")
            break
        book, theirs = read_json(e / VERIFIED), read_json(src / VERIFIED)   # (the same files: checked there once)
        for f in files:
            if f in theirs and theirs[f].get("size") == (e / f).stat().st_size:
                book[f] = theirs[f]
        try:
            (e / VERIFIED).write_text(json.dumps(book, indent=1), encoding="utf-8")
        except OSError:
            pass
        if not checked(files):
            fail(f"{src}: the Engram tables do not match their published sha256", "delete them and run the setup again")
        ok(f"the n-gram (Engram) tables: Maya's Q6_K, linked from {src} (no second copy) into {e}")
        return e
    tier = eg.get("download", "Maya-XS")
    files = pairs[tier]
    m = {"about": f"Maya's Q6_K n-gram (Engram) tables (DeepSeek's released ones; the two {tier} shards that hold them), "
                  "read instead of orcarouter's own IQ4_NL table - one copy for both orcarouter versions",
         "format": "ds4-engram", "repo": my["repo"], "revision": my["revision"], "folder": tier,
         "files": {f: v[0] for f, v in files.items()}, "sha256": {f: v[1] for f, v in files.items()},
         "download_gb": sum(v[0] for v in files.values()) / 1e9}
    shards = [e / f for f in files]
    if any(file_incomplete(m, s) for s in shards) and not offer_download(a, "the n-gram tables", e, shards, m):
        return None
    if not checked(files):
        fail(f"{e}: the Engram tables do not match their published sha256", "delete them and run the setup again")
    ok(f"the n-gram (Engram) tables: Maya's Q6_K, {e}")
    return e


def ds4_shared_tables(quant: str, d: Path, models: Path, m: dict) -> dict | None:
    """One copy of the Q6_K n-gram (Engram) tables for unsloth's UD-IQ3_S and Maya's quants, whichever is on the disk
    first (download_gguf.json: unsloth's table shard holds the same bytes as Maya's pair of shards, compared in full;
    "tables_as_maya").  Only when the other side's files are whole, by size AND sha256 (never assumed): a UD quant is
    then set up without its table shard (the Maya pair is read), a Maya quant without its pair (unsloth's shard is
    read).  None: nothing to share.  Else {"model": m without the files left out, "without": their names, "src":
    {name: path} of the copy to read, "folder": where the pack reads it from, "what": a sentence for the user}."""
    man = read_json(DS4_GGUF_MANIFEST)
    pairs = {t: [Path(n).name for n in names] for t, names in ((man.get("abliterated") or {}).get("engram") or {})
             .get("from", {}).items()}
    uds = {k: v for k, v in (man.get("quants") or {}).items() if v.get("tables_as_maya")}

    told = []

    def whole(path: Path, size: int, sha: str) -> bool:
        try:
            if path.is_file() and path.stat().st_size == size and not told:
                told.append(1)
                say("  The other model's n-gram (Engram) tables are on this disk: checking they are the same bytes "
                    "(sha256, read once - a few minutes), so that there is one copy of them ...")
            return path.is_file() and path.stat().st_size == size and sha256_checked(path, sha)
        except OSError:
            return False

    def trimmed(names: list) -> dict:
        files = {k: v for k, v in m["files"].items() if k not in names}
        return dict(m, files=files, sha256={k: v for k, v in m["sha256"].items() if k in files},
                    download_gb=sum(files.values()) / 1e9)
    mine = DS4_QUANTS[quant]
    if mine in uds:                                      # unsloth's quant, Maya's pair there?
        table = Path(uds[mine]["table"]).name
        if table not in m["files"] or not pairs:
            return None
        for tier, names in pairs.items():
            q = ds4_tier_files(man.get("maya") or {}, tier)
            full = {Path(n).name: v for n, v in q.items() if Path(n).name in names}
            for where in (models / f"{DS4}-Maya-GGUF" / tier, d.parent / "Engram-Q6_K"):   # (Maya's folder first: its
                # files were checked there, a rerun does not read the 161 GB again through the link)
                if len(full) == len(names) and all(whole(where / n, v[0], v[1]) for n, v in full.items()):
                    return {"model": trimmed([table]), "without": [table], "src": {n: where / n for n in full},
                            "folder": d.parent / "Engram-Q6_K",
                            "what": f"Maya's Q6_K n-gram tables ({tier}'s pair of shards, {where}) are read for "
                                    f"{mine}'s too: its own table shard ({table}) is left out"}
        return None
    tier = mine                                          # a Maya quant, unsloth's shard there?
    names = pairs.get(tier)
    if not names or not all(n in m["files"] for n in names):
        return None
    if all(whole(d / n, m["files"][n], m["sha256"][n]) for n in names):
        return None                                      # (its own pair is there already)
    for other, onames in pairs.items():                  # the other Maya quant's pair (or the shared Engram-Q6_K copy), #776
        if other == tier:
            continue
        q = ds4_tier_files(man.get("maya") or {}, other)
        full = {Path(n).name: v for n, v in q.items() if Path(n).name in onames}
        for where in (models / f"{DS4}-Maya-GGUF" / other, d.parent / "Engram-Q6_K"):
            if len(full) == len(onames) and all(whole(where / n, v[0], v[1]) for n, v in full.items()):
                return {"model": trimmed(names), "without": names, "src": {n: where / n for n in full},
                        "folder": d.parent / "Engram-Q6_K",
                        "what": f"Maya's Q6_K n-gram tables ({other}'s pair of shards, {where}) are read for {tier} too: "
                                f"its own pair is left out"}
    for ud, v in uds.items():
        table = Path(v["table"]).name
        want = {Path(n).name: x for n, x in v["files"].items()}[table]
        where = models / f"{DS4}-GGUF" / ud
        if whole(where / table, want[0], want[1]):
            return {"model": trimmed(names), "without": names, "src": {table: where / table},
                    "folder": d.parent / "Engram-UD-Q6_K",
                    "what": f"unsloth's {ud} table shard ({where / table}) holds the same Q6_K n-gram tables as "
                            f"{tier}'s pair of shards: read for {tier} too, the pair is left out"}
    return None


def share_file(src: Path, dst: Path) -> str:
    """dst as src's one copy: already there, else a hard link (same disk), else a symbolic link, else a copy.  Returns
    which ("there", "hard link", "symbolic link" or "copy")."""
    if dst.exists() or dst.is_symlink():
        if dst.exists() and dst.stat().st_size == src.stat().st_size:
            return "there"
        dst.unlink()
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
        return "hard link"
    except OSError:
        pass
    try:
        os.symlink(src.resolve(), dst)
        return "symbolic link"
    except OSError:
        pass
    need = src.stat().st_size / 1e9
    if shutil.disk_usage(dst.parent).free / 1e9 < need + 3:
        fail(f"not enough free space in {dst.parent} to copy {src.name} ({need:.0f} GB): it could not be linked",
             "free some space, or put the models on one drive (--models-dir)")
    shutil.copy2(src, dst)
    return "copy"


def ds4_share_step(plan: dict) -> Path:
    """Puts the shared tables where the pack reads them (plan["folder"]) and says what was done."""
    how = {share_file(p, plan["folder"] / n) for n, p in plan["src"].items()}
    ok(f"the n-gram (Engram) tables: {plan['what']}; {'/'.join(sorted(how))} into {plan['folder']}"
       + ("" if "copy" in how else " (no second copy of the 161 GB)"))
    return plan["folder"]


def ds4_kv_gb(ctx: int, gpus: int = 1) -> float:
    """GB of VRAM the context's caches take, on all the GPUs together (each holds its layers' caches, every member of
    a tensor-parallel group the group's: counted on each GPU - the upper bound)."""
    return ctx * DS4_KV_BYTES * max(1, gpus) / 2**30


def ds4_recommended_context(vram: float, ram: float, need: float, gpus: int = 1, parallel: int = 1) -> int:
    """The context the setup recommends on this PC, as MiniMax's (mm3_recommended_context): the largest from
    DS4_DEFAULT_CONTEXT up to DS4_RECOMMENDED_CONTEXT (256K) whose caches fit in what VRAM + RAM hold beyond the
    model's files (need: ds4_memory_gb) - else DS4_DEFAULT_CONTEXT (64K), also when the RAM could not be read: a PC
    that reads experts from the SSD reads a long prompt slowly.  parallel: --parallel's conversations, each a whole
    context's caches - the memory is shared among them (and below 64K for several, the largest that fits)."""
    if ram <= 0:
        return DS4_DEFAULT_CONTEXT
    spare = vram + ram - need
    n = max(1, int(parallel or 1))
    lo = DS4_DEFAULT_CONTEXT if n == 1 else DS4_CONTEXTS[0]
    fits = [c for c in DS4_CONTEXTS if lo <= c <= DS4_RECOMMENDED_CONTEXT and n * ds4_kv_gb(c, gpus) <= spare]
    return max(fits) if fits else (DS4_DEFAULT_CONTEXT if n == 1 else DS4_CONTEXTS[0])


def ds4_model(quant: str = "exl3") -> dict:
    """The DeepSeek download as a MODELS entry (offer_download's form): every file, its size, sha256 and repo."""
    q = DS4_EXL3[quant]
    man = read_json(q["manifest"])
    eg = man.get("engram") or {}
    if not man.get("files") or not eg.get("files"):
        fail(f"{q['manifest']} is missing or damaged", f"restore it (git checkout tools/ds4/{q['manifest'].name})")
    both = {**man["files"], **eg["files"]}
    return {"about": f"{DS4} in {q['by']}'s EXL3 quant at {q['bits']} bits per weight (with its DSpark drafter" +
                     (")" if q.get("tower") else " and vision tower)") + ", and the two Engram tables of DeepSeek's "
                     "base repo", "format": "ds4",
            "repo": man["repo"], "revision": man["revision"], "folder": "",
            "files": {k: v[0] for k, v in both.items()}, "sha256": {k: v[1] for k, v in both.items()},
            "sources": {k: (eg["repo"], eg["revision"]) for k in eg["files"]},
            "source_folders": {k: eg["folder"] for k in eg["files"]} if eg.get("folder") else {},
            "download_gb": sum(v[0] for v in both.values()) / 1e9}


def ds4_shards(d: Path) -> list:
    """The files a DeepSeek pack indexes (the EXL3 shards and the Engram tables, or a GGUF's shards)."""
    return sorted(p for p in d.glob("*.safetensors")) or sorted(p for p in d.glob("*.gguf"))


def ds4_dir(a, models: Path, prev: dict, quant: str = "exl3") -> Path:
    """The model's folder: --model-dir, else the folder of the DeepSeek setup before when it holds the same files
    (its pack is inside it), else <models folder>/DeepSeek-V4.1-Flash-1017 (a GGUF: .../DeepSeek-V4.1-Flash-1017-GGUF/
    <its folder>)."""
    if a.model_dir:
        return Path(a.model_dir).expanduser().resolve()
    args = prev.get("args") or []
    if "--pack" in args[:-1] and (prev.get("installer") or {}).get("ds4_quant", "exl3") == quant:
        d = Path(args[args.index("--pack") + 1]).parent
        if (d / "config.json").exists() or (quant not in DS4_EXL3 and ds4_shards(d)):
            return d
    if quant in DS4_EXL3:
        return models / DS4_EXL3[quant]["dir"]
    return models / (f"{DS4}-abliterated-GGUF" if quant in DS4_ABL else f"{DS4}-GGUF") / DS4_QUANTS[quant]


def ds4_link_engram(d: Path, m: dict, models: Path, prev: dict):
    """The Engram tables are the same for every EXL3 quant: missing in d but whole in another DeepSeek folder (the
    other quant's folder, the configs' model folders), hard-linked from it - no second download, no second copy on the
    disk - with its sha256 record.  Anything else (another drive: no hard link) leaves them to the download."""
    dirs = [models / q["dir"] for q in DS4_EXL3.values()] + [Path(p) for p in
            [((read_json(c).get("installer") or {}).get("model_dir")) for c in configs()] + [(prev.get("installer") or
            {}).get("model_dir")] if p]
    for name in (m.get("sources") or {}):
        dst = d / name
        if dst.exists():
            continue
        for o in dict.fromkeys(x.resolve() for x in dirs if x.exists()):
            src = o / name
            if o == d.resolve() or not src.exists() or src.stat().st_size != m["files"][name]:
                continue
            try:
                d.mkdir(parents=True, exist_ok=True)
                os.link(src, dst)
            except OSError:
                continue
            seen = read_json(o / VERIFIED).get(name)
            if seen:   # (a hard link: the same size and time - its check still holds)
                book = read_json(d / VERIFIED)
                book[name] = seen
                try:
                    (d / VERIFIED).write_text(json.dumps(book, indent=1), encoding="utf-8")
                except OSError:
                    pass
            ok(f"{name}: linked from {o} (the same Engram table for every quant)")
            break


def ds4_model_step(a, d: Path, m: dict, announce: bool = True):
    """The model's files in d (downloaded when they are not there): their list, or None when nothing was
    downloaded.  announce=False: the step heading was printed already."""
    if announce:
        step(5, f"the model ({DS4})")
    files = [d / n for n in m["files"]]
    if any(file_incomplete(m, f) for f in files) and not offer_download(a, DS4, d, files, m):
        return None
    if m.get("format") == "ds4-gguf":
        ok(f"{DS4} ({m['folder']}): {len(files)} files, {sum(f.stat().st_size for f in files) / 1e9:.1f} GB in {d}" +
           (" - an earlier layout of its files, used as it is" if m.get("layout") == "earlier" else ""))
        if m.get("engram"):
            eg = sorted(m["engram"].glob("*.gguf"))
            ok(f"the Engram tables: {m['engram']} ({sum(f.stat().st_size for f in eg) / 1e9:.1f} GB, unsloth's MXFP8 "
               "shard of them from an earlier setup)")
        if rotational(d):
            warn(f"{d} is on a spinning hard disk: the engine reads experts and the Engram tables from it while it "
                 "answers - use an NVMe SSD (--models-dir)")
        return files
    conf = read_json(d / "config.json")
    method = (conf.get("quantization_config") or {}).get("quant_method")
    if conf.get("model_type") != "deepseek_v41" or method != "exl3":
        fail(f"{d / 'config.json'}: a {conf.get('model_type')!r} model quantized as {method!r}, not {DS4} in EXL3 "
             "(deepseek_v41, exl3)")
    ok(f"{DS4}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e9:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: the engine reads the Engram tables from it for every token - use an "
             "NVMe SSD (--models-dir)")
    return files


def ds4_pack_step(a, d: Path, gguf: bool = False, announce: bool = True, engram: Path | None = None,
                  files: list | None = None, drafter: Path | None = None, without: list | None = None,
                  vision_dir: Path | None = None) -> Path:
    """tools/ds4/pack.py's pack in <model>/maya-ds4: the geometry, where every tensor sits in the safetensors (read
    in place), the Engram hash tables and the tokenizer - built again when its tools or the model's files changed, or
    (a GGUF) when the drafter or the vision tower arrived after it.  announce=False: no step heading (step 7 adds the
    vision tower it just made).  files: a GGUF's shards (default: every .gguf in d; the folder may also hold an
    earlier layout's).  engram: the folder of the Engram tables v1.1.16's Maya-S leaves out (ds4_earlier's) - a pack of
    other files is built again.  vision_dir (an EXL3 quant whose own tower the engine does not read): the vision folder
    whose BF16 tower pack.py takes instead - a pack without it is built again."""
    if announce:
        step(6, "the pack (the engine's index of the model files)")
    pack = d / "maya-ds4"
    tools = S.source_hash(DS4_GGUF_PACK_TOOLS if gguf else DS4_PACK_TOOLS)
    need = ("ds4.txt", "tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja")
    shards = list(files) if gguf and files else ds4_shards(d)
    if gguf and engram is not None:                    # (v1.1.16's Maya-S: its Engram tables are beside it)
        shards += sorted(engram.glob("*.gguf"))
    drafter_new = gguf and (drafter or d.parent / "dspark" / "dspark.json").is_file() and not CDS.has_drafter(
        {"args": ["--pack", str(pack)]})               # (the drafter arrived after the pack was built)
    if gguf and vision_dir is None:
        vision_dir = ds4_vision_dir(d)
    vision_new = (gguf and (vision_dir / "vision.json").is_file() and not ds4_pack_has_vision(pack)) or (
        not gguf and vision_dir is not None and not ds4_pack_tower(pack))
    if not a.repack and read_json(pack / PACK_SOURCES).get("tools") == tools and not drafter_new and \
            not vision_new and all((pack / f).exists() for f in need) and pack_stale(pack, shards) is None:
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack is written inside the model's folder")
    say(f"  Indexing every tensor of the {'GGUF' if gguf else 'safetensors'} (the weights stay where they are), the "
        "Engram hash tables and the")
    say("  tokenizer (about a minute) ...")
    run([sys.executable, str(ROOT / "tools" / "ds4" / ("pack_gguf.py" if gguf else "pack.py")), "--model", str(d),
         "--out", str(pack)] + (["--vision", str(vision_dir)] if vision_dir is not None else []) +
        (["--engram", str(engram)] if gguf and engram is not None else []) +
        (["--drafter", str(drafter)] if gguf and drafter is not None else []) +
        (["--shards"] + [Path(f).name for f in files] if gguf and files else []) +
        (["--without"] + list(without) if gguf and without else []))
    missing = [f for f in need if not (pack / f).exists()]
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    stamp_pack(pack, shards)
    rec = read_json(pack / PACK_SOURCES)
    rec["tools"] = tools
    (pack / PACK_SOURCES).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    ok(f"pack: {pack}")
    return pack


def ds4_config_path(pc: dict, quant: str) -> Path:
    """The config write_config_ds4 writes for `quant` on this PC's backend: maya-ds4.json (maya-ds4-hip.json on AMD),
    the name it always had - unless that config is another DeepSeek quant's, which stays as it is: then one of this
    quant's own (maya-deepseek-maya-s.json: maya-<its setup id>, as MiniMax's), so the quants are installed side by
    side (two used to share the one name, and the second's setup took the first off the Models page - its files then
    "files no model uses").  A quant that has a config of its own keeps it."""
    suffix = "-hip" if pc.get("backend") == "hip" else ""
    legacy = ROOT / f"maya-ds4{suffix}.json"
    own = ROOT / f"maya-{Q.setup_id('deepseek', quant) or 'deepseek-' + quant}{suffix}.json"
    if own.exists():
        return own
    # (a config without the setup's record - written by hand - is no other quant's: the setup writes it, as before)
    return own if legacy.exists() and Q.config_key(read_json(legacy)) not in (None, quant) else legacy


def write_config_ds4(a, pc: dict, meta: dict, pack: Path, ctx: int, models: Path, d: Path,
                     quant: str = "exl3", vision: bool | None = None) -> Path:
    """maya-ds4.json and run-maya-ds4.sh (a second quant beside it: a config of its own - ds4_config_path).  A setup
    again keeps what the config before says that this one does not ask: the engine settings below, the thinking level
    and sampling, where the server listens and its key - a quant's first config beside another's takes them from that
    one, as it did when it replaced it.  vision: whether the engine runs the vision tower (default: the EXL3 quant's
    own; a GGUF's when its pack has one)."""
    step(8, "the configuration and the start script")
    port = a.port or 8080
    hip = pc.get("backend") == "hip"
    cfg_path = ds4_config_path(pc, quant)
    old = read_json(cfg_path if cfg_path.exists() else ROOT / ("maya-ds4-hip.json" if hip else "maya-ds4.json"))
    # --tp is the tensor-parallel group: two equal GPUs per group (a GPU without an equal neighbour runs
    # its layers alone).  On one GPU that group must be the GPU itself: with --tp 2 the prompt path falls
    # back to 16-token windows, and the chunked path (--threads on there) is what makes prompts fast.
    # Measured on an RTX 4000 Ada 20 GB, one GPU: --tp 1 --threads on reads a 2.7K-token prompt at
    # 205 tok/s against 103 with --tp 2, and a 63.8K one at 595 against 571 (decode 10.8 against 8.5).
    gpus = len(pc["gpus"])
    args = ["--pack", str(pack), "--max-context", str(ctx), "--tp", str(min(2, gpus))]
    if gpus <= 1 and "--threads" not in args[:-1]:
        args += ["--threads", "on"]
    # (--tp follows the GPUs chosen now: the config before's is kept only for the same GPUs - #436, two GPUs -> one
    #  kept --tp 2 and the engine failed later)
    same_gpus = tp_keepable(old, pc)
    kept = keep_args(old.get("args") or [], args, (("--tp",) if same_gpus else ()) +
                     ("--prefill-chunk", "--reserve-mib", "--vram-experts", "--cpu-threads", "--threads", "--dspark",
                      "--dspark-conf", "--expert-profile", "--mmap-experts", "--parallel", "--ssd-cache",
                      "--ssd-policy"))
    if getattr(a, "parallel", None):   # given now: written, and a later setup keeps it
        set_arg(args, "--parallel", a.parallel)
        ok(f"answers at once: {a.parallel} (--parallel; each its own context's caches in VRAM)")
    if not same_gpus and flag_value(old.get("args") or [], "--tp") not in (None, flag_value(args, "--tp")):
        ok(f"--tp {flag_value(args, '--tp')} for the GPU{'s' if gpus != 1 else ''} chosen now (the config before had "
           f"--tp {flag_value(old['args'], '--tp')} for GPUs {old.get('gpu')})")
    if getattr(a, "mmap_experts", None):   # given now: written, and a later setup keeps it (issue #422)
        set_arg(args, "--mmap-experts", a.mmap_experts)
        ok(f"the CPU lane's experts: --mmap-experts {a.mmap_experts}")
    if vision is None:                   # (a GGUF has none of its own: vision_pack.py's, when the pack has it)
        vision = quant in DS4_EXL3 or ds4_pack_has_vision(pack)
    # (the EXL3 quant's tower is the same as the GGUF's - the checkpoint's vision.* in the pack, the same BF16 cuBLAS
    #  GEMMs (src/ds4/ds4_vision.cu) - so it reads pictures on every GPU the model runs on, V100 included)
    if a.no_vision or not vision:
        args += ["--vision", "off"]
    if "--dspark" not in args[:-1] and CDS.has_drafter({"args": args}):
        args += ["--dspark", "none"]                   # (until the speed calibration finds that drafting pays here)
    if getattr(a, "vram_reserve_mib", None) is not None:   # given now: written, and a later setup keeps it
        set_arg(args, "--reserve-mib", a.vram_reserve_mib)
        ok(f"VRAM kept free on each GPU: {a.vram_reserve_mib} MiB (--vram-reserve-mib)")
    elif hip and "--reserve-mib" not in args[:-1]:
        mib = hip_reserve_mib(pc["gpus"])
        set_arg(args, "--reserve-mib", mib)
        ok(f"VRAM kept free on each GPU: {mib} MiB (the desktop and other programs on an AMD GPU allocate beside the "
           "engine; --vram-reserve-mib changes it)")
    cfg = {"exe": str(DS4_EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           **({"backend": "hip"} if hip else {}),
           "model_name": DS4_ABL_MODEL_NAME if quant in DS4_ABL else DS4_MODEL_NAME,
           "gpu": [g["index"] for g in pc["gpus"]],
           "sampling": dict(DS4_SAMPLING), "reasoning_effort": DS4_EFFORT,   # (the model's own, every setup)
           "lib_dirs": meta.get("lib_dirs") or old.get("lib_dirs") or [], "port": port}
    kept += keep_server_settings(cfg, old, a, ("fast_prefill", "mixed_gpus", "mixed_exe", "mixed_gpu", "mixed_mode"))
    set_fast_prefill(cfg, getattr(a, "fast_prefill", None))
    set_mixed_gpus(cfg, getattr(a, "mixed_gpus", None))
    if hip and os.name != "nt" and "mixed_gpus" not in cfg and getattr(a, "mixed_gpus", None) is None and nvidia_beside():
        exe = Path(cfg.get("mixed_exe") or Path(cfg["exe"]).resolve().parents[1] / "build-cuda" / Path(cfg["exe"]).name)
        if exe.is_file():
            say("  two GPUs (experimental): this PC also has an NVIDIA card - the first start measures DeepSeek on the "
                "AMD GPU alone and with the card beside it (the card computes its most-routed experts) and " +
                ("keeps the faster" if DS4_BOTH_AUTO else
                 "says the result; both GPUs stay off until their answers are verified, `--mixed-gpus on` turns them on") +
                "; answers can be worded slightly differently from the AMD GPU alone; `--mixed-gpus split` "
                "gives the card the first layers instead - long prompts read faster still, decode about the same; "
                "`--mixed-gpus off` turns it off)")
        else:
            say(f"  (this PC also has an NVIDIA card: with a CUDA build of maya-ds4 at {exe} (its build stopped above) "
                "DeepSeek can run on both GPUs (`--mixed-gpus on`))")
    env = {**(old.get("env") or {}), **parse_env(a.env)}
    if env:
        cfg["env"] = env
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if kept:
        ok("kept from the config before: " + ", ".join(kept))
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    cfg["installer"] = {"family": "deepseek", "models_dir": str(models), "quant": DS4, "ds4_quant": quant,
                        "model_dir": str(d), "written": time.strftime("%Y-%m-%d %H:%M")}
    cal = saved_calibration(cfg, any_context=True)     # --tp and the CPU threads tuned on this PC for this quant before
    if cal is not None and isinstance(cal.get("args"), dict):
        had = [f for f in CAL4.KEYS if f in (old.get("args") or [])[:-1] and   # (the config before's own values win,
               (f != "--tp" or same_gpus)]                                       #  its --tp for the same GPUs only)
        new = {k: v for k, v in cal["args"].items() if k not in had}
        if new:
            cfg["args"] = CAL4.apply(cfg["args"], new)
            ok("the settings tuned for this PC earlier are used" + (f" ({cal['date']})" if cal.get("date") else "") +
               f": {ds4_tuned(new)}")
            if cal.get("context"):
                ok(f"(tuned at context {cal['context']}: {ME} --calibrate tunes this context too)")
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, cfg['port'])}")
    if ds4_seed_profile(cfg):
        ok("expert profile: the pack starts from DeepSeek's shipped one (the most-routed experts placed in VRAM and "
           "read ahead first); the engine adds what this PC routes to it")
    return cfg_path


def set_up_ds4(a, prev: dict) -> Path | None:
    """DeepSeek-V4.1-Flash-1017's steps 1-8: the PC (Linux, RTX 30 / A100 or newer, the memory), the context, its
    engine (build/maya-ds4), the files, the pack, the config.  No tuning: the engine balances its own split."""
    quant = choose_ds4_quant(a, prev)
    if prev and Q.config_key(prev) != quant:           # (quants side by side: this quant's own config's answers, if any)
        prev = next((c for c in map(read_json, configs()) if config_family(c) == "deepseek" and
                     c.get("backend", "cuda") == prev.get("backend", "cuda") and Q.config_key(c) == quant), prev)
    inst = prev.get("installer") or {}
    prev_args = prev.get("args") or []
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) if "--max-context" in prev_args[:-1] else None
    gguf = quant not in DS4_EXL3
    a.gpus_before = prev.get("gpu")                    # (the GPU question's default: this config's GPUs)
    pc = check_pc(a, "deepseek", gguf, quant)          # 1
    step(2, "your choices")                            # 2
    ok(f"files: {Q.shown('deepseek', quant)}" + (" (Maya's quant)" if quant in DS4_MAYA else
                                                 " (a gated repo: your Hugging Face token)" if quant in DS4_ABL else
                                                 " (unsloth's GGUF)" if gguf else
                                                 f" ({DS4_EXL3[quant]['by']}'s EXL3 quant at {DS4_EXL3[quant]['bits']} bits)"))
    vram = sum(g.get("vram_gb", 0) for g in pc["gpus"])
    par = getattr(a, "parallel", None) or int(flag_value(prev_args, "--parallel") or 1)   # (--parallel: kept)
    rec = ds4_recommended_context(vram, mem_gb()[0], ds4_memory_gb(quant), len(pc["gpus"]), par)
    ctx = choose_context(a, prev_ctx, DS4_CONTEXTS, rec)
    ok(f"context: {ctx} tokens" + (" (recommended for this PC's memory)" if ctx == rec else "") +
       (f", for each of {par} conversations at once (--parallel)" if par > 1 else ""))
    models = (Path(a.models_dir).expanduser().resolve() if a.models_dir else
              Path(inst["models_dir"]) if inst.get("models_dir") else ROOT.parent / "Maya-data" / "models")
    d = ds4_dir(a, models, prev, quant)
    ok(f"model folder: {d}")
    llama = tools_step(a, PY_PACKAGES + ["tokenizers"])   # 3 (the pack's Engram token map normalizes with it)
    meta = build_step(a, pc, llama, "maya-ds4")      # 4
    mixed_builds(a, pc, llama, "maya-ds4")           # (an NVIDIA card beside an AMD GPU: its engine too)
    m = ds4_gguf_model(quant) if gguf else ds4_model(quant)
    if not gguf:                                       # (the Engram tables from the other EXL3 quant's folder)
        ds4_link_engram(d, m, models, prev)
    old = ds4_earlier(m, d) if gguf else None          # (an earlier layout of the files, whole: used as it is)
    upgrade = ds4_upgrade_from(m, d) if gguf else None   # (Maya-XS v1 whole: v2 is downloaded, the old model files go last)
    if upgrade and not ds4_upgrade_wanted(a, m, d):    # (kept: the earlier layout used as it is, as ds4_earlier's)
        old, upgrade = dict(upgrade, engram=None, layout="earlier"), None
    if upgrade:
        ds4_upgrade_note(m, d)
    shared = None
    look = gguf and old is None and quant not in DS4_ABL and quant != "maya-xs"
    if look:                                           # (its checks of the other model's tables print under the heading)
        step(5, f"the model ({DS4})")
        shared = ds4_shared_tables(quant, d, models, m)
    if shared:                                         # (the same Q6_K tables are there: unsloth's UD-IQ3_S <-> Maya's)
        ok(f"{shared['what']}: {(m['download_gb'] - shared['model']['download_gb']):.0f} GB less to download")
        m = shared["model"]
    # every DeepSeek quant takes its n-gram tables from Maya's repo (Q6_K): unsloth's own table shard is left out unless
    # it is there whole already (nothing is downloaded twice), Maya-XS's tables are not among its model shards
    own_table = ds4_own_table(quant) if gguf and old is None and not shared and quant not in DS4_ABL + DS4_MAYA else None
    if own_table and own_table in m["files"] and not (
            (d / own_table).is_file() and (d / own_table).stat().st_size == m["files"][own_table]):
        m = ds4_trim_files(m, [own_table])
    else:
        own_table = None
    tabled = gguf and old is None and bool(own_table or quant == "maya-xs" or (quant in DS4_ABL and m.get("table")))
    if tabled and not ds4_abl_engram_there(d, models):  # (the tables follow: room for them too)
        m = dict(m, also_gb=ds4_abl_engram_gb(), also_what="Maya's Q6_K n-gram tables, downloaded after it")
    files = ds4_model_step(a, d, old or m, announce=not look)             # 5
    if files is None:
        return None
    if gguf and old is None and not upgrade:
        ds4_unused(m, d)
    engram = old and old["engram"]                     # (v1.1.16's Maya-S: its Engram tables beside it)
    without = [m["table"]] if quant in DS4_ABL and m.get("table") else [own_table] if own_table else None
    if without or (tabled and quant == "maya-xs"):     # (Maya's Q6_K tables; the quant's own table shard, if any, left out)
        if without and (d / without[0]).exists():
            warn(f"{d / without[0]}: the n-gram table the model's publisher ships, not used - Maya's Q6_K tables are "
                 f"read instead; you can delete it ({(d / without[0]).stat().st_size / 1e9:.1f} GB)")
        engram = ds4_abl_engram_step(a, d, models)
        if engram is None:
            return None
    if shared:
        engram, without = ds4_share_step(shared), shared["without"]
    drafter = gguf and ds4_drafter_step(a, d.parent / "dspark", quant)   # (the EXL3 quant has its own)
    own = ds4_quant_drafter(quant) if gguf else None   # (Maya-XS: its own drafter beside the shared one)
    dj = d.parent / "dspark" / own["json"] if own is not None and drafter else None
    tower = None if gguf or a.no_vision else DS4_EXL3[quant].get("tower")   # (DeepSeek's BF16 tower for the quant's own)
    vdir = ds4_vision_dir(d, models) if gguf else ds4_exl3_vision_dir(models) if tower else None
    pack = ds4_pack_step(a, d, gguf, engram=engram, files=files, drafter=dj, without=without,
                         vision_dir=vdir if gguf or vdir is not None and (vdir / "vision.bin").is_file() else None)   # 6
    if gguf:                                           # 7 (DeepSeek's checkpoint's tower, added to the pack)
        vision = ds4_vision_step(a, pc, d, mmproj=quant in DS4_UD, v=vdir)   # (UD quants: unsloth's mmproj, the same tower)
        if vision and not ds4_pack_has_vision(pack):
            pack = ds4_pack_step(a, d, gguf, announce=False, engram=engram, files=files, drafter=dj, without=without,
                                 vision_dir=vdir)
        vision = vision and ds4_pack_has_vision(pack)
    elif tower:                                        # 7 (the quant's own tower is EXL3: DeepSeek's BF16 one instead)
        vision = ds4_vision_step(a, pc, d, v=vdir)
        if vision and not ds4_pack_tower(pack):
            pack = ds4_pack_step(a, d, False, announce=False, vision_dir=vdir)
        vision = vision and ds4_pack_tower(pack)
        if not vision and (vdir / "vision.bin").is_file():
            warn("the pack did not take DeepSeek's vision tower (the reason is above): the model reads text only")
    else:
        step(7, "images (the vision tower)")
        vision = not a.no_vision
        if a.no_vision:
            ok("skipped (--no-vision): the model reads text only")
        else:
            ok("the model's own vision tower runs in the engine: its weights in RAM, no VRAM between pictures (about "
               "320 MB of the first GPU's while one is read); the server resizes the pictures")
    if upgrade:                                        # (the new files are whole and packed: the old model files go)
        ds4_upgrade_drop(upgrade, m, d)
    if gguf:                                           # (an earlier layout's copy of the shared tables: one copy, #777)
        ds4_table_dupes(pack, d)
    cfg_path = write_config_ds4(a, pc, meta, pack, ctx, models, d, quant, vision)   # 8
    ablate_step(cfg_path, "deepseek", models, a)
    if (drafter or not gguf) and CDS.has_drafter(read_json(cfg_path)):
        how = drafting_choice(a, "The DSpark drafter", "10-30")
        if how != "measure":
            keep_drafting(cfg_path, how == "on")
        else:
            if S.UI is not None:
                S.UI.step(9, "speed calibration: the DSpark drafter or none")
            speed_calibration(cfg_path, measure_asked(a))
    return cfg_path


# ------------------------------------------------------------------------------------------------ MiniMax-M3.1-Flash
def mm3_min_cc() -> int:
    """The GPUs maya-mm3's setup takes: DeepSeek's EXL3 reach (MM3_MIN_CC, V100 on) once MM3_ALL_PLATFORMS is on;
    until then RTX 30 / A100 or newer (MM3_MIN_CC_NOW), or - for testing, opt-in - from the compute capability
    MAYA_MM3_MIN_CC names (70: V100 on - the engine's portable kernels before sm_80)."""
    if MM3_ALL_PLATFORMS:
        return MM3_MIN_CC
    v = os.environ.get(MM3_TEST_CC_ENV, "").strip()
    if v.isdigit() and 70 <= int(v) < MM3_MIN_CC_NOW:
        warn(f"{MM3_TEST_CC_ENV}={v}: GPUs before RTX 30 / A100 are taken for testing - {MM3}'s engine has not been "
             "validated on them")
        return int(v)
    return MM3_MIN_CC_NOW


def mm3_gpus() -> str:
    """The NVIDIA GPUs the setup offers MiniMax on, in words."""
    return "V100 / RTX 20 or newer" if MM3_ALL_PLATFORMS else "RTX 30 / A100 or newer"


def mm3_kv_format(args: list) -> str:
    """The KV cache's format a maya-mm3 config runs with: its --kv-cache, else the engine's default."""
    return args[args.index("--kv-cache") + 1] if "--kv-cache" in args[:-1] else MM3_KV_FORMAT


def mm3_kv_gb(ctx: int, fmt: str) -> float:
    """GB of pinned RAM the KV cache takes at a context (KV streaming: every position in RAM)."""
    k, _, v = fmt.partition(",")
    half = lambda x: MM3_KV_BYTES.get(x, MM3_KV_BYTES["bf16"]) / 2
    return ctx * (half(k) + half(v or k)) / 2**30


def mm3_kv_vram_gb(ctx: int, fmt: str) -> float:
    """GB of VRAM the KV cache takes at a context with KV streaming: the ring of the newest positions, the index keys."""
    return mm3_kv_gb(min(ctx, MM3_KV_RING), fmt) + ctx * MM3_INDEX_BYTES / 2**30


def apu_kind(gpus: list) -> str | None:
    """"win" / "linux" when the GPUs include a unified-memory APU (hip_unified_memory: Strix Halo) on Windows / Linux,
    else None.  Windows gives it a fixed carve-out (its VRAM) beside the RAM it counts; Linux's engine draws the GPU's
    memory from the RAM, so there VRAM and RAM are one memory."""
    if not any(hip_unified_memory(g) for g in gpus or []):
        return None
    return "win" if WIN else "linux"


def mm3_context_limit(ctx: int, vram: float, ram: float, fmt: str, need: float = MM3_MEMORY_GB,
                      apu: str | None = None) -> str | None:
    """Why a context's KV cache (streamed: all of it pinned in RAM, a ring and the index keys in VRAM) would cost speed
    on this PC, None when it does not: "ram" - the pinned cache does not fit in the RAM the model leaves (a VRAM that has
    room beyond the model does not hold it: the cache is in RAM); "shared" - a Windows APU, the cache past what Windows
    maps for the GPU (MM3_WIN_APU_KV_SHARE of the RAM); "vram" - what the context adds in VRAM pushes experts out of it
    beyond the share Qwen's context keeps (12% of the VRAM, 6% when VRAM + RAM do not hold the model; not on a Linux
    APU, where the GPU's memory is the RAM)."""
    kv = mm3_kv_gb(ctx, fmt)
    if kv > ram - max(0.0, need - vram):
        return "ram"
    if apu == "win" and kv > MM3_WIN_APU_KV_SHARE * ram:
        return "shared"
    grown = mm3_kv_vram_gb(ctx, fmt) - mm3_kv_vram_gb(MM3_CONTEXTS[0], fmt)
    if apu != "linux" and need + mm3_kv_vram_gb(ctx, fmt) > vram and \
            grown > (0.06 if vram + ram < need else 0.12) * vram:
        return "vram"
    return None


def mm3_recommended_context(vram: float, ram: float, fmt: str, need: float = MM3_MEMORY_GB, parallel: int = 1,
                            apu: str | None = None) -> int:
    """The context the setup recommends on this PC: the largest (up to MM3_TESTED_CONTEXT) whose KV cache costs no
    speed here (mm3_context_limit: it fits in the RAM the model leaves - need: MM3_MEMORY_GB, a GGUF quant's own -, on a
    Windows APU in what Windows maps for the GPU, and pushes no more experts out of VRAM than Qwen's) - else the
    smallest; MM3_DEFAULT_CONTEXT when the RAM could not be read.  parallel: --parallel's conversations (above 1: no KV
    streaming - each conversation's whole cache in VRAM, MM3_PARALLEL_VRAM_SHARE).  apu: apu_kind(the GPUs)."""
    n = max(1, int(parallel or 1))
    if n > 1:
        fits = [c for c in MM3_CONTEXTS if c <= MM3_TESTED_CONTEXT and n * mm3_kv_gb(c, fmt) <= vram * MM3_PARALLEL_VRAM_SHARE]
        return max(fits) if fits else MM3_CONTEXTS[0]
    if ram <= 0:
        return MM3_DEFAULT_CONTEXT
    fits = [c for c in MM3_CONTEXTS if c <= MM3_TESTED_CONTEXT and mm3_context_limit(c, vram, ram, fmt, need, apu) is None]
    return max(fits) if fits else MM3_CONTEXTS[0]


def mm3_context_notes(ctx: int, vram: float, ram: float, fmt: str, need: float = MM3_MEMORY_GB,
                      parallel: int = 1, apu: str | None = None) -> None:
    """What a context costs here, and the warnings (as the dashboard's): past what was run end to end, a cache that
    costs speed here (mm3_context_limit).  With --parallel above 1 the caches are in VRAM, a whole context each."""
    n = max(1, int(parallel or 1))
    kv = mm3_kv_gb(ctx, fmt)
    why = mm3_context_limit(ctx, vram, ram, fmt, need, apu) if n == 1 and ram > 0 else None
    rec = context_label(mm3_recommended_context(vram, ram, fmt, need, n, apu))
    if n > 1:
        ok(f"{n} conversations at once (--parallel): each its own KV cache of about {kv:.1f} GB ({fmt.upper()}), "
           f"{n * kv:.1f} GB of VRAM in all - no KV streaming and no DSpark while it is above 1")
    else:
        ok(f"the context's KV cache: about {kv:.0f} GB of pinned RAM ({fmt.upper()}), its newest 32K positions in VRAM too")
    if ctx > MM3_TESTED_CONTEXT:
        warn(f"past {context_label(MM3_TESTED_CONTEXT)} the engine has not been run end to end yet, and a prompt that "
             "long reads slowly - tell us how it goes")
    if n > 1:
        if n * kv > vram * MM3_PARALLEL_VRAM_SHARE:
            warn(f"{vram:.0f} GB of VRAM: {n} caches of a {context_label(ctx)} context leave fewer experts in VRAM "
                 "(answers slower), and the engine shortens the context where they do not fit - "
                 f"{rec} fits this PC for {n}")
    elif why == "ram":
        warn(f"{vram:.0f} GB of VRAM and {ram:.0f} GB of RAM: with a {context_label(ctx)} context's cache, fewer "
             f"experts stay in RAM (answers slower) and the engine may not start - {rec} fits this PC")
    elif why == "shared":
        warn(f"{ram:.0f} GB of RAM beside the GPU's {vram:.0f} GB: a {context_label(ctx)} context's {kv:.0f} GB cache "
             "fills the RAM Windows lets the GPU use (answers slower) - "
             f"{rec} fits this PC")
    elif why == "vram":
        warn(f"{vram:.0f} GB of VRAM: a {context_label(ctx)} context's index keys "
             f"({mm3_kv_vram_gb(ctx, fmt):.1f} GB in VRAM) leave fewer experts there (answers slower) - {rec} "
             "keeps more of them")


def mm3_models() -> tuple:
    """The MiniMax download as MODELS entries (offer_download's form): the model's files, and its DSpark draft's (in
    the repo's dspark/ folder, so a folder of its own beside them)."""
    man = read_json(MM3_MANIFEST)
    if not man.get("files") or not (man.get("dspark") or {}).get("files"):
        fail(f"{MM3_MANIFEST} is missing or damaged", "restore it (git checkout tools/mm3/download.json)")

    def entry(files: dict, folder: str, about: str) -> dict:
        return {"about": about, "format": "mm3", "repo": man["repo"], "revision": man["revision"], "folder": folder,
                "files": {k: v[0] for k, v in files.items()}, "sha256": {k: v[1] for k, v in files.items()},
                "download_gb": sum(v[0] for v in files.values()) / 1e9}
    return (entry(man["files"], "", f"{MM3} in Mia-AiLab's EXL3 quant at 3.5 bits per weight (EAQ v1), with its "
                                    "vision tower"),
            entry(man["dspark"]["files"], man["dspark"]["folder"], f"{MM3}'s DSpark draft (speculative decoding)"))


def mm3_gguf_entry(quant: str) -> dict:
    """Maya's GGUF quant of MiniMax (download_gguf.json) as a MODELS entry (offer_download's form): its shards and the
    dense layers' indexers file, their sizes and sha256 (empty until published), the repo's folder (the tier's:
    Maya-XS), the VRAM + RAM that hold it (memory_gb), and "earlier_names": the names its files had before, in the same
    order."""
    man = read_json(MM3_GGUF_MANIFEST)
    q = (man.get("quants") or {}).get(MM3_QUANTS[quant]) or {}
    if not q.get("files"):
        fail(f"{MM3_GGUF_MANIFEST} is missing or damaged", "restore it (git checkout tools/mm3/download_gguf.json)")
    files = {Path(k).name: v for k, v in q["files"].items()}
    folder = Path(next(iter(q["files"]))).parent.as_posix()
    return {"about": f"Maya's quant of {MM3} ({q.get('about', 'GGUF')})", "format": "mm3-gguf",
            "repo": man["repo"], "revision": man.get("revision") or "", "folder": "" if folder == "." else folder,
            "files": {k: v[0] for k, v in files.items()}, "sha256": {k: v[1] for k, v in files.items()},
            "download_gb": sum(v[0] for v in files.values()) / 1e9, "memory_gb": q.get("memory_gb", MM3_MEMORY_GB),
            "earlier_names": list(q.get("was") or [])}   # (not "was": GLM's name patterns, old_names)


def mm3_gguf_published(quant: str) -> bool:
    """Whether Maya's quant has been uploaded: download_gguf.json names the commit and every file's sha256."""
    m = mm3_gguf_entry(quant)
    return bool(m["revision"]) and all(m["sha256"].values())


def mm3_memory_gb(quant: str, own: Path | None = None) -> float:
    """The VRAM + RAM that hold MiniMax in these files (the KV cache aside).  A folder of your own (mm3_own_dir): the
    pinned quant's headroom (MM3_MEMORY_GB less its download) on top of the folder's size."""
    if own is not None:
        size = sum(f.stat().st_size for f in own.rglob("*.safetensors")) / 1e9
        return size + max(MM3_MEMORY_GB - mm3_models()[0]["download_gb"], 0.0)
    return MM3_MEMORY_GB if quant == "exl3" else float(mm3_gguf_entry(quant)["memory_gb"])


def choose_mm3_quant(a, prev: dict) -> str:
    """--mm3-quant, else the one set up before, else asked among those this PC runs: Maya's GGUF quants once published
    and the EXL3 quant - by this PC's memory the one recommended."""
    if getattr(a, "mm3_quant", None):
        return a.mm3_quant
    if mm3_own_dir(a, "exl3") is not None:            # --model-dir names an EXL3 folder of your own: its files are the quant
        return "exl3"
    ggufs = [k for k in ("gguf-xs", "gguf-s") if mm3_gguf_published(k)]
    keys = ["exl3"] + ggufs
    before = (prev.get("installer") or {}).get("mm3_quant")
    total, _ = mem_gb()
    # by this PC's VRAM + RAM (fit_recommendation): the EXL3 quant first where it fits, else the largest GGUF, the
    # KV cache at the default context in the engine's format (mm3_kv_gb)
    rec = fit_recommendation({k: mm3_memory_gb(k) for k in keys}, setup_gpus(a), total,
                             mm3_kv_gb(MM3_DEFAULT_CONTEXT, MM3_KV_FORMAT), "exl3")
    if len(keys) == 1:
        return keys[0]
    gb = {k: mm3_models()[0]["download_gb"] if k == "exl3" else mm3_gguf_entry(k)["download_gb"] for k in keys}
    about = {"exl3": f"Mia-AiLab's quant: pictures and DSpark drafting; ~{MM3_MEMORY_GB} GB of VRAM + RAM",
             "gguf-xs": "Maya's quant: experts in ~1.6 bits, the whole model in a 128 GB PC's memory",
             "gguf-s": "Maya's quant: experts in ~2.1 bits"}
    pics = "; pictures and DSpark drafting (their files, ~4 GB, beside it)"
    label = {k: (Q.shown("minimax", k, gb[k]), about[k] + ("" if k == "exl3" else pics)) for k in keys}
    order = sorted(keys, key=lambda k: k != rec)
    default = before if before in order else rec
    pick = choose(f"Which files of {MM3}?", ["Maya's GGUF quants and the EXL3 quant run on NVIDIA from V100 on and "
                                             "on AMD:"],
                  [(label[k][0], "recommended" if k == rec else None, label[k][1]) for k in order],
                  order.index(default), a.yes or a.check)
    return order[pick]


def mm3_dir(a, models: Path, prev: dict, quant: str = "exl3") -> Path:
    """The model's folder: --model-dir, else the folder of the MiniMax setup before when it holds the same quant (its
    pack is inside it), else <models folder>/MiniMax-M3.1-Flash-EXL3-3.5bpw (a GGUF: <models folder>/<its name>)."""
    if a.model_dir:
        return Path(a.model_dir).expanduser().resolve()
    args = prev.get("args") or []
    inst = prev.get("installer") or {}
    if "--pack" in args[:-1] and inst.get("mm3_quant", "exl3") == quant and not inst.get("own"):   # (not a folder of your own)
        d = Path(args[args.index("--pack") + 1]).parent
        if (d / "config.json").exists() or (quant != "exl3" and any(d.glob("*.gguf"))):
            return d
    return models / (MM3_DIR if quant == "exl3" else MM3_QUANTS[quant])


def mm3_gguf_model_step(a, d: Path, quant: str) -> list | None:
    """Maya's GGUF quant's files in d (downloaded once published, when they are not there): the files, or None when
    nothing was downloaded.  Before it is published: the files already in d (--model-dir), by their sizes.  Files
    under the names they had before (the manifest's "was": made before the naming scheme), whole, are used as they
    are."""
    name = Q.shown("minimax", quant)
    step(5, f"the model ({name})")
    m = mm3_gguf_entry(quant)
    files = [d / n for n in m["files"]]
    missing = [f for f in files if file_incomplete(m, f)]
    old = [d / n for n in m["earlier_names"]]
    if missing and len(old) == len(files) and all(o.is_file() and o.stat().st_size == size
                                                  for o, size in zip(old, m["files"].values())):
        ok(f"{name}: its files under their earlier names ({old[0].name}, ...) - used as they are")
        files, missing = old, []
    for f in list(missing):                 # (the indexers' file as v1.1.30 named it: .gguf, the same bytes - renamed)
        o = f.with_suffix(".gguf")
        if f.suffix == ".bin" and not f.exists() and o.is_file() and o.stat().st_size == m["files"][f.name]:
            o.rename(f)
            ok(f"{o.name} renamed {f.name} (its name from v1.1.31 on)")
            missing.remove(f)
    if missing and not mm3_gguf_published(quant):
        fail(f"{name} is not published yet (and {missing[0]} is {file_incomplete(m, missing[0])})",
             f"with its files in a folder: {ME} --setup --model {Q.setup_id('minimax', quant)} --model-dir <folder>")
    if missing and not offer_download(a, name, d, files, m):
        return None
    if not mm3_gguf_published(quant):
        warn(f"{name} is not published yet: the files in {d} are taken by their sizes (no published sha256 to check)")
    else:   # (files already there - copied, or downloaded some other way - by sha256 too: checked once, then remembered;
            #  a damaged file keeps its size and the engine then fails on NaN weights, #672)
        for f, want in zip(files, m["sha256"].values()):
            if not sha256_checked(f, want):
                fail(f"{f}: the sha256 does not match the published one - the file is damaged (its size is right)",
                     f"delete it and run {ME} --setup again: the setup downloads it again")
    ok(f"{name}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e9:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: loading reads all {m['download_gb']:.0f} GB from it - use an NVMe SSD "
             "(--models-dir)")
    return files


def mm3_own_dir(a, quant: str) -> Path | None:
    """The folder --model-dir names when it holds an EXL3 quant of MiniMax that is not the pinned one (Mia-AiLab's 3.5
    bits: the shards and index download.json lists) - another bit rate, another maker: the user's own, which the setup
    takes as it is, whole by its own index.  None: the pinned download (a folder of it half downloaded included: the
    setup carries on with it)."""
    model_dir = getattr(a, "model_dir", None)
    if quant != "exl3" or not model_dir or exl3_folder_family(model_dir) != "minimax":
        return None
    d = Path(model_dir).expanduser().resolve()
    index = d / "model.safetensors.index.json"
    try:
        names = set(json.loads(index.read_text(encoding="utf-8-sig"))["weight_map"].values())
        size = index.stat().st_size
    except (OSError, ValueError, KeyError, AttributeError, TypeError):
        return None                                    # (no index of its own: nothing to go by)
    pinned = mm3_models()[0]["files"]
    shards = {n for n in pinned if n.startswith("model-0") and n.endswith(".safetensors")}
    return d if names != shards or size != pinned.get("model.safetensors.index.json") else None


def mm3_own_step(d: Path) -> list:
    """The files of a MiniMax EXL3 folder of the user's own (mm3_own_dir): its shards and its draft's (dspark/) whole by
    their own headers, the small files the pack reads there - never the pinned list, nothing is downloaded."""
    step(5, f"the model ({MM3}, the EXL3 files in your folder)")
    files = exl3_folder_files(d, ("config.json", "tokenizer.json", "special_tokens_map.json", "generation_config.json"))
    if (d / "dspark" / "model.safetensors.index.json").is_file():
        files += exl3_folder_files(d / "dspark", ("config.json",))
    else:
        warn("no dspark/ folder (MiniMax's draft) beside the model: it decodes without drafting")
    if read_json(d / "config.json").get("vision_config") and not list(d.glob("model-vision-*.safetensors")):
        warn("no model-vision-*.safetensors beside the model: it reads text only")
    ok(f"{MM3}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e9:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: loading reads all of it from there - use an NVMe SSD")
    return files


def mm3_model_step(a, d: Path) -> list | None:
    """The model's files in d and its draft's in d/dspark (downloaded when they are not there): the model's files, or
    None when nothing was downloaded."""
    step(5, f"the model ({MM3})")
    m, dm = mm3_models()
    files = [d / n for n in m["files"]]
    if any(file_incomplete(m, f) for f in files) and not offer_download(a, MM3, d, files, m):
        return None
    dd = d / dm["folder"]
    dfiles = [dd / n for n in dm["files"]]
    if any(file_incomplete(dm, f) for f in dfiles) and not offer_download(a, MM3 + " DSpark", dd, dfiles, dm):
        return None
    conf = read_json(d / "config.json")
    method = (conf.get("quantization_config") or {}).get("quant_method")
    if conf.get("model_type") != "minimax_m3_vl" or method != "exl3":
        fail(f"{d / 'config.json'}: a {conf.get('model_type')!r} model quantized as {method!r}, not {MM3} in EXL3 "
             "(minimax_m3_vl, exl3)")
    ok(f"{MM3}: {len(files) + len(dfiles)} files, {sum(f.stat().st_size for f in files + dfiles) / 1e9:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: loading reads all {m['download_gb']:.0f} GB from it - use an NVMe SSD "
             "(--models-dir)")
    return files + dfiles


def mm3_companions_step(a, d: Path, hip: bool) -> Path | None:
    """A GGUF quant's pictures and drafting: the EXL3 repo's vision tower (MM3_VISION_FILES; not with --no-vision) and
    DSpark draft (dspark/) - download.json's pinned files - in <models folder>/MiniMax-M3.1-Flash-companions, downloaded
    when they are not there (a whole EXL3 download beside the GGUF is used as it is).  The folder pack_gguf.py takes, or
    None: text only, no draft (the download declined).  NVIDIA and AMD alike (hip: kept for the callers)."""
    exl3 = d.parent / MM3_DIR
    if all((exl3 / n).exists() for n in MM3_VISION_FILES) and (exl3 / "dspark" / "config.json").exists():
        ok(f"pictures and drafting: from the EXL3 download beside it ({exl3})")
        return exl3
    m, dm = mm3_models()
    names = ["config.json"] + ([] if a.no_vision else [n for n in MM3_VISION_FILES if n != "config.json"])
    c = d.parent / MM3_COMPANIONS
    vm = dict(m, about=f"{MM3}'s vision tower and its configs from Mia-AiLab's EXL3 repo (pictures with any quant)",
              files={n: m["files"][n] for n in names}, sha256={n: m["sha256"][n] for n in names})
    vm["download_gb"] = sum(vm["files"].values()) / 1e9
    vf, dfiles = [c / n for n in names], [c / dm["folder"] / n for n in dm["files"]]
    if any(file_incomplete(vm, f) for f in vf) and not offer_download(a, "the vision tower", c, vf, vm):
        warn("pictures and drafting: not downloaded - the model reads text only, without the DSpark draft")
        return None
    if any(file_incomplete(dm, f) for f in dfiles) and not offer_download(a, MM3 + " DSpark", c / dm["folder"], dfiles,
                                                                           dm):
        warn("the DSpark draft: not downloaded - no drafting")
    ok(f"pictures{' (off: --no-vision)' if a.no_vision else ''} and drafting: the EXL3 repo's files in {c}")
    return c


def mm3_pack_step(a, d: Path, files: list, gguf: bool = False, companions: Path | None = None) -> Path:
    """tools/mm3/pack.py's pack in <model>/maya-mm3 (a GGUF's: pack_gguf.py's): the geometry, where every tensor sits in
    the safetensors / the GGUF's shards (read in place) and the tokenizer - built again when its tools or the model's
    files changed."""
    step(6, "the pack (the engine's index of the model files)")
    pack = d / "maya-mm3"
    tools = S.source_hash(MM3_GGUF_PACK_TOOLS if gguf else MM3_PACK_TOOLS)
    need = ("mm3.txt", "tensors.txt", "tokenizer/vocab.json", "tokenizer/chat_template.jinja")
    shards = [f for f in files if f.suffix == (".gguf" if gguf else ".safetensors")]
    if not a.repack and read_json(pack / PACK_SOURCES).get("tools") == tools and \
            all((pack / f).exists() for f in need) and pack_stale(pack, shards) is None:
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack is written inside the model's folder")
    say(f"  Indexing every tensor of the {'GGUF' if gguf else 'safetensors'} (the weights stay where they are) and the "
        "tokenizer ...")
    if gguf:
        run([sys.executable, str(ROOT / "tools" / "mm3" / "pack_gguf.py"), "--model", str(d), "--out", str(pack)] +
            (["--companions", str(companions)] if companions else ["--text-only"]) +
            ["--shards"] + [f.name for f in shards])
    else:
        run([sys.executable, str(ROOT / "tools" / "mm3" / "pack.py"), "--model", str(d), "--out", str(pack)])
    missing = [f for f in need if not (pack / f).exists()]
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    stamp_pack(pack, shards)
    rec = read_json(pack / PACK_SOURCES)
    rec["tools"] = tools
    (pack / PACK_SOURCES).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    ok(f"pack: {pack}")
    return pack


def write_config_mm3(a, pc: dict, meta: dict, pack: Path, ctx: int, models: Path, d: Path,
                     quant: str = "exl3", own: bool = False) -> Path:
    """maya-mm3.json and run-maya-mm3.sh (a Maya GGUF quant: maya-minimax-maya-xs.json - maya_quants.config_stem).  A
    setup again keeps what the config before says that this one does not ask: the engine settings below, the thinking
    level and sampling, where the server listens and its key - a GGUF quant's first config takes them from the EXL3
    one (they shared maya-mm3.json before the naming scheme; such a config of this quant is moved to its new name)."""
    step(8, "the configuration and the start script")
    port = a.port or 8080
    hip = pc.get("backend") == "hip"   # (AMD: a GGUF quant - a config of its own, as DeepSeek's)
    suffix = "-hip" if hip else ""
    legacy = ROOT / f"maya-mm3{suffix}.json"
    cfg_path = ROOT / f"{Q.config_stem('minimax', quant) or 'maya-mm3'}{suffix}.json"
    if own:   # (a folder of your own: a config of its own, named after the folder - the pinned quant's stays as it is)
        cfg_path = ROOT / f"maya-{re.sub(r'[^a-z0-9._-]+', '-', d.name.lower()).strip('-') or 'mm3-own'}{suffix}.json"
    old = read_json(cfg_path) if cfg_path.exists() else {} if own else read_json(legacy)
    moved = not own and cfg_path != legacy and (old.get("installer") or {}).get("mm3_quant") == quant and \
        not cfg_path.exists()
    gguf = quant != "exl3"
    args = ["--pack", str(pack), "--max-context", str(ctx)]
    if MM3_PROFILE.exists():
        args += ["--profile", str(MM3_PROFILE)]
    # (a GGUF: the layer split - the EXL3 config's tensor-parallel settings not kept; pictures / drafting as its pack
    #  has them: a kept --vision on without a tower in the pack dropped)
    # (--tp: the config before's for the same GPUs only, else worked out again below from the GPUs chosen now - #436)
    keep = ("--chunk", "--staging", "--cpu-threads", "--reserve-gb", "--resident-cap", "--kv-cache", "--kv-streaming",
            "--kv-resident", "--dspark", "--dspark-conf", "--vision-gpu", "--parallel", "--suffix-draft") + \
        (() if gguf else (("--tp",) if tp_keepable(old, pc) else ()) + ("--cache-gb",))
    kept = keep_args(old.get("args") or [], args, keep)
    if getattr(a, "parallel", None):   # given now: written, and a later setup keeps it
        set_arg(args, "--parallel", a.parallel)
        ok(f"answers at once: {a.parallel} (--parallel; each its own KV cache in VRAM)")
    par = int(flag_value(args, "--parallel") or 1)
    if "--vision" in args[:-1] and args[args.index("--vision") + 1] == "on" and not mm3_pack_has_vision(pack):
        i = args.index("--vision")
        del args[i:i + 2]
    has_draft = "dspark.num_hidden_layers" in (pack / "mm3.txt").read_text(encoding="utf-8")
    for k in ("--dspark", "--dspark-conf"):
        if k in args[:-1] and not has_draft:
            i = args.index(k)
            del args[i:i + 2]
    # KV streaming (maya-mm3 --kv-streaming on), as the GLM engine's: 32768 positions of each layer's K / V in VRAM
    # (--kv-resident N sets another window), all of them in pinned RAM - a longer context leaves the VRAM to experts;
    # MM3_KV_BYTES of RAM a context token by the cache's format (BF16: 15 GB at 128K, 120 GB at 1M)
    # (--parallel above 1: the engine runs each conversation's whole cache in VRAM and turns streaming off with a NOTE -
    #  so a new config does not ask for it; one kept from the config before stays, for the day --parallel is 1 again)
    if "--kv-streaming" not in args[:-1] and par <= 1:
        args += ["--kv-streaming", "on"]
    # tensor parallel (maya-mm3 --tp): the GPUs in groups of 4 (or 2), the first alone when the others make whole
    # groups - each layer's work spread over a group's GPUs; on 3090 + 8 x 5060 Ti decode 27.7 -> 38 tok/s (code),
    # ~19 -> 32 (prose), prompts 35-70 -> ~100 tok/s
    n = len(pc["gpus"])
    if "--tp" not in args[:-1] and not gguf:   # (a GGUF pack: the layer split - maya-mm3's tensor parallel is EXL3's)
        for tp in (4, 2):
            if n > tp and ((n - 1) % tp == 0 or n % tp == 0):
                args += ["--tp", str(tp)]
                break
    # DSpark (the draft in the pack: tools/mm3/pack.py indexes dspark/): drafts verified by the model, the same tokens
    # it would answer alone - about 1.6x the decode on the 3090 + 8x 5060 Ti it was measured on, but where most experts
    # run on the CPU the drafts' windows cost more than they save (DeepSeek's lesson): off until this PC's speed
    # calibration (tools/calibrate_ds4.py, as DeepSeek's) or the user's choice says it pays (set_up_mm3)
    if "--dspark" not in args[:-1] and "dspark.num_hidden_layers" in (pack / "mm3.txt").read_text(encoding="utf-8"):
        args += ["--dspark", "off"]
    # prompt lookup (--suffix-draft 3): on for one GPU, where it measured faster on code and tool calls; on several GPUs
    # (with DSpark) it gained ~2% and the greedy answers repeated less often run to run, so there it is the user's
    # choice (a kept --suffix-draft stays)
    if "--suffix-draft" not in args[:-1] and n == 1:
        args += ["--suffix-draft", "3"]
    # pictures, ON by default: the pack's vision tower in the engine (in RAM, on a GPU only during a picture) - a setup
    # again turns them on (a config's --vision off is not kept); --no-vision frees it for experts
    if getattr(a, "no_vision", False):
        set_arg(args, "--vision", "off")
    elif mm3_pack_has_vision(pack):
        set_arg(args, "--vision", "on")
    if getattr(a, "vram_reserve_mib", None) is not None:   # (#1026) given now: written, and a later setup keeps it
        put = set_vram_reserve(args, "minimax", a.vram_reserve_mib)
        ok(f"VRAM kept free on each GPU: {a.vram_reserve_mib} MiB (--vram-reserve-mib: {put})")
    elif hip and "--reserve-gb" not in args[:-1]:   # (as DeepSeek's on AMD: the desktop's allocations beside the engine)
        mib = hip_reserve_mib(pc["gpus"])
        set_arg(args, "--reserve-gb", f"{mib / 1024:.1f}")
        ok(f"VRAM kept free on each GPU: {mib / 1024:.1f} GB (the desktop and other programs on an AMD GPU allocate "
           "beside the engine)")
    cfg = {"exe": str(MM3_EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           **({"backend": "hip"} if hip else {}),
           "model_name": MM3_MODEL_NAME, "gpu": [g["index"] for g in pc["gpus"]],
           # (the model's own, every setup - an EXL3 folder of the user's own: its generation_config.json's)
           "sampling": folder_sampling(d, MM3_SAMPLING) if own else dict(MM3_SAMPLING),
           "reasoning_effort": MM3_EFFORT,
           "lib_dirs": meta.get("lib_dirs") or old.get("lib_dirs") or [], "port": port}
    kept += keep_server_settings(cfg, old, a)
    env = {**(old.get("env") or {}), **parse_env(a.env)}
    if env:
        cfg["env"] = env
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if kept:
        ok("kept from the config before: " + ", ".join(kept))
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    cfg["installer"] = {"family": "minimax", "models_dir": str(models),
                        "quant": MM3 if quant == "exl3" else MM3_QUANTS[quant], "mm3_quant": quant,
                        "model_dir": str(d), **({"own": True} if own else {}),
                        "written": time.strftime("%Y-%m-%d %H:%M")}
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, cfg['port'])}")
    if moved:   # (this quant's config under the name before: its dashboard settings come along, the old files go)
        shared = legacy.with_suffix(".shared-settings.json")
        if shared.exists() and not cfg_path.with_suffix(".shared-settings.json").exists():
            shared.replace(cfg_path.with_suffix(".shared-settings.json"))
        for p in (legacy, ROOT / f"run-{legacy.stem}.{'bat' if WIN else 'sh'}"):
            p.unlink(missing_ok=True)
        ok(f"{legacy.name} (this quant's config before) is now {cfg_path.name}")
    if MM3_PROFILE.exists():
        ok("expert profile: the most-routed experts (data/expert-profile-mm3.bin) are placed in VRAM")
    return cfg_path


def mm3_pack_has_vision(pack: Path) -> bool:
    """The pack indexes the model's vision tower (tools/mm3/pack.py's vision.* lines)."""
    try:
        return "vision.hidden_size" in (pack / "mm3.txt").read_text(encoding="utf-8")
    except OSError:
        return False


def set_up_mm3(a, prev: dict) -> Path | None:
    """MiniMax-M3.1-Flash's steps 1-8: the files (EXL3, or Maya's GGUF quants), the PC (NVIDIA - mm3_gpus(); a GGUF also
    AMD; the memory), the context (recommended by this PC's memory), its engine (build/maya-mm3), the files, the pack,
    pictures, the config."""
    inst = prev.get("installer") or {}
    prev_args = prev.get("args") or []
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) if "--max-context" in prev_args[:-1] else None
    quant = choose_mm3_quant(a, prev)
    gguf = quant != "exl3"
    own = mm3_own_dir(a, quant)                        # --model-dir: an EXL3 folder of your own, not the pinned download
    need = mm3_memory_gb(quant, own)
    a.gpus_before = prev.get("gpu")                    # (the GPU question's default: this config's GPUs)
    pc = check_pc(a, "minimax", gguf=gguf)             # 1
    total, _ = mem_gb()
    vram = sum(g["vram_gb"] for g in pc["gpus"])
    name = MM3 if not gguf else Q.shown("minimax", quant)
    if vram + total < need:
        warn(f"{vram:.0f} GB of VRAM and {total:.0f} GB of RAM: below the ~{need:.0f} GB that hold all of {name} "
             "(the engine reads what does not fit from the SSD as the page cache allows - slower)")
    step(2, "your choices")                            # 2
    ok(f"files: {name}" + (" (Maya's quant)" if gguf else " (your EXL3 folder)" if own else " (Mia-AiLab's EXL3 quant)"))
    fmt = mm3_kv_format(prev_args)                     # (a setup again keeps the config's --kv-cache)
    par = getattr(a, "parallel", None) or int(flag_value(prev_args, "--parallel") or 1)   # (--parallel: kept)
    apu = apu_kind(pc["gpus"])
    rec = mm3_recommended_context(vram, total, fmt, need, par, apu)
    ctx = choose_context(a, prev_ctx, MM3_CONTEXTS, rec)
    ok(f"context: {ctx} tokens" + (" (recommended for this PC's memory)" if ctx == rec else "") +
       (f", for each of {par} conversations at once (--parallel)" if par > 1 else ""))
    mm3_context_notes(ctx, vram, total, fmt, need, par, apu)
    models = (Path(a.models_dir).expanduser().resolve() if a.models_dir else
              Path(inst["models_dir"]) if inst.get("models_dir") else ROOT.parent / "Maya-data" / "models")
    d = mm3_dir(a, models, prev, quant)
    ok(f"model folder: {d}")
    llama = tools_step(a)                              # 3
    meta = build_step(a, pc, llama, "maya-mm3")        # 4
    files = mm3_gguf_model_step(a, d, quant) if gguf else mm3_own_step(d) if own else mm3_model_step(a, d)   # 5
    if files is None:
        return None
    companions = mm3_companions_step(a, d, a.backend == "hip") if gguf else None  # (a GGUF's pictures and draft)
    pack = mm3_pack_step(a, d, files, gguf, companions)  # 6
    if own:                                            # the folder's own chat template, else MiniMax's (the pack's copy)
        body, label = own_template(d)
        put_template(pack, body)
        ok(f"chat template: {label.replace('the base model', 'MiniMax')}")
    step(7, "images (the vision tower)")
    if a.no_vision:
        ok("pictures: off (--no-vision): the model reads text only")
    elif mm3_pack_has_vision(pack):
        ok("pictures: ON - the model's own vision tower runs in the engine; it stays in RAM and comes to the GPU only "
           "while a picture is read (no VRAM kept for it); the server resizes the pictures, the Chat's picture button "
           "is shown")
    elif gguf:
        ok(f"pictures: none - no vision tower beside {name}: text only")
    else:
        ok("no vision tower in this pack: text only")
    cfg_path = write_config_mm3(a, pc, meta, pack, ctx, models, d, quant, bool(own))   # 8
    ablate_step(cfg_path, "minimax", models, a)
    if S.UI is not None:                               # 9: how this PC reads prompts fastest, measured
        S.UI.step(9, "prefill calibration")
    prefill_calibration(cfg_path, "once, for this setup")
    par = parallel_of(read_json(cfg_path))             # (as written: the config before's --parallel may be another quant's)
    if par > 1:                                        # (the engine drafts for one conversation only: nothing to measure)
        ok(f"DSpark: off while {par} conversations are answered at once; --parallel 1 brings it back")
    elif CDS.has_drafter(read_json(cfg_path)):         # 9: the DSpark drafter, measured on this PC (or chosen)
        how = drafting_choice(a, "MiniMax's DSpark drafter", "10-30")
        if how != "measure":
            keep_drafting(cfg_path, how == "on")
        else:
            if S.UI is not None:
                S.UI.step(9, "speed calibration: the DSpark drafter or none")
            speed_calibration(cfg_path, measure_asked(a))
    return cfg_path


# ------------------------------------------------------------------------------------------------ Hy4-flash
def hy4_layouts(q: dict) -> dict:
    """A size's layouts as the manifest lists them, {name: {"files", "plan", "about"}}: its own ("standard") first,
    then its "layouts" (another trunk for cards the standard one does not fit) - only those with files.  "published":
    false switches a size or a layout off whatever else it lists; "hub" (a layout whose file may be uploaded after a
    release): the setup asks Hugging Face whether the file is there when no folder here has it (hy4_on_hub)."""
    out = {}
    if q.get("published", True) is False:
        return out
    if q.get("files"):
        out[HY4_STANDARD] = {"files": q["files"], "plan": q.get("plan") or {}, "about": "", "published": True}
    for k, v in (q.get("layouts") or {}).items():
        if v.get("files") and v.get("published", True) is not False:
            out[k] = {"files": v["files"], "plan": v.get("plan") or {}, "about": v.get("about", ""),
                      "published": v.get("published", True)}
    return out


_HY4_HUB = {}


def hy4_local_name(rel: str) -> str:
    """A table file's name in a model folder: its name on the Hub - a ".gguf_file" there (a GGUF the Hub's list of
    sizes leaves out: the 8 GB-card trunk) is saved as ".gguf", the name the pack and the engine read."""
    name = Path(rel).name
    return name[:-len("_file")] if name.endswith(".gguf_file") else name


def hy4_on_hub(repo: str, rel: str, size: int, sha: str, revision: str = "main"):
    """Whether Hugging Face has the file `rel` of a repo with this size and sha256 (one HEAD request, not followed to
    the CDN; asked once per run): the commit it is at (a str), False (not there, or another file), None (no answer)."""
    import urllib.error
    import urllib.request
    key = (repo, rel, revision)
    if key not in _HY4_HUB:
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *a, **k):
                return None
        url = HF.format(repo=repo, revision=revision, path=rel)
        try:
            urllib.request.build_opener(NoRedirect).open(urllib.request.Request(url, method="HEAD"), timeout=20)
            got = False                                # (a plain file, not an LFS one: not a model file)
        except urllib.error.HTTPError as e:
            h = e.headers
            if e.code in (301, 302, 303, 307, 308) and h.get("X-Linked-Size"):
                same = int(h["X-Linked-Size"]) == int(size) and (h.get("X-Linked-Etag") or "").strip('"') == sha
                got = (h.get("X-Repo-Commit") or revision) if same else False
            else:
                got = False if e.code in (401, 403, 404) else None
        except (OSError, ValueError):
            got = None
        _HY4_HUB[key] = got
    return _HY4_HUB[key]


def hy4_layout_hub(quant: str, layout: str):
    """hy4_on_hub for the trunk of a layout the table marks "published": "hub"."""
    man = read_json(HY4_MANIFEST)
    lay = hy4_layouts(hy4_quant_table(quant)).get(layout) or {}
    rel, (size, sha) = next(iter(lay["files"].items()))
    return hy4_on_hub(man.get("repo", ""), rel, size, sha)


def hy4_quant_table(quant: str) -> dict:
    """The manifest's entry of a size ("maya-xs" -> its "Hy4-Flash-Maya-XS") or of another set it lists ("q4_k_m":
    Tencent's own Q4_K_M from AngelSlim, its "others"); {} for one it does not list."""
    man = read_json(HY4_MANIFEST)
    if quant == HY4_Q4:
        return (man.get("others") or {}).get(HY4_Q4) or {}
    return (man.get("quants") or {}).get(HY4_QUANTS.get(quant, "")) or {}


def hy4_published(quant: str, layout: str = HY4_STANDARD) -> bool:
    """Whether a size's files are uploaded: download.json names the repo's commit ("revision") and the size has files
    with their sizes and sha256 (the n-gram table's too) - the one switch for the setup's list and the dashboard's.
    Tencent's own Q4_K_M: its repo and the commit it is pinned at beside them."""
    man = read_json(HY4_MANIFEST)
    q = hy4_quant_table(quant)
    lay = hy4_layouts(q).get(layout)
    files = list((lay or {}).get("files", {}).values()) + list((man.get("ngram") or {}).values())
    return bool(man.get("revision")) and bool(lay) and bool(man.get("ngram")) and \
        (quant != HY4_Q4 or bool(q.get("repo") and q.get("revision"))) and \
        all(isinstance(v, list) and len(v) == 2 and v[0] and v[1] for v in files) and \
        (lay.get("published") != "hub" or bool(hy4_layout_hub(quant, layout)))


def hy4_entry(quant: str = HY4_Q4, layout: str = HY4_STANDARD) -> dict:
    """A set of Hy4-Flash's files (download.json) as a MODELS entry (offer_download's form): its trunk and the n-gram
    table, their sizes and sha256, the repo's folder, the VRAM + RAM that hold it (memory_gb), its "plan" (hy4_fit),
    the name of its folder in the models folder ("dir") - a layout's: that layout's trunk, in a folder of its own.
    The n-gram table is one file for every set, from Maya's repo at the table's commit ("sources": where it is not in
    the trunk's folder of the repo).  No quant named: Tencent's own Q4_K_M from AngelSlim - its trunk from their repo
    at the commit it is pinned at, "mtp": the set has the model's MTP block (it drafts), "theirs": the n-gram shard of
    their own set, taken as it is where a folder already has it."""
    man = read_json(HY4_MANIFEST)
    if not isinstance(man.get("quants"), dict) or not man.get("repo") or not man.get("ngram"):
        fail(f"{HY4_MANIFEST} is missing or damaged", "restore it (git checkout tools/hy4/download.json)")
    q4 = quant == HY4_Q4
    q = hy4_quant_table(quant)
    if q4 and not (q.get("files") and q.get("repo") and q.get("revision")):
        fail(f"{HY4_MANIFEST} is missing or damaged", "restore it (git checkout tools/hy4/download.json)")
    lay = hy4_layouts(q).get(layout)
    if not lay:
        offered = [Q.setup_id("hy4", k) for k in HY4_QUANTS if hy4_layouts(hy4_quant_table(k))]
        fail(f"{Q.shown('hy4', quant)} is not available yet" + ("" if layout == HY4_STANDARD else f" in its {layout} layout"),
             "the sizes there are now: " + (", ".join(offered) or "none"))
    files = {hy4_local_name(k): v for k, v in lay["files"].items()}
    hub_names = {hy4_local_name(k): Path(k).name for k in lay["files"] if hy4_local_name(k) != Path(k).name}
    folder = Path(next(iter(lay["files"]))).parent.as_posix()
    npath, nfile = next(iter(man["ngram"].items()))
    nname, nfolder = Path(npath).name, Path(npath).parent.as_posix()
    files[nname] = nfile                               # (the trunk first, the n-gram table last: the set's order)
    rev = man.get("revision") or ""
    repo = q.get("repo") or man["repo"]
    # (a layout uploaded after the release is at a later commit than the table's: the one the Hub answered with,
    #  its file still checked against the table's sha256)
    trunk_rev = q["revision"] if q4 else \
        (_HY4_HUB.get((man["repo"], next(iter(lay["files"])), "main")) or "main") \
        if rev and lay.get("published") == "hub" else rev
    about = q.get("about", "GGUF") + (f"; {lay['about']}" if lay["about"] else "")
    m = {"about": (f"{HY4} in AngelSlim's GGUF ({about}); {q.get('license', 'the repo license applies')}" if q4 else
                   f"Maya's quant of {HY4} ({about}); {man.get('license', 'Tencent Hy License')}"),
         "format": "hy4-gguf", "repo": repo, "revision": trunk_rev, "folder": "" if folder == "." else folder,
         "files": {k: v[0] for k, v in files.items()}, "sha256": {k: v[1] for k, v in files.items()},
         "ngram": nname, "ngram_gb": nfile[0] / 1e9, "download_gb": sum(v[0] for v in files.values()) / 1e9,
         "memory_gb": float(q.get("memory_gb") or 0), "plan": dict(lay["plan"]), "layout": layout,
         "mtp": bool(q.get("mtp")), "theirs": {k: list(v) for k, v in (q.get("theirs") or {}).items()},
         "earlier": {k: list(v) for k, v in (man.get("was") or {}).items() if k in files},
         "dir": q.get("dir") if q4 else HY4_QUANTS[quant] + ("" if layout == HY4_STANDARD else f"-{layout.upper()}")}
    if hub_names:
        m["hub_names"] = hub_names
    if nfolder != folder or repo != man["repo"]:       # (a layout's folder of the repo has its trunk only; another repo's set)
        m["sources"] = {nname: (man["repo"], rev)}
        m["source_folders"] = {nname: "" if nfolder == "." else nfolder}
    return m


def hy4_kv_gb(ctx: int) -> float:
    """Hy4-flash's cache for a context of `ctx` tokens, in GB - on every GPU of a tensor-parallel group."""
    return ctx * HY4_KV_BYTES / 2**30


def hy4_recommended_context(pc: dict) -> int:
    """Tencent's own Q4_K_M: the largest of HY4_Q4_CONTEXTS whose cache takes at most 15% of the smallest GPU's VRAM
    (tensor parallel keeps all of it on every GPU of the group), else the smallest."""
    vram = min((g.get("vram_gb") or 0 for g in pc.get("gpus") or []), default=0)
    fits = [c for c in HY4_Q4_CONTEXTS if hy4_kv_gb(c) <= 0.15 * vram]
    return max(fits) if fits else HY4_Q4_CONTEXTS[0]


def hy4_q4_contexts(gpus: list) -> list:
    """The contexts offered for Tencent's own Q4_K_M (no VRAM plan in the table: the engine works the fit out at its
    start and says what does not fit): HY4_Q4_CONTEXTS, and those of HY4_Q4_LONG - up to the model's 1M - whose cache
    in 8-bit rows (what the engine's --kv auto takes where F16 rows do not fit; every GPU of a tensor-parallel group
    holds all of it) is at most HY4_Q4_LONG_SHARE of the smallest GPU's VRAM."""
    vram = min((float(g.get("vram_gb") or 0) for g in gpus or []), default=0.0)
    per = 13 * (HY4_KV_Q8_ROW + 128 * 2 // 4)
    return HY4_Q4_CONTEXTS + [c for c in HY4_Q4_LONG if c * per / 2**30 <= HY4_Q4_LONG_SHARE * vram]


def hy4_cache_bytes(ctx: int, chunk: int, q8: bool = False, resident: int = 0) -> int:
    """The bytes of maya-hy4's caches in VRAM at a context (src/hy4/hy4_model.cu's plan): the window layers' ring of
    latents (the newest 512 + max(16384, chunk) positions), the global layers' latents and pooled keys for every
    position, and the ring's hidden states.  q8: with the latent cache in 8-bit rows (the engine's --kv q8, and its
    --kv auto where F16 rows do not fit: 672 bytes a row against 1152, the indexer's rows F16 against F32).
    resident: with KV streaming (the engine's --kv-streaming) - the global layers' latents a ring of that many
    positions (a chunk's at least) with a tag each, the others in RAM (hy4_stream_ram_gib), and the picked rows'
    buffer; a context no larger than the ring is not streamed."""
    ring = min(ctx, 512 + max(16384, chunk))
    row, pool = (HY4_KV_Q8_ROW, 2) if q8 else (576 * 2, 4)
    res = max(int(resident), chunk) if resident else 0
    if res and ctx > res:
        rows = res * (row + 4)
        pick = 64 * (2048 + 4) * (row + 8) + ctx * 4 + 1024
    else:
        rows, pick = ctx * row, 0
    return 29 * ring * row + 13 * (rows + (ctx // 4 + 1) * 128 * pool + ring * 2 * 128 * pool) + ring * 3072 * 4 + pick


def hy4_stream_ram_gib(ctx: int, q8: bool = False) -> float:
    """KV streaming's RAM (GiB, pinned): every position's latent row of the 13 global layers."""
    return 13 * ctx * (HY4_KV_Q8_ROW if q8 else 576 * 2) / 2**30


def hy4_stream_need_gib(plan: dict, ctx: int, q8: bool = False, resident: int | None = None) -> float:
    """hy4_fit_need_gib with KV streaming: less the global layers' rows that leave the VRAM."""
    chunk = int(plan.get("chunk") or 1024)
    res = HY4_KV_RESIDENT if resident is None else resident
    return hy4_fit_need_gib(plan, ctx, q8) - \
        (hy4_cache_bytes(ctx, chunk, q8) - hy4_cache_bytes(ctx, chunk, q8, res)) / 2**30


def hy4_stream_rows(plan: dict, ctx: int, usable: float, ram_gib: float | None, kv: str | None = None) -> tuple | None:
    """A context that fits the VRAM only with KV streaming, as the engine's --kv-streaming auto takes it:
    (rows, ring) - "f16" then "q8", with the ring of HY4_KV_RESIDENT positions and then a prompt chunk's - where the
    RAM left for the rows (ram_gib) holds them; None: no RAM figure given, or it fits neither way.  kv "f16" / "q8":
    those rows only (the engine's --kv f16 / q8)."""
    if ram_gib is None:
        return None
    for res in (HY4_KV_RESIDENT, 1):
        for q8 in ((kv == "q8",) if kv in ("f16", "q8") else (False, True)):
            if hy4_stream_need_gib(plan, ctx, q8, res) <= usable:
                # (the engine stops at the first that fits the VRAM: the RAM must hold that one's rows)
                return ("q8" if q8 else "f16", res) if hy4_stream_ram_gib(ctx, q8) <= ram_gib else None
    return None


def hy4_ram_for_rows(q: dict, usable: float, ram_gb: float | None) -> float | None:
    """The RAM (GiB) KV streaming's rows may take on this PC: its RAM less the system's headroom (8 GiB or a tenth)
    and the size's experts outside the VRAM (the CPU lane reads them from the file cache); None: no RAM figure."""
    if not ram_gb:
        return None
    return float(ram_gb) - max(8.0, 0.1 * float(ram_gb)) - max(0.0, float(q.get("memory_gb") or 0) - usable)


def hy4_config_stream(quant: str, layout: str, ctx: int, usable: float, ram_gib: float | None,
                      chunk: int | None = None, reserve_gb: float | None = None, mtp: bool = False,
                      kv: str | None = None) -> tuple | None:
    """hy4_stream_rows for a size already set up, counted as hy4_config_need_gib counts it (the Models page's
    serve/models.py hy4_room): (rows, positions kept in VRAM, VRAM GiB) where the context fits its cards only with KV
    streaming and the RAM left for the rows (ram_gib: hy4_ram_for_rows) holds them; kv "f16" / "q8": the config's own
    rows only."""
    if ram_gib is None:
        return None
    plan = (hy4_layouts(hy4_quant_table(quant)).get(layout or HY4_STANDARD) or {}).get("plan") or {}
    ch = int(chunk or plan.get("chunk") or 1024)
    for res in (HY4_KV_RESIDENT, 1):
        for q8 in ((kv == "q8",) if kv in ("f16", "q8") else (False, True)):
            base = hy4_config_need_gib(quant, layout, ctx, chunk=chunk, reserve_gb=reserve_gb, q8=q8, mtp=mtp)
            if base is None:
                return None
            need = base - (hy4_cache_bytes(ctx, ch, q8) - hy4_cache_bytes(ctx, ch, q8, res)) / 2**30
            if need <= usable:
                return ("q8" if q8 else "f16", max(res, ch), need) if hy4_stream_ram_gib(ctx, q8) <= ram_gib else None
    return None


def hy4_need_gib(plan: dict, ctx: int, q8: bool = False) -> float:
    """The VRAM (GiB, its reserve included) maya-hy4 needs at a context for everything that is not a routed expert -
    these parts have no RAM fallback.  From the plan's table ("need_gib", by context); a context it does not list:
    the nearest one below it (else its smallest) plus what the caches grow by (hy4_cache_bytes).  q8: with the
    latent cache in 8-bit rows - the plan's own table for them ("need_gib_q8": measured on a card), else the F16
    figure less what the 8-bit rows save at that context."""
    table = {int(k): float(v) for k, v in (plan.get("need_gib") or {}).items()}
    if not table:
        return 0.0
    chunk = int(plan.get("chunk") or 1024)
    if q8:
        table8 = {int(k): float(v) for k, v in (plan.get("need_gib_q8") or {}).items()}
        if ctx in table8:
            return table8[ctx]
        return hy4_need_gib(plan, ctx) - (hy4_cache_bytes(ctx, chunk) - hy4_cache_bytes(ctx, chunk, True)) / 2**30
    if ctx in table:
        return table[ctx]
    base = max((c for c in table if c <= ctx), default=min(table))
    return table[base] + (hy4_cache_bytes(ctx, chunk) - hy4_cache_bytes(base, chunk)) / 2**30


def hy4_fit_need_gib(plan: dict, ctx: int, q8: bool = False) -> float:
    """hy4_need_gib as the fit counts it: a context the plan's table does not list (worked out, not measured on a
    card) with HY4_FIT_MARGIN_GIB on top."""
    listed = plan.get("need_gib_q8" if q8 else "need_gib") or {}
    return hy4_need_gib(plan, ctx, q8) + (0.0 if str(ctx) in listed else HY4_FIT_MARGIN_GIB)


def hy4_kv_rows(need_f16: float, need_q8: float, usable: float, kv: str | None = None) -> str | None:
    """The latent cache's rows the engine takes at a context, as its --kv auto does it on every layout: "f16" where
    the context fits in the VRAM with F16 rows, "q8" (8-bit rows) where it fits only with them, None where it fits
    neither way.  kv: a config's own --kv ("f16" / "q8": only that one)."""
    if kv != "q8" and need_f16 <= usable:
        return "f16"
    if kv != "f16" and need_q8 <= usable:
        return "q8"
    return None


def hy4_config_need_gib(quant: str, layout: str, ctx: int, chunk: int | None = None,
                        reserve_gb: float | None = None, q8: bool = False, mtp: bool = False,
                        dfly2: bool = False, margin: bool = True) -> float | None:
    """The VRAM (GiB) a size already set up needs at a context, as hy4_fit counts it (the dashboard's Models page
    asks before it writes a context into a config - serve/models.py's hy4_room): its layout's plan
    (hy4_fit_need_gib), a prompt chunk above the plan's as hy4_fit doubles it (HY4_CHUNK_UP_GIB), and a --reserve-gb
    other than the plan's by the difference.  q8: with the latent cache in 8-bit rows (what the engine's --kv auto
    takes where F16 rows do not fit).  mtp: a Maya size that drafts (--mtp) - with its draft block's VRAM
    (hy4_mtp_gib); dfly2: one that drafts with its DFly2 drafter (--dfly2: hy4_dfly2_gib).  margin False: a context
    the plan does not list without HY4_FIT_MARGIN_GIB (the sums themselves - hy4_room_measure's).  None for a size or
    a layout the manifest has no plan for.  The engine counts a drafter's room by the pack, not by the flag: the MTP
    block and the DFly2 drafter (one GPU) the pack holds, with drafting on and off - both: the larger room once."""
    plan = (hy4_layouts(hy4_quant_table(quant)).get(layout or HY4_STANDARD) or {}).get("plan") or {}
    if not plan.get("need_gib"):
        return None
    need = hy4_fit_need_gib(plan, int(ctx), q8) if margin else hy4_need_gib(plan, int(ctx), q8)
    if chunk and int(plan.get("chunk") or 0) and int(chunk) > int(plan["chunk"]):
        need += HY4_CHUNK_UP_GIB
    if reserve_gb is not None and "reserve_gb" in plan:
        need += float(reserve_gb) - float(plan["reserve_gb"])
    # (one drafter at a time: the engine counts the larger room of the two a pack holds, with drafting on and off)
    return need + max(hy4_mtp_gib(int(chunk or plan.get("chunk") or 1024)) if mtp and quant != HY4_Q4 else 0.0,
                      hy4_dfly2_gib(quant) if dfly2 else 0.0)


def hy4_amd(gpus: list) -> bool:
    """These are AMD cards (the HIP backend's: setup's amd_gpus entries, arch gfx...)."""
    return any(g.get("vendor") == "amd" or str(g.get("arch") or "").startswith("gfx") for g in gpus or [])


def amd_gpu_limit_gib() -> float | None:
    """Linux: the memory the amdgpu driver lets programs take on a GPU - its VRAM and its share of the RAM (GTT: on
    an APU nearly all of what the engine uses; half the RAM unless the kernel was told more, e.g.
    ttm.pages_limit) - the largest card's, from sysfs; None where it does not say."""
    best = None
    for d in sorted(Path("/sys/class/drm").glob("card[0-9]*/device")) if sys.platform.startswith("linux") else ():
        try:
            if (d / "vendor").read_text().strip().lower() != "0x1002":
                continue
            b = int((d / "mem_info_gtt_total").read_text()) + int((d / "mem_info_vram_total").read_text())
        except (OSError, ValueError):
            continue
        best = max(best or 0.0, b / 2**30)
    return best


def hy4_apu_gib(gpus: list) -> float:
    """The memory (GiB) a unified-memory APU (hip_unified_memory: Strix Halo / Gorgon Halo) has for the model on its
    GPU: the PC's one memory (apu_memory) less what the engine keeps free for the system and the other programs
    (apu_margin_gb, as src/hy4's own headroom) - and on Linux no more than the driver lets the GPU take
    (amd_gpu_limit_gib).  Windows (experimental for Hy4-Flash): the GPU's share Windows gives it (Variable Graphics
    Memory; HIP's figure), less what the engines leave the desktop there (hip_reserve_mib) - src/hy4 keeps within its
    dedicated memory as GLM's and DeepSeek's engines do, and the RAM beside it is Windows' and the CPU lane's."""
    vram = sum(float(g.get("vram_gb") or 0) for g in gpus or [])
    if WIN:
        return max(vram - len(gpus or []) * hip_reserve_mib(gpus) / 1024.0, 0.0)
    ram, _ = mem_gb()
    pool = apu_memory(vram, ram, not WIN) - apu_margin_gb(ram)
    lim = None if WIN else amd_gpu_limit_gib()
    return max(min(pool, lim) if lim else pool, 0.0)


def hy4_apu_note(gpus: list, q: dict, name: str) -> tuple | None:
    """On a unified-memory APU: (True, line) where its memory holds all of a size - every routed expert sits in the
    GPU's memory, none is computed on the CPU - else (False, line): what the GPU's share does not hold is computed on
    the CPU from the same memory, or read from the SSD.  None on other PCs, or with no figure for the size."""
    if not hy4_amd(gpus) or not any(hip_unified_memory(g) for g in gpus):
        return None
    need, have = float(q.get("memory_gb") or 0), hy4_apu_gib(gpus)
    if need <= 0:
        return None
    if have >= need:
        return True, (f"unified memory: all of {name} (about {need:.0f} GB with its caches) sits in the GPU's memory - "
                      f"{have:.0f} GB of this PC's memory are the model's to take; no expert is computed on the CPU")
    ram, _ = mem_gb()
    lim = None if WIN else amd_gpu_limit_gib()
    why = (f"Windows gives the GPU {have:.0f} GB of this PC's memory for the model (its share: Variable Graphics "
           "Memory in AMD Software)" if WIN else
           f"Linux lets the GPU take {lim:.0f} GB of this PC's memory (the kernel's limit for it: ttm.pages_limit, "
           "see docs/AMD_MAYA.md)" if lim and lim < apu_memory(0.0, ram, not WIN) - apu_margin_gb(ram) - 0.5 else
           f"this PC's memory leaves {have:.0f} GB for the model")
    return False, (f"unified memory: {name} takes about {need:.0f} GB with its caches and {why}: the engine keeps the "
                   "most-routed experts in the GPU's memory and computes the others on the CPU - slower; a smaller "
                   "size sits in the GPU's memory whole")


def hy4_usable_gib(gpus: list, used: dict | None = None) -> float:
    """The VRAM (GiB) the chosen cards have for the model: their total less what is in use on each now (nvidia-smi;
    at most HY4_FIT_USED_CAP_GIB a card counts - more is a program that can be closed).  No reading: one card is
    taken to drive a desktop (HY4_FIT_DISPLAY_GIB).  AMD cards (nvidia-smi does not read them): their VRAM less what
    the engines leave the desktop on each (hip_reserve_mib); a unified-memory APU: hy4_apu_gib.
    A card with "engine_room_gb" (Windows: hy4_engine_room, measured on this PC): no more than that on it - what
    the engine's CUDA gets there (Windows keeps part of the VRAM from every program, whatever nvidia-smi says)."""
    total = sum(float(g.get("vram_gb") or 0) for g in gpus or [])
    if hy4_amd(gpus):
        if any(hip_unified_memory(g) for g in gpus):
            return hy4_apu_gib(gpus)
        return max(total - len(gpus) * hip_reserve_mib(gpus) / 1024.0, 0.0)
    used = gpu_used_mib() if used is None else used
    if any(g.get("engine_room_gb") is not None for g in gpus or []):
        def card(g: dict) -> float:                    # (a card's own: as below, then no more than its room)
            mine = float(g.get("vram_gb") or 0) - (
                min(float(used.get(g.get("index"), 0)) / 1024.0, HY4_FIT_USED_CAP_GIB) if used else
                HY4_FIT_DISPLAY_GIB if g is gpus[0] else 0.0)
            room = g.get("engine_room_gb")
            return mine if room is None else min(mine, float(room))
        return max(sum(card(g) for g in gpus), 0.0)
    if not used:
        return max(total - (HY4_FIT_DISPLAY_GIB if gpus else 0.0), 0.0)
    return max(total - sum(min(float(used.get(g.get("index"), 0)) / 1024.0, HY4_FIT_USED_CAP_GIB) for g in gpus), 0.0)


def hy4_engine_room(pc: dict, probe=None, saved: dict | None = None) -> dict:
    """Windows: each chosen NVIDIA card's VRAM as the engine's CUDA gets it, measured now, kept on the card as
    "engine_room_gb" (hy4_usable_gib counts no more than that there); {index: GiB} of what was measured.  Windows
    keeps part of every card from the programs on it (the display driver's budget), which nvidia-smi does not show:
    an RTX 3050 8 GB with 0.1 GB in use gave CUDA 7.0 GB free - the engine plans with that figure (cudaMemGetInfo at
    its start), the setup's own sums with the card's total less what is in use.  So the room is that figure in the
    setup's terms: CUDA's free VRAM with a context held, plus what CUDA itself takes beside it - a context (the free
    VRAM with a second one beside it, the difference) and the part of the card the driver keeps from CUDA
    (nvidia-smi's total less CUDA's) - CUDA's own share as the setup's sums count it.  Windows keeps that share out
    of sight in its budget, so the engine measures it at its first start on this PC (hy4_room_measure: the engine's
    --room-report, kept in HY4_ROOM_STORE for this PC, card and driver) and the room counts that figure from then on
    ("engine_room_how": "measured"); until then the larger of what this PC shows of it and HY4_FIT_CUDA_OWN_GIB, the
    first guess ("estimated").  "cuda_free_gb": CUDA's own reading (the config keeps it: the Models page counts a
    figure measured later with it).  On Linux the measured part alone is nvidia-smi's total less its used (nothing
    else is kept there), so only Windows is measured.  Not measured: another OS, AMD, a unified-memory GPU, a card a
    Maya engine holds now (its own VRAM would be counted as kept: the reading as before), no answer from the driver.
    Several cards: each one's share from their record together where a start on all of them measured it
    (hy4_room_set_key - a split's or a tensor-parallel group's plan is not one card's), else each card's own.
    probe: the readings (tests; else cuda_room_bytes); saved: HY4_ROOM_STORE's records (tests; else read)."""
    gpus = (pc or {}).get("gpus") or []
    if not WIN or (pc or {}).get("backend", "cuda") != "cuda" or not gpus or hy4_amd(gpus) or \
            any(g.get("unified_memory") for g in gpus):
        return {}
    cards = [g for g in gpus if g.get("pci") and g.get("vram_gb")]   # (CUDA's devices are matched by PCI bus)
    busy = maya_engine_mib() if cards else {}
    cards = [g for g in cards if g.get("index") not in busy]
    got = (cuda_room_bytes() if probe is None else probe) if cards else {}
    saved = hy4_room_load() if saved is None else saved
    # (several cards: their record together where one start on all of them measured it - hy4_room_measure_set)
    together = ((saved.get(hy4_room_set_key(cards)) or {}).get("cards") or {}) \
        if len(cards) > 1 and len(cards) == len(gpus) else {}
    out = {}
    for g in cards:
        r = got.get(S._pci_key(str(g["pci"])))
        if r:
            one, two, tot = (float(x) / 2**30 for x in r)
            rec = saved.get(hy4_room_key(g)) or {}
            mine = together.get(S._pci_key(str(g["pci"])))
            if isinstance(mine, (int, float)):
                own, how = float(mine), "measured"
            elif isinstance(rec.get("own_gib"), (int, float)):
                own, how = float(rec["own_gib"]), "measured"
            else:
                own = max(max(one - two, 0.0) + max(float(g["vram_gb"]) - tot, 0.0), HY4_FIT_CUDA_OWN_GIB)
                how = "estimated"
            g["engine_room_gb"] = out[g["index"]] = one + own
            g["engine_room_how"], g["cuda_free_gb"] = how, one
    return out


def hy4_room_key(g: dict) -> str:
    """What a card's measured share (HY4_ROOM_STORE) holds for: this PC, the card (its name, PCI place and VRAM) and
    its driver - a new driver or another card in the slot is measured again."""
    return "|".join([platform.node() or "?", str(g.get("name") or "?"), S._pci_key(str(g.get("pci") or "?")),
                     f"{float(g.get('vram_gb') or 0):.2f}", str(g.get("driver") or "?")])


def hy4_room_load() -> dict:
    """HY4_ROOM_STORE's records ({hy4_room_key: {"own_gib": ..., ...}}); {} where there are none."""
    try:
        got = json.loads(HY4_ROOM_STORE.read_text(encoding="utf-8"))
        return got if isinstance(got, dict) else {}
    except (OSError, ValueError):
        return {}


def hy4_room_due(cfg: dict, gpus: list | None = None, saved: dict | None = None, probe=None) -> dict | None:
    """The server's, just before it starts Hy4-Flash's engine: the card to measure CUDA's own share on at this start
    (the engine's --room-report, then hy4_room_measure) - Windows, an NVIDIA card, the config on one GPU, no record
    of it for this PC, card and driver yet - with what the setup's CUDA probe sees of it now, before the engine holds
    it ("cuda_probe": cuda_room_bytes' reading; Windows' figure for another program drops as the engine fills the
    card); None otherwise, or where the probe has no reading.  The config on several GPUs (a layer split or tensor
    parallel): {"name", "cards": each card so} where these cards have no record together yet (hy4_room_set_key;
    each card is measured in it: hy4_room_measure_set).  gpus, probe: setup.py's list and the readings (tests; else
    read)."""
    if not WIN or (cfg.get("backend") or "cuda") != "cuda" or             Path(str(cfg.get("exe") or "")).stem.lower() != "maya-hy4":
        return None
    sel = cfg.get("gpu")                               # (one card, several, or "0,2" - serve's gpu_list)
    sel = [] if sel is None or sel == "" else list(sel) if isinstance(sel, (list, tuple)) else str(sel).split(",")
    sel = [int(str(x).strip()) for x in sel if str(x).strip() != ""]
    if not sel:
        return None
    found = {g["index"]: g for g in ((S.gpus() if gpus is None else gpus) or [])}
    cards = [found.get(i) for i in sel]
    if any(not g or not g.get("pci") or not g.get("vram_gb") or g.get("unified_memory") for g in cards):
        return None
    saved = hy4_room_load() if saved is None else saved
    if (hy4_room_key(cards[0]) if len(cards) == 1 else hy4_room_set_key(cards)) in saved:
        return None
    probe = cuda_room_bytes() if probe is None else probe
    out = []
    for g in cards:
        r = probe.get(S._pci_key(str(g["pci"])))
        if not r:
            return None
        out.append(dict(g, cuda_probe=[int(x) for x in r]))
    return out[0] if len(out) == 1 else {"name": f"{len(out)} cards", "cards": out}


def hy4_room_set_key(cards: list) -> str:
    """What the measured shares of several cards in one start (HY4_ROOM_STORE) hold for: this PC and the cards (each
    one's name, PCI place, VRAM and driver, as hy4_room_key) - another card in the set, or a driver, is measured
    again; a start on one of them alone has its own record (hy4_room_key)."""
    return "|".join([platform.node() or "?", "set"]) + "|" + "+".join(sorted(hy4_room_key(g).split("|", 1)[1]
                                                                              for g in cards))


def hy4_room_store(key: str, rec: dict, same) -> dict | None:
    """rec kept in HY4_ROOM_STORE under key - a record same(key) says is the same cards' under another driver gone."""
    store = {k: v for k, v in hy4_room_load().items() if not same(k)}
    store[key] = rec
    try:
        HY4_ROOM_STORE.parent.mkdir(parents=True, exist_ok=True)
        HY4_ROOM_STORE.write_text(json.dumps(store, indent=1) + "\n", encoding="utf-8")
    except OSError:
        return None
    return rec


def hy4_room_need(cfg: dict, info: dict) -> tuple:
    """(the setup's sums for the start the engine reported on - hy4_config_need_gib with the cache's rows and the
    drafters' room the engine took, no margin -, its context, size, layout); need None where there are none."""
    args, inst = [str(x) for x in cfg.get("args") or []], cfg.get("installer") or {}

    def arg(k: str):
        return args[args.index(k) + 1] if k in args[:-1] else None
    quant, layout = inst.get("hy4_quant") or "", inst.get("layout") or HY4_STANDARD
    ctx = int(arg("--max-context") or 0)
    need = hy4_config_need_gib(quant, layout, ctx, chunk=int(arg("--chunk")) if arg("--chunk") else None,
                               reserve_gb=float(arg("--reserve-gb")) if arg("--reserve-gb") else None,
                               q8=info.get("room_rows") == "q8",
                               mtp=bool(int(info.get("room_mtp") or 0)) if "room_mtp" in info else "--mtp" in args,
                               dfly2=bool(int(info.get("room_drafter") or 0)), margin=False) if ctx else None
    return need, ctx, quant, layout


def hy4_room_measure(cfg: dict, card: dict, info: dict, probe=None, save: bool = True) -> dict | None:
    """After the engine started with --room-report on `card` (hy4_room_due): CUDA's own share on it as this PC shows
    it, kept in HY4_ROOM_STORE.  The engine says how much free VRAM CUDA gave it (free) and how much less of it its
    start's choices hold with (slack: the cache's rows, the drafter's room, the pictures, that it starts - its own
    plan, so exactly what decides).  In the setup's terms that start takes the setup's sums for it (need:
    hy4_config_need_gib with the rows and the drafter the engine took, no margin) and had `slack` more: room = need +
    slack, against what the setup's CUDA probe saw of the card just before the start (card["cuda_probe"]:
    hy4_room_due's, else probe's - free with a context held) - the
    share is room - probe.  Not kept: a start that streams its cache (other sums), a figure out of all reason
    (beyond -1 / +2 GiB), no reading.  The record, or None."""
    if card.get("cards"):
        return hy4_room_measure_set(cfg, card["cards"], info, probe, save)
    try:
        gpus = str(info.get("room_gpus") or "")
        mine = S._pci_key(str(card.get("pci") or ""))
        got = next((p.split(",") for p in gpus.split(";")
                    if len(p.split(",")) == 3 and S._pci_key(p.split(",")[0]) == mine), None)
        if got is None or int(info.get("room_streamed") or 0):
            return None
        free, slack = float(got[1]) / 2**30, float(got[2]) / 2**30
        need, ctx, quant, layout = hy4_room_need(cfg, info)
        r = card.get("cuda_probe") or (probe or {}).get(mine)
        if need is None or not r:
            return None
        seen = float(r[0]) / 2**30
        own = need + slack - seen
        if not -1.0 <= own <= 2.0:
            return None
    except (TypeError, ValueError, IndexError, KeyError, AttributeError):
        return None
    rec = {"own_gib": round(own, 3), "probe_free_gib": round(seen, 3), "engine_free_gib": round(free, 3),
           "slack_gib": round(slack, 3), "need_gib": round(need, 3), "context": ctx, "rows": info.get("room_rows"),
           "drafter": bool(int(info.get("room_drafter") or 0)), "quant": quant, "layout": layout,
           "measured": time.strftime("%Y-%m-%d %H:%M:%S")}
    if save:
        key = hy4_room_key(card)                       # (this card's figure under another driver: gone)
        return hy4_room_store(key, rec, lambda k: k.split("|")[:3] == key.split("|")[:3])
    return rec


def hy4_room_measure_set(cfg: dict, cards: list, info: dict, probe=None, save: bool = True) -> dict | None:
    """hy4_room_measure for a start on several cards (a layer split or tensor parallel): the setup's sums for it are
    one figure for the cards together (need: hy4_usable_gib adds the cards' rooms), and the engine says how much less
    free on every card at once what the setup claims of the start holds with (room_joint: that it fits, the cache's
    rows, the drafters, the pictures, KV streaming - not tensor parallel against a layer split, which the sums do
    not count; each card's own figure counts the others' spare too where a layer split moves layers to a card with
    room).  So the cards' shares together are need + n * joint
    less what the probe saw of them, each card's in proportion to what the start takes on it (its probe less joint):
    a tensor-parallel group's cache on every member, a split's layers by the cards' room.  Kept as one record for
    these cards (hy4_room_set_key: "cards", each one's share by PCI place) - a start on one of them alone has its own.
    Not kept: as hy4_room_measure's, or a share beyond -2 / +2 GiB on a card."""
    try:
        if int(info.get("room_streamed") or 0) or "room_joint" not in info:
            return None
        joint = float(info["room_joint"]) / 2**30
        need, ctx, quant, layout = hy4_room_need(cfg, info)
        seen = []
        for g in cards:
            r = g.get("cuda_probe") or (probe or {}).get(S._pci_key(str(g.get("pci") or "")))
            if not r:
                return None
            seen.append(float(r[0]) / 2**30)
        takes = [x - joint for x in seen]
        if need is None or min(takes) <= 0:
            return None
        own = [t * (need / sum(takes) - 1.0) for t in takes]
        if not all(-2.0 <= x <= 2.0 for x in own):
            return None
    except (TypeError, ValueError, IndexError, KeyError, AttributeError, ZeroDivisionError):
        return None
    rec = {"cards": {S._pci_key(str(g["pci"])): round(x, 3) for g, x in zip(cards, own)},
           "probe_free_gib": [round(x, 3) for x in seen], "joint_gib": round(joint, 3), "need_gib": round(need, 3),
           "context": ctx, "rows": info.get("room_rows"), "tp": int(info.get("room_tp") or 0),
           "drafter": bool(int(info.get("room_drafter") or 0)), "mtp": bool(int(info.get("room_mtp") or 0)),
           "quant": quant, "layout": layout, "measured": time.strftime("%Y-%m-%d %H:%M:%S")}
    if save:
        key = hy4_room_set_key(cards)                  # (these cards' figures under other drivers: gone)

        def bare(k: str) -> list:                      # (a set key's cards without their drivers)
            return sorted("|".join(c.split("|")[:3]) for c in "|".join(k.split("|")[2:]).split("+"))
        return hy4_room_store(key, rec, lambda k: k.split("|")[:2] == key.split("|")[:2] and bare(k) == bare(key))
    return rec


def maya_engine_mib() -> dict:
    """{GPU index: the MiB of VRAM Maya's engines running now hold on it} by nvidia-smi's process list - None for a
    card where it names an engine without its memory (Windows gives none per program); {} where no engine runs, or
    it does not answer."""
    def query(what):
        try:
            return subprocess.run(["nvidia-smi", f"--query-{what}", "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True, timeout=30).stdout
        except (OSError, subprocess.SubprocessError):
            return ""
    index = {}
    for line in query("gpu=index,uuid").splitlines():
        i, _, uuid = (s.strip() for s in line.partition(","))
        if i.isdigit():
            index[uuid] = int(i)
    out = {}
    for line in query("compute-apps=gpu_uuid,pid,used_memory,process_name").splitlines():
        parts = [s.strip() for s in line.split(",", 3)]
        if len(parts) < 4 or parts[0] not in index:
            continue
        exe = re.split(r"[\\/]", parts[3])[-1].lower()
        exe = exe[:-4] if exe.endswith(".exe") else exe
        if exe not in ENGINE_PROCESSES:
            continue
        i = index[parts[0]]
        try:
            mib = float(parts[2])
        except ValueError:
            mib = None
        out[i] = None if mib is None or out.get(i, 0) is None else out.get(i, 0) + mib
    return out


def hy4_used_mib(before: dict | None = None, used: dict | None = None, own: dict | None = None) -> dict:
    """The setup's reading of the VRAM in use on each card (gpu_used_mib) for a PC whose model may be loaded now (the
    dashboard's Models page - serve/models.py's hy4_used): what a Maya engine holds there itself is not "in use" - it
    is the model the page sets up, or one that stops when that one starts.  A card with an engine on it: the reading
    less the engine's own (maya_engine_mib); where nvidia-smi does not say how much that is (Windows), the reading
    taken before the engine started (`before`: {index: MiB}, the server's) - and with neither, the reading as it is
    (hy4_usable_gib counts at most HY4_FIT_USED_CAP_GIB of it).  {} when nvidia-smi does not answer."""
    used = gpu_used_mib() if used is None else used
    if not used:
        return {}
    own = maya_engine_mib() if own is None else own
    out = {}
    for i, m in used.items():
        if i not in own:
            out[i] = m
        elif own[i] is not None:
            out[i] = max(m - own[i], 0)
        elif before and i in before:
            out[i] = min(before[i], m)
        else:
            out[i] = m
    return out


def hy4_fit(q: dict, gpus: list, ctx_wanted: int | None = None, used: dict | None = None,
            layout: str | None = None, pictures: bool = True, name: str | None = None,
            ram_gb: float | None = None, drafter: float = 0.0, kv: str | None = None) -> dict:
    """How a size of Hy4-Flash (q: its entry in the manifest) runs on these cards, worked out before any download:
    {"layout": "standard" where its plan fits the VRAM (hy4_usable_gib) at the context - else a small-card layout the
    size has; "context": ctx_wanted, else "recommended" - the largest of HY4_CONTEXTS that fits, up to
    HY4_TESTED_CONTEXT; "contexts": all that fit; "kv": the latent cache's rows the engine's --kv auto takes at the
    context - "f16" or "q8" (8-bit rows; never said on screen: the engine chooses at every start); "vision":
    pictures are read ("vision_small": smaller ones than the tower's largest); "chunk" and "args": the layout's own engine flags
    (none for "standard": the engine's defaults - it fills the VRAM left with the most-routed experts itself, and its
    own plan takes the prompt chunk at every start: the largest of 4096 / 2048 / 1024 tokens that these cards hold
    beside the model, its cache and the pictures' room - pictures first there too, src/hy4/hy4_plan.hpp's chunk_pick -
    so no --chunk is written and none can be one the engine would refuse; never --kv: the engine's auto); "usable",
    "need" (GiB, with the rows of "kv")}.  Pictures take no VRAM of their own (the engine lends a picture's room
    from its prompt buffers), so they are on wherever the model runs, at full size, and cost no context and no
    row's precision ("vision_small" is always False now, kept for the callers); a small-card layout reads prompts in
    chunks twice its plan's (HY4_CHUNK_UP_GIB) where that fits beside the model's own parts - with or without
    pictures.
    Nothing fits: {"error": why, "hint": what to do, "usable", "need"} ("too_big": not even at the smallest context -
    the size itself; name: the size's on-screen name for the line, else the model's).  layout: only this one (the
    files of a folder are that layout's).
    ram_gb (the PC's RAM; None: as before, VRAM alone): contexts that fit the VRAM only with KV streaming are offered
    too where the RAM holds their rows beside the system's headroom and the size's experts outside the VRAM -
    "streamed": those contexts; at one of them "kv_streaming" True, "kv_ram" (GiB of RAM for the rows) and
    "--kv-streaming auto" in "args" (the engine then streams by itself, only there).
    drafter (GiB: the DFly2 drafter the setup would load beside the model, hy4_dfly2_gib; 0: none): a small-card
    layout's chunk is not doubled where that would leave the drafter no room while the plan's own chunk does - its
    drafts are the larger gain there (an RTX 3050 8 GB at 4K: decoding +12-39%, prompts read 37% slower).
    kv ("f16" / "q8": the engine's --kv with those rows only - "--kv" in "args"; None: its auto): the faster drafting
    the setup offers on a small card (hy4_fast_drafting_fit) keeps the cache in 8-bit rows at a context where F16 fit."""
    lays = hy4_layouts(q)
    if layout is not None:
        lays = {k: v for k, v in lays.items() if k == layout}
    usable = hy4_usable_gib(gpus, used)
    kv_own = kv if kv in ("f16", "q8") else None
    measured = bool(gpus) and all(g.get("engine_room_how") == "measured" for g in gpus)   # (hy4_engine_room's)

    def need(plan: dict, c: int) -> float:             # (the least a context takes: with F16 rows, or with 8-bit rows)
        if kv_own:
            return hy4_fit_need_gib(plan, c, kv_own == "q8")
        return min(hy4_fit_need_gib(plan, c), hy4_fit_need_gib(plan, c, True))

    ram_rows = hy4_ram_for_rows(q, usable, ram_gb)     # (the RAM left for streamed rows)

    def stream(plan: dict, c: int) -> tuple | None:
        return hy4_stream_rows(plan, c, usable, ram_rows, kv_own)

    def need_at(plan: dict, c: int, q8: bool, st: tuple | None) -> float:
        return hy4_stream_need_gib(plan, c, q8, st[1]) if st else hy4_fit_need_gib(plan, c, q8)

    def in_vram(plan: dict, c: int) -> str | None:     # (the rows a context fits in VRAM with - no streaming)
        return hy4_kv_rows(hy4_fit_need_gib(plan, c), hy4_fit_need_gib(plan, c, True), usable, kv_own)

    def rows(plan: dict, c: int) -> str | None:        # (a context the table does not list: not measured on a card)
        return in_vram(plan, c) or (stream(plan, c) or (None,))[0]
    least = HY4_CONTEXTS[0]
    cards = f"{len(gpus)} cards together have" if len(gpus or []) > 1 else "this card has"
    fits = {k: [c for c in HY4_CONTEXTS if rows(v["plan"], c)] for k, v in lays.items()}
    if not any(fits.values()):
        smallest = min((need(v["plan"], least) for v in lays.values()), default=0.0)
        return {"error": f"{name or HY4} needs about {smallest:.1f} GB of free VRAM for the parts of the model that "
                         f"must sit on the GPU (at its smallest context, {context_label(least)}; the engine has no RAM "
                         f"fallback for them), and {cards} {usable:.1f} GB free",
                "hint": "a card with more VRAM (or more cards: --gpus), or close the programs that use this one",
                "usable": usable, "need": smallest, "too_big": True}
    want = int(ctx_wanted) if ctx_wanted else least
    pick = next((k for k in lays if rows(lays[k]["plan"], want)), None)
    if pick is None:                                   # (the context asked for fits no layout)
        most = max(max(v) for v in fits.values() if v)
        return {"error": f"a context of {want} tokens needs about "
                         f"{min(need(v['plan'], want) for v in lays.values()):.1f} GB of free VRAM for {HY4}, and "
                         f"{cards} {usable:.1f} GB free",
                "hint": f"the largest context that fits here: --context {most}",
                "usable": usable, "need": min(need(v["plan"], want) for v in lays.values())}
    plan = lays[pick]["plan"]
    rec = max([c for c in fits[pick] if c <= HY4_TESTED_CONTEXT] or [least])
    ctx = int(ctx_wanted) if ctx_wanted else rec
    kv = rows(plan, ctx)
    st = None if in_vram(plan, ctx) else stream(plan, ctx)
    takes = need_at(plan, ctx, kv == "q8", st)
    chunk, args = int(plan.get("chunk") or 0), []
    # (pictures: their room is lent by the engine's prompt buffers - no VRAM kept, so the rows and the context
    #  worked out above stand and the largest picture is read on every layout)
    vision, small = bool(pictures), False
    if pick != HY4_STANDARD:
        # (the chunk: doubled where that fits beside the model's own parts with the rows chosen above - the larger
        #  chunk's buffers hold the largest picture by themselves, so pictures and the chunk no longer compete)
        def beside(ch: int) -> float:                  # (with the drafter: hy4_drafter_need_gib)
            return hy4_drafter_need_gib(plan, {"context": ctx, "kv": kv, "need": takes, "chunk": ch,
                                               "room_measured": measured}, drafter)
        if chunk and takes + HY4_CHUNK_UP_GIB <= usable and \
                not (drafter and not st and beside(2 * chunk) > usable >= beside(chunk)):
            chunk *= 2
        args = (["--chunk", str(chunk)] if chunk else []) +             [x for k, flag in (("reserve_gb", "--reserve-gb"), ("vram_experts", "--vram-experts")) if k in plan
             for x in (flag, f"{float(plan[k]):g}")]
    streamed = [c for c in fits[pick] if not in_vram(plan, c)]
    if kv_own:                                         # (the rows asked for: the engine's --kv, nothing auto)
        args = args + ["--kv", kv_own]
    out = {"layout": pick, "context": ctx, "recommended": rec, "contexts": fits[pick], "chunk": chunk, "args": args,
           "kv": kv, "usable": usable, "need": takes, "vision": vision, "vision_small": small}
    if streamed:
        out["streamed"] = streamed
    if measured:                                       # (the drafter's sums then without margins: hy4_drafter_need_gib)
        out["room_measured"] = True
    if st:
        out.update(kv_streaming=True, kv_ram=hy4_stream_ram_gib(ctx, kv == "q8"),
                   args=args + ["--kv-streaming", "auto"])
    return out


def hy4_layout_in(d: Path, quant: str) -> str | None:
    """The layout whose trunk is in a folder - under its name, or one it had before (the table's "was": both layouts'
    trunks had the same name then) - by its size, or None."""
    was = read_json(HY4_MANIFEST).get("was") or {}
    for name, lay in hy4_layouts(hy4_quant_table(quant)).items():
        for rel, (size, _) in lay["files"].items():
            for n in [hy4_local_name(rel)] + list(was.get(hy4_local_name(rel)) or []):
                f = Path(d) / n
                if f.is_file() and f.stat().st_size == size:
                    return name
    return None


def hy4_for_pcs(quant: str, memory_gb: float) -> str:
    """A size's memory line in the setup's list: the PCs it is meant for as the model card says it (the table's
    "for_gb": RAM + VRAM), else the VRAM + RAM that hold it."""
    lo, hi = (list(hy4_quant_table(quant).get("for_gb") or []) + [0, 0])[:2]
    return f"for PCs with {lo:g}-{hi:g} GB of RAM + VRAM" if lo and hi else f"about {memory_gb:.0f} GB of VRAM and RAM together"


def choose_hy4_quant(a, prev: dict) -> str:
    """--hy4-quant (--model hy4-flash-maya-xs, hy4-flash-q4_k_m), else asked: Maya's sizes that are uploaded first - by
    this PC's memory the one recommended (fit_recommendation) - then Tencent's own Q4_K_M (AngelSlim).  Nothing to
    choose from: the one set up before, else the smallest size (with --model-dir: its files)."""
    if getattr(a, "hy4_quant", None):
        return a.hy4_quant
    keys = [k for k in HY4_QUANTS if hy4_published(k)]
    others = [HY4_Q4] if hy4_published(HY4_Q4) else []
    before = (prev.get("installer") or {}).get("hy4_quant")
    if not keys + others:
        return before if before in HY4_QUANTS else next(iter(HY4_QUANTS))
    if len(keys + others) == 1:
        return (keys + others)[0]
    total, _ = mem_gb()
    m = {k: hy4_entry(k) for k in keys + others}
    rec = fit_recommendation({k: m[k]["memory_gb"] for k in keys}, setup_gpus(a), total) if keys else None
    order = sorted(keys, key=lambda k: k != rec) + others
    default = before if before in order else rec if rec in order else order[0]
    rows = [(Q.shown("hy4", k, m[k]["download_gb"]), "recommended" if k == rec else None,
             f"Maya's quant: {hy4_quant_table(k).get('about', 'GGUF')}; " + hy4_for_pcs(k, m[k]["memory_gb"]))
            for k in order if k != HY4_Q4]
    rows += [(Q.shown("hy4", k, m[k]["download_gb"]), None,
              f"Tencent's own Q4_K_M (AngelSlim): {hy4_quant_table(k).get('about', 'GGUF')}; about "
              f"{m[k]['memory_gb']:.0f} GB of VRAM and RAM together; with MTP drafting - run on NVIDIA RTX 30 / A100 "
              "and newer, Linux") for k in others]
    pick = choose(f"Which files of {HY4}?", ["Maya's GGUF quants of it, on NVIDIA from V100 on or AMD (Linux; Windows experimental), and "
                                           "Tencent's own:"],
                  rows, order.index(default), a.yes or getattr(a, "check", False))
    return order[pick]


def hy4_dir(a, models: Path, prev: dict, quant: str, layout: str = HY4_STANDARD) -> Path:
    """The model's folder: --model-dir, else the folder of the Hy4-Flash setup before when it holds the same size and
    layout (its pack is inside it), else <models folder>/Hy4-Flash-Maya-XS (a layout: ...-8GB)."""
    if a.model_dir:
        return Path(a.model_dir).expanduser().resolve()
    inst = prev.get("installer") or {}
    if inst.get("hy4_quant") == quant and inst.get("layout", HY4_STANDARD) == layout and inst.get("model_dir") and \
            Path(inst["model_dir"]).is_dir():
        return Path(inst["model_dir"])
    return models / hy4_entry(quant, layout)["dir"]


def hy4_places(models: Path, name: str) -> list:
    """Where another Hy4-Flash setup here keeps the file `name` (the n-gram table: one file for every set): each
    size's and layout's default folder in the models folder, the Q4_K_M's, and the folders of the Hy4 configs."""
    dirs = []
    for k, full in HY4_QUANTS.items():
        dirs += [models / full] + [models / f"{full}-{lay.upper()}" for lay in hy4_quant_table(k).get("layouts") or {}]
    if hy4_quant_table(HY4_Q4).get("dir"):
        dirs.append(models / hy4_quant_table(HY4_Q4)["dir"])
    for c in configs():
        cfg = read_json(c)
        if config_family(cfg) == "hy4" and (cfg.get("installer") or {}).get("model_dir"):
            dirs.append(Path(cfg["installer"]["model_dir"]))
    return [x / name for x in dirs]


def hy4_model_step(a, d: Path, quant: str, layout: str = HY4_STANDARD, models: Path | None = None) -> list | None:
    """A size's files in d - [its trunk, the n-gram table] - or None when nothing was downloaded.  The n-gram table is
    one file for every size and layout: where another Hy4-Flash folder here has it (the same size and sha256) it is
    hard-linked, not downloaded again.  Missing files are downloaded once the size is published; before that, the
    files already in d (--model-dir) are taken.  Either way each is checked against its sha256 (once, then
    remembered).  The vision tower's file is hy4_vision_step's; the repo's drafter file is not fetched (the engine
    does not read it yet)."""
    name = Q.shown("hy4", quant)
    step(5, f"the model ({name})")
    m = hy4_entry(quant, layout)
    files = [d / n for n in m["files"]]
    ngram = d / m["ngram"]
    was = m.get("earlier") or {}
    for alt, (size, sha) in (m.get("theirs") or {}).items():
        # (Tencent's own set with its own n-gram shard already in the folder - --model-dir, a download before: taken
        #  as it is, checked against its own sha256; nothing is downloaded)
        if not ngram.exists() and (d / alt).is_file() and (d / alt).stat().st_size == size:
            m["files"] = {k: v for k, v in m["files"].items() if k != ngram.name}
            m["sha256"] = {k: v for k, v in m["sha256"].items() if k != ngram.name}
            m["files"][alt], m["sha256"][alt] = size, sha
            ngram = d / alt
            files = [d / n for n in m["files"]]
            break
    for f in files:   # (a folder filled before the files' names carried their quant type: the same file, linked)
        if not f.exists() and was.get(f.name):
            link_published(f, m["files"][f.name], m["sha256"][f.name], [d / o for o in was[f.name]])
    if not ngram.exists() and models is not None:
        link_published(ngram, m["files"][ngram.name], m["sha256"][ngram.name],
                       [p for n in [ngram.name] + was.get(ngram.name, []) for p in hy4_places(models, n)])
    missing = [f for f in files if file_incomplete(m, f)]
    if missing and not hy4_published(quant, layout):
        if len(missing) < len(files):                  # (a folder with a part of the set: which file, and its size)
            what = ", ".join(("the n-gram table " if f == ngram else "") + f"{f.name} ({m['files'][f.name] / 1e9:.1f} GB)"
                             + ("" if file_incomplete(m, f) == "missing" else f": {file_incomplete(m, f)}")
                             for f in missing)
            fail(f"{name} is not published yet, and {d} does not have all of its files - missing: {what}",
                 f"both files must be in the folder ({', '.join(f.name for f in files)}); the n-gram table is the "
                 f"same file for every size: a copy another {HY4} folder under the models folder has is linked by "
                 "itself (--models-dir <that models folder>)")
        fail(f"{name} is not published yet (and {missing[0]} is {file_incomplete(m, missing[0])})",
             f"with its files in a folder: {ME} --setup --model {Q.setup_id('hy4', quant)} --model-dir <folder>")
    if missing and not offer_download(a, Q.shown("hy4", quant, m["download_gb"]), d, files, m):
        return None
    for f in files:   # (downloaded, copied or linked: by sha256 too - a damaged file keeps its size)
        if not sha256_checked(f, m["sha256"][f.name]):
            fail(f"{f}: the sha256 does not match the published one - the file is damaged (its size is right)",
                 f"delete it and run {ME} --setup again" + (": the setup downloads it again"
                                                           if hy4_published(quant, layout) else ""))
    ok(f"{name}: {len(files)} files, {sum(f.stat().st_size for f in files) / 1e9:.1f} GB in {d}")
    say(f"  the n-gram table ({m['ngram_gb']:.0f} GB) is read from the SSD page by page while the model answers - it "
        "is never loaded into RAM")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: the engine reads the n-gram table ({m['ngram_gb']:.0f} GB) from it on "
             "every token, and the experts that are not in VRAM - use an NVMe SSD (--models-dir)")
    return files


def hy4_vision_entry(quant: str = HY4_Q4) -> dict | None:
    """A set's vision tower - its mmproj file (download.json's "vision") - as a MODELS entry (offer_download's form):
    Maya's own Q8_0 file, one for every Maya size, from Maya's repo at the table's commit; AngelSlim's BF16 for
    Tencent's own Q4_K_M, from their repo at the commit it is pinned at.  "name": the file's.  None: the table has
    none."""
    man = read_json(HY4_MANIFEST)
    q = hy4_quant_table(quant) if quant == HY4_Q4 else man
    src = q.get("vision") or {}
    if not src or not q.get("repo"):
        return None
    path, (size, sha) = next(iter(src.items()))
    name, folder = Path(path).name, Path(path).parent.as_posix()
    return {"about": f"{HY4}'s vision tower (mmproj, {'BF16, AngelSlim' if quant == HY4_Q4 else 'Q8_0'}); "
                     f"{q.get('license', 'Tencent Hy License')}",
            "format": "hy4-gguf", "repo": q["repo"], "revision": q.get("revision") or "main",
            "folder": "" if folder == "." else folder, "files": {name: size}, "sha256": {name: sha},
            "download_gb": size / 1e9, "name": name}


def hy4_vision_file(d: Path, quant: str = HY4_Q4) -> Path | None:
    """The vision tower's mmproj file in a set's folder: the table's for the set, else the folder's one mmproj*.gguf
    (a size's own from the repo, --model-dir) - or None."""
    v = hy4_vision_entry(quant)
    if v and (d / v["name"]).is_file():
        return d / v["name"]
    try:
        found = sorted(p for p in d.glob("*.gguf") if "mmproj" in p.name.lower())
    except OSError:
        found = []
    return found[0] if len(found) == 1 else None


def hy4_pack_has_vision(pack: Path) -> bool:
    """The pack indexes the vision tower (hy4.txt's vision.* keys)."""
    try:
        return any(ln.startswith("vision.dim ") for ln in (pack / "hy4.txt").read_text(encoding="utf-8").splitlines())
    except OSError:
        return False


def hy4_vision_step(a, d: Path, quant: str, models: Path | None = None) -> Path | None:
    """The vision tower's mmproj file in d, or None (text only, said in step 7).  A file the folder has is taken (the
    table's: checked against its sha256, once); else the table's is hard-linked from another Hy4 folder here that has
    it, else downloaded (asked like the model; not with --no-vision, and not for a size that is not published yet)."""
    have = hy4_vision_file(d, quant)
    v = hy4_vision_entry(quant)
    if have is None and not a.no_vision and v is not None:
        f = d / v["name"]
        if models is not None:
            link_published(f, v["files"][f.name], v["sha256"][f.name], hy4_places(models, f.name))
        if file_incomplete(v, f) and hy4_published(quant) and \
                offer_download(a, f"{Q.shown('hy4', quant)}'s vision tower ({v['download_gb']:.1f} GB)", d, [f], v):
            pass
        have = f if not file_incomplete(v, f) else None
    if have is not None and v is not None and have.name == v["name"] and not sha256_checked(have, v["sha256"][have.name]):
        fail(f"{have}: the sha256 does not match the published one - the file is damaged (its size is right)",
             f"delete it and run {ME} --setup again")
    return have


def hy4_mtp_entry() -> dict | None:
    """The draft block of Maya's sizes - Tencent's MTP block in a file of its own (download.json's "mtp": one file
    for every size, in Maya's repo) - as a MODELS entry (offer_download's form).  "name": the file's here (a
    ".gguf_file" of the Hub saved as ".gguf"); "published": the table names the commit the file is at
    ("mtp_revision") - before that only a copy already on this PC is taken.  None: the table has none."""
    man = read_json(HY4_MANIFEST)
    src = man.get("mtp") or {}
    if not src or not man.get("repo"):
        return None
    path, (size, sha) = next(iter(src.items()))
    name, folder = hy4_local_name(path), Path(path).parent.as_posix()
    e = {"about": f"{HY4}'s MTP draft block (Tencent's, BF16, as in AngelSlim's GGUF); "
                  f"{man.get('license', 'Tencent Hy License')}",
         "format": "hy4-gguf", "repo": man["repo"], "revision": man.get("mtp_revision") or "main",
         "folder": "" if folder == "." else folder, "files": {name: size}, "sha256": {name: sha},
         "download_gb": size / 1e9, "name": name, "published": bool(man.get("mtp_revision"))}
    if name != Path(path).name:
        e["hub_names"] = {name: Path(path).name}
    return e


def hy4_mtp_gib(chunk: int = 1024) -> float:
    """The VRAM (GiB) the draft block of Maya's sizes takes on the main GPU, as the engine counts it: its matrices
    rounded to Q8_0 and a prompt chunk of its rows (five F32 rows a token: src/hy4/hy4_plan.hpp's mtp_rows_bytes).
    0 where the table has no block."""
    e = hy4_mtp_entry()
    if e is None:
        return 0.0
    return (e["files"][e["name"]] * HY4_MTP_VRAM + int(chunk or 1024) * 3072 * 20) / 2**30


def hy4_pack_mtp_file(pack: Path) -> str | None:
    """The name of the MTP block's own file a pack reads (tensors.txt's `file` lines), or None (a set with the block
    inside, or without one)."""
    e = hy4_mtp_entry()
    if e is None:
        return None
    try:
        with (pack / "tensors.txt").open(encoding="utf-8") as f:
            for ln in f:
                if not ln.startswith("file "):
                    break
                if Path(ln.split(None, 2)[2].strip()).name == e["name"]:
                    return e["name"]
    except (OSError, IndexError):
        pass
    return None


def hy4_mtp_step(a, d: Path, quant: str, models: Path | None = None) -> Path | None:
    """The draft block's file in d for a Maya size, or None (no drafts: said in step 7).  A file the folder has is
    taken; else it is hard-linked from another Hy4-Flash folder here that has it (one file for every size), else
    downloaded once it is published (asked like the model).  Checked against its sha256 (once, then remembered)."""
    e = hy4_mtp_entry()
    if e is None or quant == HY4_Q4:
        return None
    f = d / e["name"]
    if file_incomplete(e, f) and models is not None:
        link_published(f, e["files"][f.name], e["sha256"][f.name], hy4_places(models, f.name))
    if file_incomplete(e, f) and e["published"]:
        offer_download(a, f"{HY4}'s draft block ({e['download_gb']:.1f} GB, one file for every size)", d, [f], e)
    if file_incomplete(e, f):
        return None
    if not sha256_checked(f, e["sha256"][f.name]):
        fail(f"{f}: the sha256 does not match the published one - the file is damaged (its size is right)",
             f"delete it and run {ME} --setup again")
    return f


def hy4_dfly2_entry(quant: str) -> dict | None:
    """A Maya size's DFly2 drafter - its size entry's "drafter" in download.json: Tencent's draft layers in a GGUF
    named .dfly2 on the Hub, kept under that name (the pack finds it in the model's folder) - as a MODELS entry
    (offer_download's form), from Maya's repo at the table's commit.  None: the size has none (Tencent's own Q4_K_M
    drafts with its MTP block)."""
    if quant == HY4_Q4:
        return None
    man = read_json(HY4_MANIFEST)
    src = hy4_quant_table(quant).get("drafter") or {}
    if not src or not man.get("repo"):
        return None
    path, (size, sha) = next(iter(src.items()))
    name, folder = Path(path).name, Path(path).parent.as_posix()
    return {"about": f"{HY4}'s DFly2 drafter for {Q.shown('hy4', quant)}; {man.get('license', 'Tencent Hy License')}",
            "format": "hy4-gguf", "repo": man["repo"], "revision": man.get("revision") or "main",
            "folder": "" if folder == "." else folder, "files": {name: size}, "sha256": {name: sha},
            "download_gb": size / 1e9, "name": name, "published": bool(man.get("revision"))}


def hy4_dfly2_gib(quant: str) -> float:
    """The VRAM (GiB) a size's DFly2 drafter takes on the main GPU, as the engine counts it (src/hy4/hy4_plan.hpp's
    dfly_bytes): its matrices as they are in the file, its small scratch and lists (HY4_DFLY2_EXTRA).  0 where the size
    has none."""
    e = hy4_dfly2_entry(quant)
    return 0.0 if e is None else (e["files"][e["name"]] + HY4_DFLY2_EXTRA) / 2**30


def hy4_drafter_need_gib(plan: dict, fit: dict, drafter: float) -> float:
    """The VRAM (GiB) a start hy4_fit worked out on a layout's plan needs with a drafter of `drafter` GiB beside it (the
    DFly2 drafter: hy4_dfly2_gib): the fit's own figure ("need", with the cache's rows it chose), a prompt chunk it
    doubled (HY4_CHUNK_UP_GIB), the drafter's room, and HY4_FIT_MARGIN_GIB where the fit's figure is one measured on a
    card (a start with the drafter beside it never was; a context the plan does not list carries the margin already).
    The engine loads the drafter only where its room changes nothing but the experts' places - the cache's rows, the
    prompt chunk, KV streaming (src/hy4 load()): with this figure free it does, with less it decodes without drafts.
    "room_measured" (hy4_fit's: every card's room counts CUDA's own share as the engine measured it on this PC -
    hy4_engine_room): the sums without HY4_FIT_MARGIN_GIB, neither this one nor a context's the plan does not list -
    the margin stands for sums never set against an engine, and that start set them against this PC's; a claim the
    engine then finds no room for costs a start without drafts, never a start (the contexts and the rows that fit keep
    their margins).  Measured on an RTX 3050 8 GB: so the setup claims the drafter at 4K and 5K, not at 6K and 8K,
    and finds it room at 8K beside 8-bit rows - each as its engine then does."""
    listed = plan.get("need_gib_q8" if fit.get("kv") == "q8" else "need_gib") or {}
    up = HY4_CHUNK_UP_GIB if int(fit.get("chunk") or 0) > int(plan.get("chunk") or 0) > 0 else 0.0
    if fit.get("room_measured") and not fit.get("kv_streaming"):
        own = 0.0 if str(fit.get("context")) in listed else HY4_FIT_MARGIN_GIB   # (hy4_fit_need_gib's, taken out)
        return float(fit.get("need") or 0) - own + up + drafter
    margin = HY4_FIT_MARGIN_GIB if str(fit.get("context")) in listed else 0.0
    return float(fit.get("need") or 0) + up + drafter + margin


def hy4_dfly2_need_gib(quant: str, fit: dict) -> float:
    """hy4_drafter_need_gib for a size's DFly2 drafter beside the start hy4_fit worked out for it (fit["layout"])."""
    plan = (hy4_layouts(hy4_quant_table(quant)).get(fit.get("layout") or HY4_STANDARD) or {}).get("plan") or {}
    return hy4_drafter_need_gib(plan, fit, hy4_dfly2_gib(quant))


def hy4_mtp_need_gib(quant: str, fit: dict) -> float:
    """hy4_drafter_need_gib for the MTP block (hy4_mtp_gib at the fit's prompt chunk) beside the start hy4_fit worked
    out for a size - the engine counts the block's room as the DFly2 drafter's (with drafting on and off; where it
    would change the cache's rows or KV streaming, the block is not loaded).  With both drafters in
    a pack the engine counts the larger room once (one drafter at a time): the block's."""
    plan = (hy4_layouts(hy4_quant_table(quant)).get(fit.get("layout") or HY4_STANDARD) or {}).get("plan") or {}
    return hy4_drafter_need_gib(plan, fit, hy4_mtp_gib(int(fit.get("chunk") or plan.get("chunk") or 1024)))


def hy4_fast_drafting_fit(quant: str, gpus: list, fit: dict, used: dict | None = None,
                          ram_gb: float | None = None) -> dict | None:
    """The setup's opt-in faster drafting (--fast-drafting on) for the start hy4_fit worked out (fit): the same
    context and layout with the latent cache in 8-bit rows (the engine's --kv q8, "--kv q8" in "args";
    "fast_drafting": True) where those rows are what gives the size's DFly2 drafter its room - on one GPU, where the
    start as worked out drafts with nothing: its rows F16 in VRAM (no KV streaming), no room for the drafter beside
    them nor for the MTP block, and with 8-bit rows the drafter's room as the engine counts it
    (hy4_dfly2_need_gib).  "f16_room": True where these sums find the drafter room beside the F16 rows too (the
    default then claims it; the engine, which counts its own free VRAM at every start, may still find none there -
    its start line says so - so the faster start is taken there only when asked by
    name).  Decided by this PC's own figures (its cards' free VRAM, the size's plan), no card named.  The 8-bit rows
    round the cache, so the answers are worded slightly differently from the default's (F16) - only when asked;
    within that start drafting on and off give the same tokens (the drafter's room is counted either way).  None: no
    such trade here."""
    if quant == HY4_Q4 or len(gpus or []) != 1 or "error" in fit or fit.get("kv") != "f16" or \
            fit.get("kv_streaming") or hy4_dfly2_entry(quant) is None:
        return None
    usable = float(fit.get("usable") or 0)
    if fit.get("layout") == HY4_STANDARD and hy4_mtp_entry() is not None and \
            float(fit.get("need") or 0) + hy4_mtp_gib(int(fit.get("chunk") or 1024)) <= usable:
        return None                                    # (the MTP block fits: the default drafts with it)
    f8 = hy4_fit(hy4_quant_table(quant), gpus, fit["context"], used=used, layout=fit.get("layout"),
                 pictures=bool(fit.get("vision", True)), ram_gb=ram_gb, drafter=hy4_dfly2_gib(quant), kv="q8")
    if "error" in f8 or f8.get("kv") != "q8" or f8.get("kv_streaming") or hy4_dfly2_need_gib(quant, f8) > f8["usable"]:
        return None
    room = hy4_dfly2_need_gib(quant, fit) <= usable
    if room and fit.get("layout") == HY4_STANDARD:
        return None                                    # (the standard layout: its room comes from the experts' VRAM)
    return {**f8, "fast_drafting": True, "f16_room": room}


def hy4_pack_dfly2_file(pack: Path) -> str | None:
    """The name of the DFly2 drafter's file a pack reads (hy4.txt has dfly_layers; tensors.txt's `file` line of the
    file that is no trunk shard, MTP block or mmproj), or None."""
    try:
        if not any(ln.split()[:1] == ["dfly_layers"] for ln in (pack / "hy4.txt").read_text(encoding="utf-8").splitlines()):
            return None
        with (pack / "tensors.txt").open(encoding="utf-8") as f:
            for ln in f:
                if not ln.startswith("file "):
                    break
                n = Path(ln.split(None, 2)[2].strip()).name
                if n.endswith(".dfly2") or "drafter" in n.lower():
                    return n
    except (OSError, IndexError):
        pass
    return None


def hy4_dfly2_step(a, d: Path, quant: str, models: Path | None = None) -> Path | None:
    """A Maya size's DFly2 drafter file in d, or None (no drafts: said in step 7).  A file the folder has is taken;
    else it is hard-linked from another folder of this size here (its other layout's), else downloaded (asked like
    the model).  Checked against its sha256 (once, then remembered)."""
    e = hy4_dfly2_entry(quant)
    if e is None:
        return None
    f = d / e["name"]
    if file_incomplete(e, f) and models is not None:
        link_published(f, e["files"][f.name], e["sha256"][f.name], hy4_places(models, f.name))
    if file_incomplete(e, f) and e["published"]:
        offer_download(a, f"{HY4}'s DFly2 drafter for this size ({e['download_gb']:.1f} GB)", d, [f], e)
    if file_incomplete(e, f):
        return None
    if not sha256_checked(f, e["sha256"][f.name]):
        fail(f"{f}: the sha256 does not match the published one - the file is damaged (its size is right)",
             f"delete it and run {ME} --setup again")
    return f


def hy4_pack_step(a, d: Path, files: list, vis: Path | None = None, mtp: Path | None = None,
                  dfly: Path | None = None) -> Path:
    """tools/hy4/pack_gguf.py's pack in <model>/maya-hy4: the geometry, where every tensor sits in the two GGUF files
    (read in place) - the vision tower's mmproj file's too (vis), and the drafter's of a Maya size (mtp: Tencent's
    MTP block in its own file; dfly: the size's DFly2 drafter) -, the n-gram table's small parts and the tokenizer with
    the GGUF's chat template - built again when its tools or the model's files changed, or the vision tower or a
    drafter came or went."""
    step(6, "the pack (the engine's index of the model files)")
    pack = d / HY4_PACK
    tools = S.source_hash(HY4_PACK_TOOLS)
    need = ("hy4.txt", "tensors.txt", "engram.bin", "tokenizer/vocab.json", "tokenizer/chat_template.jinja")
    every = list(files) + [f for f in (mtp, dfly, vis) if f is not None]
    if not a.repack and read_json(pack / PACK_SOURCES).get("tools") == tools and \
            all((pack / f).exists() for f in need) and pack_stale(pack, every) is None and \
            hy4_pack_has_vision(pack) == (vis is not None) and \
            hy4_pack_mtp_file(pack) == (mtp.name if mtp is not None else None) and \
            hy4_pack_dfly2_file(pack) == (dfly.name if dfly is not None else None):
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack is written inside the model's folder")
    say("  Indexing every tensor of the GGUF files (the weights stay where they are) and the tokenizer ...")
    run([sys.executable, str(ROOT / "tools" / "hy4" / "pack_gguf.py"), "--model", str(d), "--out", str(pack),
         "--mmproj", vis.name if vis is not None else "none", "--mtp", mtp.name if mtp is not None else "none",
         "--drafter", dfly.name if dfly is not None else "none", "--files"] + [f.name for f in files])
    missing = [f for f in need if not (pack / f).exists()]
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    stamp_pack(pack, every)
    rec = read_json(pack / PACK_SOURCES)
    rec["tools"] = tools
    (pack / PACK_SOURCES).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    ok(f"pack: {pack}")
    return pack


def write_config_hy4(a, pc: dict, meta: dict, pack: Path, ctx: int, models: Path, d: Path, drafting: bool = False,
                     quant: str = HY4_Q4, fit: dict | None = None, vision: bool = True, drafter: str = "mtp",
                     extra: dict | None = None) -> Path:
    """A set's config and its start script: maya-hy4-flash-maya-xs.json for a Maya size (maya_quants.config_stem: one
    config a size, whichever layout), maya-hy4-flash.json for Tencent's own Q4_K_M (--mtp: drafting with its MTP
    block; a Maya size drafts with the block, or with its DFly2 drafter - drafter "dfly2": --dfly2).  The model's own
    sampling and thinking level are written at every setup; a setup again keeps what the config before says that this one does not ask - where the server listens, its key, the env, and the engine's
    settings set by hand or on the Models page (tensor parallel, the CPU lane's threads, an expert profile, the
    drafts; in the standard layout the prompt chunk and the VRAM kept free too).  A layout's own flags (hy4_fit's
    "args") are worked out again every time.  maya-hy4 stops at a flag it does not know: only its own are written."""
    step(8, "the configuration and the start script")
    port = a.port or 8080
    fit = fit or {"layout": HY4_STANDARD, "args": []}
    cfg_path = ROOT / f"{Q.config_stem('hy4', quant)}.json"
    old = read_json(cfg_path) if cfg_path.exists() else {}
    # (--tp auto, the engine's default: tensor parallel over the largest group of equal GPUs, the fastest GPU holding
    #  the head and the drafts; the layers split where no group fits.  No --gpus: the server shows the engine only the
    #  config's "gpu" cards, and it takes all it sees)
    # (drafting on a Maya size, not by hand: --drafting auto - the engine measures once on this PC whether no drafting,
    #  the MTP block or the DFly2 drafter decodes fastest, each with its own VRAM plan, and keeps it; --drafting on:
    #  the faster drafter; mtp / dfly2 by hand, or the opt-in faster drafting's DFly2: that one)
    how = getattr(a, "drafting", None)
    if drafting and quant != HY4_Q4 and how not in ("mtp", "dfly2") and not fit.get("fast_drafting"):
        draft_args = ["--drafting", "on" if how == "on" else "auto"]
    else:
        draft_args = [("--dfly2" if drafter == "dfly2" else "--mtp")] if drafting else []
    args = ["--pack", str(pack), "--max-context", str(ctx)] + list(fit["args"]) + draft_args
    if quant != HY4_Q4 and HY4_PROFILE.exists():
        args += ["--profile", str(HY4_PROFILE)]
    # pictures: on whenever the pack has the vision tower (the engine's default - no flag; a setup again turns them
    # on: a config's --vision off is not kept); vision False (--no-vision, or no room for a picture beside the model)
    # writes --vision off - its room on the main GPU to the model
    if not vision and hy4_pack_has_vision(pack):
        args += ["--vision", "off"]
    standard = fit["layout"] == HY4_STANDARD and (old.get("installer") or {}).get("layout", HY4_STANDARD) == HY4_STANDARD
    kept = keep_args(old.get("args") or [], args, ("--tp", "--main-gpu", "--cpu-threads", "--profile", "--draft",
                                                   "--draft-floor") +
                     (("--chunk", "--reserve-gb", "--vram-experts") if standard else ()))
    if getattr(a, "vram_reserve_mib", None) is not None:   # (#1026) given now: written, and a later setup keeps it
        put = set_vram_reserve(args, "hy4", a.vram_reserve_mib)
        ok(f"VRAM kept free on each GPU: {a.vram_reserve_mib} MiB (--vram-reserve-mib: {put})")
    inst = {"family": "hy4", "hy4_quant": quant, "quant": HY4_QUANTS.get(quant, HY4), "layout": fit["layout"],
            "drafting": "on" if drafting else "off"}
    if quant != HY4_Q4 and hy4_pack_mtp_file(pack):    # (a Maya size with the draft block: "off" is then a choice)
        inst["draft_block"] = True
    if quant != HY4_Q4 and hy4_pack_dfly2_file(pack):  # (... with the DFly2 drafter: the same)
        inst["dfly2"] = True
    if drafting and quant != HY4_Q4:
        inst["drafter"] = drafter
    if fit.get("fast_drafting"):                       # (the opt-in: --kv q8 in its args; a setup again keeps it)
        inst["fast_drafting"] = True
    # (Windows: the VRAM the engine's CUDA gets on each card, as measured now - hy4_engine_room - with the card's
    #  VRAM and CUDA's own reading: the Models page counts it too while the engine holds the card, where it cannot be
    #  measured - with CUDA's own share as the engine measured it later, where it has since)
    room = {str(g["index"]): [round(float(g["engine_room_gb"]), 3), float(g["vram_gb"])] +
            ([round(float(g["cuda_free_gb"]), 3)] if g.get("cuda_free_gb") is not None else [])
            for g in pc["gpus"] if g.get("engine_room_gb") is not None and g.get("vram_gb")}
    if room:
        inst["engine_room_gb"] = room
    inst.update(extra or {})                         # (draft_choice: auto = both fit, the speed calibration picks)
    sampling, effort = Q.model_defaults({"installer": inst})   # (the model's own, every setup: its card's)
    cfg = {"exe": str(HY4_EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": HY4_MODEL_NAME, "gpu": [g["index"] for g in pc["gpus"]],
           "sampling": sampling, "reasoning_effort": effort,
           "lib_dirs": meta.get("lib_dirs") or old.get("lib_dirs") or [], "port": port,
           **({"backend": "hip"} if pc.get("backend") == "hip" else {})}
    kept += keep_server_settings(cfg, old, a)
    env = {**(old.get("env") or {}), **parse_env(a.env)}
    if env:
        cfg["env"] = env
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if kept:
        ok("kept from the config before: " + ", ".join(kept))
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    cfg["installer"] = {**inst, "models_dir": str(models), "model_dir": str(d), "written": time.strftime("%Y-%m-%d %H:%M")}
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, cfg['port'])}")
    return cfg_path


def hy4_smaller_fit(quant: str, gpus: list, pictures: bool = True) -> tuple[str, str] | None:
    """Where `quant` does not fit these cards even at its smallest context: the largest smaller Maya size that does
    (hy4_fit) and that the setup offers - its files published, the layout it takes there too: (key, layout), or None.
    Another set of files (Tencent's Q4_K_M): any Maya size."""
    keys = list(HY4_QUANTS)
    for k in reversed(keys[:keys.index(quant)] if quant in keys else keys):
        q = hy4_quant_table(k)
        if not hy4_layouts(q) or not hy4_published(k):
            continue
        f = hy4_fit(q, gpus, pictures=pictures)
        if "error" not in f and hy4_published(k, f["layout"]):
            return k, f["layout"]
    return None


def hy4_fit_step(a, pc: dict, quant: str, prev_ctx: int | None = None, have=None) -> dict:
    """The layout and the context of a size on the cards chosen (hy4_fit), said; what does not fit stops the setup
    here, before any download.  --model-dir: the layout of the trunk in that folder.  The context: --context, else
    asked among those that fit (the one set up before, else the recommended one).  A layout the table marks
    "published": "hub" must be in a folder here already (--model-dir, or have(layout): the folder the setup would
    use) or on Hugging Face now - else the setup stops with a plain line, before any build or download."""
    q = hy4_quant_table(quant)
    name = Q.shown("hy4", quant)
    if not hy4_layouts(q):
        hy4_entry(quant)                               # (a size with no files yet: its own message)
    forced = hy4_layout_in(Path(a.model_dir).expanduser(), quant) if a.model_dir else None
    ram = mem_gb()[0]
    # (the DFly2 drafter where the setup would pack it - one GPU, not --drafting mtp: its room first; with drafting off
    #  too - the engine counts the room of the drafter the pack holds either way, so on and off keep one start)
    de = hy4_dfly2_entry(quant)
    dg = hy4_dfly2_gib(quant) if de and de["published"] and len(pc["gpus"] or []) == 1 and \
        getattr(a, "drafting", None) != "mtp" else 0.0
    fit = hy4_fit(q, pc["gpus"], a.context, layout=forced, pictures=not a.no_vision, name=name, ram_gb=ram, drafter=dg)
    if "error" not in fit and hy4_layouts(q)[fit["layout"]].get("published") == "hub" and not forced and \
            read_json(HY4_MANIFEST).get("revision") and \
            not (have and hy4_layout_in(have(fit["layout"]), quant) == fit["layout"]):
        hub = hy4_layout_hub(quant, fit["layout"])     # (no folder here has this layout's file: is it uploaded?)
        std = hy4_layouts(q).get(HY4_STANDARD)
        what = "8 GB-card" if fit["layout"] == "8gb" else f"{fit['layout']}-layout"
        if hub is None:
            fail(f"{name}: Hugging Face did not answer, so the setup cannot tell whether the {what} file is there",
                 f"check the connection and run {ME} --setup again")
        if not hub:
            fail(f"{name}: the {what} file is not published yet" +
                 (f" (this PC has {fit['usable']:.1f} GB of free VRAM; the standard file needs about "
                  f"{hy4_fit_need_gib(std['plan'], HY4_CONTEXTS[0]):.1f} GB there)" if std else ""),
                 f"run {ME} --setup again later: the file is used by itself once it is on Hugging Face (no update "
                 "needed); a card with about 10 GB of VRAM or more runs the standard file now")
    if "error" not in fit and not a.context and not getattr(a, "check", False):
        ctx = choose_context(a, prev_ctx, fit["contexts"], fit["recommended"])
        fit = hy4_fit(q, pc["gpus"], ctx, layout=fit["layout"], pictures=not a.no_vision, name=name, ram_gb=ram,
                      drafter=dg)
    if "error" in fit:
        hint = fit["hint"]
        alt = hy4_smaller_fit(quant, pc["gpus"], not a.no_vision) if fit.get("too_big") and not forced else None
        if alt:                                        # (a smaller size runs here: said, with its command)
            k, lay = alt
            where = "these cards" if len(pc["gpus"] or []) > 1 else "this card"
            hint = (f"the smaller {Q.shown('hy4', k)} fits {where}" + (" (its 8 GB-card file)" if lay == "8gb" else "")
                    + f": {ME} --setup --model {Q.setup_id('hy4', k)} - or {hint}")
        fail(fit["error"] + (f" (the files in {a.model_dir} are the {forced} layout's)" if forced else ""), hint)
    if fit["layout"] == HY4_STANDARD and hy4_amd(pc["gpus"]) and any(hip_unified_memory(g) for g in pc["gpus"]):
        # (a unified-memory APU: the next line - set_up_hy4's hy4_apu_note - says whether the experts all sit there)
        ok(f"{name} fits: about {fit['need']:.1f} GB for what must sit on the GPU, of the {fit['usable']:.1f} GB of "
           "this PC's memory the GPU can take")
    elif fit["layout"] == HY4_STANDARD:
        ok(f"{name} fits: about {fit['need']:.1f} GB of the {fit['usable']:.1f} GB of VRAM for what must sit there; "
           "the engine fills the rest with the most-routed experts and computes the others on the CPU; it reads "
           "prompts in the largest chunk these cards hold beside the model and the pictures (up to 4096 tokens - its "
           "start line says which)")
    else:
        ok(f"{name} in its {fit['layout']} layout ({hy4_layouts(q)[fit['layout']]['about']}): about "
           f"{fit['need']:.1f} GB of the {fit['usable']:.1f} GB of VRAM for what must sit there; the routed experts "
           f"are computed on the CPU, prompts are read in chunks of {fit['chunk']} tokens")
    ok(f"context: {fit['context']} tokens" + (" (recommended for this VRAM)" if fit["context"] == fit["recommended"] else ""))
    if fit.get("kv_streaming"):
        ok(f"at this context the attention cache's older rows live in RAM: about {fit['kv_ram']:.1f} GB of it, kept "
           "pinned (the answers are the same; long prompts and answers are somewhat slower)")
    if max(fit["contexts"]) < HY4_CONTEXTS[-1]:        # (the model reaches 1M: what keeps this PC below it, said once)
        ok(f"a context above {context_label(max(fit['contexts']))} needs more VRAM than "
           f"{'these cards have' if len(pc['gpus']) > 1 else 'this card has'} free (the model's own reach: "
           f"{context_label(HY4_CONTEXTS[-1])})")
    if fit["context"] > HY4_TESTED_CONTEXT:
        warn(f"past {context_label(HY4_TESTED_CONTEXT)} the engine has not been run end to end yet")
    return fit


HY4_FAST_DRAFTING_SAID = "faster drafting, slightly different wording than the default"


def hy4_fast_drafting_step(a, pc: dict, quant: str, fit: dict, before: bool = False) -> tuple:
    """Step 2's faster drafting for a Maya size (hy4_fast_drafting_fit: the cache in 8-bit rows where that gives the
    DFly2 drafter its room): (the start the setup goes on with, the faster one where it applies here but was not
    taken - step 5's drafting line names it - else None).  --fast-drafting on: the faster where it applies (else
    said why not; the default); off: the default; not given: the config before's choice (before) where it still
    applies, else asked where the setup's sums give the drafter no room beside the F16 rows - the default first
    (--yes: the default).  Never the default by itself: its answers are not the default's."""
    want, how = getattr(a, "fast_drafting", None), getattr(a, "drafting", None)
    if how in ("off", "mtp"):
        if want == "on":
            warn(f"faster drafting: not set up with --drafting {how} - it is for the DFly2 drafter; the cache's rows "
                 "stay the default's")
        return fit, None
    fast = hy4_fast_drafting_fit(quant, pc["gpus"], fit, ram_gb=mem_gb()[0])
    if fast is None:
        if want == "on" or (want is None and before):
            usable = float(fit.get("usable") or 0)
            why = (f"the DFly2 drafter runs on one GPU for now (this PC: {len(pc['gpus'])})" if len(pc["gpus"]) != 1
                   else "this size has no DFly2 drafter" if hy4_dfly2_entry(quant) is None
                   else "at this context the cache is in 8-bit rows by default already" if fit.get("kv") == "q8"
                   else "at this context the cache's older rows live in RAM" if fit.get("kv_streaming")
                   else "the MTP block drafts here, with the default's answers"
                   if fit.get("layout") == HY4_STANDARD and hy4_mtp_entry() is not None and
                   float(fit.get("need") or 0) + hy4_mtp_gib(int(fit.get("chunk") or 1024)) <= usable
                   else f"even with the cache in 8-bit rows the DFly2 drafter has no room at this context "
                        f"({usable:.1f} GB free)")
            (warn if want == "on" else ok)(f"faster drafting: not set up - {why}; the default start")
        return fit, None
    if want is None and not before and fast["f16_room"]:
        return fit, None                               # (the default claims the drafter: nothing asked, nothing said)
    if want is None and not before:
        rows = [("Default: the cache in F16 rows, no drafting at this context", "recommended",
                 "The answers as Maya's default writes them."),
                ("Faster drafting: the cache in 8-bit rows, so the DFly2 drafter fits", None,
                 f"Decoding is faster; {HY4_FAST_DRAFTING_SAID}.")]
        pick = choose("Faster drafting?",
                      [f"At a context of {context_label(fit['context'])} the DFly2 drafter has no room on this card beside "
                       "the cache's F16 rows.", "With the cache in 8-bit rows it fits: faster drafting, slightly "
                       "different wording than the default."], rows, 0, a.yes or getattr(a, "check", False))
        want = "on" if pick == 1 else "off"
    if want == "off":                                  # (not asked for by name: step 5 names the faster one)
        return fit, (fast if getattr(a, "fast_drafting", None) is None else None)
    ok("faster drafting: on" + (" (as set up before)" if getattr(a, "fast_drafting", None) is None else "") +
       f" - the cache in 8-bit rows at this context, so the DFly2 drafter fits" +
       (" also where the engine finds no room for it beside F16 rows (its start line says which)"
        if fast["f16_room"] else "") +
       f": {HY4_FAST_DRAFTING_SAID} (F16 rows; {ME} --setup --fast-drafting off sets the default up again)")
    return fast, None


def set_up_hy4(a, prev: dict) -> Path | None:
    """Hy4-Flash's steps 1-8: the files (Maya's sizes, or Tencent's own Q4_K_M), the PC (NVIDIA from V100 on; AMD on
    Linux - under Windows experimental: hy4_usable_gib counts a unified-memory APU's memory, hy4_apu_note says what
    it holds), the
    context - a Maya size: with the layout that fits its VRAM, worked out before any download; the Q4_K_M: recommended
    by the smallest GPU's VRAM, and its drafting - the engine (build/maya-hy4), the files (pinned, sha256-checked), the
    pack, the config.  A Maya size drafts with Tencent's MTP block from its own file (hy4_mtp_step: one for every
    size) where the block fits in the VRAM beside the model's own parts (hy4_mtp_gib) - on by default, --drafting off
    or a setup before that chose off: without.  Nothing to measure yet."""
    inst = prev.get("installer") or {}
    prev_args = [str(x) for x in prev.get("args") or []]
    quant = choose_hy4_quant(a, prev)
    same = inst.get("hy4_quant") == quant              # (the context and the drafting before: this set's own)
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) \
        if same and "--max-context" in prev_args[:-1] else None
    name = Q.shown("hy4", quant)
    a.gpus_before = prev.get("gpu")                    # (the GPU question's default: this config's GPUs)
    pc = check_pc(a, "hy4")                            # 1
    total, _ = mem_gb()
    vram = sum(g["vram_gb"] for g in pc["gpus"])
    room = hy4_engine_room(pc)                         # (Windows: what the engine's CUDA gets on each card, measured)
    if room and len(room) == len(pc["gpus"]) and \
            sum(room.values()) < hy4_usable_gib([{k: v for k, v in g.items() if k != "engine_room_gb"}
                                                 for g in pc["gpus"]]):
        how = {g.get("engine_room_how") for g in pc["gpus"]}
        ok(f"VRAM: Windows keeps part of {'these cards' if len(pc['gpus']) > 1 else 'this card'} from the programs "
           f"on it - the setup counts {sum(room.values()):.1f} of its {vram:.1f} GB, from CUDA's own reading (the "
           "figure the engine plans with) and CUDA's own share " +
           ("as the engine measured it at its first start here (measured)" if how == {"measured"} else
            f"guessed until the engine's first start here measures it (estimated: {HY4_FIT_CUDA_OWN_GIB:g} GB)"
            if how == {"estimated"} else "measured on some cards, guessed on the others (estimated)"))
    need = float(hy4_quant_table(quant).get("memory_gb") or 0)
    models = (Path(a.models_dir).expanduser().resolve() if a.models_dir else
              Path(inst["models_dir"]) if inst.get("models_dir") else ROOT.parent / "Maya-data" / "models")
    step(2, "your choices")                            # 2
    drafting, plain_fit, fast_alt = False, None, None
    if quant == HY4_Q4:
        m = hy4_entry()
        ok(f"files: {Q.shown('hy4', quant, m['download_gb'])} - Tencent's own Q4_K_M (AngelSlim's GGUF; the repo's "
           "LICENSE and Tencent's terms for Hy4-flash apply)")
        if WIN or any(int(g.get("arch") or 0) < HY4_Q4_TESTED_CC for g in pc["gpus"]):
            warn("these files with their drafting have been run on Linux with NVIDIA RTX 30 / A100 and newer cards; "
                 f"on this PC Maya's own sizes are the ones tested ({ME} --setup --model hy4-flash)")
        if total > 0 and vram + total < need:
            warn(f"{vram:.0f} GB of VRAM and {total:.0f} GB of RAM: below the ~{need:.0f} GB that hold all of {name} - "
                 "slower: parts of the model are then read from the SSD")
        rec = hy4_recommended_context(pc)
        q4_ctxs = hy4_q4_contexts(pc.get("gpus") or [])
        ctx = choose_context(a, prev_ctx, q4_ctxs, rec)
        ok(f"context: {ctx} tokens" + (" (recommended for this PC's VRAM)" if ctx == rec else "") +
           f"; its cache: up to {hy4_kv_gb(ctx):.1f} GB of VRAM (on each GPU of a tensor-parallel group)")
        if q4_ctxs[-1] < HY4_Q4_LONG[-1]:
            ok(f"a context above {context_label(q4_ctxs[-1])} needs more VRAM on each card (the model's own reach: "
               f"{context_label(HY4_Q4_LONG[-1])})")
        if ctx > HY4_Q4_CONTEXTS[-1]:
            warn(f"past {context_label(HY4_Q4_CONTEXTS[-1])} these files have not been run end to end yet")
        how = getattr(a, "drafting", None)
        if how == "measure":
            warn(f"no speed calibration for {HY4} yet: it drafts with its MTP block (--drafting off: without)")
        drafting = how != "off" if how else ("--mtp" in prev_args or not same)
        ok("drafting: the model's MTP block drafts tokens for it to check - the same answers" if drafting else
           "drafting: off (--drafting on drafts with the model's MTP block)")
        fit = {"layout": HY4_STANDARD, "args": [], "context": ctx, "vision": True}
    else:
        ok(f"files: {name} (Maya's quant)")
        fit = hy4_fit_step(a, pc, quant, prev_ctx, lambda lay: hy4_dir(a, models, prev, quant, lay))
        plain_fit = fit
        fit, fast_alt = hy4_fast_drafting_step(a, pc, quant, fit, same and bool(inst.get("fast_drafting")))
        note = hy4_apu_note(pc["gpus"], hy4_quant_table(quant), name)
        if note:
            (ok if note[0] else warn)(note[1])
        elif total > 0 and vram + total < need:
            warn(f"{vram:.0f} GB of VRAM and {total:.0f} GB of RAM: below the ~{need:.0f} GB that hold all of {name} - "
                 "slower: parts of the model are then read from the SSD")
    d = hy4_dir(a, models, prev, quant, fit["layout"])
    ok(f"model folder: {d}")
    llama = tools_step(a)                              # 3
    meta = build_step(a, pc, llama, "maya-hy4")        # 4
    files = hy4_model_step(a, d, quant, fit["layout"], models)   # 5
    if files is None:
        return None
    vis = hy4_vision_step(a, d, quant, models)
    # a Maya size's drafts: the block where it fits on the main GPU beside the model's own parts at this context
    # (pictures take no VRAM of their own) - a small-card layout has no room for it, and it is not fetched there;
    # where it does not fit (or --drafting dfly2), the size's DFly2 drafter where that fits - on one GPU (the engine
    # runs it with the layers it reads and the head on one card)
    how = getattr(a, "drafting", None) if quant != HY4_Q4 else None
    mtp, mtp_why, dfly, dfly_why = None, "", None, ""
    # (the faster drafting's start: the block counted on the default's, where it has no room - hy4_fast_drafting_fit)
    f0 = plain_fit or fit
    usable = float(fit.get("usable") or 0)
    # (the engine loads the block only where its room leaves the start as it is but for the experts' places - the
    #  cache's rows, KV streaming - and counts that room with drafting off too: hy4_mtp_need_gib)
    if quant != HY4_Q4 and hy4_mtp_entry() is not None and how != "dfly2":
        room = hy4_mtp_need_gib(quant, f0)
        if f0["layout"] != HY4_STANDARD:
            mtp_why = (f"the draft block needs about {hy4_mtp_gib():.1f} GB of VRAM beside the model's own parts, "
                       "more than this card's layout leaves")
        elif f0.get("kv_streaming") or room > usable:
            mtp_why = (f"the draft block needs about {hy4_mtp_gib():.1f} GB of VRAM beside the model's own parts " +
                       ("(at this context the cache's older rows live in RAM, and beside it fewer would stay in VRAM)"
                        if f0.get("kv_streaming") else
                        f"({room:.1f} GB with it at this context, {usable:.1f} GB free on this PC's cards)") +
                       ": the engine loads it only where it leaves the rest of the start as it is (the cache's rows, "
                       "KV streaming), so the answers with drafts are those without")
            q, used, ram = hy4_quant_table(quant), gpu_used_mib(), mem_gb()[0]
            for c in sorted((c for c in f0.get("contexts") or [] if c < f0["context"]), reverse=True):
                f = hy4_fit(q, pc["gpus"], c, used=used, layout=f0["layout"], pictures=bool(f0.get("vision", True)),
                            ram_gb=ram)
                if "error" not in f and not f.get("kv_streaming") and hy4_mtp_need_gib(quant, f) <= f["usable"]:
                    mtp_why += f" (at a context of {context_label(c)} it fits: {ME} --setup --context {c})"
                    break
        else:
            mtp = hy4_mtp_step(a, d, quant, models)
            if mtp is None:
                mtp_why = ("the draft block's file is not in the model's folder" +
                           ("" if hy4_mtp_entry()["published"] else " (it is not published yet)"))
    # (where the block fits too, the drafter beside it: which of the two drafts is measured on this PC - the speed
    #  calibration's short A/B at the first start - and until then the faster for this size where it was measured)
    #  The engine loads the drafter only where its room leaves the start as it is but for the experts' places (the
    #  cache's rows, the prompt chunk, KV streaming): elsewhere it decodes without drafts, and the setup says so
    if how != "mtp" and hy4_dfly2_entry(quant) is not None:
        room = hy4_dfly2_need_gib(quant, fit)
        if len(pc["gpus"]) > 1:
            dfly_why = f"the DFly2 drafter runs on one GPU for now (this PC: {len(pc['gpus'])})"
        elif fit.get("kv_streaming") or room > usable:
            dfly_why = (f"the DFly2 drafter needs about {hy4_dfly2_gib(quant):.1f} GB beside the model's own parts " +
                        ("(at this context the cache's older rows live in RAM, and beside it fewer would stay in VRAM)"
                         if fit.get("kv_streaming") else
                         f"({room:.1f} GB with it at this context, {usable:.1f} GB free)") +
                        ": the engine loads it only where it leaves the rest of the start as it is (the cache's rows, "
                        "the prompt chunk), so the answers with drafts are those without")
            q, used, ram = hy4_quant_table(quant), gpu_used_mib(), mem_gb()[0]
            for c in sorted((c for c in fit.get("contexts") or [] if c < fit["context"]), reverse=True):
                f = hy4_fit(q, pc["gpus"], c, used=used, layout=fit["layout"], pictures=bool(fit.get("vision", True)),
                            ram_gb=ram, drafter=hy4_dfly2_gib(quant))
                if "error" not in f and not f.get("kv_streaming") and hy4_dfly2_need_gib(quant, f) <= f["usable"]:
                    dfly_why += f" (at a context of {context_label(c)} it fits: {ME} --setup --context {c})"
                    break
            if fast_alt:                               # (the opt-in: this context, the cache in 8-bit rows)
                dfly_why += (f"; with the cache in 8-bit rows it fits at {context_label(fit['context'])}: "
                             f"{HY4_FAST_DRAFTING_SAID} ({ME} --setup --fast-drafting on)")
        else:
            dfly = hy4_dfly2_step(a, d, quant, models)
            if dfly is None:
                dfly_why = ("the DFly2 drafter's file is not in the model's folder" +
                            ("" if hy4_dfly2_entry(quant)["published"] else " (it is not published yet)"))
                if fit.get("fast_drafting"):           # (no drafter: the 8-bit rows would only change the answers)
                    warn("faster drafting: not set up - " + dfly_why + "; the default start")
                    fit = plain_fit
    pack = hy4_pack_step(a, d, files, vis, mtp, dfly)  # 6
    step(7, "images (the vision tower)")               # 7
    pictures = hy4_pack_has_vision(pack) and not a.no_vision and bool(fit.get("vision", True))
    if not hy4_pack_has_vision(pack):
        (ok if a.no_vision else warn)("pictures: off (--no-vision): the model reads text only" if a.no_vision else
                                      f"no vision tower beside {name}'s files: the model reads text only ({ME} "
                                      "--setup asks for it again)")
    elif a.no_vision:
        ok("pictures: off (--no-vision; the pack has the vision tower: Images on the Models page turns them on)")
    elif not pictures:
        warn(f"pictures: OFF on this PC - at a context of {fit['context']} tokens the model's own parts fill the VRAM "
             f"({fit.get('usable', 0):.1f} GB free, the model's own parts {fit.get('need', 0):.1f} GB): text works as "
             "before, a picture sent is refused in plain words")
    elif fit.get("vision_small"):
        ok("pictures: ON - the model's own vision tower runs in the engine, a picture at a time: its weights stay in "
           "the file (RAM as the page cache) and no VRAM is kept for pictures. At this context the card is a little "
           "short of the tower's largest picture, so the server makes very big pictures slightly smaller before the "
           "model reads them (the engine's start line says the size)")
    else:
        ok("pictures: ON - the model's own vision tower runs in the engine, a picture at a time: its weights stay in "
           "the file (RAM as the page cache) and no VRAM is kept for pictures (a picture's room, under 1 GB, is lent "
           "by the prompt buffers while it is read); the server resizes the pictures, the Chat's picture button is "
           "shown")
    drafter = "dfly2" if dfly is not None else "mtp"
    extra, measure_now = {}, False
    if quant != HY4_Q4 and mtp is not None and dfly is not None:   # (both fit: the faster on this PC, measured)
        cal = inst.get("speed_calibration") or {}
        carried = same and inst.get("layout") == fit["layout"] and inst.get("draft_choice") == "auto" and \
            cal.get("kept") in ("mtp", "dfly2") and how != "measure"
        drafter = cal["kept"] if carried else HY4_DRAFT_RULE.get(quant, "mtp")
        extra["draft_choice"] = "auto"
        if carried:
            extra["speed_calibration"] = cal
        elif getattr(a, "no_speed_calibration", False):
            extra["speed_calibration"] = {"date": time.strftime("%Y-%m-%d %H:%M"), "version": maya_version(),
                                          "skipped": "chosen", "kept": drafter}
        had = inst.get("draft_block") or inst.get("dfly2")
        drafting = how != "off" if how else not (same and had and inst.get("drafting") == "off")
        measure_now = drafting and how == "measure"
        one = {"mtp": "the MTP block", "dfly2": "the DFly2 drafter"}
        if not drafting:
            ok("drafting: off (--drafting on, or Drafting on the Models page, drafts with the MTP block or the DFly2 "
               "drafter: both are in the pack)")
        elif how not in ("mtp", "dfly2", "on"):         # (--drafting auto: the engine's own measurement)
            ok("drafting: auto - Tencent's MTP block and DFly2 drafter both fit here; each drafts tokens the model "
               "checks, so only tokens the model itself would pick are written. The engine measures once on this PC, "
               "at its first start (before it is ready), whether no drafting, the MTP block or the DFly2 drafter "
               "decodes fastest - each with its own VRAM plan, all of it for the experts without drafting - and keeps "
               f"that one ({ME} --speed-calibration measures again; --setup --drafting off|mtp|dfly2 picks one by "
               "hand: another plan, so an answer may be worded differently)")
        elif carried:
            ok(f"drafting: {one[drafter]} - the faster of the two on this PC (the speed calibration of "
               f"{cal.get('date', 'before')}; {ME} --speed-calibration measures again)")
        else:
            ok("drafting: Tencent's MTP block and DFly2 drafter both fit here - each drafts tokens the model checks, "
               "so only tokens the model itself would pick are written. " +
               ("Maya measures now which is faster on this PC" if measure_now else
                f"Until a speed calibration measures which is faster on this PC ({ME} --speed-calibration; "
                "otherwise once, at the first start that is not unattended)" if "speed_calibration" not in extra else
                "Not measured (--no-speed-calibration)") +
               f": {one[drafter]} (the faster for {name} where it was measured)")
    elif quant != HY4_Q4:
        if mtp is None and dfly is None:
            (warn if how in ("on", "measure", "mtp", "dfly2") else ok)(
                f"drafting: off - {mtp_why or ('no draft block for these files' if how != 'dfly2' else '--drafting dfly2')}"
                + (f"; {dfly_why}" if dfly_why else ""))
        else:
            what = "the MTP block" if dfly is None else "the DFly2 drafter"
            extra["draft_choice"] = how if how in ("mtp", "dfly2") else "only"
            measure_now = how == "measure"                 # (it against no drafting: the engine's start measures it)
            had = inst.get("draft_block") if dfly is None else inst.get("dfly2")
            # (the faster drafting drafts: its 8-bit rows are there for the drafter)
            chose_off = same and had and inst.get("drafting") == "off" and not fit.get("fast_drafting")
            drafting = how != "off" if how else not chose_off
            if drafting and how not in ("mtp", "dfly2", "on"):   # (--drafting auto: the engine's own measurement)
                ok(f"drafting: auto - {what} fits here (it drafts tokens the model checks, so only tokens the model "
                   "itself would pick are written). The engine measures once on this PC, at its first start (before "
                   f"it is ready), whether {what} or no drafting decodes faster - each with its own VRAM plan, all of "
                   f"it for the experts without drafting - and keeps that one ({ME} --speed-calibration measures "
                   "again; --setup --drafting off|on picks by hand: another plan, so an answer may be worded "
                   "differently)" + (f" (the MTP block: {mtp_why})" if mtp_why and dfly is not None else
                                     f" ({dfly_why})" if dfly_why and dfly is None else ""))
            elif dfly is None:
                ok("drafting: Tencent's MTP block drafts tokens and the model checks every one - only tokens the "
                   f"model itself would pick are written; about {hy4_mtp_gib(int(fit.get('chunk') or 1024)):.1f} GB "
                   "of the main GPU's VRAM" if drafting else
                   "drafting: off (--drafting on, or Drafting on the Models page, drafts with the MTP block)")
            else:
                ok("drafting: Tencent's DFly2 drafter drafts tokens and the model checks every one - only tokens the "
                   f"model itself would pick are written; about {hy4_dfly2_gib(quant):.1f} GB of the GPU's VRAM" +
                   (f" (the MTP block: {mtp_why})" if mtp_why else "") if drafting else
                   "drafting: off (--drafting on, or Drafting on the Models page, drafts with the DFly2 drafter)")
    cfg_path = write_config_hy4(a, pc, meta, pack, fit["context"], models, d, drafting, quant, fit, pictures,
                                drafter, extra)   # 8
    if measure_now:
        speed_calibration(cfg_path, measure_asked(a))
    return cfg_path


# ------------------------------------------------------------------------------------------------ CYBER-FROST-3.8-PS
def qwen_entry(quant: str = "q5_k_m") -> dict:
    """CYBER-FROST-3.8-PS's download (tools/qwen/download.json) as a MODELS entry (offer_download's form); quant
    QWEN_UD: unsloth's UD-IQ2_XS of Qwen3.8-Flash-Next (tools/qwen/download_unsloth.json: its n-gram shard from Maya's
    repo, its MTP head from the repo's top - "sources")."""
    if quant == QWEN_UD:
        man = read_json(QWEN_UD_MANIFEST)
        if not man.get("files") or not man.get("revision") or not man.get("ngram") or not man.get("mtp"):
            fail(f"{QWEN_UD_MANIFEST} is missing or damaged",
                 "restore it (git checkout tools/qwen/download_unsloth.json)")
        src = man.get("sources") or {}
        return {"about": f"Qwen3.8-Flash-Next in unsloth's UD-IQ2_XS GGUF ({man.get('about', '')}); "
                         f"{man.get('license', 'the Qwen Community License')}",
                "format": "qwen-gguf", "repo": man["repo"], "revision": man["revision"],
                "folder": man.get("folder", ""),
                "files": {k: v[0] for k, v in man["files"].items()},
                "sha256": {k: v[1] for k, v in man["files"].items()},
                "sources": {k: (v["repo"], v["revision"]) for k, v in src.items()},
                "source_folders": {k: v.get("folder", "") for k, v in src.items()},
                "ngram": man["ngram"], "mtp": man["mtp"],
                "download_gb": sum(v[0] for v in man["files"].values()) / 1e9,
                "ngram_gb": man["files"][man["ngram"]][0] / 1e9,
                "memory_gb": float(man.get("memory_gb") or 47)}
    man = read_json(QWEN_MANIFEST)
    if not man.get("files") or not man.get("revision"):
        fail(f"{QWEN_MANIFEST} is missing or damaged", "restore it (git checkout tools/qwen/download.json)")
    return {"about": f"{QWEN_SHOWN} in peasantsmith's GGUF ({man.get('about', 'Q5_K_M')}); "
                     f"{man.get('license', 'the Qwen Community License')}",
            "format": "qwen-gguf", "repo": man["repo"], "revision": man["revision"], "folder": "",
            "files": {k: v[0] for k, v in man["files"].items()}, "sha256": {k: v[1] for k, v in man["files"].items()},
            "download_gb": sum(v[0] for v in man["files"].values()) / 1e9,
            "memory_gb": float(man.get("memory_gb") or 84)}


def qwen_kv_gb(ctx: int) -> float:
    """GB of VRAM the context's KV cache takes (the GatedDeltaNet layers' state is fixed: ~0.1 GB)."""
    return ctx * QWEN_KV_BYTES / 2**30


def qwen_recommended_context(vram: float, starved: bool = False) -> int:
    """The context the setup recommends here: the largest whose KV cache takes at most ~12% of the VRAM (the rest stays
    the experts' - a longer context makes every answer slower), at least the smallest.  On a PC where VRAM and RAM
    together cannot hold every expert (starved) the rest are read from the SSD, and every expert slot in VRAM saves
    those reads - the KV cache then takes at most ~6% (a V100 32 GB with 30 GB of RAM: 128K, not 256K)."""
    fits = [c for c in QWEN_CONTEXTS if qwen_kv_gb(c) <= (0.06 if starved else 0.12) * vram]
    return max(fits) if fits else QWEN_CONTEXTS[0]


def qwen_dir(a, models: Path, prev: dict, quant: str = "q5_k_m") -> Path:
    """The model's folder: --model-dir, else the folder of the setup before of the same files (its pack is inside it),
    else <models folder>/CYBER-FROST-3.8-PS (unsloth's UD-IQ2_XS: <models folder>/Qwen3.8-Flash-Next-UD-IQ2_XS)."""
    if a.model_dir:
        return Path(a.model_dir).expanduser().resolve()
    args = prev.get("args") or []
    if "--native" in args[:-1] and ((prev.get("installer") or {}).get("qwen_quant") or "q5_k_m") == quant:
        d = Path(args[args.index("--native") + 1]).parent
        if d.is_dir():
            return d
    return models / (QWEN_UD_DIR if quant == QWEN_UD else QWEN)


def link_published(dst: Path, size: int, sha: str, places: list) -> Path | None:
    """dst missing, the same published file (its size, then its sha256 - never assumed) whole in one of `places`:
    hard-linked from it (one copy on the disk, no second download), with its sha256 record.  The source linked, or None
    (none there, or on another drive: the download fetches it)."""
    if dst.exists():
        return None
    for src in dict.fromkeys(Path(p) for p in places):
        try:
            if src.resolve() == dst.resolve() or not src.is_file() or src.stat().st_size != size:
                continue
            if not sha256_checked(src, sha):
                continue
            dst.parent.mkdir(parents=True, exist_ok=True)
            os.link(src, dst)
        except OSError:
            continue
        seen = read_json(src.parent / VERIFIED).get(src.name)
        if seen:   # (a hard link: the same size and time - its check still holds)
            book = read_json(dst.parent / VERIFIED)
            book[dst.name] = seen
            try:
                (dst.parent / VERIFIED).write_text(json.dumps(book, indent=1), encoding="utf-8")
            except OSError:
                pass
        ok(f"{dst.name}: linked from {src.parent} (the same file, one copy)")
        return src
    return None


def qwen_places(models: Path, rel: str) -> list:
    """Where another Qwen setup keeps the file `rel` (relative to its model folder): both quants' default folders in the
    models folder and the folders of the Qwen configs here."""
    dirs = [models / QWEN, models / QWEN_UD_DIR]
    for c in configs():
        cfg = read_json(c)
        if config_family(cfg) == "qwen" and (cfg.get("installer") or {}).get("model_dir"):
            dirs.append(Path(cfg["installer"]["model_dir"]))
    return [x / rel for x in dirs]


def qwen_model_step(a, d: Path, quant: str = "q5_k_m", models: Path | None = None) -> list | None:
    """The model's files in d (downloaded when they are not there, then checked against their published sha256): [the
    GGUF, or unsloth's shards and MTP head - the first shard first], or None when nothing was downloaded.  A file
    another Qwen setup here already has (the same sha256) is hard-linked from it."""
    step(5, f"the model ({Q.shown('qwen', quant)})")
    m = qwen_entry(quant)
    files = [d / n for n in m["files"]]
    for f in files:
        if file_incomplete(m, f) and models is not None:
            link_published(f, m["files"][f.name], m["sha256"][f.name], qwen_places(models, f.name))
    missing = [f for f in files if file_incomplete(m, f)]
    if missing and not offer_download(a, Q.shown("qwen", quant, m["download_gb"]), d, files, m):
        return None
    for f in files:                                    # (a file that was already there: checked once, then remembered)
        if not sha256_checked(f, m["sha256"][f.name]):
            fail(f"{f.name}: the sha256 does not match the published one - still being downloaded, or damaged",
                 f"let the download finish, or delete the file and run {ME} --download-model")
    ok(f"{Q.shown('qwen', quant)}: {sum(f.stat().st_size for f in files) / 1e9:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: the engine reads the n-gram table ({m.get('ngram_gb', 52):.0f} GB) from "
             "it row by row on every token, and the experts RAM cannot hold - use an NVMe SSD (--models-dir)")
    return files


def qwen_pack_step(a, d: Path, files: list, llama: Path, mtp_gguf: Path | None = None) -> Path:
    """tools/iq_pack.py's pack in <model>/pack (the index of every tensor, the small float weights, the tokenizer with
    the GGUF's chat template), and the MTP drafter built from the GGUF's NextN block (<pack>/mtp, tools/mtp_from_gguf.py;
    mtp_gguf: unsloth's separate MTP head) - built again when its tools or the model files changed."""
    step(6, "the pack (the engine's index of the model file) and the MTP drafter")
    pack = d / "pack"
    tools = S.source_hash(QWEN_PACK_TOOLS)
    need = PACK_FILES + ("mtp/dense.txt", "mtp/experts.bin")
    if not a.repack and read_json(pack / PACK_SOURCES).get("tools") == tools and \
            all((pack / f).exists() for f in need) and pack_stale(pack, files) is None:
        ok(f"pack already built: {pack}")
        return pack
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the pack is written next to the GGUF file")
    env = dict(os.environ, MAYA_GGUF_PY=str(llama / "gguf-py"))
    say("  Writing the index of every tensor, the small float weights (about 1.5 GB) and the tokenizer; the experts and")
    say("  the n-gram table stay in the GGUF file (a few minutes) ...")
    # (--compat-bf16: the projections the engine reads as BF16 - the hyper-connections' mixes, the router - are
    #  quantized in this GGUF: they are dequantized and rounded to BF16, the experts and the n-gram table unchanged)
    run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(files[0]), "--out", str(pack),
         "--compat-bf16"], env=env)
    missing = [f for f in PACK_FILES if not (pack / f).exists()]
    if missing:
        fail(f"the pack builder did not write {pack / missing[0]}", "the reason is in the messages above")
    say("  The MTP drafter from the model's NextN block (its 512 experts requantized to 2 bits for the drafts; a few "
        "minutes) ...")
    run([sys.executable, str(ROOT / "tools" / "mtp_from_gguf.py"), "--gguf", str(mtp_gguf or files[0]),
         "--out", str(pack / "mtp")], env=env)
    if QWEN_PROFILE.exists():                          # (the shipped ranking of the experts: which start in VRAM)
        shutil.copyfile(QWEN_PROFILE, pack / "expert-profile.bin")
    stamp_pack(pack, files)
    rec = read_json(pack / PACK_SOURCES)
    rec["tools"] = tools
    (pack / PACK_SOURCES).write_text(json.dumps(rec, indent=1), encoding="utf-8")
    ok(f"pack: {pack}")
    return pack


def write_config_qwen(a, pc: dict, meta: dict, pack: Path, gguf: Path, ctx: int, models: Path, d: Path,
                      vision: dict | None = None, quant: str = "q5_k_m", ple: Path | None = None) -> Path:
    """maya-cyber-frost.json (maya-cyber-frost-hip.json on AMD; unsloth's UD-IQ2_XS: maya-qwen-ud-iq2_xs.json) and its
    start script.  A setup again keeps what the config before says that this one does not ask: its engine settings,
    where the server listens and its key; the model's own sampling and thinking level every setup.  ple: the shard
    that holds the n-gram table (unsloth's shard 3), else the GGUF itself."""
    step(8, "the configuration and the start script")
    port = a.port or 8080
    hip = pc.get("backend") == "hip"
    cfg_path = ROOT / f"{Q.config_stem('qwen', quant)}{'-hip' if hip else ''}.json"
    old = read_json(cfg_path)
    profile = pack / "expert-profile.bin"
    # the Qwen path: the pack, the model file for its native weights and its n-gram table, decode windows of up to 4
    # tokens (prompt-lookup drafts; the MTP drafter where it measures faster - qwen_draft_calibration), prompts in
    # chunks sized from the expert cache it lends them, the expert cache sized from the free VRAM
    args = ["--pack", str(pack), "--native", str(gguf), "--ple-gguf", str(ple or gguf), "--spec", "4", "--prefill",
            "auto", "--expert-cache", "auto", "--max-context", str(ctx), "--kv", QWEN_KV]
    if quant == QWEN_UD:                               # (its n-gram table memory-mapped: the page cache, never loaded)
        args += ["--ple-io", "mmap"]
    if profile.exists():
        args += ["--expert-profile", str(profile)]
    # (the engine's server needs the MTP drafter loaded - the setup always builds it; drafting off is --mtp-max-t 1)
    if (pack / "mtp" / "dense.txt").exists():
        args += ["--mtp", str(pack / "mtp")]
    if vision:                                         # (pictures: GENI requests, each image's rows at its pad tokens)
        args.append("--vision")
    kept = keep_args(old.get("args") or [], args, ("--mtp-max-t", "--kv", "--vram-reserve-mib", "--pool-workers",
                                                   "--experts-from-gguf", "--ple-io"))
    if getattr(a, "vram_reserve_mib", None) is not None:   # (#1026) given now: written, and a later setup keeps it
        put = set_vram_reserve(args, "qwen", a.vram_reserve_mib)
        ok(f"VRAM kept free beside the expert cache: {a.vram_reserve_mib} MiB ({put})")
    inst = {"family": "qwen", "quant": QWEN_UD_DIR if quant == QWEN_UD else QWEN, "qwen_quant": quant}
    sampling, effort = Q.model_defaults({"installer": inst})   # (the model's own, every setup: its card's)
    cfg = {"exe": str(EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           **({"backend": "hip"} if hip else {}),
           "model_name": QWEN_UD_MODEL_NAME if quant == QWEN_UD else QWEN_MODEL_NAME,
           "gpu": [g["index"] for g in pc["gpus"]],
           "sampling": sampling, "reasoning_effort": effort,
           "lib_dirs": meta.get("lib_dirs") or old.get("lib_dirs") or [], "port": port,
           **({"vision": vision} if vision else {})}
    kept += keep_server_settings(cfg, old, a, ("layer_split",))
    env = {**(old.get("env") or {}), **parse_env(a.env)}
    if env:
        cfg["env"] = env
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if kept:
        ok("kept from the config before: " + ", ".join(kept))
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    cfg["installer"] = {**inst, "models_dir": str(models), "model_dir": str(d), "written": time.strftime("%Y-%m-%d %H:%M")}
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, cfg['port'])}")
    return cfg_path


def qwen_decode_rate(cfg: dict, ids: list, extra: list, log) -> float | None:
    """The config's engine through its server protocol, as the dashboard runs it (a layer split over several GPUs
    works only there): a warm-up answer, then `ids` answered greedily, 192 tokens - its decode rate, or None."""
    sys.path.insert(0, str(ROOT))
    import serve.server as SV
    try:
        p = subprocess.Popen([cfg["exe"], "--serve"] + SV.engine_args(cfg) + extra, stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=log, text=True, env=SV.child_env(cfg),
                             cwd=cfg.get("cwd") or str(ROOT), bufsize=1)
    except OSError:
        return None

    def gen(n_new):
        p.stdin.write(f"GEN {n_new} temperature=0 " + ",".join(map(str, ids)) + "\n")
        p.stdin.flush()
        for line in p.stdout:
            if line.startswith("DONE") or line.startswith("ERR"):
                return line.split()
        return ["EOF"]

    try:
        for line in p.stdout:
            if line.startswith("READY"):
                break
        else:
            return None
        gen(32)                                          # (warm-up: the first answer after a start is slower)
        f = gen(192)
        p.stdin.write("QUIT\n")
        p.stdin.flush()
        p.wait(timeout=120)
    except (OSError, ValueError, subprocess.TimeoutExpired):
        p.kill()
        return None
    # DONE <generated> <prompt> <prompt ms> <decode ms> ...
    return int(f[1]) / float(f[4]) * 1000.0 if f[0] == "DONE" and len(f) > 4 and float(f[4]) > 0 else None


def qwen_draft_calibration(cfg_path: Path) -> bool | None:
    """The MTP drafter, measured on this PC: the same answer decoded with its drafts and without them (the engine's
    own decode rate, greedy - the same tokens either way).  The engine's server always loads the drafter (--mtp); its
    drafts are turned off with --mtp-max-t 1 (windows of one token, prompt-lookup drafts only), kept only when drafting
    is not at least 3% faster.  True / False: drafting on / off; None: not measured (the reason said)."""
    cfg = read_json(cfg_path)
    args = cfg.get("args") or []
    pack = Path(args[args.index("--pack") + 1])
    mtp = pack / "mtp"
    if not (mtp / "dense.txt").exists():
        warn("no MTP drafter in the pack: decoding without it")
        return None
    sys.path.insert(0, str(ROOT / "tools"))
    import strata_tokenizer as ST
    tok = ST.Tokenizer.from_dir(Path(cfg["tokenizer"]))
    ids = tok.encode("<|im_start|>user\n" + BENCH_TOPICS[0] + "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n",
                     parse_special=True)
    for k in ("--mtp", "--mtp-max-t"):
        if k in args[:-1]:
            i = args.index(k)
            del args[i:i + 2]
    args += ["--mtp", str(mtp)]
    say()
    say("  Measuring the MTP drafter on this PC: one answer written with and without it (the model loads twice,")
    say("  a few minutes) ...")
    with open(ROOT / "maya-mtp-calibration.log", "w", encoding="utf-8") as log:
        off = qwen_decode_rate(dict(cfg, args=args), ids, ["--mtp-max-t", "1"], log)
        on = qwen_decode_rate(dict(cfg, args=args), ids, [], log)
    if off is None or on is None:
        warn("the drafter could not be measured (maya-mtp-calibration.log says why): decoding without it")
        return None
    use = on >= off * 1.03
    if not use:
        args += ["--mtp-max-t", "1"]
    cfg["args"] = args
    cfg.setdefault("installer", {})["mtp_calibration"] = {"off": off, "on": on, "date": time.strftime("%Y-%m-%d")}
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    ok(f"MTP drafting {'ON' if use else 'off'}: {on / off:.2f}x the decode without it on this PC")
    return use


def qwen_vision_model() -> dict | None:
    """The model's vision tower as offer_download's form (tools/qwen/download.json's "vision": llama.cpp's mmproj of the
    base model's tower, which the fine-tune left as it was), or None when the manifest names none."""
    man = read_json(QWEN_MANIFEST).get("vision") or {}
    if not man.get("files") or not man.get("repo") or not man.get("revision"):
        return None
    return {"about": f"{QWEN_SHOWN}'s vision tower (it lets the model read pictures): llama.cpp's mmproj of the base "
                     "model Qwen3.8-Flash-Next's tower, the same weights", "format": "mmproj",
            "repo": man["repo"], "revision": man["revision"], "folder": man.get("folder", ""),
            "files": {k: v[0] for k, v in man["files"].items()}, "sha256": {k: v[1] for k, v in man["files"].items()},
            "download_gb": sum(v[0] for v in man["files"].values()) / 1e9}


def vision_encoder_ready(a, pc: dict, meta: dict, llama: Path) -> bool:
    """build-vision's maya-vision for this PC's GPUs, compiled when it is not there or its source changed."""
    src = S.source_hash(("tools/vision",))
    vmeta = read_json(VSTAMP)
    if VEXE.exists() and not a.rebuild and vmeta.get("src") == src \
            and set(meta.get("archs") or pc.get("archs") or []) <= set(vmeta.get("archs", [])):
        return True
    if compile_vision(pc, meta, llama, src):
        return True
    warn("the vision encoder did not compile (the reason is above): the model reads text only - "
         f"{ME} --setup --rebuild tries again")
    return False


def qwen_vision_step(a, pc: dict, meta: dict, llama: Path, d: Path, gguf: Path,
                     models: Path | None = None) -> dict | None:
    """Step 7: the config's "vision" entry - maya-vision with the model's mmproj (in <model folder>/vision) and the
    GGUF's vocabulary - or None (text only, said why).  The server starts the encoder when a request carries a picture;
    the engine places its rows at the <|image_pad|> tokens with their M-RoPE positions (--vision: GENI requests).  The
    same mmproj serves CYBER-FROST and unsloth's GGUFs of the base model: another Qwen setup's copy is hard-linked."""
    step(7, "images (the vision encoder)")
    if a.no_vision:
        ok("skipped (--no-vision): the model reads text only")
        return None
    m = qwen_vision_model()
    if m is None:
        warn(f"{QWEN_MANIFEST} names no vision tower: the model reads text only")
        return None
    vd = d / "vision"
    files = [vd / n for n in m["files"]]
    for f in files:
        if models is not None and file_incomplete(m, f):
            link_published(f, m["files"][f.name], m["sha256"][f.name], qwen_places(models, f"vision/{f.name}"))
    if any(file_incomplete(m, f) for f in files) and not offer_download(a, "the vision tower", vd, files, m):
        warn(f"the vision tower was not downloaded: the model reads text only ({ME} --setup asks again)")
        return None
    for f in files:                                    # (offer_download checked the ones it fetched; these too)
        if not sha256_checked(f, m["sha256"][f.name]):
            f.unlink(missing_ok=True)                  # else the next run finds it complete and fails here again
            fail(f"{f}: the sha256 does not match the published one (the file was removed)", "run the setup again")
    if not vision_encoder_ready(a, pc, meta, llama):
        return None
    ok(f"pictures: ON - the vision encoder {VEXE} with {files[0].name} ({m['download_gb']:.1f} GB on the SSD); it "
       "starts only when a picture arrives, no VRAM kept for it")
    return {"exe": str(VEXE), "mmproj": str(files[0]), "model": str(gguf), "gpu": True}


def choose_qwen_quant(a, prev: dict) -> str:
    """Which Qwen files: --qwen-quant (or a --model id that names them), else asked - CYBER-FROST-3.8-PS's Q5_K_M or
    unsloth's UD-IQ2_XS of the base model; the files set up before are the default."""
    given = getattr(a, "qwen_quant", None)
    if given:
        return given
    before = (prev.get("installer") or {}).get("qwen_quant") or "q5_k_m"
    default = before if before in QWEN_QUANTS else "q5_k_m"
    rows = []
    for q in QWEN_QUANTS:
        m = qwen_entry(q)
        rows.append((Q.shown("qwen", q, m["download_gb"]), None,
                     "Qwen3.8-Flash-Next itself in unsloth's UD-IQ2_XS GGUF (its n-gram table from Maya's repo, its MTP "
                     f"head, the experts in about {m['memory_gb']:.0f} GB of VRAM and RAM together)" if q == QWEN_UD else
                     "Blackfrost-AI's cybersecurity fine-tune in peasantsmith's GGUF (Q5_K experts, all of them in about "
                     f"{m['memory_gb']:.0f} GB of VRAM and RAM together)"))
    pick = choose("Which Qwen3.8-Flash-Next?", ["The files to set up:"], rows, QWEN_QUANTS.index(default),
                  getattr(a, "yes", False))
    return QWEN_QUANTS[pick]


def set_up_qwen(a, prev: dict) -> Path | None:
    """CYBER-FROST-3.8-PS's steps 1-8: the PC (NVIDIA from V100, or AMD), the context (recommended by this PC's VRAM),
    the engine (build/maya), the file (pinned, sha256-checked), the pack and the MTP drafter, the config; then the
    drafter measured."""
    inst = prev.get("installer") or {}
    quant = choose_qwen_quant(a, prev)
    if (inst.get("qwen_quant") or "q5_k_m") != quant:   # (the other Qwen files' setup before: its own config's answers)
        prev = next((c for c in map(read_json, configs()) if config_family(c) == "qwen" and
                     ((c.get("installer") or {}).get("qwen_quant") or "q5_k_m") == quant), {})
        inst = prev.get("installer") or inst
    prev_args = prev.get("args") or []
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) if "--max-context" in prev_args[:-1] else None
    m = qwen_entry(quant)
    a.gpus_before = prev.get("gpu")                    # (the GPU question's default: this config's GPUs)
    pc = check_pc(a, "qwen")                           # 1
    total, _ = mem_gb()
    vram = sum(g["vram_gb"] for g in pc["gpus"])
    if vram + total < m["memory_gb"]:
        warn(f"{vram:.0f} GB of VRAM and {total:.0f} GB of RAM: below the ~{m['memory_gb']:.0f} GB that hold all of its "
             "experts - the engine reads the others from the SSD when they are routed (slower)")
    step(2, "your choices")                            # 2
    ok(f"model: {Q.shown('qwen', quant, m['download_gb'])} - " +
       ("unsloth's GGUF of Qwen3.8-Flash-Next " if quant == QWEN_UD else "a fine-tune of Qwen3.8-Flash-Next ") +
       "(Qwen Community License 1.0: LICENSE.qwen-community-1.0; a Model-as-a-Service business needs Qwen's own license)")
    rec = qwen_recommended_context(vram, vram + total < m["memory_gb"])
    ctx = choose_context(a, prev_ctx, QWEN_CONTEXTS, rec)
    ok(f"context: {ctx} tokens" + (" (recommended for this PC's VRAM)" if ctx == rec else "") +
       f"; its KV cache: {qwen_kv_gb(ctx):.1f} GB of VRAM")
    models = (Path(a.models_dir).expanduser().resolve() if a.models_dir else
              Path(inst["models_dir"]) if inst.get("models_dir") else ROOT.parent / "Maya-data" / "models")
    d = qwen_dir(a, models, prev, quant)
    ok(f"model folder: {d}")
    llama = tools_step(a)                              # 3
    meta = build_step(a, pc, llama)                    # 4
    files = qwen_model_step(a, d, quant, models)       # 5
    if files is None:
        return None
    ud = quant == QWEN_UD
    pack = qwen_pack_step(a, d, files, llama, d / m["mtp"] if ud else None)   # 6
    vision = qwen_vision_step(a, pc, meta, llama, d, files[0], models)       # 7
    cfg_path = write_config_qwen(a, pc, meta, pack, files[0], ctx, models, d, vision, quant,
                                 d / m["ngram"] if ud else None)                # 8
    if "mtp_calibration" not in (read_json(cfg_path).get("installer") or {}) or a.calibrate:
        how = drafting_choice(a, "the MTP drafter", "10-40")
        if how != "measure":
            keep_drafting(cfg_path, how == "on")
        else:
            if S.UI is not None:
                S.UI.step(9, "speed calibration: the MTP drafter or none")
            qwen_draft_calibration(cfg_path)
    return cfg_path


# ------------------------------------------------------------------------------------------------ Inkling-2-Mini
def ik2_entry() -> dict:
    """Inkling-2-Mini's download (tools/ik2/download.json: unsloth's UD-Q4_K_XL) as a MODELS entry (offer_download's
    form)."""
    man = read_json(IK2_MANIFEST)
    if not man.get("files") or not man.get("revision") or man.get("gguf") not in (man.get("files") or {}):
        fail(f"{IK2_MANIFEST} is missing or damaged", "restore it (git checkout tools/ik2/download.json)")
    src = man.get("sources") or {}
    return {"about": f"{IK2_SHOWN} in unsloth's GGUF ({man.get('about', 'UD-Q4_K_XL')}); "
                     f"{man.get('license', 'Apache 2.0')}",
            "format": "ik2-gguf", "repo": man["repo"], "revision": man["revision"], "folder": man.get("folder", ""),
            "files": {k: v[0] for k, v in man["files"].items()}, "sha256": {k: v[1] for k, v in man["files"].items()},
            "sources": {k: (v["repo"], v["revision"]) for k, v in src.items()},
            "source_folders": {k: v.get("folder", "") for k, v in src.items()},
            "gguf": man["gguf"],
            "download_gb": sum(v[0] for v in man["files"].values()) / 1e9}


def ik2_about() -> str:
    """The setup's model row for Inkling-2-Mini."""
    files = read_json(IK2_MANIFEST).get("files") or {}   # (damaged: the setup says so when it is picked)
    gb = f"{sum(v[0] for v in files.values()) / 1e9:.0f} GB" if files else "about 18 GB"
    return ("An EXPERIMENTAL PREVIEW: thinkingmachines' Inkling-2-Mini (26B, a looped model: 8 of 128 experts a token) "
            f"on its own engine, in unsloth's UD-Q4_K_XL GGUF ({gb}); NVIDIA V100 / "
            "RTX 20 and newer with 8 GB or more (experts that do not fit in VRAM stay in RAM: about 13 GB of RAM with an 8 GB card), or "
            "AMD RDNA 3 / 3.5 / 4 or a Halo / APU with 24 GB (every expert in memory); Linux only; tool calls, "
            "its thinking levels, up to a 256K context, pictures (its vision tower, 0.6 GB), drafting on RTX 30 / A100 "
            "and newer with every expert in VRAM, or AMD (its own drafter, 0.6 GB, and the prompt lookup: measured once on this PC, the same answers); "
            "one answer at a time.")


def ik2_dir(a, models: Path, prev: dict) -> Path:
    """The model's folder: --model-dir, else the folder of the setup before, else <models folder>/Inkling-2-Mini."""
    if a.model_dir:
        return Path(a.model_dir).expanduser().resolve()
    d = (prev.get("installer") or {}).get("model_dir")
    if d and Path(d).is_dir() and config_family(prev) == "ik2":
        return Path(d)
    return models / IK2


def ik2_model_step(a, d: Path, models: Path) -> Path | None:
    """The model's GGUF in d, downloaded when it is not there, then checked against its published sha256; None when
    it was not downloaded."""
    step(5, f"the model ({Q.shown('ik2', Q.IK2_QUANT)})")
    m = ik2_entry()
    files = [d / m["gguf"]]
    missing = [f for f in files if file_incomplete(m, f)]
    if missing and not offer_download(a, Q.shown("ik2", Q.IK2_QUANT, m["download_gb"]), d, files, m):
        return None
    for f in files:                                    # (a file that was already there: checked once, then remembered)
        if not sha256_checked(f, m["sha256"][f.name]):
            fail(f"{f.name}: the sha256 does not match the published one - still being downloaded, or damaged",
                 f"let the download finish, or delete the file and run {ME} --download-model")
    ok(f"{Q.shown('ik2', Q.IK2_QUANT)}: {sum(f.stat().st_size for f in files) / 1e9:.1f} GB in {d}")
    if rotational(d):
        warn(f"{d} is on a spinning hard disk: every start copies the whole model from it into VRAM - an NVMe SSD "
             "starts it in seconds (--models-dir)")
    return files[0]


def ik2_tokenizer_step(a, d: Path, gguf: Path) -> Path:
    """The tokenizer from the GGUF with thinkingmachines' own chat template (the GGUF carries the conversion's, which
    differs) in <model folder>/maya-ik2/tokenizer, round-trip checked (tools/strata_tokenizer.py --check); built again
    when its tools or the GGUF changed."""
    step(6, "the tokenizer and the chat template")
    out = d / IK2_DIR
    tools = S.source_hash(IK2_TOKENIZER_TOOLS)
    rec = read_json(out / PACK_SOURCES)
    vocab = out / "tokenizer" / "vocab.json"
    same = rec.get("tools") == tools and rec.get("gguf") == {"name": gguf.name, "size": gguf.stat().st_size}
    if not a.repack and same and vocab.exists() and (out / "tokenizer" / "chat_template.jinja").exists():
        ok(f"tokenizer already exported: {vocab.parent}")
        return out / "tokenizer"
    if not os.access(d, os.W_OK):
        fail(f"{d} is not writable: the tokenizer is written next to the GGUF file", "pass --model-dir with a writable folder")
    r = run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(gguf), "--out", str(out),
             "--chat-template", str(IK2_TEMPLATE), "--check"], check=False)
    if r.returncode != 0 or not vocab.exists():
        fail(f"the tokenizer could not be exported from {gguf.name}", "the reason is in the messages above")
    out.mkdir(parents=True, exist_ok=True)
    # (the files the config runs from, for the Models page's size and its delete: serve/models.py's pack_shards)
    files = [{"name": gguf.name, "size": gguf.stat().st_size}]
    (out / PACK_SOURCES).write_text(json.dumps({"tools": tools, "gguf": files[0], "files": files}, indent=1),
                                    encoding="utf-8")
    ok(f"tokenizer: {vocab.parent} (thinkingmachines' chat template)")
    return out / "tokenizer"


def ik2_vision_model() -> dict | None:
    """The model's vision tower as offer_download's form (tools/ik2/download.json's "vision": unsloth's mmproj of
    Inkling-2-Mini's tower, from the GGUF's repo and revision), or None when the manifest names none."""
    man = read_json(IK2_MANIFEST).get("vision") or {}
    if not man.get("files") or not man.get("repo") or not man.get("revision"):
        return None
    return {"about": f"{IK2_SHOWN}'s vision tower (it lets the model read pictures): unsloth's mmproj of the model's "
                     "own tower (BF16)", "format": "mmproj",
            "repo": man["repo"], "revision": man["revision"], "folder": man.get("folder", ""),
            "files": {k: v[0] for k, v in man["files"].items()}, "sha256": {k: v[1] for k, v in man["files"].items()},
            "download_gb": sum(v[0] for v in man["files"].values()) / 1e9}


def ik2_drafter_model() -> dict | None:
    """The model's drafter as offer_download's form (tools/ik2/download.json's "drafter": thinkingmachines' own
    mtp.safetensors at a pinned revision), or None when the manifest names none."""
    man = read_json(IK2_MANIFEST).get("drafter") or {}
    if not man.get("files") or not man.get("repo") or not man.get("revision"):
        return None
    return {"about": f"{IK2_SHOWN}'s drafter (it drafts tokens the model checks: faster answers, the same ones): "
                     f"thinkingmachines' own ({man.get('license', 'Apache 2.0')})", "format": "safetensors",
            "repo": man["repo"], "revision": man["revision"], "folder": man.get("folder", ""),
            "files": {k: v[0] for k, v in man["files"].items()}, "sha256": {k: v[1] for k, v in man["files"].items()},
            "download_gb": sum(v[0] for v in man["files"].values()) / 1e9}


def ik2_drafting(a) -> str:
    """The engine's --drafting from the setup's: off by hand; on by hand (the drafter and the prompt lookup, not
    measured); else auto - measured once on the PC at the engine's first start (measure, --yes, another model's
    mtp / dfly2: the same as auto)."""
    how = getattr(a, "drafting", None)
    return how if how in ("on", "off") else "auto"


def ik2_drafter_step(a, d: Path, vram: float, ctx: int, kv: str) -> Path | None:
    """The drafter beside the GGUF (in d), downloaded when it is not there and checked against its published sha256,
    or None: drafting off (--drafting off), or not downloaded - the engine then drafts with the prompt lookup alone.
    Truthful about this card: where the drafter's room does not fit beside the model at this context, the engine
    leaves it out at its start (the plan as without it)."""
    how = ik2_drafting(a)
    if how == "off":
        ok("drafting: off (--drafting off) - every token from a full model step")
        return None
    m = ik2_drafter_model()
    if m is None:
        warn(f"{IK2_MANIFEST} names no drafter: drafting with the prompt lookup alone")
        return None
    files = [d / n for n in m["files"]]
    if any(file_incomplete(m, f) for f in files) and not offer_download(a, "the drafter", d, files, m):
        warn(f"the drafter was not downloaded: drafting with the prompt lookup alone ({ME} --setup asks again)")
        return None
    for f in files:
        if not sha256_checked(f, m["sha256"][f.name]):
            f.unlink(missing_ok=True)                  # else the next run finds it complete and fails here again
            fail(f"{f}: the sha256 does not match the published one (the file was removed)", "run the setup again")
    fits = IK2F.draft_fits(vram, ctx, kv)
    ok(f"drafting: {how} - {files[0].name} ({m['download_gb']:.1f} GB) and the prompt lookup; "
       + ("measured once at the engine's first start on this PC (a minute), the faster kept - the answers are the "
          "same with drafting or without" if how == "auto" else "on (--drafting on), not measured"))
    if not fits:
        warn(f"at {context_label(ctx)} the drafter's {IK2F.draft_gib(ctx):.1f} GB does not fit beside the model on this "
             f"card: the engine leaves it out (the prompt lookup still drafts) - a smaller context leaves it room")
    return files[0]


def ik2_vision_step(a, pc: dict, meta: dict, llama: Path, d: Path) -> dict | None:
    """Step 7: the config's "vision" entry - maya-vision with the model's mmproj (beside the GGUF, in d) - or None
    (text only, said why).  No text model in it: the pinned llama.cpp does not open Inkling's GGUF, and maya-vision
    needs none for its pictures (the server writes the template's markers).  The server starts the encoder only while
    a request's pictures are encoded, on the GPU where its free VRAM holds it, else on the CPU; the engine puts the
    rows in place of the image tokens (GENI requests)."""
    step(7, "images (the vision encoder)")
    if a.no_vision:
        ok("skipped (--no-vision): the model reads text only")
        return None
    m = ik2_vision_model()
    if m is None:
        warn(f"{IK2_MANIFEST} names no vision tower: the model reads text only")
        return None
    files = [d / n for n in m["files"]]
    if any(file_incomplete(m, f) for f in files) and not offer_download(a, "the vision tower", d, files, m):
        warn(f"the vision tower was not downloaded: the model reads text only ({ME} --setup asks again)")
        return None
    for f in files:                                    # (offer_download checked the ones it fetched; these too)
        if not sha256_checked(f, m["sha256"][f.name]):
            f.unlink(missing_ok=True)                  # else the next run finds it complete and fails here again
            fail(f"{f}: the sha256 does not match the published one (the file was removed)", "run the setup again")
    if not vision_encoder_ready(a, pc, meta, llama):
        return None
    ok(f"pictures: ON - the vision encoder {VEXE} with {files[0].name} ({m['download_gb']:.1f} GB on the SSD); it "
       "starts only when a picture arrives, no VRAM kept for it")
    return {"exe": str(VEXE), "mmproj": str(files[0]), "gpu": True}


def write_config_ik2(a, pc: dict, meta: dict, gguf: Path, tok: Path, ctx: int, models: Path, d: Path,
                     vision: dict | None = None, drafter: Path | None = None) -> Path:
    """maya-inkling-2-mini.json and its start script.  A setup again keeps what the config before says that this one
    does not ask (its engine flags it names, where the server listens and its key); the model's own sampling and
    thinking level every setup; pictures (`vision`, step 7) when they were set up; drafting (--drafting auto by
    default: the drafter where it was downloaded, and the prompt lookup)."""
    step(8, "the configuration and the start script")
    port = a.port or 8080
    cfg_path = ROOT / f"{Q.config_stem('ik2', Q.IK2_QUANT)}.json"
    old = read_json(cfg_path)
    args = ["--model", str(gguf), "--max-context", str(ctx)]
    how = ik2_drafting(a)
    args += (["--draft", str(drafter)] if drafter and how != "off" else []) + ["--drafting", how]
    kept = keep_args(old.get("args") or [], args, ("--threads", "--eos", "--cache-gb"))
    inst = {"family": "ik2", "quant": Q.OTHERS[("ik2", Q.IK2_QUANT)], "ik2_quant": Q.IK2_QUANT, "drafting": how}
    sampling, effort = Q.model_defaults({"installer": inst})   # (the model's own, every setup)
    cfg = {"exe": str(IK2_EXE), "args": args, "cwd": str(ROOT), "tokenizer": str(tok), "model_name": IK2_MODEL_NAME,
           **({"backend": "hip"} if pc.get("backend") == "hip" else {}),
           "gpu": [g["index"] for g in pc["gpus"]], "sampling": sampling, "reasoning_effort": effort,
           "lib_dirs": meta.get("lib_dirs") or old.get("lib_dirs") or [], "port": port,
           **({"vision": vision} if vision else {})}
    kept += keep_server_settings(cfg, old, a)
    # (its cache in FP8 - half the bytes, the faster one deep in a context; the user's own variables kept)
    env = dict(old.get("env") or {})
    env.setdefault("IK2_KV", IK2F.DEFAULT_KV)
    # (a card that does not hold every expert: the offload path's VRAM tier starts from Maya's profile of the
    #  experts Inkling-2-Mini picks most - data/expert-profile-ik2mini.bin; where every expert fits, none)
    vram = float(pc["gpus"][0].get("vram_gb") or 0) if pc.get("gpus") else 0.0
    seed = str(IK2_PROFILE)
    if IK2F.plan(vram, ctx, env["IK2_KV"])["layout"] == "offload" and IK2_PROFILE.is_file():
        env.setdefault("IK2_WARM", seed)
    elif env.get("IK2_WARM") == seed:
        env.pop("IK2_WARM")
    env.update(parse_env(a.env))
    cfg["env"] = env
    if a.host:
        cfg["host"] = a.host
    if a.api_key:
        cfg["api_key"] = a.api_key
    if kept:
        ok("kept from the config before: " + ", ".join(kept))
    cfg["log"] = str(cfg_path.with_suffix(".log"))
    cfg["installer"] = {**inst, "models_dir": str(models), "model_dir": str(d), "experimental": True,
                        "written": time.strftime("%Y-%m-%d %H:%M")}
    cfg_path.write_text(json.dumps(N.env_out(cfg), indent=1), encoding="utf-8")
    ok(f"config: {cfg_path}")
    ok(f"start script: {write_run_script(cfg_path, cfg['port'])}")
    return cfg_path


def set_up_ik2(a, prev: dict) -> Path | None:
    """Inkling-2-Mini's steps 1-8 (an experimental preview): the PC (refused in one line where the preview does not
    run: a Mac, Windows, two GPUs or more asked for, a card older than RTX 30 / A100 or under 24 GB, an AMD GPU older
    than RDNA 3 or with less memory - an APU counts the RAM it shares, less what is kept free), the context
    (the largest that keeps every expert in VRAM on this card), the engine (build/maya-ik2), the files (pinned,
    sha256-checked), the tokenizer with the model's own template, the pictures (its mmproj and maya-vision), the
    config."""
    inst = prev.get("installer") or {}
    prev_args = prev.get("args") or []
    prev_ctx = int(prev_args[prev_args.index("--max-context") + 1]) if "--max-context" in prev_args[:-1] else None
    a.gpus_before = prev.get("gpu")                    # (the GPU question's default: this config's GPU)
    pc = check_pc(a, "ik2")                            # 1
    vram = float(pc["gpus"][0].get("vram_gb") or 0)
    amd = pc.get("backend") == "hip"
    if amd:   # (AMD, Linux: RDNA 3 / 3.5 / 4 with every expert in memory; an APU's is the RAM, less what is kept free)
        g = dict(pc["gpus"][0])
        if hip_unified_memory(g):
            ram = mem_gb()[0]
            g.update(unified_memory=True, memory_gb=apu_memory(vram, ram, not WIN) - apu_margin_gb(ram))
            vram = g["memory_gb"]
        why = IK2F.unsupported("hip", WIN, len(pc["gpus"]), [g])
        if why:
            fail(why)
    step(2, "your choices")                            # 2
    warn(f"{IK2_SHOWN} is an experimental preview: run by its contributor on RTX 3090 cards (Linux), not yet by "
         "Maya's team on every card - tell us how it goes in an issue" if not amd else
         f"{IK2_SHOWN} on AMD is an experimental preview: run by Maya's team on a Radeon 8065S (Gorgon Halo, "
         "Linux), not yet on Radeon RX 7000 / 9000 cards - tell us how it goes in an issue")
    m = ik2_entry()
    ok(f"model: {Q.shown('ik2', Q.IK2_QUANT, m['download_gb'])} - thinkingmachines' Inkling-2-Mini in unsloth's GGUF "
       "(Apache 2.0)")
    kv = str((prev.get("env") or {}).get("IK2_KV") or IK2F.DEFAULT_KV)
    rec = IK2F.recommended_context(vram, kv)
    ctx = choose_context(a, prev_ctx, IK2F.CONTEXTS, rec)
    fit = IK2F.plan(vram, ctx, kv)
    ok(f"context: {ctx} tokens" + (" (recommended for this card's VRAM)" if ctx == rec else "") +
       f"; its cache: {ctx * IK2F.KV_BYTES.get(kv, IK2F.KV_BYTES['f16']) / 2**30:.1f} GB of VRAM ({kv.upper()})")
    if not fit["resident"]:
        word, line = IK2F.verdict(vram, mem_gb()[0], ctx, kv)
        keep = (f" (a context of {context_label(rec)} keeps everything in VRAM)"
                if IK2F.plan(vram, rec, kv)["resident"] else "")
        warn(f"at {context_label(ctx)} the model and its cache do not all fit in this card's {vram:.0f} GB: "
             f"{line}{keep}" + (" - it may not start" if word == "tight" else ""))
    models = (Path(a.models_dir).expanduser().resolve() if a.models_dir else
              Path(inst["models_dir"]) if inst.get("models_dir") else ROOT.parent / "Maya-data" / "models")
    d = ik2_dir(a, models, prev)
    ok(f"model folder: {d}")
    llama = tools_step(a)                              # 3
    meta = build_step(a, pc, llama, "maya-ik2")        # 4
    if not IK2_EXE.exists():
        fail(f"the engine was not built ({IK2_EXE})", "the reason is in the build's messages above (maya-ik2 needs "
             + ("an RDNA 3 / 3.5 / 4 GPU in the build's list, on Linux)" if amd else
                "a V100 / RTX 20 or newer card in the build's GPU list)"))
    gguf = ik2_model_step(a, d, models)               # 5
    if gguf is None:
        return None
    off = IK2F.drafting_off_reason(None if amd else pc["gpus"][0].get("arch"), vram, ctx, kv, amd)
    if off:                                            # (Volta / Turing, the offload path: no drafter, drafting off)
        ok(off)
        a.drafting = "off"
        drafter = None
    else:
        drafter = ik2_drafter_step(a, d, vram, ctx, kv)   # (5: its drafter)
    tok = ik2_tokenizer_step(a, d, gguf)              # 6
    vision = ik2_vision_step(a, pc, meta, llama, d)   # 7
    return write_config_ik2(a, pc, meta, gguf, tok, ctx, models, d, vision, drafter)   # 8


def set_up_and_serve(a, prev: dict) -> tuple | None:
    """On the setup's screen: the setup, then Maya running on the same screen - its server's output there, Ctrl+C
    stops it.  (The config, the server's exit code or None when it was not started), or None when the model was not
    downloaded."""
    cfg_path = set_up(a, prev)
    if cfg_path is None or a.no_start:
        return cfg_path and (cfg_path, None)
    return cfg_path, serve_on_screen(cfg_path, a)


# ------------------------------------------------------------------------------------------------ main
def maya_version() -> str:
    return (HERE / "VERSION").read_text(encoding="utf-8").strip() if (HERE / "VERSION").exists() else "?"


def model_option(a) -> str | None:
    """--model as typed -> what the setup reads (in place): a.model a GLM download (a MODELS key), DS4 or MM3, and a
    DeepSeek / MiniMax / Hy4-Flash quant's files as --ds4-quant / --mm3-quant / --hy4-quant.  It takes a setup id (glm-maya-s, deepseek-maya-xs,
    minimax-maya-xs ...), a quant's full name (MiniMax-M3.1-Flash-Maya-XS) or any name it took before (Maya-S24,
    EXL3-3.05bpw, DeepSeek-V4.1-Flash-1017 ...; --mm3-quant gguf-xs as maya-xs).  The error for a name it does not
    know, or for a quant --ds4-quant / --mm3-quant names otherwise; else None."""
    if getattr(a, "mm3_quant", None):
        a.mm3_quant = Q.MM3_QUANT_ALIASES.get(a.mm3_quant, a.mm3_quant)
    if not a.model:
        return None
    hit = Q.resolve(a.model, list(glm_downloads()))   # (GLM-5.3's downloads included)
    if hit is None:
        return (f"--model {a.model}: not a model Maya knows - one of " + ", ".join(i for _, _, _, i in Q.QUANTS) +
                f", {DS4}, {MM3}, {HY4}, cyber-frost = {QWEN_SHOWN}, {Q.IK2_IDS[0]} = {IK2_SHOWN} (experimental), or a GLM-5.3-Flash download: " + ", ".join(MODELS))
    family, key = hit
    if family == "glm":
        a.model = key
        return None
    if family == "qwen":                                 # (CYBER-FROST's GGUF, or unsloth's UD-IQ2_XS: --qwen-quant)
        given = getattr(a, "qwen_quant", None)
        if key is not None and given and given != key:
            return (f"--model {a.model} is {Q.shown(family, key)}, but --qwen-quant says {given}: give one of them")
        a.model = QWEN
        if key is not None:
            a.qwen_quant = key
        return None
    if family == "ik2":                                  # (Inkling-2-Mini: its one file set, unsloth's UD-Q4_K_XL)
        a.model = IK2
        return None
    a.model, flag = {"deepseek": (DS4, "ds4_quant"), "hy4": (HY4, "hy4_quant")}.get(family, (MM3, "mm3_quant"))
    if key is not None:
        given = getattr(a, flag, None)
        if given and given != key:
            return (f"--model {Q.setup_id(family, key)} is {Q.shown(family, key)}, but --{flag.replace('_', '-')} "
                    f"says {given}: give one of them")
        setattr(a, flag, key)
    return None


def main() -> int:
    N.alias_env(os.environ)   # (MAYA_* settings under the names the code and the engines read, for all it starts)
    stop_signals()            # (SIGTERM / SIGHUP stop it as Ctrl+C does: the server it runs with it)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cuda", "hip", "metal"],
                    help="GPU backend (HIP: gfx1100/gfx1201/gfx1151 on Linux and Windows; Metal: experimental, "
                         "Apple Silicon Macs, text only; the default on a Mac)")
    ap.add_argument("--setup", action="store_true", help="set up again instead of starting the installed model")
    ap.add_argument("--ablation", choices=["on", "off"],
                    help="GLM, DeepSeek and MiniMax: the model's refusal direction loaded (on - each request still "
                         "asks for it, off by default) or left out (off). Not given: a new config loads it, a setup "
                         "again keeps what its config had")
    ap.add_argument("--check", action="store_true", help="only check this PC and exit")
    ap.add_argument("--no-start", action="store_true", help="set up, but do not start the dashboard")
    ap.add_argument("--yes", action="store_true",
                    help="take the recommended answers (the model download still needs --download-model)")
    ap.add_argument("--plain", action="store_true",
                    help="the setup in plain text, the answers typed (not on the setup's own screen)")
    ap.add_argument("--import-model", metavar="FOLDER",
                    help="use model files downloaded some other way (a download manager, a copy from another PC, a "
                         "Hugging Face cache): finds Maya's published files in FOLDER by size and sha256 (any names, "
                         "any subfolders) and moves them into the models folder, or copies them with --copy when "
                         "they are on another drive")
    ap.add_argument("--copy", action="store_true",
                    help="with --import-model: copy files that cannot be moved (another drive); the originals stay")
    ap.add_argument("--download-model", action="store_true",
                    help="download the model without asking (the commands and the size are still printed)")
    ap.add_argument("--model", metavar="MODEL",
                    help="which model and files (default: asked): a Maya quant - "
                         + ", ".join(i for _, _, _, i in Q.QUANTS) +
                         f" - or {DS4} / {MM3} / {HY4} (their files asked, or --ds4-quant / --mm3-quant / --hy4-quant), "
                         f"cyber-frost = {QWEN_SHOWN}, {Q.IK2_IDS[0]} = {IK2_SHOWN} (an experimental preview),"
                         " or a GLM-5.3-Flash "
                         "download by its name (" + ", ".join(MODELS) + "; every name used before still works)")
    ap.add_argument("--model-dir", help=f"use the files you already have of {DS4}, MiniMax, {HY4} or {QWEN_SHOWN}: their "
                                        "folder (a GGUF's shards, or the EXL3 quant's files and the Engram shards 47 / "
                                        "48 of DeepSeek's repo), or an EXL3 folder of GLM-5.3-Flash or "
                                        "MiniMax-M3.1-Flash at any bit rate (whole by its own index; its own sampling "
                                        "and chat template are used); it must be writable - the pack is written inside "
                                        "it")
    ap.add_argument("--ds4-quant", choices=list(DS4_QUANTS) + [DS4_ABL_ALIAS],
                    help=f"{DS4}'s files: maya-s = Maya's quant (the experts in 2-3 bits; the default once "
                         "published; maya-m: 3.4 bits, once published; maya-xs: ~1.6 bits, with its own "
                         "2-bit drafter: the whole model in a 128 GB PC's memory), gguf-q8 = unsloth's UD-Q8_K_XL GGUF, gguf-q4 = its UD-Q4_K_XL "
                         "(dense weights Q8_0: less VRAM; chosen by hand only, not in the setup's list), gguf-iq1_s / gguf-iq1_m / "
                         "gguf-iq2_xxs / gguf-q2_k_xl / gguf-iq3_xxs / gguf-iq3_s / gguf-q3_k_xl = its smaller UD quants (experts "
                         "1.6-3.5 bits; chosen by hand too), exl3 = LibertAIDAI's EXL3 quant at 2.9 bits, exl3-3.5 = its "
                         "3.5-bit quant (closer to the original; ~280 GB of VRAM + RAM), exl3-2.25 = turboderp's EXL3 "
                         f"quant at 2.25 bits (~{DS4_EXL3['exl3-2.25']['memory_gb']} GB of VRAM + RAM; all three V100 or "
                         "newer, or AMD); "
                         "abliterated-128g / abliterated-256g = orcarouter's abliterated model (its refusals removed) "
                         "as its Mix-128G / Mix-256G GGUF (~114 / ~236 GB of VRAM + RAM, with Maya's Q6_K n-gram "
                         "tables; a gated repo: HF_TOKEN or `hf auth login`), abliterated = the one that fits this "
                         "PC; a Maya "
                         "quant also as --model deepseek-maya-s / -maya-m / -maya-xs")
    ap.add_argument("--mm3-quant", choices=list(MM3_QUANTS) + list(Q.MM3_QUANT_ALIASES),
                    help=f"{MM3}'s files: exl3 = Mia-AiLab's EXL3 quant at 3.5 bits (NVIDIA; pictures, DSpark), "
                         "maya-xs (or gguf-xs) = Maya's MiniMax-M3.1-Flash-Maya-XS (experts ~1.6 bits, ~93 GB: a 128 GB "
                         "PC), maya-s (gguf-s) = MiniMax-M3.1-Flash-Maya-S (~2.1 bits, ~122 GB) - the GGUFs on NVIDIA "
                         "from V100 on and on AMD; with --model-dir: the files you have (the same as --model "
                         "minimax-maya-xs / minimax-maya-s)")
    ap.add_argument("--hy4-quant", choices=list(HY4_QUANTS) + [HY4_Q4],
                    help=f"{HY4}'s files, Maya's own GGUF quants of it (NVIDIA from V100 on, Linux or Windows; AMD on Linux, "
                         "and under Windows experimental): maya-xs = Hy4-Flash-Maya-XS (routed experts in 2 bits), maya-s / maya-m / "
                         "maya-l once they are published; q4_k_m = Tencent's own Q4_K_M from AngelSlim, with MTP "
                         "drafting (--drafting off: without; run on RTX 30 / A100 and newer, Linux); with "
                         "--model-dir: the files you have (the same as --model hy4-flash-maya-xs, hy4-flash-q4_k_m ...)")
    ap.add_argument("--qwen-quant", choices=list(QWEN_QUANTS),
                    help=f"Qwen3.8-Flash-Next's files: q5_k_m = {Q.shown('qwen', 'q5_k_m')} (peasantsmith's GGUF), "
                         f"{QWEN_UD} = Qwen3.8-Flash-Next in unsloth's UD-IQ2_XS GGUF (~81 GB with its vision tower; the same as --model "
                         "qwen-ud-iq2_xs)")
    ap.add_argument("--draft", choices=list(DRAFTS),
                    help="how the model drafts: mtp = its MTP block (the default), dflash2 = a DFlash2 drafter "
                         f"(downloads {DFLASH['bytes'] / 1e9:.1f} GB; NVIDIA RTX 30 / A100 and newer, Apple Silicon, "
                         "and one AMD Ryzen AI Max APU)")
    ap.add_argument("--no-vision", action="store_true", help="text only: no image encoder (saves its build and "
                                                             "its 1.1 GB of files)")
    ap.add_argument("--gguf-dir", help="use GLM-5.3-Flash GGUF files you already have: the folder with all of them, "
                                       "or the .gguf file (the first one of a split model) when the folder holds "
                                       "several models; it must be writable - the pack is written inside it "
                                       "(without this option the setup lists the ones it finds in ~/models*)")
    ap.add_argument("--models-dir", help="where downloaded models go, each in a folder named after it, e.g. "
                                         "<DIR>/Maya-L/ (default: Maya-data/models next to this folder); use "
                                         "a fast NVMe SSD with ~100 GB free")
    ap.add_argument("--gpu", type=int, help="run on this one GPU (CUDA: nvidia-smi; HIP: KFD topology order, on Windows hipInfo's)")
    ap.add_argument("--gpus", help=f"split the model's layers across these GPUs (up to {MAX_GPUS}), e.g. 0,1 or "
                                   "0,1,2,3")
    ap.add_argument("--context", type=int, help=f"context length in tokens (default {DEFAULT_CONTEXT}; a setup flag: "
                                                 "a start of an installed config keeps its own, see --config)")
    ap.add_argument("--parallel", type=int, choices=range(1, 9), metavar="N",
                    help="GLM-5.3-Flash, DeepSeek-V4.1-Flash and MiniMax-M3.1-Flash: answer up to N requests at once "
                         "(1-8, default 1), kept in the config: they are decoded together, more tokens per second in "
                         "all; each extra conversation keeps its own context in VRAM, which the expert cache gives "
                         "up (MiniMax: no KV streaming and no DSpark while N is above 1); Qwen-3.8-Flash-Next and "
                         "Hy4-Flash serve one request at a time")
    ap.add_argument("--force-fit", action="store_true",
                    help="GLM-5.3-Flash: set up a quant even where the setup works out that it cannot start on this "
                         "PC's one NVIDIA card, not even with its experts on the CPU (it then warns instead of "
                         "stopping before the download)")
    ap.add_argument("--vram-reserve-mib", type=int, metavar="N",
                    help="VRAM in MiB the expert pool leaves free on each GPU for the driver, the desktop and other "
                         "programs (default 700; DeepSeek 600, MiniMax 1024, Hy4-Flash 715), kept in the config - given on a start "
                         "too, it is written into the config that starts")
    ap.add_argument("--mmap-experts", choices=["auto", "on", "off"],
                    help="DeepSeek: the experts VRAM does not hold - read in place from the files (auto, the default), "
                         "or off: copied into RAM (with DS4_REQUIRE_ALL_RAM=1 all of them); kept in the config")
    ap.add_argument("--fast-prefill", choices=["on", "off"],
                    help="DeepSeek: fast prompt reading - long prompts read faster (the prompt skip as a per-layer "
                         "cone); answers can shift slightly, so it is off by default; kept in the config")
    ap.add_argument("--mixed-gpus", type=mixed_gpus_value, metavar="on|off|helper|split|LAYERS",
                    help="experimental - DeepSeek or GLM on an AMD GPU with an NVIDIA card beside it (Linux): on - both "
                         "GPUs, in the layout that suits the model (DeepSeek: the card computes the AMD GPU's most-used "
                         "experts, or `split`: it runs the first layers - as many as its VRAM holds, or LAYERS; GLM: "
                         "the card leads, the AMD GPU computes the other experts); off - the AMD GPU alone; not given - "
                         "the first start measures both and keeps the faster (GLM-5.3 744B: only with on); answers can be worded slightly "
                         "differently from one GPU alone; kept in the config")
    ap.add_argument("--port", type=int, help="the dashboard's and the API's port (default 8080)")
    ap.add_argument("--host", help="where the server listens: 127.0.0.1 = this machine only (default), 0.0.0.0 = also "
                                   "other devices on your network (set --api-key too)")
    ap.add_argument("--api-key", help="require this key from API clients and the dashboard")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="an engine setting kept in the config, e.g. MAYA_GLM_RAM_HEADROOM_GB=4 (docs/SETTINGS.md)")
    ap.add_argument("--nvcc", help="the CUDA toolkit's nvcc to build with (default: the newest that fits your GPUs)")
    ap.add_argument("--host-compiler", help="the C++ compiler nvcc uses, e.g. g++-12 (when the default g++ is newer "
                                            "than your CUDA accepts)")
    ap.add_argument("--llama-dir", help="a llama.cpp checkout at the pinned commit, instead of downloading its source")
    ap.add_argument("--rebuild", action="store_true", help="compile the engine again")
    ap.add_argument("--repack", action="store_true", help="build the pack again")
    ap.add_argument("--report", action="store_true", help="write maya-report.txt - this PC, the setup and the engine's "
                                                          "speed lines - to attach when you report a problem or a "
                                                          "speed (nothing is sent anywhere)")
    ap.add_argument("--bench", action="store_true", help="a standard speed test of the installed model (a few "
                                                         "minutes, with Maya stopped): decode on three questions and "
                                                         "prefill at 2k / 8k tokens; writes maya-bench.txt")
    ap.add_argument("--calibrate", action="store_true",
                    help="tune the engine's CPU lane for this PC (the PCIe share and the CPU threads, ~10-15 minutes; "
                         f"{DS4}: the tensor parallel groups and the CPU threads, an engine start each, ~25-40 "
                         "minutes on 9 GPUs), then start the model (with --no-start: only tune)")
    ap.add_argument("--speed-calibration", action="store_true",
                    help="measure on this PC whether the drafter (DFlash2, or DeepSeek's DSpark) decodes faster than "
                         "without it (sustained speed, ~5-30 minutes, the model loaded twice) and keep the faster; "
                         "then start the model "
                         "(with --no-start: only measure)")
    ap.add_argument("--drafting", choices=("measure", "on", "off", "mtp", "dfly2"),
                    help="the drafter (DFlash2 with --draft dflash2, DeepSeek's DSpark): measure = the speed "
                         "calibration keeps it only if it is faster on this PC (the default with --yes; asked "
                         "otherwise), on = draft with it, off = without it - both skip the measuring.  Hy4-Flash's "
                         "sizes: on = the MTP block where it fits in the VRAM, else the size's DFly2 drafter; mtp / "
                         "dfly2 = that one alone (another model: the same as on)")
    ap.add_argument("--fast-drafting", choices=["on", "off"],
                    help="Hy4-Flash's sizes on a small card (a setup setting, kept in the config): where the DFly2 "
                         "drafter has no room beside the cache's F16 rows at the context chosen, on keeps that context "
                         "with the cache in 8-bit rows so the drafter fits - faster drafting (an 8 GB card at 8K: "
                         "decoding about +8-27%%), slightly different wording than the default; off = the default "
                         "(F16 rows, Maya's default answers). Not given: asked where it applies (--yes: the default)")
    ap.add_argument("--no-speed-calibration", action="store_true",
                    help="the same as --drafting on: draft with the drafter without measuring it")
    ap.add_argument("--config", type=Path, help="the installed JSON config to start (its run-maya-<model>.sh does), or "
                                              "for --bench or --report (default: most recently used). Starting an "
                                              "installed config uses the settings saved in it: setup flags such as "
                                              "--context or --model are ignored then (only the start's own apply: "
                                              "--port, --host, --api-key, --gpus, and the few written into the config, "
                                              "such as --parallel); add --setup to set up again with them")
    a = ap.parse_args()
    why = model_option(a)
    if why:
        ap.error(why)
    if a.vram_reserve_mib is not None and a.vram_reserve_mib < 0:
        ap.error("--vram-reserve-mib takes a number of MiB, 0 or more, e.g. --vram-reserve-mib 2048")
    if a.config is not None:
        if not a.config.is_file():
            ap.error(f"config not found: {a.config}")
        if a.backend and read_json(a.config).get("backend", "cuda") != a.backend:
            ap.error(f"{a.config}: config backend does not match --backend {a.backend}")
    if not WIN and sys.prefix == sys.base_prefix and not os.environ.get("MAYA_SH"):
        # started as `python3 maya.py`: a system Python takes no pip installs (PEP 668, "externally-managed-
        # environment") - maya.sh makes the private .venv and runs this file with its Python
        say("Starting through ./maya.sh, which runs Maya with its own Python environment (.venv) ...")
        sys.stdout.flush()
        os.execve("/bin/sh", ["/bin/sh", str(HERE / "maya.sh"), *sys.argv[1:]], dict(os.environ, MAYA_SH="1"))
    version = maya_version()
    say(f"Project Maya v{version} - GLM-5.3-Flash, {DS4}, {MM3} or {HY4} on your own GPU(s).")
    if a.report:
        return report(version, a.backend, a.config)
    if a.import_model:
        return import_model(a, a.import_model)
    if a.bench:
        have = [a.config] if a.config is not None else configs()
        if a.backend:
            have = [p for p in have if read_json(p).get("backend", "cuda") == a.backend]
        if not have:
            fail("Maya is not set up here yet", f"run {ME} first")
        return bench(have[0], version)

    have = configs() if a.config is None else [a.config.resolve()]   # (--config: that one, no question)
    if a.backend:
        have = [p for p in have if read_json(p).get("backend", "cuda") == a.backend]
    elif have:
        a.backend = read_json(have[0]).get("backend") if read_json(have[0]).get("backend") in ("hip", "metal") else "cuda"
    else:
        # nothing set up yet (START-MAYA.bat / ./maya.sh without --backend): Metal on a Mac with Apple Silicon, HIP on
        # a PC with a supported AMD GPU and no NVIDIA one (e.g. Strix Halo)
        a.backend = first_run_backend()
    setting_up = a.setup or a.check or a.gguf_dir or a.model or a.model_dir or a.rebuild or a.repack or \
        a.download_model
    if have and not setting_up and (a.calibrate or a.speed_calibration or not a.no_start):
        pick = have[0]
        if len(have) > 1:
            say()
            for i, c in enumerate(have, 1):
                say(f"  {i}) {c.name}")
            pick = have[int(ask("Which one?", [str(i) for i in range(1, len(have) + 1)], "1", a.yes)) - 1]
        check_parallel(a, config_family(read_json(pick)))
        if getattr(a, "fast_drafting", None):   # (a setup setting: the cache's rows are worked out with the context)
            warn(f"--fast-drafting is a setup setting of Hy4-Flash: {ME} --setup --fast-drafting {a.fast_drafting} "
                 "sets it up; this start keeps the config as it is")
        if getattr(a, "ablation", None):       # (a setup setting, kept in the config)
            warn(f"--ablation is a setup setting: {ME} --setup --ablation {a.ablation} sets it up; this start keeps "
                 "the config as it is")
        if getattr(a, "fast_prefill", None):   # (given on a start too: written into a DeepSeek config, which keeps it)
            cfg = read_json(pick)
            if config_family(cfg) != "deepseek":
                warn("--fast-prefill is a DeepSeek setting: this config does not use it")
            else:
                set_fast_prefill(cfg, a.fast_prefill)
                pick.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if getattr(a, "mixed_gpus", None):   # (given on a start too: written into the config, which keeps it)
            cfg = read_json(pick)
            if mixed_family(cfg) is None:
                warn("--mixed-gpus is a DeepSeek and GLM setting: this config does not use it")
            else:
                set_mixed_gpus(cfg, a.mixed_gpus)
                pick.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
        if a.mmap_experts:   # (#422: given on a start too - written into a DeepSeek config, which keeps it)
            cfg = read_json(pick)
            if config_family(cfg) != "deepseek":
                warn("--mmap-experts is a DeepSeek setting: this GLM-5.3-Flash config does not use it")
            else:
                set_arg(cfg.setdefault("args", []), "--mmap-experts", a.mmap_experts)
                pick.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
                ok(f"{pick.name}: --mmap-experts {a.mmap_experts}")
        if a.vram_reserve_mib is not None:   # (#1026: given on a start too - written into the config, which keeps it)
            cfg = read_json(pick)
            put = set_vram_reserve(cfg.setdefault("args", []), config_family(cfg), a.vram_reserve_mib,
                                   cfg.get("backend"))
            if put is None:
                warn("--vram-reserve-mib: this model's engine has no VRAM reserve setting - the start keeps the config "
                     "as it is")
            else:
                pick.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
                ok(f"{pick.name}: VRAM kept free on each GPU {a.vram_reserve_mib} MiB ({put})")
        if (a.calibrate or a.speed_calibration) and read_json(pick).get("backend") == "metal":
            ok("the Metal engine has no CPU lane to tune and no drafter to measure: nothing to tune")
            a.calibrate = a.speed_calibration = False
            if a.no_start:
                return 0
        # (DeepSeek too: --speed-calibration measures its DSpark drafter against none - speed_calibration_ds4 - and keeps
        #  the faster; it used to stop here with "nothing to tune", so a config the setup wrote with --dspark none until
        #  that measurement could never get it - #393.  --calibrate tunes its --tp and CPU threads)
        if a.calibrate:
            refresh_engine(read_json(pick))            # (the engine the tuning measures is the current one)
            if not calibrate_config(pick):
                say()
                warn("this PC is NOT tuned (the reason is above): the model " +
                     ("keeps" if a.no_start else "starts with") + " the engine's own settings")
            if a.no_start:
                return 0
        if a.speed_calibration:
            refresh_engine(read_json(pick))            # (the engine it measures is the current one)
            speed_calibration(pick, "asked for")
            if mixed_why_not(read_json(pick)) is None and \
                    read_json(pick).get("mixed_gpus") is not False:   # (the user's --mixed-gpus off stays)
                link_calibration(pick, "asked for")
            if a.no_start:
                return 0
        return start(pick, a)
    if not have and a.calibrate:
        say("Nothing is set up yet: the setup runs first, then the tuning.")
    prev = read_json(have[0]) if have else {}
    if a.check:
        hy4 = a.model == HY4 or bool(getattr(a, "hy4_quant", None))
        ik2 = a.model == IK2                           # (Inkling-2-Mini's preview: its refusals, or this PC's card)
        ds4 = not hy4 and not ik2 and (a.model == DS4 or bool(a.model_dir) and exl3_folder_family(a.model_dir) is None)
        quant = choose_ds4_quant(a, prev) if ds4 else None
        pc = check_pc(a, "hy4" if hy4 else "ik2" if ik2 else "deepseek" if ds4 else "glm",
                      gguf=ds4 and quant != "exl3")
        if hy4:                                        # (the layout and the context that fit this VRAM, or why none does)
            quant = choose_hy4_quant(a, prev if config_family(prev) == "hy4" else {})
            if quant != HY4_Q4:                        # (Tencent's own Q4_K_M: the table has no VRAM plan for it)
                hy4_fit_step(a, pc, quant)
        say()
        if a.backend == "metal":
            say(f"This Mac can run Maya on Metal (experimental). Run {ME} without --check to set it up.")
        elif a.backend == "hip":
            pick = (f"--gpus {','.join(str(g['index']) for g in pc['gpus'])}" if len(pc["gpus"]) > 1
                    else f"--gpu {pc['gpus'][0]['index']}")
            say(f"HIP prerequisites found. Run {ME} --backend hip {pick} to set it up.")
        else:
            say(f"This PC can run Maya. Run {ME} without --check to set it up.")
        return 0
    rc = None                                          # the server's exit code, when it ran on the setup's screen
    if setup_screen(a):
        import setup_tui
        cfg_path, rc = setup_tui.run(lambda: set_up_and_serve(a, prev), version, ROOT / "maya-setup.log",
                                     ME) or (None, None)
    else:
        cfg_path = set_up(a, prev)
    if cfg_path is None:                               # the model was not downloaded (what to do next is above)
        return 0
    if rc is not None:
        return after_server(rc, a)
    if a.no_start:
        say()
        say(f"All set. Start it with {ME} (or " + (f"run-{cfg_path.stem}.bat" if WIN else f"./run-{cfg_path.stem}.sh")
            + ").")
        return 0
    return start(cfg_path, a)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("\nstopped.")
        sys.exit(1)

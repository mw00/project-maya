"""serve/glm_dsa_product.py - GLM-5.3 (744B-A40B, GGUF arch glm-dsa) on Maya's GLM engine: what tells a glm-dsa pack
from a GLM-5.3-Flash one, its names and its memory figures.

GLM-5.3 runs on the same engine, server and dashboard as GLM-5.3-Flash under Maya's own name (released in v1.1.57).
"""
from __future__ import annotations

import re
from pathlib import Path

ARCH = "glm-dsa"                        # its general.architecture (llama.cpp's name)
MODEL_NAME = "glm-5.3"                  # its API name (a config's model_name)
SHOWN = "GLM-5.3"
LABEL = "GLM-5.3 (744B)"                # on the dashboard, before the quant: "GLM-5.3 (744B) · UD-IQ2_M" (the owner,
#                                         2026-11-06) - GLM-5.3-Flash keeps "GLM-5.3 · Maya-S24"
ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = ROOT / "data" / "glm-dsa" / "templates" / "glm-5.3.official.jinja"   # Z.ai's chat template, byte for byte
# GLM-5.3's own expert usage profile (tools/strata/glm_dsa_profile.py, made on a GPU day; 79 layers x 256 experts):
# the setup passes it to a glm-dsa config when the pack has no profile of its own (tools/strata/maya_sync.profile_for)
PROFILE = ROOT / "data" / "expert-profile-glm53.bin"
_HEADER = re.compile(r"#\s*glm-dsa:")   # the line tools/strata/glm_dsa_pack.py adds to a pack's index.txt
_DSA = re.compile(r"#\s*glm-dsa:\s*trunk\s+(\d+),.*?\bmtp\s+(\w+),\s*indexer full\s+([\d,]*)")

# What the engine serves on glm-dsa today, for the setup and the Models page (each True once the engine has it):
# KV streaming (--kv-streaming on / --kv-resident N) - glm_state_carve refuses a resident window smaller than the
# context on glm-dsa (HANDOFF "Remaining work" 3a), so a GLM-5.3 config keeps its whole cache in VRAM: the setup
# writes --kv-streaming off (with on, every context above 32768 would not start) and the Models page neither shows nor
# takes it
KV_STREAMING = False
# --parallel's conversations decoded together (batched rows, 3b): on glm-dsa they take turns (the pipelined path)
BATCHED_ROWS = False
# Memory guidance (neutral, no speeds): from about this much VRAM + RAM together every expert of unsloth's UD-IQ2_M
# (238.6 GB) stays in memory; with less the engine reads the rest from the SSD while it answers
MEMORY_GB = 245
# The setup's context choice while the whole cache stays in VRAM (KV_STREAMING False), on discrete GPUs: a context's
# cache - one a conversation (--parallel) - (KV_BYTES a token: GLM-5.3's INT8 latent + FP32 index keys, the setup's
# default; kv_bytes_of(GEOMETRY, ...) for the formats an --env asks for - kv_bytes_per_token of its pack says the same),
# the dense weights (DENSE_GB: 4.90 + 4.83 + 4.64 GB measured on 3x V100 with UD-IQ2_M, GPU day 1) and on each GPU its
# working memory (GPU_WORK_GB: activations, MTP, scratch, the prompt path ~0.7-0.9 GB) and the VRAM the engine leaves
# free beside the expert pool (the reserve: --vram-reserve-mib, RESERVE_MIB by default; the HIP setup's
# STRATA_GLM_RESERVE_MB) must leave a tenth of the VRAM to the experts
GEOMETRY = {"trunk": 78, "mtp": True, "full": 21, "kv_lora": 512, "kv_row": 576, "idx_key": 128}   # (geometry()'s)
KV_BYTES = 64352
DENSE_GB = 14.5
GPU_WORK_GB = 0.8
RESERVE_MIB = 700
# --calibrate (tools/calibrate_glm.py: one engine run, ~20 groups of three 128-token answers) on GLM-5.3: its load takes
# a few minutes and its answers come slower than GLM-5.3-Flash's
CALIBRATE_MINUTES = "30-60"


def pack_is_glm_dsa(pack) -> bool:
    """`pack` (a --glm-pack folder) was built from a GLM-5.3 (glm-dsa) GGUF: its index.txt says so in a header line.
    False for any other pack, or none."""
    if not pack:
        return False
    try:
        with open(Path(pack) / "index.txt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.startswith("#"):
                    return False
                if _HEADER.match(line):
                    return True
    except OSError:
        return False
    return False


def config_pack(cfg: dict) -> Path | None:
    """The --glm-pack of a run config's engine arguments, or None."""
    args = cfg.get("args") if isinstance(cfg, dict) else None
    if isinstance(args, list) and "--glm-pack" in args[:-1]:
        return Path(str(args[args.index("--glm-pack") + 1]))
    return None


def config_is_glm_dsa(cfg: dict) -> bool:
    return pack_is_glm_dsa(config_pack(cfg))


def geometry(pack) -> dict | None:
    """The attention caches' shape of a glm-dsa pack, from its index.txt: the header's trunk, NextN block and full-
    indexer layers, and the tensor rows' widths - kv_lora (attn_kv_a_norm), the latent row kv_lora + rope (attn_kv_a_mqa's
    output) and the index key (indexer.attn_k's output).  None for any other pack, or one that does not read."""
    want = {"blk.0.attn_kv_a_norm.weight": "kv_lora", "blk.0.attn_kv_a_mqa.weight": "kv_row",
            "blk.0.indexer.attn_k.weight": "idx_key"}
    got: dict = {}
    try:
        with open(Path(pack) / "index.txt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("#"):
                    m = _DSA.match(line)
                    if m:
                        got["trunk"], got["mtp"] = int(m.group(1)), m.group(2) != "none"
                        got["full"] = len([x for x in m.group(3).split(",") if x])
                    continue
                name, _, rest = line.partition(" ")
                if name in want:                       # (name, then 17 fields: ne0 the 7th, ne1 the 8th)
                    f17 = rest.split()
                    got[want[name]] = int(f17[6]) if want[name] == "kv_lora" else int(f17[7])
                    if len(got) == 6:
                        break
    except (OSError, TypeError, ValueError, IndexError):
        return None
    if len(got) != 6 or not 0 < got["kv_lora"] < got["kv_row"] or got["idx_key"] <= 0:
        return None
    return got


def kv_bytes_per_token(pack, int8: bool = True, idx_fp16: bool = False) -> int | None:
    """What one context position of a glm-dsa model takes in its caches, as the engine lays them out
    (src/core/glm_state_layout.cpp): every layer and the NextN block its latent - INT8 (the default: kv_lora codes and
    an FP16 scale per 32) or FP16 - and its rope part k_pe (FP16); every full-indexer layer and the NextN block its index
    key (FP32; FP16 with STRATA_GLM_IDX_FP16=1).  GLM-5.3: 79 x 672 + 22 x 512 = 64,352 bytes (FP16 latent: 102,272).
    None for any other pack."""
    g = geometry(pack)
    return None if g is None else kv_bytes_of(g, int8, idx_fp16)


def kv_bytes_of(g: dict, int8: bool = True, idx_fp16: bool = False) -> int:
    """kv_bytes_per_token's count for a geometry() (GEOMETRY: GLM-5.3's own, before its pack is built)."""
    lat = g["kv_lora"] + g["kv_lora"] // 16 if int8 else 2 * g["kv_lora"]
    layers, full = g["trunk"] + g["mtp"], g["full"] + g["mtp"]
    return layers * (lat + 2 * (g["kv_row"] - g["kv_lora"])) + full * g["idx_key"] * (2 if idx_fp16 else 4)

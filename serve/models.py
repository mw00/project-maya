"""serve/models.py - the web app's Models page: the run configs the setup wrote (maya-<model>.json beside maya.py), what
each one runs with, the engine settings the page changes in them, and a model deleted (delete_plan, delete).

A config's settings, as the page shows them (None: the engine's own default, or "auto"):
  both:      context, images, gpus, reserve_mib, env
  GLM:       parallel, draft ("mtp" | "dflash2"), expert_cache, prefill, kv_streaming, kv_resident
  DeepSeek:  parallel, draft ("dspark" | "none"), tp, mmap ("auto" | "on" | "off"), prefill_chunk, cpu_threads
  MiniMax:   parallel, draft ("dspark" | "none"), tp (0: the layers split, no groups), kv_cache, kv_streaming,
             kv_resident, cpu_threads (maya-mm3 stops at a flag it does not know: only its own are written)
  CYBER-FROST (the `maya` engine's Qwen path; its server always loads the MTP drafter - "none" is --mtp-max-t 1):
             draft ("mtp" | "none"), expert_cache, prefill, kv ("fp16" | "int8" |
             "k8v4" | "q4_0"), ram_gb (the experts' RAM tier, MAYA_EXPERT_RAM_GB); no images (text only)
  Hy4-flash (maya-hy4: it stops at a flag it does not know - only its own are written): draft ("mtp" | "dfly2" |
             "none": the drafters the pack names), tp ("auto" | 0: the layers split | a group's size), reserve_gb,
             prefill_chunk, cpu_threads, kv ("auto" | "q8": the latent cache's rows, --kv - 8-bit rows are the setup's
             opt-in faster drafting on a small card, "f16" kept where a config has it); images (the pack's mmproj
             file: --vision off when off); one answer at a time
apply() writes them back into the config's "args", "env", "gpu" and "vision" - its other keys as they are."""
from __future__ import annotations

import copy
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import maya_names  # noqa: E402  (tools/) a setting under either of its names
import maya_quants  # noqa: E402  (tools/) the quants' names users see, and each config's family
from serve import glm_dsa_product as G53  # noqa: E402  GLM-5.3 (744B, glm-dsa): what its engine serves today

SWITCH_FILE = ".maya-switch.json"           # the server's note to maya.py: start this config next (a switch)
DELETE_FILE = ".maya-delete.json"           # ... and to the next server: delete this config (it was the one running)
CONTEXTS = [8192, 32768, 65536, 131072]       # the setup's choices (a config's own size is offered too)
DS4_CONTEXTS = [32768, 65536, 131072, 262144, 524288, 1048576]   # ... for DeepSeek (maya.py's DS4_CONTEXTS)
MM3_CONTEXTS = DS4_CONTEXTS                                       # ... and MiniMax (maya.py's MM3_CONTEXTS)
QWEN_CONTEXTS = [32768, 65536, 131072, 262144]                    # ... and CYBER-FROST (maya.py's QWEN_CONTEXTS)
HY4_CONTEXTS = [4096, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]          # ... and Hy4-Flash (maya.py's HY4_CONTEXTS)
CONTEXT_MIN = 4096          # the dashboard's Context size: the smallest it offers ...
CONTEXT_FALLBACK_MAX = 262144   # ... and the largest when the model file does not say what it was trained for
# the largest context the dashboard offers without an "untested" warning, by engine: maya-ds4 its model's own 1M; the
# GLM engine's 256K, and maya-mm3's 256K (MiniMax-M3.1 is offered up to its 1M; past 256K it has not been run end to
# end yet)
# (maya-hy4: Tencent's own Q4_K_M run end to end up to 256K; Maya's sizes up to 32K so far - the model is trained for 1M)
CONTEXT_TESTED = {"ds4": 1048576, "mm3": 262144, "hy4": 32768, "hy4-q4": 262144}
CONTEXT_TESTED_DEFAULT = 262144
QWEN_KV =("fp16", "int8", "k8v4", "q4_0")                        # the Qwen path's --kv
MM3_KV = ("bf16", "int8", "fp8", "fp4")                           # maya-mm3 --kv-cache (one format for K and V)
# files the setup's quant list (tools/maya_quants.py) does not name, by their config's installer key
OTHER_QUANTS = {("deepseek", "abliterated"): "Mix-256G"}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
# variables that change which code the engine's process loads or runs (a library preloaded, the program search path):
# never set or changed from the page - with remote access on, the page is a network client; a config's own are kept
SECRET = re.compile(r"key|token|secret|password", re.I)   # (the server's report: the same names left out)
MASK = "***"
ENV_LOADER = re.compile(r"^(?:LD_\w*|DYLD_\w*|PYTHON\w*|PATH|PATHEXT|BASH_ENV|ENV|IFS|COMSPEC|SYSTEMROOT)$", re.I)
GLM_KEYS = ("context", "images", "gpus", "reserve_mib", "env", "parallel", "draft", "expert_cache", "prefill",
            "kv_streaming", "kv_resident")
DS4_KEYS = ("context", "images", "gpus", "reserve_mib", "env", "parallel", "draft", "tp", "mmap", "prefill_chunk",
            "cpu_threads", "prompt_lookup", "dspark_conf", "fast_prefill")
MM3_KEYS = ("context", "images", "gpus", "env", "parallel", "draft", "tp", "kv_cache", "kv_streaming", "kv_resident",
            "cpu_threads", "reserve_gb", "prefill_chunk", "cache_gb", "dspark_conf")
QWEN_KEYS = ("context", "images", "gpus", "reserve_mib", "env", "draft", "expert_cache", "prefill", "kv", "ram_gb")
HY4_KEYS = ("context", "images", "gpus", "env", "draft", "tp", "reserve_gb", "prefill_chunk", "cpu_threads", "kv")
HY4_KV_BYTES = 13 * (2 * (512 + 64) + 128 * 4 // 4)   # (maya.py's: its cache a context token, on each GPU of a group)
HY4_Q4_CONTEXTS = [32768, 65536, 131072, 262144]   # AngelSlim's Q4_K_M: the setup's choices for it (maya.py's HY4_Q4_CONTEXTS)
# (... and its longer ones, up to the model's 1M, where the cache in the engine's smaller rows is at most this share of
#  the smallest card's VRAM - maya.py's HY4_Q4_LONG, HY4_Q4_LONG_SHARE, hy4_q4_contexts)
HY4_Q4_LONG, HY4_Q4_LONG_SHARE = [524288, 1048576], 0.25
HY4_KV_Q8_BYTES = 13 * (512 + 512 // 32 * 2 + 64 * 2 + 128 * 2 // 4)
# (KV streaming: the RAM a context token takes - the 13 global layers' rows, F16 or 8-bit - maya.py's hy4_stream_ram_gib)
HY4_STREAM_RAM_BYTES = {"f16": 13 * (512 + 64) * 2, "q8": 13 * (512 + 512 // 32 * 2 + 64 * 2)}
# what maya-hy4 does not have yet, said in plain words when a client asks for it (apply; each one's "off" - HY4_OFF,
# or nothing - is taken as it is: the page sends back what it was shown)
HY4_OFF = {"parallel": 1, "kv_streaming": False}
HY4_NOT_YET = {"parallel": "Answers at once: Hy4-Flash answers one request at a time for now",
               "kv_cache": "KV cache: Hy4-Flash's cache rows are auto or 8-bit (its setting kv)",
               "kv_streaming": "KV streaming: Hy4-Flash's engine turns it on by itself where the context needs it",
               "kv_resident": "KV streaming: Hy4-Flash's engine turns it on by itself where the context needs it"}


# ------------------------------------------------------------------ the engine's arguments
def flag(args: list, name: str, default=None):
    """The value after `name` in the arguments, else `default`."""
    args = [str(a) for a in args]
    return args[args.index(name) + 1] if name in args[:-1] else default


def set_flag(args: list, name: str, value) -> list:
    """The arguments with `name value` (replaced where it was, else added at the end)."""
    out = [str(a) for a in args]
    if name in out[:-1]:
        out[out.index(name) + 1] = str(value)
    else:
        out += [name, str(value)]
    return out


def drop_flag(args: list, name: str) -> list:
    out, i, args = [], 0, [str(a) for a in args]
    while i < len(args):
        if args[i] == name and i + 1 < len(args):
            i += 2
            continue
        out.append(args[i])
        i += 1
    return out


def _int(v, default=None):
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _float(v, default=None):
    try:
        f = float(str(v).strip())
    except (TypeError, ValueError):
        return default
    return f if f == f else default                    # (not NaN)


def family(cfg: dict) -> str:
    """"minimax", "deepseek", "qwen", "hy4" or "glm" (maya_quants.family_of, as maya.py and the server read it)."""
    return maya_quants.family_of(cfg)


def gpus_of(cfg: dict) -> list:
    g = cfg.get("gpu")
    if isinstance(g, list):
        return [int(x) for x in g if str(x).strip().isdigit()]
    if isinstance(g, int):
        return [g]
    if isinstance(g, str):
        return [int(x) for x in g.split(",") if x.strip().isdigit()]
    return []


def pack_of(cfg: dict) -> Path | None:
    args = cfg.get("args") or []
    p = flag(args, "--glm-pack") or flag(args, "--pack")
    return Path(p) if p else None


# ------------------------------------------------------------------ the context a model was trained for
def trained_context(model_dir) -> int | None:
    """The context the model was trained for (`<arch>.context_length` in its first GGUF's header; GLM-5.3-Flash:
    1,048,576): the top of the dashboard's Context size.  None when there is no readable GGUF."""
    try:
        from gguf_reader import GGUFFile                # (tools/, on the path above)
        shards = sorted(Path(model_dir).glob("*-00001-of-*.gguf")) or sorted(Path(model_dir).glob("*.gguf"))
        if not shards:
            return None
        md = GGUFFile(shards[0]).metadata
        n = md.get(f"{md.get('general.architecture', '')}.context_length")
        return int(n) if isinstance(n, int) and n > 0 else None
    except Exception:  # noqa: BLE001 - a header it can't read: the fallback range
        return None


def trained_context_ds4(pack) -> int | None:
    """DeepSeek's, MiniMax's and Hy4-flash's: the context its pack says the model was trained for (ds4.txt's / mm3.txt's
    max_position_embeddings, from the model's config.json or a GGUF's context_length; hy4.txt's context_length;
    V4.1-Flash, M3.1-Flash and Hy4-flash: 1,048,576, their native reach), the top of the dashboard's Context size.
    None when the pack does not say."""
    for name, key in (("ds4.txt", "max_position_embeddings"), ("mm3.txt", "max_position_embeddings"),
                      ("hy4.txt", "context_length")):
        try:
            with open(Path(pack) / name, encoding="utf-8", errors="replace") as f:
                for line in f:
                    k, _, v = line.strip().partition(" ")
                    if k == key:
                        n = int(v.split()[0])
                        return n if n > 0 else None
        except (OSError, ValueError, IndexError):
            continue
    return None


_trained: dict = {}


def trained_context_of(cfg: dict) -> int | None:
    """A config's model's trained context, as the server reads it when it runs that config (GLM: its GGUF header
    beside --glm-pack; DeepSeek and MiniMax: --pack's own file); kept once found (a header read is not free)."""
    args = [str(x) for x in cfg.get("args") or []]
    if "--glm-pack" in args[:-1]:
        key, read_it = ("gguf", str(Path(args[args.index("--glm-pack") + 1]).resolve().parent)), trained_context
    elif "--pack" in args[:-1]:
        key, read_it = ("pack", args[args.index("--pack") + 1]), trained_context_ds4
    else:
        return None
    if key not in _trained:
        n = read_it(key[1])
        if n is None:
            return None                                 # (not kept: the files may still be on their way)
        _trained[key] = n
    return _trained[key]


def context_range(trained: int | None, current: int, engine_kind: str = "") -> dict:
    """The dashboard's Context size slider for one model (Settings > Model for the running one, the Models page's
    Engine settings for each): CONTEXT_MIN up to what the model was trained for (else CONTEXT_FALLBACK_MAX), never
    below the size it has now; `tested`: past it the page says the size is untested on this engine."""
    return {"min": CONTEXT_MIN, "max": max(trained or CONTEXT_FALLBACK_MAX, int(current or 0)), "trained": trained,
            "tested": CONTEXT_TESTED.get(engine_kind, CONTEXT_TESTED_DEFAULT)}


# ------------------------------------------------------------------ what a config can be switched to
def dflash_file(cfg: dict) -> Path | None:
    """GLM's DFlash2 drafter: the one the config drafts with, else one a setup downloaded into its models folder."""
    p = flag(cfg.get("args") or [], "--dflash")
    if p:
        return Path(p)
    models = (cfg.get("installer") or {}).get("models_dir")
    d = Path(models) / "GLM-5.3-Flash-DFlash2" if models else None
    try:
        found = sorted(d.glob("*.gguf")) if d is not None and d.is_dir() else []
    except OSError:
        found = []
    return found[0] if found else None


def ds4_has_drafter(cfg: dict) -> bool:
    """DeepSeek's pack holds the DSpark drafter (tools/calibrate_ds4.py's has_drafter)."""
    pack = pack_of(cfg)
    t = pack / "tensors.txt" if pack else None
    try:
        with t.open(encoding="utf-8", errors="replace") as f:
            return any(line.startswith("t mtp.") for line in f)
    except (OSError, AttributeError):
        return False


def ds4_has_vision(cfg: dict) -> bool:
    pack = pack_of(cfg)
    try:
        return any(line.startswith("vision.num_hidden_layers ")
                   for line in (pack / "ds4.txt").read_text(encoding="utf-8", errors="replace").splitlines())
    except (OSError, TypeError):
        return False


def qwen_mtp(cfg: dict) -> Path | None:
    """CYBER-FROST's MTP drafter: the one the config drafts with, else the one its setup built in the pack (<pack>/mtp,
    tools/mtp_from_gguf.py), or None."""
    p = flag(cfg.get("args") or [], "--mtp")
    pack = pack_of(cfg)
    for d in ([Path(p)] if p else []) + ([pack / "mtp"] if pack else []):
        if (d / "dense.txt").is_file():
            return d
    return None


def hy4_has_mtp(cfg: dict) -> bool:
    """Hy4-flash's pack names the model's MTP block (hy4.txt: n_mtp 1) - its drafter."""
    pack = pack_of(cfg)
    try:
        return any(line.split()[:2] == ["n_mtp", "1"]
                   for line in (pack / "hy4.txt").read_text(encoding="utf-8", errors="replace").splitlines())
    except (OSError, TypeError):
        return False


def hy4_has_dfly2(cfg: dict) -> bool:
    """Hy4-flash's pack names a DFly2 drafter (hy4.txt: dfly_layers) - its other drafter (--dfly2)."""
    pack = pack_of(cfg)
    try:
        return any(line.split()[:1] == ["dfly_layers"]
                   for line in (pack / "hy4.txt").read_text(encoding="utf-8", errors="replace").splitlines())
    except (OSError, TypeError):
        return False


def hy4_has_vision(cfg: dict) -> bool:
    """Hy4-flash's pack has the vision tower (hy4.txt's vision.* keys: tools/hy4/pack_gguf.py with the mmproj file)."""
    pack = pack_of(cfg)
    try:
        return any(line.startswith("vision.dim ")
                   for line in (pack / "hy4.txt").read_text(encoding="utf-8", errors="replace").splitlines())
    except (OSError, TypeError):
        return False


def mm3_pack_has(cfg: dict, key: str) -> bool:
    """MiniMax's pack indexes `key` (mm3.txt: "dspark.num_hidden_layers" its drafter, "vision.hidden_size" its vision
    tower - maya.py's checks)."""
    pack = pack_of(cfg)
    try:
        return any(line.startswith(key + " ")
                   for line in (pack / "mm3.txt").read_text(encoding="utf-8", errors="replace").splitlines())
    except (OSError, TypeError):
        return False


# ------------------------------------------------------------------ refusal ablation (an optional direction file)
# The engines that project a direction out per request (GEN ablate=1; loaded with --ablate FILE --ablate-default off):
# maya-ds4, maya-mm3 and the `maya` engine's GLM models (GLM-5.3-Flash, GLM-5.3; not on the Mac engine).  Qwen
# (CYBER-FROST) has none; without a direction file the dashboard shows no switch.
ABLATE_FAMILIES = ("deepseek", "minimax", "glm")
# a model repo's direction file (phase 2: fitted and shipped by us): <Model>-refusal-direction.gguf - a control vector
# GGUF (general.architecture "controlvector", direction.1 F32 [n_embd]) with maya.ablate.scale / .fit_layer / .model
# / .fitted_on / .version metadata; the setup downloads it with the model and records it as cfg["ablate"]["file"]
ABLATE_SUFFIX = "-refusal-direction.gguf"


def ablate_file(cfg: dict) -> Path | None:
    """The config's direction file (cfg["ablate"]["file"], else an --ablate FILE in its args) when it is on disk."""
    a = cfg.get("ablate") if isinstance(cfg.get("ablate"), dict) else {}
    f = a.get("file") or flag(cfg.get("args") or [], "--ablate")
    p = Path(str(f)).expanduser() if f else None
    return p if p is not None and p.is_file() else None


def ablate_choice(cfg: dict) -> dict:
    """{"available": bool, "why": the reason when not} - the Models page's and the Chat's switch (hidden without)."""
    fam = family(cfg)
    if fam not in ABLATE_FAMILIES or (fam == "glm" and cfg.get("backend") == "metal"):
        return {"available": False, "why": "not available for this model"}
    if ablate_file(cfg) is None:
        return {"available": False, "why": "no refusal direction file for this model (its repo ships none yet)"}
    if fam == "minimax" and (_int(flag(cfg.get("args") or [], "--tp"), 0) or 0) > 0:
        return {"available": False, "why": "not with the tensor-parallel split (Tensor parallel: 0 offers it)"}
    return {"available": True, "why": ""}


def ablate_default(cfg: dict) -> bool:
    """The Models page's default for requests that leave `ablate` out: cfg["ablate"]["default"] (off unless set), or
    the config's sampling.ablate."""
    a = cfg.get("ablate") if isinstance(cfg.get("ablate"), dict) else {}
    return a.get("default") is True or (cfg.get("sampling") or {}).get("ablate") is True


def ablate_args(cfg: dict, args: list) -> list:
    """args with the direction loaded, off by default (--ablate FILE --ablate-default off), where it can be; without
    a direction file (or on an engine without the projection) none of the three flags - the command as before."""
    for f in ("--ablate", "--ablate-default", "--ablate-layer", "--ablate-scale"):
        args = drop_flag(args, f)
    p = ablate_file(cfg)
    if p is None or family(cfg) not in ABLATE_FAMILIES or (family(cfg) == "glm" and cfg.get("backend") == "metal"):
        return args
    a = cfg.get("ablate") if isinstance(cfg.get("ablate"), dict) else {}
    args = set_flag(args, "--ablate", str(p))
    args = set_flag(args, "--ablate-default", "off")
    if a.get("scale") is not None:   # (the file's own maya.ablate.scale otherwise)
        args = set_flag(args, "--ablate-scale", a["scale"])
    return args


def choices(cfg: dict) -> dict:
    """What this config can be set to beyond what it is: a drafter and an image encoder that exist on this PC, and the
    refusal-ablation switch (ablate_choice)."""
    out = _choices(cfg)
    out["ablate"] = ablate_choice(cfg)
    return out


def _choices(cfg: dict) -> dict:
    if family(cfg) == "minimax":
        return {"draft": ["dspark", "none"] if mm3_pack_has(cfg, "dspark.num_hidden_layers") else ["none"],
                "images": mm3_pack_has(cfg, "vision.hidden_size")}
    if family(cfg) == "deepseek":
        return {"draft": ["dspark", "none"] if ds4_has_drafter(cfg) else ["none"],
                "images": ds4_has_vision(cfg) or "--vision" not in (cfg.get("args") or [])}
    if family(cfg) == "hy4":                           # (images: the pack's mmproj file)
        return {"draft": (["mtp"] if hy4_has_mtp(cfg) else []) + (["dfly2"] if hy4_has_dfly2(cfg) else []) + ["none"],
                "images": hy4_has_vision(cfg)}
    if family(cfg) == "qwen":                          # (images: the setup's maya-vision entry, on or kept off)
        return {"draft": ["mtp", "none"] if qwen_mtp(cfg) is not None else ["none"],
                "images": bool(cfg.get("vision") or cfg.get("vision_off"))}
    if G53.config_is_glm_dsa(cfg):
        # GLM-5.3 (744B): text only (no vision tower), its NextN block drafts (DFlash2's drafter is GLM-5.3-Flash's),
        # and what its engine serves today - KV streaming, answers at once decoded together (glm_dsa_product)
        return {"draft": ["mtp"], "images": False, "text_only": True, "dflash2": False,
                "kv_streaming": G53.KV_STREAMING, "batched": G53.BATCHED_ROWS}
    f = dflash_file(cfg)
    return {"draft": ["mtp", "dflash2"] if f is not None else ["mtp"],
            "images": bool(cfg.get("vision") or cfg.get("vision_off"))}


# ------------------------------------------------------------------ settings <-> the config
def settings_of(cfg: dict) -> dict:
    args = cfg.get("args") or []
    env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    out = {"context": _int(flag(args, "--max-context"), 32768), "gpus": gpus_of(cfg), "env": env,
           # (refusal ablation: the default for requests that leave `ablate` out - ablate_default)
           "ablate": ablate_default(cfg)}
    if family(cfg) == "minimax":                       # (maya-mm3's own defaults where the config names none)
        vision, kvs = flag(args, "--vision"), flag(args, "--kv-streaming")
        out.update(images=vision == "on" or (vision is None and mm3_pack_has(cfg, "vision.hidden_size")),
                   parallel=_int(flag(args, "--parallel"), 1),
                   draft="dspark" if flag(args, "--dspark", "off") not in ("off", "none") else "none",
                   tp=_int(flag(args, "--tp"), 0), kv_cache=flag(args, "--kv-cache", "int8"),
                   kv_streaming=kvs == "on" or (kvs is None and _int(flag(args, "--kv-resident"), 0) > 0),
                   kv_resident=_int(flag(args, "--kv-resident")), cpu_threads=_int(flag(args, "--cpu-threads")),
                   reserve_gb=_float(flag(args, "--reserve-gb")), prefill_chunk=_int(flag(args, "--chunk")),
                   cache_gb=_float(flag(args, "--cache-gb")), dspark_conf=_float(flag(args, "--dspark-conf")))
        return out
    if family(cfg) == "hy4":                           # (maya-hy4's own defaults where the config names none)
        tp = flag(args, "--tp", "auto")
        out.update(images=flag(args, "--vision") != "off" and hy4_has_vision(cfg),
                   draft="mtp" if "--mtp" in [str(a) for a in args] else
                   "dfly2" if "--dfly2" in [str(a) for a in args] else "none",
                   tp="auto" if tp == "auto" else _int(tp, "auto"), reserve_gb=_float(flag(args, "--reserve-gb")),
                   prefill_chunk=_int(flag(args, "--chunk")), cpu_threads=_int(flag(args, "--cpu-threads")),
                   kv=flag(args, "--kv", "auto"))     # (the latent cache's rows: auto, the engine's default)
        return out
    if family(cfg) == "qwen":
        ec, pf = flag(args, "--expert-cache", "auto"), flag(args, "--prefill", "auto")
        ram = maya_names.lookup(env, "MAYA_EXPERT_RAM_GB")
        out.update(images=bool(cfg.get("vision")), reserve_mib=_int(flag(args, "--vram-reserve-mib")),
                   draft="mtp" if flag(args, "--mtp") and flag(args, "--mtp-max-t") != "1" else "none",
                   expert_cache=None if ec == "auto" else _int(ec), prefill=None if pf == "auto" else _int(pf),
                   kv=flag(args, "--kv", "fp16"), ram_gb=_int(ram))
        return out
    if family(cfg) == "deepseek":
        dspark = flag(args, "--dspark")
        out.update(images=flag(args, "--vision") != "off",
                   reserve_mib=_int(flag(args, "--reserve-mib")), parallel=_int(flag(args, "--parallel"), 1),
                   draft="none" if dspark == "none" or (dspark is None and not ds4_has_drafter(cfg)) else "dspark",
                   tp=_int(flag(args, "--tp"), 2), mmap=flag(args, "--mmap-experts", "auto"),   # (2: the engine's own)
                   prefill_chunk=_int(flag(args, "--prefill-chunk")), cpu_threads=_int(flag(args, "--cpu-threads")),
                   # (--suffix-draft: the prompt lookup's shortest match, 3 by the engine; 0 off)
                   prompt_lookup=_int(flag(args, "--suffix-draft"), 3) != 0,
                   dspark_conf=_float(flag(args, "--dspark-conf")),
                   # (fast prompt reading: the long prompt's skip as a per-layer cone, DS4_YOCO_CONE=1 in the engine's
                   #  environment - serve/server.py's child_env; off by default, answers can shift slightly)
                   fast_prefill=cfg.get("fast_prefill") is True)
        if out["mmap"] not in ("auto", "on", "off"):
            out["mmap"] = "auto"
        return out
    ec, pf = flag(args, "--expert-cache", "auto"), flag(args, "--prefill", "auto")
    out.update(images=bool(cfg.get("vision")), reserve_mib=_int(flag(args, "--vram-reserve-mib")),
               parallel=_int(flag(args, "--parallel"), 1), draft="dflash2" if flag(args, "--dflash") else "mtp",
               expert_cache=None if ec == "auto" else _int(ec), prefill=None if pf == "auto" else _int(pf),
               kv_streaming=glm_streams(cfg), kv_resident=_int(flag(args, "--kv-resident")))
    if not G53.KV_STREAMING and G53.config_is_glm_dsa(cfg):   # (GLM-5.3: its whole cache in VRAM - apply() writes it)
        out.update(kv_streaming=False, kv_resident=None)
    return out


def glm53_kv_bytes(cfg: dict) -> int | None:
    """A GLM-5.3 (glm-dsa) config's cache bytes a context token (glm_dsa_product.kv_bytes_per_token) with the formats
    its engine runs: the INT8 latent unless --kv fp16 or MAYA_GLM_KV_INT8=0 says FP16 (the environment wins, as in the
    engine), the index keys FP32 unless MAYA_GLM_IDX_FP16=1.  None for any other config."""
    if not G53.config_is_glm_dsa(cfg):
        return None
    env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    int8 = maya_names.lookup(env, "MAYA_GLM_KV_INT8")
    int8 = int8.strip() != "0" if int8 is not None else flag(cfg.get("args") or [], "--kv") in (None, "int8")
    return G53.kv_bytes_per_token(G53.config_pack(cfg), int8,
                                  str(maya_names.lookup(env, "MAYA_GLM_IDX_FP16", "")).strip() == "1")


def glm_streams(cfg: dict) -> bool:
    """GLM's KV streaming as the engine runs it: --kv-streaming on/off, else only a window given (--kv-resident or the
    environment's GLM_KV_RESIDENT) streams - a config from before the flag keeps every cell in VRAM."""
    args, env = cfg.get("args") or [], cfg.get("env") or {}
    kvs = flag(args, "--kv-streaming")
    if kvs is not None:
        return kvs != "off"
    return (_int(flag(args, "--kv-resident"), 0) or 0) > 0 or \
        (_int(maya_names.lookup(env, "MAYA_GLM_KV_RESIDENT"), 0) or 0) > 0


def _num(s: dict, key: str, lo: int, hi: int, what: str, none_ok: bool = True):
    v = s.get(key)
    if v is None or v == "":
        if none_ok:
            return None
        raise ValueError(f"{what}: a number is needed")
    if isinstance(v, bool) or _int(v) is None:
        raise ValueError(f"{what}: a whole number, please")
    v = _int(v)
    if not lo <= v <= hi:
        raise ValueError(f"{what}: from {lo} to {hi}")
    return v


def _dec(s: dict, key: str, lo: float, hi: float, what: str):
    """A decimal setting: None where empty (the engine's own), else a number from lo to hi."""
    v = s.get(key)
    if v is None or v == "":
        return None
    if isinstance(v, bool) or _float(v) is None:
        raise ValueError(f"{what}: a number, please")
    v = _float(v)
    if not lo <= v <= hi:
        raise ValueError(f"{what}: from {lo:g} to {hi:g}")
    return v


def apply(cfg: dict, settings: dict) -> dict:
    """The config with `settings` written into it (a ValueError says what is wrong, nothing changed).  Keys left out
    keep the config's value."""
    if not isinstance(settings, dict):
        raise ValueError("the settings must be a JSON object")
    fam = family(cfg)
    ds4 = fam == "deepseek"
    known = {"minimax": MM3_KEYS, "deepseek": DS4_KEYS, "qwen": QWEN_KEYS, "hy4": HY4_KEYS}.get(fam, GLM_KEYS)
    known += ("ablate",)
    if fam == "hy4":
        not_yet = sorted(k for k in set(settings) & set(HY4_NOT_YET)
                         if settings[k] not in (None, "") and settings[k] != HY4_OFF.get(k))
        if not_yet:
            raise ValueError(HY4_NOT_YET[not_yet[0]])
        known += tuple(HY4_NOT_YET)
    unknown = sorted(set(settings) - set(known))
    if unknown:
        raise ValueError(f"not a setting of this model: {', '.join(unknown)}")
    s = {**settings_of(cfg), **settings}
    can = choices(cfg)
    out = copy.deepcopy(cfg)
    # refusal ablation's default (cfg["ablate"]["default"]: requests without `ablate` take it; not an engine setting -
    # no restart, the direction stays loaded and each request asks); on only where the switch is (ablate_choice)
    if not isinstance(s.get("ablate"), bool):
        raise ValueError("Refusal ablation: on or off")
    if s["ablate"] and settings.get("ablate") is True and not can["ablate"]["available"]:
        raise ValueError("Refusal ablation: " + can["ablate"]["why"])
    if isinstance(out.get("ablate"), dict):
        if s["ablate"]:
            out["ablate"]["default"] = True
        else:
            out["ablate"].pop("default", None)
    if not s["ablate"] and isinstance(out.get("sampling"), dict):
        out["sampling"].pop("ablate", None)
    args = [str(a) for a in out.get("args") or []]
    args = set_flag(args, "--max-context", _num(s, "context", 1024, 1048576, "Context size", none_ok=False))

    gpus = s.get("gpus")
    if not isinstance(gpus, list) or not gpus or not all(isinstance(g, int) and not isinstance(g, bool) and 0 <= g < 64
                                                        for g in gpus):
        raise ValueError("GPUs: pick at least one")
    if gpus != gpus_of(cfg):
        out["gpu"] = sorted(set(gpus))

    env = s.get("env") or {}
    if not isinstance(env, dict):
        raise ValueError("Environment: name = value pairs")
    clean = {}
    for k, v in env.items():
        k = str(k).strip()
        if not k:
            continue
        if not ENV_NAME.match(k):
            raise ValueError(f"Environment: {k!r} is not a variable name (letters, digits and _)")
        if str(v) == MASK and SECRET.search(k) and k in (cfg.get("env") or {}):
            v = (cfg.get("env") or {})[k]                # (the page's mask of a secret: its value as it was)
        if ENV_LOADER.match(k) and str((cfg.get("env") or {}).get(k)) != str(v):
            raise ValueError(f"Environment: {k} changes which programs or libraries the engine loads - not from the "
                             "page (the setup's --env sets it)")
        clean[k] = str(v)
    if clean:
        out["env"] = clean
    else:
        out.pop("env", None)

    reserve = _num(s, "reserve_mib", 0, 65536, "VRAM kept free") if fam not in ("minimax", "hy4") else None
    if fam == "qwen":
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: this pack has no MTP drafter (the setup builds it)" if s.get("draft") == "mtp"
                             else "Drafting: MTP or none")
        if qwen_mtp(cfg) is not None:   # (loaded either way - the server needs it; "none" keeps its drafts off)
            args = set_flag(args, "--mtp", qwen_mtp(cfg))
        args = drop_flag(args, "--mtp-max-t") if s["draft"] == "mtp" else set_flag(args, "--mtp-max-t", 1)
        # Images: the config's "vision" (maya-vision with the model's mmproj) and the engine's bare --vision flag (GENI
        # requests); off keeps the entry as "vision_off", so Images can be switched on again
        if s.get("images") and not out.get("vision"):
            if not out.get("vision_off"):
                raise ValueError("Images: this model was set up without its vision encoder (the setup adds it)")
            out["vision"] = out.pop("vision_off")
        elif not s.get("images") and out.get("vision"):
            out["vision_off"] = out.pop("vision")
        args = [x for x in args if x != "--vision"] + (["--vision"] if out.get("vision") else [])
        args = set_flag(args, "--vram-reserve-mib", reserve) if reserve is not None else \
            drop_flag(args, "--vram-reserve-mib")
        ec = _num(s, "expert_cache", 1, 1000000, "Expert cache")
        args = set_flag(args, "--expert-cache", "auto" if ec is None else ec)
        pf = _num(s, "prefill", 64, 65536, "Prompt chunk")
        args = set_flag(args, "--prefill", "auto" if pf is None else pf)
        kv = str(s.get("kv") or "fp16")
        if kv not in QWEN_KV:
            raise ValueError("KV cache: " + ", ".join(f.upper() for f in QWEN_KV))
        args = set_flag(args, "--kv", kv) if kv != "fp16" or "--kv" in args else args
        ram = _num(s, "ram_gb", 0, 4096, "Experts in RAM (GB)")
        env2 = dict(out.get("env") or {})
        if ram is None:
            for name in maya_names.both_names("MAYA_EXPERT_RAM_GB"):   # (set under either name)
                env2.pop(name, None)
        else:
            for name in maya_names.both_names("MAYA_EXPERT_RAM_GB"):
                env2.pop(name, None)
            env2["MAYA_EXPERT_RAM_GB"] = str(ram)
        if env2:
            out["env"] = env2
        else:
            out.pop("env", None)
    elif fam == "hy4":
        # (maya-hy4: --max-context above, the GPUs, drafting, tensor parallel, the VRAM kept free, the prompt chunk and
        #  the CPU lane's threads - empty: the engine's own.  --gpus counts the cards the config picks, in their order:
        #  the server shows the engine only those)
        if gpus != gpus_of(cfg):
            n = len(set(gpus))
            args = set_flag(args, "--gpus", ",".join(str(i) for i in range(n))) if n > 1 else drop_flag(args, "--gpus")
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: this pack has no MTP block" if s.get("draft") == "mtp" else
                             "Drafting: this pack has no DFly2 drafter" if s.get("draft") == "dfly2" else
                             "Drafting: MTP, DFly2 or none")
        args = [x for x in args if x not in ("--mtp", "--dfly2")] + \
            (["--mtp"] if s["draft"] == "mtp" else ["--dfly2"] if s["draft"] == "dfly2" else [])
        # (auto, the engine's default: tensor parallel over the largest group of equal GPUs; 0 splits the layers)
        tp = s.get("tp")
        if tp in (None, "", "auto"):
            args = drop_flag(args, "--tp")
        else:
            args = set_flag(args, "--tp", _num(s, "tp", 0, max(2, len(set(gpus))), "Tensor parallel", none_ok=False))
        v = _dec(s, "reserve_gb", 0, 64, "VRAM kept free")
        args = set_flag(args, "--reserve-gb", f"{v:g}") if v is not None else drop_flag(args, "--reserve-gb")
        ch = _num(s, "prefill_chunk", 64, 65536, "Prompt chunk")
        args = set_flag(args, "--chunk", ch) if ch is not None else drop_flag(args, "--chunk")
        ct = _num(s, "cpu_threads", 1, 512, "CPU threads")
        args = set_flag(args, "--cpu-threads", ct) if ct is not None else drop_flag(args, "--cpu-threads")
        # (the latent cache's rows: auto - the engine's default, F16 where they fit - writes no flag; 8-bit rows (the
        #  setup's opt-in faster drafting: the DFly2 drafter's room on a small card; slightly different wording than
        #  the default) --kv q8; a config's own --kv f16 stays offered as it is)
        kv = str(s.get("kv") or "auto")
        if kv not in ("auto", "q8") and not (kv == "f16" and flag(cfg.get("args") or [], "--kv") == "f16"):
            raise ValueError("KV cache: auto or 8-bit")
        args = drop_flag(args, "--kv") if kv == "auto" else set_flag(args, "--kv", kv)
        # (images: on is the engine's default when the pack has the tower - no flag; off gives its room to the model.
        #  On a card with no room for a picture beside the model the engine itself starts text only and says so)
        if s.get("images"):
            if not can["images"]:
                raise ValueError("Images: this pack has no vision tower (the setup packs the model with its mmproj "
                                 "file)")
            args = drop_flag(args, "--vision")
        elif can["images"] or "--vision" in args:
            args = set_flag(args, "--vision", "off")
        args, by_hand = hy4_chunk_again(cfg, {**out, "args": args})
        if by_hand != bool((out.get("installer") or {}).get(HY4_CHUNK_BY_HAND)):
            out["installer"] = {k: v for k, v in (out.get("installer") or {}).items() if k != HY4_CHUNK_BY_HAND}
            if by_hand:
                out["installer"][HY4_CHUNK_BY_HAND] = True
        if hy4_context_check(cfg, {**out, "args": args}) and flag(args, "--kv-streaming") is None:
            args = set_flag(args, "--kv-streaming", "auto")   # (streams only where nothing else fits, as the setup's)
    elif fam == "minimax":
        # (maya-mm3 --parallel: n conversations at once - each its own KV cache in VRAM, so no KV streaming and no DSpark
        #  while it is above 1; those two settings stay as chosen for the day it is 1 again; written where it changes)
        par = _num(s, "parallel", 1, 8, "Answers at once", none_ok=False)
        args = set_flag(args, "--parallel", par) if par > 1 or "--parallel" in args else args
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: this pack has no DSpark drafter" if s.get("draft") == "dspark"
                             else "Drafting: DSpark or none")
        if s["draft"] == "dspark" or "--dspark" in args:
            args = set_flag(args, "--dspark", "on" if s["draft"] == "dspark" else "off")
        if s.get("images") and not can["images"]:
            raise ValueError("Images: this pack has no vision tower")
        args = set_flag(args, "--vision", "on" if s.get("images") else "off") \
            if can["images"] or "--vision" in args else args
        # (0: the layers split across the GPUs; 1, 2 or 4: the tensor-parallel path in groups of that many)
        tp = _num(s, "tp", 0, 4, "Tensor parallel", none_ok=False)
        if tp == 3:
            raise ValueError("Tensor parallel: 1, 2 or 4 GPUs a group, or 0 to split the layers")
        args = set_flag(args, "--tp", tp) if tp else drop_flag(args, "--tp")
        kv = str(s.get("kv_cache") or "int8")
        if kv not in MM3_KV and kv != flag(cfg.get("args") or [], "--kv-cache"):   # (a K,V mix the config has: kept)
            raise ValueError("KV cache: " + ", ".join(f.upper() for f in MM3_KV))
        args = set_flag(args, "--kv-cache", kv) if kv != "int8" or "--kv-cache" in args else args
        args = set_flag(args, "--kv-streaming", "on" if s.get("kv_streaming") else "off")
        kr = _num(s, "kv_resident", 1024, 1048576, "Positions kept in VRAM")
        args = set_flag(args, "--kv-resident", kr) if kr is not None and s.get("kv_streaming") else \
            drop_flag(args, "--kv-resident")
        ct = _num(s, "cpu_threads", 1, 512, "CPU threads")
        args = set_flag(args, "--cpu-threads", ct) if ct is not None else drop_flag(args, "--cpu-threads")
        # (empty: the engine's own - --reserve-gb 1.0, 0.3 with --tp; --chunk the prefill calibration's; --cache-gb 1.5,
        #  read with --tp only; --dspark-conf 0.7)
        for key, name, lo, hi, what in (("reserve_gb", "--reserve-gb", 0, 64, "VRAM kept free"),
                                        ("cache_gb", "--cache-gb", 0, 256, "Expert cache"),
                                        ("dspark_conf", "--dspark-conf", 0, 1, "Draft confidence")):
            v = _dec(s, key, lo, hi, what)
            args = set_flag(args, name, f"{v:g}") if v is not None else drop_flag(args, name)
        ch = _num(s, "prefill_chunk", 16, 65536, "Prompt chunk")
        args = set_flag(args, "--chunk", ch) if ch is not None else drop_flag(args, "--chunk")
    elif ds4:
        # (maya-ds4 --parallel: n conversations at once - each its own context's caches in VRAM; written where it changes)
        par = _num(s, "parallel", 1, 8, "Answers at once", none_ok=False)
        args = set_flag(args, "--parallel", par) if par > 1 or "--parallel" in args else args
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: this pack has no DSpark drafter" if s.get("draft") == "dspark"
                             else "Drafting: DSpark or none")
        args = set_flag(args, "--dspark", "on" if s["draft"] == "dspark" else "none") \
            if ds4_has_drafter(cfg) or "--dspark" in args else args
        if s.get("images"):
            if not can["images"]:
                raise ValueError("Images: this pack has no vision tower")
            args = drop_flag(args, "--vision")
        else:
            args = set_flag(args, "--vision", "off")
        # (a group of up to every GPU picked - the calibration's 2, 4, 8, 16 or all of them; 2 on one GPU runs its
        #  layers alone, as before)
        tp = _num(s, "tp", 1, max(2, len(set(gpus))), "Tensor parallel", none_ok=False)
        args = set_flag(args, "--tp", tp)
        args = set_flag(args, "--reserve-mib", reserve) if reserve is not None else drop_flag(args, "--reserve-mib")
        if s.get("mmap") not in ("auto", "on", "off"):
            raise ValueError("Experts VRAM does not hold: auto, on or off")
        args = drop_flag(args, "--mmap-experts") if s["mmap"] == "auto" else set_flag(args, "--mmap-experts", s["mmap"])
        for key, name, lo, hi, what in (("prefill_chunk", "--prefill-chunk", 16, 65536, "Prompt chunk"),
                                        ("cpu_threads", "--cpu-threads", 1, 512, "CPU threads")):
            v = _num(s, key, lo, hi, what)
            args = set_flag(args, name, v) if v is not None else drop_flag(args, name)
        # the prompt lookup: on keeps a config's own shortest match (else the engine's 3), off is --suffix-draft 0
        if not isinstance(s.get("prompt_lookup"), bool):
            raise ValueError("Prompt lookup: on or off")
        if not s["prompt_lookup"]:
            args = set_flag(args, "--suffix-draft", 0)
        elif not _int(flag(args, "--suffix-draft"), 0):
            args = drop_flag(args, "--suffix-draft")
        conf = _dec(s, "dspark_conf", 0, 1, "Draft confidence")
        args = set_flag(args, "--dspark-conf", f"{conf:g}") if conf is not None else drop_flag(args, "--dspark-conf")
        if not isinstance(s.get("fast_prefill"), bool):
            raise ValueError("Fast prompt reading: on or off")
        if s["fast_prefill"]:
            out["fast_prefill"] = True
        else:
            out.pop("fast_prefill", None)               # (off: the engine's default, nothing in the config)
    else:
        dsa = G53.config_is_glm_dsa(cfg)                # GLM-5.3 (744B): what its engine takes (choices)
        par = _num(s, "parallel", 1, 8, "Answers at once", none_ok=False)
        args = set_flag(args, "--parallel", par) if par > 1 or "--parallel" in args else args
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: GLM-5.3 drafts with its own MTP (NextN) block" if dsa else
                             "Drafting: the DFlash2 drafter is not on this PC (the setup's --draft dflash2 downloads "
                             "it)" if s.get("draft") == "dflash2" else "Drafting: MTP or DFlash2")
        if s["draft"] == "dflash2":
            args = set_flag(args, "--dflash", dflash_file(cfg))
        else:
            args = drop_flag(args, "--dflash")
        if dsa and s.get("images"):
            raise ValueError("Images: GLM-5.3 reads text only (the model has no vision tower)")
        if s.get("images") and not out.get("vision"):
            if not out.get("vision_off"):
                raise ValueError("Images: this model was set up without its image encoder (the setup adds it)")
            out["vision"] = out.pop("vision_off")
        elif not s.get("images") and out.get("vision"):
            out["vision_off"] = out.pop("vision")       # (kept, so Images can be switched on again)
        args = set_flag(args, "--vram-reserve-mib", reserve) if reserve is not None else \
            drop_flag(args, "--vram-reserve-mib")
        # (a flag the config does not have is written only when its value changes: a config from an older setup, run
        #  by the engine it was built with, keeps its command - that engine may not know the newer flags)
        ec = _num(s, "expert_cache", 1, 1000000, "Expert cache")
        if ec is not None or "--expert-cache" in args:
            args = set_flag(args, "--expert-cache", "auto" if ec is None else ec)
        pf = _num(s, "prefill", 64, 65536, "Prompt chunk")
        if pf is not None or "--prefill" in args:
            args = set_flag(args, "--prefill", "auto" if pf is None else pf)
        if dsa and not G53.KV_STREAMING:
            # GLM-5.3: the engine keeps its whole cache in VRAM for now (a window below the context does not start) -
            # asked for, a clear no; a config with it (an older setup's "on", a window) gets it off, so every context
            # starts
            if settings.get("kv_streaming") or (_int(settings.get("kv_resident"), 0) or 0) > 0:
                raise ValueError("KV streaming: not served on GLM-5.3 yet - its whole attention cache stays in VRAM")
            if "--kv-streaming" in args or glm_streams(cfg):
                args = set_flag(args, "--kv-streaming", "off")
            args = drop_flag(args, "--kv-resident")
        else:
            if "--kv-streaming" in args or bool(s.get("kv_streaming")) != glm_streams(cfg):
                args = set_flag(args, "--kv-streaming", "on" if s.get("kv_streaming") else "off")
            kr = _num(s, "kv_resident", 1024, 1048576, "Positions kept in VRAM")
            args = set_flag(args, "--kv-resident", kr) if kr is not None and s.get("kv_streaming") else \
                drop_flag(args, "--kv-resident")
    if isinstance(out.get("ablate"), dict):   # (a direction the setup recorded: loaded, off by default)
        args = ablate_args(out, args)
    out["args"] = args
    return out


def shown_env(env: dict | None) -> dict:
    """A config's environment as the page gets it: a secret's value (a name with key, token, secret or password in it,
    as the server's report leaves them out) masked - a save that sends the mask back keeps the config's value."""
    return {str(k): (MASK if SECRET.search(str(k)) and str(v) else str(v)) for k, v in (env or {}).items()}


def shown_settings(cfg: dict) -> dict:
    return {**settings_of(cfg), "env": shown_env(cfg.get("env"))}


def command(cfg: dict) -> dict:
    """What the config runs: the engine, its arguments and its environment (the page's preview, secrets masked)."""
    return {"exe": Path(str(cfg.get("exe") or "maya")).name, "args": [str(a) for a in cfg.get("args") or []],
            "env": shown_env(cfg.get("env"))}


# ------------------------------------------------------------------ the list
def config_paths(root: Path) -> list:
    """The installed models' configs, the most recently used first (maya.py's configs())."""
    try:
        found = [p for p in Path(root).glob("maya-*.json") if not p.name.endswith(".shared-settings.json")]
        return sorted(found, key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return []


def read(path: Path) -> dict:
    try:
        cfg = json.loads(Path(path).read_text(encoding="utf-8-sig"))
        return cfg if isinstance(cfg, dict) else {}
    except (OSError, ValueError):
        return {}


def write(path: Path, cfg: dict):
    p = Path(path)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    os.replace(tmp, p)


def folder_size(pack: Path | None) -> int | None:
    """The model's files: its folder's (the GGUF shards beside the pack) and the pack's, one level each."""
    if pack is None:
        return None
    total = 0
    for d in (pack.parent, pack):
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_file(follow_symlinks=False):
                        total += e.stat(follow_symlinks=False).st_size
        except OSError:
            continue
    return total or None


def quant_label(cfg: dict, path: Path) -> str:
    """The files a config runs, as shown after the model's name ("Maya-XS", "UD-Q8_K_XL", "EXL3 3.5 bpw" - the setup's
    names, maya_quants); else its installer key, or the config's own name for one the setup did not write."""
    fam, key = family(cfg), maya_quants.config_key(cfg)
    if maya_quants.is_glm53(cfg):                      # (GLM-5.3, 744B: "UD-IQ2_M" after its own name)
        return str(maya_quants.glm53_quant(key) or path.stem.removeprefix("maya-").removesuffix("-hip"))
    what = maya_quants.tier(fam, key) or maya_quants.OTHERS.get((fam, key)) or OTHER_QUANTS.get((fam, str(key).lower()))
    what = maya_quants.exl3_bits(cfg) or what          # (an EXL3 quant set up by hand: its folder's bits, maya_quants)
    return str(what or key or path.stem.removeprefix("maya-").removesuffix("-hip"))


def missing(cfg: dict) -> str | None:
    """What keeps a config from starting (maya.py stops at it - a switch to it would leave no model running): None when
    its engine and pack are named and the pack is on this PC."""
    pack = pack_of(cfg)
    if not cfg.get("exe") or pack is None:
        return "its engine or pack is not set"
    if not pack.is_dir():
        return f"its model files are not on this PC ({pack.parent})"
    return None


def describe(path: Path, running: str | None) -> dict:
    cfg = read(path)
    pack = pack_of(cfg)
    why = missing(cfg)
    inst = cfg.get("installer") or {}
    script = next((s.name for s in (path.with_name(f"run-{path.stem}.sh"), path.with_name(f"run-{path.stem}.bat"))
                   if s.is_file()), None)
    try:
        used = path.stat().st_mtime
    except OSError:
        used = None
    cur = shown_settings(cfg)
    rng = context_range(trained_context_of(cfg), cur.get("context"),
                        {"deepseek": "ds4", "minimax": "mm3",
                         "hy4": "hy4-q4" if hy4_plain(cfg) else "hy4"}.get(family(cfg), ""))
    hy4_fits = hy4_contexts(cfg) if family(cfg) == "hy4" else []
    if family(cfg) == "hy4" and not hy4_plain(cfg):
        rng["max"] = min(rng["max"], max(hy4_fits + [CONTEXT_MIN, int(cur.get("context") or 0)]))
    setup = dict(inst.get("as_set_up") or cur)
    setup["env"] = shown_env(setup.get("env"))
    # GLM-5.3 (744B): its name on screen beside the quant, and its cache's bytes a context token for the Context size's
    # line (the page's GLM_KV_BYTES is GLM-5.3-Flash's) - keys only its configs have
    own = {"shown": G53.LABEL, "kv_bytes": glm53_kv_bytes(cfg)} if G53.config_is_glm_dsa(cfg) else {}
    if family(cfg) == "hy4":
        # (never said on the page: from which context the engine keeps its cache in the smaller rows - the Context
        #  size's GB figure counts them; and whether a context above the largest offered needs more VRAM - one line)
        room = None if hy4_plain(cfg) else hy4_room(cfg)
        small = (room or {}).get("contexts_q8") or []
        # (... and the contexts KV streaming takes: their rows' RAM a token and the positions kept in VRAM - the
        #  Context size's line then counts RAM, not VRAM)
        st = {c: room["streamed"](c) for c in room["contexts"]} if room else {}
        own.update(kv_streamed={str(c): {"ram": HY4_STREAM_RAM_BYTES[v[0]], "win": v[1]}
                                for c, v in st.items() if v} or None)
        own.update(kv_small_from=min(small) if small else None,
                   context_more=bool(hy4_fits) and max(hy4_fits) < (HY4_Q4_LONG if hy4_plain(cfg) else HY4_CONTEXTS)[-1])
    return {**own, "file": path.name, "path": str(path), "model": cfg.get("model_name") or "", "family": family(cfg),
            "quant": quant_label(cfg, path), "backend": cfg.get("backend") or "cuda", "written": inst.get("written"),
            "used": used, "running": running is not None and str(path.resolve()) == str(Path(running).resolve()),
            "port": cfg.get("port"), "script": script, "folder": str(pack.parent) if pack else None,
            "size": folder_size(pack), "ok": why is None, "missing": why,
            "settings": cur, "setup": setup, "choices": choices(cfg), "command": command(cfg),
            # (Hy4-Flash: only the contexts that fit this config's cards - hy4_contexts; the slider ends at the largest)
            "contexts": {"deepseek": DS4_CONTEXTS, "minimax": MM3_CONTEXTS, "qwen": QWEN_CONTEXTS,
                         "hy4": hy4_fits}.get(family(cfg), CONTEXTS),
            # (the Engine settings' Context size slider: the same range as Settings > Model's for this model)
            "context_range": rng}


_gpu_cache: dict = {}


def hy4_plain(cfg: dict) -> bool:
    """Hy4-flash in files the table has no VRAM plan for - AngelSlim's Q4_K_M: its contexts are the setup's own list
    for it (HY4_Q4_CONTEXTS; the cache by HY4_KV_BYTES), with no fit worked out."""
    return maya_quants.config_key(cfg) == "q4_k_m"


_HY4_BEFORE = {}                                    # {GPU index: MiB in use before this server started its engine}
_hy4_used_cache = {}


def hy4_note_before_start():
    """The server's, just before it starts Hy4-Flash's engine: the VRAM in use on each card then (the setup's reading,
    maya.py's gpu_used_mib) - what hy4_used falls back on while the engine runs on a PC whose nvidia-smi does not say
    how much of a card's VRAM a program holds (Windows).  Nothing kept where nvidia-smi does not answer."""
    try:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import maya
        got = maya.gpu_used_mib()
    except Exception:  # noqa: BLE001 - no reading: hy4_used counts the card as it reads now
        got = {}
    _HY4_BEFORE.clear()
    _HY4_BEFORE.update(got)
    _hy4_used_cache.clear()


def hy4_ram_gb() -> float | None:
    """This PC's RAM in GiB as the setup reads it (maya.py's mem_gb - hy4_fit's ram_gb); None where it cannot."""
    try:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import maya                                    # (the setup's own reading; nothing runs on import)
        return float(maya.mem_gb()[0]) or None
    except Exception:  # noqa: BLE001
        return None


def hy4_used(gpus: list) -> dict:
    """{GPU index: MiB of VRAM in use} on a config's cards as the setup reads it (maya.py's gpu_used_mib, the same
    nvidia-smi figure; hy4_usable_gib then counts at most HY4_FIT_USED_CAP_GIB a card, as for the setup) - less what
    a Maya engine loaded now holds there itself (maya.py's hy4_used_mib: the model this page sets up, or one that
    stops when it starts, is not "in use"; else the page would refuse the config that is running).  Asked at most
    every 10 s.  {} when nvidia-smi does not answer: the setup's sum for a PC with no reading."""
    now = time.time()
    hit = _hy4_used_cache.get("used")
    if not hit or now - hit[0] >= 10:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import maya
        hit = _hy4_used_cache["used"] = (now, maya.hy4_used_mib(before=dict(_HY4_BEFORE)))
    return {i: hit[1][i] for i in gpus if i in hit[1]} if hit[1] else {}


def hy4_cards(cfg: dict, gpus: list) -> dict:
    """{index: card} of machine_gpus' cards, each with the VRAM the engine's CUDA gets on it as the setup measured it
    under Windows (installer.engine_room_gb - maya.py's hy4_engine_room: Windows keeps part of every card, and while
    the engine holds it that cannot be measured again) where the card's VRAM is still the one measured then:
    hy4_usable_gib counts no more than that there, as the setup did.  Where the engine has since measured CUDA's own
    share on that card (maya.py's HY4_ROOM_STORE, this PC, card and driver - hy4_room_own): CUDA's reading the setup
    kept (the entry's third figure) with that share, as the next setup counts it ("engine_room_how": "measured" -
    the drafter's sums then as the setup's, without margins)."""
    cards = {g["index"]: g for g in gpus}
    room = (cfg.get("installer") or {}).get("engine_room_gb") or {}
    own = hy4_room_own() if any(isinstance(v, list) and len(v) >= 3 for v in room.values()) else {}
    for k, v in room.items():
        try:
            i, gib, vram = int(k), float(v[0]), float(v[1])
            how = "measured" if len(v) >= 3 and own.get(i) is not None else None
            if how:
                gib = float(v[2]) + own[i]
        except (TypeError, ValueError, IndexError):
            continue
        if i in cards and abs(float(cards[i].get("vram_gb") or 0) - vram) < 0.01:
            cards[i] = {**cards[i], "engine_room_gb": gib, **({"engine_room_how": how} if how else {})}
    return cards


def hy4_room_due(cfg: dict) -> dict | None:
    """The server's, before it starts an engine: Hy4-Flash's card on which CUDA's own share is to be measured at this
    start (maya.py's hy4_room_due: Windows, one NVIDIA card, none kept for this PC, card and driver); None else."""
    try:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import maya
        return maya.hy4_room_due(cfg)
    except Exception:  # noqa: BLE001 - nothing to measure: the start as it was
        return None


def hy4_room_measured(cfg: dict, card: dict, info: dict) -> dict | None:
    """After that start (the engine's --room-report in its INFO keys): the figure kept (maya.py's hy4_room_measure),
    said in the server window; None where none was."""
    try:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import maya
        rec = maya.hy4_room_measure(cfg, card, info)
    except Exception:  # noqa: BLE001 - no figure: the setup keeps its first guess
        rec = None
    _gpu_cache.pop("room_own", None)
    if rec:
        print(f"[maya] VRAM: CUDA's own share of {card.get('name') or 'the card'} measured at this first start: "
              f"{rec['own_gib']:.2f} GB (the setup and the Models page count it from now on)", flush=True)
    return rec


def hy4_room_own() -> dict:
    """{GPU index: GiB} - CUDA's own share of each of this PC's NVIDIA cards as the engine measured it (maya.py's
    HY4_ROOM_STORE, for this PC, the card and its driver now); {} where none is kept or nothing can be read."""
    now = time.time()
    hit = _gpu_cache.get("room_own")
    if hit and now - hit[0] < 60:
        return hit[1]
    out = {}
    try:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import maya                                    # (the setup's own records; nothing runs on import)
        saved = maya.hy4_room_load()
        if saved:
            for g in maya.S.gpus() or []:
                rec = saved.get(maya.hy4_room_key(g)) or {}
                if isinstance(rec.get("own_gib"), (int, float)):
                    out[int(g["index"])] = float(rec["own_gib"])
    except Exception:  # noqa: BLE001 - no record to read: the setup's figure stands
        out = {}
    _gpu_cache["room_own"] = (now, out)
    return out


def hy4_room(cfg: dict) -> dict | None:
    """What Hy4-Flash's config has in VRAM for a context, worked out as the setup does it (maya.py's
    hy4_config_need_gib and hy4_usable_gib: the plan of its size and layout - installer.hy4_quant / layout - in
    tools/hy4/download.json, its --chunk and --reserve-gb): {"usable": GiB on its cards, "need": GiB by context (a
    function: the least it takes - with the latent cache in F16 rows, or in the 8-bit rows the engine's --kv auto
    takes where F16 does not fit; a config's own --kv f16 / q8: that one's), "contexts": those of HY4_CONTEXTS that
    fit, "contexts_q8": those of them that fit only in 8-bit rows, "cards": how many, "streamed": a function - for a
    context that fits only with KV streaming (the global layers' older rows in RAM, where this PC's RAM holds them:
    the engine's --kv-streaming auto; "need" is then that VRAM) maya.hy4_config_stream's (rows, ring, GiB), else None}.  The other parts counted have no RAM fallback: the
    engine stops at its start where they do not fit.  Each card's VRAM in use is the setup's own
    reading (hy4_used: nvidia-smi's figure now, at most HY4_FIT_USED_CAP_GIB of it counted, a loaded Maya engine's own
    VRAM left out), so the page and the setup offer the same contexts.  None where it cannot be worked out: a card of
    the config this PC does not list with its VRAM, a size the table has no plan for."""
    inst, args, gpus = cfg.get("installer") or {}, cfg.get("args") or [], gpus_of(cfg)
    cards = hy4_cards(cfg, machine_gpus(cfg.get("backend") or "cuda"))
    if not gpus or any(not (cards.get(i) or {}).get("vram_gb") for i in gpus):
        return None
    try:
        root = str(Path(__file__).resolve().parents[1])
        if root not in sys.path:
            sys.path.insert(0, root)
        import maya                                    # (the setup's own plan and sums; nothing runs on import)

        kv = flag(args, "--kv")
        # (the prompt chunk: the config's own at the context it has; at another one the page writes the layout's
        #  choice again - hy4_chunk_again - so only a chunk set by hand counts there)
        cur, own = _int(flag(args, "--max-context"), 32768), hy4_chunk_by_hand(cfg)

        def one(ctx: int, q8: bool):
            return maya.hy4_config_need_gib(inst.get("hy4_quant") or "", inst.get("layout") or maya.HY4_STANDARD, ctx,
                                            chunk=_int(flag(args, "--chunk")) if own or ctx == cur else None,
                                            reserve_gb=_float(flag(args, "--reserve-gb")), q8=q8,
                                            mtp="--mtp" in [str(x) for x in args],
                                            dfly2="--dfly2" in [str(x) for x in args])

        def need_vram(ctx: int):
            if kv in ("f16", "q8") or one(ctx, False) is None:
                return one(ctx, kv == "q8")
            return min(one(ctx, False), one(ctx, True))
        if need_vram(HY4_CONTEXTS[0]) is None:
            return None
        usable = maya.hy4_usable_gib([cards[i] for i in gpus], hy4_used(gpus))
        rows = {c: maya.hy4_kv_rows(one(c, False), one(c, True), usable, kv) for c in HY4_CONTEXTS}
        # (KV streaming, the engine's --kv-streaming auto: a context the VRAM holds only with the global layers' older
        #  rows in RAM, where this PC's RAM holds them - as the setup's hy4_fit offers it)
        ram = None if flag(args, "--kv-streaming") == "off" else maya.hy4_ram_for_rows(
            maya.hy4_quant_table(inst.get("hy4_quant") or ""), usable, hy4_ram_gb())
        streamed = {}

        def stream(ctx: int):
            if ctx not in streamed:
                streamed[ctx] = maya.hy4_config_stream(
                    inst.get("hy4_quant") or "", inst.get("layout") or maya.HY4_STANDARD, ctx, usable, ram,
                    chunk=_int(flag(args, "--chunk")) if own or ctx == cur else None,
                    reserve_gb=_float(flag(args, "--reserve-gb")), mtp="--mtp" in [str(x) for x in args], kv=kv)
            return streamed[ctx]

        def need(ctx: int):
            n = need_vram(ctx)
            st = stream(ctx) if n is not None and n > usable else None
            return st[2] if st else n
        for c in HY4_CONTEXTS:
            if not rows[c]:
                stream(c)
    except Exception:  # noqa: BLE001 - no plan to read: the page keeps to the context the setup wrote
        return None
    return {"usable": usable, "need": need, "contexts": [c for c in HY4_CONTEXTS if rows[c] or streamed.get(c)],
            "contexts_q8": [c for c in HY4_CONTEXTS if rows[c] == "q8"], "cards": len(gpus),
            "streamed": lambda ctx: stream(ctx) if need_vram(ctx) is not None and need_vram(ctx) > usable else None}


HY4_CHUNK_BY_HAND = "chunk_by_hand"                 # (installer's: the page's Prompt chunk was changed by its user)


def _hy4_plan(cfg: dict):
    """(maya.py, the plan of the config's size and layout in tools/hy4/download.json) - the plan {} where the table
    has none."""
    root = str(Path(__file__).resolve().parents[1])
    if root not in sys.path:
        sys.path.insert(0, root)
    import maya                                        # (the setup's own plan and sums; nothing runs on import)
    inst = cfg.get("installer") or {}
    lay = maya.hy4_layouts(maya.hy4_quant_table(inst.get("hy4_quant") or "")).get(inst.get("layout") or maya.HY4_STANDARD)
    return maya, (lay or {}).get("plan") or {}


def hy4_chunk_by_hand(cfg: dict) -> bool:
    """Whether Hy4-Flash's config has a prompt chunk of its user's, which a change of context leaves alone, as a setup
    again leaves the flags set by hand: every --chunk of the standard layout (the engine's own choice there: the setup
    writes none and keeps one it finds); in a small-card layout one the page's user changed (installer.chunk_by_hand),
    or one that is not the layout's (its plan's chunk, or twice that: hy4_fit's two), or none at all."""
    try:
        maya, plan = _hy4_plan(cfg)
    except Exception:  # noqa: BLE001 - no plan to read: the chunk stays as it is
        return True
    inst, chunk = cfg.get("installer") or {}, _int(flag(cfg.get("args") or [], "--chunk"))
    base = int(plan.get("chunk") or 0)
    if (inst.get("layout") or maya.HY4_STANDARD) == maya.HY4_STANDARD or not base:
        return True
    return bool(inst.get(HY4_CHUNK_BY_HAND)) or chunk not in (base, 2 * base)


def hy4_refit(new: dict) -> dict | None:
    """The setup's choice (maya.py's hy4_fit: pictures first, then the larger prompt chunk) for Hy4-Flash's config
    `new` at its context, in its layout, on its cards as hy4_room takes them (hy4_used: the setup's own reading);
    pictures switched off (--vision off, or a pack without the tower) are not counted, as with the setup's
    --no-vision.  None where it cannot be worked out, or where the context does not fit (hy4_context_check says so)."""
    inst, args, gpus = new.get("installer") or {}, new.get("args") or [], gpus_of(new)
    cards = hy4_cards(new, machine_gpus(new.get("backend") or "cuda"))
    if hy4_plain(new) or not gpus or any(not (cards.get(i) or {}).get("vram_gb") for i in gpus):
        return None
    try:
        maya, plan = _hy4_plan(new)
        if not plan.get("need_gib"):
            return None
        dfly = "--dfly2" in [str(x) for x in args]     # (the DFly2 drafter's room first, as the setup's fit)
        fit = maya.hy4_fit(maya.hy4_quant_table(inst.get("hy4_quant") or ""), [cards[i] for i in gpus],
                           _int(flag(args, "--max-context"), 32768),
                           used=hy4_used(gpus),
                           layout=inst.get("layout") or maya.HY4_STANDARD,
                           pictures=flag(args, "--vision") != "off" and hy4_has_vision(new),
                           drafter=maya.hy4_dfly2_gib(inst.get("hy4_quant") or "") if dfly else 0.0,
                           kv=flag(args, "--kv"))      # (the config's own rows: --kv q8, the faster drafting's)
    except Exception:  # noqa: BLE001
        return None
    return None if fit.get("error") else fit


def hy4_chunk_again(cfg: dict, new: dict) -> tuple:
    """apply's: (the arguments of Hy4-Flash's config `new` - cfg with the page's settings - with the prompt chunk the
    setup would choose at its context, whether the chunk is its user's).  A small-card layout's chunk is the setup's
    choice for one context (hy4_fit: a picture's room first, the larger chunk only beside the largest picture's), so
    a config whose context the page changes gets that choice again for the new one: the larger chunk never takes a
    picture's room there.  Left as they are: a chunk set by hand - one this save changes, or hy4_chunk_by_hand -
    the standard layout, a save that keeps the context, and what cannot be worked out.  The larger chunk only with
    the layout's own VRAM kept free (the sums of hy4_fit are for that), its rows the config's own (--kv: hy4_refit)."""
    args = list(new.get("args") or [])
    if hy4_plain(cfg):
        return args, False
    marked = bool((cfg.get("installer") or {}).get(HY4_CHUNK_BY_HAND))
    try:
        maya, plan = _hy4_plan(new)
    except Exception:  # noqa: BLE001
        return args, marked
    base = int(plan.get("chunk") or 0)
    if ((new.get("installer") or {}).get("layout") or maya.HY4_STANDARD) == maya.HY4_STANDARD or not base:
        return args, False
    was, chunk = _int(flag(cfg.get("args") or [], "--chunk")), _int(flag(args, "--chunk"))
    ctx, cur = _int(flag(args, "--max-context"), 32768), settings_of(cfg)["context"]
    fit = hy4_refit(new)
    own = base
    if fit and fit.get("chunk"):
        plain = _float(flag(args, "--reserve-gb")) == float(plan.get("reserve_gb", -1))   # (its rows: hy4_refit's kv)
        own = int(fit["chunk"]) if plain else base
    if chunk != was:
        # (changed in this save: its user's from now on - unless it is the very one the setup would write here, or
        #  the one it did write: the page's Reset)
        setup = (cfg.get("installer") or {}).get("as_set_up") or {}
        back = chunk is not None and chunk == setup.get("prefill_chunk") and ctx == setup.get("context")
        return args, not (back or (fit is not None and chunk == own))
    if hy4_chunk_by_hand(cfg):
        return args, marked
    if ctx == cur or fit is None:
        return args, False
    return set_flag(args, "--chunk", own), False


def hy4_contexts(cfg: dict) -> list:
    """The contexts the Models page offers for Hy4-Flash's config: those of HY4_CONTEXTS that fit its cards
    (hy4_room) and those up to the one it has (the setup worked that one out); no VRAM figure: only the latter."""
    if hy4_plain(cfg):
        # (no VRAM plan: the setup's own list, and up to the model's 1M where the smallest card is large enough for the
        #  cache - maya.py's hy4_q4_contexts; the one the config has stays offered)
        cards, cur = {g["index"]: g for g in machine_gpus(cfg.get("backend") or "cuda")}, settings_of(cfg)["context"]
        vram = min((float((cards.get(i) or {}).get("vram_gb") or 0) for i in gpus_of(cfg)), default=0.0)
        return HY4_Q4_CONTEXTS + [c for c in HY4_Q4_LONG if c <= cur or c * HY4_KV_Q8_BYTES / 2**30 <= HY4_Q4_LONG_SHARE * vram]
    cur, room = settings_of(cfg)["context"], hy4_room(cfg)
    return [c for c in HY4_CONTEXTS if c <= cur or (room is not None and c in room["contexts"])]


def hy4_context_check(cfg: dict, new: dict):
    """apply's: a ValueError where Hy4-Flash's config `new` (cfg with the page's settings) asks for a context that
    does not fit its cards - the engine would stop at every start from then on.  A context up to the config's own on
    the same cards with the same VRAM kept free passes as it is (the setup worked it out; less never needs more).
    True: the context fits only with KV streaming (apply then writes --kv-streaming auto)."""
    if hy4_plain(cfg):
        return False
    ctx, cur = _int(flag(new.get("args") or [], "--max-context"), 32768), settings_of(cfg)["context"]
    same = sorted(set(gpus_of(new))) == sorted(set(gpus_of(cfg))) and         flag(new.get("args") or [], "--reserve-gb") == flag(cfg.get("args") or [], "--reserve-gb")
    if same and ctx <= cur:
        return False
    room = hy4_room(new)
    if room is None:
        if ctx > cur:
            raise ValueError(f"Context: the VRAM of this config's cards could not be read, so Hy4-Flash keeps at most "
                             f"the {cur} tokens its setup worked out")
        return False
    if room["need"](ctx) <= room["usable"]:
        return bool(room["streamed"](ctx))
    fits = [c for c in room["contexts"] if c < ctx] + ([cur] if same else [])
    cards = f"these {room['cards']} cards have" if room["cards"] > 1 else "this card has"
    raise ValueError(f"Context: Hy4-Flash's parts that must sit on the GPU need about {room['need'](ctx):.1f} GB of VRAM "
                     f"at {ctx} tokens; {cards} {room['usable']:.1f} GB for them - "
                     + (f"the largest that fits is {max(fits)}" if fits else "no context fits"))


def machine_gpus(backend: str = "cuda") -> list:
    """This PC's GPUs as the setup numbers them ([{index, name, vram_gb}]), asked once a minute."""
    now = time.time()
    hit = _gpu_cache.get(backend)
    if hit and now - hit[0] < 60:
        return hit[1]
    found = []
    try:
        import setup as SU                              # (the project's setup.py: nvidia-smi / the KFD topology)
        raw = SU.amd_gpus() if backend == "hip" else SU.gpus()
        # (an AMD card keeps what says so: maya.py's hy4_usable_gib counts its memory another way - a unified-memory
        #  APU's is the PC's)
        found = [{"index": int(g["index"]), "name": g.get("name"), "vram_gb": g.get("vram_gb"),
                  **({"arch": g.get("arch"), "vendor": "amd"} if backend == "hip" else {})} for g in raw or []]
    except (ImportError, OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        found = []
    _gpu_cache[backend] = (now, found)
    return found


def listing(root: Path, running: str | None) -> dict:
    models = [describe(p, running) for p in config_paths(root)]
    backend = next((m["backend"] for m in models if m["running"]), models[0]["backend"] if models else "cuda")
    gpus = machine_gpus(backend)
    seen = {g["index"] for g in gpus}
    for m in models:                                   # (a GPU a config names that the PC does not list now)
        for i in m["settings"]["gpus"]:
            if i not in seen:
                seen.add(i)
                gpus.append({"index": i, "name": None, "vram_gb": None})
    return {"models": models, "gpus": sorted(gpus, key=lambda g: g["index"]), "contexts": CONTEXTS}


def own_defaults(path: Path) -> bool:
    """A switch to the config at `path` (the Models page): its model's own sampling and thinking level
    (maya_quants.MODEL_DEFAULTS, as the setup writes them) written into it where they differ - a config from an older
    setup, or with another model's values - the one before kept beside it (<config>.prev).  True when it changed."""
    cfg = read(path)
    if not cfg:
        return False
    sampling, effort = maya_quants.model_defaults(cfg)
    budget = maya_quants.thinking_budget_for(cfg)      # (GLM's 8192 where none was set, or still the old 32768)
    if cfg.get("sampling") == sampling and cfg.get("reasoning_effort") == effort and budget is None:
        return False
    p = Path(path)
    p.with_name(p.name + ".prev").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    st = p.stat()
    write(p, {**cfg, "sampling": sampling, "reasoning_effort": effort,
              **({} if budget is None else {"thinking_budget": budget})})
    os.utime(p, (st.st_atime, st.st_mtime))
    return True


def _commit(path: Path, cfg: dict, new: dict) -> dict:
    """`new` written over the config at `path` (`cfg` as it was): its settings before kept beside it (<config>.prev), and
    the first time the page changes it, what the setup wrote (installer.as_set_up).  "changed": the engine's command or
    environment is not what it was - a restart is due (a Turbo marker alone is not one)."""
    engine = ("args", "env", "gpu", "vision", "vision_off")
    changed = any(new.get(k) != cfg.get(k) for k in engine)
    if new == cfg:
        return {"changed": False}
    inst = new.setdefault("installer", {})
    if "as_set_up" not in inst:
        inst["as_set_up"] = settings_of(cfg)
    prev = Path(path).with_name(Path(path).name + ".prev")
    prev.write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    st = Path(path).stat()
    write(path, new)
    os.utime(path, (st.st_atime, st.st_mtime))         # (its time says when it last ran: a start picks the latest)
    return {"changed": changed, "prev": prev.name}


def save(path: Path, settings: dict) -> dict:
    """The settings written into the config at `path` (its settings before kept beside it, <config>.prev, and the
    first time the page changes it, what the setup wrote: installer.as_set_up, the page's Reset)."""
    cfg = read(path)
    if not cfg:
        raise ValueError(f"{Path(path).name}: not a config Maya can read")
    return _commit(path, cfg, apply(cfg, settings))


# ------------------------------------------------------------------ Turbo (the page's Turbo button)
# The fastest settings the engine has, for people who play with speed and run benchmarks.  What the engine needs a start
# for is written into the config (the same "settings saved, Maya starts again" flow); the config keeps a "turbo" note:
# what it was before, so that switching Turbo off puts every one of these back.  The request-level half (greedy
# decoding, the fastest thinking level, coding prompts) lives in the page (serve/web/turbo-core.js).
TURBO_CONTEXT = 32768                               # GLM and MiniMax: a smaller context leaves VRAM to the experts ...
TURBO_CONTEXT_FAMILIES = ("glm", "minimax")         # ... DeepSeek's context is cheap, Qwen keeps its own
# Turbo trades quality for speed on purpose (benchmarks, playing with the engine): it goes past the safeguards.
# Expert skipping, where the engine has it (docs/SETTINGS.md): GLM's MISS_SKIP leaves out the light experts VRAM does not
# hold on layers 23 and up, at the highest value the docs measured (0.16: 30.9 vs 27.4 tok/s on Maya-S24, KL 0.075;
# _FROM stays 23 - the docs warn the early layers notice it far more); DeepSeek's RAM_SKIP the same rule for every
# expert VRAM does not hold, at the highest value documented (Maya-XS, DSpark: 27.8 -> 39.7 tok/s at 0.14).  MiniMax and
# Qwen have no skipping.  DeepSeek's DSPARK_CUM is lowered with the draft floor below (longer verified windows).
# Not used: MAYA_GLM_TRUNK_KEEP(_MAP) (its layer map is per model, a uniform value hurts the early layers),
# MAYA_*_DISK_SKIP_* (only for experts that fit neither VRAM nor RAM), MAYA_GLM_DFLASH_MIN/_CUM (meant for per-request
# tuning, no measured fast value).  The loop guard (MAYA_LOOP_GUARD) is deliberately NOT touched: it costs no speed and
# stops greedy runaway loops, which would ruin a benchmark.
# Where the whole model is in VRAM those skips have nothing to leave out, so Turbo also caps the routed experts a token
# computes, wherever they live (MAYA_*_RESIDENT_TOPK, docs/SETTINGS.md; the weight rule MAYA_*_RESIDENT_SKIP_W is not
# used: on the AMD Strix Halo test PC 0.20 already gave garbage, the rank cap degrades more gently).  The values are the
# fastest whose output was still sound there (a three.js page that parses with `node --check`, short factual answers):
# DeepSeek IQ1_M keeps 3 of its 6 experts a token (40.8 -> 49.5 tok/s with Turbo's other settings), GLM Maya-M-Derisked 4
# of its 8 (33.2 -> 38.8; 3 gave a page with a duplicate import and slower drafts).
TURBO_ENV = {"glm": {"MAYA_GLM_MISS_SKIP_W": "0.16", "MAYA_GLM_MISS_SKIP_FROM": "23", "MAYA_GLM_RESIDENT_TOPK": "4"},
             "deepseek": {"MAYA_DS4_RAM_SKIP_W": "0.14", "MAYA_DS4_DSPARK_CUM": "0.3", "MAYA_DS4_RESIDENT_TOPK": "3"}}
# GLM-5.3 (744B, glm-dsa) has its own: TURBO_ENV's GLM values were tuned on GLM-5.3-Flash's 45 layers.  Measured on
# GLM-5.3 Maya-XS (greedy, 512 tokens): MAYA_GLM_ROUTE_AFFINITY (added to a VRAM-resident expert's selection score, so
# more of a token's experts are ones VRAM holds) at 0.2 decodes 22-25% faster, KL 0.20 against the exact route, the loop
# test and a 20K-token document question as without it; 0.5 and up drifted into echoing the prompt, and
# MAYA_GLM_RESIDENT_TOPK 6 to 4 broke the code answers (not used).
TURBO_ENV_G53 = {"MAYA_GLM_ROUTE_AFFINITY": "0.2"}
# the drafter that goes fastest where the pack has it: DSpark (DeepSeek, MiniMax), MTP (Qwen).  GLM keeps the drafter the
# setup measured for this PC (MTP or DFlash2: which is faster depends on the GPUs).
TURBO_DRAFT = {"deepseek": "dspark", "minimax": "dspark", "qwen": "mtp"}
# DSpark's draft confidence floor (default 0.7), lowered so that longer windows are verified
TURBO_DSPARK_CONF = 0.3
TURBO_DSPARK_FAMILIES = ("deepseek", "minimax")
# The smallest quants (GLM and DeepSeek Maya-XS, experts at 1-2 bits) keep every expert a token routes to: on 3x V100 the
# cap gained at most 2-6% there, and GLM Maya-XS with 4 of 8 experts invented three.js classes and rambled.
TURBO_NO_TOPK = ("maya-xs",)


def turbo_smallest_quant(cfg: dict) -> bool:
    """Whether cfg runs one of TURBO_NO_TOPK's quants (the setup's record of the quant, else the pack's folder)."""
    inst = cfg.get("installer") or {}
    names = [str(inst.get(k) or "") for k in ("ds4_quant", "quant", "model_dir")]
    args = cfg.get("args") or []
    names += [str(args[i + 1]) for i, a in enumerate(args[:-1]) if a in ("--pack", "--glm-pack")]
    return any(t in n.lower().replace("_", "-") for n in names for t in TURBO_NO_TOPK)


def turbo_patch(cfg: dict) -> dict:
    """The engine settings Turbo changes in `cfg` (only those that differ from what it has now), for apply()."""
    fam, cur, can = family(cfg), settings_of(cfg), choices(cfg)
    patch = {}
    if fam in TURBO_CONTEXT_FAMILIES and (cur.get("context") or 0) > TURBO_CONTEXT:
        patch["context"] = TURBO_CONTEXT
    if (cur.get("parallel") or 1) > 1:                 # (one answer at a time is the fastest one)
        patch["parallel"] = 1
    want = TURBO_DRAFT.get(fam)
    if want and want in can["draft"] and cur.get("draft") != want:
        patch["draft"] = want
    if fam in TURBO_DSPARK_FAMILIES and (want in can["draft"] or cur.get("draft") == "dspark")             and cur.get("dspark_conf") != TURBO_DSPARK_CONF:
        patch["dspark_conf"] = TURBO_DSPARK_CONF
    if fam == "deepseek" and cur.get("prompt_lookup") is False:
        patch["prompt_lookup"] = True
    env = TURBO_ENV_G53 if G53.config_is_glm_dsa(cfg) else TURBO_ENV.get(fam)
    if env and turbo_smallest_quant(cfg):              # (no expert cap on the 1-2 bit quants: see TURBO_NO_TOPK)
        env = {k: v for k, v in env.items() if not k.endswith("_RESIDENT_TOPK")}
    if env and any(cur["env"].get(k) != v for k, v in env.items()):
        patch["env"] = {**cur["env"], **env}
    return patch


def turbo_on(cfg: dict) -> bool:
    return isinstance(cfg.get("turbo"), dict)


def turbo_info(cfg: dict) -> dict:
    """What the page needs: whether Turbo is on, and what switching it on would change (before -> after)."""
    cur = settings_of(cfg)
    if turbo_on(cfg):
        return {"on": True, "family": family(cfg), "since": cfg["turbo"].get("since"), "plan": {}}
    patch = turbo_patch(cfg)
    return {"on": False, "family": family(cfg),
            "plan": {k: {"from": cur.get(k), "to": v} for k, v in patch.items() if k != "env"},
            "env": {k: v for k, v in (patch.get("env") or {}).items() if cur["env"].get(k) != v}}


def turbo_enable(path: Path) -> dict:
    """Turbo on: its settings written into the config at `path`, and the note that undoes them.  "changed": the engine
    must start again."""
    cfg = read(path)
    if not cfg:
        raise ValueError(f"{Path(path).name}: not a config Maya can read")
    if turbo_on(cfg):
        raise ValueError("Turbo is already on")
    cur, patch = settings_of(cfg), turbo_patch(cfg)
    new = apply(cfg, patch)
    new["turbo"] = {"since": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "before": {k: cur.get(k) for k in patch},            # (the settings Turbo changed, as they were)
                    "args": list(cfg.get("args") or []), "env": cfg.get("env"),   # (and the command and environment whole)
                    "applied": {"args": list(new.get("args") or []), "env": new.get("env")}}
    return _commit(path, cfg, new)


def turbo_disable(path: Path) -> dict:
    """Turbo off: the config as it was before.  The command and environment come back whole when nothing else changed
    them since; if they were changed meanwhile (the context raised, a setting edited), only the settings Turbo changed go
    back to what they were."""
    cfg = read(path)
    if not cfg:
        raise ValueError(f"{Path(path).name}: not a config Maya can read")
    note = cfg.get("turbo")
    if not isinstance(note, dict):
        raise ValueError("Turbo is not on")
    applied = note.get("applied") or {}
    if list(cfg.get("args") or []) == applied.get("args") and cfg.get("env") == applied.get("env"):
        new = copy.deepcopy(cfg)
        new["args"] = list(note.get("args") or [])
        if note.get("env"):
            new["env"] = note["env"]
        else:
            new.pop("env", None)
    else:
        new = apply(cfg, dict(note.get("before") or {}))
    new.pop("turbo", None)
    return _commit(path, cfg, new)


# ------------------------------------------------------------------ deleting a model (the page's Delete)
# Off the list: its config, its start scripts and what Maya keeps beside it.  With its files: what the setup downloaded
# for it and the pack Maya built - never a file another model here runs from (hard-linked Engram tables, a vision tower
# or drafter several configs use, a pack built in another model's folder: they stay, and the page says so), nor one
# outside the models folders the setup downloaded into (a GGUF of your own stays, and a pack built beside it).  Its
# DFlash2 drafter goes too when no other model drafts with it.
GGUF_PART = re.compile(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", re.I)


def _res(p) -> Path | None:
    try:
        return Path(p).expanduser().resolve() if p else None
    except (OSError, RuntimeError, ValueError, TypeError):
        return None


def _within(p: Path, d: Path) -> bool:
    """`p` is `d` or in it."""
    return p == d or d in p.parents


def drafter_of(cfg: dict) -> Path | None:
    """The DFlash2 drafter a config drafts with, or the one its setup downloaded for it (installer.draft)."""
    p = flag(cfg.get("args") or [], "--dflash")
    if p:
        return Path(p)
    return dflash_file(cfg) if (cfg.get("installer") or {}).get("draft") == "dflash2" else None


def vision_files(cfg: dict) -> list:
    """GLM's image encoder: its mmproj and vocab files (on or switched off)."""
    out = []
    for key in ("vision", "vision_off"):
        v = cfg.get(key)
        if isinstance(v, dict):
            out += [Path(v[k]) for k in ("mmproj", "model") if isinstance(v.get(k), str) and v[k]]
    return out


def pack_shards(pack: Path) -> list:
    """The model files a pack indexes, in its folder: its sources.json's list, else the GGUF its native_experts.txt
    names (every part of a split one)."""
    names = []
    try:
        rec = json.loads((pack / "sources.json").read_text(encoding="utf-8")).get("files") or []
        names = [r["name"] for r in rec if isinstance(r, dict) and isinstance(r.get("name"), str)]
    except (OSError, ValueError, AttributeError):
        names = []
    if not names:
        try:
            with open(pack / "native_experts.txt", encoding="utf-8") as f:
                m = re.search(r"absolute offsets in ([^,)]+)", f.readline())
        except OSError:
            m = None
        first = Path(m.group(1).strip()).name if m else None
        part = GGUF_PART.match(first) if first else None
        names = ([f"{part.group(1)}-{i:05d}-of-{part.group(3)}.gguf" for i in range(1, int(part.group(3)) + 1)]
                 if part else [first] if first else [])
    return [pack.parent / Path(n).name for n in names]   # (by name: nothing outside the pack's folder)


def pack_reads(pack: Path) -> list:
    """The files a DeepSeek, MiniMax or Hy4-Flash pack reads: its tensors.txt's `file <i> <path>` lines (they come
    first: read up to the first tensor only), each relative to the model_dir its ds4.txt / mm3.txt / hy4.txt names -
    the model's own shards, and
    what it takes from other folders: the Engram tables of a GGUF set beside it, the DSpark drafter, the vision tower."""
    model = None
    for name in ("ds4.txt", "mm3.txt", "hy4.txt"):
        try:
            with open(pack / name, encoding="utf-8", errors="replace") as f:
                model = next((ln[len("model_dir "):].strip() for ln in f if ln.startswith("model_dir ")), None)
        except OSError:
            continue
        if model:
            break
    base = Path(model) if model and Path(model).is_dir() else pack.parent   # (a pack moved since: its own folder's)
    out = []
    try:
        with open(pack / "tensors.txt", encoding="utf-8", errors="replace") as f:
            for ln in f:
                w = ln.split()
                if len(w) < 3 or w[0] != "file":
                    break
                out.append(base / w[2])
    except OSError:
        pass
    return out


def runs_from(cfg: dict) -> list:
    """Every path a config runs from: its pack and the model files in it (a DeepSeek or MiniMax pack's from other
    folders too), its drafter, image encoder and tokenizer, and the folder or GGUF its setup named."""
    pack = pack_of(cfg)
    found = ([pack] + pack_shards(pack) + pack_reads(pack)) if pack else []
    inst = cfg.get("installer") or {}
    found += [Path(inst[k]) for k in ("model_dir", "gguf", "gguf_dir") if isinstance(inst.get(k), str) and inst[k]]
    found += [Path(v) for v in (flag(cfg.get("args") or [], "--dflash"), flag(cfg.get("args") or [], "--mtp"),
                                cfg.get("tokenizer")) if isinstance(v, str) and v]
    found += vision_files(cfg)
    return [r for r in map(_res, found) if r is not None]


def _size(p: Path) -> int:
    try:
        st = os.lstat(p)
    except OSError:
        return 0
    if not stat.S_ISDIR(st.st_mode):
        return st.st_size if stat.S_ISREG(st.st_mode) else 0
    total = 0
    for dp, _, files in os.walk(p):                     # (symlinked folders: not followed)
        for n in files:
            try:
                st = os.lstat(os.path.join(dp, n))
            except OSError:
                continue
            # (a hard link another model's folder also holds - the Engram tables the setup links between folders - frees
            # nothing when this one goes: the data stays with the last link, so it is not counted)
            total += st.st_size if stat.S_ISREG(st.st_mode) and st.st_nlink <= 1 else 0
    return total


def _linkers(roots: list, p: Path) -> list:
    """The names of `roots` ([(name, folder)]: other models' folders) under which a symbolic link points into `p`:
    deleting `p` would leave it dangling (the setup links the Engram tables between model folders when it cannot hard
    link them)."""
    seen, out = 0, []
    for name, r in roots:
        if name in out or not r.is_dir():
            continue
        for dp, dirs, files in os.walk(r):
            if any(os.path.islink(os.path.join(dp, n)) and _within(_res(os.path.join(dp, n)) or Path(dp, n), p)
                   for n in dirs + files):
                out.append(name)
                break
            seen += len(dirs) + len(files)
            if seen > 20000:
                return out
    return out


def _hard_links(p: Path) -> int:
    """How many files in `p` (or `p` itself) are hard links another folder holds too: deleting them frees nothing."""
    try:
        st = os.lstat(p)
    except OSError:
        return 0
    if not stat.S_ISDIR(st.st_mode):
        return int(stat.S_ISREG(st.st_mode) and st.st_nlink > 1)
    n = 0
    for dp, _, files in os.walk(p):
        for f in files:
            try:
                s = os.lstat(os.path.join(dp, f))
            except OSError:
                continue
            n += int(stat.S_ISREG(s.st_mode) and s.st_nlink > 1)
    return n


def delete_plan(root: Path, path: Path) -> dict:
    """What deleting the config at `path` with its files would delete: {"ok", "why" (when not ok), "items": [{"path",
    "what" (folder, pack, gguf, vision, drafter), "bytes", "linked" (hard links in it: they free nothing)}], "bytes",
    "folder", "gguf" (model files), "pack", "vision", "drafter" (its bytes, or None), "drafter_users" (the other configs
    drafting with its drafter: it stays then), "shared" (what stays because another model uses it: [{"path", "what",
    "users" (their configs), "files" (model files, for "gguf")}]), "linked" (hard links in all), "kept" (what else stays,
    said)}.  Only what is inside a models folder the setup downloaded into (models_dirs: never the folder itself, a drive's
    root, the project or the home folder), by its real path: a link out of it is never followed."""
    path = Path(path)
    cfg = read(path)
    me = _res(path)
    others = [(p, read(p)) for p in config_paths(root) if _res(p) != me]
    refs = [(p.name, runs_from(c)) for p, c in others]
    # (their folders, and what they run from: a link in one of them into this model's files keeps those)
    homes = [(p.name, h.parent) for p, h in ((p, _res(pack_of(c))) for p, c in others) if h is not None]
    homes += [(n, r) for n, rs in refs for r in rs]
    out = {"ok": False, "why": None, "items": [], "bytes": 0, "folder": None, "gguf": 0, "pack": False,
           "vision": False, "drafter": None, "drafter_users": [], "shared": [], "linked": 0, "kept": []}
    models = _res((cfg.get("installer") or {}).get("models_dir"))
    pack = _res(pack_of(cfg))
    if models is None or pack is None:
        out["why"] = "Maya's setup did not download this model's files, so it does not know which ones are its own"
        return out
    # (the folders the setup downloads into: every config's --models-dir and its own Maya-data/models - never a drive's
    # root, the home folder or one it is in)
    home = Path.home().resolve()
    known = [k for k in [_res((read(p).get("installer") or {}).get("models_dir")) for p in config_paths(root)] +
             [_res(Path(root).parent / "Maya-data" / "models")]
             if k is not None and k.is_dir() and k != Path(k.anchor) and not _within(home, k)]
    if models not in known:
        out["why"] = f"its models folder {models} is not one Maya downloads into, so Maya deletes nothing in it"
        return out
    folder = pack.parent
    out["folder"] = str(folder)
    project = _res(Path(root))
    items: list = []

    def inside(p: Path) -> bool:                        # (in a models folder Maya knows, not the folder itself)
        return any(_within(p, k) and p != k for k in known) and not (project and _within(project, p))

    def users(p: Path) -> list:                         # (another model runs from it, from something in it, or links to it)
        names = [n for n, rs in refs if any(_within(r, p) for r in rs)]
        return names + [n for n in _linkers(homes, p) if n not in names]

    def keep(p: Path, what: str, names: list, files: int = 0):
        out["shared"].append({"path": str(p), "what": what, "users": names, **({"files": files} if files else {})})

    def covered(p: Path) -> bool:
        return any(_within(p, it) for it, _ in items)

    def add(p: Path, what: str):
        if covered(p) or not (p.exists() or p.is_symlink()):
            return
        if inside(p):
            items.append((p, what))
        elif what == "pack":
            out["kept"].append(f"The pack Maya built in {p} stays: it is outside Maya's models folders, so Maya does "
                               "not delete it. Delete it by hand if you no longer want it.")

    downloaded = _within(folder, models)
    in_folder = users(folder) if downloaded else []
    if (downloaded and folder != models and not in_folder and not _within(project, folder)
            and not _within(Path.home().resolve(), folder)):
        add(folder, "folder")                           # (its own folder: the files, the pack, the image encoder)
    else:                                               # (a folder other models' files are in too, or your own)
        if in_folder and folder != models:
            keep(folder, "folder", in_folder)
        u = users(pack)
        if u:
            keep(pack, "pack", u)
        else:
            add(pack, "pack")
        if downloaded:
            held: dict = {}
            for f in pack_shards(pack):
                u = users(f)
                if u:
                    held.setdefault(tuple(u), []).append(f)
                else:
                    add(f, "gguf")
            for u, fs in held.items():
                keep(folder, "gguf", list(u), len(fs))
        else:
            out["kept"].insert(0, f"Your own model files in {folder} stay" + ("; only the pack Maya built from them goes."
                                   if any(it == pack for it, _ in items) else "."))
    for v in map(_res, vision_files(cfg)):
        if v is not None and _within(v, models) and not covered(v):
            u = users(v)
            if u:
                keep(v, "vision", u)
            else:
                add(v, "vision")
    d = _res(drafter_of(cfg))
    if d is not None and d.is_file() and _within(d, models):
        out["drafter_users"] = [n for n in users(d)]
        if out["drafter_users"]:                        # (another model drafts with it: it stays theirs)
            keep(d, "drafter", out["drafter_users"])
        else:
            add(d, "drafter")
    for p, what in items:
        n = _size(p)
        out["items"].append({"path": str(p), "what": what, "bytes": n, "linked": _hard_links(p)})
        out["linked"] += out["items"][-1]["linked"]
        out["bytes"] += n
        if what == "drafter":
            out["drafter"] = n
        elif what == "vision":
            out["vision"] = True
        elif what == "pack":
            out["pack"] = True
        elif what == "gguf":
            out["gguf"] += 1
        else:                                           # (its folder: the GGUF files in it, the pack, the encoder)
            try:
                out["gguf"] += sum(1 for e in p.iterdir() if e.suffix.lower() == ".gguf" and e.is_file())
                out["pack"] = out["pack"] or pack.is_dir()
                out["vision"] = out["vision"] or any(v is not None and _within(v, p) and v.exists()
                                                     for v in map(_res, vision_files(cfg)))
            except OSError:
                pass
    out["ok"] = True
    return out


def delete(root: Path, path: Path, files: bool = False) -> dict:
    """The config at `path` off the list: it, its start scripts, the settings kept beside it (.prev,
    .shared-settings.json); with `files`, delete_plan's files first (a file that cannot be deleted keeps the config,
    and says which).  {"deleted", "freed" (bytes), "drafter_users" (always [] now: a drafter another model drafts with
    stays)}."""
    path = Path(path)
    freed, users, errors = 0, [], []
    if files:
        plan = delete_plan(root, path)
        if not plan["ok"]:
            raise ValueError(plan["why"])
        for it in plan["items"]:
            p = Path(it["path"])
            try:
                if p.is_dir() and not p.is_symlink():
                    shutil.rmtree(p)
                else:
                    p.unlink()
                freed += it["bytes"]
            except FileNotFoundError:
                pass
            except OSError as e:
                errors.append(f"{p} ({e.strerror or e})")
                continue
            if it["what"] == "drafter":                 # (only a drafter no other model drafts with: delete_plan)
                try:
                    p.parent.rmdir()                    # (its folder, when nothing else is in it)
                except OSError:
                    pass
        if errors:
            raise OSError("could not delete " + "; ".join(errors) + " - the model stays on the list")
    for p in (path, path.with_name(f"run-{path.stem}.sh"), path.with_name(f"run-{path.stem}.bat"),
              path.with_name(path.name + ".prev"), path.with_name(path.stem + ".shared-settings.json")):
        try:
            p.unlink()
        except FileNotFoundError:
            pass
    return {"deleted": path.name, "freed": freed, "files": bool(files), "drafter_users": users}


# ------------------------------------------------------------------ files no model here runs from
# What the models folders hold that no config here names: a download whose config was deleted by hand or by an older
# version, a quant set up again into another folder.  Only what is in a folder the setup downloaded into: at its top a
# folder or a file of 100 MB or more (a download's log is not listed), and one level down in a folder another model
# runs from only a folder with a pack of its own (GLM's quants side by side in GLM-5.3-Flash/: DeepSeek's dspark/ and
# vision/ beside its GGUF folder are read through the pack, so a folder without one is never taken for unused).  A
# folder a pack reads from is used too (pack_reads: a DeepSeek EXL3 quant's Engram tables in the GGUF set's folder,
# its vision tower in another's).  Never a hidden one, the project or the home folder.  Maya may not have downloaded it: the page says what is in it, and
# deletes one at a time.
LEFTOVER_MIN = 10 ** 8
PACK_MARKS = ("sources.json", "tensors.txt", "mm3.txt", "ds4.txt", "hy4.txt", "native_experts.txt", "index.txt")


def models_dirs(root: Path) -> list:
    """The folders the setup downloaded into: each config's, and the setup's own (Maya-data/models beside the project)
    - not the home folder, nor one the project is in."""
    project, home = _res(root), Path.home().resolve()
    found = [_res((read(p).get("installer") or {}).get("models_dir")) for p in config_paths(root)]
    found.append(_res(Path(root).parent / "Maya-data" / "models"))
    out = []
    for d in found:
        if d is None or d in out or not d.is_dir() or _within(home, d) or (project and _within(project, d)):
            continue
        out.append(d)
    return out


def _has_pack(p: Path) -> bool:
    """A pack Maya built is in `p` (or `p` is one), two folders deep at most."""
    try:
        return any(next(p.glob(pat + m), None) is not None for pat in ("", "*/", "*/*/") for m in PACK_MARKS)
    except OSError:
        return False


def leftovers(root: Path) -> dict:
    """{"folders": the models folders, "items": [{"path", "name", "bytes", "folder" (a folder, else a file), "pack" (a
    pack Maya built is in it), "hub" (downloaded from the Hugging Face Hub), "modified"}], "bytes"}: what no model here
    runs from, largest first."""
    project, home = _res(root), Path.home().resolve()
    refs = [r for p in config_paths(root) for r in runs_from(read(p))]
    dirs, items = models_dirs(root), []

    def entries(d: Path) -> list:
        try:
            return [e for e in d.iterdir() if not e.name.startswith(".") and not e.is_symlink()]
        except OSError:
            return []

    def add(e: Path):
        n = _size(e)
        if n >= LEFTOVER_MIN:
            items.append({"path": str(e), "name": e.name, "bytes": n, "folder": e.is_dir(),
                          "pack": e.is_dir() and _has_pack(e), "hub": (e / ".cache" / "huggingface").is_dir(),
                          "modified": e.stat().st_mtime})

    for d in dirs:
        for e in entries(d):
            try:
                if not (e.is_dir() or e.is_file()) or (project and _within(project, e)) or _within(home, e):
                    continue
                if not any(_within(r, e) for r in refs):
                    add(e)
                elif e.is_dir() and not any(_within(e, r) for r in refs):   # (another model runs from a part of it)
                    for c in entries(e):
                        if c.is_dir() and not any(_within(r, c) for r in refs) and _has_pack(c):
                            add(c)
            except OSError:
                continue
    items.sort(key=lambda i: -i["bytes"])
    return {"folders": [str(d) for d in dirs], "items": items, "bytes": sum(i["bytes"] for i in items)}


def delete_leftover(root: Path, path) -> dict:
    """One of leftovers()'s items deleted, checked again first (a model set up since may run from it now).
    {"deleted", "freed" (bytes)}."""
    item = next((i for i in leftovers(root)["items"] if i["path"] == str(path or "")), None)
    if item is None:
        raise ValueError(f"{path} is not a file no model uses here (gone, or a model runs from it now)")
    p = Path(item["path"])
    if item["folder"]:
        shutil.rmtree(p)
    else:
        p.unlink()
    return {"deleted": str(p), "freed": item["bytes"]}

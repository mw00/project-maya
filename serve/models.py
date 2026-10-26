"""serve/models.py - the web app's Models page: the run configs the setup wrote (maya-<model>.json beside maya.py), what
each one runs with, and the engine settings the page changes in them.

A config's settings, as the page shows them (None: the engine's own default, or "auto"):
  both:      context, images, gpus, reserve_mib, env
  GLM:       parallel, draft ("mtp" | "dflash2"), expert_cache, prefill, kv_streaming, kv_resident
  DeepSeek:  draft ("dspark" | "none"), tp, mmap ("auto" | "on" | "off"), prefill_chunk, cpu_threads
apply() writes them back into the config's "args", "env", "gpu" and "vision" - its other keys as they are."""
from __future__ import annotations

import copy
import json
import os
import re
import subprocess
import time
from pathlib import Path

SWITCH_FILE = ".maya-switch.json"           # the server's note to maya.py: start this config next (a switch)
CONTEXTS = [8192, 32768, 65536, 131072]       # the setup's choices (a config's own size is offered too)
DS4_QUANTS = {"gguf-q8": "UD-Q8_K_XL", "gguf-q4": "UD-Q4_K_XL", "maya-s": "Maya-S", "maya-m": "Maya-M",
              "maya-xs": "Maya-XS", "exl3": "EXL3 2.9 bpw", "exl3-3.5": "EXL3 3.5 bpw"}
ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")
GLM_KEYS = ("context", "images", "gpus", "reserve_mib", "env", "parallel", "draft", "expert_cache", "prefill",
            "kv_streaming", "kv_resident")
DS4_KEYS = ("context", "images", "gpus", "reserve_mib", "env", "draft", "tp", "mmap", "prefill_chunk", "cpu_threads")


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


def family(cfg: dict) -> str:
    args = [str(a) for a in cfg.get("args") or []]
    return "deepseek" if (cfg.get("installer") or {}).get("family") == "deepseek" or "--pack" in args[:-1] else "glm"


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


def choices(cfg: dict) -> dict:
    """What this config can be set to beyond what it is: a drafter and an image encoder that exist on this PC."""
    if family(cfg) == "deepseek":
        return {"draft": ["dspark", "none"] if ds4_has_drafter(cfg) else ["none"],
                "images": ds4_has_vision(cfg) or "--vision" not in (cfg.get("args") or [])}
    f = dflash_file(cfg)
    return {"draft": ["mtp", "dflash2"] if f is not None else ["mtp"],
            "images": bool(cfg.get("vision") or cfg.get("vision_off"))}


# ------------------------------------------------------------------ settings <-> the config
def settings_of(cfg: dict) -> dict:
    args = cfg.get("args") or []
    env = {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}
    out = {"context": _int(flag(args, "--max-context"), 32768), "gpus": gpus_of(cfg), "env": env}
    if family(cfg) == "deepseek":
        dspark = flag(args, "--dspark")
        out.update(images=flag(args, "--vision") != "off",
                   reserve_mib=_int(flag(args, "--reserve-mib")),
                   draft="none" if dspark == "none" or (dspark is None and not ds4_has_drafter(cfg)) else "dspark",
                   tp=_int(flag(args, "--tp"), 1), mmap=flag(args, "--mmap-experts", "auto"),
                   prefill_chunk=_int(flag(args, "--prefill-chunk")), cpu_threads=_int(flag(args, "--cpu-threads")))
        if out["mmap"] not in ("auto", "on", "off"):
            out["mmap"] = "auto"
        return out
    ec, pf = flag(args, "--expert-cache", "auto"), flag(args, "--prefill", "auto")
    out.update(images=bool(cfg.get("vision")), reserve_mib=_int(flag(args, "--vram-reserve-mib")),
               parallel=_int(flag(args, "--parallel"), 1), draft="dflash2" if flag(args, "--dflash") else "mtp",
               expert_cache=None if ec == "auto" else _int(ec), prefill=None if pf == "auto" else _int(pf),
               kv_streaming=flag(args, "--kv-streaming", "on") != "off", kv_resident=_int(flag(args, "--kv-resident")))
    return out


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


def apply(cfg: dict, settings: dict) -> dict:
    """The config with `settings` written into it (a ValueError says what is wrong, nothing changed).  Keys left out
    keep the config's value."""
    if not isinstance(settings, dict):
        raise ValueError("the settings must be a JSON object")
    ds4 = family(cfg) == "deepseek"
    known = DS4_KEYS if ds4 else GLM_KEYS
    unknown = sorted(set(settings) - set(known))
    if unknown:
        raise ValueError(f"not a setting of this model: {', '.join(unknown)}")
    s = {**settings_of(cfg), **settings}
    can = choices(cfg)
    out = copy.deepcopy(cfg)
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
        clean[k] = str(v)
    if clean:
        out["env"] = clean
    else:
        out.pop("env", None)

    reserve = _num(s, "reserve_mib", 0, 65536, "VRAM kept free")
    if ds4:
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
        tp = _num(s, "tp", 1, 2, "Tensor parallel", none_ok=False)
        args = set_flag(args, "--tp", tp)
        args = set_flag(args, "--reserve-mib", reserve) if reserve is not None else drop_flag(args, "--reserve-mib")
        if s.get("mmap") not in ("auto", "on", "off"):
            raise ValueError("Experts VRAM does not hold: auto, on or off")
        args = drop_flag(args, "--mmap-experts") if s["mmap"] == "auto" else set_flag(args, "--mmap-experts", s["mmap"])
        for key, name, lo, hi, what in (("prefill_chunk", "--prefill-chunk", 16, 65536, "Prompt chunk"),
                                        ("cpu_threads", "--cpu-threads", 1, 512, "CPU threads")):
            v = _num(s, key, lo, hi, what)
            args = set_flag(args, name, v) if v is not None else drop_flag(args, name)
    else:
        par = _num(s, "parallel", 1, 8, "Answers at once", none_ok=False)
        args = set_flag(args, "--parallel", par) if par > 1 or "--parallel" in args else args
        if s.get("draft") not in can["draft"]:
            raise ValueError("Drafting: the DFlash2 drafter is not on this PC (the setup's --draft dflash2 downloads "
                             "it)" if s.get("draft") == "dflash2" else "Drafting: MTP or DFlash2")
        if s["draft"] == "dflash2":
            args = set_flag(args, "--dflash", dflash_file(cfg))
        else:
            args = drop_flag(args, "--dflash")
        if s.get("images") and not out.get("vision"):
            if not out.get("vision_off"):
                raise ValueError("Images: this model was set up without its image encoder (the setup adds it)")
            out["vision"] = out.pop("vision_off")
        elif not s.get("images") and out.get("vision"):
            out["vision_off"] = out.pop("vision")       # (kept, so Images can be switched on again)
        args = set_flag(args, "--vram-reserve-mib", reserve) if reserve is not None else \
            drop_flag(args, "--vram-reserve-mib")
        ec = _num(s, "expert_cache", 1, 1000000, "Expert cache")
        args = set_flag(args, "--expert-cache", "auto" if ec is None else ec)
        pf = _num(s, "prefill", 64, 65536, "Prompt chunk")
        args = set_flag(args, "--prefill", "auto" if pf is None else pf)
        args = set_flag(args, "--kv-streaming", "on" if s.get("kv_streaming") else "off")
        kr = _num(s, "kv_resident", 1024, 1048576, "Positions kept in VRAM")
        args = set_flag(args, "--kv-resident", kr) if kr is not None and s.get("kv_streaming") else \
            drop_flag(args, "--kv-resident")
    out["args"] = args
    return out


def command(cfg: dict) -> dict:
    """What the config runs: the engine, its arguments and its environment (the page's preview)."""
    return {"exe": Path(str(cfg.get("exe") or "maya")).name, "args": [str(a) for a in cfg.get("args") or []],
            "env": {str(k): str(v) for k, v in (cfg.get("env") or {}).items()}}


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
    inst = cfg.get("installer") or {}
    if family(cfg) == "deepseek":
        q = str(inst.get("ds4_quant") or inst.get("quant") or "")
        return DS4_QUANTS.get(q.lower(), q) or path.stem.removeprefix("maya-")
    return str(inst.get("quant") or path.stem.removeprefix("maya-").removesuffix("-hip"))


def describe(path: Path, running: str | None) -> dict:
    cfg = read(path)
    pack = pack_of(cfg)
    inst = cfg.get("installer") or {}
    script = next((s.name for s in (path.with_name(f"run-{path.stem}.sh"), path.with_name(f"run-{path.stem}.bat"))
                   if s.is_file()), None)
    try:
        used = path.stat().st_mtime
    except OSError:
        used = None
    cur = settings_of(cfg)
    return {"file": path.name, "path": str(path), "model": cfg.get("model_name") or "", "family": family(cfg),
            "quant": quant_label(cfg, path), "backend": cfg.get("backend") or "cuda", "written": inst.get("written"),
            "used": used, "running": running is not None and str(path.resolve()) == str(Path(running).resolve()),
            "port": cfg.get("port"), "script": script, "folder": str(pack.parent) if pack else None,
            "size": folder_size(pack), "ok": bool(cfg.get("exe") and pack is not None),
            "settings": cur, "setup": inst.get("as_set_up") or cur, "choices": choices(cfg), "command": command(cfg)}


_gpu_cache: dict = {}


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
        found = [{"index": int(g["index"]), "name": g.get("name"), "vram_gb": g.get("vram_gb")} for g in raw or []]
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


def save(path: Path, settings: dict) -> dict:
    """The settings written into the config at `path` (its settings before kept beside it, <config>.prev, and the
    first time the page changes it, what the setup wrote: installer.as_set_up, the page's Reset)."""
    cfg = read(path)
    if not cfg:
        raise ValueError(f"{Path(path).name}: not a config Maya can read")
    new = apply(cfg, settings)
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
    return {"changed": True, "prev": prev.name}

"""tools/maya_quants.py - Maya's quants by the names users see: the model first, then the tier by its routed experts'
bits, the same meaning for every model - XS ~1.6-1.7, S ~2-2.5, M ~3-3.5, L ~4+ (a rebuild: -v2, -v3).

    on screen       "MiniMax M3.1 · Maya-XS"                        shown(family, key), config_label(cfg)
    setup ids       --model minimax-maya-xs                         resolve(name)
    files           <Model>-Maya-<Tier>-<maintype>-0000N-of-0000M.gguf, one repo per model (<Model>-Maya-GGUF)
    configs         maya-minimax-maya-xs.json                       config_stem(family, key)

Every name a quant had before stays an alias, so nothing installed or typed before breaks: a GLM download's name
(--model Maya-S24, Maya-S-v2-IQ2_XXS, EXL3-3.05bpw ...), --ds4-quant maya-xs, --mm3-quant gguf-xs, and the installer
records of configs written before (their "quant" / "ds4_quant" / "mm3_quant" keys are what the setup still writes).
The files already published keep their names (their manifests list them).
"""
from __future__ import annotations

from pathlib import Path

FAMILIES = ("glm", "deepseek", "minimax")
MODEL = {"glm": "GLM-5.3-Flash", "deepseek": "DeepSeek-V4.1-Flash-1017", "minimax": "MiniMax-M3.1-Flash"}
SHOWN = {"glm": "GLM-5.3", "deepseek": "DeepSeek V4.1", "minimax": "MiniMax M3.1"}   # the model's name on screen
SEP = " · "

# Maya's own quants: (family, the setup's key for it, its tier, its setup id).  The key is what the setup and the
# configs' installer records use: a GLM download's MODELS key, a --ds4-quant / --mm3-quant value
QUANTS = (
    ("glm", "Maya-S-v2-IQ2_XXS", "Maya-S", "glm-maya-s"),
    ("glm", "Maya-S24", "Maya-S24", "glm-maya-s24"),
    ("glm", "Maya-M", "Maya-M", "glm-maya-m"),
    ("glm", "Maya-M-Derisked", "Maya-M-Derisked", "glm-maya-m-derisked"),
    ("glm", "Maya-L", "Maya-L", "glm-maya-l"),
    ("deepseek", "maya-xs", "Maya-XS", "deepseek-maya-xs"),
    ("deepseek", "maya-s", "Maya-S", "deepseek-maya-s"),
    ("deepseek", "maya-m", "Maya-M", "deepseek-maya-m"),
    ("minimax", "gguf-xs", "Maya-XS", "minimax-maya-xs"),
    ("minimax", "gguf-s", "Maya-S", "minimax-maya-s"),
)
# the other files the setup offers, as shown after the model's name (a GLM download not listed: its own name)
OTHERS = {("deepseek", "gguf-q8"): "UD-Q8_K_XL", ("deepseek", "gguf-q4"): "UD-Q4_K_XL",
          ("deepseek", "exl3"): "EXL3 2.9 bpw", ("deepseek", "exl3-3.5"): "EXL3 3.5 bpw",
          ("deepseek", "abliterated-128g"): "orcarouter Abliterated Mix-128G",
          ("deepseek", "abliterated-256g"): "orcarouter Abliterated Mix-256G",
          ("minimax", "exl3"): "EXL3 3.5 bpw"}
# --mm3-quant also takes the tier as DeepSeek's --ds4-quant does (maya-xs), beside its first names (gguf-xs)
MM3_QUANT_ALIASES = {"maya-xs": "gguf-xs", "maya-s": "gguf-s"}


def tier(family: str, key: str | None) -> str | None:
    """A Maya quant's tier ("Maya-XS"), or None for any other files."""
    return next((t for f, k, t, _ in QUANTS if f == family and k == key), None)


def setup_id(family: str, key: str | None) -> str | None:
    """A Maya quant's setup id ("minimax-maya-xs"), or None."""
    return next((i for f, k, _, i in QUANTS if f == family and k == key), None)


def full_name(family: str, key: str | None) -> str | None:
    """A Maya quant's full name, model first ("MiniMax-M3.1-Flash-Maya-XS"), or None."""
    t = tier(family, key)
    return f"{MODEL[family]}-{t}" if t else None


def shown(family: str, key: str | None = None, gb: float | None = None) -> str:
    """The files as users see them, the model's name first: "MiniMax M3.1 · Maya-XS", with gb "... (93 GB)"; never a
    bare tier.  Files that are not a Maya quant by their own name ("DeepSeek V4.1 · UD-Q8_K_XL"); no key: the model."""
    name = SHOWN.get(family, family)
    what = tier(family, key) or OTHERS.get((family, key)) or key
    return (name + SEP + what if what else name) + (f" ({gb:.0f} GB)" if gb else "")


def config_stem(family: str, key: str | None) -> str | None:
    """A Maya quant's config name under the scheme (maya-minimax-maya-xs), or None where the config keeps the name it
    always had (GLM's maya-<download>.json, DeepSeek's maya-ds4.json, MiniMax's EXL3 maya-mm3.json - installs and
    their run scripts point at them).  MiniMax's GGUFs, new with the scheme, are the first named by it."""
    return f"maya-{setup_id(family, key)}" if family == "minimax" and setup_id(family, key) else None


def resolve(name: str, glm_downloads=()) -> tuple | None:
    """(family, the setup's key) for a model named on the command line, or None for a name Maya does not know:
    a setup id (glm-maya-s, deepseek-maya-xs, minimax-maya-xs ...) or a quant's full name (MiniMax-M3.1-Flash-Maya-XS),
    any case; a GLM download by its name before (glm_downloads: the setup's MODELS keys, Maya-S24 ...); or a model's
    own name (DeepSeek-V4.1-Flash-1017, MiniMax-M3.1-Flash: its files then asked, or --ds4-quant / --mm3-quant) -
    key None."""
    low = (name or "").strip().lower()
    for f, k, _, i in QUANTS:
        if low in (i, full_name(f, k).lower()):
            return f, k
    for k in glm_downloads:
        if low == k.lower():
            return "glm", k
    for f in ("deepseek", "minimax"):
        if low == MODEL[f].lower():
            return f, None
    return None


def family_of(cfg: dict) -> str:
    """"minimax" for a MiniMax-M3.1-Flash config (maya-mm3), "deepseek" for a DeepSeek-V4.1-Flash config (maya-ds4's
    --pack: also the configs from before the programs' names of v1.1.16), else "glm"."""
    inst = cfg.get("installer") or {}
    exe = Path(str(cfg.get("exe", ""))).name
    if inst.get("family") == "minimax" or exe.startswith("maya-mm3"):
        return "minimax"
    args = cfg.get("args") or []
    return "deepseek" if inst.get("family") == "deepseek" or "--pack" in args[:-1] or \
        exe.startswith(("maya-ds4", "strata-ds4")) else "glm"   # (the engine's name before v1.1.16)


def config_key(cfg: dict) -> str | None:
    """The setup's key of a config's files, from its installer record (configs of every release: DeepSeek's and
    MiniMax's before their quants were recorded are the EXL3 quant's); None for a config the setup did not write."""
    inst = cfg.get("installer")
    if not isinstance(inst, dict):
        return None
    fam = family_of(cfg)
    if fam == "deepseek":
        return inst.get("ds4_quant") or "exl3"
    if fam == "minimax":
        return inst.get("mm3_quant") or "exl3"
    return inst.get("quant")


def config_label(cfg: dict) -> str | None:
    """What a config runs, as users see it ("GLM-5.3 · Maya-S24"), or None for a config the setup did not write."""
    if not isinstance(cfg.get("installer"), dict):
        return None
    return shown(family_of(cfg), config_key(cfg))

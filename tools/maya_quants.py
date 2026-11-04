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

import re
from pathlib import Path

FAMILIES = ("glm", "deepseek", "minimax", "qwen")
# ("qwen": CYBER-FROST-3.8-PS, the Qwen3.8-Flash-Next fine-tune - qwen4exp, on the `maya` engine's own Qwen path)
MODEL = {"glm": "GLM-5.3-Flash", "deepseek": "DeepSeek-V4.1-Flash-1017", "minimax": "MiniMax-M3.1-Flash",
         "qwen": "CYBER-FROST-3.8-PS"}
SHOWN = {"glm": "GLM-5.3", "deepseek": "DeepSeek V4.1", "minimax": "MiniMax M3.1",
         "qwen": "Qwen-3.8-Flash-Next (CYBER-FROST)"}   # the model's name on screen (the chat header: serve/web/lib.js's
                                                         # modelName(id, true), "Qwen-3.8-Flash-Next"); its ids stay
SEP = " · "

# Each model's own defaults - what its config gets (the setup, the dashboard's switch to it) and what a config without
# them runs with: its sampling as its developer publishes it (no top-k, min-p or penalties unless they name one) and the
# thinking level a request without one gets.  Every quant of a model the same (Maya's, unsloth's, orcarouter's, EXL3).
#   GLM-5.3-Flash: zai-org/GLM-5.3-Flash generation_config.json - temperature 1.0, top_p 0.95; the level its card's
#     default, Max (the owner, 2026-10-27; it was Maya's High)
#   DeepSeek-V4.1-Flash-1017: deepseek-ai's card ("Recommended sampling parameters") - temperature 1.0, top_p 1.0;
#     the level its encoder's default, high (a reasoning effort of 75); orcarouter's card names none (its base's)
#   MiniMax-M3.1-Flash: MiniMaxAI's card and generation_config.json (54e4e69) - temperature 1.0, top_p 0.95; no
#     default effort named, so "auto" (the template's own: the model decides whether to think)
MODEL_DEFAULTS = {
    "glm": ({"temperature": 1.0, "top_p": 0.95}, "max"),
    "deepseek": ({"temperature": 1.0, "top_p": 1.0}, "high"),
    "minimax": ({"temperature": 1.0, "top_p": 0.95}, "auto"),
    # CYBER-FROST-3.8-PS: Blackfrost-AI/CYBER-FROST-3.8-BF16 generation_config.json and card - temperature 1.0, top_p
    # 0.95, top_k 20; thinking on, its template's default effort xhigh
    "qwen": ({"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "xhigh"),
}


# A quant with its own sampling (the owner validated it on that quant, 2026-11-02): GLM-5.3-Flash Maya-XS - temperature
# 0.7, top_p 0.95, min_p 0.05, no penalties.  Every other quant keeps its model's (MODEL_DEFAULTS).
QUANT_SAMPLING = {("glm", "Maya-XS"): {"temperature": 0.7, "top_p": 0.95, "min_p": 0.05}}


def model_defaults(cfg: dict) -> tuple:
    """(sampling, reasoning_effort): a config's model's own defaults (MODEL_DEFAULTS by its family, the quant's own
    sampling where it has one: QUANT_SAMPLING), copies."""
    fam = family_of(cfg)
    sampling, effort = MODEL_DEFAULTS.get(fam, MODEL_DEFAULTS["glm"])
    return dict(QUANT_SAMPLING.get((fam, tier_of(cfg)), sampling)), effort


# The thinking budget (the config's "thinking_budget": tokens, 0 = no limit; the reasoning ends there and the answer
# starts).  GLM-5.3-Flash Maya-XS thinks past 32K on a plain three.js prompt and never answers; 8192 gave complete,
# valid answers fastest.  Only the quants listed here get another default (the owner, 2026-11-02: Maya-XS only); every
# other quant and model keeps 32768.  Keyed by (family, tier): a quant's catalog entry in QUANTS (and maya.MODELS)
# carries the tier, so adding the GLM Maya-XS entry there is all it needs to pick this up.
OLD_THINK_BUDGET = 32768
GLM_THINK_BUDGET = 8192
QUANT_THINK_BUDGET = {("glm", "Maya-XS"): GLM_THINK_BUDGET}


def tier_of(cfg: dict) -> str:
    """The tier ("Maya-XS") of the quant a config was set up with (installer.quant: a catalog key or a tier)."""
    fam, quant = family_of(cfg), str((cfg.get("installer") or {}).get("quant") or "")
    for f, key, tier, _ in QUANTS:
        if f == fam and quant in (key, tier):
            return tier
    return quant
THINK_BUDGET_MIN, THINK_BUDGET_MAX = 256, 262144


def default_thinking_budget(cfg: dict) -> int:
    """The thinking budget a config without one runs with: its quant's (QUANT_THINK_BUDGET), else 32768."""
    return QUANT_THINK_BUDGET.get((family_of(cfg), tier_of(cfg)), OLD_THINK_BUDGET)


def thinking_budget_for(cfg: dict) -> int | None:
    """The "thinking_budget" a setup or a switch writes into `cfg`, or None for 'leave it as it is'.  A value the user
    set (installer.thinking_budget_set, written by the dashboard's field) or any value but the old default stays; a
    config of a quant with its own default (QUANT_THINK_BUDGET) and none set, or still at the old 32768, gets it."""
    new = default_thinking_budget(cfg)
    if new == OLD_THINK_BUDGET or (cfg.get("installer") or {}).get("thinking_budget_set"):
        return None
    value = cfg.get("thinking_budget")
    return new if value is None or value == OLD_THINK_BUDGET else None


def check_thinking_budget(value) -> int:
    """A thinking budget from the dashboard: 0 (no limit) or 256..262144 tokens; ValueError otherwise."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value != int(value):
        raise ValueError("the thinking budget must be a whole number of tokens")
    value = int(value)
    if value != 0 and not THINK_BUDGET_MIN <= value <= THINK_BUDGET_MAX:
        raise ValueError(f"the thinking budget is 0 (no limit) or {THINK_BUDGET_MIN} to {THINK_BUDGET_MAX} tokens")
    return value

# Maya's own quants: (family, the setup's key for it, its tier, its setup id).  The key is what the setup and the
# configs' installer records use: a GLM download's MODELS key, a --ds4-quant / --mm3-quant value
QUANTS = (
    ("glm", "Maya-S-v2-IQ2_XXS", "Maya-S", "glm-maya-s"),
    ("glm", "Maya-S24", "Maya-S24", "glm-maya-s24"),
    ("glm", "Maya-XS-IQ1_S", "Maya-XS", "glm-maya-xs"),
    ("glm", "Maya-M", "Maya-M", "glm-maya-m"),
    ("glm", "Maya-M-Derisked", "Maya-M-Derisked", "glm-maya-m-derisked"),
    ("glm", "Maya-L", "Maya-L", "glm-maya-l"),
    ("deepseek", "maya-xs", "Maya-XS", "deepseek-maya-xs"),
    ("deepseek", "maya-s", "Maya-S", "deepseek-maya-s"),
    ("deepseek", "maya-m", "Maya-M", "deepseek-maya-m"),
    ("minimax", "gguf-xs", "Maya-XS", "minimax-maya-xs"),
    ("minimax", "gguf-s", "Maya-S", "minimax-maya-s"),
)
# the other files the setup offers, as shown after the model's name (a GLM download not listed: its own name); also
# the ones an install can still have though the list no longer offers them (DeepSeek's UD-Q4_K_XL: maya.DS4_BY_HAND)
OTHERS = {("deepseek", "gguf-q8"): "UD-Q8_K_XL", ("deepseek", "gguf-q4"): "UD-Q4_K_XL",
          # (unsloth's smaller UD quants: maya.DS4_UD, chosen by hand too)
          ("deepseek", "gguf-iq1_s"): "UD-IQ1_S", ("deepseek", "gguf-iq1_m"): "UD-IQ1_M",
          ("deepseek", "gguf-iq2_xxs"): "UD-IQ2_XXS", ("deepseek", "gguf-q2_k_xl"): "UD-Q2_K_XL",
          ("deepseek", "gguf-iq3_xxs"): "UD-IQ3_XXS", ("deepseek", "gguf-iq3_s"): "UD-IQ3_S",
          ("deepseek", "gguf-q3_k_xl"): "UD-Q3_K_XL",
          ("deepseek", "exl3"): "EXL3 2.9 bpw", ("deepseek", "exl3-3.5"): "EXL3 3.5 bpw",
          ("deepseek", "abliterated-128g"): "orcarouter Abliterated Mix-128G",
          ("deepseek", "abliterated-256g"): "orcarouter Abliterated Mix-256G",
          ("minimax", "exl3"): "EXL3 3.5 bpw", ("qwen", "q5_k_m"): "Q5_K_M"}
# CYBER-FROST-3.8-PS: its one GGUF (peasantsmith/CYBER-FROST-3.8-PS-GUFF) - the setup ids it is asked for by
QWEN_IDS = ("cyber-frost", "cyber-frost-3.8", "cyber-frost-3.8-ps", "cyber-frost-3.8-ps-q5_k_m", "cyber-frost-q5_k_m")
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
    if family == "qwen":
        return "maya-cyber-frost"
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
    if low in QWEN_IDS:
        return "qwen", "q5_k_m"
    for f in ("deepseek", "minimax", "qwen"):
        if low == MODEL[f].lower():
            return f, None
    return None


def family_of(cfg: dict) -> str:
    """"minimax" for a MiniMax-M3.1-Flash config (maya-mm3), "qwen" for CYBER-FROST-3.8-PS (the `maya` engine with its
    n-gram table, --ple-gguf), "deepseek" for a DeepSeek-V4.1-Flash config (maya-ds4's --pack: also the configs from
    before the programs' names of v1.1.16), else "glm"."""
    inst = cfg.get("installer") or {}
    exe = Path(str(cfg.get("exe", ""))).name
    if inst.get("family") == "minimax" or exe.startswith("maya-mm3"):
        return "minimax"
    args = cfg.get("args") or []
    if inst.get("family") == "qwen" or "--ple-gguf" in args[:-1]:
        return "qwen"
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
    if fam == "qwen":
        return inst.get("qwen_quant") or "q5_k_m"
    return inst.get("quant")


EXL3_BPW = re.compile(r"(?<![\d.])(\d+(?:\.\d+)?)[ _-]?bpw(?![a-z])", re.I)


def exl3_bits(cfg: dict) -> str | None:
    """"EXL3 2.25 bpw" from the model folder's name (DeepSeek-V4.1-Flash-1017-EXL3-2.25bpw: a DeepSeek or MiniMax EXL3
    quant of other bits than the setup's, set up by hand under the "exl3" key), else None - the setup's own label then."""
    fam, key = family_of(cfg), config_key(cfg)
    if fam not in ("deepseek", "minimax") or not str(key).lower().startswith("exl3"):
        return None
    inst = cfg.get("installer") or {}
    args = [str(a) for a in cfg.get("args") or []]
    pack = next((args[i + 1] for i, a in enumerate(args[:-1]) if a in ("--glm-pack", "--pack")), None)
    for d in (inst.get("model_dir"), Path(pack).parent if pack else None):
        m = EXL3_BPW.search(Path(str(d)).name) if d else None
        if m:
            return f"EXL3 {m.group(1)} bpw"
    return None


def config_label(cfg: dict) -> str | None:
    """What a config runs, as users see it ("GLM-5.3 · Maya-S24"), or None for a config the setup did not write."""
    if not isinstance(cfg.get("installer"), dict):
        return None
    fam, bits = family_of(cfg), exl3_bits(cfg)
    return SHOWN.get(fam, fam) + SEP + bits if bits else shown(fam, config_key(cfg))

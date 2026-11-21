"""tools/ik2_fit.py - Inkling-2-Mini on maya-ik2 (an experimental preview): what fits on one card, for the setup
(maya.py) and the Models page (serve/models.py) alike.

maya-ik2 keeps every expert of Inkling-2-Mini in VRAM where they fit (the resident token path, docs/INKLING2.md 8) and
falls back to its cache path - every answer slower - where they do not; it measures that itself at every start.  What
is worked out here is only which context to write into the config so that the resident path is the one taken: the
engine's own figures (docs/INKLING2.md 8, an RTX 3090 measured) - 22.3 GiB at a 262,144 context with an FP8 cache, of
which the cache is 16 KiB a token (the 8 visits of the layers without a window), so 18.3 GiB beside the cache.  Every
figure scales with the card's own VRAM: nothing here is one PC's constant but the model's sizes.  (The setup offers no
drafter: the engine's --draft is by hand, docs/INKLING2.md.)"""
from __future__ import annotations

GIB = 1 << 30
BASE_GIB = 18.3                                     # the model's weights, the windowed layers' cache, the scratch
KV_BYTES = {"fp8": 16384, "f16": 32768}             # the cache of a context token (IK2_KV)
MARGIN_GIB = 0.5                                    # the driver's and the desktop's share of the card
CONTEXTS = [32768, 65536, 131072, 262144]           # the setup's choices (the engine keeps every expert in VRAM up
                                                    # to 352,256 with an FP8 cache on a 24 GB card: text only)
DEFAULT_KV = "fp8"
MIN_VRAM_GB = 22.0                                  # the engine's own floor (check_preview: 22 GiB - a "24 GB" card)
MIN_CC = 80                                         # the engine's kernels: mma.sync m16n8k16, cp.async (Ampere on)


def need_gib(ctx: int, kv: str = DEFAULT_KV) -> float:
    """GiB of the card the engine takes at this context with every expert in VRAM."""
    return BASE_GIB + ctx * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB


def plan(vram_gb: float, ctx: int, kv: str = DEFAULT_KV) -> dict:
    """What this card holds at `ctx`: {"resident": every expert in VRAM, "need_gib"}.  A card whose VRAM is not known
    (0): resident - the engine measures at its start."""
    if not vram_gb:
        return {"resident": True, "need_gib": None}
    need = need_gib(ctx, kv)
    return {"resident": float(vram_gb) - MARGIN_GIB >= need, "need_gib": need}


def recommended_context(vram_gb: float, kv: str = DEFAULT_KV) -> int:
    """The largest of CONTEXTS whose cache leaves every expert in VRAM, else the smallest."""
    fits = [c for c in CONTEXTS if plan(vram_gb, c, kv)["resident"]]
    return max(fits) if fits else CONTEXTS[0]


def unsupported(backend: str, windows: bool, gpus_asked: int, cards: list) -> str | None:
    """Why this PC cannot run the preview, in one plain line, or None (said before anything is built or downloaded;
    the engine's own check_preview says the same at its start).  backend: cuda / hip / metal; gpus_asked: how many GPUs
    --gpu / --gpus named (0: none named); cards: the NVIDIA GPUs the model would pick from ({"arch", "vram_gb"},
    nvidia-smi's) - empty when they are not read yet."""
    if backend in ("hip", "metal"):
        return ("Inkling 2 Mini is an experimental preview that runs on NVIDIA GPUs only for now (no AMD or Mac "
                "engine yet)")
    if windows:
        return "Inkling 2 Mini is an experimental preview that runs on Linux only for now (not on Windows yet)"
    if gpus_asked > 1:
        return "Inkling 2 Mini is an experimental preview that runs on one GPU only for now: pick one (--gpu N)"
    if cards:
        new = [g for g in cards if int(g.get("arch") or 0) >= MIN_CC]
        if not new:
            return ("Inkling 2 Mini needs an NVIDIA RTX 30 / A100 or newer GPU (compute capability 8.0 or newer) - "
                    "older cards are not supported by its engine")
        if max(float(g.get("vram_gb") or 0) for g in new) < MIN_VRAM_GB:
            return "Inkling 2 Mini needs one NVIDIA GPU with 24 GB of VRAM or more (its whole model sits on the card)"
    return None

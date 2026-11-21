"""tools/ik2_fit.py - Inkling-2-Mini on maya-ik2 (an experimental preview): what fits on one card, for the setup
(maya.py) and the Models page (serve/models.py) alike.

maya-ik2 keeps every expert of Inkling-2-Mini in VRAM where they fit (the resident token path, docs/INKLING2.md 8) and
falls back to its cache path - every answer slower - where they do not; it measures that itself at every start.  What
is worked out here is only which context and drafter room to write into the config so that the resident path is
the one taken: the engine's own figures (docs/INKLING2.md 8, an RTX 3090 measured) - 22.3 GiB at a 262,144 context
with an FP8 cache, of which the cache is 16 KiB a token (the 8 visits of the layers without a window), so 18.3 GiB
beside the cache; the drafter's weights (mtp.safetensors, 0.59 GiB - 0.75 counted with its rings and verify windows) and its layer without a window, 4 KiB a position
in FP16 (IK2_DRAFT_CTX caps those positions: 65,536 keeps 262,144 resident on a 24 GB card).  Every figure scales
with the card's own VRAM: nothing here is one PC's constant but the model's sizes."""
from __future__ import annotations

GIB = 1 << 30
BASE_GIB = 18.3                                     # the model's weights, the windowed layers' cache, the scratch
KV_BYTES = {"fp8": 16384, "f16": 32768}             # the cache of a context token (IK2_KV)
DRAFT_GIB = 0.75                                    # the drafter's weights (0.59 GiB), its rings and windows
DRAFT_ROW = 4096                                    # its layer without a window: FP16, a position
DRAFT_CTX_MIN = 16384                               # a drafter that sees fewer positions than this: off instead
MARGIN_GIB = 0.5                                    # the driver's and the desktop's share of the card
CONTEXTS = [32768, 65536, 131072, 262144]           # the setup's choices (the engine keeps every expert in VRAM up
                                                    # to 352,256 with an FP8 cache on a 24 GB card: text only)
DEFAULT_KV = "fp8"
MIN_VRAM_GB = 22.0                                  # the engine's own floor (check_preview: 22 GiB - a "24 GB" card)
MIN_CC = 80                                         # the engine's kernels: mma.sync m16n8k16, cp.async (Ampere on)


def need_gib(ctx: int, kv: str = DEFAULT_KV, draft_ctx: int | None = None, drafting: bool = False) -> float:
    """GiB of the card the engine takes at this context with every expert in VRAM; with the drafter, its weights and
    `draft_ctx` positions of its layer without a window (None: the whole context)."""
    g = BASE_GIB + ctx * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB
    if drafting:
        g += DRAFT_GIB + min(ctx, draft_ctx or ctx) * DRAFT_ROW / GIB
    return g


def plan(vram_gb: float, ctx: int, kv: str = DEFAULT_KV, drafting: bool = True) -> dict:
    """What this card holds at `ctx`: {"resident": every expert in VRAM, "drafting": the drafter fits beside them,
    "draft_ctx": the drafter's positions to write (IK2_DRAFT_CTX; None: the whole context), "need_gib"}.  A card whose
    VRAM is not known (0): resident, the drafter with the whole context - the engine measures at its start."""
    if not vram_gb:
        return {"resident": True, "drafting": drafting, "draft_ctx": None, "need_gib": None}
    room = float(vram_gb) - MARGIN_GIB - need_gib(ctx, kv)
    if room < 0:
        return {"resident": False, "drafting": False, "draft_ctx": None, "need_gib": need_gib(ctx, kv)}
    if not drafting:
        return {"resident": True, "drafting": False, "draft_ctx": None, "need_gib": need_gib(ctx, kv)}
    if room >= DRAFT_GIB + ctx * DRAFT_ROW / GIB:
        return {"resident": True, "drafting": True, "draft_ctx": None,
                "need_gib": need_gib(ctx, kv, None, True)}
    fit = int((room - DRAFT_GIB) * GIB // DRAFT_ROW) if room > DRAFT_GIB else 0
    cap = 1 << (fit.bit_length() - 1) if fit > 0 else 0    # (a power of two: 65,536, 32,768 ...)
    if cap < DRAFT_CTX_MIN:
        return {"resident": True, "drafting": False, "draft_ctx": None, "need_gib": need_gib(ctx, kv)}
    return {"resident": True, "drafting": True, "draft_ctx": cap, "need_gib": need_gib(ctx, kv, cap, True)}


def recommended_context(vram_gb: float, kv: str = DEFAULT_KV, drafting: bool = True) -> int:
    """The largest of CONTEXTS whose cache leaves every expert in VRAM (the drafter beside them where it fits), else
    the smallest."""
    fits = [c for c in CONTEXTS if plan(vram_gb, c, kv, drafting)["resident"]]
    return max(fits) if fits else CONTEXTS[0]


def unsupported(backend: str, windows: bool, gpus_asked: int, cards: list) -> str | None:
    """Why this PC cannot run the preview, in one plain line, or None.  backend: cuda / hip / metal; gpus_asked: how
    many GPUs --gpu / --gpus named (0: none named); cards: the NVIDIA GPUs the model would pick from ({"arch", "vram_gb"},
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

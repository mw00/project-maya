"""tools/ik2_fit.py - Inkling-2-Mini on maya-ik2 (an experimental preview): what fits on one card, for the setup
(maya.py) and the Models page (serve/models.py) alike.

maya-ik2 keeps every expert of Inkling-2-Mini in VRAM where they fit (the resident token path, docs/INKLING2.md 8) and
falls back to its cache path - every answer slower - where they do not; it measures that itself at every start.  What
is worked out here is only which context to write into the config so that the resident path is the one taken: the
engine's own figures (docs/INKLING2.md 8, an RTX 3090 measured) - 22.3 GiB at a 262,144 context with an FP8 cache, of
which the cache is 16 KiB a token (the 8 visits of the layers without a window), so 18.3 GiB beside the cache.  Every
figure scales with the card's own VRAM: nothing here is one PC's constant but the model's sizes.  The drafter
(draft_gib) takes room only where the plan stays as it is without it: the engine checks at its start."""
from __future__ import annotations

GIB = 1 << 30
BASE_GIB = 18.3                                     # the model's weights, the windowed layers' cache, the scratch
KV_BYTES = {"fp8": 16384, "f16": 32768}             # the cache of a context token (IK2_KV)
MARGIN_GIB = 0.5                                    # the driver's and the desktop's share of the card
CONTEXTS = [32768, 65536, 131072, 262144]           # the setup's choices (the engine keeps every expert in VRAM up
                                                    # to 352,256 with an FP8 cache on a 24 GB card: text only)
DEFAULT_KV = "fp8"
MIN_VRAM_GB = 22.0                                  # the engine's own floor (check_preview: 22 GiB - a "24 GB" card)
MIN_CC = 70                                         # the engine's kernels: Volta on (V100: mma.m8n8k4; RTX 20, Ampere on)
AMD_ARCHS = ("gfx11", "gfx12")                      # AMD: RDNA 3 / 3.5 / 4 - the matrix-core prompt kernels (HIP)


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


DRAFT_BASE_GIB = 0.88                               # the drafter (mtp.safetensors 0.59 GiB, its fold through the head,
DRAFT_TOKEN_BYTES = 4864                            # its rings, the taps): maya-ik2's draft_room; its layer without a
                                                    # window's context 4 KiB a token + its scratch's 768 B


def draft_gib(ctx: int) -> float:
    """GiB the drafter takes beside the model at `ctx` (maya-ik2's draft_room: it sees the whole context)."""
    return DRAFT_BASE_GIB + ctx * DRAFT_TOKEN_BYTES / GIB


def draft_fits(vram_gb: float, ctx: int, kv: str = DEFAULT_KV) -> bool:
    """Whether the drafter's room fits beside the model at `ctx` on this card with every expert in VRAM - where it
    does not, the engine leaves it out at its start (the plan stays as without it; the prompt lookup still drafts).
    A card whose VRAM is not known: True (the engine measures)."""
    if not vram_gb:
        return True
    return plan(vram_gb, ctx, kv)["resident"] and float(vram_gb) - MARGIN_GIB >= need_gib(ctx, kv) + draft_gib(ctx)


def recommended_context(vram_gb: float, kv: str = DEFAULT_KV) -> int:
    """The largest of CONTEXTS whose cache leaves every expert in VRAM, else the smallest."""
    fits = [c for c in CONTEXTS if plan(vram_gb, c, kv)["resident"]]
    return max(fits) if fits else CONTEXTS[0]


def amd_unsupported(cards: list) -> str | None:
    """The AMD half of unsupported(): maya-ik2's HIP build (Linux) runs on RDNA 3 / 3.5 / 4 (gfx11 / gfx12: its prompt
    kernels are the matrix cores') with every expert in memory - 24 GB of VRAM, or an APU whose memory is the RAM
    ("unified_memory": its "memory_gb", what the engine may take of it).  cards: {"arch": "gfx1151", "vram_gb", ...}."""
    if not cards:
        return None
    rdna = [g for g in cards if str(g.get("arch") or "").startswith(AMD_ARCHS)]
    if not rdna:
        return ("Inkling 2 Mini on AMD needs a Radeon RX 7000 / 9000, a Radeon PRO W7000 / AI PRO R9700 or a Ryzen AI "
                "Max (Strix / Gorgon Halo) GPU - RDNA 3, 3.5 or 4 (gfx11 / gfx12); older AMD GPUs are not supported "
                "by its engine")
    room = max(float(g.get("memory_gb") if g.get("unified_memory") else g.get("vram_gb") or 0) for g in rdna)
    if room < MIN_VRAM_GB:
        return ("Inkling 2 Mini needs one AMD GPU with 24 GB of VRAM or more, or a Ryzen AI Max APU with that much "
                "memory free for it (its whole model sits in the GPU's memory)")
    return None


def unsupported(backend: str, windows: bool, gpus_asked: int, cards: list) -> str | None:
    """Why this PC cannot run the preview, in one plain line, or None (said before anything is built or downloaded;
    the engine's own check_preview says the same at its start).  backend: cuda / hip / metal; gpus_asked: how many GPUs
    --gpu / --gpus named (0: none named); cards: the GPUs the model would pick from (NVIDIA: {"arch", "vram_gb"},
    nvidia-smi's; AMD: amd_unsupported's) - empty when they are not read yet."""
    if backend == "metal":
        return ("Inkling 2 Mini is an experimental preview that runs on NVIDIA and AMD GPUs only for now (no Mac "
                "engine yet)")
    if backend == "hip":
        if windows:
            return ("Inkling 2 Mini on AMD runs on Linux only for now (its engine is not built for AMD on Windows "
                    "yet)")
        if gpus_asked > 1:
            return "Inkling 2 Mini is an experimental preview that runs on one GPU only for now: pick one (--gpu N)"
        return amd_unsupported(cards)
    if windows:
        return "Inkling 2 Mini is an experimental preview that runs on Linux only for now (not on Windows yet)"
    if gpus_asked > 1:
        return "Inkling 2 Mini is an experimental preview that runs on one GPU only for now: pick one (--gpu N)"
    if cards:
        new = [g for g in cards if int(g.get("arch") or 0) >= MIN_CC]
        if not new:
            return ("Inkling 2 Mini needs an NVIDIA V100 / RTX 20 or newer GPU (compute capability 7.0 or newer) - "
                    "older cards are not supported by its engine")
        if max(float(g.get("vram_gb") or 0) for g in new) < MIN_VRAM_GB:
            return "Inkling 2 Mini needs one NVIDIA GPU with 24 GB of VRAM or more (its whole model sits on the card)"
    return None

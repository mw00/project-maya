"""tools/ik2_fit.py - Inkling-2-Mini on maya-ik2 (an experimental preview): what fits on one card, for the setup
(maya.py) and the Models page (serve/models.py) alike.

maya-ik2 keeps every expert of Inkling-2-Mini in VRAM where they fit (the resident token path, docs/INKLING2.md 8);
where they do not - a card under 24 GB, or a context past what one holds - it takes the offload path: a tier of the
most-used experts in VRAM, the others in pinned RAM, fetched by the GPU (or computed by the CPU lane) as tokens pick
them (docs/INKLING2.md 8).  It measures that itself at every start; worked out here: which context to write into the
config, what share of the experts the card keeps, and the PCs that cannot run it at all.  The engine's own figures
(docs/INKLING2.md 8, an RTX 3090 measured) - 22.3 GiB at a 262,144 context with an FP8 cache, of which the cache is
16 KiB a token (the 8 visits of the layers without a window), so 18.3 GiB beside the cache, 14.7 GiB of it the
experts.  Every figure scales with the card's own VRAM and the PC's RAM: nothing here is one PC's constant but the
model's sizes.  (The setup offers no drafter: the engine's --draft is by hand, docs/INKLING2.md.)"""
from __future__ import annotations

GIB = 1 << 30
BASE_GIB = 18.3                                     # the model's weights, the windowed layers' cache, the scratch
EXPERTS_GIB = 14.7                                  # the routed experts (UD-Q4_K_XL) - of BASE_GIB
OFF_ROOM_GIB = 1.2                                  # the offload path's room for prompts and its staging slots
HOST_GIB = 1.5                                      # the engine's RAM beside the experts (embedding rows, buffers)
KV_BYTES = {"fp8": 16384, "f16": 32768}             # the cache of a context token (IK2_KV)
MARGIN_GIB = 0.5                                    # the driver's and the desktop's share of the card
CONTEXTS = [32768, 65536, 131072, 262144]           # the setup's choices (the engine keeps every expert in VRAM up
                                                    # to 352,256 with an FP8 cache on a 24 GB card: text only)
DEFAULT_KV = "fp8"
MIN_VRAM_GB = 7.0                                   # the engine's own floor (check_preview: 7 GiB - an "8 GB" card)
MIN_TOTAL_GB = 24.0                                 # RAM + VRAM that hold the model (no SSD tier: refused below)
MIN_CC = 80                                         # the engine's kernels: mma.sync m16n8k16, cp.async (Ampere on)


def need_gib(ctx: int, kv: str = DEFAULT_KV) -> float:
    """GiB of the card the engine takes at this context with every expert in VRAM."""
    return BASE_GIB + ctx * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB


def plan(vram_gb: float, ctx: int, kv: str = DEFAULT_KV) -> dict:
    """What this card holds at `ctx`: {"resident": every expert in VRAM, "need_gib", "layout": "resident" or
    "offload", "tier_gib": the experts' GiB the card keeps, "ram_gib": the engine's RAM (the offload path pins every
    expert in RAM)}.  A card whose VRAM is not known (0): resident - the engine measures at its start."""
    if not vram_gb:
        return {"resident": True, "need_gib": None, "layout": "resident", "tier_gib": EXPERTS_GIB, "ram_gib": HOST_GIB}
    need = need_gib(ctx, kv)
    if float(vram_gb) - MARGIN_GIB >= need:
        return {"resident": True, "need_gib": need, "layout": "resident", "tier_gib": EXPERTS_GIB, "ram_gib": HOST_GIB}
    tier = float(vram_gb) - MARGIN_GIB - (need - EXPERTS_GIB) - OFF_ROOM_GIB
    return {"resident": False, "need_gib": need, "layout": "offload",
            "tier_gib": round(max(0.0, min(EXPERTS_GIB, tier)), 1), "ram_gib": EXPERTS_GIB + HOST_GIB}


def recommended_context(vram_gb: float, kv: str = DEFAULT_KV) -> int:
    """The largest of CONTEXTS whose cache leaves every expert in VRAM, else the smallest (the one that leaves the
    most experts on the card)."""
    fits = [c for c in CONTEXTS if plan(vram_gb, c, kv)["resident"]]
    return max(fits) if fits else CONTEXTS[0]


def verdict(vram_gb: float, ram_gb: float, ctx: int, kv: str = DEFAULT_KV) -> tuple:
    """(word, line) for the setup and the Models page: "fast" (every expert in VRAM), "runs" (the offload path),
    "tight" (the offload path, with less RAM free than it pins - it may not start), "does not fit" (RAM + VRAM under
    the model).  ram_gb 0: not known."""
    p = plan(vram_gb, ctx, kv)
    if p["resident"]:
        return "fast", "every expert in VRAM"
    if ram_gb and ram_gb + vram_gb < MIN_TOTAL_GB:
        return "does not fit", too_small_line(ram_gb, vram_gb)
    share = 100.0 * p["tier_gib"] / EXPERTS_GIB
    line = (f"about {p['tier_gib']:.1f} of the experts' {EXPERTS_GIB:.1f} GB on the card ({share:.0f}%), the rest in "
            f"RAM - slower than with every expert in VRAM")
    if ram_gb and ram_gb < p["ram_gib"] + 2.0:
        return "tight", line + f"; it pins {p['ram_gib']:.0f} GB of RAM, this PC has {ram_gb:.0f} GB"
    return "runs", line


def too_small_line(ram_gb: float, vram_gb: float) -> str:
    return (f"Inkling 2 Mini needs about {MIN_TOTAL_GB:.0f} GB of RAM and VRAM together to hold the model (this PC "
            f"has {ram_gb:.0f} GB of RAM and {vram_gb:.0f} GB of VRAM)")


def unsupported(backend: str, windows: bool, gpus_asked: int, cards: list, ram_gb: float = 0.0) -> str | None:
    """Why this PC cannot run the preview, in one plain line, or None (said before anything is built or downloaded;
    the engine's own check_preview says the same at its start).  backend: cuda / hip / metal; gpus_asked: how many GPUs
    --gpu / --gpus named (0: none named); cards: the NVIDIA GPUs the model would pick from ({"arch", "vram_gb"},
    nvidia-smi's) - empty when they are not read yet; ram_gb: the PC's RAM (0: not read yet)."""
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
        vram = max(float(g.get("vram_gb") or 0) for g in new)
        if vram < MIN_VRAM_GB:
            return "Inkling 2 Mini needs one NVIDIA GPU with 8 GB of VRAM or more"
        if ram_gb and vram - MARGIN_GIB < need_gib(CONTEXTS[0]) and ram_gb + vram < MIN_TOTAL_GB:
            return too_small_line(ram_gb, vram)
    return None

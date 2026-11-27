"""tools/ik2_fit.py - Inkling-2-Mini on maya-ik2 : what fits on one card, for the setup
(maya.py) and the Models page (serve/models.py) alike.

maya-ik2 keeps every expert of Inkling-2-Mini in VRAM where they fit (the resident token path, docs/INKLING2.md 8);
where they do not - a card under 24 GB (NVIDIA from 8 GB, AMD from 12 GB), or a context past what one holds - it takes
the offload path: a tier of the most-used experts in VRAM, the others in pinned RAM, fetched by the GPU (or computed by
the CPU lane) as tokens pick them (docs/INKLING2.md 8; an AMD APU, whose memory is the RAM: every expert stays).  It measures that itself at
every start; worked out here: which context to write into the config, what share of the experts the card keeps, and
the PCs that cannot run it at all.  The engine's own figures (docs/INKLING2.md 8, an RTX 3090 measured) - 22.3 GiB at
a 262,144 context with an FP8 cache, of which the cache is 16 KiB a token (the 8 visits of the layers without a
window), so 18.3 GiB beside the cache, 14.7 GiB of it the experts.  Every figure scales with the card's own VRAM and
the PC's RAM: nothing here is one PC's constant but the model's sizes.  The drafter (draft_gib) takes room only where
the plan stays as it is without it, every expert in VRAM: the engine checks at its start."""
from __future__ import annotations

GIB = 1 << 30
BASE_GIB = 18.3                                     # the model's weights, the windowed layers' cache, the scratch
EXPERTS_GIB = 14.7                                  # the routed experts (UD-Q4_K_XL) - of BASE_GIB
OFF_ROOM_GIB = 1.2                                  # the offload path's room for prompts and its staging slots
HOST_GIB = 1.5                                      # the engine's RAM beside the experts (embedding rows, buffers)
OFF_HOST_GIB = 0.5                                  # the offload path's RAM beside the experts it pins (Jupiter, an 8
                                                    # GB card: 12.3 GiB anonymous with 12.1 of experts pinned)
OS_GIB = 1.5                                       # the system's and the server's RAM beside the engine ("tight" under)
KV_BYTES = {"fp8": 16384, "f16": 32768}             # the cache of a context token (IK2_KV)
MARGIN_GIB = 0.5                                    # the driver's and the desktop's share of the card
CONTEXTS = [32768, 65536, 131072, 262144, 524288, 1048576]   # the setup's choices, up to the trained 1,048,576 (the
                                                    # engine keeps every expert and the cache in VRAM up to 352,256
                                                    # with an FP8 cache on a 24 GB card; past it the offload path
                                                    # with the cache in VRAM, KV streaming only past that)
RECOMMEND_MAX = 262144                              # the largest context the setup recommends by itself
FULL_VISITS = 8                                     # the visits of the layers without a window: KV_BYTES / 8 each
SCRATCH_TOKEN_BYTES = 1152                          # the attention's scratch a context token past 262,144 (48 x 24 heads)
RING_OF_VISIT = 0.1                                 # the tokens' copy ring beside a full visit (V100: 2 x 2 slots of a
                                                    # twentieth of a visit's rows; more SMs, smaller slots)
STREAM_HOST_GIB = 4.0                              # the RAM the engine leaves the system beside the streamed rows
OFF_TIER_MIN_GIB = 0.5                              # an offload layout streams only where its cache leaves the tier less
                                                    # (the engine: else its cache in VRAM, as in v1.1.73)
DEFAULT_KV = "fp8"
MIN_VRAM_GB = 7.0                                   # the engine's own floor (check_hardware: 7 GiB - an "8 GB" card)
AMD_MIN_VRAM_GB = 11.0                              # AMD, a discrete card (check_hardware: 11 GiB - a "12 GB" card)
AMD_APU_MIN_GB = 22.0                               # an AMD APU: every expert in its memory (check_hardware: 22 GiB)
MIN_TOTAL_GB = 24.0                                 # RAM + VRAM that hold the model, as the refusal says it (no SSD
                                                    # tier): the floor itself is the RAM a card leaves (ram_gib)
MIN_CC = 70                                         # the engine's kernels: Volta on (V100: mma.m8n8k4; RTX 20, Ampere on)
AMD_ARCHS = ("gfx11", "gfx12")                      # AMD: RDNA 3 / 3.5 / 4 - the matrix-core prompt kernels (HIP)


def need_gib(ctx: int, kv: str = DEFAULT_KV) -> float:
    """GiB of the card the engine takes at this context with every expert in VRAM."""
    return BASE_GIB + ctx * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB


def plan(vram_gb: float, ctx: int, kv: str = DEFAULT_KV, apu: bool = False) -> dict:
    """What this card holds at `ctx`: {"resident": every expert in VRAM, "need_gib", "layout": "resident" or
    "offload", "tier_gib": the experts' GiB the card keeps, "ram_gib": the engine's RAM (the offload path pins the
    experts the card does not keep; KV streaming pins the cache's rows it keeps in RAM), "kv_streamed": the cache of
    some full-attention visits in RAM, "kv_ram_gib": those rows' GiB}.  Where the model and the cache do not fit
    together the experts leave the card first (a token reads the whole cache, but only ~8 experts a layer): the
    offload path with the cache in VRAM; maya-ik2 --kv-streaming auto streams the cache only where it alone does not
    fit beside the dense weights and a tier of OFF_TIER_MIN_GIB.  An AMD APU (apu: its memory is the RAM - no offload
    path there): every expert stays, the cache streams.  NVIDIA and AMD cards alike otherwise.  A card whose VRAM is not known (0): resident - the engine measures at its start."""
    if not vram_gb:
        return {"resident": True, "need_gib": None, "layout": "resident", "tier_gib": EXPERTS_GIB, "ram_gib": HOST_GIB,
                "kv_streamed": False, "kv_ram_gib": 0.0}
    room = float(vram_gb) - MARGIN_GIB
    need = need_gib(ctx, kv)
    if room >= need:
        return {"resident": True, "need_gib": need, "layout": "resident", "tier_gib": EXPERTS_GIB, "ram_gib": HOST_GIB,
                "kv_streamed": False, "kv_ram_gib": 0.0}
    kv_gib = ctx * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB
    vis = kv_gib / FULL_VISITS
    base = BASE_GIB + max(0, ctx - 262144) * SCRATCH_TOKEN_BYTES / GIB   # (beside the full visits' cache)
    # (an AMD APU) KV streaming with every expert in memory: as many full visits in VRAM as fit beside the tokens' ring, one
    # of them (the swing visit) the prompts' staging pair - its rows in RAM too
    ring = vis * RING_OF_VISIT
    n_res = min(FULL_VISITS, int((room - base - ring) // vis)) if room >= base + ring + vis else -1
    if apu and n_res >= 1:
        kv_ram = round((FULL_VISITS - n_res + 1) * vis, 1)
        return {"resident": True, "need_gib": round(base + n_res * vis + ring, 1), "layout": "resident",
                "tier_gib": EXPERTS_GIB, "ram_gib": round(HOST_GIB + kv_ram + STREAM_HOST_GIB, 1),
                "kv_streamed": True, "kv_ram_gib": kv_ram}
    # the offload path: its cache in VRAM where that leaves the tier OFF_TIER_MIN_GIB, else streamed
    dense = base - EXPERTS_GIB
    tier = room - dense - kv_gib - OFF_ROOM_GIB
    kv_ram = 0.0
    if tier < OFF_TIER_MIN_GIB:
        n_res = max(1, min(FULL_VISITS, int((room - dense - OFF_ROOM_GIB - OFF_TIER_MIN_GIB - ring) // vis)))
        kv_ram = (FULL_VISITS - n_res + 1) * vis
        tier = room - dense - n_res * vis - ring - OFF_ROOM_GIB
    tier = round(max(0.0, min(EXPERTS_GIB, tier)), 1)
    return {"resident": False, "need_gib": need, "layout": "offload", "tier_gib": tier,
            "ram_gib": round(EXPERTS_GIB - tier + OFF_HOST_GIB + (kv_ram + STREAM_HOST_GIB if kv_ram else 0.0), 1),
            "kv_streamed": kv_ram > 0, "kv_ram_gib": round(kv_ram, 1)}


TWO_ROOM_GIB = 0.3                                  # the second card's room beside its experts (token buffers, slack)


def plan_two(vram0_gb: float, vram1_gb: float, ctx: int, kv: str = DEFAULT_KV) -> dict:
    """Two NVIDIA cards (maya-ik2 --gpus 2, the larger first): the first alone where it holds every expert (the
    resident path - faster there than the split: 2x V100 86.7 against ~74 tok/s), else the split - the first card on
    the offload path (the dense weights, the cache, every prompt, its tier), the second with the next experts, the
    rest in pinned RAM.  plan()'s keys plus "gpus" (1 or 2), "layout" "resident" / "split" (every expert on the two
    cards) / "split+ram", "gpu1_gib" (the second card's experts).  "ram_gib" as one card's offload path: the prompts
    stream every expert the first card does not hold, the second card's too."""
    p0 = plan(vram0_gb, ctx, kv)
    if p0["resident"]:
        return dict(p0, gpus=1, gpu1_gib=0.0)
    g1 = round(max(0.0, min(EXPERTS_GIB - p0["tier_gib"], float(vram1_gb or 0) - MARGIN_GIB - TWO_ROOM_GIB)), 1)
    return dict(p0, gpus=2, gpu1_gib=g1, layout="split" if p0["tier_gib"] + g1 >= EXPERTS_GIB - 0.05 else "split+ram")


def verdict_two(vram0_gb: float, vram1_gb: float, ram_gb: float, ctx: int, kv: str = DEFAULT_KV) -> tuple:
    """verdict() for two cards (plan_two): "fast" - every expert in VRAM (on the first alone, or across the two),
    "runs" / "tight" / "does not fit" as verdict() with the split's share."""
    p = plan_two(vram0_gb, vram1_gb, ctx, kv)
    if p["gpus"] == 1:
        return verdict(vram0_gb, ram_gb, ctx, kv)
    if ram_gb and ram_gb < p["ram_gib"]:
        return "does not fit", too_small_line(ram_gb, vram0_gb, p["ram_gib"])
    held = p["tier_gib"] + p["gpu1_gib"]
    if p["layout"] == "split":
        line = (f"every expert in VRAM across the two cards (about {p['tier_gib']:.1f} GB on the first, "
                f"{p['gpu1_gib']:.1f} GB on the second; the second's also pinned in RAM for prompts)")
        word = "fast"
    else:
        line = (f"about {held:.1f} of the experts' {EXPERTS_GIB:.1f} GB on the two cards ({100.0 * held / EXPERTS_GIB:.0f}%"
                f"), the other {EXPERTS_GIB - held:.1f} GB pinned in RAM - slower than with every expert in VRAM")
        word = "runs"
    if ram_gb and ram_gb < p["ram_gib"] + OS_GIB:
        return "tight", line + f"; it takes about {p['ram_gib']:.0f} GB of RAM, this PC has {ram_gb:.0f} GB"
    return word, line


DRAFT_BASE_GIB = 0.86                               # the drafter (mtp.safetensors 0.59 GiB, its fold through the head,
DRAFT_TOKEN_BYTES = 4096                            # its 2 rings, its attention's partials, the taps): maya-ik2's
                                                    # draft_room; its layer without a window's context 4 KiB a token
                                                    # (its attention's logits in shared memory: no scratch a token)


PLAN_DENSE_GIB = 1.1                                # the engine's own sum for the drafter's room (maya-ik2's fit block):
PLAN_RESERVE_GIB = 1.5                              # the experts, the dense weights as blocks, the cache and its
                                                    # reserve (work buffers, room) - less than BASE_GIB's measured use
DRAFT_KEEP_GIB = 0.28125                            # + what the engine keeps free beside every expert at the drafter (256 + 32 MiB)
OFF_WIN_STAGE_GIB = 0.7                             # the offload path's verify window: 64 staging slots a slot size
DRAFT_MIN_ROWS = 65536                              # the fewest context rows the engine plans for the drafter there


def draft_gib(ctx: int) -> float:
    """GiB the drafter takes beside the model at `ctx` (maya-ik2's draft_room: it sees the whole context)."""
    return DRAFT_BASE_GIB + ctx * DRAFT_TOKEN_BYTES / GIB


def draft_fits(vram_gb: float, ctx: int, kv: str = DEFAULT_KV) -> bool:
    """Whether the drafter's room fits beside the model at `ctx` on this card - where it does not, the engine leaves
    it out at its start (the plan stays as without it; the prompt lookup still drafts).  With every expert in VRAM
    the engine's own check: its plan's sum (the experts, the dense weights, the cache, its reserve) and the drafter
    with FP16 context rows, else FP8 ones - so a 24 GB card drafts at 262,144 as the engine does.  A card whose VRAM
    is not known: True (the engine measures)."""
    if not vram_gb:
        return True
    p = plan(vram_gb, ctx, kv)
    if p["resident"]:    # (its context rows FP16, else FP8: half of them)
        # (the engine measures the free VRAM after its work buffers and the drafter: the attention's scratch grows with
        #  the context beyond the fixed reserve - SCRATCH_TOKEN_BYTES a token - and it keeps DRAFT_KEEP_GIB of room
        #  beside every expert: 256 MiB of the cache's and 32 MiB of slack.  Without both the tool said "fits" where
        #  the engine left the drafter out: 32 GB at 720,896 tokens, 0.03 GiB short.)
        used = (EXPERTS_GIB + PLAN_DENSE_GIB + ctx * KV_BYTES.get(kv, KV_BYTES["f16"]) / GIB + PLAN_RESERVE_GIB
                + ctx * SCRATCH_TOKEN_BYTES / GIB + DRAFT_KEEP_GIB)
        return float(vram_gb) - MARGIN_GIB >= used + draft_gib(ctx) - ctx * DRAFT_TOKEN_BYTES / GIB / 2
    # the offload path with the whole cache in VRAM: the tier gives the drafter's room, at most half of its own - the
    # verify window's staging and at least DRAFT_MIN_ROWS of FP8 context rows (the last positions) with it
    least = DRAFT_BASE_GIB + OFF_WIN_STAGE_GIB + min(ctx, DRAFT_MIN_ROWS) * DRAFT_TOKEN_BYTES / GIB / 2
    return not p["kv_streamed"] and (p["tier_gib"] + OFF_ROOM_GIB + MARGIN_GIB) / 2 >= least   # (the engine measures)


def drafting_off_reason(arch, vram_gb: float, ctx: int, kv: str = DEFAULT_KV, hip: bool = False,
                        apu: bool = False, gpus2: bool = False) -> str | None:
    """Why drafting is off on this card (one line), or None.  Every NVIDIA class the engine runs on drafts - Volta
    and Turing (sm_70 / sm_75) have their own verify-window kernels since v1.1.75.  The offload path (the card does
    not hold every expert: a small card, or a long context) drafts where its whole cache stays in VRAM and the experts'
    tier has the drafter's room, on AMD and NVIDIA alike, and on two NVIDIA cards (gpus2: the split, plan_two's "gpus"
    2 - its cache is whole in VRAM): --drafting auto measures it once on the PC (on NVIDIA against plain decoding in
    the PC's own mode - lane parity where that is faster - and keeps the faster, the same at every start; on a V100
    it may well keep plain decoding).  With every expert in VRAM it is kept (an AMD APU always).  A VRAM not known
    counts as not a reason; arch is no reason any more (kept for the callers)."""
    off_cache = ("drafting off: at this context part of the cache streams from RAM, where drafting does not run - a "
                 "smaller context or a bigger card brings it back")
    off_room = ("drafting off: at this context the experts' tier has no room for the drafter - a smaller context or a "
                "bigger card brings it back")
    if hip:
        if apu or not vram_gb or plan(vram_gb, ctx, kv)["resident"]:
            return None
        if plan(vram_gb, ctx, kv)["kv_streamed"]:
            return off_cache
        return None if draft_fits(vram_gb, ctx, kv) else off_room
    if vram_gb and not plan(vram_gb, ctx, kv)["resident"]:
        if gpus2:           # (the two-card split: its cache whole in VRAM, mode A's arithmetic already)
            return None if draft_fits(vram_gb, ctx, kv) else off_room
        if plan(vram_gb, ctx, kv)["kv_streamed"]:
            return off_cache
        if not draft_fits(vram_gb, ctx, kv):
            return off_room
    return None


def recommended_context(vram_gb: float, kv: str = DEFAULT_KV) -> int:
    """The largest of CONTEXTS whose cache leaves every expert in VRAM, else the smallest (the one that leaves the
    most experts on the card)."""
    fits = [c for c in CONTEXTS if c <= RECOMMEND_MAX and plan(vram_gb, c, kv)["resident"]
            and not plan(vram_gb, c, kv)["kv_streamed"]]
    return max(fits) if fits else CONTEXTS[0]


def amd_unsupported(cards: list, ram_gb: float = 0.0) -> str | None:
    """The AMD half of unsupported(): maya-ik2's HIP build (Linux) runs on RDNA 3 / 3.5 / 4 (gfx11 / gfx12: its prompt
    kernels are the matrix cores') - a card of 12 GB or more (under 24 GB the offload path: the experts it does not
    hold in RAM, as on NVIDIA), or an APU whose memory is the RAM with every expert in it ("unified_memory": its
    "memory_gb", what the engine may take of it).  cards: {"arch": "gfx1151", "vram_gb", ...}; ram_gb: the PC's RAM
    (0: not read yet)."""
    if not cards:
        return None
    rdna = [g for g in cards if str(g.get("arch") or "").startswith(AMD_ARCHS)]
    if not rdna:
        return ("Inkling 2 Mini on AMD needs a Radeon RX 7000 / 9000, a Radeon PRO W7000 / AI PRO R9700 or a Ryzen AI "
                "Max (Strix / Gorgon Halo) GPU - RDNA 3, 3.5 or 4 (gfx11 / gfx12); older AMD GPUs are not supported "
                "by its engine")
    apus = [g for g in rdna if g.get("unified_memory")]
    if apus and max(float(g.get("memory_gb") or 0) for g in apus) >= AMD_APU_MIN_GB:
        return None
    cards_ = [g for g in rdna if not g.get("unified_memory")]
    vram = max([float(g.get("vram_gb") or 0) for g in cards_] or [0.0])
    if vram < AMD_MIN_VRAM_GB:
        return ("Inkling 2 Mini needs one AMD GPU with 12 GB of VRAM or more, or a Ryzen AI Max APU with 24 GB of "
                "memory free for it")
    if ram_gb and vram - MARGIN_GIB < need_gib(CONTEXTS[0]) and ram_gb < plan(vram, CONTEXTS[0])["ram_gib"]:
        return too_small_line(ram_gb, vram)
    return None


def verdict(vram_gb: float, ram_gb: float, ctx: int, kv: str = DEFAULT_KV, apu: bool = False) -> tuple:
    """(word, line) for the setup and the Models page: "fast" (every expert in VRAM), "runs" (the offload path),
    "tight" (the offload path, with less RAM beside what it takes than the system needs - it may not start),
    "does not fit" (RAM under what the offload path takes with this card: RAM + VRAM under the model).  ram_gb 0: not known."""
    p = plan(vram_gb, ctx, kv, apu)
    if ram_gb and ram_gb < p["ram_gib"] and p["kv_streamed"]:
        return "does not fit", (f"at this context the cache keeps about {p['kv_ram_gib']:.0f} GB in RAM (KV "
                                f"streaming): about {p['ram_gib']:.0f} GB of RAM in all, this PC has {ram_gb:.0f} GB - "
                                "a smaller context")
    if p["resident"] and p["kv_streamed"]:
        return "streamed", (f"every expert in VRAM; about {p['kv_ram_gib']:.0f} GB of the cache in RAM (KV streaming), "
                            "read over the bus at every token - slower the deeper a chat goes")
    if p["resident"]:
        return "fast", "every expert in VRAM"
    if ram_gb and ram_gb < p["ram_gib"]:
        return "does not fit", too_small_line(ram_gb, vram_gb, p["ram_gib"])
    share = 100.0 * p["tier_gib"] / EXPERTS_GIB
    line = (f"about {p['tier_gib']:.1f} of the experts' {EXPERTS_GIB:.1f} GB on the card ({share:.0f}%), the other "
            f"{EXPERTS_GIB - p['tier_gib']:.1f} GB pinned in RAM - slower than with every expert in VRAM")
    if p["kv_streamed"]:
        line += f"; about {p['kv_ram_gib']:.0f} GB of the cache in RAM too (KV streaming)"
    else:
        line += "; the whole cache stays on the card"
    if ram_gb and ram_gb < p["ram_gib"] + OS_GIB:
        return "tight", line + f"; it takes about {p['ram_gib']:.0f} GB of RAM, this PC has {ram_gb:.0f} GB"
    return "runs", line


def too_small_line(ram_gb: float, vram_gb: float, need: float | None = None) -> str:
    need = plan(vram_gb, CONTEXTS[0])["ram_gib"] if need is None else need
    return (f"Inkling 2 Mini needs about {MIN_TOTAL_GB:.0f} GB of RAM and VRAM together to hold the model: about "
            f"{need:.0f} GB of RAM with this {vram_gb:.0f} GB card (this PC has {ram_gb:.0f} GB of RAM)")


def unsupported(backend: str, windows: bool, gpus_asked: int, cards: list, ram_gb: float = 0.0) -> str | None:
    """Why this PC cannot run Inkling 2, in one plain line, or None (said before anything is built or downloaded;
    the engine's own check_hardware says the same at its start).  backend: cuda / hip / metal; gpus_asked: how many GPUs
    --gpu / --gpus named (0: none named); cards: the GPUs the model would pick from (NVIDIA: {"arch", "vram_gb"},
    nvidia-smi's; AMD: amd_unsupported's) - empty when they are not read yet; ram_gb: the PC's RAM (0: not read yet;
    the offload path's)."""
    if backend == "metal":
        return ("Inkling 2 Mini runs on NVIDIA and AMD GPUs only for now (no Mac "
                "engine yet)")
    if backend == "hip":
        if windows:
            return ("Inkling 2 Mini on AMD runs on Linux only for now (its engine is not built for AMD on Windows "
                    "yet)")
        if gpus_asked > 1:
            return "Inkling 2 Mini on AMD runs on one GPU for now: pick one (--gpu N)"
        return amd_unsupported(cards, ram_gb)
    # (NVIDIA: Linux and Windows alike, the same card rules; two cards on Linux only - untested on Windows)
    if windows and gpus_asked > 1:
        return "Inkling 2 Mini on Windows runs on one GPU for now (two are Linux only): pick one (--gpu N)"
    if gpus_asked > 2:
        return ("Inkling 2 Mini runs on one or two NVIDIA GPUs for now (three or more GPUs are not supported): pick one or "
                "two (--gpu N, --gpus N,M)")
    if cards:
        new = [g for g in cards if int(g.get("arch") or 0) >= MIN_CC]
        if not new:
            return ("Inkling 2 Mini needs an NVIDIA V100 / RTX 20 or newer GPU (compute capability 7.0 or newer) - "
                    "older cards are not supported by its engine")
        vram = max(float(g.get("vram_gb") or 0) for g in new)
        if vram < MIN_VRAM_GB:
            return "Inkling 2 Mini needs one NVIDIA GPU with 8 GB of VRAM or more"
        if gpus_asked == 2 and min(float(g.get("vram_gb") or 0) for g in cards) < MIN_VRAM_GB:
            return "Inkling 2 Mini on two GPUs needs 8 GB of VRAM or more on each of them (or one GPU: --gpu N)"
        if ram_gb and vram - MARGIN_GIB < need_gib(CONTEXTS[0]) and ram_gb < plan(vram, CONTEXTS[0])["ram_gib"]:
            return too_small_line(ram_gb, vram)
    return None

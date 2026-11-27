"""tools/ling_fit.py - Unified memory & fit planner for Ling-3.2-air (bailingmoe3, 128B MoE) on maya-ling.

Single source of truth used by:
  - setup (maya.py: set_up_ling)
  - Models page / server (serve/models.py)
  - tests verifying setup claims match maya-ling's runtime start-plan at every offered context.

Exact Ling-3.2-air (42 trunk layers = 35 KDA + 7 MLA, 40 MoE layers x 512 experts top-8, optional MTP blk.42):
  - Non-expert trunk weights (Q8_0 dense + emb + lm_head): 3.73 GiB
  - KDA constant recurrent state (35 layers, context-independent): 74.92 MiB (0.073 GiB)
  - MLA KV cache across the 7 trunk MLA layers (kv_lora=512, n_rot=64):
      f16 (default): 7 * (512*2 + 64*2) = 8,064 B/tok (0.246 GiB @ 32K, 0.984 GiB @ 128K, 1.969 GiB @ 262K)
      q8  (opt-in):  7 * (512 + 32 + 128) = 4,704 B/tok (0.144 GiB @ 32K, 0.574 GiB @ 128K, 1.148 GiB @ 262K)
  - MTP drafter (blk.42, optional): 0.055 GiB dense + 1 MLA layer KV (1,152 B/tok f16; 672 B/tok q8),
    loaded AFTER trunk expert-pool planning from headroom so loading the drafter NEVER changes the trunk plan.
  - Vision tower (mmproj-BF16): 0.00 GiB permanent VRAM (~1.21 GiB lent transiently while encoding a picture).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

GIB = 1 << 30
MIB = 1 << 20

CONTEXTS = [8192, 16384, 32768, 65536, 131072, 262144]
DEFAULT_KV = "f16"
KV_FORMATS = ("f16", "q8")

# Exact bytes per token across the 7 trunk MLA layers
KV_BYTES = {
    "f16": 7 * (512 * 2 + 64 * 2),         # 8,064 B/tok (default, bit-faithful to FP16)
    "q8":  7 * (512 + (512 // 32) * 2 + 64 * 2),  # 4,704 B/tok (opt-in INT8 latent)
}
# Extra KV bytes per token for optional MTP blk.42 (1 MLA layer)
MTP_KV_BYTES = {
    "f16": 512 * 2 + 64 * 2,               # 1,152 B/tok
    "q8":  512 + (512 // 32) * 2 + 64 * 2, # 672 B/tok
}

N_LAYERS = 42
N_KDA_LAYERS = 35
N_MLA_LAYERS = 7
N_MOE_LAYERS = 40
N_EXPERTS = 512
N_ACTIVE = 8
TOTAL_TRUNK_EXPERTS = N_MOE_LAYERS * N_EXPERTS  # 20,480

EMB_GIB = 0.398
OUT_GIB = 0.398
TRUNK_BODY_DENSE_GIB = 2.933
DENSE_GIB = EMB_GIB + OUT_GIB + TRUNK_BODY_DENSE_GIB  # 3.729 GiB
KDA_STATE_GIB = (N_KDA_LAYERS * (32 * 128 * 128 + 3 * 4096 * 3) * 4) / GIB  # 0.0732 GiB
MTP_DENSE_GIB = 0.055
VISION_TRANSIENT_GIB = 1.21
CUDA_CTX_GIB = 0.40
HEADROOM_GIB = 1.25  # matches Model::Model headroom (absorbs optional MTP dense+KV so trunk plan is MTP-invariant)
RAM_SPARE_GIB = 8.0
MIN_VRAM_GIB = 7.5   # 8 GB class card minimum (zero-expert VRAM pool works)
MIN_CC = 70          # Volta / Turing+ (sm_70+)
# AMD (HIP, Linux): wave32 RDNA 3 / 3.5 / 4 (gfx11 / gfx12), the HIP build's own family - run on a Gorgon Halo's
# Radeon 8065S (gfx1151); one GPU for now (the layer split across AMD cards is not run yet)
AMD_ARCHS = ("gfx11", "gfx12")

# Neutral fallback sizing when no local GGUF file size is supplied (maintainers add official quants in download.json).
DEFAULT_EXPERTS_GIB = 52.0
DEFAULT_QUANT = "local"


def download_manifest() -> dict:
    p = Path(__file__).resolve().parent / "ling" / "download.json"
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {"name": "Ling-3.2-air", "default_kv": DEFAULT_KV, "quants": {}}


def quant_files(quant: str | None, manifest: dict | None = None) -> dict:
    """A size's files as the table lists them ({name: [bytes, sha256]}); {} for a size not made yet or not listed."""
    q = ((manifest or download_manifest()).get("quants") or {}).get(quant or "") or {}
    return dict(q.get("files") or {})


def listed(manifest: dict | None = None) -> list:
    """The sizes whose files the table lists (their sizes and sha256 known), in the table's order."""
    m = manifest or download_manifest()
    return [q for q in (m.get("quants") or {}) if quant_files(q, m)]


def published(quant: str | None, manifest: dict | None = None) -> bool:
    """Whether the setup may download a size: its files are listed and the table names the repo and its commit."""
    m = manifest or download_manifest()
    return bool(quant_files(quant, m) and m.get("repo") and m.get("revision"))


def quant_spec(quant: str | None = None, gguf_gib: float | None = None,
               experts_gib: float | None = None) -> dict:
    """What a size takes: the table's own figures where it gives them, else from its files' bytes (a listed size) or
    the GGUF on the disk (gguf_gib); a size with no file yet has no figure of its own (the neutral fallback)."""
    manifest = download_manifest()
    q = (manifest.get("quants") or {}).get(quant or "") or {}
    if q.get("experts_gib"):
        return {
            "dense_gib": float(q.get("dense_gib", DENSE_GIB)),
            "experts_gib": float(q["experts_gib"]),
            "memory_gb": float(q.get("memory_gb", DENSE_GIB + float(q["experts_gib"]) + 4.0)),
        }
    files = quant_files(quant, manifest)
    if files and not (gguf_gib and gguf_gib > 0) and not (experts_gib and experts_gib > 0):
        gguf_gib = sum(int(v[0]) for v in files.values()) / GIB
    if experts_gib is not None and experts_gib > 0:
        eg = float(experts_gib)
    elif gguf_gib is not None and gguf_gib > 0:
        eg = max(1.0, float(gguf_gib) - DENSE_GIB)
    else:
        eg = DEFAULT_EXPERTS_GIB
    return {
        "dense_gib": round(DENSE_GIB, 2),
        "experts_gib": round(eg, 2),
        "memory_gb": round(DENSE_GIB + eg + 4.0, 1),
    }


def normalize_kv(kv: str | None) -> str:
    k = str(kv or DEFAULT_KV).strip().lower()
    if k in ("q8", "int8"):
        return "q8"
    return "f16"


def kv_gib(ctx: int, kv: str = DEFAULT_KV, include_mtp: bool = False) -> float:
    k = normalize_kv(kv)
    b = int(ctx) * (KV_BYTES[k] + (MTP_KV_BYTES[k] if include_mtp else 0))
    return b / GIB


def _prefill_chunk_for_free(free_after_weights_gib: float) -> tuple[int, float]:
    """Mirror Model::Model adaptive prefill chunk P and its scratch footprint per GPU."""
    if free_after_weights_gib < 3.0:
        return 128, 0.35
    if free_after_weights_gib < 6.0:
        return 256, 0.50
    if free_after_weights_gib < 10.0:
        return 512, 0.72
    if free_after_weights_gib < 14.0:
        return 1024, 1.05
    return 2048, 1.55


def _layer_split(n_gpu: int) -> list[tuple[int, int]]:
    """Mirror Model::Model layer split across GPUs (matches ling_model.cu)."""
    n_gpu = max(1, int(n_gpu))
    out = []
    for gi in range(n_gpu):
        l0 = (gi * N_LAYERS) // n_gpu
        l1 = ((gi + 1) * N_LAYERS) // n_gpu
        out.append((l0, l1))
    return out


def plan(vram_gbs: list[float] | float, ctx: int, kv: str = DEFAULT_KV, quant: str | None = None,
         ram_gb: float = 0.0, reserve_gb: float = 0.0, gguf_gib: float | None = None,
         experts_gib: float | None = None) -> dict:
    """Compute exact per-GPU & host placement plan matching maya-ling's runtime planner.

    Invariant: trunk KV format and trunk_vram_slots do NOT depend on whether MTP drafting is loaded.
    """
    kv = normalize_kv(kv)
    qs = quant_spec(quant, gguf_gib=gguf_gib, experts_gib=experts_gib)
    gbs = [float(vram_gbs)] if isinstance(vram_gbs, (int, float)) else [float(x) for x in vram_gbs]
    if not gbs:
        gbs = [0.0]
    ngpu = len(gbs)
    splits = _layer_split(ngpu)
    kv_per_mla_gib = (int(ctx) * (KV_BYTES[kv] / N_MLA_LAYERS)) / GIB
    kda_per_layer_gib = KDA_STATE_GIB / N_KDA_LAYERS
    exp_per_layer_gib = qs["experts_gib"] / N_MOE_LAYERS
    exp_blob_gib = exp_per_layer_gib / N_EXPERTS
    # MLA layer indices in Ling-3.2-air: layers 5, 11, 17, 23, 29, 35, 41 (1-based every 6th -> 0-based 5,11,17,23,29,35,41)
    mla_layers = {5, 11, 17, 23, 29, 35, 41}

    per_gpu = []
    total_vram_slots = 0
    total_vram_exp_gib = 0.0
    total_non_exp_gib = 0.0
    fits_trunk = True

    for gi, (vram_gb, (l0, l1)) in enumerate(zip(gbs, splits)):
        n_l = l1 - l0
        n_mla = sum(1 for il in range(l0, l1) if il in mla_layers)
        n_kda = n_l - n_mla
        n_moe = max(0, l1 - max(l0, 2))
        dense_g = (EMB_GIB if gi == 0 else 0.0) + (OUT_GIB if gi == ngpu - 1 else 0.0) + \
            TRUNK_BODY_DENSE_GIB * (n_l / N_LAYERS)
        state_kv_g = n_mla * kv_per_mla_gib + n_kda * kda_per_layer_gib
        eff_vram = max(0.0, vram_gb - float(reserve_gb))
        free_before_scratch = eff_vram - CUDA_CTX_GIB - dense_g - state_kv_g
        p_chunk, scratch_g = _prefill_chunk_for_free(max(0.0, free_before_scratch))
        need_non_exp = CUDA_CTX_GIB + dense_g + state_kv_g + scratch_g
        free_before_pool = eff_vram - need_non_exp
        if free_before_pool < 0.25:
            fits_trunk = False
        budget = max(0.0, free_before_pool - HEADROOM_GIB)
        gpu_moe_gib = n_moe * exp_per_layer_gib
        frac = min(1.0, budget / gpu_moe_gib) if gpu_moe_gib > 0 else 0.0
        slots_per_layer = min(N_EXPERTS, int(math.floor(frac * N_EXPERTS)))
        gpu_slots = slots_per_layer * n_moe
        gpu_exp_gib = gpu_slots * exp_blob_gib
        total_vram_slots += gpu_slots
        total_vram_exp_gib += gpu_exp_gib
        total_non_exp_gib += need_non_exp
        per_gpu.append({
            "gpu": gi,
            "layers": [l0, l1],
            "vram_gb": round(vram_gb, 2),
            "dense_gib": round(dense_g, 2),
            "kv_gib": round(state_kv_g, 2),
            "scratch_gib": round(scratch_g, 2),
            "prefill_chunk": p_chunk,
            "free_before_pool_gib": round(max(0.0, free_before_pool), 2),
            "vram_slots": gpu_slots,
            "moe_experts": n_moe * N_EXPERTS,
            "vram_exp_gib": round(gpu_exp_gib, 2),
        })

    # Host RAM vs SSD split (matching Model::Model 8 GiB RAM spare)
    offload_layers = [g for g in per_gpu if g["vram_slots"] < g["moe_experts"]]
    all_resident = len(offload_layers) == 0 and fits_trunk
    ram_budget = max(0.0, float(ram_gb) - RAM_SPARE_GIB) if ram_gb > 0 else qs["experts_gib"]
    ram_used_gib = 0.0
    ssd_used_gib = 0.0
    ram_slots = 0
    ssd_slots = 0
    for g in per_gpu:
        n_moe = g["moe_experts"] // N_EXPERTS
        n_res_l = (g["vram_slots"] // n_moe) if n_moe > 0 else N_EXPERTS
        if n_res_l >= N_EXPERTS:
            continue
        non_vram = N_EXPERTS - n_res_l
        for _ in range(n_moe):
            if ram_used_gib + exp_per_layer_gib <= ram_budget:
                ram_used_gib += exp_per_layer_gib
                ram_slots += non_vram
            else:
                ssd_used_gib += non_vram * exp_blob_gib
                ssd_slots += non_vram

    vram_pct = 100.0 * total_vram_slots / TOTAL_TRUNK_EXPERTS
    active_tok_gib = (qs["experts_gib"] / N_EXPERTS) * N_ACTIVE + qs["dense_gib"]
    return {
        "fits_trunk": fits_trunk,
        "resident": all_resident,
        "kv": kv,
        "ctx": int(ctx),
        "kv_bytes_per_tok": KV_BYTES[kv],
        "kv_gib": round(kv_gib(ctx, kv), 3),
        "kda_state_gib": round(KDA_STATE_GIB, 3),
        "dense_gib": round(qs["dense_gib"], 2),
        "experts_gib": round(qs["experts_gib"], 2),
        "active_per_tok_gib": round(active_tok_gib, 2),
        "mtp_gib": round(MTP_DENSE_GIB + kv_gib(ctx, kv, include_mtp=True) - kv_gib(ctx, kv), 3),
        "vision_permanent_gib": 0.0,
        "vision_transient_gib": VISION_TRANSIENT_GIB,
        "vram_slots": total_vram_slots,
        "total_experts": TOTAL_TRUNK_EXPERTS,
        "vram_pct": round(vram_pct, 1),
        "vram_exp_gib": round(total_vram_exp_gib, 2),
        "ram_slots": ram_slots,
        "ram_gib": round(ram_used_gib, 2),
        "ssd_slots": ssd_slots,
        "ssd_gib": round(ssd_used_gib, 2),
        "need_non_exp_gib": round(total_non_exp_gib, 2),
        "gpus": ngpu,
        "per_gpu": per_gpu,
    }


def recommended_context(vram_gbs: list[float] | float, kv: str = DEFAULT_KV, quant: str | None = None) -> int:
    """Choose default context that leaves healthy VRAM room for experts (full 262,144 stays available as an option)."""
    gbs = [float(vram_gbs)] if isinstance(vram_gbs, (int, float)) else [float(x) for x in vram_gbs]
    tot = sum(gbs)
    min_g = min(gbs) if gbs else 0.0
    if min_g < 10.0:
        return 16384
    if min_g < 16.0:
        return 32768
    if tot >= 44.0:
        return 131072
    return 32768  # 1x 24 GB card: 32K FP16 KV = 0.25 GiB, leaving ~17+ GiB for experts


def unsupported(backend: str, win: bool, gpus: list[dict], ram_gb: float = 0.0, avx2: bool = True) -> str | None:
    """Clear refusal message for unsupported machines (None = supported).  AMD (backend "hip"): the setup's amd_gpus
    entries ({"arch": "gfx1151", "vram_gb", "unified_memory": an APU whose memory is the RAM - no VRAM minimum}).
    Windows: NVIDIA only (maya-ling is not built for AMD there)."""
    if backend == "hip":
        if win:
            return ("Ling-3.2-air on AMD runs on Linux only for now; on Windows maya-ling runs on NVIDIA GPUs only")
        if len(gpus) > 1:
            return "Ling-3.2-air on AMD runs on one GPU for now: pick one (--gpu N)"
        if gpus and not str(gpus[0].get("arch") or "").startswith(AMD_ARCHS):
            return ("Ling-3.2-air on AMD needs a Radeon RX 7000 / 9000, a Radeon PRO W7000 / AI PRO R9700 or a Ryzen "
                    "AI Max (Strix / Gorgon Halo) GPU - RDNA 3, 3.5 or 4 (gfx11 / gfx12); other AMD GPUs are not "
                    "supported by its engine yet")
        v = float(gpus[0].get("vram_gb") or 0) if gpus else 0.0
        if gpus and not gpus[0].get("unified_memory") and 0 < v < MIN_VRAM_GIB:
            return (f"GPU {gpus[0].get('name', '?')} has {v:.1f} GB VRAM (minimum {MIN_VRAM_GIB:.1f} GB for "
                    "Ling-3.2-air trunk + KV)")
        if not avx2:
            return "This CPU lacks AVX2; maya-ling's CPU expert lane requires AVX2"
        return None
    if win and backend != "cuda":
        return "Ling-3.2-air (maya-ling) on Windows runs on NVIDIA GPUs only; on an AMD GPU it needs Linux for now"
    if backend == "metal":
        return "Ling-3.2-air (maya-ling) requires an NVIDIA CUDA GPU (compute capability >= 7.0); Metal is not supported yet"
    if not gpus:
        return "Ling-3.2-air requires at least one NVIDIA CUDA GPU with >= 8 GB VRAM"
    for g in gpus:
        cc = g.get("cc") or g.get("compute_cap")
        if cc is not None:
            try:
                cc_num = int(str(cc).replace(".", ""))
                if cc_num < 10:
                    cc_num *= 10
                if cc_num < MIN_CC:
                    return f"GPU {g.get('name', '?')} has compute capability {cc} (< 7.0); maya-ling requires Volta/Turing (sm_70) or newer"
            except ValueError:
                pass
        v = float(g.get("vram_gb") or 0)
        if v > 0 and v < MIN_VRAM_GIB:
            return f"GPU {g.get('name', '?')} has {v:.1f} GB VRAM (minimum {MIN_VRAM_GIB:.1f} GB for Ling-3.2-air trunk + KV)"
    if not avx2:
        return "This CPU lacks AVX2; maya-ling's CPU expert lane requires AVX2"
    return None


def smaller_quant_hint(vram_gbs: list[float] | float, ram_gb: float, ctx: int, kv: str,
                       current_quant: str | None = None, gguf_gib: float | None = None) -> str:
    """Suggest a smaller published quant (when download.json lists any), shorter context, or --kv q8."""
    sizes = {q: quant_spec(q)["memory_gb"] for q in listed()}   # (only the sizes that exist)
    if sizes:
        cur_p = plan(vram_gbs, ctx, kv, current_quant, ram_gb=ram_gb, gguf_gib=gguf_gib)
        cur_mem = quant_spec(current_quant, gguf_gib=gguf_gib)["memory_gb"]
        for q, mem in sorted(sizes.items(), key=lambda kvp: kvp[1]):
            if mem < cur_mem:
                p = plan(vram_gbs, ctx, kv, q, ram_gb=ram_gb)
                if p["fits_trunk"] and (p["ssd_slots"] == 0 or p["ssd_slots"] < cur_p["ssd_slots"]):
                    note = "fits RAM without SSD streaming" if p["ssd_slots"] == 0 else f"cuts SSD spill to {p['ssd_gib']:.1f} GiB"
                    return f"try {q} (~{mem:.0f} GB total, {note})"
    if ctx > 16384:
        return "try a smaller quant, a shorter context (--context 16384 or 32768), or --kv q8"
    return "try a smaller quant or use --force-fit to proceed anyway"


def verdict(vram_gbs: list[float] | float, ram_gb: float, ctx: int, kv: str = DEFAULT_KV,
            quant: str | None = None, reserve_gb: float = 0.0, gguf_gib: float | None = None) -> tuple[str, str]:
    """Return (verdict, human_line): verdict in {'fits', 'runs slowly', 'does not fit'}."""
    p = plan(vram_gbs, ctx, kv, quant, ram_gb=ram_gb, reserve_gb=reserve_gb, gguf_gib=gguf_gib)
    if not p["fits_trunk"]:
        hint = smaller_quant_hint(vram_gbs, ram_gb, ctx, kv, quant, gguf_gib=gguf_gib)
        return "does not fit", (
            f"trunk + KV ({p['need_non_exp_gib']:.1f} GiB non-expert) does not fit available VRAM at {ctx} ctx ({kv}); {hint}"
        )
    if p["resident"]:
        return "fits", (
            f"all {p['total_experts']} routed experts fit in VRAM ({p['vram_exp_gib']:.1f} GiB) at {ctx} ctx ({kv.upper()} KV {p['kv_gib']:.2f} GiB)"
        )
    if p["vram_slots"] == 0:
        hint = smaller_quant_hint(vram_gbs, ram_gb, ctx, kv, quant, gguf_gib=gguf_gib)
        where = f"RAM + SSD ({p['ssd_gib']:.1f} GiB on NVMe)" if p["ssd_slots"] > 0 else f"host RAM ({p['ram_gib']:.1f} GiB)"
        return "runs slowly", (
            f"zero-expert VRAM pool (trunk + KV leave no room for expert slots); all routed experts run from {where}; {hint}"
        )
    if p["ssd_slots"] > 0:
        hint = smaller_quant_hint(vram_gbs, ram_gb, ctx, kv, quant, gguf_gib=gguf_gib)
        return "runs slowly", (
            f"{p['vram_pct']:.1f}% of experts in VRAM ({p['vram_exp_gib']:.1f} GiB), "
            f"{p['ram_gib']:.1f} GiB in RAM, {p['ssd_gib']:.1f} GiB streamed from NVMe SSD; {hint}"
        )
    return "fits", (
        f"{p['vram_pct']:.1f}% of experts in VRAM ({p['vram_slots']}/{p['total_experts']}, {p['vram_exp_gib']:.1f} GiB), "
        f"rest in host RAM ({p['ram_gib']:.1f} GiB), {kv.upper()} KV {p['kv_gib']:.2f} GiB"
    )

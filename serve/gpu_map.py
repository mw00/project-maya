"""The Monitor's map of a GPU ("What fills the GPU"): the GLM engine's VRAM lines - what each part of the model holds on
its card, measured once it has loaded - joined with what NVML sees on that card now: the engine's own process, the
vision encoder's, other programs and the free memory.  What the engine's parts do not explain of its own process is the
CUDA context and the small buffers (the conversation snapshots, the KV streaming maps, the sampler); with the memory the
driver keeps outside every process (its reserve), that is "CUDA and the rest"."""
from __future__ import annotations

# the VRAM line's byte counts (Glm5Model::VramReport, src/program/generate.cpp; strata-ds4's info_lines, whose vision
# tower runs in the engine's own process: vision_own, and which also tells its prompt chunks' buffers apart (prompt)
# and what its tensor parallel copies (tp, of a group of tp_n GPUs - out of its weights and kv))
BYTES = ("total", "free", "weights", "kv", "kv_ram", "act", "drafter", "window", "pool", "tail", "borrow", "vision",
         "vision_own", "prompt", "tp", "helper")


def parse(line: str) -> dict | None:
    """`VRAM dev=0 pci=0000:01:00.0 total=... layers=0-45` -> {"dev": 0, "pci": "0000:01:00", ..., "layers": [0, 45]};
    None for a line that is not one."""
    if not line.startswith("VRAM "):
        return None
    out: dict = {}
    for kv in line.split()[1:]:
        k, _, v = kv.partition("=")
        if k == "pci":
            out["pci"] = pci_key(v)
        elif k == "layers":
            a, _, b = v.partition("-")
            out["layers"] = [int(a), int(b)] if a.isdigit() and b.isdigit() else None
        elif v.lstrip("-").isdigit():
            out[k] = int(v)
    return out if "dev" in out else None


def pci_key(s: str | None) -> str | None:
    """CUDA's "0000:01:00.0" and NVML's "00000000:01:00.0" (or the telemetry's "0000:01:00") as "0000:01:00"."""
    if not s or s == "-":
        return None
    f = s.strip().lower().split(":")
    if len(f) < 3:
        return None
    try:
        return f"{int(f[-3], 16):04x}:{int(f[-2], 16):02x}:{int(f[-1].split('.')[0], 16):02x}"
    except ValueError:
        return None


def merge(lines: list[dict]) -> list[dict]:
    """One entry a device: two parts of a split on one card add up (the device's own figures are taken once)."""
    by: dict[int, dict] = {}
    for r in lines:
        d = by.get(r["dev"])
        if d is None:
            by[r["dev"]] = dict(r)
            continue
        for k in BYTES:
            if k not in ("total", "free"):
                d[k] = d.get(k, 0) + r.get(k, 0)
        d["slots"] = d.get("slots", 0) + r.get("slots", 0)
        d["free"] = min(d.get("free", 0), r.get("free", 0))
        if d.get("layers") and r.get("layers"):
            d["layers"] = [min(d["layers"][0], r["layers"][0]), max(d["layers"][1], r["layers"][1])]
    return [by[k] for k in sorted(by)]


def as_pids(p) -> set:
    """A pid, a collection of pids or None, as a set."""
    return set() if p is None else {p} if isinstance(p, int) else set(p)


def held_by(procs: dict, pids: set) -> int | None:
    """What the processes `pids` hold on a card, as NVML lists them; None when it lists none of them (or can't say)."""
    b = [procs[p] for p in pids if p in procs]
    return None if not b or None in b else sum(b)


def view(report: list[dict], cards: list[dict] | None, engine_pid, vision_pid=None,
         vision_lent: bool = False) -> list[dict]:
    """Each GPU the engine reports, as the Monitor draws it: its NVML number and name, the total, and the bytes of each
    part - the engine's (weights, kv, act, prompt, tp, drafter + window, pool), the vision encoder's, the CUDA context
    and the rest, other programs, free.  `cards`: the telemetry's per-card readings ([] without NVML: the start's
    measurement only); `engine_pid`, `vision_pid`: the pid - or the pids: a launcher and what it started - NVML may
    list the engine's and the encoder's process under; `vision_lent`: the on-demand encoder has the first GPU's pool
    tail now (the engine freed it).  A card that lists none of the engine's pids (NVML can't list processes, or names
    them in a pid namespace the server does not share) is drawn from the start's measurement."""
    cards = cards or []
    mine, enc_pids = as_pids(engine_pid), as_pids(vision_pid)
    by_pci = {c.get("pci"): c for c in cards if c.get("pci")}
    out = []
    for k, r in enumerate(merge(report)):
        card = by_pci.get(r.get("pci")) or (cards[k] if k < len(cards) and not by_pci else None)
        total = (card or {}).get("total") or r.get("total", 0)
        used = (card or {}).get("used")
        procs = (card or {}).get("procs") or {}
        lent = bool(vision_lent and r.get("vision") and r["dev"] == 0)
        pool = max(0, r.get("pool", 0) - (r.get("tail", 0) if lent else 0))
        parts = {"weights": r.get("weights", 0), "kv": r.get("kv", 0), "act": r.get("act", 0),
                 "prompt": r.get("prompt", 0), "tp": r.get("tp", 0),
                 "drafter": r.get("drafter", 0) + r.get("window", 0), "pool": pool, "helper": r.get("helper", 0)}
        engine = sum(parts.values())
        own, enc = held_by(procs, mine), held_by(procs, enc_pids)
        inside = r.get("vision_own", 0) if enc is None else 0   # (a vision tower in the engine's own process)
        engine += inside
        vision = enc if enc is not None else r.get("tail", 0) if lent else inside
        start = r.get("total", 0) - r.get("free", 0)     # what the card held once the engine had loaded
        if used is None:                                 # (no NVML: that measurement - a lent tail moved, not added)
            used = start
        if own is not None:                              # (NVML says what the engine's process holds here)
            # other programs: what other processes hold, as NVML lists them - the card's used memory beyond every
            # process (the driver's reserve, ~0.45 GB a card; what it keeps outside them) is the rest's, not theirs
            held = [b for pid, b in procs.items() if pid not in mine and pid not in enc_pids]
            other = sum(held) if None not in held else max(0, used - own - (enc or 0))
            rest = max(0, used - engine - (enc or 0) - other)
        else:                                            # (it can't: what the parts did not explain at the start -
            rest = max(0, start - engine - (vision if lent else 0))   # other programs then among it)
            other = max(0, used - engine - rest - (vision - inside))
        parts.update(vision=vision, rest=rest, other=other)
        over = sum(parts.values()) - total               # (never wider than the card: what NVML rounds, trimmed)
        for key in ("other", "rest"):
            cut = min(max(0, over), parts[key])
            parts[key] -= cut
            over -= cut
        out.append({"dev": r["dev"], "index": (card or {}).get("index"), "name": (card or {}).get("name"),
                    "total": total, "free": max(0, total - sum(parts.values())), "parts": parts,
                    "window": r.get("window", 0), "drafter": r.get("drafter", 0), "tail": r.get("tail", 0),
                    "borrow": r.get("borrow", 0), "lendable": r.get("vision", 0), "vision_lent": lent,
                    "vision_own": inside,
                    "slots": r.get("slots", 0), "layers": r.get("layers"), "ctx": r.get("ctx", 0),
                    "tp_n": r.get("tp_n", 0),
                    "kv_ram": r.get("kv_ram", 0), "measured": "nvml" if own is not None else "start"})
    return out

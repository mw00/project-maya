"""The dashboard on Maya's screen once it runs (setup_tui.py): the web Monitor's numbers (serve/web/app.js, render) from
the server's GET /metrics, read once a second - the throughput, where the experts live, the GPUs, the hardware and the
recent requests - in binsider's boxes and the dashboard's colours (setup_look.py).  While the model loads, a box of its
own instead: how far the start is (the engine log's warm-up of each GPU's experts), which GPU, how long."""
from __future__ import annotations

import json
import threading
import time
import urllib.request
from datetime import datetime

from rich.console import Group
from rich.table import Table
from rich.text import Text
from textual.app import ComposeResult
from textual.containers import Center, Horizontal, VerticalScroll
from textual.widgets import Static

from setup_bridge import LoadWatch
from setup_look import (ACCENT, ACCENT2, ACCENT_TEXT, DANGER, FAINT, INFO_TEXT, INK, INK_SOFT, MUTED, OK, QUADRANTS,
                        WARN, Card, bar, clock, gradient, mix, title)

# font8x8's digits (public domain, as the logo's letters), for the throughput in big
DIGITS = {"0": (0x3E, 0x63, 0x73, 0x7B, 0x6F, 0x67, 0x3E, 0x00), "1": (0x0C, 0x0E, 0x0C, 0x0C, 0x0C, 0x0C, 0x3F, 0x00),
          "2": (0x1E, 0x33, 0x30, 0x1C, 0x06, 0x33, 0x3F, 0x00), "3": (0x1E, 0x33, 0x30, 0x1C, 0x30, 0x33, 0x1E, 0x00),
          "4": (0x38, 0x3C, 0x36, 0x33, 0x7F, 0x30, 0x78, 0x00), "5": (0x3F, 0x03, 0x1F, 0x30, 0x30, 0x33, 0x1E, 0x00),
          "6": (0x1C, 0x06, 0x03, 0x1F, 0x33, 0x33, 0x1E, 0x00), "7": (0x3F, 0x33, 0x30, 0x18, 0x0C, 0x0C, 0x0C, 0x00),
          "8": (0x1E, 0x33, 0x33, 0x1E, 0x33, 0x33, 0x1E, 0x00), "9": (0x1E, 0x33, 0x33, 0x3E, 0x30, 0x18, 0x0E, 0x00),
          ".": (0x00, 0x00, 0x00, 0x00, 0x00, 0x0C, 0x0C, 0x00), "-": (0x00, 0x00, 0x00, 0x3F, 0x00, 0x00, 0x00, 0x00),
          "%": (0x00, 0x63, 0x33, 0x18, 0x0C, 0x66, 0x63, 0x00)}
BLOCKS = "▁▂▃▄▅▆▇█"
TIERS = (("ram_fetch", "From RAM", INFO_TEXT, "/tok"), ("disk", "From SSD", WARN, "/tok"),
         ("promo", "Promoted", ACCENT, "/tok"))   # per token
# DeepSeek's (maya-ds4): experts the CPU lane computed, MB the storage read (Linux), the cache's moves, the n-gram
# memory's (Engram) read time - each a token
TIERS_DS4 = (("cpu", "On the CPU", INFO_TEXT, "/tok"), ("disk_mb", "Disk reads", WARN, " MB/tok"),
             ("promo", "Promoted", ACCENT, "/tok"), ("engram_ms", "N-gram memory", ACCENT, " ms/tok"))
# MiniMax's (maya-mm3): GLM's keys, but the experts VRAM does not hold are computed on the CPU from pinned RAM
TIERS_MM3 = (("ram_fetch", "On the CPU", INFO_TEXT, "/tok"), ("disk", "From SSD", WARN, "/tok"),
             ("promo", "Promoted", ACCENT, "/tok"))
TIPS = {   # the Monitor's advice under the tiers, said shorter as the box narrows (one line: the longest that fits)
    "disk": ("Many experts are read from the SSD for every token: more RAM, a smaller model or a shorter context "
             "makes answers faster.", "Many experts come from the SSD: more RAM or a shorter context is faster.",
             "Experts come from the SSD: more RAM helps."),
    "ram": ("Most misses come from RAM over PCIe: more VRAM (or a second GPU) keeps more experts on the card.",
            "Most misses come from RAM: more VRAM keeps more experts on the GPU.",
            "Misses come from RAM: more VRAM helps."),
    "ds4_disk": ("Much of every token is read from the SSD: more RAM keeps more of the model's experts cached, and "
                 "answers get faster.", "Much of every token comes from the SSD: more RAM is faster.",
                 "Reads from the SSD: more RAM helps."),
    "ds4_cpu": ("Most experts run on the CPU: more VRAM (or a second GPU) keeps more of them on the card.",
                "Most experts run on the CPU: more VRAM keeps more on the GPU.", "CPU-bound: more VRAM helps.")}
FINISH = {"stop": ("Done", OK), "length": ("Max tokens", INK_SOFT), "cancel": ("Stopped", WARN),
          "disconnect": ("Closed", WARN), "error": ("Error", DANGER)}
API = {"web": "Chat", "openai": "OpenAI", "anthropic": "Anthropic"}


def num(n):
    """A number from /metrics - the engine's INFO facts come as text ("107.8") - else None."""
    try:
        return None if n is None or isinstance(n, bool) else float(n)
    except (TypeError, ValueError):
        return None


def fmt(n, d: int = 0) -> str:
    n = num(n)
    return "–" if n is None else f"{n:,.{d}f}"


def kfmt(n) -> str:
    return "–" if n is None else f"{n / 1000:,.{0 if n >= 10000 else 1}f}k" if n >= 1000 else fmt(n)


def ctxfmt(n) -> str:
    return f"{n // 1024}K" if n and n % 1024 == 0 else kfmt(n)


def gb(b, d: int = 1) -> str:                           # binary GB, as the web Monitor shows memory
    return "–" if b is None else fmt(b / 2**30, d)


def big(s: str) -> list:
    """`s` (digits, ".", "-") in font8x8, 2 x 2 pixels to a character; a "." keeps only its own two columns."""
    rows = [""] * 4
    for c in s:
        px = [[DIGITS[c][r] >> b & 1 for b in range(8)] for r in range(8)]
        cells = range(1, 2) if c == "." else range(4)
        for i, r in enumerate(range(0, 8, 2)):
            rows[i] += "".join(QUADRANTS[px[r][2 * x] | px[r][2 * x + 1] << 1 | px[r + 1][2 * x] << 2 |
                                         px[r + 1][2 * x + 1] << 3] for x in cells)
    return rows


def spark(values, width: int, top=None, color: str | None = None) -> Text:
    """A sparkline of the last minute, one character a column (the newest on the right); nothing to draw: a faint
    baseline."""
    v = [0.0 if x is None else float(x) for x in (values or [])]
    if len(v) > width:                                  # (more samples than columns: each column's peak)
        step = len(v) / width
        v = [max(v[int(i * step):max(int(i * step) + 1, int((i + 1) * step))]) for i in range(width)]
    hi = max([top or 0.0, *v, 1e-9])
    t = Text(" " * (width - len(v)))
    for i, x in enumerate(v):
        level = min(len(BLOCKS) - 1, int(x / hi * (len(BLOCKS) - 1) + 0.5)) if x > 0 else -1
        t.append(BLOCKS[max(level, 0)], FAINT if level < 0 else color or mix(ACCENT, ACCENT2, i / max(1, len(v) - 1)))
    return t


def prefill_rate(r: dict):
    """A request's prefill: the server's figure, else (as the web Monitor) the tokens it read over its prompt time."""
    if r.get("prefill_tok_s") is not None:
        return r["prefill_tok_s"]
    read = (r.get("prompt_tokens") or 0) - (r.get("reused") or 0)
    return read / (r["prompt_ms"] / 1000) if (r.get("prompt_ms") or 0) > 0 and read > 0 else None


def clip(s: str, width: int) -> str:
    return s if len(s) <= width else s[:max(0, width - 1)] + "…"


def label(k: str, v, unit: str = "", width: int = 0) -> Text:
    return Text.assemble((f"{k}  ".ljust(width), MUTED), (v if isinstance(v, str) else fmt(v), f"bold {INK}"),
                         (unit if unit == "%" else f" {unit}" if unit else "", MUTED))


def speed(m: dict, width: int):
    """The Monitor's hero: the decode speed in big (now, else the last request's), the prefill, the last minute,
    what the model is doing, and four headline numbers."""
    live, h, eng, totals = m.get("live") or {}, m.get("history") or {}, m.get("engine") or {}, m.get("totals") or {}
    last = (m.get("requests") or [None])[0]
    tn, state = (m.get("tiers") or {}).get("now"), live.get("state")
    gen, reading = state == "generating", state == "reading"
    rate = live.get("tok_s") if gen else last and last.get("decode_tok_s")
    prefill = live.get("prefill_tok_s_mean") if state != "idle" else last and prefill_rate(last)
    side = [Text("tok/s", f"bold {MUTED}"),
            Text("decode · " + ("writing now" if gen else "reading the prompt" if reading else
                                "last request" if last else "waiting for a request"), MUTED),
            Text.assemble(("prefill ", MUTED), (fmt(prefill), f"bold {INK}"), (" tok/s", MUTED)), Text("")]
    digits = big(fmt(rate, 1).replace(",", "") if rate is not None else "-")
    lines = [Text.assemble(gradient(row, "bold"), "   ", s) for row, s in zip(digits, side)]
    lines.append(spark(h.get("tok_s"), width))
    if state == "reading":
        what = (f"{fmt(live.get('prompt_read'))} / {fmt(live.get('prompt_total'))} prompt tokens"
                if live.get("prompt_total") else f"{fmt(live.get('prompt_tokens'))} prompt tokens")
        badge, tone, frac = "Reading", INFO_TEXT, (live.get("prompt_read") or 0) / max(1, live.get("prompt_total") or 1)
    elif gen:
        phase = live.get("phase") or "writing"
        what = f"{fmt(live.get('generated'))} tokens · {clock(live.get('elapsed_s') or 0)}"
        badge, tone = phase[:1].upper() + phase[1:], ACCENT_TEXT
        frac = (live.get("generated") or 0) / max(1, live.get("max_tokens") or 1)
    else:
        what = f"last answer: {fmt(last.get('output_tokens'))} tokens · {clock(last.get('duration_s') or 0)}" \
            if last else "waiting for a request"
        badge, tone, frac = "Idle", MUTED, 0.0
    if live.get("queued"):
        badge += f" · {live['queued']} queued"
    lines.append(Text.assemble(("● ", tone), (badge, f"bold {tone}"), ("  " + what, MUTED)))
    lines.append(bar(frac, width))
    ms = tn.get("ms_tok") if gen and tn else 1000 / rate if rate else None
    fresh = tn and tn.get("time") and time.time() - tn["time"] <= 30
    hit = tn.get("vram_hit") if fresh else last and last.get("hit_rate")
    used = (live.get("prompt_tokens") or 0) + (live.get("generated") or 0) if state != "idle" else \
        (last.get("prompt_tokens") or 0) + (last.get("output_tokens") or 0) if last else 0
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_row(label("Time per token", fmt(ms, 1), "ms"), label("VRAM hit", fmt(hit and 100 * hit, 1), "%"))
    grid.add_row(label("Context", f"{kfmt(used)} / {ctxfmt(eng.get('max_context'))}"),
                 label("Requests", totals.get("requests")))
    return Group(*lines, grid)


def tiers(m: dict, width: int):
    """Where the experts live (the GLM engine's STAT lines): the bar, how many each tier holds, what a token takes
    from RAM and the SSD - and the Monitor's advice when the numbers call for it."""
    t, eng = m.get("tiers") or {}, m.get("engine") or {}
    n, h = t.get("now"), t.get("history") or {}
    if not n:
        return Text("The engine tells where its experts live while it writes an answer.", f"italic {MUTED}")
    ds4 = eng.get("engine_kind") == "ds4"
    tier = ds4 and int(eng.get("ram_experts") or 0) > 0
    if ds4:   # DeepSeek: the VRAM slots, the RAM tier's (when on), the rest read in place from the model's files
        vu = int(eng.get("expert_slots") or 0)
        ru = int(eng.get("ram_experts") or 0) if tier else int(eng.get("cpu_experts") or 0)
        disk = max(0, int(eng.get("cpu_experts") or 0) - ru) if tier else 0
        total = max(1, vu + ru + disk)
    else:
        vu, ru = n.get("vram_used") or 0, n.get("ram_used") or 0
        total = eng.get("experts") or max(1, vu + ru)
        disk = max(0, total - vu - ru)
    whole = max(total, vu + ru)                         # (the RAM tier can hold experts VRAM has too: never wider)
    cols = [round(width * vu / whole)]
    cols.append(min(width - cols[0], round(width * ru / whole)))
    line = Text.assemble(gradient("█" * cols[0]), ("█" * cols[1], INFO_TEXT), ("█" * max(0, width - sum(cols)), WARN))
    legend = Table.grid(padding=(0, 1))
    parts = (((ACCENT, "VRAM", vu, ""), (INFO_TEXT, "Pinned RAM", ru, ""), (WARN, "SSD only", disk, "")) if tier else
             ((ACCENT, "VRAM", vu, ""), (INFO_TEXT, "RAM cache / SSD (in place)", ru, ""))) if ds4 else \
        ((ACCENT, "VRAM", vu, f" · {fmt(n.get('vram_gb'), 1)} GB"),
         (INFO_TEXT, "Pinned RAM", ru, f" · {fmt(n.get('ram_gb'), 1)} GB"), (WARN, "SSD only", disk, ""))
    for color, what, k, extra in parts:
        legend.add_row(Text("■", color), Text(what, MUTED), Text(fmt(k), f"bold {INK}"), Text("experts" + extra, MUTED))
    rows = Table.grid(padding=(0, 2))
    mm3 = eng.get("engine_kind") == "mm3"
    for key, name, color, unit in TIERS_DS4 if ds4 else TIERS_MM3 if mm3 else TIERS:
        rows.add_row(Text(name, MUTED), Text.assemble((fmt(n.get(key), 2), f"bold {INK}"), (unit, MUTED)),
                     spark(h.get(key), max(8, width - 32), color=color))
    stale = time.time() - (n.get("time") or 0) > 30
    if ds4:
        tips = () if stale else TIPS["ds4_disk"] if (n.get("disk_mb") or 0) >= 500 else \
            TIPS["ds4_cpu"] if (n.get("vram_hit") or 1) < 0.5 else ()
    elif mm3:
        tips = () if stale else TIPS["disk"] if (n.get("disk") or 0) >= 3 else \
            TIPS["ds4_cpu"] if (n.get("vram_hit") or 1) < 0.5 else ()
    else:
        tips = () if stale else TIPS["disk"] if (n.get("disk") or 0) >= 3 else \
            TIPS["ram"] if (n.get("vram_hit") or 1) < 0.85 and (n.get("ram_fetch") or 0) >= 4 else ()
    tip = next((t for t in tips if len(t) <= width), tips[-1] if tips else "")   # (one line: the longest that fits)
    return Group(line, legend, Text(""), rows,
                 *([Text(tip, f"italic {INK_SOFT}", no_wrap=True, overflow="ellipsis")] if tip else []))


def tiers_sub(m: dict) -> Text:
    n, eng = (m.get("tiers") or {}).get("now"), m.get("engine") or {}
    if not n:
        return Text("")
    age = time.time() - (n.get("time") or 0)
    return Text(f"last reading {clock(age)} ago" if age > 30 else f"live · {fmt(eng.get('experts'))} routed experts",
                f"italic {FAINT}")


def gpus(m: dict) -> Table:
    """One row a GPU: its load, memory, power, temperature and PCIe link (the Monitor's GPUs table)."""
    hw, st = m.get("hardware") or {}, m.get("hardware_static") or {}
    cards = hw.get("gpus") or ([{"index": "", **{k[4:]: v for k, v in hw.items() if k.startswith("gpu_")}}]
                               if hw.get("gpu_mem_total") else [])   # (one card: the server sends its gpu_* only)
    names = (st.get("gpu_name") or "").split(" + ")
    t = Table(box=None, expand=True, padding=(0, 1), header_style=f"bold {MUTED}", pad_edge=False)
    for col, right in (("GPU", False), ("Load", False), ("VRAM", True), ("Power", True), ("Temp", True),
                       ("PCIe", True)):
        t.add_column(col, justify="right" if right else "left", no_wrap=True)
    for i, g in enumerate(cards):
        name = names[i] if len(names) == len(cards) else ""
        name = name.replace("NVIDIA ", "").replace("GeForce ", "").replace("AMD ", "").replace("Radeon ", "")
        util, temp = g.get("util"), g.get("temp")
        gen = g.get("pcie_gen_max") or g.get("pcie_gen")
        rx = g.get("pcie_rx_mb")
        t.add_row(Text.assemble((f"{g.get('index')} ", f"bold {INK}"), (name, MUTED)),
                  Text.assemble(bar((util or 0) / 100, 10), (f" {fmt(util):>3}%", INK)),
                  Text.assemble((gb(g.get("mem_used")), INK), (f" / {gb(g.get('mem_total'), 0)} GB", MUTED)),
                  Text.assemble((fmt(g.get("power")), INK), (f" / {fmt(g.get('power_limit'))} W", MUTED)),
                  Text(f"{fmt(temp)} °C", DANGER if (temp or 0) >= 87 else WARN if (temp or 0) >= 80 else INK),
                  Text.assemble((f"Gen{gen}" if gen else "–", INK),
                                (f" x{g.get('pcie_width')}" if g.get("pcie_width") else "", MUTED),
                                (f" · {fmt(rx, 1 if rx < 10 else 0)} MB/s" if rx is not None else "", MUTED)))
    if not cards:
        t.add_row(Text("no GPU readings (nvidia-smi / NVML did not answer)", MUTED), "", "", "", "", "")
    return t


def hardware(m: dict, width: int) -> Table:
    """The Monitor's hardware cards: the GPUs together (load, memory, the hottest, power, PCIe), then the CPU, RAM and
    disk - each with its last minute."""
    hw, st, h, eng = (m.get(k) or {} for k in ("hardware", "hardware_static", "history", "engine"))

    def rate(mb):
        return "–" if mb is None else f"{mb / 1024:.1f} GB/s" if mb >= 1000 else f"{mb:,.{1 if mb < 10 else 0}f} MB/s"
    n = len(hw.get("gpus") or [])
    tiles = [("GPU load", fmt(hw.get("gpu_util")), "%", f"mean of {n} cards" if n > 1 else "", h.get("gpu_util"), 100),
             ("VRAM", gb(hw.get("gpu_mem_used")), f"/ {gb(hw.get('gpu_mem_total'), 0)} GB",
              f"{fmt(eng.get('expert_slots'))} experts cached" if eng.get("expert_slots") else "",
              h.get("gpu_mem_used"), hw.get("gpu_mem_total")),
             ("Temperature", fmt(hw.get("gpu_temp")), "°C", "the hottest card" if n > 1 else "", h.get("gpu_temp"), 90),
             ("Power", fmt(hw.get("gpu_power")), "W", f"of {fmt(hw.get('gpu_power_limit'))} W" if
              hw.get("gpu_power_limit") else "", h.get("gpu_power"), hw.get("gpu_power_limit")),
             ("PCIe", f"Gen{hw.get('gpu_pcie_gen_max') or hw.get('gpu_pcie_gen') or '?'}",
              f"x{hw.get('gpu_pcie_width')}" if hw.get("gpu_pcie_width") else "",
              f"to GPU {rate(hw.get('gpu_pcie_rx_mb'))}", h.get("gpu_pcie_rx_mb"), None),
             ("CPU", fmt(hw.get("cpu")), "%", f"{st.get('cores')} cores · {st.get('threads')} threads"
              if st.get("threads") else "", h.get("cpu"), 100),
             ("RAM", gb(hw.get("ram_used")), f"/ {gb(hw.get('ram_total'), 0)} GB",
              f"{fmt(eng.get('ram_gb'), 1)} GB pinned for experts" if eng.get("ram_gb") else "", h.get("ram_used"),
              hw.get("ram_total")),
             ("Disk read", rate(hw.get("disk_read_mb")), "", f"write {fmt(hw.get('disk_write_mb'), 1)} MB/s"
              if hw.get("disk_write_mb") is not None else "", h.get("disk_read_mb"), None)]
    per_row = 4 if width >= 96 else 2
    cell = max(10, width // per_row - 3)
    grid = Table.grid(expand=True, padding=(0, 3))
    for _ in range(per_row):
        grid.add_column(ratio=1)
    cells = [Text("\n").join([Text(k, MUTED), label("", v, u)[2:], Text(clip(sub or " ", cell), FAINT),
                              spark(hist, cell, top)]) for k, v, u, sub, hist, top in tiles]
    for i in range(0, len(cells), per_row):
        grid.add_row(*cells[i:i + per_row])
        if i + per_row < len(cells):
            grid.add_row(*([Text("")] * per_row))
    return grid


def requests(m: dict, width: int, rows: int = 8):
    """The last requests, the newest first (the Monitor's table), and the totals since the start."""
    reqs, t = m.get("requests") or [], m.get("totals") or {}
    if not reqs:
        return Text("No requests yet: they show here as the dashboard's chat and the API send them.", MUTED)
    table = Table(box=None, expand=True, padding=(0, 1), header_style=f"bold {MUTED}", pad_edge=False)
    wide = width >= 90
    for col in ("Time", "Status", *(("API",) if wide else ()), "Prompt", *(("Reused",) if wide else ()), "Prefill",
                "Output", "Decode", "VRAM hit", "Duration"):
        table.add_column(col, justify="left" if col in ("Time", "Status", "API") else "right", no_wrap=True)
    for r in reqs[:rows]:
        status, tone = FINISH.get(r.get("finish"), (r.get("finish") or "–", INK))
        pre = prefill_rate(r)
        hit = r.get("hit_rate")
        table.add_row(Text(datetime.fromtimestamp(r.get("time") or 0).strftime("%H:%M:%S"), MUTED), Text(status, tone),
                      *((Text(API.get(r.get("api"), r.get("api") or ""), MUTED),) if wide else ()),
                      fmt(r.get("prompt_tokens")), *((fmt(r.get("reused")),) if wide else ()), fmt(pre),
                      fmt(r.get("output_tokens")), Text(fmt(r.get("decode_tok_s"), 1), f"bold {INK}"),
                      "–" if hit is None else f"{100 * hit:.1f}%", f"{fmt(r.get('duration_s'), 1)} s")
    out = [table]
    if t.get("requests"):
        read = (t.get("prompt_tokens") or 0) - (t.get("reused") or 0)
        p = f" at {fmt(read / (t['prompt_ms'] / 1000))} tok/s" if t.get("prompt_ms") and read > 0 else ""
        o = f" at {fmt(t['output_tokens'] / (t['decode_ms'] / 1000), 1)} tok/s" if t.get("decode_ms") and \
            t.get("output_tokens") else ""
        since = datetime.fromtimestamp(t.get("since") or 0).strftime("%a %H:%M")
        out.append(Text(f"Since {since}: {fmt(t['requests'])} requests · {fmt(read)} prompt tokens read{p} "
                        f"({fmt(t.get('reused'))} reused) · {fmt(t.get('output_tokens'))} written{o}", MUTED))
    return Group(*out)


def banners(m: dict, fails: int) -> Text:
    """What the Monitor puts above everything: a reload, a hot GPU, a server that stopped answering."""
    out = []
    rl = m.get("reload") or {}
    if rl.get("state") in ("waiting", "running"):
        out.append((f"Reloading the model with a {ctxfmt(rl.get('to'))} context: requests wait until it is back.",
                    INFO_TEXT))
    for g in (m.get("hardware") or {}).get("gpus") or []:
        if (g.get("temp") or 0) >= 87:
            out.append((f"GPU {g.get('index')} is at {fmt(g['temp'])} °C: it may slow itself down to cool off.", WARN))
    if fails >= 3:
        out.append(("The server does not answer: these numbers are from before.", DANGER))
    return Text("\n").join(Text.assemble(("! ", f"bold {tone}"), (msg, tone)) for msg, tone in out)


def loading(w: LoadWatch, width: int):
    """The box while the model loads: the share in big, what the engine does now, the time, and each GPU's part."""
    frac, took = w.fraction(), time.monotonic() - w.t0
    n, parts = len(w.gpus), w.parts

    def gpu(k):
        return f"GPU {w.gpus[k]}" if k < n else f"part {k + 1}"
    now = [k for k in sorted(parts) if parts[k][0] < parts[k][1]]   # (the GPU warming now; before it starts: the next)
    if not parts:
        what = "reading the model's weights"
    elif len(parts) >= n and not now:
        what = "the engine's last checks - almost ready"
    else:
        k = now[-1] if now else max(parts) + 1
        what = f"warming the experts on {gpu(k)}" + (f" (part {k + 1} of {n})" if n > 1 else "")
    left = f" · about {clock(took * (1 - frac) / frac)} left" if frac and 0.03 < frac < 1 and took > 10 else ""
    warm, (total, known) = sum(done for done, _ in parts.values()), w.total()
    of = f"{fmt(total, 1)}" if known else f"about {fmt(total)}"
    side = [Text(f"{fmt(warm, 1)} of {of} GB of experts warm" if total else "Starting the engine",
                 f"bold {INK}"), Text(what[:1].upper() + what[1:], INK_SOFT),
            Text(f"{clock(took)} so far{left}", MUTED), Text("")]
    if frac is None:                                    # (no engine log: the time only)
        return Group(*side[:3])
    lines = [Text.assemble(gradient(row, "bold"), "    ", s) for row, s in zip(big(f"{round(frac * 100)}%"), side)]
    rows = Table.grid(padding=(0, 2))
    for k in range(max(n, len(parts))):
        done, whole = parts.get(k, (0.0, 0.0))
        state = Text("waiting", FAINT) if k not in parts else \
            Text.assemble(("✓ ", OK), (f"{fmt(whole, 1)} GB", INK)) if done >= whole else \
            Text(f"{fmt(done, 1)} of {fmt(whole, 1)} GB", INK)
        rows.add_row(Text(gpu(k), INK if k in parts else MUTED), bar(done / whole if whole else 0.0, 24), state)
    return Group(*lines, Text(""), bar(frac, width), Text(""), rows, Text(""),
                 Text("The engine reads the experts into RAM and VRAM: the PC can be slow until the model is ready.",
                      f"italic {MUTED}"))


class Dashboard(VerticalScroll):
    """The boxes above, filled from GET /metrics once a second while they show."""

    SCOPED_CSS = False                                  # (.narrow is the screen's: the top boxes stack on 80 columns)
    DEFAULT_CSS = """
    Dashboard { height: 1fr; scrollbar-size-vertical: 1; scrollbar-color: $maya-edge;
                scrollbar-background: $background; }
    Dashboard Static { height: auto; }
    #dash-banner { margin-bottom: 1; }
    #dash-load { width: 90; max-width: 100%; border: solid $maya-line; padding: 1 2; margin-bottom: 1;
                 border-title-align: center; }
    #dash-card { width: 76; max-width: 100%; height: auto; border: solid $maya-line; padding: 0 1; margin-bottom: 1; }
    #dash-row { height: auto; }
    #dash-speed, #dash-tiers, #dash-gpus, #dash-hw, #dash-reqs { border: solid $maya-line; padding: 0 1;
                 margin-bottom: 1; border-title-align: center; border-subtitle-align: right; }
    #dash-speed, #dash-tiers { width: 1fr; }
    #dash-speed { margin-right: 1; }
    .narrow #dash-row { layout: vertical; }
    .narrow #dash-speed { margin-right: 0; }
    """

    def compose(self) -> ComposeResult:
        yield Static(id="dash-banner")
        with Center():
            yield Card(id="dash-card")
        with Center():
            yield Static(id="dash-load")
        with Horizontal(id="dash-row"):
            yield Static(id="dash-speed")
            yield Static(id="dash-tiers")
        yield Static(id="dash-gpus")
        yield Static(id="dash-hw")
        yield Static(id="dash-reqs")

    def on_mount(self) -> None:
        self.metrics, self.fails, self.drawn, self.load = None, 0, None, LoadWatch(None, [])
        for box, name in (("load", "Loading the model"), ("speed", "Throughput"), ("tiers", "Where the experts live"),
                          ("gpus", "GPUs"), ("hw", "Hardware"), ("reqs", "Recent requests")):
            self.query_one(f"#dash-{box}").border_title = title(name)
        self.query_one("#dash-banner").display = False
        self.set_interval(1.0, self.draw)

    def follow(self, url: str, key: str = "", log=None, gpus=(), vram=None) -> None:
        """GET /metrics once a second from now on, in a thread: the screen never waits for the server.  log, gpus, vram:
        the engine log, the start's GPUs and their VRAM, for the loading box (LoadWatch)."""
        self.load = LoadWatch(log, gpus, vram)

        def poll():
            req = urllib.request.Request(url, headers={"Authorization": f"Bearer {key}"} if key else {})
            while True:
                try:
                    with urllib.request.urlopen(req, timeout=3) as r:
                        self.metrics, self.fails = json.loads(r.read()), 0
                except (OSError, ValueError):
                    self.fails += 1
                time.sleep(1)
        threading.Thread(target=poll, name="dashboard", daemon=True).start()

    def draw(self) -> None:
        if not self.display:
            return
        starting = (getattr(self.app, "serving", None) or {}).get("state") != "ready"   # (no numbers before it is)
        for box in ("#dash-row", "#dash-gpus", "#dash-hw", "#dash-reqs"):
            self.query_one(box).display = not starting
        self.query_one("#dash-load").display = starting
        if starting:
            self.load.read()
            self.query_one("#dash-banner").display = False
            width = min(84, self.size.width - 12)
            self.query_one("#dash-load", Static).update(loading(self.load, width))
            return
        m = self.metrics
        if m is None or (m is self.drawn and self.fails < 3):
            return
        self.drawn = m
        try:
            self.fill(m)
        except Exception as e:                         # noqa: BLE001 - numbers it cannot show never stop the screen
            self.query_one("#dash-banner", Static).update(Text(f"! the dashboard could not show these numbers ({e})",
                                                               DANGER))
            self.query_one("#dash-banner").display = True

    def fill(self, m: dict) -> None:
        half = (self.size.width - 1) // (1 if self.screen.has_class("narrow") else 2) - 4
        full = self.size.width - 5
        note = banners(m, self.fails)
        self.query_one("#dash-banner", Static).update(note)
        self.query_one("#dash-banner").display = bool(note.plain)
        self.query_one("#dash-speed", Static).update(speed(m, half))
        self.query_one("#dash-tiers", Static).update(tiers(m, half))
        self.query_one("#dash-tiers").border_subtitle = tiers_sub(m)
        gpu_count = len((m.get("hardware") or {}).get("gpus") or [])
        self.query_one("#dash-gpus").border_subtitle = Text(f"{gpu_count} cards" if gpu_count > 1 else "",
                                                            f"italic {FAINT}")
        self.query_one("#dash-gpus", Static).update(gpus(m))
        self.query_one("#dash-hw", Static).update(hardware(m, full))
        self.query_one("#dash-reqs", Static).update(requests(m, full))

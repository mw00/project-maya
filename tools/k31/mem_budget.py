#!/usr/bin/env python3
"""tools/k31/mem_budget.py - how many bytes a Kimi K3.1 build can have for its experts on this machine, measured with
everything that runs here now still running.

B_pool is the memory the build can take without pushing anything into swap:
  - RAM: MemAvailable, plus ZFS's ARC above its floor (the kernel does not count it, the engines do:
    src/ds4/ds4_model.cu ram_avail);
  - VRAM: each GPU's free memory as nvidia-smi reports it (--gpus picks which GPUs count; default all).
E_gb is what is left for the routed experts once the fixed costs are taken off: the always-on part at Maya's quant
floors (docs/QUANT-PLAN.md, attention / KDA / MLA / shared experts / latent projections at Q6_K, dense layer 0 at
Q8_0, head and embeddings at Q6_K, the BF16 router; params from docs/KIMI-K31.md §3.1), CUDA contexts and scratch,
the MLA latent and indexer keys at --ctx tokens and the KDA state (§3.3), the KDA checkpoints in host RAM, host
buffers and a 4 GB RAM margin.

The "services only" column is the same machine without the big jobs: the processes named with --job (a regular
expression matched against each process's command line and working directory; everything such a process started
counts too) give back their private memory (Pss_Anon, else RssAnon), the RAM their swapped pages hold in zram, and
their GPU memory. What is left running is listed by its systemd unit. A service whose pages are in swap may page them
back in; the table says how much that could take back.

A service that is stopped holds nothing, so its memory counts as free here: the numbers assume the K3.1 build runs in
its place. --reserve takes back what such a service holds when it runs (ram=GB, or gpuN=GB for one GPU's VRAM), off
both columns.

Read-only: it reads /proc, /sys and nvidia-smi and changes nothing. Linux; NVIDIA GPUs only (no nvidia-smi: RAM only).

    python tools/k31/mem_budget.py
    python tools/k31/mem_budget.py --job 'llama-quantize|convert_hf_to_gguf' --gpus 0,1,3 --json budget.json
    python tools/k31/mem_budget.py --reserve ram=6 --reserve gpu3=3.8

GB in the table is 10^9 bytes, as in docs/KIMI-K31.md; the JSON holds bytes.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import subprocess
import sys

GB = 10 ** 9

# ggml block layouts: type -> (values a block, bytes a block)
BLOCK = {"Q6_K": (256, 210), "Q8_0": (32, 34), "IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ2_XXS": (256, 66)}


def qbytes(params: int, kind: str) -> int:
    n, b = BLOCK[kind]
    return params * b // n


# The always-on part at the floors: (part, bytes, how). Params are docs/KIMI-K31.md §3.1's checkpoint counts; the
# router is BF16 in the checkpoint, so its bytes are the checkpoint's; the indexer (BF16 keys, F32 proj) and the F32
# norms / attn-res scores are rounded up.
ALWAYS_ON = [
    ("KDA, 69 layers", qbytes(30_618_088_704, "Q6_K"), "Q6_K"),
    ("MLA, 24 layers", qbytes(5_572_657_152, "Q6_K"), "Q6_K"),
    ("shared experts, 92 layers", qbytes(12_155_092_992, "Q6_K"), "Q6_K"),
    ("latent projections, 92 layers", qbytes(4_726_980_608, "Q6_K"), "Q6_K"),
    ("dense layer-0 MLP", qbytes(726_663_168, "Q8_0"), "Q8_0"),
    ("lm_head", qbytes(1_174_405_120, "Q6_K"), "Q6_K"),
    ("token_embd (host RAM)", qbytes(1_174_405_120, "Q6_K"), "Q6_K"),
    ("router gate + bias", 1_182_074_880, "BF16 + F32"),
    ("indexer, 24 layers", 400_000_000, "BF16 / F32"),
    ("norms, attn-res scores", 50_000_000, "F32"),
]
TOKEN_EMBD = ALWAYS_ON[6][1]

CUDA_PER_GPU = 1 * GB                 # context and scratch a GPU
KV_PER_TOKEN = 24 * 576 * 2 + 24 * 128 * 2   # MLA latent (f16) + indexer keys (BF16), §3.3
KDA_STATE = 69 * 96 * 128 * 128 * 4 + 30_000_000   # one sequence's KDA state + conv state, §3.3
KDA_CHECKPOINT = 443 << 20            # one host checkpoint
HOST = 6 * GB                         # host buffers, pinned staging, CPU-lane scratch, 4 GB RAM margin

EXPERT_PARAMS = 3 * 3584 * 3072       # one routed expert (gate, up, down), §3.1
MOE_LAYERS = 92


def read(path: pathlib.Path) -> str:
    try:
        return path.read_text(errors="replace")
    except OSError:
        return ""


def kv_kb(text: str) -> dict:
    """'Key:   123 kB' lines -> {key: bytes}."""
    out = {}
    for line in text.splitlines():
        m = re.match(r"(\w+(?:\(\w+\))?):\s+(\d+)(?:\s+kB)?", line)
        if m:
            out[m.group(1)] = int(m.group(2)) * (1024 if line.rstrip().endswith("kB") else 1)
    return out


def ram(root: pathlib.Path) -> dict:
    mi = kv_kb(read(root / "proc/meminfo"))
    total, avail = mi.get("MemTotal", 0), mi.get("MemAvailable", 0)
    arc = {}
    for line in read(root / "proc/spl/kstat/zfs/arcstats").splitlines()[2:]:
        f = line.split()
        if len(f) == 3 and f[2].isdigit():
            arc[f[0]] = int(f[2])
    arc_free = max(0, arc.get("size", 0) - arc.get("c_min", 0)) if "size" in arc and "c_min" in arc else 0
    arc_free = min(arc_free, max(0, total - avail))
    swap_used = zram_used = 0
    for line in read(root / "proc/swaps").splitlines()[1:]:
        f = line.split()
        if len(f) >= 4 and f[3].isdigit():
            swap_used += int(f[3]) << 10
            if os.path.basename(f[0]).startswith("zram"):
                zram_used += int(f[3]) << 10
    z_orig = z_mem = 0
    for d in sorted((root / "sys/block").glob("zram*")):
        f = read(d / "mm_stat").split()
        if len(f) >= 3:
            z_orig += int(f[0])
            z_mem += int(f[2])
    return {"mem_total": total, "mem_available": avail, "zfs_arc_reclaimable": arc_free,
            "ram_avail": avail + arc_free, "swap_total": mi.get("SwapTotal", 0), "swap_used": swap_used,
            "zram_swap_used": zram_used, "zram_orig": z_orig, "zram_mem": z_mem}


def zram_ram_of(swapped: int, r: dict) -> int:
    """The RAM that `swapped` bytes of a process's swap hold, if they sit in zram as the whole swap does on average."""
    if r["swap_used"] <= 0 or r["zram_orig"] <= 0:
        return 0
    return int(swapped * r["zram_swap_used"] / r["swap_used"] * r["zram_mem"] / r["zram_orig"])


def smi(exe: str, args: list[str]) -> list[list[str]] | None:
    try:
        out = subprocess.run([exe] + args, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return None
    if out.returncode != 0:
        return None
    return [[c.strip() for c in line.split(",")] for line in out.stdout.splitlines() if line.strip()]


def gpus(exe: str) -> tuple[list[dict], dict]:
    """[{index, name, uuid, total, used, free}] (bytes) and {pid: {gpu index: bytes}}."""
    rows = smi(exe, ["--query-gpu=index,uuid,name,memory.total,memory.used,memory.free",
                     "--format=csv,noheader,nounits"])
    if rows is None:
        return [], {}
    out, by_uuid = [], {}
    for r in rows:
        if len(r) < 6 or not r[0].isdigit():
            continue
        g = {"index": int(r[0]), "uuid": r[1], "name": r[2], "total": int(r[3]) << 20,
             "used": int(r[4]) << 20, "free": int(r[5]) << 20}
        out.append(g)
        by_uuid[g["uuid"]] = g["index"]
    apps = {}
    for r in smi(exe, ["--query-compute-apps=pid,gpu_uuid,used_memory", "--format=csv,noheader,nounits"]) or []:
        if len(r) >= 3 and r[0].isdigit() and r[1] in by_uuid and r[2].isdigit():
            d = apps.setdefault(int(r[0]), {})
            d[by_uuid[r[1]]] = d.get(by_uuid[r[1]], 0) + (int(r[2]) << 20)
    return out, apps


def unit_of(cgroup: str) -> str:
    """The systemd unit (or scope) a process runs in, from its cgroup v2 line."""
    path = ""
    for line in cgroup.splitlines():
        if line.startswith("0::"):
            path = line[3:]
    leaf = path.rstrip("/").rsplit("/", 1)[-1] or "-"
    m = re.match(r"docker-([0-9a-f]{12})[0-9a-f]*\.scope$", leaf)
    return f"docker-{m.group(1)}" if m else leaf


def procs(root: pathlib.Path) -> dict:
    """{pid: {ppid, name, cmd, cwd, unit, anon, swap}} for every user-space process (bytes)."""
    out = {}
    for d in (root / "proc").glob("[0-9]*"):
        st = read(d / "status")
        if "VmRSS" not in st:
            continue
        kv = kv_kb(st)
        name = re.search(r"^Name:\s*(.*)$", st, re.M)
        anon = kv.get("RssAnon", 0)
        roll = kv_kb(read(d / "smaps_rollup"))
        if "Pss_Anon" in roll:
            anon = roll["Pss_Anon"]
        try:
            cwd = os.readlink(d / "cwd")
        except OSError:
            cwd = ""
        cmd = read(d / "cmdline").replace("\0", " ").strip()
        out[int(d.name)] = {"ppid": kv.get("PPid", 0), "name": name.group(1) if name else "?", "cmd": cmd, "cwd": cwd,
                            "unit": unit_of(read(d / "cgroup")), "anon": anon, "swap": kv.get("VmSwap", 0)}
    return out


def jobs_of(ps: dict, patterns: list[str], skip: set) -> set:
    """The pids that match a --job pattern (command line or working directory), and everything they started; `skip`
    (this tool and the shells that started it, whose command lines hold the patterns) never matches."""
    if not patterns:
        return set()
    rx = re.compile("|".join(f"(?:{p})" for p in patterns))
    hit = {p for p, v in ps.items() if p not in skip and (rx.search(v["cmd"]) or rx.search(v["cwd"]))}
    kids = {}
    for p, v in ps.items():
        kids.setdefault(v["ppid"], []).append(p)
    todo, seen = list(hit), set(hit)
    while todo:
        for c in kids.get(todo.pop(), []):
            if c not in seen and c not in skip:
                seen.add(c)
                todo.append(c)
    return seen


def ancestors(ps: dict, pid: int) -> set:
    out = set()
    while pid in ps and pid not in out and pid > 1:
        out.add(pid)
        pid = ps[pid]["ppid"]
    return out


def fixed_costs(n_gpus: int, ctx: int, checkpoints: int) -> list[tuple[str, int, str]]:
    """[(cost, bytes, where)] with where 'vram' or 'ram' (where the build would put it)."""
    on = sum(b for _, b, _ in ALWAYS_ON)
    return [
        ("always-on part at the floors, on the GPUs", on - TOKEN_EMBD, "vram"),
        ("token_embd (lookup only)", TOKEN_EMBD, "ram"),
        (f"CUDA contexts and scratch ({n_gpus} GPUs)", n_gpus * CUDA_PER_GPU, "vram"),
        (f"MLA latent + indexer keys at {ctx} tokens", ctx * KV_PER_TOKEN, "vram"),
        ("KDA state + conv (one sequence)", KDA_STATE, "vram"),
        (f"KDA checkpoints in host RAM ({checkpoints})", checkpoints * KDA_CHECKPOINT, "ram"),
        ("host buffers + staging + CPU scratch + 4 GB margin", HOST, "ram"),
    ]


def budget(ram_avail: int, free: dict, fixed: list) -> dict:
    """B_pool, E_gb and where the experts would sit (VRAM first: the always-on part takes the GPUs before them; per-GPU
    fit is not checked)."""
    vram = sum(free.values())
    b_pool = ram_avail + vram
    fx = sum(b for _, b, _ in fixed)
    fv = sum(b for _, b, w in fixed if w == "vram")
    fr = fx - fv
    e = b_pool - fx
    ev = vram - fv
    er = ram_avail - fr + min(0, ev)
    per = EXPERT_PARAMS * MOE_LAYERS
    return {"ram_avail": ram_avail, "vram_free": vram, "b_pool": b_pool, "fixed": fx, "e": e,
            "experts_in_vram": max(0, ev), "experts_in_ram": er,
            "experts_a_layer": {k: max(0, e) // qbytes(per, k) for k in ("IQ1_S", "IQ1_M", "IQ2_XXS")}}


def gb(x: int) -> str:
    return f"{x / GB:8.2f}"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--job", action="append", default=[],
                    help="a regular expression naming a big job (command line or working directory); repeatable")
    ap.add_argument("--gpus", help="the GPU indices that count, comma-separated (default: all)")
    ap.add_argument("--ctx", type=int, default=65536, help="context length for the KV and indexer keys (65536)")
    ap.add_argument("--checkpoints", type=int, default=2, help="KDA host checkpoints (2)")
    ap.add_argument("--candidates", default="90,108", help="expert budgets to check, GB (90,108)")
    ap.add_argument("--reserve", action="append", default=[], metavar="WHERE=GB",
                    help="memory a stopped service holds when it runs (ram=GB or gpuN=GB); off both columns; repeatable")
    ap.add_argument("--json", metavar="PATH", help="write the numbers as JSON ('-': stdout instead of the table)")
    ap.add_argument("--root", default="/", help=argparse.SUPPRESS)
    ap.add_argument("--nvidia-smi", default="nvidia-smi", help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    root = pathlib.Path(a.root)

    gl, apps = gpus(a.nvidia_smi)
    r = ram(root)
    ps = procs(root)
    res_ram, res_vram = 0, {}
    for x in a.reserve:
        m = re.fullmatch(r"(?:ram|gpu(\d+))=(\d+(?:\.\d*)?)", x.strip())
        if not m or (m.group(1) is not None and int(m.group(1)) not in {g["index"] for g in gl}):
            print(f"mem_budget: --reserve {x!r}: want ram=GB, or gpuN=GB for a GPU nvidia-smi lists", file=sys.stderr)
            return 2
        if m.group(1) is None:
            res_ram += int(float(m.group(2)) * GB)
        else:
            res_vram[int(m.group(1))] = res_vram.get(int(m.group(1)), 0) + int(float(m.group(2)) * GB)
    if a.gpus:
        want = {int(x) for x in a.gpus.split(",") if x.strip()}
        missing = want - {g["index"] for g in gl}
        if missing:
            print(f"mem_budget: no GPU {sorted(missing)} (nvidia-smi lists {[g['index'] for g in gl]})", file=sys.stderr)
            return 2
        gl = [g for g in gl if g["index"] in want]
    counted = {g["index"] for g in gl}
    fixed = fixed_costs(len(gl), a.ctx, a.checkpoints)

    # this tool and its shells hold the patterns on their command lines; a --root tree's pids are not ours
    me = ancestors(ps, os.getpid()) | {os.getpid()} if root.resolve() == pathlib.Path("/") else set()
    jobs = jobs_of(ps, a.job, me)
    job_rows = []
    for p in sorted(jobs, key=lambda p: -ps[p]["anon"]):
        v = ps[p]
        vr = {i: b for i, b in apps.get(p, {}).items()}
        job_rows.append({"pid": p, "name": v["name"], "cmd": v["cmd"], "unit": v["unit"], "anon": v["anon"],
                         "swap": v["swap"], "swap_zram_ram": zram_ram_of(v["swap"], r), "vram": vr})
    j_anon = sum(j["anon"] for j in job_rows)
    j_zram = sum(j["swap_zram_ram"] for j in job_rows)
    j_vram = {i: sum(j["vram"].get(i, 0) for j in job_rows) for i in counted}

    units = {}
    for p, v in ps.items():
        if p in jobs or p in me:
            continue
        u = units.setdefault(v["unit"], {"unit": v["unit"], "procs": 0, "anon": 0, "swap": 0, "vram": {}})
        u["procs"] += 1
        u["anon"] += v["anon"]
        u["swap"] += v["swap"]
        for i, b in apps.get(p, {}).items():
            u["vram"][i] = u["vram"].get(i, 0) + b
    svc = sorted(units.values(), key=lambda u: -(u["anon"] + u["swap"] + sum(u["vram"].values())))
    s_swap = sum(u["swap"] for u in svc)
    s_swap_back = max(0, s_swap - zram_ram_of(s_swap, r))

    free_now = {g["index"]: g["free"] for g in gl}
    free_so = {i: free_now[i] + j_vram.get(i, 0) for i in free_now}
    res_vram = {i: b for i, b in res_vram.items() if i in counted}
    off = lambda free: {i: b - res_vram.get(i, 0) for i, b in free.items()}
    now = budget(r["ram_avail"] - res_ram, off(free_now), fixed)
    so = budget(r["ram_avail"] - res_ram + j_anon + j_zram, off(free_so), fixed)
    cands = [float(x) for x in a.candidates.split(",") if x.strip()]

    doc = {
        "units": "bytes", "ctx": a.ctx, "checkpoints": a.checkpoints, "gpus_counted": sorted(counted),
        "ram": r, "gpus": [dict(g, free_services_only=free_so[g["index"]]) for g in gl],
        "reserve": {"ram": res_ram, "vram": res_vram},
        "always_on": [{"part": n, "bytes": b, "type": t} for n, b, t in ALWAYS_ON],
        "fixed": [{"cost": n, "bytes": b, "where": w} for n, b, w in fixed],
        "now": now, "services_only": so,
        "candidates_gb": {f"{c:g}": {"now": now["e"] >= c * GB, "services_only": so["e"] >= c * GB} for c in cands},
        "jobs": job_rows, "jobs_freed": {"ram_anon": j_anon, "ram_zram": j_zram, "vram": j_vram},
        "services": svc[:40], "services_swap": s_swap, "services_swap_back": s_swap_back,
    }
    if a.json and a.json != "-":
        pathlib.Path(a.json).write_text(json.dumps(doc, indent=1) + "\n")
    if a.json == "-":
        print(json.dumps(doc, indent=1))
        return 0

    two = bool(a.job)
    head = f"{'':58s} {'now':>8s}" + (f" {'services only':>14s}" if two else "")
    def row(label, x, y=None):
        print(f"{label:58s} {gb(x)}" + (f" {gb(y):>14s}" if two and y is not None else ""))
    print("Memory pool (GB = 10^9 bytes)")
    print(head)
    row("RAM: MemAvailable", r["mem_available"], r["mem_available"] + j_anon + j_zram)
    row("RAM: ZFS ARC above its floor", r["zfs_arc_reclaimable"], r["zfs_arc_reclaimable"])
    for g in gl:
        row(f"GPU {g['index']} {g['name']} ({g['total'] / 2 ** 30:.0f} GiB): free", g["free"], free_so[g["index"]])
    if res_ram:
        row("reserved for stopped services (--reserve): RAM", -res_ram, -res_ram)
    for i, b in sorted(res_vram.items()):
        row(f"reserved for stopped services (--reserve): GPU {i}", -b, -b)
    row("B_pool", now["b_pool"], so["b_pool"])
    print()
    print("Fixed costs of the K3.1 build at the floors")
    for n, b, w in fixed:
        row(f"{n} [{w}]", b)
    row("total fixed", now["fixed"])
    print()
    print(head)
    row("E_gb: bytes left for the experts", now["e"], so["e"])
    row("  of it in VRAM (after the always-on part)", now["experts_in_vram"], so["experts_in_vram"])
    row("  of it in RAM", now["experts_in_ram"], so["experts_in_ram"])
    for k in ("IQ1_S", "IQ1_M", "IQ2_XXS"):
        mb = qbytes(EXPERT_PARAMS * MOE_LAYERS, k) / GB
        print(f"{'  experts a layer at ' + k + f' ({mb:.3f} GB each)':58s} {now['experts_a_layer'][k]:8d}"
              + (f" {so['experts_a_layer'][k]:14d}" if two else ""))
    for c in cands:
        ok = lambda e: "fits" if e >= c * GB else f"short {(c * GB - e) / GB:.1f}"
        print(f"{f'  candidate E_gb = {c:g}':58s} {ok(now['e']):>8s}" + (f" {ok(so['e']):>14s}" if two else ""))
    if two:
        print()
        print(f"Big jobs (--job): {len(job_rows)} processes, {gb(j_anon).strip()} GB private RAM, "
              f"{gb(j_zram).strip()} GB of RAM in zram, {gb(sum(j_vram.values())).strip()} GB VRAM")
        for j in job_rows[:15]:
            if j["anon"] + j["swap"] + sum(j["vram"].values()) < 50 << 20:
                continue
            vr = " ".join(f"gpu{i} {b / GB:.1f}" for i, b in j["vram"].items())
            print(f"  {j['pid']:>8d} {j['anon'] / GB:6.2f} GB ram {j['swap'] / GB:6.2f} swap {vr:12s} {j['cmd'][:70]}")
    print()
    print("Still running, by unit (private RAM, swap, VRAM; GB)")
    for u in svc[:12]:
        vr = " ".join(f"gpu{i} {b / GB:.1f}" for i, b in sorted(u["vram"].items()))
        print(f"  {u['unit'][:44]:44s} {u['procs']:4d} procs {u['anon'] / GB:6.2f} {u['swap'] / GB:6.2f}  {vr}")
    print(f"These hold {s_swap / GB:.2f} GB in swap; paging all of it back in would take up to {s_swap_back / GB:.2f} GB "
          f"from the pool.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

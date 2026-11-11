#!/usr/bin/env python3
"""tools/k31/tier_replay.py - replay Kimi K3.1 routing through VRAM / RAM / NVMe expert tiers and predict decode tok/s.

Two steps:

  extract  routing traces from calibration MoE dumps (layer_NNN.npz: h = the MoE input, ids = full-model top-16), optionally
           re-routed among a keep set exactly as a GGUF pruned to it routes (noaux_tc: top-16 of sigmoid(h W^T) + bias,
           the bias of every non-kept expert = -inf). Needs the router weights (safetensors) and h; numpy only.
           -> trace.npz: ids int16 [L, T, K], layers [L], n_expert, top_k, stride, keep (json) and provenance.
  replay   a trace through a tier configuration and a policy, then prices every (layer, token) request with the lane
           model below. numpy only (Maya's tools CI).

Cost model (one decode request = one layer's K picks for one token), in ds4_pcie.hpp's terms extended by a zero-copy
(ZC) fan-out lane:
  - VRAM picks run on the GPU from VRAM: g_v ms each.
  - every other pick runs on one of two lanes that work at the same time, split to finish together:
      CPU lane:  c ms an expert (bytes / the CPU lane's GB/s at its threads) + oc ms a call;
      ZC lane:   z ms an expert (bytes / (n_gpu x the per-GPU zero-copy GB/s), the experts' rows split over the GPUs,
                 capped by the RAM bus) + oz ms a request (fan-out bookkeeping);
    a pick that is not in RAM (cold) is read from NVMe first: disk ms (bytes / NVMe GB/s), on either lane; the split
    minimises max(CPU, ZC) over the counts (warm and cold picks are interchangeable within a class, so the best split
    depends only on how many of each go to the ZC lane: O(K^2) instead of ds4's 2^n masks).
  - t_layer = max(lanes) + VRAM picks' GPU time + t_dense (the layer's attention / shared expert / norms), and a token
    is the sum over the model's MoE layers, using the traced layers' mean request time per layer.
Every rate is an input. Defaults: one 16-core Zen 4 box with 4 DDR5 channels and RTX 3090s, measured 2026-11; override
them for yours (they are printed with each run).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

K31_LAYERS_MOE = 92
K31_EXPERT_PARAMS = 3 * 3072 * 3584          # gate + up + down of one routed expert (33.0M)


# ============================================================== trace extraction (needs the calibration dumps + checkpoint)

def _npz_memmap(path: Path, member: str) -> np.ndarray:
    """memmap one stored (uncompressed) member of an .npz, so a 364 MB h never lands in RAM."""
    import zipfile
    zi = zipfile.ZipFile(path).getinfo(member + ".npy")
    if zi.compress_type != 0:
        return np.load(path)[member]
    with open(path, "rb") as f:
        f.seek(zi.header_offset)
        hdr = f.read(30)
        f.seek(zi.header_offset + 30 + int.from_bytes(hdr[26:28], "little") + int.from_bytes(hdr[28:30], "little"))
        ver = np.lib.format.read_magic(f)
        rh = np.lib.format.read_array_header_1_0 if ver == (1, 0) else np.lib.format.read_array_header_2_0
        shape, fortran, dtype = rh(f)
        off = f.tell()
    return np.memmap(path, dtype=dtype, mode="r", offset=off, shape=shape, order="F" if fortran else "C")


def _bf16_to_f32(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a).astype(np.uint16).astype(np.uint32) << 16).view(np.float32)


def _st_tensor(model_dir: Path, index: dict, name: str) -> np.ndarray:
    """one tensor from a safetensors shard, read by header offsets (no safetensors package needed)."""
    path = model_dir / index[name]
    with open(path, "rb") as f:
        n = int.from_bytes(f.read(8), "little")
        hdr = json.loads(f.read(n))
        meta = hdr[name]
        a, b = meta["data_offsets"]
        f.seek(8 + n + a)
        raw = np.frombuffer(f.read(b - a), dtype=np.uint8)
    dt = meta["dtype"]
    if dt == "BF16":
        return _bf16_to_f32(raw.view(np.uint16)).reshape(meta["shape"])
    if dt in ("F32", "F16"):
        return raw.view(np.float32 if dt == "F32" else np.float16).astype(np.float32).reshape(meta["shape"])
    raise ValueError(f"{name}: dtype {dt} not handled")


def route_kept(h: np.ndarray, Wr: np.ndarray, bias: np.ndarray, keeps: list, k: int, chunk: int = 4096) -> list:
    """noaux_tc selection for several keep sets at once: top-k of sigmoid(h Wr^T) + bias with
    the bias of non-kept experts = -inf (None = all experts). The router scores are computed once per chunk."""
    biases = []
    for keep in keeps:
        b = bias.astype(np.float32).copy()
        if keep is not None:
            b[~keep] = -np.inf
        biases.append(b)
    outs = [np.empty((h.shape[0], k), dtype=np.int16) for _ in keeps]
    for s in range(0, h.shape[0], chunk):
        x = _bf16_to_f32(h[s:s + chunk]) if h.dtype in (np.int16, np.uint16) else np.asarray(h[s:s + chunk], np.float32)
        p = 1.0 / (1.0 + np.exp(-(x @ Wr.T)))
        for b, out in zip(biases, outs):
            sc = p + b
            sel = np.argpartition(-sc, k - 1, axis=1)[:, :k]
            order = np.argsort(-np.take_along_axis(sc, sel, 1), axis=1)
            out[s:s + chunk] = np.take_along_axis(sel, order, 1)
    return outs


def cmd_extract(a) -> None:
    """one pass over each layer's dump; with --keep-freq N1,N2,... one trace per count (--out with {n})."""
    dump = Path(a.dump)
    meta = json.loads((dump / "meta.json").read_text())
    layers = [int(x) for x in a.layers.split(",")] if a.layers else list(meta["layers"])
    sets = {}                                                  # label -> {layer: kept ids} (None = all experts)
    if a.keep:
        sets["keep"] = {int(l): v for l, v in json.loads(Path(a.keep).read_text())["layers"].items()}
    if a.keep_freq:
        rep = json.loads(Path(a.reap).read_text())["experts"]
        for n in (int(x) for x in str(a.keep_freq).split(",")):
            sets[str(n)] = {l: sorted(int(e) for e in rep[str(l)]["rank_freq"][:n]) for l in layers}
    if not sets:
        sets["full"] = None
    model = Path(a.model) if a.model else None
    index = json.loads(Path(a.index or (model / "model.safetensors.index.json")).read_text())["weight_map"] if model else None
    per = {lab: [] for lab in sets}
    for il in layers:
        f = dump / f"layer_{il:03d}.npz"
        labs = list(sets)
        if sets.get("full", 0) is None and len(sets) == 1:
            per["full"].append(np.asarray(_npz_memmap(f, "ids")).astype(np.int16))
        else:
            if model is None:
                sys.exit("extract: re-routing among a keep set needs --model (router weights)")
            pre = f"language_model.model.layers.{il}.block_sparse_moe."
            Wr = _st_tensor(model, index, pre + "gate.weight")
            bias = _st_tensor(model, index, pre + "gate.e_score_correction_bias")
            keeps = []
            for lab in labs:
                kp = np.zeros(Wr.shape[0], dtype=bool)
                kp[sets[lab][il]] = True
                keeps.append(kp)
            for lab, ids in zip(labs, route_kept(_npz_memmap(f, "h"), Wr, bias, keeps, a.top_k)):
                per[lab].append(ids)
        print(f"layer {il}: done", file=sys.stderr, flush=True)
    for lab, ids_all in per.items():
        T = min(x.shape[0] for x in ids_all)
        out = a.out.format(n=lab) if "{n}" in a.out else a.out
        ks = sets[lab]
        np.savez(out, ids=np.stack([x[:T] for x in ids_all]), layers=np.array(layers), n_expert=a.n_expert,
                 top_k=a.top_k, stride=int(meta.get("stride", 1)),
                 keep=json.dumps({str(k): v for k, v in ks.items()} if ks else None),
                 source=json.dumps({"dump": str(dump), "tokens": meta.get("tokens"), "rows": meta.get("rows"),
                                    "keep": lab}))
        print(f"wrote {out}: {len(layers)} layers x {T} tokens ({lab})", file=sys.stderr)


# ================================================================================================ residency policies

def _freq(ids_l: np.ndarray, n_expert: int) -> np.ndarray:
    return np.bincount(ids_l.ravel().astype(np.int64), minlength=n_expert)


def residency_static(ids_l: np.ndarray, n_expert: int, kept: np.ndarray, vram: int, ram: int, warm: float):
    """tier per expert (0 VRAM, 1 RAM, 2 NVMe) by frequency on the first `warm` fraction of tokens; the rest is
    scored. Returns per-token counts (v, w, d) of the scored tokens."""
    T = ids_l.shape[0]
    t0 = int(T * warm)
    f = _freq(ids_l[:t0], n_expert).astype(np.float64)
    f[~kept] = -1                                              # never routed in a pruned build
    order = np.argsort(-f, kind="stable")
    tier = np.full(n_expert, 2, dtype=np.int8)
    tier[order[:vram]] = 0
    tier[order[vram:vram + ram]] = 1
    t = tier[ids_l[t0:].astype(np.int64)]
    return np.stack([(t == 0).sum(1), (t == 1).sum(1), (t == 2).sum(1)], 1)


def residency_cache(ids_l: np.ndarray, n_expert: int, kept: np.ndarray, vram: int, ram: int, warm: float, kind: str):
    """VRAM = static hot set (by frequency over the warm-up part); RAM = an LRU or LFU cache of `ram` slots over the
    rest, filled from NVMe on a miss (the slot table starts with the next most frequent experts)."""
    T = ids_l.shape[0]
    t0 = int(T * warm)
    f = _freq(ids_l[:t0], n_expert).astype(np.float64)
    f[~kept] = -1
    order = np.argsort(-f, kind="stable")
    in_vram = np.zeros(n_expert, dtype=bool)
    in_vram[order[:vram]] = True
    cache: OrderedDict = OrderedDict((int(e), 0) for e in order[vram:vram + ram])
    hits = np.zeros(n_expert, dtype=np.int64)                  # LFU counts (since start, warm-up included)
    for t in range(t0):
        for e in ids_l[t]:
            hits[e] += 1
    out = np.zeros((T - t0, 3), dtype=np.int64)
    for i, t in enumerate(range(t0, T)):
        v = w = d = 0
        for e in ids_l[t].astype(np.int64):
            hits[e] += 1
            if in_vram[e]:
                v += 1
            elif e in cache:
                w += 1
                if kind == "lru":
                    cache.move_to_end(e)
            else:
                d += 1
                if ram > 0:
                    if len(cache) >= ram:
                        if kind == "lru":
                            cache.popitem(last=False)
                        else:                                  # LFU: evict the least used resident expert
                            victim = min(cache.keys(), key=lambda x: hits[x])
                            del cache[victim]
                    cache[e] = 0
        out[i] = (v, w, d)
    return out


# ======================================================================================================= cost model

class Rates:
    """every number the model uses, in GB/s or ms; printed with each result."""

    def __init__(self, a):
        self.bpe = a.bytes_per_expert                          # bytes of one routed expert in its format
        self.cpu_gbs = a.cpu_gbs                               # CPU lane: expert bytes it gets through a second
        self.zc_gbs_gpu = a.zc_gbs                             # one GPU reading pinned RAM (measured kernel rate)
        self.n_gpu = a.zc_gpus                                 # GPUs in the fan-out
        self.bus_gbs = a.bus_gbs                               # RAM bus cap for ZC + CPU together
        self.nvme_gbs = a.nvme_gbs
        self.vram_gbs = a.vram_gbs                             # effective VRAM read rate for VRAM-resident experts
        self.oc, self.oz = a.oc_ms, a.oz_ms
        self.t_dense = a.dense_ms                              # per layer, everything but the routed experts

    def ms(self, gbs: float) -> float:
        return self.bpe / (gbs * 1e9) * 1e3

    def describe(self) -> dict:
        return {k: getattr(self, k) for k in ("bpe", "cpu_gbs", "zc_gbs_gpu", "n_gpu", "bus_gbs", "nvme_gbs",
                                              "vram_gbs", "oc", "oz", "t_dense")}


def request_ms(r: Rates, v: int, w: int, d: int) -> float:
    """one layer's routed experts for one token: VRAM picks on the GPU; warm (RAM) and cold (NVMe) picks split between
    the CPU and ZC lanes so they finish together; returns ms (without t_dense). When both lanes stream and their sum
    exceeds the RAM bus, both slow down in proportion (measured: 3 GPUs + 16 threads share ~123 GB/s)."""
    zc_full = min(r.n_gpu * r.zc_gbs_gpu, r.bus_gbs) if r.n_gpu > 0 else 0.0
    both = zc_full + r.cpu_gbs
    shrink = min(1.0, r.bus_gbs / both) if both > 0 else 1.0
    disk = r.ms(r.nvme_gbs)
    best = math.inf
    for sw in range(w + 1):                                    # warm picks on the ZC lane
        for sd in range(d + 1):                                # cold picks on the ZC lane
            nz, nc = sw + sd, (w - sw) + (d - sd)
            if nz and zc_full <= 0:
                continue
            f = shrink if (nz and nc) else 1.0
            tc = nc * r.ms(r.cpu_gbs * f) + (d - sd) * disk + (r.oc if nc else 0.0) if nc else 0.0
            tz = nz * r.ms(zc_full * f) + sd * disk + (r.oz if nz else 0.0) if nz else 0.0
            best = min(best, max(tc, tz))
    return best + v * r.ms(r.vram_gbs)


def price(counts: np.ndarray, r: Rates, n_layers: int = K31_LAYERS_MOE) -> dict:
    """counts (n, 3) of (v, w, d) per request, pooled over the traced layers -> per-token time for n_layers."""
    uniq, inv = np.unique(counts, axis=0, return_inverse=True)
    costs = np.array([request_ms(r, int(v), int(w), int(d)) for v, w, d in uniq])
    per_req = costs[inv.ravel()]
    t_layer = float(per_req.mean()) + r.t_dense
    t_tok = n_layers * t_layer
    K = counts.sum(1).mean()
    frac = counts.mean(0) / K
    return {"tok_s": 1e3 / t_tok, "ms_per_token": t_tok, "experts_ms_per_layer": float(per_req.mean()),
            "frac_vram": float(frac[0]), "frac_ram": float(frac[1]), "frac_nvme": float(frac[2]),
            "nvme_gb_per_token": float(counts[:, 2].mean() * n_layers * r.bpe / 1e9),
            "ram_gb_per_token": float(counts[:, 1].mean() * n_layers * r.bpe / 1e9)}


# ============================================================================================================ replay

def slots(budget_gb: float, bpe: float, n_layers: int, n_kept: int) -> int:
    return int(max(0, min(n_kept, math.floor(budget_gb * 1e9 / (n_layers * bpe)))))


def replay(trace: dict, a) -> dict:
    ids = trace["ids"]
    E = int(trace["n_expert"])
    keep = json.loads(str(trace["keep"])) if trace.get("keep") is not None else None
    r = Rates(a)
    vram_slots = slots(a.vram_gb, r.bpe, K31_LAYERS_MOE, a.n_kept)
    ram_slots = slots(max(0.0, a.ram_gb - a.ram_reserve_gb), r.bpe, K31_LAYERS_MOE, a.n_kept - vram_slots)
    pooled = []
    for li, il in enumerate(trace["layers"]):
        kept = np.zeros(E, dtype=bool)
        if keep and keep != "null":
            kept[keep[str(int(il))]] = True
        else:
            kept[:] = True
        if a.policy == "static":
            pooled.append(residency_static(ids[li], E, kept, vram_slots, ram_slots, a.warm))
        else:
            pooled.append(residency_cache(ids[li], E, kept, vram_slots, ram_slots, a.warm, a.policy))
    res = price(np.concatenate(pooled), r)
    res.update({"vram_slots_per_layer": vram_slots, "ram_slots_per_layer": ram_slots, "n_kept": a.n_kept,
                "policy": a.policy, "rates": r.describe()})
    return res


def replay_freq(counts_by_layer: dict, a, n_samples: int = 4000, seed: int = 0) -> dict:
    """approximate replay from per-layer usage counts (reap-report 'count' / score_freq over all 92 layers) when no
    keep-set trace exists: the build keeps the top n_kept by count; a pruned build's routing mass is taken as the kept
    experts' counts renormalized (masked routing hands a dropped pick to some kept expert); static placement (VRAM =
    most used, then RAM, then NVMe); each request = K independent draws from the layer's kept distribution (no
    temporal locality, so it suits static placement only). Priced with the same lane model."""
    r = Rates(a)
    vram_slots = slots(a.vram_gb, r.bpe, K31_LAYERS_MOE, a.n_kept)
    ram_slots = slots(max(0.0, a.ram_gb - a.ram_reserve_gb), r.bpe, K31_LAYERS_MOE, a.n_kept - vram_slots)
    rng = np.random.default_rng(seed)
    K = 16
    pooled = []
    for il, cnt in sorted(counts_by_layer.items()):
        c = np.asarray(cnt, dtype=np.float64)
        order = np.argsort(-c, kind="stable")[:a.n_kept]
        p = c[order] / c[order].sum()
        tier = np.full(a.n_kept, 2, dtype=np.int8)
        tier[:vram_slots] = 0
        tier[vram_slots:vram_slots + ram_slots] = 1
        picks = np.stack([rng.choice(a.n_kept, K, replace=False, p=p) for _ in range(n_samples // 8)])
        t = tier[picks]
        pooled.append(np.stack([(t == 0).sum(1), (t == 1).sum(1), (t == 2).sum(1)], 1))
    res = price(np.concatenate(pooled), r)
    res.update({"vram_slots_per_layer": vram_slots, "ram_slots_per_layer": ram_slots, "n_kept": a.n_kept,
                "policy": "freq-static (approx.)", "layers": len(counts_by_layer), "rates": r.describe()})
    return res


def load_trace(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: (z[k] if z[k].shape != () else z[k][()]) for k in z.files}


def add_rate_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-kept", type=int, required=True, help="experts per layer in the build")
    p.add_argument("--bytes-per-expert", type=float, default=K31_EXPERT_PARAMS * 1.5625 / 8,
                   help="bytes of one routed expert (default IQ1_S: 6.45e6)")
    p.add_argument("--vram-gb", type=float, default=30.0, help="VRAM left for routed experts (all GPUs)")
    p.add_argument("--ram-gb", type=float, default=123.0)
    p.add_argument("--ram-reserve-gb", type=float, default=20.0, help="RAM not available to experts (OS, services)")
    p.add_argument("--policy", choices=("static", "lru", "lfu"), default="static")
    p.add_argument("--warm", type=float, default=0.25, help="leading fraction of the trace used to rank / warm")
    p.add_argument("--cpu-gbs", type=float, default=22.0 * 1.27, help="CPU lane, expert bytes a second (default: 16 Zen 4 cores, AVX-512 IQ1_S kernel)")
    p.add_argument("--zc-gbs", type=float, default=23.5, help="one 3090 running the IQ1_S kernel on pinned RAM")
    p.add_argument("--zc-gpus", type=int, default=2, help="GPUs in the zero-copy fan-out")
    p.add_argument("--bus-gbs", type=float, default=123.0, help="RAM bus cap for ZC + CPU together (4 channels)")
    p.add_argument("--nvme-gbs", type=float, default=1.4, help="NVMe read rate (default: a hot, throttled drive; ~3.5 when cooled)")
    p.add_argument("--vram-gbs", type=float, default=650.0)
    p.add_argument("--oc-ms", type=float, default=0.02)
    p.add_argument("--oz-ms", type=float, default=0.09, help="fan-out bookkeeping a request (3 ops x ~30 us)")
    p.add_argument("--dense-ms", type=float, default=1.0, help="per layer: attention, shared expert, norms, launches")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract", help="routing traces from calibration MoE dumps")
    e.add_argument("--dump", required=True)
    e.add_argument("--layers", default=None)
    e.add_argument("--keep", default=None, help="a *.prune.json sidecar or {layers: {L: ids}}")
    e.add_argument("--keep-freq", default=None, help="keep the top N of rank_freq from --reap; N1,N2,... -> one trace each")
    e.add_argument("--reap", default=None)
    e.add_argument("--model", default=None, help="HF checkpoint dir (router weights), needed with a keep set")
    e.add_argument("--index", default=None)
    e.add_argument("--n-expert", type=int, default=896)
    e.add_argument("--top-k", type=int, default=16)
    e.add_argument("--out", required=True, help="output .npz; use {n} for one file per --keep-freq count")
    r = sub.add_parser("replay", help="one trace through one tier configuration")
    r.add_argument("trace")
    add_rate_args(r)
    r.add_argument("--json", action="store_true")
    q = sub.add_parser("replay-freq", help="approximate: per-layer usage counts (all layers), static placement")
    q.add_argument("reap", help="reap_plan report: experts[layer]['count']")
    add_rate_args(q)
    q.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "replay-freq":
        rep = json.loads(Path(a.reap).read_text())["experts"]
        res = replay_freq({int(l): v["count"] for l, v in rep.items()}, a)
        print(json.dumps(res, indent=1) if a.json else
              f"{a.n_kept} experts, {a.bytes_per_expert / 1e6:.2f} MB each, RAM {a.ram_gb} GB: {res['tok_s']:.2f} tok/s "
              f"(approx.); picks VRAM {res['frac_vram']:.1%} RAM {res['frac_ram']:.1%} NVMe {res['frac_nvme']:.1%}")
        return 0
    if a.cmd == "extract":
        cmd_extract(a)
        return 0
    res = replay(load_trace(a.trace), a)
    if a.json:
        print(json.dumps(res, indent=1))
    else:
        print(f"{a.n_kept} experts, {a.bytes_per_expert / 1e6:.2f} MB each, VRAM {a.vram_gb} GB, RAM {a.ram_gb} GB, "
              f"{a.policy}: {res['tok_s']:.2f} tok/s ({res['ms_per_token']:.0f} ms/token); picks VRAM "
              f"{res['frac_vram']:.1%} RAM {res['frac_ram']:.1%} NVMe {res['frac_nvme']:.1%} "
              f"({res['nvme_gb_per_token']:.2f} GB/token from NVMe)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

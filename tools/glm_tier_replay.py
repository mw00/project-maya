#!/usr/bin/env python3
"""Replay STRATA_GLM_ROUTE_LOG traces through a model of the GLM fast path's VRAM expert tier - the engine's own rules
and alternatives - to measure an eviction or promotion idea on recorded routes before building it.

Recording a trace: start the engine as usual with STRATA_GLM_ROUTE_LOG=<prefix> (the config's "env" or the shell).
Every route goes to <prefix>.<device> (0 = a split's first GPU, 1 = its second), one line per route:
    layer e0 .. e7 fetch miss cpu [| n0 .. n15]
the route's experts in selection order, then bit masks over e0..e7: fetched over PCIe, read from disk, computed by
the CPU lane (an expert in none of them was in VRAM); the near misses after '|' are there when the build logs them.
Run the workload to study, stop the engine, then replay with the usage file of that run (the engine seeds its
residents from it and rewrites it when it stops):

    glm_tier_replay.py <prefix> <pack>/expert_usage.txt <split> <slots0> <slots1> [plan] [--policy ...]

<split> is the split layer; <slotsN> the per-layer slots of device N's pool (the "expert pool ..., N-N slots/layer"
line of the engine's log); [plan] the CPU lane's split table as 9 digits, plan[f] = how many of a route's f RAM-tier
misses the CPU takes (the "of 0..8 RAM-tier experts the CPU takes" line; default 011233456). Validate first: trace a
run with STRATA_GLM_PREFETCH_N unset and compare "hit" (the replay) with "live" (what the trace's own masks show) -
with the engine's policy they agree within a few points.

The model (src/kernels/cuda/glm_fast.cu route kernel, src/core/glm_fast_path.cu service thread + fast_boundary):
  - a MoE layer's slots = residents + 3 spares (gf::kSpares); the start fills the residents with the layer's
    most-used experts by the usage file;
  - aged counts: seeded max(1, 32 * usage / layer max), +1 per route use, halved every 4096 routes of a device;
  - of a route's f misses the CPU lane takes plan[f], the coldest by count (ties: the later in route order); the
    others are fetched over PCIe into the layer's free spares (promoted: resident from then on) or into scratch;
  - at each token boundary a layer's used spares are refilled by evicting residents - the engine evicts the lowest
    aged count (ties: the least recently used);
  - draft routes (the NextN layer, --nextn) leave their misses out from route rank --keep on (STRATA_GLM_MTP_KEEP).

What it leaves out: time - "off-card" is per route max(n_cpu * c_ms, n_fetch * p_ms) (set both from your own
STRATA_GLM_TIMING numbers), with no overlap between layers, no pipeline between the devices and no prompt reads;
the RAM tier and the disk are one level (a disk miss counts as a fetch); the prefetch (its copies are not routes);
anything the engine does to its residents outside these rules.

Policies (--policy key:value,key:value; several --policy flags are compared in one run):
  evict:lfu   the engine (default)        evict:lru     least recently used
  evict:slru  segmented LRU (pfrac:0.8)   evict:slfu    the engine's rule among residents not re-used since arrival
  evict:protect<N>  keep residents used in the last N tokens, else the engine's rule
  evict:reuse  furthest predicted next use from the expert's mean reuse gap (alpha:0.3, def:400 routes)
  evict:soft   lowest decayed use count where a near miss adds w (w:0.3, nn:16 near misses)
  evict:nlru   LRU where a near miss refreshes too (delta:0)
  evict:belady furthest real next use - hindsight, the ceiling for any eviction rule
  pick:cnt (the engine: the CPU lane takes the coldest) | pick:recent (the least recently used) | pick:soft
  promote:all  the CPU lane's experts are also copied into free spares ("copies" counts them)
  spares:<n>   spare slots a layer (default 3)        half:<routes>  the count-halving period (default 4096)
"""
import argparse
import bisect
import glob
from collections import defaultdict


def load_usage(path):
    """layer -> {expert: routes} from an expert_usage.txt ("layer e:count e:count ...", '#' comments)."""
    usage = {}
    with open(path) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            usage[int(fields[0])] = {int(e): int(c) for e, c in (x.split(":") for x in fields[1:])}
    return usage


def parse_route(line):
    """(layer, [e0..e7], off-card mask, [near misses]) from one route-log line, or None for a short line."""
    f = line.split()
    if len(f) < 12:
        return None
    near = [int(x) for x in f[f.index("|") + 1:]] if "|" in f else []
    return int(f[0]), [int(x) for x in f[1:9]], int(f[9]) | int(f[10]) | int(f[11]), near


def load_traces(prefix):
    """device -> routes, from <prefix>.<device> files."""
    traces = {}
    for fn in sorted(glob.glob(prefix + ".*")):
        dev = fn.rsplit(".", 1)[1]
        if not dev.isdigit():
            continue
        with open(fn) as f:
            traces[int(dev)] = [r for r in (parse_route(line) for line in f) if r is not None]
    return traces


def parse_policy(text):
    p = {"evict": "lfu", "pick": "cnt", "spares": 3, "half": 4096}
    for kv in (text or "").split(","):
        if ":" in kv:
            k, v = kv.split(":", 1)
            p[k] = v
    p["spares"] = int(p["spares"])
    p["half"] = int(p["half"])
    return p


def count_tokens(rows):
    """A token starts where the layer index goes back (the routes of one device run in layer order)."""
    return max(1, sum(1 for k in range(1, len(rows)) if rows[k][0] <= rows[k - 1][0]) + 1)


def replay(rows, usage, cap, policy, plan, nextn=45, keep=2, p_ms=0.67, c_ms=0.32):
    """One device's routes through the tier model; per-token averages of what the tier did."""
    P = parse_policy(policy)
    S = P["spares"]
    res = defaultdict(dict)          # layer -> {expert: last-use clock}: the residents
    cnt, dcnt = {}, {}               # (layer, e) -> aged count (host / device)
    spares_left = {}                 # layer -> spares still free this token
    lastuse, gap, soft, ntouch = {}, {}, {}, {}
    prot = defaultdict(set)
    for il in {r[0] for r in rows}:
        u = usage.get(il, {})
        top = max(u.values()) if u else 1
        for e, c in u.items():
            cnt[(il, e)] = dcnt[(il, e)] = max(1, 32 * c // top)
        for e in sorted(u, key=lambda x: -u[x])[:cap - S]:
            res[il][e] = 0
        spares_left[il] = S
    pos = defaultdict(list)          # (layer, e) -> route indices that use it (hindsight)
    for j, (il_, ids_, _, _n) in enumerate(rows):
        for e_ in ids_:
            pos[(il_, e_)].append(j)

    def next_use(l, e, i):
        q = pos.get((l, e))
        if not q:
            return 1 << 60
        k = bisect.bisect_right(q, i)
        return q[k] if k < len(q) else 1 << 60

    def victim(l2, R, clock, ridx):
        ev = P["evict"]
        if ev == "lru":
            return min(R, key=lambda e: R[e])
        if ev == "belady":
            return max(R, key=lambda e: next_use(l2, e, ridx))
        if ev == "soft":
            return min(R, key=lambda e: (soft.get((l2, e), 0.0), R[e]))
        if ev == "nlru":
            dl = int(P.get("delta", 0))
            return min(R, key=lambda e: max(R[e], ntouch.get((l2, e), -1 << 40) - dl))
        if ev in ("slru", "slfu"):
            pr = prot[l2]
            pool = [e for e in R if e not in pr] or list(R)
            v = (min(pool, key=lambda e: R[e]) if ev == "slru"
                 else min(pool, key=lambda e: (cnt.get((l2, e), 0), R[e])))
            pr.discard(v)
            return v
        if ev.startswith("reuse"):
            dflt = float(P.get("def", 400))
            return max(R, key=lambda e: R[e] + gap.get((l2, e), dflt))
        if ev.startswith("protect"):
            w = int(ev[7:] or 64) * 42
            cand = [e for e in R if R[e] + w <= clock]
            return (min(cand, key=lambda e: (cnt.get((l2, e), 0), R[e])) if cand
                    else min(R, key=lambda e: R[e]))
        return min(R, key=lambda e: (cnt.get((l2, e), 0), R[e]))      # the engine

    clock = routes = 0
    hits = uses = fetch = cpu = promo = live_hits = copies = 0
    offms = 0.0
    prev_il = None
    for ridx, (il, ids, live_off, near) in enumerate(rows):
        if prev_il is not None and il <= prev_il:          # a new token: the boundary refills the spares
            for l2, left in spares_left.items():
                R = res[l2]
                for _ in range(S - left):
                    if not R:
                        break
                    del R[victim(l2, R, clock, ridx)]
                spares_left[l2] = S
        prev_il = il
        clock += 1
        routes += 1
        if routes % P["half"] == 0:
            for k in soft:
                soft[k] *= 0.5
            for k in cnt:
                cnt[k] >>= 1
            for k in dcnt:
                dcnt[k] >>= 1
        wn = float(P.get("w", 0.3))
        for e in near[:int(P.get("nn", 16))]:
            soft[(il, e)] = soft.get((il, e), 0.0) + wn
            ntouch[(il, e)] = clock
        for e in ids:
            soft[(il, e)] = soft.get((il, e), 0.0) + 1.0
        draft = il == nextn
        skip_from = keep if draft else 8
        R = res[il]
        cand = []                                          # the CPU lane's pick among this route's misses
        for i, e in enumerate(ids):
            c = dcnt[(il, e)] = dcnt.get((il, e), 0) + 1
            if i < skip_from and e not in R:
                cand.append((i, c, e))
        host = set()
        for _ in range(min(plan[min(len(cand), 8)], len(cand))):
            best = None
            for i, c, e in cand:
                if i in host:
                    continue
                key = (c if P["pick"] == "cnt" else soft.get((il, e), 0.0) if P["pick"] == "soft"
                       else lastuse.get((il, e), -1))
                if best is None or key <= best[0]:
                    best = (key, i)
            host.add(best[1])
        n_cpu = n_fetch = 0
        for i, e in enumerate(ids):
            if not draft:                                  # a draft's skipped misses leave no bit in the log:
                uses += 1                                  # the hit rates count the trunk layers only
                if not (live_off >> i) & 1:
                    live_hits += 1
            cnt[(il, e)] = cnt.get((il, e), 0) + 1
            if e in R:
                if not draft:
                    hits += 1
                R[e] = clock
                if P["evict"] in ("slru", "slfu"):
                    pr = prot[il]
                    pr.add(e)
                    if len(pr) > int((cap - S) * float(P.get("pfrac", 0.8))):
                        pr.discard(min(pr, key=lambda x: R.get(x, -1)))   # the oldest back on probation
            elif i >= skip_from:
                pass                                       # a draft leaves it out
            elif i in host:
                cpu += 1
                n_cpu += 1
                if P.get("promote") == "all" and spares_left[il] > 0:
                    spares_left[il] -= 1
                    R[e] = clock
                    promo += 1
                    copies += 1
            else:
                fetch += 1
                n_fetch += 1
                if spares_left[il] > 0:
                    spares_left[il] -= 1
                    R[e] = clock
                    promo += 1
            if (il, e) in lastuse:
                g0 = clock - lastuse[(il, e)]
                a = float(P.get("alpha", 0.3))
                gap[(il, e)] = g0 if (il, e) not in gap else (1 - a) * gap[(il, e)] + a * g0
            lastuse[(il, e)] = clock
        offms += max(n_cpu * c_ms, n_fetch * p_ms)
    toks = count_tokens(rows)
    return dict(hit=100 * hits / max(1, uses), live=100 * live_hits / max(1, uses), fetch=fetch / toks,
                cpu=cpu / toks, promo=promo / toks, copies=copies / toks, offms=offms / toks, toks=toks)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("prefix", help="the STRATA_GLM_ROUTE_LOG prefix (<prefix>.0, <prefix>.1 are read)")
    ap.add_argument("usage", help="the run's expert_usage.txt")
    ap.add_argument("split", type=int, help="the split layer (informational)")
    ap.add_argument("cap_head", type=int, help="device 0's slots per layer")
    ap.add_argument("cap_tail", type=int, help="device 1's slots per layer")
    ap.add_argument("plan", nargs="?", default="011233456", help="the CPU lane's split table, 9 digits")
    ap.add_argument("--policy", action="append", default=None, help="see the module docstring (repeatable)")
    ap.add_argument("--nextn", type=int, default=45, help="the NextN (draft) layer")
    ap.add_argument("--keep", type=int, default=2, help="STRATA_GLM_MTP_KEEP of the traced run")
    ap.add_argument("--p_ms", type=float, default=0.67, help="ms per expert fetched over PCIe")
    ap.add_argument("--c_ms", type=float, default=0.32, help="ms per expert on the CPU lane")
    args = ap.parse_args(argv)
    if len(args.plan) != 9 or not args.plan.isdigit():
        ap.error("plan must be 9 digits, plan[f] = experts the CPU takes of f misses")
    plan = [int(ch) for ch in args.plan]
    usage = load_usage(args.usage)
    traces = load_traces(args.prefix)
    if not traces:
        ap.error(f"no trace files {args.prefix}.<device>")
    for pol in args.policy or ["evict:lfu,pick:cnt"]:
        tot = worst = 0.0
        out = []
        for dev, rows in sorted(traces.items()):
            cap = args.cap_head if dev == 0 else args.cap_tail
            r = replay(rows, usage, cap, pol, plan, args.nextn, args.keep, args.p_ms, args.c_ms)
            out.append(f"dev{dev}: hit {r['hit']:5.1f} % (live {r['live']:5.1f}) fetch {r['fetch']:5.1f} "
                       f"cpu {r['cpu']:5.1f} promo {r['promo']:4.1f} (+copies {r['copies']:4.1f}) /tok, "
                       f"off-card {r['offms']:5.1f} ms/tok")
            tot += r["offms"]
            worst = max(worst, r["offms"])
        # the two devices run as a pipeline: the slower one's off-card time paces a token
        print(f"{args.plan} {pol:34s} | " + " | ".join(out) + f" | sum {tot:5.1f}, slower {worst:5.1f} ms/tok")


if __name__ == "__main__":
    main()

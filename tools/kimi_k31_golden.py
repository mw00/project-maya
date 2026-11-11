#!/usr/bin/env python3
"""tools/kimi_k31_golden.py - golden vectors for the Kimi K3.1 MoE router and SiTU-GLU, for engine kernels'
*_parity --selftest harnesses. numpy only; no GPU, no network; a second or two.

Both functions are ref/kimi_k31.py's own code paths (its default float32 mode), pulled out of moe_ffn so a kernel
can be held to them without running a model; tools/test_kimi_k31_golden.py checks that they still equal the
oracle's moe_ffn bit for bit on the synthetic fixture.

Router (the released config: E = 896 routed experts, k = 16, routed_scaling_factor 1.0, noaux_tc):
    probs  = sigmoid(logits)                          float64, rounded to float32
    biased = float32(probs + e_score_correction_bias)  the bias picks experts, it never weights them
    ids    = the k largest biased scores, descending, ties toward the LOWER expert id (ggml_top_k, top_k_desc)
    w      = probs[ids] / (sum(probs[ids]) + 1e-20) * routed_scale      float64, rounded to float32
SiTU-GLU (configuration: activation_situ_beta 4.0, activation_situ_linear_beta 25.0):
    situ(g, u) = beta*tanh(g/beta)*sigmoid(g) * lbeta*tanh(u/lbeta)     float64, rounded to float32

Router cases (each E = 896 unless noted, logits [E, T] float32, bias [E] float32):
    rand      ordinary logits and biases, T = 64
    decode1   one token (the decode shape)
    window16  16 tokens (a verify window)
    ties      bitwise-equal biased scores straddling the cut, at the top and everywhere (all 896 equal):
              the lower-id rule decides the set
    bias_flip biases large enough that the biased top-k differs from the unbiased top-k (weights must still
              come from the unbiased probs)
    tiny      every selected sigmoid near 1e-13..1e-9: +1e-20 renormalisation gives ordinary weights, a 2^-14
              clamp (ref/glm.py's, llama.cpp's) gives ~1e-9 - the one case that separates them
    saturate  logits >= 18, where sigmoid rounds to exactly 1.0f: the bias alone orders, with ties
    inf       +-inf logits (sigmoid 1 and 0)
    scaled    routed_scale 2.5 (not the released value) so a kernel that drops the scale fails somewhere
SiTU cases (g, u float32 pairs): gate sweep (u = 1), up sweep (g = 1), random pairs over 1e-30..1e4,
    the saturation edges, signed zeros and denormals, and +-inf.

Out of range on purpose: logits below about -88 for every selected expert (a float32 sigmoid underflows to 0
there, the oracle's float64 one does not; no router produces that), and NaN inputs.

Output (--out DIR): ref/kimi_k31.py's raw-dump layout (`write_raw_dumps`: one `<name>.f32` file per array in ggml
order, features fastest, int64 stored as int32, plus manifest.txt `name d0 d1 ...`) and golden.json: the cases,
the formulas, the tolerance rules with the ULP budgets measured here against a plain float32 kernel
(`emulate_*`), and the SHA-256 of every file.

    python tools/kimi_k31_golden.py --out DIR [--budgets]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pathlib
import sys
import warnings

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ref"))
import kimi_k31 as K  # noqa: E402  (ref/kimi_k31.py)

F32, F64 = np.float32, np.float64
SEED = 3116
E_RELEASED, K_RELEASED = 896, 16
SITU_BETA, SITU_LINEAR_BETA = 4.0, 25.0
RENORM_EPS = 1e-20
MIN_NORMAL_F32 = float(np.finfo(F32).tiny)           # 2^-126
SITU_UNDERFLOW_G = -87.0     # below this a float32 sigmoid(g) leaves the normal range (expf(-g) overflows near -88.7)
SITU_UNDERFLOW_ABS = 1e-35   # there: |out - ref| <= this (|ref| <= 100 * sigmoid(-87) ~ 1.7e-36)
VERSION = 2   # 2: route cases eps_dom and denormal; "denormal" left out of the router budget


# --------------------------------------------------------------------------- the two functions (the oracle's)

def route(logits: np.ndarray, bias: np.ndarray, k: int, scale: float = 1.0, norm: bool = True,
          eps: float = RENORM_EPS) -> dict:
    """ref/kimi_k31.py moe_ffn's routing for logits [E, T] (float32) and bias [E] (float32)."""
    logits = np.asarray(logits, dtype=F32)
    bias = np.asarray(bias, dtype=F32)
    E, T = logits.shape
    probs = K.sigmoid(logits, F32)
    biased = (probs + bias[:, None]).astype(F32)
    ids = np.empty((k, T), dtype=np.int64)
    w_raw = np.empty((k, T), dtype=F32)
    w_norm = np.empty((k, T), dtype=F32)
    w = np.empty((k, T), dtype=F32)
    margin = np.zeros(T, dtype=F64)
    for t in range(T):
        sel = K.top_k_desc(biased[:, t], k)
        ids[:, t] = sel
        x = probs[sel, t].astype(F64)
        w_raw[:, t] = x
        if norm:
            x = x / (x.sum() + eps)
        w_norm[:, t] = x
        x = x * scale
        w[:, t] = x
        ordered = np.sort(biased[:, t].astype(F64))[::-1]
        margin[t] = ordered[k - 1] - ordered[k] if E > k else np.inf
    return {"probs": probs, "biased": biased, "ids": ids, "w_raw": w_raw, "w_norm": w_norm, "w": w,
            "margin": margin}


def situ(g: np.ndarray, u: np.ndarray, beta: float = SITU_BETA, lbeta: float = SITU_LINEAR_BETA,
         dt=F32) -> np.ndarray:
    """ref/kimi_k31.py situ (float64 inside), rounded to dt."""
    with np.errstate(over="ignore"):                    # exp(-g) overflows to inf for g << 0: sigmoid 0, as intended
        return K.situ(np.asarray(g, dtype=F32), np.asarray(u, dtype=F32), beta, lbeta, dt)


# --------------------------------------------------------------------------- what a plain float32 kernel does

def emulate_route(logits, bias, k, scale=1.0, eps=RENORM_EPS):
    """A straightforward float32 kernel: 1/(1+expf(-x)), float32 bias add, the same tie rule, a float32 sum in
    pick order, float32 division and scale. The ULP budgets are measured from this."""
    l32 = np.asarray(logits, dtype=F32)
    b32 = np.asarray(bias, dtype=F32)
    with np.errstate(over="ignore"):
        p = (F32(1) / (F32(1) + np.exp(-l32))).astype(F32)
    s_b = (p + b32[:, None]).astype(F32)
    T = l32.shape[1]
    ids = np.empty((k, T), dtype=np.int64)
    w = np.empty((k, T), dtype=F32)
    for t in range(T):
        sel = np.argsort(-s_b[:, t].astype(F64), kind="stable")[:k]
        ids[:, t] = sel
        acc = F32(0)
        for e in sel:
            acc = F32(acc + p[e, t])
        w[:, t] = (p[sel, t] / F32(acc + F32(eps)) * F32(scale)).astype(F32)
    return ids, w


def emulate_situ(g, u, beta=SITU_BETA, lbeta=SITU_LINEAR_BETA):
    """A straightforward float32 kernel: beta*tanhf(g/beta) * 1/(1+expf(-g)) * lbeta*tanhf(u/lbeta)."""
    g32, u32 = np.asarray(g, dtype=F32), np.asarray(u, dtype=F32)
    with np.errstate(over="ignore", invalid="ignore"):
        a = (F32(beta) * np.tanh(g32 / F32(beta))).astype(F32)
        s = (F32(1) / (F32(1) + np.exp(-g32))).astype(F32)
        ub = (F32(lbeta) * np.tanh(u32 / F32(lbeta))).astype(F32)
        return (a * s * ub).astype(F32)


# --------------------------------------------------------------------------- error measures and the gates

def ulp_err(got: np.ndarray, ref64: np.ndarray) -> np.ndarray:
    """|got - ref| in float32 ULPs of ref. Below the smallest normal float32 the unit is 2^-126 (an absolute
    floor), so a kernel that flushes denormals to zero is not penalised for it."""
    got = np.asarray(got, dtype=F64)
    ref64 = np.asarray(ref64, dtype=F64)
    with np.errstate(invalid="ignore", over="ignore"):
        return _ulp_err(got, ref64)


def _ulp_err(got, ref64):
    unit = np.maximum(np.spacing(np.abs(ref64.astype(F32))).astype(F64), MIN_NORMAL_F32)
    unit = np.where(np.abs(ref64) < MIN_NORMAL_F32, MIN_NORMAL_F32, unit)
    err = np.abs(got - ref64)
    same_inf = np.isinf(got) & np.isinf(ref64) & (np.sign(got) == np.sign(ref64))
    return np.where(same_inf, 0.0, err / unit)


def tie_tol(score: float) -> float:
    """How close to the cut a score may be for a set difference to be a float32 rounding tie: 4 ULPs."""
    return 4.0 * float(np.spacing(np.abs(F32(score))))


def check_route(gold: dict, ids: np.ndarray, w: np.ndarray, budget_ulp: float) -> tuple[bool, dict]:
    """The router gate. Per token: the selected set equals the golden set, or both sets have k members and every
    expert in their symmetric difference scores within tie_tol of the cut (a near tie; ties broken bitwise -
    margin 0 - must match exactly, since equal inputs give equal scores in any kernel). Weights are matched by
    expert id and must be within budget_ulp of the golden float64 value; on a near-tie token only the common
    experts are compared. Returns (ok, stats)."""
    g_ids, g_w, biased, margin = gold["ids"], gold["w64"], gold["biased"], gold["margin"]
    k, T = g_ids.shape
    bad_sets, near, worst = 0, 0, 0.0
    for t in range(T):
        a, b = [int(i) for i in g_ids[:, t]], [int(i) for i in np.asarray(ids)[:, t]]
        if len(b) != k or len(set(b)) != k:
            bad_sets += 1
            continue
        if set(a) != set(b):
            ok = margin[t] > 0 and K.topk_sets_agree(biased[:, t], a, b, tol=tie_tol(np.sort(biased[:, t])[::-1][k - 1]))
            if not ok:
                bad_sets += 1
                continue
            near += 1
        gw = {e: g_w[i, t] for i, e in enumerate(a)}
        for i, e in enumerate(b):
            if e in gw:
                worst = max(worst, float(ulp_err(np.asarray(w)[i, t], gw[e])))
    ok = bad_sets == 0 and worst <= budget_ulp
    return ok, {"bad_sets": bad_sets, "near_ties": near, "worst_ulp": worst}


def check_situ(gold: dict, out: np.ndarray, budget_ulp: float) -> tuple[bool, dict]:
    """The SiTU gate: within budget_ulp ULPs of the float64 reference wherever g >= SITU_UNDERFLOW_G; below it
    (where any float32 sigmoid leaves the normal range) an absolute bound of SITU_UNDERFLOW_ABS; infinities and
    NaN positions must match."""
    out64 = np.asarray(out, dtype=F64)
    ref64 = gold["ref64"]
    under = np.asarray(gold["g"], dtype=F64) < SITU_UNDERFLOW_G
    e = ulp_err(out64, ref64)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)              # all-NaN slices: counted by nan_mismatch below
        e_norm = e[~under]
        worst = float(np.nanmax(e_norm)) if e_norm.size else 0.0
        abs_under = np.abs(out64[under] - ref64[under])
        worst_under = float(np.nanmax(abs_under)) if abs_under.size else 0.0
    nan_bad = int(np.sum(np.isnan(out64) != np.isnan(ref64)))
    ok = worst <= budget_ulp and worst_under <= SITU_UNDERFLOW_ABS and nan_bad == 0
    return ok, {"worst_ulp": worst, "worst_abs_underflow": worst_under, "nan_mismatch": nan_bad}


# --------------------------------------------------------------------------- the cases

def route_cases(seed: int = SEED) -> dict:
    """name -> (logits [E,T] f32, bias [E] f32, k, scale)."""
    rng = np.random.default_rng(seed)
    E, k = E_RELEASED, K_RELEASED
    cases = {}

    def rand(T, lsd=2.0, bsd=0.3):
        return rng.normal(0, lsd, (E, T)).astype(F32), rng.normal(0, bsd, E).astype(F32)

    cases["rand"] = (*rand(64), k, 1.0)
    cases["decode1"] = (*rand(1), k, 1.0)
    cases["window16"] = (*rand(16, 1.0, 0.05), k, 1.0)

    # ties: 8 tokens, each with a block of bitwise-equal scores straddling rank k-1/k, plus a top tie and an all-equal token
    T = 10
    lg = rng.normal(-2.0, 0.5, (E, T)).astype(F32)
    bs = np.zeros(E, dtype=F32)
    for t in range(8):
        lg[:, t] = rng.normal(-3.0, 0.5, E).astype(F32)
        top = rng.choice(E, k - 4, replace=False)
        lg[top, t] = F32(4.0) + rng.random(k - 4).astype(F32)          # 12 clear winners
        blk = rng.choice(np.setdiff1d(np.arange(E), top), 9, replace=False)
        lg[blk, t] = F32(1.25)                                         # 9 equal for the last 4 places
    lg[:, 8] = F32(0.5)
    lg[rng.choice(E, 20, replace=False), 8] = F32(3.0)                 # 20 equal at the top, k = 16 of them win
    lg[:, 9] = F32(0.7)                                                # all 896 equal: ids 0..15
    cases["ties"] = (lg, bs, k, 1.0)

    # bias_flip: the bias decides the set; unbiased top-k is a different set
    lg, _ = rand(16, 1.0)
    bs = np.zeros(E, dtype=F32)
    boost = rng.choice(E, 40, replace=False)
    bs[boost] = F32(2.0)
    cases["bias_flip"] = (lg, bs, k, 1.0)

    # tiny: all selected sigmoids ~1e-13..1e-9
    lg = rng.uniform(-30.0, -21.0, (E, 12)).astype(F32)
    cases["tiny"] = (lg, rng.normal(0, 1e-12, E).astype(F32), k, 1.0)

    # saturate: sigmoid == 1.0f exactly; bias orders, with bitwise ties in the bias
    lg = rng.uniform(18.0, 60.0, (E, 8)).astype(F32)
    bs = (rng.integers(0, 6, E).astype(F32) * F32(0.125))
    cases["saturate"] = (lg, bs, k, 1.0)

    # inf: +-inf logits mixed with finite ones
    lg = rng.normal(0, 2.0, (E, 6)).astype(F32)
    for t in range(6):
        lg[rng.choice(E, 5 + 2 * t, replace=False), t] = np.inf
        lg[rng.choice(E, 50, replace=False), t] = -np.inf
    cases["inf"] = (lg, rng.normal(0, 0.1, E).astype(F32), k, 1.0)

    cases["scaled"] = (*rand(8), k, 2.5)

    # eps_dom: every sigmoid ~1e-23..1e-21, so sum(p) < 1e-20 and the + 1e-20 decides each weight (dropping it, or
    # max(sum, eps), moves the weights by x1.2 .. x30); appended after the others so their draws are unchanged
    lg = rng.uniform(-52.0, -48.0, (E, 8)).astype(F32)
    cases["eps_dom"] = (lg, rng.normal(0, 1e-24, E).astype(F32), k, 1.0)

    # denormal: every sigmoid below 2^-126 and a zero bias: a flush-to-zero kernel routes by id with weight 0
    lg = rng.uniform(-103.9, -88.0, (E, 8)).astype(F32)
    cases["denormal"] = (lg, np.zeros(E, dtype=F32), k, 1.0)
    return cases


def situ_cases(seed: int = SEED) -> dict:
    """name -> (g f32, u f32)."""
    rng = np.random.default_rng(seed + 1)
    one = lambda n: np.ones(n, dtype=F32)                                       # noqa: E731
    g_sweep = np.concatenate([np.linspace(-100, 100, 8001), np.linspace(-12, 12, 4801),
                              np.geomspace(1e-30, 1e4, 600), -np.geomspace(1e-30, 1e4, 600)]).astype(F32)
    u_sweep = np.concatenate([np.linspace(-400, 400, 8001), np.linspace(-30, 30, 4801),
                              np.geomspace(1e-30, 1e4, 600), -np.geomspace(1e-30, 1e4, 600)]).astype(F32)
    mag = 10.0 ** rng.uniform(-30, 4, 20000)
    rg = (mag * rng.choice([-1, 1], 20000)).astype(F32)
    mag = 10.0 ** rng.uniform(-30, 4, 20000)
    ru = (mag * rng.choice([-1, 1], 20000)).astype(F32)
    edges = np.array([0.0, -0.0, 1e-45, -1e-45, 1e-39, -1e-39, 1.1754944e-38, 40.0, -40.0, 87.0, -87.0, 88.7, -88.7,
                      89.0, -89.0, 103.0, -103.0, 250.0, -250.0, 1e30, -1e30, 3.4e38, -3.4e38], dtype=F32)
    eg, eu = np.meshgrid(edges, edges, indexing="ij")
    inf = np.array([np.inf, -np.inf, np.inf, -np.inf, 1.0, 1.0, np.inf, -3.0], dtype=F32)
    infu = np.array([1.0, 1.0, -2.0, 5.0, np.inf, -np.inf, np.inf, -np.inf], dtype=F32)
    return {"gate_sweep": (g_sweep, one(g_sweep.size)), "up_sweep": (one(u_sweep.size), u_sweep),
            "random": (rg, ru), "edges": (eg.ravel(), eu.ravel()), "inf": (inf, infu)}


# --------------------------------------------------------------------------- building and writing

def build(seed: int = SEED) -> tuple[dict, dict]:
    """Return (arrays for the raw files, golden.json metadata without hashes)."""
    arrays: dict[str, np.ndarray] = {}
    meta = {"version": VERSION, "seed": seed, "source": "ref/kimi_k31.py moe_ffn routing and situ (float32 mode)",
            "router": {"formula": "probs=sigmoid(l); biased=f32(probs+bias); ids=top-k(biased) desc, ties to lower id; "
                                  "w=probs[ids]/(sum+1e-20)*scale", "renorm_eps": RENORM_EPS, "cases": {}},
            "situ": {"formula": "beta*tanh(g/beta)*sigmoid(g)*lbeta*tanh(u/lbeta)", "beta": SITU_BETA,
                     "linear_beta": SITU_LINEAR_BETA, "cases": {}}}
    worst_r, worst_s, near_total = 0.0, 0.0, 0
    for name, (lg, bs, k, scale) in route_cases(seed).items():
        r = route(lg, bs, k, scale)
        p = f"route.{name}."
        arrays.update({p + "logits": lg, p + "bias": bs, p + "probs": r["probs"], p + "biased": r["biased"],
                       p + "ids": r["ids"], p + "w": r["w"], p + "w_raw": r["w_raw"], p + "w_norm": r["w_norm"],
                       p + "margin": r["margin"].astype(F32)})
        gold = gold_route(lg, bs, k, scale, r)
        ids_e, w_e = emulate_route(lg, bs, k, scale)
        _, st = check_route(gold, ids_e, w_e, budget_ulp=np.inf)
        if name not in BUDGET_EXCLUDED:
            worst_r = max(worst_r, st["worst_ulp"])
        near_total += st["near_ties"]
        meta["router"]["cases"][name] = {"E": int(lg.shape[0]), "T": int(lg.shape[1]), "k": int(k), "scale": scale,
                                         "zero_margin_tokens": int(np.sum(r["margin"] == 0)),
                                         "bias_changed_set": int(sum(set(r["ids"][:, t]) != set(K.top_k_desc(r["probs"][:, t], k))
                                                                     for t in range(lg.shape[1]))),
                                         "emulated_f32": st}
    for name, (g, u) in situ_cases(seed).items():
        out = situ(g, u)
        ref64 = situ(g, u, dt=F64)
        hi = ref64.astype(F32)
        with np.errstate(invalid="ignore"):
            lo = np.where(np.isfinite(ref64), ref64 - hi.astype(F64), 0.0).astype(F32)
        p = f"situ.{name}."
        arrays.update({p + "g": g, p + "u": u, p + "out": out, p + "ref_hi": hi, p + "ref_lo": lo})
        _, st = check_situ({"ref64": ref64, "g": g}, emulate_situ(g, u), budget_ulp=np.inf)
        worst_s = max(worst_s, st["worst_ulp"])
        meta["situ"]["cases"][name] = {"n": int(g.size), "emulated_f32": st}
    meta["router"]["budget_ulp"] = budget_from(worst_r)
    meta["router"]["tie_rule"] = ("sets equal; or both k distinct ids and every id in the symmetric difference within "
                                  "4 float32 ULPs of the cut score, only when margin > 0 (bitwise ties must match)")
    meta["router"]["weights_rule"] = "matched by expert id, |w - w64| <= budget_ulp ULPs (unit floor 2^-126)"
    meta["router"]["emulated_worst_ulp"], meta["router"]["emulated_near_ties"] = worst_r, near_total
    meta["situ"]["budget_ulp"] = budget_from(worst_s)
    meta["situ"]["rule"] = (f"|out - (ref_hi+ref_lo)| <= budget_ulp ULPs of the reference (unit floor 2^-126) where "
                            f"g >= {SITU_UNDERFLOW_G}; below it |out - ref| <= {SITU_UNDERFLOW_ABS}; inf must match")
    meta["situ"]["underflow_g"], meta["situ"]["underflow_abs"] = SITU_UNDERFLOW_G, SITU_UNDERFLOW_ABS
    meta["situ"]["emulated_worst_ulp"] = worst_s
    return arrays, meta


# cases left out of the router budget's derivation: "denormal" is there to fail a flush-to-zero kernel (ids by id,
# weights 0: off by orders of magnitude), and a plain float32 kernel's denormal sigmoids lose precision there (5.6
# ULPs measured), which doubled would loosen the gate for every case from 9 to 12 ULPs; the case is still gated
BUDGET_EXCLUDED = ("denormal",)


def budget_from(measured: float) -> float:
    """Twice what a plain float32 kernel measured, at least 4 ULPs, as a whole number."""
    return float(max(4.0, np.ceil(2.0 * measured)))


def gold_route(lg, bs, k, scale, r=None) -> dict:
    """The router reference for check_route: golden ids, float64 weights, biased scores, margins."""
    r = r if r is not None else route(lg, bs, k, scale)
    w64 = np.empty(r["ids"].shape, dtype=F64)
    for t in range(r["ids"].shape[1]):
        x = r["probs"][r["ids"][:, t], t].astype(F64)                     # the oracle's float32 probs, as moe_ffn uses
        w64[:, t] = x / (x.sum() + RENORM_EPS) * scale
    return {"ids": r["ids"], "w64": w64, "biased": r["biased"], "margin": r["margin"]}


def write(out: pathlib.Path, seed: int = SEED) -> dict:
    arrays, meta = build(seed)
    K.write_raw_dumps(arrays, out)
    meta["files"] = {f"{n}.f32": hashlib.sha256((out / f"{n}.f32").read_bytes()).hexdigest() for n in arrays}
    meta["files"]["manifest.txt"] = hashlib.sha256((out / "manifest.txt").read_bytes()).hexdigest()
    (out / "golden.json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="write Kimi K3.1 router and SiTU golden vectors")
    ap.add_argument("--out", default="kimi-k31-golden", help="output directory")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--budgets", action="store_true", help="print the measured budgets and coverage")
    a = ap.parse_args()
    meta = write(pathlib.Path(a.out), a.seed)
    print(f"golden vectors -> {a.out}: {len(meta['files']) - 1} arrays")
    if a.budgets:
        r, s = meta["router"], meta["situ"]
        print(f"router: budget {r['budget_ulp']} ULP (float32 kernel measured {r['emulated_worst_ulp']:.2f}, "
              f"{r['emulated_near_ties']} near-tie tokens)")
        for n, c in r["cases"].items():
            print(f"  {n:9s} E {c['E']} T {c['T']:3d} k {c['k']} scale {c['scale']}  zero-margin {c['zero_margin_tokens']}"
                  f"  bias-changed {c['bias_changed_set']}  f32 {c['emulated_f32']['worst_ulp']:.2f} ULP")
        print(f"situ: budget {s['budget_ulp']} ULP (float32 kernel measured {s['emulated_worst_ulp']:.2f})")
        for n, c in s["cases"].items():
            print(f"  {n:10s} n {c['n']:6d}  f32 {c['emulated_f32']['worst_ulp']:.2f} ULP"
                  f"  (underflow abs {c['emulated_f32']['worst_abs_underflow']:.2e})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

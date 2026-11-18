#!/usr/bin/env python3
"""tools/kimi_k31_indexer_dump.py - the DSA lightning indexer seams of ref/kimi_k31.py, for k31_index's parity test
(tests/k31_index_test.cu --dumps DIR).  numpy only, no GPU, under a minute.

Two sources, both through ref/kimi_k31.py's own indexer():
  fix    the fixture forward (tools/kimi_k31_synth_gguf.py's TINY, or KIMI_K31_GGUF=...): every MLA layer, the
         fixture's top-k (6 of up to 32 keys), one prefill call
  long   the first MLA layer called directly with random layer inputs at the released top-k 2048: a 2200-token
         prefill, then a 40-token continuation that reads the carried key cache (query positions 2200..2239), so
         the selection discards keys.  Columns 600..899 of the prefill and 10..19 of the continuation repeat column
         599: 310 identical keys, so scores tie exactly and the lower-position rule decides at the cut.

Per call C (fix.L, long.0 and long.1) in the raw-dump layout:
    idx.C.q         [T][H][D]      indexer_q (attn_q_b @ qr, per head)
    idx.C.kpre      [T][D]         indexer.attn_k @ x (before the key LayerNorm)
    idx.C.w         [T][H]         indexer_weights (indexer.proj @ x, before the (H*D)^-0.5 prescale)
    idx.C.k         [T][D]         indexer_k (the LayerNorm'd keys this call adds to the cache)
    idx.C.score_hi  [T][n_kv]      indexer_score, float64 as hi + lo float32 (-inf: future keys, lo 0)
    idx.C.score_lo  [T][n_kv]
    idx.C.top       [T][k']        top_k: positions, descending score, ties to the lower position, -1 = none
per layer L: idx.ln.L.w / idx.ln.L.b [D] (indexer.k_norm), and index.json (H, D, eps, prescale, the calls).

    python tools/kimi_k31_indexer_dump.py --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ref"))
import kimi_k31_synth_gguf as synth  # noqa: E402
import kimi_k31 as K  # noqa: E402  (ref/kimi_k31.py)

F32, F64 = np.float32, np.float64
LONG_T, LONG_T2, LONG_TOPK = 2200, 40, 2048
TIE_SRC, TIE_LO, TIE_HI, TIE2_LO, TIE2_HI = 599, 600, 900, 10, 20


def one_call(m, il, x, qr, st, pos, tag, arrays, calls):
    """Run m.indexer on (x, qr) at positions pos, check the seams close, record them."""
    c = m.cfg
    m.dumps = {}
    sel = m.indexer(x, qr, il, st, np.asarray(pos))
    d = m.dumps
    iq, ik, iw = d[f"indexer_q-{il}"], d[f"indexer_k-{il}"], d[f"indexer_weights-{il}"]
    score, top = d[f"indexer_score-{il}"], d[f"top_k-{il}"]
    kpre = m.layer_t(il, "indexer.attn_k") @ x
    lw, lb = m.layer_t(il, "indexer.k_norm"), m.layer_t(il, "indexer.k_norm", "bias")
    if not np.array_equal(K.layer_norm(kpre, lw, lb, c.eps_indexer_ln, F32), ik):
        raise RuntimeError(f"{tag}: the key LayerNorm on kpre does not reproduce indexer_k")
    for t, s in enumerate(sel):
        if not np.array_equal(top[:len(s), t], s) or np.any(top[len(s):, t] != -1):
            raise RuntimeError(f"{tag}: top_k column {t} is not the selection")
    # cb() keeps seams in float32: the float64 score again, with indexer()'s own expressions on the same inputs
    ik_all = st.idx_k[il]
    H, D, n_kv = c.idx_n_head, c.idx_key_dim, ik_all.shape[1]
    w64 = iw.astype(F64) * (H * D) ** -0.5
    kq = np.einsum("dht,ds->sht", iq.astype(F64), ik_all.astype(F64))
    s64 = (np.maximum(kq, 0.0) * w64[None, :, :]).sum(axis=1)
    s64[np.arange(n_kv)[:, None] > np.asarray(pos)[None, :]] = -np.inf
    if not np.array_equal(s64.astype(F32), score):
        raise RuntimeError(f"{tag}: the float64 score does not round to the dumped indexer_score")
    for t, p_ in enumerate(pos):
        if not np.array_equal(K.top_k_desc(s64[:p_ + 1, t], c.idx_top_k), sel[t]):
            raise RuntimeError(f"{tag}: the float64 score's top-k is not the selection at query {t}")
    hi = s64.astype(F32)
    with np.errstate(invalid="ignore"):
        lo = np.where(np.isfinite(s64), s64 - hi.astype(F64), 0.0).astype(F32)
    p = f"idx.{tag}."
    arrays.update({p + "q": iq, p + "kpre": kpre, p + "w": iw, p + "k": ik, p + "score_hi": hi, p + "score_lo": lo,
                   p + "top": top})
    calls.append({"name": tag, "layer": il, "T": int(x.shape[1]), "p0": int(pos[0]), "n_kv": int(n_kv),
                  "k": int(top.shape[0]), "top_k": int(c.idx_top_k),
                  "over_top_k": int(sum(p_ + 1 > c.idx_top_k for p_ in pos))})
    return lw, lb


def dump(out: pathlib.Path, gguf: pathlib.Path | None = None) -> dict:
    arrays: dict[str, np.ndarray] = {}
    calls: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="strata-k31-idx-") as tmp:
        if gguf is None:
            gguf = pathlib.Path(tmp) / "tiny.gguf"
            vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
            md, ts = synth.to_gguf(vals, synth.TINY)
            synth.write_gguf(gguf, md, ts)
        # the fixture forward, every MLA layer
        m = K.KimiK31(gguf, cache_weights=True)
        try:
            c = m.cfg
            m.forward(synth.sample_tokens(synth.TINY, synth.SEED))
            fwd = dict(m.dumps)
            layers = c.mla_layers()
            for il in layers:
                x, qr = fwd[f"attn_norm-{il}"], fwd[f"qr-{il}"]
                st = m.new_state()
                lw, lb = one_call(m, il, x, qr, st, np.arange(x.shape[1]), f"fix.{il}", arrays, calls)
                if not np.array_equal(arrays[f"idx.fix.{il}.top"], fwd[f"top_k-{il}"]):
                    raise RuntimeError(f"layer {il}: the direct call does not reproduce the forward's top_k")
                arrays[f"idx.ln.{il}.w"], arrays[f"idx.ln.{il}.b"] = lw, lb
            H, D, eps, n_embd, q_lora = c.idx_n_head, c.idx_key_dim, c.eps_indexer_ln, c.n_embd, c.q_lora_rank
        finally:
            m.close()
        # the long call at the released top-k
        m = K.KimiK31(gguf, cache_weights=True, index_top_k=LONG_TOPK)
        try:
            rng = np.random.default_rng(synth.SEED + 77)
            il = layers[0]
            x = rng.normal(0, 1, (n_embd, LONG_T)).astype(F32)
            qr = rng.normal(0, 1, (q_lora, LONG_T)).astype(F32)
            x[:, TIE_LO:TIE_HI] = x[:, TIE_SRC:TIE_SRC + 1]
            st = m.new_state()
            one_call(m, il, x, qr, st, np.arange(LONG_T), "long.0", arrays, calls)
            x2 = rng.normal(0, 1, (n_embd, LONG_T2)).astype(F32)
            q2 = rng.normal(0, 1, (q_lora, LONG_T2)).astype(F32)
            x2[:, TIE2_LO:TIE2_HI] = x[:, TIE_SRC:TIE_SRC + 1]
            one_call(m, il, x2, q2, st, np.arange(LONG_T, LONG_T + LONG_T2), "long.1", arrays, calls)
        finally:
            m.close()
    meta = {"layers": layers, "H": H, "D": D, "eps": eps, "prescale": (H * D) ** -0.5, "calls": calls,
            "ties": {"source": TIE_SRC, "prefill": [TIE_LO, TIE_HI], "continuation": [TIE2_LO, TIE2_HI]},
            "source": "ref/kimi_k31.py indexer() (float32 mode): the fixture forward's MLA layers at its top-k, and a "
                      "long random call (2200 + 40 tokens) at top-k 2048"}
    K.write_raw_dumps(arrays, out)
    (out / "index.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="dump K3.1 indexer parity data from ref/kimi_k31.py")
    ap.add_argument("--out", default="kimi-k31-index", help="output directory")
    a = ap.parse_args()
    env = os.environ.get("KIMI_K31_GGUF")
    meta = dump(pathlib.Path(a.out), pathlib.Path(env) if env else None)
    print(f"indexer dumps -> {a.out}: {meta['H']} heads x {meta['D']}; calls " +
          ", ".join(f"{c['name']} (T {c['T']}, p0 {c['p0']}, n_kv {c['n_kv']}, top {c['top_k']}, {c['over_top_k']} over)"
                    for c in meta["calls"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

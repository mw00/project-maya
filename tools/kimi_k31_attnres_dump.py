#!/usr/bin/env python3
"""tools/kimi_k31_attnres_dump.py - Block Attention Residuals seams of ref/kimi_k31.py, for k31_attn_res's parity test
(tests/k31_attn_res_test.cu --dumps DIR).  numpy only, no GPU, a few seconds.

Runs the oracle on the synthetic fixture (tools/kimi_k31_synth_gguf.py, or KIMI_K31_GGUF=...) and writes, in
ref/kimi_k31.py's raw-dump layout (write_raw_dumps: manifest.txt `name d0 d1 ...`, ggml order):

    ar.inp_embd           [T][n_embd]        the embeddings (block 0)
    ar.L.prefix_attn      [T][n_embd]        prefix_sum_attn-L: the partial sum after layer L's attention
    ar.L.l_out            [T][n_embd]        l_out-L: the partial sum after layer L's FFN
    ar.L.score_attn       [n_embd]           attn_res_score-L (the fused res_norm * res_proj vector)
    ar.L.score_ffn        [n_embd]           ffn_res_score-L
    ar.L.mix_attn / probs_attn               res_mix_attn-L [T][n_embd] / res_probs_attn-L [T][n_cand]  (L >= 1)
    ar.L.mix_ffn  / probs_ffn                res_mix_ffn-L / res_probs_ffn-L
    ar.score_out, ar.mix_out, ar.probs_out   output_res_score, the final mix and its probabilities

plus attnres.json (n_layers, block_size, eps, T, n_embd, and per site the candidate count).  The candidates of every
site are fully determined by these states and the block rule (tests/k31_attn_res_test.cu rebuilds them):
    attn site of layer L >= 1 : blocks pushed at j < L, then the partial l_out-(L-1)
    ffn  site of layer L      : blocks pushed at j <= L, then prefix_sum_attn-L
    out  site                 : every block, then l_out-(last)
    block pushed at j (j % block_size == 0): inp_embd for j = 0, else l_out-(j-1)
This script checks that rule against the candidates the oracle actually mixed (it records res_mix's inputs).

    python tools/kimi_k31_attnres_dump.py --out DIR
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

F32 = np.float32


def candidates(states: dict, block: int, site: str, il: int, n_layers: int) -> list[np.ndarray]:
    """The block rule (see the module docstring)."""
    def pushed(j):
        return states["inp_embd"] if j == 0 else states[f"l_out-{j - 1}"]
    if site == "attn":
        return [pushed(j) for j in range(0, il, block)] + [states[f"l_out-{il - 1}"]]
    if site == "ffn":
        return [pushed(j) for j in range(0, il + 1, block)] + [states[f"prefix_sum_attn-{il}"]]
    return [pushed(j) for j in range(0, n_layers, block)] + [states[f"l_out-{n_layers - 1}"]]


def dump(out: pathlib.Path, gguf: pathlib.Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="strata-k31-ar-") as tmp:
        if gguf is None:
            gguf = pathlib.Path(tmp) / "tiny.gguf"
            vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
            md, ts = synth.to_gguf(vals, synth.TINY)
            synth.write_gguf(gguf, md, ts)
        m = K.KimiK31(gguf, cache_weights=True)
        seen = []
        orig = m.res_mix

        def rec(blocks, cur, score_w, il, site):
            seen.append((site, il, [b.copy() for b in blocks] + [cur.copy()]))
            return orig(blocks, cur, score_w, il, site)

        m.res_mix = rec
        try:
            c = m.cfg
            m.forward(synth.sample_tokens(synth.TINY, synth.SEED))
            d = dict(m.dumps)
            B, L = c.attn_res_block_size, c.n_layers
            # the block rule must give exactly the candidates the oracle mixed
            for site, il, cands in seen:
                if site == "attn" and il == 0:
                    continue                                  # no blocks yet: res_mix returns its input unmixed
                want = candidates(d, B, site, il, L)
                if len(want) != len(cands) or any(not np.array_equal(a, b) for a, b in zip(want, cands)):
                    raise RuntimeError(f"block rule disagrees with the oracle at {site} {il}")
            arrays = {"ar.inp_embd": d["inp_embd"], "ar.score_out": m.t("output_res_score.weight"),
                      "ar.mix_out": d["res_mix"], "ar.probs_out": d["res_probs"]}
            sites = {}
            for il in range(L):
                p = f"ar.{il}."
                arrays.update({p + "prefix_attn": d[f"prefix_sum_attn-{il}"], p + "l_out": d[f"l_out-{il}"],
                               p + "score_attn": m.layer_t(il, "attn_res_score"), p + "score_ffn": m.layer_t(il, "ffn_res_score"),
                               p + "mix_ffn": d[f"res_mix_ffn-{il}"], p + "probs_ffn": d[f"res_probs_ffn-{il}"]})
                sites[f"ffn.{il}"] = int(d[f"res_probs_ffn-{il}"].shape[0])
                if il >= 1:
                    arrays.update({p + "mix_attn": d[f"res_mix_attn-{il}"], p + "probs_attn": d[f"res_probs_attn-{il}"]})
                    sites[f"attn.{il}"] = int(d[f"res_probs_attn-{il}"].shape[0])
            sites["out"] = int(d["res_probs"].shape[0])
            x = d["inp_embd"]
            meta = {"n_layers": L, "block_size": B, "eps": float(c.eps_rms), "T": int(x.shape[1]), "n_embd": int(x.shape[0]),
                    "sites": sites, "source": "ref/kimi_k31.py on the synthetic fixture (float32 mode)"}
        finally:
            m.close()
    K.write_raw_dumps(arrays, out)
    (out / "attnres.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="dump K3.1 Block Attention Residuals parity data from ref/kimi_k31.py")
    ap.add_argument("--out", default="kimi-k31-attnres", help="output directory")
    a = ap.parse_args()
    env = os.environ.get("KIMI_K31_GGUF")
    meta = dump(pathlib.Path(a.out), pathlib.Path(env) if env else None)
    print(f"attn-res dumps -> {a.out}: {meta['n_layers']} layers, block {meta['block_size']}, T {meta['T']}, "
          f"n_embd {meta['n_embd']}, {len(meta['sites'])} sites (max {max(meta['sites'].values())} candidates); "
          f"the block rule reproduces every site's candidates")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""tools/kimi_k31_moe_dump.py - per-pick expert outputs and the latent-combine seams of ref/kimi_k31.py, for
k31_moe_combine's parity test (tests/k31_moe_combine_test.cu --dumps DIR).  numpy only, no GPU, a few seconds.

ref/kimi_k31.py dumps the routed sum (ffn_moe_out) but not the single experts' outputs.  This script runs the oracle
on the synthetic fixture (tools/kimi_k31_synth_gguf.py, or KIMI_K31_GGUF=...), recomputes every pick's
down_e(situ(gate_e z, up_e z)) exactly as moe_ffn does (float32 GEMVs, float32 SiTU), checks that the oracle's own
float64 weighted sum of those parts reproduces ffn_moe_out bit for bit, and writes, per MoE layer L:

    moe.L.parts      [T][k][latent]   each pick's down output (float32), in rank order
    moe.L.w          [T][k]           the routing weights (ffn_moe_weights_scaled, float32)
    moe.L.norm_w     [latent]         ffn_routed_norm
    moe.L.up_w       [latent][n_embd] ffn_routed_up (out = n_embd fastest: a column-major (n_embd x latent) matrix)
    moe.L.shexp      [T][n_embd]      the shared experts' output (ffn_shexp)
    moe.L.moe_out    [T][latent]      oracle: the routed sum (ffn_moe_out)
    moe.L.normed     [T][latent]      oracle: after routed_norm (ffn_routed_normed)
    moe.L.routed_out [T][n_embd]      oracle: after routed_up (ffn_routed_out)
    moe.L.ffn_out    [T][n_embd]      oracle: routed + shared (ffn_out)

in ref/kimi_k31.py's raw-dump layout (write_raw_dumps: manifest.txt `name d0 d1 ...`, ggml order), plus moe.json
(layers, T, k, latent, n_embd, eps).

    python tools/kimi_k31_moe_dump.py --out DIR
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


def dump(out: pathlib.Path, gguf: pathlib.Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="strata-k31-moe-") as tmp:
        if gguf is None:
            gguf = pathlib.Path(tmp) / "tiny.gguf"
            vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
            md, ts = synth.to_gguf(vals, synth.TINY)
            synth.write_gguf(gguf, md, ts)
        m = K.KimiK31(gguf, cache_weights=True)
        try:
            c = m.cfg
            m.forward(synth.sample_tokens(synth.TINY, synth.SEED))
            d = dict(m.dumps)
            arrays: dict[str, np.ndarray] = {}
            layers = list(range(c.n_layer_dense_lead, c.n_layers))
            for il in layers:
                z = d[f"ffn_routed_in-{il}"]                       # (latent, T)
                ids = d[f"ffn_moe_topk-{il}"]                      # (k, T)
                w32 = d[f"ffn_moe_weights_scaled-{il}"]            # (k, T)
                k, T = ids.shape
                lat = z.shape[0]
                parts = np.empty((lat, k, T), dtype=F32)
                for t in range(T):
                    zt = z[:, t]
                    for i in range(k):
                        e = int(ids[i, t])
                        h = K.situ(m.expert_w(il, "ffn_gate_exps", e) @ zt, m.expert_w(il, "ffn_up_exps", e) @ zt,
                                   c.situ_beta, c.situ_linear_beta, F32)
                        parts[:, i, t] = m.expert_w(il, "ffn_down_exps", e) @ h
                # the oracle's own float64 sum of these parts (with its unrounded float64 weights) must be ffn_moe_out
                probs = d[f"ffn_moe_probs-{il}"]
                re = np.empty((lat, T), dtype=F32)
                for t in range(T):
                    x = probs[ids[:, t], t].astype(F64)
                    wt = x / (x.sum() + c.eps_router_renorm) * c.expert_weights_scale if c.expert_weights_norm else x * c.expert_weights_scale
                    re[:, t] = _acc(parts[:, :, t], wt)
                if not np.array_equal(re, d[f"ffn_moe_out-{il}"]):
                    raise RuntimeError(f"layer {il}: the recomputed parts do not reproduce ffn_moe_out")
                up = m.layer_t(il, "ffn_routed_up")                # (n_embd, latent) = (out, in)
                p = f"moe.{il}."
                arrays.update({p + "parts": parts, p + "w": w32, p + "norm_w": m.layer_t(il, "ffn_routed_norm"),
                               p + "up_w": up, p + "shexp": d[f"ffn_shexp-{il}"], p + "moe_out": d[f"ffn_moe_out-{il}"],
                               p + "normed": d[f"ffn_routed_normed-{il}"], p + "routed_out": d[f"ffn_routed_out-{il}"],
                               p + "ffn_out": d[f"ffn_out-{il}"]})
            meta = {"layers": layers, "T": int(T), "k": int(k), "latent": int(lat), "n_embd": int(up.shape[0]),
                    "eps": float(c.eps_rms), "source": "ref/kimi_k31.py moe_ffn on the synthetic fixture (float32 mode)"}
        finally:
            m.close()
    K.write_raw_dumps(arrays, out)
    (out / "moe.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def _acc(parts_t: np.ndarray, wt: np.ndarray) -> np.ndarray:
    """moe_ffn's accumulation: acc += w[i] * part_i in float64, pick by pick, then one rounding."""
    acc = np.zeros(parts_t.shape[0], dtype=F64)
    for i in range(parts_t.shape[1]):
        acc += wt[i] * parts_t[:, i].astype(F64)
    return acc.astype(F32)


def main() -> int:
    ap = argparse.ArgumentParser(description="dump K3.1 latent-combine parity data from ref/kimi_k31.py")
    ap.add_argument("--out", default="kimi-k31-moe", help="output directory")
    a = ap.parse_args()
    env = os.environ.get("KIMI_K31_GGUF")
    meta = dump(pathlib.Path(a.out), pathlib.Path(env) if env else None)
    print(f"moe combine dumps -> {a.out}: layers {meta['layers']}, T {meta['T']}, k {meta['k']}, latent {meta['latent']}, "
          f"n_embd {meta['n_embd']}; parts reproduce ffn_moe_out bit for bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())

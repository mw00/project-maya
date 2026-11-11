#!/usr/bin/env python3
"""tools/kimi_k31_mla_dump.py - the MLA attention seams of ref/kimi_k31.py, for k31_mla's parity test
(tests/k31_mla_test.cu --dumps DIR).  numpy only, no GPU, a few seconds.

Runs the oracle on the synthetic fixture (tools/kimi_k31_synth_gguf.py, or KIMI_K31_GGUF=...) and, for every MLA
layer L, rebuilds the attention's inputs from its own seams (qr, q_absorbed, kv_cmpr, k_pe, top_k, attn_norm), checks
that mla_layer's formula on them reproduces kqv_out bit for bit, and writes (raw-dump layout, file order shown):

    mla.L.q        [T][H][576]       the absorbed query (512) then the un-rotated query part (64), per head
    mla.L.kv       [n_kv][576]       the cache row: kv_cmpr (512, the values) then k_pe (64)
    mla.L.sel      [T][k]            the indexer's picks (top_k, -1 = none), descending score
    mla.L.ctx      [T][H][512]       oracle: the softmax-weighted latent, float64 rounded to float32
    mla.L.wv_b     [H][v_head][512]  attn_v_b
    mla.L.kqv_out  [T][H*v_head]     oracle: per-head value projection, row e + v_head*h
    mla.L.gate_pre [T][H*v_head]     attn_gate @ x (the gate's pre-activation)
    mla.L.gated    [T][H*v_head]     oracle: kqv_out * sigmoid(gate_pre)
    mla.L.wo       [H*v_head][n_embd] attn_output, out = n_embd fastest (a column-major n_embd x H*v_head matrix)
    mla.L.out      [T][n_embd]       oracle: mla_out

plus mla.json (layers, H, T, n_kv, kv_lora, rope, v_head, scale, top_k, n_embd).

    python tools/kimi_k31_mla_dump.py --out DIR
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
from glm import split_heads  # noqa: E402  (ref/glm.py)

F32, F64 = np.float32, np.float64


def dump(out: pathlib.Path, gguf: pathlib.Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="strata-k31-mla-") as tmp:
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
            H, R, nope, vh = c.n_head, c.n_rot, c.qk_nope, c.n_head_v_mla
            scale = c.n_head_k_mla ** -0.5
            arrays: dict[str, np.ndarray] = {}
            layers = c.mla_layers()
            for il in layers:
                x = d[f"attn_norm-{il}"]
                q = split_heads(m.layer_t(il, "attn_q_b") @ d[f"qr-{il}"], c.n_head_k_mla, H)
                q_pe = q[nope:]                                                    # (64, H, T)
                q_abs = d[f"q_absorbed-{il}"]                                      # (512, H, T)
                kv_cmpr, k_pe = d[f"kv_cmpr-{il}"], d[f"k_pe-{il}"]                # (512, n_kv), (64, n_kv)
                top = d[f"top_k-{il}"]                                             # (k, T), -1 = none
                T, n_kv = x.shape[1], kv_cmpr.shape[1]
                wv_b = m.layer_t(il, "attn_v_b").astype(F64)                       # (H, v_head, kv_lora)
                ctx32 = np.empty((c.kv_lora_rank, H, T), dtype=F32)
                kqv = np.empty((vh * H, T), dtype=F32)
                for t in range(T):
                    cells = np.sort(top[top[:, t] >= 0, t])
                    L, P = kv_cmpr[:, cells].astype(F64), k_pe[:, cells].astype(F64)
                    s = (q_abs[:, :, t].T.astype(F64) @ L + q_pe[:, :, t].T.astype(F64) @ P) * scale
                    p = K.softmax_cols(s, axis=1)
                    ctx = L @ p.T                                                  # (kv_lora, H)
                    ctx32[:, :, t] = ctx
                    kqv[:, t] = np.einsum("hvc,ch->hv", wv_b, ctx).reshape(-1)
                if not np.array_equal(kqv, d[f"kqv_out-{il}"]):
                    raise RuntimeError(f"layer {il}: rebuilt inputs do not reproduce kqv_out")
                gate_pre = m.layer_t(il, "attn_gate") @ x
                p = f"mla.{il}."
                arrays.update({p + "q": np.concatenate([q_abs, q_pe], axis=0), p + "kv": np.concatenate([kv_cmpr, k_pe], axis=0),
                               p + "sel": top, p + "ctx": ctx32, p + "wv_b": np.ascontiguousarray(m.layer_t(il, "attn_v_b").transpose(2, 1, 0)),
                               p + "kqv_out": d[f"kqv_out-{il}"], p + "gate_pre": gate_pre, p + "gated": d[f"mla_gated-{il}"],
                               p + "wo": m.layer_t(il, "attn_output"), p + "out": d[f"mla_out-{il}"]})
            meta = {"layers": layers, "H": H, "T": int(T), "n_kv": int(n_kv), "kv_lora": c.kv_lora_rank, "rope": R, "v_head": vh,
                    "scale": scale, "top_k": int(top.shape[0]), "n_embd": int(x.shape[0]),
                    "source": "ref/kimi_k31.py mla_layer on the synthetic fixture (float32 mode), one prefill call"}
        finally:
            m.close()
    K.write_raw_dumps(arrays, out)
    (out / "mla.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="dump K3.1 MLA attention parity data from ref/kimi_k31.py")
    ap.add_argument("--out", default="kimi-k31-mla", help="output directory")
    a = ap.parse_args()
    env = os.environ.get("KIMI_K31_GGUF")
    meta = dump(pathlib.Path(a.out), pathlib.Path(env) if env else None)
    print(f"mla dumps -> {a.out}: layers {meta['layers']}, H {meta['H']}, T {meta['T']}, n_kv {meta['n_kv']}, top_k {meta['top_k']}, "
          f"row {meta['kv_lora'] + meta['rope']}; the rebuilt inputs reproduce kqv_out bit for bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())

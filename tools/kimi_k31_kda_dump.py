#!/usr/bin/env python3
"""tools/kimi_k31_kda_dump.py - the KDA layer seams of ref/kimi_k31.py, for k31_kda's parity test
(tests/k31_kda_test.cu --dumps DIR).  numpy only, no GPU, a few seconds.

Runs the oracle on the synthetic fixture (tools/kimi_k31_synth_gguf.py, or KIMI_K31_GGUF=...) and, for every KDA
layer L, recomputes the GEMV outputs the engine hands the KDA kernels (they are plain projections of the layer input
attn_norm-L), checks that ref/kimi_k31.py's own functions on them reproduce the oracle's KDA seams bit for bit, and
writes (raw-dump layout, file order shown; one prefill call, zero state):

    kda.L.xq / xk / xv     [T][d_inner]          attn_q / attn_k / attn_v @ x   (the conv inputs)
    kda.L.conv_q / _k / _v [d_inner][d_conv]     ssm_conv1d_{q,k,v}, taps oldest first (glm_kda_conv's layout)
    kda.L.f_pre            [T][d_inner]          ssm_f_b @ (ssm_f_a @ x)         (the decay gate, before dt_bias)
    kda.L.dt_bias          [d_inner]
    kda.L.ssm_a            [n_head]              -exp(A_log), sliced to the head count by the converter
    kda.L.beta_pre         [T][n_head]           ssm_beta @ x
    kda.L.g2               [T][d_inner]          ssm_g @ x                        (the full-rank output gate)
    kda.L.norm_w           [head_dim]            ssm_norm
    kda.L.wo               [d_inner][n_embd]     attn_output (out = n_embd fastest: column-major n_embd x d_inner)
  oracle seams:
    kda.L.q_conv / k_conv / v_conv [T][d_inner]  kda_{q,k,v}_conv  (silu(conv))
    kda.L.g1               [T][d_inner]          kda_g1   (lower_bound * sigmoid(exp(A_log) * (f_pre + dt_bias)))
    kda.L.beta             [T][n_head]           kda_beta
    kda.L.q_l2 / k_l2      [T][d_inner]          the per-head l2-normalised q, k (the eps is kda.l2_norm_epsilon)
    kda.L.scan_out         [T][d_inner]          kda_scan_out (includes the 1/sqrt(head_dim) scale)
    kda.L.gated            [T][d_inner]          rms_norm(scan_out, ssm_norm) * sigmoid(g2) (attn_output's input)
    kda.L.out              [T][n_embd]           kda_out
    kda.L.state            [n_head][dk][dv]      the recurrent state after the call (float64 rounded once), S[h] row-major,
                                                 row = KEY axis: what the next call starts from

plus kda.json (layers, n_head, head_dim, d_conv, T, n_embd, lower_bound, eps_l2, eps_rms).

    python tools/kimi_k31_kda_dump.py --out DIR
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
from glm import join_heads, split_heads  # noqa: E402  (ref/glm.py)

F32, F64 = np.float32, np.float64


def dump(out: pathlib.Path, gguf: pathlib.Path | None = None) -> dict:
    with tempfile.TemporaryDirectory(prefix="strata-k31-kda-") as tmp:
        if gguf is None:
            gguf = pathlib.Path(tmp) / "tiny.gguf"
            vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
            md, ts = synth.to_gguf(vals, synth.TINY)
            synth.write_gguf(gguf, md, ts)
        m = K.KimiK31(gguf, cache_weights=True)
        try:
            c = m.cfg
            st = m.new_state()
            m.forward(synth.sample_tokens(synth.TINY, synth.SEED), st)
            d = dict(m.dumps)
            H, dk = c.n_head, c.kda_head_dim
            arrays: dict[str, np.ndarray] = {}
            layers = [il for il in range(c.n_layers) if c.is_recr(il)]
            for il in layers:
                x = d[f"attn_norm-{il}"]
                T = x.shape[1]
                p = f"kda.{il}."
                convs = []
                for s in "qkv":
                    xp = m.layer_t(il, f"attn_{s}") @ x
                    wraw = m.layer_t(il, f"ssm_conv1d_{s}").reshape(c.d_inner, c.ssm_d_conv)
                    y, _ = K.short_conv(xp, wraw.T, np.zeros((c.d_inner, c.ssm_d_conv - 1), dtype=F32), F32)
                    if not np.array_equal(y, d[f"kda_{s}_conv-{il}"]):
                        raise RuntimeError(f"layer {il}: conv {s} does not reproduce kda_{s}_conv")
                    convs.append(y)
                    arrays.update({p + "x" + s: xp, p + "conv_" + s: wraw.T, p + s + "_conv": y})
                f_pre = m.layer_t(il, "ssm_f_b") @ (m.layer_t(il, "ssm_f_a") @ x)
                dt_bias = m.layer_t(il, "ssm_dt", "bias")
                ssm_a = m.t(f"blk.{il}.ssm_a")
                pre = f_pre + dt_bias[:, None]
                arg = (split_heads(pre, dk, H).astype(F64) * (-ssm_a.astype(F64))[None, :, None]).astype(F32)
                g1 = (c.kda_gate_lower_bound * K.sigmoid(arg, F32).astype(F64)).astype(F32)
                if not np.array_equal(g1, d[f"kda_g1-{il}"]):
                    raise RuntimeError(f"layer {il}: gate does not reproduce kda_g1")
                beta_pre = m.layer_t(il, "ssm_beta") @ x
                q = K.l2_norm(split_heads(convs[0], dk, H), c.eps_kda_l2, F32)
                kk = K.l2_norm(split_heads(convs[1], dk, H), c.eps_kda_l2, F32)
                g2 = m.layer_t(il, "ssm_g") @ x
                normed = d[f"kda_normed-{il}"]
                gated = normed * K.sigmoid(split_heads(g2, dk, H), F32)
                arrays.update({p + "f_pre": f_pre, p + "dt_bias": dt_bias, p + "ssm_a": ssm_a, p + "beta_pre": beta_pre,
                               p + "g2": g2, p + "norm_w": m.layer_t(il, "ssm_norm"), p + "wo": m.layer_t(il, "attn_output"),
                               p + "g1": join_heads(d[f"kda_g1-{il}"]), p + "beta": d[f"kda_beta-{il}"],
                               p + "q_l2": join_heads(q), p + "k_l2": join_heads(kk),
                               p + "scan_out": join_heads(d[f"kda_scan_out-{il}"]), p + "gated": join_heads(gated),
                               p + "out": d[f"kda_out-{il}"],
                               # (H, dk, dv) C order is the engine's layout; write_raw_dumps reverses the axes
                               p + "state": np.ascontiguousarray(st.ssm[il].astype(F32).transpose(2, 1, 0))})
                # the carried conv history is the last d_conv-1 conv inputs, oldest first
                for j, s in enumerate("qkv"):
                    if not np.array_equal(st.conv[il][j], arrays[p + "x" + s][:, -(c.ssm_d_conv - 1):]):
                        raise RuntimeError(f"layer {il}: conv history {s} is not the last inputs")
                # the oracle's out is attn_output @ join_heads(gated): check the seam chain closes
                if not np.array_equal(m.layer_t(il, "attn_output") @ join_heads(gated), d[f"kda_out-{il}"]):
                    raise RuntimeError(f"layer {il}: gated does not reproduce kda_out")
            meta = {"layers": layers, "n_head": H, "head_dim": dk, "d_conv": c.ssm_d_conv, "T": int(T), "n_embd": int(x.shape[0]),
                    "lower_bound": c.kda_gate_lower_bound, "eps_l2": c.eps_kda_l2, "eps_rms": float(c.eps_rms),
                    "source": "ref/kimi_k31.py kda_layer on the synthetic fixture (float32 mode), one prefill call, zero state"}
        finally:
            m.close()
    K.write_raw_dumps(arrays, out)
    (out / "kda.json").write_text(json.dumps(meta, indent=1) + "\n")
    return meta


def main() -> int:
    ap = argparse.ArgumentParser(description="dump K3.1 KDA layer parity data from ref/kimi_k31.py")
    ap.add_argument("--out", default="kimi-k31-kda", help="output directory")
    a = ap.parse_args()
    env = os.environ.get("KIMI_K31_GGUF")
    meta = dump(pathlib.Path(a.out), pathlib.Path(env) if env else None)
    print(f"kda dumps -> {a.out}: layers {meta['layers']}, {meta['n_head']} x {meta['head_dim']}, d_conv {meta['d_conv']}, "
          f"T {meta['T']}; the rebuilt inputs reproduce the conv, gate and output seams bit for bit")
    return 0


if __name__ == "__main__":
    sys.exit(main())

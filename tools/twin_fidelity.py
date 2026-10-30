#!/usr/bin/env python3
"""tools/twin_fidelity.py - how far the engine's load-time Q8_0 copies sit from the BF16 rows they replace.

For every keep16 (kind-4) row of the pack that the fast path reads only through q() (the rows that get the Q8_0
copy), the script computes the twin with the engine's recipe (ggml quantize_row_q8_0_ref: per 32-block d = amax/127
in F32, stored as FP16 bits the kernel decodes with __half2float; q = round(x/d)) from the pack's own dense.bin,
dequantizes it the way the kernel does, and measures the error against the BF16 bits.  Rows the b16()/q_or_b16()
lane reads keep their exact BF16 copy - listed only for size.  CPU only; reads dense.bin and index.txt.
"""
import sys

import numpy as np

Q_ONLY = {"attn_q.weight", "attn_k.weight", "attn_v.weight", "attn_q_b.weight", "attn_output.weight",
          "output.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
          "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"}


def cls(name):
    n = name.split(".", 2)[2] if name.startswith("blk.") else name
    if n == "output.weight":
        return "output head"
    if n in ("attn_q_a.weight", "attn_kv_a_mqa.weight"):
        return "mla q_a/kv_a (q_or_b16, exact)"
    if n.endswith((".weight",)) and n in Q_ONLY:
        if "shexp" in n:
            return "shared experts"
        if n.startswith("ffn_"):
            return "dense ffn"
        return "attention/dense proj"
    if "ffn_gate_inp" in n:
        return "router (b16, exact)"
    if "index" in n:
        return "indexer (b16, exact)"
    if "ssm_" in n or "hc_" in n:
        return "ssm/hc (b16, exact)"
    if "attn_k_b" in n or "attn_v_b" in n:
        return "mla k_b/v_b (b16, exact)"
    return "other"


def fp16_bits(d):
    """float32 -> IEEE binary16 bits, round-to-nearest-even (what __float2half_rn produces)."""
    return np.ascontiguousarray(d, dtype=np.float16).view(np.uint16)


def twin_q8(bf16_bits):
    """The engine's Q8_0 twin: (nblocks, 34) uint8 -> dequantized weights, the way the kernel reads them."""
    x = (bf16_bits.astype(np.uint32) << 16).view(np.float32)          # exact BF16 -> F32
    nb = (x.size + 31) // 32
    xb = np.zeros(nb * 32, dtype=np.float32)
    xb[: x.size] = x
    xb = xb.reshape(nb, 32)
    amax = np.abs(xb).max(axis=1)
    d = amax / 127.0
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0)
    ax = xb * inv[:, None]
    q = (np.floor(np.abs(ax) + 0.5) * np.sign(ax)).astype(np.int64)    # std::round: half away from zero
    q = np.clip(q, -127, 127).astype(np.int8)
    dh = fp16_bits(d)                                                   # the FP16 field the kernel decodes
    d_served = dh.view(np.float16).astype(np.float32)                   # __half2float(dh)
    return (q.astype(np.float32) * d_served[:, None]).ravel()[: x.size], x


def main(pack):
    rows = []
    for line in open(pack + "/index.txt"):
        if line.startswith("#") or not line.strip():
            continue
        p = line.split()
        if p[2] == "4":
            rows.append((p[0], int(p[3]), int(p[4])))
    dense = np.memmap(pack + "/dense.bin", dtype=np.uint8, mode="r")

    agg = {}
    for name, off, nbytes in rows:
        bits = dense[off: off + nbytes].view("<u2")
        served, src = twin_q8(bits)
        c = cls(name)
        err = served - src
        denom = np.linalg.norm(src.astype(np.float64))
        rel = np.linalg.norm(err.astype(np.float64)) / denom if denom else 0.0
        a = agg.setdefault(c, {"n": 0, "bytes16": 0, "bytes8": 0, "rel": [], "maxabs": 0.0})
        exact = c.endswith(", exact)")
        a["n"] += 1
        a["bytes16"] += nbytes
        a["bytes8"] += (bits.size // 32) * 34
        if exact:
            a["rel"] = [0.0]
            continue
        a["rel"].append(rel)
        a["maxabs"] = max(a["maxabs"], float(np.abs(err).max()) if err.size else 0.0)

    print("%-24s %6s %10s %10s %9s %9s %9s" % ("class", "rows", "BF16 MB", "Q8 MB", "rel-rms", "worst row", "max|err|"))
    tot16 = tot8 = 0
    for c in sorted(agg):
        a = agg[c]
        exact = c.endswith(", exact)")
        rel = max(a["rel"]) if a["rel"] else 0.0
        print("%-24s %6d %10.1f %10.1f %9s %9.5f %9.4f" %
              (c, a["n"], a["bytes16"] / 1e6, a["bytes8"] / 1e6,
               "exact" if exact else "%.5f" % (sum(a['rel']) / len(a['rel'])),
               0.0 if exact else rel, a["maxabs"]))
        tot16 += a["bytes16"]
        tot8 += a["bytes8"]
    print("\nall keep16 rows: BF16 %.2f GB vs served copies %.2f GB (the q-only rows: %.2f GB BF16 -> %.2f GB Q8_0)"
          % (tot16 / 1e9, tot8 / 1e9,
             sum(a["bytes16"] for c, a in agg.items() if not c.endswith(", exact)")) / 1e9,
             sum(a["bytes8"] for c, a in agg.items() if not c.endswith(", exact)")) / 1e9))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")

"""tools/glm_mtp_fixture.py - the synthetic glm5-next GGUF (tools/glm_synth_gguf.py) with a NextN block, for maya-metal's
MTP tests: the trunk is the small or big fixture unchanged, and one more block (index n_layers) carries the NextN tensors
GLM-5.3-Flash's MTP drafter has - eh_proj, enorm, hnorm, shared_head_norm, a nope-MLA + DSA attention layer and a MoE.
The block's weights are random: drafting with it must still decode exactly as plain decoding does, which is what the
test checks (the drafts only decide how many tokens a verify window keeps).

    python tools/glm_mtp_fixture.py --out FILE [--big] [--no-nextn] [--q8-experts]

--q8-experts stores the routed experts as Q8_0 (the others stay F32): a prompt's slices then take ggml-metal's
quantized matrix-matrix expert kernel, which F32 experts never reach.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_synth_gguf as S  # noqa: E402


def q8_0(a: np.ndarray) -> bytes:
    """ggml's Q8_0: blocks of 32 along the last axis, an fp16 scale (max |x| / 127) then 32 int8."""
    x = np.ascontiguousarray(a, dtype=np.float32).reshape(-1, 32)
    d = np.abs(x).max(axis=1) / 127.0
    q = np.round(x / np.where(d == 0, 1, d)[:, None]).clip(-127, 127).astype(np.int8)
    out = np.empty((x.shape[0], 34), dtype=np.uint8)
    out[:, :2] = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.tobytes()


def write(path: Path, md, ts, q8: bool):
    """write_gguf's layout, with the routed experts as Q8_0 when q8"""
    if not q8:
        S.write_gguf(path, md, ts)
        return
    import struct
    header = b"GGUF" + struct.pack("<IQQ", 3, len(ts), len(md))
    kvs = b"".join(S._s(k) + v for k, v in md)
    infos, blobs, offset = b"", [], 0
    for name, arr in ts:
        arr = np.ascontiguousarray(arr, dtype="<f4")
        dims = list(reversed(arr.shape))
        quant = "_exps." in name and dims[0] % 32 == 0
        blob = q8_0(arr) if quant else arr.tobytes()
        infos += S._s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<I", 8 if quant else S.GGML_F32) + struct.pack("<Q", offset)   # 8: GGML_TYPE_Q8_0
        blobs.append(blob)
        offset += len(blob)
        offset = (offset + S.ALIGN - 1) // S.ALIGN * S.ALIGN
    data_pad = (len(header) + len(kvs) + len(infos) + S.ALIGN - 1) // S.ALIGN * S.ALIGN
    pad = b"\0" * (data_pad - len(header) - len(kvs) - len(infos))
    body = b"".join(b.ljust(len(b) + (-len(b) % S.ALIGN), b"\0") for b in blobs)
    path.write_bytes(header + kvs + infos + pad + body)


def build(big: bool, seed: int = 4242, nextn: bool = True, wide: str = ""):
    gw = {"": None, "64": S.WIDE, "512": S.WIDE512}[wide]
    g = S.BIG if big else gw or S.SMALL
    md, ts = S.build_big(777) if big else S._build(gw, 1234, False) if gw else S.build(1234)
    if not nextn:
        return md, ts
    n, e = g["n_layers"], g["n_embd"]
    rng = np.random.default_rng(seed)

    def w(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_w(k):
        return (1.0 + rng.standard_normal(k) * 0.02).astype(np.float32)

    # the per-layer arrays and the block count gain the NextN block (an attention layer)
    swap = {
        "glm5-next.block_count": S.kv_u32(n + 1),
        "glm5-next.attention.head_count_kv": S.arr_i32(list(g["n_head_kv"]) + [1]),
        "glm5-next.expert_feed_forward_length": S.arr_i32([g["n_ff_exp"]] * (n + 1)),
        "glm5-next.attention.indexer.types": S.arr_i32([1] * (n + 1)),
        "glm5-next.swiglu_clamp_exp": S.arr_f32([10.0] * (n + 1)),
        "glm5-next.swiglu_clamp_shexp": S.arr_f32([10.0] * (n + 1)),
    }
    md = [(k, swap.get(k, v)) for k, v in md] + [("glm5-next.nextn_predict_layers", S.kv_u32(1))]
    q_lora, kv_lora, mla_k, mla_v = g["q_lora"], g["kv_lora"], g["mla_k"], g["mla_v"]
    nh, idx_heads, idx_dim, kpool = g["n_head"], g["idx_heads"], g["idx_dim"], g["kpool"]
    ne, ff = g["n_expert"], g["n_ff_exp"]
    p = f"blk.{n}."
    ts += [
        (p + "nextn.eh_proj.weight", w(e, 2 * e)),
        (p + "nextn.enorm.weight", norm_w(e)),
        (p + "nextn.hnorm.weight", norm_w(e)),
        (p + "nextn.shared_head_norm.weight", norm_w(e)),
        (p + "attn_norm.weight", norm_w(e)),
        (p + "ffn_norm.weight", norm_w(e)),
        (p + "attn_q_a_norm.weight", norm_w(q_lora)),
        (p + "attn_kv_a_norm.weight", norm_w(kv_lora)),
        (p + "attn_q_a.weight", w(q_lora, e)),
        (p + "attn_q_b.weight", w(nh * mla_k, q_lora)),
        (p + "attn_kv_a_mqa.weight", w(kv_lora, e)),
        (p + "attn_k_b.weight", w(nh, kv_lora, mla_k)),
        (p + "attn_v_b.weight", w(nh, mla_v, kv_lora)),
        (p + "attn_output.weight", w(e, nh * mla_v)),
        (p + "indexer.k_norm.weight", norm_w(idx_dim)),
        (p + "indexer.k_norm.bias", w(idx_dim)),
        (p + "indexer.proj.weight", w(idx_heads, e)),
        (p + "indexer.attn_k.weight", w(idx_dim, e)),
        (p + "indexer.attn_q_b.weight", w(idx_heads * idx_dim, q_lora)),
        (p + "indexer_compressor_gate.weight", w(idx_dim, e)),
        (p + "indexer_compressor_ape.weight", w(kpool, idx_dim, scale=0.1)),
        (p + "ffn_gate_inp.weight", w(ne, e)),
        (p + "exp_probs_b.bias", w(ne, scale=0.1)),
        (p + "ffn_gate_exps.weight", w(ne, ff, e)),
        (p + "ffn_down_exps.weight", w(ne, e, ff)),
        (p + "ffn_up_exps.weight", w(ne, ff, e)),
        (p + "ffn_gate_shexp.weight", w(ff, e)),
        (p + "ffn_down_shexp.weight", w(e, ff)),
        (p + "ffn_up_shexp.weight", w(ff, e)),
    ]
    return md, ts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--big", action="store_true", help="the 45-layer fixture's trunk")
    ap.add_argument("--no-nextn", action="store_true", help="the trunk alone (no NextN block)")
    ap.add_argument("--q8-experts", action="store_true", help="the routed experts as Q8_0")
    ap.add_argument("--wide", nargs="?", const="64", default="", choices=["64", "512"],
                    help="the small trunk with 16 heads (glm_synth_gguf.WIDE; 512: WIDE512's 512-wide latent)")
    a = ap.parse_args()
    md, ts = build(a.big, nextn=not a.no_nextn, wide=a.wide)
    write(Path(a.out), md, ts, a.q8_experts)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

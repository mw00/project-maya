"""tools/glm_synth_gguf.py - a tiny, structurally complete synthetic glm5-next GGUF.

Phase 1 of docs/GLM5-FLASH.md needs a model the reference forward pass (`ref/glm.py`) and the
future engine kernels can run against without the 320B download: same tensor set, same metadata
keys, every path exercised (KDA layers, nope-MLA + k-pool DSA layers, dense lead FFN, MoE with
shared expert, mHC on both halves), dimensions scaled down and values drawn from a seeded RNG
(norm weights near 1, everything else small; `ssm_a` negative, since it holds -exp(A_log)).

The writer here is deliberately minimal and self-contained (GGUF v3, F32 tensors only) so the
synthetic artifact has no dependency on conversion-time choices; `ref/load.py`'s dequantization
path is exercised through F32 and the repo's own reader validates the result.

Usage:
    python tools/glm_synth_gguf.py --out glm5-synth.gguf [--seed 1234]

The tokenizer is deliberately absent: the oracle runs on token ids, and the real tokenizer is a
conversion concern (tools/strata_tokenizer.py), not a reference-model one.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

import numpy as np

# GGUF metadata value types (ggml spec order)
U8, I8, U16, I16, U32, I32, F32, BOOL, STRING, ARRAY, U64, I64, F64 = range(13)
GGML_F32 = 0

ALIGN = 32


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def kv_u32(v): return struct.pack("<I", U32) + struct.pack("<I", v)
def kv_i32(v): return struct.pack("<I", I32) + struct.pack("<i", v)
def kv_f32(v): return struct.pack("<I", F32) + struct.pack("<f", v)
def kv_bool(v): return struct.pack("<I", BOOL) + struct.pack("<B", 1 if v else 0)
def kv_str(v): return struct.pack("<I", STRING) + _s(v)


# array elements carry NO per-element type tag: the array header names the element type once
def kv_arr(elem_tag: int, elems: list[bytes]) -> bytes:
    return struct.pack("<I", ARRAY) + struct.pack("<I", elem_tag) + struct.pack("<Q", len(elems)) + b"".join(elems)


def arr_i32(vs): return kv_arr(I32, [struct.pack("<i", v) for v in vs])
def arr_f32(vs): return kv_arr(F32, [struct.pack("<f", v) for v in vs])


def write_gguf(path: Path, metadata: list[tuple[str, bytes]], tensors: list[tuple[str, np.ndarray]]) -> None:
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    kvs = b"".join(_s(k) + v for k, v in metadata)
    infos, blobs, offset = b"", [], 0
    for name, arr in tensors:
        arr = np.ascontiguousarray(arr, dtype="<f4")
        dims = list(reversed(arr.shape))                 # ne0 varies fastest
        infos += _s(name) + struct.pack("<I", len(dims))
        infos += b"".join(struct.pack("<Q", d) for d in dims)
        infos += struct.pack("<I", GGML_F32) + struct.pack("<Q", offset)
        blob = arr.tobytes()
        blobs.append(blob)
        offset += len(blob)
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN   # tensors start aligned, like gguf.cpp
    data_pad = (len(header) + len(kvs) + len(infos) + ALIGN - 1) // ALIGN * ALIGN
    pad = b"\0" * (data_pad - len(header) - len(kvs) - len(infos))
    body = b"".join(b.ljust(len(b) + (-len(b) % ALIGN), b"\0") for b in blobs)
    path.write_bytes(header + kvs + infos + pad + body)


def arr_str(vs): return kv_arr(STRING, [_s(v) for v in vs])


# The seed-1234 fixture geometry: every structural path of glm5-next, none of the size.
SMALL = dict(n_embd=128, n_vocab=512, n_layers=6, n_head=8, kda_head_dim=32, d_conv=4,
             q_lora=96, kv_lora=64, mla_k=48, mla_v=48, n_expert=12, n_exp_used=3,
             n_ff_exp=64, n_shared=1, n_ff_dense=192, dense_lead=1, hc=4, sinkhorn=3,
             idx_heads=4, idx_dim=16, idx_topk=8, kpool=2, ctx=4096, hot_ffn=0.05, dt_spikes=0.0,
             n_head_kv=[0, 0, 0, 1, 0, 1])

# The second synthetic: the REAL shape of GLM-5.3-Flash scaled down but with real magnitudes.
#   * 45 layers (odd), the real cadence - a DSA layer at il%4==3, layers 3..43, so the layer-type
#     array ends on a DSA layer and every per-layer array has odd length;
#   * n_embd / n_ff_exp / q_lora / kv_lora multiples of 256, so the IQ row width (256) is
#     exercised the moment this model is packed (the 128-wide seed-1234 fixture cannot);
#   * top-8 of 32 experts (the real n_expert_used), kpool 4 (the real pool width) with
#     top_k 256 so the 512-token sequences cross 128 pools and DISCARD half of them;
#   * sinkhorn iterations 5 - different from both the real 20 and the fixture's 3;
#   * hot FFN gate/up scales (0.25 vs 0.05) so the SwiGLU clamps actually fire, and +8 spikes on
#     every 8th ssm_dt channel so KDA decay gates actually reach the structural lower bound -
#     the coverage audit's two zero branches on the gentle fixture (tools/cover_audit.py).
BIG = dict(n_embd=256, n_vocab=512, n_layers=45, n_head=8, kda_head_dim=32, d_conv=4,
           q_lora=256, kv_lora=128, mla_k=64, mla_v=64, n_expert=32, n_exp_used=8,
           n_ff_exp=256, n_shared=1, n_ff_dense=512, dense_lead=3, hc=4, sinkhorn=5,
           idx_heads=8, idx_dim=64, idx_topk=256, kpool=4, ctx=1024, hot_ffn=0.25, dt_spikes=8.0,
           n_head_kv=[1 if il % 4 == 3 else 0 for il in range(45)])


# the small fixture with the heads the DSA products' matrix kernels need: ggml-metal takes mul_mm from 9 columns and a
# 64-long inner product (glm_metal.cpp dsa_mm), which SMALL/BIG's 8 heads never reach - and a top-k past the 64-cell pad
WIDE = dict(SMALL, n_head=16, idx_heads=16, idx_dim=64, idx_topk=128)
# and with GLM-5.3-Flash's 512-wide latent: ggml_sparse_mla's D=512 kernel (third_party/ggml/metal-sparse-mla.patch)
WIDE512 = dict(WIDE, kv_lora=512)


def build(seed: int, vocab: bool = False) -> tuple[list[tuple[str, bytes]], list[tuple[str, np.ndarray]]]:
    return _build(SMALL, seed, vocab)


def build_big(seed: int = 777, vocab: bool = False) -> tuple[list[tuple[str, bytes]], list[tuple[str, np.ndarray]]]:
    return _build(BIG, seed, vocab)


def _build(g: dict, seed: int, vocab: bool) -> tuple[list[tuple[str, bytes]], list[tuple[str, np.ndarray]]]:
    n_embd, n_vocab, n_layers = g["n_embd"], g["n_vocab"], g["n_layers"]
    n_head, kda_head_dim, d_conv = g["n_head"], g["kda_head_dim"], g["d_conv"]
    d_inner = n_head * kda_head_dim
    q_lora, kv_lora, mla_k, mla_v = g["q_lora"], g["kv_lora"], g["mla_k"], g["mla_v"]
    n_expert, n_exp_used, n_ff_exp, n_shared = g["n_expert"], g["n_exp_used"], g["n_ff_exp"], g["n_shared"]
    n_ff_dense, dense_lead = g["n_ff_dense"], g["dense_lead"]
    hc, sinkhorn = g["hc"], g["sinkhorn"]
    hc_mix_dim = (2 + hc) * hc
    idx_heads, idx_dim, idx_topk, kpool = g["idx_heads"], g["idx_dim"], g["idx_topk"], g["kpool"]
    n_head_kv = g["n_head_kv"]
    hot, dt_spikes = g["hot_ffn"], g["dt_spikes"]
    rng = np.random.default_rng(seed)

    def w(*shape, scale=0.05):                            # a plain weight
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_w(n):                                        # an RMS/LayerNorm gain, near 1
        return (1.0 + rng.standard_normal(n) * 0.02).astype(np.float32)

    md = [
        ("general.architecture", kv_str("glm5-next")),
        ("general.name", kv_str("glm5-synth")),
        ("general.alignment", kv_u32(ALIGN)),
        ("glm5-next.embedding_length", kv_u32(n_embd)),
        ("glm5-next.block_count", kv_u32(n_layers)),
        ("glm5-next.vocab_size", kv_u32(n_vocab)),
        ("glm5-next.attention.head_count", kv_u32(n_head)),
        ("glm5-next.attention.head_count_kv", arr_i32(n_head_kv)),
        ("glm5-next.attention.key_length_mla", kv_u32(mla_k)),
        ("glm5-next.attention.value_length_mla", kv_u32(mla_v)),
        ("glm5-next.attention.q_lora_rank", kv_u32(q_lora)),
        ("glm5-next.attention.kv_lora_rank", kv_u32(kv_lora)),
        ("glm5-next.attention.layer_norm_rms_epsilon", kv_f32(1e-5)),
        ("glm5-next.attention.layer_norm_epsilon", kv_f32(1e-5)),
        # nope-only MLA: without an explicit rope dim llama.cpp derives n_rot from the generic
        # head math and sizes attn_kv_a_mqa as kv_lora + rope (the released config is
        # qk_rope_head_dim 0 / mla_use_nope)
        ("glm5-next.rope.dimension_count", kv_u32(0)),
        # the K-cache row width follows attention.key_length (the FULL head dim); the MLA cache
        # holds the compressed latent, so the key length is kv_lora_rank + rope (= kv_lora here) -
        # the same convention as llama.cpp's deepseek2 GGUFs.  Without it the generic default
        # (n_embd / n_head) sizes the cache and the scatter aborts on the width mismatch.
        ("glm5-next.attention.key_length", kv_u32(kv_lora)),
        ("glm5-next.ssm.conv_kernel", kv_u32(d_conv)),
        ("glm5-next.kda.head_dim", kv_u32(kda_head_dim)),
        ("glm5-next.kda.gate_lower_bound", kv_f32(-5.0)),
        ("glm5-next.expert_count", kv_u32(n_expert)),
        ("glm5-next.expert_used_count", kv_u32(n_exp_used)),
        ("glm5-next.expert_feed_forward_length", arr_i32([n_ff_exp] * n_layers)),
        ("glm5-next.expert_shared_count", kv_u32(n_shared)),
        ("glm5-next.leading_dense_block_count", kv_u32(dense_lead)),
        ("glm5-next.feed_forward_length", kv_u32(n_ff_dense)),
        ("glm5-next.expert_weights_scale", kv_f32(2.5)),
        ("glm5-next.expert_weights_norm", kv_bool(True)),
        ("glm5-next.expert_gating_func", kv_u32(2)),   # LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
                                                       # (llama-hparams.h; the key is a u32 enum)
        ("glm5-next.attention.indexer.head_count", kv_u32(idx_heads)),
        ("glm5-next.attention.indexer.key_length", kv_u32(idx_dim)),
        ("glm5-next.attention.indexer.top_k", kv_u32(idx_topk)),
        ("glm5-next.attention.indexer.kpool", kv_u32(kpool)),
        ("glm5-next.attention.indexer.kpool_select_tail", kv_bool(True)),
        ("glm5-next.attention.indexer.types", arr_i32([1] * n_layers)),
        ("glm5-next.hyper_connection.count", kv_u32(hc)),
        ("glm5-next.hyper_connection.sinkhorn_iterations", kv_u32(sinkhorn)),
        ("glm5-next.hyper_connection.epsilon", kv_f32(1e-6)),
        ("glm5-next.swiglu_clamp_exp", arr_f32([10.0] * n_layers)),
        ("glm5-next.swiglu_clamp_shexp", arr_f32([10.0] * n_layers)),
    ]
    if vocab:
        # everything upstream llama.cpp needs to LOAD the model (the oracle runs on ids and needs
        # none of this - the seed-1234 fixture GGUF is built without it and stays byte-stable).
        # pre="glm5" is llama-vocab.cpp's CHATGLM4 scheme with ignore_merges and a nulled BOS;
        # the model is never tokenized here, the vocab only has to parse.
        md += [
            ("glm5-next.context_length", kv_u32(g["ctx"])),
            ("tokenizer.ggml.model", kv_str("gpt2")),
            ("tokenizer.ggml.pre", kv_str("glm5")),
            ("tokenizer.ggml.tokens", arr_str(["tok%d" % i for i in range(n_vocab)])),
            ("tokenizer.ggml.scores", arr_f32([0.0] * n_vocab)),
            ("tokenizer.ggml.token_type", arr_i32([1] * n_vocab)),
            ("tokenizer.ggml.merges", arr_str([])),
            # no bos/eos keys: the "glm5" pre branch nulls the BOS (llama-vocab.cpp) and llama.cpp
            # treats absent ids as LLAMA_TOKEN_NULL; they would have to be u32 if present
        ]

    ts: list[tuple[str, np.ndarray]] = [
        ("token_embd.weight", w(n_vocab, n_embd)),
        ("output_norm.weight", norm_w(n_embd)),
        ("output.weight", w(n_vocab, n_embd)),
    ]
    for il in range(n_layers):
        p = f"blk.{il}."
        ts += [
            (p + "attn_norm.weight", norm_w(n_embd)),
            (p + "ffn_norm.weight", norm_w(n_embd)),
            (p + "hc_attn_fn.weight", w(hc_mix_dim, hc * n_embd)),     # GGUF ne0 is the IN dim: numpy is (out, in)
            (p + "hc_attn_base.weight", rng.standard_normal(hc_mix_dim).astype(np.float32) * 0.1),
            (p + "hc_attn_scale.weight", (1.0 + rng.standard_normal(3) * 0.1).astype(np.float32)),
            (p + "hc_ffn_fn.weight", w(hc_mix_dim, hc * n_embd)),
            (p + "hc_ffn_base.weight", rng.standard_normal(hc_mix_dim).astype(np.float32) * 0.1),
            (p + "hc_ffn_scale.weight", (1.0 + rng.standard_normal(3) * 0.1).astype(np.float32)),
        ]
        if n_head_kv[il] == 0:                            # KDA layer
            ts += [
                (p + "attn_q.weight", w(d_inner, n_embd)),
                (p + "attn_k.weight", w(d_inner, n_embd)),
                (p + "attn_v.weight", w(d_inner, n_embd)),
                (p + "ssm_conv1d_q.weight", w(d_inner, 1, d_conv)),
                (p + "ssm_conv1d_k.weight", w(d_inner, 1, d_conv)),
                (p + "ssm_conv1d_v.weight", w(d_inner, 1, d_conv)),
                (p + "ssm_f_a.weight", w(kda_head_dim, n_embd)),
                (p + "ssm_f_b.weight", w(d_inner, kda_head_dim)),
                (p + "ssm_beta.weight", w(n_head, n_embd)),
                # ssm_a holds -exp(A_log): negative by contract
                (p + "ssm_a", (-np.exp(rng.uniform(0.5, 1.5, n_head))).astype(np.float32)),
                # dt spikes (+dt_spikes on every 8th channel) push some decay gates onto the
                # structural lower bound so the saturation branch is exercised (no extra draws);
                # the f32 cast BEFORE the 0.1 scale matches the original byte-exact expression
                (p + "ssm_dt.bias",
                 _with_spikes(rng.standard_normal(d_inner).astype(np.float32) * 0.1, dt_spikes)),
                (p + "ssm_g_a.weight", w(kda_head_dim, n_embd)),
                (p + "ssm_g_b.weight", w(d_inner, kda_head_dim)),
                (p + "ssm_norm.weight", norm_w(kda_head_dim)),
                (p + "attn_output.weight", w(n_embd, d_inner)),
            ]
        else:                                             # nope-MLA + DSA layer
            ts += [
                (p + "attn_q_a_norm.weight", norm_w(q_lora)),
                (p + "attn_kv_a_norm.weight", norm_w(kv_lora)),
                (p + "attn_q_a.weight", w(q_lora, n_embd)),
                (p + "attn_q_b.weight", w(n_head * mla_k, q_lora)),
                (p + "attn_kv_a_mqa.weight", w(kv_lora, n_embd)),
                (p + "attn_k_b.weight", w(n_head, kv_lora, mla_k)),
                (p + "attn_v_b.weight", w(n_head, mla_v, kv_lora)),
                (p + "attn_output.weight", w(n_embd, n_head * mla_v)),
                (p + "indexer.k_norm.weight", norm_w(idx_dim)),
                (p + "indexer.k_norm.bias", rng.standard_normal(idx_dim).astype(np.float32) * 0.05),
                (p + "indexer.proj.weight", w(idx_heads, n_embd)),
                (p + "indexer.attn_k.weight", w(idx_dim, n_embd)),
                (p + "indexer.attn_q_b.weight", w(idx_heads * idx_dim, q_lora)),
                (p + "indexer_compressor_gate.weight", w(idx_dim, n_embd)),
                (p + "indexer_compressor_ape.weight", (rng.standard_normal((kpool, idx_dim)) * 0.1).astype(np.float32)),
            ]
        if il < dense_lead:                               # dense lead FFN
            ts += [
                (p + "ffn_gate.weight", w(n_ff_dense, n_embd, scale=hot)),
                (p + "ffn_down.weight", w(n_embd, n_ff_dense)),
                (p + "ffn_up.weight", w(n_ff_dense, n_embd, scale=hot)),
            ]
        else:                                             # MoE + shared expert
            ts += [
                (p + "ffn_gate_inp.weight", w(n_expert, n_embd)),
                (p + "exp_probs_b.bias", rng.standard_normal(n_expert).astype(np.float32) * 0.1),
                (p + "ffn_gate_exps.weight", w(n_expert, n_ff_exp, n_embd, scale=hot)),
                (p + "ffn_down_exps.weight", w(n_expert, n_embd, n_ff_exp)),
                (p + "ffn_up_exps.weight", w(n_expert, n_ff_exp, n_embd, scale=hot)),
                (p + "ffn_gate_shexp.weight", w(n_ff_exp * n_shared, n_embd, scale=hot)),
                (p + "ffn_down_shexp.weight", w(n_embd, n_ff_exp * n_shared)),
                (p + "ffn_up_shexp.weight", w(n_ff_exp * n_shared, n_embd, scale=hot)),
            ]
    return md, ts


def _with_spikes(dt: np.ndarray, spike: float) -> np.ndarray:
    if spike:
        dt = dt.copy()
        dt[::8] += spike
    return dt.astype(np.float32)


def selftest(path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    gf = GGUFFile(path)
    by = {t.name: t for t in gf.tensors}
    assert gf.metadata["general.architecture"] == "glm5-next", "architecture key lost"
    assert gf.metadata["glm5-next.attention.head_count_kv"] == [0, 0, 0, 1, 0, 1], "array KV mangled"
    assert list(by["blk.3.attn_k_b.weight"].shape) == [48, 64, 8], f"3-D shape {by['blk.3.attn_k_b.weight'].shape}"
    assert list(by["blk.1.ffn_gate_exps.weight"].shape) == [128, 64, 12], "expert shape wrong"
    assert "blk.0.ffn_gate_exps.weight" not in by and "blk.0.ffn_gate.weight" in by, "dense lead wrong"
    assert "blk.0.attn_q.weight" in by and "blk.3.attn_q_a.weight" in by, "layer types wrong"
    n = len(gf.tensors)
    print(f"selftest ok: {n} tensors, {path.stat().st_size / 1e6:.2f} MB")


def selftest_big(path: Path) -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gguf_reader import GGUFFile
    gf = GGUFFile(path)
    by = {t.name: t for t in gf.tensors}
    md = gf.metadata
    assert md["general.architecture"] == "glm5-next", "architecture key lost"
    kv = md["glm5-next.attention.head_count_kv"]
    assert len(kv) == 45 and sum(kv) == 11, f"layer pattern {kv}"
    assert [i for i, v in enumerate(kv) if v] == [3, 7, 11, 15, 19, 23, 27, 31, 35, 39, 43], "cadence"
    assert md["glm5-next.hyper_connection.sinkhorn_iterations"] == 5, "sinkhorn"
    assert md["glm5-next.attention.indexer.top_k"] == 256 and md["glm5-next.attention.indexer.kpool"] == 4, "indexer"
    assert list(by["blk.43.attn_k_b.weight"].shape) == [64, 128, 8], "big 3-D shape"
    assert list(by["blk.3.ffn_gate_exps.weight"].shape) == [256, 256, 32], "big expert shape"
    assert "blk.44.ffn_gate_exps.weight" in by and "blk.44.attn_q.weight" in by, "last layer KDA MoE"
    n_exp_used = md["glm5-next.expert_used_count"]
    assert n_exp_used == 8, "top-8"
    print(f"selftest ok: {len(gf.tensors)} tensors, {path.stat().st_size / 1e6:.2f} MB (big)")


def main() -> int:
    ap = argparse.ArgumentParser(description="write a tiny synthetic glm5-next GGUF")
    ap.add_argument("--out", default="glm5-synth.gguf")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--vocab", action="store_true",
                    help="add the tokenizer metadata upstream llama.cpp needs to LOAD the model "
                         "(byte-different from the default; the fixture stays vocab-less)")
    ap.add_argument("--big", action="store_true",
                    help="the 45-layer real-cadence geometry (BIG) instead of the 6-layer fixture")
    args = ap.parse_args()
    md, ts = build_big(args.seed, vocab=args.vocab) if args.big else build(args.seed, vocab=args.vocab)
    out = Path(args.out)
    write_gguf(out, md, ts)
    (selftest_big if args.big else selftest)(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

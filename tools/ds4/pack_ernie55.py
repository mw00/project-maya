"""tools/ds4/pack_ernie55.py - maya-ds4's pack for an ERNIE-5.5-Flash GGUF set (baidu/ERNIE-5.5-Flash-Preview's Maya
quants, general.architecture ernie55: the checkpoint's own tensor names, its dense FP8 kept as published), beside the
model's files.

    python tools/ds4/pack_ernie55.py --model ~/models/ERNIE-5.5-Flash-Preview-Maya-S

writes <model>/maya-ds4/ in pack_gguf.py's format, so the engine reads it as it reads a DeepSeek-V4.1 GGUF's:
  ds4.txt       the geometry (config.json's names for what the two models share, ERNIE's own beside them), arch ernie55,
                model_dir, format gguf
  tensors.txt   every tensor the engine reads under its names (layers.N.attn.*, layers.N.ffn.*, ...): `file <i> <path>`
                lines, then `t <name> <dtype> <file> <offset> <bytes> <dims...>` (dims outermost first); the dense FP8
                as dtype F8_E4M3 (one E4M3 byte a value) with its F32 scale a 128 x 128 block beside it
                (<name>_scale_inv), the routed experts an entry an expert (layers.N.ffn.experts.E.w1.weight), the
                grouped wo_a an entry a group (attn.wo_a.slice.G.weight and its scale's rows)
  ernie55.bin   the few tensors stored in another layout than the engine's: the hyper-connections' mixers (the
                checkpoint keeps them [in][out]: here [out][in], as DeepSeek's hc_*_fn), their three alpha scalars as
                one [3] (hc_*_scale), mhc_contract's head mixer the same way
  tokenizer/    the server's tokenizer from the GGUF's tokenizer.huggingface.json (pre "ernie5"), and ERNIE's chat
                template (tools/ds4/ernie55_chat_template.jinja)

Only names and offsets are written: the GGUF's bytes are read in place.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from gguf_reader import GGUFFile  # noqa: E402

# ggml type -> (name in tensors.txt, values a block, bytes a block); I8 is the FP8 the set stores raw (its raw_f8_e4m3)
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2), 8: ("Q8_0", 32, 34), 24: ("I8", 1, 1),
         10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
         14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
         19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
         23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56)}
HCA, MLA = 128, -2

# the checkpoint's name under model.layers.N. -> the engine's under layers.N. (both layer kinds; None: handled below)
COMMON = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.vha_postmix_U": "attn.vha_postmix_U",
    "self_attn.vha_postmix_V": "attn.vha_postmix_V",
    "self_attn.gate_proj.weight": "attn.gate_proj.weight",
    "self_attention_hyper_connection.bias": "hc_attn_base",
    "mlp_hyper_connection.bias": "hc_ffn_base",
    # the dense first layer's FFN
    "mlp.gate_proj.weight": "ffn.dense.w1.weight",
    "mlp.up_proj.weight": "ffn.dense.w3.weight",
    "mlp.down_proj.weight": "ffn.dense.w2.weight",
    # the latent MoE
    "block_sparse_moe.gate.weight": "ffn.gate.weight",
    "block_sparse_moe.gate.weight_1": "ffn.gate.weight_1",
    "block_sparse_moe.e_score_correction_bias": "ffn.gate.bias",
    "block_sparse_moe.gate.routed_scaling_factor_param": "ffn.gate.scale",
    "block_sparse_moe.fc1_latent_proj.weight": "ffn.fc1_latent.weight",
    "block_sparse_moe.fc2_latent_proj.weight": "ffn.fc2_latent.weight",
    "block_sparse_moe.latent_norm.weight": "ffn.latent_norm.weight",
    "block_sparse_moe.shared_experts.w1.weight": "ffn.shared_experts.w1.weight",
    "block_sparse_moe.shared_experts.w2.weight": "ffn.shared_experts.w2.weight",
    "block_sparse_moe.shared_experts.w3.weight": "ffn.shared_experts.w3.weight",
}
HCA_NAMES = {
    "self_attn.linear_q_down_proj.weight": "attn.wq_a.weight",
    "self_attn.q_layernorm.weight": "attn.q_norm.weight",
    "self_attn.linear_q_up_proj.weight": "attn.wq_b.weight",
    "self_attn.linear_kv_proj.weight": "attn.wkv.weight",
    "self_attn.kv_layernorm.weight": "attn.kv_norm.weight",
    "self_attn.linear_o_group_proj": "attn.wo_a",               # one entry a group (the engine's slices)
    "self_attn.o_proj.weight": "attn.wo_b.weight",
    "self_attn.core_attention.attn_sink": "attn.attn_sink",
    "self_attn.core_attention.compressor.linear_wkv.weight": "attn.compressor.wkv.weight",
    "self_attn.core_attention.compressor.linear_wgate.weight": "attn.compressor.wgate.weight",
    "self_attn.core_attention.compressor.norm.weight": "attn.compressor.norm.weight",
    "self_attn.core_attention.compressor.ape": "attn.compressor.ape",
}
MLA_NAMES = {
    "self_attn.q_a_proj.weight": "attn.wq_a.weight",
    "self_attn.q_a_layernorm.weight": "attn.q_norm.weight",
    "self_attn.q_b_proj.weight": "attn.wq_b.weight",
    "self_attn.kv_a_proj_with_mqa.weight": "attn.wkv.weight",
    "self_attn.kv_a_layernorm.weight": "attn.kv_norm.weight",
    "self_attn.kv_b_proj.weight": "attn.kv_b.weight",
    "self_attn.o_proj.weight": "attn.wo_b.weight",
    "self_attn.core_attention.softmax_offset": "attn.attn_sink",
    "self_attn.core_attention.indexer.wq_b.weight": "attn.indexer.wq_b.weight",
    "self_attn.core_attention.indexer.wk.weight": "attn.indexer.wk.weight",
    "self_attn.core_attention.indexer.weights_proj.weight": "attn.indexer.weights_proj.weight",
    "self_attn.core_attention.indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "self_attn.core_attention.indexer.k_norm.bias": "attn.indexer.k_norm.bias",
}
# not read: FlashIndexer's head gate (config.json's enable_flashindexer is off: every indexer head scores)
SKIP = ("self_attn.core_attention.indexer.linear_gate.weight", "self_attn.core_attention.indexer.flashindexer_bias")
GLOBAL = {
    "model.embed_tokens.weight": "embed.weight",
    "lm_head.weight": "head.weight",
    "model.norm.weight": "norm.weight",
    "model.lm_head.multimax_ranges": "multimax.ranges",
    "model.lm_head.multimax_ts": "multimax.ts",
    "model.mhc_contract.hc_head_base": "hc_head_base",
    "model.mhc_contract.hc_head_scale": "hc_head_scale",
}


def geometry(c: dict) -> dict:
    """ds4.txt's lines: DeepSeek-V4.1's names for what ERNIE-5.5 shares, its own for the rest."""
    nl = c["num_hidden_layers"]
    ratios = [int(r) for r in c["compress_ratios"][:nl]]
    bad = sorted(set(ratios) - {HCA, MLA})
    if bad:
        raise SystemExit("compress_ratios %s: only %d (HCA) and %d (MLA) are covered" % (bad, HCA, MLA))
    need = {"hidden_act": "situ", "moe_split_feature_routing": True, "routed_scaling_factor_learnable": True,
            "latent_moe_use_norm": True, "multimax": "lm_head", "n_group": 1, "topk_group": 1,
            "scoring_func": "sigmoid", "use_vha_attention": True, "use_gated_attn": True}
    wrong = {k: c.get(k) for k, v in need.items() if c.get(k, v) != v}
    if wrong:
        raise SystemExit("this ERNIE-5.5 checkpoint sets %s; the engine implements %s" % (wrong, need))
    if c.get("enable_flashindexer"):
        raise SystemExit("enable_flashindexer: the FlashIndexer head gate is not implemented")
    rs = c.get("rope_scaling") or {}
    gen = {
        "arch": "ernie55",
        "vocab_size": c["vocab_size"], "hidden_size": c["hidden_size"], "num_hidden_layers": nl,
        "num_attention_heads": c["num_attention_heads"], "head_dim": c["head_dim"],
        "qk_rope_head_dim": c["qk_rope_head_dim"], "q_lora_rank": c["q_lora_rank"], "o_lora_rank": c["o_lora_rank"],
        "o_groups": c["o_groups"], "n_routed_experts": c["n_routed_experts"],
        "num_experts_per_tok": c["num_experts_per_tok"], "moe_intermediate_size": c["moe_intermediate_size"],
        "sliding_window": c["sliding_window"], "hc_mult": c["hc_mult"], "hc_sinkhorn_iters": c["hc_sinkhorn_iters"],
        "hc_eps": c["hc_eps"], "rms_norm_eps": c["rms_norm_eps"],
        "index_n_heads": c["index_n_heads"], "index_head_dim": c["index_head_dim"], "index_topk": c["index_topk"],
        "rope_theta": c["rope_theta"], "compress_rope_theta": c["compress_rope_theta"],
        "rope_scaling.factor": rs.get("factor", 1.0), "rope_scaling.beta_fast": rs.get("beta_fast", 32),
        "rope_scaling.beta_slow": rs.get("beta_slow", 1),
        "rope_scaling.original_max_position_embeddings": rs.get("original_max_position_embeddings", 0),
        # the chat turn's end (<|ernie:im_end|>): the answer's stop token
        "eos_token_id": 8,
        "routed_scaling_factor": 1.0,     # (each expert's own learned factor instead: ffn.gate.scale)
        "swiglu_limit": 0.0,
        "compress_ratios": ratios,
        # every HCA layer compresses its own KV; the MLA layers keep theirs apart (no shared source, no shared indexer)
        "kv_source_layer_ids": [i for i, r in enumerate(ratios) if r == HCA],
        "index_source_layer_ids": [],
        "candidate_source_layer_id": -1,
        # ERNIE-5.5's own
        "first_k_dense_replace": c["first_k_dense_replace"], "intermediate_size": c["intermediate_size"],
        "moe_latent_size": c["moe_latent_size"], "n_shared_experts": c["n_shared_experts"],
        "activation_situ_beta": c["activation_situ_beta"],
        "activation_situ_linear_beta": c["activation_situ_linear_beta"],
        "norm_topk_prob": int(bool(c.get("norm_topk_prob", True))),
        "mla_layer_ids": [i for i, r in enumerate(ratios) if r == MLA],
        "mla_num_heads": c["hybrid_mla_num_attention_heads"], "mla_q_lora_rank": c["hybrid_mla_q_lora_rank"],
        "mla_kv_lora_rank": c["hybrid_mla_kv_lora_rank"], "mla_qk_nope_head_dim": c["hybrid_mla_qk_nope_head_dim"],
        "mla_qk_rope_head_dim": c["hybrid_mla_qk_rope_head_dim"], "mla_v_head_dim": c["hybrid_mla_v_head_dim"],
        "vha_rank": 16,
    }
    return gen


def write_tokenizer(out: pathlib.Path, hf: dict, template: str):
    """The server's tokenizer (tools/strata_tokenizer.py's files) from the HF tokenizer.json the GGUF carries."""
    m = hf["model"]
    if m.get("type") != "BPE":
        raise SystemExit("tokenizer.huggingface.json: a %s model, not BPE" % m.get("type"))
    vocab = dict(m["vocab"])
    types = {}
    for a in hf.get("added_tokens", []):
        vocab[a["content"]] = a["id"]
        types[a["id"]] = 3 if a.get("special") else 4   # GGUF: CONTROL / USER_DEFINED
    n = max(vocab.values()) + 1
    tokens = [""] * n
    for t, i in vocab.items():
        tokens[i] = t
    missing = [i for i, t in enumerate(tokens) if t == ""]
    for i in missing:   # (ids no string names: placeholders that never match)
        tokens[i] = "<|ERNIE_unused_%d|>" % i
        types[i] = 5
    merges = [" ".join(x) if isinstance(x, list) else x for x in m.get("merges", [])]
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps([types.get(i, 1) for i in range(n)]))
    ids = {a["content"]: a["id"] for a in hf.get("added_tokens", [])}
    special = {"bos": ids.get("<|ERNIE_BEGIN|>", 1), "eos": ids.get("<|ernie:im_end|>", 8),
               "pad": ids.get("<|ERNIE_PAD|>", 0)}
    (out / "tokenizer.json").write_text(json.dumps({"pre": "ernie5", "special_ids": special}, indent=1))
    (out / "chat_template.jinja").write_text(template, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the folder of the ERNIE-5.5 GGUF shards")
    ap.add_argument("--out", help="the pack's folder (default <model>/maya-ds4)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser().resolve() if a.out else model / "maya-ds4"
    shards = sorted(p for p in model.glob("*.gguf"))
    if not shards:
        raise SystemExit("%s: no .gguf files" % model)
    gs = [GGUFFile(p) for p in shards]
    meta = gs[0].metadata
    if meta.get("general.architecture") != "ernie55":
        raise SystemExit("%s: general.architecture %r, not ernie55" % (shards[0].name, meta.get("general.architecture")))
    cfg = json.loads(meta["ernie55.config_json"])
    fp8 = set(meta.get("ernie55.raw_f8_e4m3", []))
    geo = geometry(cfg)
    ratios = geo["compress_ratios"]
    nl = geo["num_hidden_layers"]
    files = [p.name for p in shards] + ["maya-ds4/ernie55.bin"]
    side = len(files) - 1
    tensors, unknown = [], []
    by_name = {}
    for fi, g in enumerate(gs):
        for ti in g.tensors:
            if ti.type_id not in TYPES:
                raise SystemExit("%s: %s has ggml type %d, which the engine does not read" % (shards[fi].name, ti.name,
                                                                                                ti.type_id))
            tname, per, bpb = TYPES[ti.type_id]
            nel = 1
            for d in ti.shape:
                nel *= d
            shape = list(reversed(ti.shape))   # ggml ne0 innermost -> torch order, outermost first
            by_name[ti.name] = (tname, fi, g.data_start + ti.offset, nel // per * bpb, shape)
    sidebuf = bytearray()

    def side_add(name, arr_bytes, shape):
        off = len(sidebuf)
        sidebuf.extend(arr_bytes)
        while len(sidebuf) % 64:
            sidebuf.append(0)
        tensors.append((name, "F32", side, off, len(arr_bytes), shape))

    def read(name):
        tname, fi, off, nb, shape = by_name[name]
        with open(model / files[fi], "rb") as f:
            f.seek(off)
            return f.read(nb)

    def transposed_f32(name):
        """A [in][out] F32 matrix as [out][in]."""
        tname, fi, off, nb, shape = by_name[name]
        if tname != "F32" or len(shape) != 2:
            raise SystemExit("%s: %s %s, an F32 matrix expected" % (name, tname, shape))
        rows, cols = shape
        v = struct.unpack("<%df" % (rows * cols), read(name))
        t = [v[r * cols + c] for c in range(cols) for r in range(rows)]
        return struct.pack("<%df" % len(t), *t), [cols, rows]

    def emit(ck, name):
        tname, fi, off, nb, shape = by_name[name]
        if tname == "I8":
            if name not in fp8:
                raise SystemExit("%s: I8 without the set's raw_f8_e4m3 mark" % name)
            tname = "F8_E4M3"
        tensors.append((ck, tname, fi, off, nb, shape))

    used = set()
    for name, (tname, fi, off, nb, shape) in by_name.items():
        if name.endswith("_scale_inv"):
            continue   # (emitted with their matrix)
        if name in GLOBAL:
            emit(GLOBAL[name], name)
            used.add(name)
            continue
        m = re.match(r"model\.layers\.(\d+)\.(.+)$", name)
        if not m:
            unknown.append(name)
            continue
        li, rest = int(m.group(1)), m.group(2)
        if li >= nl:   # (a draft block's: not in this pack)
            continue
        P = "layers.%d." % li
        kind = HCA_NAMES if ratios[li] == HCA else MLA_NAMES
        if rest in SKIP:
            used.add(name)
            continue
        em = re.match(r"block_sparse_moe\.experts\.(\d+)\.(w[123])\.weight$", rest)
        hcm = re.match(r"(self_attention|mlp)_hyper_connection\.(mapping_proj\.weight|alpha_pre|alpha_post|alpha_res)$",
                       rest)
        if em:
            ck = "%sffn.experts.%s.%s.weight" % (P, em.group(1), em.group(2))
        elif hcm:
            kindhc = "attn" if hcm.group(1) == "self_attention" else "ffn"
            if hcm.group(2) == "mapping_proj.weight":
                b, sh = transposed_f32(name)
                side_add("%shc_%s_fn" % (P, kindhc), b, sh)
            elif hcm.group(2) == "alpha_pre":   # the three as one [pre, post, res]
                H = "model.layers.%d.%s_hyper_connection." % (li, hcm.group(1))
                vals = b"".join(read(H + x) for x in ("alpha_pre", "alpha_post", "alpha_res"))
                side_add("%shc_%s_scale" % (P, kindhc), vals, [3])
            used.add(name)
            continue
        elif rest in COMMON:
            ck = P + COMMON[rest]
        elif rest in kind:
            ck = P + kind[rest]
        else:
            unknown.append(name)
            continue
        used.add(name)
        if ck.endswith(".attn.wo_a"):   # [o_groups * o_lora][in] -> a group an entry, its scale's rows with it
            G, R = geo["o_groups"], geo["o_lora_rank"]
            sname = name + "_scale_inv"
            stn, sfi, soff, snb, sshape = by_name[sname]
            if shape[0] != G * R or R % 128 or sshape[0] * 128 != shape[0]:
                raise SystemExit("%s: %s does not split in %d groups of %d rows" % (name, shape, G, R))
            part, spart = nb // G, snb // G
            for i in range(G):
                tensors.append(("%s.slice.%d.weight" % (ck, i), "F8_E4M3", fi, off + i * part, part, [R, shape[1]]))
                tensors.append(("%s.slice.%d.weight_scale_inv" % (ck, i), "F32", sfi, soff + i * spart, spart,
                                [R // 128, sshape[1]]))
            used.add(sname)
            continue
        emit(ck, name)
        sname = name[:-len(".weight")] + ".weight_scale_inv" if name.endswith(".weight") else None
        if sname and sname in by_name:
            stn, sfi, soff, snb, sshape = by_name[sname]
            tensors.append((ck[:-len(".weight")] + ".weight_scale_inv", stn, sfi, soff, snb, sshape))
            used.add(sname)
        elif tensors[-1][1] == "F8_E4M3":
            raise SystemExit("%s: FP8 without its block scale" % name)
    # the head's mixer, [in][out] in the checkpoint
    b, sh = transposed_f32("model.mhc_contract.hc_head_fn")
    side_add("hc_head_fn", b, sh)
    used.add("model.mhc_contract.hc_head_fn")
    unknown = [n for n in unknown if n not in used]
    if unknown:
        raise SystemExit("tensors the pack does not map (%d): %s" % (len(unknown), ", ".join(unknown[:12])))
    out.mkdir(parents=True, exist_ok=True)
    (out / "ernie55.bin").write_bytes(bytes(sidebuf))
    with (out / "ds4.txt").open("w") as f:
        for k, v in geo.items():
            f.write("%s %s\n" % (k, ",".join(str(x) for x in v) if isinstance(v, list) else v))
        f.write("format gguf\n")
        f.write("model_dir %s\n" % model)
    with (out / "tensors.txt").open("w") as f:
        for i, n in enumerate(files):
            f.write("file %d %s\n" % (i, n))
        for name, dt, fi, off, nb, shape in tensors:
            f.write("t %s %s %d %d %d %s\n" % (name, dt, fi, off, nb, " ".join(map(str, shape))))
    tpl = HERE / "ernie55_chat_template.jinja"   # (the server knows the model by it: never an empty one)
    if not tpl.is_file() or not tpl.read_text(encoding="utf-8").strip():
        print("pack_ernie55: %s is missing or empty" % tpl, file=sys.stderr)
        return 1
    write_tokenizer(out / "tokenizer", json.loads(meta["tokenizer.huggingface.json"]), tpl.read_text(encoding="utf-8"))
    by = {}
    for t in tensors:
        by[t[1]] = by.get(t[1], 0) + t[4]
    print("%s: %d tensors in %d files (%s) -> %s" % (model.name, len(tensors), len(files),
                                                      ", ".join("%s %.1f GB" % (k, v / 1e9) for k, v in sorted(by.items())),
                                                      out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

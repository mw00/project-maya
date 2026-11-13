"""tools/inkling2_gguf.py - Inkling-2-Small's checkpoint as an `inkling2` GGUF, in the layout llama.cpp's Inkling 2
conversion writes (ggml-org/llama.cpp PR #32187, the format unsloth/Inkling-2-Small-GGUF ships), with no value
changed beyond that layout's own folds. What Maya's Inkling 2 reference (ref/inkling2.py) and engine read; its
experts the original bits.

    python tools/inkling2_gguf.py --src ~/Models/Inkling-2-Small --out ~/Models/Inkling-2-Small/maya-bf16 \
        [--vocab-from Inkling-2-Small-UD-IQ1_S-00001-of-00005.gguf] [--layers 0,5] [--split-gb 48]
    python tools/inkling2_gguf.py --src DIR --plan [--against tensors.json --kv-against FIRST_SHARD.gguf]

The layout (written from the PR's conversion and checked against unsloth's files' headers, `--plan --against`):
  - names llama.cpp's: token_embd(.norm), output, output_norm, output_res_score; a layer's attn_q / k / v / r /
    output / gate / sink_proj, attn_q_clip / r_clip (the scaled clips' scales), attn_rel_proj, attn_norm (sliding)
    or attn_{q,k,v}_in_norm (full), attn_res_score (the attention residual's query: layer l's is the checkpoint's
    attn_queries.(l-1); a full layer's 3 of them), ffn_res_score, ffn_norm, ffn_gate_inp, exp_probs_b.bias,
    shortconv_{attn,mlp,k,v} and a full layer's shortconv_{k,v}_dyn, ffn_{gate,up,down}_exps(.scale).
  - the short convolutions folded: every static conv [C, taps] with tap taps-1 the current token, its residual
    added to that tap and its output scale multiplied in - (w + e_last) / sqrt(taps + 1); a full layer's static
    taps (the checkpoint's [taps, C] by lag) flipped to that order, (w + e_last) / sqrt(2 taps + 1), and its
    per-token projection ([kv_heads * taps, n_embd], row = head * taps + lag, lag 0 the current token - not
    flipped) times 1 / sqrt(2 taps + 1). So a conv is y[t] = sum_j w[c, j] x[t - (taps-1) + j] (+ a full layer's
    sum_lag dyn[t, head(c) * taps + lag] x[t - lag]), nothing else.
  - experts NVFP4 (layers 0-39) as ggml's NVFP4, bit-exact, with each expert's scale (amax / denominator) as F32
    `.scale` beside gate, up and down; MXFP8 (40-47) as BF16, exact; w13's rows gate even, up odd.
  - F32: norms, the router and its bias, the clips' scales, rel_proj, the residual queries, the convs; the other
    matrices as the checkpoint holds them (BF16). The head and the embedding keep the padded vocabulary's rows.
  - metadata: llama.cpp's standard keys plus inkling2.d_rel, rel_extent, rel_extent_swa, shortconv_kernel,
    log_scaling_n_floor / alpha, unpadded_vocab_size, qk_clip_threshold, rel_clip_threshold, attn_res.block_size,
    attn_res_normalize_values, mlp_gate_softcap / up_softcap, value_norm, attention.sliding_window_pattern (true:
    a sliding layer). The drafter (mtp.*) and the vision tower are not written (the PR leaves them out).

The tokenizer: --vocab-from copies an existing inkling2 GGUF's tokenizer.* keys (unsloth's first shard is 12 MB of
metadata); without it, a checkpoint with no tokenizer.json (the tiny test model) gets placeholder tokens.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import struct
import sys

import numpy as np
import torch

import gguf

NVFP4 = gguf.GGMLQuantizationType.NVFP4
BF16, F32 = gguf.GGMLQuantizationType.BF16, gguf.GGMLQuantizationType.F32
SIZES = {NVFP4: (64, 36), BF16: (1, 2), F32: (1, 4)}
ARCH = "inkling2"


class Src:
    """the checkpoint's tensors by name: their bytes from the shards, their dtype and shape from the shards' headers
    (or --meta's JSON, {name: [dtype, shape]}, for a plan without the weights)"""

    def __init__(self, path, meta=None):
        self.path = path
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        idx = os.path.join(path, "model.safetensors.index.json")
        self.where = json.load(open(idx))["weight_map"]
        self.metas = json.load(open(meta)) if meta else {}
        self.files = {}

    def meta(self, k):
        if k not in self.metas:
            f = os.path.join(self.path, self.where[k])
            with open(f, "rb") as fh:
                n, = struct.unpack("<Q", fh.read(8))
                h = json.loads(fh.read(n))
            for name, v in h.items():
                if name != "__metadata__":
                    self.metas[name] = [v["dtype"], v["shape"]]
        return self.metas[k]

    def get(self, k):
        from safetensors import safe_open
        f = self.where[k]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.path, f), "pt")
        return self.files[f].get_tensor(k)


# ---- the names -------------------------------------------------------------------------------------------------
TOP = {"embed.weight": "token_embd.weight", "embed_norm.weight": "token_embd_norm.weight",
       "unembed.weight": "output.weight", "norm.weight": "output_norm.weight",
       "attn_res.output_query": "output_res_score.weight"}
NORMS = {"attn_norm": "attn_norm", "q_input_norm": "attn_q_in_norm", "k_input_norm": "attn_k_in_norm",
         "v_input_norm": "attn_v_in_norm", "mlp_norm": "ffn_norm"}
ATTN = {"wq_du.weight": "attn_q", "wk_dv.weight": "attn_k", "wv_dv.weight": "attn_v", "wr_du.weight": "attn_r",
        "wo_ud.weight": "attn_output", "gate_attn.weight": "attn_gate", "attn_sink_proj.weight": "attn_sink_proj",
        "q_l2_clip.scale": "attn_q_clip", "rel_l2_clip.scale": "attn_r_clip", "rel_logits_proj.proj": "attn_rel_proj"}
SKIP = re.compile(r"model\.(visual|vision|audio|mtp)\.|^mtp\.|mlp\.experts\.")   # (the experts: experts())


def plan_of(k):
    """a checkpoint tensor -> (GGUF name, how: plain | conv | conv_static | conv_dyn | f32) or None"""
    n = k.replace("model.llm.", "")
    if n in TOP:
        return TOP[n], ("f32" if n == "attn_res.output_query" else "plain")
    m = re.fullmatch(r"attn_res\.(attn|mlp)_queries\.(\d+)", n)
    if m:
        L = int(m[2]) + (m[1] == "attn")
        return f"blk.{L}.{'attn' if m[1] == 'attn' else 'ffn'}_res_score.weight", "f32"
    m = re.fullmatch(r"layers\.(\d+)\.(.+)", n)
    assert m, k
    L, rest = int(m[1]), m[2]
    root, _, leaf = rest.partition(".")
    if root in NORMS:
        return f"blk.{L}.{NORMS[root]}.{leaf}", "plain"
    if root in ("attn_sconv", "mlp_sconv"):
        return f"blk.{L}.shortconv_{'attn' if root == 'attn_sconv' else 'mlp'}.weight", "conv"
    if root == "attn":
        if leaf in ATTN:
            return f"blk.{L}.{ATTN[leaf]}.weight", "plain"
        kv, _, part = leaf.partition("_sconv.")
        how = {"weight": "conv", "static_w": "conv_static", "weight_proj": "conv_dyn"}[part]
        return f"blk.{L}.shortconv_{kv}{'_dyn' if part == 'weight_proj' else ''}.weight", how
    if root == "mlp" and leaf == "gate.weight":
        return f"blk.{L}.ffn_gate_inp.weight", "plain"
    if root == "mlp" and leaf == "gate.bias":
        return f"blk.{L}.exp_probs_b.bias", "f32"
    raise ValueError(f"unexpected tensor {k}")


# F32 whatever the checkpoint's dtype (llama.cpp's: 1-d tensors, norms, the router; the PR's forced ones)
def forced_f32(name):
    return (name.endswith(("_norm.weight", "attn_rel_proj.weight", "_clip.weight", "_res_score.weight", ".bias",
                           "ffn_gate_inp.weight")) or ".shortconv_" in name)


# ---- the folds -------------------------------------------------------------------------------------------------
def fold_static(w, dynamic):
    """[C, taps] (tap taps-1 the current token) -> (w + e_last) * (taps (x2 if dynamic) + 1)^-1/2"""
    w = w.float().clone()
    k = w.shape[-1]
    w[:, k - 1] += 1.0
    return w * (k * (2 if dynamic else 1) + 1) ** -0.5


def tensor_of(src, k, how, taps):
    t = src.get(k)
    if how == "conv":            # [C, 1, taps] -> [C, taps]
        return fold_static(t.squeeze(1), False)
    if how == "conv_static":     # [taps, C] by lag -> [C, taps], tap taps-1 the current token
        return fold_static(t.flip(0).T.contiguous(), True)
    if how == "conv_dyn":
        return t.float() * (2 * taps + 1) ** -0.5
    return t


def to_bytes(t, gtype):
    if gtype == F32:
        return t.float().contiguous().numpy()
    assert t.dtype == torch.bfloat16, t.dtype
    return t.contiguous().view(torch.int16).numpy()


# ---- the experts -----------------------------------------------------------------------------------------------
def nvfp4_rows(w, s):
    """checkpoint NVFP4 rows (w [R, K/2] u8: value 2i in the low nibble; s [R, K/16] e4m3) -> ggml NVFP4 [R, K/64*36]:
    a block of 64 = 4 UE4M3 scales then 32 bytes, sub-block j's value i in byte 4 + 8j + i's low nibble, i+8 high"""
    R = w.shape[0]
    d = s.view(torch.uint8)
    bad = ((d & 0x80) != 0) | ((d & 0x7F) == 0x7F)
    assert not bad.any(), f"{int(bad.sum())} block scales negative or NaN: not UE4M3"
    v = torch.stack([w & 0x0F, w >> 4], dim=-1).reshape(R, -1, 16)
    qs = (v[:, :, :8] | (v[:, :, 8:] << 4)).to(torch.uint8)
    nb = qs.shape[1] // 4
    return torch.cat([d.reshape(R, nb, 4), qs.reshape(R, nb, 32)], dim=-1).reshape(R, nb * 36)


def mxfp8_bf16(w, s):
    """checkpoint MXFP8 rows (w [R, K] e4m3, s [R, K/32] ue8m0) -> BF16, every value exact (else refused)"""
    e = s.view(torch.uint8).float()
    assert e.min() >= 3 and e.max() <= 246, "MXFP8 exponents past [3, 246]: not exact in BF16"
    f = w.float() * torch.exp2(e - 127).repeat_interleave(32, dim=-1)
    b = f.to(torch.bfloat16)
    assert torch.equal(b.float(), f), "an MXFP8 value not exact in BF16"
    return b


def expert_plan(src, L):
    """layer L's expert tensors: (name, logical shape [n_exp, rows, K], type)"""
    p = f"model.llm.layers.{L}.mlp.experts."
    dt, (n_exp, R2, kb) = src.meta(p + "w13_weight")
    _, (_, H, kb2) = src.meta(p + "w2_weight")
    nv = dt == "U8"
    t = NVFP4 if nv else BF16
    out = [(f"blk.{L}.ffn_gate_exps.weight", [n_exp, R2 // 2, kb * (2 if nv else 1)], t),
           (f"blk.{L}.ffn_up_exps.weight", [n_exp, R2 // 2, kb * (2 if nv else 1)], t),
           (f"blk.{L}.ffn_down_exps.weight", [n_exp, H, kb2 * (2 if nv else 1)], t)]
    if nv:
        out += [(f"blk.{L}.ffn_{nm}_exps.scale", [n_exp], F32) for nm in ("gate", "up", "down")]
    return out


def expert_data(src, L):
    p = f"model.llm.layers.{L}.mlp.experts."
    nv = src.meta(p + "w13_weight")[0] == "U8"
    out = {}
    for proj, parts in (("w13", (("gate", 0), ("up", 1))), ("w2", (("down", None),))):
        w, s = src.get(p + proj + "_weight"), src.get(p + proj + "_weight_scale")
        for nm, sel in parts:
            rows = []
            for e in range(w.shape[0]):
                we, se = (w[e], s[e]) if sel is None else (w[e, sel::2], s[e, sel::2])
                rows.append(nvfp4_rows(we, se).numpy().view(np.int8) if nv else
                            mxfp8_bf16(we, se).view(torch.int16).numpy())
            out[f"blk.{L}.ffn_{nm}_exps.weight"] = np.stack(rows)
        if nv:
            sc = (src.get(p + proj + "_weight_amax").float() /
                  src.get(p + proj + "_weight_outer_scale_denominator").float()).numpy().astype(np.float32)
            for nm, _ in parts:
                out[f"blk.{L}.ffn_{nm}_exps.scale"] = sc
    return out


# ---- the metadata ----------------------------------------------------------------------------------------------
def check_config(tc):
    """what the layout (and Maya's reference) implements; anything else refused, not written"""
    for a, b in (("swa_head_dim", "head_dim"), ("swa_num_attention_heads", "num_attention_heads"),
                 ("swa_num_key_value_heads", "num_key_value_heads")):
        assert tc.get(a, tc[b]) == tc[b], f"{a} {tc.get(a)} is not {b} {tc[b]}"
    for k in ("q_bias", "o_bias", "use_global_scale", "pre_norm_pre_affine", "final_norm_pre_affine",
              "n_shared_experts"):
        assert not tc.get(k), f"{k} set: not implemented"
    assert tc.get("use_value_norm_no_gamma", False) and tc.get("attn_res_normalize_values", False)


def sliding_layers(tc):
    n = tc["num_hidden_layers"]
    local = tc.get("local_layer_ids")
    if local is None:
        local = [i for i in range(n) if i % 6 != 5]
    return [i in set(local) for i in range(n)]


def meta(w, tc):
    w.add_block_count(tc["num_hidden_layers"])
    w.add_context_length(tc.get("model_max_length", tc.get("max_position_embeddings")))
    w.add_embedding_length(tc["hidden_size"])
    w.add_feed_forward_length(tc["intermediate_size"])
    w.add_head_count(tc["num_attention_heads"])
    w.add_head_count_kv(tc["num_key_value_heads"])
    w.add_layer_norm_rms_eps(tc["rms_norm_eps"])
    w.add_expert_count(tc["n_routed_experts"])
    w.add_expert_used_count(tc["num_experts_per_tok"])
    w.add_key_length(tc["head_dim"])
    w.add_value_length(tc["head_dim"])
    w.add_vocab_size(tc["vocab_size"])
    w.add_expert_feed_forward_length(tc["intermediate_size"])
    w.add_expert_weights_scale(tc["route_scale"])
    w.add_final_logit_softcapping(tc["final_logit_softcapping"])
    w.add_sliding_window(tc["sliding_window_size"])
    w.add_sliding_window_pattern(sliding_layers(tc))
    w.add_uint32(f"{ARCH}.d_rel", tc["d_rel"])
    w.add_uint32(f"{ARCH}.rel_extent", tc["rel_extent"])
    w.add_uint32(f"{ARCH}.rel_extent_swa", tc["sliding_window_size"])
    w.add_uint32(f"{ARCH}.shortconv_kernel", tc["sconv_kernel_size"])
    w.add_uint32(f"{ARCH}.log_scaling_n_floor", int(tc.get("log_scaling_n_floor") or 0))
    w.add_float32(f"{ARCH}.log_scaling_alpha", tc.get("log_scaling_alpha", 0.0))
    w.add_uint32(f"{ARCH}.unpadded_vocab_size", tc.get("unpadded_vocab_size") or tc["vocab_size"])
    w.add_float32(f"{ARCH}.qk_clip_threshold", tc["qk_l2_clip_threshold"])
    w.add_float32(f"{ARCH}.rel_clip_threshold", tc["rel_l2_clip_threshold"])
    w.add_attn_res_block_size(tc["n_layers_per_attn_res_block"])
    w.add_bool(f"{ARCH}.attn_res_normalize_values", bool(tc.get("attn_res_normalize_values", False)))
    w.add_float32(f"{ARCH}.mlp_gate_softcap", float(tc.get("mlp_gate_softcap", 0.0)))
    w.add_float32(f"{ARCH}.mlp_up_softcap", float(tc.get("mlp_up_softcap", 0.0)))
    w.add_bool(f"{ARCH}.value_norm", bool(tc.get("use_value_norm_no_gamma", False)))


def vocab(w, tc, vocab_from):
    if vocab_from:
        r = gguf.GGUFReader(vocab_from)
        for f in r.fields.values():
            if not f.name.startswith("tokenizer."):
                continue
            t, v = f.types[0], f.contents()   # (python values: gguf-py's writer refuses numpy's scalars)
            if t == gguf.GGUFValueType.ARRAY:
                w.add_key_value(f.name, v, t, sub_type=f.types[-1])
            else:
                w.add_key_value(f.name, v, t)
        return
    n = tc["vocab_size"]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("inkling")
    w.add_token_list([f"<t{i}>" for i in range(n)])
    w.add_token_types([gguf.TokenType.NORMAL] * n)
    w.add_token_merges([])
    w.add_eos_token_id(int(tc.get("eos_token_id", 0) or 0))


# ---- the plan and the file -------------------------------------------------------------------------------------
def plan(src, layers):
    """every tensor to write: (kind, key, GGUF name, logical shape (torch order), type, how)"""
    tc = src.cfg["text_config"]
    out = []
    for k in src.where:
        if SKIP.search(k):
            continue
        m = re.match(r"model\.llm\.layers\.(\d+)\.", k)
        if m and int(m[1]) not in layers:
            continue
        name, how = plan_of(k)
        m = re.match(r"blk\.(\d+)\.", name)
        if m and int(m[1]) not in layers:
            continue
        dt, sh = src.meta(k)
        sh = list(sh)
        if how == "conv":
            sh = [sh[0], sh[2]]
        elif how == "conv_static":
            sh = [sh[1], sh[0]]
        gtype = F32 if (forced_f32(name) or dt == "F32") else BF16
        assert gtype == F32 or dt == "BF16", (k, dt)
        out.append(("plain", k, name, sh, gtype, how))
    for L in layers:
        out += [("expert", L, name, sh, t, None) for name, sh, t in expert_plan(src, L)]
    _ = tc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--layers", default="")
    ap.add_argument("--split-gb", type=float, default=48.0)
    ap.add_argument("--vocab-from", default="")
    ap.add_argument("--plan", action="store_true", help="list the tensors (name, GGUF dims, type), write nothing")
    ap.add_argument("--meta", default="", help="the checkpoint's tensors' dtype and shape as JSON (--plan only)")
    ap.add_argument("--against", default="", help="--plan: a GGUF's tensor list as JSON [[name, dims, type]]")
    ap.add_argument("--kv-against", default="", help="--plan: a GGUF (its first shard): its inkling2.* keys")
    a = ap.parse_args()
    src = Src(a.src, a.meta or None)
    tc = src.cfg["text_config"]
    check_config(tc)
    layers = list(map(int, a.layers.split(","))) if a.layers else list(range(tc["num_hidden_layers"]))
    P = plan(src, layers)
    if a.plan:
        return compare(P, tc, a)
    w = gguf.GGUFWriter(a.out + ".gguf", ARCH, split_max_size=int(a.split_gb * 1e9))
    w.add_name("Inkling-2-Small")
    w.add_string("general.quantized_by", "Maya tools/inkling2_gguf.py: the original weights")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    meta(w, tc)
    vocab(w, tc, a.vocab_from)
    for kind, k, name, sh, t, how in P:
        bs, ts = SIZES[t]
        nbytes = int(np.prod(sh)) // bs * ts
        dt = {F32: np.float32, BF16: np.int16}.get(t, np.int8)
        # (the shape in values: a dtype other than uint8 keeps gguf-py from reading it as bytes)
        w.add_tensor_info(name, sh, np.dtype(dt), nbytes, raw_dtype=t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    taps = tc["sconv_kernel_size"]
    pending, done = {}, set()
    for kind, k, name, sh, t, how in P:
        if kind == "plain":
            data = to_bytes(tensor_of(src, k, how, taps), t)
        else:
            if k not in done:
                pending = expert_data(src, k)
                done.add(k)
            data = pending.pop(name)
        w.write_tensor_data(data)
        print(f"wrote {name} {sh} {t.name}", file=sys.stderr, flush=True)
    w.close()


def compare(P, tc, a):
    """the plan against another inkling2 GGUF's tensors (names, dims) and keys: every difference printed"""
    ours = {name: list(reversed(sh)) for _, _, name, sh, _, _ in P}
    f32 = {name for _, _, name, _, t, _ in P if t == F32}
    for name in sorted(ours):
        print(name, ours[name], "F32" if name in f32 else "")
    bad = 0
    if a.against:
        theirs = {n: d for n, d, _ in json.load(open(a.against))}
        t_types = {n: t for n, _, t in json.load(open(a.against))}
        for n in sorted(set(ours) | set(theirs)):
            if n not in theirs or n not in ours or ours[n] != theirs[n]:
                bad += 1
                print(f"DIFF {n}: ours {ours.get(n)} theirs {theirs.get(n)}")
        # (a quantized file keeps F32 where the converter wrote F32: ours F32 must be theirs F32 and back)
        for n in sorted(set(ours) & set(theirs)):
            if (n in f32) != (t_types[n] == 0):
                bad += 1
                print(f"TYPE {n}: ours {'F32' if n in f32 else 'BF16/NVFP4'} theirs type {t_types[n]}")
        print(f"tensors: ours {len(ours)}, theirs {len(theirs)}")
    if a.kv_against:
        import tempfile
        w = gguf.GGUFWriter(os.path.join(tempfile.mkdtemp(), "x.gguf"), ARCH)
        meta(w, tc)
        mine = {k: v.value for k, v in w.kv_data[0].items() if k.startswith(ARCH + ".")}
        r = gguf.GGUFReader(a.kv_against)
        for f in r.fields.values():
            if not f.name.startswith(ARCH + "."):
                continue
            v = f.contents()
            m = mine.pop(f.name, "MISSING")
            same = (np.float32(m) == np.float32(v) if isinstance(v, float) else   # (written as f32)
                    [bool(x) for x in m] == [bool(x) for x in v] if isinstance(v, list) else m == v)
            if not same:
                bad += 1
                print(f"KEY {f.name}: ours {m} theirs {v}")
        for k in mine:
            bad += 1
            print(f"KEY {k}: ours only")
    print("plan:", "the same" if not bad else f"{bad} differences")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main() or 0)

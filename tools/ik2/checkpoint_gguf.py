#!/usr/bin/env python3
"""tools/ik2/checkpoint_gguf.py - an Inkling-2-Mini checkpoint (thinkingmachines' inkling2_loop,
modelopt NVFP4 / MXFP8 experts) as an inkling2 GGUF in llama.cpp master's looped layout - the one
unsloth/Inkling-2-Mini-GGUF ships and maya-ik2 reads.

    python tools/ik2/checkpoint_gguf.py --src <checkpoint dir> --out <out.gguf> \
        --vocab-from <unsloth Inkling-2-Mini-UD-Q4_K_XL.gguf> [--name NAME]

The mapping (checkpoint -> GGUF), verified against unsloth's UD-Q4_K_XL tensor list:
  - embed/unembed/norm/embed_norm/attn_res.output_query -> token_embd/output/output_norm/token_embd_norm/
    output_res_query (the head and the embedding requantized Q8_0, the norms F32)
  - attn_res.(attn|mlp)_queries.(N) -> blk.{N(+1 for attn)}.(attn|ffn)_res_query.weight (F32)
  - middle_cycle_cross_pass_(kv_)queries -> loop_(kv_)res_query.weight (F32, direct)
  - layers.N.attn.{wq_du,wk_dv,wv_dv,wo_ud,wr_du,gate_attn} -> blk.N.attn_{q,k,v,output,rel,gate}.weight (Q8_0)
  - layers.N.attn.attn_sink_proj -> blk.N.attn_sink_proj.weight (BF16, direct)
  - layers.N.attn.{q,rel}_l2_clip.scale -> blk.N.attn_{q,rel}_clip.weight (F32)
  - layers.N.attn.rel_logits_proj.proj -> blk.N.attn_rel_bias.weight (F32, transposed [extent, d_rel])
  - layers.N.attn.{k,v}_sconv.weight -> blk.N.attn_{k,v}_conv.weight (F32, [C,1,taps] -> [C,taps])
  - layers.N.attn.{k,v}_sconv.static_w (full layers) -> blk.N.attn_{k,v}_conv.weight (F32, [taps,C] by lag
    flipped to [C,taps], tap taps-1 the current token; master's layout leaves the fold to the reader)
  - layers.N.attn.{k,v}_sconv.weight_proj (full layers) -> blk.N.attn_{k,v}_conv_proj.weight (BF16, direct,
    unscaled - the reader applies 1/sqrt(2 taps + 1))
  - layers.N.{attn,mlp}_sconv.weight -> blk.N.{attn,mlp}_post_conv.weight (F32, [C,1,taps] -> [C,taps])
  - layers.N.{q,k,v}_input_norm (full layers) / attn_norm (sliding) -> blk.N.attn_{q,k,v}_inp_norm / attn_norm
  - layers.N.mlp.gate.{weight,bias} -> blk.N.ffn_gate_inp.weight (F32) / blk.N.exp_probs_b.bias (F32)
  - layers.N.mlp_norm -> blk.N.ffn_norm.weight (F32)
  - layers.N.mlp.experts.w13/w2 (+ _scale, _amax, _outer_scale_denominator) ->
    blk.N.ffn_{gate_up,down}_exps.weight + .scale: layers 0-19 NVFP4 (the original bits, master's merged
    tensor with gate's rows first under one per-expert scale = amax / outer_scale_denominator); layers 20-23
    MXFP8 -> exact BF16 -> requantized Q8_0 (unsloth's choice), no .scale
  - layers.N.controllers.{attn,k,v,mlp}.{gain,forget,down}_weight (core layers 6-17) ->
    blk.N.loop_{attn,k,v,mlp}_{gain,forget,down}.weight (F32, direct)

Skipped: the vision tower (model.visual.*, unsloth's mmproj covers pictures) and the drafter (markov_head.*,
mask_embed, the bare norm.weight - the engine takes mtp.safetensors by --draft).

The tensor shapes were read from the checkpoint's safetensors headers and the target layout from
unsloth/Inkling-2-Mini-GGUF UD-Q4_K_XL; tools/ik2/checkpoint_gguf_validate.py checks a conversion against
the checkpoint and unsloth's file (plan, NVFP4 bits, F32/BF16 exactness, Q8_0 bounds, expert scales).
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
from safetensors import safe_open

import gguf

ARCH = "inkling2"
F32 = gguf.GGMLQuantizationType.F32
BF16 = gguf.GGMLQuantizationType.BF16
Q8_0 = gguf.GGMLQuantizationType.Q8_0
NVFP4 = gguf.GGMLQuantizationType.NVFP4


class Src:
    """the checkpoint's tensors by name: their bytes from the shards, their dtype and shape from the shards' headers"""

    def __init__(self, path):
        self.path = path
        self.cfg = json.load(open(os.path.join(path, "config.json")))
        idx = os.path.join(path, "model.safetensors.index.json")
        self.where = json.load(open(idx))["weight_map"]
        self.metas = {}
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
        f = self.where[k]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.path, f), "pt")
        return self.files[f].get_tensor(k)


# ---- the names ---------------------------------------------------------------------------------------------------
SKIP = re.compile(r"model\.visual\.|markov_head\.|^mask_embed$|^norm\.weight$|mlp\.experts\.")

TOP = {
    "embed.weight": ("token_embd.weight", "q8"),
    "embed_norm.weight": ("token_embd_norm.weight", "f32"),
    "unembed.weight": ("output.weight", "q8"),
    "norm.weight": ("output_norm.weight", "f32"),
    "attn_res.output_query": ("output_res_query.weight", "f32"),
    "middle_cycle_cross_pass_queries": ("loop_res_query.weight", "f32"),
    "middle_cycle_cross_pass_kv_queries": ("loop_kv_res_query.weight", "f32"),
}

ATTN = {
    "wq_du.weight": "attn_q.weight", "wk_dv.weight": "attn_k.weight", "wv_dv.weight": "attn_v.weight",
    "wr_du.weight": "attn_rel.weight", "wo_ud.weight": "attn_output.weight", "gate_attn.weight": "attn_gate.weight",
    "attn_sink_proj.weight": "attn_sink_proj.weight",
    "q_l2_clip.scale": "attn_q_clip.weight", "rel_l2_clip.scale": "attn_rel_clip.weight",
    "rel_logits_proj.proj": "attn_rel_bias.weight",
}


def plan_of(k):
    """a checkpoint tensor -> (GGUF name, kind) where kind: f32 | f32_t (transpose) | conv | conv_static |
    conv_dyn | bf16 | q8 | expert_gate_up | expert_down | scale_gate_up | scale_down"""
    n = k.replace("model.llm.", "")
    if n in TOP:
        return TOP[n][0], TOP[n][1]
    m = re.fullmatch(r"attn_res\.(attn|mlp)_queries\.(\d+)", n)
    if m:
        L = int(m[2]) + (m[1] == "attn")
        return f"blk.{L}.{'attn' if m[1] == 'attn' else 'ffn'}_res_query.weight", "f32"
    m = re.fullmatch(r"layers\.(\d+)\.(.+)", n)
    assert m, k
    L, rest = int(m[1]), m[2]
    root, _, leaf = rest.partition(".")
    if root == "attn_norm":
        return f"blk.{L}.attn_norm.weight", "f32"
    if root in ("q_input_norm", "k_input_norm", "v_input_norm"):
        return f"blk.{L}.attn_{root[0]}_inp_norm.weight", "f32"
    if root == "mlp_norm":
        return f"blk.{L}.ffn_norm.weight", "f32"
    if root == "attn":
        if leaf in ATTN:
            if leaf == "attn_sink_proj.weight":
                kind = "bf16"
            elif leaf == "rel_logits_proj.proj":
                kind = "f32_t"
            elif leaf.endswith("_clip.scale"):
                kind = "f32"
            else:
                kind = "q8"
            return f"blk.{L}.{ATTN[leaf]}", kind
        kv, _, part = leaf.partition("_sconv.")
        if part not in ("weight", "static_w", "weight_proj"):
            raise ValueError(f"unexpected short-conv tensor {k}")
        if part == "weight":
            return f"blk.{L}.attn_{kv}_conv.weight", "conv"
        if part == "static_w":
            return f"blk.{L}.attn_{kv}_conv.weight", "conv_static"
        return f"blk.{L}.attn_{kv}_conv_proj.weight", "bf16"
    if root == "attn_sconv":
        return f"blk.{L}.attn_post_conv.weight", "conv"
    if root == "mlp_sconv":
        return f"blk.{L}.ffn_post_conv.weight", "conv"
    if root == "mlp":
        if leaf == "gate.weight":
            return f"blk.{L}.ffn_gate_inp.weight", "f32"
        if leaf == "gate.bias":
            return f"blk.{L}.exp_probs_b.bias", "f32"
    if root == "controllers":
        mod, _, leaf2 = leaf.partition(".")
        if not leaf2.endswith("_weight"):
            raise ValueError(f"unexpected controller tensor {k}")
        return f"blk.{L}.loop_{mod}_{leaf2[:-len('_weight')]}.weight", "f32"
    raise ValueError(f"unexpected tensor {k}")


# ---- the folds and the dtypes ------------------------------------------------------------------------------------
def to_f32(t):
    return t.float().contiguous()


def q8_0(t):
    """[R, K] float -> Q8_0 blocks [R, K/32, 34] int8 (ggml's quantize_row_q8_0: d = amax/127 as fp16, then
    32 int8 values round(x/d))"""
    w = t.float().reshape(t.shape[0], -1, 32)
    amax = w.abs().amax(-1)
    d = amax / 127.0
    d = torch.where(d > 0, d, torch.ones_like(d))
    d16 = d.to(torch.float16)
    q = torch.round(w / d16.float().unsqueeze(-1)).clamp(-128, 127).to(torch.int8)
    out = torch.cat([d16.reshape(*d16.shape, 1).view(torch.int8), q], dim=-1)
    return out.reshape(t.shape[0], -1).contiguous().numpy()


def nvfp4_rows(w, s):
    """checkpoint NVFP4 rows (w [R, K/2] u8: value 2i in the low nibble; s [R, K/16] e4m3) -> ggml NVFP4
    [R, K/64*36]: a block of 64 = 4 UE4M3 scales then 32 bytes, sub-block j's value i in byte 4 + 8j + i's
    low nibble, i+8 high"""
    R = w.shape[0]
    assert w.shape[1] % 32 == 0, f"row width {w.shape[1]} bytes is not a whole number of 64-value blocks"
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


def bf16_bytes(t):
    if t.dtype != torch.bfloat16:
        raise ValueError(f"{t.dtype} is not BF16 (a derivative changed a matrix's dtype?)")
    return t.contiguous().view(torch.int16).numpy()


# ---- the experts -------------------------------------------------------------------------------------------------
def expert_data(src, L, gate_up_order):
    """layer L's experts: {gguf name: numpy data}. gate_up_order: 'interleaved' (w13's rows gate even, up
    odd) or 'stacked' (gate's rows first)"""
    p = f"model.llm.layers.{L}.mlp.experts."
    dt13, sh13 = src.meta(p + "w13_weight")
    dt2, _ = src.meta(p + "w2_weight")
    dt13s, _ = src.meta(p + "w13_weight_scale")
    dt2s, _ = src.meta(p + "w2_weight_scale")
    if dt13 == "U8":
        kind = "nvfp4"
        if dt13s != "F8_E4M3" or dt2 != "U8" or dt2s != "F8_E4M3":
            raise ValueError(f"layer {L}: NVFP4 weights (U8) expect F8_E4M3 scales, got w13 {dt13s}, "
                             f"w2 {dt2}/{dt2s} - a quantization this tool does not know")
    elif dt13 == "F8_E4M3":
        kind = "mxfp8"
        if dt13s != "U8" or dt2 != "F8_E4M3" or dt2s != "U8":
            raise ValueError(f"layer {L}: MXFP8 weights (F8_E4M3) expect U8 scales, got w13 {dt13s}, "
                             f"w2 {dt2}/{dt2s} - a quantization this tool does not know")
    else:
        raise ValueError(f"layer {L}: expert weights are {dt13}, not U8 (NVFP4) or F8_E4M3 (MXFP8) - "
                         "a quantization this tool does not know")
    out = {}
    if kind == "nvfp4":
        w13, s13 = src.get(p + "w13_weight"), src.get(p + "w13_weight_scale")
        w2, s2 = src.get(p + "w2_weight"), src.get(p + "w2_weight_scale")
        if gate_up_order == "interleaved":
            gate13, up13 = w13[:, 0::2, :], w13[:, 1::2, :]
            gsc, usc = s13[:, 0::2, :], s13[:, 1::2, :]
        else:
            F = w13.shape[1] // 2
            gate13, up13 = w13[:, :F, :], w13[:, F:, :]
            gsc, usc = s13[:, :F, :], s13[:, F:, :]
        merged_w = torch.cat([gate13, up13], dim=1)      # [E, R_total, K/2] master's: gate's rows first
        merged_s = torch.cat([gsc, usc], dim=1)
        rows = [nvfp4_rows(merged_w[e], merged_s[e]).numpy().view(np.int8) for e in range(merged_w.shape[0])]
        out["blk.%d.ffn_gate_up_exps.weight" % L] = np.stack(rows)
        sc = (src.get(p + "w13_weight_amax").float() /
              src.get(p + "w13_weight_outer_scale_denominator").float()).numpy().astype(np.float32)
        out["blk.%d.ffn_gate_up_exps.scale" % L] = sc
        rows = [nvfp4_rows(w2[e], s2[e]).numpy().view(np.int8) for e in range(w2.shape[0])]
        out["blk.%d.ffn_down_exps.weight" % L] = np.stack(rows)
        sc = (src.get(p + "w2_weight_amax").float() /
              src.get(p + "w2_weight_outer_scale_denominator").float()).numpy().astype(np.float32)
        out["blk.%d.ffn_down_exps.scale" % L] = sc
    else:                      # MXFP8 -> exact BF16 -> Q8_0 requant (unsloth's choice), no .scale
        w13, s13 = src.get(p + "w13_weight"), src.get(p + "w13_weight_scale")
        w2, s2 = src.get(p + "w2_weight"), src.get(p + "w2_weight_scale")
        F = w13.shape[1] // 2
        if gate_up_order == "interleaved":
            gsel, usel = (slice(None), slice(None, None, 2), slice(None)), (slice(None), slice(1, None, 2), slice(None))
        else:
            gsel, usel = (slice(None), slice(None, F), slice(None)), (slice(None), slice(F, None), slice(None))
        merged = torch.cat([mxfp8_bf16(w13[gsel], s13[gsel]), mxfp8_bf16(w13[usel], s13[usel])], dim=1)
        out["blk.%d.ffn_gate_up_exps.weight" % L] = q8_0(merged)
        out["blk.%d.ffn_down_exps.weight" % L] = q8_0(mxfp8_bf16(w2, s2))
    return out


# ---- the metadata ------------------------------------------------------------------------------------------------
def check_config(tc):
    for a, b in (("swa_head_dim", "head_dim"), ("swa_num_attention_heads", "num_attention_heads"),
                 ("swa_num_key_value_heads", "num_key_value_heads")):
        if tc.get(a, tc[b]) != tc[b]:
            raise ValueError(f"{a} {tc.get(a)} is not {b} {tc[b]}")
    for k in ("q_bias", "o_bias", "use_global_scale", "pre_norm_pre_affine", "final_norm_pre_affine",
              "n_shared_experts"):
        if tc.get(k):
            raise ValueError(f"{k} set: not implemented")
    if not (tc.get("use_value_norm_no_gamma", False) and tc.get("attn_res_normalize_values", False)):
        raise ValueError("value_norm / attn_res_normalize_values off: not implemented")
    lc = tc["loop_config"]
    n_layers = tc["num_hidden_layers"]
    if lc["num_passes"] <= 1:
        raise ValueError("num_passes 1: this converter is for the looped Inkling-2-Mini; "
                         "tools/inkling2_gguf.py converts the non-looped Inkling-2-Small")
    if not (0 < lc["core_start"] < lc["core_end"] <= n_layers):
        raise ValueError(f"loop core {lc['core_start']}..{lc['core_end']} is not inside 1..{n_layers}")
    block = tc["n_layers_per_attn_res_block"]
    if lc["core_start"] % block or (lc["core_end"] - lc["core_start"]) % block:
        raise ValueError(f"the loop core {lc['core_start']}..{lc['core_end']} is not a multiple "
                         f"of the attention-residual block size {block}")


def visit_layer(v, core0, core1, n_pass):
    """visit v -> its physical layer"""
    n_core = core1 - core0
    if v < core1:
        return v
    if v < core1 + (n_pass - 1) * n_core:
        return core0 + (v - core1) % n_core
    return v - (n_pass - 1) * n_core


def meta(w, tc):
    lc = tc["loop_config"]
    n_pass, core0, core1 = lc["num_passes"], lc["core_start"], lc["core_end"]
    n_visits = tc["num_hidden_layers"] + (n_pass - 1) * (core1 - core0)
    local = set(tc["local_layer_ids"])
    w.add_block_count(n_visits)
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
    w.add_sliding_window_pattern([visit_layer(v, core0, core1, n_pass) in local for v in range(n_visits)])
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
    # the loop
    w.add_uint32(f"{ARCH}.num_loops", n_pass)
    w.add_uint32(f"{ARCH}.loop.core_start", core0)
    w.add_uint32(f"{ARCH}.loop.core_end", core1)
    w.add_uint32(f"{ARCH}.loop.gate_rank", lc["gate_rank"])
    w.add_uint32(f"{ARCH}.attention.key_length_swa", tc["head_dim"])
    w.add_uint32(f"{ARCH}.attention.value_length_swa", tc["head_dim"])
    w.add_uint32(f"{ARCH}.rope.dimension_count", 0)


def vocab(w, vocab_from):
    r = gguf.GGUFReader(vocab_from)
    arch = r.get_field("general.architecture")
    if arch is None or bytes(arch.parts[arch.data[0]]).decode() != ARCH:
        raise ValueError(f"{vocab_from} is not an {ARCH} GGUF: its tokenizer may not match")
    for f in r.fields.values():
        if not f.name.startswith("tokenizer."):
            continue
        t, v = f.types[0], f.contents()   # (python values: gguf-py's writer refuses numpy's scalars)
        if t == gguf.GGUFValueType.ARRAY:
            w.add_key_value(f.name, v, t, sub_type=f.types[-1])
        else:
            w.add_key_value(f.name, v, t)


# ---- the plan and the file ---------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="the checkpoint folder")
    ap.add_argument("--out", default="", help="the output GGUF path")
    ap.add_argument("--vocab-from", default="", help="an inkling2 GGUF to copy the tokenizer from")
    ap.add_argument("--gate-up-order", choices=("interleaved", "stacked"), default="interleaved",
                    help="how the checkpoint's w13 rows hold gate and up (bit-exact check decides)")
    ap.add_argument("--split-gb", type=float, default=48.0,
                    help="split the output into shards of this many GB (one file up to it)")
    ap.add_argument("--plan", action="store_true", help="list the tensors (name, GGUF dims, type), write nothing")
    ap.add_argument("--name", default="", help="general.name of the GGUF (default: the checkpoint folder's name)")
    a = ap.parse_args()
    src = Src(a.src)
    tc = src.cfg["text_config"]
    check_config(tc)
    lc = tc["loop_config"]
    n_pass, core0, core1 = lc["num_passes"], lc["core_start"], lc["core_end"]
    n_core = core1 - core0
    n_visits = tc["num_hidden_layers"] + (n_pass - 1) * n_core
    n_layers = tc["num_hidden_layers"]

    P = []   # (key, gguf name, torch shape, dtype kind)
    for k in src.where:
        if SKIP.search(k):
            continue
        name, kind = plan_of(k)
        dt, sh = src.meta(k)
        sh = list(sh)
        if kind == "conv":
            sh = [sh[0], sh[2]]           # [C, 1, taps] -> [C, taps]
        elif kind == "conv_static":
            sh = [sh[1], sh[0]]           # [taps, C] -> [C, taps]
        elif kind == "f32_t":
            sh = [sh[1], sh[0]]           # [d_rel, extent] -> [extent, d_rel]
        P.append((k, name, sh, kind))
    for L in range(n_layers):
        p = f"model.llm.layers.{L}.mlp.experts."
        dt13, sh13 = src.meta(p + "w13_weight")
        _, (_, H, kb2) = src.meta(p + "w2_weight")
        E = tc["n_routed_experts"]
        if dt13 == "U8":
            P.append((None, f"blk.{L}.ffn_gate_up_exps.weight", [E, sh13[1], sh13[2] * 2], "nvfp4_gu"))
            P.append((None, f"blk.{L}.ffn_gate_up_exps.scale", [E], "scale_gu"))
            P.append((None, f"blk.{L}.ffn_down_exps.weight", [E, H, kb2 * 2], "nvfp4_down"))
            P.append((None, f"blk.{L}.ffn_down_exps.scale", [E], "scale_down"))
        else:
            P.append((None, f"blk.{L}.ffn_gate_up_exps.weight", [E, sh13[1], sh13[2]], "q8_gu"))
            P.append((None, f"blk.{L}.ffn_down_exps.weight", [E, H, kb2], "q8_down"))

    if a.plan:
        for k, name, sh, kind in P:
            print(name, list(reversed(sh)), kind)
        print(f"tensors: {len(P)}")
        return
    if not a.out:
        raise SystemExit("--out is required (or --plan to list the tensors without writing)")

    KIND_GGUF = {"f32": F32, "f32_t": F32, "conv": F32, "conv_static": F32, "bf16": BF16,
                 "q8": Q8_0, "nvfp4_gu": NVFP4, "nvfp4_down": NVFP4, "q8_gu": Q8_0,
                 "q8_down": Q8_0, "scale_gu": F32, "scale_down": F32}
    SIZES = {F32: (1, 4), BF16: (1, 2), Q8_0: (32, 34), NVFP4: (64, 36)}
    w = gguf.GGUFWriter(a.out, ARCH, split_max_size=int(a.split_gb * 1e9))
    name = a.name or os.path.basename(os.path.normpath(a.src))
    w.add_name(name)
    w.add_string("general.quantized_by", "tools/ik2/checkpoint_gguf.py: " + name + ", the original weights")
    # (general.file_type: the --vocab-from file's - the same per-tensor types; a mixed file has no single one)
    if a.vocab_from:
        ft = gguf.GGUFReader(a.vocab_from).fields.get("general.file_type")
        if ft is not None:
            w.add_file_type(int(ft.contents()))
    meta(w, tc)
    vocab(w, a.vocab_from)
    for k, name, sh, kind in P:
        gtype = KIND_GGUF[kind]
        bs, ts = SIZES[gtype]
        nbytes = int(np.prod(sh)) // bs * ts
        dt = np.float32 if gtype == F32 else (np.int16 if gtype == BF16 else np.int8)
        w.add_tensor_info(name, sh, np.dtype(dt), nbytes, raw_dtype=gtype)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    taps = tc["sconv_kernel_size"]
    pending, done = {}, set()
    for k, name, sh, kind in P:
        if k is not None:
            t = src.get(k)
            if kind == "f32":
                data = to_f32(t).numpy()
            elif kind == "f32_t":
                data = to_f32(t.transpose(0, 1)).numpy()
            elif kind == "conv":
                data = to_f32(t.squeeze(1)).numpy()
            elif kind == "conv_static":
                data = to_f32(t.flip(0).T.contiguous()).numpy()
            elif kind == "bf16":
                data = bf16_bytes(t)
            elif kind == "q8":
                data = q8_0(t)
            else:
                raise ValueError(kind)
        else:
            L = int(re.match(r"blk\.(\d+)\.", name).group(1))  # the name always starts blk.N
            if L not in done:
                pending = expert_data(src, L, a.gate_up_order)
                done.add(L)
            data = pending.pop(name)
        w.write_tensor_data(data)
        print(f"wrote {name} {sh} {kind}", file=sys.stderr, flush=True)
    w.close()
    print(f"done: {a.out}")


if __name__ == "__main__":
    main()

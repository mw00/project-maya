"""tools/ds4/pack_gguf.py - the engine's pack for a DeepSeek-V4.1 GGUF (unsloth's DeepSeek-V4.1-Flash-1017 UD-Q8_K_XL
and the like: dense BF16, routed experts MXFP4, Engram tables MXFP8 or Q6_K), beside the model's files (no weights
copied).

    python tools/ds4/pack_gguf.py --model ~/models/DeepSeek-V4.1-Flash-1017-GGUF/UD-Q8_K_XL

writes <model>/maya-ds4/ in pack.py's format, so the engine reads both:
  ds4.txt       the geometry under config.json's names (the GGUF's deepseek41.* keys translated), model_dir, format gguf
  tensors.txt   every tensor under the checkpoint's names: `file <i> <name>` lines, then
                `t <name> <dtype> <file> <offset> <bytes> <dims...>` - dims outermost first (torch order: a matrix is
                [out][in]), dtype BF16 / F32 / MXFP4 / MXFP8 (ggml's block layouts, 32 values a block); the routed
                experts as one tensor a layer and role ([n_exp][out][in])
  engram.bin    the Engram hash tables, from the GGUF's metadata (pack.py's layout; no tokenizers package needed)
  tokenizer/    the server's tokenizer from the GGUF's vocabulary and merges (pre "deepseek3"), and its chat template

A GGUF has no DSpark drafter: drafter_pack.py makes it from DeepSeek's checkpoint into ../dspark/, and the pack takes it
from there when it is present.  Nor a vision tower: vision_pack.py makes it from DeepSeek's checkpoint into ../vision/
(--vision: another folder), and the pack takes it and its geometry (vision.<key>, image_token_id) from there when it is
present; without it the engine reads text only.

A quant that leaves the two Engram tables out (v1.1.16's Maya-S; Maya's quants hold them since) takes them from a set
of GGUF files that holds them, beside it: --engram, else an Engram-architecture file in its own folder (ISTA-DASLab's
GSQ-RCO sets ship DeepSeek-V4.1-Flash-Engram-*.gguf beside the shards), else ../Engram/ (unsloth's MXFP8 shard of
them, where setups before v1.1.17 put it) when it is whole.

The ISTA-DASLab GSQ-RCO quants (general.tags 'gsq-rco': dense Q5_K / Q8_0, routed experts IQ1_M / IQ2_XXS /
IQ2_XS) spell several things their own way, and the pack reads all of it as it is - no requantizing, no folding:
the Engram's gate as separate q and k (the runtime multiplies them at load), its kv as engram_wkv, its hash
parameters as integer tensors of the first shard (engram_token_map I32, engram_primes / engram_offsets /
engram_multipliers I64) instead of metadata, the candidate / source-layer / dspark keys under their flat names, and
attn_output_a as one 3-D [groups][o_lora][in] tensor.  That recovery is adapted from Strata 96ca4383 (MIT; see
LICENSE.strata-gsq).  A pack is only written when every piece is found and consistent: a missing or ill-sized hash
parameter, table or tensor is an error that names it, never a silent gap.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import re
import struct
import sys

TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 8: ("Q8_0", 32, 34), 30: ("BF16", 1, 2), 39: ("MXFP4", 32, 17),
         67: ("MXFP8", 32, 33),          # ggml type -> (name, values a block, bytes a block); 67: unsloth's fork
         43: ("MXFP8", 32, 33),          # llama.cpp's own id (unsloth's GGUFs from 2026-10-19 on)
         # the low-bit expert types (the Maya quants' experts; the engine's ds4_iq kernels)
         10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
         14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
         19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
         23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56)}
# the Engram set a quant without the tables takes them from by default, beside it (or inside its folder): unsloth's
# MXFP8 shard of them, where setups before v1.1.17 put it for Maya-S (--engram: any other)
ENGRAM_SETS = ("Engram",)
# the Engram's hash parameters carried as integer tensors of the first shard (the ISTA-DASLab GSQ sets, which leave
# them out of the metadata): name -> (the engram.* key, the struct field, its byte size)
ENGRAM_HASH_TENSORS: dict[str, tuple[str, str, int, int]] = {
    "engram_token_map.weight": ("token_map", "i", 4, 26),
    "engram_primes.weight": ("primes", "q", 8, 27),
    "engram_offsets.weight": ("offsets", "q", 8, 27),
    "engram_multipliers.weight": ("multipliers", "q", 8, 27),
}


class Gguf:
    """One GGUF shard's metadata and tensor infos (offsets absolute in the file)."""

    def __init__(self, path: pathlib.Path):
        self.path = path
        with path.open("rb") as f:
            self.f = f
            magic = f.read(4)
            if magic != b"GGUF":
                raise SystemExit("%s: not a GGUF file" % path)
            self.version, nt, nkv = struct.unpack("<IQQ", f.read(20))
            self.kv = {}
            for _ in range(nkv):
                k = self._str()
                self.kv[k] = self._val(struct.unpack("<I", f.read(4))[0])
            self.tensors = []
            for _ in range(nt):
                name = self._str()
                nd = struct.unpack("<I", f.read(4))[0]
                dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
                typ, off = struct.unpack("<IQ", f.read(12))
                self.tensors.append((name, typ, dims, off))
            align = int(self.kv.get("general.alignment", 32))
            pos = f.tell()
            self.data_start = (pos + align - 1) // align * align
        del self.f

    def _str(self):
        n = struct.unpack("<Q", self.f.read(8))[0]
        return self.f.read(n).decode("utf-8", "replace")

    def _val(self, t):
        fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
        if t in fmt:
            n = struct.calcsize(fmt[t])
            return struct.unpack("<" + fmt[t], self.f.read(n))[0]
        if t == 8:
            return self._str()
        if t == 9:
            at, n = struct.unpack("<IQ", self.f.read(12))
            if at in fmt:
                n_b = struct.calcsize(fmt[at])
                return list(struct.unpack("<%d%s" % (n, fmt[at]), self.f.read(n * n_b)))
            return [self._val(at) for _ in range(n)]
        raise SystemExit("%s: metadata type %d" % (self.path, t))


# GGUF name (blk.N. stripped) -> the checkpoint's name under layers.N. (pack.py / the engine)
LAYER = {
    "attn_norm.weight": "attn_norm.weight",
    "attn_sinks.weight": "attn.attn_sink",
    "attn_q_a.weight": "attn.wq_a.weight",
    "attn_q_a_norm.weight": "attn.q_norm.weight",
    "attn_q_b.weight": "attn.wq_b.weight",
    "attn_kv.weight": "attn.wkv.weight",
    "attn_kv_a_norm.weight": "attn.kv_norm.weight",
    "attn_output_a.weight": "attn.wo_a.slice",        # one entry a group (the engine's slices)
    "attn_output_b.weight": "attn.wo_b.weight",
    "attn_compressor_kv.weight": "attn.compressor.wkv.weight",
    "attn_compressor_gate.weight": "attn.compressor.wgate.weight",
    "attn_compressor_norm.weight": "attn.compressor.norm.weight",
    "indexer.attn_q_b.weight": "attn.indexer.wq_b.weight",
    "indexer.attn_k.weight": "attn.indexer.wk.weight",
    "indexer.k_norm.weight": "attn.indexer.k_norm.weight",
    "indexer.proj.weight": "attn.indexer.weights_proj.weight",
    "hc_attn_fn.weight": "hc_attn_fn",
    "hc_attn_base.weight": "hc_attn_base",
    "hc_attn_scale.weight": "hc_attn_scale",
    "hc_ffn_fn.weight": "hc_ffn_fn",
    "hc_ffn_base.weight": "hc_ffn_base",
    "hc_ffn_scale.weight": "hc_ffn_scale",
    "ffn_norm.weight": "ffn_norm.weight",
    "ffn_gate_inp.weight": "ffn.gate.weight",
    "exp_probs_b.bias": "ffn.gate.bias",
    "exp_probs_b_vl.bias": "ffn.gate.bias_vl",
    "ffn_gate_shexp.weight": "ffn.shared_experts.w1.weight",
    "ffn_up_shexp.weight": "ffn.shared_experts.w3.weight",
    "ffn_down_shexp.weight": "ffn.shared_experts.w2.weight",
    "ffn_gate_exps.weight": "ffn.experts.w1",             # [n_exp][ff][dim]: one entry an expert
    "ffn_up_exps.weight": "ffn.experts.w3",
    "ffn_down_exps.weight": "ffn.experts.w2",             # [n_exp][dim][ff]
    "engram_embd.weight": "engram.embed.weight",
    "engram_kv.weight": "engram.wkv.weight",
    "engram_gate.weight": "engram.gate.weight",          # q_weight * k_weight, folded by the converter
    # the ISTA-DASLab GSQ sets keep q and k apart (the runtime multiplies them at load) and the kv as engram_wkv
    "engram_wkv.weight": "engram.wkv.weight",
    "engram_q.weight": "engram.q_weight",
    "engram_k.weight": "engram.k_weight",
}
GLOBAL = {"token_embd.weight": "embed.weight", "output_norm.weight": "norm.weight", "output.weight": "head.weight"}

# the same quantity under another key (the ISTA-DASLab GSQ sets write the flat spelling where unsloth's GGUFs write
# the dotted one): geometry's key -> its alternative spellings in the GGUF (the GGUF's own key wins when it is there)
ALIASES = {
    "attention.kv_source_layers": ("kv_source_layer_ids",),
    "attention.indexer.source_layers": ("index_source_layer_ids",),
    "attention.indexer.candidate_source_layer": ("attention.candidate_source_layer_id",),
    "attention.indexer.candidate_top_k_blocks": ("attention.candidate_topk_blocks",),
    "attention.indexer.candidate_block_size": ("attention.candidate_block_size",),
    "dspark.target_layers": ("dspark_target_layer_ids",),
    "dspark.block_size": ("dspark_block_size",),
    "dspark.noise_token_id": ("dspark_noise_token_id",),
    "dspark.markov_rank": ("dspark_markov_rank",),
    "dspark.expert_count": ("dspark_n_routed_experts",),
    "dspark.expert_used_count": ("dspark_num_experts_per_tok",),
    "engram.layers": ("engram_layer_ids",),
}


def geometry(kv: dict, vocab: int) -> dict:
    """config.json's text_config names (what the engine reads) from the GGUF's deepseek41.* keys."""
    a = kv["general.architecture"]

    kv = dict(kv)
    for key, alternatives in ALIASES.items():
        canonical = f"{a}.{key}"
        if kv.get(canonical) is not None:
            continue
        for alternative in alternatives:
            value = kv.get(f"{a}.{alternative}")
            if value is not None:
                kv[canonical] = value
                break
    g = lambda k, d=None: kv.get(f"{a}.{k}", d)  # noqa: E731
    nl = g("block_count")
    ratios = list(g("attention.compress_ratios", []))
    out = {
        "vocab_size": vocab,
        "hidden_size": g("embedding_length"),
        "num_hidden_layers": nl,
        "num_attention_heads": g("attention.head_count"),
        "head_dim": g("attention.key_length"),
        "qk_rope_head_dim": g("rope.dimension_count"),
        "q_lora_rank": g("attention.q_lora_rank"),
        "o_lora_rank": g("attention.output_lora_rank"),
        "o_groups": g("attention.output_group_count"),
        "n_routed_experts": g("expert_count"),
        "num_experts_per_tok": g("expert_used_count"),
        "moe_intermediate_size": g("expert_feed_forward_length"),
        "sliding_window": g("attention.sliding_window"),
        "hc_mult": g("hyper_connection.count"),
        "hc_sinkhorn_iters": g("hyper_connection.sinkhorn_iterations"),
        "hc_eps": g("hyper_connection.epsilon"),
        "index_n_heads": g("attention.indexer.head_count"),
        "index_head_dim": g("attention.indexer.key_length"),
        "index_topk": g("attention.indexer.top_k"),
        "candidate_source_layer_id": g("attention.indexer.candidate_source_layer"),
        "candidate_topk_blocks": g("attention.indexer.candidate_top_k_blocks"),
        "candidate_block_size": g("attention.indexer.candidate_block_size"),
        "rope_scaling.original_max_position_embeddings": g("rope.scaling.original_context_length"),
        "rope_scaling.factor": g("rope.scaling.factor"),
        "rope_scaling.beta_fast": g("rope.scaling.yarn_beta_fast"),
        "rope_scaling.beta_slow": g("rope.scaling.yarn_beta_slow"),
        "eos_token_id": kv.get("tokenizer.ggml.eos_token_id", 1),
        "rms_norm_eps": g("attention.layer_norm_rms_epsilon"),
        "routed_scaling_factor": g("expert_weights_scale"),
        "swiglu_limit": (g("swiglu_clamp_exp") or [0.0])[0],
        "rope_theta": g("rope.freq_base"),
        "compress_rope_theta": g("attention.compress_rope_freq_base"),
        "compress_ratios": ratios,
        "kv_source_layer_ids": list(g("attention.kv_source_layers", [])),
        "index_source_layer_ids": list(g("attention.indexer.source_layers", [])),
        "engram_layer_ids": list(g("engram.layers", [])),
        "max_position_embeddings": g("context_length"),
        "dspark_block_size": g("dspark.block_size", 0),
        "dspark_noise_token_id": g("dspark.noise_token_id", 0),
        "dspark_markov_rank": g("dspark.markov_rank", 0),
        "dspark_n_routed_experts": g("dspark.expert_count", 0),
        "dspark_num_experts_per_tok": g("dspark.expert_used_count", 0),
        "dspark_target_layer_ids": list(g("dspark.target_layers", [])),
    }
    clamp = g("swiglu_clamp_exp") or []
    if clamp and any(abs(c - clamp[0]) > 1e-6 for c in clamp):
        raise SystemExit("swiglu_clamp_exp differs between layers %s: the engine takes one swiglu_limit" % clamp[:8])
    return out


def ple_to_engram(kv: dict):
    """llama.cpp's own names for the Engram metadata (unsloth's GGUFs from 2026-10-19 on: ple.*, the hash heads' offsets
    counted across the layers' tables) -> the engram.* keys this pack reads (each layer's offsets from its own table)."""
    a = kv.get("general.architecture")
    g = lambda k, d=None: kv.get("%s.%s" % (a, k), d)  # noqa: E731
    if g("engram.layers") or not g("ple.layers"):
        return
    layers, ngram, heads = list(g("ple.layers")), g("ple.ngram_size"), g("ple.heads_per_ngram")
    offs, sizes = list(g("ple.head_offsets")), list(g("ple.head_vocab_sizes"))
    cols = (ngram - 1) * heads
    if len(offs) != len(layers) * cols or len(sizes) != len(offs):
        raise SystemExit("ple: %d head offsets, %d sizes for %d layers x %d columns" %
                         (len(offs), len(sizes), len(layers), cols))
    per, rows = [], []
    for i in range(len(layers)):
        o, s = offs[i * cols:(i + 1) * cols], sizes[i * cols:(i + 1) * cols]
        per += [x - o[0] for x in o]
        rows.append(o[-1] + s[-1] - o[0])
    for k, v in (("layers", layers), ("max_ngram_size", ngram), ("head_count", heads),
                 ("head_dim", g("embedding_length_per_layer_input")), ("multipliers", list(g("ple.layer_multipliers"))),
                 ("primes", sizes), ("offsets", per), ("num_embeddings", rows)):
        kv["%s.engram.%s" % (a, k)] = v


def read_engram_hash_tensor(
    shard: Gguf, tensor: tuple[str, int, list[int], int]
) -> tuple[str, list[int]]:
    """Read a recognized Engram hash tensor, checking its integer type and file bounds."""
    name, typ, dims, off = tensor
    key, fmt, size, expected_type = ENGRAM_HASH_TENSORS[name]
    if typ != expected_type or not dims or any(d <= 0 for d in dims):
        raise SystemExit(
            f"engram hash tensor {name}: expected ggml type {expected_type} and nonempty dimensions"
        )
    count = 1
    for dim in dims:
        count *= dim
    start, length = shard.data_start + off, count * size
    if start < shard.data_start or start + length > shard.path.stat().st_size:
        raise SystemExit(
            f"engram hash tensor {name}: payload lies outside {shard.path.name}"
        )
    with shard.path.open("rb") as stream:
        stream.seek(start)
        raw = stream.read(length)
    if len(raw) != length:
        raise SystemExit(f"engram hash tensor {name}: truncated payload")
    return key, list(struct.unpack(f"<{count}{fmt}", raw))


def int_tensors_to_engram(gs: list[Gguf], kv: dict[str, object]) -> None:
    """Recover Engram hash metadata without accepting conflicting or mistyped payloads."""
    arch = kv.get("general.architecture")
    for shard in gs:
        for tensor in shard.tensors:
            if tensor[0] not in ENGRAM_HASH_TENSORS:
                continue
            key, values = read_engram_hash_tensor(shard, tensor)
            field = f"{arch}.engram.{key}"
            if field in kv and kv[field] != values:
                raise SystemExit(
                    f"engram hash tensor {tensor[0]}: conflicts with existing {field}"
                )
            kv[field] = values
    for dotted, alias in (
        ("engram.layers", "engram.layer_ids"),
        ("engram.max_ngram_size", "engram_max_ngram_size"),
        ("engram.head_count", "engram_n_heads"),
        ("engram.head_dim", "engram_head_dim"),
        ("engram.pad_token_id", "engram_pad_token_id"),
    ):
        field, alternative = f"{arch}.{dotted}", f"{arch}.{alias}"
        if field not in kv and alternative in kv:
            kv[field] = kv[alternative]


def is_engram_file(g: Gguf) -> bool:
    """A GGUF of the Engram tables themselves: architecture 'engram' (the first release) or 'deepseek41-engram'
    (ISTA-DASLab's final files), which ships beside the model's shards in its own folder."""
    arch = str(g.kv.get("general.architecture") or "")
    return arch == "engram" or arch.endswith("-engram")


def model_shard(gs: list[Gguf]) -> Gguf | None:
    """Find a DeepSeek model shard rather than selecting an auxiliary Engram file."""
    return next(
        (g for g in gs if g.kv.get("general.architecture") == "deepseek41"), None
    )


def split_ple(g: Gguf, kv: dict):
    """llama.cpp's one Engram tensor (per_layer_token_embd: every layer's table, one after the other) -> each layer's
    own blk.N.engram_embd as the earlier GGUFs had it: the same bytes, read in place."""
    a = kv["general.architecture"]
    ple = [t for t in g.tensors if t[0] == "per_layer_token_embd.weight"]
    if not ple:
        return
    name, typ, dims, off = ple[0]
    layers, rows = (
        kv.get("%s.engram.layers" % a, []),
        kv.get("%s.engram.num_embeddings" % a, []),
    )
    if (
        len(dims) != 2
        or typ not in TYPES
        or len(rows) != len(layers)
        or sum(rows) != dims[1]
    ):
        raise SystemExit(
            "%s: %s %s does not split in the Engram layers %s of %s rows"
            % (g.path.name, name, dims, layers, rows)
        )
    _, per, bpb = TYPES[typ]
    row = dims[0] // per * bpb
    parts, start = [], 0
    for il, n in zip(layers, rows):
        parts.append(
            ("blk.%d.engram_embd.weight" % il, typ, [dims[0], n], off + start * row)
        )
        start += n
    g.tensors = [t for t in g.tensors if t[0] != name] + parts


def short(g: Gguf):
    """Why a GGUF file is shorter than its own tensor directory says (a download still running), or None."""
    need = g.data_start
    for _, typ, dims, off in g.tensors:
        if typ in TYPES:
            _, per, bpb = TYPES[typ]
            nel = 1
            for d in dims:
                nel *= d
            need = max(need, g.data_start + off + nel // per * bpb)
    have = g.path.stat().st_size
    return None if have >= need else "%s: %d of %d bytes" % (g.path.name, have, need)


def engram_set(model: pathlib.Path, explicit: str | None):
    """The Engram tables' GGUF set for a quant that leaves them out: (its folder, its files read), or (None, []).
    explicit (--engram): that folder, whole; else the first whole one of ENGRAM_SETS beside the quant or in its
    folder - one not whole yet (its download still running) is passed over."""
    dirs = (
        [pathlib.Path(explicit).expanduser().resolve()]
        if explicit
        else [model] + [d for n in ENGRAM_SETS for d in (model / n, model.parent / n)]
    )
    for d in dirs:
        paths = (
            [d] if d.is_file() else sorted(d.glob("*.gguf")) if d.is_dir() else []
        )  # (--engram: folder or file)
        if not paths:
            if explicit:
                raise SystemExit(
                    "%s: no .gguf files (--engram: the folder of the Engram tables' GGUF set)"
                    % d
                )
            continue
        try:
            egs = [Gguf(p) for p in paths]
            if not explicit and d == model:
                egs = [shard for shard in egs if is_engram_file(shard)]
                if not egs:
                    continue
            why = next((w for w in map(short, egs) if w), None)
        except (Exception, SystemExit) as e:  # noqa: BLE001 - a header cut short raises anything
            why = str(e) or type(e).__name__
        if why is None:
            return d, egs
        if explicit:
            raise SystemExit("%s: not whole (%s)" % (d, why))
        print("%s: not whole (%s) - not used" % (d, why))
    return None, []


def engram_int_array(kv: dict[str, object], key: str) -> list[int]:
    """Read an Engram integer array without coercing floats or booleans to integers."""
    raw = kv.get(key)
    if not isinstance(raw, list):
        raise SystemExit(f"Engram metadata {key}: missing integer array")
    values: list[int] = []
    for value in raw:
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not -(2**63) <= value < 2**63
        ):
            raise SystemExit(f"Engram metadata {key}: expected signed int64 values")
        values.append(value)
    return values


def validate_engram_metadata(kv: dict[str, object], vocab: int) -> None:
    """Reject incomplete hash parameters before writing any pack artifacts."""
    prefix = f"{kv['general.architecture']}.engram."
    if not kv.get(prefix + "layers"):
        return
    arrays = {
        name: engram_int_array(kv, prefix + name)
        for name in ("layers", "token_map", "multipliers", "primes", "offsets")
    }
    scalars: dict[str, int] = {}
    for name in ("max_ngram_size", "head_count", "head_dim", "pad_token_id"):
        value = kv.get(prefix + name)
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or not 0 <= value < 2**31
        ):
            raise SystemExit(
                f"Engram metadata {prefix}{name}: expected nonnegative int32"
            )
        scalars[name] = value
    ngram, heads, width = (
        scalars[name] for name in ("max_ngram_size", "head_count", "head_dim")
    )
    layers, token_map = arrays["layers"], arrays["token_map"]
    if (
        not 0 < vocab < 2**31
        or not 2 <= ngram <= 8
        or heads < 1
        or width < 32
        or width % 32
    ):
        raise SystemExit(
            "Engram metadata: invalid vocabulary, ngram count, head count, or row width"
        )
    if (
        len(layers) > 8
        or len(set(layers)) != len(layers)
        or any(layer < 0 for layer in layers)
    ):
        raise SystemExit(
            "Engram metadata: expected at most eight distinct nonnegative layers"
        )
    if len(token_map) > vocab or any(not 0 <= value < 2**31 for value in token_map):
        raise SystemExit(
            "Engram metadata: token map exceeds vocabulary or contains invalid int32 IDs"
        )
    if not 0 <= scalars["pad_token_id"] < len(token_map):
        raise SystemExit("Engram metadata: pad token id is outside the token map")
    cols, count = (ngram - 1) * heads, len(layers)
    mult, primes, offsets = (
        arrays[name] for name in ("multipliers", "primes", "offsets")
    )
    if (
        len(mult) != count * ngram
        or len(primes) != count * cols
        or len(offsets) != count * cols
    ):
        raise SystemExit(
            f"engram: {len(mult)} multipliers, {len(primes)} primes, {len(offsets)} offsets "
            f"for {count} layers x {cols} columns (ngram {ngram})"
        )
    if any(prime <= 0 for prime in primes):
        raise SystemExit("Engram primes must be positive")
    if any(offset < 0 for offset in offsets):
        raise SystemExit("Engram offsets must be nonnegative")


def write_engram(out: pathlib.Path, kv: dict, vocab: int):
    validate_engram_metadata(kv, vocab)
    a = kv["general.architecture"]
    g = lambda k, d=None: kv.get("%s.%s" % (a, k), d)  # noqa: E731
    layers = list(g("engram.layers", []))
    if not layers:
        return 0, 0
    ngram, heads, hd = g("engram.max_ngram_size"), g("engram.head_count"), g("engram.head_dim")
    tm = list(g("engram.token_map"))
    mult, primes, offs = list(g("engram.multipliers")), list(g("engram.primes")), list(g("engram.offsets"))
    cols = (ngram - 1) * heads
    nl = len(layers)
    pad = int(tm[g("engram.pad_token_id")])
    full = tm + [0] * max(0, vocab - len(tm))
    with out.open("wb") as f:
        f.write(struct.pack("<4s6i", b"DSE1", vocab, nl, ngram, heads, hd, pad))
        f.write(struct.pack("<%di" % vocab, *full[:vocab]))
        f.write(struct.pack("<%dq" % len(mult), *mult))
        f.write(struct.pack("<%dq" % len(primes), *primes))
        f.write(struct.pack("<%dq" % len(offs), *offs))
    return nl, cols


def write_tokenizer(out: pathlib.Path, kv: dict, template: str | None):
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(tokens)
    merges = kv.get("tokenizer.ggml.merges") or []
    if len(set(tokens)) != len(tokens):
        raise SystemExit("the GGUF's vocabulary has duplicate token strings")
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps([int(t) for t in types]))
    special = {"bos": int(kv.get("tokenizer.ggml.bos_token_id", 0)), "eos": int(kv.get("tokenizer.ggml.eos_token_id", 1)),
               "pad": int(kv.get("tokenizer.ggml.padding_token_id", 2))}
    (out / "tokenizer.json").write_text(json.dumps({"pre": "deepseek3", "special_ids": special}, indent=1))
    if template:
        (out / "chat_template.jinja").write_text(template, encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the directory of the GGUF's shards")
    ap.add_argument("--out", help="default: <model>/maya-ds4")
    ap.add_argument("--vision", help="vision_pack.py's folder (default: <model>/../vision; taken when it is there)")
    ap.add_argument("--drafter", help="the DSpark drafter's .json (drafter_pack.py's; default <model>/../dspark/dspark.json)")
    ap.add_argument("--engram", help="an Engram GGUF file or folder, for a quant that leaves its tables out "
                                     "(default: <model>/../Engram, when it is whole)")
    ap.add_argument("--shards", nargs="+", help="the GGUF's files in --model (default: every .gguf there - one "
                                                "model's; the setup names them: a folder may hold an earlier layout)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser() if a.out else model / "maya-ds4"
    shards = [model / n for n in a.shards] if a.shards else sorted(model.glob("*.gguf"))
    if not shards or not all(p.is_file() for p in shards):
        raise SystemExit("%s: no .gguf files" % model if not shards else
                         "%s: not there" % next(p for p in shards if not p.is_file()))
    gs = [Gguf(p) for p in shards]
    eg_here = []                           # an Engram file in the model's own folder (ISTA-DASLab's sets ship one)
    if not a.shards:                       # a folder's *.gguf: the Engram tables' GGUF (architecture 'engram' or
        keep = [i for i, g in enumerate(gs) if not is_engram_file(g)]   # 'deepseek41-engram') is not a model shard
        eg_here = [g for g in gs if is_engram_file(g)]
        shards, gs = [shards[i] for i in keep], [gs[i] for i in keep]
        if not gs:
            raise SystemExit(f"{model}: no model shards, only an Engram table")
    source = model_shard(gs)
    if source is None:
        arch = next((g.kv.get("general.architecture") for g in gs), None)
        raise SystemExit(f"{model}: not a DeepSeek-V4.1 GGUF (architecture {arch!r})")
    kv = source.kv
    split = int(kv.get("split.count", 1))
    if split != len(shards):
        counts = sorted({m.group(1) for m in (re.search(r"-of-(\d{5})\.gguf$", p.name) for p in shards) if m})
        raise SystemExit("%s: %d of the %d shards are here" % (model, len(shards), split) if len(counts) < 2 else
                         "%s: the shards of more than one GGUF split (-of-%s): name the model's own with --shards" %
                         (model, ", -of-".join(counts)))
    ple_to_engram(kv)
    for g in gs:
        split_ple(g, kv)
    int_tensors_to_engram(gs, kv)
    # the Engram tables may come from beside the quant (the same tensor names) - its engram tensors fill in the ones
    # the quant's shards leave out, its engram metadata too.  v1.1.16's Maya-S: unsloth's UD-Q8_K_XL shard that holds
    # only the tables, as it is (00008 of 9; before 2026-10-19 shards 00008 and 00009 of 10, a table each), so no
    # shard count is checked - every Engram layer's tables must be found, below.  Looked for only when the quant's
    # shards leave some out.
    have = {t[0] for g in gs for t in g.tensors}
    arch = kv["general.architecture"]
    layers = kv.get("%s.engram.layers" % arch, [])

    def has_part(il: int, part: str, names: set[str]) -> bool:
        """The quant (or an Engram set) holds the layer's part: the GSQ sets keep the gate's q and k apart and the
        kv as engram_wkv, where the earlier GGUFs had one engram_gate / engram_kv tensor."""
        if f"blk.{il}.{part}.weight" in names:
            return True
        if part == "engram_kv":
            return f"blk.{il}.engram_wkv.weight" in names
        if part == "engram_gate":
            return f"blk.{il}.engram_q.weight" in names and f"blk.{il}.engram_k.weight" in names
        return False

    lacking = not layers or any(not has_part(il, part, have)
                                for il in layers for part in ("engram_embd", "engram_kv", "engram_gate"))
    eg_dir, egs = None, []
    if lacking or a.engram:
        if a.engram:
            eg_dir, egs = engram_set(model, a.engram)
        elif eg_here:
            why = next((w for w in map(short, eg_here) if w), None)
            if why is not None:        # (the tables sit in the folder the user asked to pack: say so, do not pass over)
                raise SystemExit(f"{model}: not whole ({why})")
            eg_dir, egs = model, eg_here
            print(f"Engram tables from the model's folder ({', '.join(g.path.name for g in eg_here)})")
        else:
            eg_dir, egs = engram_set(model, None)
    eg_from = {}
    if eg_dir is not None:
        eshards = [g.path for g in egs]
        for g in egs:
            for k, v in g.kv.items():
                if ".engram." in k or ".ple." in k or k.endswith(".embedding_length_per_layer_input"):
                    kv.setdefault(k, v)
        ple_to_engram(kv)
        for g in egs:
            split_ple(g, kv)
        for p, g in zip(eshards, egs):
            picked = [t for t in g.tensors if re.match(r"blk\.\d+\.engram_", t[0]) and t[0] not in have]
            picked = [t for t in picked if t[0].endswith("engram_embd.weight") or not has_part(
                int(t[0].split(".")[1]), "engram_kv" if t[0].endswith("engram_kv.weight") else "engram_gate", have)]
            if picked:
                g.tensors = picked
                eg_from[p] = g
    for il in kv.get("%s.engram.layers" % arch, []):
        for part in ("engram_embd", "engram_kv", "engram_gate"):
            nm = "blk.%d.%s.weight" % (il, part)
            if not has_part(il, part, have) and \
                    not any(t[0] == nm for g in eg_from.values() for t in g.tensors):
                raise SystemExit("%s: missing - not in the quant's shards nor in an Engram set (--engram, or "
                                 "<model>/../Engram/: unsloth's UD-Q8_K_XL shard of the Engram tables)" % nm)
    rel = lambda p: os.path.relpath(p, model).replace(os.sep, "/")  # noqa: E731
    gs_all = gs + list(eg_from.values())
    names = [g.path.name for g in gs] + [rel(p) for p in eg_from]

    tensors, unknown, skipped = [], [], []
    for fi, g in enumerate(gs_all):
        for name, typ, dims, off in g.tensors:
            if name in ENGRAM_HASH_TENSORS:
                skipped.append(f"{name} (the Engram's hash parameters, read into the metadata)")
                continue
            if typ not in TYPES:
                raise SystemExit("%s: %s has ggml type %d, which the engine does not read" % (g.path.name, name, typ))
            tname, per, bpb = TYPES[typ]
            nel = 1
            for d in dims:
                nel *= d
            if dims[0] % per:
                raise SystemExit("%s: %s rows of %d are not whole %s blocks" % (g.path.name, name, dims[0], tname))
            nbytes = nel // per * bpb
            m = re.match(r"blk\.(\d+)\.(.+)$", name)
            if m and m.group(2) in LAYER:
                ck = "layers.%s.%s" % (m.group(1), LAYER[m.group(2)])
            elif name in GLOBAL:
                ck = GLOBAL[name]
            else:
                unknown.append(name)
                continue
            shape = list(reversed(dims))     # ggml ne0 innermost -> torch order, outermost first
            base = g.data_start + off
            if ck.endswith(".attn.wo_a.slice"):
                # [o_groups * o_lora][in] -> o_groups entries of [o_lora][in] (group g: rows g * o_lora ..)
                n = int(kv["%s.attention.output_group_count" % kv["general.architecture"]])
                if len(shape) == 3 and shape[0] == n:
                    shape = [shape[0] * shape[1], shape[2]]   # (the GSQ sets: [groups][o_lora][in] in one tensor)
                if len(shape) != 2 or shape[0] % n:
                    raise SystemExit("%s: %s %s does not split in %d groups" % (g.path.name, name, shape, n))
                part = nbytes // n
                for i in range(n):
                    tensors.append(("%s.%d.weight" % (ck, i), tname, fi, base + i * part, part,
                                    [shape[0] // n, shape[1]]))
                continue
            if ".ffn.experts." in ck:
                # [n_exp][out][in] -> an entry an expert: layers.N.ffn.experts.E.w1.weight
                n, part = shape[0], nbytes // shape[0]
                pre, role = ck.rsplit(".", 1)
                for i in range(n):
                    tensors.append(("%s.%d.%s.weight" % (pre, i, role), tname, fi, base + i * part, part, shape[1:]))
                continue
            tensors.append((ck, tname, fi, base, nbytes, shape))
    if unknown:
        raise SystemExit("tensors the pack does not map (%d): %s" % (len(unknown), ", ".join(unknown[:12])))
    if skipped:
        print(f"not weights, left out ({len(skipped)}): {', '.join(n.split(' (')[0] for n in skipped[:8])}")
    # the DSpark drafter (drafter_pack.py's dspark.bin beside the quants' folders, from DeepSeek's checkpoint): its
    # three blocks as mtp.0-2, so the engine drafts with them
    files = names
    if eg_from:
        print("Engram tables from %s (%d shards)" % (eg_dir, len(eg_from)))
    drafter = pathlib.Path(a.drafter).expanduser().resolve() if a.drafter else model.parent / "dspark" / "dspark.json"
    if a.drafter and not drafter.is_file():
        raise SystemExit("%s: no drafter there" % drafter)
    drafter_in = drafter.is_file() and drafter.with_suffix(".bin").is_file()
    if drafter_in:
        files.append(rel(drafter.with_suffix(".bin")))
        groups = int(kv["%s.attention.output_group_count" % kv["general.architecture"]])
        for name, dt, off, nb, shape in json.loads(drafter.read_text())["tensors"]:
            if name.endswith(".attn.wo_a.weight"):   # (one entry a group, as the trunk's)
                for i in range(groups):
                    tensors.append(("%s.slice.%d.weight" % (name[:-len(".weight")], i), dt, len(files) - 1,
                                    off + i * (nb // groups), nb // groups, [shape[0] // groups, shape[1]]))
                continue
            tensors.append((name, dt, len(files) - 1, off, nb, shape))
    # the vision tower (vision_pack.py's vision.bin, from DeepSeek's checkpoint): its tensors under the checkpoint's
    # names, the picture span's marker rows, and the geometry config.json gives them
    vdir = pathlib.Path(a.vision).expanduser().resolve() if a.vision else model.parent / "vision"
    vision = {}
    if (vdir / "vision.json").is_file() and (vdir / "vision.bin").is_file():
        try:
            rel = os.path.relpath(vdir / "vision.bin", model).replace(os.sep, "/")
        except ValueError:  # (Windows: another drive)
            raise SystemExit(
                "%s: the vision folder must be on the model's drive" % vdir
            )
        if " " in rel:
            raise SystemExit(
                "%s: the vision folder's path relative to the model has a space" % vdir
            )
        files.append(rel)
        info = json.loads((vdir / "vision.json").read_text())
        for name, dt, off, nb, shape in info["tensors"]:
            tensors.append((name, dt, len(files) - 1, off, nb, shape))
        vision = info["config"]
    vocab = next(t[5][0] for t in tensors if t[0] == "embed.weight")
    geo = geometry(kv, vocab)
    validate_engram_metadata(kv, vocab)
    prefix = f"{kv['general.architecture']}.engram."
    layers = kv.get(prefix + "layers", [])
    cols = (kv.get(prefix + "max_ngram_size", 1) - 1) * kv.get(prefix + "head_count", 0)
    for index, layer in enumerate(layers):
        table = next(
            t for t in tensors if t[0] == f"layers.{layer}.engram.embed.weight"
        )
        shape = table[5]
        if len(shape) != 2 or shape[1] != kv[prefix + "head_dim"]:
            raise SystemExit(
                f"Engram table layer {layer}: shape does not match head dimension"
            )
        for offset, prime in zip(
            kv[prefix + "offsets"][index * cols : (index + 1) * cols],
            kv[prefix + "primes"][index * cols : (index + 1) * cols],
        ):
            if offset + prime > shape[0]:
                raise SystemExit(
                    f"Engram hash range exceeds table rows at layer {layer}"
                )
    # every tensor's bytes must lie within its file: a wrong offset or size is an error here, not a load failure
    end = [0] * len(files)
    for name, dt, fi, off, nb, shape in tensors:
        end[fi] = max(end[fi], off + nb)
    for i, fname in enumerate(files):
        size = (model / fname).stat().st_size
        if end[i] > size:
            over = next(t for t in tensors if t[2] == i and t[3] + t[4] == end[i])
            raise SystemExit(f"{fname}: {over[0]} ends at {end[i]} of {size} bytes")

    out.mkdir(parents=True, exist_ok=True)
    with (out / "ds4.txt").open("w") as f:
        for k, v in geo.items():
            if v is None:
                continue
            if isinstance(v, bool):
                v = int(v)
            f.write("%s %s\n" % (k, ",".join(str(x) for x in v) if isinstance(v, list) else v))
        for k, v in vision.items():
            f.write("%s %s\n" % (k, v))
        f.write("format gguf\n")
        f.write("model_dir %s\n" % model)
    with (out / "tensors.txt").open("w") as f:
        for i, name in enumerate(files):
            f.write("file %d %s\n" % (i, name))
        for name, dt, fi, off, nb, shape in tensors:
            f.write("t %s %s %d %d %d %s\n" % (name, dt, fi, off, nb, " ".join(map(str, shape))))
    nl, cols = write_engram(out / "engram.bin", kv, vocab)
    # the server's chat template (pack.py's, so both formats chat alike); the GGUF's own when it is not there
    template = pathlib.Path(__file__).resolve().parent / "chat_template.jinja"
    write_tokenizer(out / "tokenizer", kv,
                    template.read_text(encoding="utf-8") if template.exists() else kv.get("tokenizer.chat_template"))
    by = {}
    for t in tensors:
        by[t[1]] = by.get(t[1], 0) + t[4]
    extras = [w for w, there in (("the DSpark drafter", drafter_in), ("the vision tower", bool(vision))) if there]
    print("%s: %d tensors in %d files (%s), Engram %d layers x %d hash columns%s -> %s" %
          (model.name, len(tensors), len(files), ", ".join("%s %.1f GB" % (k, v / 1e9) for k, v in sorted(by.items())),
           nl, cols, "".join(", " + w for w in extras), out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

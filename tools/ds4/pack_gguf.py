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
of GGUF files that holds them, beside it: --engram, else ../Engram/ (unsloth's MXFP8 shard of them, where setups before
v1.1.17 put it) when it is whole.
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
    "engram_wkv.weight": "engram.wkv.weight",            # (newer converters: the checkpoint's names,
    "engram_q.weight": "engram.q_weight",                #  q and k apart - the engine folds them as from the
    "engram_k.weight": "engram.k_weight",                #  checkpoint)
}
GLOBAL = {"token_embd.weight": "embed.weight", "output_norm.weight": "norm.weight", "output.weight": "head.weight"}


def geometry(kv: dict, vocab: int) -> dict:
    """config.json's text_config names (what the engine reads) from the GGUF's deepseek41.* keys."""
    a = kv["general.architecture"]
    g = lambda k, d=None: kv.get("%s.%s" % (a, k), d)  # noqa: E731
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


# DeepSeek-V4.1-Flash's own DSpark geometry (its checkpoint's config.json - the drafter drafter_pack.py builds is that
# checkpoint's), for a GGUF of it that leaves the keys out: ISTA's GSQ-RCO carries only the target layers (#500), and a
# drafter packed with block size 0 loads, VRAM and all, and never drafts.  The trunk it is for, as the GGUF gives it.
DSPARK_KEYS = ("dspark_block_size", "dspark_noise_token_id", "dspark_markov_rank", "dspark_n_routed_experts",
               "dspark_num_experts_per_tok", "dspark_target_layer_ids")
DSPARK_V41_FLASH = {"dspark_block_size": 5, "dspark_noise_token_id": 128799, "dspark_markov_rank": 256,
                    "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 3,
                    "dspark_target_layer_ids": [37, 38, 39]}
V41_FLASH_TRUNK = {"hidden_size": 5120, "num_hidden_layers": 40, "n_routed_experts": 384}


def drafter_experts(info: dict):
    """The routed experts of the drafter's first block (its .json's tensor names), or None when it names none."""
    ids = [int(m.group(1)) for m in (re.match(r"mtp\.0\.ffn\.experts\.(\d+)\.", t[0]) for t in info.get("tensors", []))
           if m]
    return max(ids) + 1 if ids else None


def dspark_fill(geo: dict, info: dict, vocab: int):
    """The DSpark keys geo (the GGUF's) lacks or has as 0, for the drafter `info` (its .json): from that .json's own
    "config" (drafter_pack.py writes the checkpoint's when its config.json is beside the shards), else
    DeepSeek-V4.1-Flash's when the trunk and the drafter are that model's.  (values, where from), or (None, the keys
    missing) when nothing gives them - the drafter would never draft."""
    missing = [k for k in DSPARK_KEYS if not geo.get(k)]
    if not missing:
        return {}, None
    own = info.get("config") or {}
    fill = {k: own[k] for k in missing if own.get(k)}
    src = "the drafter's .json" if fill else ""
    rest = [k for k in missing if k not in fill]
    flash = all(geo.get(k) == v for k, v in V41_FLASH_TRUNK.items()) and vocab > DSPARK_V41_FLASH["dspark_noise_token_id"]
    if rest and flash and drafter_experts(info) in (None, DSPARK_V41_FLASH["dspark_n_routed_experts"]):
        fill.update({k: DSPARK_V41_FLASH[k] for k in rest})
        src += (" and " if src else "") + "DeepSeek-V4.1-Flash's own values"
        rest = []
    if rest:
        return None, rest
    return fill, src


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


# newer converters' spellings of the metadata -> the ones this pack reads: (theirs, ours)
NEWER_KEYS = (("kv_source_layer_ids", "attention.kv_source_layers"),
              ("index_source_layer_ids", "attention.indexer.source_layers"),
              ("attention.candidate_source_layer_id", "attention.indexer.candidate_source_layer"),
              ("attention.candidate_topk_blocks", "attention.indexer.candidate_top_k_blocks"),
              ("attention.candidate_block_size", "attention.indexer.candidate_block_size"),
              ("dspark_target_layer_ids", "dspark.target_layers"),
              ("engram.layer_ids", "engram.layers"), ("engram_max_ngram_size", "engram.max_ngram_size"),
              ("engram_n_heads", "engram.head_count"), ("engram_head_dim", "engram.head_dim"),
              ("engram_pad_token_id", "engram.pad_token_id"))
# its Engram hash parameters as tensors (I64 / I32) -> the engram.* lists write_engram takes (each layer's in turn)
HASH_TENSORS = {"engram_multipliers.weight": "engram.multipliers", "engram_primes.weight": "engram.primes",
                "engram_offsets.weight": "engram.offsets", "engram_token_map.weight": "engram.token_map"}
INT_TYPES = {26: ("<i", 4), 27: ("<q", 8)}   # ggml I32 / I64


def newer_keys(gs: list, kv: dict):
    """A GGUF that spells the metadata as NEWER_KEYS's first names, with the Engram hash parameters as tensors: their
    values under the names the rest of this pack reads (an existing key wins), the tensors read and left out."""
    a = kv["general.architecture"]
    for theirs, ours in NEWER_KEYS:
        if "%s.%s" % (a, theirs) in kv:
            kv.setdefault("%s.%s" % (a, ours), kv["%s.%s" % (a, theirs)])
    for g in gs:
        keep = []
        for name, typ, dims, off in g.tensors:
            if name not in HASH_TENSORS:
                keep.append((name, typ, dims, off))
                continue
            if typ not in INT_TYPES:
                raise SystemExit("%s: %s has ggml type %d (I32 / I64 expected)" % (g.path.name, name, typ))
            fmt, size = INT_TYPES[typ]
            n = 1
            for d in dims:
                n *= d
            with g.path.open("rb") as f:
                f.seek(g.data_start + off)
                vals = list(struct.unpack(fmt[0] + "%d" % n + fmt[1], f.read(n * size)))
            kv.setdefault("%s.%s" % (a, HASH_TENSORS[name]), vals)   # (ggml ne0 innermost: each layer's row in turn)
        g.tensors = keep


def split_ple(g: Gguf, kv: dict):
    """llama.cpp's one Engram tensor (per_layer_token_embd: every layer's table, one after the other) -> each layer's
    own blk.N.engram_embd as the earlier GGUFs had it: the same bytes, read in place."""
    a = kv["general.architecture"]
    ple = [t for t in g.tensors if t[0] == "per_layer_token_embd.weight"]
    if not ple:
        return
    name, typ, dims, off = ple[0]
    layers, rows = kv.get("%s.engram.layers" % a, []), kv.get("%s.engram.num_embeddings" % a, [])
    if len(dims) != 2 or typ not in TYPES or len(rows) != len(layers) or sum(rows) != dims[1]:
        raise SystemExit("%s: %s %s does not split in the Engram layers %s of %s rows" % (g.path.name, name, dims, layers,
                                                                                         rows))
    _, per, bpb = TYPES[typ]
    row = dims[0] // per * bpb
    parts, start = [], 0
    for il, n in zip(layers, rows):
        parts.append(("blk.%d.engram_embd.weight" % il, typ, [dims[0], n], off + start * row))
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
    dirs = ([pathlib.Path(explicit).expanduser().resolve()] if explicit else
            [d for n in ENGRAM_SETS for d in (model / n, model.parent / n)])
    for d in dirs:
        paths = sorted(d.glob("*.gguf")) if d.is_dir() else []
        if not paths:
            if explicit:
                raise SystemExit("%s: no .gguf files (--engram: the folder of the Engram tables' GGUF set)" % d)
            continue
        try:
            egs = [Gguf(p) for p in paths]
            why = next((w for w in map(short, egs) if w), None)
        except (Exception, SystemExit) as e:   # noqa: BLE001 - a header cut short raises anything
            why = str(e) or type(e).__name__
        if why is None:
            return d, egs
        if explicit:
            raise SystemExit("%s: not whole (%s)" % (d, why))
        print("%s: not whole (%s) - not used" % (d, why))
    return None, []


def write_engram(out: pathlib.Path, kv: dict, vocab: int):
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
    if len(mult) != nl * ngram or len(primes) != nl * cols or len(offs) != nl * cols:
        raise SystemExit("engram: %d multipliers, %d primes, %d offsets for %d layers x %d columns (ngram %d)" %
                         (len(mult), len(primes), len(offs), nl, cols, ngram))
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
    ap.add_argument("--engram", help="the folder of the Engram tables' GGUF set, for a quant that leaves them out "
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
    # (a folder of shards beside a separate Engram GGUF: its tables, read below, not a shard)
    side = [g for g in gs if g.kv.get("general.architecture") == "deepseek41-engram"]
    gs = [g for g in gs if g not in side]
    kv = next((g.kv for g in gs if "general.architecture" in g.kv), None)
    if kv is None or kv["general.architecture"] not in ("deepseek41",):
        raise SystemExit("%s: not a DeepSeek-V4.1 GGUF (architecture %r)" % (model, kv and kv.get("general.architecture")))
    split = int(kv.get("split.count", 1))
    if split != len(gs):
        counts = sorted({m.group(1) for m in (re.search(r"-of-(\d{5})\.gguf$", p.name) for p in shards) if m})
        raise SystemExit("%s: %d of the %d shards are here" % (model, len(shards), split) if len(counts) < 2 else
                         "%s: the shards of more than one GGUF split (-of-%s): name the model's own with --shards" %
                         (model, ", -of-".join(counts)))
    newer_keys(gs, kv)
    ple_to_engram(kv)
    for g in gs:
        split_ple(g, kv)
    # the Engram tables may come from a shared set of GGUF files beside the quant (the same tensor names: engram_set) -
    # its engram tensors fill in the ones the quant's shards leave out, its engram metadata too.  v1.1.16's Maya-S:
    # unsloth's UD-Q8_K_XL shard that holds only the tables, as it is (00008 of 9; before 2026-10-19 shards 00008 and
    # 00009 of 10, a table each), so no shard count is checked - every Engram layer's tables must be found, below.
    # Looked for only when the quant's shards leave some out.
    have = {t[0] for g in gs for t in g.tensors}
    arch = kv["general.architecture"]
    layers = kv.get("%s.engram.layers" % arch, [])
    # (each Engram layer: its tables, its kv projection and its gate - folded, or its q and k apart)
    parts = (("engram_embd",), ("engram_kv", "engram_wkv"), ("engram_gate", "engram_q"))
    lacking = not layers or any(all("blk.%d.%s.weight" % (il, p) not in have for p in alts)
                                for il in layers for alts in parts)
    # the tables in a GGUF of their own beside the shards (the model's engram.file names it)
    own = kv.get("%s.engram.file" % arch)
    if lacking and not a.engram and (side or (own and (model / own).is_file())):
        eg_dir, egs = model, side or [Gguf(model / own)]
    else:
        eg_dir, egs = engram_set(model, a.engram) if lacking or a.engram else (None, [])
    eg_from = {}
    if eg_dir is not None:
        eshards = [g.path for g in egs]
        for g in egs:
            for k, v in g.kv.items():
                if ".engram." in k or ".ple." in k or k.endswith(".embedding_length_per_layer_input"):
                    # (the tables' own hash parameters: a quant's that differ would read other rows - the set's win,
                    #  e.g. Maya's Q6_K tables under another quant)
                    if k in kv and kv[k] != v and ".engram." in k:
                        print("Engram %s: the tables' own (%s) over the quant's" % (k.split(".engram.", 1)[1],
                                                                                str(v)[:60]))
                    kv[k] = v if ".engram." in k else kv.get(k, v)
        ple_to_engram(kv)
        for g in egs:
            split_ple(g, kv)
        for p, g in zip(eshards, egs):
            # (what the quant's shards hold under another name stays the quant's: a set's folded kv / gate beside
            #  a quant's own wkv / q / k would be the same tensors twice)
            alias = {"engram_kv": "engram_wkv", "engram_gate": "engram_q"}
            picked = [t for t in g.tensors if re.match(r"blk\.\d+\.engram_", t[0]) and t[0] not in have and
                      not any(t[0].endswith(".%s.weight" % x) and t[0].replace(x, y) in have for x, y in alias.items())]
            if picked:
                g.tensors = picked
                eg_from[p] = g
    for il in kv.get("%s.engram.layers" % arch, []):
        for alts in parts:
            nms = ["blk.%d.%s.weight" % (il, p) for p in alts]
            if not any(nm in have or any(t[0] == nm for g in eg_from.values() for t in g.tensors) for nm in nms):
                raise SystemExit("%s: missing - not in the quant's shards nor in an Engram set (--engram, or "
                                 "<model>/../Engram/: unsloth's UD-Q8_K_XL shard of the Engram tables)" % nms[0])
    rel = lambda p: os.path.relpath(p, model).replace(os.sep, "/")  # noqa: E731
    gs_all = gs + list(eg_from.values())
    names = [g.path.name for g in gs] + [rel(p) for p in eg_from]

    tensors, unknown = [], []
    for fi, g in enumerate(gs_all):
        for name, typ, dims, off in g.tensors:
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
                if len(shape) == 3 and shape[0] == n:   # (newer GGUFs: [o_groups][o_lora][in], the same bytes)
                    shape = [n * shape[1], shape[2]]
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
    # the DSpark drafter (drafter_pack.py's dspark.bin beside the quants' folders, from DeepSeek's checkpoint): its
    # three blocks as mtp.0-2, so the engine drafts with them
    files = names
    if eg_from:
        print("Engram tables from %s (%d shards)" % (eg_dir, len(eg_from)))
    drafter = pathlib.Path(a.drafter).expanduser().resolve() if a.drafter else model.parent / "dspark" / "dspark.json"
    if a.drafter and not drafter.is_file():
        raise SystemExit("%s: no drafter there" % drafter)
    drafter_in = drafter.is_file() and drafter.with_suffix(".bin").is_file()
    vocab = next(t[5][0] for t in tensors if t[0] == "embed.weight")
    geo = geometry(kv, vocab)
    dinfo = json.loads(drafter.read_text()) if drafter_in else {}
    if drafter_in:   # (its geometry: the GGUF's, else filled in - a drafter without it never drafts: left out, said so)
        fill, src = dspark_fill(geo, dinfo, vocab)
        if fill is None:
            print("WARNING: the DSpark drafter %s is left out of the pack: the GGUF has no %s for it (and this is not "
                  "DeepSeek-V4.1-Flash's trunk, whose own values the pack would take) - with them 0 the engine would "
                  "load the drafter and never draft.  Drafting stays off; a GGUF with the dspark.* metadata drafts"
                  % (drafter, ", ".join(src)), file=sys.stderr)
            drafter_in = False
        elif fill:
            geo.update(fill)
            print("DSpark geometry not in the GGUF, taken from %s: %s" %
                  (src, ", ".join("%s %s" % (k, fill[k]) for k in DSPARK_KEYS if k in fill)))
    if drafter_in:
        files.append(rel(drafter.with_suffix(".bin")))
        groups = int(kv["%s.attention.output_group_count" % kv["general.architecture"]])
        for name, dt, off, nb, shape in dinfo["tensors"]:
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
        except ValueError:     # (Windows: another drive)
            raise SystemExit("%s: the vision folder must be on the model's drive" % vdir)
        if " " in rel:
            raise SystemExit("%s: the vision folder's path relative to the model has a space" % vdir)
        files.append(rel)
        info = json.loads((vdir / "vision.json").read_text())
        for name, dt, off, nb, shape in info["tensors"]:
            tensors.append((name, dt, len(files) - 1, off, nb, shape))
        vision = info["config"]

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

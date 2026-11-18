"""tools/hy4/pack_gguf.py - maya-hy4's pack for a Hy4-flash GGUF set (Tencent's Hy4-flash, the GGUF's `hy_v4_flash`
layout: AngelSlim's Q4_K_M / Q3_K / IQ2 builds), beside the model's files (no weights copied).

    python tools/hy4/pack_gguf.py --model ~/models/Hy4-flash-GGUF

writes <model>/maya-hy4/:
  hy4.txt       the geometry, one `key value` a line (lists comma-separated), from the GGUF's hy_v4_flash.* keys
  tensors.txt   `file <i> <path>` lines (relative to the model's folder, model_dir in hy4.txt), then `t <name> <type> <file> <offset> <bytes> <ne0> <ne1> ...` a tensor: the
                GGUF's own names and dims (ne0 innermost, as ggml counts them), type ggml's name (Q4_K, Q8_0, BF16,
                F32, F8_E4M3 ...), offset absolute in its file
  engram.bin    the Engram n-gram hash ('HYE1'): int32 vocab, modules, ngram, heads_per_ngram, head_dim, pad (the token
                map's value of the pad token), table_rows; int32 token_map[vocab]; int32 layers[modules]; uint64
                multipliers[modules][ngram], primes[modules][cols], offsets[modules][cols] (cols = (ngram - 1)
                heads_per_ngram; the offsets index the one table, per_layer_token_embd)
  tokenizer/    the server's tokenizer from the GGUF's vocabulary and merges (pre "hy_v4") and the model's own chat
                template (tokenizer.chat_template)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import re
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
ARCH = "hy_v4_flash"

_spec = importlib.util.spec_from_file_location("ds4_pack_gguf", HERE.parent / "ds4" / "pack_gguf.py")
_ds4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ds4)

# ggml type -> (name, values a block, bytes a block): the types a Hy4-flash GGUF holds and the engine reads
TYPES = {0: ("F32", 1, 4), 1: ("F16", 1, 2), 30: ("BF16", 1, 2), 8: ("Q8_0", 32, 34), 43: ("F8_E4M3", 1, 1),
         10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110), 12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176),
         14: ("Q6_K", 256, 210), 16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98),
         19: ("IQ1_S", 256, 50), 20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82),
         23: ("IQ4_XS", 256, 136), 29: ("IQ1_M", 256, 56)}


class Gguf(_ds4.Gguf):
    """maya-ds4's GGUF reader, with ggml type 43 as ggml's F8_E4M3 (one byte a value: Hy4-flash's Engram table)."""

    def _type43(self, align: int):
        pass


def nbytes(typ: int, dims: list) -> int:
    _, per, bpb = TYPES[typ]
    if dims[0] % per:
        raise SystemExit("a %s row of %d values is not whole blocks" % (TYPES[typ][0], dims[0]))
    n = 1
    for d in dims[1:]:
        n *= d
    return dims[0] // per * bpb * n


def find_shards(model: pathlib.Path, names: list | None) -> list:
    paths = [model / n for n in names] if names else sorted(model.glob("*.gguf"))
    if not paths:
        raise SystemExit("%s: no .gguf files" % model)
    gs = [Gguf(p) for p in paths]
    first = [g for g in gs if g.kv.get("general.architecture")]
    if not first or first[0].kv["general.architecture"] != ARCH:
        raise SystemExit("%s: not a %s GGUF (general.architecture %r)" %
                         (model, ARCH, first[0].kv.get("general.architecture") if first else None))
    # one quant a folder: a set's shards all name the same count
    want = int(first[0].kv.get("split.count", 1))
    if len(gs) != want:
        raise SystemExit("%s: %d .gguf files, the set has %d (a folder of one quant, or --files)" % (model, len(gs), want))
    for g in gs:
        why = short(g)
        if why:
            raise SystemExit("not whole yet: " + why)
    return gs


def short(g: Gguf):
    need = g.data_start
    for _, typ, dims, off in g.tensors:
        if typ not in TYPES:
            raise SystemExit("%s: a tensor of ggml type %d, which maya-hy4 does not read" % (g.path.name, typ))
        need = max(need, g.data_start + off + nbytes(typ, dims))
    have = g.path.stat().st_size
    return None if have >= need else "%s: %d of %d bytes" % (g.path.name, have, need)


def geometry(kv: dict) -> dict:
    g = lambda k, d=None: kv.get("%s.%s" % (ARCH, k), d)  # noqa: E731
    n_all = int(g("block_count"))
    mtp = int(g("nextn_predict_layers", 0))
    heads = g("attention.head_count")
    heads = list(heads) if isinstance(heads, list) else [int(heads)] * n_all
    swa = [int(bool(x)) for x in g("attention.sliding_window_pattern")]
    swa += [0] * (n_all - len(swa))
    idx = [int(bool(x)) for x in (g("attention.indexer.types") or [])]
    n_layer = n_all - mtp
    idx += [1] * (n_all - len(idx))            # (the MTP block always owns an indexer: the hy_v4_flash layout)
    clamp = g("swiglu_clamp_exp", 0.0)
    clamp = list(clamp) if isinstance(clamp, list) else [float(clamp)] * n_all
    if len(heads) != n_all or len(clamp) != n_all:
        raise SystemExit("head_count / swiglu_clamp_exp: %d / %d values for %d blocks" % (len(heads), len(clamp), n_all))
    if any(int(x) != 1 for x in ([g("attention.head_count_kv")] if not isinstance(g("attention.head_count_kv"), list)
                                 else g("attention.head_count_kv"))):
        raise SystemExit("head_count_kv is not 1: not the MLA layout maya-hy4 runs")
    for il in range(n_layer):
        if idx[il] and swa[il]:
            raise SystemExit("layer %d is a sliding-window layer with an indexer" % il)
    geo = {
        "arch": ARCH,
        "n_layer": n_layer, "n_mtp": mtp, "n_embd": int(g("embedding_length")),
        "n_vocab": int(g("vocab_size", len(kv["tokenizer.ggml.tokens"]))),
        "context_length": int(g("context_length")),
        "n_ff": int(g("feed_forward_length")), "n_dense_lead": int(g("leading_dense_block_count", 0)),
        "n_expert": int(g("expert_count")), "n_expert_used": int(g("expert_used_count")),
        "n_ff_exp": int(g("expert_feed_forward_length")), "n_expert_shared": int(g("expert_shared_count", 1)),
        "n_expert_latent": int(g("expert_latent_length", 0) or g("embedding_length")),
        "expert_weights_scale": float(g("expert_weights_scale", 1.0)),
        "expert_weights_norm": int(bool(g("expert_weights_norm", False))),
        "expert_gating_func": int(g("expert_gating_func", 0)),
        "swiglu_limit": clamp,
        "rms_eps": float(g("attention.layer_norm_rms_epsilon")),
        "q_lora": int(g("attention.q_lora_rank")), "kv_lora": int(g("attention.kv_lora_rank")),
        "k_mla": int(g("attention.key_length_mla")), "v_mla": int(g("attention.value_length_mla")),
        "rope_dim": int(g("rope.dimension_count")),
        "rope_base": float(g("rope.freq_base")), "rope_base_swa": float(g("rope.freq_base_swa")),
        "n_swa": int(g("attention.sliding_window")),
        "heads": [int(x) for x in heads], "swa": swa, "indexer": idx,
        "ix_heads": int(g("attention.indexer.head_count", 0)), "ix_dim": int(g("attention.indexer.key_length", 0)),
        "ix_top_k": int(g("attention.indexer.top_k", 0)), "ix_kpool": int(g("attention.indexer.kpool", 0)),
        "ix_select_tail": int(bool(g("attention.indexer.kpool_select_tail", False))),
        "hc": int(g("hyper_connection.count")), "hc_eps": float(g("hyper_connection.epsilon")),
        "hc_magnitude": float(g("hyper_connection.magnitude")),
        "bos": int(kv["tokenizer.ggml.bos_token_id"]), "eos": int(kv["tokenizer.ggml.eos_token_id"]),
    }
    if geo["expert_gating_func"] != 2:
        raise SystemExit("expert_gating_func %d: maya-hy4 runs the sigmoid router (2)" % geo["expert_gating_func"])
    if geo["k_mla"] - geo["rope_dim"] < 1 or geo["kv_lora"] % 32 or geo["ix_kpool"] < 2 or \
            geo["ix_top_k"] % geo["ix_kpool"]:
        raise SystemExit("an attention geometry maya-hy4 does not run: %s" %
                         {k: geo[k] for k in ("k_mla", "rope_dim", "kv_lora", "ix_top_k", "ix_kpool")})
    return geo


def geometry_text(geo: dict, model: pathlib.Path) -> str:
    lines = []
    for k, v in geo.items():
        if isinstance(v, list):
            v = ",".join(repr(x) if isinstance(x, float) else str(x) for x in v)
        elif isinstance(v, float):
            v = repr(v)
        lines.append("%s %s" % (k, v))
    lines.append("model_dir %s" % model)
    return "\n".join(lines) + "\n"


def engram_bytes(kv: dict, tensors: dict, vocab: int) -> tuple:
    """(Engram layers, engram.bin's bytes - None: no Engram)"""
    g = lambda k, d=None: kv.get("%s.%s" % (ARCH, k), d)  # noqa: E731
    layers = list(g("ple.layers") or [])
    if not layers:
        return 0, None
    ngram, heads, hd = int(g("ple.ngram_size")), int(g("ple.heads_per_ngram")), int(g("embedding_length_per_layer_input"))
    mult, offs, primes = list(g("ple.layer_multipliers")), list(g("ple.head_offsets")), list(g("ple.head_vocab_sizes"))
    tm = list(g("engram.token_map"))
    cols, nm = (ngram - 1) * heads, len(layers)
    if len(mult) != nm * ngram or len(offs) != nm * cols or len(primes) != nm * cols:
        raise SystemExit("engram: %d multipliers, %d offsets, %d primes for %d layers x %d columns" %
                         (len(mult), len(offs), len(primes), nm, cols))
    if "per_layer_token_embd.weight" not in tensors:
        raise SystemExit("engram: the GGUF names Engram layers but has no per_layer_token_embd table")
    typ, dims = tensors["per_layer_token_embd.weight"][:2]
    if TYPES[typ][0] != "F8_E4M3" or dims[0] != hd:
        raise SystemExit("engram: the table is %s %s, maya-hy4 reads F8_E4M3 rows of %d" % (TYPES[typ][0], dims, hd))
    rows = dims[1]
    for c in range(nm * cols):
        if primes[c] < 2 or offs[c] + primes[c] > rows:
            raise SystemExit("engram: head %d's buckets [%d, %d) are outside the table's %d rows" %
                             (c, offs[c], offs[c] + primes[c], rows))
    pad_id = int(g("engram.pad_token_id"))
    if pad_id >= len(tm):
        raise SystemExit("engram: the pad token %d is outside the token map (%d)" % (pad_id, len(tm)))
    full = [int(x) for x in tm] + [-1] * max(0, vocab - len(tm))
    b = struct.pack("<4s7i", b"HYE1", vocab, nm, ngram, heads, hd, int(tm[pad_id]), rows)
    b += struct.pack("<%di" % vocab, *full[:vocab])
    b += struct.pack("<%di" % nm, *[int(x) for x in layers])
    b += struct.pack("<%dQ" % len(mult), *[int(x) & 0xFFFFFFFFFFFFFFFF for x in mult])
    b += struct.pack("<%dQ" % len(primes), *[int(x) for x in primes])
    b += struct.pack("<%dQ" % len(offs), *[int(x) for x in offs])
    return nm, b


def tokenizer_files(kv: dict) -> dict:
    """{file name: text} of the pack's tokenizer/ folder"""
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(tokens)
    merges = kv.get("tokenizer.ggml.merges") or []
    if len(set(tokens)) != len(tokens):
        raise SystemExit("the GGUF's vocabulary has duplicate token strings")
    if kv.get("tokenizer.ggml.model") != "gpt2" or kv.get("tokenizer.ggml.pre") != "hy_v4":
        raise SystemExit("the GGUF's tokenizer is %r / %r, not the byte-level BPE with the hy_v4 pre-tokenizer" %
                         (kv.get("tokenizer.ggml.model"), kv.get("tokenizer.ggml.pre")))
    special = {"bos": int(kv["tokenizer.ggml.bos_token_id"]), "eos": int(kv["tokenizer.ggml.eos_token_id"])}
    if "tokenizer.ggml.padding_token_id" in kv:
        special["pad"] = int(kv["tokenizer.ggml.padding_token_id"])
    files = {"vocab.json": json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
             "merges.txt": "\n".join(merges), "token_type.json": json.dumps([int(t) for t in types])}
    # the server reads Hy4's answers as GLM's (serve/frontend.py's ChatTemplate.hy4): each suffixed tag token (<think:
    # 6124c78e>, <tool_call:6124c78e>, <arg_key:6124c78e> ...) under GLM's spelling, the <tool_calls> wrapper as nothing
    aliases = {}
    for i, t in enumerate(tokens):
        m = re.fullmatch(r"<(/?)(think|tool_calls|tool_call|arg_key|arg_value):[0-9A-Za-z]+>", t)
        if m and int(types[i]) in (3, 4):
            aliases[str(i)] = "" if m.group(2) == "tool_calls" else "<%s%s>" % (m.group(1), m.group(2))
    files["tokenizer.json"] = json.dumps({"pre": "hy_v4", "special_ids": special, "aliases": aliases}, indent=1,
                                         ensure_ascii=False)
    tpl = kv.get("tokenizer.chat_template")
    if not tpl:
        raise SystemExit("the GGUF has no chat template")
    files["chat_template.jinja"] = tpl
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the folder of the GGUF set")
    ap.add_argument("--files", nargs="*", help="the set's shards (default: every .gguf in the folder)")
    ap.add_argument("--out", help="the pack's folder (default: <model>/maya-hy4)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser().resolve() if a.out else model / "maya-hy4"
    gs = find_shards(model, a.files)
    kv = {}
    for g in gs:
        for k, v in g.kv.items():
            kv.setdefault(k, v)
    geo = geometry(kv)
    tensors, lines = {}, []
    for i, g in enumerate(gs):
        lines.append("file %d %s" % (i, os.path.relpath(g.path, model)))
        for name, typ, dims, off in g.tensors:
            if name in tensors:
                raise SystemExit("%s twice in the set" % name)
            tensors[name] = (typ, dims, i, g.data_start + off)
    for name, (typ, dims, i, off) in tensors.items():
        lines.append("t %s %s %d %d %d %s" % (name, TYPES[typ][0], i, off, nbytes(typ, dims), " ".join(map(str, dims))))
    need = ["token_embd.weight", "output.weight", "output_norm.weight", "output_hc_fn.weight"]
    need += ["blk.%d.attn_q_a.weight" % il for il in range(geo["n_layer"])]
    missing = [n for n in need if n not in tensors]
    if missing:
        raise SystemExit("the set lacks %s" % ", ".join(missing[:5]))
    # everything checked before anything is written: a refused set leaves no part of a pack (tensors.txt last - the
    # file the engine and the setup look for)
    ne, engram = engram_bytes(kv, tensors, geo["n_vocab"])
    tok = tokenizer_files(kv)
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    for name, text in tok.items():
        (out / "tokenizer" / name).write_text(text, encoding="utf-8")
    if engram is not None:
        (out / "engram.bin").write_bytes(engram)
    elif (out / "engram.bin").exists():
        (out / "engram.bin").unlink()
    (out / "hy4.txt").write_text(geometry_text(geo, model))
    (out / "tensors.txt").write_text("\n".join(lines) + "\n")
    print("%s: %d tensors from %d files, %d layers (+%d MTP), %d Engram layers" %
          (out, len(tensors), len(gs), geo["n_layer"], geo["n_mtp"], ne))
    return 0


if __name__ == "__main__":
    sys.exit(main())

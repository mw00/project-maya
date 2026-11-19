"""tools/hy4/pack_gguf.py - maya-hy4's pack for a Hy4-flash GGUF set (Tencent's Hy4-flash, the GGUF's `hy_v4_flash`
layout: AngelSlim's Q4_K_M / Q3_K / IQ2 builds), beside the model's files (no weights copied).

    python tools/hy4/pack_gguf.py --model ~/models/Hy4-flash-GGUF [--mmproj FILE|none] [--mtp FILE|none]

writes <model>/maya-hy4/:
  hy4.txt       the geometry, one `key value` a line (lists comma-separated), from the GGUF's hy_v4_flash.* keys; with
                the vision tower, its vision.* keys (from the mmproj file's clip.* keys, and the vocabulary's image
                start / end / pad tokens)
  tensors.txt   `file <i> <path>` lines (relative to the model's folder, model_dir in hy4.txt), then `t <name> <type> <file> <offset> <bytes> <ne0> <ne1> ...` a tensor: the
                GGUF's own names and dims (ne0 innermost, as ggml counts them), type ggml's name (Q4_K, Q8_0, BF16,
                F32, F8_E4M3 ...), offset absolute in its file; the mmproj file (the vision tower: v.* and mm.*
                tensors) is the last file when there is one
  engram.bin    the Engram n-gram hash ('HYE1'): int32 vocab, modules, ngram, heads_per_ngram, head_dim, pad (the token
                map's value of the pad token), table_rows; int32 token_map[vocab]; int32 layers[modules]; uint64
                multipliers[modules][ngram], primes[modules][cols], offsets[modules][cols] (cols = (ngram - 1)
                heads_per_ngram; the offsets index the one table, per_layer_token_embd)
  tokenizer/    the server's tokenizer from the GGUF's vocabulary and merges (pre "hy_v4") and the model's own chat
                template (tokenizer.chat_template)

The vision tower: AngelSlim's mmproj-Hy4-flash-*.gguf (clip.projector_type hy_v4_flash) in the model's folder, found
there unless --mmproj names one (or `none`: text only).  The model's shards are the folder's other .gguf files.

The MTP draft block (maya-hy4's --mtp): in the set itself (AngelSlim's builds: hy_v4_flash.nextn_predict_layers 1), or -
for a set made from the public release, which has none - in a file of its own in the model's folder (general.type
"mtp": tools/hy4/mtp_gguf.py writes it from a set that has the block), found there unless --mtp names one (or `none`).
Its tensors join the pack as one more file and the geometry gains the block (n_mtp 1); a set with its own block keeps it.
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


def is_mmproj(g: Gguf) -> bool:
    return g.kv.get("general.type") == "mmproj" or g.kv.get("general.architecture") == "clip"


def is_mtp(g: Gguf) -> bool:
    return g.kv.get("general.type") == "mtp" and g.kv.get("general.architecture") == ARCH


def find_shards(model: pathlib.Path, names: list | None) -> list:
    paths = [model / n for n in names] if names else sorted(model.glob("*.gguf"))
    if not paths:
        raise SystemExit("%s: no .gguf files" % model)
    if names:
        gs = [Gguf(p) for p in paths]
    else:
        # the folder's other files (a vision projector: another architecture; a drafter: general.type "drafter") are
        # not the set's shards - left alone, said once; the set's own count is still checked below
        gs = []
        for p in paths:
            try:
                g = Gguf(p)
            except (Exception, SystemExit) as e:  # noqa: BLE001 - a file that is no GGUF this reader takes: not a shard
                print("%s: not read (%s), left out of the set" % (p.name, e))
                continue
            arch, kind = g.kv.get("general.architecture"), g.kv.get("general.type")
            if (arch is None or arch == ARCH) and kind in (None, "model"):
                gs.append(g)
            elif not is_mtp(g):                    # (the MTP block's own file: find_mtp's)
                print("%s: %s / %s, not a shard of the model, left out" % (p.name, arch, kind))
        if not gs:
            raise SystemExit("%s: no %s model .gguf files" % (model, ARCH))
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


def engram_scales(model: pathlib.Path, rows: int):
    """The Engram table's row scales beside the set (a table of per-row scaled codes: <name>.engram-scales.bf16, one
    BF16 a row) - its name, or None.  A file of another size is not this table's: refused."""
    found = sorted(model.glob("*.engram-scales.bf16"))
    if not found:
        return None
    if len(found) > 1 or found[0].stat().st_size != rows * 2:
        raise SystemExit("engram: %s is not the table's row scales (%d rows of BF16 expected)" % (found[0].name, rows))
    return found[0].name


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


def find_mmproj(model: pathlib.Path, arg: str | None) -> Gguf | None:
    """The vision tower's file: --mmproj's, else the folder's one mmproj GGUF of this projector (None: text only)."""
    if arg == "none":
        return None
    if arg:
        p = pathlib.Path(arg).expanduser()
        p = p if p.is_absolute() else (model / p if (model / p).exists() else p.resolve())
        cands = [Gguf(p)]
    else:
        cands = [g for g in (Gguf(p) for p in sorted(model.glob("*.gguf")) if "mmproj" in p.name.lower())
                 if is_mmproj(g)]
    cands = [g for g in cands if is_mmproj(g)]
    if arg and not cands:
        raise SystemExit("%s: not an mmproj GGUF" % arg)
    mine = [g for g in cands if g.kv.get("clip.projector_type") == ARCH]
    if arg and not mine:
        raise SystemExit("%s: clip.projector_type %r, not %s" % (arg, cands[0].kv.get("clip.projector_type"), ARCH))
    if len(mine) > 1:
        raise SystemExit("%s: %d mmproj files of %s (name one with --mmproj)" % (model, len(mine), ARCH))
    if not mine:
        return None
    why = short(mine[0])
    if why:
        raise SystemExit("not whole yet: " + why)
    return mine[0]


def find_mtp(model: pathlib.Path, arg: str | None) -> Gguf | None:
    """The MTP block's own file: --mtp's, else the folder's one GGUF of general.type "mtp" (None: there is none)."""
    if arg == "none":
        return None
    if arg:
        p = pathlib.Path(arg).expanduser()
        p = p if p.is_absolute() else (model / p if (model / p).exists() else p.resolve())
        cands = [Gguf(p)]
        if not is_mtp(cands[0]):
            raise SystemExit("%s: not a %s MTP block's file (general.type %r)" %
                             (arg, ARCH, cands[0].kv.get("general.type")))
    else:
        cands = []
        for p in sorted(model.glob("*.gguf")):
            try:
                g = Gguf(p)
            except (Exception, SystemExit):  # noqa: BLE001 - not a GGUF this reader takes: not the block's file
                continue
            if is_mtp(g):
                cands.append(g)
    if len(cands) > 1:
        raise SystemExit("%s: %d MTP block files (name one with --mtp)" % (model, len(cands)))
    if not cands:
        return None
    why = short(cands[0])
    if why:
        raise SystemExit("not whole yet: " + why)
    return cands[0]


# the keys a trunk and its MTP block must agree on (the block's matrices have the trunk's shapes), and the per-block ones
MTP_SAME = ("embedding_length", "vocab_size", "expert_count", "expert_used_count", "expert_feed_forward_length",
            "expert_latent_length", "expert_shared_count", "expert_gating_func", "attention.q_lora_rank",
            "attention.kv_lora_rank", "attention.key_length_mla", "attention.value_length_mla", "rope.dimension_count",
            "attention.indexer.head_count", "attention.indexer.key_length", "attention.layer_norm_rms_epsilon")
MTP_PER_BLOCK = ("attention.head_count", "swiglu_clamp_exp")


def with_mtp(kv: dict, m: Gguf) -> dict:
    """The set's keys with the block of the MTP file `m` after the trunk's last: block_count and nextn_predict_layers
    the file's, the per-block lists one entry longer (the file's own value for its block).  Refused when the file's
    block is not this trunk's next one, or its geometry another model's."""
    a = ARCH + "."
    n = int(kv[a + "block_count"])
    n_all, mtp = int(m.kv.get(a + "block_count", 0)), int(m.kv.get(a + "nextn_predict_layers", 0))
    if mtp != 1 or n_all - mtp != n:
        raise SystemExit("%s: the MTP block of a model of %d blocks, this set has %d" % (m.path.name, n_all - mtp, n))
    for k in MTP_SAME:
        if a + k in kv and a + k in m.kv and kv[a + k] != m.kv[a + k]:
            raise SystemExit("%s: %s %r, the set's is %r - not this model's MTP block" %
                             (m.path.name, k, m.kv[a + k], kv[a + k]))
    prefix = "blk.%d." % n
    odd = [name for name, _, _, _ in m.tensors if not name.startswith(prefix)]
    if odd or prefix + "nextn.eh_proj.weight" not in {t[0] for t in m.tensors}:
        raise SystemExit("%s: not block %d alone (%s)" % (m.path.name, n, odd[0] if odd else "no nextn.eh_proj"))
    out = dict(kv)
    out[a + "block_count"], out[a + "nextn_predict_layers"] = n_all, mtp
    for k in MTP_PER_BLOCK:
        v, mv = kv.get(a + k), m.kv.get(a + k)
        if mv is None:
            continue
        tail = list(mv)[n:n_all] if isinstance(mv, list) else [mv]
        if len(tail) != 1:
            raise SystemExit("%s: %s has %d values for %d blocks" % (m.path.name, k, len(mv), n_all))
        if isinstance(v, list):
            if len(v) != n:
                raise SystemExit("%s: %d values for %d blocks" % (k, len(v), n))
            out[a + k] = list(v) + tail
        elif v is not None and v != tail[0]:
            out[a + k] = [v] * n + tail
    return out


def vision_geometry(g: Gguf, tokens: list, n_embd: int) -> dict:
    """hy4.txt's vision.* keys from the mmproj's clip.* keys and the model's vocabulary."""
    kv = g.kv
    c = lambda k, d=None: kv.get("clip.vision." + k, kv.get("clip." + k, d))  # noqa: E731
    t = {name: (typ, dims) for name, typ, dims, _ in g.tensors}
    pos = t.get("v.position_embd.weight")
    if pos is None or "v.patch_embd.weight" not in t or "mm.0.weight" not in t or "v.image_newline" not in t:
        raise SystemExit("%s: not the hy_v4_flash vision tower's tensors" % g.path.name)
    grid = int(round(pos[1][1] ** 0.5))
    if grid * grid != pos[1][1]:
        raise SystemExit("%s: a position grid of %d rows is not square" % (g.path.name, pos[1][1]))

    def tok(prefix):
        ids = [i for i, s in enumerate(tokens) if s.startswith(prefix)]
        if len(ids) != 1:
            raise SystemExit("the vocabulary has %d tokens starting %s" % (len(ids), prefix))
        return ids[0]

    geo = {
        "vision.dim": int(c("embedding_length")), "vision.heads": int(c("attention.head_count")),
        "vision.ff": int(c("feed_forward_length")), "vision.layers": int(c("block_count")),
        "vision.patch": int(c("patch_size")), "vision.merge": int(c("spatial_merge_size", 2)),
        "vision.grid": grid, "vision.out_dim": int(c("projection_dim")),
        "vision.eps": float(c("attention.layer_norm_epsilon", 1e-5)),
        "vision.min_pixels": int(c("image_min_pixels", 4096)), "vision.max_pixels": int(c("image_max_pixels", 4194304)),
        "vision.mean": [float(x) for x in c("image_mean", [0.5] * 3)],
        "vision.std": [float(x) for x in c("image_std", [0.5] * 3)],
        "vision.image_token": tok("<｜hy_image▁pad"), "vision.image_start": tok("<｜hy_image▁start"),
        "vision.image_end": tok("<｜hy_image▁end"),
    }
    if geo["vision.out_dim"] != n_embd or geo["vision.merge"] != 2 or geo["vision.dim"] % geo["vision.heads"]:
        raise SystemExit("%s: a vision geometry maya-hy4 does not run (projection %d for a model of %d, merge %d)" %
                         (g.path.name, geo["vision.out_dim"], n_embd, geo["vision.merge"]))
    missing = ["v.blk.%d.%s" % (i, n) for i in range(geo["vision.layers"])
               for n in ("attn_q.weight", "ffn_down.weight") if "v.blk.%d.%s" % (i, n) not in t]
    if missing:
        raise SystemExit("%s: lacks %s" % (g.path.name, ", ".join(missing[:3])))
    return geo


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
    ap.add_argument("--mmproj", help="the vision tower's GGUF (default: the folder's mmproj file of Hy4-flash; "
                                     "none: text only)")
    ap.add_argument("--mtp", help="the MTP draft block's own GGUF, for a set without the block (default: the folder's "
                                  "file of general.type mtp; none: no drafts)")
    ap.add_argument("--engram-eps", type=float, help="the Engram norms' eps (default: the engine's, FP32's machine "
                    "epsilon; 0.0078125: the release run in BF16)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser().resolve() if a.out else model / "maya-hy4"
    gs = find_shards(model, a.files)
    vis = find_mmproj(model, a.mmproj)
    kv = {}
    for g in gs:
        for k, v in g.kv.items():
            kv.setdefault(k, v)
    mtp = None
    if int(kv.get("%s.nextn_predict_layers" % ARCH, 0)) == 0:
        mtp = find_mtp(model, a.mtp)
        if mtp is not None:
            kv = with_mtp(kv, mtp)
    elif a.mtp and a.mtp != "none":   # (a set with its own block keeps it; a block's file in its folder is not read)
        raise SystemExit("--mtp %s: the set has its own MTP block" % a.mtp)
    geo = geometry(kv)
    if vis is not None:
        geo.update(vision_geometry(vis, kv["tokenizer.ggml.tokens"], geo["n_embd"]))
    tensors, lines = {}, []
    for i, g in enumerate(gs + ([mtp] if mtp is not None else []) + ([vis] if vis is not None else [])):
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
    extra = ""
    if engram is not None:
        sc = engram_scales(model, tensors["per_layer_token_embd.weight"][1][1])
        if sc:
            extra += "engram_scales %s\n" % sc
        if a.engram_eps is not None:
            if not 0.0 < a.engram_eps <= 1.0:
                raise SystemExit("--engram-eps %r" % a.engram_eps)
            extra += "engram_eps %r\n" % a.engram_eps
    (out / "hy4.txt").write_text(geometry_text(geo, model) + extra)
    (out / "tensors.txt").write_text("\n".join(lines) + "\n")
    print("%s: %d tensors from %d files, %d layers (+%d MTP%s), %d Engram layers, %s" %
          (out, len(tensors), len(gs), geo["n_layer"], geo["n_mtp"],
           ": %s" % mtp.path.name if mtp is not None else "", ne,
           "the vision tower (%s)" % vis.path.name if vis is not None else "text only"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

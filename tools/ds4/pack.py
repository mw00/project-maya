"""tools/ds4/pack.py - the engine's pack for a DeepSeek-V4.1 EXL3 model, beside the model's files (no weights copied).

    python tools/ds4/pack.py --model ~/models/DSV4

writes <model>/maya-ds4/:
  ds4.txt       the geometry, `key value` lines (lists comma-separated), from config.json's text_config (and its
                vision_config as vision.<key>, image_token_id)
  tensors.txt   every tensor of the directory's safetensors (the index's 43 shards and the base repo's Engram shards
                47 / 48 beside them): `file <i> <name>` lines, then `t <name> <dtype> <file> <offset> <bytes> <dims...>`
  engram.bin    the Engram hash tables (tools/ds4/engram_hash.py): int32 magic 'DSE1', vocab, layers, max_ngram,
                heads, head_dim, pad (compressed); int32 token_map[vocab]; int64 mult[layers][max_ngram];
                int64 primes[layers][cols]; int64 offsets[layers][cols]   (cols = (max_ngram - 1) * heads)
  tokenizer/    the server's tokenizer (tools/strata_tokenizer.py's layout, pre "deepseek3") and the model repo's own
                chat_template.jinja (a port of DeepSeek's encoding/encoding.py, byte for byte its prompts) - this
                directory's chat_template.jinja only for a repo without one

A repo's exl3_k_map.json (each tensor group's EXL3 bits and the codebook) is checked against the tensors: the engine
reads every matrix's K from its trellis shape, so the map is a cross-check, not an input.

The compressed token map needs the `tokenizers` package (it normalizes every token as DeepSeek's training did).
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import engram_hash as EH  # noqa: E402
from st import SafeTensors  # noqa: E402

TEMPLATE = HERE / "chat_template.jinja"   # (a model repo without its own chat_template.jinja)
MUL1 = 0x83DCD12D                          # the engine's mul1 codebook multiplier
TOWER = ("vision.", "aligner.")            # the vision tower's and its aligner's tensors
MARKS = ("image_start", "image_newline", "image_end")   # the picture span's marker rows (BF16 too)
SHARD1 = "model-00001-of-00048.safetensors"   # DeepSeek's checkpoint: shard 1 holds only the tower, in BF16


def flat_patch(name: str, shape: tuple) -> tuple:
    """The patch embedding as the engine reads it, [hidden][3 * patch * patch] - a [hidden][3][p][p] copy is the same
    bytes in the same order."""
    if name == "vision.patch_embed.proj.weight" and len(shape) > 2:
        return (shape[0], int(np.prod(shape[1:])))
    return tuple(shape)


def vision_bf16(st, model: pathlib.Path, vdir: pathlib.Path | None) -> str | None:
    """The engine reads the vision tower in BF16 only: a quant that ships its own copy in another type (turboderp's
    EXL3 quants: F16) had its tensors count as missing and pictures off (#886).  Those tensors are taken from a BF16
    copy instead: DeepSeek's shard 1 in the model's folder (SafeTensors keeps a name's later copies in .dup), else
    vision_pack.py's folder (vdir: its vision.bin / vision.json, as a GGUF install makes).  The line to print: what was
    taken from where, or why pictures stay off.  None: the tower is BF16 already, or there is none."""
    bad = [k for k in st.t if (k.startswith(TOWER) or k in MARKS) and st.t[k][1] != "BF16"]
    if not bad:
        return None
    kinds = ", ".join(sorted({st.t[k][1] for k in bad}))
    src = set()
    for k in bad:
        e = next((e for e in st.dup.get(k, ()) if e[1] == "BF16"), None)
        if e is not None:
            fn, dt, shape, off, nb = e
            st.t[k] = (fn, dt, flat_patch(k, shape), off, nb)
            src.add(fn)
    left = [k for k in bad if st.t[k][1] != "BF16"]
    if left and vdir is not None and (vdir / "vision.json").is_file() and (vdir / "vision.bin").is_file():
        have = {e[0]: e for e in json.loads((vdir / "vision.json").read_text())["tensors"]}
        try:
            rel = os.path.relpath(vdir / "vision.bin", model).replace(os.sep, "/")
        except ValueError:     # (Windows: another drive)
            rel = None
        if rel is not None and " " not in rel and all(k in have and have[k][1] == "BF16" for k in left):
            if rel not in st.files:
                st.files.append(rel)
            for k in left:
                _, dt, off, nb, shape = have[k]
                st.t[k] = (rel, dt, flat_patch(k, tuple(shape)), off, nb)
            src.add(rel)
            left = []
    if not left:
        return "  the vision tower: %d tensors in BF16 from %s (the quant's own copy is %s, which the engine does not " \
               "read)" % (len(bad), ", ".join(sorted(src)), kinds)
    return "  WARNING - the vision tower is %s here and the engine reads it in BF16 only: pictures stay off.  Put shard " \
           "1 of DeepSeek's checkpoint (deepseek-ai/DeepSeek-V4.1-Flash-1017, %s, 0.97 GB: the tower in BF16) in %s " \
           "and run this again" % (kinds, SHARD1, model)


def need(st: SafeTensors, t: dict) -> list[str]:
    """The tensors the engine reads; a missing one stops the pack here instead of the engine's start."""
    miss = []
    for li in range(t["num_hidden_layers"]):
        L = "layers.%d." % li
        for n in ("attn.wq_a.trellis", "attn.wq_b.trellis", "attn.wkv.trellis", "attn.wo_b.trellis",
                  "attn.attn_sink", "attn.q_norm.weight", "attn.kv_norm.weight", "attn_norm.weight", "ffn_norm.weight",
                  "ffn.gate.weight", "ffn.gate.bias", "hc_attn_fn", "hc_ffn_fn", "ffn.shared_experts.w1.trellis"):
            # (a dense matrix as EXL3, or as BF16 / Q8_0 - fp8_dense.py's for a quant that kept DeepSeek's FP8)
            if not st.has(L + n) and not (n.endswith(".trellis") and st.has(L + n[:-len(".trellis")] + ".weight")):
                miss.append(L + n)
        if li in t["engram_layer_ids"] and not st.has(L + "engram.embed.weight"):
            miss.append(L + "engram.embed.weight (shards 47 / 48 of deepseek-ai/DeepSeek-V4.1-Flash-1017)")
        elif li in t["engram_layer_ids"] and st.t[L + "engram.embed.weight"][1] == "U8":
            # (a quant's own INT4 Engram tables, U8 + embed.scale: the tables are DeepSeek's, unchanged by the quant)
            miss.append(L + "engram.embed.weight: an INT4 (U8) layout the engine does not read - put DeepSeek's FP8 "
                        "shards 47 / 48 of deepseek-ai/DeepSeek-V4.1-Flash-1017 in their place")
    for n in ("embed.weight", "norm.weight", "head.trellis"):
        # (the head as EXL3, or kept as a BF16 head.weight - the engine reads either)
        if not st.has(n) and not (n == "head.trellis" and st.has("head.weight")):
            miss.append(n)
    return miss


def type_ordered(st: SafeTensors) -> str | None:
    """A layer whose experts share one layout (each part's K and pieces' bytes) is read by the engine as one region an
    expert, its trellis / suh / svh close together in one file.  Shards that write their tensors grouped by type (all
    trellis, then all suh, ...) put every expert's pieces far apart - the engine would stop at its start.
    -> the first such layer's message, None when every layer is fine."""
    layers: dict[str, list[int]] = {}
    for n in st.t:
        p, _, rest = n.partition(".ffn.experts.")
        if rest.endswith(".w1.trellis") and rest[:-len(".w1.trellis")].isdigit():
            layers.setdefault(p + ".ffn.experts.", []).append(int(rest[:-len(".w1.trellis")]))
    for P, es in layers.items():
        sigs, apart = set(), None
        for e in sorted(es):
            ts = [st.t.get("%s%d.%s.%s" % (P, e, r, q)) for r in ("w1", "w3", "w2") for q in ("trellis", "suh", "svh")]
            if any(x is None for x in ts):
                sigs = None   # (a missing piece: the engine names it)
                break
            sigs.add(tuple(x[4] for x in ts) + tuple(ts[i][2][-1] for i in (0, 3, 6)))
            nb = sum(x[4] for x in ts)
            lo, hi = min(x[3] for x in ts), max(x[3] + x[4] for x in ts)
            if apart is None and len({x[0] for x in ts}) == 1 and hi - lo > 2 * nb + (1 << 20):
                apart = e
        if sigs is not None and len(sigs) == 1 and apart is not None:
            return ("%s%d: its pieces are not together in their file (the shards' tensors ordered by type, not expert "
                    "by expert?) - rewrite those shards with each expert's tensors back to back (the same tensors, "
                    "only their order changes)" % (P, apart))
    return None


def write_engram(out: pathlib.Path, model: pathlib.Path, t: dict):
    lay = EH.Layout.from_config(t)
    tm, n = EH.compressed_token_map(model / "tokenizer.json")
    if n != t["engram_compressed_vocab_size"]:
        raise SystemExit("engram: the compressed vocab has %d ids, the config says %d" %
                         (n, t["engram_compressed_vocab_size"]))
    vocab = t["vocab_size"]
    full = np.zeros(vocab, dtype=np.int32)                      # ids past the tokenizer's (none in use) map to 0
    full[:len(tm)] = tm
    mult = EH.multipliers(lay, n)
    cols = (lay.max_ngram - 1) * lay.n_heads
    with out.open("wb") as f:
        f.write(struct.pack("<4s6i", b"DSE1", vocab, len(lay.layer_ids), lay.max_ngram, lay.n_heads, lay.head_dim,
                            int(tm[t["engram_pad_token_id"]])))
        f.write(full.astype("<i4").tobytes())
        f.write(mult.astype("<i8").tobytes())
        f.write(lay.flat_primes().astype("<i8").tobytes())
        f.write(lay.offsets().astype("<i8").tobytes())
    return len(lay.layer_ids), cols


def write_tokenizer(out: pathlib.Path, model: pathlib.Path):
    """vocab.json (id order), merges.txt, token_type.json, tokenizer.json {pre, special_ids} and the chat template."""
    tj = json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))
    vocab = tj["model"]["vocab"]
    added = {a["id"]: a for a in tj["added_tokens"]}
    n = max(max(vocab.values()), max(added)) + 1
    tokens = ["<｜unused▁%d｜>" % i for i in range(n)]       # (ids no token names: none in this vocabulary)
    types = [1] * n                                            # llama.cpp: 1 normal, 3 control, 4 user-defined
    for s, i in vocab.items():
        tokens[i] = s
    for i, a in added.items():
        tokens[i] = a["content"]
        types[i] = 3 if a["special"] else 4
    merges = tj["model"]["merges"]
    merges = [m if isinstance(m, str) else " ".join(m) for m in merges]
    out.mkdir(parents=True, exist_ok=True)
    if len(set(tokens)) != n:
        raise SystemExit("tokenizer.json: duplicate token strings")
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types))
    (out / "tokenizer.json").write_text(json.dumps({"pre": "deepseek3", "special_ids": {
        "bos": 0, "eos": 1, "pad": 2}}, indent=1))
    tpl = model / "chat_template.jinja"
    tpl = tpl if tpl.exists() else TEMPLATE
    if tpl.exists():
        (out / "chat_template.jinja").write_text(tpl.read_text(encoding="utf-8"), encoding="utf-8")
    return tpl


def check_k_map(model: pathlib.Path, st: SafeTensors) -> int | None:
    """exl3_k_map.json (if the repo has one) against every EXL3 matrix: its K (the trellis's last dim / 16) and the
    codebook; a mismatch stops the pack.  -> the matrices checked, None without a map."""
    path = model / "exl3_k_map.json"
    if not path.exists():
        return None
    km = json.loads(path.read_text())
    if km.get("codebook", "mul1") != "mul1" or int(km.get("mul1_multiplier", MUL1)) != MUL1:
        raise SystemExit("%s: codebook %r, multiplier %r - the engine decodes mul1 (0x%X) alone" %
                         (path, km.get("codebook"), km.get("mul1_multiplier"), MUL1))

    def want(name: str):
        if name.startswith("mtp."):
            return km.get("mtp_bits") if ".ffn.experts." in name else km.get("head_bits")
        if name.startswith("head."):
            return km.get("head_bits")
        parts = name.split(".")
        if parts[0] != "layers" or len(parts) < 4:
            return None
        L, rest = parts[1], ".".join(parts[2:-1])
        if rest.startswith("ffn.experts."):
            return km["routed"].get(L)
        if rest.startswith("ffn.shared_experts."):
            override = (km.get("shared_overrides") or {}).get(L)
            if isinstance(override, dict) and "note" in override:
                return None
            return override if override is not None else km["shared"].get(L)
        if rest.startswith("engram."):
            return (km.get("engram") or {}).get(L)
        if rest == "attn.indexer.wk":
            return km.get("indexer_wk_bits", km.get("attn_default"))
        if L == "0" and rest in ("attn.wq_a", "attn.wkv"):
            return km.get("head_bits", km.get("attn_default"))
        return km.get("attn_default")

    bad, n = [], 0
    for name, (_, _, shape, _, _) in st.t.items():
        if not name.endswith(".trellis"):
            continue
        w = want(name)
        if w is None:
            continue
        n += 1
        if shape[-1] // 16 != int(w):
            bad.append("%s: K %d, the map %d" % (name, shape[-1] // 16, int(w)))
    if bad:
        raise SystemExit("%s disagrees with the tensors:\n  %s" % (path, "\n  ".join(bad[:20])))
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", help="default: <model>/maya-ds4")
    ap.add_argument("--vision", help="vision_pack.py's folder, the BF16 vision tower for a quant whose own is not BF16 "
                                     "and without DeepSeek's shard 1 beside it (default: <model>/../vision)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser() if a.out else model / "maya-ds4"
    cfg = json.loads((model / "config.json").read_text())
    t = cfg["text_config"]
    q = cfg.get("quantization_config", {})
    if q.get("quant_method") != "exl3":
        raise SystemExit("%s: not an EXL3 model (quant_method %r)" % (model, q.get("quant_method")))
    st = SafeTensors(model)
    dense = model / "maya-dense-bf16.safetensors"   # (fp8_dense.py: the FP8 dense weights as BF16, in their place)
    if dense.is_file():
        with dense.open("rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            h = json.loads(f.read(n))
        for k, v in h.items():
            if k != "__metadata__":
                a, b = v["data_offsets"]
                st.t[k] = (dense.name, v["dtype"], tuple(v["shape"]), 8 + n + a, b - a)
        if dense.name not in st.files:
            st.files.append(dense.name)
    # (an FP8 tensor the EXL3 set also has quantized is not the pack's: the Engram tables' shards from DeepSeek's repo
    #  carry layers.1/14.engram.wkv.weight in FP8 beside the EXL3 set's own wkv - #376)
    elif any(v[1] == "F8_E4M3" and not k.startswith("vision.") and ".engram.embed." not in k and
             not (k.endswith(".weight") and k[:-len("weight")] + "trellis" in st.t) for k, v in st.t.items()):
        raise SystemExit("%s keeps dense weights in FP8: python tools/ds4/fp8_dense.py --model %s first" % (model, model))
    vis = vision_bf16(st, model, pathlib.Path(a.vision).expanduser().resolve() if a.vision else model.parent / "vision")
    miss = need(st, t)
    if miss:
        raise SystemExit("missing tensors:\n  " + "\n  ".join(miss[:20]))
    apart = type_ordered(st)
    if apart:
        raise SystemExit(apart)
    out.mkdir(parents=True, exist_ok=True)

    with (out / "ds4.txt").open("w") as f:
        for k, v in t.items():
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, (int, float)):
                f.write("%s %s\n" % (k, v))
            elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                f.write("%s %s\n" % (k, ",".join(str(x) for x in v)))
            elif isinstance(v, dict):
                for k2, v2 in v.items():
                    if isinstance(v2, (int, float)) and not isinstance(v2, bool):
                        f.write("%s.%s %s\n" % (k, k2, v2))
        # the vision tower's geometry (vision.<key>) and the token an image's span carries
        for k, v in (cfg.get("vision_config") or {}).items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                f.write("vision.%s %s\n" % (k, v))
        if isinstance(cfg.get("image_token_id"), int):
            f.write("image_token_id %d\n" % cfg["image_token_id"])
        f.write("model_dir %s\n" % model)
    with (out / "tensors.txt").open("w") as f:
        for i, fn in enumerate(st.files):
            f.write("file %d %s\n" % (i, fn))
        for name, (fn, dt, shape, off, nb) in st.t.items():
            f.write("t %s %s %d %d %d %s\n" % (name, dt, st.files.index(fn), off, nb, " ".join(map(str, shape))))
    nk = check_k_map(model, st)
    nl, cols = write_engram(out / "engram.bin", model, t)
    tpl = write_tokenizer(out / "tokenizer", model)
    print("%s: %d tensors in %d files, Engram %d layers x %d hash columns -> %s" %
          (model.name, len(st.t), len(st.files), nl, cols, out))
    print("  chat template: %s%s" % (tpl, "" if tpl.parent == model else " (the repo has none)"))
    if vis:
        print(vis)
    if nk is not None:
        print("  exl3_k_map.json: every one of %d EXL3 matrices' K agrees" % nk)
    return 0


if __name__ == "__main__":
    sys.exit(main())

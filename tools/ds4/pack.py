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
    for n in ("embed.weight", "norm.weight", "head.trellis"):
        if not st.has(n):
            miss.append(n)
    return miss


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
            return km.get("mtp_bits")
        if name.startswith("head."):
            return km.get("head_bits")
        parts = name.split(".")
        if parts[0] != "layers" or len(parts) < 4:
            return None
        L, rest = parts[1], ".".join(parts[2:-1])
        if rest.startswith("ffn.experts."):
            return km["routed"].get(L)
        if rest.startswith("ffn.shared_experts."):
            return (km.get("shared_overrides") or {}).get(L, km["shared"].get(L))
        if rest.startswith("engram."):
            return (km.get("engram") or {}).get(L)
        if rest == "attn.indexer.wk":
            return km.get("indexer_wk_bits", km.get("attn_default"))
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
    elif any(v[1] == "F8_E4M3" and not k.startswith("vision.") and ".engram.embed." not in k for k, v in st.t.items()):
        raise SystemExit("%s keeps dense weights in FP8: python tools/ds4/fp8_dense.py --model %s first" % (model, model))
    miss = need(st, t)
    if miss:
        raise SystemExit("missing tensors:\n  " + "\n  ".join(miss[:20]))
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
    if nk is not None:
        print("  exl3_k_map.json: every one of %d EXL3 matrices' K agrees" % nk)
    return 0


if __name__ == "__main__":
    sys.exit(main())

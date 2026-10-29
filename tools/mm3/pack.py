"""tools/mm3/pack.py - the engine's pack for a MiniMax-M3.1-Flash EXL3 model, beside the model's files (no weights copied).

    python tools/mm3/pack.py --model ~/models/MiniMax-M3.1-Flash-EXL3-3.5bpw

writes <model>/maya-mm3/:
  mm3.txt       the geometry, `key value` lines (lists comma-separated), from config.json's text_config (its
                sparse_attention_config flattened to sparse_attention_config.<key>), the DSpark draft's config as
                dspark.<key> (dflash_config.<key> as dspark.dflash.<key>), the special token ids the engine needs and
                the per-layer EXL3 bits of the experts (exl3.<layer>.w1 / w3 / w2)
  tensors.txt   every tensor of the model's safetensors and of dspark/'s: `file <i> <path>` lines (paths relative to
                the model directory), then `t <name> <dtype> <file> <offset> <bytes> <dims...>` - the draft's names
                prefixed `dspark.`
  tokenizer/    the server's tokenizer (tools/strata_tokenizer.py's layout, pre "minimax") and Maya's chat template
                (this directory's chat_template.jinja: MiniMax's own plus thinking turned off - the same for every
                quant, whichever template its files carry)

The pack stops on anything the engine does not cover: an EXL3 matrix whose codebook is not mul1 or whose bits are
not whole, experts of one layer at different bits for the same matrix, an index-value branch, init blocks.
The norm weights stay as stored (Gemma's weight + 1 is the engine's, at load).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from st import SafeTensors  # noqa: E402

TEMPLATE = HERE / "chat_template.jinja"     # (Maya's: MiniMax's own template plus thinking off)
MUL1 = 0x83DCD12D                          # the mul1 codebook multiplier, the one the engine decodes
P = "language_model.model."


def need(st: SafeTensors, t: dict) -> list[str]:
    """The tensors the engine reads; a missing one stops the pack here instead of the engine's start."""
    sa = t.get("sparse_attention_config") or {}
    sparse = sa.get("sparse_attention_freq", [0] * t["num_hidden_layers"])
    miss = []

    def want(name: str):
        if not st.has(name):
            miss.append(name)

    for li in range(t["num_hidden_layers"]):
        L = "%slayers.%d." % (P, li)
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight", "self_attn.q_norm.weight",
                  "self_attn.k_norm.weight"):
            want(L + n)
        for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
            want(L + "self_attn.%s.trellis" % n)
        if sparse[li]:
            for n in ("index_q_proj.weight", "index_k_proj.weight", "index_q_norm.weight", "index_k_norm.weight"):
                want(L + "self_attn." + n)
        if t["moe_layer_freq"][li]:
            m = L + "block_sparse_moe."
            want(m + "gate.weight")
            want(m + "e_score_correction_bias")
            for n in ("gate_proj", "up_proj", "down_proj"):
                want(m + "shared_experts.%s.trellis" % n)
            for e in range(t["num_local_experts"]):
                for n in ("w1", "w2", "w3"):
                    want(m + "experts.%d.%s.trellis" % (e, n))
        else:
            for n in ("gate_proj", "up_proj", "down_proj"):
                want(L + "mlp.%s.trellis" % n)
    want(P + "embed_tokens.weight")
    want(P + "norm.weight")
    want("language_model.lm_head.trellis")
    return miss


def check_exl3(st: SafeTensors) -> tuple[int, list[str]]:
    """Every EXL3 matrix: mul1, whole bits, suh / svh beside it. -> (matrices, problems)."""
    bad, n = [], 0
    for name, (_, _, shape, _, _) in st.t.items():
        if not name.endswith(".trellis"):
            continue
        n += 1
        pre = name[:-len(".trellis")]
        if not st.has(pre + ".mul1"):
            bad.append("%s: codebook %s, the engine decodes mul1" % (pre, "mcg" if st.has(pre + ".mcg") else "3inst"))
        elif int(st.raw(pre + ".mul1").view("<u4")[0]) != MUL1:
            bad.append("%s: mul1 multiplier 0x%X" % (pre, int(st.raw(pre + ".mul1").view("<u4")[0])))
        if shape[-1] % 16:
            bad.append("%s: %g bits (fractional trellis)" % (pre, shape[-1] / 16))
        for s in (".suh", ".svh"):
            if not st.has(pre + s):
                bad.append(pre + s + " missing")
    return n, bad


def expert_bits(st: SafeTensors, t: dict) -> tuple[dict, list[str]]:
    """{layer: {w1: K, w3: K, w2: K}} for the MoE layers; a layer whose experts differ for one matrix is a problem."""
    out, bad = {}, []
    for li in range(t["num_hidden_layers"]):
        if not t["moe_layer_freq"][li]:
            continue
        m = "%slayers.%d.block_sparse_moe.experts." % (P, li)
        out[li] = {}
        for w in ("w1", "w3", "w2"):
            ks = {st.t["%s%d.%s.trellis" % (m, e, w)][2][-1] // 16 for e in range(t["num_local_experts"])}
            if len(ks) != 1:
                bad.append("layer %d %s: experts at bits %s" % (li, w, sorted(ks)))
            out[li][w] = max(ks)
    return out, bad


def write_tokenizer(out: pathlib.Path, model: pathlib.Path) -> dict:
    """vocab.json (id order), merges.txt, token_type.json, tokenizer.json {pre, special_ids} and the chat template."""
    tj = json.loads((model / "tokenizer.json").read_text(encoding="utf-8"))
    pre = tj.get("pre_tokenizer") or {}
    split = [p for p in pre.get("pretokenizers", []) if p.get("type") == "Split"]
    if (tj.get("normalizer") or {}).get("type") != "NFC" or len(split) != 1:
        raise SystemExit("tokenizer.json: not the NFC + one-Split pre-tokenizer the `minimax` scheme transcribes")
    vocab = tj["model"]["vocab"]
    added = {a["id"]: a for a in tj["added_tokens"]}
    n = max(max(vocab.values()), max(added)) + 1
    tokens = ["<|unused_%d|>" % i for i in range(n)]
    types = [1] * n                                            # llama.cpp: 1 normal, 3 control, 4 user-defined
    for s, i in vocab.items():
        tokens[i] = s
    for i, a in added.items():
        tokens[i] = a["content"]
        types[i] = 3 if a["special"] else 4
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    if len(set(tokens)) != n:
        raise SystemExit("tokenizer.json: duplicate token strings")
    stm = json.loads((model / "special_tokens_map.json").read_text())
    ids = {a["content"]: i for i, a in added.items()}

    def tok(v):
        return ids[v["content"] if isinstance(v, dict) else v]

    special = {k.replace("_token", ""): tok(v) for k, v in stm.items() if k.endswith("_token")}
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps(types))
    (out / "tokenizer.json").write_text(json.dumps({"pre": "minimax", "special_ids": special}, indent=1))
    (out / "chat_template.jinja").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    return special


def flat(f, prefix: str, d: dict):
    for k, v in d.items():
        if isinstance(v, bool):
            v = int(v)
        if isinstance(v, (int, float)):
            f.write("%s%s %s\n" % (prefix, k, v))
        elif isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v):
            f.write("%s%s %s\n" % (prefix, k, ",".join(str(int(x) if isinstance(x, bool) else x) for x in v)))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", help="default: <model>/maya-mm3")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser() if a.out else model / "maya-mm3"
    cfg = json.loads((model / "config.json").read_text())
    t = cfg["text_config"]
    q = cfg.get("quantization_config", {})
    if q.get("quant_method") != "exl3":
        raise SystemExit("%s: not an EXL3 model (quant_method %r)" % (model, q.get("quant_method")))
    sa = t.get("sparse_attention_config") or {}
    if sa.get("sparse_init_block", 0):
        raise SystemExit("sparse_init_block %r: init blocks are not covered" % sa["sparse_init_block"])
    if any(s and not d for s, d in zip(sa.get("sparse_attention_freq", []), sa.get("sparse_disable_index_value", []))):
        raise SystemExit("an index-value branch (sparse_disable_index_value 0) is not covered")
    st = SafeTensors(model)
    miss = need(st, t)
    if miss:
        raise SystemExit("missing tensors (%d):\n  %s" % (len(miss), "\n  ".join(miss[:20])))
    nmat, bad = check_exl3(st)
    bits, bad2 = expert_bits(st, t)
    if bad or bad2:
        raise SystemExit("not covered:\n  " + "\n  ".join((bad + bad2)[:20]))
    files = [(fn, fn, st) for fn in st.files]
    names = [(name, name, st) for name in st.t]
    dsp = model / "dspark"
    dcfg = None
    if (dsp / "model.safetensors.index.json").exists():
        ds = SafeTensors(dsp)
        files += [(fn, "dspark/" + fn, ds) for fn in ds.files]
        names += [(name, "dspark." + name, ds) for name in ds.t]
        dcfg = json.loads((dsp / "config.json").read_text())
    out.mkdir(parents=True, exist_ok=True)
    special = write_tokenizer(out / "tokenizer", model)
    gen = json.loads((model / "generation_config.json").read_text())

    with (out / "mm3.txt").open("w") as f:
        flat(f, "", t)
        flat(f, "sparse_attention_config.", sa)
        f.write("bos_token_id %d\n" % gen["bos_token_id"])
        f.write("eos_token_id %d\n" % gen["eos_token_id"])
        for k in ("image_token_index", "video_token_index"):
            if isinstance(cfg.get(k), int):
                f.write("%s %d\n" % (k, cfg[k]))
        for k, v in special.items():
            f.write("special.%s %d\n" % (k, v))
        for li, b in bits.items():
            f.write("exl3.%d %d,%d,%d\n" % (li, b["w1"], b["w3"], b["w2"]))
        vc = cfg.get("vision_config")
        if vc and "vision_tower.vision_model.pre_layrnorm.weight" in st.t:
            # the vision tower's geometry, its preprocessor's and the image span's tokens (image_processor.py's
            # smart_resize: sides multiples of patch x merge, max_pixels 672 x 672 and min_pixels 4 x 28 x 28 are its
            # class defaults - preprocessor_config.json does not set them)
            pp = json.loads((model / "preprocessor_config.json").read_text())
            added = {a["content"]: a["id"] for a in json.loads((model / "tokenizer.json").read_text(
                encoding="utf-8"))["added_tokens"]}
            flat(f, "vision.", vc)
            flat(f, "vision.", vc.get("img_token_compression_config") or {})
            flat(f, "vision.", {"image_mean": pp["image_mean"], "image_std": pp["image_std"],
                                "projector_hidden_size": cfg["projector_hidden_size"], "max_pixels": 672 * 672,
                                "min_pixels": 4 * 28 * 28, "image_start": added["]<]start of image[>["],
                                "image_end": added["]<]end of image[>["], "image_token": added["]<]image[>["]})
        if dcfg is not None:
            flat(f, "dspark.", {k: v for k, v in dcfg.items() if k not in ("text_config", "vision_config")})
            flat(f, "dspark.dflash.", dcfg.get("dflash_config") or {})
            flat(f, "dspark.cfg.", dcfg.get("dspark_config") or {})
        f.write("model_dir %s\n" % model)
    fidx = {}
    with (out / "tensors.txt").open("w") as f:
        for i, (fn, path, src) in enumerate(files):
            fidx[(id(src), fn)] = i
            f.write("file %d %s\n" % (i, path))
        for name, outname, src in names:
            fn, dt, shape, off, nb = src.t[name]
            f.write("t %s %s %d %d %d %s\n" % (outname, dt, fidx[(id(src), fn)], off, nb, " ".join(map(str, shape))))
    ks = sorted({tuple(b.values()) for b in bits.values()})
    print("%s: %d tensors in %d files (%d EXL3 matrices, all mul1, whole bits) -> %s" %
          (model.name, len(names), len(files), nmat, out))
    print("  experts' bits (w1, w3, w2) per layer: %s" % ", ".join("%d/%d/%d" % k for k in ks))
    print("  special ids: %s" % special)
    print("  DSpark draft: %s" % ("%d tensors" % sum(1 for n in names if n[1].startswith("dspark."))
                                   if dcfg else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

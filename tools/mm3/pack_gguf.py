"""tools/mm3/pack_gguf.py - the engine's pack for a MiniMax-M3.1-Flash GGUF set (Project Maya's MiniMax-M3.1-Flash-Maya-XS /
-S: llama.cpp's `minimax-m3` layout, dense matrices in Q4_K..Q8_0, routed experts in IQ1_S..IQ3_XXS), beside the model's
files (no weights copied).

    python tools/mm3/pack_gguf.py --model ~/models/MiniMax-M3.1-Flash-Maya-XS

writes <model>/maya-mm3/ in pack.py's format, so maya-mm3 reads both:
  mm3.txt       the geometry under config.json's text_config names (from the GGUF's minimax-m3.* keys; swiglu_alpha /
                swiglu_limit MiniMax-M3's 1.702 / 7, as llama.cpp's graph holds them), format gguf, quant_name, model_dir
  tensors.txt   every tensor under the checkpoint's names (language_model.model.layers.N.self_attn.q_proj.weight ...):
                `file <i> <name>` lines, then `t <name> <dtype> <file> <offset> <bytes> <dims...>` - dims outermost first
                (a matrix [out][in]), dtype ggml's type name (Q6_K, IQ1_S, F32 ...); the routed experts an entry each
                (layers.N.block_sparse_moe.experts.E.w1.weight: expert E's slice of the GGUF's stacked tensor)
  tokenizer/    the server's tokenizer from the GGUF's vocabulary and merges (pre "minimax"), and Maya's chat template
                (this directory's chat_template.jinja - the same for every MiniMax pack)

MiniMax-M3.1 runs its sparse attention's indexer in all 60 layers; stock llama.cpp's `minimax-m3` (M3) creates indexers
only from the first MoE layer on, so the Maya sets keep layers 0-2's (12 tensors) in a side file beside the shards,
<name>-maya-indexers.bin (a GGUF; .gguf before v1.1.31 - the Hub took a lone .gguf for the model): this pack takes
it (--indexers names another).  Without it the pack stops: the engine runs
the model as MiniMax defines it, never those layers on full attention in silence.

The GGUF's norms already hold Gemma's 1 + w (llama.cpp's converter adds it): mm3.txt says so (norms_plus_one 1) and the
engine adds nothing.

A GGUF set has no vision tower and no DSpark draft.  Both come with the EXL3 repo (Mia-AiLab's, tools/mm3/download.json)
as files of their own - model-vision-00001-of-00001.safetensors with config.json / preprocessor_config.json, and dspark/
- which do not depend on the experts' quant: --companions names a folder holding them (the setup downloads them there:
<models>/MiniMax-M3.1-Flash-companions, the repo's own layout; a whole EXL3 download beside the GGUF serves as well), and
the pack indexes them as tools/mm3/pack.py does - pictures and drafting with the GGUF's weights.
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
TEMPLATE = HERE / "chat_template.jinja"

_spec = importlib.util.spec_from_file_location("ds4_pack_gguf", HERE.parent / "ds4" / "pack_gguf.py")
_ds4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_ds4)
Gguf, TYPES = _ds4.Gguf, _ds4.TYPES   # (maya-ds4's GGUF reader: the shards' metadata and tensor infos)
sys.path.insert(0, str(HERE))
import pack as exl3_pack  # noqa: E402  (tools/mm3/pack.py: its mm3.txt writer for the vision / DSpark keys)
from st import SafeTensors  # noqa: E402

COMPANIONS = "MiniMax-M3.1-Flash-companions"   # (the setup's folder of the EXL3 repo's vision tower and DSpark draft)
EXL3_DIR = "MiniMax-M3.1-Flash-EXL3-3.5bpw"    # (a whole EXL3 download: maya.py's MM3_DIR)


def st_tensors(path: pathlib.Path) -> list:
    """A .safetensors file's tensors: [(name, dtype, shape, absolute offset, bytes)]."""
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    return [(k, v["dtype"], list(v["shape"]), 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0])
            for k, v in h.items() if k != "__metadata__"]


def find_companions(model: pathlib.Path, explicit: str | None) -> pathlib.Path | None:
    """The folder of the EXL3 repo's vision tower / DSpark draft: --companions, else beside the model (the setup's, or
    a whole EXL3 download), else None."""
    if explicit:
        d = pathlib.Path(explicit).expanduser().resolve()
        if not d.is_dir():
            raise SystemExit("%s: not a folder (--companions)" % d)
        return d
    for d in (model.parent / COMPANIONS, model.parent / EXL3_DIR):
        if (d / "config.json").is_file():
            return d
    return None


def companion_parts(comp: pathlib.Path, rel, tokens: list, f_index: int) -> tuple:
    """(files, tensors, vision keys, dspark config) of the companions folder: the vision tower when its file and
    configs are whole there, the DSpark draft when dspark/ is."""
    files, tensors, vision, dcfg = [], [], {}, None
    cfg = json.loads((comp / "config.json").read_text()) if (comp / "config.json").is_file() else {}
    vfiles = sorted(comp.glob("model-vision-*.safetensors"))
    pp = comp / "preprocessor_config.json"
    if vfiles and pp.is_file() and cfg.get("vision_config"):
        vt = [(fp, t) for fp in vfiles for t in st_tensors(fp)]
        if any(t[0] == "vision_tower.vision_model.pre_layrnorm.weight" for _, t in vt):
            for fp in vfiles:
                files.append(rel(fp))
            for fp, (name, dt, shape, off, nb) in vt:
                tensors.append((name, dt, f_index + files.index(rel(fp)), off, nb, shape))
            ppj = json.loads(pp.read_text())
            tok = {t: i for i, t in enumerate(tokens)}
            vc = cfg["vision_config"]
            vision = {"_vc": vc, "_comp": vc.get("img_token_compression_config") or {},
                      "_extra": {"image_mean": ppj["image_mean"], "image_std": ppj["image_std"],
                                 "projector_hidden_size": cfg["projector_hidden_size"], "max_pixels": 672 * 672,
                                 "min_pixels": 4 * 28 * 28, "image_start": tok["]<]start of image[>["],
                                 "image_end": tok["]<]end of image[>["], "image_token": tok["]<]image[>["]},
                      "_ids": {k: cfg[k] for k in ("image_token_index", "video_token_index") if isinstance(cfg.get(k), int)}}
    dsp = comp / "dspark"
    if (dsp / "model.safetensors.index.json").is_file() and (dsp / "config.json").is_file():
        ds = SafeTensors(dsp)
        base = f_index + len(files)
        for fn in ds.files:
            files.append(rel(dsp / fn))
        for name, (fn, dt, shape, off, nb) in ds.t.items():
            tensors.append(("dspark." + name, dt, base + ds.files.index(fn), off, nb, list(shape)))
        dcfg = json.loads((dsp / "config.json").read_text())
    return files, tensors, vision, dcfg

ARCH = "minimax-m3"
P = "language_model.model."
# GGUF name (blk.N. stripped) -> the checkpoint's under layers.N. (pack.py / the engine)
LAYER = {
    "attn_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "indexer.q_proj.weight": "self_attn.index_q_proj.weight",
    "indexer.k_proj.weight": "self_attn.index_k_proj.weight",
    "indexer.q_norm.weight": "self_attn.index_q_norm.weight",
    "indexer.k_norm.weight": "self_attn.index_k_norm.weight",
    "ffn_gate_inp.weight": "block_sparse_moe.gate.weight",
    "exp_probs_b.bias": "block_sparse_moe.e_score_correction_bias",
    "ffn_gate_shexp.weight": "block_sparse_moe.shared_experts.gate_proj.weight",
    "ffn_up_shexp.weight": "block_sparse_moe.shared_experts.up_proj.weight",
    "ffn_down_shexp.weight": "block_sparse_moe.shared_experts.down_proj.weight",
    "ffn_gate_exps.weight": "block_sparse_moe.experts.w1",   # [n_exp][ff][dim]: an entry an expert
    "ffn_up_exps.weight": "block_sparse_moe.experts.w3",
    "ffn_down_exps.weight": "block_sparse_moe.experts.w2",   # [n_exp][dim][ff]
    "ffn_gate.weight": "mlp.gate_proj.weight",                # (the leading dense layers)
    "ffn_up.weight": "mlp.up_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
}
GLOBAL = {"token_embd.weight": P + "embed_tokens.weight", "output_norm.weight": P + "norm.weight",
          "output.weight": "language_model.lm_head.weight"}
INDEXER = ("indexer.q_proj.weight", "indexer.k_proj.weight", "indexer.q_norm.weight", "indexer.k_norm.weight")
SWIGLU_ALPHA, SWIGLU_LIMIT = 1.702, 7.0   # MiniMax-M3's (config.json's; llama.cpp's LLM_FFN_SWIGLU_OAI_MOE)


def geometry(kv: dict, vocab: int, shared_ff: int) -> dict:
    """config.json's text_config names (what the engine reads) from the GGUF's minimax-m3.* keys."""
    g = lambda k, d=None: kv.get("%s.%s" % (ARCH, k), d)  # noqa: E731
    nl, lead = int(g("block_count")), int(g("leading_dense_block_count", 0))
    hd, rot = int(g("attention.key_length")), int(g("rope.dimension_count"))
    return {
        "hidden_size": g("embedding_length"),
        "num_hidden_layers": nl,
        "num_attention_heads": g("attention.head_count"),
        "num_key_value_heads": g("attention.head_count_kv"),
        "head_dim": hd,
        "rotary_dim": rot,
        "partial_rotary_factor": rot / hd,
        "vocab_size": vocab,
        "num_local_experts": g("expert_count"),
        "num_experts_per_tok": g("expert_used_count"),
        "intermediate_size": g("expert_feed_forward_length"),
        "dense_intermediate_size": g("feed_forward_length"),
        "shared_intermediate_size": shared_ff * int(g("expert_shared_count", 1)),
        "rope_theta": g("rope.freq_base"),
        "rms_norm_eps": g("attention.layer_norm_rms_epsilon"),
        "routed_scaling_factor": g("expert_weights_scale", 1.0),
        "swiglu_alpha": g("swiglu_alpha", SWIGLU_ALPHA),
        "swiglu_limit": g("swiglu_limit", SWIGLU_LIMIT),
        "max_position_embeddings": g("context_length"),
        "moe_layer_freq": [0] * lead + [1] * (nl - lead),
        "sparse_attention_config.sparse_attention_freq": [1] * nl,
        "sparse_attention_config.sparse_disable_index_value": [1] * nl,
        "sparse_attention_config.sparse_num_index_heads": g("attention.indexer.head_count"),
        "sparse_attention_config.sparse_index_dim": g("attention.indexer.key_length"),
        "sparse_attention_config.sparse_topk_blocks": g("attention.indexer.top_k"),
        "sparse_attention_config.sparse_block_size": g("attention.indexer.block_size"),
        "sparse_attention_config.sparse_local_block": g("attention.indexer.local_blocks", 1),
        "sparse_attention_config.sparse_init_block": 0,
        "norms_plus_one": 1,
    }


def write_tokenizer(out: pathlib.Path, kv: dict) -> dict:
    """pack.py's layout from the GGUF's vocabulary: vocab.json (id order), merges.txt, token_type.json,
    tokenizer.json {pre minimax, special_ids}, and Maya's chat template."""
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(tokens)
    merges = kv.get("tokenizer.ggml.merges") or []
    if len(set(tokens)) != len(tokens):
        raise SystemExit("the GGUF's vocabulary has duplicate token strings")
    if kv.get("tokenizer.ggml.model") != "gpt2" or not kv.get("tokenizer.ggml.normalizer.nfc", True):
        raise SystemExit("the GGUF's tokenizer is not the NFC byte-level BPE the `minimax` scheme transcribes")
    special = {"bos": int(kv["tokenizer.ggml.bos_token_id"]), "eos": int(kv["tokenizer.ggml.eos_token_id"])}
    out.mkdir(parents=True, exist_ok=True)
    (out / "vocab.json").write_text(json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
                                    encoding="utf-8")
    (out / "merges.txt").write_text("\n".join(merges), encoding="utf-8")
    (out / "token_type.json").write_text(json.dumps([int(t) for t in types]))
    (out / "tokenizer.json").write_text(json.dumps({"pre": "minimax", "special_ids": special}, indent=1))
    (out / "chat_template.jinja").write_text(TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")
    return special


SIDE_BIN = "-maya-indexers.bin"                     # (a GGUF; the Hub reads a lone .gguf as the model)
SIDE_NAMES = (SIDE_BIN, "-maya-indexers.gguf")       # (.gguf: its name up to v1.1.30)


def find_shards(model: pathlib.Path, names: list[str] | None, indexers: str | None):
    """(the main set's shards, the indexers' side file or None) - every .gguf in the folder (and the indexers'
    -maya-indexers.bin) unless named."""
    paths = [model / n for n in names] if names else sorted(model.glob("*.gguf")) + sorted(model.glob("*" + SIDE_BIN))
    missing = [p for p in paths if not p.is_file()]
    if not paths or missing:
        raise SystemExit("%s: no .gguf files" % model if not paths else "%s: not there" % missing[0])
    side = [p for p in paths if p.name.endswith(SIDE_NAMES)]
    main = [p for p in paths if p not in side]
    if indexers:
        side = [pathlib.Path(indexers).expanduser().resolve()]
        if not side[0].is_file():
            raise SystemExit("%s: not there (--indexers)" % side[0])
    return main, (side[0] if side else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the directory of the GGUF's shards")
    ap.add_argument("--out", help="default: <model>/maya-mm3")
    ap.add_argument("--shards", nargs="+", help="the set's files in --model (default: every .gguf there)")
    ap.add_argument("--indexers", help="the dense layers' indexers side file (default: the folder's "
                                       "*-maya-indexers.bin, or .gguf)")
    ap.add_argument("--companions", help="the EXL3 repo's vision tower / DSpark draft files (default: <model>/../"
                                         "%s, or a whole EXL3 download beside the model)" % COMPANIONS)
    ap.add_argument("--text-only", action="store_true", help="leave the companions' vision tower and draft out")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser() if a.out else model / "maya-mm3"
    paths, side = find_shards(model, a.shards, a.indexers)
    gs = [Gguf(p) for p in paths]
    kv = next((g.kv for g in gs if "general.architecture" in g.kv), None)
    if kv is None or kv.get("general.architecture") != ARCH:
        raise SystemExit("%s: not a MiniMax-M3 GGUF (architecture %r)" % (model, kv and kv.get("general.architecture")))
    split = int(kv.get("split.count", 1))
    if split != len(gs):
        counts = sorted({m.group(1) for m in (re.search(r"-of-(\d{5})\.gguf$", p.name) for p in paths) if m})
        raise SystemExit("%s: %d of the %d shards are here" % (model, len(gs), split) if len(counts) < 2 else
                         "%s: the shards of more than one GGUF split (-of-%s): name the model's own with --shards" %
                         (model, ", -of-".join(counts)))
    why = next((w for w in map(_ds4.short, gs) if w), None)
    if why:
        raise SystemExit("%s: not whole (%s) - a download still running?" % (model, why))
    sg = None
    if side is not None:
        sg = Gguf(side)
        if sg.kv.get("general.architecture") != ARCH or _ds4.short(sg):
            raise SystemExit("%s: not a whole MiniMax-M3 indexers file" % side)
        for_, name = sg.kv.get("maya.indexers.for"), str(kv.get("general.name", "")).replace(" ", "-")
        if for_ and name and for_ != name:
            print("note: %s says it is for %s (the set: %s)" % (side.name, for_, name))
    rel = lambda p: os.path.relpath(p, model).replace(os.sep, "/")  # noqa: E731
    files = [rel(p) for p in paths] + ([rel(side)] if sg else [])
    if any(" " in f for f in files):
        raise SystemExit("a file name with a space: %s" % [f for f in files if " " in f])
    n_exp = int(kv["%s.expert_count" % ARCH])
    nl = int(kv["%s.block_count" % ARCH])
    tensors, unknown, have = [], [], set()
    for fi, g in enumerate(gs + ([sg] if sg else [])):
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
                ck = "%slayers.%s.%s" % (P, m.group(1), LAYER[m.group(2)])
                have.add((int(m.group(1)), m.group(2)))
            elif name in GLOBAL:
                ck = GLOBAL[name]
            else:
                unknown.append(name)
                continue
            shape = list(reversed(dims))     # ggml ne0 innermost -> torch order, outermost first
            base = g.data_start + off
            if ck.endswith((".experts.w1", ".experts.w2", ".experts.w3")):
                if shape[0] != n_exp:
                    raise SystemExit("%s: %s holds %d experts, not %d" % (g.path.name, name, shape[0], n_exp))
                part = nbytes // n_exp
                pre, role = ck.rsplit(".", 1)
                for i in range(n_exp):
                    tensors.append(("%s.%d.%s.weight" % (pre, i, role), tname, fi, base + i * part, part, shape[1:]))
                continue
            tensors.append((ck, tname, fi, base, nbytes, shape))
    if unknown:
        raise SystemExit("tensors the pack does not map (%d): %s" % (len(unknown), ", ".join(unknown[:12])))
    lacking = [li for li in range(nl) if any((li, n) not in have for n in INDEXER)]
    if lacking:
        raise SystemExit(
            "the sparse-attention indexer of layer%s %s is not in the set%s: MiniMax-M3.1 runs it in every layer. The "
            "Maya sets keep layers 0-2's in <name>-maya-indexers.bin beside the shards - download it into %s (or name "
            "it with --indexers)" % ("s" if len(lacking) > 1 else "", ", ".join(map(str, lacking[:8])),
                                      "" if sg is None else " nor in " + side.name, model))
    # the EXL3 repo's vision tower and DSpark draft beside the GGUF's weights (the same files, any quant)
    comp = None if a.text_only else find_companions(model, a.companions)
    vision, dcfg = {}, None
    if comp is not None:
        try:
            cfiles, ctensors, vision, dcfg = companion_parts(comp, rel, kv["tokenizer.ggml.tokens"], len(files))
        except ValueError:   # (Windows: os.path.relpath across drives)
            raise SystemExit("%s: the companions must be on the model's drive" % comp)
        if any(" " in f for f in cfiles):
            raise SystemExit("the companions' path relative to the model has a space: %s" % cfiles[0])
        files += cfiles
        tensors += ctensors
    embd = next(t for t in tensors if t[0] == GLOBAL["token_embd.weight"])
    shexp = next((t for t in tensors if t[0].endswith("shared_experts.gate_proj.weight")), None)
    geo = geometry(kv, embd[5][0], shexp[5][0] if shexp else 0)

    out.mkdir(parents=True, exist_ok=True)
    special = write_tokenizer(out / "tokenizer", kv)
    dense = sorted({t[1] for t in tensors if ".experts." not in t[0] and t[1] != "F32"})
    exp_types = {}   # (each layer and role once: its expert 0's entry)
    for t in tensors:
        if ".experts.0." in t[0]:
            exp_types[t[1]] = exp_types.get(t[1], 0) + 1
    recipe = {}
    try:
        recipe = json.loads(kv.get("maya.recipe", "{}"))
    except ValueError:
        pass
    with (out / "mm3.txt").open("w", newline="\n") as f:
        for k, v in geo.items():
            if v is None:
                continue
            if isinstance(v, bool):
                v = int(v)
            f.write("%s %s\n" % (k, ",".join(str(x) for x in v) if isinstance(v, list) else v))
        f.write("bos_token_id %d\n" % special["bos"])
        f.write("eos_token_id %d\n" % special["eos"])
        for k, v in special.items():
            f.write("special.%s %d\n" % (k, v))
        f.write("format gguf\n")
        f.write("quant_name %s\n" % str(kv.get("general.name", model.name)).replace(" ", "-"))
        if recipe.get("tier"):
            f.write("maya.tier %s\n" % recipe["tier"])
        f.write("gguf.dense %s\n" % ",".join(dense))
        f.write("gguf.experts %s\n" % ",".join("%sx%d" % (k, v) for k, v in sorted(exp_types.items())))
        f.write("gguf.indexers %s\n" % (rel(side) if sg else "-"))
        if vision:   # (pack.py's keys: the tower's geometry, its preprocessor's, the picture span's tokens)
            for k, v in vision["_ids"].items():
                f.write("%s %d\n" % (k, v))
            exl3_pack.flat(f, "vision.", vision["_vc"])
            exl3_pack.flat(f, "vision.", vision["_comp"])
            exl3_pack.flat(f, "vision.", vision["_extra"])
        if dcfg is not None:
            exl3_pack.flat(f, "dspark.", {k: v for k, v in dcfg.items() if k not in ("text_config", "vision_config")})
            exl3_pack.flat(f, "dspark.dflash.", dcfg.get("dflash_config") or {})
            exl3_pack.flat(f, "dspark.cfg.", dcfg.get("dspark_config") or {})
        f.write("model_dir %s\n" % model)
    with (out / "tensors.txt").open("w", newline="\n") as f:
        for i, name in enumerate(files):
            f.write("file %d %s\n" % (i, name))
        for name, dt, fi, off, nb, shape in tensors:
            f.write("t %s %s %d %d %d %s\n" % (name, dt, fi, off, nb, " ".join(map(str, shape))))
    by = {}
    for t in tensors:
        by[t[1]] = by.get(t[1], 0) + t[4]
    print("%s: %d tensors in %d files (%s)%s -> %s" %
          (model.name, len(tensors), len(files), ", ".join("%s %.1f GB" % (k, v / 1e9) for k, v in sorted(by.items())),
           ", the dense layers' indexers from " + side.name if sg else "", out))
    print("  experts (layer x role): %s" % ", ".join("%s %d" % (k, v) for k, v in sorted(exp_types.items())))
    print("  special ids: %s" % special)
    print("  vision tower: %s; DSpark draft: %s" % (
        "from %s" % comp if vision else "none (text only)",
        "%d tensors from %s" % (sum(1 for t in tensors if t[0].startswith("dspark.")), comp) if dcfg else "none"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

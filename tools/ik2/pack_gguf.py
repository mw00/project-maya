"""tools/ik2/pack_gguf.py - maya-ik2's pack for an Inkling 2 GGUF set (Thinking Machines' Inkling 2 Small,
the GGUF's `inkling2` layout: unsloth's UD-Q2_K_XL build), beside the model's files (no weights copied).

    python tools/ik2/pack_gguf.py --model ~/models/Inkling-2-Small-GGUF/UD-Q2_K_XL [--log-floor 8192]

writes <model>/maya-ik2/:
  inkling2.txt  the geometry, one `key value` a line (lists comma-separated), from the GGUF's inkling2.* keys
                (log_scaling_n_floor: the GGUF's, with its stale 128000 read as 8192 - STALE_FLOOR; --log-floor
                overrides it)
  tensors.txt   `file <i> <path>` lines (relative to the model's folder, model_dir in inkling2.txt), then
                `t <name> <type> <file> <offset> <bytes> <ne0> <ne1> ...` a tensor: the GGUF's own names and dims (ne0
                innermost, as ggml counts them; the routed experts stacked, ne2 = the expert), type ggml's name,
                offset absolute in its file
  tokenizer/    the server's tokenizer files from the GGUF's vocabulary and merges (pre "gpt-4o") and thinkingmachines' own
                chat template (tools/ik2/chat_template.jinja; the GGUF's is the llama.cpp conversion's)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
ARCH = "inkling2"
# Thinking Machines' first config.json said log_scaling_n_floor 128000 (the GGUFs converted then carry it). Their
# 2026-11-17 commit 1df2d97d45 changed that one value to 8192, which is also NeMo AutoModel's inkling2 default. The
# pack takes the fix. Only the global layers' Q and R past position 8192 see it.
STALE_FLOOR = {128000: 8192}

_spec = importlib.util.spec_from_file_location("hy4_pack_gguf", HERE.parent / "hy4" / "pack_gguf.py")
_hy4 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_hy4)
Gguf, TYPES, nbytes, short = _hy4.Gguf, _hy4.TYPES, _hy4.nbytes, _hy4.short


def find_shards(model: pathlib.Path, names: list | None) -> list:
    paths = [model / n for n in names] if names else sorted(model.glob("*.gguf"))
    gs = []
    for p in paths:
        g = Gguf(p)
        arch, kind = g.kv.get("general.architecture"), g.kv.get("general.type")
        if arch in (None, ARCH) and kind in (None, "model"):
            gs.append(g)
        else:
            print("%s: %s / %s, not a shard of the model, left out" % (p.name, arch, kind))
    first = [g for g in gs if g.kv.get("general.type")]
    if not first or first[0].kv.get("general.architecture") != ARCH:
        raise SystemExit("%s: not an %s GGUF set" % (model, ARCH))
    want = int(first[0].kv.get("split.count", 1))
    if len(gs) != want:
        raise SystemExit("%s: %d .gguf files, the set has %d (a folder of one quant, or --files)" % (model, len(gs), want))
    for g in gs:
        why = short(g)
        if why:
            raise SystemExit("not whole yet: " + why)
    return gs


def geometry(kv: dict, log_floor: int | None) -> dict:
    def g(*keys, d=None):
        for k in keys:
            if "%s.%s" % (ARCH, k) in kv:
                return kv["%s.%s" % (ARCH, k)]
        if d is None:
            raise SystemExit("the GGUF has no %s.%s" % (ARCH, keys[0]))
        return d

    n_layer = int(g("block_count"))
    swa = [int(bool(x)) for x in g("attention.sliding_window_pattern")]
    floor = int(g("attention.log_scale_floor", "log_scaling_n_floor"))
    if len(swa) != n_layer:
        raise SystemExit("sliding_window_pattern: %d values for %d blocks" % (len(swa), n_layer))
    geo = {
        "arch": ARCH,
        "n_layer": n_layer, "n_embd": int(g("embedding_length")),
        "n_vocab": int(g("vocab_size")), "n_vocab_unpadded": int(g("vocab_size_unpadded", "unpadded_vocab_size")),
        "context_length": int(g("context_length")),
        "n_head": int(g("attention.head_count")), "n_head_kv": int(g("attention.head_count_kv")),
        "head_dim": int(g("attention.key_length")),
        "rms_eps": float(g("attention.layer_norm_rms_epsilon")),
        "n_expert": int(g("expert_count")), "n_expert_used": int(g("expert_used_count")),
        "n_ff_exp": int(g("expert_feed_forward_length")),
        "expert_weights_scale": float(g("expert_weights_scale")),
        "final_logit_softcapping": float(g("final_logit_softcapping")),
        "n_swa": int(g("attention.sliding_window")), "swa": swa,
        "d_rel": int(g("attention.relative_length", "d_rel")),
        "rel_extent": int(g("attention.relative_extent", "rel_extent")),
        "shortconv_kernel": int(g("shortconv.l_cache", "shortconv_kernel")),
        "log_scaling_n_floor": log_floor or STALE_FLOOR.get(floor, floor),
        "log_scaling_alpha": float(g("attention.log_scale_alpha", "log_scaling_alpha")),
        "qk_clip": float(g("attention.qk_clip", "qk_clip_threshold")),
        "rel_clip": float(g("attention.relative_clip", "rel_clip_threshold")),
        "attn_res_block": int(g("attn_res.block_size")),
        "situ_beta": float(g("activation.situ_beta", "mlp_gate_softcap")),
        "situ_linear_beta": float(g("activation.situ_linear_beta", "mlp_up_softcap")),
        "bos": int(kv["tokenizer.ggml.bos_token_id"]), "eos": int(kv["tokenizer.ggml.eos_token_id"]),
    }
    if int(g("attention.value_length")) != geo["head_dim"] or geo["n_head"] % geo["n_head_kv"] or \
            geo["shortconv_kernel"] != 4 or geo["n_embd"] % 256 or geo["n_ff_exp"] % 256 or geo["n_expert_used"] > 16:
        raise SystemExit("a geometry maya-ik2 does not run")
    return geo


def tokenizer_files(kv: dict) -> dict:
    """{file name: text} of the pack's tokenizer/ folder"""
    tokens = kv["tokenizer.ggml.tokens"]
    types = kv.get("tokenizer.ggml.token_type") or [1] * len(tokens)
    merges = kv.get("tokenizer.ggml.merges") or []
    special = {"bos": int(kv["tokenizer.ggml.bos_token_id"]), "eos": int(kv["tokenizer.ggml.eos_token_id"])}
    files = {"vocab.json": json.dumps({t: i for i, t in enumerate(tokens)}, ensure_ascii=False),
             "merges.txt": "\n".join(merges), "token_type.json": json.dumps([int(t) for t in types]),
             "tokenizer.json": json.dumps({"pre": kv.get("tokenizer.ggml.pre"), "special_ids": special,
                                           "add_bos": bool(kv.get("tokenizer.ggml.add_bos_token", False))}, indent=1)}
    # the chat template: thinkingmachines' own (tools/ik2/chat_template.jinja, the one serve/frontend.py's
    # ChatTemplate.inkling reads) - the GGUF carries the llama.cpp conversion's, which differs
    official = HERE.parent / "ik2" / "chat_template.jinja"
    if official.exists():
        files["chat_template.jinja"] = official.read_text(encoding="utf-8")
    elif kv.get("tokenizer.chat_template"):
        files["chat_template.jinja"] = kv["tokenizer.chat_template"]
    return files


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="the folder of the GGUF set")
    ap.add_argument("--files", nargs="*", help="the set's shards (default: every .gguf in the folder)")
    ap.add_argument("--out", help="the pack's folder (default: <model>/maya-ik2)")
    ap.add_argument("--log-floor", type=int, help="log_scaling_n_floor in place of the GGUF's (default: the GGUF's, "
                                                  "its stale 128000 read as Thinking Machines' fixed 8192)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    out = pathlib.Path(a.out).expanduser().resolve() if a.out else model / "maya-ik2"
    gs = find_shards(model, a.files)
    kv = {}
    for g in gs:
        for k, v in g.kv.items():
            kv.setdefault(k, v)
    geo = geometry(kv, a.log_floor)
    tensors, lines = {}, []
    for i, g in enumerate(gs):
        lines.append("file %d %s" % (i, os.path.relpath(g.path, model)))
        for name, typ, dims, off in g.tensors:
            if name in tensors:
                raise SystemExit("%s twice in the set" % name)
            tensors[name] = (typ, dims, i, g.data_start + off)
    for name, (typ, dims, i, off) in tensors.items():
        lines.append("t %s %s %d %d %d %s" % (name, TYPES[typ][0], i, off, nbytes(typ, dims), " ".join(map(str, dims))))
    E, ff, ne = geo["n_embd"], geo["n_ff_exp"], geo["n_expert"]
    need = {"token_embd.weight": [E, geo["n_vocab"]], "output.weight": [E, geo["n_vocab"]]}
    for il in range(geo["n_layer"]):
        need["blk.%d.ffn_gate_up_exps.weight" % il] = [E, 2 * ff, ne]
        need["blk.%d.ffn_down_exps.weight" % il] = [ff, E, ne]
        need["blk.%d.ffn_gate_inp.weight" % il] = [E, ne]
    for n, dims in need.items():
        if n not in tensors or list(tensors[n][1]) != dims:
            raise SystemExit("%s: %s, %s expected" % (n, tensors[n][1] if n in tensors else "missing", dims))
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    for name, text in tokenizer_files(kv).items():
        (out / "tokenizer" / name).write_text(text, encoding="utf-8")
    text = []
    for k, v in geo.items():
        v = ",".join(map(str, v)) if isinstance(v, list) else repr(v) if isinstance(v, float) else str(v)
        text.append("%s %s" % (k, v))
    text.append("model_dir %s" % model)
    (out / "inkling2.txt").write_text("\n".join(text) + "\n")
    (out / "tensors.txt").write_text("\n".join(lines) + "\n")   # (last: the file the engine looks for)
    print("%s: %d tensors from %d files, %d layers, log_scaling_n_floor %d" %
          (out, len(tensors), len(gs), geo["n_layer"], geo["log_scaling_n_floor"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

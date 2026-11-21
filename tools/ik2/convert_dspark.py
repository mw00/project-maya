"""tools/ik2/convert_dspark.py - Inkling 2 Small's DSpark drafter (Thinking Machines' mtp.safetensors, 84 BF16
tensors, with its config.json's mtp_config) as a GGUF, and maya-ik2's index of it (docs/INKLING2_DSPARK.md).

    nice -n 19 python tools/ik2/convert_dspark.py --mtp <models>/Inkling-2-Small-MTP \
        --tokenizer-gguf <target set>/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf [--type bf16|q8_0|both]

writes into --out (default: the --mtp folder):
  Inkling-2-Small-DSpark-BF16.gguf   the matrices as they are (BF16), every 1-D tensor and dspark_fuse F32
  Inkling-2-Small-DSpark-Q8_0.gguf   the matrices as ggml Q8_0 blocks (rounded here: gguf.quants), the rest F32
  maya-ik2-dspark-<type>/       dspark.txt (the geometry, `key value` lines, lists comma-separated) and
                                     tensors.txt (`file 0 <gguf>`, then `t <name> <type> <file> <offset> <bytes>
                                     <ne0> <ne1> ...`, as tools/ik2/pack_gguf.py writes the model's)

The GGUF is what llama.cpp PR #32218's `convert_hf_to_gguf.py --dspark` writes (conversion/inkling2.py's
Inkling2DSparkModel, architecture `dflash`): its tensor names, its dims (ne0 innermost, the checkpoint's [out, in]
reversed) and its keys, so that a reference llama.cpp run can load the same file.  The tokenizer keys are copied from
the target's GGUF (--tokenizer-gguf: the same vocabulary, 200058 tokens); without it the file has no tokenizer, which
maya-ik2 does not need and llama.cpp does.

No torch, no safetensors: the checkpoint's header is read here and its tensors are mapped.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

ARCH = "dflash"
N_TENSORS = 84
# the checkpoint's names -> the GGUF's (gguf-py's tensor_mapping.py with the PR's additions)
GLOBAL = {
    "norm.weight": "output_norm.weight",
    "mask_embed": "mask_embd.weight",
    "fuse_weight": "dspark_fuse.weight",
    "markov_head.markov_w1.weight": "markov_w1.weight",
    "markov_head.rnn_proj.weight": "joint_proj.weight",
    "markov_head.rnn_proj.bias": "joint_proj.bias",
    "markov_head.hidden_out.weight": "dspark_hidden_out.weight",
    "markov_head.hidden_out.bias": "dspark_hidden_out.bias",
    "markov_head.dflare_norm.weight": "dspark_norm.weight",
}
LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "dflare_norm.weight": "dflare_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight",
    "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight",
    "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.k_proj_ctx.weight": "attn_k_ctx.weight",
    "self_attn.v_proj_ctx.weight": "attn_v_ctx.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight",
    "self_attn.k_norm.weight": "attn_k_norm.weight",
    "self_attn.k_norm_ctx.weight": "attn_k_ctx_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight",
    "mlp.up_proj.weight": "ffn_up.weight",
    "mlp.down_proj.weight": "ffn_down.weight",
}
TOKENIZER_KEYS = ("tokenizer.ggml.model", "tokenizer.ggml.pre", "tokenizer.ggml.tokens", "tokenizer.ggml.token_type",
                  "tokenizer.ggml.merges", "tokenizer.ggml.add_bos_token")


def gguf_name(name: str) -> str:
    if name in GLOBAL:
        return GLOBAL[name]
    p = name.split(".", 2)
    if len(p) == 3 and p[0] == "layers" and p[1].isdigit() and p[2] in LAYER:
        return "blk.%s.%s" % (p[1], LAYER[p[2]])
    raise SystemExit("mtp.safetensors: a tensor this converter does not know: " + name)


class Safetensors:
    def __init__(self, path: pathlib.Path):
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            head = json.loads(f.read(n))
        self.base = 8 + n
        self.t = {k: v for k, v in head.items() if k != "__metadata__"}
        self.map = np.memmap(path, dtype=np.uint8, mode="r")

    def bf16(self, name: str) -> np.ndarray:
        """the tensor's BF16 words (a view of the file), in the checkpoint's shape"""
        v = self.t[name]
        if v["dtype"] != "BF16":
            raise SystemExit("%s: %s, not BF16" % (name, v["dtype"]))
        a, b = v["data_offsets"]
        return self.map[self.base + a:self.base + b].view(np.uint16).reshape(v["shape"])


def f32(bf: np.ndarray) -> np.ndarray:
    """BF16 words -> F32, exact (the words are the F32s' upper halves)"""
    return (bf.astype(np.uint32) << 16).view(np.float32)


def check_config(cfg: dict) -> tuple:
    text, mtp = cfg.get("text_config", cfg), cfg.get("mtp_config")
    if not mtp:
        raise SystemExit("config.json has no mtp_config")
    targets = mtp["target_layer_ids"]
    if (len(targets) < 2 or targets != sorted(set(targets)) or targets[0] < 0 or
            targets[-1] != text["num_hidden_layers"] or mtp["hidden_size"] != text["hidden_size"] or
            mtp["markov_head_type"] != "rnn_hidden" or mtp.get("q_bias") or mtp.get("o_bias") or
            mtp["block_size"] != 8):   # (the PR's own checks)
        raise SystemExit("an mtp_config this converter (and llama.cpp PR #32218) does not take")
    return text, mtp


def expect_shapes(st: Safetensors, text: dict, mtp: dict) -> None:
    E, R, F = mtp["hidden_size"], mtp["markov_rank"], mtp["intermediate_size"]
    hd, nq, nk, nl = mtp["head_dim"], mtp["num_attention_heads"], mtp["num_key_value_heads"], mtp["num_hidden_layers"]
    want = {"norm.weight": [E], "mask_embed": [E], "fuse_weight": [nl + 1, len(mtp["target_layer_ids"])],
            "markov_head.markov_w1.weight": [text["vocab_size"], R],
            "markov_head.rnn_proj.weight": [3 * R, 2 * R + 2 * E], "markov_head.rnn_proj.bias": [3 * R],
            "markov_head.hidden_out.weight": [E, R], "markov_head.hidden_out.bias": [E],
            "markov_head.dflare_norm.weight": [E]}
    for i in range(nl):
        p = "layers.%d." % i
        for n in ("input_layernorm", "post_attention_layernorm", "dflare_norm"):
            want[p + n + ".weight"] = [E]
        for n in ("q_norm", "k_norm", "k_norm_ctx"):
            want[p + "self_attn." + n + ".weight"] = [hd]
        for n, s in (("q_proj", [hd * nq, E]), ("k_proj", [hd * nk, E]), ("v_proj", [hd * nk, E]),
                     ("o_proj", [E, hd * nq]), ("k_proj_ctx", [hd * nk, E]), ("v_proj_ctx", [hd * nk, E])):
            want[p + "self_attn." + n + ".weight"] = s
        for n, s in (("gate_proj", [F, E]), ("up_proj", [F, E]), ("down_proj", [E, F])):
            want[p + "mlp." + n + ".weight"] = s
    got = {k: v["shape"] for k, v in st.t.items()}
    if got != want:
        bad = sorted(set(got) ^ set(want)) + sorted(k for k in set(got) & set(want) if got[k] != want[k])
        raise SystemExit("mtp.safetensors does not have mtp_config's tensors: " + ", ".join(bad[:8]))
    assert len(want) == N_TENSORS


def write(gguf, st: Safetensors, text: dict, mtp: dict, tok, out: pathlib.Path, kind: str) -> None:
    Q = gguf.GGMLQuantizationType
    w = gguf.GGUFWriter(str(out), ARCH)
    nl = mtp["num_hidden_layers"]
    # general.* and TextModel.set_gguf_parameters' keys, then Inkling2DSparkModel's
    w.add_type("model")
    w.add_name("Inkling-2-Small DSpark")
    w.add_string("general.base_model.0.repo_url", "https://huggingface.co/thinkingmachines/Inkling-2-Small")
    w.add_uint32("general.base_model.count", 1)
    w.add_quantization_version(2)
    w.add_file_type(int(gguf.LlamaFileType.MOSTLY_BF16 if kind == "bf16" else gguf.LlamaFileType.MOSTLY_Q8_0))
    w.add_block_count(nl)
    w.add_context_length(mtp["max_position_embeddings"])
    w.add_embedding_length(mtp["hidden_size"])
    w.add_feed_forward_length(mtp["intermediate_size"])
    w.add_head_count(mtp["num_attention_heads"])
    w.add_head_count_kv(mtp["num_key_value_heads"])
    w.add_rope_freq_base(float(mtp["rope_theta"]))
    w.add_layer_norm_rms_eps(float(mtp["rms_norm_eps"]))
    w.add_key_length(mtp["head_dim"])
    w.add_value_length(mtp["head_dim"])
    w.add_uint32(ARCH + ".vocab_size", text["vocab_size"])
    w.add_uint32(ARCH + ".vocab_size_unpadded", text["unpadded_vocab_size"])
    w.add_uint32(ARCH + ".block_size", mtp["block_size"])
    w.add_array(ARCH + ".target_layers", [int(x) for x in mtp["target_layer_ids"]])
    w.add_uint32(ARCH + ".dspark.markov_rank", mtp["markov_rank"])
    w.add_bool(ARCH + ".sample_from_anchor", False)
    w.add_bool(ARCH + ".has_confidence_head", False)
    w.add_bool(ARCH + ".attention.causal", False)
    w.add_uint32(ARCH + ".attention.sliding_window", mtp["sliding_window_size"])
    w.add_array(ARCH + ".attention.sliding_window_pattern", [i in mtp["local_layer_ids"] for i in range(nl)])
    w.add_uint32(ARCH + ".rope.dimension_count", mtp["head_dim"])
    w.add_float32(ARCH + ".final_logit_softcapping", float(text["final_logit_softcapping"]))
    if tok is not None:   # (_set_inkling2_vocab: gpt2, the tokens and merges, eos = eot, no BOS; then the mask token)
        eos = int(text.get("eos_token_id", 200006))
        for k in TOKENIZER_KEYS:
            f = tok.fields[k]
            v = f.contents()
            if k.endswith(".tokens") and len(v) != text["unpadded_vocab_size"]:
                raise SystemExit("--tokenizer-gguf: %d tokens, the model has %d" % (len(v), text["unpadded_vocab_size"]))
            if isinstance(v, list):
                w.add_array(k, v)
            elif isinstance(v, bool):
                w.add_bool(k, v)
            else:
                w.add_string(k, v)
        w.add_uint32("tokenizer.ggml.eos_token_id", eos)
        w.add_uint32("tokenizer.ggml.eot_token_id", eos)
        w.add_uint32("tokenizer.ggml.mask_token_id", 0)
    for name in sorted(st.t, key=lambda k: st.t[k]["data_offsets"][0]):
        bf = st.bf16(name)
        new = gguf_name(name)
        if bf.ndim < 2 or name == "fuse_weight":   # (1-D: F32, as the converter writes them; the PR forces the fuse)
            w.add_tensor(new, np.ascontiguousarray(f32(bf)), raw_dtype=Q.F32)
        elif kind == "bf16":
            w.add_tensor(new, bf, raw_dtype=Q.BF16)   # (the file's own words)
        else:
            if bf.shape[-1] % 32:
                raise SystemExit("%s: rows of %d values, not whole Q8_0 blocks" % (name, bf.shape[-1]))
            w.add_tensor(new, gguf.quants.quantize(f32(bf), Q.Q8_0), raw_dtype=Q.Q8_0)
        print("  %-34s <- %s %s" % (new, name, list(bf.shape)), flush=True)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()


def index(gguf, path: pathlib.Path, text: dict, mtp: dict, kind: str) -> pathlib.Path:
    r = gguf.GGUFReader(str(path))
    d = path.parent / ("maya-ik2-dspark-" + kind)
    d.mkdir(exist_ok=True)
    nl = mtp["num_hidden_layers"]
    geo = [("format", "maya-ik2-dspark 1"), ("model_dir", ".."), ("type", kind), ("n_layer", nl),
           ("n_embd", mtp["hidden_size"]), ("n_ff", mtp["intermediate_size"]), ("n_head", mtp["num_attention_heads"]),
           ("n_head_kv", mtp["num_key_value_heads"]), ("head_dim", mtp["head_dim"]), ("eps", mtp["rms_norm_eps"]),
           ("rope_theta", mtp["rope_theta"]), ("n_swa", mtp["sliding_window_size"]),
           ("swa", ",".join(str(int(i in mtp["local_layer_ids"])) for i in range(nl))),
           ("block_size", mtp["block_size"]), ("markov_rank", mtp["markov_rank"]),
           ("target_layers", ",".join(str(x) for x in mtp["target_layer_ids"])),
           ("target_n_layer", text["num_hidden_layers"]), ("n_vocab", text["vocab_size"]),
           ("n_vocab_unpadded", text["unpadded_vocab_size"]), ("softcap", text["final_logit_softcapping"])]
    (d / "dspark.txt").write_text("".join("%s %s\n" % kv for kv in geo))
    lines = ["file 0 %s\n" % path.name]
    for t in r.tensors:
        lines.append("t %s %s 0 %d %d %s\n" % (t.name, t.tensor_type.name, t.data_offset, t.n_bytes,
                                              " ".join(str(int(x)) for x in t.shape)))
    if len(r.tensors) != N_TENSORS:
        raise SystemExit("%s: %d tensors, not %d" % (path, len(r.tensors), N_TENSORS))
    (d / "tensors.txt").write_text("".join(lines))
    return d


def verify(gguf, path: pathlib.Path, st: Safetensors, kind: str) -> None:
    """every tensor of the GGUF against the checkpoint: BF16 / F32 to the bit, Q8_0 within half a block step"""
    r = gguf.GGUFReader(str(path))
    back = {gguf_name(k): k for k in st.t}
    worst = 0.0
    for t in r.tensors:
        src = st.bf16(back[t.name])
        if list(reversed([int(x) for x in t.shape])) != list(src.shape):
            raise SystemExit("%s: dims %s, the checkpoint's %s" % (t.name, list(t.shape), list(src.shape)))
        ty = t.tensor_type.name
        if ty == "BF16":
            ok = np.array_equal(np.asarray(t.data).view(np.uint16).ravel(), src.ravel())
        elif ty == "F32":
            ok = np.array_equal(np.asarray(t.data).view(np.uint32).ravel(), f32(src).view(np.uint32).ravel())
        else:
            raw = np.asarray(t.data).reshape(-1, 34)   # (a block: an F16 step d, then 32 int8)
            dstep = raw[:, :2].copy().view(np.float16).astype(np.float32)
            q = raw[:, 2:].view(np.int8).astype(np.float32) * dstep
            err = np.abs(q - f32(src).reshape(-1, 32))
            # (rounded to the nearest of a block's levels: half a step, and the step's own F16 rounding over 127 levels)
            # (a step under 6.1e-5 is an F16 subnormal, 2^-24 apart: joint_proj's small weights have such blocks)
            lim = np.abs(dstep) * (0.5 + 127.0 * 2.0 ** -11) * 1.001 + 127.0 * 2.0 ** -25 * 1.001
            ok = bool((err <= lim).all())
            ref = np.abs(f32(src)).reshape(-1, 32).max(1, keepdims=True)
            worst = max(worst, float((err / np.maximum(ref, 1e-30)).max()))
        if not ok:
            raise SystemExit("%s: %s differs from the checkpoint" % (path.name, t.name))
    print("%s: %d tensors checked against mtp.safetensors (%s)%s" % (
        path.name, len(r.tensors), "to the bit" if kind == "bf16" else "the F32 ones to the bit, Q8_0 within its rounding",
        "" if kind == "bf16" else "; worst Q8_0 error %.4f of its block's largest value" % worst))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtp", required=True, help="the folder with mtp.safetensors and the model's config.json")
    ap.add_argument("--tokenizer-gguf", help="the target's GGUF (the shard with its tokenizer): its tokenizer keys copied")
    ap.add_argument("--out", help="where the files go (default: the --mtp folder)")
    ap.add_argument("--type", choices=("bf16", "q8_0", "both"), default="both")
    ap.add_argument("--gguf-py", help="a llama.cpp checkout's gguf-py folder (default: the installed gguf package)")
    ap.add_argument("--no-verify", action="store_true", help="do not read the written files back")
    a = ap.parse_args()
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    import gguf

    src = pathlib.Path(a.mtp)
    out = pathlib.Path(a.out) if a.out else src
    text, mtp = check_config(json.loads((src / "config.json").read_text()))
    st = Safetensors(src / "mtp.safetensors")
    expect_shapes(st, text, mtp)
    tok = gguf.GGUFReader(a.tokenizer_gguf) if a.tokenizer_gguf else None
    if tok is None:
        print("no --tokenizer-gguf: the files have no tokenizer (llama.cpp needs one, maya-ik2 does not)")
    for kind in (("bf16", "q8_0") if a.type == "both" else (a.type,)):
        path = out / ("Inkling-2-Small-DSpark-%s.gguf" % kind.upper())
        print("%s:" % path, flush=True)
        write(gguf, st, text, mtp, tok, path, kind)
        d = index(gguf, path, text, mtp, kind)
        print("%s: %.2f GB; index %s" % (path.name, path.stat().st_size / 1e9, d))
        if not a.no_verify:
            verify(gguf, path, st, kind)
    return 0


if __name__ == "__main__":
    sys.exit(main())

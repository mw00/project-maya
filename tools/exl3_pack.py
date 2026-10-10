"""tools/exl3_pack.py - a Maya pack for an EXL3 (exllamav3) GLM-5.3-Flash model, without copying its weights.

    python tools/exl3_pack.py --model ~/models2/GLM-5.3-Flash-exl3-3.05bpw
    python tools/exl3_pack.py --model <dir> --check <a GGUF of the same model, shard 1>

The engine serves a pack (index.txt, dense.bin, native_experts.txt, tokenizer/) from beside the model's files.  An
EXL3 model is a directory of safetensors whose big matrices are trellis-coded (include/strata/kernels/exl3.hpp); the
packer leaves those where they are and writes:

  <model>/maya-exl3.gguf   what the engine reads its geometry from (llama.cpp's glm5-next keys), the tokenizer, and
                           every unquantized tensor under llama.cpp's name and layout, with the conversions llama.cpp's
                           converter makes (kv_b -> attn_k_b / attn_v_b, A_log -> ssm_a = -exp(A_log), dt_bias ->
                           ssm_dt.bias, the fused q|k|v conv split in three)
  pack/index.txt           as for a GGUF model; the EXL3 matrices are rows of kind x
  pack/dense.bin           the float rows (BF16 where the engine reads BF16, else F32)
  pack/exl3.txt            the EXL3 matrices (file, trellis / suh / svh offsets, k, n, bits, codebook) and every
                           routed expert's three parts (gate, up, down: [suh | svh | codebook word | trellis] each),
                           read in place by the engine
  pack/native_experts.txt  the header naming maya-exl3.gguf (no rows: the experts are in exl3.txt)
  pack/tokenizer/          from the model's tokenizer.json, typed as llama.cpp's converter types it, with Maya's
                           GLM-5.3 chat template (tools/glm_chat_template.jinja: the one Maya's own GGUFs carry - images,
                           video and audio, clear_thinking, the tool-result ordering fixes); --model-template keeps the
                           model directory's chat_template.jinja instead

The fused KDA q|k|v projection stays one EXL3 matrix (attn_qkv): the engine serves q, k and v as column views of it.
The NextN (MTP) block is packed too when the files carry it (block_count counts it, its floats go into the GGUF and
its matrices and experts into exl3.txt); --no-mtp leaves it out.

--check <gguf>: the converted float tensors against the same model's GGUF (exact where both keep them unquantized,
BF16 rounding where the GGUF has F32), and the EXL3-decoded weights against its quantized ones (cosine - two quants
of one model agree to ~0.95+; a layout or codebook mistake shows as ~0).
"""
from __future__ import annotations

import argparse
import io
import json
import math
import pathlib
import re
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gguf_writer import GGUFWriter, GGUF_MAGIC  # noqa: E402

ALIGN = 256
GGUF_NAME = "maya-exl3.gguf"
KIND_EXL3 = "x"
TYPE_EXL3 = 200          # the engine's kTypeEXL3
ROLES = ("gate", "up", "down")
GGUF_TYPE = {"F32": 0, "F16": 1, "BF16": 30}


# ------------------------------------------------------------------ safetensors, read in place
class SafeTensors:
    """Every tensor of the model directory: name -> (file, dtype, shape, absolute offset, bytes).  The index's own
    files first; any other .safetensors beside them (mtp.safetensors, kpool_aux.safetensors) only adds names the
    index does not have."""

    def __init__(self, d: pathlib.Path):
        self.dir = d
        ix = json.loads((d / "model.safetensors.index.json").read_text())["weight_map"]
        primary = sorted(set(ix.values()))
        others = sorted(p.name for p in d.glob("*.safetensors") if p.name not in primary)
        self.files: list[str] = []
        self.t: dict[str, tuple] = {}
        self.mm: dict[str, np.memmap] = {}
        for fn in primary + others:
            p = d / fn
            with p.open("rb") as f:
                n = struct.unpack("<Q", f.read(8))[0]
                h = json.loads(f.read(n))
            base = 8 + n
            self.files.append(fn)
            for k, v in h.items():
                if k == "__metadata__" or k in self.t:
                    continue
                if fn in primary and ix.get(k) not in (None, fn):
                    continue
                s, e = v["data_offsets"]
                self.t[k] = (fn, v["dtype"], list(v["shape"]), base + s, e - s)
            self.mm[fn] = np.memmap(p, dtype=np.uint8, mode="r")

    def has(self, name: str) -> bool:
        return name in self.t

    def raw(self, name: str) -> np.ndarray:
        fn, _, _, off, nb = self.t[name]
        return self.mm[fn][off:off + nb]

    def f32(self, name: str) -> np.ndarray:
        """The tensor as float32 (exact for F32/F16/BF16), shaped as stored."""
        _, dt, shape, _, _ = self.t[name]
        r = self.raw(name)
        if dt == "F32":
            a = r.view("<f4")
        elif dt == "F16":
            a = r.view("<f2").astype(np.float32)
        elif dt == "BF16":
            a = (r.view("<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError("%s is %s, not a float tensor" % (name, dt))
        return a.reshape(shape)


def to_bf16(a: np.ndarray) -> tuple[np.ndarray, int]:
    """float32 -> BF16 bits, round to nearest even; also the count of values that did not survive exactly."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    r = ((u.astype(np.uint64) + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)
    lost = int(np.count_nonzero((r.astype(np.uint32) << 16) != u))
    return r, lost


# ------------------------------------------------------------------ the EXL3 format in numpy (the --check reference)
def _sylvester(n: int) -> np.ndarray:
    h = np.ones((1, 1), dtype=np.float64)
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / math.sqrt(n)


HAD = _sylvester(128)
_P = np.arange(256)
_ROW = (_P >> 3 & 3) * 2 + (_P & 1) + 8 * (_P >> 1 & 1)
_COL = (_P >> 5) + 8 * (_P >> 2 & 1)


def exl3_states(words: np.ndarray, K: int) -> np.ndarray:
    """(..., 8 K) uint32 tile words -> (..., 256) 16-bit states (MSB-first stream, tail-biting)."""
    W = 8 * K
    b1 = (_P + 1 + 256) * K
    i1, i0 = (b1 - 1) >> 5, (b1 - 16) >> 5
    sh = ((i1 + 1) << 5) - b1
    m = (words[..., i0 % W].astype(np.uint64) << np.uint64(32)) | words[..., i1 % W].astype(np.uint64)
    return ((m >> sh.astype(np.uint64)) & np.uint64(0xFFFF)).astype(np.uint32)


def exl3_values(st: np.ndarray, cb: int) -> np.ndarray:
    f16 = lambda b: np.asarray(b, dtype=np.uint16).view(np.float16).astype(np.float64)  # noqa: E731
    if cb == 2:
        x = (st.astype(np.uint64) * 0x83DCD12D) & 0xFFFFFFFF
        bs = (x & 0xFF) + (x >> 8 & 0xFF) + (x >> 16 & 0xFF) + (x >> 24 & 0xFF)
        v = (1024.0 + bs.astype(np.float64)) * f16(0x1EEE) + f16(0xC931)
        return v.astype(np.float16).astype(np.float32)
    x = st.astype(np.uint64) * (0xCBAC1FED if cb == 1 else 89226354)
    if cb == 0:
        x += 64248484
    x = ((x & 0xFFFFFFFF) & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (x & 0xFFFF).astype(np.uint16).view(np.float16).astype(np.float64)
    hi = (x >> 16).astype(np.uint16).view(np.float16).astype(np.float64)
    return (lo + hi).astype(np.float16).astype(np.float32)


def exl3_rows(trellis: np.ndarray, suh: np.ndarray, svh: np.ndarray, K: int, cb: int, c0: int = 0,
              nc: int | None = None) -> np.ndarray:
    """The full weight's rows (n x k, the GGUF layout) of output columns [c0, c0 + nc)."""
    kt, nt, _ = trellis.shape
    nc = nt * 16 if nc is None else nc
    w = np.ascontiguousarray(trellis[:, c0 // 16:(c0 + nc) // 16, :]).view(np.uint32)
    st = exl3_states(w, K)
    vals = exl3_values(st, cb)                                       # (kt, nt', 256) in step order
    tile = np.empty(vals.shape, dtype=np.float32)
    tile[..., _ROW * 16 + _COL] = vals
    k = kt * 16
    inner = tile.reshape(kt, nc // 16, 16, 16).transpose(0, 2, 1, 3).reshape(k, nc).astype(np.float64)
    inner = (inner.reshape(k, nc // 128, 128) @ HAD).reshape(k, nc)
    inner = (HAD @ inner.reshape(k // 128, 128, nc)).reshape(k, nc)
    full = inner * suh.astype(np.float64)[:, None] * svh[c0:c0 + nc].astype(np.float64)[None, :]
    return full.T.astype(np.float32)


# ------------------------------------------------------------------ the plan
class Plan:
    def __init__(self, st: SafeTensors, cfg: dict):
        self.st, self.cfg = st, cfg
        self.floats: list[tuple] = []     # (gguf name, gguf shape, "F32" | "BF16" | "BF16RAW", producer)
        self.mats: list[tuple] = []       # (gguf name, hf prefix, k, n, K, cb)
        self.layers: list[dict] = []      # per MoE layer: the expert parts
        self.notes: list[str] = []

    def f32(self, gname, shape, fn):
        self.floats.append((gname, shape, "F32", fn))

    def bf16(self, gname, shape, fn):
        self.floats.append((gname, shape, "BF16", fn))

    def gguf_only(self, n0: int):
        """The floats added since index n0 go into maya-exl3.gguf but not into index.txt / dense.bin."""
        self.floats[n0:] = [(n, sh, ty + ":gguf", fn) for n, sh, ty, fn in self.floats[n0:]]

    def mat(self, gname: str, hf: str):
        st = self.st
        if not st.has(hf + ".trellis"):
            if st.has(hf + ".weight"):
                raise SystemExit("%s is stored unquantized; the packer expects it as EXL3" % hf)
            raise SystemExit("%s: no tensor %s.trellis" % (gname, hf))
        if not st.has(hf + ".suh") or not st.has(hf + ".svh"):
            raise SystemExit("%s: EXL3 without suh/svh (an old packed-sign checkpoint); not supported" % hf)
        kt, nt, w = st.t[hf + ".trellis"][2]
        if w % 16:
            raise SystemExit("%s: half-integer bitrate (%d words a tile); not supported yet" % (hf, w))
        K = w // 16
        cb = 2 if st.has(hf + ".mul1") else 1 if st.has(hf + ".mcg") else 0
        if st.t[hf + ".suh"][1] != "F16" or st.t[hf + ".svh"][1] != "F16":
            raise SystemExit("%s: suh/svh are not F16" % hf)
        self.mats.append((gname, hf, kt * 16, nt * 16, K, cb))

    def build(self, mtp: bool):
        st, t = self.st, self.cfg
        L = "model.language_model."
        n_layers = t["num_hidden_layers"]
        n_head, nope, vhd = t["num_attention_heads"], t["qk_nope_head_dim"], t["v_head_dim"]
        kda = t["linear_attn_config"]
        di = kda["num_heads"] * kda["head_dim"]
        first_dense = t.get("first_k_dense_replace", 3)
        g = lambda n: (lambda: st.f32(n))  # noqa: E731

        self.bf16("token_embd.weight", None, None)   # the BF16 embedding, copied raw (see write)
        self.f32("output_norm.weight", [t["hidden_size"]], g(L + "norm.weight"))
        self.mat("output.weight", "lm_head")
        for il in range(n_layers + (1 if mtp else 0)):
            P, H = "blk.%d." % il, L + "layers.%d." % il
            A = H + "self_attn."
            n_float0 = len(self.floats)
            self.f32(P + "attn_norm.weight", None, g(H + "input_layernorm.weight"))
            self.f32(P + "ffn_norm.weight", None, g(H + "post_attention_layernorm.weight"))
            for s in ("attn", "ffn") if il < n_layers else ():
                self.bf16(P + "hc_%s_fn.weight" % s, None, g(H + "hc_%s_fn" % s))
                self.f32(P + "hc_%s_base.weight" % s, None, g(H + "hc_%s_base" % s))
                self.f32(P + "hc_%s_scale.weight" % s, None, g(H + "hc_%s_scale" % s))
            if il < n_layers and t["layer_types"][il] == "linear_attention":
                self.mat(P + "attn_qkv.weight", A + "qkv_proj")
                self.mat(P + "attn_output.weight", A + "o_proj")
                self.f32(P + "ssm_a", None, lambda A=A: -np.exp(st.f32(A + "A_log").reshape(-1)[:n_head]))
                self.f32(P + "ssm_dt.bias", None, g(A + "dt_bias"))
                self.bf16(P + "ssm_beta.weight", None, g(A + "b_proj.weight"))
                for nm in ("f_a", "f_b", "g_a", "g_b"):
                    self.bf16(P + "ssm_%s.weight" % nm, None, g(A + "%s_proj.weight" % nm))
                self.f32(P + "ssm_norm.weight", None, g(A + "o_norm.weight"))
                for i, q in enumerate("qkv"):
                    # torch [d_inner, 1, d_conv] per part = GGUF [d_conv, 1, d_inner]
                    self.f32(P + "ssm_conv1d_%s.weight" % q, [kda["short_conv_kernel_size"], 1, di],
                             lambda A=A, i=i: self._conv(A, i, di))
            else:
                self.mat(P + "attn_q_a.weight", A + "q_a_proj")
                self.mat(P + "attn_q_b.weight", A + "q_b_proj")
                self.mat(P + "attn_kv_a_mqa.weight", A + "kv_a_proj_with_mqa")
                self.mat(P + "attn_output.weight", A + "o_proj")
                self.mat(P + "indexer.attn_q_b.weight", A + "indexer.wq_b")
                self.f32(P + "attn_q_a_norm.weight", None, g(A + "q_a_layernorm.weight"))
                self.f32(P + "attn_kv_a_norm.weight", None, g(A + "kv_a_layernorm.weight"))
                kvb = A + "kv_b_proj.weight"
                # llama.cpp: kv_b [n_head (nope + v), kv_lora] -> k_b = [n_head, kv_lora, nope], v_b = [n_head, v, kv_lora]
                self.bf16(P + "attn_k_b.weight", None,
                          lambda kvb=kvb: st.f32(kvb).reshape(n_head, nope + vhd, -1)[:, :nope, :].transpose(0, 2, 1))
                self.bf16(P + "attn_v_b.weight", None,
                          lambda kvb=kvb: st.f32(kvb).reshape(n_head, nope + vhd, -1)[:, nope:, :])
                I = A + "indexer."
                self.bf16(P + "indexer.attn_k.weight", None, g(I + "wk.weight"))
                self.bf16(P + "indexer.proj.weight", None, g(I + "weights_proj.weight"))
                self.f32(P + "indexer.k_norm.weight", None, g(I + "k_norm.weight"))
                self.f32(P + "indexer.k_norm.bias", None, g(I + "k_norm.bias"))
                self.bf16(P + "indexer_compressor_gate.weight", None, g(I + "index_kpool_compress_gate"))
                self.f32(P + "indexer_compressor_ape.weight", None, g(I + "index_kpool_compress_ape"))
            if il == n_layers:
                # the NextN block's own inputs and output norm
                self.mat(P + "nextn.eh_proj.weight", H + "eh_proj")
                self.f32(P + "nextn.enorm.weight", None, g(H + "enorm.weight"))
                self.f32(P + "nextn.hnorm.weight", None, g(H + "hnorm.weight"))
                self.f32(P + "nextn.shared_head_norm.weight", None, g(H + "shared_head.norm.weight"))
            M = H + "mlp."
            if il < first_dense:
                for r in ROLES:
                    self.mat(P + "ffn_%s.weight" % r, M + "%s_proj" % r)
            else:
                self.bf16(P + "ffn_gate_inp.weight", None, g(M + "gate.weight"))
                self.f32(P + "exp_probs_b.bias", None, g(M + "gate.e_score_correction_bias"))
                for r in ROLES:
                    self.mat(P + "ffn_%s_shexp.weight" % r, M + "shared_experts.%s_proj" % r)
                self.layers.append(self._experts(il, M, t["n_routed_experts"]))
            if il == n_layers:
                self.gguf_only(n_float0)

    def _conv(self, A: str, i: int, di: int) -> np.ndarray:
        st = self.st
        if st.has(A + "conv1d.weight"):
            w = st.f32(A + "conv1d.weight")              # [3 d_inner, 1, d_conv], q | k | v
            return w.reshape(3, di, -1)[i]
        return st.f32(A + "%s_conv1d.weight" % "qkv"[i]).reshape(di, -1)

    def _experts(self, il: int, M: str, n_expert: int) -> dict:
        """Every expert's nine pieces (trellis, suh, svh of gate, up and down) wherever the files keep them; the
        engine assembles its blob as [gate | up | down], each part [trellis | suh | svh] (every piece a multiple of
        256 bytes).  Every expert of the layer must share the parts' shapes, bits and codebook."""
        st = self.st
        lay = None
        rows = []
        for e in range(n_expert):
            parts, shape = [], []
            for r in ROLES:
                p = M + "experts.%d.%s_proj." % (e, r)
                for x in ("trellis", "suh", "svh"):
                    if not st.has(p + x):
                        raise SystemExit("layer %d expert %d: no %s%s" % (il, e, p, x))
                kt, nt, w = st.t[p + "trellis"][2]
                if w % 16:
                    raise SystemExit("layer %d expert %d %s: half-integer bitrate; not supported yet" % (il, e, r))
                if st.t[p + "suh"][1] != "F16" or st.t[p + "svh"][1] != "F16":
                    raise SystemExit("layer %d expert %d %s: suh / svh are not F16" % (il, e, r))
                cb = 2 if st.has(p + "mul1") else 1 if st.has(p + "mcg") else 0
                shape.append((kt * 16, nt * 16, w // 16, cb))
                parts.append((self.st.files.index(st.t[p + "trellis"][0]), st.t[p + "trellis"][3],
                              self.st.files.index(st.t[p + "suh"][0]), st.t[p + "suh"][3],
                              self.st.files.index(st.t[p + "svh"][0]), st.t[p + "svh"][3]))
            shape = tuple(shape)
            if lay is None:
                lay = shape
            elif shape != lay:
                raise SystemExit("layer %d expert %d: shapes / bits differ from expert 0's" % (il, e))
            rows.append((e, parts))
        (gk, gn, gK, gcb), (uk, un, uK, ucb), (dk, dn, dK, dcb) = lay
        if (gk, gn) != (uk, un) or gK != uK or not (gcb == ucb == dcb):
            raise SystemExit("layer %d: gate and up differ, or the codebooks do" % il)
        part = lambda k, n, K: k * n * K // 8 + 2 * k + 2 * n  # noqa: E731
        return {"layer": il, "cb": gcb, "K": (gK, uK, dK), "shape": (gk, gn, dk, dn),
                "gu_bytes": part(gk, gn, gK), "dn_bytes": part(dk, dn, dK), "rows": rows}


# ------------------------------------------------------------------ the tokenizer, as llama.cpp's converter types it
MAYA_TEMPLATE = HERE / "glm_chat_template.jinja"


def tokenizer_meta(d: pathlib.Path, vocab_size: int, template: pathlib.Path | None = None) -> dict:
    tj = json.loads((d / "tokenizer.json").read_text(encoding="utf-8"))
    tc = json.loads((d / "tokenizer_config.json").read_text(encoding="utf-8"))
    if tj["model"]["type"] != "BPE":
        raise SystemExit("tokenizer.json is %s, not BPE" % tj["model"]["type"])
    vocab = dict(tj["model"]["vocab"])
    added = {a["content"]: a for a in tj.get("added_tokens", [])}
    for a in added.values():
        vocab[a["content"]] = a["id"]
    rev = {i: s for s, i in vocab.items()}
    tokens, types = [], []
    for i in range(vocab_size):
        if i not in rev:
            tokens.append("[PAD%d]" % i)
            types.append(5)                       # UNUSED
            continue
        s = rev[i]
        if s in added:
            a = added[s]
            looks = (s.startswith("<|") and s.endswith("|>")) or s in ("<pad>", "<mask>", "<2mass>", "[@BOS@]") or \
                (s.startswith("<｜") and s.endswith("｜>")) or (s.startswith("<unused") and s.endswith(">"))
            if a.get("special") or looks:
                types.append(3)                   # CONTROL
            else:
                s = s.replace("▁", " ")
                types.append(4)                   # USER_DEFINED
        else:
            types.append(1)                       # NORMAL
        tokens.append(s)
    merges = [m if isinstance(m, str) else " ".join(m) for m in tj["model"]["merges"]]
    av = {a["content"]: a["id"] for a in added.values()}

    def tid(s):
        if s not in av:
            raise SystemExit("tokenizer: %r is not an added token" % s)
        return av[s]
    tpl = template if template is not None else d / "chat_template.jinja"
    meta = {
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.pre": "glm4",
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.token_type": types,
        "tokenizer.ggml.merges": merges,
        # the special ids llama.cpp's _set_vocab_glm sets (SpecialVocab adds eos / pad from tokenizer_config)
        "tokenizer.ggml.eos_token_id": tid(tc.get("eos_token", "<|endoftext|>")),
        "tokenizer.ggml.padding_token_id": tid(tc.get("pad_token", "<|endoftext|>")),
        "tokenizer.ggml.bos_token_id": tid("[gMASK]"),
        "tokenizer.ggml.eot_token_id": tid("<|user|>"),
        "tokenizer.ggml.unknown_token_id": tid("<|endoftext|>"),
        "tokenizer.ggml.eom_token_id": tid("<|observation|>"),
    }
    if tpl.exists():
        meta["tokenizer.chat_template"] = tpl.read_text(encoding="utf-8")
    elif "chat_template" in tc:
        meta["tokenizer.chat_template"] = tc["chat_template"]
    return meta


def model_meta(t: dict, n_layers: int, nextn: int = 0) -> list[tuple]:
    """llama.cpp's glm5-next keys; the NextN block counts into block_count (a DSA layer with a full indexer)."""
    P = "glm5-next."
    kda = t["linear_attn_config"]
    kv = [0 if x == "linear_attention" else 1 for x in t["layer_types"][:n_layers]] + [1] * nextn
    full = [x == "full" for x in t["indexer_types"][:n_layers]] + [True] * nextn
    n_blocks = n_layers + nextn
    m = [
        ("general.architecture", "glm5-next", "string"),
        ("general.name", "GLM-5.3-Flash EXL3", "string"),
        (P + "block_count", n_blocks, "u32"),
        (P + "context_length", t.get("max_position_embeddings", 1048576), "u32"),
        (P + "embedding_length", t["hidden_size"], "u32"),
        (P + "feed_forward_length", t["intermediate_size"], "u32"),
        (P + "attention.head_count", t["num_attention_heads"], "u32"),
        (P + "attention.layer_norm_rms_epsilon", float(t["rms_norm_eps"]), "f32"),
        (P + "expert_count", t["n_routed_experts"], "u32"),
        (P + "expert_used_count", t["num_experts_per_tok"], "u32"),
        (P + "expert_group_count", t.get("n_group", 1), "u32"),
        (P + "expert_group_used_count", t.get("topk_group", 1), "u32"),
        (P + "expert_gating_func", 2, "u32"),
        (P + "vocab_size", t["vocab_size"], "u32"),
        (P + "attention.head_count_kv", kv, "array:i32"),
        (P + "attention.q_lora_rank", t["q_lora_rank"], "u32"),
        (P + "attention.kv_lora_rank", t["kv_lora_rank"], "u32"),
        (P + "rope.dimension_count", t["qk_rope_head_dim"], "u32"),
        (P + "attention.key_length", t["kv_lora_rank"] + t["qk_rope_head_dim"], "u32"),
        (P + "attention.value_length", t["kv_lora_rank"], "u32"),
        (P + "attention.key_length_mla", t["qk_nope_head_dim"] + t["qk_rope_head_dim"], "u32"),
        (P + "attention.value_length_mla", t["v_head_dim"], "u32"),
        (P + "attention.layer_norm_epsilon", 1e-6, "f32"),
        (P + "ssm.conv_kernel", kda["short_conv_kernel_size"], "u32"),
        (P + "kda.head_dim", kda["head_dim"], "u32"),
        (P + "attention.indexer.head_count", t["index_n_heads"], "u32"),
        (P + "attention.indexer.key_length", t["index_head_dim"], "u32"),
        (P + "attention.indexer.top_k", t["index_topk"], "u32"),
        (P + "attention.indexer.kpool", t["index_kpool"], "u32"),
        (P + "attention.indexer.kpool_select_tail", bool(t.get("index_kpool_always_select_tail", True)), "bool"),
        (P + "attention.indexer.types", full, "array:bool"),
        (P + "hyper_connection.count", t["hc_mult"], "u32"),
        (P + "hyper_connection.sinkhorn_iterations", t["hc_sinkhorn_iters"], "u32"),
        (P + "hyper_connection.epsilon", float(t["hc_eps"]), "f32"),
        (P + "expert_feed_forward_length", t["moe_intermediate_size"], "u32"),
        (P + "expert_shared_feed_forward_length", t["moe_intermediate_size"] * t["n_shared_experts"], "u32"),
        (P + "expert_shared_count", t["n_shared_experts"], "u32"),
        (P + "leading_dense_block_count", t["first_k_dense_replace"], "u32"),
        (P + "expert_weights_scale", float(t["routed_scaling_factor"]), "f32"),
        (P + "expert_weights_norm", bool(t["norm_topk_prob"]), "bool"),
        (P + "nextn_predict_layers", nextn, "u32"),
    ]
    if kda.get("gate_lower_bound") is not None:
        m.append((P + "kda.gate_lower_bound", float(kda["gate_lower_bound"]), "f32"))
    if t.get("swiglu_limit") is not None:
        m.append((P + "swiglu_clamp_exp", [float(t["swiglu_limit"])] * n_blocks, "array:f32"))
        m.append((P + "swiglu_clamp_shexp", [float(t["swiglu_limit"])] * n_blocks, "array:f32"))
    return m


# ------------------------------------------------------------------ writing
def gguf_shape(a_shape, given):
    return list(given) if given is not None else list(reversed(a_shape))


def replace_atomic(path: pathlib.Path):
    """A file written beside path and renamed over it when the with-block ends: an engine that has the old one mapped
    (a model being served while the pack is rewritten) keeps reading the old one."""
    import contextlib

    @contextlib.contextmanager
    def cm():
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("wb") as f:
            yield f
        import os
        os.replace(tmp, path)
    return cm()


def write_all(plan: Plan, d: pathlib.Path, out: pathlib.Path, t: dict, nextn: int,
              template: pathlib.Path | None) -> None:
    st = plan.st
    n_layers = t["num_hidden_layers"]
    meta = model_meta(t, n_layers, nextn)
    tok = tokenizer_meta(d, t["vocab_size"], template)
    w = GGUFWriter(alignment=32)
    for k, v, ty in meta:
        w.add(k, v, ty)
    for k, v in tok.items():
        if k == "tokenizer.ggml.token_type":
            w.add(k, v, "array:i32")
        elif k.endswith("_id"):
            w.add(k, v, "u32")
        else:
            w.add(k, v)
    # every float tensor, materialised once (~1 GB without the embedding), in the engine's types
    tensors = []        # (name, gguf shape, type, bytes-or-memmap)
    rows = []           # index rows
    lost_total = 0
    for name, shape, ty, fn in plan.floats:
        if name == "token_embd.weight":
            src = "model.language_model.embed_tokens.weight"
            fn_, dt, shp, _, _ = st.t[src]
            if dt != "BF16":
                raise SystemExit("token_embd is %s; the engine reads the embedding row of a GGUF type (BF16 here)" % dt)
            tensors.append((name, list(reversed(shp)), "BF16", st.raw(src)))
            rows.append([name, "0", "0", "0", "0", "0", "0", str(shp[1]), str(shp[0]), "8", "0", "32"] + ["0"] * 7)
            continue
        gguf_only = ty.endswith(":gguf")
        ty = ty.split(":")[0]
        a = np.ascontiguousarray(fn(), dtype=np.float32)
        gs = gguf_shape(a.shape, shape)
        if ty == "BF16":
            b, lost = to_bf16(a)
            lost_total += lost
            data = b.tobytes()
            kind = "4"
        else:
            data = a.tobytes()
            kind = "2"
        tensors.append((name, gs, ty, data))
        if gguf_only:
            continue
        ne0 = gs[0] * gs[1] if len(gs) >= 3 else gs[0]
        ne1 = gs[2] if len(gs) >= 3 else (gs[1] if len(gs) > 1 else 0)
        rows.append([name, "0", kind, "", str(len(data)), "0", str(len(data)), str(ne0), str(ne1), "0", "0", "1"] +
                    ["0"] * 7)
    if lost_total:
        plan.notes.append("BF16: %d values rounded (F32 sources such as hc_*_fn; F16 sources convert exactly when "
                          "they came from BF16)" % lost_total)
    for name, hf, k, n, K, cb in plan.mats:
        if name.startswith("blk.") and int(name.split(".")[1]) >= n_layers:
            continue   # the NextN block: load_mtp takes it from exl3.txt
        rows.append([name, "0", KIND_EXL3, "0", "0", "0", "0", str(k), str(n), "0", "0", "0"] + ["0"] * 7)

    # ---- maya-exl3.gguf (streamed: the embedding is 1.27 GB)
    gpath = d / GGUF_NAME
    with replace_atomic(gpath) as f:
        f.write(struct.pack("<IIQQ", GGUF_MAGIC, 3, len(tensors), len(w.metadata)))
        for k, (ty, v) in w.metadata.items():
            f.write(w._kv_bytes(k, ty, v))
        off = 0
        for name, gs, ty, data in tensors:
            f.write(struct.pack("<Q", len(name)) + name.encode())
            f.write(struct.pack("<I", len(gs)) + struct.pack("<%dQ" % len(gs), *gs))
            f.write(struct.pack("<IQ", GGUF_TYPE[ty], off))
            off = (off + len(data) + 31) // 32 * 32
        f.write(b"\0" * ((-f.tell()) % 32))
        for name, gs, ty, data in tensors:
            mv = memoryview(data) if not isinstance(data, np.ndarray) else memoryview(np.ascontiguousarray(data))
            for i in range(0, len(mv), 1 << 28):
                f.write(mv[i:i + (1 << 28)])
            f.write(b"\0" * ((-len(mv)) % 32))
    print("%s: %d tensors, %.2f GB" % (gpath.name, len(tensors), gpath.stat().st_size / 1e9))

    # ---- dense.bin + index.txt (the float rows; the embedding stays in the GGUF, read per token)
    out.mkdir(parents=True, exist_ok=True)
    at = 0
    by = {n: data for n, _, _, data in tensors}
    with replace_atomic(out / "dense.bin") as f:
        for r in rows:
            if r[2] not in ("2", "4"):
                continue
            data = by[r[0]]
            r[3] = str(at)
            f.write(data)
            pad = (-len(data)) % ALIGN
            f.write(b"\0" * pad)
            at += len(data) + pad
    pool = 0
    for r in rows:
        r[5] = str(pool)
        pool += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with replace_atomic(out / "index.txt") as fb, io.TextIOWrapper(fb, encoding="utf-8", newline="\n") as f:
        f.write("# strata pack index v3 -- generated by tools/exl3_pack.py (EXL3) from %s\n" % d.name)
        f.write("# align %d pool %d tensors %d\n" % (ALIGN, pool, len(rows)))
        for r in rows:
            f.write(" ".join(r) + "\n")
    with replace_atomic(out / "native_experts.txt") as fb, io.TextIOWrapper(fb, encoding="utf-8", newline="\n") as f:
        f.write("# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                "[shard | gate_shard up_shard down_shard] (n_expert %d, total 0; absolute offsets in %s, or in the "
                "named shard(s) beside it) - EXL3: the experts are in exl3.txt\n" % (t["n_routed_experts"], GGUF_NAME))

    # ---- exl3.txt
    with replace_atomic(out / "exl3.txt") as fb, io.TextIOWrapper(fb, encoding="utf-8", newline="\n") as f:
        f.write("# maya exl3 pack v1 -- generated by tools/exl3_pack.py from %s\n" % d.name)
        f.write("# file <i> <name>  |  mat <name> <file> <trellis_off> <suh_off> <svh_off> <k> <n> <K> <cb>  |\n")
        f.write("# layer <l> <cb> <K_gate> <K_up> <K_down> <k_gate> <n_gate> <k_down> <n_down> <gu_bytes> <dn_bytes>"
                "  |  e <l> <expert> then per part (gate, up, down): <file> <trellis_off> <file> <suh_off> <file> "
                "<svh_off>  (absolute; the blob is [gate | up | down], a part [trellis | suh | svh])\n")
        for i, fn in enumerate(st.files):
            f.write("file %d %s\n" % (i, fn))
        for name, hf, k, n, K, cb in plan.mats:
            tf = st.t[hf + ".trellis"]
            f.write("mat %s %d %d %d %d %d %d %d %d\n" % (name, st.files.index(tf[0]), tf[3], st.t[hf + ".suh"][3],
                                                          st.t[hf + ".svh"][3], k, n, K, cb))
        for L in plan.layers:
            f.write("layer %d %d %d %d %d %d %d %d %d %d %d\n" % (
                L["layer"], L["cb"], *L["K"], *L["shape"], L["gu_bytes"], L["dn_bytes"]))
            for e, parts in L["rows"]:
                f.write("e %d %d %s\n" % (L["layer"], e, " ".join("%d %d %d %d %d %d" % p for p in parts)))
    n_exp = sum(len(L["rows"]) for L in plan.layers)
    print("pack: %d float rows (dense.bin %.2f GB), %d EXL3 matrices, %d MoE layers x %d experts"
          % (sum(1 for r in rows if r[2] in "24"), at / 1e9, len(plan.mats), len(plan.layers),
             n_exp // max(1, len(plan.layers))))

    # ---- tokenizer/ through the GGUF, as for every other pack
    import strata_tokenizer
    cfg = strata_tokenizer.extract(gpath, out)
    print("tokenizer/: vocab %d, merges %d, pre %s" % (cfg["vocab_size"], cfg["n_merges"], cfg["pre"]))


# ------------------------------------------------------------------ --check
def check(plan: Plan, d: pathlib.Path, ref: pathlib.Path, n_cols: int) -> int:
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGUFReader
    from gguf.quants import dequantize
    shards = [ref]
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", ref.name)
    if m:
        shards = [ref.parent / ("%s-%05d-of-%s.gguf" % (m.group(1), i, m.group(2))) for i in range(1, int(m.group(2)) + 1)]
    R = {}
    for s in shards:
        for tt in GGUFReader(s).tensors:
            R[tt.name] = tt
    # unsloth / Maya spell the arch without the hyphen, the tensors the same

    def ref_rows(name, r0=None, r1=None):
        # gguf-py shapes a tensor's data (rows, row bytes) / (rows, cols): slice the rows before dequantizing
        tt = R[name]
        raw = tt.data if r0 is None else tt.data.reshape(-1, tt.data.shape[-1])[r0:r1]
        return dequantize(raw, tt.tensor_type).astype(np.float32)
    bad = 0
    print("float tensors (converted vs the GGUF):")
    for name, shape, ty, fn in plan.floats:
        if fn is None or name not in R:
            continue
        ty = ty.split(":")[0]
        a = np.ascontiguousarray(fn(), dtype=np.float32).reshape(-1)
        if ty == "BF16":
            a = (to_bf16(a)[0].astype(np.uint32) << 16).view(np.float32)
        b = ref_rows(name).reshape(-1)
        if a.size != b.size:
            print("  %-40s size %d vs %d  FAIL" % (name, a.size, b.size))
            bad += 1
            continue
        if R[name].tensor_type.name not in ("F32", "F16", "BF16"):
            # the GGUF quantized it: the layout still shows (a transposed k_b would be ~0)
            cos = float((a * b).sum() / math.sqrt((a * a).sum() * (b * b).sum()))
            ok = cos > 0.99
            if not ok or re.match(r"blk\.(0|3)\.", name):
                print("  %-40s cos %.5f %s (GGUF %s)" % (name, cos, "ok" if ok else "FAIL", R[name].tensor_type.name))
            bad += not ok
            continue
        d_ = np.abs(a - b).max() / max(1e-30, np.abs(b).max())
        ok = d_ <= 1e-2
        if not ok or re.match(r"blk\.(0|3)\.|output_norm", name):
            print("  %-40s max rel %.2e %s (GGUF %s)" % (name, d_, "ok" if ok else "FAIL", R[name].tensor_type.name))
        bad += not ok
    print("EXL3 matrices (decoded vs the GGUF's quant, first %d rows):" % n_cols)
    st = plan.st
    picks = [x for x in plan.mats if re.match(r"(blk\.(0|3|20|44|45)\.|output)", x[0])]
    for name, hf, k, n, K, cb in picks:
        nc = min(max(128, n_cols // 128 * 128), n)   # whole output rotation blocks
        tr = st.raw(hf + ".trellis").view(np.int16).reshape(st.t[hf + ".trellis"][2])
        suh = st.f32(hf + ".suh")
        svh = st.f32(hf + ".svh")
        views = [(name, 0)]
        if name.endswith("attn_qkv.weight"):
            di = n // 3
            views = [(name.replace("attn_qkv", "attn_" + q), i * di) for i, q in enumerate("qkv")]
        for rname, c0 in views:
            if rname not in R:
                print("  %-40s not in the GGUF" % rname)
                continue
            got = exl3_rows(tr, suh, svh, K, cb, c0, nc)
            want = ref_rows(rname, 0, nc)
            cos = float((got * want).sum() / math.sqrt((got * got).sum() * (want * want).sum()))
            ok = cos > 0.9
            print("  %-40s K%d cos %.4f  %s (GGUF %s)" % (rname, K, cos, "ok" if ok else "FAIL",
                                                     R[rname].tensor_type.name))
            bad += not ok
    for L in plan.layers:
        if L["layer"] not in (3, 25, 44, 45):
            continue
        il, e = L["layer"], 7
        for ri, r in enumerate(ROLES):
            p = "model.language_model.layers.%d.mlp.experts.%d.%s_proj." % (il, e, r)
            tr = st.raw(p + "trellis").view(np.int16).reshape(st.t[p + "trellis"][2])
            got = exl3_rows(tr, st.f32(p + "suh"), st.f32(p + "svh"), L["K"][ri], L["cb"])
            tt = R["blk.%d.ffn_%s_exps.weight" % (il, r)]
            want = dequantize(tt.data[e], tt.tensor_type).astype(np.float32)
            cos = float((got * want).sum() / math.sqrt((got * got).sum() * (want * want).sum()))
            ok = cos > 0.85
            print("  blk.%d expert %d %-5s K%d cos %.4f  %s (GGUF %s)" % (il, e, r, L["K"][ri], cos,
                                                                    "ok" if ok else "FAIL", tt.tensor_type.name))
            bad += not ok
    print("check: %s" % ("PASS" if bad == 0 else "%d FAILED" % bad))
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="the EXL3 model directory (config.json, *.safetensors)")
    ap.add_argument("--out", help="the pack directory (default: <model>/pack)")
    ap.add_argument("--check", help="a GGUF of the same model (shard 1): compare instead of / before packing")
    ap.add_argument("--check-rows", type=int, default=512, help="rows of each EXL3 matrix --check decodes")
    ap.add_argument("--check-only", action="store_true", help="--check without writing the pack")
    ap.add_argument("--no-mtp", action="store_true", help="leave the NextN (MTP) draft block out")
    ap.add_argument("--model-template", action="store_true",
                    help="the model directory's chat_template.jinja instead of Maya's GLM template")
    a = ap.parse_args()
    d = pathlib.Path(a.model).expanduser().absolute()
    out = pathlib.Path(a.out).expanduser().absolute() if a.out else d / "pack"
    cfg = json.loads((d / "config.json").read_text())
    if cfg.get("architectures", [None])[0] != "Glm5NextForConditionalGeneration":
        print("%s is %s; the EXL3 packer knows GLM-5.3-Flash (Glm5NextForConditionalGeneration)"
              % (d.name, cfg.get("architectures")))
        return 1
    q = cfg.get("quantization_config", {})
    if q.get("quant_method") != "exl3":
        print("%s is not an EXL3 model (quantization_config.quant_method %r)" % (d.name, q.get("quant_method")))
        return 1
    t = cfg["text_config"]
    print("%s: EXL3 %s, %s bpw (head %s), codebook %s" % (d.name, q.get("version"), q.get("bits"), q.get("head_bits"),
                                                          q.get("codebook", "3inst")))
    st = SafeTensors(d)
    plan = Plan(st, t)
    nextn = int(t.get("num_nextn_predict_layers", 0) or 0)
    mtp = nextn > 0 and not a.no_mtp and st.has("model.language_model.layers.%d.eh_proj.trellis" % t["num_hidden_layers"])
    if nextn > 0 and not mtp and not a.no_mtp:
        print("note: the config names a NextN block but the files do not carry it - packed without MTP")
    plan.build(mtp=mtp)
    if a.check:
        rc = check(plan, d, pathlib.Path(a.check).expanduser().absolute(), a.check_rows)
        if rc or a.check_only:
            return rc
    template = None if a.model_template else MAYA_TEMPLATE
    if template is not None and not template.exists():
        print("%s is missing; use --model-template" % template)
        return 1
    write_all(plan, d, out, t, 1 if mtp else 0, template)
    for n in plan.notes:
        print("note: " + n)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

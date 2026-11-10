"""tools/glm_dsa_synth_gguf.py - synthetic glm-dsa (GLM-5.3 744B-A40B layout) GGUFs for glm-dsa.

BUILD-PLAN B3.2.  Every file is a REAL GGUF written by gguf-py's GGUFWriter, with the tensor names,
ggml dims and metadata keys llama.cpp's glm-dsa loader reads (src/models/glm-dsa.cpp
load_arch_hparams/load_arch_tensors and src/llama-arch.cpp at aa94f208, gguf-py MODEL_ARCH.GLM_DSA,
conversion/glm.py GlmMoeDsaModel + conversion/deepseek.py DeepseekV2Model).  The oracle
(`ref/glm_dsa.py`), the engine's runner tests and llama.cpp itself all load the same file.

Presets (geometry in PRESETS below):
  SMALL   8 trunk layers + MTP, n_embd 128, kv_lora 32 + rope 16, indexer 4 x 64, top-k 8, F32.
          The CI fixture: indexer schedule {0,1,2,6} written explicitly, hot SwiGLU rows (|x| > 10
          so a clamp-10 misreading shows), a tiny-variance indexer key projection (so the LayerNorm
          eps shows), indexer weights ALSO on the shared layers (early unsloth GLM-5.2 files do this,
          llama.cpp #25407; a correct reader ignores them, the shared-as-full misreading uses them).
  LONG    SMALL geometry with the real top-k 2048 (n > 2048 exercises selection); no
          indexer.types key, so readers must apply llama.cpp's back-compat rule (ctx >= 1048576 ->
          the GLM-5.2 pattern).
  BIG     78 trunk layers + MTP, real MLA head geometry (attn_k_b {192,512,64}, attn_v_b
          {512,256,64}), real indexer 32 x 128 top-k 2048, per-layer KV arrays with 79 entries.
          About 2.3 GB; generated streaming (one tensor in memory at a time), never in CI.
  WIDE    n_embd 6144 and the real expert matrix 6144 x 2048 in community types (IQ2_XXS/Q2_K),
          4 layers + MTP.  About 1 GB, never in CI.
  TIERS   256 routed experts x top-8 (the real expert count) in IQ types, 6 layers + MTP.
  PACK    a community-file look-alike for tools/iq_pack.py and the pack upcasts (B1.6): K/IQ
          types on the 256-wide tensors, attn_k_b/attn_v_b in a 32-block type (--kb-type: Q8_0
          default, Q4_0, Q5_0, IQ4_NL, F16, BF16), router and norms F32.
  PACK9   PACK with 10 MoE layers whose (gate/up, down) expert types form 9 size classes (B3.7).
  PFIQ1M  a geometry the batched prompt path covers (16 heads, 2 x n_ff_exp < n_embd) with IQ1_M experts
          in gate/up, in down and in both, beside MMQ types: the prompt path's FP16 expert products.

Determinism: every tensor is drawn from its own PCG64 stream seeded by (seed, crc32(name)), turned
into values with IEEE-exact arithmetic only (no libm), quantized with deterministic code, and
written in a fixed order; so the bytes depend on (preset, seed, options) and the gguf-py version,
not on the machine.  `--sha256` prints the file hash; data/glm-dsa/sha256.txt records the
reference hashes at the default seeds.

Quantized tensors: the 32-block types (Q8_0, Q4_0, Q4_1, Q5_0, Q5_1) and BF16 are produced by
gguf-py's quantizers from float weights, IQ4_NL by a nearest-value quantizer here; the K and IQ
types are "random valid blocks": random code bytes with the block scales set so the dequantized
values have the target RMS.  Either way the dequantized values ARE the model: the oracle and the
engine both start from the same bytes.

Usage:
    python tools/glm_dsa_synth_gguf.py --preset SMALL --out small.gguf [--seed N] [--sha256]
    python tools/glm_dsa_synth_gguf.py --preset PACK --kb-type IQ4_NL --out pack-iq4nl.gguf
    python tools/glm_dsa_synth_gguf.py --all --out-dir DIR [--include-large]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import math
import pathlib
import sys
import zlib
from typing import Callable

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))


def _gguf():
    from strata_gguf import import_gguf
    return import_gguf()


ARCH = "glm-dsa"
DEFAULT_SEED = 5300

# GLM-5.3 indexer schedule rule (config index_topk_freq 4, index_skip_topk_offset 3): layer i is
# "full" iff max(i - 3 + 1, 0) % 4 == 0, i.e. 0, 1, 2, 6, 10, ..., 74 (llama.cpp
# GLM_5_2_DEFAULT_INDEXER_TYPES has the same 78 entries).
def glm_indexer_full(il: int) -> bool:
    return max(il - 3 + 1, 0) % 4 == 0


# ----------------------------------------------------------------------------- presets
# n_layer is the TRUNK count; block_count = n_layer + nextn.  Types map tensor roles to ggml types.
F32_TYPES: dict[str, str] = {}

COMMUNITY_TYPES = dict(          # what an unsloth UD-IQ2_M-like recipe puts where (by shape class)
    token_embd="Q4_K", output="Q6_K",
    attn_q_a="Q4_K", attn_q_b="Q4_K", attn_kv_a_mqa="Q6_K", attn_k_b="Q8_0", attn_v_b="Q8_0",
    attn_output="Q4_K",
    idx_attn_q_b="Q8_0", idx_attn_k="Q8_0", idx_proj="F32",
    ffn_gate="Q4_K", ffn_up="Q4_K", ffn_down="Q5_K",
    exp_gate="IQ2_XXS", exp_up="IQ2_XXS", exp_down="Q2_K",
    shexp_gate="Q5_K", shexp_up="Q5_K", shexp_down="Q6_K",
    eh_proj="Q8_0",
)

PRESETS: dict[str, dict] = {
    "SMALL": dict(
        n_layer=8, nextn=1, n_embd=128, n_vocab=512, n_head=4, q_lora=64, kv_lora=32, rope=16,
        qk_nope=48, v_head=32, idx_heads=16, idx_dim=64, top_k=8, n_expert=16, n_used=4,
        n_ff_exp=64, n_ff=192, dense_lead=3, ctx=1048576, write_types=True, idx_on_shared=True,
        hot_rows=8, hot_gain=6.0, idx_k_gain=2e-3, arrays79=False, types=F32_TYPES),
    "LONG": dict(
        n_layer=8, nextn=1, n_embd=128, n_vocab=512, n_head=4, q_lora=64, kv_lora=32, rope=16,
        qk_nope=48, v_head=32, idx_heads=16, idx_dim=64, top_k=2048, n_expert=16, n_used=4,
        n_ff_exp=64, n_ff=192, dense_lead=3, ctx=1048576, write_types=False, idx_on_shared=False,
        hot_rows=0, hot_gain=1.0, idx_k_gain=1.0, arrays79=False, types=F32_TYPES),
    "BIG": dict(
        n_layer=78, nextn=1, n_embd=256, n_vocab=1024, n_head=64, q_lora=256, kv_lora=512, rope=64,
        qk_nope=192, v_head=256, idx_heads=32, idx_dim=128, top_k=2048, n_expert=16, n_used=8,
        n_ff_exp=256, n_ff=512, dense_lead=3, ctx=1048576, write_types=True, idx_on_shared=False,
        hot_rows=0, hot_gain=1.0, idx_k_gain=1.0, arrays79=True,
        types=dict(attn_q_b="Q8_0", attn_k_b="Q8_0", attn_v_b="Q8_0", attn_output="Q8_0",
                   exp_gate="IQ2_XXS", exp_up="IQ2_XXS", exp_down="Q2_K")),
    "WIDE": dict(
        n_layer=4, nextn=1, n_embd=6144, n_vocab=2048, n_head=8, q_lora=2048, kv_lora=512, rope=64,
        qk_nope=192, v_head=256, idx_heads=32, idx_dim=128, top_k=2048, n_expert=12, n_used=8,
        n_ff_exp=2048, n_ff=12288, dense_lead=1, ctx=1048576, write_types=True, idx_on_shared=False,
        hot_rows=0, hot_gain=1.0, idx_k_gain=1.0, arrays79=False, types=COMMUNITY_TYPES),
    "TIERS": dict(
        n_layer=6, nextn=1, n_embd=256, n_vocab=1024, n_head=4, q_lora=256, kv_lora=512, rope=64,
        qk_nope=192, v_head=256, idx_heads=8, idx_dim=128, top_k=2048, n_expert=256, n_used=8,
        n_ff_exp=256, n_ff=512, dense_lead=1, ctx=1048576, write_types=True, idx_on_shared=False,
        hot_rows=0, hot_gain=1.0, idx_k_gain=1.0, arrays79=False,
        types=dict(COMMUNITY_TYPES, exp_gate="IQ2_XXS", exp_up="IQ2_XXS", exp_down="IQ3_XXS")),
    "PACK": dict(
        n_layer=6, nextn=1, n_embd=256, n_vocab=1024, n_head=4, q_lora=256, kv_lora=512, rope=64,
        qk_nope=192, v_head=256, idx_heads=8, idx_dim=128, top_k=2048, n_expert=16, n_used=8,
        n_ff_exp=256, n_ff=512, dense_lead=2, ctx=1048576, write_types=True, idx_on_shared=False,
        hot_rows=0, hot_gain=1.0, idx_k_gain=1.0, arrays79=False, types=COMMUNITY_TYPES,
        # per MoE layer (gate=up type, down type); cycles if shorter than the MoE layer count
        expert_classes=[("IQ2_XXS", "Q2_K"), ("IQ2_S", "IQ3_XXS"), ("IQ2_XS", "Q3_K"),
                        ("IQ1_S", "IQ2_XXS"), ("Q2_K", "Q4_K")]),
    "PACK9": dict(
        n_layer=12, nextn=1, n_embd=256, n_vocab=1024, n_head=4, q_lora=256, kv_lora=512, rope=64,
        qk_nope=192, v_head=256, idx_heads=8, idx_dim=128, top_k=2048, n_expert=16, n_used=8,
        n_ff_exp=256, n_ff=512, dense_lead=2, ctx=1048576, write_types=True, idx_on_shared=False,
        hot_rows=0, hot_gain=1.0, idx_k_gain=1.0, arrays79=False, types=COMMUNITY_TYPES,
        # 10 MoE layers + MTP, 9 distinct (gate/up, down) row-byte classes (the 10th repeats the 1st)
        expert_classes=[("IQ2_XXS", "Q2_K"), ("IQ2_S", "IQ3_XXS"), ("IQ2_XS", "Q3_K"),
                        ("IQ1_S", "IQ2_XXS"), ("Q2_K", "Q4_K"), ("IQ3_XXS", "IQ3_S"),
                        ("Q3_K", "Q5_K"), ("IQ3_S", "Q6_K"), ("IQ4_XS", "IQ4_XS")]),
    "PFIQ1M": dict(   # (n_embd 768 > 2 x n_ff_exp: the down's outputs overlap later h rows, as at 6144 / 2048)
        n_layer=4, nextn=1, n_embd=768, n_vocab=1024, n_head=16, q_lora=256, kv_lora=512, rope=64,
        qk_nope=192, v_head=256, idx_heads=8, idx_dim=128, top_k=2048, n_expert=16, n_used=8,
        n_ff_exp=256, n_ff=512, dense_lead=1, ctx=1048576, write_types=True, idx_on_shared=False,
        hot_rows=0, hot_gain=1.0, idx_k_gain=1.0, arrays79=False, types=COMMUNITY_TYPES,
        # 3 MoE layers + MTP (the MTP block repeats the first class)
        expert_classes=[("IQ1_M", "IQ2_XXS"), ("IQ1_S", "IQ1_M"), ("IQ1_M", "IQ1_M")]),
}
LARGE_PRESETS = ("BIG", "WIDE")
KB_TYPES = ("Q8_0", "Q4_0", "Q5_0", "Q4_1", "Q5_1", "IQ4_NL", "F16", "BF16", "F32")

# ----------------------------------------------------------------------------- deterministic values
class Streams:
    """One PCG64 stream per tensor name.  SeedSequence and the PCG64 raw output are stable across
    numpy versions and platforms; the conversions below use only IEEE + and * (no libm)."""

    def __init__(self, seed: int):
        self.seed = int(seed)

    def gen(self, name: str) -> np.random.Generator:
        ss = np.random.SeedSequence([self.seed, zlib.crc32(name.encode("utf-8"))])
        return np.random.Generator(np.random.PCG64(ss))

    def raw(self, name: str, n: int) -> np.ndarray:
        return self.gen(name).bit_generator.random_raw(n).astype("<u8")

    def normal(self, name: str, n: int) -> np.ndarray:
        """n float64 values, mean 0, variance 1 (Irwin-Hall of 4 uniforms, rescaled)."""
        r = self.raw(name, 4 * n).reshape(4, n)
        u = (r >> np.uint64(11)).astype(np.float64) * (2.0 ** -53)
        s = ((u[0] + u[1]) + u[2]) + u[3]
        return (s - 2.0) * 1.7320508075688772          # sqrt(12/4)

    def bytes_(self, name: str, n: int) -> np.ndarray:
        return self.raw(name, (n + 7) // 8).view(np.uint8)[:n].copy()


# ----------------------------------------------------------------------------- quantizers
IQ4NL_VALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 88, 113],
                        dtype=np.float32)


def quantize_iq4_nl(x: np.ndarray) -> np.ndarray:
    """Rows of float32 -> IQ4_NL blocks (uint8, 18 bytes per 32 values).  The scale maps the
    largest-magnitude value onto -127 (ggml's sign trick); codes are the nearest table entry."""
    b = x.reshape(-1, 32).astype(np.float32)
    idx = np.argmax(np.abs(b), axis=1)
    mx = b[np.arange(b.shape[0]), idx]
    d = (mx / np.float32(-127.0)).astype(np.float16)
    df = d.astype(np.float32)
    inv = np.where(df != 0, np.float32(1.0) / np.where(df != 0, df, 1), 0).astype(np.float32)
    q = (b * inv[:, None])[:, :, None]
    L = np.argmin(np.abs(q - IQ4NL_VALUES[None, None, :]), axis=2).astype(np.uint8)
    qs = (L[:, :16] | (L[:, 16:] << 4)).astype(np.uint8)
    out = np.concatenate([d.view(np.uint8).reshape(-1, 2), qs], axis=1)
    return out.reshape(x.shape[:-1] + (x.shape[-1] // 32 * 18,))


# (f16 scale field byte offsets, dmin/d ratio for the field at index 1 if any) for random blocks
_RANDOM_LAYOUT: dict[str, tuple[tuple[int, ...], float]] = {
    "Q2_K": ((80, 82), 1.5), "Q3_K": ((108,), 0.0), "Q4_K": ((0, 2), 7.5), "Q5_K": ((0, 2), 15.5),
    "Q6_K": ((208,), 0.0), "IQ2_XXS": ((0,), 0.0), "IQ2_XS": ((0,), 0.0), "IQ2_S": ((0,), 0.0),
    "IQ3_XXS": ((0,), 0.0), "IQ3_S": ((0,), 0.0), "IQ1_S": ((0,), 0.0), "IQ4_XS": ((0,), 0.0),
    "IQ1_M": ((), 0.0),   # (its f16 scale is the top nibbles of the 4 scale words, bytes 48..55: random_blocks)
}
FLOAT_QUANT = ("Q8_0", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "BF16")


def quant_geometry(type_name: str) -> tuple[int, int]:
    g = _gguf()
    return g.constants.GGML_QUANT_SIZES[g.GGMLQuantizationType[type_name]]


def random_blocks(st: Streams, name: str, type_name: str, n_values: int, rms: float) -> np.ndarray:
    """Random code bytes of a K/IQ type with the block scales set so the dequantized values have
    root-mean-square `rms`.  Dequantization is linear in the f16 scale fields (dmin is tied to d by a
    fixed ratio that centres K-quant blocks), so one calibration on the first blocks fixes them."""
    g = _gguf()
    blk, bsz = quant_geometry(type_name)
    if n_values % blk:
        raise ValueError(f"{name}: {n_values} values is not a multiple of the {type_name} block {blk}")
    nb = n_values // blk
    offs, ratio = _RANDOM_LAYOUT[type_name]
    data = st.bytes_(name, nb * bsz).reshape(nb, bsz)

    def set_scales(d: float) -> None:
        if type_name == "IQ1_M":
            h = int(np.array([d], dtype=np.float16).view(np.uint16)[0])
            sc = np.ascontiguousarray(data[:, 48:56]).view("<u2").copy()
            for k in range(4):
                sc[:, k] = (sc[:, k] & 0x0FFF) | (((h >> (4 * k)) & 0xF) << 12)
            data[:, 48:56] = sc.view(np.uint8).reshape(nb, 8)
            return
        vals = [d, d * ratio] if len(offs) == 2 else [d]
        for off, v in zip(offs, vals):
            data[:, off:off + 2] = np.array([v], dtype=np.float16).view(np.uint8)

    set_scales(1.0)
    nc = min(nb, 64)
    probe = np.asarray(g.quants.dequantize(data[:nc].reshape(-1), g.GGMLQuantizationType[type_name]),
                       dtype=np.float64).reshape(-1)
    cur = math.sqrt(math.fsum((probe * probe).tolist()) / probe.size)
    if not cur > 0:
        raise ValueError(f"{name}: degenerate {type_name} probe")
    set_scales(rms / cur)
    return data.reshape(-1)


def byte_symbols() -> list[str]:
    """GPT-2's byte -> printable unicode map (the byte-level BPE alphabet), in byte order."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("\u00a1"), ord("\u00ac") + 1)) \
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return [chr(c) for _, c in sorted(zip(bs, cs))]


# ----------------------------------------------------------------------------- tensor plan
@dataclasses.dataclass
class TSpec:
    name: str
    ne: tuple[int, ...]                  # ggml dims, ne0 fastest
    type_name: str
    make: Callable[[], np.ndarray]       # -> bytes (uint8, flat) of the encoded tensor

    @property
    def n_values(self) -> int:
        return int(np.prod(self.ne))

    def nbytes(self) -> int:
        if self.type_name in ("F32",):
            return self.n_values * 4
        if self.type_name in ("F16", "BF16"):
            return self.n_values * 2
        blk, bsz = quant_geometry(self.type_name)
        return self.n_values // blk * bsz


def encode(values: np.ndarray, type_name: str) -> np.ndarray:
    """Float values shaped (..., ne0) -> flat uint8 encoding of `type_name`."""
    if type_name == "F32":
        return np.ascontiguousarray(values, dtype="<f4").reshape(-1).view(np.uint8)
    if type_name == "F16":
        return np.ascontiguousarray(values, dtype="<f2").reshape(-1).view(np.uint8)
    if type_name == "IQ4_NL":
        return quantize_iq4_nl(np.asarray(values, dtype=np.float32)).reshape(-1)
    if type_name in FLOAT_QUANT:
        g = _gguf()
        q = g.quants.quantize(np.asarray(values, dtype=np.float32), g.GGMLQuantizationType[type_name])
        return np.ascontiguousarray(q).reshape(-1).view(np.uint8)
    raise ValueError(f"cannot encode float values as {type_name}")


class Builder:
    def __init__(self, preset: str, seed: int, kb_type: str | None = None):
        if preset not in PRESETS:
            raise ValueError(f"unknown preset {preset}; known: {', '.join(PRESETS)}")
        self.preset = preset
        self.p = dict(PRESETS[preset])
        self.types = dict(self.p["types"])
        if kb_type:
            if kb_type not in KB_TYPES:
                raise ValueError(f"--kb-type {kb_type}: one of {', '.join(KB_TYPES)}")
            self.types["attn_k_b"] = kb_type
            self.types["attn_v_b"] = kb_type
        self.seed = seed
        self.st = Streams(seed)
        p = self.p
        self.n_block = p["n_layer"] + p["nextn"]
        self.k_mla = p["qk_nope"] + p["rope"]
        self.idx_full = [glm_indexer_full(il) for il in range(p["n_layer"])]

    # ---- value generators (float64 -> float32)
    def _w(self, name: str, ne: tuple[int, ...], gain: float, hot: bool = False) -> np.ndarray:
        n = int(np.prod(ne))
        v = self.st.normal(name, n) * (gain / math.sqrt(ne[0]))
        v = v.reshape(tuple(reversed(ne)))                    # numpy (…, ne1, ne0)
        if hot and self.p["hot_rows"]:
            # every hot_rows-th output row (ne1 index) is hot: its SwiGLU inputs leave [-10, 10]
            v[..., :: self.p["hot_rows"], :] *= self.p["hot_gain"]
        return v.astype(np.float32)

    def _norm(self, name: str, n: int, base: float = 1.0, spread: float = 0.1) -> np.ndarray:
        return (base + spread * self.st.normal(name, n)).astype(np.float32)

    def type_of(self, role: str) -> str:
        return self.types.get(role, "F32")

    def mat(self, name: str, ne: tuple[int, ...], role: str, gain: float, hot: bool = False) -> TSpec:
        t = self.type_of(role)
        if t in _RANDOM_LAYOUT:
            n = int(np.prod(ne))
            return TSpec(name, ne, t, lambda: random_blocks(self.st, name, t, n, gain / math.sqrt(ne[0])))
        return TSpec(name, ne, t, lambda: encode(self._w(name, ne, gain, hot), t))

    def vec(self, name: str, n: int, base: float = 1.0, spread: float = 0.1) -> TSpec:
        return TSpec(name, (n,), "F32", lambda: encode(self._norm(name, n, base, spread), "F32"))

    # ---- metadata
    def metadata(self) -> list[tuple[str, str, object]]:
        """(key, kind, value) in write order; kind in u32/f32/bool/str/arr_u32/arr_bool/arr_str/arr_i32."""
        p = self.p
        a = ARCH
        nb = self.n_block
        md: list[tuple[str, str, object]] = [
            ("general.name", "str", f"glm-dsa-synth-{self.preset.lower()}"),
            (f"{a}.block_count", "u32", nb),
            (f"{a}.context_length", "u32", p["ctx"]),
            (f"{a}.embedding_length", "u32", p["n_embd"]),
        ]
        if p["arrays79"]:
            md += [(f"{a}.feed_forward_length", "arr_i32", [p["n_ff"]] * nb),
                   (f"{a}.attention.head_count", "arr_i32", [p["n_head"]] * nb),
                   (f"{a}.attention.head_count_kv", "arr_i32", [1] * nb)]
        else:
            md += [(f"{a}.feed_forward_length", "u32", p["n_ff"]),
                   (f"{a}.attention.head_count", "u32", p["n_head"]),
                   (f"{a}.attention.head_count_kv", "u32", 1)]
        md += [
            (f"{a}.rope.freq_base", "f32", 8000000.0),
            (f"{a}.attention.layer_norm_rms_epsilon", "f32", 1e-5),
            (f"{a}.expert_used_count", "u32", p["n_used"]),
            (f"{a}.expert_gating_func", "u32", 2),
            (f"{a}.leading_dense_block_count", "u32", p["dense_lead"]),
            (f"{a}.vocab_size", "u32", p["n_vocab"]),
            (f"{a}.attention.q_lora_rank", "u32", p["q_lora"]),
            (f"{a}.attention.kv_lora_rank", "u32", p["kv_lora"]),
            (f"{a}.attention.key_length", "u32", p["kv_lora"] + p["rope"]),
            (f"{a}.attention.value_length", "u32", p["kv_lora"]),
            (f"{a}.attention.key_length_mla", "u32", self.k_mla),
            (f"{a}.attention.value_length_mla", "u32", p["v_head"]),
        ]
        if p["arrays79"]:
            md += [(f"{a}.expert_feed_forward_length", "arr_i32", [p["n_ff_exp"]] * nb)]
        else:
            md += [(f"{a}.expert_feed_forward_length", "u32", p["n_ff_exp"])]
        md += [
            (f"{a}.expert_count", "u32", p["n_expert"]),
            (f"{a}.expert_shared_count", "u32", 1),
            (f"{a}.expert_group_count", "u32", 1),
            (f"{a}.expert_group_used_count", "u32", 1),
            (f"{a}.expert_weights_scale", "f32", 2.5),
            (f"{a}.expert_weights_norm", "bool", True),
            (f"{a}.rope.dimension_count", "u32", p["rope"]),
            (f"{a}.nextn_predict_layers", "u32", p["nextn"]),
            (f"{a}.attention.indexer.head_count", "u32", p["idx_heads"]),
            (f"{a}.attention.indexer.key_length", "u32", p["idx_dim"]),
            (f"{a}.attention.indexer.top_k", "u32", p["top_k"]),
        ]
        if p["write_types"]:
            md += [(f"{a}.attention.indexer.types", "arr_bool", list(self.idx_full))]
        # a parseable byte-level BPE vocabulary so upstream llama.cpp can load the file (the
        # oracle runs on ids): the 256 GPT-2 byte symbols, one merge, then filler tokens
        toks = byte_symbols() + ["\u0120t"]
        toks += [f"<t{i}>" for i in range(len(toks), p["n_vocab"])]
        md += [
            ("tokenizer.ggml.model", "str", "gpt2"),
            ("tokenizer.ggml.pre", "str", "glm4"),
            ("tokenizer.ggml.tokens", "arr_str", toks),
            ("tokenizer.ggml.token_type", "arr_i32", [1] * p["n_vocab"]),
            ("tokenizer.ggml.merges", "arr_str", ["\u0120 t"]),
        ]
        return md

    # ---- tensors, in llama.cpp's create_tensor order
    def tensors(self) -> list[TSpec]:
        p = self.p
        E, V, H = p["n_embd"], p["n_vocab"], p["n_head"]
        out = [
            self.mat("token_embd.weight", (E, V), "token_embd", gain=math.sqrt(E)),
            self.vec("output_norm.weight", E),
            self.mat("output.weight", (E, V), "output", gain=3.0),
        ]
        moe_layers = [il for il in range(self.n_block) if il >= p["dense_lead"]]
        classes = p.get("expert_classes")
        for il in range(self.n_block):
            b = f"blk.{il}."
            is_mtp = il >= p["n_layer"]
            out += [
                self.vec(b + "attn_norm.weight", E),
                self.vec(b + "attn_q_a_norm.weight", p["q_lora"]),
                self.vec(b + "attn_kv_a_norm.weight", p["kv_lora"]),
                self.mat(b + "attn_q_a.weight", (E, p["q_lora"]), "attn_q_a", 1.0),
                self.mat(b + "attn_q_b.weight", (p["q_lora"], H * self.k_mla), "attn_q_b", 1.0),
                self.mat(b + "attn_kv_a_mqa.weight", (E, p["kv_lora"] + p["rope"]), "attn_kv_a_mqa", 1.0),
                self.mat(b + "attn_k_b.weight", (p["qk_nope"], p["kv_lora"], H), "attn_k_b", 1.5),
                self.mat(b + "attn_v_b.weight", (p["kv_lora"], p["v_head"], H), "attn_v_b", 1.0),
                self.mat(b + "attn_output.weight", (H * p["v_head"], E), "attn_output", 0.7),
                self.vec(b + "ffn_norm.weight", E),
            ]
            full = is_mtp or self.idx_full[il]
            if full or p["idx_on_shared"]:
                D, NH = p["idx_dim"], p["idx_heads"]
                out += [
                    self.vec(b + "indexer.k_norm.weight", D),
                    self.vec(b + "indexer.k_norm.bias", D, base=0.0, spread=0.1),
                    self.mat(b + "indexer.proj.weight", (E, NH), "idx_proj", 1.0),
                    self.mat(b + "indexer.attn_k.weight", (E, D), "idx_attn_k", p["idx_k_gain"]),
                    self.mat(b + "indexer.attn_q_b.weight", (p["q_lora"], NH * D), "idx_attn_q_b", 1.0),
                ]
            if il < p["dense_lead"]:
                out += [
                    self.mat(b + "ffn_gate.weight", (E, p["n_ff"]), "ffn_gate", 1.0, hot=True),
                    self.mat(b + "ffn_down.weight", (p["n_ff"], E), "ffn_down", 0.5),
                    self.mat(b + "ffn_up.weight", (E, p["n_ff"]), "ffn_up", 1.0, hot=True),
                ]
            else:
                F, X = p["n_ff_exp"], p["n_expert"]
                saved = dict(self.types)
                if classes:
                    gu, dn = classes[moe_layers.index(il) % len(classes)]
                    self.types.update(exp_gate=gu, exp_up=gu, exp_down=dn)
                out += [
                    self.mat(b + "ffn_gate_inp.weight", (E, X), "router", 1.0),
                    self.vec(b + "exp_probs_b.bias", X, base=0.0, spread=0.3),
                    self.mat(b + "ffn_gate_exps.weight", (E, F, X), "exp_gate", 1.0, hot=True),
                    self.mat(b + "ffn_down_exps.weight", (F, E, X), "exp_down", 0.5),
                    self.mat(b + "ffn_up_exps.weight", (E, F, X), "exp_up", 1.0, hot=True),
                    self.mat(b + "ffn_gate_shexp.weight", (E, F), "shexp_gate", 1.0, hot=True),
                    self.mat(b + "ffn_down_shexp.weight", (F, E), "shexp_down", 0.5),
                    self.mat(b + "ffn_up_shexp.weight", (E, F), "shexp_up", 1.0, hot=True),
                ]
                self.types = saved
            if is_mtp:
                out += [
                    self.mat(b + "nextn.eh_proj.weight", (2 * E, E), "eh_proj", 1.0),
                    self.vec(b + "nextn.enorm.weight", E),
                    self.vec(b + "nextn.hnorm.weight", E),
                    self.vec(b + "nextn.shared_head_norm.weight", E),
                ]
        return out


# ----------------------------------------------------------------------------- writer
def write(path: str | pathlib.Path, preset: str, seed: int = DEFAULT_SEED, kb_type: str | None = None,
          split_max_size: int = 0, verbose: bool = False) -> list[pathlib.Path]:
    """Write the preset (streaming: one tensor's values in memory at a time).  Returns the paths
    written (more than one when split_max_size splits the file into shards)."""
    g = _gguf()
    b = Builder(preset, seed, kb_type)
    path = pathlib.Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    w = g.GGUFWriter(path, ARCH, split_max_size=split_max_size)
    VT = g.GGUFValueType
    for key, kind, val in b.metadata():
        if kind == "u32":
            w.add_uint32(key, int(val))
        elif kind == "f32":
            w.add_float32(key, float(val))
        elif kind == "bool":
            w.add_bool(key, bool(val))
        elif kind == "str":
            w.add_string(key, str(val))
        elif kind == "arr_bool":
            w.add_key_value(key, [bool(x) for x in val], VT.ARRAY, sub_type=VT.BOOL)
        elif kind == "arr_i32":
            w.add_key_value(key, [int(x) for x in val], VT.ARRAY, sub_type=VT.INT32)
        elif kind == "arr_str":
            w.add_key_value(key, list(val), VT.ARRAY, sub_type=VT.STRING)
        else:
            raise ValueError(kind)
    specs = b.tensors()
    for s in specs:
        nbytes = s.nbytes()
        if s.type_name == "F32":
            w.add_tensor_info(s.name, tuple(reversed(s.ne)), np.dtype(np.float32), nbytes)
        elif s.type_name == "F16":
            w.add_tensor_info(s.name, tuple(reversed(s.ne)), np.dtype(np.float16), nbytes)
        else:
            qt = g.GGMLQuantizationType[s.type_name]
            blk, bsz = quant_geometry(s.type_name)
            byte_shape = tuple(reversed(s.ne[1:])) + (s.ne[0] // blk * bsz,)
            w.add_tensor_info(s.name, byte_shape, np.dtype(np.uint8), nbytes, raw_dtype=qt)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    # the files the writer opened: the shards, or just `path` when the model is smaller than split_max_size (gguf-py
    # then logs "not splitting" and writes one file)
    written = [pathlib.Path(p) for p in w.format_shard_names(path)]
    for i, s in enumerate(specs):
        data = s.make()
        if data.dtype != np.uint8 or data.size != s.nbytes():
            raise AssertionError(f"{s.name}: encoded {data.size} B of {data.dtype}, expected {s.nbytes()}")
        w.write_tensor_data(data)
        if verbose and (i % 50 == 0 or i == len(specs) - 1):
            print(f"  {i + 1}/{len(specs)} {s.name} {s.type_name}", flush=True)
    w.close()
    return written


def sha256(path: str | pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest()


def check(paths, preset: str) -> None:
    """Read the file back (tools/gguf_reader + gguf-py dequant) and check names, dims, finiteness."""
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ref"))
    from glm_dsa_load import GgufAny
    b = Builder(preset, DEFAULT_SEED)
    want = {s.name: s.ne for s in b.tensors()}
    with GgufAny(paths[0] if isinstance(paths, list) else paths) as g:
        got = set(g.names())
        if got != set(want):
            raise AssertionError(f"tensor set differs: missing {sorted(set(want) - got)[:5]}, "
                                 f"extra {sorted(got - set(want))[:5]}")
        for n, ne in want.items():
            if tuple(g.info(n).shape) != tuple(ne):
                raise AssertionError(f"{n}: dims {g.info(n).shape} != {ne}")
        for n in ("token_embd.weight", "blk.0.attn_k_b.weight", "output.weight"):
            x = g.tensor(n)
            if not np.isfinite(x).all():
                raise AssertionError(f"{n}: non-finite values")
        if g.metadata.get("general.architecture") != ARCH:
            raise AssertionError("general.architecture")


def default_set(include_large: bool) -> list[tuple[str, str, str | None]]:
    """(file stem, preset, kb_type) for --all."""
    out = [("glm-dsa-small", "SMALL", None), ("glm-dsa-long", "LONG", None),
           ("glm-dsa-tiers", "TIERS", None), ("glm-dsa-pack", "PACK", None),
           ("glm-dsa-pack-q4_0", "PACK", "Q4_0"), ("glm-dsa-pack-iq4_nl", "PACK", "IQ4_NL"),
           ("glm-dsa-pack-f16", "PACK", "F16"), ("glm-dsa-pack9", "PACK9", None)]
    if include_large:
        out += [("glm-dsa-big", "BIG", None), ("glm-dsa-wide", "WIDE", None)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="synthetic glm-dsa GGUF generator (glm-dsa B3.2)")
    ap.add_argument("--preset", choices=sorted(PRESETS), help="one preset")
    ap.add_argument("--out", help="output .gguf (with --preset)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--kb-type", default=None, help=f"attn_k_b/attn_v_b type ({', '.join(KB_TYPES)})")
    ap.add_argument("--split-max-size", type=int, default=0, help="split into shards of at most N bytes")
    ap.add_argument("--all", action="store_true", help="write the standard set into --out-dir")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--include-large", action="store_true", help="with --all: also BIG (~2.3 GB) and WIDE (~1 GB)")
    ap.add_argument("--sha256", action="store_true", help="print the sha256 of each file written")
    ap.add_argument("--check", action="store_true", help="read each file back and validate it")
    a = ap.parse_args()
    if a.all:
        if not a.out_dir:
            ap.error("--all needs --out-dir")
        jobs = [(pathlib.Path(a.out_dir) / f"{stem}.gguf", pre, kb) for stem, pre, kb in default_set(a.include_large)]
    else:
        if not (a.preset and a.out):
            ap.error("give --preset and --out, or --all --out-dir")
        jobs = [(pathlib.Path(a.out), a.preset, a.kb_type)]
    for out, pre, kb in jobs:
        paths = write(out, pre, a.seed, kb, a.split_max_size, verbose=pre in LARGE_PRESETS)
        if a.check:
            check(paths, pre)
        for p in paths:
            line = f"{p.name}\t{p.stat().st_size}"
            if a.sha256:
                line = f"{sha256(p)}  {p.name}\t{p.stat().st_size}"
            print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""ref/load.py - P1.S4: GGUF -> FP32 tensors on demand, with LAZY per-expert dequantization.

The plan is explicit: "experts are dequantized lazily per (layer, expert) - do not materialize 120B
parameters". That is the whole design constraint here, and it is what `expert()` below exists for: the
routed-expert tensors are fused `[in, out, experts]`, and touching one expert must not touch the other
511.

Dequantization of the older encodings is NOT reimplemented. `gguf-py` handles the twelve standard
encodings and is already the reference this project validates its C++ against; `Q2_0` (type 42) is the
one gguf-py cannot represent at all, so it is served by this project's own decoder. That split is
deliberate: it means a bug in `tools/gguf_writer.dequantize_q2_0` cannot hide behind gguf-py and vice
versa.

Kimi K3.1's routed-expert encodings - MXFP4 (type 39) and the i-quants IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
IQ2_S, IQ3_XXS and IQ3_S - are decoded HERE, in numpy, for the same reason: the tests hold these
decoders to gguf-py and to ggml's own C `to_float` bit for bit (tools/test_kimi_k31_load.py), which
checks nothing if one of them is the other. Their codebooks are not copied: they are parsed from
`third_party/ggml/ggml-common.h`, the header Maya's IQ kernels compile.

    from ref.load import Gguf
    g = Gguf("path.gguf")
    w = g.tensor("blk.0.attn_gate.weight")      # (out, in) fp32, whole tensor
    e = g.expert("blk.0.ffn_gate_exps.weight", 7)  # (out, in) fp32, ONE expert only
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "tools"))
from gguf_reader import GGUFFile                     # noqa: E402
from gguf_writer import dequantize_q2_0              # noqa: E402

# gguf-py's member name for each encoding.  The second group is decoded by this module (below); naming it
# here lets gguf_py_dequant() put the same bytes through gguf-py as a cross-check.
_GGUF_PY_NAME = {"Q4_0": "Q4_0", "Q8_0": "Q8_0", "Q4_K": "Q4_K", "Q5_K": "Q5_K", "Q6_K": "Q6_K",
                 "Q3_K": "Q3_K", "IQ4_NL": "IQ4_NL", "IQ4_XS": "IQ4_XS", "Q5_0": "Q5_0",
                 "MXFP4": "MXFP4", "IQ1_S": "IQ1_S", "IQ1_M": "IQ1_M", "IQ2_XXS": "IQ2_XXS",
                 "IQ2_XS": "IQ2_XS", "IQ2_S": "IQ2_S", "IQ3_XXS": "IQ3_XXS", "IQ3_S": "IQ3_S"}

# (elements per block, bytes per block) - mirrors block_geometry() in src/artifact/gguf_reader.cpp;
# the i-quants and MXFP4 are ggml-common.h's block_* sizes (its static_asserts)
GEOM = {"F32": (1, 4), "F16": (1, 2), "BF16": (1, 2),
        "Q4_0": (32, 18), "Q5_0": (32, 22), "Q8_0": (32, 34), "IQ4_NL": (32, 18),
        "Q3_K": (256, 110), "Q4_K": (256, 144), "Q5_K": (256, 176), "Q6_K": (256, 210),
        "IQ4_XS": (256, 136), "Q2_0": (64, 18),
        "MXFP4": (32, 17), "IQ1_S": (256, 50), "IQ1_M": (256, 56), "IQ2_XXS": (256, 66),
        "IQ2_XS": (256, 74), "IQ2_S": (256, 82), "IQ3_XXS": (256, 98), "IQ3_S": (256, 110)}


def _add_gguf_py_types() -> None:
    """Every other type gguf-py knows (the i-quants, MXFP4, ...) with gguf-py's own geometry and
    decoder, so a real K3.1 file (IQ1_S experts) can be read; the entries above take precedence.
    Without gguf-py installed nothing is added and those types stay unknown, as before."""
    try:
        from gguf.constants import GGML_QUANT_SIZES
    except ImportError:
        return
    for q, geom in GGML_QUANT_SIZES.items():
        GEOM.setdefault(q.name, tuple(geom))
        _GGUF_PY_NAME.setdefault(q.name, q.name)


_add_gguf_py_types()


def _dequant_flat(type_name: str, raw: np.ndarray) -> np.ndarray:
    """One dequantization dispatch, shared by tensor()/expert()/rows().

    Q2_0 is this project's own decoder because gguf-py's enum cannot represent type 42 at all;
    MXFP4 and the i-quants are this module's numpy decoders (below); everything else goes through
    gguf-py, which is already the reference this project validates its C++ against.  Keeping one
    dispatch means a type added in one path cannot be forgotten in another.
    """
    raw = _byte_array(raw)
    if type_name == "Q2_0":
        return dequantize_q2_0(raw.tobytes())
    if type_name in _DECODERS:
        return _DECODERS[type_name](raw)
    if type_name == "BF16":
        return _bf16_to_f32(raw)
    if type_name == "F32":
        return raw.view(np.float32)
    if type_name == "F16":
        return raw.view(np.float16).astype(np.float32)
    return gguf_py_dequant(type_name, raw)


def gguf_py_dequant(type_name: str, raw: np.ndarray, quants=None) -> np.ndarray:
    """gguf-py's decode of `raw` (whole blocks of `type_name`), as float32.

    `quants` is a gguf-py `quants` module; the default is whichever `gguf` imports.  The loader tests
    pass a specific checkout's module, so the comparison names the gguf-py it was made against.
    """
    raw = _byte_array(raw)
    if quants is None:
        from gguf import quants
    q = quants.GGMLQuantizationType[_GGUF_PY_NAME[type_name]]
    return quants.dequantize(raw, q).astype(np.float32)


def _byte_array(raw: np.ndarray) -> np.ndarray:
    """`raw` (a uint8 array of whole blocks, any shape) as one contiguous flat run of bytes.  Any other
    dtype is refused: converting it by value would decode numbers that are not the file's bytes."""
    if not isinstance(raw, np.ndarray) or raw.dtype != np.uint8:
        raise TypeError(f"raw blocks must be a uint8 array, not {getattr(raw, 'dtype', type(raw).__name__)}")
    return np.ascontiguousarray(raw).ravel()


# ---- MXFP4 and the i-quants, in numpy
#
# Each decoder below is ggml-quants.c's dequantize_row_<type> written over whole arrays of blocks: the
# same float32 products in the same order (`d * (0.5f + s) * 0.25f` is `(d * (0.5 + s)) * 0.25`, and
# a sign is a multiply by -1.0, so the sign of a zero comes out as ggml's does).  numpy keeps float32
# operands in float32, so every value is bit-equal to ggml's, which the loader tests check.
#
# The codebooks come from third_party/ggml/ggml-common.h, read once.  A grid row is the table entry's
# bytes in memory order (the C code casts `iq2xxs_grid + i` to `uint8_t *`), so a uint64 entry viewed as
# 8 little-endian bytes is the row; iq1s_grid's bytes are int8 values -1/0/+1.

F32 = np.float32
_GGML_COMMON = pathlib.Path(__file__).resolve().parent.parent / "third_party" / "ggml" / "ggml-common.h"
_TABLE_RE = re.compile(r"^GGML_TABLE_BEGIN\((\w+),\s*(\w+),\s*(\w+)\)[ \t]*$(.*?)^GGML_TABLE_END\(\)",
                       re.S | re.M)
_CTYPES = {"uint8_t": "u1", "int8_t": "i1", "uint16_t": "<u2", "uint32_t": "<u4", "uint64_t": "<u8"}
_GRIDS: dict | None = None


def parse_ggml_tables(text: str) -> dict[str, np.ndarray]:
    """Every `GGML_TABLE_BEGIN(type, name, size) ... GGML_TABLE_END()` in a ggml-common.h, as numpy
    arrays.  Refuses a table whose entry count differs from its declared size, and a name seen twice."""
    defines = dict(re.findall(r"^#define[ \t]+(\w+)[ \t]+(\d+)[ \t]*$", text, re.M))
    out: dict[str, np.ndarray] = {}
    for ctype, name, size, body in _TABLE_RE.findall(text):
        if ctype not in _CTYPES:
            continue                                    # (no table the decoders use is of another type)
        if name in out:
            raise ValueError(f"ggml-common.h: table {name} is defined twice")
        vals = [int(v, 0) for v in re.sub(r"//[^\n]*", "", body).replace(",", " ").split()]
        n = int(defines.get(size, size))
        if len(vals) != n:
            raise ValueError(f"ggml-common.h: table {name} has {len(vals)} entries, declared {n}")
        out[name] = np.array(vals, dtype=np.dtype(_CTYPES[ctype]))
    return out


def _ggml_float_define(text: str, name: str) -> np.float32:
    m = re.search(rf"^#define[ \t]+{name}[ \t]+([0-9.eE+-]+)f?[ \t]*$", text, re.M)
    if not m:
        raise ValueError(f"ggml-common.h: no #define {name}")
    return F32(float(m.group(1)))


def _grids() -> dict:
    """The codebooks, parsed from ggml-common.h on first use."""
    global _GRIDS
    if _GRIDS is None:
        if not _GGML_COMMON.is_file():
            raise FileNotFoundError(f"{_GGML_COMMON}: the i-quant codebooks are read from this header")
        text = _GGML_COMMON.read_text()
        t = parse_ggml_tables(text)
        kmask = t["kmask_iq2xs"]
        # a sign byte's 8 multipliers: bit j set -> -1.0 for value j (ggml: `signs & kmask_iq2xs[j]`)
        sign = np.where((np.arange(256, dtype=np.uint8)[:, None] & kmask[None, :]) != 0,
                        F32(-1.0), F32(1.0)).astype(F32)
        _GRIDS = {
            "sign": sign,                                                    # (256, 8) by raw sign byte
            "ksigns": t["ksigns_iq2xs"],                                     # (128,) 7 bits -> sign byte
            "iq2xxs": t["iq2xxs_grid"].view(np.uint8).reshape(256, 8),
            "iq2xs": t["iq2xs_grid"].view(np.uint8).reshape(512, 8),
            "iq2s": t["iq2s_grid"].view(np.uint8).reshape(1024, 8),
            "iq3xxs": t["iq3xxs_grid"].view(np.uint8).reshape(256, 4),
            "iq3s": t["iq3s_grid"].view(np.uint8).reshape(512, 4),
            "iq1s": t["iq1s_grid"].view(np.int8).reshape(2048, 8),
            "fp4": t["kvalues_fp4"],                                         # (16,) int8, e2m1 doubled
            "iq1s_delta": _ggml_float_define(text, "IQ1S_DELTA"),
        }
    return _GRIDS


def _blocks(raw: np.ndarray, nbytes: int) -> np.ndarray:
    raw = _byte_array(raw)
    if raw.size % nbytes:
        raise ValueError(f"{raw.size} bytes is not a whole number of {nbytes}-byte blocks")
    return raw.reshape(-1, nbytes)


def _le(b: np.ndarray, dtype: str) -> np.ndarray:
    """Little-endian fields from the trailing byte axis of `b` (a copy, so any slice works)."""
    return np.ascontiguousarray(b).view(dtype)


def _f16(b: np.ndarray) -> np.ndarray:
    """ggml_half fields (the trailing 2-byte axis) as float32; f16 -> f32 is exact."""
    return _le(b, "<f2")[..., 0].astype(F32)


def _dq_mxfp4(raw: np.ndarray) -> np.ndarray:
    """block_mxfp4 {uint8 e (E8M0); uint8 qs[16]}: y[j] = kvalues_mxfp4[lo nibble] * d, y[j+16] = the
    high nibble's; d = ggml_e8m0_to_fp32_half(e) = 2^(e-128), with e = 0, 1 the two denormals."""
    G = _grids()
    b = _blocks(raw, 17)
    e = b[:, 0].astype(np.uint32)
    bits = np.where(e < 2, np.uint32(0x00200000) << e, (np.maximum(e, 1) - 1) << np.uint32(23))
    d = bits.astype(np.uint32).view(F32)
    qs = b[:, 1:]
    v = np.concatenate([G["fp4"][qs & 15], G["fp4"][qs >> 4]], axis=1).astype(F32)   # (n, 32)
    with np.errstate(over="ignore"):                    # e = 255 overflows to inf, in ggml too
        return (v * d[:, None]).ravel()


def _dq_iq2_xxs(raw: np.ndarray) -> np.ndarray:
    """block_iq2_xxs {half d; uint16 qs[32]}; per 32 values 8 bytes: 4 grid indices, then a uint32 with
    four 7-bit sign indices and the 4-bit scale on top: db = d * (0.5 + s) * 0.25."""
    G = _grids()
    b = _blocks(raw, 66)
    n = b.shape[0]
    d = _f16(b[:, :2])                                                   # (n,)
    q = b[:, 2:].reshape(n, 8, 8)
    aux = _le(q[:, :, 4:], "<u4")[..., 0]                                # (n, 8)
    db = d[:, None] * (F32(0.5) + (aux >> 28).astype(F32)) * F32(0.25)
    sidx = (aux[:, :, None] >> (np.uint32(7) * np.arange(4, dtype=np.uint32))) & 127
    sign = G["sign"][G["ksigns"][sidx]]                                  # (n, 8, 4, 8)
    grid = G["iq2xxs"][q[:, :, :4]].astype(F32)                          # (n, 8, 4, 8)
    return (db[:, :, None, None] * grid * sign).ravel()


def _dq_iq2_xs(raw: np.ndarray) -> np.ndarray:
    """block_iq2_xs {half d; uint16 qs[32]; uint8 scales[8]}: qs = 9-bit grid index | 7-bit sign index;
    the two nibbles of scales[ib] scale the 32-value group's halves."""
    G = _grids()
    b = _blocks(raw, 74)
    n = b.shape[0]
    d = _f16(b[:, :2])
    qs = _le(b[:, 2:66], "<u2").reshape(n, 8, 4)
    sc = b[:, 66:74]
    lo = d[:, None] * (F32(0.5) + (sc & 15).astype(F32)) * F32(0.25)
    hi = d[:, None] * (F32(0.5) + (sc >> 4).astype(F32)) * F32(0.25)
    dl = np.stack([lo, lo, hi, hi], axis=-1)                             # (n, 8, 4): db[l / 2]
    grid = G["iq2xs"][qs & 511].astype(F32)
    sign = G["sign"][G["ksigns"][qs >> 9]]
    return (dl[..., None] * grid * sign).ravel()


def _dq_iq2_s(raw: np.ndarray) -> np.ndarray:
    """block_iq2_s {half d; uint8 qs[64]; uint8 qh[8]; uint8 scales[8]}: qs[0:32] are the grid indices'
    low bytes, qs[32:64] the sign bytes, qh[ib] the indices' bits 8-9."""
    G = _grids()
    b = _blocks(raw, 82)
    n = b.shape[0]
    d = _f16(b[:, :2])
    qs = b[:, 2:34].reshape(n, 8, 4).astype(np.int32)
    signs = b[:, 34:66].reshape(n, 8, 4)
    qh = b[:, 66:74].astype(np.int32)
    sc = b[:, 74:82]
    idx = qs | ((qh[:, :, None] << np.array([8, 6, 4, 2], dtype=np.int32)) & 0x300)
    lo = d[:, None] * (F32(0.5) + (sc & 15).astype(F32)) * F32(0.25)
    hi = d[:, None] * (F32(0.5) + (sc >> 4).astype(F32)) * F32(0.25)
    dl = np.stack([lo, lo, hi, hi], axis=-1)
    return (dl[..., None] * G["iq2s"][idx].astype(F32) * G["sign"][signs]).ravel()


def _dq_iq3_xxs(raw: np.ndarray) -> np.ndarray:
    """block_iq3_xxs {half d; uint8 qs[96]}: qs[0:64] grid indices (two 4-value rows per 8 values),
    qs[64:96] one uint32 per 32 values with four 7-bit sign indices and the scale: d * (0.5 + s) * 0.5."""
    G = _grids()
    b = _blocks(raw, 98)
    n = b.shape[0]
    d = _f16(b[:, :2])
    grid = G["iq3xxs"][b[:, 2:66].reshape(n, 8, 4, 2)].reshape(n, 8, 4, 8).astype(F32)
    aux = _le(b[:, 66:98], "<u4").reshape(n, 8)
    db = d[:, None] * (F32(0.5) + (aux >> 28).astype(F32)) * F32(0.5)
    sidx = (aux[:, :, None] >> (np.uint32(7) * np.arange(4, dtype=np.uint32))) & 127
    sign = G["sign"][G["ksigns"][sidx]]
    return (db[:, :, None, None] * grid * sign).ravel()


def _dq_iq3_s(raw: np.ndarray) -> np.ndarray:
    """block_iq3_s {half d; uint8 qs[64]; uint8 qh[8]; uint8 signs[32]; uint8 scales[4]}: group k of 32
    values reads qs[8k..8k+8), qh[k] (index bit 8), signs[4k..4k+4) and the nibble k%2 of scales[k/2]:
    db = d * (1 + 2 s)."""
    G = _grids()
    b = _blocks(raw, 110)
    n = b.shape[0]
    d = _f16(b[:, :2])
    qs = b[:, 2:66].reshape(n, 8, 4, 2).astype(np.int32)
    qh = b[:, 66:74].astype(np.int32)[:, :, None]
    signs = b[:, 74:106].reshape(n, 8, 4)
    sc = b[:, 106:110]
    nib = np.stack([sc & 15, sc >> 4], axis=-1).reshape(n, 8).astype(np.int32)
    db = d[:, None] * (1 + 2 * nib).astype(F32)
    l2 = 2 * np.arange(4, dtype=np.int32)
    idx = np.stack([qs[..., 0] | ((qh << (8 - l2)) & 256), qs[..., 1] | ((qh << (7 - l2)) & 256)], axis=-1)
    grid = G["iq3s"][idx].reshape(n, 8, 4, 8).astype(F32)
    return (db[:, :, None, None] * grid * G["sign"][signs]).ravel()


def _dq_iq1_s(raw: np.ndarray) -> np.ndarray:
    """block_iq1_s {half d; uint8 qs[32]; uint16 qh[8]}: per 32 values qh = 3 index bits per 8 values,
    a 3-bit scale (bits 12-14) and the delta's sign (bit 15): y = d * (2 s + 1) * (grid + delta)."""
    G = _grids()
    b = _blocks(raw, 50)
    n = b.shape[0]
    d = _f16(b[:, :2])
    qs = b[:, 2:34].reshape(n, 8, 4).astype(np.int32)
    qh = _le(b[:, 34:50], "<u2").reshape(n, 8).astype(np.int32)
    dl = d[:, None] * (2 * ((qh >> 12) & 7) + 1).astype(F32)
    delta = np.where((qh & 0x8000) != 0, -G["iq1s_delta"], G["iq1s_delta"]).astype(F32)
    idx = qs | (((qh[:, :, None] >> (3 * np.arange(4, dtype=np.int32))) & 7) << 8)
    grid = G["iq1s"][idx].astype(F32)                                    # (n, 8, 4, 8)
    return (dl[:, :, None, None] * (grid + delta[:, :, None, None])).ravel()


def _dq_iq1_m(raw: np.ndarray) -> np.ndarray:
    """block_iq1_m {uint8 qs[32]; uint8 qh[16]; uint8 scales[8]}: the f16 scale is the top nibbles of the
    four uint16 scale words; each word's 3-bit fields scale 16 values; qh holds 3 index bits and the
    delta's sign per 8 values.  (ggml's dequant uses IQ1S_DELTA here too.)"""
    G = _grids()
    b = _blocks(raw, 56)
    n = b.shape[0]
    qs = b[:, :32].reshape(n, 8, 4).astype(np.int32)
    qh = b[:, 32:48].reshape(n, 8, 2).astype(np.int32)
    sc = _le(b[:, 48:56], "<u2").reshape(n, 4).astype(np.int32)
    u16 = (sc[:, 0] >> 12) | ((sc[:, 1] >> 8) & 0x00F0) | ((sc[:, 2] >> 4) & 0x0F00) | (sc[:, 3] & 0xF000)
    d = u16.astype(np.uint16).view(np.float16).astype(F32)
    s = sc[:, np.arange(8) // 2]                                         # (n, 8): sc[ib / 2]
    sh = 6 * (np.arange(8, dtype=np.int32) % 2)
    dl1 = d[:, None] * (2 * ((s >> sh) & 7) + 1).astype(F32)
    dl2 = d[:, None] * (2 * ((s >> (sh + 3)) & 7) + 1).astype(F32)
    dl = np.stack([dl1, dl1, dl2, dl2], axis=-1)                         # (n, 8, 4)
    h0, h1 = qh[..., 0], qh[..., 1]
    idx = np.stack([qs[..., 0] | ((h0 << 8) & 0x700), qs[..., 1] | ((h0 << 4) & 0x700),
                    qs[..., 2] | ((h1 << 8) & 0x700), qs[..., 3] | ((h1 << 4) & 0x700)], axis=-1)
    neg = np.stack([h0 & 0x08, h0 & 0x80, h1 & 0x08, h1 & 0x80], axis=-1) != 0
    delta = np.where(neg, -G["iq1s_delta"], G["iq1s_delta"]).astype(F32)
    grid = G["iq1s"][idx].astype(F32)
    return (dl[..., None] * (grid + delta[..., None])).ravel()


_DECODERS = {"MXFP4": _dq_mxfp4, "IQ1_S": _dq_iq1_s, "IQ1_M": _dq_iq1_m, "IQ2_XXS": _dq_iq2_xxs,
             "IQ2_XS": _dq_iq2_xs, "IQ2_S": _dq_iq2_s, "IQ3_XXS": _dq_iq3_xxs, "IQ3_S": _dq_iq3_s}


def _bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def _discover_shards(path) -> list[pathlib.Path]:
    """Resolve a GGUF argument to its shard list.

    The real artifact is TWO files and the split is not incidental: shard 2 holds exactly one tensor,
    `per_layer_token_embd.weight` (26.82 GiB). A reader that opens one file can therefore load the
    whole model EXCEPT the PLE table and will not fail until a forward pass reaches layer 1 - so
    shard discovery is not a convenience, it is a correctness requirement for the real model.

    Accepts a directory (all `*.gguf` in it), any single shard (siblings found by the
    `-NNNNN-of-MMMMM` suffix), or an explicit list.
    """
    if isinstance(path, (list, tuple)):
        return [pathlib.Path(p) for p in path]
    p = pathlib.Path(path)
    if p.is_dir():
        # A directory is ambiguous in practice: the project root holds BOTH the model shards (in a
        # subdirectory) and an unrelated `mmproj-*.gguf`.  Globbing `*.gguf` there silently loaded
        # the multimodal projector instead of the model.  So prefer an explicit shard set, accept a
        # single unambiguous file, and REFUSE to guess otherwise.
        shards = sorted(p.glob("*-[0-9][0-9][0-9][0-9][0-9]-of-[0-9][0-9][0-9][0-9][0-9].gguf"))
        if shards:
            totals = {re.match(r"^.*-(\d{5})-of-(\d{5})\.gguf$", s.name).group(2) for s in shards}
            if len(totals) != 1:
                raise ValueError(f"{p}: shard files from more than one split: {sorted(totals)}")
            return shards
        found = sorted(p.glob("*.gguf"))
        if len(found) == 1:
            return found
        if not found:
            raise FileNotFoundError(f"no .gguf files in {p}")
        raise ValueError(
            f"{p}: {len(found)} .gguf files and none is a shard set, so which is the model is "
            f"ambiguous: {[f.name for f in found]}. Pass one file explicitly.")
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", p.name)
    if m:
        stem, _, total = m.groups()
        sibs = sorted(p.parent.glob(f"{stem}-*-of-{total}.gguf"))
        if len(sibs) != int(total):
            raise FileNotFoundError(
                f"{p.name}: expected {total} shards, found {len(sibs)} in {p.parent}")
        return sibs
    return [p]


class Gguf:
    """Read-only, demand-paged view of a (possibly multi-shard) GGUF.

    Nothing is dequantized until asked for.  `g` is the FIRST shard's `GGUFFile`, so metadata and the
    tensor directory are reachable as before; reads route to whichever shard owns the tensor.
    """

    def __init__(self, path):
        self.paths = _discover_shards(path)
        self.path = self.paths[0]
        self.shards: list[tuple[GGUFFile, object]] = []
        self._by_name: dict[str, tuple[int, object]] = {}
        for i, sp in enumerate(self.paths):
            gf = GGUFFile(sp)
            self.shards.append((gf, sp.open("rb")))
            for t in gf.tensors:
                if t.name in self._by_name:
                    raise ValueError(f"{sp.name}: duplicate tensor {t.name} across shards")
                self._by_name[t.name] = (i, t)
        self.g = self.shards[0][0]           # metadata / directory / alignment live in shard 1

    def close(self) -> None:
        """Close the shard handles.  Windows refuses to unlink an open file, so callers that
        delete the GGUF while still alive (the oracle test's temp dir) must close first."""
        for _, fh in self.shards:
            fh.close()

    # ---- raw bytes
    def _raw(self, name: str, nbytes: int, byte_offset: int = 0) -> bytes:
        i, t = self._by_name[name]
        gf, fh = self.shards[i]
        fh.seek(gf.data_start + t.offset + byte_offset)
        return fh.read(nbytes)

    def info(self, name: str):
        return self._by_name[name][1]

    def row_bytes(self, t) -> int:
        el, by = GEOM[t.type_name]
        return t.shape[0] // el * by

    def blocks(self, name: str, start: int, count: int) -> np.ndarray:
        """`count` raw blocks of a tensor from block `start` on, as a (count, bytes per block) uint8
        array: the slice a dequant cross-check reads, without reading the rest of the tensor."""
        t = self.info(name)
        el, by = GEOM[t.type_name]
        n_blocks = t.elements // el
        if start < 0 or count < 0 or start + count > n_blocks:
            raise ValueError(f"{name}: blocks {start}..{start + count} outside 0..{n_blocks}")
        raw = np.frombuffer(self._raw(name, count * by, start * by), dtype=np.uint8)
        if raw.size != count * by:
            raise ValueError(f"{name}: the file ends inside blocks {start}..{start + count}")
        return raw.reshape(count, by)

    # ---- full tensors
    def tensor(self, name: str) -> np.ndarray:
        """Dequantize a whole tensor to FP32, shape as GGUF order reversed to (out, in[, experts])."""
        t = self.info(name)
        el, by = GEOM[t.type_name]
        if t.elements % el:
            raise ValueError(f"{name}: {t.elements} not a multiple of block {el}")
        raw = np.frombuffer(self._raw(name, t.elements // el * by), dtype=np.uint8)
        flat = _dequant_flat(t.type_name, raw)
        # GGUF dim 0 varies fastest; numpy's last axis does, so reverse and transpose to (out, in).
        return flat.reshape(tuple(reversed(t.shape)))

    # ---- the lazy row path
    def rows(self, name: str, idx) -> np.ndarray:
        """Gather specific ROWS of a 2-D tensor without decoding the rest of it.

        This exists for `per_layer_token_embd.weight`, which is `[160, 320001536]` IQ4_NL = **26.82
        GiB** - 43% of the artifact, the entire second shard - and is read 16 rows per token by one
        layer (see round 98's finding). Calling `tensor()` on it would dequantize 320 million rows to
        FP32, about 205 GB, to use 2.5 KB of them.

        A row is contiguous in GGUF (ne0 varies fastest), so one row is `row_bytes` at
        `row * row_bytes`. The type's block size must divide the row width; for this table it does
        (160 = 5 x 32 for IQ4_NL).
        """
        t = self.info(name)
        if len(t.shape) != 2:
            raise ValueError(f"{name}: rows() needs a 2-D tensor, got shape {t.shape}")
        el, by = GEOM[t.type_name]
        width = t.shape[0]
        if width % el:
            raise ValueError(f"{name}: row width {width} is not a multiple of block {el}")
        rb = self.row_bytes(t)
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= t.shape[1]):
            raise ValueError(f"{name}: row index out of range (0..{t.shape[1] - 1})")
        out = np.empty((idx.size, width), dtype=np.float32)
        for k, r in enumerate(idx):
            raw = np.frombuffer(self._raw(name, rb, int(r) * rb), dtype=np.uint8)
            out[k] = _dequant_flat(t.type_name, raw)
        return out

    def row_range(self, name: str, r0: int, r1: int) -> np.ndarray:
        """Rows r0..r1-1 of a 2-D tensor as one contiguous read, (r1 - r0, width) FP32: rows() for a
        slab, e.g. an output head taken a few thousand vocabulary rows at a time."""
        t = self.info(name)
        if len(t.shape) != 2:
            raise ValueError(f"{name}: row_range() needs a 2-D tensor, got shape {t.shape}")
        el, _ = GEOM[t.type_name]
        width = t.shape[0]
        if width % el:
            raise ValueError(f"{name}: row width {width} is not a multiple of block {el}")
        if not 0 <= r0 <= r1 <= t.shape[1]:
            raise ValueError(f"{name}: rows {r0}..{r1} out of range (0..{t.shape[1]})")
        rb = self.row_bytes(t)
        raw = np.frombuffer(self._raw(name, (r1 - r0) * rb, r0 * rb), dtype=np.uint8)
        return _dequant_flat(t.type_name, raw).reshape(r1 - r0, width)

    # ---- the lazy path
    def expert(self, name: str, expert_id: int) -> np.ndarray:
        """One expert out of a fused `[in, out, experts]` tensor, WITHOUT touching the others.

        In GGUF order the expert index is the LAST axis, so expert e's matrix occupies a contiguous
        run of `in*out` elements at offset e*in*out. Only that run is read and decoded - which is the
        plan's "do not materialize 120B parameters" requirement, and the reason the engine can run at
        all on a machine where the expert set is 34 GB.
        """
        t = self.info(name)
        if len(t.shape) != 3:
            raise ValueError(f"{name}: not a fused expert tensor (shape {t.shape})")
        in_f, out_f, n_exp = t.shape
        if not 0 <= expert_id < n_exp:
            raise ValueError(f"{name}: expert {expert_id} out of range (0..{n_exp - 1})")
        el, by = GEOM[t.type_name]
        per_expert = in_f * out_f
        if per_expert % el:
            raise ValueError(f"{name}: per-expert element count {per_expert} not a multiple of {el}")
        # byte offset of expert e: whole experts before it, in blocks
        start = expert_id * (per_expert // el) * by
        raw = np.frombuffer(self._raw(name, per_expert // el * by, start), dtype=np.uint8)
        flat = _dequant_flat(t.type_name, raw)
        return flat.reshape((out_f, in_f))

    def close(self) -> None:
        for _, fh in self.shards:
            fh.close()

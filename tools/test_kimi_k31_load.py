"""tools/test_kimi_k31_load.py - ref/load.py's MXFP4 and i-quant decoders (Kimi K3.1's routed-expert
encodings: MXFP4, IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S), bit for bit.

Always (numpy only: no ggml, no model files):
  * the codebooks parse from third_party/ggml/ggml-common.h with their declared sizes and value sets,
    and the parser refuses a short table and a table defined twice;
  * GEOM agrees with tools/gguf_reader.py's BLOCK_GEOMETRY;
  * hand-built blocks decode to the values the formats define (an MXFP4 scale of 2^0, the IQ1_S
    delta's sign);
  * deterministic synthetic blocks (finite scales, with +0, -0, a subnormal and the largest half among
    them; every MXFP4 exponent) decode to float32 whose SHA-256 equals the one recorded from ggml's C
    `to_float` (dequantize_row_*), so a machine with numpy alone checks the decoders against the C;
  * the same blocks decode bit-equal under the gguf-py that `import gguf` finds (skipped without one);
  * tensor(), expert(), rows() and blocks() address these types by whole blocks: on a small GGUF written
    here, one expert, chosen rows and a block run equal the matching slice of the whole tensor.

With a llama.cpp checkout and real K3.1 GGUFs:
  KIMI_K31_PORT_DIR  the checkout: its gguf-py/ and a built libggml-base (build*/bin/libggml-base.so,
                     or KIMI_K31_GGML_LIB);
  KIMI_K31_REAL_GGUF real GGUFs, os.pathsep-separated (a file still being written is read only below
                     its current size); KIMI_K31_REAL_BLOCKS blocks per type and file (default 1024);
  * ggml's type traits give each type's name, block size and bytes as GEOM and gguf_reader do;
  * the synthetic blocks decode bit-equal under the checkout's C `to_float` and its gguf-py;
  * >= 1,000 real blocks per type, read in a few contiguous runs from a few tensors, decode bit-equal
    three ways (this module, the checkout's gguf-py, its C `to_float`). The types the files hold are read
    as written; every i-quant is also made from real weight rows by the checkout's ggml_quantize_chunk
    (the function llama-quantize calls), so the types no file holds are covered by real-weight blocks;
  * expert() on a real expert tensor equals the C decode of the same bytes.

--dump DIR writes the random-block references tests/k31_fmt_test.cpp reads (plan T2): for each type, the
synthetic blocks above (--blocks N of them, default N_SYNTH) and, with KIMI_K31_PORT_DIR and KIMI_K31_REAL_GGUF,
the real and requantized blocks the real tier reads, each set as
    <stem>.blocks   the raw blocks, as a GGUF stores them (n x block bytes)
    <stem>.f32      gguf-py's decode of them, float32 little-endian (n x block values)
and one line per set in DIR/manifest.txt:
    <stem> <type> <ggml type id> <block values> <block bytes> <n blocks> <sha256 .blocks> <sha256 .f32>
('#' lines name the gguf-py, the ggml library and each set's source).  A set is written only when this module,
gguf-py and (with the checkout) ggml's C decode it bit for bit alike.  The files are ~8 MB for 1,024 blocks a type,
so they are made, not committed.

    python -m unittest tools.test_kimi_k31_load
    KIMI_K31_PORT_DIR=... KIMI_K31_REAL_GGUF=a.gguf:b.gguf python -m unittest -v tools.test_kimi_k31_load
    KIMI_K31_PORT_DIR=... python tools/test_kimi_k31_load.py --record      (prints the C reference hashes)
    [KIMI_K31_PORT_DIR=... KIMI_K31_REAL_GGUF=...] python tools/test_kimi_k31_load.py --dump DIR [--blocks N]
"""
from __future__ import annotations

import argparse
import contextlib
import ctypes
import glob
import hashlib
import importlib
import importlib.machinery
import importlib.util
import os
import io
import pathlib
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "ref"))

import numpy as np  # noqa: E402

import gguf_reader  # noqa: E402
import gguf_writer  # noqa: E402
import load as L  # noqa: E402  (ref/load.py)

TYPES = ("MXFP4", "IQ1_S", "IQ1_M", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S")
IQ_TYPES = TYPES[1:]
TYPE_ID = {name: tid for tid, name in gguf_reader.GGML_TYPES.items() if name in TYPES}
N_SYNTH = 1024                     # synthetic blocks per type
N_REAL = int(os.environ.get("KIMI_K31_REAL_BLOCKS", "1024"))   # real blocks per type and source
N_RUNS = 8                         # ... read as this many contiguous runs, one tensor each

# SHA-256 of the float32 (little-endian) that ggml's C to_float makes of _synthetic(type, N_SYNTH),
# recorded with `--record` from llama.cpp's libggml-base (ggml 0.29.0).
C_SHA256 = {
    "MXFP4": "407802b2911b217c7ca755c10089fa27bf63a7288671004160fa497519f40842",
    "IQ1_S": "40b795215c0b24556f0a51d52024c71394cace36711b992b4ad811d187613243",
    "IQ1_M": "c09d1bb0f0307b3213ade099d68bdd80a56614c586b8a8237ec96e01032c1443",
    "IQ2_XXS": "6ca70b8b172f293bfe145c0815c8c477860b00159082421a6eb5f86b6ea163de",
    "IQ2_XS": "d0722cea77772b4b3c9f9409f1f8846972367b78dc971587dbe42b56d50653f7",
    "IQ2_S": "9f44f196b0648ecdee08d2626c4be50b5745ae32200af496ab7aad2fc6b3cb06",
    "IQ3_XXS": "eeef83ed760cc376794838dc1910a37feb88243e9553dfce5f827fc2001d79fe",
    "IQ3_S": "ff8f3a34d8b03a465b5c7c90c8c29cb936c08068d44a59c1efa30d06cdebc512",
}


# ---------------------------------------------------------------- synthetic blocks
def _stream(tag: str, n: int) -> np.ndarray:
    """n deterministic pseudo-random bytes (SHA-256 in counter mode: the same on every numpy)."""
    out = bytearray()
    i = 0
    while len(out) < n:
        out += hashlib.sha256(f"kimi-k31-load:{tag}:{i}".encode()).digest()
        i += 1
    return np.frombuffer(bytes(out[:n]), dtype=np.uint8).copy()


def _synthetic(type_name: str, n: int = N_SYNTH) -> np.ndarray:
    """n blocks of `type_name`, every field random except the scale: a half scale with exponent 31
    (inf/nan) has the exponent's top bit cleared, and blocks 0-3 get +0, -0, the smallest subnormal
    and the largest finite half; MXFP4's first 256 blocks take every E8M0 exponent 0..255."""
    _, by = L.GEOM[type_name]
    b = _stream(type_name, n * by).reshape(n, by)
    special = np.array([0x0000, 0x8000, 0x0001, 0x7BFF], dtype=np.uint16)
    if type_name == "MXFP4":
        b[:min(n, 256), 0] = np.arange(min(n, 256), dtype=np.uint8)
    elif type_name == "IQ1_M":
        # the half is the top nibbles of the four uint16 scale words (bytes 48..55): word k gives bits 4k..4k+3
        sc = np.ascontiguousarray(b[:, 48:56]).view("<u2").copy()             # (n, 4)
        d = (sc[:, 0] >> 12) | ((sc[:, 1] >> 8) & 0xF0) | ((sc[:, 2] >> 4) & 0xF00) | (sc[:, 3] & 0xF000)
        d = np.where((d & 0x7C00) == 0x7C00, d & 0xBFFF, d).astype(np.uint16)
        d[:4] = special
        for k in range(4):
            sc[:, k] = (sc[:, k] & 0x0FFF) | (((d >> (4 * k)) & 0xF) << 12)
        b[:, 48:56] = sc.astype("<u2").view(np.uint8).reshape(n, 8)
    else:                                                                       # ggml_half d first
        d = np.ascontiguousarray(b[:, :2]).view("<u2")[:, 0].copy()
        d = np.where((d & 0x7C00) == 0x7C00, d & 0xBFFF, d).astype(np.uint16)
        d[:4] = special
        b[:, :2] = d.astype("<u2").view(np.uint8).reshape(n, 2)
    return b


def _bits(x: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)


def _sha(x: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(x, dtype="<f4").tobytes()).hexdigest()


def _first_diff(a: np.ndarray, b: np.ndarray, el: int) -> str:
    bad = np.flatnonzero(_bits(a) != _bits(b))
    if not bad.size:
        return "equal"
    i = int(bad[0])
    return (f"{bad.size} of {a.size} values differ; first at value {i} (block {i // el}): "
            f"{a[i]!r} ({_bits(a)[i]:#010x}) vs {b[i]!r} ({_bits(b)[i]:#010x})")


# ---------------------------------------------------------------- the llama.cpp checkout
class Ggml:
    """ggml's C to_float and ggml_quantize_chunk, through ctypes on a built libggml-base."""

    class _Traits(ctypes.Structure):
        _fields_ = [("type_name", ctypes.c_char_p), ("blck_size", ctypes.c_int64),
                    ("blck_size_interleave", ctypes.c_int64), ("type_size", ctypes.c_size_t),
                    ("is_quantized", ctypes.c_bool), ("to_float", ctypes.c_void_p),
                    ("from_float_ref", ctypes.c_void_p)]

    _TO_FLOAT = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64)

    def __init__(self, lib: str):
        self.path = lib
        self.lib = ctypes.CDLL(lib)
        self.lib.ggml_get_type_traits.restype = ctypes.POINTER(self._Traits)
        self.lib.ggml_get_type_traits.argtypes = [ctypes.c_int]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t
        self.lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
                                                 ctypes.c_int64, ctypes.c_int64, ctypes.c_int64,
                                                 ctypes.c_void_p]

    def traits(self, type_name: str):
        return self.lib.ggml_get_type_traits(TYPE_ID[type_name]).contents

    def to_float(self, type_name: str, raw: np.ndarray) -> np.ndarray:
        el, by = L.GEOM[type_name]
        raw = L._byte_array(raw)
        assert raw.size % by == 0
        out = np.empty(raw.size // by * el, dtype=np.float32)
        fn = self._TO_FLOAT(self.traits(type_name).to_float)
        fn(raw.ctypes.data, out.ctypes.data, out.size)
        return out

    def quantize(self, type_name: str, rows: np.ndarray) -> np.ndarray:
        """(nrows, n_per_row) float32 -> the blocks ggml_quantize_chunk writes (imatrix of ones)."""
        el, by = L.GEOM[type_name]
        rows = np.ascontiguousarray(rows, dtype=np.float32)
        nrows, n_per_row = rows.shape
        dst = np.zeros(nrows * n_per_row // el * by, dtype=np.uint8)
        imatrix = np.ones(n_per_row, dtype=np.float32)
        got = self.lib.ggml_quantize_chunk(TYPE_ID[type_name], rows.ctypes.data, dst.ctypes.data, 0,
                                           nrows, n_per_row, imatrix.ctypes.data)
        assert got == dst.size, (type_name, got, dst.size)
        return dst


def _port_dir() -> pathlib.Path | None:
    p = os.environ.get("KIMI_K31_PORT_DIR")
    return pathlib.Path(p).expanduser() if p else None


def _ggml_lib(port: pathlib.Path) -> str | None:
    if os.environ.get("KIMI_K31_GGML_LIB"):
        return os.environ["KIMI_K31_GGML_LIB"]
    cands = [str(port / "build" / "bin" / "libggml-base.so"),
             *sorted(glob.glob(str(port / "build*" / "bin" / "libggml-base.so")))]
    return next((c for c in cands if os.path.isfile(c)), None)


def _port_quants(port: pathlib.Path):
    """The checkout's gguf-py `quants`, loaded under its own package name so that it cannot be confused
    with (or replace) whichever `gguf` the process imports."""
    name = "_kimi_k31_port_gguf"
    if name not in sys.modules:
        spec = importlib.machinery.ModuleSpec(name, None, is_package=True)
        spec.submodule_search_locations = [str(port / "gguf-py" / "gguf")]
        sys.modules[name] = importlib.util.module_from_spec(spec)
    return importlib.import_module(name + ".quants")


def _default_quants():
    """The gguf-py `import gguf` finds after tools/_paths (CI: llama.cpp's pinned gguf-py), or None."""
    try:
        import _paths
        _paths.add_gguf_py()
    except SystemExit:
        pass
    try:
        from gguf import quants
    except ImportError:
        return None
    return quants


def _real_paths() -> list[pathlib.Path]:
    v = os.environ.get("KIMI_K31_REAL_GGUF", "")
    return [pathlib.Path(p).expanduser() for p in v.split(os.pathsep) if p]


def _open_real() -> list:
    """(path, Gguf, readable limit) for each KIMI_K31_REAL_GGUF file.  A GGUF being written holds final bytes
    only below its current size (the writer is sequential), so a tensor is used only if it ends below it."""
    return [(p, L.Gguf(p), p.stat().st_size) for p in _real_paths()]


def _complete(g, t, limit) -> bool:
    return g.g.data_start + t.offset + t.expected_bytes() <= limit


def _runs(g, type_name: str, limit: int, total: int, rng):
    """Up to N_RUNS (tensor, first block, count) runs of `type_name` adding up to `total` blocks."""
    cands = [t for t in g.g.tensors if t.type_name == type_name and _complete(g, t, limit)]
    if not cands:
        return []
    pick = rng.choice(len(cands), size=min(N_RUNS, len(cands)), replace=False)
    per = -(-total // len(pick))
    out = []
    for i in pick:
        t = cands[int(i)]
        nb = t.elements // L.GEOM[type_name][0]
        run = min(per, nb)
        out.append((t.name, int(rng.integers(0, nb - run + 1)), run))
    return out


def _source_rows(files, total_values: int, rng):
    """Real weight rows (row length a multiple of 256), decoded by this module: MXFP4 expert rows if a file
    has them (the released experts), else BF16 or Q8_0 matrices.  -> (source type, [(where, rows)])."""
    for src in ("MXFP4", "BF16", "Q8_0"):
        for p, g, limit in files:
            cands = [t for t in g.g.tensors if t.type_name == src and len(t.shape) >= 2
                     and t.shape[0] % 256 == 0 and _complete(g, t, limit)]
            if not cands:
                continue
            pick = rng.choice(len(cands), size=min(N_RUNS, len(cands)), replace=False)
            groups = []
            for i in pick:
                t = cands[int(i)]
                width = t.shape[0]
                nrows = t.elements // width
                want = min(nrows, -(-total_values // (len(pick) * width)))
                r0 = int(rng.integers(0, nrows - want + 1))
                el = L.GEOM[src][0]
                raw = g.blocks(t.name, r0 * width // el, want * width // el)
                groups.append((f"{p.name}:{t.name} rows {r0}+{want}",
                               L._dequant_flat(src, raw).reshape(want, width)))
            return src, groups
    return None, []


# ---------------------------------------------------------------- always
class TablesTest(unittest.TestCase):
    def test_codebooks(self):
        G = L._grids()
        self.assertEqual(G["iq1s"].shape, (2048, 8))
        self.assertEqual(sorted(np.unique(G["iq1s"]).tolist()), [-1, 0, 1])
        for k, rows in (("iq2xxs", 256), ("iq2xs", 512), ("iq2s", 1024)):
            self.assertEqual(G[k].shape, (rows, 8))
            self.assertEqual(sorted(np.unique(G[k]).tolist()), [8, 25, 43], k)
        self.assertEqual(G["iq3xxs"].shape, (256, 4))
        self.assertEqual(sorted(np.unique(G["iq3xxs"]).tolist()), [4, 12, 20, 28, 36, 44, 52, 62])
        self.assertEqual(G["iq3s"].shape, (512, 4))
        self.assertEqual(sorted(np.unique(G["iq3s"]).tolist()), list(range(1, 16, 2)))
        self.assertEqual(G["fp4"].tolist(), [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12])
        self.assertEqual(float(G["iq1s_delta"]), 0.125)
        # ksigns: the 7 stored bits plus a parity bit 7, so every sign byte has an even number of -1s
        ks = G["ksigns"].astype(np.int64)
        self.assertTrue(np.array_equal(ks & 127, np.arange(128)))
        self.assertTrue(all(bin(int(v)).count("1") % 2 == 0 for v in ks))
        # grid rows are distinct (a wrong byte order or stride would repeat or scramble them)
        for k in ("iq2xxs", "iq2xs", "iq2s", "iq3xxs", "iq3s", "iq1s"):
            self.assertEqual(len({r.tobytes() for r in G[k]}), G[k].shape[0], k)
        # iq1s_grid's first entries, by hand from the header (0xffffffffffffffff, 0xffffffffffffff01)
        self.assertEqual(G["iq1s"][0].tolist(), [-1] * 8)
        self.assertEqual(G["iq1s"][1].tolist(), [1] + [-1] * 7)

    def test_parser_refuses(self):
        good = "#define N 3\nGGML_TABLE_BEGIN(uint8_t, t, N)\n    1, 2, 3,\nGGML_TABLE_END()\n"
        self.assertEqual(L.parse_ggml_tables(good)["t"].tolist(), [1, 2, 3])
        with self.assertRaisesRegex(ValueError, "has 2 entries, declared 3"):
            L.parse_ggml_tables(good.replace("1, 2, 3", "1, 2"))
        with self.assertRaisesRegex(ValueError, "defined twice"):
            L.parse_ggml_tables(good + good.replace("#define N 3\n", ""))

    def test_geometry(self):
        for t in TYPES:
            self.assertEqual(L.GEOM[t], gguf_reader.BLOCK_GEOMETRY[t], t)
            self.assertIn(t, L._DECODERS)
            self.assertEqual(L._GGUF_PY_NAME[t], t)

    def test_decoders_refuse_partial_blocks(self):
        for t in TYPES:
            with self.assertRaises(ValueError, msg=t):
                L._dequant_flat(t, np.zeros(L.GEOM[t][1] + 1, dtype=np.uint8))

    def test_refuse_non_byte_buffers(self):
        # a buffer of any other dtype is refused, not converted by value (which would decode other bytes)
        for t in (*TYPES, "Q8_0", "BF16", "F32", "Q2_0"):
            _, by = L.GEOM[t]
            for bad in (np.full(by, 300.7, dtype=np.float32), np.zeros(by, dtype=np.uint16),
                        np.zeros(by, dtype=np.int8), bytes(by)):
                with self.assertRaises(TypeError, msg=f"{t} {type(bad).__name__}"):
                    L._dequant_flat(t, bad)
        with self.assertRaises(TypeError):
            L.gguf_py_dequant("Q8_0", np.zeros(34, dtype=np.uint16), quants=None)


class HandBuiltTest(unittest.TestCase):
    def test_mxfp4_values(self):
        # e = 127: scale 2^-1 on the doubled e2m1 values, i.e. the e2m1 values themselves
        blk = np.zeros(17, dtype=np.uint8)
        blk[0] = 127
        blk[1:] = np.arange(16, dtype=np.uint8) | (np.arange(16, dtype=np.uint8)[::-1] << 4)
        e2m1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]
        y = L._dequant_flat("MXFP4", blk)
        self.assertEqual(y[:16].tolist(), e2m1)
        self.assertEqual(y[16:].tolist(), e2m1[::-1])
        blk[0] = 0                                          # 2^-128: a denormal, not zero
        self.assertEqual(L._dequant_flat("MXFP4", blk)[7], np.float32(6 * 2.0 ** -127))

    def test_iq1_s_values(self):
        # grid 0 (all -1), scale field 0 (factor 1), delta +0.125, then delta -0.125 with factor 3
        blk = np.zeros(50, dtype=np.uint8)
        blk[:2] = np.array([1.0], dtype="<f2").view(np.uint8)
        y = L._dequant_flat("IQ1_S", blk)
        self.assertTrue(np.all(y == np.float32(-0.875)))
        qh = np.full(8, 0x8000 | (1 << 12), dtype="<u2")
        blk[34:50] = qh.view(np.uint8)
        y = L._dequant_flat("IQ1_S", blk)
        self.assertTrue(np.all(y == np.float32(3 * -1.125)))


@unittest.skipIf(not all(C_SHA256.values()), "C reference hashes not recorded (run with --record)")
class CReferenceHashTest(unittest.TestCase):
    def test_synthetic_blocks_match_recorded_c(self):
        for t in TYPES:
            y = L._dequant_flat(t, _synthetic(t))
            self.assertEqual(_sha(y), C_SHA256[t],
                             f"{t}: the decode differs from ggml's C; KIMI_K31_PORT_DIR=... locates the value")


class GgufPyTest(unittest.TestCase):
    def test_synthetic_blocks_match_gguf_py(self):
        quants = _default_quants()
        if quants is None:
            self.skipTest("no gguf-py importable")
        for t in TYPES:
            if not hasattr(quants, t):
                self.skipTest(f"this gguf-py has no {t}")
            raw = _synthetic(t)
            with np.errstate(over="ignore"):                # MXFP4's e = 255 blocks overflow to inf
                a, b = L._dequant_flat(t, raw), L.gguf_py_dequant(t, raw, quants)
            self.assertTrue(np.array_equal(_bits(a), _bits(b)), f"{t}: {_first_diff(a, b, L.GEOM[t][0])}")


class LazyAddressingTest(unittest.TestCase):
    """expert()/rows()/blocks() read only their slice; their offsets are whole blocks of these types."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        path = pathlib.Path(cls.tmp.name) / "iq.gguf"
        w = gguf_writer.GGUFWriter()
        w.add("general.architecture", "test")
        cls.shapes = {}
        for t in TYPES:
            el, by = L.GEOM[t]
            for name, shape in ((f"exps.{t}.weight", [512, 3, 4]), (f"rows.{t}.weight", [512, 6])):
                n = int(np.prod(shape)) // el
                w.tensors.append(gguf_writer._Tensor(name, shape, t, _synthetic(t, n).tobytes()))
                cls.shapes[name] = shape
        with mock.patch.dict(gguf_writer.TYPE_IDS, TYPE_ID):
            w.write(path)
        cls.g = L.Gguf(path)

    @classmethod
    def tearDownClass(cls):
        cls.g.close()
        cls.tmp.cleanup()

    def test_expert_rows_blocks(self):
        for t in TYPES:
            el, _ = L.GEOM[t]
            whole = self.g.tensor(f"exps.{t}.weight")
            self.assertEqual(whole.shape, (4, 3, 512))
            for e in range(4):
                self.assertTrue(np.array_equal(_bits(self.g.expert(f"exps.{t}.weight", e)), _bits(whole[e])), t)
            m = self.g.tensor(f"rows.{t}.weight")
            self.assertEqual(m.shape, (6, 512))
            idx = [5, 0, 3]
            self.assertTrue(np.array_equal(_bits(self.g.rows(f"rows.{t}.weight", idx)), _bits(m[idx])), t)
            blk = self.g.blocks(f"exps.{t}.weight", 5, 7)
            self.assertTrue(np.array_equal(_bits(L._dequant_flat(t, blk)),
                                           _bits(whole.ravel()[5 * el:12 * el])), t)
            with self.assertRaises(ValueError):
                self.g.blocks(f"exps.{t}.weight", 4 * 3 * 512 // el - 2, 3)


class DumpTest(unittest.TestCase):
    """--dump writes what its docstring says, and each set's .f32 is the decode of its .blocks."""

    def test_dump_round_trip(self):
        if _default_quants() is None and not _port_ready():
            self.skipTest("no gguf-py importable")
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "fmt"
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(_dump(out, 32), 0)
                self.assertEqual(_dump(out, 32), 2, "a second dump into the same directory is refused")
            lines = [ln.split() for ln in (out / "manifest.txt").read_text().splitlines()
                     if ln and not ln.startswith("#")]
            self.assertEqual({ln[0] for ln in lines if ln[0].endswith(".synthetic")},
                             {f"{t}.synthetic" for t in TYPES})
            for stem, t, tid, el, by, n, sha_b, sha_f in lines:
                el, by, n = int(el), int(by), int(n)
                self.assertEqual((int(tid), (el, by)), (TYPE_ID[t], L.GEOM[t]), stem)
                rb, fb = (out / f"{stem}.blocks").read_bytes(), (out / f"{stem}.f32").read_bytes()
                self.assertEqual((len(rb), len(fb)), (n * by, n * el * 4), stem)
                self.assertEqual((hashlib.sha256(rb).hexdigest(), hashlib.sha256(fb).hexdigest()), (sha_b, sha_f))
                self.assertGreaterEqual(n, 32, stem)
                y = L._dequant_flat(t, np.frombuffer(rb, dtype=np.uint8))
                self.assertTrue(np.array_equal(_bits(y), np.frombuffer(fb, dtype="<u4")), stem)
            syn = (out / "IQ1_S.synthetic.blocks").read_bytes()
            self.assertEqual(syn, _synthetic("IQ1_S", 32).tobytes())


# ---------------------------------------------------------------- with the llama.cpp checkout
def _port_ready() -> bool:
    p = _port_dir()
    return bool(p and (p / "gguf-py" / "gguf" / "quants.py").is_file() and _ggml_lib(p))


class _PortCase(unittest.TestCase):
    """The checkout's C to_float and gguf-py, and the three-way comparison (no tests of its own)."""

    @classmethod
    def setUpClass(cls):
        port = _port_dir()
        cls.ggml = Ggml(_ggml_lib(port))
        cls.quants = _port_quants(port)

    def three_way(self, t: str, raw: np.ndarray, where: str) -> None:
        el = L.GEOM[t][0]
        ours = L._dequant_flat(t, raw)
        c = self.ggml.to_float(t, raw)
        with np.errstate(over="ignore"):
            py = L.gguf_py_dequant(t, raw, self.quants)
        self.assertTrue(np.array_equal(_bits(ours), _bits(c)), f"{t} {where} vs ggml C: {_first_diff(ours, c, el)}")
        self.assertTrue(np.array_equal(_bits(py), _bits(c)), f"{t} {where} gguf-py vs ggml C: {_first_diff(py, c, el)}")


@unittest.skipUnless(_port_ready(), "set KIMI_K31_PORT_DIR to a llama.cpp checkout with gguf-py and a built "
                                    "libggml-base")
class PortTest(_PortCase):
    def test_traits(self):
        for t in TYPES:
            tr = self.ggml.traits(t)
            self.assertEqual(tr.type_name.decode(), t.lower())
            self.assertEqual((tr.blck_size, tr.type_size), L.GEOM[t])
            self.assertEqual(self.quants.GGML_QUANT_SIZES[self.quants.GGMLQuantizationType[t]], L.GEOM[t])

    def test_synthetic_blocks(self):
        for t in TYPES:
            raw = _synthetic(t)
            self.three_way(t, raw, "synthetic")
            if C_SHA256[t]:
                self.assertEqual(_sha(self.ggml.to_float(t, raw)), C_SHA256[t], f"{t}: recorded hash is stale")


@unittest.skipUnless(_port_ready() and _real_paths(), "set KIMI_K31_PORT_DIR and KIMI_K31_REAL_GGUF")
class RealBlocksTest(_PortCase):
    """Real blocks from real K3.1 GGUFs. Each file is read in N_RUNS runs per type (a few MB at most)."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.files = _open_real()
        cls.rng = np.random.default_rng(3101)
        cls.report = []

    @classmethod
    def tearDownClass(cls):
        for _, g, _ in cls.files:
            g.close()
        if cls.report:
            print("\n" + "\n".join(cls.report), file=sys.stderr)

    def test_file_blocks(self):
        """The i-quant and MXFP4 tensors the files hold, as written."""
        seen = 0
        for p, g, limit in self.files:
            for t in TYPES:
                runs = _runs(g, t, limit, N_REAL, self.rng)
                if not runs:
                    continue
                n = 0
                for name, start, count in runs:
                    self.three_way(t, g.blocks(name, start, count), f"{p.name}:{name}@{start}+{count}")
                    n += count
                self.assertGreaterEqual(n, N_REAL, f"{p.name}: {t}")
                self.report.append(f"real {t:8s} {n:5d} blocks bit-equal x3  {p.name} ({len(runs)} tensors)")
                seen += 1
        self.assertGreater(seen, 0, "no MXFP4 or i-quant tensor in KIMI_K31_REAL_GGUF")

    def test_requantized_real_rows(self):
        """Every i-quant, made by ggml_quantize_chunk from real weight rows, decoded three ways."""
        src, groups = _source_rows(self.files, N_REAL * 256, self.rng)
        if not groups:
            self.skipTest("no MXFP4, BF16 or Q8_0 matrix with rows a multiple of 256 in KIMI_K31_REAL_GGUF")
        for t in IQ_TYPES:
            n = 0
            for where, rows in groups:
                raw = self.ggml.quantize(t, rows)
                self.three_way(t, raw, f"requantized {where}")
                y = L._dequant_flat(t, raw).reshape(rows.shape)
                rel = float(np.linalg.norm(y - rows) / max(np.linalg.norm(rows), 1e-30))
                self.assertLess(rel, 1.0, f"{t} {where}: the quantizer's output is not near its input")
                n += raw.size // L.GEOM[t][1]
            self.assertGreaterEqual(n, N_REAL, t)
            self.report.append(f"real {t:8s} {n:5d} blocks bit-equal x3  requantized from {src} rows "
                               f"({len(groups)} tensors)")

    def test_expert_matches_c(self):
        """expert() on a real fused expert tensor: the same bytes as the C decode of that expert's run."""
        for p, g, limit in self.files:
            for t in g.g.tensors:
                if t.type_name in TYPES and len(t.shape) == 3 and _complete(g, t, limit):
                    in_f, out_f, n_exp = t.shape
                    e = int(self.rng.integers(0, n_exp))
                    el = L.GEOM[t.type_name][0]
                    per = in_f * out_f // el
                    c = self.ggml.to_float(t.type_name, g.blocks(t.name, e * per, per)).reshape(out_f, in_f)
                    self.assertTrue(np.array_equal(_bits(g.expert(t.name, e)), _bits(c)), f"{p.name}:{t.name}[{e}]")
                    self.report.append(f"real expert() {t.type_name} {p.name}:{t.name}[{e}] = C")
                    break


def _record() -> int:
    port = _port_dir()
    if not _port_ready():
        print("set KIMI_K31_PORT_DIR to a llama.cpp checkout with gguf-py and a built libggml-base")
        return 2
    ggml = Ggml(_ggml_lib(port))
    print(f"# {ggml.path}")
    for t in TYPES:
        print(f'    "{t}": "{_sha(ggml.to_float(t, _synthetic(t)))}",')
    return 0


def _gguf_py_label(quants) -> str:
    """Where a gguf-py `quants` module comes from: its directory and the llama.cpp commit, or its version."""
    qdir = pathlib.Path(quants.__file__).resolve().parent.parent
    if (qdir.parent / ".git").exists():
        return f"{qdir}, git {_git_rev(qdir.parent)}"
    try:
        from importlib import metadata
        return f"{qdir}, gguf {metadata.version('gguf')}"
    except Exception:                                  # noqa: BLE001  (a label only)
        return str(qdir)


def _git_rev(path) -> str:
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "--short=10", "HEAD"], capture_output=True,
                           text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return "-"
    return r.stdout.strip() or "-"


def _dump(out: pathlib.Path, n_blocks: int) -> int:
    """--dump: the random-block references for tests/k31_fmt_test.cpp (the module docstring has the format)."""
    port = _port_dir() if _port_ready() else None
    quants = _port_quants(port) if port else _default_quants()
    if quants is None:
        print("--dump needs gguf-py (KIMI_K31_PORT_DIR, or one tools/_paths finds): the references are its decode")
        return 2
    if n_blocks < 4:
        print("--blocks must be at least 4 (blocks 0-3 carry the special scales)")
        return 2
    if (out / "manifest.txt").exists():
        print(f"{out / 'manifest.txt'} exists: dump into a new directory")
        return 2
    ggml = Ggml(_ggml_lib(port)) if port else None
    head = ["# k31_fmt references, written by tools/test_kimi_k31_load.py --dump",
            "# <stem> <type> <ggml type id> <block values> <block bytes> <n blocks> <sha256 .blocks> <sha256 .f32>",
            f"# .f32: gguf-py's decode ({_gguf_py_label(quants)})",
            "# checked bit for bit against ref/load.py" +
            (f" and ggml's C to_float ({ggml.path}, git {_git_rev(port)})" if ggml else
             " (not against ggml's C: KIMI_K31_PORT_DIR is unset)")]
    sets = [(f"{t}.synthetic", t, _synthetic(t, n_blocks), f"synthetic blocks, _synthetic({t!r}, {n_blocks})")
            for t in TYPES]
    if port and _real_paths():
        rng = np.random.default_rng(3101)
        files = _open_real()
        try:
            for p, g, limit in files:
                for t in TYPES:
                    runs = _runs(g, t, limit, n_blocks, rng)
                    if runs:
                        raw = np.concatenate([g.blocks(name, start, count) for name, start, count in runs])
                        sets.append((f"{t}.{p.stem}", t, raw, f"{p.name}, as written: " +
                                     "; ".join(f"{name}@{start}+{count}" for name, start, count in runs)))
            src, groups = _source_rows(files, n_blocks * 256, rng)
            for t in IQ_TYPES if groups else ():
                raw = np.concatenate([ggml.quantize(t, rows) for _, rows in groups])
                sets.append((f"{t}.requant", t, raw, f"ggml_quantize_chunk (imatrix of ones) of real {src} rows: " +
                             "; ".join(where for where, _ in groups)))
        finally:
            for _, g, _ in files:
                g.close()
    done = []
    for stem, t, raw, where in sets:                   # every set is checked before anything is written
        el, by = L.GEOM[t]
        with np.errstate(over="ignore"):               # MXFP4's e = 255 blocks overflow to inf
            ref = L.gguf_py_dequant(t, raw, quants)
        for who, y in (("ref/load.py", L._dequant_flat(t, raw)),
                       *((("ggml C", ggml.to_float(t, raw)),) if ggml else ())):
            if not np.array_equal(_bits(y), _bits(ref)):
                print(f"{stem}: {who} differs from gguf-py: {_first_diff(y, ref, el)}; nothing written")
                return 1
        done.append((stem, t, raw.tobytes(), ref.astype("<f4").tobytes(), where))
    out.mkdir(parents=True, exist_ok=True)
    lines = []
    for stem, t, rb, fb, where in done:
        el, by = L.GEOM[t]
        (out / f"{stem}.blocks").write_bytes(rb)
        (out / f"{stem}.f32").write_bytes(fb)
        head.append(f"# {stem}: {where}")
        lines.append(f"{stem} {t} {TYPE_ID[t]} {el} {by} {len(rb) // by} "
                     f"{hashlib.sha256(rb).hexdigest()} {hashlib.sha256(fb).hexdigest()}")
        print(f"{stem:28s} {len(rb) // by:6d} blocks")
    (out / "manifest.txt").write_text("\n".join(head + lines) + "\n")
    print(f"{len(lines)} sets in {out}")
    return 0


if __name__ == "__main__":
    if "--record" in sys.argv or "--dump" in sys.argv:
        ap = argparse.ArgumentParser(description="the C reference hashes (--record) or the k31_fmt references "
                                                 "(--dump DIR); see the module docstring")
        mode = ap.add_mutually_exclusive_group(required=True)
        mode.add_argument("--record", action="store_true", help="print the C reference hashes of the synthetic blocks")
        mode.add_argument("--dump", metavar="DIR", type=pathlib.Path, help="write the k31_fmt references into DIR")
        ap.add_argument("--blocks", type=int, default=N_SYNTH, help=f"blocks per type and set (default {N_SYNTH})")
        a = ap.parse_args()
        sys.exit(_record() if a.record else _dump(a.dump, a.blocks))
    unittest.main()

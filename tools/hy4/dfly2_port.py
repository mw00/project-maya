"""tools/hy4/dfly2_port.py - our NumPy port of the drafter's public description, for checking maya-hy4's --dfly2 pass
against a dump (HY4_DFLY_DUMP).

    python tools/hy4/dfly2_port.py --drafter FILE.gguf --dump DIR [--model-gguf SHARD ...] [--q8-inputs]
                                   [--block-rows own|dump] [--from-dump] [--rope-base 1e7] [--head NAME]
    python tools/hy4/dfly2_port.py --drafter FILE.gguf --check      every tensor of the file dequantised: finite, sane
    python tools/hy4/dfly2_port.py --selftest                       the dequantisers, the RoPE and the pass, on tiny data

The dump (one draft round, little-endian raw): meta.txt (`tok`, `pos`, `n`, `q8`, `mask`, `drafts ...`, `conf ...`),
h_last.f32 [E], emb.f32 [2][E] (the rows of tok and of the mask token), x0.f32 [N+1][E] (after the fusion); a draft
layer i: keys_i.i32 (the last row's key list: the history positions, then pos .. pos+N), cache_i.f16 [pos][kv+rope]
(the target layer's latent rows of positions 0 .. pos-1), x{i+1}.f32 (the block after the layer), qrows_i.f16
[N+1][kv+rope] (the rows the draft layer wrote); h.f32 (after final_norm); hc_s.f32 [E] a produced draft.

The pass (a matrix of dims [k, n] is n rows of k values, y = W @ x; every norm y = x / sqrt(mean(x^2) + 1e-5) * w):
the query block [tok, mask x N] at positions pos .. pos+N, the fusion on row 0, the draft layers (the latent attention
over the target layer's cache rows at the history positions and the block's own rows up to the row's, a sink term in
the softmax's sum, gated values; a SwiGLU FFN), final_norm, then a draft a step from the corrected state.  With
--model-gguf the model's embedding rows and head give the port's own tokens; without, step 0's state alone.

--q8-inputs: the engine's GEMV inputs (a vector rounded to 8 bits a block of 32 before every matrix product).  The
key matrix's transpose is held rounded the same way by the engine in both modes: so here.

Products in float32; the report's differences in float64: max |port - dump|, that over the dump's max |value|, and the
RMS of the difference over the dump's RMS.
"""
from __future__ import annotations

import argparse
import importlib.util
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("hy4_pack_gguf", HERE / "pack_gguf.py")
_pack = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_pack)
Gguf, TYPES = _pack.Gguf, _pack.TYPES

PREFIX = "model.dfly_layers."
EPS = np.float32(1e-5)
F32 = np.float32


# ---------------------------------------------------------------------------------------------------------------------
# ggml's formats -> float32 (raw: the uint8 bytes of whole blocks; the result flat)

def deq_f32(raw: np.ndarray) -> np.ndarray:
    return raw.view("<f4").astype(F32)


def deq_f16(raw: np.ndarray) -> np.ndarray:
    return raw.view("<f2").astype(F32)


def deq_bf16(raw: np.ndarray) -> np.ndarray:
    return (raw.view("<u2").astype(np.uint32) << 16).view(F32)


def deq_q8_0(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 34)
    d = np.ascontiguousarray(b[:, :2]).view("<f2").astype(F32)          # [nb, 1]
    return (d * b[:, 2:].view(np.int8).astype(F32)).reshape(-1)


def deq_q6_k(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 210)
    nb = b.shape[0]
    ql = b[:, :128].reshape(nb, 2, 64)
    qh = b[:, 128:192].reshape(nb, 2, 32)
    sc = b[:, 192:208].view(np.int8).reshape(nb, 2, 8, 1).astype(F32)
    d = np.ascontiguousarray(b[:, 208:210]).view("<f2").astype(F32).reshape(nb, 1, 1, 1)
    a, c = ql[:, :, :32], ql[:, :, 32:]
    q = np.stack([(a & 0xF) | ((qh & 3) << 4), (c & 0xF) | (((qh >> 2) & 3) << 4),
                  (a >> 4) | (((qh >> 4) & 3) << 4), (c >> 4) | (((qh >> 6) & 3) << 4)], axis=2)   # [nb, 2, 4, 32]
    q = (q.astype(np.int8) - np.int8(32)).reshape(nb, 2, 8, 16).astype(F32)
    return ((d * sc) * q).reshape(-1)


def _k4_scales(s: np.ndarray) -> tuple:
    """Q4_K / Q5_K's 12 scale bytes [nb, 12] -> the 8 6-bit scales and mins, uint8 [nb, 8]."""
    lo_sc, lo_m = s[:, 0:4] & 63, s[:, 4:8] & 63
    hi_sc = (s[:, 8:12] & 0xF) | ((s[:, 0:4] >> 6) << 4)
    hi_m = (s[:, 8:12] >> 4) | ((s[:, 4:8] >> 6) << 4)
    return np.concatenate([lo_sc, hi_sc], axis=1), np.concatenate([lo_m, hi_m], axis=1)


def _k45(b: np.ndarray, q: np.ndarray) -> np.ndarray:
    """d * scale * q - dmin * min, q uint8 [nb, 8, 32] (the 8 sub-blocks of 32 in order)."""
    nb = b.shape[0]
    d = np.ascontiguousarray(b[:, 0:2]).view("<f2").astype(F32).reshape(nb, 1, 1)
    dmin = np.ascontiguousarray(b[:, 2:4]).view("<f2").astype(F32).reshape(nb, 1, 1)
    sc, m = _k4_scales(b[:, 4:16])
    return ((d * sc.astype(F32)[:, :, None]) * q.astype(F32) - dmin * m.astype(F32)[:, :, None]).reshape(-1)


def deq_q4_k(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 144)
    qs = b[:, 16:144].reshape(-1, 4, 32)
    return _k45(b, np.stack([qs & 0xF, qs >> 4], axis=2).reshape(-1, 8, 32))


def deq_q5_k(raw: np.ndarray) -> np.ndarray:
    b = raw.reshape(-1, 176)
    qh = b[:, 16:48].reshape(-1, 1, 32)
    qs = b[:, 48:176].reshape(-1, 4, 32)
    sh = (2 * np.arange(4, dtype=np.uint8)).reshape(1, 4, 1)
    lo = (qs & 0xF) + (((qh >> sh) & 1) << 4)
    hi = (qs >> 4) + (((qh >> (sh + 1)) & 1) << 4)
    return _k45(b, np.stack([lo, hi], axis=2).reshape(-1, 8, 32))


DEQ = {"F32": deq_f32, "F16": deq_f16, "BF16": deq_bf16, "Q8_0": deq_q8_0, "Q6_K": deq_q6_k, "Q4_K": deq_q4_k,
       "Q5_K": deq_q5_k}


class Weights:
    """The tensors of some GGUF files (the first that has a name wins), dequantised on demand."""

    def __init__(self, paths: list):
        self.files = []
        self.index = {}
        for p in paths:
            g = Gguf(pathlib.Path(p))
            mm = np.memmap(g.path, dtype=np.uint8, mode="r")
            self.files.append((g, mm))
            for name, typ, dims, off in g.tensors:
                self.index.setdefault(name, (len(self.files) - 1, typ, list(dims), off))

    def has(self, name: str) -> bool:
        return name in self.index

    def kind(self, name: str) -> str:
        return TYPES[self.index[name][1]][0]

    def shape(self, name: str) -> tuple:
        """(rows, values a row)."""
        dims = self.index[name][2]
        if len(dims) > 2:
            raise SystemExit("%s: %d dims" % (name, len(dims)))
        return (dims[1] if len(dims) == 2 else 1), dims[0]

    def rows(self, name: str, r0: int = 0, r1: int | None = None) -> np.ndarray:
        """Rows r0 .. r1-1 of a matrix, float32 [rows, k]."""
        fi, typ, dims, off = self.index[name]
        tname, per, bpb = TYPES[typ]
        if tname not in DEQ:
            raise SystemExit("%s: a %s tensor, not read by this port" % (name, tname))
        n, k = self.shape(name)
        r1 = n if r1 is None else r1
        if k % per:
            raise SystemExit("%s: a %s row of %d values is not whole blocks" % (name, tname, k))
        rb = k // per * bpb
        g, mm = self.files[fi]
        at = g.data_start + off
        raw = np.asarray(mm[at + r0 * rb: at + r1 * rb])
        return DEQ[tname](raw).reshape(r1 - r0, k)

    def get(self, name: str) -> np.ndarray:
        """A matrix [n, k]; a vector flat."""
        w = self.rows(name)
        return w.reshape(-1) if len(self.index[name][2]) == 1 else w


class Dict(Weights):
    """Weights from arrays (the self-test's)."""

    def __init__(self, d: dict):   # noqa: D107
        self.d = d

    def has(self, name):
        return name in self.d

    def kind(self, name):
        return "F32"

    def shape(self, name):
        a = self.d[name]
        return (a.shape[0], a.shape[1]) if a.ndim == 2 else (1, a.shape[0])

    def rows(self, name, r0=0, r1=None):
        a = self.d[name]
        return (a if a.ndim == 2 else a[None])[r0:r1].astype(F32)

    def get(self, name):
        return self.d[name].astype(F32)


# ---------------------------------------------------------------------------------------------------------------------
# the math

def rms(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    x = x.astype(F32, copy=False)
    return (x / np.sqrt(np.mean(x * x, axis=-1, keepdims=True, dtype=F32) + EPS) * w).astype(F32)


def silu(x: np.ndarray) -> np.ndarray:
    return (x / (F32(1) + np.exp(-x))).astype(F32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (F32(1) / (F32(1) + np.exp(-x))).astype(F32)


def q8in(x: np.ndarray) -> np.ndarray:
    """The engine's GEMV input: the last axis rounded a block of 32 - sc = max|x| / 127, rint(x / sc) * sc."""
    k = x.shape[-1]
    pad = (-k) % 32
    v = x.astype(F32, copy=False)
    if pad:
        v = np.concatenate([v, np.zeros(v.shape[:-1] + (pad,), F32)], axis=-1)
    b = v.reshape(v.shape[:-1] + (-1, 32))
    sc = (np.max(np.abs(b), axis=-1, keepdims=True) / F32(127)).astype(F32)
    safe = np.where(sc == 0, F32(1), sc)
    out = (np.rint(b / safe) * sc).astype(F32).reshape(v.shape)
    return out[..., :k] if pad else out


def round_q8_rows(w: np.ndarray) -> np.ndarray:
    """A matrix as the engine holds one it made itself: each row in Q8_0 blocks of 32 (the scale a float16, the
    values rounded half away from zero)."""
    k = w.shape[-1]
    pad = (-k) % 32
    v = w.astype(F32, copy=False)
    if pad:
        v = np.concatenate([v, np.zeros(v.shape[:-1] + (pad,), F32)], axis=-1)
    b = v.reshape(v.shape[:-1] + (-1, 32))
    d32 = (np.max(np.abs(b), axis=-1, keepdims=True) / F32(127)).astype(F32)
    with np.errstate(divide="ignore"):
        inv = np.where(d32 != 0, F32(1) / d32, F32(0)).astype(F32)
    t = (b * inv).astype(F32)
    q = (np.sign(t) * np.floor(np.abs(t) + F32(0.5))).astype(np.int8)
    out = (d32.astype(np.float16).astype(F32) * q.astype(F32)).reshape(v.shape)
    return out[..., :k] if pad else out


def mv(w: np.ndarray, x: np.ndarray, q8: bool) -> np.ndarray:
    """W [n, k] @ x [..., k] -> [..., n]."""
    x = q8in(x) if q8 else x.astype(F32, copy=False)
    return (x.reshape(-1, x.shape[-1]) @ w.T).reshape(x.shape[:-1] + (w.shape[0],)).astype(F32, copy=False)


def rope(x: np.ndarray, pos, base: float) -> np.ndarray:
    """The pairs (x[2i], x[2i+1]) of the last axis turned by pos * base^(-2i/dim); pos broadcasts over x[..., 0]."""
    dim = x.shape[-1]
    inv = np.float64(base) ** (-2.0 * np.arange(dim // 2) / dim)
    ang = np.asarray(pos, np.float64)[..., None] * inv
    c, s = np.cos(ang).astype(F32), np.sin(ang).astype(F32)
    a, b = x[..., 0::2].astype(F32), x[..., 1::2].astype(F32)
    out = np.empty(np.broadcast(a, c).shape[:-1] + (dim,), F32)
    out[..., 0::2] = a * c - b * s
    out[..., 1::2] = b * c + a * s
    return out


class Shape:
    """The drafter's sizes, from its file."""

    def __init__(self, w: Weights, heads: int | None = None, n_layers: int | None = None):
        p = PREFIX + "0.self_attn."
        self.q_lora, self.E = w.shape(p + "q_a_proj.weight")
        kvr, self.kv_lora = w.shape(p + "kv_b_proj.weight")
        self.H = heads or w.shape(p + "learnable_sink_param")[1]
        self.rope = w.shape(p + "kv_a_proj_with_mqa.weight")[0] - self.kv_lora
        self.v = w.shape(p + "linear_gate.weight")[0] // self.H
        self.nope = kvr // self.H - self.v
        self.row = self.kv_lora + self.rope
        if n_layers is None:
            n_layers = 0
            while w.has("%s%d.self_attn.q_a_proj.weight" % (PREFIX, n_layers)):
                n_layers += 1
        self.layers = n_layers


def fusion(w: Weights, emb_tok: np.ndarray, emb_mask: np.ndarray, h_last: np.ndarray, n: int, q8: bool) -> np.ndarray:
    p = PREFIX + "mtp_fusion."
    x = np.concatenate([emb_tok[None], np.repeat(emb_mask[None], n, axis=0)]).astype(F32)
    c = np.concatenate([rms(x[0], w.get(p + "enorm.weight")), rms(h_last, w.get(p + "hnorm.weight"))])
    x[0] += w.get(p + "gate") * mv(w.get(p + "eh_proj.weight"), c, q8)
    return x


def layer(w: Weights, i: int, x: np.ndarray, pos: int, hist: np.ndarray, sh: Shape, q8: bool, base: float,
          block_rows: np.ndarray | None = None) -> tuple:
    """Draft layer i on the block x [R, E] at positions pos .. pos+R-1; hist [P, kv+rope]: the history's cache rows;
    block_rows: the block's own key rows to use (None: the ones computed here, rounded to float16).
    -> (the block after the layer, the rows it wrote [R, kv+rope] float16)."""
    p = "%s%d." % (PREFIX, i)
    a = p + "self_attn."
    R, H = x.shape[0], sh.H
    at = pos + np.arange(R)
    xa = rms(x, w.get(p + "input_layernorm.weight"))
    qr = rms(mv(w.get(a + "q_a_proj.weight"), xa, q8), w.get(a + "q_a_layernorm.weight"))
    q = mv(w.get(a + "q_b_proj.weight"), qr, q8).reshape(R, H, sh.nope + sh.rope)
    kv = mv(w.get(a + "kv_a_proj_with_mqa.weight"), xa, q8)
    own = np.concatenate([rms(kv[:, :sh.kv_lora], w.get(a + "kv_a_layernorm.weight")),
                          rope(kv[:, sh.kv_lora:], at, base)], axis=1).astype(np.float16)
    gate = mv(w.get(a + "linear_gate.weight"), xa, q8).reshape(R, H, sh.v)
    kvb = w.get(a + "kv_b_proj.weight").reshape(H, sh.nope + sh.v, sh.kv_lora)
    wkt = round_q8_rows(np.ascontiguousarray(kvb[:, :sh.nope].transpose(0, 2, 1)))   # [H, kv_lora, nope]
    wv = kvb[:, sh.nope:]                                                             # [H, v, kv_lora]
    qn = q[:, :, :sh.nope]
    qn = q8in(qn) if q8 else qn
    ql = np.einsum("hkn,rhn->rhk", wkt, qn, optimize=True).astype(F32)
    qfull = np.concatenate([ql, rope(q[:, :, sh.nope:], at[:, None], base)], axis=2)  # [R, H, kv+rope]
    keys = np.concatenate([hist.astype(F32), (own if block_rows is None else block_rows).astype(F32)])
    P = hist.shape[0]
    s = (qfull.reshape(R * H, sh.row) @ keys.T).reshape(R, H, P + R) / F32(np.sqrt(sh.nope + sh.rope))
    later = np.arange(R)[None, :] > np.arange(R)[:, None]                             # [row, block key]: not seen
    s[:, :, P:][np.broadcast_to(later[:, None, :], (R, H, R))] = -np.inf
    sink = w.get(a + "learnable_sink_param").reshape(1, H, 1)
    m = np.maximum(s.max(axis=-1, keepdims=True), sink)
    e = np.exp(s - m)
    prob = (e / (e.sum(axis=-1, keepdims=True) + np.exp(sink - m))).astype(F32)
    o = (prob.reshape(R * H, P + R) @ keys[:, :sh.kv_lora]).reshape(R, H, sh.kv_lora)
    o = q8in(o) if q8 else o
    v = np.einsum("hvk,rhk->rhv", wv, o, optimize=True).astype(F32) * sigmoid(gate)
    y = mv(w.get(a + "o_proj.weight"), v.reshape(R, H * sh.v), q8)
    x = x + rms(y, w.get(a + "post_proj_norm.weight"))
    xf = rms(x, w.get(p + "post_attention_layernorm.weight"))
    f = silu(mv(w.get(p + "mlp.gate_proj.weight"), xf, q8)) * mv(w.get(p + "mlp.up_proj.weight"), xf, q8)
    return (x + mv(w.get(p + "mlp.down_proj.weight"), f, q8)).astype(F32), own


def correct(w: Weights, h_row: np.ndarray, emb_prev: np.ndarray, q8: bool) -> np.ndarray:
    p = PREFIX + "hidden_correction."
    c = np.concatenate([rms(h_row, w.get(p + "hidden_norm.weight")), rms(emb_prev, w.get(p + "embed_norm.weight"))])
    f = silu(mv(w.get(p + "gate_proj.weight"), c, q8)) * mv(w.get(p + "up_proj.weight"), c, q8)
    return (h_row + mv(w.get(p + "down_proj.weight"), f, q8)).astype(F32)


def head(model: Weights, name: str, hc: np.ndarray, q8: bool, chunk: int = 8192) -> tuple:
    """(argmax, its softmax probability) of name @ hc, the matrix read a chunk of rows at a time."""
    n, _ = model.shape(name)
    x = q8in(hc) if q8 else hc.astype(F32)
    logits = np.empty(n, F32)
    for r0 in range(0, n, chunk):
        r1 = min(n, r0 + chunk)
        logits[r0:r1] = model.rows(name, r0, r1) @ x
    best = int(np.argmax(logits))
    z = logits.astype(np.float64) - float(logits[best])
    return best, float(1.0 / np.exp(z).sum())


# ---------------------------------------------------------------------------------------------------------------------
# the dump and the report

def diff(name: str, mine: np.ndarray, ref: np.ndarray) -> float:
    a, b = np.asarray(mine, np.float64).reshape(-1), np.asarray(ref, np.float64).reshape(-1)
    if a.shape != b.shape:
        print("%-10s shapes differ: the port's %d values, the dump's %d" % (name, a.size, b.size))
        return float("inf")
    d = np.abs(a - b)
    top, peak, r = float(d.max()), float(np.abs(b).max()), float(np.sqrt(np.mean(b * b)))
    print("%-10s max|d| %.3e  /max %.3e  rms-rel %.3e" % (name, top, top / peak if peak else float("inf"),
                                                           float(np.sqrt(np.mean(d * d))) / r if r else float("inf")))
    return top


class Dump:
    def __init__(self, path: pathlib.Path):
        self.path = path
        self.meta = {}
        for line in (path / "meta.txt").read_text().splitlines():
            k, _, v = line.strip().partition(" ")
            if k:
                self.meta[k] = v.split()
        self.tok, self.pos, self.n = (int(self.meta[k][0]) for k in ("tok", "pos", "n"))
        self.q8 = int(self.meta.get("q8", ["0"])[0]) != 0
        self.mask = int(self.meta.get("mask", ["-1"])[0])
        self.drafts = [int(x) for x in self.meta.get("drafts", [])]
        self.conf = [float(x) for x in self.meta.get("conf", [])]

    def has(self, name: str) -> bool:
        return (self.path / name).exists()

    def read(self, name: str, dtype: str, cols: int | None = None) -> np.ndarray:
        a = np.fromfile(self.path / name, dtype=dtype)
        if cols:
            if a.size % cols:
                raise SystemExit("%s: %d values, not rows of %d" % (name, a.size, cols))
            a = a.reshape(-1, cols)
        return a


def run(a) -> int:
    w = Weights([a.drafter])
    d = Dump(pathlib.Path(a.dump))
    sh = Shape(w)
    R = d.n + 1
    q8 = a.q8_inputs
    mode = a.block_rows or ("dump" if d.q8 else "own")
    print("tok %d pos %d n %d (the dump's q8 %d); %d layers, E %d, %d heads; q8 inputs %s, the block's rows: %s%s" %
          (d.tok, d.pos, d.n, d.q8, sh.layers, sh.E, sh.H, "on" if q8 else "off", mode,
           ", each stage from the dump's input" if a.from_dump else ""))
    emb = d.read("emb.f32", "<f4", sh.E)
    x = fusion(w, emb[0], emb[1], d.read("h_last.f32", "<f4"), d.n, q8)
    worst = 0.0

    def check(name: str, mine: np.ndarray, cols: int, dtype: str = "<f4") -> np.ndarray:
        nonlocal worst
        if not d.has(name):
            return mine
        ref = d.read(name, dtype, cols)
        worst = max(worst, diff(name.split(".")[0], mine, ref))
        return ref.astype(F32) if a.from_dump and dtype == "<f4" else mine

    x = check("x0.f32", x, sh.E)
    for i in range(sh.layers):
        keys = d.read("keys_%d.i32" % i, "<i4")
        want = list(range(d.pos, d.pos + R))
        if list(keys[-R:]) != want:
            print("keys_%d: the list does not end with the block's positions %d .. %d" % (i, want[0], want[-1]))
        hist = keys[:-R]
        cache = d.read("cache_%d.f16" % i, "<f2", sh.row)
        if hist.size and (hist.min() < 0 or hist.max() >= cache.shape[0]):
            raise SystemExit("keys_%d: a history position outside the cache's %d rows" % (i, cache.shape[0]))
        qrows = d.read("qrows_%d.f16" % i, "<f2", sh.row) if d.has("qrows_%d.f16" % i) else None
        if mode == "dump" and qrows is None:
            raise SystemExit("--block-rows dump: no qrows_%d.f16" % i)
        x, own = layer(w, i, x, d.pos, cache[hist], sh, q8, a.rope_base, qrows if mode == "dump" else None)
        if qrows is not None:
            diff("qrows_%d" % i, own, qrows)
        x = check("x%d.f32" % (i + 1), x, sh.E)
    h = check("h.f32", rms(x, w.get(PREFIX + "final_norm.weight")), sh.E)

    model = Weights(a.model_gguf) if a.model_gguf else None
    table = model is not None and model.has("token_embd.weight") and model.kind("token_embd.weight") in DEQ
    heads = model is not None and model.has(a.head) and model.kind(a.head) in DEQ
    if model is not None and not (table and heads):
        print("token_embd.weight / %s: missing from --model-gguf or of a type this port does not read - step 0's "
              "state alone" % a.head)
    steps = max(len(d.drafts), sum(1 for s in range(d.n) if d.has("hc_%d.f32" % s)))
    steps = max(1, min(steps, d.n, len(d.drafts) + 1)) if table else 1
    mine, same = [], 0
    for s in range(steps):
        prev = emb[0] if s == 0 else model.rows("token_embd.weight", d.drafts[s - 1], d.drafts[s - 1] + 1)[0]
        hc = check("hc_%d.f32" % s, correct(w, h[s + 1], prev, q8), sh.E)
        if heads:
            t, pr = head(model, a.head, hc, q8)
            mine.append(t)
            theirs = d.drafts[s] if s < len(d.drafts) else None
            same += t == theirs
            print("draft %d: the port's %d (p %.4f), the dump's %s%s%s" %
                  (s, t, pr, theirs, " (conf %.4f)" % d.conf[s] if s < len(d.conf) else "",
                   "" if t == theirs else "  DIFFERENT"))
    if heads:
        print("drafts: the port's %s, the dump's %s - %d of %d the same" % (mine, d.drafts, same, len(mine)))
    else:
        print("drafts: the dump's %s (no head: not checked)" % d.drafts)
    print("largest max|d| of the states: %.3e" % worst)
    return 0


def check_file(path: str) -> int:
    """Every tensor of the drafter's file dequantised: all finite, no all-zero matrix, a plausible scale."""
    w = Weights([path])
    bad, lo, hi, n = [], float("inf"), 0.0, 0
    for name in w.index:
        if w.kind(name) not in DEQ:
            bad.append("%s: a %s tensor, not read" % (name, w.kind(name)))
            continue
        x = w.rows(name)
        n += 1
        if not np.isfinite(x).all():
            bad.append("%s: not finite" % name)
            continue
        r, top = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))), float(np.abs(x).max())
        if x.shape[0] > 1:
            lo, hi = min(lo, r), max(hi, r)
            if r == 0 or r > 10 or top > 1e3:
                bad.append("%s: rms %.3g, max %.3g" % (name, r, top))
        elif top > 1e4:
            bad.append("%s: max %.3g" % (name, top))
    print("%s: %d tensors dequantised, the matrices' rms %.3g .. %.3g, %d odd" % (pathlib.Path(path).name, n, lo, hi,
                                                                                   len(bad)))
    for b in bad[:20]:
        print("  " + b)
    return 1 if bad else 0


# ---------------------------------------------------------------------------------------------------------------------
# the self-test

def _q6k_scalar(blk: bytes) -> list:
    """One Q6_K block, value by value (ggml's dequantize_row_q6_K)."""
    ql, qh = blk[:128], blk[128:192]
    sc = np.frombuffer(blk[192:208], np.int8)
    d = F32(np.frombuffer(blk[208:210], "<f2")[0])
    y = [F32(0)] * 256
    for half in range(2):
        L, Hh, S, Y = ql[64 * half:], qh[32 * half:], sc[8 * half:], 128 * half
        for l in range(32):   # noqa: E741
            i = l // 16
            q1 = ((L[l] & 0xF) | (((Hh[l] >> 0) & 3) << 4)) - 32
            q2 = ((L[l + 32] & 0xF) | (((Hh[l] >> 2) & 3) << 4)) - 32
            q3 = ((L[l] >> 4) | (((Hh[l] >> 4) & 3) << 4)) - 32
            q4 = ((L[l + 32] >> 4) | (((Hh[l] >> 6) & 3) << 4)) - 32
            y[Y + l] = d * F32(S[i]) * F32(q1)
            y[Y + l + 32] = d * F32(S[i + 2]) * F32(q2)
            y[Y + l + 64] = d * F32(S[i + 4]) * F32(q3)
            y[Y + l + 96] = d * F32(S[i + 6]) * F32(q4)
    return y


def _k45_scalar(blk: bytes, five: bool) -> list:
    """One Q4_K / Q5_K block, value by value (ggml's dequantize_row_q4_K / q5_K)."""
    d, dmin = (F32(v) for v in np.frombuffer(blk[:4], "<f2"))
    s = blk[4:16]
    qh, q = (blk[16:48], blk[48:176]) if five else (None, blk[16:144])

    def sm(j):
        if j < 4:
            return s[j] & 63, s[j + 4] & 63
        return (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4), (s[j + 4] >> 4) | ((s[j] >> 6) << 4)
    y = []
    for i in range(4):
        (s1, m1), (s2, m2) = sm(2 * i), sm(2 * i + 1)
        c = q[32 * i: 32 * i + 32]
        y += [d * F32(s1) * F32((c[l] & 0xF) + (16 if five and qh[l] & (1 << (2 * i)) else 0)) - dmin * F32(m1)
              for l in range(32)]   # noqa: E741
        y += [d * F32(s2) * F32((c[l] >> 4) + (16 if five and qh[l] & (2 << (2 * i)) else 0)) - dmin * F32(m2)
              for l in range(32)]   # noqa: E741
    return y


def _layer_slow(w: Dict, i: int, x: np.ndarray, pos: int, hist: np.ndarray, sh: Shape, base: float) -> np.ndarray:
    """The layer a row, a head and a key at a time (float64), from the description."""
    g = lambda n: w.get("%s%d.%s" % (PREFIX, i, n)).astype(np.float64)  # noqa: E731

    def norm(v, wn):
        return v / np.sqrt(np.mean(v * v) + 1e-5) * wn

    def rot(v, p):
        out = v.copy()
        for j in range(len(v) // 2):
            t = p * base ** (-2.0 * j / len(v))
            out[2 * j] = v[2 * j] * np.cos(t) - v[2 * j + 1] * np.sin(t)
            out[2 * j + 1] = v[2 * j + 1] * np.cos(t) + v[2 * j] * np.sin(t)
        return out
    R, H, km = x.shape[0], sh.H, sh.nope + sh.rope
    kvb = g("self_attn.kv_b_proj.weight")
    x = x.astype(np.float64).copy()
    rows, qs, gates = [], [], []
    for j in range(R):
        xa = norm(x[j], g("input_layernorm.weight"))
        qs.append(g("self_attn.q_b_proj.weight") @ norm(g("self_attn.q_a_proj.weight") @ xa,
                                                        g("self_attn.q_a_layernorm.weight")))
        kv = g("self_attn.kv_a_proj_with_mqa.weight") @ xa
        row = np.concatenate([norm(kv[:sh.kv_lora], g("self_attn.kv_a_layernorm.weight")), rot(kv[sh.kv_lora:], pos + j)])
        rows.append(row.astype(np.float16).astype(np.float64))
        gates.append(g("self_attn.linear_gate.weight") @ xa)
    out = x.copy()
    for j in range(R):
        keys = [r.astype(np.float64) for r in hist] + rows[:j + 1]
        vs = []
        for h in range(H):
            q = qs[j][h * km:(h + 1) * km]
            wk = kvb[h * (sh.nope + sh.v): h * (sh.nope + sh.v) + sh.nope]
            wv = kvb[h * (sh.nope + sh.v) + sh.nope: (h + 1) * (sh.nope + sh.v)]
            wkt = round_q8_rows(wk.T.astype(F32)).astype(np.float64)
            ql, qp = wkt @ q[:sh.nope], rot(q[sh.nope:], pos + j)
            sc = np.array([(ql @ k[:sh.kv_lora] + qp @ k[sh.kv_lora:]) / np.sqrt(km) for k in keys])
            e = np.exp(sc)
            pr = e / (e.sum() + np.exp(g("self_attn.learnable_sink_param")[h]))
            o = sum(p_ * k[:sh.kv_lora] for p_, k in zip(pr, keys))
            vs.append((wv @ o) / (1.0 + np.exp(-gates[j][h * sh.v:(h + 1) * sh.v])))
        y = g("self_attn.o_proj.weight") @ np.concatenate(vs)
        r = x[j] + norm(y, g("self_attn.post_proj_norm.weight"))
        xf = norm(r, g("post_attention_layernorm.weight"))
        ga = g("mlp.gate_proj.weight") @ xf
        out[j] = r + g("mlp.down_proj.weight") @ (ga / (1.0 + np.exp(-ga)) * (g("mlp.up_proj.weight") @ xf))
    return out


def selftest() -> int:
    rng = np.random.default_rng(7)
    ok = True

    def said(name: str, good: bool, more: str = ""):
        nonlocal ok
        ok = ok and bool(good)
        print("%-34s %s%s" % (name, "ok" if good else "FAILED", more))

    raw = rng.integers(0, 256, 3 * 210, dtype=np.uint8)
    raw.reshape(3, 210)[:, 208:210] = np.array([0.0123, -0.5, 2.0], "<f2").view(np.uint8).reshape(3, 2)
    want = np.array(sum((_q6k_scalar(bytes(raw[i * 210:(i + 1) * 210])) for i in range(3)), []), F32)
    said("Q6_K (3 blocks, value by value)", np.array_equal(deq_q6_k(raw), want))
    blk = np.zeros(210, np.uint8)   # ql 0x21: q1 = q2 = 1 - 32, q3 = q4 = 2 - 32 with qh 0; scales 1 .. 16; d 0.5
    blk[:128], blk[192:208], blk[208:210] = 0x21, np.arange(1, 17), np.array([0.5], "<f2").view(np.uint8)
    y = deq_q6_k(blk)
    said("Q6_K (by hand)", (y[0], y[16], y[32], y[64], y[127], y[128], y[255]) ==
         (0.5 * 1 * -31, 0.5 * 2 * -31, 0.5 * 3 * -31, 0.5 * 5 * -30, 0.5 * 8 * -30, 0.5 * 9 * -31, 0.5 * 16 * -30))
    blk[128] = 0b11100100   # l = 0: the high bits 0, 1, 2, 3 of q1 .. q4
    y = deq_q6_k(blk)
    said("Q6_K (the high bits)", (y[0], y[32], y[64], y[96]) ==
         (0.5 * 1 * -31, 0.5 * 3 * (17 - 32), 0.5 * 5 * (34 - 32), 0.5 * 7 * (50 - 32)))

    q = rng.integers(-127, 128, (2, 32)).astype(np.int8)
    d = np.array([0.03125, -1.5], "<f2")
    raw = np.concatenate([np.concatenate([d[i:i + 1].view(np.uint8), q[i].view(np.uint8)]) for i in range(2)])
    said("Q8_0", np.array_equal(deq_q8_0(raw), (d.astype(F32)[:, None] * q.astype(F32)).reshape(-1)))
    said("BF16", np.array_equal(deq_bf16(np.array([0x3F80, 0xC000, 0x3E00], "<u2").view(np.uint8)),
                                np.array([1.0, -2.0, 0.125], F32)))
    for name, nb, f, five in (("Q4_K", 144, deq_q4_k, False), ("Q5_K", 176, deq_q5_k, True)):
        raw = rng.integers(0, 256, 2 * nb, dtype=np.uint8)
        raw.reshape(2, nb)[:, :4] = np.array([[0.01, 0.02], [0.3, -0.07]], "<f2").view(np.uint8).reshape(2, 4)
        want = np.array(sum((_k45_scalar(bytes(raw[i * nb:(i + 1) * nb]), five) for i in range(2)), []), F32)
        said("%s (2 blocks, value by value)" % name, np.array_equal(f(raw), want))

    v = rng.standard_normal(64).astype(F32)
    u = rng.standard_normal(64).astype(F32)
    said("RoPE (position 0: as it was)", np.array_equal(rope(v, 0, 1e7), v))
    r = rope(np.array([1, 0, 1, 0], F32), 3, 1e4)   # (pair 0 by 3 rad, pair 1 by 3 * 1e4^(-1/2) = 0.03 rad)
    said("RoPE (by hand)", np.allclose(r, [np.cos(3), np.sin(3), np.cos(0.03), np.sin(0.03)], atol=1e-6))
    said("RoPE (lengths, relative turns)",
         np.allclose(np.linalg.norm(rope(v, 1234, 1e7)), np.linalg.norm(v), rtol=1e-5) and
         np.allclose(rope(v, 1005, 1e7) @ rope(u, 1002, 1e7), rope(v, 3, 1e7) @ u, atol=2e-3))
    vs = rng.standard_normal((3, 2, 64)).astype(F32)
    rr = rope(vs, np.array([5, 6, 7])[:, None], 1e7)
    said("RoPE (rows x heads)", np.array_equal(rr[1, 1], rope(vs[1, 1], 6, 1e7)))

    x = rng.standard_normal((2, 96)).astype(F32)
    x[1, 32:64] = 0
    y = q8in(x)
    sc = np.abs(x[0, :32]).max() / F32(127)
    said("the 8-bit input", np.array_equal(y[0, :32], (np.rint(x[0, :32] / sc) * sc).astype(F32)) and
         not y[1, 32:64].any() and float(np.abs(y - x).max()) <= float(np.abs(x).max()) / 254 * 1.001 and
         np.array_equal(q8in(x[:, :40])[:, :32], y[:, :32]))
    t = round_q8_rows(x)
    said("the 8-bit rows", float(np.abs(t - x).max()) <= float(np.abs(x).max()) / 254 * 1.01 and not t[1, 32:64].any())

    # the pass on a tiny drafter: 2 layers of E 64, 2 heads, keys of 32 + 8 from a latent of 32, values of 16
    E, H, ql_, kvl, nope, rd, vv, ff, n, pos, P = 64, 2, 32, 32, 32, 8, 16, 96, 3, 11, 6
    m = lambda *s: (rng.standard_normal(s) / np.sqrt(s[-1])).astype(F32)   # noqa: E731
    g1 = lambda k: (1 + 0.1 * rng.standard_normal(k)).astype(F32)          # noqa: E731
    W = {PREFIX + "mtp_fusion.eh_proj.weight": m(E, 2 * E), PREFIX + "mtp_fusion.gate": g1(E),
         PREFIX + "mtp_fusion.enorm.weight": g1(E), PREFIX + "mtp_fusion.hnorm.weight": g1(E),
         PREFIX + "final_norm.weight": g1(E), PREFIX + "hidden_correction.hidden_norm.weight": g1(E),
         PREFIX + "hidden_correction.embed_norm.weight": g1(E), PREFIX + "hidden_correction.gate_proj.weight": m(E, 2 * E),
         PREFIX + "hidden_correction.up_proj.weight": m(E, 2 * E), PREFIX + "hidden_correction.down_proj.weight": m(E, E)}
    for i in range(2):
        p = "%s%d." % (PREFIX, i)
        W.update({p + "input_layernorm.weight": g1(E), p + "post_attention_layernorm.weight": g1(E),
                  p + "self_attn.q_a_proj.weight": m(ql_, E), p + "self_attn.q_a_layernorm.weight": g1(ql_),
                  p + "self_attn.q_b_proj.weight": m(H * (nope + rd), ql_),
                  p + "self_attn.kv_a_proj_with_mqa.weight": m(kvl + rd, E), p + "self_attn.kv_a_layernorm.weight": g1(kvl),
                  p + "self_attn.kv_b_proj.weight": m(H * (nope + vv), kvl),
                  p + "self_attn.linear_gate.weight": m(H * vv, E), p + "self_attn.o_proj.weight": m(E, H * vv),
                  p + "self_attn.learnable_sink_param": rng.standard_normal(H).astype(F32),
                  p + "self_attn.post_proj_norm.weight": g1(E), p + "mlp.gate_proj.weight": m(ff, E),
                  p + "mlp.up_proj.weight": m(ff, E), p + "mlp.down_proj.weight": m(E, ff)})
    w = Dict(W)
    sh = Shape(w)
    said("the sizes from the file", (sh.E, sh.H, sh.q_lora, sh.kv_lora, sh.nope, sh.rope, sh.v, sh.layers) ==
         (E, H, ql_, kvl, nope, rd, vv, 2))
    emb, hl = rng.standard_normal((2, E)).astype(F32), rng.standard_normal(E).astype(F32)
    x0 = fusion(w, emb[0], emb[1], hl, n, False)
    said("the fusion (row 0 alone)", np.array_equal(x0[1:], np.repeat(emb[1:2], n, 0)) and not np.array_equal(x0[0], emb[0]))
    x, top = x0, 0.0
    for i in range(2):
        hist = rng.standard_normal((P, kvl + rd)).astype(np.float16)
        slow = _layer_slow(w, i, x, pos, hist, sh, 1e7)
        x1, own = layer(w, i, x, pos, hist, sh, False, 1e7)
        top = max(top, float(np.abs(x1 - slow).max() / np.abs(slow).max()))
        again, _ = layer(w, i, x, pos, hist, sh, False, 1e7, block_rows=own)
        cut, _ = layer(w, i, x[:2], pos, hist, sh, False, 1e7)   # (causal: a shorter block's rows the same)
        ok8, _ = layer(w, i, x, pos, hist, sh, True, 1e7)
        said("layer %d (the rows it wrote, causal)" % i, np.array_equal(again, x1) and own.shape == (n + 1, kvl + rd) and
             np.allclose(cut, x1[:2], atol=1e-5) and np.isfinite(ok8).all() and
             0 < float(np.abs(ok8 - x1).max()) < 0.2 * float(np.abs(x1).max()))
        x = x1
    said("the layers against the slow pass", top < 2e-5, "  (%.1e of the largest value)" % top)
    h = rms(x, w.get(PREFIX + "final_norm.weight"))
    hc = correct(w, h[1], emb[0], False)
    W["output.weight"] = m(50, E)
    t, pr = head(w, "output.weight", hc, False, chunk=16)
    lg = W["output.weight"].astype(np.float64) @ hc
    said("the correction and the head", hc.shape == (E,) and t == int(np.argmax(lg)) and
         abs(pr - np.exp(lg[t]) / np.exp(lg).sum()) < 1e-5)
    print("self-test: %s" % ("all ok" if ok else "FAILED"))
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="our NumPy port of the drafter's public description: maya-hy4's --dfly2 "
                                             "pass checked against a dump (HY4_DFLY_DUMP)")
    ap.add_argument("--drafter", help="the DFly2 drafter's GGUF")
    ap.add_argument("--dump", help="the engine's dump folder of one draft round")
    ap.add_argument("--model-gguf", nargs="+", help="the model's shards (token_embd.weight and the head: the port's "
                                                    "own tokens, the steps after the first)")
    ap.add_argument("--head", default="output.weight", help="the head's tensor (default output.weight)")
    ap.add_argument("--q8-inputs", action="store_true", help="the engine's GEMV inputs: 8 bits a block of 32")
    ap.add_argument("--block-rows", choices=("own", "dump"), help="the block's own key rows: the port's (rounded to "
                    "float16) or the dump's qrows (default: dump when the dump's q8 is 1, else own)")
    ap.add_argument("--from-dump", action="store_true", help="each stage from the dump's input, not the port's own "
                                                             "(where a difference starts)")
    ap.add_argument("--rope-base", type=float, default=1e7)
    ap.add_argument("--check", action="store_true", help="dequantise every tensor of --drafter: finite, a sane scale")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        return selftest()
    if not a.drafter:
        ap.error("--drafter is needed")
    if a.check:
        return check_file(a.drafter)
    if not a.dump:
        ap.error("--dump is needed")
    return run(a)


if __name__ == "__main__":
    sys.exit(main())

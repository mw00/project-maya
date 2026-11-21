"""tools/ds4/ref_gguf.py - tools/ds4/ref.py's oracle over a GGUF (tools/ds4/pack_gguf.py's pack): the same forward,
each weight dequantised in torch from the GGUF's own blocks (BF16, Q8_0, MXFP4, MXFP8) - a check of maya-ds4's GGUF
path that shares none of its code.

    python tools/ds4/ref_gguf.py --pack <model>/maya-ds4 --ids 0,671,6102 [--dump out.npz] [--gen 8]

The token ids come from the server's tokenizer (maya-ds4 --serve, or the engine's own --ids runs); --dump writes
what tools/ds4/compare.py reads.

--workers N runs the routed low-bit experts (the ones gguf-py dequantises) in N spawned CPU worker processes, each
re-opening the pack: for speed when screening (finding real bugs or large gaps).  Per layer the workers match serial
to float tolerance; end to end the logits are close, not identical, about as close as two serial runs with different
thread counts (on real text mean KL around 2e-3), so differences that small between engines are within the oracle's
own noise either way.  Default 1 is the serial path; 0 means one worker per CPU (affinity mask), up to 8.
"""
from __future__ import annotations

import argparse
import math
import multiprocessing
import os
import pathlib
import struct
import sys
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ref as R  # noqa: E402

BLOCK = {"Q8_0": 34, "MXFP4": 17, "MXFP8": 33}
E2M1 = torch.tensor([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=torch.float32)


class Pack:
    """pack_gguf.py's tensors.txt over the GGUF's files (memory-mapped), and ds4.txt's geometry."""

    def __init__(self, d: pathlib.Path):
        self.kv = {}
        for line in (d / "ds4.txt").read_text().splitlines():
            k, _, v = line.partition(" ")
            self.kv[k] = v
        model = pathlib.Path(self.kv["model_dir"])
        self.mm, self.t = {}, {}
        for line in (d / "tensors.txt").read_text().splitlines():
            f = line.split()
            if f[0] == "file":
                self.mm[int(f[1])] = np.memmap(model / f[2], dtype=np.uint8, mode="r")
            elif f[0] == "t":
                self.t[f[1]] = (f[2], int(f[3]), int(f[4]), int(f[5]), tuple(int(x) for x in f[6:]))
        self._embed = None

    def has(self, name):
        return name in self.t

    def raw(self, name) -> np.ndarray:
        dt, fi, off, nb, _ = self.t[name]
        return self.mm[fi][off:off + nb]

    def num(self, k, d=0.0):
        return float(self.kv.get(k, d))

    def ints(self, k):
        return [int(x) for x in self.kv.get(k, "").split(",") if x]


def dequant(p: Pack, name: str, dev) -> torch.Tensor:
    """A tensor as F32 on dev, shaped as stored ([out][in] for a matrix)."""
    dt, _, _, _, shape = p.t[name]
    r = torch.from_numpy(np.array(p.raw(name))).to(dev)
    if dt == "F32":
        v = r.view(torch.float32)
    elif dt == "F16":
        v = r.view(torch.float16).float()
    elif dt == "BF16":
        v = r.view(torch.bfloat16).float()
    elif dt == "Q8_0":
        b = r.view(-1, 34)
        v = (b[:, :2].contiguous().view(torch.float16).float() * b[:, 2:].view(torch.int8).float()).reshape(-1)
    elif dt == "MXFP4":
        b = r.view(-1, 17)
        s = torch.ldexp(torch.ones(1, device=dev), b[:, :1].int() - 128)          # 2^(e - 128)
        q = b[:, 1:].long()
        lut = E2M1.to(dev)
        v = (torch.cat([lut[q & 15], lut[q >> 4]], 1) * s).reshape(-1)          # values 0..15, then 16..31
    elif dt == "MXFP8":
        b = r.view(-1, 33)
        s = torch.ldexp(torch.ones(1, device=dev), b[:, :1].int() - 127)
        v = (b[:, 1:].contiguous().view(torch.float8_e4m3fn).float() * s).reshape(-1)
    else:   # the low-bit expert types (IQ2_XXS .. Q4_K): gguf-py's numpy dequantisers (ggml's dequantize_row_*)
        v = torch.from_numpy(gq_values(np.array(p.raw(name)), dt, name)).to(dev)
    return v.reshape(shape)


def gq_values(raw: np.ndarray, dt: str, name: str) -> np.ndarray:
    """ggml blocks of type dt as F32 (flat) through gguf-py's dequantisers - the installed package, else a llama.cpp
    checkout's ($GGUF_PY, ~/llama.cpp/gguf-py)."""
    try:
        try:
            import gguf
        except ImportError:
            for d in (os.environ.get("GGUF_PY", ""), os.path.expanduser("~/llama.cpp/gguf-py")):
                if d and os.path.isdir(d):
                    sys.path.insert(0, d)
            import gguf
        from gguf.quants import dequantize as gq_dequant
        qt = gguf.GGMLQuantizationType[dt]
    except (ImportError, KeyError):
        raise SystemExit("%s: dtype %s (no gguf-py with a dequantiser for it: GGUF_PY=<llama.cpp>/gguf-py)" % (name, dt))
    return np.ascontiguousarray(gq_dequant(raw, qt), dtype=np.float32).reshape(-1)


TORCH_DTS = ("F32", "F16", "BF16", "Q8_0", "MXFP4", "MXFP8")   # the types dequant() decodes in torch

_PAR_ST = None
_PAR_TPC = None   # keeps a threadpool_limits context alive for a worker's whole life


def _par_init(pack_dir):
    """Worker initializer (spawn: a fresh interpreter): reopen the pack here and cap this worker at one thread."""
    global _PAR_ST, _PAR_TPC
    try:
        torch.set_num_threads(1)
    except Exception:
        pass
    try:
        import threadpoolctl
        _PAR_TPC = threadpoolctl.threadpool_limits(1)
        _PAR_TPC.__enter__()
    except Exception:
        pass
    _PAR_ST = Pack(pathlib.Path(pack_dir))


_CAPS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
# the caps are process-wide, so they are shared by every live pool (two RefGguf objects with workers > 1, say):
# the first pool's creation saves the old values and sets the caps, the last pool's close() restores them
_CAPS_LOCK = threading.Lock()
_CAPS_LIVE = 0        # pools that currently hold the caps
_CAPS_SAVED = None    # the values saved when the first of them was created


def _caps_acquire():
    """A new pool needs the caps: the first live one saves the old values and sets them to 1."""
    global _CAPS_LIVE, _CAPS_SAVED
    with _CAPS_LOCK:
        if _CAPS_LIVE == 0:
            _CAPS_SAVED = {v: os.environ.get(v) for v in _CAPS}
            for v in _CAPS:
                os.environ[v] = "1"
        _CAPS_LIVE += 1


def _caps_release():
    """A pool is gone: when it was the last live one, put OMP/OPENBLAS/MKL_NUM_THREADS back as they were."""
    global _CAPS_LIVE, _CAPS_SAVED
    with _CAPS_LOCK:
        if _CAPS_LIVE <= 0:
            return
        _CAPS_LIVE -= 1
        if _CAPS_LIVE == 0:
            saved, _CAPS_SAVED = _CAPS_SAVED, None
            for v, old in (saved or {}).items():
                if old is None:
                    os.environ.pop(v, None)
                else:
                    os.environ[v] = old


def _par_expert(job):
    """One routed expert in a worker: (li, e, xs, wrow, lim) -> the expert's float32 output rows.

    Every failure - gguf-py's SystemExit over a dtype it can't dequantise included - becomes a RuntimeError
    naming the expert, so it pickles back and raises in moe() instead of looking like a hang.  A worker's own
    KeyboardInterrupt is re-raised as it is, not wrapped into a task failure."""
    li, e, xs, wrow, lim = job
    P = "layers.%d.ffn.experts.%d." % (li, e)
    try:
        w1 = dequant(_PAR_ST, P + "w1.weight", "cpu").numpy()
        w3 = dequant(_PAR_ST, P + "w3.weight", "cpu").numpy()
        w2 = dequant(_PAR_ST, P + "w2.weight", "cpu").numpy()
        with np.errstate(over="ignore"):   # silu's exp overflows to inf for large negative a; a / inf is the right -0
            a = np.clip(xs @ w1.T, None, lim)
            h = (a / (1.0 + np.exp(-a))) * np.clip(xs @ w3.T, -lim, lim)
        return (h * wrow[:, None]) @ w2.T
    except (Exception, SystemExit) as ex:   # SystemExit included: a worker must report, not exit quietly
        raise RuntimeError("%s: %s" % (P.rstrip("."), ex)) from ex


class Hasher:
    """engram_hash.Hasher from the pack's engram.bin (the token map, multipliers, primes and offsets)."""

    def __init__(self, path: pathlib.Path):
        b = path.read_bytes()
        vocab, nl, ngram, heads, hd, pad = struct.unpack_from("<6i", b, 4)
        cols = (ngram - 1) * heads
        o = 28
        self.tm = np.frombuffer(b, np.int32, vocab, o).astype(np.int64)
        o += 4 * vocab
        self.mult = np.frombuffer(b, np.int64, nl * ngram, o).reshape(nl, ngram)
        o += 8 * nl * ngram
        self.primes = np.frombuffer(b, np.int64, nl * cols, o).reshape(nl, ngram - 1, heads)
        o += 8 * nl * cols
        self.off = np.frombuffer(b, np.int64, nl * cols, o).reshape(nl, cols)
        self.ngram, self.pad, self.hd = ngram, pad, hd

    def ids(self, tokens) -> np.ndarray:   # engram_hash.Hasher.ids
        c = self.tm[np.asarray(tokens, dtype=np.int64)]
        T = len(c)
        pos = np.arange(T)
        toks, blocked = [], np.zeros(T, dtype=bool)
        for s in range(self.ngram):
            src = c[np.clip(pos - s, 0, None)]
            blocked = blocked | (pos < s)
            toks.append(np.where(blocked, self.pad, src))
        toks = np.stack(toks, axis=-1)
        prod = toks[:, None, :] * self.mult[None]
        rolling, out = prod[..., 0], []
        for i in range(1, self.ngram):
            rolling = np.bitwise_xor(rolling, prod[..., i])
            out.append(rolling[..., None] % self.primes[None, :, i - 1])
        return np.concatenate(out, axis=-1) + self.off[None]


def q6k_rows(rows: np.ndarray, name: str) -> np.ndarray:
    """Engram rows of Q6_K superblocks as F32 (ggml's dequantize_row_q6_K, through gguf-py)."""
    return gq_values(rows.reshape(-1), "Q6_K", name)


class RefGguf(R.Ref):
    def __init__(self, pack: pathlib.Path, dev: str = "cuda:0", max_seq: int = 8192, workers: int = 1):
        self.dir = pack
        self.st = p = Pack(pack)
        g = p.num
        t = {k: int(g(k)) for k in ("hidden_size", "hc_mult", "num_hidden_layers", "candidate_source_layer_id",
                                    "qk_rope_head_dim", "num_experts_per_tok", "sliding_window", "o_groups",
                                    "num_attention_heads", "index_topk", "index_n_heads", "index_head_dim", "head_dim",
                                    "candidate_topk_blocks", "candidate_block_size")}
        t.update(hc_eps=g("hc_eps"), hc_sinkhorn_iters=int(g("hc_sinkhorn_iters")),
                 rms_norm_eps=g("rms_norm_eps"), routed_scaling_factor=g("routed_scaling_factor"),
                 swiglu_limit=g("swiglu_limit"), rope_theta=g("rope_theta"), compress_rope_theta=g("compress_rope_theta"),
                 compress_ratios=p.ints("compress_ratios"), kv_source_layer_ids=p.ints("kv_source_layer_ids"),
                 index_source_layer_ids=p.ints("index_source_layer_ids"),
                 rope_scaling={"factor": g("rope_scaling.factor"), "beta_fast": g("rope_scaling.beta_fast"),
                               "beta_slow": g("rope_scaling.beta_slow"),
                               "original_max_position_embeddings": int(g("rope_scaling.original_max_position_embeddings"))})
        self.cfg = {"text_config": t}
        self.t = t
        self.dev = dev
        self.dim, self.hc = t["hidden_size"], t["hc_mult"]
        self.eps = t["rms_norm_eps"]
        self.nl = t["num_hidden_layers"]
        self.ratios = t["compress_ratios"]
        self.kv_src = set(t["kv_source_layer_ids"])
        self.ix_src = set(t["index_source_layer_ids"])
        self.cand_src = t["candidate_source_layer_id"]
        rs, rd = t["rope_scaling"], t["qk_rope_head_dim"]
        self.fc_window = R.freqs_cis(rd, max_seq, 0, t["rope_theta"], rs["factor"], rs["beta_fast"], rs["beta_slow"], dev)
        self.fc_comp = R.freqs_cis(rd, max_seq, rs["original_max_position_embeddings"], t["compress_rope_theta"],
                                   rs["factor"], rs["beta_fast"], rs["beta_slow"], dev)
        self.hasher = Hasher(pack / "engram.bin")
        self.lay = SimpleNamespace(layer_ids=tuple(p.ints("engram_layer_ids")), head_dim=self.hasher.hd)
        self.timing = {}
        self._embed = None
        self._workers = workers
        self._pool = None
        self._caps_held = False   # this object's pool holds a share of the process-wide thread caps

    def W(self, prefix: str) -> torch.Tensor:
        return dequant(self.st, prefix + ".weight", self.dev)

    def V(self, name: str) -> torch.Tensor:
        return dequant(self.st, name, self.dev).float()

    def close(self):
        """Shut down the expert worker pool, if one was started, cancelling the work that has not started yet,
        and release its share of the thread-cap env vars: the last live pool in the process puts them back (no-op
        when no pool was ever started).  The caps are released in a finally: a failing shutdown must not leave
        them set.  A RefGguf is also a context manager, so `with RefGguf(pack) as r:` calls this on any exit,
        including an exception; there a failing close() is reported on stderr and the block's own exception is
        the one that propagates."""
        pool, self._pool = self._pool, None
        held, self._caps_held = self._caps_held, False
        try:
            if pool is not None:
                pool.shutdown(wait=True, cancel_futures=True)
        finally:
            if held:
                _caps_release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.close()      # nothing in flight: a failing close() is the error to report
            return False
        try:
            self.close()
        except BaseException as cex:   # don't mask the block's exception with the shutdown's
            print("ref_gguf: closing the expert pool failed too (%s: %s); raising the original error"
                  % (type(cex).__name__, cex), file=sys.stderr, flush=True)
        return False

    def _pool_new(self):
        """The expert pool, started at the first expert that needs one: spawn (Windows-safe, safe after CUDA
        init), each worker re-opening the pack in its initializer.  The OMP/OPENBLAS/MKL_NUM_THREADS caps are
        set in this process's env and stay set for the pool's whole life - a ProcessPoolExecutor spawns its
        workers lazily, on any submit, and a child reads these vars only as it starts (this process, whose
        numpy and torch loaded long ago, is unaffected).  They are shared by every live pool in the process:
        the first one saves the old values, and the last one's close() restores them.  While any pool is live they
        are set for the whole process, so any other subprocess the caller starts inherits them too: end that with
        `with RefGguf(...)` or close()."""
        _caps_acquire()
        try:
            pool = ProcessPoolExecutor(max_workers=self._workers, mp_context=multiprocessing.get_context("spawn"),
                                       initializer=_par_init, initargs=(str(self.dir),))
            self._caps_held = True   # the caps stay set while the pool lives; close() releases them
            return pool
        except BaseException:        # no pool is running: nothing else will ever release this share
            _caps_release()
            raise

    def engram(self, li, x, hash_ids):
        P = "layers.%d.engram." % li
        T, cols = hash_ids.shape
        hd = self.lay.head_dim
        dt = self.st.t[P + "embed.weight"][0]
        rb = hd // 256 * 210 if dt == "Q6_K" else hd // 32 * 33        # (a row's bytes: Q6_K or MXFP8 blocks)
        tab = self.st.raw(P + "embed.weight").reshape(-1, rb)
        flat = hash_ids.reshape(-1)
        order = np.argsort(flat)
        rows = np.empty((flat.size, rb), np.uint8)
        rows[order] = tab[flat[order]]
        if dt == "Q6_K":
            e = torch.from_numpy(q6k_rows(rows, P)).to(self.dev).view(T, cols * hd)
        else:
            b = torch.from_numpy(rows).to(self.dev).view(-1, 33)
            s = torch.ldexp(torch.ones(1, device=self.dev), b[:, :1].int() - 127)
            e = (b[:, 1:].contiguous().view(torch.float8_e4m3fn).float() * s).view(T, cols * hd)
        e = e.to(torch.bfloat16).float()
        kv = e @ self.W(P + "wkv").T
        d, hc = self.dim, self.hc
        key, value = kv[:, :hc * d].view(T, hc, d), kv[:, hc * d:]
        w = self.V(P + "gate.weight")                                     # q_weight * k_weight (folded)
        h = x.float()
        rstd = torch.rsqrt(h.square().mean(-1) + self.eps) * torch.rsqrt(key.square().mean(-1) + self.eps)
        dot = (h * w * key).sum(-1) * rstd * d ** -0.5
        gate = torch.sigmoid(torch.copysign(dot.abs().clamp_min(1e-6).sqrt(), dot))
        return h + gate.unsqueeze(-1) * value.unsqueeze(1)

    def moe(self, li, x):
        t = self.t
        M = "layers.%d.ffn." % li
        T = x.shape[0]
        gw = self.V(M + "gate.weight")
        scores = F.softplus(x.float() @ gw.T).sqrt()
        idx = (scores + self.V(M + "gate.bias")).topk(t["num_experts_per_tok"], dim=-1)[1]
        wts = scores.gather(1, idx)
        wts = wts / (wts.sum(-1, keepdim=True) + 1e-20) * t["routed_scaling_factor"]
        lim = t["swiglu_limit"]
        y = torch.zeros(T, self.dim, device=self.dev)
        t0 = time.time()
        experts = torch.unique(idx).tolist()
        par, pend = {}, deque()   # par: submitted, not yet applied (expert id -> (rows, future)); pend: to submit
        if self._workers > 1:   # the low-bit experts go to the worker pool; the pool starts at the first one that needs it
            for e in experts:
                P = M + "experts.%d." % e
                if any(self.st.t[P + w + ".weight"][0] not in TORCH_DTS for w in ("w1", "w3", "w2")):
                    pend.append(e)
            if pend and self._pool is None:
                self._pool = self._pool_new()
        used = bool(pend)
        try:
            for e in experts:   # applied in expert id order, whatever order the workers finish in
                while pend and len(par) < 2 * self._workers:   # sliding window: at most 2 * workers in flight
                    e2 = pend.popleft()
                    rows, slot = torch.where(idx == e2)
                    par[e2] = (rows, self._pool.submit(_par_expert, (li, e2, x[rows].float().cpu().numpy(),
                                                                     wts[rows, slot].float().cpu().numpy(), lim)))
                if e in par:    # a dead or failed worker raises here (BrokenProcessPool / RuntimeError), not hangs
                    rows, fut = par.pop(e)
                    y.index_add_(0, rows, torch.from_numpy(fut.result()).to(self.dev))
                else:
                    P = M + "experts.%d." % e
                    rows, slot = torch.where(idx == e)
                    xs = x[rows]
                    h = F.silu((xs @ self.W(P + "w1").T).clamp(max=lim)) * (xs @ self.W(P + "w3").T).clamp(-lim, lim)
                    y.index_add_(0, rows, (h * wts[rows, slot, None]) @ self.W(P + "w2").T)
        except BaseException:   # an error or Ctrl-C must not wait for the rest of this layer: drop its queued work
            for _, fut in par.values():
                fut.cancel()
            raise
        S = M + "shared_experts."
        h = F.silu((x @ self.W(S + "w1").T).clamp(max=lim)) * (x @ self.W(S + "w3").T).clamp(-lim, lim)
        y += h @ self.W(S + "w2").T
        self.last_route = (idx, wts)
        if used:   # only a layer that sent experts to the pool has a parallel time worth reporting
            print("layer %d/%d: %d experts, %.1f s" % (li, self.nl, len(experts), time.time() - t0),
                  file=sys.stderr, flush=True)
        return y

    @torch.inference_mode()
    def forward(self, ids, dump=None):
        # ref.py's forward reads the embedding rows as BF16: a Q8_0 table is dequantised for the rows asked
        dt = self.st.t["embed.weight"][0]
        if dt != "BF16":
            full = self.st.raw("embed.weight").reshape(-1, self.dim // 32 * 34)
            sel = torch.from_numpy(np.array(full[np.asarray(ids)])).to(self.dev).view(-1, 34)
            v = (sel[:, :2].contiguous().view(torch.float16).float() * sel[:, 2:].view(torch.int8).float())
            rows = v.reshape(len(ids), self.dim).to(torch.bfloat16).view(torch.uint16).cpu().numpy()
            raw0 = self.st.raw
            emb = np.zeros((max(ids) + 1, self.dim), np.uint16)
            emb[np.asarray(ids)] = rows
            self.st.raw = lambda n, e=emb, r=raw0: e.view(np.uint8) if n == "embed.weight" else r(n)
            try:
                return super().forward(ids, dump)
            finally:
                self.st.raw = raw0
        return super().forward(ids, dump)


WORKERS_CAP = 8   # --workers 0 means one per CPU, up to this


def cpus() -> int:
    """CPUs this process may use: its affinity mask where the OS has one, else os.cpu_count()."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):   # Windows has no sched_getaffinity
        return os.cpu_count() or 1


def workers_of(v: str) -> int:
    """--workers: 0 means one worker per CPU this process may use (up to WORKERS_CAP); a negative count is
    argparse's error."""
    n = int(v)
    if n < 0:
        raise argparse.ArgumentTypeError("must be >= 0 (0: one worker per CPU, up to %d)" % WORKERS_CAP)
    return min(cpus(), WORKERS_CAP) if n == 0 else n


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="pack_gguf.py's pack (<model>/maya-ds4)")
    ap.add_argument("--ids", required=True, help="comma-separated token ids")
    ap.add_argument("--gen", type=int, default=0, help="greedy tokens to append (a whole forward each)")
    ap.add_argument("--dump", help="write the per-layer residual streams, the logits and the ids (.npz)")
    ap.add_argument("--no-hs", action="store_true", help="--dump without the per-layer residual streams")
    ap.add_argument("--dev", default="cuda:0")
    ap.add_argument("--workers", type=workers_of, default=1,
                    help="dequant and matmul the routed low-bit experts in N CPU worker processes: for speed when "
                         "screening (finding real bugs or large gaps).  Per layer the workers match serial to float "
                         "tolerance; end to end the logits are close, not identical, about as close as two serial "
                         "runs with different thread counts (on real text mean KL around 2e-3), so differences that "
                         "small between engines are within the oracle's own noise either way.  1 (default): today's "
                         "serial path. 0: one worker per CPU this process may use, up to %d" % WORKERS_CAP)
    return ap


def main() -> int:
    a = build_parser().parse_args()
    ids = [int(x) for x in a.ids.split(",")]
    # `with`, not try/finally: if forward() raises and closing the pool fails too, __exit__ reports the close
    # failure on stderr and the original error is the one that propagates
    with RefGguf(pathlib.Path(a.pack).expanduser(), a.dev, workers=a.workers) as r:
        print("tokens:", len(ids), ids[:32], flush=True)
        dump = [] if a.dump and not a.no_hs else None
        t0 = time.time()
        logits = r.forward(ids, dump)
        print("forward %.1fs" % (time.time() - t0), flush=True)
        lp = torch.log_softmax(logits.float(), -1)
        if len(ids) > 1:
            tgt = torch.tensor(ids[1:], device=lp.device)
            nll = -lp[:-1].gather(1, tgt[:, None]).squeeze(1)
            print("ppl over %d tokens: %.4f" % (len(ids) - 1, math.exp(nll.mean().item())))
        top = lp[-1].topk(8)
        print("next:", list(zip(top.indices.tolist(), [round(v, 3) for v in top.values.tolist()])))
        if a.dump:
            np.savez(a.dump, ids=np.array(ids), logits=logits.cpu().numpy(), hs=np.stack(dump) if dump else np.zeros(0))
        out = list(ids)
        for _ in range(a.gen):
            nxt = int(lp[-1].argmax())
            out.append(nxt)
            print("gen ids:", out[len(ids):], flush=True)
            if nxt == 1:
                break
            lp = torch.log_softmax(r.forward(out).float(), -1)
        return 0


if __name__ == "__main__":
    sys.exit(main())

"""tools/test_ds4_ref_gguf.py - moe()'s opt-in CPU worker pool in tools/ds4/ref_gguf.py: parallel == serial
(mixed and all-quantised packs), worker failures raise instead of hanging, lazy pool start, the 2 * workers
in-flight window, a failing close() not masking the original error, two live pools sharing the thread caps,
check_ref_par.py refusing a layer that never used the pool, and the --workers argument.

    python -m unittest tools.test_ds4_ref_gguf

Needs torch and llama.cpp's gguf-py; without either the whole file skips.
"""
from __future__ import annotations

import contextlib
import io
import os
import pathlib
import struct
import sys
import tempfile
import threading
import time
import unittest
from unittest import mock
from concurrent.futures.process import BrokenProcessPool

try:
    import numpy as np
except ImportError:
    np = None

HERE = pathlib.Path(__file__).resolve().parent


def _probe_gguf():
    try:
        import gguf
        return gguf
    except ImportError:
        pass
    for d in (os.environ.get("GGUF_PY", ""), os.environ.get("MAYA_GGUF_PY", "")):
        if d and (pathlib.Path(d) / "gguf").is_dir():
            sys.path.insert(0, d)
    try:
        import gguf
        return gguf
    except ImportError:
        pass
    try:
        sys.path.insert(0, str(HERE))
        from _paths import add_gguf_py
        add_gguf_py()
        import gguf
        return gguf
    except (ImportError, SystemExit):
        return None


GGUF = _probe_gguf()
try:
    import torch
except ImportError:
    torch = None

ref_gguf = None
if np is not None and GGUF is not None and torch is not None:
    sys.path.insert(0, str(HERE / "ds4"))
    try:
        import ref_gguf
    except (ImportError, SystemExit):
        ref_gguf = None

needs = unittest.skipUnless(ref_gguf is not None, "needs numpy, torch and llama.cpp's gguf-py")


@needs
class ExpertPoolTest(unittest.TestCase):
    """moe()'s opt-in worker pool: parallel == serial (all-Q5_0 and mixed F32/Q5_0 packs), a worker failure
    raises instead of hanging, the pool starts lazily (only when an expert needs one), at most 2 * workers
    expert jobs are in flight, --workers parses, and `with r:` closes the pool and hands the thread caps back."""

    def make_pack(self, root: pathlib.Path, expert_dt="Q5_0", bogus_e=None, bias=None, nexp=4, mixed=False):
        """A tiny pack moe() alone can run on: F32 gate/shared, `expert_dt` routed experts (with the
        bogus_e-th one's w1 labelled "BOGUS" and a gate `bias` strong enough to always route to it).
        mixed: even-numbered experts F32, odd ones Q5_0, whatever expert_dt says."""
        rng = np.random.RandomState(1)
        dim, ffn = 64, 32
        tensors = []

        def f32(shape):
            return rng.randn(int(np.prod(shape))).astype("<f4").tobytes()

        def q5(shape):
            n = int(np.prod(shape))
            try:
                arr = rng.randn(n).astype(np.float32)
                return np.ascontiguousarray(GGUF.quants.quantize(arr, GGUF.GGMLQuantizationType.Q5_0)).tobytes()
            except Exception:   # a gguf-py without quantize for this type: valid blocks (d, then random)
                bs, ts = GGUF.GGML_QUANT_SIZES[GGUF.GGMLQuantizationType.Q5_0]
                out = bytearray(n // bs * ts)
                for b in range(n // bs):
                    struct.pack_into("<e", out, b * ts, 0.01)
                    out[b * ts + 2:(b + 1) * ts] = rng.randint(0, 256, ts - 2, dtype=np.uint8).tobytes()
                return bytes(out)

        def add(name, dt, data, shape):
            tensors.append((name, dt, data, shape))

        add("layers.0.ffn.gate.weight", "F32", f32((nexp, dim)), (nexp, dim))
        add("layers.0.ffn.gate.bias", "F32",
            f32((nexp,)) if bias is None else np.asarray(bias, np.float32).astype("<f4").tobytes(), (nexp,))
        for w, shape in (("w1", (ffn, dim)), ("w3", (ffn, dim)), ("w2", (dim, ffn))):
            add("layers.0.ffn.shared_experts.%s.weight" % w, "F32", f32(shape), shape)
            for e in range(nexp):
                dt = ("Q5_0" if e % 2 else "F32") if mixed else expert_dt
                add("layers.0.ffn.experts.%d.%s.weight" % (e, w),
                    "BOGUS" if (e == bogus_e and w == "w1") else dt,
                    q5(shape) if dt == "Q5_0" else f32(shape), shape)

        blob, lines, off = bytearray(), ["file 0 weights.bin"], 0
        for name, dt, data, shape in tensors:
            blob += data
            lines.append("t %s %s 0 %d %d %s" % (name, dt, off, len(data), " ".join(str(x) for x in shape)))
            off += len(data)
        (root / "weights.bin").write_bytes(bytes(blob))
        (root / "tensors.txt").write_text("\n".join(lines) + "\n")
        (root / "ds4.txt").write_text("model_dir %s\n" % root)
        return dim

    def make_ref(self, root: pathlib.Path, dim: int, workers: int):
        r = object.__new__(ref_gguf.RefGguf)   # moe alone: no attention / engram geometry needed
        r.dir = root
        r.st = ref_gguf.Pack(root)
        r.dev = "cpu"
        r.dim = dim
        r.nl = 1
        r.t = {"num_experts_per_tok": 2, "routed_scaling_factor": 2.5, "swiglu_limit": 10.0}
        r._workers, r._pool, r._caps_held = workers, None, False
        return r

    def call_guarded(self, fn, timeout=60):
        """fn() in a daemon thread, its raised exception returned (None if it returned): a regression that hangs
        fails the test after `timeout` s instead of hanging the suite."""
        box = {}

        def run():
            try:
                fn()
            except BaseException as ex:
                box["err"] = ex
        th = threading.Thread(target=run, daemon=True)
        th.start()
        th.join(timeout)
        if th.is_alive():
            self.fail("moe() neither raised nor returned within %d s (a hung worker pool?)" % timeout)
        return box.get("err")

    CAPS = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")

    @contextlib.contextmanager
    def quiet_fd_stderr(self):
        """Quiet fd 2 (not just sys.stderr): a spawn worker writes its traceback to the inherited fd, so a
        failing initializer's logging would otherwise print straight over the test run."""
        saved = os.dup(2)
        try:
            with open(os.devnull, "wb") as dn:
                os.dup2(dn.fileno(), 2)
                yield
        finally:
            os.dup2(saved, 2)
            os.close(saved)

    def test_parallel_matches_serial(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root)          # Q5_0 experts: every routed expert goes to the pool
            r = self.make_ref(root, dim, 1)
            torch.manual_seed(0)
            x = torch.randn(5, dim, dtype=torch.float32)
            before = {v: os.environ.get(v) for v in self.CAPS}
            os.environ["OMP_NUM_THREADS"] = "7"   # a distinctive prior value the pool must give back
            try:
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    y_ser = r.moe(0, x)         # the serial path: no pool, no new stderr lines, no caps set
                self.assertIsNone(r._pool)
                self.assertEqual(err.getvalue(), "")
                self.assertEqual({v: os.environ.get(v) for v in self.CAPS}, {**before, "OMP_NUM_THREADS": "7"})
                r._workers = 2
                with contextlib.redirect_stderr(io.StringIO()):
                    y_par = r.moe(0, x)
                self.assertIsNotNone(r._pool)
                # the caps stay set for the pool's whole life: a later lazy spawn still inherits them
                self.assertEqual({v: os.environ.get(v) for v in self.CAPS}, dict.fromkeys(self.CAPS, "1"))
            finally:
                r.close()
                after = {v: os.environ.get(v) for v in self.CAPS}   # what close() leaves behind
                for v, old in before.items():   # test hygiene: leave no cap behind even if close() failed
                    if old is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = old
            self.assertEqual(after, {**before, "OMP_NUM_THREADS": "7"})   # close() restored the creation-time values
        maxabs = float((y_ser - y_par).abs().max())
        maxrel = maxabs / max(float(y_ser.abs().max()), 1e-30)
        self.assertLess(maxrel, 1e-5, "max abs %.3e, max rel %.3e" % (maxabs, maxrel))

    def test_context_manager_closes(self):
        # `with RefGguf(...) as r:` gives back the same object and closes it on an exceptional exit too: the pool
        # started inside the block is shut down and the thread caps its creation set are back where they were
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root)
            r = self.make_ref(root, dim, 2)
            before = {v: os.environ.get(v) for v in self.CAPS}
            os.environ["OMP_NUM_THREADS"] = "7"   # what the pool's creation must hand back
            torch.manual_seed(0)
            x = torch.randn(5, dim, dtype=torch.float32)
            try:
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        with r as got:
                            self.assertIs(got, r)
                            got.moe(0, x)
                            self.assertIsNotNone(got._pool)   # the pool was started inside the block
                            raise ValueError("out of the with-block")
                except ValueError:
                    pass
                else:
                    self.fail("__exit__ swallowed the block's exception")
                self.assertIsNone(r._pool)
                r.close()   # close() is a no-op the second time
                self.assertEqual({v: os.environ.get(v) for v in self.CAPS}, {**before, "OMP_NUM_THREADS": "7"})
            finally:
                r.close()
                for v, old in before.items():
                    if old is None:
                        os.environ.pop(v, None)
                    else:
                        os.environ[v] = old

    def test_worker_error_raises(self):
        # expert 0's w1 labelled "BOGUS", a dtype gguf-py can't dequantise, with a bias that always routes to
        # it - first of 8 experts, so the other 7 are still queued on the pool when moe() must raise
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root, bogus_e=0, bias=[100.0] + [0.0] * 7, nexp=8)
            r = self.make_ref(root, dim, 2)
            torch.manual_seed(0)
            x = torch.randn(5, dim, dtype=torch.float32)
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    t0 = time.perf_counter()
                    err = self.call_guarded(lambda: r.moe(0, x), timeout=15)
                    t_call = time.perf_counter() - t0
                t0 = time.perf_counter()
                r.close()
                t_close = time.perf_counter() - t0
            finally:
                r.close()
        self.assertIsInstance(err, RuntimeError, "got %r" % (err,))
        self.assertIn("experts.0", str(err))     # names the expert
        self.assertIn("BOGUS", str(err))         # and the worker's original message
        self.assertLess(t_call, 15.0, "moe() waited for the whole layer, not just the failed expert (%.1f s)" % t_call)
        self.assertLess(t_close, 5.0, "close() waited on pending work (%.1f s)" % t_close)

    def test_failing_initializer_raises_broken_pool(self):
        # the pool's initializer re-opens the pack: point dir at one that isn't there, the initializer dies in
        # every worker, and moe() surfaces BrokenProcessPool (at the first expert) instead of hanging
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root)
            r = self.make_ref(root, dim, 2)
            r.dir = root / "no-such-pack"        # moe() reads r.st; only the initializer opens dir
            torch.manual_seed(0)
            x = torch.randn(5, dim, dtype=torch.float32)
            with self.quiet_fd_stderr():         # through close(): late workers die while it shuts down
                try:
                    with contextlib.redirect_stderr(io.StringIO()):
                        err = self.call_guarded(lambda: r.moe(0, x), timeout=60)
                finally:
                    r.close()
        self.assertIsInstance(err, BrokenProcessPool, "got %r" % (err,))

    def test_lazy_pool_torch_types_only(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root, expert_dt="F32")   # every expert decodes in torch: nothing needs a worker
            r = self.make_ref(root, dim, 2)
            torch.manual_seed(0)
            x = torch.randn(5, dim, dtype=torch.float32)
            try:
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    y = r.moe(0, x)
                self.assertIsNone(r._pool)       # the pool was never started
                self.assertEqual(err.getvalue(), "")   # nor did the layer print a parallel timing line
            finally:
                r.close()
        self.assertEqual(tuple(y.shape), (5, dim))
        self.assertTrue(torch.isfinite(y).all())

    def test_mixed_pack_parallel_matches_serial(self):
        # even experts F32 (serial in both runs), odd ones Q5_0 (the pool when parallel): the layer still matches.
        # the gate bias forces expert 1, a Q5_0 one, into every token's route: the pool is used, not just available
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root, mixed=True, bias=[0.0, 100.0, 0.0, 0.0])
            r = self.make_ref(root, dim, 1)
            torch.manual_seed(0)
            x = torch.randn(5, dim, dtype=torch.float32)
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    y_ser = r.moe(0, x)
                self.assertIsNone(r._pool)
                r._workers = 2
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    y_par = r.moe(0, x)
                self.assertTrue(bool((r.last_route[0] == 1).any(-1).all()))   # expert 1 (Q5_0) in every token's route
                self.assertIsNotNone(r._pool)    # the Q5_0 experts did start the pool
                self.assertIn("layer 0/1:", err.getvalue())   # and the layer that used it printed its timing line
            finally:
                r.close()
        maxabs = float((y_ser - y_par).abs().max())
        maxrel = maxabs / max(float(y_ser.abs().max()), 1e-30)
        self.assertLess(maxrel, 1e-5, "max abs %.3e, max rel %.3e" % (maxabs, maxrel))

    def test_inflight_window_bounded(self):
        # at most 2 * workers expert jobs between submit and result(): wrap the pool's submit, counting the
        # futures it handed out that moe() has not collected yet.  The routing is forced, not seed-driven: 6 experts
        # per token and a gate bias that puts experts 0-5 (all Q5_0) in every token's route, so exactly 6 jobs go
        # to a pool whose window is 4
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root, nexp=8, bias=[100.0] * 6 + [0.0] * 2)
            r = self.make_ref(root, dim, 2)
            r.t["num_experts_per_tok"] = 6
            torch.manual_seed(0)
            x = torch.randn(16, dim, dtype=torch.float32)
            st = {"out": 0, "max": 0, "n": 0}

            class CountingFuture:
                def __init__(self, fut):
                    self.f = fut

                def result(self, timeout=None):
                    v = self.f.result(timeout)
                    st["out"] -= 1
                    return v

                def cancel(self):
                    return self.f.cancel()

            class CountingPool:
                def __init__(self, pool):
                    self.pool = pool

                def submit(self, fn, *args):
                    fut = self.pool.submit(fn, *args)
                    st["out"] += 1
                    st["n"] += 1
                    st["max"] = max(st["max"], st["out"])
                    return CountingFuture(fut)

                def shutdown(self, *args, **kw):
                    self.pool.shutdown(*args, **kw)

            real_new = r._pool_new
            r._pool_new = lambda: CountingPool(real_new())
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    y = r.moe(0, x)
                self.assertIsNotNone(r._pool)
                self.assertEqual(sorted(torch.unique(r.last_route[0]).tolist()), list(range(6)))   # the forced route
                self.assertEqual(st["n"], 6)
                self.assertGreater(st["n"], 2 * r._workers,
                                   "only %d experts routed: too few to show a window of %d" % (st["n"], 2 * r._workers))
                self.assertLessEqual(st["max"], 2 * r._workers,
                                     "%d expert jobs in flight at once (window %d)" % (st["max"], 2 * r._workers))
            finally:
                r.close()
        self.assertEqual(tuple(y.shape), (16, dim))

    def restore_caps(self, before):
        for v, old in before.items():
            if old is None:
                os.environ.pop(v, None)
            else:
                os.environ[v] = old

    def test_close_failure_keeps_original_error(self):
        # the pool's shutdown raises while the with-block's own exception is in flight: that exception (ValueError)
        # is the one that arrives, the close failure is reported on stderr, and the caps are still released
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root)
            r = self.make_ref(root, dim, 2)
            before = {v: os.environ.get(v) for v in self.CAPS}
            os.environ["OMP_NUM_THREADS"] = "7"
            torch.manual_seed(0)
            x = torch.randn(5, dim, dtype=torch.float32)
            try:
                err = io.StringIO()
                with contextlib.redirect_stderr(err), self.assertRaises(ValueError):
                    with r:
                        r.moe(0, x)
                        real = r._pool

                        class FailingShutdown:
                            def shutdown(self, *a, **kw):
                                real.shutdown(*a, **kw)
                                raise OSError("shutdown failed")
                        r._pool = FailingShutdown()
                        raise ValueError("the real error")
                self.assertIn("closing the expert pool failed too", err.getvalue())
                self.assertIn("shutdown failed", err.getvalue())
                self.assertIsNone(r._pool)
                self.assertEqual({v: os.environ.get(v) for v in self.CAPS}, {**before, "OMP_NUM_THREADS": "7"})

                # with nothing in flight, the close failure itself is the error
                class Boom:
                    def shutdown(self, *a, **kw):
                        raise OSError("shutdown failed")
                r._pool = Boom()
                with self.assertRaises(OSError):
                    with r:
                        pass
            finally:
                r.close()
                self.restore_caps(before)

    def test_main_close_failure_keeps_original_error(self):
        # main() itself: forward() raises, then close() raises; the forward() error is what main() raises
        class Stub(ref_gguf.RefGguf):
            def __init__(self, *a, **kw):
                pass

            def forward(self, ids, dump=None):
                raise ValueError("forward failed")

            def close(self):
                raise OSError("shutdown failed")

        err = io.StringIO()
        with mock.patch.object(ref_gguf, "RefGguf", Stub), \
                mock.patch.object(sys, "argv", ["ref_gguf.py", "--pack", "x", "--ids", "0,1", "--dev", "cpu"]), \
                contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(err):
            with self.assertRaises(ValueError) as cm:
                ref_gguf.main()
        self.assertIn("forward failed", str(cm.exception))
        self.assertIn("shutdown failed", err.getvalue())

    def test_two_pools_share_caps(self):
        # two live RefGguf pools: closing the first keeps the caps for the second; the last close restores them
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root)
            r1 = self.make_ref(root, dim, 2)
            r2 = self.make_ref(root, dim, 2)
            before = {v: os.environ.get(v) for v in self.CAPS}
            os.environ["OMP_NUM_THREADS"] = "7"
            torch.manual_seed(0)
            x = torch.randn(5, dim, dtype=torch.float32)
            ones = dict.fromkeys(self.CAPS, "1")
            try:
                with contextlib.redirect_stderr(io.StringIO()):
                    r1.moe(0, x)
                    r2.moe(0, x)
                self.assertIsNotNone(r1._pool)
                self.assertIsNotNone(r2._pool)
                self.assertEqual({v: os.environ.get(v) for v in self.CAPS}, ones)
                r1.close()
                self.assertEqual({v: os.environ.get(v) for v in self.CAPS}, ones)   # r2's pool still needs them
                r1.close()   # a second close() must not release r2's share
                self.assertEqual({v: os.environ.get(v) for v in self.CAPS}, ones)
                with contextlib.redirect_stderr(io.StringIO()):
                    r2.moe(0, x)   # r2's pool still works
                r2.close()
                self.assertEqual({v: os.environ.get(v) for v in self.CAPS}, {**before, "OMP_NUM_THREADS": "7"})
            finally:
                r1.close()
                r2.close()
                self.restore_caps(before)

    def run_check_ref_par(self, root, dim):
        import check_ref_par
        r = self.make_ref(root, dim, 1)
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(check_ref_par.RG, "RefGguf", lambda pack, dev: r), \
                mock.patch.object(sys, "argv", ["check_ref_par.py", "--pack", str(root), "--layer", "0",
                                                "--tokens", "5", "--workers", "2"]), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = check_ref_par.main()
        self.assertIsNone(r._pool)   # closed on the way out
        return rc, out.getvalue(), err.getvalue()

    def test_check_ref_par_pool_not_exercised(self):
        # a layer whose experts all decode in torch never starts the pool: a warning and a non-zero exit, not a PASS
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root, expert_dt="F32")
            rc, out, err = self.run_check_ref_par(root, dim)
        self.assertNotEqual(rc, 0)
        self.assertNotIn("PASS", out)
        self.assertIn("layer 0 has no low-bit routed experts: the pool was not exercised; pick another --layer", err)

    def test_check_ref_par_pass(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            dim = self.make_pack(root)   # Q5_0 experts: the pool is used
            rc, out, err = self.run_check_ref_par(root, dim)
        self.assertEqual(rc, 0, out + err)
        self.assertIn("PASS", out)

    def test_workers_arg(self):
        p = ref_gguf.build_parser()
        base = ["--pack", "x", "--ids", "0"]
        self.assertEqual(ref_gguf.WORKERS_CAP, 8)
        self.assertEqual(p.parse_args(base).workers, 1)
        self.assertEqual(p.parse_args(base + ["--workers", "3"]).workers, 3)
        try:
            ncpu = len(os.sched_getaffinity(0))   # --workers 0: the affinity mask, not the raw cpu_count
        except (AttributeError, OSError):
            ncpu = os.cpu_count() or 1
        self.assertEqual(p.parse_args(base + ["--workers", "0"]).workers, min(ncpu, 8))
        err = io.StringIO()
        with contextlib.redirect_stderr(err), self.assertRaises(SystemExit):
            p.parse_args(base + ["--workers", "-1"])
        self.assertIn("--workers", err.getvalue())


if __name__ == "__main__":
    unittest.main()

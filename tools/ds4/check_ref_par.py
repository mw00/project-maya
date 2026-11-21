"""tools/ds4/check_ref_par.py - ref_gguf.moe's parallel routed-expert path vs the serial one, CPU only.

    GGUF_PY=<llama.cpp>/gguf-py python tools/ds4/check_ref_par.py --pack <model>/maya-ds4 \
        [--layer 3] [--workers N (0: one per CPU, up to the cap; 1, or 0 on a 1-CPU box, runs as 2)] [--tokens 128]

Runs layer L's moe on a random --tokens-row x three times - parallel (cold: the pool starts in it, its pages come
off the disk), serial, then parallel again (warm page cache) - and prints the three wall times plus the max abs /
max rel difference of y (serial vs the WARM parallel, so no page-cache bias): PASS when the rel diff is below 1e-5.
A layer whose routed experts all decode in torch never starts the pool: that is a warning and exit 2, not a PASS.
"""
import argparse
import pathlib
import sys
import time

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import ref_gguf as RG  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", required=True, help="pack_gguf.py's pack (<model>/maya-ds4)")
    ap.add_argument("--layer", type=int, default=3)
    ap.add_argument("--tokens", type=int, default=128, help="rows of random x (tokens)")
    ap.add_argument("--workers", default=None, metavar="N",
                    help="expert worker processes; 0: one per CPU this process may use, up to %d; the check needs 2, "
                         "so 1 - and 0 on a 1-CPU box - runs as 2" % RG.WORKERS_CAP)
    a = ap.parse_args()
    if a.workers is None:
        w, asked_for_one = min(RG.WORKERS_CAP, RG.cpus()), False
    else:
        try:
            w = RG.workers_of(a.workers)
        except (argparse.ArgumentTypeError, ValueError) as ex:
            ap.error("--workers: %s" % ex)
        asked_for_one = a.workers.strip() == "1"
    # a 1-worker "parallel" is just the serial path again: use 2, and say which 1 the user asked for
    if w > 1:
        nw, why = w, ""
    elif asked_for_one:
        nw, why = 2, ", --workers 1: using 2 instead"
    else:
        nw, why = 2, ", only 1 CPU: using 2 workers to exercise the pool"
    with RG.RefGguf(pathlib.Path(a.pack).expanduser(), dev="cpu") as r:
        torch.manual_seed(0)
        x = torch.randn(a.tokens, r.dim, dtype=torch.float32)
        r._workers = nw            # parallel first, so the serial run and the second parallel one read a warm cache
        t0 = time.perf_counter()
        r.moe(a.layer, x)          # cold: the pool starts in it, and its pages come off the disk
        t_cold = time.perf_counter() - t0
        if r._pool is None:        # every routed expert decoded in torch: "parallel" was the serial path again
            print("layer %d has no low-bit routed experts: the pool was not exercised; pick another --layer"
                  % a.layer, file=sys.stderr, flush=True)
            print("NOT EXERCISED")
            return 2
        r._workers = 1
        t0 = time.perf_counter()
        y_ser = r.moe(a.layer, x)
        t_ser = time.perf_counter() - t0
        r._workers = nw
        t0 = time.perf_counter()
        y_par = r.moe(a.layer, x)
        t_par = time.perf_counter() - t0
    d = (y_ser - y_par).abs()      # serial vs the WARM parallel: the cold run carries the pool start
    maxabs = float(d.max())
    maxrel = maxabs / max(float(y_ser.abs().max()), 1e-30)
    print("layer %d: max abs diff %.3e, max rel diff %.3e (serial vs warm parallel)" % (a.layer, maxabs, maxrel))
    print("layer %d: parallel cold (%d workers%s) %.1f s, serial %.1f s, parallel warm %.1f s"
          % (a.layer, nw, why, t_cold, t_ser, t_par))
    ok = maxrel < 1e-5
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

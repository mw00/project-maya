"""tools/mm3/compare.py - maya-mm3's logits (--dump-logits) against tools/mm3/ref.py's (--dump), position by position.

    python tools/mm3/compare.py engine.bin ref.npz

Per position both share: the cosine of the logits, the largest difference, whether the top token is the same, and
KL(ref || engine) of the softmax.  The ids of both must agree.
"""
from __future__ import annotations

import struct
import sys

import numpy as np


def read_engine(path: str):
    with open(path, "rb") as f:
        if f.read(4) != b"MM3L":
            raise SystemExit(f"{path}: not a maya-mm3 logits dump")
        rows, vocab, n = struct.unpack("<3i", f.read(12))
        ids = np.frombuffer(f.read(4 * n), dtype="<i4")
        pos = np.frombuffer(f.read(4 * rows), dtype="<i4")
        lg = np.frombuffer(f.read(4 * rows * vocab), dtype="<f4").reshape(rows, vocab)
    return ids, pos, lg


def log_softmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    m = x.max(-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(-1, keepdims=True))


def main() -> int:
    ids_e, pos_e, lg_e = read_engine(sys.argv[1])
    r = np.load(sys.argv[2])
    ids_r, pos_r, lg_r = r["ids"], r["pos"], r["logits"]
    n = min(len(ids_e), len(ids_r))
    if not np.array_equal(ids_e[:n], ids_r[:n]):
        raise SystemExit("the ids differ")
    common = sorted(set(pos_e.tolist()) & set(pos_r.tolist()))
    if not common:
        raise SystemExit("no position in both dumps")
    same, kls = 0, []
    for p in common:
        a = lg_r[list(pos_r).index(p)].astype(np.float64)
        b = lg_e[list(pos_e).index(p)].astype(np.float64)
        cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))
        la, lb = log_softmax(a), log_softmax(b)
        kl = float((np.exp(la) * (la - lb)).sum())
        ta, tb = int(a.argmax()), int(b.argmax())
        same += ta == tb
        kls.append(kl)
        print(f"pos {p:6d}: cos {cos:.6f}  max|d| {np.abs(a - b).max():8.4f}  top ref {ta:6d} eng {tb:6d}"
              f"{'' if ta == tb else '  DIFF'}  KL {kl:.5f}")
    print(f"{len(common)} positions: top-1 same {same}/{len(common)}, mean KL {np.mean(kls):.5f}, "
          f"max KL {np.max(kls):.5f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

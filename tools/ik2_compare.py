"""tools/ik2_compare.py - maya-ik2's dump (--dump DIR: blk.L.out.f32, logprobs.f32) against ref/inkling2.py's seams
(--dump-dir: blk.L.out.npy, logprobs.npy), the same tokens through both.

    python tools/ik2_compare.py ENGINE_DUMP_DIR REF_SEAMS_DIR [--width 4608]
Prints each layer's relative error of its output (|a - b| / |b| over the whole [T][C]) and, at the end, the
log-probabilities' agreement: top-1, max |dlogp| of the reference's top token, KL(ref || engine) mean and max.
"""
import argparse
import pathlib

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("engine")
    ap.add_argument("ref")
    ap.add_argument("--width", type=int, default=4608)
    a = ap.parse_args()
    E, R = pathlib.Path(a.engine), pathlib.Path(a.ref)
    L = 0
    while (R / f"blk.{L}.out.npy").exists():
        r = np.load(R / f"blk.{L}.out.npy").astype(np.float64)
        f = E / f"blk.{L}.out.f32"
        if not f.exists():
            print(f"layer {L}: no engine dump")
            break
        e = np.fromfile(f, np.float32).astype(np.float64).reshape(r.shape)
        rel = np.linalg.norm(e - r) / np.linalg.norm(r)
        cos = (e * r).sum(-1) / (np.linalg.norm(e, axis=-1) * np.linalg.norm(r, axis=-1))
        extra = ""
        fa, ra = E / f"blk.{L}.attn_out.f32", R / f"blk.{L}.attn_out.npy"
        if fa.exists() and ra.exists():
            rr = np.load(ra).astype(np.float64)
            ea = np.fromfile(fa, np.float32).astype(np.float64).reshape(rr.shape)
            extra += f", attn_out rel err {np.linalg.norm(ea - rr) / np.linalg.norm(rr):.2e}"
        fp, rp = E / f"blk.{L}.picks.i32", R / f"blk.{L}.picks.npy"
        if fp.exists() and rp.exists():
            rpk = np.load(rp).astype(np.int64)
            epk = np.fromfile(fp, np.int32).astype(np.int64).reshape(rpk.shape)
            same = np.mean([len(set(a) & set(b)) for a, b in zip(rpk, epk)])
            extra += f", experts same {same:.2f}/{rpk.shape[1]} ({np.sum([set(a) != set(b) for a, b in zip(rpk, epk)])} tokens differ)"
        print(f"layer {L:2d}: rel err {rel:.2e}, worst token cos {cos.min():.6f}{extra}")
        L += 1
    r = np.load(R / "logprobs.npy").astype(np.float64)
    e = np.fromfile(E / "logprobs.f32", np.float32).astype(np.float64).reshape(r.shape)
    top_r, top_e = r.argmax(-1), e.argmax(-1)
    pr = np.exp(r)
    kl = (pr * (r - e)).sum(-1)
    d = np.abs(np.take_along_axis(e, top_r[:, None], -1) - np.take_along_axis(r, top_r[:, None], -1))
    print(f"logprobs: {r.shape[0]} positions, top-1 agree {np.mean(top_r == top_e) * 100:.1f}%, "
          f"max |dlogp| of the reference's top {d.max():.3e}, KL mean {kl.mean():.3e} max {kl.max():.3e}")


if __name__ == "__main__":
    main()

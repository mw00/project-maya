#!/usr/bin/env python3
"""tools/kimi_k31_ref_pin.py - pins ref/kimi_k31.py's arithmetic on the synthetic fixture, bit for bit.

tools/kimi_k31_gate.py holds the oracle to Moonshot's code within tolerances; this pins its exact bits, so
a change meant to leave the oracle alone (lazy reads, a truncated model, a memory-bounded indexer, a new
option at its default) can be shown to change nothing. data/kimi-k31-ref-pin.json holds, per run, a
sha256 (16 hex digits) of every seam and the logits:

  * float32 and float64, a prefill of the fixture's 32 suggested tokens, then two decode steps on its
    state (the fixed tokens below), on tools/kimi_k31_synth_gguf.py's fixture (seed 3101);
  * "probe": a hash of a few float32 / float64 products, einsums, reductions and transcendentals of
    the kinds the forward uses. Bits depend on numpy's SIMD dispatch and the BLAS kernels, so the pin
    is checked only where the probe matches the machine that made it; elsewhere the golden gate covers
    the oracle.

    python tools/kimi_k31_ref_pin.py --check                 # this checkout against the committed pin
    python tools/kimi_k31_ref_pin.py --oracle DIR --write    # (re)make the pin from DIR/kimi_k31.py
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import pathlib
import sys
import tempfile

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parent
PIN = REPO / "data" / "kimi-k31-ref-pin.json"
DECODE = (3, 7)          # decode steps feed the fixture tokens at these indexes, after the prefill


def platform_probe() -> str:
    """A hash of the rounding of the operation kinds the forward is made of, on fixed inputs."""
    rng = np.random.default_rng(20261105)
    a = rng.standard_normal((160, 256))
    b = rng.standard_normal((256, 32))
    v = rng.standard_normal((32, 6, 9))
    parts = []
    for dt in (np.float32, np.float64):
        A, B = a.astype(dt), b.astype(dt)
        parts += [A @ B, A @ B[:, :1], A @ B[:, :5], A.T[:, :32] @ B[:32].T[:, :16]]
        parts += [np.exp(-A), np.tanh(A / 4), np.sqrt(np.abs(A) + 1e-5), 1.0 / (1.0 + np.exp(-A)),
                  np.log1p(np.abs(A)), A.sum(axis=0), A.sum(axis=1), (A ** 2).mean(axis=0)]
    parts += [np.einsum("d,dnt->nt", v[:, 0, 0], v), np.einsum("dnt,nt->dt", v, v[0]),
              np.einsum("dht,ds->sht", v, a[:32, :40]), np.einsum("hvc,ch->hv", v.transpose(1, 2, 0), v[:, :, 0])]
    h = hashlib.sha256()
    for p in parts:
        h.update(np.ascontiguousarray(p).tobytes())
    return h.hexdigest()[:16]


def digest(a: np.ndarray) -> str:
    a = np.ascontiguousarray(a)
    h = hashlib.sha256(f"{a.dtype.str}{a.shape}".encode())
    h.update(a.tobytes())
    return h.hexdigest()[:16]


def pin_runs(K, gguf: pathlib.Path, tokens: list[int]) -> dict[str, dict[str, str]]:
    """{run: {seam: digest}} of oracle module K (ref/kimi_k31.py or an older copy) on `gguf`."""
    runs = {}
    for dt in ("float32", "float64"):
        m = K.KimiK31(gguf, dtype=dt)
        st = m.new_state()
        for step, toks in [("prefill", tokens)] + [(f"decode{i}", [tokens[j]]) for i, j in enumerate(DECODE)]:
            logits = m.forward(toks, st)
            d = {k: digest(v) for k, v in m.dumps.items()}
            d["logits"] = digest(logits)
            runs[f"{dt}/{step}"] = d
        m.close()
    return runs


def fixture(d: pathlib.Path) -> tuple[pathlib.Path, list[int]]:
    sys.path.insert(0, str(HERE))
    import kimi_k31_synth_gguf as synth
    vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
    p = d / "tiny.gguf"
    synth.write_gguf(p, *synth.to_gguf(vals, synth.TINY))
    return p, synth.sample_tokens(synth.TINY, synth.SEED)


def load_oracle(ref_dir: pathlib.Path):
    sys.path.insert(0, str(ref_dir))                    # its own ref/load.py
    spec = importlib.util.spec_from_file_location("kimi_k31_pinned", ref_dir / "kimi_k31.py")
    K = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(K)
    return K


def compare(got: dict, want: dict) -> list[str]:
    """Differences, one line each (missing runs or seams, extra seams, other bits)."""
    out = []
    for run in sorted(set(want) | set(got)):
        g, w = got.get(run), want.get(run)
        if g is None or w is None:
            out.append(f"{run}: {'missing' if g is None else 'not in the pin'}")
            continue
        out += [f"{run} {k}: {'missing' if k not in g else 'extra' if k not in w else 'differs'}"
                for k in sorted(set(g) | set(w)) if g.get(k) != w.get(k)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--oracle", default=str(REPO / "ref"), help="directory holding kimi_k31.py and load.py")
    ap.add_argument("--pin", default=str(PIN))
    act = ap.add_mutually_exclusive_group(required=True)
    act.add_argument("--write", action="store_true")
    act.add_argument("--check", action="store_true")
    args = ap.parse_args()
    K = load_oracle(pathlib.Path(args.oracle).resolve())
    with tempfile.TemporaryDirectory(prefix="kimi-k31-pin-") as d:
        gguf, tokens = fixture(pathlib.Path(d))
        runs = pin_runs(K, gguf, tokens)
    pin = pathlib.Path(args.pin)
    if args.write:
        pin.write_text(json.dumps({"probe": platform_probe(), "numpy": np.__version__, "runs": runs},
                                  indent=1, sort_keys=True) + "\n")
        print(f"{pin}: {sum(len(r) for r in runs.values())} digests in {len(runs)} runs")
        return 0
    want = json.loads(pin.read_text())
    if want["probe"] != platform_probe():
        print("pin: this platform rounds differently from the one the pin was made on; not checked")
        return 0
    diff = compare(runs, want["runs"])
    for line in diff[:40]:
        print(line)
    print(f"pin: {'OK' if not diff else f'{len(diff)} differences'}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())

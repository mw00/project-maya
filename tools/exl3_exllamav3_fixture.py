"""tools/exl3_exllamav3_fixture.py - EXL3 decode fixtures from exllamav3 itself, for the engine's EXL3 tests.

    <exllamav3 venv>/bin/python -I tools/exl3_exllamav3_fixture.py --out data/exl3-exllamav3
    <exllamav3 venv>/bin/python -I tools/exl3_exllamav3_fixture.py --model <EXL3 model dir> --out <dir> \\
        --tensor model.language_model.layers.3.mlp.experts.0.gate_proj lm_head ...

Runs in an exllamav3 environment (torch + the exllamav3_ext extension, exllamav3 1.6.2 here) on a CUDA device - its
packers and its reconstruct are CUDA kernels (CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=<n> picks one).  The
engine's decoders (the CPU reference, the device kernels, the CPU lane, tools/exl3_pack.py's numpy decoder) are then
checked against what exllamav3 writes, bit for bit, by tests/exl3_test.cu (--exllamav3, --expect) and
tools/test_exl3_exllamav3.py - not only against each other.

Random tiles (default): for each bitrate and codebook, random 16-bit trellis states packed by exllamav3's own packer
(pack_trellis), then its unpacker's windows (unpack_trellis: the state of each of a tile's 256 steps) and its
reconstruct's W' (the decoded matrix in the rotated basis, fp16).  exllamav3's pack(unpack(words)) == words is checked
first.  The bitrate is given as k2 = 2 K, exllamav3's own unit; codebooks 0 3inst, 1 mcg, 2 mul1 (exl3.hpp).  Each
<out>/exl3_k2_<k2>_cb<cb>.bin is
    "EXH1", int32 k2, ki, nj, cb, words (16 K uint16 a tile)
    int16   trellis [ki][nj][words]       the tiles as a model file stores them (exllamav3's packed int16)
    uint16  states  [ki][nj][256]         each step's 16-bit state
    fp16    w_hat   [16 ki][16 nj]        W' row-major

Model tensors (--model): each named EXL3 matrix (the safetensors prefix of its .trellis / .suh / .svh) as stored, with
exllamav3's decode of it: <base>.trellis.i16 (the stored int16 [k/16][n/16][words]), <base>.suh.f16, <base>.svh.f16,
<base>.what.f16 (reconstruct: W' [k][n]) and <base>.w.f16 (reconstruct_had_slice: W = diag(suh) H W' H diag(svh) in
the original basis [k][n], rounded by exllamav3's fused kernel), all raw little-endian, and a line
"<base> <k2> <cb> <k> <n>" appended to <out>/manifest.txt - for exl3_test --expect <out>.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np
import torch                      # before the extension: it links against torch's libraries
import exllamav3_ext as ext       # noqa: E402

MAGIC = b"EXH1"
CODEBOOKS = {0: "3inst", 1: "mcg", 2: "mul1"}
DEFAULT_SETS = [(2 * K, 2) for K in range(1, 9)] + [(6, 0), (6, 1)]   # K 1..8 mul1; K 3 3inst and mcg


def bitrate_ok(k2: int, cb: int) -> bool:
    """The bitrates exllamav3 1.6 writes: K 1..8 with any codebook (k2 even)."""
    return 2 <= k2 <= 16 and k2 % 2 == 0 and cb in CODEBOOKS


def pack(idx: torch.Tensor, k2: int) -> torch.Tensor:
    ki, nj, _ = idx.shape
    packed = torch.zeros((ki, nj, 8 * k2), dtype=torch.int16, device=idx.device)
    ext.pack_trellis(packed, idx.contiguous(), k2 // 2)
    return packed


def unpack(packed: torch.Tensor, k2: int) -> torch.Tensor:
    ki, nj, _ = packed.shape
    st = torch.zeros((ki, nj, 256), dtype=torch.int16, device=packed.device)
    ext.unpack_trellis(st, packed.contiguous(), k2 // 2)
    return st


def w_hat(packed: torch.Tensor, k2: int, cb: int) -> torch.Tensor:
    ki, nj, _ = packed.shape
    w = torch.zeros((ki * 16, nj * 16), dtype=torch.half, device=packed.device)
    ext.reconstruct(w, packed.contiguous(), k2 / 2, cb == 1, cb == 2)
    return w


def exllamav3_version() -> str:
    try:
        from importlib.metadata import version
        return version("exllamav3")
    except Exception:  # noqa: BLE001 - a label only
        return "?"


def random_tiles(out: pathlib.Path, sets: list[tuple[int, int]], seed: int, ki: int, nj: int) -> None:
    dev = torch.device("cuda")
    if nj % 8:
        raise SystemExit("--nj %d: exllamav3's reconstruct takes n a multiple of 128 (nj a multiple of 8)" % nj)
    for k2, cb in sets:
        if not bitrate_ok(k2, cb):
            raise SystemExit("k2 %d cb %d: not a bitrate / codebook exllamav3 writes" % (k2, cb))
        rng = np.random.default_rng(seed + 1000 * k2 + 100 * cb)
        idx = torch.from_numpy(rng.integers(-(1 << 15), 1 << 15, size=(ki, nj, 256), dtype=np.int16)).to(dev)
        packed = pack(idx, k2)
        states = unpack(packed, k2)
        if not torch.equal(pack(states, k2), packed):   # exllamav3's own consistency
            raise SystemExit("k2 %d: exllamav3's pack(unpack(words)) != words" % k2)
        wh = w_hat(packed, k2, cb)
        torch.cuda.synchronize()
        path = out / ("exl3_k2_%d_cb%d.bin" % (k2, cb))
        with open(path, "wb") as f:
            f.write(MAGIC)
            f.write(struct.pack("<iiiii", k2, ki, nj, cb, 8 * k2))
            f.write(packed.cpu().numpy().astype("<i2").tobytes())
            f.write(states.cpu().numpy().astype("<i2").tobytes())
            f.write(wh.cpu().numpy().astype("<f2").tobytes())
        print("wrote %s (%d bytes): K %g %s, %d x %d tiles, %d words a tile, exllamav3 %s" %
              (path, path.stat().st_size, k2 / 2, CODEBOOKS[cb], ki, nj, 8 * k2, exllamav3_version()))


def model_tensors(out: pathlib.Path, model: pathlib.Path, bases: list[str]) -> None:
    from safetensors import safe_open
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    dev = torch.device("cuda")
    for base in bases:
        if base + ".trellis" not in index:
            raise SystemExit("%s: no %s.trellis in %s" % (base, base, model / "model.safetensors.index.json"))
        with safe_open(str(model / index[base + ".trellis"]), framework="pt", device="cpu") as f:
            tr = f.get_tensor(base + ".trellis")
        parts = {}
        for x in ("suh", "svh"):
            with safe_open(str(model / index[base + "." + x]), framework="pt", device="cpu") as f:
                parts[x] = f.get_tensor(base + "." + x)
        cb = 2 if base + ".mul1" in index else 1 if base + ".mcg" in index else 0
        words = int(tr.shape[2])
        if tr.dtype != torch.int16 or words % 8:
            raise SystemExit("%s: trellis %s %s is not an EXL3 trellis" % (base, tr.dtype, tuple(tr.shape)))
        k2 = words // 8
        if not bitrate_ok(k2, cb):
            raise SystemExit("%s: %d words a tile (K %g) with codebook %s: not a bitrate this tool decodes" %
                             (base, words, k2 / 2, CODEBOOKS[cb]))
        trd = tr.to(dev).contiguous()
        wh = w_hat(trd, k2, cb)
        k, n = wh.shape
        w = torch.zeros((k, n), dtype=torch.half, device=dev)
        ext.reconstruct_had_slice(w, trd, parts["suh"].to(dev).half().contiguous(),
                                  parts["svh"].to(dev).half().contiguous(), k2 / 2, cb == 1, cb == 2, 0)
        torch.cuda.synchronize()
        (out / (base + ".trellis.i16")).write_bytes(tr.numpy().astype("<i2").tobytes())
        (out / (base + ".suh.f16")).write_bytes(parts["suh"].half().numpy().astype("<f2").tobytes())
        (out / (base + ".svh.f16")).write_bytes(parts["svh"].half().numpy().astype("<f2").tobytes())
        (out / (base + ".what.f16")).write_bytes(wh.cpu().numpy().astype("<f2").tobytes())
        (out / (base + ".w.f16")).write_bytes(w.cpu().numpy().astype("<f2").tobytes())
        line = "%s %d %d %d %d" % (base, k2, cb, k, n)
        with open(out / "manifest.txt", "a") as f:
            f.write(line + "\n")
        print(line, "(K %g %s)" % (k2 / 2, CODEBOOKS[cb]))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, help="the fixture directory (made if missing)")
    ap.add_argument("--set", nargs="*", default=None, metavar="K2:CB",
                    help="random-tile sets, k2 = 2 K and the codebook (default: K 1..8 mul1, K 3 3inst and mcg)")
    ap.add_argument("--seed", type=int, default=20261018)
    ap.add_argument("--ki", type=int, default=2, help="tile rows (16 inputs each)")
    ap.add_argument("--nj", type=int, default=8, help="tile columns (16 outputs each; a multiple of 8)")
    ap.add_argument("--model", help="an EXL3 model directory: write its --tensor matrices instead of random tiles")
    ap.add_argument("--tensor", nargs="*", default=[], help="with --model: the matrices' safetensors prefixes")
    a = ap.parse_args()
    if not torch.cuda.is_available():
        print("no CUDA device: exllamav3's packers and reconstruct are CUDA kernels", file=sys.stderr)
        return 2
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    if a.model:
        if not a.tensor:
            raise SystemExit("--model needs --tensor <prefix> ...")
        model_tensors(out, pathlib.Path(a.model), a.tensor)
        return 0
    sets = DEFAULT_SETS
    if a.set is not None:
        try:
            sets = [(int(s.split(":")[0]), int(s.split(":")[1])) for s in a.set]
        except (ValueError, IndexError):
            raise SystemExit("--set takes K2:CB pairs, e.g. 6:2 (K 3, mul1)")
    random_tiles(out, sets, a.seed, a.ki, a.nj)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

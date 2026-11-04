"""tools/ds4/fp8_dense.py - a DeepSeek-V4.1 EXL3 quant's dense FP8 weights as BF16, for the engine.

    python tools/ds4/fp8_dense.py --model ~/models/<model>

Some EXL3 quants (the mixed-precision ones) keep the attention, the shared experts and the indexer in DeepSeek's own
FP8 (E4M3 values, an E8M0 power-of-two scale a block of rows x columns) instead of EXL3. The engine reads dense
matrices as EXL3, BF16 or Q8_0: this writes every such FP8 weight (not the vision tower's) as BF16 - exactly, each FP8
value times its block's scale is a BF16 number - into <model>/maya-dense-bf16.safetensors, under the names the engine
reads (attn.wo_a as its groups' slices, attn.wo_a.slice.<g>.weight, as pack_gguf.py lays it out). pack.py then indexes
those in place of the FP8 originals. About twice the FP8 bytes (DeepSeek-V4.1-Flash: ~14 GB).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from st import SafeTensors  # noqa: E402

OUT = "maya-dense-bf16.safetensors"


def e4m3_table() -> np.ndarray:
    """The 256 E4M3 (fn) codes as float32: 1 sign, 4 exponent (bias 7), 3 mantissa bits; 0x7F / 0xFF are NaN."""
    t = np.zeros(256, dtype=np.float32)
    for c in range(256):
        s, e, m = (c >> 7) & 1, (c >> 3) & 15, c & 7
        if e == 15 and m == 7:
            v = np.nan
        elif e == 0:
            v = (m / 8.0) * 2.0 ** -6
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        t[c] = -v if s else v
    return t


E4M3 = e4m3_table()


def dequant(w: np.ndarray, sc: np.ndarray, shape: tuple) -> np.ndarray:
    """FP8 codes [rows, cols] and E8M0 scale codes [rows / br, cols / bc] -> float32 [rows, cols]."""
    rows, cols = shape
    v = E4M3[w.reshape(rows, cols)]
    sr, scn = sc.shape
    br, bc = -(-rows // sr), -(-cols // scn)
    s = np.ldexp(np.float32(1.0), sc.astype(np.int32) - 127).astype(np.float32)
    s = np.repeat(np.repeat(s, br, axis=0), bc, axis=1)[:rows, :cols]
    return v * s


def to_q8_0(a: np.ndarray) -> np.ndarray:
    """float32 [rows, cols] -> ggml Q8_0 bytes: each 32 values one FP16 scale (the block's largest |value| / 127) and
    32 int8 - the format the engine's fastest dense kernels read (as the GGUF quants' dense weights)."""
    rows, cols = a.shape
    b = a.reshape(rows, cols // 32, 32).astype(np.float32)
    d = np.abs(b).max(axis=2) / 127.0
    inv = np.where(d > 0, 1.0 / np.where(d > 0, d, 1.0), 0.0).astype(np.float32)
    q = np.clip(np.rint(b * inv[..., None]), -127, 127).astype(np.int8)
    out = np.empty((rows, cols // 32, 34), dtype=np.uint8)
    out[..., :2] = d.astype(np.float16).view(np.uint8).reshape(rows, cols // 32, 2)
    out[..., 2:] = q.view(np.uint8)
    return out


def to_bf16(a: np.ndarray) -> np.ndarray:
    """float32 -> BF16 bits; exact here (an FP8 value times a power of two has at most 4 significant bits)."""
    return (a.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--q8", action="store_true", help="Q8_0 instead of BF16: half the bytes and the GGUF quants' "
                                                     "dense kernels (a 32-value block's own scale: not exact)")
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser().resolve()
    t = json.loads((model / "config.json").read_text())["text_config"]
    groups = int(t.get("o_groups", 0))
    st = SafeTensors(model)
    jobs = []   # (out name, source name, row range or None)
    for name, (fn, dt, shape, off, nb) in st.t.items():
        if fn == OUT or dt != "F8_E4M3" or not name.endswith(".weight") or name.startswith("vision.") or \
                ".engram.embed." in name:   # (the Engram tables: the engine reads them as they are)
            continue
        base = name[:-len(".weight")]
        if not st.has(base + ".scale") or len(shape) != 2:
            raise SystemExit("%s: FP8 without a block scale (or not a matrix)" % name)
        if base.endswith(".attn.wo_a"):
            if groups < 1 or shape[0] % groups:
                raise SystemExit("%s %s: does not split in o_groups %d" % (name, shape, groups))
            per = shape[0] // groups
            for g in range(groups):
                jobs.append(("%s.slice.%d.weight" % (base, g), name, (g * per, (g + 1) * per)))
        else:
            jobs.append((name, name, None))
    if not jobs:
        print("%s: no FP8 dense weights - nothing to do" % model.name)
        return 0
    # the header first: each output's shape and offsets
    head, off = {}, 0
    for out, src, rr in jobs:
        rows, cols = st.t[src][2]
        if rr is not None:
            rows = rr[1] - rr[0]
        nb = rows * cols // 32 * 34 if a.q8 else rows * cols * 2
        head[out] = {"dtype": "Q8_0" if a.q8 else "BF16", "shape": [rows, cols], "data_offsets": [off, off + nb]}
        off += nb
    head["__metadata__"] = {"format": "pt", "source": "fp8_dense.py: the model's FP8 dense weights as BF16"}
    hb = json.dumps(head, separators=(",", ":")).encode()
    hb += b" " * (-len(hb) % 8)
    tmp = model / (OUT + ".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<Q", len(hb)))
        f.write(hb)
        cache = {}
        for i, (out, src, rr) in enumerate(jobs):
            if src not in cache:
                cache.clear()
                shape = st.t[src][2]
                sc = st.raw(src[:-len(".weight")] + ".scale").reshape(st.t[src[:-len(".weight")] + ".scale"][2])
                cache[src] = dequant(st.raw(src), sc, shape)
            full = cache[src]
            part = full if rr is None else full[rr[0]:rr[1]]
            part = np.ascontiguousarray(part)
            f.write((to_q8_0(part) if a.q8 else to_bf16(part)).tobytes())
            if i % 64 == 0:
                print("  %d / %d %s" % (i, len(jobs), out), flush=True)
    tmp.replace(model / OUT)
    print("%s: %d FP8 weights as BF16 (%.1f GB) -> %s" % (model.name, len(jobs), off / 1e9, OUT))
    return 0


if __name__ == "__main__":
    sys.exit(main())

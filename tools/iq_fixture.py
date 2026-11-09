#!/usr/bin/env python3
"""tools/iq_fixture.py - small real-format GGUF expert fixtures for the K-quant and i-quant parity tools (B13.1).

    python3 tools/iq_fixture.py --out FILE.gguf [--types Q2_K,IQ2_XXS,...] [--n-embd 1024] [--n-ff 512]
                                [--n-expert 8] [--seed 1] [--manifest FILE.json] [--check] [--ref-out FILE.f32]

Writes one GGUF ("general.architecture" iq-fixture) whose layer i holds blk.i.ffn_{gate,up,down}_exps.weight in the
i-th type of --types: gate/up {n_embd, n_ff, n_expert}, down {n_ff, n_embd, n_expert} (GGUF order), the layout of a
routed-expert tensor in a llama.cpp MoE file.  src/kernels/native_expert_parity.cpp reads it as
`native_expert_parity FILE.gguf 0 1 2 ...` (float reference vs ggml-cpu vs GPU on the same blocks); the manifest
lists layer -> type and the sha256 of each tensor's bytes.

How the blocks are made:
  * exact block construction for every K-quant and i-quant type (Q2_K..Q6_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS,
    IQ3_S, IQ1_S, IQ4_XS, IQ4_NL): seeded random bytes, then each block's fp16 super-scale(s) (d, and dmin
    on Q2_K/Q4_K/Q5_K) overwritten with a finite positive value so every weight is finite and of order 1e-2..1.
    Every bit pattern of the other fields is a valid encoding (grid indices, sign bits, scales), so the fixture
    covers the whole code space, not only what a quantizer would emit;
  * gguf-py quantization of seeded normal weights for the types gguf-py can quantize (Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    BF16, F16, F32).
The output is a pure function of the arguments (byte-identical on every machine).  --check dequantizes every tensor
with gguf-py and verifies the values are finite and the shapes right.  --ref-out writes gguf-py's float32 dequant of
expert REF_EXPERT's first REF_ROWS gate rows and first REF_ROWS down rows per layer, in layer order: the oracle
tests/glm_dsa/iq_fixture_cpu_test.cpp holds ggml's own to_float to.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_gguf import import_gguf  # noqa: E402
gguf = import_gguf()  # PYTHONPATH's gguf-py, else where tools/_paths.py looks
from gguf import GGMLQuantizationType as T  # noqa: E402
from gguf.constants import GGML_QUANT_SIZES  # noqa: E402

# the expert types of the GLM-5.3 candidates (BUILD-PLAN section 3.2 A4, moe_supported) in the default order
DEFAULT_TYPES = ("Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S",
                 "IQ4_XS")

# byte offsets of the fp16 super-block scales inside one block: (d offset, dmin offset or None)
CONSTRUCTED = {
    "Q2_K": (80, 82), "Q3_K": (108, None), "Q4_K": (0, 2), "Q5_K": (0, 2), "Q6_K": (208, None),
    "IQ2_XXS": (0, None), "IQ2_XS": (0, None), "IQ2_S": (0, None), "IQ3_XXS": (0, None), "IQ3_S": (0, None),
    "IQ1_S": (0, None), "IQ4_XS": (0, None), "IQ4_NL": (0, None),
}
# the scale range per type: |w| stays O(1e-2..1) whatever the random sub-scales and quants are
D_RANGE = {
    "Q2_K": (2e-3, 6e-3), "Q3_K": (4e-4, 1.2e-3), "Q4_K": (1e-4, 4e-4), "Q5_K": (5e-5, 2e-4), "Q6_K": (2e-4, 6e-4),
    "IQ2_XXS": (2e-3, 6e-3), "IQ2_XS": (2e-3, 6e-3), "IQ2_S": (2e-3, 6e-3), "IQ3_XXS": (2e-3, 6e-3),
    "IQ3_S": (2e-3, 6e-3), "IQ1_S": (5e-3, 1.5e-2), "IQ4_XS": (2e-4, 6e-4), "IQ4_NL": (2e-4, 6e-4),
}
REF_EXPERT, REF_ROWS = 7, 16   # the expert native_expert_parity reads; rows the reference dump carries
QUANTIZED = ("Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "BF16", "F16", "F32")


def type_of(name: str) -> T:
    try:
        return T[name.upper()]
    except KeyError:
        raise SystemExit(f"iq_fixture: unknown type {name!r}")


def constructed_blocks(rng: np.random.Generator, qtype: str, n_blocks: int) -> np.ndarray:
    """n_blocks valid blocks of qtype: random bytes with finite positive fp16 super-scales."""
    _, bsz = GGML_QUANT_SIZES[T[qtype]]
    b = rng.integers(0, 256, size=(n_blocks, bsz), dtype=np.uint8)
    lo, hi = D_RANGE[qtype]
    d_off, m_off = CONSTRUCTED[qtype]
    for off in (d_off, m_off):
        if off is None:
            continue
        d = rng.uniform(lo, hi, size=n_blocks).astype(np.float16)
        b[:, off:off + 2] = d.view(np.uint8).reshape(n_blocks, 2)
    if qtype == "Q6_K":   # int8 sub-scales: keep them away from -128 (the activation corner B14.8 checks separately)
        sc = b[:, 192:208].view(np.int8)
        np.clip(sc, -127, 127, out=sc)
    return b


def tensor_bytes(rng: np.random.Generator, qtype: str, rows: int, cols: int) -> np.ndarray:
    """(rows, row_bytes) uint8 of a {cols, rows} GGUF matrix (cols contiguous) in qtype."""
    blk, bsz = GGML_QUANT_SIZES[T[qtype]]
    if cols % blk:
        raise SystemExit(f"iq_fixture: {qtype} needs rows of a multiple of {blk} values, not {cols}")
    if qtype in CONSTRUCTED:
        return constructed_blocks(rng, qtype, rows * (cols // blk)).reshape(rows, cols // blk * bsz)
    if qtype in QUANTIZED:
        w = rng.normal(0.0, 0.05, size=(rows, cols)).astype(np.float32)
        q = gguf.quants.quantize(w, T[qtype])
        return np.ascontiguousarray(q).view(np.uint8).reshape(rows, -1)
    raise SystemExit(f"iq_fixture: type {qtype} is neither constructed nor quantizable here")


def build(out: Path, types: list[str], n_embd: int, n_ff: int, n_expert: int, seed: int) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    w = gguf.GGUFWriter(str(out), "iq-fixture")
    w.add_name("iq-fixture")
    w.add_uint32("iq-fixture.block_count", len(types))
    w.add_uint32("iq-fixture.embedding_length", n_embd)
    w.add_uint32("iq-fixture.expert_feed_forward_length", n_ff)
    w.add_uint32("iq-fixture.expert_count", n_expert)
    w.add_uint32("iq-fixture.seed", seed)
    w.add_array("iq-fixture.layer_types", list(types))
    manifest = {"file": out.name, "seed": seed, "n_embd": n_embd, "n_ff": n_ff, "n_expert": n_expert, "layers": []}
    for il, ty in enumerate(types):
        # one generator per layer and role: adding a type at the end keeps the earlier layers' bytes
        entry = {"layer": il, "type": ty, "sha256": {}}
        for role, rows, cols in (("gate", n_ff, n_embd), ("up", n_ff, n_embd), ("down", n_embd, n_ff)):
            rng = np.random.default_rng([seed, il, ("gate", "up", "down").index(role), int(T[ty])])
            per = [tensor_bytes(rng, ty, rows, cols) for _ in range(n_expert)]
            arr = np.stack(per)                                  # (n_expert, rows, row_bytes)
            name = f"blk.{il}.ffn_{role}_exps.weight"
            if ty in ("F32", "F16"):
                arr = arr.view(np.float32 if ty == "F32" else np.float16).reshape(n_expert, rows, cols)
                w.add_tensor(name, arr)
            else:
                w.add_tensor(name, arr, raw_dtype=T[ty])
            entry["sha256"][role] = hashlib.sha256(arr.tobytes()).hexdigest()
        manifest["layers"].append(entry)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    manifest["file_sha256"] = hashlib.sha256(out.read_bytes()).hexdigest()
    return manifest


def check(path: Path, types: list[str], n_embd: int, n_ff: int, n_expert: int) -> list[str]:
    """Dequantize every tensor with gguf-py: shapes, types and finiteness.  Returns the problems."""
    bad = []
    r = gguf.GGUFReader(str(path))
    by = {t.name: t for t in r.tensors}
    for il, ty in enumerate(types):
        for role, shape in (("gate", (n_embd, n_ff, n_expert)), ("up", (n_embd, n_ff, n_expert)),
                            ("down", (n_ff, n_embd, n_expert))):
            t = by.get(f"blk.{il}.ffn_{role}_exps.weight")
            if t is None:
                bad.append(f"layer {il}: no {role} tensor")
                continue
            if t.tensor_type != T[ty] or tuple(int(x) for x in t.shape) != shape:
                bad.append(f"layer {il} {role}: {t.tensor_type.name} {tuple(t.shape)}, expected {ty} {shape}")
                continue
            v = gguf.quants.dequantize(t.data, t.tensor_type)
            if not np.all(np.isfinite(v)):
                bad.append(f"layer {il} {role}: non-finite values")
            elif float(np.abs(v).max()) == 0.0:
                bad.append(f"layer {il} {role}: all zero")
    return bad


def write_ref(path: Path, out: Path, types: list[str], n_expert: int) -> None:
    """gguf-py float32 dequant: per layer, gate rows [0, REF_ROWS) then down rows [0, REF_ROWS) of REF_EXPERT."""
    r = gguf.GGUFReader(str(path))
    by = {t.name: t for t in r.tensors}
    e = min(REF_EXPERT, n_expert - 1)
    with open(out, "wb") as f:
        for il in range(len(types)):
            for role in ("gate", "down"):
                t = by[f"blk.{il}.ffn_{role}_exps.weight"]
                rows = np.asarray(t.data)[e][:REF_ROWS]          # (REF_ROWS, row_bytes or row values)
                v = gguf.quants.dequantize(rows, t.tensor_type).astype(np.float32)
                f.write(np.ascontiguousarray(v).tobytes())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--types", default=",".join(DEFAULT_TYPES),
                    help="comma list, one layer per type (default: the 12 GLM-5.3 expert types)")
    ap.add_argument("--n-embd", type=int, default=1024)
    ap.add_argument("--n-ff", type=int, default=512)
    ap.add_argument("--n-expert", type=int, default=8, help=">= 8: native_expert_parity reads expert 7")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--manifest", type=Path, help="JSON manifest (default: <out>.json)")
    ap.add_argument("--check", action="store_true", help="dequantize everything with gguf-py afterwards")
    ap.add_argument("--ref-out", type=Path, help="gguf-py float32 reference rows (see the module doc)")
    a = ap.parse_args(argv)
    types = [t.strip().upper() for t in a.types.split(",") if t.strip()]
    for t in types:
        type_of(t)
    if a.n_expert < 1 or a.n_embd < 1 or a.n_ff < 1:
        raise SystemExit("iq_fixture: sizes must be positive")
    man = build(a.out, types, a.n_embd, a.n_ff, a.n_expert, a.seed)
    mpath = a.manifest or a.out.with_suffix(a.out.suffix + ".json")
    mpath.write_text(json.dumps(man, indent=1) + "\n")
    print(f"iq_fixture: wrote {a.out} ({a.out.stat().st_size / 1e6:.1f} MB, {len(types)} layers: "
          f"{', '.join(f'{i}={t}' for i, t in enumerate(types))}), manifest {mpath}")
    if a.ref_out:
        write_ref(a.out, a.ref_out, types, a.n_expert)
        print(f"iq_fixture: reference rows (expert {min(REF_EXPERT, a.n_expert - 1)}, {REF_ROWS} gate + {REF_ROWS} "
              f"down rows a layer) in {a.ref_out}")
    if a.check:
        bad = check(a.out, types, a.n_embd, a.n_ff, a.n_expert)
        for b in bad:
            print("iq_fixture: CHECK FAIL " + b)
        if bad:
            return 1
        print("iq_fixture: check OK (gguf-py dequantizes every tensor to finite values)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate a small GLM pack source with the real CUDA kernel geometry.

Set MAYA_GGUF_PY to the pinned llama.cpp/gguf-py, then pack this GGUF with
`tools/iq_pack.py`. No downloaded model weights are used.
"""

import argparse
from pathlib import Path
from unittest import mock

from _paths import add_gguf_py

add_gguf_py()
import struct

import glm_synth_gguf as s
import numpy as np
from gguf import GGMLQuantizationType as Q
from gguf import quants
from iq_pack import GLM5NEXT_NATIVE_OK, needs_bf16
from numpy.typing import NDArray


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--embd",
        type=int,
        default=1024,
        choices=[512, 1024],
        help="embedding width for constrained GPU tests",
    )
    parser.add_argument(
        "--mtp", action="store_true", help="include the native NextN drafter"
    )
    args = parser.parse_args()
    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    g = dict(
        s.BIG,
        n_embd=args.embd,
        n_layers=2,
        n_head=16,
        n_head_kv=[0, 1],
        kda_head_dim=128,
        n_expert=64,
        n_exp_used=8,
        idx_heads=64,
        kv_lora=512,
        n_ff_exp=256,
        n_ff_dense=512,
        dense_lead=1,
    )
    if args.mtp:
        import glm_mtp_fixture as mtp

        def build_native_trunk(
            seed: int, vocab: bool = False
        ) -> tuple[list[tuple[str, bytes]], list[tuple[str, NDArray[np.float32]]]]:
            return s._build(g, seed, True)

        with (
            mock.patch.object(s, "SMALL", g),
            mock.patch.object(s, "build", build_native_trunk),
        ):
            metadata, tensors = mtp.build(False)
    else:
        metadata, tensors = s._build(g, 1234, True)
    infos = []
    blobs = []
    offset = 0
    for name, array in tensors:
        if needs_bf16(name, "F32", "glm5-next"):
            kind = Q.BF16
        elif "_exps.weight" in name or name.endswith(
            GLM5NEXT_NATIVE_OK + ("nextn.eh_proj.weight",)
        ):
            kind = Q.Q8_0
        else:
            kind = Q.F32
        raw = quants.quantize(np.ascontiguousarray(array, dtype="<f4"), kind).tobytes()
        dims = list(reversed(array.shape))
        info = (
            s._s(name)
            + struct.pack("<I", len(dims))
            + b"".join(struct.pack("<Q", d) for d in dims)
            + struct.pack("<IQ", int(kind), offset)
        )
        infos.append(info)
        blobs.append(raw + b"\0" * (-len(raw) % 32))
        offset += len(blobs[-1])
    header = (
        b"GGUF"
        + struct.pack("<IQQ", 3, len(tensors), len(metadata))
        + b"".join(s._s(k) + v for k, v in metadata)
        + b"".join(infos)
    )
    out.write_bytes(header + b"\0" * (-len(header) % 32) + b"".join(blobs))
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()

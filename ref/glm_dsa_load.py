"""ref/glm_dsa_load.py - GGUF -> float tensors for the glm-dsa oracle, for ANY ggml block type.

`ref/load.py` (Maya) serves the Flash artifact: F32/F16/BF16, a fixed list of 32- and 256-block
types and Maya's own Q2_0.  A community GLM-5.3 GGUF carries more: routed experts in IQ2_XXS,
IQ2_S, IQ3_XXS, IQ1_S, Q2_K-Q6_K, IQ4_XS; `attn_k_b`/`attn_v_b` in a 32-block fallback type
(Q8_0, Q4_0, Q5_0, IQ4_NL) or F16, because ne0 = 192 forbids K/IQ types; indexer and attention
tensors in whatever the recipe picked.  This module keeps Maya's lazy access pattern (whole
tensor, chosen rows, ONE expert of a fused [in, out, n_expert] tensor) and dequantizes every type
through gguf-py, the decoder llama.cpp's own Python tooling validates against ggml.  Nothing is
re-implemented here except the three trivial float formats.

    from glm_dsa_load import GgufAny
    g = GgufAny("model-00001-of-00006.gguf")     # sibling shards found by the -NNNNN-of-MMMMM suffix
    w = g.tensor("blk.0.attn_k_b.weight")       # (64, 512, 192) float32: GGUF dims reversed
    e = g.expert("blk.3.ffn_gate_exps.weight", 17)   # (n_ff, n_embd), only expert 17 is read

Shapes follow ref/load.py: numpy axes are the GGUF dims reversed, so a 2-D weight is (out, in)
and `W @ x` is `ggml_mul_mat(W, x)`.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE.parent / "tools"))

from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402
from load import _discover_shards  # noqa: E402  (shard discovery is shared with Maya's loader)


def _gguf_py():
    """Import gguf-py: the one on sys.path/PYTHONPATH if any, else where tools/_paths.py looks."""
    from strata_gguf import import_gguf
    import_gguf()
    from gguf import GGMLQuantizationType, quants
    return GGMLQuantizationType, quants


def bf16_to_f32(raw: np.ndarray) -> np.ndarray:
    return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)


def dequant(type_id: int, type_name: str, raw: np.ndarray) -> np.ndarray:
    """Flat float32 values of `raw` (whole blocks of one ggml type)."""
    raw = np.ascontiguousarray(raw, dtype=np.uint8)
    if type_name == "F32":
        return raw.view(np.float32).copy()
    if type_name == "F16":
        return raw.view(np.float16).astype(np.float32)
    if type_name == "BF16":
        return bf16_to_f32(raw)
    if type_name == "F64":
        return raw.view(np.float64).astype(np.float32)
    Q, quants = _gguf_py()
    try:
        qt = Q(type_id)
    except ValueError as e:
        raise NotImplementedError(f"ggml type {type_id} ({type_name}) is unknown to gguf-py") from e
    return np.asarray(quants.dequantize(raw, qt), dtype=np.float32).reshape(-1)


class GgufAny:
    """Read-only, demand-paged view of a (possibly multi-shard) GGUF of any ggml block type."""

    def __init__(self, path):
        self.paths = _discover_shards(path)
        self.path = self.paths[0]
        self.shards: list[tuple[GGUFFile, object]] = []
        self._by_name: dict[str, tuple[int, object]] = {}
        for i, sp in enumerate(self.paths):
            gf = GGUFFile(sp)
            self.shards.append((gf, sp.open("rb")))
            for t in gf.tensors:
                if t.name in self._by_name:
                    raise ValueError(f"{sp.name}: duplicate tensor {t.name} across shards")
                self._by_name[t.name] = (i, t)
        self.g = self.shards[0][0]
        self.metadata = self.g.metadata

    def close(self) -> None:
        for _, fh in self.shards:
            fh.close()
        self.shards = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---- directory
    def has(self, name: str) -> bool:
        return name in self._by_name

    def names(self) -> list[str]:
        return list(self._by_name)

    def info(self, name: str):
        return self._by_name[name][1]

    @staticmethod
    def geometry(t) -> tuple[int, int]:
        geom = BLOCK_GEOMETRY.get(t.type_name)
        if geom is None:
            raise NotImplementedError(f"{t.name}: no block geometry for {t.type_name}")
        return geom

    def row_bytes(self, t) -> int:
        el, by = self.geometry(t)
        if t.shape[0] % el:
            raise ValueError(f"{t.name}: ne0 {t.shape[0]} is not a multiple of the {t.type_name} block {el}")
        return t.shape[0] // el * by

    # ---- raw bytes
    def _raw(self, name: str, nbytes: int, byte_offset: int = 0) -> np.ndarray:
        i, t = self._by_name[name]
        gf, fh = self.shards[i]
        fh.seek(gf.data_start + t.offset + byte_offset)
        b = fh.read(nbytes)
        if len(b) != nbytes:
            raise ValueError(f"{name}: short read ({len(b)} of {nbytes} bytes)")
        return np.frombuffer(b, dtype=np.uint8)

    # ---- reads
    def tensor(self, name: str) -> np.ndarray:
        """The whole tensor as float32, numpy shape = GGUF dims reversed."""
        t = self.info(name)
        el, by = self.geometry(t)
        if t.elements % el:
            raise ValueError(f"{name}: {t.elements} elements is not a multiple of block {el}")
        flat = dequant(t.type_id, t.type_name, self._raw(name, t.elements // el * by))
        return flat.reshape(tuple(reversed(t.shape)))

    def rows(self, name: str, idx) -> np.ndarray:
        """Rows `idx` of a 2-D tensor (a row is ne0 contiguous values), without reading the rest."""
        t = self.info(name)
        if len(t.shape) != 2:
            raise ValueError(f"{name}: rows() needs a 2-D tensor, got {t.shape}")
        rb = self.row_bytes(t)
        idx = np.asarray(idx, dtype=np.int64).ravel()
        if idx.size and (idx.min() < 0 or idx.max() >= t.shape[1]):
            raise ValueError(f"{name}: row index out of range (0..{t.shape[1] - 1})")
        out = np.empty((idx.size, t.shape[0]), dtype=np.float32)
        for k, r in enumerate(idx):
            out[k] = dequant(t.type_id, t.type_name, self._raw(name, rb, int(r) * rb))
        return out

    def expert(self, name: str, e: int) -> np.ndarray:
        """Expert `e` of a fused [in, out, n_expert] tensor as (out, in); only its bytes are read."""
        t = self.info(name)
        if len(t.shape) != 3:
            raise ValueError(f"{name}: not a fused expert tensor (shape {t.shape})")
        n_in, n_out, n_exp = t.shape
        if not 0 <= e < n_exp:
            raise ValueError(f"{name}: expert {e} out of range (0..{n_exp - 1})")
        per = self.row_bytes(t) * n_out
        flat = dequant(t.type_id, t.type_name, self._raw(name, per, e * per))
        return flat.reshape(n_out, n_in)

    def expert_bytes(self, name: str) -> int:
        """Bytes of one expert's matrix (the pack's blob unit)."""
        t = self.info(name)
        return self.row_bytes(t) * t.shape[1]


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="list or dequantize tensors of a GGUF (any ggml type)")
    ap.add_argument("gguf")
    ap.add_argument("--tensor", default=None, help="dequantize this tensor and print stats")
    a = ap.parse_args()
    with GgufAny(a.gguf) as g:
        if a.tensor:
            x = g.tensor(a.tensor)
            print(f"{a.tensor} {g.info(a.tensor).type_name} {x.shape}: mean {x.mean():.6g} "
                  f"std {x.std():.6g} min {x.min():.6g} max {x.max():.6g}")
            return 0
        for n in g.names():
            t = g.info(n)
            print(f"{n}\t{t.type_name}\t{'x'.join(str(d) for d in t.shape)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

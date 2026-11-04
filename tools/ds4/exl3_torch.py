"""tools/ds4/exl3_torch.py - EXL3 (exllamav3) matrices decoded on the GPU with torch, for the DeepSeek-V4.1 reference.

The format is include/strata/kernels/exl3.hpp's; this is tools/exl3_pack.py's numpy decoder (exl3_states, exl3_values,
exl3_rows) on torch tensors.  decode(st, prefix) returns the full weight as [n, k] (out x in, F.linear's layout):
W = diag(suh) H W' H diag(svh), so x W is H(H(x * suh) W') * svh.
"""
from __future__ import annotations

import math

import numpy as np
import torch

_P = torch.arange(256)
_ROW = (_P >> 3 & 3) * 2 + (_P & 1) + 8 * (_P >> 1 & 1)
_COL = (_P >> 5) + 8 * (_P >> 2 & 1)
_PERM = _ROW * 16 + _COL                     # tile[..., _PERM[p]] = value of step p


def _sylvester(n: int) -> torch.Tensor:
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / math.sqrt(n)


_HAD = {}


def had(dev) -> torch.Tensor:
    if dev not in _HAD:
        _HAD[dev] = _sylvester(128).to(dev, torch.float32)
    return _HAD[dev]


def _f16(b: int) -> float:
    return float(np.array([b], dtype=np.uint16).view(np.float16)[0])


K_INV, K_BIAS = _f16(0x1EEE), _f16(0xC931)


def states(words: torch.Tensor, K: int) -> torch.Tensor:
    """(..., 8 K) int32 tile words -> (..., 256) 16-bit states (MSB-first stream, tail-biting), int64."""
    W = 8 * K
    dev = words.device
    p = _P.to(dev)
    b1 = (p + 1 + 256) * K
    i1, i0 = (b1 - 1) >> 5, (b1 - 16) >> 5
    sh = ((i1 + 1) << 5) - b1
    w = words.to(torch.int64) & 0xFFFFFFFF
    m = (w[..., i0 % W] << 32) | w[..., i1 % W]
    return (m >> sh) & 0xFFFF


def values(st: torch.Tensor, cb: int) -> torch.Tensor:
    """States -> FP16-rounded codebook values (float32)."""
    if cb == 2:
        x = (st * 0x83DCD12D) & 0xFFFFFFFF
        bs = (x & 0xFF) + (x >> 8 & 0xFF) + (x >> 16 & 0xFF) + (x >> 24 & 0xFF)
        v = (1024.0 + bs.to(torch.float64)) * K_INV + K_BIAS
        return v.to(torch.float16).to(torch.float32)
    x = st * (0xCBAC1FED if cb == 1 else 89226354)
    if cb == 0:
        x = x + 64248484
    x = ((x & 0xFFFFFFFF) & 0x8FFF8FFF) ^ 0x3B603B60
    lo = (x & 0xFFFF).to(torch.int16).view(torch.float16).to(torch.float64)   # bit patterns as FP16
    hi = (x >> 16).to(torch.int16).view(torch.float16).to(torch.float64)
    return (lo + hi).to(torch.float16).to(torch.float32)


def decode_raw(trellis: torch.Tensor, suh: torch.Tensor, svh: torch.Tensor, cb: int) -> torch.Tensor:
    """trellis int16 [k/16, n/16, 16 K] (device), suh F16 [k], svh F16 [n] -> W [n, k] float32."""
    kt, nt, w16 = trellis.shape
    K = w16 // 16
    k, n = kt * 16, nt * 16
    dev = trellis.device
    words = trellis.contiguous().view(torch.int32)                    # [kt, nt, 8K]
    out = torch.empty(n, k, dtype=torch.float32, device=dev)
    H = had(dev)
    step = max(1, (1 << 27) // (nt * 256 * 8))                       # 16-row slices per chunk (bounded scratch)
    step = max(8, step - step % 8)                                    # whole 128-row Hadamard blocks
    perm = _PERM.to(dev)
    for a in range(0, kt, step):
        b = min(kt, a + step)
        v = values(states(words[a:b], K), cb)                          # [b-a, nt, 256] in step order
        tile = torch.empty_like(v)
        tile[..., perm] = v
        inner = tile.view(b - a, nt, 16, 16).permute(0, 2, 1, 3).reshape((b - a) * 16, n)
        inner = (inner.view(-1, n // 128, 128) @ H).view(-1, n)       # over n, in blocks of 128
        inner = (H @ inner.view(-1, 128, n)).view(-1, n)              # over k, in blocks of 128
        out[:, a * 16:b * 16] = inner.T
    out *= svh.to(torch.float32)[:, None]
    out *= suh.to(torch.float32)[None, :]
    return out


def decode_many(st, prefixes: list[str], dev) -> list[torch.Tensor]:
    """Several EXL3 matrices of one shape (a layer's experts' w1, say) decoded together: their trellises side by side
    along n (both Hadamards act inside blocks of 128, so the stacked matrix decodes as each would), then each one's
    channel scales.  Returns [n, k] float32 views, one a prefix."""
    trs = [tensor(st, p + ".trellis", dev) for p in prefixes]
    kt, nt, _ = trs[0].shape
    cb = codebook(st, prefixes[0])
    ones_k = torch.ones(kt * 16, dtype=torch.float16, device=dev)
    ones_n = torch.ones(nt * 16 * len(prefixes), dtype=torch.float16, device=dev)
    inner = decode_raw(torch.cat(trs, dim=1), ones_k, ones_n, cb)          # [N n, k]
    n = nt * 16
    out = []
    for i, p in enumerate(prefixes):
        w = inner[i * n:(i + 1) * n]
        w *= tensor(st, p + ".svh", dev).to(torch.float32)[:, None]
        w *= tensor(st, p + ".suh", dev).to(torch.float32)[None, :]
        out.append(w)
    return out


def tensor(st, name: str, dev) -> torch.Tensor:
    _, dt, shape, _, _ = st.t[name]
    r = st.raw(name)
    np_dt = {"F32": np.float32, "F16": np.float16, "I16": np.int16, "I32": np.int32, "BF16": np.uint16}[dt]
    a = torch.from_numpy(np.array(r.view(np_dt)).reshape(shape))
    if dt == "BF16":
        a = a.view(torch.int16).view(torch.bfloat16)
    return a.to(dev)


def is_exl3(st, prefix: str) -> bool:
    return st.has(prefix + ".trellis")


def codebook(st, prefix: str) -> int:
    return 2 if st.has(prefix + ".mul1") else 1 if st.has(prefix + ".mcg") else 0


def decode(st, prefix: str, dev) -> torch.Tensor:
    """The EXL3 matrix at `prefix` (.trellis/.suh/.svh) as W [n, k] float32 on dev."""
    return decode_raw(tensor(st, prefix + ".trellis", dev), tensor(st, prefix + ".suh", dev),
                      tensor(st, prefix + ".svh", dev), codebook(st, prefix))


def weight(st, prefix: str, dev) -> torch.Tensor:
    """A linear weight [n, k] float32: EXL3 decoded, or the stored float tensor (prefix.weight)."""
    if is_exl3(st, prefix):
        return decode(st, prefix, dev)
    return torch.from_numpy(np.array(st.f32(prefix + ".weight"))).to(dev)

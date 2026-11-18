"""tools/kimi_k31_fla_standin.py - pure-torch stand-ins for the fla 0.6.0 pieces Moonshot's Kimi code imports.

Moonshot's modeling_kimi_linear.py (HF moonshotai/Kimi-K3 @ f831ab6, lines 46-53) imports

    from fla.modules import FusedRMSNormGated, ShortConvolution
    from fla.ops.kda import chunk_kda, fused_recurrent_kda
    from fla.ops.utils.index import prepare_cu_seqlens_from_mask, prepare_lens_from_mask
    from fla.utils import tensor_cache

and fla's Triton kernels need a GPU (on a machine without one `import fla` raises at import time). This
module provides the same names in plain torch so the published code runs on a CPU, and so the tiny
fixture's golden outputs can be reproduced anywhere. `install()` registers them as `fla.*` modules in
sys.modules; call it BEFORE importing Moonshot's code, and never in the same process as the real fla.

What each stand-in computes (fla 0.6.0 sources the formulas were read from):

  chunk_kda / fused_recurrent_kda   the KDA delta rule as fla's fused_recurrent kernel writes it
      (ops/kda/fused_recurrent.py:151-199), one token at a time:
        q, k  <- x / sqrt(sum(x^2) + 1e-6)          (use_qk_l2norm_in_kernel)
        q     <- q * scale, scale = K^-0.5 by default
        gk    <- lower_bound * sigmoid(exp(A_log[h]) * (g + dt_bias))   (use_gate_in_kernel, lower_bound set;
                 -exp(A_log[h]) * softplus(g + dt_bias) when lower_bound is None)
        beta  <- sigmoid(beta)                      (use_beta_sigmoid_in_kernel)
        S     <- S * exp(gk)[:, None];  v' = beta * (v - S^T k);  S <- S + k v'^T;  o = S^T q
      A_log is read per head as A_log[h] for h < H (fused_recurrent.py:163, gate.py:133/199), so a
      longer A_log only has its first H entries used. The chunk kernel computes the same recurrence
      chunk-wise (64-token chunks, TF32 dots in the intra-chunk solve), so it differs from this
      token loop only in rounding. State layout follows transpose_state_layout / state_v_first:
      [N, H, V, K] when set, [N, H, K, V] otherwise.
  ShortConvolution   depthwise causal conv with SiLU, cache = the last W inputs [N, D, W], tap W-1 on
      the current token (modules/causal_conv1d/ops.py:1118-1320).
  FusedRMSNormGated  RMS(x, eps) * weight * sigmoid(g) for activation='sigmoid', x * silu(g) gating for
      'swish' (modules/norm/fused_norm_gate.py).

All arithmetic runs in float32 (float64 when the inputs are float64), and outputs are cast back to the
input dtype like the kernels do. Padding/varlen (cu_seqlens) is supported for the recurrence and the
conv; the oracle itself never pads.
"""
from __future__ import annotations

import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F

L2_EPS = 1e-6   # fla's in-kernel q/k l2 norm (fused_recurrent.py:157-158)


def _ct(*ts) -> torch.dtype:
    """Compute dtype: float64 if any input is float64, else float32 (what the Triton kernels use)."""
    return torch.float64 if any(t is not None and t.dtype == torch.float64 for t in ts) else torch.float32


def _segments(B: int, T: int, cu_seqlens):
    if cu_seqlens is None:
        return [(b, 0, T) for b in range(B)]
    assert B == 1, "cu_seqlens needs batch size 1 (fla convention)"
    cs = [int(c) for c in cu_seqlens.tolist()]
    return [(0, cs[i], cs[i + 1]) for i in range(len(cs) - 1)]


def kda_reference(q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
                  use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False, use_beta_sigmoid_in_kernel=False,
                  A_log=None, dt_bias=None, lower_bound=None, cu_seqlens=None,
                  transpose_state_layout=False, state_v_first=False, safe_gate=False,
                  allow_neg_eigval=False, **unused):
    """The KDA recurrence (see the module docstring). q, k: [B,T,H,K]; v: [B,T,H,V]; g: [B,T,H,K];
    beta: [B,T,H]. Returns (o [B,T,H,V] in v's dtype, final state or None)."""
    v_first = bool(state_v_first or transpose_state_layout)
    B, T, H, K = q.shape
    HV, V = v.shape[2], v.shape[3]
    if HV != H:
        raise NotImplementedError("grouped-value KDA (HV != H) is not used by Kimi")
    ct = _ct(q, k, v, g)
    if scale is None:
        scale = K ** -0.5
    qf, kf, vf, gf, bf = (t.to(ct) for t in (q, k, v, g, beta))
    if use_qk_l2norm_in_kernel:
        qf = qf / torch.sqrt((qf * qf).sum(-1, keepdim=True) + L2_EPS)
        kf = kf / torch.sqrt((kf * kf).sum(-1, keepdim=True) + L2_EPS)
    qf = qf * scale
    if use_gate_in_kernel:
        if dt_bias is not None:
            gf = gf + dt_bias.to(ct).view(H, K)                       # dt_bias[h*K + k]
        a = A_log.to(ct).reshape(-1)[:H].view(H, 1).exp() if A_log is not None else 1.0   # A_log[h], h < H
        if lower_bound is not None:
            gf = lower_bound * torch.sigmoid(a * gf)
        else:
            gf = -a * F.softplus(gf)
    if use_beta_sigmoid_in_kernel:
        bf = bf.sigmoid()
    if allow_neg_eigval:
        bf = bf * 2
    o = torch.zeros(B, T, H, V, dtype=ct, device=q.device)
    finals = []
    for n, (b, s, e) in enumerate(_segments(B, T, cu_seqlens)):
        if initial_state is not None:
            S = initial_state[n].to(ct)
            S = S.transpose(-1, -2).clone() if v_first else S.clone()   # work in [H, K, V]
        else:
            S = torch.zeros(H, K, V, dtype=ct, device=q.device)
        for t in range(s, e):
            S = S * gf[b, t].exp()[:, :, None]                           # decay each key channel
            u = vf[b, t] - torch.einsum("hk,hkv->hv", kf[b, t], S)       # v - S^T k
            u = u * bf[b, t][:, None]                                    # * beta
            S = S + kf[b, t][:, :, None] * u[:, None, :]                 # + k u^T
            o[b, t] = torch.einsum("hk,hkv->hv", qf[b, t], S)            # S^T q
        finals.append(S.transpose(-1, -2) if v_first else S)
    fs = torch.stack(finals) if output_final_state else None            # fp32 like fla's state (fp64 runs: fp64)
    return o.to(v.dtype), fs


def chunk_kda(q, k, v, g, beta, **kw):
    return kda_reference(q, k, v, g, beta, **kw)


def fused_recurrent_kda(q, k, v, g, beta, **kw):
    return kda_reference(q, k, v, g, beta, **kw)


class FusedRMSNormGated(nn.Module):
    def __init__(self, hidden_size, elementwise_affine=True, eps=1e-5, activation="swish", **unused):
        super().__init__()
        self.hidden_size, self.eps, self.activation = hidden_size, eps, activation
        self.weight = nn.Parameter(torch.ones(hidden_size)) if elementwise_affine else None

    def forward(self, x, g, residual=None, prenorm=False, residual_in_fp32=False, **unused):
        assert residual is None and not prenorm
        ct = _ct(x, g)
        xf, gf = x.to(ct), g.to(ct)
        y = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + self.eps)
        if self.weight is not None:
            y = y * self.weight.to(ct)
        if self.activation == "sigmoid":
            y = y * torch.sigmoid(gf)
        elif self.activation in ("swish", "silu"):
            y = y * gf * torch.sigmoid(gf)
        else:
            raise NotImplementedError(self.activation)
        return y.to(x.dtype)


class ShortConvolution(nn.Conv1d):
    def __init__(self, hidden_size, kernel_size, bias=False, activation="silu", backend=None, device=None,
                 dtype=None, **unused):
        super().__init__(hidden_size, hidden_size, kernel_size, groups=hidden_size, padding=kernel_size - 1,
                         bias=bias, device=device, dtype=dtype)
        self.hidden_size, self.activation = hidden_size, activation

    def forward(self, x, residual=None, mask=None, cache=None, output_final_state=False, cu_seqlens=None,
                **unused):
        assert residual is None and mask is None
        B, T, D = x.shape
        W = self.kernel_size[0]
        ct = _ct(x)
        w = self.weight[:, 0, :].to(ct)                                     # [D, W]; w[:, W-1] = current token
        y = torch.empty(B, T, D, dtype=ct, device=x.device)
        states = []
        for n, (b, s, e) in enumerate(_segments(B, T, cu_seqlens)):
            prev = cache[n].to(ct) if cache is not None else torch.zeros(D, W, dtype=ct, device=x.device)
            seq = torch.cat([prev, x[b, s:e].to(ct).t()], dim=1)           # [D, W + L]: last W inputs, then new
            out = F.conv1d(seq[None, :, 1:], w[:, None, :], groups=D)[0]    # [D, L]
            if self.bias is not None:
                out = out + self.bias.to(ct)[:, None]
            if self.activation in ("silu", "swish"):
                out = F.silu(out)
            y[b, s:e] = out.t()
            states.append(seq[:, -W:])
        fs = torch.stack(states).to(x.dtype) if output_final_state else None
        if cache is not None and fs is not None:
            cache.copy_(fs)                                                 # the real step() updates in place
            fs = cache
        return y.to(x.dtype), fs


def prepare_lens_from_mask(mask):
    return mask.sum(dim=-1, dtype=torch.int32)


def prepare_cu_seqlens_from_mask(mask, dtype=torch.int32):
    return F.pad(prepare_lens_from_mask(mask).cumsum(dim=0, dtype=dtype), (1, 0))


def tensor_cache(fn):
    return fn


def install() -> None:
    """Register the stand-ins as the fla modules Moonshot's code imports. Refuses if the real fla is loaded."""
    if "fla" in sys.modules and not getattr(sys.modules["fla"], "_kimi_standin", False):
        raise RuntimeError("the real fla is already imported in this process; the stand-in cannot replace it")
    names = ["fla", "fla.modules", "fla.ops", "fla.ops.kda", "fla.ops.utils", "fla.ops.utils.index", "fla.utils"]
    mods = {n: types.ModuleType(n) for n in names}
    mods["fla"]._kimi_standin = True
    mods["fla"].__version__ = "standin"
    mods["fla.modules"].FusedRMSNormGated = FusedRMSNormGated
    mods["fla.modules"].ShortConvolution = ShortConvolution
    mods["fla.ops.kda"].chunk_kda = chunk_kda
    mods["fla.ops.kda"].fused_recurrent_kda = fused_recurrent_kda
    mods["fla.ops.utils.index"].prepare_lens_from_mask = prepare_lens_from_mask
    mods["fla.ops.utils.index"].prepare_cu_seqlens_from_mask = prepare_cu_seqlens_from_mask
    mods["fla.utils"].tensor_cache = tensor_cache
    sys.modules.update(mods)

"""tools/kimi_k31_moonshot_indexer.py - the Kimi K3.1 lightning indexer, patched onto Moonshot's K3 MLA module.

Kimi K3.1 is Kimi K3 plus a per-token DSA ("lightning") indexer on every MLA layer
(config text_config.dsa_config). Moonshot ships no K3.1 modeling code, so the independent oracle
(tools/kimi_k31_moonshot_oracle.py) runs Moonshot's K3 modeling_kimi_linear.py unchanged and adds only
this module. The indexer semantics follow the two public K3.1 serving implementations, which agree on
every point used here:

  * SGLang, branch kimi-k3.1-support @ accf045c (python/sglang/srt/...):
      models/deepseek_common/attention_forward_methods/forward_mla.py:449-474
          the indexer query input is q_lora = q_a_layernorm(q_a_proj(x)) (after the RMSNorm);
          the indexer key/weight input is the attention input x itself.
      layers/attention/dsa/dsa_indexer.py:308   skip_rope = (index_rope_dim == 0)   -> no RoPE
      layers/attention/dsa/dsa_indexer.py:393   k_norm = LayerNorm(head_dim, fp32, eps=rms_norm_eps)
      configs/kimi_linear.py:137-173            index_topk_freq must be 1, index_rope_dim must be 0,
                                                index_k_norm_type "layernorm", no Hadamard rotation,
                                                k_norm eps = rms_norm_eps
      layers/attention/dsa/dsa_indexer.py:435, 615-619   weights * n_heads^-0.5 * head_dim^-0.5
      layers/attention/dsa/utils.py:465-494     logits = sum_h relu(q_h . k) * w_h
      layers/attention/dsa_backend.py:1482-1522 query at position p sees keys 0..p (inclusive)
  * vLLM PR #65685 @ 9d6ac73f:
      vllm/models/kimi_k3/nvidia/dsa.py:80-101  KimiK3DSAIndexerKNorm:
          F.layer_norm(x.float(), (dim,), weight, bias, eps).type_as(x), eps = rms_norm_eps
      vllm/models/kimi_k3/nvidia/dsa.py:131, 185, 370
          weight_scale = index_head_dim^-0.5 * index_num_heads^-0.5
      tests/models/kimi_k3/test_dsa_indexer_fused.py:551-553
          _logits_ref = (einsum("thd,nd->thn", q, k).relu() * w[..., None]).sum(1)

Per MLA layer, with x [B,T,hidden] the attention input and qr = q_a_layernorm(q_a_proj(x)):

    q      = wq_b(qr).view(B, T, H, D)                                  no RoPE, no Hadamard
    k      = LayerNorm_fp32(wk(x); k_norm.weight, k_norm.bias, eps)     cast back to x's dtype (vLLM)
    w      = float(weights_proj(x)) * (D^-0.5 * H^-0.5)                 = 1/64 for the release
    score  = sum_h w[t,h] * relu(q[t,h] . k[s])                         fp32 (fp64 for an fp64 model)
    score  = -inf where key position s > query position p               causal; scores can be negative
    sel[t] = topk(score[t], min(index_topk, S))  restricted to visible keys (a set; ties -> earlier key)
    MLA    : additive mask 0 on sel[t], -inf elsewhere, folded into the attention mask

Numerics followed: vLLM PR #65685's unfused path without its FP8 q/k quantisation (the plain fp32/bf16
reference both engines' tests compare against). SGLang's extra bf16 rounding of w*H^-0.5 (its
index_bf16_head_gate) is NOT applied; in an fp32 model all of this is plain fp32. Rows with at most
index_topk visible keys select every visible key, so the layer is then exactly Moonshot's dense causal
MLA. There are no always-kept (sink/local) tokens. index_topk_freq = 1 and index_skip_topk_offset = 1
mean every DSA layer runs its own indexer; any other index_topk_freq, or index_rope_dim != 0, is refused.

Ties are possible: a key with q.k <= 0 for every head scores exactly 0, and the cut can fall inside such a
group (a 4-head version of the tiny fixture hit it on 24 rows). The engines leave tie-breaking unspecified
(set semantics), and torch.topk breaks ties differently for rows of different length, which made prefill
and decode select different keys. This oracle therefore breaks ties deterministically: among equal scores
the EARLIER key position is kept (a stable descending sort). That rule is a property of this oracle, not of
K3.1: a golden whose logits depend on it would fail an implementation that is correct. The default
fixture's test sequences are therefore picked tie-free, and selections are compared as sets with a tie
tolerance at the cut. The recorded trace carries the full score rows and the cut margin for that.
"""
from __future__ import annotations

import functools

import torch
import torch.nn as nn
import torch.nn.functional as F

NEG_INF = float("-inf")


def dsa_layers_0idx(text_config: dict) -> list[int]:
    """Validate text_config.dsa_config the way SGLang's _init_dsa does and return the 0-indexed DSA layers."""
    dsa = text_config.get("dsa_config")
    if not dsa:
        raise ValueError("text_config has no dsa_config: this is not a Kimi K3.1 config")
    if dsa.get("index_topk_freq", 1) != 1:
        raise ValueError(f"index_topk_freq must be 1 (every DSA layer runs its own indexer), got "
                         f"{dsa.get('index_topk_freq')}")
    if dsa.get("index_rope_dim", 0) != 0:
        raise ValueError(f"index_rope_dim must be 0 (NoPE indexer), got {dsa.get('index_rope_dim')}")
    if dsa.get("index_k_norm_type", "layernorm") != "layernorm":
        raise ValueError(f"index_k_norm_type must be 'layernorm', got {dsa.get('index_k_norm_type')}")
    full = (text_config.get("linear_attn_config") or {}).get("full_attn_layers")
    if sorted(dsa.get("dsa_layers") or []) != sorted(full or []):
        raise ValueError(f"dsa_layers {dsa.get('dsa_layers')} must equal full_attn_layers {full}")
    for key in ("index_topk", "index_head_dim", "index_num_heads"):
        if not isinstance(dsa.get(key), int) or dsa[key] <= 0:
            raise ValueError(f"dsa_config.{key} must be a positive integer, got {dsa.get(key)}")
    return sorted(i - 1 for i in dsa["dsa_layers"])          # config lists are 1-indexed


class LightningIndexer(nn.Module):
    """Parameter names match the checkpoint: self_attn.indexer.{wq_b,wk,k_norm,weights_proj}."""

    def __init__(self, hidden_size: int, q_lora_rank: int, n_heads: int, head_dim: int, topk: int,
                 k_norm_eps: float):
        super().__init__()
        self.n_heads, self.head_dim, self.topk, self.k_norm_eps = n_heads, head_dim, topk, k_norm_eps
        self.wq_b = nn.Linear(q_lora_rank, n_heads * head_dim, bias=False)
        self.wk = nn.Linear(hidden_size, head_dim, bias=False)
        self.k_norm = nn.LayerNorm(head_dim, eps=k_norm_eps)            # weight AND bias
        self.weights_proj = nn.Linear(hidden_size, n_heads, bias=False)
        self.weight_scale = head_dim ** -0.5 * n_heads ** -0.5          # vLLM dsa.py:370 (two rounded factors)
        self.trace: list | None = None                                  # set to a list to record each call

    @staticmethod
    def _ct(x: torch.Tensor) -> torch.dtype:
        return torch.float64 if x.dtype == torch.float64 else torch.float32

    def index_keys(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,hidden] -> [B,T,D]: LayerNorm in fp32 with weight and bias, cast back to x's dtype."""
        ct = self._ct(x)
        k = self.wk(x)
        k = F.layer_norm(k.to(ct), (self.head_dim,), self.k_norm.weight.to(ct), self.k_norm.bias.to(ct),
                         self.k_norm_eps)
        return k.to(x.dtype)

    def scores(self, x: torch.Tensor, qr: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """x [B,T,hidden], qr [B,T,q_lora], keys [B,S,D] -> raw scores [B,T,S] (no mask yet)."""
        B, T, _ = x.shape
        ct = self._ct(x)
        q = self.wq_b(qr).view(B, T, self.n_heads, self.head_dim)
        w = self.weights_proj(x).to(ct) * self.weight_scale                     # [B,T,H]
        dots = torch.einsum("bthd,bsd->bths", q.to(ct), keys.to(ct)).relu()    # relu BEFORE weighting
        return (dots * w[..., None]).sum(2)                                     # [B,T,S]

    def select(self, score: torch.Tensor, q_pos: torch.Tensor):
        """Causal -inf mask, then top-k among the visible keys. Returns (masked score, keep [B,T,S] bool)."""
        S = score.shape[-1]
        k_pos = torch.arange(S, device=score.device)
        visible = k_pos[None, :] <= q_pos[:, None]                              # [T,S]
        score = score.masked_fill(~visible, NEG_INF)
        # top-k with ties broken toward the earlier key (stable sort keeps index order among equal scores)
        idx = score.sort(dim=-1, descending=True, stable=True).indices[..., :min(self.topk, S)]
        keep = torch.zeros(score.shape, dtype=torch.bool, device=score.device)  # rows with < k visible keys
        keep.scatter_(-1, idx, True)                                            # pick some -inf ones ...
        return score, keep & visible                                            # ... which this drops


def _fold_mask(attention_mask, keep: torch.Tensor, dtype: torch.dtype):
    """Combine the model's causal mask with the selection. keep: [B,T,S] (True = attend)."""
    S = keep.shape[-1]
    if attention_mask is None:
        return torch.zeros(keep[:, None].shape, dtype=dtype, device=keep.device).masked_fill(
            ~keep[:, None], NEG_INF)
    if attention_mask.dim() != 4:
        raise ValueError(f"expected a 4-D attention mask for MLA, got {tuple(attention_mask.shape)}")
    m = attention_mask[..., :S]
    if m.dtype == torch.bool:
        return m & keep[:, None]
    return m + torch.zeros(keep[:, None].shape, dtype=m.dtype, device=m.device).masked_fill(~keep[:, None],
                                                                                           NEG_INF)


def _dsa_forward(self, orig_forward, hidden_states, attention_mask=None, position_ids=None, past_key_values=None,
                 **kwargs):
    """KimiMLAAttention.forward with the indexer's selection folded into its attention mask."""
    B, T, _ = hidden_states.shape
    # The same modules, on the same input, that Moonshot's forward applies (modeling_kimi_linear.py:419):
    qr = self.q_a_layernorm(self.q_a_proj(hidden_states))
    k_new = self.indexer.index_keys(hidden_states)
    past = 0
    if past_key_values is not None:
        store = past_key_values.__dict__.setdefault("_k31_index_keys", {})
        prev = store.get(self.layer_idx)
        past = 0 if prev is None else prev.shape[1]
        mla_past = past_key_values.get_seq_length(self.layer_idx)
        if past != mla_past:
            raise RuntimeError(f"layer {self.layer_idx}: indexer cache has {past} keys, MLA cache {mla_past}")
        keys = k_new if prev is None else torch.cat([prev, k_new], dim=1)
        store[self.layer_idx] = keys
    else:
        keys = k_new
    q_pos = torch.arange(past, past + T, device=hidden_states.device)
    score, keep = self.indexer.select(self.indexer.scores(hidden_states, qr, keys), q_pos)
    if self.indexer.trace is not None:
        self.indexer.trace.append({"q_pos": q_pos.detach().cpu(), "score": score.detach().cpu(),
                                   "keep": keep.detach().cpu(), "index_k": k_new.detach().cpu(),
                                   "qr": qr.detach().cpu()})
    mask = _fold_mask(attention_mask, keep, hidden_states.dtype)
    return orig_forward(hidden_states, attention_mask=mask, position_ids=position_ids,
                        past_key_values=past_key_values, **kwargs)


def attach(model, text_config: dict, mla_class, k_norm_eps: float) -> list[int]:
    """Add an indexer to every DSA layer of a Moonshot KimiLinearForCausalLM and route its MLA through it.
    Returns the 0-indexed DSA layers. The indexer parameters are then part of model.state_dict()
    (model.layers.N.self_attn.indexer.*) and load with the rest of the checkpoint."""
    layers = dsa_layers_0idx(text_config)
    dsa = text_config["dsa_config"]
    mla = [i for i, layer in enumerate(model.model.layers) if isinstance(layer.self_attn, mla_class)]
    if mla != layers:
        raise ValueError(f"DSA layers {layers} != the model's MLA layers {mla}")
    for i in layers:
        attn = model.model.layers[i].self_attn
        if attn.q_lora_rank is None:
            raise ValueError("the K3.1 indexer reads the q_a latent; q_lora_rank must be set")
        attn.indexer = LightningIndexer(text_config["hidden_size"], attn.q_lora_rank, dsa["index_num_heads"],
                                        dsa["index_head_dim"], dsa["index_topk"], k_norm_eps)
        orig = functools.partial(type(attn).forward, attn)
        attn.forward = functools.partial(_dsa_forward, attn, orig)
    return layers

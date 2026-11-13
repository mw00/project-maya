"""tests/inkling2_real_check.py - the real Inkling-2-Small through ref/inkling2.py, against transformers on the original
checkpoint: the same text through both, their log-probabilities compared, and a greedy answer to a chat prompt.

    PYTHONPATH=<transformers main>:<llama.cpp>/gguf-py python tests/inkling2_real_check.py \
        --src ~/Models/Inkling-2-Small --gguf ~/Models/Inkling-2-Small/maya-bf16-00001-of-00009.gguf \
        [--tokens 256] [--answer 24] [--gpu 0] [--skip-ref]

transformers' side: its own decoder layers (models/inkling2), one at a time on the GPU in fp32, every dense weight
the checkpoint's; the experts decoded from the checkpoint's NVFP4 (e2m1 x e4m3 x amax / 2688) and MXFP8 (e4m3 x
2^(e - 127)) when a token picks them - every value exact in fp32. The reference's side: the GGUF tools/inkling2_gguf.py
wrote from the same checkpoint, in float64 on the CPU (~40 s a layer at 256 tokens). The answer is transformers'
(greedy, the whole sequence again each token: ~3 min a token, the weights read once a pass).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

from transformers import AutoConfig, AutoTokenizer
from transformers.models.inkling2 import modeling_inkling2 as M

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ref"))
import inkling2 as R  # noqa: E402

E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]
# (written for this test: no one's text, and no passage repeated - a repeat is copied, and its perplexity says nothing)
TEXT = ("The ferry left the north bank at six, as it had every morning since the bridge washed out. Its pilot, a "
        "woman who had learned the river from her father, read the current by the color of the water: brown where "
        "the silt ran fast, green in the slack pools behind the gravel bars. Most passengers were farmers carrying "
        "eggs and onions to the market on the far side, but on Tuesdays a schoolteacher crossed with a crate of "
        "borrowed books, and on Fridays the doctor came over to see the patients who could not make the trip. The "
        "town council had argued for three years about rebuilding the bridge. One side wanted a steel span that "
        "would carry trucks; the other wanted a footbridge that the town could pay for without a loan. In the end "
        "the river decided for them. A spring flood took the last of the old pilings, and the engineers who came to "
        "survey the site said the bed had moved forty meters to the east. Any bridge would have to start over on "
        "new ground. So the ferry kept running. The pilot added a second trip at noon, then a third at dusk, and "
        "the market on the far side began to stay open later to meet the last boat. Children who grew up in those "
        "years remember the crossing more than the town: the smell of diesel and wet rope, the dog that slept on "
        "the coil of the bow line, the way the pilot let them take the wheel for a minute in the middle of the "
        "river, where the water was deepest and nothing could go wrong. When the new bridge finally opened, a "
        "decade later and two kilometers downstream, the ferry made one last crossing with the whole town on "
        "board, and nobody paid.")
QUESTION = "In two sentences: why does a short convolution before attention help a language model?"

# the checkpoint's names -> transformers' text model's (conversion_mapping.py "inkling2_text", in order)
RENAMES = [
    (r"model\.llm\.unembed\.weight", "lm_head.weight"), (r"model\.llm\.", "model."),
    (r"embed\.weight", "embed_tokens.weight"), (r"embed_norm\.", "embed_tokens.embed_norm."),
    (r"mlp\.gate\.bias", "mlp.gate.e_score_correction_bias"),
    (r"attn\.wq_du", "self_attn.q_proj"), (r"attn\.wk_dv", "self_attn.k_proj"), (r"attn\.wv_dv", "self_attn.v_proj"),
    (r"attn\.wr_du", "self_attn.r_proj"), (r"attn\.wo_ud", "self_attn.o_proj"),
    (r"attn\.gate_attn", "self_attn.gate_proj"), (r"attn\.attn_sink_proj", "self_attn.sink_proj"),
    (r"\.attn\.(q_l2_clip|k_l2_clip|rel_l2_clip|k_sconv|v_sconv|rel_logits_proj)\.", r".self_attn.\1."),
    (r"attn_sconv\.", "post_attention_convolution."), (r"mlp_sconv\.", "post_mlp_convolution."),
    (r"(convolution|sconv)\.weight(?!_)", r"\1.conv1d.weight"), (r"(convolution|sconv)\.static_w", r"\1.static_weight"),
    (r"(convolution|sconv)\.weight_proj", r"\1.weight_proj.weight"),
    (r"mlp_norm", "post_attention_layernorm"), (r"attn_norm", "input_layernorm"),
    (r"(q|k|v)_input_norm", r"\1_input_layernorm"),
]


def rename(k):
    m = re.fullmatch(r"model\.llm\.attn_res\.attn_queries\.(\d+)", k)
    if m:
        return f"model.layers.{int(m[1]) + 1}.attention_input.query"
    m = re.fullmatch(r"model\.llm\.attn_res\.mlp_queries\.(\d+)", k)
    if m:
        return f"model.layers.{m[1]}.mlp_input.query"
    if k == "model.llm.attn_res.output_query":
        return "model.final_input.query"
    for pat, rep in RENAMES:
        k = re.sub(pat, rep, k)
    return k


class Src:
    def __init__(self, path):
        self.path = path
        self.where = json.load(open(os.path.join(path, "model.safetensors.index.json")))["weight_map"]
        self.files = {}

    def get(self, k):
        f = self.where[k]
        if f not in self.files:
            self.files[f] = safe_open(os.path.join(self.path, f), "pt")
        return self.files[f].get_tensor(k)

    def layer_keys(self, L):
        p = f"model.llm.layers.{L}."
        ks = [k for k in self.where if k.startswith(p) and ".mlp.experts." not in k]
        if L > 0:
            ks.append(f"model.llm.attn_res.attn_queries.{L - 1}")
        return ks + [f"model.llm.attn_res.mlp_queries.{L}"]


class StreamExperts(torch.nn.Module):
    """Inkling2Experts' forward over the checkpoint's expert bytes, an expert decoded when a token picks it"""

    def __init__(self, src, L, dev, act_fn):
        super().__init__()
        p = f"model.llm.layers.{L}.mlp.experts."
        self.nv = src.get(p + "w13_weight").dtype == torch.uint8
        self.t = {}
        for proj in ("w13", "w2"):
            self.t[proj] = src.get(p + proj + "_weight").to(dev)
            self.t[proj + "_s"] = src.get(p + proj + "_weight_scale").to(dev)
            if self.nv:
                self.t[proj + "_o"] = (src.get(p + proj + "_weight_amax").float() /
                                       src.get(p + proj + "_weight_outer_scale_denominator").float()).to(dev)
        self.lut = torch.tensor(E2M1, device=dev)
        self.act_fn = act_fn

    def weight(self, proj, e):
        w, s = self.t[proj][e], self.t[proj + "_s"][e]
        if self.nv:
            codes = torch.stack([w & 0x0F, w >> 4], -1).reshape(w.shape[0], -1).long()
            return self.lut[codes] * s.float().repeat_interleave(16, dim=1) * self.t[proj + "_o"][e]
        return w.float() * torch.exp2(s.view(torch.uint8).float() - 127).repeat_interleave(32, dim=1)

    def forward(self, x, top_k_index, top_k_weights):
        out = torch.zeros_like(x, dtype=torch.float32)
        for e in torch.unique(top_k_index).tolist():
            tok, pos = torch.where(top_k_index == e)
            gu = F.linear(x[tok], self.weight("w13", e))
            h = self.act_fn(gu[:, 0::2], gu[:, 1::2])      # (w13's rows: gate even, up odd)
            out.index_add_(0, tok, F.linear(h, self.weight("w2", e)) * top_k_weights[tok, pos, None])
        return out


def tf_forward(src, cfg, ids, dev):
    """transformers' forward over the whole sequence, a layer at a time: log-probabilities over the real tokens"""
    T = len(ids)
    emb = M.Inkling2NormedEmbedding(cfg.vocab_size, cfg.hidden_size, cfg.pad_token_id, cfg.rms_norm_eps)
    emb.weight.data = src.get("model.llm.embed.weight").float()
    emb.embed_norm.weight.data = src.get("model.llm.embed_norm.weight").float()
    h = emb(torch.tensor(ids)[None]).to(dev)
    del emb
    blocks = h.new_zeros(1, T, 0, cfg.hidden_size)
    pos = torch.arange(T, device=dev)
    d = pos[:, None] - pos[None, :]
    masks = {}
    for sliding in (True, False):
        ok = (d >= 0) & ((d < cfg.sliding_window_size) if sliding else True)
        masks[sliding] = torch.zeros(T, T, device=dev).masked_fill(~ok, torch.finfo(torch.float32).min)[None, None]
    for L in range(cfg.num_hidden_layers):
        sliding = cfg.layer_types[L] == "hybrid_sliding"
        layer = (M.Inkling2SlidingDecoderLayer if sliding else M.Inkling2FullDecoderLayer)(cfg, L)
        act = layer.mlp.experts.act_fn
        layer.mlp.experts = torch.nn.Identity()   # (transformers' own fp32 experts would be 49 GB: streamed instead)
        layer = layer.to(dev).float().eval()
        sd = {rename(k)[len(f"model.layers.{L}."):]: src.get(k).to(dev, torch.float32) for k in src.layer_keys(L)}
        missing, unexpected = layer.load_state_dict(sd, strict=False)
        missing = [m for m in missing if "e_score_correction_bias" not in m]
        assert not missing and not unexpected, (missing, unexpected)
        layer.mlp.gate.e_score_correction_bias = src.get(f"model.llm.layers.{L}.mlp.gate.bias").float().to(dev)
        layer.mlp.experts = StreamExperts(src, L, dev, act)
        new = layer(h, blocks, attention_mask=masks[sliding], conv_mask=None, past_key_values=None,
                    position_ids=pos[None])
        if layer.is_block_boundary:
            blocks = torch.cat([blocks, h.unsqueeze(-2)], dim=-2)
        h = new
        del layer
        torch.cuda.empty_cache()
    final = M.Inkling2BlockAttentionResidual(cfg).to(dev)
    final.query.data = src.get("model.llm.attn_res.output_query").float().to(dev)
    norm = M.Inkling2RMSNorm(cfg.hidden_size, cfg.rms_norm_eps).to(dev)
    norm.weight.data = src.get("model.llm.norm.weight").float().to(dev)
    h = norm(final(blocks, h)[0])[0]
    head = src.get("model.llm.unembed.weight")
    logits = torch.cat([h @ head[i:i + 50000].to(dev).float().T for i in range(0, head.shape[0], 50000)], -1)
    logits = torch.tanh(logits / cfg.final_logit_softcapping) * cfg.final_logit_softcapping
    return torch.log_softmax(logits[:, :cfg.unpadded_vocab_size].double(), -1).cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--gguf", required=True)
    ap.add_argument("--tokens", type=int, default=256)
    ap.add_argument("--answer", type=int, default=24)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--skip-ref", action="store_true")
    a = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.set_grad_enabled(False)
    dev = torch.device(f"cuda:{a.gpu}")
    cfg = AutoConfig.from_pretrained(a.src).text_config
    cfg._attn_implementation = "eager"
    src = Src(a.src)
    tok = AutoTokenizer.from_pretrained(a.src)

    ids = tok(TEXT, add_special_tokens=False)["input_ids"][:a.tokens]
    t0 = time.time()
    want = tf_forward(src, cfg, ids, dev)
    t1 = time.time()
    nll = lambda lp: -lp[:-1].gather(-1, torch.tensor(ids[1:])[:, None]).mean().item()  # noqa: E731
    print(f"transformers: {len(ids)} tokens in {t1 - t0:.0f} s; perplexity {math.exp(nll(want)):.3f}, next token "
          f"right {(want[:-1].argmax(-1) == torch.tensor(ids[1:])).double().mean() * 100:.1f}%", flush=True)
    if not a.skip_ref:
        got = R.Ref(a.gguf).forward(ids)
        t2 = time.time()
        kl = (want.exp() * (want - got)).sum(-1)
        print(f"reference:    {len(ids)} tokens in {t2 - t1:.0f} s; perplexity {math.exp(nll(got)):.3f}; against "
              f"transformers max |dlogp| {(got - want).abs().max():.2e}, mean KL {kl.mean():.2e}, max KL "
              f"{kl.max():.2e}, top-1 the same {(got.argmax(-1) == want.argmax(-1)).double().mean() * 100:.1f}%",
              flush=True)

    if a.answer:
        # (the template opens no model turn: the model writes its own, <|message_model|> first. Effort "none": the
        # answer without a long thinking first; the turn ends at <|content_model_end_sampling|>)
        text = tok.apply_chat_template([{"role": "user", "content": QUESTION}], tokenize=False,
                                       reasoning_effort="none")
        chat = tok(text, add_special_tokens=False)["input_ids"]
        stop = {tok.convert_tokens_to_ids("<|content_model_end_sampling|>"), tok.eos_token_id}
        out = []
        for _ in range(a.answer):
            nxt = int(tf_forward(src, cfg, chat + out, dev)[-1].argmax())
            out.append(nxt)
            print(f"answer so far: {tok.decode(out)!r}", flush=True)
            if nxt in stop:
                break


if __name__ == "__main__":
    main()

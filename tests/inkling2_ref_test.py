"""tests/inkling2_ref_test.py - ref/inkling2.py against transformers' own Inkling2ForCausalLM, end to end: a small random
Inkling 2 written as the checkpoint is (its names, NVFP4 experts for layers 0-9 with a scale an expert, MXFP8 for
10-11, w13's gate and up rows interleaved, the dense weights BF16), converted by tools/inkling2_gguf.py, read back by
the reference - whose log-probabilities must be transformers' within fp32's rounding.

    PYTHONPATH=<transformers main>:<llama.cpp>/gguf-py python tests/inkling2_ref_test.py

300 tokens against a window and a relative extent of 64 and a log-scaling floor of 100: every edge of the
attention crossed; layers 5 and 11 full (the dynamic convs, three residual queries); blocks of 6 (a boundary at 0
and 6). Skipped (exit 0) without a transformers that has inkling2 (its main from 109a7d19bf, 2026-11-12).
"""
import json
import pathlib
import re
import subprocess
import sys
import tempfile

import torch

ROOT = pathlib.Path(__file__).resolve().parent.parent
try:
    from transformers.models.inkling2 import configuration_inkling2 as CI
    from transformers.models.inkling2 import modeling_inkling2 as M
except ImportError:
    print("[inkling2_ref_test] skipped: no transformers with inkling2")
    sys.exit(0)
from safetensors.torch import save_file  # noqa: E402

sys.path.insert(0, str(ROOT / "ref"))
import inkling2 as R  # noqa: E402

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
TC = dict(hidden_size=256, num_hidden_layers=12, num_attention_heads=4, num_key_value_heads=2, head_dim=64,
          swa_num_attention_heads=4, swa_num_key_value_heads=2, swa_head_dim=64, n_routed_experts=16,
          num_experts_per_tok=4, intermediate_size=128, moe_intermediate_size=128, vocab_size=1024,
          unpadded_vocab_size=1000, sliding_window_size=64, rel_extent=64, log_scaling_n_floor=100,
          d_rel=16, sconv_kernel_size=4, n_layers_per_attn_res_block=6, route_scale=8.0, mlp_gate_softcap=4.0,
          mlp_up_softcap=25.0, final_logit_softcapping=30.0, qk_l2_clip_threshold=22.627416997969522,
          rel_l2_clip_threshold=8.0, log_scaling_alpha=0.1, rms_norm_eps=1e-6, model_max_length=1048576,
          use_value_norm_no_gamma=True, attn_res_normalize_values=True, n_shared_experts=0, eos_token_id=1)
N_NV = 10   # layers 0..9 NVFP4, the rest MXFP8 (the real model: 0-39, 40-47)


def nvfp4(W):
    """[R, K] -> the checkpoint's NVFP4: codes [R, K/2] (value 2i low nibble), e4m3 scales [R, K/16], amax"""
    amax = W.abs().max()
    outer = amax / 2688.0
    blk = W.reshape(W.shape[0], -1, 16)
    s = (blk.abs().amax(-1) / 6.0 / outer).clamp(min=2 ** -9).to(torch.float8_e4m3fn)
    x = blk / (s.float()[..., None] * outer)
    idx = (x.abs()[..., None] - E2M1).abs().argmin(-1)
    code = (idx | ((x < 0).long() << 3)).reshape(W.shape[0], -1)
    return (code[:, 0::2] | (code[:, 1::2] << 4)).to(torch.uint8), s, amax


def nvfp4_values(w, s, amax):
    lut = torch.cat([E2M1, -E2M1])
    lut[8] = 0.0
    codes = torch.stack([w & 0x0F, w >> 4], -1).reshape(w.shape[0], -1).long()
    return lut[codes] * s.float().repeat_interleave(16, 1) * (amax / 2688.0)


def mxfp8(W):
    blk = W.reshape(W.shape[0], -1, 32)
    e = torch.ceil(torch.log2(blk.abs().amax(-1).clamp(min=1e-30) / 448.0)).clamp(-120, 120)
    v = (blk / torch.exp2(e)[..., None]).to(torch.float8_e4m3fn).reshape(W.shape)
    return v, (e + 127).to(torch.uint8)


def mxfp8_values(v, s):
    return v.float() * torch.exp2(s.float() - 127).repeat_interleave(32, 1)


ATTN = {"q_proj.weight": "wq_du.weight", "k_proj.weight": "wk_dv.weight", "v_proj.weight": "wv_dv.weight",
        "r_proj.weight": "wr_du.weight", "o_proj.weight": "wo_ud.weight", "gate_proj.weight": "gate_attn.weight",
        "sink_proj.weight": "attn_sink_proj.weight", "k_sconv.conv1d.weight": "k_sconv.weight",
        "v_sconv.conv1d.weight": "v_sconv.weight", "k_sconv.static_weight": "k_sconv.static_w",
        "v_sconv.static_weight": "v_sconv.static_w", "k_sconv.weight_proj.weight": "k_sconv.weight_proj",
        "v_sconv.weight_proj.weight": "v_sconv.weight_proj"}
LAYER = {"post_attention_convolution.conv1d.weight": "attn_sconv.weight",
         "post_mlp_convolution.conv1d.weight": "mlp_sconv.weight", "input_layernorm.weight": "attn_norm.weight",
         "post_attention_layernorm.weight": "mlp_norm.weight", "q_input_layernorm.weight": "q_input_norm.weight",
         "k_input_layernorm.weight": "k_input_norm.weight", "v_input_layernorm.weight": "v_input_norm.weight",
         "mlp.gate.weight": "mlp.gate.weight", "mlp.gate.e_score_correction_bias": "mlp.gate.bias"}


def checkpoint_name(n):
    """transformers' text-model name -> the checkpoint's (conversion_mapping.py "inkling2_text", inverted)"""
    top = {"lm_head.weight": "unembed.weight", "model.embed_tokens.weight": "embed.weight",
           "model.embed_tokens.embed_norm.weight": "embed_norm.weight", "model.norm.weight": "norm.weight",
           "model.final_input.query": "attn_res.output_query"}
    if n in top:
        return "model.llm." + top[n]
    L, rest = re.fullmatch(r"model\.layers\.(\d+)\.(.+)", n).groups()
    L = int(L)
    if rest == "attention_input.query":
        return f"model.llm.attn_res.attn_queries.{L - 1}"
    if rest == "mlp_input.query":
        return f"model.llm.attn_res.mlp_queries.{L}"
    if rest.startswith("self_attn."):
        a = rest[len("self_attn."):]
        return f"model.llm.layers.{L}.attn." + ATTN.get(a, a)
    return f"model.llm.layers.{L}." + LAYER[rest]


def main():
    torch.manual_seed(0)
    out = pathlib.Path(tempfile.mkdtemp(prefix="inkling2_ref_"))
    cfg = CI.Inkling2TextConfig(**TC)
    cfg._attn_implementation = "eager"
    model = M.Inkling2ForCausalLM(cfg).float().eval()
    sd = {}
    with torch.no_grad():
        for n, p in model.named_parameters():   # (random everywhere: the zero-initialized queries and convs too)
            if "norm" in n and p.dim() == 1:
                p.copy_(1 + 0.1 * torch.randn_like(p))
            else:
                p.copy_(torch.randn_like(p) * (0.5 if p.dim() == 1 else 1 / p.shape[-1] ** 0.5))
        for layer in model.model.layers:
            layer.mlp.gate.e_score_correction_bias.copy_(0.01 * torch.randn(TC["n_routed_experts"]))
        for n, t in model.state_dict().items():
            if ".mlp.experts." in n:
                continue
            k = checkpoint_name(n)
            keep_f32 = "attn_res" in k or k.endswith("gate.bias")
            sd[k] = t.clone() if keep_f32 else t.to(torch.bfloat16)
            if not keep_f32:   # (the model the BF16 values the checkpoint holds)
                t.copy_(sd[k].float())
        for L, layer in enumerate(model.model.layers):
            ex, p = layer.mlp.experts, f"model.llm.layers.{L}.mlp.experts."
            gu, dn = ex.gate_up_proj.data, ex.down_proj.data
            ff = gu.shape[1] // 2
            w13 = torch.stack([gu[:, :ff], gu[:, ff:]], 2).reshape(gu.shape)   # (gate row i at 2i, up at 2i + 1)
            dec = {}
            for proj, W in (("w13", w13), ("w2", dn)):
                if L < N_NV:
                    parts = [nvfp4(W[e]) for e in range(W.shape[0])]
                    sd[p + proj + "_weight"] = torch.stack([x[0] for x in parts])
                    sd[p + proj + "_weight_scale"] = torch.stack([x[1] for x in parts])
                    sd[p + proj + "_weight_amax"] = torch.stack([x[2] for x in parts]).float()
                    sd[p + proj + "_weight_outer_scale_denominator"] = torch.tensor([2688.0])
                    dec[proj] = torch.stack([nvfp4_values(*x) for x in parts])
                else:
                    parts = [mxfp8(W[e]) for e in range(W.shape[0])]
                    sd[p + proj + "_weight"] = torch.stack([x[0] for x in parts])
                    sd[p + proj + "_weight_scale"] = torch.stack([x[1] for x in parts])
                    dec[proj] = torch.stack([mxfp8_values(*x) for x in parts])
            gu.copy_(torch.cat([dec["w13"][:, 0::2], dec["w13"][:, 1::2]], 1))   # (the model the decoded values)
            dn.copy_(dec["w2"])
    save_file({k: v.contiguous() for k, v in sd.items()}, str(out / "model-00001-of-00001.safetensors"))
    json.dump({"weight_map": {k: "model-00001-of-00001.safetensors" for k in sd}},
              open(out / "model.safetensors.index.json", "w"))
    json.dump({"architectures": ["Inkling2ForConditionalGeneration"], "text_config": TC},
              open(out / "config.json", "w"))
    subprocess.run([sys.executable, str(ROOT / "tools" / "inkling2_gguf.py"), "--src", str(out), "--out",
                    str(out / "tiny")], check=True, stderr=subprocess.DEVNULL)
    ids = torch.randint(0, TC["unpadded_vocab_size"], (300,))
    with torch.no_grad():
        want = torch.log_softmax(model(input_ids=ids[None]).logits[0].double(), -1)
    got = R.Ref(str(out / "tiny.gguf")).forward(ids.tolist())
    d = (got - want).abs().max().item()
    kl = (want.exp() * (want - got)).sum(-1).abs().max().item()
    top1 = (got.argmax(-1) == want.argmax(-1)).double().mean().item()
    print(f"[inkling2_ref_test] 300 tokens: max |dlogp| {d:.2e}, max |KL| {kl:.2e}, top-1 {top1 * 100:.1f}%")
    ok = d < 1e-4 and kl < 1e-6 and top1 == 1.0
    print("[inkling2_ref_test]", "ok" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

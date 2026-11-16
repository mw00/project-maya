"""tools/strata/glm_dsa_hf_xcheck.py - ref/glm_dsa.py (hf mode) vs transformers in float64 (A-46).

BUILD-PLAN B3.1 acceptance: "transformers f64 on a tiny checkpoint agrees within 1e-9".  This
script builds a transformers GlmMoeDsaForCausalLM (model_type glm_moe_dsa, transformers >= 5.15)
from a synthetic glm-dsa GGUF's geometry, loads the GGUF's weights into it (dequantized), runs
both models teacher-forced on the same tokens in float64 and reports the max logit difference.

STATUS: written in a container without torch or transformers (BUILD-PLAN rule: do not install
torch there), so it has NOT been run.  The HF parameter names below come from GLM-5.3-SPEC.md and
llama.cpp's tensor_mapping/conversion code (the inverse mapping); the expert layout is detected
from the instantiated model's state dict (per-expert modules, or transformers 5's fused
gate_up_proj/down_proj).  Any name or shape it cannot place is reported, not guessed.

    pip install torch 'transformers>=5.15'
    python tools/glm_dsa_synth_gguf.py --preset SMALL --out /tmp/small.gguf
    python tools/strata/glm_dsa_hf_xcheck.py /tmp/small.gguf [--tokens 24] [--tol 1e-9]
"""
from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "ref"))

from glm_dsa import GlmDsa, Options  # noqa: E402


def hf_config(c):
    from transformers import AutoConfig
    types = ["full" if f else "shared" for f in c.idx_full]
    return AutoConfig.for_model(
        "glm_moe_dsa",
        vocab_size=c.n_vocab, hidden_size=c.n_embd, num_hidden_layers=c.n_layer,
        num_attention_heads=c.n_head, num_key_value_heads=c.n_head,
        q_lora_rank=c.q_lora, kv_lora_rank=c.kv_lora, qk_nope_head_dim=c.qk_nope,
        qk_rope_head_dim=c.n_rot, v_head_dim=c.v_mla, head_dim=c.n_rot,
        rope_parameters={"rope_theta": c.rope_base, "rope_type": "default"},
        rope_interleave=True, indexer_rope_interleave=True,
        max_position_embeddings=max(c.n_ctx_train, 4096), rms_norm_eps=c.rms_eps,
        index_topk=c.top_k, index_n_heads=c.idx_heads, index_head_dim=c.idx_dim,
        indexer_types=types, first_k_dense_replace=c.dense_lead,
        mlp_layer_types=["dense" if i < c.dense_lead else "sparse" for i in range(c.n_layer)],
        intermediate_size=c.n_ff, moe_intermediate_size=c.n_ff_exp,
        n_routed_experts=c.n_expert, num_experts_per_tok=c.n_used, n_shared_experts=c.n_shared,
        n_group=1, topk_group=1, norm_topk_prob=c.w_norm, routed_scaling_factor=c.w_scale,
        scoring_func="sigmoid", topk_method="noaux_tc", tie_word_embeddings=False,
        num_nextn_predict_layers=0, attention_bias=False, hidden_act="silu",
    )


def gguf_to_hf(m: GlmDsa) -> dict[str, np.ndarray]:
    """GGUF tensors -> HF names (per-expert form; fused forms are assembled by load())."""
    c = m.cfg
    out = {"model.embed_tokens.weight": m.W("token_embd.weight"),
           "model.norm.weight": m.W("output_norm.weight"),
           "lm_head.weight": m.W("output.weight")}
    for il in range(c.n_layer):
        p, L = f"model.layers.{il}.", (lambda b, s="weight": m.L(il, b, s))
        out[p + "input_layernorm.weight"] = L("attn_norm")
        out[p + "post_attention_layernorm.weight"] = L("ffn_norm")
        a = p + "self_attn."
        out[a + "q_a_proj.weight"] = L("attn_q_a")
        out[a + "q_a_layernorm.weight"] = L("attn_q_a_norm")
        out[a + "q_b_proj.weight"] = L("attn_q_b")
        out[a + "kv_a_proj_with_mqa.weight"] = L("attn_kv_a_mqa")
        out[a + "kv_a_layernorm.weight"] = L("attn_kv_a_norm")
        kb, vb = L("attn_k_b"), L("attn_v_b")           # (H, kv, nope), (H, v, kv)
        out[a + "kv_b_proj.weight"] = np.concatenate([kb.transpose(0, 2, 1), vb], axis=1).reshape(-1, c.kv_lora)
        out[a + "o_proj.weight"] = L("attn_output")
        if c.idx_full[il]:
            x = a + "indexer."
            out[x + "wq_b.weight"] = L("indexer.attn_q_b")
            out[x + "wk.weight"] = L("indexer.attn_k")
            out[x + "k_norm.weight"] = L("indexer.k_norm")
            out[x + "k_norm.bias"] = L("indexer.k_norm", "bias")
            out[x + "weights_proj.weight"] = L("indexer.proj")
        f = p + "mlp."
        if il < c.dense_lead:
            out[f + "gate_proj.weight"] = L("ffn_gate")
            out[f + "up_proj.weight"] = L("ffn_up")
            out[f + "down_proj.weight"] = L("ffn_down")
        else:
            out[f + "gate.weight"] = L("ffn_gate_inp")
            out[f + "gate.e_score_correction_bias"] = L("exp_probs_b", "bias")
            for e in range(c.n_expert):
                out[f + f"experts.{e}.gate_proj.weight"] = m.expert(il, "ffn_gate_exps", e)
                out[f + f"experts.{e}.up_proj.weight"] = m.expert(il, "ffn_up_exps", e)
                out[f + f"experts.{e}.down_proj.weight"] = m.expert(il, "ffn_down_exps", e)
            out[f + "shared_experts.gate_proj.weight"] = L("ffn_gate_shexp")
            out[f + "shared_experts.up_proj.weight"] = L("ffn_up_shexp")
            out[f + "shared_experts.down_proj.weight"] = L("ffn_down_shexp")
    return out


def load(model, src: dict[str, np.ndarray], n_expert: int) -> list[str]:
    """Copy src into the model's state dict; returns the HF keys left unset."""
    import torch
    sd = model.state_dict()
    done, problems = set(), []
    for k, t in sd.items():
        if k in src:
            v = src[k]
        elif k.endswith("experts.gate_up_proj") or k.endswith("experts.down_proj"):
            base = k.rsplit("experts.", 1)[0] + "experts."
            if k.endswith("gate_up_proj"):
                g = np.stack([src[f"{base}{e}.gate_proj.weight"] for e in range(n_expert)])
                u = np.stack([src[f"{base}{e}.up_proj.weight"] for e in range(n_expert)])
                v = np.concatenate([g, u], axis=1)              # (E, 2ff, hidden)
            else:
                v = np.stack([src[f"{base}{e}.down_proj.weight"] for e in range(n_expert)])
            if tuple(v.shape) != tuple(t.shape) and tuple(v.transpose(0, 2, 1).shape) == tuple(t.shape):
                v = v.transpose(0, 2, 1)
        else:
            if "rotary" not in k and "inv_freq" not in k:
                problems.append(f"unset: {k} {tuple(t.shape)}")
            continue
        if tuple(v.shape) != tuple(t.shape):
            problems.append(f"shape: {k} model {tuple(t.shape)} gguf {v.shape}")
            continue
        with torch.no_grad():
            t.copy_(torch.from_numpy(np.ascontiguousarray(v)))
        done.add(k)
    model.load_state_dict(sd)
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("gguf")
    ap.add_argument("--tokens", type=int, default=24)
    ap.add_argument("--tol", type=float, default=1e-9)
    ap.add_argument("--rope-angles", choices=("hf", "f64"), default="f64",
                    help="oracle angle model: f64 when transformers computes RoPE in the model dtype, "
                         "hf when it keeps float32 inv_freq/angles (then expect about 1e-6)")
    a = ap.parse_args()
    try:
        import torch
        from transformers import AutoModelForCausalLM
    except ImportError as e:
        print(f"glm_dsa_hf_xcheck: needs torch and transformers>=5.15 ({e})")
        return 2
    m = GlmDsa(a.gguf, Options(mode="hf", rope_angles=a.rope_angles), cache=True)
    c = m.cfg
    toks = [(i * 37 + 11) % c.n_vocab for i in range(a.tokens)]
    ours = m.forward(toks)
    model = AutoModelForCausalLM.from_config(hf_config(c), torch_dtype=torch.float64,
                                             attn_implementation="eager")
    problems = load(model, gguf_to_hf(m), c.n_expert)
    if problems:
        print("\n".join(problems))
        return 1
    model.eval()
    with torch.no_grad():
        hf = model(torch.tensor([toks])).logits[0].double().numpy().T
    d = float(np.abs(hf - ours).max())
    print(f"max |logit(hf) - logit(oracle)| = {d:.3g} over {a.tokens} tokens (tol {a.tol:g})")
    return 0 if d <= a.tol else 1


if __name__ == "__main__":
    sys.exit(main())

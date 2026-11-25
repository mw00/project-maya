"""tools/ling_quant/ling_map.py - the Ling-3.2-air FP8 checkpoint (inclusionAI, block-FP8 safetensors) read tensor by
tensor, mapped onto llama.cpp's bailingmoe3 GGUF names, shapes and layouts (conversion/bailingmoe3.py's transforms),
in fp32 from the exact dequantisation (FP8 x its 128 x 128 block scale).

    ck = Checkpoint("/path/to/Ling-3.2-air")
    for name, make in gguf_tensors(ck): w = make()   # float32 torch tensor, GGUF numpy order

`python ling_map.py verify <model dir> <BF16 GGUF> [layers]` compares every mapped tensor of those layers (and the
globals) with the BF16 GGUF's: experts must be equal, everything else within BF16 rounding; a wrong transform shows as
a cosine far from 1.  `python ling_map.py names <model dir> <BF16 GGUF>` checks the two name sets are the same.
"""
from __future__ import annotations

import os
import pathlib
import sys

import ling_paths as P  # noqa: E402  (this folder's path defaults)

import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent / "maya_quant"))
from hf_map import Checkpoint  # noqa: E402,F401

P = "model."


def text(ck: Checkpoint) -> dict:
    return ck.text


def is_full_attention(T: dict, il: int) -> bool:
    n, g = int(T["num_hidden_layers"]), int(T["layer_group_size"])
    return il >= n or (il + 1) % g == 0 or il >= n // g * g


def gguf_tensors(ck: Checkpoint, layers=None, experts: bool = True, mtp: bool = True):
    """(gguf name, thunk -> float32 tensor in GGUF numpy order) for every text tensor (or those of `layers`)."""
    T = ck.text
    n_head = int(T["num_attention_heads"])
    nope, vdim = int(T["qk_nope_head_dim"]), int(T["v_head_dim"])
    n_exp = int(T["num_experts"])
    out = []
    if layers is None:
        out += [("token_embd.weight", lambda: ck.get(P + "word_embeddings.weight")),
                ("output_norm.weight", lambda: ck.get(P + "norm.weight")),
                ("output.weight", lambda: ck.get("lm_head.weight"))]
        layers = range(int(T["num_hidden_layers"]) + (int(T.get("num_nextn_predict_layers", 0)) if mtp else 0))
    for il in layers:
        L = f"{P}layers.{il}."
        B = f"blk.{il}."

        def g(n, L=L):
            return lambda: ck.get(L + n)
        out += [(B + "attn_norm.weight", g("input_layernorm.weight")),
                (B + "ffn_norm.weight", g("post_attention_layernorm.weight"))]
        if not is_full_attention(T, il):   # KDA
            def conv(c, L=L):
                return lambda: ck.get(L + f"attention.{c}_conv1d.weight").reshape(1, -1, 1, int(T["short_conv_kernel_size"]))
            out += [(B + "attn_q.weight", g("attention.q_proj.weight")),
                    (B + "attn_k.weight", g("attention.k_proj.weight")),
                    (B + "attn_v.weight", g("attention.v_proj.weight")),
                    (B + "ssm_conv1d_q.weight", conv("q")),
                    (B + "ssm_conv1d_k.weight", conv("k")),
                    (B + "ssm_conv1d_v.weight", conv("v")),
                    (B + "ssm_f_a.weight", g("attention.f_proj.weight")),
                    (B + "ssm_beta.weight", g("attention.b_proj.weight")),
                    (B + "ssm_a", lambda L=L: torch.exp(ck.get(L + "attention.A_log")).reshape(-1, 1)),
                    (B + "ssm_dt.bias", g("attention.dt_bias")),
                    (B + "ssm_g_a.weight", g("attention.g_proj.weight")),
                    (B + "ssm_norm.weight", g("attention.o_norm.weight")),
                    (B + "attn_output.weight", g("attention.o_proj.weight"))]
        else:   # gated MLA
            def kv_b(part, L=L):
                def f():
                    w = ck.get(L + "attention.kv_b_proj.weight")
                    w = w.view(n_head, nope + vdim, w.shape[-1])
                    if part == "k":
                        return w[:, :nope, :].transpose(1, 2).contiguous()   # [n_head, kv_lora, nope]
                    return w[:, nope:, :].contiguous()                          # [n_head, v, kv_lora]
                return f
            out += [(B + "attn_q.weight", g("attention.q_proj.weight")),
                    (B + "attn_kv_a_mqa.weight", g("attention.kv_a_proj_with_mqa.weight")),
                    (B + "attn_kv_a_norm.weight", g("attention.kv_a_layernorm.weight")),
                    (B + "attn_k_b.weight", kv_b("k")),
                    (B + "attn_v_b.weight", kv_b("v")),
                    (B + "attn_gate.weight", g("attention.g_proj.weight")),
                    (B + "attn_output.weight", g("attention.dense.weight"))]
        if il < int(T["first_k_dense_replace"]):
            out += [(B + "ffn_gate.weight", g("mlp.gate_proj.weight")),
                    (B + "ffn_up.weight", g("mlp.up_proj.weight")),
                    (B + "ffn_down.weight", g("mlp.down_proj.weight"))]
        else:
            out += [(B + "ffn_gate_inp.weight", g("mlp.gate.weight")),
                    (B + "exp_probs_b.bias", g("mlp.gate.expert_bias")),
                    (B + "ffn_gate_shexp.weight", g("mlp.shared_experts.gate_proj.weight")),
                    (B + "ffn_up_shexp.weight", g("mlp.shared_experts.up_proj.weight")),
                    (B + "ffn_down_shexp.weight", g("mlp.shared_experts.down_proj.weight"))]
            if experts:
                for role in ("gate", "up", "down"):
                    out.append((B + f"ffn_{role}_exps.weight",
                                lambda L=L, role=role: torch.stack(
                                    [ck.get(f"{L}mlp.experts.{e}.{role}_proj.weight") for e in range(n_exp)])))
        if ck.has(L + "eh_proj.weight"):   # the NextN (MTP) layer
            out += [(B + "nextn.eh_proj.weight", g("eh_proj.weight")),
                    (B + "nextn.enorm.weight", g("enorm.weight")),
                    (B + "nextn.hnorm.weight", g("hnorm.weight")),
                    (B + "layer_output_norm.weight", g("final_layernorm.weight"))]
    return out


def _readers(first: str):
    import re
    sys.path.insert(0, P.GGUF_PY)
    import gguf
    paths = sorted(pathlib.Path(first).parent.glob(re.sub(r"-00001-of-", "-*-of-", pathlib.Path(first).name)))
    return gguf, [gguf.GGUFReader(str(p)) for p in paths]


def _names(model: str, bf16: str):
    gguf, readers = _readers(bf16)
    theirs = {t.name for r in readers for t in r.tensors}
    mine = {n for n, _ in gguf_tensors(Checkpoint(model))}
    print(f"mine {len(mine)}, the GGUF's {len(theirs)}")
    for n in sorted(mine - theirs):
        print("  only mine:", n)
    for n in sorted(theirs - mine):
        print("  only the GGUF's:", n)
    print("SAME" if mine == theirs else "DIFFERENT")


def _verify(model: str, bf16: str, layers):
    import numpy as np
    gguf, readers = _readers(bf16)
    by = {t.name: t for r in readers for t in r.tensors}
    ck = Checkpoint(model)
    worst, worst_rel = 1.0, 0.0
    items = gguf_tensors(ck)
    want = {f"blk.{il}." for il in layers}
    for name, make in items:
        if name.startswith("blk.") and not any(name.startswith(w) for w in want):
            continue
        t = by.get(name)
        if t is None:
            print(f"{name}: NOT IN THE GGUF")
            continue
        mine = make().float()
        ref = torch.from_numpy(np.ascontiguousarray(gguf.quants.dequantize(np.asarray(t.data), t.tensor_type),
                                                     dtype=np.float32))
        ref = ref.reshape(tuple(reversed([int(v) for v in t.shape])))
        if tuple(mine.shape) != tuple(ref.shape):
            print(f"{name}: SHAPE {tuple(mine.shape)} vs {tuple(ref.shape)}   <-- CHECK")
            continue
        a, b = mine.reshape(-1).double(), ref.reshape(-1).double()
        cos = float((a @ b) / (a.norm() * b.norm() + 1e-30))
        rel = float(((a - b).abs() / a.abs().clamp(min=1e-30))[a != 0].max()) if (a != 0).any() else 0.0
        exact = bool((a.to(torch.bfloat16).double() == b).all())
        worst, worst_rel = min(worst, cos), max(worst_rel, rel)
        flag = "" if cos > 0.9999 and rel < 5e-3 else "   <-- CHECK"
        print(f"{name:36s} {str(tuple(mine.shape)):26s} {t.tensor_type.name:5s} cos {cos:.7f} max rel {rel:.1e} "
              f"{'= bf16(mine)' if exact else ''}{flag}")
    print(f"worst cosine {worst:.7f}, worst relative difference {worst_rel:.2e}")


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "verify":
        ls = [int(x) for x in sys.argv[4].split(",")] if len(sys.argv) > 4 else [0, 2, 5, 41, 42]
        _verify(sys.argv[2], sys.argv[3], ls)
    elif len(sys.argv) >= 4 and sys.argv[1] == "names":
        _names(sys.argv[2], sys.argv[3])
    else:
        print(__doc__)

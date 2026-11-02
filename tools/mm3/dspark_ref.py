#!/usr/bin/env python3
# DSpark draft reference (torch, CPU) + fixture generator for the Metal port.
# Implements the algorithm of src/mm3/mm3_dspark.hpp/.cu:
#   context: ctx = sum_j tap_j @ fc_j^T (taps rounded to BF16 first), xn = rms(ctx, hidden_norm+1);
#            per layer K/V = xn @ W^T (MXFP8 dequantized exactly), K per-head RMS norm (k_norm+1) +
#            partial NeoX RoPE, stored in a K/V cache at absolute positions.
#   propose: block [anchor, mask x (B-1)] embeddings at positions P..P+B-1; per layer: rms(input_norm+1),
#            Q/K/V, Q/K per-head norm + RoPE, block K/V into the cache, attention (not causal) over the
#            last `window` positions ending at P+B-1, o-proj residual, rms(post_norm+1), swigluoai MLP,
#            down residual. Final rms(final_norm+1) -> target head logits; sequential Markov chain:
#            pick_i = argmax(logit_i + W2 @ W1[prev]); conf_i = sigmoid(cw . [hid_i ; W1[prev]] + cb).
# Norms in the safetensors are raw (the +1 is applied at use, Gemma style). Activations F32.
import json, math, sys
from pathlib import Path

import torch

PRE = "language_model.model.dspark."


def e4m3_dec(b):
    b = b.to(torch.int32)
    sign = torch.where((b & 0x80) != 0, -1.0, 1.0)
    exp = (b >> 3) & 0xF
    man = (b & 0x7).to(torch.float32)
    val = torch.where(exp == 0, man / 8.0 * (2.0 ** -6),
                      (1.0 + man / 8.0) * torch.pow(2.0, (exp - 7).to(torch.float32)))
    return sign * val


def mx_dequant(w_u8, s_u8):
    if w_u8.dtype != torch.uint8:
        w_u8 = w_u8.view(torch.uint8)
    N, K = w_u8.shape
    v = e4m3_dec(w_u8)
    scale = torch.pow(2.0, s_u8.to(torch.float32) - 127.0)
    scale = scale.repeat_interleave(32, dim=1)
    return (v * scale).to(torch.bfloat16).to(torch.float32)


def mx_quant(x):
    # float [N][K] -> (E4M3 bytes, UE8M0 bytes), per-32-block scaling (fixture generation)
    N, K = x.shape
    xb = x.reshape(N, K // 32, 32)
    amax = xb.abs().amax(dim=2).clamp(min=1e-12)
    e = torch.ceil(torch.log2(amax / 448.0)).clamp(-127, 127)
    scale = torch.pow(2.0, e)
    v = (xb / scale.unsqueeze(2)).reshape(N, K)
    sign = (v < 0).to(torch.uint8) << 7
    a = v.abs()
    sub = a < 2.0 ** -6
    exp = torch.where(sub, torch.tensor(-7.0), torch.floor(torch.log2(a.clamp(min=2.0 ** -6))))
    man = torch.where(sub, torch.round(a / (2.0 ** -6) * 8.0),
                      torch.round((a / torch.pow(2.0, exp) - 1.0) * 8.0))
    carry = man == 8.0
    man = torch.where(carry, torch.tensor(0.0), man)
    exp = torch.where(carry, exp + 1.0, exp)
    code = (((exp + 7).clamp(0, 15).to(torch.uint8)) << 3) | man.to(torch.uint8)
    return sign | code, (e + 127.0).to(torch.uint8)


def rms(x, w1, eps):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps) * w1


def rope(x, pos, n_rot, theta):
    half = n_rot // 2
    inv = theta ** (-torch.arange(half).float() * 2.0 / n_rot)
    ang = pos.float().unsqueeze(-1) * inv
    cos, sin = torch.cos(ang), torch.sin(ang)
    a, b = x[..., :half], x[..., half:n_rot]
    out = x.clone()
    out[..., :half] = a * cos - b * sin
    out[..., half:n_rot] = b * cos + a * sin
    return out


def swigluoai(g, u, alpha, limit):
    g = g.clamp(max=limit)
    u = u.clamp(min=-limit, max=limit)
    return g * torch.sigmoid(alpha * g) * (u + 1.0)


class Draft:
    def __init__(self, tensors, cfg):
        self.t = tensors
        tc = cfg.get("text_config", cfg)
        self.D = tc["hidden_size"]
        self.V = tc["vocab_size"]
        self.n_layers = tc["num_hidden_layers"]
        self.n_head = tc["num_attention_heads"]
        self.n_kv = tc["num_key_value_heads"]
        self.hd = tc["head_dim"]
        self.n_rot = int(tc.get("rotary_dim", round(tc["head_dim"] * tc.get("partial_rotary_factor", 1.0))))
        self.eps = tc["rms_norm_eps"]
        self.theta = float(tc["rope_theta"])
        self.alpha = tc["swiglu_alpha"]
        self.limit = tc["swiglu_limit"]
        self.B = tc["dspark_block_size"]
        self.R = tc["dspark_markov_rank"]
        self.mask = tc["dspark_noise_token_id"]
        self.window = tc.get("sliding_window", 4096)
        self.taps = tc["dspark_target_layer_ids"]
        self.NT = len(self.taps)
        self.kc = [None] * self.n_layers
        self.vc = [None] * self.n_layers

    def W(self, name):
        return self.t[PRE + name].float()

    def mx(self, name):
        return mx_dequant(self.t[PRE + name + ".weight"], self.t[PRE + name + ".weight_scale_inv"])

    def _grow(self, l, upto):
        if self.kc[l] is None or self.kc[l].shape[0] < upto:
            grow = max(upto, 1024)
            nk = torch.zeros(grow, self.n_kv * self.hd)
            nv = torch.zeros(grow, self.n_kv * self.hd)
            if self.kc[l] is not None:
                nk[:self.kc[l].shape[0]] = self.kc[l]
                nv[:self.vc[l].shape[0]] = self.vc[l]
            self.kc[l], self.vc[l] = nk, nv

    def context(self, taps, pos0):
        NT, T, D = taps.shape
        ctx = torch.zeros(T, D)
        fc = self.W("fc.weight")
        for j in range(NT):
            r = taps[j].to(torch.bfloat16).to(torch.float32)
            ctx = ctx + r @ fc[:, j * D:(j + 1) * D].t()
        xn = rms(ctx, self.W("hidden_norm.weight") + 1.0, self.eps)
        pos = torch.arange(pos0, pos0 + T)
        for l in range(self.n_layers):
            P = f"layers.{l}.decoder_layer.self_attn."
            k = (xn @ self.mx(P + "k_proj").t()).reshape(T, self.n_kv, self.hd)
            v = (xn @ self.mx(P + "v_proj").t()).reshape(T, self.n_kv * self.hd)
            k = rope(rms(k, self.W(P + "k_norm.weight") + 1.0, self.eps),
                     pos.unsqueeze(1), self.n_rot, self.theta).reshape(T, self.n_kv * self.hd)
            self._grow(l, pos0 + T)
            self.kc[l][pos0:pos0 + T] = k
            self.vc[l][pos0:pos0 + T] = v
        return xn

    def propose(self, emb, head, anchor, P, n):
        D, B = self.D, self.B
        h = torch.stack([emb[anchor].float()] + [emb[self.mask].float()] * (B - 1))
        pos = torch.arange(P, P + B)
        hi = P + B - 1
        lo = max(0, hi - self.window + 1)
        for l in range(self.n_layers):
            P_ = f"layers.{l}.decoder_layer."
            xn = rms(h, self.W(P_ + "input_layernorm.weight") + 1.0, self.eps)
            q = (xn @ self.mx(P_ + "self_attn.q_proj").t()).reshape(B, self.n_head, self.hd)
            k = (xn @ self.mx(P_ + "self_attn.k_proj").t()).reshape(B, self.n_kv, self.hd)
            v = (xn @ self.mx(P_ + "self_attn.v_proj").t()).reshape(B, self.n_kv * self.hd)
            q = rope(rms(q, self.W(P_ + "self_attn.q_norm.weight") + 1.0, self.eps),
                     pos.unsqueeze(1), self.n_rot, self.theta)
            k = rope(rms(k, self.W(P_ + "self_attn.k_norm.weight") + 1.0, self.eps),
                     pos.unsqueeze(1), self.n_rot, self.theta)
            self._grow(l, P + B)
            self.kc[l][P:P + B] = k.reshape(B, self.n_kv * self.hd)
            self.vc[l][P:P + B] = v.reshape(B, self.n_kv * self.hd)
            T = hi + 1 - lo
            K = self.kc[l][lo:hi + 1].reshape(T, self.n_kv, self.hd).repeat_interleave(self.n_head // self.n_kv, dim=1)
            V = self.vc[l][lo:hi + 1].reshape(T, self.n_kv, self.hd).repeat_interleave(self.n_head // self.n_kv, dim=1)
            s = torch.einsum("bhd,thd->bht", q, K) / math.sqrt(self.hd)
            a = torch.softmax(s, dim=-1)
            att = torch.einsum("bht,thd->bhd", a, V).reshape(B, self.n_head * self.hd)
            h = h + att @ self.mx(P_ + "self_attn.o_proj").t()
            xn = rms(h, self.W(P_ + "post_attention_layernorm.weight") + 1.0, self.eps)
            gt = xn @ self.mx(P_ + "mlp.gate_proj").t()
            up = xn @ self.mx(P_ + "mlp.up_proj").t()
            h = h + swigluoai(gt, up, self.alpha, self.limit) @ self.mx(P_ + "mlp.down_proj").t()
        hid = rms(h, self.W("final_norm.weight") + 1.0, self.eps)
        logits = hid.to(torch.bfloat16).to(torch.float32) @ head.float().t()
        w1 = self.W("markov_head.markov_w1.weight")
        w2 = self.W("markov_head.markov_w2.weight")
        cw = self.W("confidence_head.proj.weight").reshape(-1)
        cb = float(self.W("confidence_head.proj.bias").reshape(-1)[0])
        drafts, confs = [], []
        prev = anchor
        for i in range(n):
            e = w1[prev]
            pick = int(torch.argmax(logits[i] + w2 @ e))
            conf = float(torch.sigmoid(torch.cat([hid[i], e]) @ cw + cb))
            drafts.append(pick)
            confs.append(conf)
            prev = pick
        return drafts, confs, hid, logits


def load_st(path):
    from safetensors.torch import load_file
    return load_file(str(path))


def fixture(outdir):
    g = torch.Generator().manual_seed(20261101)
    rnd = lambda *s: torch.randn(*s, generator=g) * 0.05
    tc = {
        "hidden_size": 256, "vocab_size": 512, "num_hidden_layers": 2,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 64,
        "rotary_dim": 32, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
        "swiglu_alpha": 1.702, "swiglu_limit": 7.0,
        "dspark_block_size": 4, "dspark_markov_rank": 32, "dspark_noise_token_id": 509,
        "dspark_target_layer_ids": [1, 3], "sliding_window": 64,
        "dspark_ffn_hidden_size": 512, "intermediate_size": 512,
    }
    cfg = {"architectures": ["DSparkMiniMaxDraftModel"], "text_config": tc,
           "dspark_block_size": 4}
    D, V, R, B, FF = 256, 512, 32, 4, 512
    t = {}

    def bf16(name, x):
        t[PRE + name] = x.to(torch.bfloat16)

    def mx(name, x):
        w, s = mx_quant(x.to(torch.bfloat16).float())
        t[PRE + name + ".weight"] = w.view(torch.float8_e4m3fn)   # stored as F8_E4M3, like the real file
        t[PRE + name + ".weight_scale_inv"] = s

    bf16("fc.weight", rnd(D, 2 * D) * 4)
    bf16("hidden_norm.weight", rnd(D))
    bf16("final_norm.weight", rnd(D))
    bf16("markov_head.markov_w1.weight", rnd(V, R))
    bf16("markov_head.markov_w2.weight", rnd(V, R))
    bf16("confidence_head.proj.weight", rnd(1, D + R))
    bf16("confidence_head.proj.bias", rnd(1))
    for l in range(2):
        P = f"layers.{l}.decoder_layer."
        bf16(P + "input_layernorm.weight", rnd(D))
        bf16(P + "post_attention_layernorm.weight", rnd(D))
        bf16(P + "self_attn.q_norm.weight", rnd(64))
        bf16(P + "self_attn.k_norm.weight", rnd(64))
        mx(P + "self_attn.q_proj", rnd(256, D))
        mx(P + "self_attn.k_proj", rnd(64, D))
        mx(P + "self_attn.v_proj", rnd(64, D))
        mx(P + "self_attn.o_proj", rnd(D, 256))
        mx(P + "mlp.gate_proj", rnd(FF, D))
        mx(P + "mlp.up_proj", rnd(FF, D))
        mx(P + "mlp.down_proj", rnd(D, FF))
    emb = rnd(V, D).to(torch.bfloat16)
    head = rnd(V, D).to(torch.bfloat16)
    out = Path(outdir)
    (out / "dspark").mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    save_file(t, str(out / "dspark" / "model-dspark-00001.safetensors"), metadata={"format": "pt"})
    save_file({"emb": emb, "head": head}, str(out / "emb_head.safetensors"), metadata={"format": "pt"})
    (out / "dspark" / "config.json").write_text(json.dumps(cfg, indent=1))

    d = Draft(t, cfg)
    taps = torch.randn(2, 6, D, generator=g) * 0.5          # [NT][T][D], positions 0..5
    d.context(taps[:], 0)                                    # all 6 in one call here; the test splits 4+2
    drafts, confs, hid, logits = d.propose(emb, head, anchor=37, P=6, n=B)
    exp = {
        "taps": taps.tolist(), "anchor": 37, "pos": 6, "n": B,
        "drafts": drafts, "confs": confs,
        "hid": hid.tolist(), "logits": logits.tolist(),
    }
    (out / "expected.json").write_text(json.dumps(exp))
    import struct as _st
    blob = b"".join([
        taps.numpy().astype("float32").tobytes(),
        hid.numpy().astype("float32").tobytes(),
        logits.numpy().astype("float32").tobytes(),
        _st.pack("<%di" % B, *drafts),
        _st.pack("<%df" % B, *confs),
    ])
    (out / "expected.bin").write_bytes(blob)
    print("fixture written:", out, "drafts:", drafts, "confs:", [round(c, 4) for c in confs])


if __name__ == "__main__":
    if sys.argv[1] == "fixture":
        fixture(sys.argv[2])
    else:
        print("usage: dspark_ref.py fixture <dir>")

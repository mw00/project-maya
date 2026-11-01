"""tools/mm3_metal_fixture.py - the synthetic MiniMax-M3.1-Flash fixture for the Metal engine's parity
tests (src/metal/metal_mm3_test.cpp), in the spirit of tools/glm_big_fixture.py.

    python tools/mm3_metal_fixture.py --out build-metal/minimax-m3-synth.gguf [--seed 20261101]

Writes the GGUF (all F32: the oracle's arithmetic stays exact; the quantized expert types are covered
by the real-model run), the set's -maya-indexers.bin side file holding layer 0's indexer tensors (so
the loader's side-file path is exercised), and <out>.expected.bin with a numpy port of
tools/mm3/ref.py's forward over the same weights: the ids, the last position's logits, and the greedy
continuation.  The numpy oracle rounds k / v / iq / ik to BF16 before use, as the engines' caches do.
"""
from __future__ import annotations

import argparse
import math
import pathlib
import struct
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm_synth_gguf as G  # noqa: E402  (write_gguf + the kv_* encoders)

GEO = dict(dim=256, layers=3, dense_lead=1, n_head=8, n_kv=2, hd=128, n_rot=64, theta=1e7, eps=1e-6,
           vocab=512, dense_ff=384, exp_ff=128, shared_ff=128, n_exp=8, k_exp=4, scale=2.0,
           alpha=1.702, limit=7.0, ix_heads=2, ix_dim=128, blk=128, topk=16, local=1, ctx=1048576)
T = 2176        # 17 index blocks: the last queries must drop a block (topk is 16)
NGEN = 8


def bf16(x: np.ndarray) -> np.ndarray:
    u = x.astype(np.float32).view(np.uint32)
    u = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return u.view(np.float32)


def gemma_rms(x: np.ndarray, stored: np.ndarray, eps: float) -> np.ndarray:
    # ref.py's gemma_rms with w = stored - 1: x * rsqrt(mean x^2 + eps) * (1 + w) == * stored
    return (x * (1.0 / np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + eps)) * stored).astype(np.float32)


def rope_tables(n_rot: int, theta: float, pos: np.ndarray):
    inv = 1.0 / (theta ** (np.arange(0, n_rot, 2, dtype=np.float32) / n_rot))
    fr = pos.astype(np.float32)[:, None] * inv[None, :]
    emb = np.concatenate([fr, fr], -1)
    return np.cos(emb).astype(np.float32), np.sin(emb).astype(np.float32)


def rope(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    r = cos.shape[-1]
    xr, xp = x[..., :r], x[..., r:]
    h = r // 2
    rot = np.concatenate([-xr[..., h:], xr[..., :h]], -1)
    c, s = cos[:, None, :], sin[:, None, :]
    return np.concatenate([xr * c + rot * s, xp], -1).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def swigluoai(g: np.ndarray, u: np.ndarray, alpha: float, limit: float) -> np.ndarray:
    g = np.minimum(g, limit)
    u = np.clip(u, -limit, limit)
    return (g * sigmoid((g * alpha).astype(np.float32)) * (u + 1.0)).astype(np.float32)


def msa_select(iq: np.ndarray, ik: np.ndarray, pos: np.ndarray, blk: int, topk: int, local: int):
    """Kept blocks [H, Tq, topk] (-1 padded), the numpy twin of ref.py's msa_select."""
    Tq, H, _ = iq.shape
    Tk = ik.shape[0]
    nblk = -(-Tk // blk)
    sc = np.einsum("qhd,kd->hqk", iq, ik).astype(np.float32)
    future = np.arange(Tk)[None, None, :] > pos[None, :, None]
    sc = np.where(future, -np.inf, sc)
    pad = nblk * blk - Tk
    if pad:
        sc = np.pad(sc, ((0, 0), (0, 0), (0, pad)), constant_values=-np.inf)
    bs = sc.reshape(H, Tq, nblk, blk).max(-1)
    if local > 0:
        qb = pos // blk
        for j in range(local):
            li = np.clip(qb - j, 0, None)
            bs[np.arange(H)[:, None], np.arange(Tq)[None, :], li[None, :]] = np.inf
    k = min(topk, nblk)
    ix = np.argsort(-bs, axis=-1, kind="stable")[..., :k]
    v = np.take_along_axis(bs, ix, axis=-1)
    ix = np.where(v == -np.inf, -1, ix)
    if k < topk:
        ix = np.pad(ix, ((0, 0), (0, 0), (0, topk - k)), constant_values=-1)
    return ix


def msa_attend(q, k, v, kept, pos, blk, scale):
    Tq, Hq, D = q.shape
    Tk, Hkv, _ = k.shape
    Gr = Hq // Hkv
    nblk = -(-Tk // blk)
    keep = np.zeros((Hkv, Tq, nblk + 1), dtype=bool)
    kk = np.where(kept < 0, nblk, kept)
    np.put_along_axis(keep, kk, True, axis=-1)
    keep = np.repeat(keep[..., :nblk], blk, axis=-1)[..., :Tk]
    keep &= ~(np.arange(Tk)[None, None, :] > pos[None, :, None])
    qg = q.reshape(Tq, Hkv, Gr, D).transpose(1, 2, 0, 3)          # [Hkv, G, Tq, D]
    s = (qg @ k.transpose(1, 2, 0)[:, None]) * scale              # [Hkv, G, Tq, Tk]
    s = np.where(keep[:, None], s, -np.inf)
    s = s - s.max(-1, keepdims=True)
    p = np.exp(s)
    p = p / p.sum(-1, keepdims=True)
    o = p @ v.transpose(1, 0, 2)[:, None]                          # [Hkv, G, Tq, D]
    return o.transpose(2, 0, 1, 3).reshape(Tq, Hq, D).astype(np.float32)


def build_weights(seed: int):
    rng = np.random.default_rng(seed)
    g = GEO

    def mat(out: int, inn: int) -> np.ndarray:
        return (rng.standard_normal((out, inn), dtype=np.float32) / math.sqrt(inn)).astype(np.float32)

    def norm(n: int) -> np.ndarray:   # the GGUF stores 1 + w
        return (1.0 + 0.1 * rng.standard_normal(n, dtype=np.float32)).astype(np.float32)

    w: dict[str, np.ndarray] = {}
    side: dict[str, np.ndarray] = {}
    w["token_embd.weight"] = (0.1 * rng.standard_normal((g["vocab"], g["dim"]), dtype=np.float32)).astype(np.float32)
    w["output_norm.weight"] = norm(g["dim"])
    w["output.weight"] = mat(g["vocab"], g["dim"])
    for l in range(g["layers"]):
        b = f"blk.{l}."
        w[b + "attn_norm.weight"] = norm(g["dim"])
        w[b + "ffn_norm.weight"] = norm(g["dim"])
        w[b + "attn_q.weight"] = mat(g["n_head"] * g["hd"], g["dim"])
        w[b + "attn_k.weight"] = mat(g["n_kv"] * g["hd"], g["dim"])
        w[b + "attn_v.weight"] = mat(g["n_kv"] * g["hd"], g["dim"])
        w[b + "attn_output.weight"] = mat(g["dim"], g["n_head"] * g["hd"])
        w[b + "attn_q_norm.weight"] = norm(g["hd"])
        w[b + "attn_k_norm.weight"] = norm(g["hd"])
        ix = {
            b + "indexer.q_proj.weight": mat(g["ix_heads"] * g["ix_dim"], g["dim"]),
            b + "indexer.k_proj.weight": mat(g["ix_dim"], g["dim"]),
            b + "indexer.q_norm.weight": norm(g["ix_dim"]),
            b + "indexer.k_norm.weight": norm(g["ix_dim"]),
        }
        (side if l == 0 else w).update(ix)   # layer 0's indexers ride in the side file, as in the real sets
        if l < g["dense_lead"]:
            w[b + "ffn_gate.weight"] = mat(g["dense_ff"], g["dim"])
            w[b + "ffn_up.weight"] = mat(g["dense_ff"], g["dim"])
            w[b + "ffn_down.weight"] = mat(g["dim"], g["dense_ff"])
        else:
            w[b + "ffn_gate_inp.weight"] = mat(g["n_exp"], g["dim"])
            w[b + "exp_probs_b.bias"] = (0.5 * rng.standard_normal(g["n_exp"], dtype=np.float32)).astype(np.float32)
            w[b + "ffn_gate_shexp.weight"] = mat(g["shared_ff"], g["dim"])
            w[b + "ffn_up_shexp.weight"] = mat(g["shared_ff"], g["dim"])
            w[b + "ffn_down_shexp.weight"] = mat(g["dim"], g["shared_ff"])
            w[b + "ffn_gate_exps.weight"] = np.stack([mat(g["exp_ff"], g["dim"]) for _ in range(g["n_exp"])])
            w[b + "ffn_up_exps.weight"] = np.stack([mat(g["exp_ff"], g["dim"]) for _ in range(g["n_exp"])])
            w[b + "ffn_down_exps.weight"] = np.stack([mat(g["dim"], g["exp_ff"]) for _ in range(g["n_exp"])])
    return w, side


def metadata() -> list:
    g = GEO
    p = "minimax-m3."
    return [
        ("general.architecture", G.kv_str("minimax-m3")),
        ("general.name", G.kv_str("MiniMax-M3.1-Flash (synthetic metal fixture)")),
        (p + "block_count", G.kv_u32(g["layers"])),
        (p + "embedding_length", G.kv_u32(g["dim"])),
        (p + "attention.head_count", G.kv_u32(g["n_head"])),
        (p + "attention.head_count_kv", G.kv_u32(g["n_kv"])),
        (p + "attention.key_length", G.kv_u32(g["hd"])),
        (p + "rope.dimension_count", G.kv_u32(g["n_rot"])),
        (p + "rope.freq_base", G.kv_f32(g["theta"])),
        (p + "attention.layer_norm_rms_epsilon", G.kv_f32(g["eps"])),
        (p + "feed_forward_length", G.kv_u32(g["dense_ff"])),
        (p + "expert_feed_forward_length", G.kv_u32(g["exp_ff"])),
        (p + "expert_count", G.kv_u32(g["n_exp"])),
        (p + "expert_used_count", G.kv_u32(g["k_exp"])),
        (p + "expert_weights_scale", G.kv_f32(g["scale"])),
        (p + "swiglu_alpha", G.kv_f32(g["alpha"])),
        (p + "swiglu_limit", G.kv_f32(g["limit"])),
        (p + "context_length", G.kv_u32(g["ctx"])),
        (p + "leading_dense_block_count", G.kv_u32(g["dense_lead"])),
        (p + "attention.indexer.head_count", G.kv_u32(g["ix_heads"])),
        (p + "attention.indexer.key_length", G.kv_u32(g["ix_dim"])),
        (p + "attention.indexer.top_k", G.kv_u32(g["topk"])),
        (p + "attention.indexer.block_size", G.kv_u32(g["blk"])),
        (p + "attention.indexer.local_blocks", G.kv_u32(g["local"])),
        ("tokenizer.ggml.eos_token_id", G.kv_u32(511)),
    ]


def forward(w: dict, side: dict, ids: np.ndarray) -> np.ndarray:
    """ref.py's forward over the fixture weights (whole sequence, last-position logits aside: all rows)."""
    g = GEO
    W = dict(w)
    W.update(side)
    Tn = len(ids)
    h = W["token_embd.weight"][ids].astype(np.float32)
    pos = np.arange(Tn)
    cos, sin = rope_tables(g["n_rot"], g["theta"], pos)
    scale = 1.0 / math.sqrt(g["hd"])
    for l in range(g["layers"]):
        b = f"blk.{l}."
        x = gemma_rms(h, W[b + "attn_norm.weight"], g["eps"])
        q = (x @ W[b + "attn_q.weight"].T).reshape(Tn, g["n_head"], g["hd"])
        k = (x @ W[b + "attn_k.weight"].T).reshape(Tn, g["n_kv"], g["hd"])
        v = (x @ W[b + "attn_v.weight"].T).reshape(Tn, g["n_kv"], g["hd"])
        q = rope(gemma_rms(q, W[b + "attn_q_norm.weight"], g["eps"]), cos, sin)
        k = rope(gemma_rms(k, W[b + "attn_k_norm.weight"], g["eps"]), cos, sin)
        k, v = bf16(k), bf16(v)
        iq = (x @ W[b + "indexer.q_proj.weight"].T).reshape(Tn, g["ix_heads"], g["ix_dim"])
        ik = (x @ W[b + "indexer.k_proj.weight"].T).reshape(Tn, 1, g["ix_dim"])
        iq = bf16(rope(gemma_rms(iq, W[b + "indexer.q_norm.weight"], g["eps"]), cos, sin))
        ik = bf16(rope(gemma_rms(ik, W[b + "indexer.k_norm.weight"], g["eps"]), cos, sin)[:, 0])
        kept = msa_select(iq, ik, pos, g["blk"], g["topk"], g["local"])
        o = msa_attend(q, k, v, kept, pos, g["blk"], scale)
        h = h + (o.reshape(Tn, -1) @ W[b + "attn_output.weight"].T)
        x = gemma_rms(h, W[b + "ffn_norm.weight"], g["eps"])
        if l < g["dense_lead"]:
            f = swigluoai(x @ W[b + "ffn_gate.weight"].T, x @ W[b + "ffn_up.weight"].T,
                          g["alpha"], g["limit"]) @ W[b + "ffn_down.weight"].T
        else:
            logits = x @ W[b + "ffn_gate_inp.weight"].T
            s = sigmoid(logits)
            top = np.argsort(-(s + W[b + "exp_probs_b.bias"]), axis=-1, kind="stable")[:, : g["k_exp"]]
            wt = np.take_along_axis(s, top, axis=-1)
            wt = wt / wt.sum(-1, keepdims=True)
            y = np.zeros_like(x)
            for j in range(g["k_exp"]):
                e = top[:, j]
                ge = W[b + "ffn_gate_exps.weight"][e]
                ue = W[b + "ffn_up_exps.weight"][e]
                de = W[b + "ffn_down_exps.weight"][e]
                hh = swigluoai(np.einsum("td,tfd->tf", x, ge), np.einsum("td,tfd->tf", x, ue),
                               g["alpha"], g["limit"])
                y += (np.einsum("tf,tdf->td", hh, de) * wt[:, j : j + 1]).astype(np.float32)
            sh = swigluoai(x @ W[b + "ffn_gate_shexp.weight"].T, x @ W[b + "ffn_up_shexp.weight"].T,
                           g["alpha"], g["limit"]) @ W[b + "ffn_down_shexp.weight"].T
            f = (y * g["scale"] + sh).astype(np.float32)
        h = h + f
    h = gemma_rms(h, W["output_norm.weight"], g["eps"])
    return (h @ W["output.weight"].T).astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", required=True, help="the fixture .gguf to write")
    ap.add_argument("--seed", type=int, default=20261101)
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    w, side = build_weights(a.seed)
    G.write_gguf(out, metadata(), sorted(w.items()))
    stem = out.with_suffix("")
    side_path = pathlib.Path(str(stem) + "-maya-indexers.bin")
    G.write_gguf(side_path, [("general.architecture", G.kv_str("minimax-m3"))], sorted(side.items()))

    rng = np.random.default_rng(a.seed + 1)
    ids = rng.integers(0, GEO["vocab"], size=T, dtype=np.int32)
    logits = forward(w, side, ids)
    greedy = []
    seq = ids.copy()
    cur = logits
    for _ in range(NGEN):
        nxt = int(np.argmax(cur[-1]))
        greedy.append(nxt)
        seq = np.append(seq, np.int32(nxt))
        cur = forward(w, side, seq)
    exp = pathlib.Path(str(out) + ".expected.bin")
    with exp.open("wb") as f:
        f.write(b"MM3EXP1\0")
        f.write(struct.pack("<III", T, NGEN, GEO["vocab"]))
        ids.astype("<i4").tofile(f)
        logits[-1].astype("<f4").tofile(f)
        np.array(greedy, dtype="<i4").tofile(f)
    print(f"{out}: {out.stat().st_size} bytes; side {side_path.name}; expected {exp.name} "
          f"(T={T}, greedy={greedy})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

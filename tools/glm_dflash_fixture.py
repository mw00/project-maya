"""tools/glm_dflash_fixture.py - a synthetic DFlash2 drafter GGUF (arch "dflash") that fits the 45-layer synthetic
glm5-next model (tools/glm_big_fixture.py: n_embd 256, vocab 512), for maya-metal's drafting test.  Random weights: the
drafts are mostly wrong, which is the point - drafting must still decode exactly as plain decoding does.  A small
sliding window (64) makes the drafter's KV ring wrap within the test.

    python tools/glm_dflash_fixture.py --out FILE
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_synth_gguf as S  # noqa: E402


def build(seed: int = 99):
    g = S.BIG
    e, v = g["n_embd"], g["n_vocab"]
    n_layer, n_head, n_head_kv, hd, ff = 2, 2, 1, 128, 512
    taps, group, rank, top_k, block, window = 2, 16, 32, 8, 8, 64
    targets = [3, 15, 30, 44]
    rng = np.random.default_rng(seed)

    def w(*shape, scale=0.05):
        return (rng.standard_normal(shape) * scale).astype(np.float32)

    def norm_w(k):
        return (1.0 + rng.standard_normal(k) * 0.02).astype(np.float32)

    md = [
        ("general.architecture", S.kv_str("dflash")),
        ("general.name", S.kv_str("glm5-dflash-synth")),
        ("general.alignment", S.kv_u32(S.ALIGN)),
        ("dflash.block_count", S.kv_u32(n_layer)),
        ("dflash.embedding_length", S.kv_u32(e)),
        ("dflash.feed_forward_length", S.kv_u32(ff)),
        ("dflash.attention.head_count", S.kv_u32(n_head)),
        ("dflash.attention.head_count_kv", S.kv_u32(n_head_kv)),
        ("dflash.attention.causal", S.kv_bool(False)),
        ("dflash.rope.freq_base", S.kv_f32(10000.0)),
        ("dflash.attention.layer_norm_rms_epsilon", S.kv_f32(1e-5)),
        ("dflash.attention.key_length", S.kv_u32(hd)),
        ("dflash.attention.value_length", S.kv_u32(hd)),
        ("dflash.block_size", S.kv_u32(block)),
        ("dflash.conv_kernel_size", S.kv_u32(taps)),
        ("dflash.conv_group_size", S.kv_u32(group)),
        ("dflash.selector_rank", S.kv_u32(rank)),
        ("dflash.selector_top_k", S.kv_u32(top_k)),
        ("dflash.target_layers", S.arr_i32(targets)),
        ("dflash.attention.sliding_window", S.kv_u32(window)),
        ("tokenizer.ggml.mask_token_id", S.kv_u32(v - 1)),
    ]
    k = len(targets)
    dyn = 2 * taps * e // group
    ts = [   # numpy (out, in): GGUF ne0 is the in dim
        ("fc.weight", w(e, k * e)),
        ("enc.output_norm.weight", norm_w(e)),
        ("output_norm.weight", norm_w(e)),
        ("selector_hidden.weight", w(rank, e)),
        ("selector_predecessor.weight", w(v, rank, scale=0.5)),
        ("selector_successor.weight", w(v, rank, scale=0.5)),
    ]
    for il in range(n_layer):
        p = f"blk.{il}."
        ts += [
            (p + "attn_norm.weight", norm_w(e)),
            (p + "ffn_norm.weight", norm_w(e)),
            (p + "attn_q_norm.weight", norm_w(hd)),
            (p + "attn_k_norm.weight", norm_w(hd)),
            (p + "attn_conv_base", w(2, taps, e, scale=0.5)),
            (p + "ffn_conv_base", w(2, taps, e, scale=0.5)),
            (p + "attn_conv_proj.weight", w(dyn, e)),
            (p + "ffn_conv_proj.weight", w(dyn, e)),
            (p + "attn_q.weight", w(n_head * hd, e)),
            (p + "attn_k.weight", w(n_head_kv * hd, e)),
            (p + "attn_v.weight", w(n_head_kv * hd, e)),
            (p + "attn_output.weight", w(e, n_head * hd)),
            (p + "ffn_gate.weight", w(ff, e)),
            (p + "ffn_up.weight", w(ff, e)),
            (p + "ffn_down.weight", w(e, ff)),
        ]
    return md, ts


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    md, ts = build()
    S.write_gguf(Path(a.out), md, ts)
    print(f"wrote {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

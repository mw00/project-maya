"""tools/kimi_k31_synth_gguf.py - a tiny, structurally complete synthetic Kimi K3.1 fixture (and a big one).

Phase 1 of a Kimi K3.1 port needs a model that the reference forward pass, an independent model
built from Moonshot's own modeling code, upstream llama.cpp and the future engine kernels can all
run without the 1.56 TB download. This tool draws ONE seeded set of weights and writes it three ways,
so every consumer reads the same numbers:

    <out>/tiny.gguf          general.architecture "kimi-k31", every tensor F32. The tensors and the
                             "kimi-k31." keys are exactly what llama.cpp's conversion/kimi_k3.py (the
                             kimi-k3 converter) writes for hf-f32 with `--outtype f32`, extended for
                             K3.1 the way a kimi-k31 subclass would: five indexer tensors per MLA layer
                             (blk.N.indexer.{attn_q_b,attn_k,k_norm,proj}) and four keys
                             (attention.indexer.{head_count,key_length,top_k},
                             attention.layer_norm_epsilon). Converting hf-mxfp4 gives the same tensors
                             with the routed experts as ggml MXFP4 (same values). general.name and
                             the dummy tokenizer are the fixture's own.
    <out>/hf-f32/            config.json + model.safetensors with the real checkpoint names
                             (language_model.model.layers.N...., language_model.lm_head.weight), all F32.
    <out>/hf-mxfp4/          the same values in the real checkpoint's dtypes: routed experts as
                             compressed-tensors "mxfp4-pack-quantized" (weight_packed U8 + weight_scale
                             U8 E8M0, group 32), A_log / dt_bias / conv / o_norm / e_score_correction_bias
                             F32, everything else BF16; config.json carries the released
                             quantization_config.
    <out>/manifest.json      seed, geometry, the text_config, sha256 and size of every file, the layer
                             layout and the test sequences (TEST_TOKENS: picked so that no top-k cut
                             lies near a tie; for the big geometry also test_sequences_pick, the pick's
                             bounds and smallest margins).
--geometry big writes big.gguf in place of tiny.gguf (below).

Bit-identity across the three: every BF16-stored weight is drawn on the bf16 grid, every expert weight
is an FP4 E2M1 code (never the -0 code) times an E8M0 power-of-two group scale, so the F32 files, the
BF16 upcast and the MXFP4 dequantization are the same float32 bits. `ssm_a = -exp(A_log)` is the one
computed value: A_log entries are drawn so that exp() in float64 lies within 0.05 ulp of a float32,
which makes any faithful float32 exp (numpy's, torch's) round to the same value the converter writes.

Fixture geometry (TINY, seed 3101). Released K3.1 values in brackets where they differ:
  * vocab 320 [163840], hidden 256 [7168], 9 layers [93], untied lm_head, max_position_embeddings
    4096 [1048576]; bos 64, eos 66 (<|end_of_msg|>), pad 319, media 85 (the real ids minus 163520).
  * Layers (0-indexed): 0-2 KDA, 3 MLA, 4-6 KDA, 7-8 MLA. Config lists are 1-indexed like the
    release: full_attn_layers = dsa_layers = [4, 8, 9], kda_layers = [1, 2, 3, 5, 6, 7]. The last two
    layers are both MLA (the release's 91, 92), with one more MLA layer earlier and KDA between.
  * Block attention residuals, attn_res_block_size 3 [12]: blocks pushed at layers 0, 3 and 6, so the
    stream crosses two block boundaries and the output mix sees 4 candidates [emb, B(0-2), B(3-5),
    partial(6-8)]. One MLA layer (3) and one KDA layer (6) sit on a boundary; MLA 7 and 8 do not.
  * KDA: 6 heads x head_dim 32 [96 x 128], short conv 4, gate_lower_bound -5.0, full-rank g_proj.
    The checkpoint stores A_log at head_dim length (32) with entries 6..31 exactly +0.0, as the real
    checkpoint does (128 entries, 96..127 zero); the GGUF ssm_a is sliced to linear_attn_config.num_heads.
    KDA heads equal the MLA heads, as released, and they have to: the GGUF format has a single
    attention.head_count, from which llama.cpp's kimi-k3 sizes both. So a converter or loader that
    slices A_log by num_attention_heads instead of linear_attn_config.num_heads cannot be told apart
    here; one that slices by any other count writes an ssm_a of the wrong length, which readers refuse.
  * MLA (NoPE, output gate): 6 heads [96], q_lora 160 [1536], kv_lora 512, qk_nope 128, qk_rope 64,
    v_head 128 - the real 576-wide kv_a_proj_with_mqa row and the real 192/128 head dims.
  * Lightning indexer on every MLA layer: 12 heads x 32 dims [32 x 128], index_topk 6 [2048],
    k_norm LayerNorm (weight + bias), index_rope_dim 0, index_topk_freq 1, index_skip_topk_offset 1.
    A 32-80 token sequence therefore selects 6 of up to 80 keys per query.
  * Sizes that a mix-up could swap are kept distinct: KDA head_dim 32 != qk_rope 64 (they differ in
    the release too, 128 vs 64), q_lora 160 != the 192-wide q head, 12 x 32 indexer q width 384 !=
    hidden 256. What still coincides does so as released (v_head = qk_nope = 128, KDA heads = MLA
    heads, routed_scaling_factor 1.0) or across unrelated modules (KDA head_dim = indexer head_dim 32,
    KDA width 6 x 32 = the 192-wide MLA q head).
  * Layer 0 dense SiTU-GLU MLP, intermediate 448 [33792] (first_k_dense_replace 1).
  * Layers 1-8 latent MoE: 24 routed experts [896], top-10 [16] (> 8 on purpose), 2 shared experts,
    moe_intermediate 64 [3072], routed_expert_hidden_size 128 [3584] < hidden, latent_moe_use_norm,
    noaux_tc sigmoid router with e_score_correction_bias, moe_renormalize, routed_scaling_factor 1.0.
  * SiTU beta 4.0, linear beta 25.0, rms_norm_eps 1e-5, as released. The transformers generation
    defaults the release carries in text_config (top_k 50, max_length 20, ...) are kept, so a consumer
    that mistakes top_k for the MoE top-k fails here too.
  * Weight scales are modest (std about 0.8/sqrt(fan_in), norm gains near 1), since float32 error in
    the reference grows with large weights.

Eps: the GGUF carries attention.layer_norm_rms_epsilon = attention.layer_norm_epsilon = 1e-5 (the
indexer k LayerNorm; engine consensus, Moonshot ships no K3.1 code). Moonshot's K3 code builds
q_a_layernorm / kv_a_layernorm with the RMSNorm class default 1e-6 while engines use 1e-5: the
fixture does not pick one, reference models must state theirs.

The vocabulary is a dummy (tok0..tok319, the special ids above as control tokens) so a llama.cpp
build that knows kimi-k31 can load tiny.gguf; the oracle runs on token ids. `--no-vocab` drops it. The HF folders carry
no tokenizer and no vision tower.

The big geometry (`--geometry big`, BIG, the same seed; ~2.9 GB written: big.gguf ~1.3 GB, hf-f32 ~1.3 GB,
hf-mxfp4 ~0.3 GB). The engine's fixture for the paths the tiny one cannot reach: i-quant rows, the
real top-16, an indexer that selects among hundreds of keys, KDA past several of fla's chunks.
  * Every row an i-quant reads is a multiple of 256 (QK_K): hidden 1024, latent 512, moe_intermediate
    256, q_lora 768, kv_lora 512, dense intermediate 1792, KDA and MLA output widths 8 x 128 = 1024,
    shared experts 2 x 256 = 512. The only rows that are not are the ones that are not in the release
    either: attn_k_b (qk_nope 128), ssm_f_b (KDA head_dim 128) and the conv taps (4). So a port's
    llama-quantize can write an IQ1_S copy with every routed expert, projection and head in an IQ type.
  * 64 routed experts, top-16 (the released k), 2 shared experts; 10 layers, 0-indexed MLA 3, 8, 9
    (full_attn_layers [4, 9, 10]), KDA 0-2 and 4-7; attn_res_block_size 3 pushes blocks at 0, 3, 6
    and 9, so the last layer opens a one-layer block and the output mix sees 5 candidates.
  * KDA 8 heads x 128 [96 x 128]; MLA 8 heads, kv_lora 512 + 64 un-rotated (the 576 row), q_lora 768;
    the indexer at its released shape, 32 heads x 128, with index_topk 32.
  * Sequences (BIG_TEST_TOKENS): seq0 96 tokens (the indexer selects 32 of up to 96 keys, the prefill
    crosses fla's 64-token chunk), seq1 = its first 32 tokens (= index_topk: dense causal, the last row
    exactly at the cut), seq2 300 tokens (4 chunk boundaries; decode from 260, past the last one).
    Picked tie-free like the tiny ones, but against big's own noise and under both eps presets
    (kimi_k31_gate.py TIE_BANDS / PICK_EPS). The manifest records the pick (BIG_PICK: bounds, smallest
    margins per preset, and the geometry and files it ran on, which selftest pins) and the float32-vs-
    float64 reference gap on these sequences (BIG_GAP, as reference_gap).
  * `--unit-imatrix PATH` writes an all-ones importance matrix, so the port's llama-quantize can make
    an IQ1_S copy (it refuses IQ1_S without one): every routed expert goes to IQ1_S, the rest to
    llama-quantize's usual IQ1_S mix, and only attn_k_b / ssm_f_b (128-wide rows) fall back (IQ4_NL).
  * What coincides, and why it is harmless here: the KDA width 8 x 128 equals hidden 1024 and the
    shared-expert width 512 equals latent 512 (the tiny fixture keeps both apart; a swap of two
    same-shape tensors still changes every value the gate compares).
  * Vocabulary 1024, special ids the real ones minus 162816 (bos 768, eos 770, media 789, pad 1023).

Usage:
    python tools/kimi_k31_synth_gguf.py --out kimi-k31-synth [--seed 3101] [--no-vocab] [--geometry big]
    python tools/kimi_k31_synth_gguf.py --selftest-only --out kimi-k31-synth [--geometry big]
        [--unit-imatrix PATH]     (an all-ones importance matrix, for llama-quantize IQ1_S / IQ2 copies)
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import struct
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from glm_synth_gguf import (ALIGN, GGML_F32, _s, arr_i32, arr_str, kv_bool, kv_f32, kv_str,  # noqa: E402
                            kv_u32, write_gguf)  # noqa: F401 (write_gguf: tests write fixtures through this module)

ARCH = "kimi-k31"
SEED = 3101

TINY = dict(
    vocab=320, n_embd=256, n_layers=9, attn_res_block=3, ctx=4096,
    full_attn_layers=[4, 8, 9],                         # 1-indexed, as in the release config
    n_head=6, kda_head_dim=32, d_conv=4, gate_lower_bound=-5.0,
    q_lora=160, kv_lora=512, qk_nope=128, qk_rope=64, v_head=128,
    n_ff_dense=448, first_k_dense=1,
    n_expert=24, n_expert_used=10, n_shared=2, n_ff_exp=64, latent=128,
    idx_heads=12, idx_dim=32, idx_topk=6,
    rms_eps=1e-5, situ_beta=4.0, situ_linear_beta=25.0, routed_scaling=1.0,
    bos=64, eos=66, pad=319, media=85,
)

# The big fixture (module docstring). Same keys as TINY, in the same order.
BIG = dict(
    vocab=1024, n_embd=1024, n_layers=10, attn_res_block=3, ctx=4096,
    full_attn_layers=[4, 9, 10],                        # 1-indexed: 0-indexed MLA 3, 8, 9
    n_head=8, kda_head_dim=128, d_conv=4, gate_lower_bound=-5.0,
    q_lora=768, kv_lora=512, qk_nope=128, qk_rope=64, v_head=128,
    n_ff_dense=1792, first_k_dense=1,
    n_expert=64, n_expert_used=16, n_shared=2, n_ff_exp=256, latent=512,
    idx_heads=32, idx_dim=128, idx_topk=32,
    rms_eps=1e-5, situ_beta=4.0, situ_linear_beta=25.0, routed_scaling=1.0,
    bos=768, eos=770, pad=1023, media=789,
)

GEOMETRIES = {"tiny": TINY, "big": BIG}
QK_K = 256                                              # the i-quant super-block: rows must be a multiple
# GGUF tensors whose rows are not a multiple of QK_K in the release either (qk_nope, KDA head_dim, conv taps)
IQ_ROW_EXEMPT = ("attn_k_b.weight", "ssm_f_b.weight", "ssm_conv1d_q.weight", "ssm_conv1d_k.weight",
                 "ssm_conv1d_v.weight")


def geometry_name(g: dict) -> str | None:
    for name, geo in GEOMETRIES.items():
        if g == geo:
            return name
    return None


def gguf_name(g: dict) -> str:
    """The fixture's GGUF file name: big.gguf for BIG, tiny.gguf otherwise."""
    return "big.gguf" if geometry_name(g) == "big" else "tiny.gguf"

# The release's quantization_config (text_config of moonshotai/Kimi-K3.1-Pro), verbatim.
QUANT_CONFIG = {
    "config_groups": {
        "group_0": {
            "format": "mxfp4-pack-quantized",
            "input_activations": None,
            "output_activations": None,
            "targets": ["Linear"],
            "weights": {
                "actorder": None, "block_structure": None, "dynamic": False, "group_size": 32,
                "num_bits": 4, "observer": "minmax", "observer_kwargs": {}, "scale_dtype": "torch.uint8",
                "strategy": "group", "symmetric": True, "type": "float", "zp_dtype": None,
            },
        }
    },
    "format": "mxfp4-pack-quantized",
    "global_compression_ratio": None,
    "ignore": [
        "re:.*self_attn.*",
        "re:.*shared_experts.*",
        "re:.*mlp\\.(gate|up|gate_up|down)_proj.*",
        "re:.*block_sparse_moe\\.gate$",
        "re:.*block_sparse_moe\\.routed_expert_(up|down)_proj.*",
        "re:.*lm_head.*",
        "re:.*vision_tower.*",
        "re:.*mm_projector.*",
    ],
    "kv_cache_scheme": None,
    "quant_method": "compressed-tensors",
    "quantization_status": "compressed",
}

# transformers-4.x generation defaults the release serializes into text_config. They are not model
# parameters (the MoE top-k is num_experts_per_token), but real configs carry them, so this one does.
_GEN_DEFAULTS = {
    "add_cross_attention": False, "bad_words_ids": None, "begin_suppress_tokens": None,
    "chunk_size_feed_forward": 0, "cross_attention_hidden_size": None, "decoder_start_token_id": None,
    "diversity_penalty": 0.0, "do_sample": False, "early_stopping": False,
    "encoder_no_repeat_ngram_size": 0, "exponential_decay_length_penalty": None,
    "finetuning_task": None, "forced_bos_token_id": None, "forced_eos_token_id": None,
    "id2label": {"0": "LABEL_0", "1": "LABEL_1"}, "is_decoder": False, "is_encoder_decoder": False,
    "label2id": {"LABEL_0": 0, "LABEL_1": 1}, "length_penalty": 1.0, "max_length": 20,
    "min_length": 0, "no_repeat_ngram_size": 0, "num_beam_groups": 1, "num_beams": 1,
    "num_return_sequences": 1, "output_attentions": False, "output_hidden_states": False,
    "output_scores": False, "prefix": None, "problem_type": None, "pruned_heads": {},
    "remove_invalid_values": False, "repetition_penalty": 1.0, "return_dict": True,
    "return_dict_in_generate": False, "sep_token_id": None, "suppress_tokens": None,
    "task_specific_params": None, "temperature": 1.0, "tf_legacy_loss": False,
    "tie_encoder_decoder": False, "tokenizer_class": None, "top_k": 50, "top_p": 1.0,
    "torchscript": False, "typical_p": 1.0, "use_bfloat16": False,
}

P = "language_model.model."
FP4_VALUES = np.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
                       -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0], dtype=np.float32)
MX_GROUP = 32


# ------------------------------------------------------------------ geometry helpers
def full_layers0(g: dict) -> list[int]:
    """0-indexed MLA (= DSA) layers."""
    return [i - 1 for i in g["full_attn_layers"]]


def is_mla(g: dict, il: int) -> bool:
    return (il + 1) in g["full_attn_layers"]


def text_config(g: dict) -> dict:
    n = g["n_layers"]
    full = list(g["full_attn_layers"])
    kda = [i for i in range(1, n + 1) if i not in full]
    cfg = {
        "_name_or_path": "",
        "activation_situ_beta": g["situ_beta"],
        "activation_situ_linear_beta": g["situ_linear_beta"],
        "architectures": ["KimiLinearForCausalLM"],
        "attn_res_block_size": g["attn_res_block"],
        "bos_token_id": g["bos"],
        "dsa_config": {
            "dsa_layers": full,
            "index_topk": g["idx_topk"],
            "index_head_dim": g["idx_dim"],
            "index_num_heads": g["idx_heads"],
            "index_rope_dim": 0,
            "index_k_norm_type": "layernorm",
            "index_topk_freq": 1,
            "index_skip_topk_offset": 1,
        },
        "dtype": "float32",
        "eos_token_id": g["eos"],
        "first_k_dense_replace": g["first_k_dense"],
        "hidden_act": "situ",
        "hidden_size": g["n_embd"],
        "initializer_range": 0.02,
        "intermediate_size": g["n_ff_dense"],
        "kv_lora_rank": g["kv_lora"],
        "latent_moe_use_norm": True,
        "linear_attn_config": {
            "full_attn_layers": full,
            "gate_lower_bound": g["gate_lower_bound"],
            "head_dim": g["kda_head_dim"],
            "kda_layers": kda,
            "num_heads": g["n_head"],
            "short_conv_kernel_size": g["d_conv"],
            "use_full_rank_gate": True,
        },
        "max_position_embeddings": g["ctx"],
        "mla_use_nope": True,
        "mla_use_output_gate": True,
        "model_type": "kimi_linear",
        "moe_intermediate_size": g["n_ff_exp"],
        "moe_layer_freq": 1,
        "moe_renormalize": True,
        "moe_router_activation_func": "sigmoid",
        "num_attention_heads": g["n_head"],
        "num_expert_group": 1,
        "num_experts": g["n_expert"],
        "num_experts_per_token": g["n_expert_used"],
        "num_hidden_layers": n,
        "num_key_value_heads": g["n_head"],
        "num_nextn_predict_layers": 0,
        "num_shared_experts": g["n_shared"],
        "pad_token_id": g["pad"],
        "q_lora_rank": g["q_lora"],
        "qk_nope_head_dim": g["qk_nope"],
        "qk_rope_head_dim": g["qk_rope"],
        "rms_norm_eps": g["rms_eps"],
        "routed_expert_hidden_size": g["latent"],
        "routed_scaling_factor": g["routed_scaling"],
        "tie_word_embeddings": False,
        "topk_group": 1,
        "topk_method": "noaux_tc",
        "transformers_version": "4.57.6",
        "use_cache": True,
        "use_grouped_topk": True,
        "v_head_dim": g["v_head"],
        "vocab_size": g["vocab"],
    }
    cfg.update(copy.deepcopy(_GEN_DEFAULTS))
    return dict(sorted(cfg.items()))


def hf_config(g: dict, mxfp4: bool) -> dict:
    tc = text_config(g)
    if mxfp4:
        tc["dtype"] = "bfloat16"
        tc["quantization_config"] = copy.deepcopy(QUANT_CONFIG)
        tc = dict(sorted(tc.items()))
    return {
        "architectures": ["KimiK31ForConditionalGeneration"],
        "bos_token_id": g["bos"],
        "dtype": "bfloat16" if mxfp4 else "float32",
        "eos_token_id": g["eos"],
        "ignore_index": -100,
        "image_placeholder": "<|kimi_image_placeholder|>",
        "media_placeholder_token_id": g["media"],
        "model_type": "kimi_k31",
        "pad_token_id": g["pad"],
        "text_config": tc,
        "tie_word_embeddings": False,
    }


# ------------------------------------------------------------------ number formats
def bf16_round(x: np.ndarray) -> np.ndarray:
    """Round float32 to the nearest bf16 (ties to even); the result is still float32."""
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32).astype(np.uint64)
    b = (b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF0000
    return b.astype(np.uint32).view(np.float32)


def f32_to_bf16_bits(x: np.ndarray) -> np.ndarray:
    b = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    if np.any(b & 0xFFFF):
        raise ValueError("value not on the bf16 grid")
    return (b >> 16).astype(np.uint16)


def bf16_bits_to_f32(bits: np.ndarray) -> np.ndarray:
    return (np.ascontiguousarray(bits, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def mxfp4_pack(codes: np.ndarray, scales: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """compressed-tensors mxfp4-pack-quantized: weight_packed [out, in/2] with element 2i in the low
    nibble and 2i+1 in the high one; weight_scale [out, in/32] = the biased E8M0 exponent."""
    out, cols = codes.shape
    assert cols % MX_GROUP == 0 and scales.shape == (out, cols // MX_GROUP)
    c = codes.astype(np.uint8)
    packed = (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8)
    return packed, scales.astype(np.uint8)


def mxfp4_dequant(packed: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """The inverse, written independently of mxfp4_pack: E2M1 code table times 2^(s - 127)."""
    out, half = packed.shape
    lo = packed & 0x0F
    hi = packed >> 4
    codes = np.empty((out, half * 2), dtype=np.uint8)
    codes[:, 0::2] = lo
    codes[:, 1::2] = hi
    s = np.ldexp(np.float32(1.0), scale.astype(np.int32) - 127).astype(np.float32)
    return (FP4_VALUES[codes] * np.repeat(s, MX_GROUP, axis=1)).astype(np.float32)


def _exp_safe(a: np.ndarray) -> np.ndarray:
    """True where float64 exp(a) lies within 0.05 ulp of a float32 (any faithful exp agrees)."""
    e = np.exp(a.astype(np.float64))
    r = e.astype(np.float32)
    ulp = np.spacing(r).astype(np.float64)
    return np.abs(e - r.astype(np.float64)) < 0.05 * ulp


def neg_exp(a: np.ndarray) -> np.ndarray:
    """-exp(a) rounded once to float32 (the converter's -torch.exp(A_log.float()))."""
    return (-np.exp(a.astype(np.float64))).astype(np.float32)


# ------------------------------------------------------------------ the weights
def make_weights(g: dict, seed: int):
    """Returns (vals, dtypes, mx): vals = checkpoint name -> logical float32 values (experts
    unpacked), dtypes = name -> dtype in the real checkpoint ("BF16" / "F32" / "MXFP4"),
    mx = expert name -> (FP4 codes, E8M0 scale bytes)."""
    D, V, L = g["n_embd"], g["vocab"], g["n_layers"]
    H, hd = g["n_head"], g["kda_head_dim"]
    d_inner = H * hd
    q_lora, kv_lora, nope, rope, vh = g["q_lora"], g["kv_lora"], g["qk_nope"], g["qk_rope"], g["v_head"]
    E, F, S, lat = g["n_expert"], g["n_ff_exp"], g["n_shared"], g["latent"]
    Hi, Di = g["idx_heads"], g["idx_dim"]
    rng = np.random.default_rng(seed)
    vals: dict[str, np.ndarray] = {}
    dtypes: dict[str, str] = {}
    mx: dict[str, tuple[np.ndarray, np.ndarray]] = {}

    def lin(name, out, inp, gain=0.8):                  # a bf16 Linear weight [out, in]
        vals[name] = bf16_round(rng.standard_normal((out, inp)).astype(np.float32) * (gain / np.sqrt(inp)))
        dtypes[name] = "BF16"

    def gain_w(name, n, std=0.05):                      # an RMS/LayerNorm gain near 1
        vals[name] = bf16_round((1.0 + rng.standard_normal(n) * std).astype(np.float32))
        dtypes[name] = "BF16"

    def bf16(name, arr):
        vals[name] = bf16_round(np.asarray(arr, dtype=np.float32))
        dtypes[name] = "BF16"

    def f32(name, arr):
        vals[name] = np.ascontiguousarray(arr, dtype=np.float32)
        dtypes[name] = "F32"

    def expert(name, out, inp):
        # FP4 codes without 8 (-0): +0 is the only zero, so every dequantizer gives the same bits
        codes = rng.integers(0, 15, size=(out, inp), dtype=np.uint8)
        codes = np.where(codes >= 8, codes + 1, codes).astype(np.uint8)
        e0 = int(np.round(np.log2(0.8 / np.sqrt(inp) / 2.93)))   # 2.93 = rms of the E2M1 table
        scales = (127 + e0 + rng.integers(-1, 2, size=(out, inp // MX_GROUP))).astype(np.uint8)
        mx[name] = (codes, scales)
        vals[name] = mxfp4_dequant(*mxfp4_pack(codes, scales))
        dtypes[name] = "MXFP4"

    bf16(P + "embed_tokens.weight", rng.standard_normal((V, D)).astype(np.float32) * 0.5)
    for il in range(L):
        p = f"{P}layers.{il}."
        gain_w(p + "input_layernorm.weight", D)
        gain_w(p + "post_attention_layernorm.weight", D)
        for site in ("self_attention_res", "mlp_res"):
            gain_w(p + f"{site}_norm.weight", D)
            bf16(p + f"{site}_proj.weight", rng.standard_normal((1, D)).astype(np.float32) * 0.1)
        a = p + "self_attn."
        if not is_mla(g, il):                               # KDA
            for x in ("q", "k", "v"):
                lin(a + f"{x}_proj.weight", d_inner, D)
            for x in ("q", "k", "v"):
                f32(a + f"{x}_conv1d.weight", rng.standard_normal((d_inner, 1, g["d_conv"])) * 0.4)
            lin(a + "f_a_proj.weight", hd, D)
            lin(a + "f_b_proj.weight", d_inner, hd)
            lin(a + "b_proj.weight", H, D)
            a_log = np.zeros(hd, dtype=np.float32)         # stored at head_dim length, zero tail
            got = []
            while len(got) < H:
                c = rng.uniform(-0.75, 2.5, size=4 * H).astype(np.float32)
                got += list(c[_exp_safe(c)])
            a_log[:H] = np.array(got[:H], dtype=np.float32)
            f32(a + "A_log", a_log)
            f32(a + "dt_bias", rng.standard_normal(d_inner) * 1.5 - 1.0)
            lin(a + "g_proj.weight", d_inner, D)
            f32(a + "o_norm.weight", 1.0 + rng.standard_normal(hd) * 0.05)
            lin(a + "o_proj.weight", D, d_inner)
        else:                                               # NoPE MLA + output gate + indexer
            lin(a + "q_a_proj.weight", q_lora, D)
            gain_w(a + "q_a_layernorm.weight", q_lora)
            lin(a + "q_b_proj.weight", H * (nope + rope), q_lora)
            lin(a + "kv_a_proj_with_mqa.weight", kv_lora + rope, D)
            gain_w(a + "kv_a_layernorm.weight", kv_lora)
            lin(a + "kv_b_proj.weight", H * (nope + vh), kv_lora)
            lin(a + "o_proj.weight", D, H * vh)
            lin(a + "g_proj.weight", H * vh, D)
            lin(a + "indexer.wq_b.weight", Hi * Di, q_lora)
            lin(a + "indexer.wk.weight", Di, D)
            gain_w(a + "indexer.k_norm.weight", Di, std=0.02)
            bf16(a + "indexer.k_norm.bias", rng.standard_normal(Di).astype(np.float32) * 0.05)
            lin(a + "indexer.weights_proj.weight", Hi, D, gain=1.0)
        if il < g["first_k_dense"]:                         # dense SiTU-GLU MLP
            lin(p + "mlp.gate_proj.weight", g["n_ff_dense"], D)
            lin(p + "mlp.up_proj.weight", g["n_ff_dense"], D)
            lin(p + "mlp.down_proj.weight", D, g["n_ff_dense"])
        else:                                               # latent MoE
            m = p + "block_sparse_moe."
            lin(m + "gate.weight", E, D, gain=1.0)
            f32(m + "gate.e_score_correction_bias", rng.standard_normal(E) * 0.1)
            lin(m + "routed_expert_down_proj.weight", lat, D)
            lin(m + "routed_expert_up_proj.weight", D, lat)
            gain_w(m + "routed_expert_norm.weight", lat)
            lin(m + "shared_experts.gate_proj.weight", F * S, D)
            lin(m + "shared_experts.up_proj.weight", F * S, D)
            lin(m + "shared_experts.down_proj.weight", D, F * S)
            for e in range(E):
                expert(m + f"experts.{e}.w1.weight", F, lat)
                expert(m + f"experts.{e}.w2.weight", lat, F)
                expert(m + f"experts.{e}.w3.weight", F, lat)
    gain_w(P + "norm.weight", D)
    gain_w(P + "output_attn_res_norm.weight", D)
    bf16(P + "output_attn_res_proj.weight", rng.standard_normal((1, D)).astype(np.float32) * 0.1)
    lin("language_model.lm_head.weight", V, D, gain=1.0)
    return vals, dtypes, mx


# The test sequences for TINY / SEED, picked by `python tools/kimi_k31_gate.py pick`: each is grown one
# token at a time (candidates from default_rng(SEED + 1) for seq0, SEED + 2 for seq2), keeping a token
# only if, at its position, no indexer or router top-k cut lies within 12x the gate's tolerance of a tie
# (indexer cut margin >= 3.6e-3, router >= 6e-4, on every MLA / MoE layer). The engines leave
# tie-breaking unspecified, so a golden judged on logits must not depend on it. The sequences depend on
# the weights: pick again after any change to TINY, SEED or make_weights.
TEST_TOKENS = {
    "seq0": [289, 263, 135, 169, 256, 73, 144, 40, 257, 176, 21, 233, 65, 220, 22, 270, 199, 41, 305, 5,
             44, 108, 250, 107, 30, 248, 59, 284, 34, 40, 141, 303],
    "seq2": [166, 79, 299, 219, 157, 270, 86, 231, 128, 304, 99, 74, 197, 244, 285, 98, 14, 255, 143, 239,
             280, 319, 296, 187, 216, 246, 102, 109, 93, 117, 224, 18, 23, 61, 205, 138, 183, 218, 195, 273,
             22, 149, 130, 154, 128, 84, 271, 159, 266, 138, 162, 300, 54, 209, 51, 242, 81, 302, 289, 222,
             127, 6, 146, 244, 119, 60, 180, 230, 234, 309, 299, 218, 298, 221, 298, 22, 18, 292, 173, 311],
}


# The test sequences for BIG / SEED, picked the same way (`python tools/kimi_k31_gate.py pick --geometry big`:
# candidates from default_rng(SEED + 1) for seq0 and SEED + 2 for seq2, kept only where the cut margins clear
# kimi_k31_gate.pick_bounds("big") under both eps presets) and re-checked as whole-sequence prefills under
# each preset (`pick --geometry big --verify`). BIG_PICK is what the two reported, and pins the geometry and
# the files they ran on (selftest checks both); BIG_GAP is `kimi_k31_gate.py gap --geometry big` on them,
# rounded to 3 digits. The manifest records both.
BIG_TEST_TOKENS: dict[str, list[int]] = {
    "seq0": [780, 894, 69, 761, 564, 96, 109, 129, 451, 9, 416, 501, 373, 821, 74, 229, 357, 773, 353, 1003, 337, 62,
             694, 180, 344, 39, 580, 256, 1014, 694, 670, 142, 255, 823, 240, 491, 650, 249, 366, 197, 954, 872, 43,
             578, 76, 943, 779, 937, 25, 785, 576, 236, 687, 928, 656, 378, 305, 260, 249, 1013, 86, 44, 433, 974,
             704, 807, 424, 681, 200, 943, 345, 353, 34, 803, 675, 852, 664, 174, 1006, 620, 804, 810, 737, 477, 779,
             604, 893, 340, 345, 143, 577, 449, 256, 524, 831, 642],
    "seq2": [533, 276, 741, 238, 783, 290, 315, 897, 83, 299, 75, 195, 658, 125, 441, 586, 700, 357, 73, 622, 495,
             447, 982, 774, 74, 259, 969, 924, 961, 16, 192, 736, 957, 501, 954, 81, 97, 939, 883, 879, 304, 945,
             826, 466, 785, 668, 139, 443, 468, 83, 67, 569, 323, 451, 135, 802, 1012, 465, 606, 478, 21, 88, 1015,
             144, 301, 310, 664, 463, 343, 122, 760, 221, 724, 522, 308, 442, 637, 802, 136, 775, 930, 106, 701, 844,
             133, 425, 349, 934, 404, 124, 126, 404, 95, 41, 88, 95, 974, 607, 823, 30, 553, 486, 814, 977, 155, 630,
             425, 215, 600, 837, 124, 876, 473, 868, 75, 459, 169, 345, 282, 254, 353, 364, 119, 842, 660, 343, 95,
             845, 497, 992, 387, 95, 339, 721, 74, 374, 827, 659, 301, 268, 28, 661, 618, 844, 548, 764, 739, 917,
             925, 473, 311, 830, 38, 594, 197, 622, 826, 710, 633, 384, 61, 557, 805, 789, 699, 215, 14, 704, 705,
             59, 650, 972, 193, 291, 73, 664, 299, 774, 658, 236, 885, 665, 521, 601, 853, 293, 84, 648, 705, 530,
             939, 853, 789, 73, 596, 427, 309, 601, 93, 379, 861, 764, 943, 646, 500, 672, 536, 101, 160, 1012, 16,
             470, 202, 746, 628, 433, 155, 121, 969, 619, 780, 335, 303, 271, 742, 264, 1020, 445, 212, 557, 889,
             781, 730, 831, 621, 625, 231, 871, 592, 839, 282, 948, 962, 557, 426, 607, 585, 85, 602, 419, 847, 728,
             560, 314, 961, 237, 499, 391, 305, 338, 150, 539, 632, 874, 9, 787, 114, 69, 592, 708, 943, 260, 376,
             135, 928, 747, 541, 616, 796, 108, 948, 386, 997, 1002, 12, 109, 827, 433, 646, 796, 359, 657, 485, 99,
             393, 1011, 573, 823, 568, 872],
}
BIG_PICK: dict = {
    "tool": "python tools/kimi_k31_gate.py pick --geometry big",
    "rule": "grown one token at a time; a candidate is kept only if, at its position, every MoE layer's router "
            "cut margin (k-th minus (k+1)-th biased score) and every MLA layer's indexer cut margin (when more "
            "than index_topk keys are visible) clear the bounds, PICK_ROOM 12 x the big tie bands "
            "(kimi_k31_gate.py TIE_BANDS), in a float64 reference under each eps preset; seq1 is the start of "
            "seq0",
    "bounds": {"router": 0.0015, "indexer": 0.0048000000000000004},
    "presets": ["moonshot", "llama.cpp"],
    "picked_on": {"geometry_sha256": "1838c2d5ffca9d11a93a137e5f5365043deb9b4b4e9f02132c3c1ec65a4a58b0",
                  "sha256": {"big.gguf": "bfe94395a585bfa39c236a286cd030093992cbafad0e4552e1adc00ed340ff88",
                             "hf-f32/model.safetensors": "fdc248360c0aeb2464c79a9a6df136e907952d9ad647c5de11034955262fac1a",
                             "hf-mxfp4/model.safetensors": "5304cf4c6964bd37dbea252b19bb46cac5ad578b8ad6a43e1f7107f52061c508"}},
    "seq0": {"tokens": 96, "candidates_from": "default_rng(seed + 1)", "draws": 620,
             "min_router_margin": 0.001501178660194502, "min_indexer_margin": 0.004912044160501949},
    "seq2": {"tokens": 300, "candidates_from": "default_rng(seed + 2)", "draws": 3663,
             "min_router_margin": 0.0015058268772893957, "min_indexer_margin": 0.004803953426508953},
    "verified": {"tool": "python tools/kimi_k31_gate.py pick --geometry big --verify",
                 "min_router_margin": {"moonshot": {"seq0": 0.0015011786601948351, "seq1": 0.001539559598301321,
                                                    "seq2": 0.0015063953520315643},
                                       "llama.cpp": {"seq0": 0.0015013453233087004, "seq1": 0.001523183632570868,
                                                     "seq2": 0.0015058268772893957}},
                 "min_indexer_margin": {"moonshot": {"seq0": 0.004912044160503615, "seq1": "inf",
                                                     "seq2": 0.0048102835764465834},
                                        "llama.cpp": {"seq0": 0.004912628751032044, "seq1": "inf",
                                                      "seq2": 0.00480395342652494}}},
}
BIG_GAP: dict = {
    "tool": "python tools/kimi_k31_gate.py gap --geometry big",
    "what": "float32 reference against float64, the same eps preset, each sequence one prefill; max = largest "
            "|difference|; all: max over the sequences, rms pooled over them; seams: layer-3 MLA and layer-0 KDA "
            "attention outputs; margin_shift = largest change of a row's cut margin; rows_differ = rows whose "
            "selected set differs",
    "presets": {"moonshot": {"all": {"logits_max": 0.00145, "logits_rms": 4.35e-05, "logits_peak": 4.86,
                                     "mla_seam_max": 4.11e-06, "mla_seam_rms": 3.27e-07, "kda_seam_max": 1.6e-06,
                                     "kda_seam_rms": 2.41e-07, "router_score_max": 0.00012,
                                     "router_margin_shift_max": 5.8e-05, "expert_rows_differ": 0,
                                     "indexer_rows_differ": 0, "indexer_score_max": 0.000373,
                                     "indexer_score_peak": 3.8, "indexer_margin_shift_max": 0.00022},
                             "seq0": {"logits_max": 0.00145, "logits_rms": 5.91e-05, "logits_peak": 4.86,
                                      "mla_seam_max": 4.11e-06, "mla_seam_rms": 3.87e-07, "kda_seam_max": 1.39e-06,
                                      "kda_seam_rms": 2.31e-07, "router_score_max": 8.88e-05,
                                      "router_margin_shift_max": 4.78e-05, "expert_rows_differ": 0,
                                      "indexer_rows_differ": 0, "indexer_score_max": 0.000301,
                                      "indexer_score_peak": 3.07, "indexer_margin_shift_max": 0.00022},
                             "seq1": {"logits_max": 0.00023, "logits_rms": 2.04e-05, "logits_peak": 4.86,
                                      "mla_seam_max": 3.95e-06, "mla_seam_rms": 3.78e-07, "kda_seam_max": 1.17e-06,
                                      "kda_seam_rms": 2.09e-07, "router_score_max": 5.04e-05,
                                      "router_margin_shift_max": 1.58e-05, "expert_rows_differ": 0,
                                      "indexer_rows_differ": 0, "indexer_score_max": 0.00013,
                                      "indexer_score_peak": 2.35, "indexer_margin_shift_max": 0.0},
                             "seq2": {"logits_max": 0.000921, "logits_rms": 3.93e-05, "logits_peak": 4.63,
                                      "mla_seam_max": 3.57e-06, "mla_seam_rms": 2.99e-07, "kda_seam_max": 1.6e-06,
                                      "kda_seam_rms": 2.47e-07, "router_score_max": 0.00012,
                                      "router_margin_shift_max": 5.8e-05, "expert_rows_differ": 0,
                                      "indexer_rows_differ": 0, "indexer_score_max": 0.000373,
                                      "indexer_score_peak": 3.8, "indexer_margin_shift_max": 0.000125}},
                "llama.cpp": {"all": {"logits_max": 0.0015, "logits_rms": 4.55e-05, "logits_peak": 4.86,
                                      "mla_seam_max": 5.3e-06, "mla_seam_rms": 3.64e-07, "kda_seam_max": 1.71e-06,
                                      "kda_seam_rms": 2.41e-07, "router_score_max": 0.000152,
                                      "router_margin_shift_max": 7.23e-05, "expert_rows_differ": 0,
                                      "indexer_rows_differ": 0, "indexer_score_max": 0.000367,
                                      "indexer_score_peak": 3.8, "indexer_margin_shift_max": 0.000174},
                              "seq0": {"logits_max": 0.0015, "logits_rms": 5.6e-05, "logits_peak": 4.86,
                                       "mla_seam_max": 4.23e-06, "mla_seam_rms": 3.86e-07,
                                       "kda_seam_max": 1.32e-06, "kda_seam_rms": 2.31e-07,
                                       "router_score_max": 0.000111, "router_margin_shift_max": 4.57e-05,
                                       "expert_rows_differ": 0, "indexer_rows_differ": 0,
                                       "indexer_score_max": 0.000218, "indexer_score_peak": 3.07,
                                       "indexer_margin_shift_max": 0.000107},
                              "seq1": {"logits_max": 0.000307, "logits_rms": 2.52e-05, "logits_peak": 4.86,
                                       "mla_seam_max": 4.09e-06, "mla_seam_rms": 3.49e-07,
                                       "kda_seam_max": 1.25e-06, "kda_seam_rms": 2.09e-07,
                                       "router_score_max": 5.69e-05, "router_margin_shift_max": 2.3e-05,
                                       "expert_rows_differ": 0, "indexer_rows_differ": 0,
                                       "indexer_score_max": 0.000202, "indexer_score_peak": 2.35,
                                       "indexer_margin_shift_max": 0.0},
                              "seq2": {"logits_max": 0.000914, "logits_rms": 4.33e-05, "logits_peak": 4.63,
                                       "mla_seam_max": 5.3e-06, "mla_seam_rms": 3.59e-07, "kda_seam_max": 1.71e-06,
                                       "kda_seam_rms": 2.47e-07, "router_score_max": 0.000152,
                                       "router_margin_shift_max": 7.23e-05, "expert_rows_differ": 0,
                                       "indexer_rows_differ": 0, "indexer_score_max": 0.000367,
                                       "indexer_score_peak": 3.8, "indexer_margin_shift_max": 0.000174}}},
    "margin_shift_vs_f64_moonshot": {"router": 9.07e-05, "indexer": 0.000341},
}

# Per geometry: (length, decode_from) of each sequence; seq1 is always the start of seq0.
#   tiny: seq0 > index_topk 6, seq1 <= it, seq2 past one 64-token fla chunk;
#   big:  seq0 > index_topk 32 and past one chunk, seq1 = index_topk exactly, seq2 past four chunks.
SEQ_PLAN = {
    "tiny": {"seq0": (32, 7), "seq1": (5, 1), "seq2": (80, 66)},
    "big": {"seq0": (96, 33), "seq1": (32, 1), "seq2": (300, 260)},
}
FLA_CHUNK = 64
_PICKED = {("tiny", SEED): TEST_TOKENS, ("big", SEED): BIG_TEST_TOKENS}


def seq_plan(g: dict) -> dict:
    return SEQ_PLAN[geometry_name(g) or "tiny"]


def picked_tokens(g: dict, seed: int) -> dict[str, list[int]] | None:
    """The picked (tie-free) seq0 / seq2 for this geometry and seed, or None."""
    toks = _PICKED.get((geometry_name(g), seed))
    return toks or None


def sample_tokens(g: dict, seed: int, n: int | None = None) -> list[int]:
    """The suggested test sequence: the picked seq0 (TEST_TOKENS / BIG_TEST_TOKENS) for TINY or BIG at
    SEED, otherwise random tokens from their own stream (does not move the weights). n defaults to
    the geometry's seq0 length."""
    n = seq_plan(g)["seq0"][0] if n is None else n
    picked = picked_tokens(g, seed)
    if picked and n == len(picked["seq0"]):
        return list(picked["seq0"])
    rng = np.random.default_rng(seed + 1)
    return [int(t) for t in rng.integers(0, g["vocab"], size=n)]


def gate_sequences(g: dict, seed: int) -> list[dict]:
    """The sequences the independent model and the gate run (written to manifest.json). For TINY / BIG at
    SEED they are the picked, tie-free TEST_TOKENS / BIG_TEST_TOKENS; otherwise random (and then not
    known to be tie-free)."""
    plan = seq_plan(g)
    picked_toks = picked_tokens(g, seed)
    picked = picked_toks is not None
    base = sample_tokens(g, seed)
    n1, n2 = plan["seq1"][0], plan["seq2"][0]
    if picked:
        long = list(picked_toks["seq2"])
    else:
        long = [int(t) for t in np.random.default_rng(seed + 2).integers(0, g["vocab"], size=n2)]
    k = g["idx_topk"]
    chunks = (n2 - 1) // FLA_CHUNK
    crosses = (f"crosses fla's {FLA_CHUNK}-token KDA chunk; decode continues past the chunk boundary"
               if chunks == 1 else
               f"crosses {chunks} of fla's {FLA_CHUNK}-token KDA chunk boundaries; decode continues past the last")
    return [
        {"name": "seq0", "tokens": base, "decode_from": plan["seq0"][1], "tie_free": picked,
         "why": f"{len(base)} tokens > index_topk {k}: the indexer selects for positions >= {k}"},
        {"name": "seq1", "tokens": base[:n1], "decode_from": plan["seq1"][1], "tie_free": picked,
         "why": f"{n1} tokens <= index_topk {k}: every row is dense causal MLA"},
        {"name": "seq2", "tokens": long, "decode_from": plan["seq2"][1], "tie_free": picked,
         "why": f"{n2} tokens: {crosses}"},
    ]


# ------------------------------------------------------------------ checkpoint -> GGUF (the converter)
def to_gguf(vals: dict[str, np.ndarray], g: dict, vocab: bool = True):
    """What conversion/kimi_k3.py writes from this checkpoint with --outtype f32, plus the kimi-k31
    additions. vals are float32 values under the checkpoint names (experts unpacked)."""
    D, L = g["n_embd"], g["n_layers"]
    H = g["n_head"]
    nope, rope, vh, kv_lora = g["qk_nope"], g["qk_rope"], g["v_head"], g["kv_lora"]
    E = g["n_expert"]
    full = g["full_attn_layers"]
    a = ARCH + "."
    md = [
        ("general.architecture", kv_str(ARCH)),
        ("general.type", kv_str("model")),
        ("general.name", kv_str("kimi-k31-synth")),
        # TextModel.set_gguf_parameters
        (a + "block_count", kv_u32(L)),
        (a + "context_length", kv_u32(g["ctx"])),
        (a + "embedding_length", kv_u32(D)),
        (a + "feed_forward_length", kv_u32(g["n_ff_dense"])),
        (a + "attention.head_count", kv_u32(H)),
        # MLA served as MQA (one KV head); 0 marks a KDA layer. 1-indexed lists, as KimiLinearConfig
        (a + "attention.head_count_kv", arr_i32([1 if (il + 1) in full else 0 for il in range(L)])),
        (a + "attention.layer_norm_rms_epsilon", kv_f32(g["rms_eps"])),
        (a + "expert_count", kv_u32(E)),
        (a + "expert_used_count", kv_u32(g["n_expert_used"])),
        (a + "expert_group_used_count", kv_u32(1)),            # topk_group
        (a + "expert_gating_func", kv_u32(2)),                 # LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID
        ("general.file_type", kv_u32(0)),                     # LLAMA_FTYPE_ALL_F32
        # KimiK3Model.set_gguf_parameters
        (a + "vocab_size", kv_u32(g["vocab"])),
        (a + "ssm.conv_kernel", kv_u32(g["d_conv"])),
        (a + "kda.head_dim", kv_u32(g["kda_head_dim"])),
        (a + "kda.gate_lower_bound", kv_f32(g["gate_lower_bound"])),
        (a + "attention.q_lora_rank", kv_u32(g["q_lora"])),
        (a + "attention.kv_lora_rank", kv_u32(kv_lora)),
        (a + "rope.dimension_count", kv_u32(rope)),            # sizes the un-absorbed part of K (NoPE)
        (a + "attention.key_length", kv_u32(kv_lora + rope)),
        (a + "attention.value_length", kv_u32(kv_lora)),
        (a + "attention.key_length_mla", kv_u32(nope + rope)),
        (a + "attention.value_length_mla", kv_u32(vh)),
        (a + "expert_feed_forward_length", kv_u32(g["n_ff_exp"])),
        (a + "expert_shared_count", kv_u32(g["n_shared"])),
        (a + "leading_dense_block_count", kv_u32(g["first_k_dense"])),
        (a + "expert_weights_scale", kv_f32(g["routed_scaling"])),
        (a + "expert_weights_norm", kv_bool(True)),
        (a + "expert_latent_length", kv_u32(g["latent"])),
        (a + "activation.situ_beta", kv_f32(g["situ_beta"])),
        (a + "activation.situ_linear_beta", kv_f32(g["situ_linear_beta"])),
        (a + "attn_res.block_size", kv_u32(g["attn_res_block"])),
        # kimi-k31: the lightning indexer (every MLA layer runs its own; no kpool / types keys)
        (a + "attention.indexer.head_count", kv_u32(g["idx_heads"])),
        (a + "attention.indexer.key_length", kv_u32(g["idx_dim"])),
        (a + "attention.indexer.top_k", kv_u32(g["idx_topk"])),
        (a + "attention.layer_norm_epsilon", kv_f32(g["rms_eps"])),   # the indexer k LayerNorm
        ("general.quantization_version", kv_u32(2)),
    ]
    if vocab:
        special = {g["bos"]: "[BOS]", g["bos"] + 1: "[EOS]", g["eos"]: "<|end_of_msg|>",
                   g["media"]: "<|media_pad|>", g["pad"]: "[PAD]"}
        md += [
            ("tokenizer.ggml.model", kv_str("gpt2")),
            ("tokenizer.ggml.pre", kv_str("kimi-k2")),
            ("tokenizer.ggml.tokens", arr_str([special.get(i, "tok%d" % i) for i in range(g["vocab"])])),
            ("tokenizer.ggml.token_type", arr_i32([3 if i in special else 1 for i in range(g["vocab"])])),
            ("tokenizer.ggml.merges", arr_str([])),
            ("tokenizer.ggml.bos_token_id", kv_u32(g["bos"])),
            ("tokenizer.ggml.eos_token_id", kv_u32(g["eos"])),
            ("tokenizer.ggml.padding_token_id", kv_u32(g["pad"])),
        ]

    def res(prefix):                                          # _try_fuse_res: norm * proj, flattened
        return (vals[prefix + "_norm.weight"].flatten() * vals[prefix + "_proj.weight"].flatten()).astype(np.float32)

    ts: list[tuple[str, np.ndarray]] = [("token_embd.weight", vals[P + "embed_tokens.weight"])]
    for il in range(L):
        p, b = f"{P}layers.{il}.", f"blk.{il}."
        s = p + "self_attn."
        ts += [
            (b + "attn_norm.weight", vals[p + "input_layernorm.weight"]),
            (b + "attn_res_score.weight", res(p + "self_attention_res")),
            (b + "ffn_res_score.weight", res(p + "mlp_res")),
            (b + "ffn_norm.weight", vals[p + "post_attention_layernorm.weight"]),
        ]
        if not is_mla(g, il):
            ts += [(b + f"attn_{x}.weight", vals[s + f"{x}_proj.weight"]) for x in ("q", "k", "v")]
            for x in ("q", "k", "v"):                         # [d_inner, 1, d_conv] -> ne [d_conv, 1, d_inner, 1]
                w = vals[s + f"{x}_conv1d.weight"]
                ts.append((b + f"ssm_conv1d_{x}.weight", w.reshape(1, w.shape[0], 1, w.shape[2])))
            ts += [
                (b + "ssm_f_a.weight", vals[s + "f_a_proj.weight"]),
                (b + "ssm_f_b.weight", vals[s + "f_b_proj.weight"]),
                (b + "ssm_beta.weight", vals[s + "b_proj.weight"]),
                # -exp(A_log) folded; sliced to the KDA head count (linear_attn_config.num_heads)
                (b + "ssm_a", neg_exp(vals[s + "A_log"][:H])),
                (b + "ssm_dt.bias", vals[s + "dt_bias"]),
                (b + "ssm_g.weight", vals[s + "g_proj.weight"]),          # g_proj on a KDA layer
                (b + "ssm_norm.weight", vals[s + "o_norm.weight"]),
                (b + "attn_output.weight", vals[s + "o_proj.weight"]),
            ]
        else:
            kv_b = vals[s + "kv_b_proj.weight"].reshape(H, nope + vh, kv_lora)
            ts += [
                (b + "attn_q_a.weight", vals[s + "q_a_proj.weight"]),
                (b + "attn_q_a_norm.weight", vals[s + "q_a_layernorm.weight"]),
                (b + "attn_q_b.weight", vals[s + "q_b_proj.weight"]),
                (b + "attn_kv_a_mqa.weight", vals[s + "kv_a_proj_with_mqa.weight"]),
                (b + "attn_kv_a_norm.weight", vals[s + "kv_a_layernorm.weight"]),
                # kv_b split for absorption: k_b transposed to [H, kv_lora, nope], v_b [H, v, kv_lora]
                (b + "attn_k_b.weight", np.ascontiguousarray(kv_b[:, :nope, :].transpose(0, 2, 1))),
                (b + "attn_v_b.weight", np.ascontiguousarray(kv_b[:, nope:, :])),
                (b + "attn_gate.weight", vals[s + "g_proj.weight"]),          # g_proj on an MLA layer
                (b + "attn_output.weight", vals[s + "o_proj.weight"]),
                (b + "indexer.attn_q_b.weight", vals[s + "indexer.wq_b.weight"]),
                (b + "indexer.attn_k.weight", vals[s + "indexer.wk.weight"]),
                (b + "indexer.k_norm.weight", vals[s + "indexer.k_norm.weight"]),
                (b + "indexer.k_norm.bias", vals[s + "indexer.k_norm.bias"]),
                (b + "indexer.proj.weight", vals[s + "indexer.weights_proj.weight"]),
            ]
        if il < g["first_k_dense"]:
            ts += [
                (b + "ffn_gate.weight", vals[p + "mlp.gate_proj.weight"]),
                (b + "ffn_up.weight", vals[p + "mlp.up_proj.weight"]),
                (b + "ffn_down.weight", vals[p + "mlp.down_proj.weight"]),
            ]
        else:
            m = p + "block_sparse_moe."
            ts += [
                (b + "ffn_gate_inp.weight", vals[m + "gate.weight"]),
                (b + "exp_probs_b.bias", vals[m + "gate.e_score_correction_bias"]),
                (b + "ffn_routed_down.weight", vals[m + "routed_expert_down_proj.weight"]),
                (b + "ffn_routed_up.weight", vals[m + "routed_expert_up_proj.weight"]),
                (b + "ffn_routed_norm.weight", vals[m + "routed_expert_norm.weight"]),
                (b + "ffn_gate_shexp.weight", vals[m + "shared_experts.gate_proj.weight"]),
                (b + "ffn_up_shexp.weight", vals[m + "shared_experts.up_proj.weight"]),
                (b + "ffn_down_shexp.weight", vals[m + "shared_experts.down_proj.weight"]),
            ]
            for wid, gname in (("w1", "ffn_gate_exps"), ("w2", "ffn_down_exps"), ("w3", "ffn_up_exps")):
                ts.append((b + gname + ".weight",
                           np.stack([vals[m + f"experts.{e}.{wid}.weight"] for e in range(E)])))
    ts += [
        ("output_norm.weight", vals[P + "norm.weight"]),
        ("output_res_score.weight", res(P + "output_attn_res")),
        ("output.weight", vals["language_model.lm_head.weight"]),
    ]
    return md, ts


# ------------------------------------------------------------------ safetensors (no dependency)
_ST_ITEMSIZE = {"F32": 4, "BF16": 2, "U8": 1}


def write_safetensors(path: Path, entries: dict[str, tuple[str, np.ndarray]]) -> None:
    """entries: name -> (dtype, raw array). Laid out like the release's shards: no __metadata__,
    tensors ordered by element size (largest first) then name, so every offset is aligned. The
    header is laid out from the shapes and the tensors are written one at a time."""
    order = sorted(entries, key=lambda n: (-_ST_ITEMSIZE[entries[n][0]], n))
    header, off = {}, 0
    for n in order:
        dt, arr = entries[n]
        assert arr.itemsize == _ST_ITEMSIZE[dt], n
        size = int(np.prod(arr.shape)) * _ST_ITEMSIZE[dt]
        header[n] = {"dtype": dt, "shape": list(arr.shape), "data_offsets": [off, off + size]}
        off += size
    h = json.dumps(header, separators=(",", ":")).encode()
    h += b" " * (-len(h) % 8)
    with open(path, "wb") as fh:
        fh.write(struct.pack("<Q", len(h)))
        fh.write(h)
        for n in order:
            blob = np.ascontiguousarray(entries[n][1]).tobytes()
            assert len(blob) == header[n]["data_offsets"][1] - header[n]["data_offsets"][0], n
            fh.write(blob)


def read_safetensors(path: Path) -> dict[str, tuple[str, np.ndarray]]:
    """name -> (dtype, read-only array mapped from the file)."""
    np_dt = {"F32": "<f4", "BF16": "<u2", "U8": "u1"}
    path = Path(path)
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        header = json.loads(fh.read(n))
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    base = 8 + n
    out = {}
    end = 0
    for name, info in header.items():
        if name == "__metadata__":
            continue
        s, e = info["data_offsets"]
        end = max(end, e)
        arr = raw[base + s:base + e].view(np_dt[info["dtype"]]).reshape(info["shape"])
        out[name] = (info["dtype"], arr)
    assert base + end == raw.size, "safetensors: trailing bytes"
    return out


def write_gguf_stream(path: Path, metadata: list[tuple[str, bytes]], tensors: list[tuple[str, np.ndarray]]) -> None:
    """tools/glm_synth_gguf.py write_gguf's bytes (GGUF v3, F32 tensors), with the tensor data
    written one tensor at a time instead of assembled in memory (the big fixture's 1.3 GB would
    be held three times over)."""
    header = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(metadata))
    kvs = b"".join(_s(k) + v for k, v in metadata)
    infos, offset = [], 0
    for name, arr in tensors:
        dims = list(reversed(arr.shape))                 # ne0 varies fastest
        infos.append(_s(name) + struct.pack("<I", len(dims)) + b"".join(struct.pack("<Q", d) for d in dims)
                     + struct.pack("<I", GGML_F32) + struct.pack("<Q", offset))
        offset += 4 * int(np.prod(arr.shape, dtype=np.int64))
        offset = (offset + ALIGN - 1) // ALIGN * ALIGN   # tensors start aligned, like gguf.cpp
    head = header + kvs + b"".join(infos)
    with open(path, "wb") as fh:
        fh.write(head + b"\0" * (-len(head) % ALIGN))
        for _, arr in tensors:
            blob = np.ascontiguousarray(arr, dtype="<f4").tobytes()
            fh.write(blob + b"\0" * (-len(blob) % ALIGN))


def write_unit_imatrix(gguf: Path, out: Path) -> int:
    """An importance matrix of ones (uniform importance) for every matrix in `gguf`, in llama.cpp's
    legacy imatrix layout (n_entries; per entry name, ncall 1, ne0 x n_expert values; chunk count and
    dataset name). llama-quantize refuses IQ1_S / IQ2_XXS / IQ2_XS without one, and the fixture has no
    text to compute a real one from (its vocabulary is a dummy). Returns the number of entries."""
    from gguf_reader import GGUFFile
    ents = [(t.name, t.shape) for t in GGUFFile(gguf).tensors if len(t.shape) >= 2]   # shape: ne0 first
    with open(out, "wb") as fh:
        fh.write(struct.pack("<i", len(ents)))
        for name, shape in ents:
            nval = int(shape[0]) * int(np.prod(shape[2:], dtype=np.int64))
            b = name.encode()
            fh.write(struct.pack("<i", len(b)) + b + struct.pack("<ii", 1, nval))
            fh.write(np.ones(nval, dtype="<f4").tobytes())
        ds = b"unit (kimi_k31_synth_gguf.py --unit-imatrix)"
        fh.write(struct.pack("<ii", 1, len(ds)) + ds)
    return len(ents)


# ------------------------------------------------------------------ build
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2) + "\n", encoding="utf-8")


def build(out: Path, seed: int = SEED, vocab: bool = True, g: dict = TINY) -> dict:
    out = Path(out)
    (out / "hf-f32").mkdir(parents=True, exist_ok=True)
    (out / "hf-mxfp4").mkdir(parents=True, exist_ok=True)
    vals, dtypes, mx = make_weights(g, seed)
    gname = gguf_name(g)

    md, ts = to_gguf(vals, g, vocab=vocab)
    write_gguf_stream(out / gname, md, ts)
    del md, ts                                          # the stacked experts: a second copy of most bytes

    write_safetensors(out / "hf-f32" / "model.safetensors", {n: ("F32", v) for n, v in vals.items()})
    _write_json(out / "hf-f32" / "config.json", hf_config(g, mxfp4=False))

    ents: dict[str, tuple[str, np.ndarray]] = {}
    for n, v in vals.items():
        if dtypes[n] == "MXFP4":
            packed, scale = mxfp4_pack(*mx[n])
            stem = n[:-len(".weight")]
            ents[stem + ".weight_packed"] = ("U8", packed)
            ents[stem + ".weight_scale"] = ("U8", scale)
        elif dtypes[n] == "BF16":
            ents[n] = ("BF16", f32_to_bf16_bits(v))
        else:
            ents[n] = ("F32", v)
    write_safetensors(out / "hf-mxfp4" / "model.safetensors", ents)
    del ents
    _write_json(out / "hf-mxfp4" / "config.json", hf_config(g, mxfp4=True))

    files = [gname, "hf-f32/config.json", "hf-f32/model.safetensors",
             "hf-mxfp4/config.json", "hf-mxfp4/model.safetensors"]
    full0 = full_layers0(g)
    manifest = {
        "generator": "tools/kimi_k31_synth_gguf.py",
        "generator_sha256": _sha256(Path(__file__)),
        "seed": seed,
        "vocab_in_gguf": vocab,
        "geometry": g,
        "layers": {
            "mla_dsa_0idx": full0,
            "kda_0idx": [i for i in range(g["n_layers"]) if i not in full0],
            "dense_ffn_0idx": list(range(g["first_k_dense"])),
            "attn_res_block_pushes_0idx": list(range(0, g["n_layers"], g["attn_res_block"])),
        },
        "eps": {
            "rms_norm_eps": g["rms_eps"],
            "indexer_k_layernorm (attention.layer_norm_epsilon)": g["rms_eps"],
            "note": "q_a/kv_a RMSNorm eps is not fixed by the files: Moonshot's K3 code uses the "
                    "class default 1e-6, engines and llama.cpp use rms_norm_eps 1e-5",
        },
        "test_tokens": sample_tokens(g, seed),
        "test_sequences": gate_sequences(g, seed),
    }
    pick = pick_record(g, seed)
    if pick is not None:
        manifest["test_sequences_pick"] = pick
    gap = gap_record(g, seed)
    if gap is not None:
        manifest["reference_gap"] = gap
    manifest["files"] = {f: {"sha256": _sha256(out / f), "bytes": (out / f).stat().st_size} for f in files}
    manifest["config"] = hf_config(g, mxfp4=False)
    _write_json(out / "manifest.json", manifest)
    return manifest


def pick_record(g: dict, seed: int) -> dict | None:
    """What the manifest records about the pick of the test sequences: BIG_PICK for BIG / SEED (the
    tiny fixture's is the TEST_TOKENS comment, and its manifest stays as it was)."""
    if geometry_name(g) == "big" and picked_tokens(g, seed) is not None and BIG_PICK:
        return copy.deepcopy(BIG_PICK)
    return None


def gap_record(g: dict, seed: int) -> dict | None:
    """The float32-vs-float64 reference gap on the picked sequences (BIG_GAP, `kimi_k31_gate.py gap
    --geometry big`) for BIG / SEED; None otherwise (tiny's is in kimi_k31_gate.py's docstring)."""
    if geometry_name(g) == "big" and picked_tokens(g, seed) is not None and BIG_GAP:
        return copy.deepcopy(BIG_GAP)
    return None


def geometry_sha256(g: dict) -> str:
    """What BIG_PICK pins the geometry by: sha256 of the geometry dict as sorted-key JSON."""
    return hashlib.sha256(json.dumps(g, sort_keys=True).encode()).hexdigest()


# ------------------------------------------------------------------ selftest
def _gguf_tensors(path: Path) -> tuple[dict, dict[str, np.ndarray]]:
    from gguf_reader import GGUFFile
    gf = GGUFFile(path)
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    out = {}
    for t in gf.tensors:
        assert t.type_name == "F32", f"{t.name}: {t.type_name}"
        n = t.elements
        start = gf.data_start + t.offset
        out[t.name] = raw[start:start + 4 * n].view("<f4").reshape(tuple(reversed(t.shape)))
    return gf.metadata, out


def check_requirements(g: dict, seqs: list[dict], kind: str | None = None) -> None:
    """The fixture requirements a K3.1 port depends on, for geometry g and its test sequences
    (gate_sequences' list); raises AssertionError naming the first one that does not hold. kind
    ("tiny" / "big", default: which of the two g is) selects the geometry's own requirements."""
    kind = geometry_name(g) if kind is None else kind
    H, hd, Hi, Di = g["n_head"], g["kda_head_dim"], g["idx_heads"], g["idx_dim"]
    full0 = full_layers0(g)
    L, blk = g["n_layers"], g["attn_res_block"]
    assert len([i for i in range(1, L) if i % blk == 0]) >= 2, "two attn-res block boundaries"
    assert full0[-2:] == [L - 2, L - 1] and len(full0) >= 3, "last two layers MLA, one more earlier"
    assert all(i not in full0 for i in range(full0[-3] + 1, L - 2)), "KDA between the MLA layers"
    assert all(i not in full0 for i in range(full0[0])), "KDA before the first MLA layer"
    assert g["first_k_dense"] == 1 and g["n_expert_used"] > 8 and g["n_shared"] == 2, \
        "one dense lead layer, top-k > 8, 2 shared experts"
    assert g["latent"] < g["n_embd"] and g["kv_lora"] + g["qk_rope"] == 576 and H < hd, \
        "latent < hidden, the 576-wide MLA row, KDA heads < head_dim"
    assert hd != g["qk_rope"] and g["q_lora"] != g["qk_nope"] + g["qk_rope"] and Hi * Di != g["n_embd"], \
        "no dimension coincidence a mix-up could hide behind"
    assert Hi >= 8, "enough indexer heads that a key rarely scores exactly 0 (every head's q.k <= 0)"

    # the sequences: one where the indexer selects, one dense, one past fla's chunk
    s = {q["name"]: q for q in seqs}
    k = g["idx_topk"]
    assert len(s["seq0"]["tokens"]) > k, "seq0 longer than index_topk: the indexer selects"
    assert len(s["seq1"]["tokens"]) <= k, "seq1 no longer than index_topk: dense causal MLA"
    assert s["seq1"]["tokens"] == s["seq0"]["tokens"][:len(s["seq1"]["tokens"])], "seq1 starts seq0"
    assert len(s["seq2"]["tokens"]) > FLA_CHUNK, "seq2 crosses fla's KDA chunk"
    assert all(0 < q["decode_from"] < len(q["tokens"]) for q in seqs), "decode_from inside the sequence"
    assert s["seq2"]["decode_from"] > FLA_CHUNK, "decode continues past a chunk boundary"
    assert all(0 <= t < g["vocab"] for q in seqs for t in q["tokens"]), "token ids in the vocabulary"

    if kind == "big":
        widths = {"n_embd": g["n_embd"], "latent": g["latent"], "n_ff_exp": g["n_ff_exp"],
                  "q_lora": g["q_lora"], "kv_lora": g["kv_lora"], "n_ff_dense": g["n_ff_dense"],
                  "KDA width": H * hd, "MLA value width": H * g["v_head"],
                  "shared-expert width": g["n_ff_exp"] * g["n_shared"]}
        bad = {n: w for n, w in widths.items() if w % QK_K}
        assert not bad, f"rows an i-quant reads must be multiples of {QK_K}: {bad}"
        assert (g["n_expert"], g["n_expert_used"]) == (64, 16), "16 of 64 routed experts"
        assert len([i for i in range(1, L) if i % blk == 0]) >= 3, "pushes at 3, 6 and 9"
        assert (Hi, Di) == (32, 128), "the indexer at its released shape"
        assert len(s["seq2"]["tokens"]) > 4 * FLA_CHUNK, "seq2 crosses several of fla's chunks"
        assert len(s["seq2"]["tokens"]) > 8 * k, "the indexer selects a small share of seq2's keys"
    else:
        assert k < 24, "index_topk small enough for a 24-32 token sequence"


def selftest(out: Path, g: dict = TINY) -> None:
    """Re-read every file and check it against the others and against the fixture requirements."""
    out = Path(out)
    man = json.loads((out / "manifest.json").read_text())
    assert man["geometry"] == g, "manifest.json was written for another geometry (--geometry)"
    for f, info in man["files"].items():
        assert _sha256(out / f) == info["sha256"], f"{f}: sha256 differs from the manifest"
    H, hd, Hi, Di = g["n_head"], g["kda_head_dim"], g["idx_heads"], g["idx_dim"]
    nope, vh, kv_lora = g["qk_nope"], g["v_head"], g["kv_lora"]
    full0 = full_layers0(g)
    L, blk = g["n_layers"], g["attn_res_block"]

    # the fixture requirements, and the manifest's sequences (and pick record) are this tool's
    seqs = gate_sequences(g, man["seed"])
    assert man["test_sequences"] == seqs, "manifest test_sequences differ from the tool's"
    assert man["test_tokens"] == sample_tokens(g, man["seed"]), "manifest test_tokens"
    assert man.get("test_sequences_pick") == pick_record(g, man["seed"]), "manifest test_sequences_pick"
    assert man.get("reference_gap") == gap_record(g, man["seed"]), "manifest reference_gap"
    check_requirements(g, seqs)
    pick = pick_record(g, man["seed"])
    if pick is not None:                                # the files are the ones the sequences were picked on
        on = pick["picked_on"]
        assert on["geometry_sha256"] == geometry_sha256(g), "BIG changed since the big sequences were picked"
        for f, h in on["sha256"].items():
            if f == gguf_name(g) and not man["vocab_in_gguf"]:
                continue                                # --no-vocab: other GGUF bytes, the same weights
            assert man["files"][f]["sha256"] == h, \
                f"{f}: not the file the big sequences were picked on (re-pick: kimi_k31_gate.py pick --geometry big)"

    # hf-f32 is the source of truth from here on (arrays mapped from the file, never written)
    f32 = read_safetensors(out / "hf-f32" / "model.safetensors")
    assert all(dt == "F32" for dt, _ in f32.values())
    vals = {n: np.asarray(a) for n, (_, a) in f32.items()}

    # hf-mxfp4 holds the same bits
    mx = read_safetensors(out / "hf-mxfp4" / "model.safetensors")
    n_packed = 0
    for n, (dt, a) in mx.items():
        if n.endswith(".weight_scale"):
            continue
        if n.endswith(".weight_packed"):
            stem = n[:-len("_packed")]
            sdt, scale = mx[stem + "_scale"]
            assert dt == "U8" and sdt == "U8" and ".block_sparse_moe.experts." in n, n
            assert scale.shape == (a.shape[0], a.shape[1] * 2 // MX_GROUP), n
            v = mxfp4_dequant(a, scale)
            ref = vals[stem]
            n_packed += 1
        else:
            v = bf16_bits_to_f32(a) if dt == "BF16" else a
            ref = vals[n]
        assert v.shape == ref.shape and np.array_equal(v.view(np.uint32), ref.view(np.uint32)), f"hf-mxfp4 {n}"
    assert n_packed == 3 * g["n_expert"] * (L - g["first_k_dense"]), "expert count in hf-mxfp4"
    assert len(mx) == len(vals) + n_packed, "hf-mxfp4 tensor set"
    for n, (dt, _) in mx.items():
        want = ("U8" if ".experts." in n else
                "F32" if n.endswith(("A_log", "dt_bias", "conv1d.weight", "o_norm.weight", "correction_bias"))
                else "BF16")
        assert dt == want, f"{n}: {dt}, the release stores {want}"

    # A_log: head_dim entries, a +0.0 tail past the KDA heads
    for il in range(L):
        if il not in full0:
            al = vals[f"{P}layers.{il}.self_attn.A_log"]
            assert al.shape == (hd,) and not np.any(al[H:].view(np.uint32)), f"layer {il} A_log tail"

    # configs
    for sub, q in (("hf-f32", False), ("hf-mxfp4", True)):
        cfg = json.loads((out / sub / "config.json").read_text())
        assert cfg == hf_config(g, mxfp4=q), f"{sub}/config.json"
        tc = cfg["text_config"]
        assert tc["dsa_config"]["dsa_layers"] == tc["linear_attn_config"]["full_attn_layers"]
        assert ("quantization_config" in tc) == q

    # the GGUF: the converter's output from the re-read checkpoint, tensor by tensor
    md, gts = _gguf_tensors(out / gguf_name(g))
    want_md, want_ts = to_gguf(vals, g, vocab=man["vocab_in_gguf"])
    assert [n for n, _ in want_ts] == list(gts), "GGUF tensor set / order"
    for n, w in want_ts:
        assert gts[n].shape == w.shape and np.array_equal(gts[n].view(np.uint32), w.view(np.uint32)), n
    del want_ts
    if geometry_name(g) == "big":                       # every row an i-quant could read is QK_K-aligned
        rows = {n: a.shape[-1] for n, a in gts.items() if a.ndim >= 2 and not n.endswith(IQ_ROW_EXEMPT)}
        bad = {n: r for n, r in rows.items() if r % QK_K}
        assert not bad, f"GGUF rows not a multiple of {QK_K}: {dict(list(bad.items())[:4])}"
    a = ARCH + "."
    assert md["general.architecture"] == ARCH and len(md) == len(want_md)
    assert md[a + "attention.head_count_kv"] == [1 if i in full0 else 0 for i in range(L)]
    assert md[a + "attention.key_length"] == 576 and md[a + "attention.key_length_mla"] == 192
    assert md[a + "attention.indexer.top_k"] == g["idx_topk"] and md[a + "attn_res.block_size"] == blk
    assert not any(k.startswith(a + "attention.indexer.") and k.rsplit(".", 1)[1] in ("kpool", "types")
                   for k in md), "kpool / types keys must not be written"

    # the transforms, restated independently of to_gguf
    b3, s3 = f"blk.{full0[0]}.", f"{P}layers.{full0[0]}.self_attn."
    kv_b = vals[s3 + "kv_b_proj.weight"]
    for h in (0, H - 1):
        rows = kv_b[h * (nope + vh):(h + 1) * (nope + vh)]
        assert np.array_equal(gts[b3 + "attn_k_b.weight"][h], rows[:nope].T), "k_b"
        assert np.array_equal(gts[b3 + "attn_v_b.weight"][h], rows[nope:]), "v_b"
    assert gts[b3 + "attn_k_b.weight"].shape == (H, kv_lora, nope)
    assert gts[b3 + "indexer.attn_q_b.weight"].shape == (Hi * Di, g["q_lora"])
    assert gts[b3 + "attn_gate.weight"].shape == (H * vh, g["n_embd"]) and b3 + "ssm_g.weight" not in gts
    p0 = f"{P}layers.0."
    a_log = vals[p0 + "self_attn.A_log"][:H].astype(np.float64)
    assert np.array_equal(gts["blk.0.ssm_a"], (-np.exp(a_log)).astype(np.float32))
    assert np.all(gts["blk.0.ssm_a"] < 0) and gts["blk.0.ssm_a"].shape == (H,)
    assert gts["blk.0.ssm_conv1d_q.weight"].shape == (1, H * hd, 1, g["d_conv"])
    assert np.array_equal(gts["blk.0.ssm_conv1d_q.weight"][0, :, 0, :],
                          vals[p0 + "self_attn.q_conv1d.weight"][:, 0, :])
    assert np.array_equal(gts["blk.0.attn_res_score.weight"],
                          vals[p0 + "self_attention_res_norm.weight"] * vals[p0 + "self_attention_res_proj.weight"][0])
    assert "blk.0.ffn_gate.weight" in gts and "blk.0.ffn_gate_exps.weight" not in gts, "dense layer 0"
    m1 = f"{P}layers.1.block_sparse_moe."
    ge, de = gts["blk.1.ffn_gate_exps.weight"], gts["blk.1.ffn_down_exps.weight"]
    assert ge.shape == (g["n_expert"], g["n_ff_exp"], g["latent"])
    assert de.shape == (g["n_expert"], g["latent"], g["n_ff_exp"])
    assert np.array_equal(ge[5], vals[m1 + "experts.5.w1.weight"])
    assert np.array_equal(de[7], vals[m1 + "experts.7.w2.weight"])
    assert np.array_equal(gts["blk.1.exp_probs_b.bias"], vals[m1 + "gate.e_score_correction_bias"])
    # the bias must be able to move the selection: spread comparable to the sigmoid score gaps
    assert float(np.std(vals[m1 + "gate.e_score_correction_bias"])) > 0.02
    print(f"selftest ok: {len(gts)} GGUF tensors, {len(vals)} checkpoint tensors "
          f"({n_packed} MXFP4 experts), {sum(i['bytes'] for i in man['files'].values()) / 1e6:.1f} MB")


def main() -> int:
    ap = argparse.ArgumentParser(description="write the synthetic Kimi K3.1 fixture (GGUF + HF folders)")
    ap.add_argument("--out", default="kimi-k31-synth", help="output directory")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--geometry", default="tiny", choices=sorted(GEOMETRIES),
                    help="tiny (tiny.gguf, the default) or big (big.gguf: i-quant rows, 16 of 64 experts)")
    ap.add_argument("--no-vocab", action="store_true",
                    help="leave the dummy tokenizer out of the GGUF (upstream llama.cpp then cannot load it)")
    ap.add_argument("--selftest-only", action="store_true", help="check an existing output directory")
    ap.add_argument("--unit-imatrix", metavar="PATH", default=None,
                    help="also write an all-ones importance matrix for the GGUF (legacy .dat layout) to PATH, "
                         "for llama-quantize's IQ1_S / IQ2 types")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    g = GEOMETRIES[args.geometry]
    if not args.selftest_only:
        man = build(out, seed=args.seed, vocab=not args.no_vocab, g=g)
        for f, info in man["files"].items():
            print(f"  {f:28s} {info['bytes'] / 1e6:7.2f} MB  {info['sha256'][:16]}")
    selftest(out, g)
    if args.unit_imatrix:
        n = write_unit_imatrix(out / gguf_name(g), Path(args.unit_imatrix).expanduser())
        print(f"  {args.unit_imatrix}: unit importance matrix, {n} entries")
    if not picked_tokens(g, args.seed):
        print(f"note: the {args.geometry} sequences for seed {args.seed} are not picked, so not known to be tie-free "
              f"(python tools/kimi_k31_gate.py pick --geometry {args.geometry} --seed {args.seed})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

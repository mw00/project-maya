"""Index MiniMax DSpark/vision GGUFs in the engine's existing pack. No weights are copied."""
from __future__ import annotations

import json
import re

DP = "dspark.language_model.model.dspark."
VP = "vision_tower.vision_model."


def config(kv, key):
    try:
        value = json.loads(kv[key])
        if not isinstance(value, dict):
            raise ValueError("not an object")
        return value
    except (KeyError, TypeError, ValueError) as e:
        raise SystemExit(f"the selected GGUF needs valid {key} metadata: {e}") from e


def entries(g, types, fi, rename):
    out = []
    for name, typ, dims, off in g.tensors:
        if typ not in (0, 30):
            raise SystemExit(f"{g.path.name}: {name}: this companion needs F32/BF16 weights")
        mapped = rename(name)
        if mapped is None:
            raise SystemExit(f"{g.path.name}: unsupported companion tensor {name}")
        dtype, per, size = types[typ]
        count = 1
        for d in dims:
            count *= d
        out.append((mapped, dtype, fi, g.data_start + off, count // per * size, list(reversed(dims))))
    return out


def draft(g, types, fi, layer_map, hidden):
    if g.kv.get("general.architecture") != "dflash" or g.kv.get("dflash.backbone") != "minimax-m3":
        raise SystemExit(f"{g.path}: not a MiniMax-M3 DSpark GGUF")
    if g.kv.get("dflash.embedding_length") != hidden:
        raise SystemExit(f"{g.path}: the drafter's hidden size does not match the model")
    cfg = config(g.kv, "dflash.minimax.source_config")
    cfg = dict(cfg.get("text_config", cfg))
    # The GGUF converter already added one to Gemma-style norm weights.
    cfg["norms_plus_one"] = 1
    names = {"fc.weight": "fc.weight", "enc.output_norm.weight": "hidden_norm.weight",
             "output_norm.weight": "final_norm.weight", "conf_proj.weight": "confidence_head.proj.weight",
             "conf_proj.bias": "confidence_head.proj.bias", "markov_w1.weight": "markov_head.markov_w1.weight",
             "markov_w2.weight": "markov_head.markov_w2.weight"}

    def rename(name):
        if name in names:
            return DP + names[name]
        m = re.fullmatch(r"blk\.(\d+)\.(.+)", name)
        if m and m[2] in layer_map:
            return DP + f"layers.{m[1]}.decoder_layer." + layer_map[m[2]]
        return None

    return entries(g, types, fi, rename), cfg


def vision(g, types, fi, tokens, hidden):
    if g.kv.get("general.architecture") != "clip" or g.kv.get("clip.projector_type") != "minimax_m3":
        raise SystemExit(f"{g.path}: not a MiniMax-M3 vision GGUF")
    if g.kv.get("clip.vision.projection_dim") != hidden:
        raise SystemExit(f"{g.path}: the vision projector's width does not match the model")
    cfg = config(g.kv, "clip.vision.minimax.config")
    pp = config(g.kv, "clip.vision.minimax.preprocessor_config")
    vc = cfg["vision_config"]
    tok = {t: i for i, t in enumerate(tokens)}
    info = {"_vc": vc, "_comp": vc.get("img_token_compression_config") or cfg.get("img_token_compression_config") or {},
            "_extra": {"image_mean": pp["image_mean"], "image_std": pp["image_std"],
                       "projector_hidden_size": cfg["projector_hidden_size"], "max_pixels": 672 * 672,
                       "min_pixels": 4 * 28 * 28, "image_start": tok["]<]start of image[>["],
                       "image_end": tok["]<]end of image[>["], "image_token": tok["]<]image[>["]},
            "_ids": {k: cfg[k] for k in ("image_token_index", "video_token_index") if isinstance(cfg.get(k), int)}}
    layers = {"ln1": "layer_norm1", "ln2": "layer_norm2", "attn_q": "self_attn.q_proj",
              "attn_k": "self_attn.k_proj", "attn_v": "self_attn.v_proj", "attn_out": "self_attn.out_proj",
              "ffn_up": "mlp.fc1", "ffn_down": "mlp.fc2"}
    names = {"v.patch_embd.weight": VP + "embeddings.patch_embedding.weight",
             "v.patch_embd.weight.1": VP + "embeddings.patch_embedding.weight.1"}
    for suffix in ("weight", "bias"):
        names["v.pre_ln." + suffix] = VP + "pre_layrnorm." + suffix
        for src, dst in (("mm.1", "multi_modal_projector.linear_1"), ("mm.2", "multi_modal_projector.linear_2"),
                         ("mm.merger.fc1", "patch_merge_mlp.linear_1"), ("mm.merger.fc2", "patch_merge_mlp.linear_2")):
            names[src + "." + suffix] = dst + "." + suffix

    def rename(name):
        if name in names:
            return names[name]
        m = re.fullmatch(r"v\.blk\.(\d+)\.([^.]+)\.(weight|bias)", name)
        if m and m[2] in layers:
            return VP + f"encoder.layers.{m[1]}." + layers[m[2]] + "." + m[3]
        return None

    return entries(g, types, fi, rename), info

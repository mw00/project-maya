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


def entries(g, types, fi, rename, allowed=(0, 30)):
    out = []
    for name, typ, dims, off in g.tensors:
        if typ not in allowed:
            formats = "/".join(types[t][0] for t in allowed)
            raise SystemExit(f"{g.path.name}: {name}: this companion needs {formats} weights")
        mapped = rename(name)
        if mapped is None:
            raise SystemExit(f"{g.path.name}: unsupported companion tensor {name}")
        dtype, per, size = types[typ]
        count = 1
        for d in dims:
            count *= d
        out.append((mapped, dtype, fi, g.data_start + off, count // per * size, list(reversed(dims))))
    return out


def is_minimax_draft(meta):
    if meta.get("general.architecture") != "dflash":
        return False
    backbone = meta.get("dflash.backbone")
    if backbone is not None:
        return backbone == "minimax-m3"
    return (meta.get("dflash.hidden_activation") == "swigluoai"
            and meta.get("dflash.has_confidence_head") is True
            and meta.get("dflash.sample_from_anchor") is True
            and isinstance(meta.get("dflash.target_layers"), list)
            and bool(meta["dflash.target_layers"]))


def draft_config(g, hidden, model_kv=None):
    if "dflash.minimax.source_config" in g.kv:
        cfg = config(g.kv, "dflash.minimax.source_config")
        return dict(cfg.get("text_config", cfg))
    # Standard llama.cpp fields. target_layers names layer inputs; Maya taps the preceding layer's output.
    kv = g.kv
    def integer(key, minimum=1):
        v = kv.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < minimum:
            raise SystemExit(f"{g.path.name}: missing or invalid {key}")
        return v
    targets = kv.get("dflash.target_layers")
    if not isinstance(targets, list) or not targets or any(type(x) is not int or x < 1 for x in targets):
        raise SystemExit(f"{g.path.name}: invalid dflash.target_layers (target layer inputs must follow a layer)")
    if len(set(targets)) != len(targets):
        raise SystemExit(f"{g.path.name}: duplicate dflash.target_layers")
    if model_kv:
        for suffix in ("attention.head_count", "attention.head_count_kv", "attention.key_length",
                       "attention.value_length", "rope.dimension_count"):
            value = integer("dflash." + suffix)
            if model_kv.get("minimax-m3." + suffix) != value:
                raise SystemExit(f"{g.path.name}: drafter {suffix} does not match the model")
        if max(targets) > model_kv["minimax-m3.block_count"]:
            raise SystemExit(f"{g.path.name}: dflash.target_layers exceeds the model's layer count")
    ts = {name: dims for name, _, dims, _ in g.tensors}
    markov = ts.get("markov_w1.weight")
    if not markov or len(markov) != 2 or markov[0] <= 0 or ts.get("markov_w2.weight") != markov:
        raise SystemExit(f"{g.path.name}: missing or inconsistent DSpark Markov tables")
    if ts.get("fc.weight") != [len(targets) * hidden, hidden]:
        raise SystemExit(f"{g.path.name}: fc.weight does not match the target layers and hidden size")
    if ts.get("conf_proj.weight") != [hidden + markov[0], 1]:
        raise SystemExit(f"{g.path.name}: confidence head does not match the Markov rank")
    return {"num_hidden_layers": integer("dflash.block_count"), "hidden_size": hidden,
            "dspark_ffn_hidden_size": integer("dflash.feed_forward_length"),
            "dspark_markov_rank": markov[0], "dspark_block_size": integer("dflash.block_size"),
            "dspark_noise_token_id": integer("tokenizer.ggml.mask_token_id", 0),
            "dspark_target_layer_ids": [x - 1 for x in targets],
            "dflash_config": {"sliding_window": integer("dflash.attention.sliding_window")}}


def draft(g, types, fi, layer_map, hidden, model_kv=None):
    if not is_minimax_draft(g.kv):
        raise SystemExit(f"{g.path}: not a MiniMax-M3 DSpark GGUF")
    if g.kv.get("dflash.embedding_length") != hidden:
        raise SystemExit(f"{g.path}: the drafter's hidden size does not match the model")
    cfg = draft_config(g, hidden, model_kv)
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

    return entries(g, types, fi, rename, allowed=(0, 30, 8)), cfg


def vision_config(g):
    # Earlier converters embedded the HF JSONs; llama.cpp/Unsloth now writes the CLIP fields directly.
    if "clip.vision.minimax.config" in g.kv:
        return config(g.kv, "clip.vision.minimax.config"), config(g.kv, "clip.vision.minimax.preprocessor_config")
    kv = g.kv
    required = {"hidden_size": "embedding_length", "intermediate_size": "feed_forward_length",
                "num_hidden_layers": "block_count", "num_attention_heads": "attention.head_count",
                "patch_size": "patch_size"}
    vc = {}
    for dst, src in required.items():
        key = "clip.vision." + src
        value = kv.get(key)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise SystemExit(f"{g.path.name}: missing or invalid {key}")
        vc[dst] = value
    pp = {}
    for key in ("image_mean", "image_std"):
        value = kv.get("clip.vision." + key)
        if not isinstance(value, (list, tuple)) or len(value) != 3 or not all(isinstance(x, (int, float)) for x in value):
            raise SystemExit(f"{g.path.name}: missing or invalid clip.vision.{key}")
        pp[key] = value
    vc["layer_norm_eps"] = kv.get("clip.vision.attention.layer_norm_epsilon", 1e-5)
    vc["rope_theta"] = kv.get("clip.vision.rope.freq_base", 10000.0)
    vc["img_token_compression_config"] = {
        "spatial_merge_size": kv.get("clip.vision.spatial_merge_size", 2),
        "temporal_patch_size": 2,  # minimax_m3's two separately stored patch convolutions
    }
    projector = next((dims for name, _, dims, _ in g.tensors if name == "mm.1.weight"), None)
    if projector is None or len(projector) != 2 or projector[0] != vc["hidden_size"]:
        raise SystemExit(f"{g.path.name}: missing or invalid mm.1.weight vision projector")
    return {"vision_config": vc, "projector_hidden_size": projector[1]}, pp


def vision(g, types, fi, tokens, hidden):
    if g.kv.get("general.architecture") != "clip" or g.kv.get("clip.projector_type") != "minimax_m3":
        raise SystemExit(f"{g.path}: not a MiniMax-M3 vision GGUF")
    if g.kv.get("clip.vision.projection_dim") != hidden:
        raise SystemExit(f"{g.path}: the vision projector's width does not match the model")
    cfg, pp = vision_config(g)
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

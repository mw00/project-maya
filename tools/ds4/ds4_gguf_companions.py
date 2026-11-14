"""Index selected DeepSeek DSpark/mmproj GGUFs in place, without writing weight copies."""
from __future__ import annotations

import re

from vision_pack import MMPROJ_CONFIG, MMPROJ_MARKS, mmproj_names


def entry(g, types, fi, name, tensor):
    source, typ, dims, off = tensor
    if typ not in types:
        raise SystemExit(f"{g.path.name}: {source}: unsupported ggml type {typ}")
    dtype, per, size = types[typ]
    count = 1
    for d in dims:
        count *= d
    if dims[0] % per:
        raise SystemExit(f"{source}: incomplete {dtype} blocks")
    return name, dtype, fi, g.data_start + off, count // per * size, list(reversed(dims))


def drafter(g, types, fi, layer_map, geo, vocab):
    kv = g.kv
    if kv.get("general.architecture") != "dflash" or not kv.get("dflash.hyper_connection.count"):
        raise SystemExit(f"{g.path}: not a DeepSeek DSpark GGUF")
    for key, target in (("embedding_length", "hidden_size"), ("attention.head_count", "num_attention_heads"),
                        ("attention.key_length", "head_dim"), ("attention.q_lora_rank", "q_lora_rank"),
                        ("attention.output_group_count", "o_groups"), ("attention.output_lora_rank", "o_lora_rank"),
                        ("expert_feed_forward_length", "moe_intermediate_size"), ("hyper_connection.count", "hc_mult")):
        if kv.get("dflash." + key) != geo[target]:
            raise SystemExit(f"{g.path}: dflash.{key} does not match the target model")
    blocks = int(kv["dflash.block_count"])
    taps = list(kv.get("dflash.target_layers", []))
    # Standard llama.cpp exports tap indices including the embedding output (+1).
    # dsv41_semantics exports the checkpoint's indices, which Maya uses directly.
    if not kv.get("dflash.dsv41_semantics", False):
        taps = [t - 1 for t in taps]
    infos = {t[0]: t for t in g.tensors}
    w1 = infos.get("markov_w1.weight")
    if not w1 or len(w1[2]) != 2 or w1[2][1] != vocab:
        raise SystemExit(f"{g.path}: Markov vocabulary does not match the target model")
    rank = w1[2][0]
    noise = kv.get("tokenizer.ggml.mask_token_id", -1)
    block_size = int(kv.get("dflash.block_size", 0))
    if blocks <= 0 or len(taps) != blocks or any(t < 0 or t >= geo["num_hidden_layers"] for t in taps) or \
            not 0 <= noise < vocab or rank <= 0 or block_size <= 1:
        raise SystemExit(f"{g.path}: incomplete DSpark block / tap / noise / rank metadata")
    out_geo = {"dspark_block_size": block_size, "dspark_noise_token_id": noise,
               "dspark_markov_rank": rank, "dspark_n_routed_experts": kv["dflash.expert_count"],
               "dspark_num_experts_per_tok": kv["dflash.expert_used_count"], "dspark_target_layer_ids": taps,
               "compress_ratios": geo["compress_ratios"][:geo["num_hidden_layers"]] + [0] * blocks}
    roots = {"fc.weight": "mtp.0.main_proj.weight", "enc.output_norm.weight": "mtp.0.main_norm.weight",
             "output_norm.weight": f"mtp.{blocks - 1}.norm.weight",
             "conf_proj.weight": f"mtp.{blocks - 1}.confidence_head.proj.weight",
             "markov_w1.weight": f"mtp.{blocks - 1}.markov_head.embed.weight",
             "markov_w2.weight": f"mtp.{blocks - 1}.markov_head.head.weight"}
    result = []
    groups = int(geo["o_groups"])
    for t in g.tensors:
        name = roots.get(t[0])
        m = re.fullmatch(r"blk\.(\d+)\.(.+)", t[0])
        if m and int(m[1]) < blocks and m[2] in layer_map:
            name = f"mtp.{m[1]}." + layer_map[m[2]]
        if name is None:
            raise SystemExit(f"{g.path.name}: unsupported DSpark tensor {t[0]}")
        n, dt, fi, off, nb, shape = entry(g, types, fi, name, t)
        if name.endswith(".attn.wo_a.slice"):
            if len(shape) == 3 and shape[0] == groups:
                shape = [groups * shape[1], shape[2]]
            if len(shape) != 2 or shape[0] % groups:
                raise SystemExit(f"{name}: cannot split output groups")
            for i in range(groups):
                result.append((f"{n}.{i}.weight", dt, fi, off + i * (nb // groups), nb // groups,
                               [shape[0] // groups, shape[1]]))
        elif ".ffn.experts." in name:
            pre, role = name.rsplit(".", 1)
            for i in range(shape[0]):
                result.append((f"{pre}.{i}.{role}.weight", dt, fi, off + i * (nb // shape[0]),
                               nb // shape[0], shape[1:]))
        else:
            result.append((n, dt, fi, off, nb, shape))
    return result, out_geo


def vision(g, types, fi, hidden):
    if g.kv.get("general.architecture") != "clip" or g.kv.get("clip.projector_type") != "deepseek41v":
        raise SystemExit(f"{g.path}: not a DeepSeek-V4.1 mmproj GGUF")
    if g.kv.get("clip.vision.projection_dim") != hidden:
        raise SystemExit(f"{g.path}: vision projection does not match the target model")
    cfg = {k: g.kv.get(key, default) if key else default for k, (key, default) in MMPROJ_CONFIG.items()}
    cfg["vision.max_image_tokens"] = g.kv.get("clip.vision.image_max_tokens", 1024)
    cfg["vision.rms_norm_eps"] = g.kv.get("clip.vision.attention.layer_norm_epsilon", 1e-6)
    infos = {t[0]: t for t in g.tensors}
    result = []
    names = mmproj_names(int(cfg["vision.num_hidden_layers"]))
    names.update({name: [source] for name, source in MMPROJ_MARKS.items()})
    for name, sources in names.items():
        for i, source in enumerate(sources):
            if source not in infos:
                raise SystemExit(f"{g.path.name}: missing vision tensor {source}")
            t = infos[source]
            if t[1] not in (0, 1, 8, 30):
                raise SystemExit(f"{source}: vision reads F32/F16/BF16/Q8_0 weights")
            n = name if len(sources) == 1 else f"{name}.part.{i}"
            e = entry(g, types, fi, n, t)
            if name == "vision.patch_embed.proj.weight":
                shape = e[5]
                e = (*e[:5], [shape[0], shape[1] * shape[2] * shape[3]])
            result.append(e)
    unknown = set(infos) - {s for ss in names.values() for s in ss}
    if unknown:
        raise SystemExit(f"{g.path.name}: unmapped vision tensors: {', '.join(sorted(unknown)[:8])}")
    return result, cfg

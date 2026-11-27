"""tools/ling/convert_mmproj.py - Extract Ling-3.2-air (bailing_moe_v3_vl) vision tower + linear_proj into a BF16 mmproj GGUF.

Reads `model.visual.*` and `linear_proj.*` directly from the HF checkpoint safetensors
(all 333 BF16 tensors live in shard 1 of the official FP8/BF16 release) and writes a
self-contained GGUF v3 file with architecture metadata (`clip.projector_type`, geometry,
image mean/std, special token IDs) for `maya-ling --mmproj`.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np

GGUF_MAGIC = b"GGUF"
GGUF_VERSION = 3
GGUF_ALIGN = 32
TYPE_F32 = 0
TYPE_BF16 = 30

META_U32 = 4
META_I32 = 5
META_F32 = 6
META_BOOL = 7
META_STR = 8
META_ARR = 9


def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _pack_kv(k: str, vtype: int, payload: bytes) -> bytes:
    return _pack_str(k) + struct.pack("<I", vtype) + payload


def _kv_str(k: str, v: str) -> bytes:
    return _pack_kv(k, META_STR, _pack_str(v))


def _kv_u32(k: str, v: int) -> bytes:
    return _pack_kv(k, META_U32, struct.pack("<I", int(v)))


def _kv_i32(k: str, v: int) -> bytes:
    return _pack_kv(k, META_I32, struct.pack("<i", int(v)))


def _kv_f32(k: str, v: float) -> bytes:
    return _pack_kv(k, META_F32, struct.pack("<f", float(v)))


def _kv_bool(k: str, v: bool) -> bytes:
    return _pack_kv(k, META_BOOL, struct.pack("<B", 1 if v else 0))


def _kv_f32_arr(k: str, vals: list[float]) -> bytes:
    body = struct.pack("<IQ", META_F32, len(vals)) + struct.pack(f"<{len(vals)}f", *[float(x) for x in vals])
    return _pack_kv(k, META_ARR, body)


def _kv_i32_arr(k: str, vals: list[int]) -> bytes:
    body = struct.pack("<IQ", META_I32, len(vals)) + struct.pack(f"<{len(vals)}i", *[int(x) for x in vals])
    return _pack_kv(k, META_ARR, body)


def _read_safetensors_header(path: pathlib.Path) -> tuple[dict, int]:
    with path.open("rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(hlen).decode("utf-8"))
    return hdr, 8 + hlen


def _bf16_bytes_to_f32(raw: bytes) -> bytes:
    u16 = np.frombuffer(raw, dtype=np.uint16)
    u32 = u16.astype(np.uint32) << 16
    return u32.view(np.float32).tobytes()


def convert(src_dir: pathlib.Path, out_path: pathlib.Path) -> None:
    cfg = json.loads((src_dir / "config.json").read_text(encoding="utf-8"))
    vcfg = cfg["vision_config"]
    tcfg = cfg["text_config"]
    pcfg = json.loads((src_dir / "preprocessor_config.json").read_text(encoding="utf-8"))
    idx_path = src_dir / "model.safetensors.index.json"
    if idx_path.exists():
        wmap = json.loads(idx_path.read_text(encoding="utf-8"))["weight_map"]
        vis_keys = sorted(k for k in wmap if k.startswith("model.visual.") or k.startswith("linear_proj."))
        shards = {k: src_dir / wmap[k] for k in vis_keys}
    else:
        sf = src_dir / "model.safetensors"
        hdr, _ = _read_safetensors_header(sf)
        vis_keys = sorted(k for k in hdr if k.startswith("model.visual.") or k.startswith("linear_proj."))
        shards = {k: sf for k in vis_keys}

    if not vis_keys:
        raise RuntimeError(f"No model.visual.* or linear_proj.* tensors found in {src_dir}")

    # Load headers per shard
    shard_headers: dict[pathlib.Path, tuple[dict, int]] = {}
    for p in set(shards.values()):
        shard_headers[p] = _read_safetensors_header(p)

    # Collect tensor descriptors; promote 1D norms/biases to F32 for direct CUDA kernel loads, keep 2D+/weights BF16
    tensors: list[tuple[str, tuple[int, ...], int, bytes]] = []
    for k in vis_keys:
        sp = shards[k]
        hdr, data_off = shard_headers[sp]
        meta = hdr[k]
        dtype = meta["dtype"]
        shape = tuple(int(x) for x in meta["shape"])
        s0, s1 = meta["data_offsets"]
        with sp.open("rb") as f:
            f.seek(data_off + s0)
            raw = f.read(s1 - s0)

        # Keep exact HF names so ling_vision.cu matches model.visual.* & linear_proj.* unambiguously
        is_1d = (len(shape) == 1)
        if dtype == "BF16":
            if is_1d:
                raw = _bf16_bytes_to_f32(raw)
                gtype = TYPE_F32
            else:
                gtype = TYPE_BF16
        elif dtype == "F32":
            gtype = TYPE_F32
        elif dtype == "F16":
            arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            if is_1d:
                raw = arr.tobytes()
                gtype = TYPE_F32
            else:
                u32 = arr.view(np.uint32)
                raw = ((u32 + 0x7FFF + ((u32 >> 16) & 1)) >> 16).astype(np.uint16).tobytes()
                gtype = TYPE_BF16
        else:
            raise RuntimeError(f"Unsupported dtype {dtype} for {k}")

        # GGUF supports up to 4 dims; flatten Conv3d patch_embed.proj.weight [out, C, T, H, W] -> [out, C*T*H*W]
        if len(shape) > 4:
            prod = 1
            for d in shape[1:]:
                prod *= d
            shape = (shape[0], prod)
        # GGUF dims are reversed relative to PyTorch/numpy row-major shape (ne[0] is contiguous inner dim)
        gguf_dims = tuple(reversed(shape))
        tensors.append((k, gguf_dims, gtype, raw))

    kvs = [
        _kv_str("general.architecture", "clip"),
        _kv_str("general.name", "Ling-3.2-air-mmproj"),
        _kv_str("clip.projector_type", "bailing_qwen3_vit"),
        _kv_bool("clip.has_vision_encoder", True),
        _kv_u32("clip.vision.block_count", vcfg["depth"]),
        _kv_u32("clip.vision.embedding_length", vcfg["hidden_size"]),
        _kv_u32("clip.vision.feed_forward_length", vcfg["intermediate_size"]),
        _kv_u32("clip.vision.attention.head_count", vcfg["num_heads"]),
        _kv_u32("clip.vision.patch_size", vcfg["patch_size"]),
        _kv_u32("clip.vision.spatial_merge_size", vcfg["spatial_merge_size"]),
        _kv_u32("clip.vision.temporal_patch_size", vcfg["temporal_patch_size"]),
        _kv_u32("clip.vision.num_position_embeddings", vcfg["num_position_embeddings"]),
        _kv_u32("clip.vision.projection_dim", tcfg["hidden_size"]),
        _kv_f32("clip.vision.attention.layer_norm_epsilon", 1e-6),
        _kv_f32("clip.vision.rope_theta", 10000.0),
        _kv_u32("clip.vision.min_pixels", pcfg.get("min_pixels", 4096)),
        _kv_u32("clip.vision.max_pixels", pcfg.get("max_pixels", 4194304)),
        _kv_f32_arr("clip.vision.image_mean", pcfg["image_mean"]),
        _kv_f32_arr("clip.vision.image_std", pcfg["image_std"]),
        _kv_i32("clip.vision.image_token_id", cfg.get("image_token_id", 157157)),
        _kv_i32("clip.vision.vision_start_token_id", cfg.get("vision_start_token_id", 157158)),
        _kv_i32("clip.vision.vision_end_token_id", cfg.get("vision_end_token_id", 157159)),
        _kv_i32_arr("clip.vision.mrope_section", cfg.get("mrope_section", [8, 12, 12])),
    ]

    # Compute aligned tensor offsets
    tinfo_buf = bytearray()
    offset = 0
    offsets = []
    for name, dims, gtype, raw in tensors:
        rem = offset % GGUF_ALIGN
        if rem:
            offset += GGUF_ALIGN - rem
        offsets.append(offset)
        tinfo_buf += _pack_str(name)
        tinfo_buf += struct.pack("<I", len(dims))
        for d in dims:
            tinfo_buf += struct.pack("<Q", d)
        tinfo_buf += struct.pack("<IQ", gtype, offset)
        offset += len(raw)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = bytearray()
    header += GGUF_MAGIC
    header += struct.pack("<IQQ", GGUF_VERSION, len(tensors), len(kvs))
    for kv in kvs:
        header += kv
    header += tinfo_buf
    pad = (-len(header)) % GGUF_ALIGN
    header += b"\x00" * pad

    with out_path.open("wb") as f:
        f.write(header)
        cur = 0
        for (name, dims, gtype, raw), off in zip(tensors, offsets):
            if off > cur:
                f.write(b"\x00" * (off - cur))
                cur = off
            f.write(raw)
            cur += len(raw)

    total_mib = out_path.stat().st_size / (1024 * 1024)
    print(f"Wrote {out_path} ({len(tensors)} tensors, {total_mib:.1f} MiB)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Export Ling-3.2-air ViT + linear_proj to BF16 mmproj GGUF")
    ap.add_argument("--src", required=True, type=pathlib.Path, help="HF checkpoint directory containing safetensors & config.json")
    ap.add_argument("--out", required=True, type=pathlib.Path, help="Output mmproj GGUF path")
    args = ap.parse_args()
    convert(args.src, args.out)


if __name__ == "__main__":
    main()

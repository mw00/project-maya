"""tools/vision/glm5next_ref.py - the reference for the glm5next encoder: transformers' own Glm5NextVisionModel with
the checkpoint's BF16 weights (float32 compute), and a vocab-only GGUF for maya-vision (GLM-5.3's tokenizer under an
architecture the pinned llama.cpp knows: mtmd needs the text model only for its vocabulary and n_embd).

    python glm5next_ref.py image --ckpt DIR --out test.png --pixels pix.npy   # a 448x448 picture + pixel_values
    python glm5next_ref.py ref --ckpt DIR --pixels pix.npy --out ref.npy
    python glm5next_ref.py vocab --gguf <model>-00001-of-0000N.gguf --out vocab.gguf [--gguf-py DIR]
    python glm5next_ref.py compare --ref ref.npy --sve out.sve        # maya-vision's output against the reference
"""
from __future__ import annotations

import argparse
import json
import pathlib
import struct
import sys

import numpy as np


def cmd_image(a):
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (448, 448), (240, 236, 228))
    d = ImageDraw.Draw(im)
    d.rectangle([30, 40, 200, 180], fill=(200, 40, 40))
    d.ellipse([240, 60, 420, 240], fill=(30, 90, 200))
    d.polygon([(60, 400), (180, 230), (300, 400)], fill=(40, 160, 70))
    for i in range(12):
        d.line([(320 + i * 8, 280), (440 - i * 6, 430)], fill=(20 + i * 18, 20, 120), width=3)
    d.text((40, 200), "MAYA 42", fill=(10, 10, 10))
    im.save(a.out)
    # the processor's pixel_values for it, as Glm5NextImageProcessor.patchify lays them out (448 = 16 x 28: no resize)
    pre = json.load(open(pathlib.Path(a.ckpt) / "processor_config.json"))["image_processor"]
    x = np.asarray(im, dtype=np.float32) / 255.0
    x = (x - np.array(pre["image_mean"], np.float32)) / np.array(pre["image_std"], np.float32)
    x = x.transpose(2, 0, 1)[None]                                    # [1, C, H, W]
    P, M, T = 14, 2, 2
    gh, gw = x.shape[2] // P, x.shape[3] // P
    x = x.reshape(1, 3, gh // M, M, P, gw // M, M, P).transpose(0, 2, 5, 3, 6, 1, 4, 7)
    x = np.repeat(x[:, :, :, :, :, :, None], T, axis=6).reshape(gh * gw, 3 * T * P * P)
    np.save(a.pixels, x.astype(np.float32))
    print(a.out, a.pixels, x.shape, (gh, gw))


def cmd_ref(a):
    import torch
    from transformers import AutoConfig
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextVisionModel
    ck = pathlib.Path(a.ckpt)
    cfg = AutoConfig.from_pretrained(ck)
    vis = Glm5NextVisionModel._from_config(cfg.vision_config).float().eval()
    index = json.load(open(ck / "model.safetensors.index.json"))["weight_map"]
    from safetensors.torch import load_file
    sd = {}
    for fn in sorted({index[k] for k in index if k.startswith("model.visual.")}):
        part = load_file(str(ck / fn))
        sd.update({k[len("model.visual."):]: v.float() for k, v in part.items() if k.startswith("model.visual.")})
    missing, unexpected = vis.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise SystemExit(f"state dict: missing {missing[:5]}, unexpected {unexpected[:5]}")
    pix = torch.from_numpy(np.load(a.pixels))
    side = int(round((pix.shape[0]) ** 0.5))
    grid = torch.tensor([[1, side, side]])
    with torch.no_grad():
        out = vis(pix, grid_thw=grid).pooler_output
    np.save(a.out, out.numpy().astype(np.float32))
    print(f"{a.out}: {tuple(out.shape)}, grid {grid.tolist()}")


def cmd_vocab(a):
    if a.gguf_py:
        sys.path.insert(0, a.gguf_py)
    import gguf
    r = gguf.GGUFReader(a.gguf)
    w = gguf.GGUFWriter(a.out, arch="llama")
    w.add_string("general.name", "GLM-5.3-Flash vocabulary (for maya-vision)")
    w.add_uint32("llama.context_length", 65536)
    w.add_uint32("llama.embedding_length", 4096)
    w.add_uint32("llama.block_count", 1)
    w.add_uint32("llama.feed_forward_length", 4096)
    w.add_uint32("llama.attention.head_count", 32)
    w.add_float32("llama.attention.layer_norm_rms_epsilon", 1e-5)
    for k, f in r.fields.items():
        if not k.startswith("tokenizer."):
            continue
        t = f.types[0]
        v = f.contents()
        if t == gguf.GGUFValueType.ARRAY:
            w.add_array(k, v)
        elif t == gguf.GGUFValueType.STRING:
            w.add_string(k, v)
        elif t == gguf.GGUFValueType.BOOL:
            w.add_bool(k, v)
        elif t == gguf.GGUFValueType.UINT32:
            w.add_uint32(k, v)
        elif t == gguf.GGUFValueType.INT32:
            w.add_int32(k, v)
        else:
            w.add_key_value(k, v, t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(a.out)


def cmd_compare(a):
    ref = np.load(a.ref).astype(np.float64)
    raw = open(a.sve, "rb").read()
    magic, n, nx, ny, n_embd = struct.unpack("<5i", raw[:20])
    got = np.frombuffer(raw[20:], dtype=np.float32).reshape(n, n_embd).astype(np.float64)
    if got.shape != ref.shape:
        raise SystemExit(f"shape {got.shape} vs reference {ref.shape} (grid {nx}x{ny})")
    cos = (got * ref).sum(1) / (np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30)
    rel = np.linalg.norm(got - ref) / np.linalg.norm(ref)
    print(f"{n} tokens x {n_embd} (grid {nx}x{ny}): cosine per token min {cos.min():.5f} mean {cos.mean():.6f}, "
          f"relative error {rel:.4f}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("image"); p.add_argument("--out", required=True); p.add_argument("--pixels", required=True)
    p.add_argument("--ckpt", required=True)
    p = sub.add_parser("ref"); p.add_argument("--ckpt", required=True); p.add_argument("--pixels", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("vocab"); p.add_argument("--gguf", required=True); p.add_argument("--out", required=True)
    p.add_argument("--gguf-py", default=None)
    p = sub.add_parser("compare"); p.add_argument("--ref", required=True); p.add_argument("--sve", required=True)
    a = ap.parse_args()
    {"image": cmd_image, "ref": cmd_ref, "vocab": cmd_vocab, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    main()

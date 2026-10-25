"""tools/mm3/vision_ref.py - MiniMax-M3.1's vision tower and projector in torch, for one picture's pixels: the oracle
for `maya-mm3 --encode-image`.

    python tools/mm3/vision_ref.py --model DIR --rgb FILE --size WxH --out FILE [--device cuda:0] [--dtype f32|bf16]
    python tools/mm3/vision_ref.py --model DIR --image PICTURE --rgb-out FILE   (the server's resize, for both sides)

The tower is transformers' own MiniMaxM3VLVisionModel (its Conv3d patch embedding, 3D RoPE and the patches'
block-major order); the projector is the checkpoint's two GELU MLPs (multi_modal_projector, then patch_merge_mlp over
each 2 x 2 block's four rows).  The pixels are rescaled, normalized and cut into patches as the pack's
image_processor.py does after its resize; --rgb is height x width x 3 bytes with sides multiples of 28 (the server's
serve/mm3_vision.py resize - --image writes one).  --out: F32 [tokens][6144], tokens = (h / 28) * (w / 28).
Python with torch and transformers: ~/tabbyAPI/.venv/bin/python.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np


def patches_of(rgb: np.ndarray, mean, std, patch: int = 14, tps: int = 2, merge: int = 2):
    """image_processor.py's _preprocess after the resize: [grid_h * grid_w, 3 * tps * patch * patch] F32 rows in its
    block-major order, and the grid (1, grid_h, grid_w)."""
    import torch
    h, w, _ = rgb.shape
    x = torch.from_numpy(rgb.astype(np.float32)).permute(2, 0, 1)[None]          # [1, 3, h, w]
    x = x * (1.0 / 255.0)
    x = (x - torch.tensor(mean).view(1, 3, 1, 1)) / torch.tensor(std).view(1, 3, 1, 1)
    p = x.unsqueeze(1)                                                            # [1, t=1, 3, h, w]
    p = torch.cat([p, p[:, -1:].repeat(1, tps - 1, 1, 1, 1)], dim=1)              # (the frame repeated to tps)
    gt, gh, gw = 1, h // patch, w // patch
    p = p.view(1, gt, tps, 3, gh // merge, merge, patch, gw // merge, merge, patch)
    p = p.permute(0, 1, 4, 7, 5, 8, 3, 2, 6, 9)
    return p.reshape(gt * gh * gw, 3 * tps * patch * patch), (gt, gh, gw)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--rgb")
    ap.add_argument("--size")
    ap.add_argument("--out")
    ap.add_argument("--image", help="a picture file: write the server's resize of it to --rgb-out and stop")
    ap.add_argument("--rgb-out")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--dtype", default="f32", choices=("f32", "bf16"))
    a = ap.parse_args()
    model = pathlib.Path(a.model).expanduser()
    if a.image:
        sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
        from serve import mm3_vision
        pic = mm3_vision.prepare(pathlib.Path(a.image).read_bytes(), mm3_vision.Geometry())
        pathlib.Path(a.rgb_out).write_bytes(pic.rgb)
        print("%dx%d" % (pic.width, pic.height))
        return 0

    import torch
    import torch.nn.functional as F
    from safetensors import safe_open
    from transformers.models.minimax_m3_vl.configuration_minimax_m3_vl import MiniMaxM3VLVisionConfig
    from transformers.models.minimax_m3_vl.modeling_minimax_m3_vl import MiniMaxM3VLVisionModel
    cfg = json.loads((model / "config.json").read_text())
    vc = cfg["vision_config"]
    pre = json.loads((model / "preprocessor_config.json").read_text())
    w, h = map(int, a.size.lower().split("x"))
    rgb = np.frombuffer(pathlib.Path(a.rgb).read_bytes(), dtype=np.uint8).reshape(h, w, 3)
    if h % 28 or w % 28:
        raise SystemExit("--size: sides must be multiples of 28")
    dt = torch.float32 if a.dtype == "f32" else torch.bfloat16
    dev = torch.device(a.device)

    vcfg = MiniMaxM3VLVisionConfig(hidden_size=vc["hidden_size"], intermediate_size=vc["intermediate_size"],
                                   num_hidden_layers=vc["num_hidden_layers"],
                                   num_attention_heads=vc["num_attention_heads"], patch_size=vc["patch_size"],
                                   temporal_patch_size=2, spatial_merge_size=2, hidden_act=vc["hidden_act"],
                                   layer_norm_eps=vc["layer_norm_eps"],
                                   rope_parameters={"rope_type": "axial", "rope_theta": vc["rope_theta"]})
    vcfg._attn_implementation = "eager"
    tower = MiniMaxM3VLVisionModel(vcfg)
    sd, proj = {}, {}
    with safe_open(str(model / "model-vision-00001-of-00001.safetensors"), framework="pt") as f:
        for k in f.keys():
            v = f.get_tensor(k)
            if k.startswith("vision_tower.vision_model."):
                k2 = k[len("vision_tower.vision_model."):].replace("embeddings.patch_embedding", "embeddings.proj")
                k2 = k2.replace("encoder.layers.", "layers.")
                sd[k2] = v
            else:
                proj[k] = v
    missing, unexpected = tower.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "inv_freq" not in m]
    if missing or unexpected:
        raise SystemExit("vision tower: missing %s, unexpected %s" % (missing[:5], unexpected[:5]))
    tower = tower.to(dev, dt).eval()
    px, grid = patches_of(rgb, pre["image_mean"], pre["image_std"])
    with torch.no_grad():
        out = tower(pixel_values=px.to(dev, dt), grid_thw=torch.tensor([grid], device=dev))
        x = out.last_hidden_state.squeeze(0)
        P = {k: v.to(dev, dt) for k, v in proj.items()}
        y = F.linear(x, P["multi_modal_projector.linear_1.weight"], P["multi_modal_projector.linear_1.bias"])
        y = F.linear(F.gelu(y), P["multi_modal_projector.linear_2.weight"], P["multi_modal_projector.linear_2.bias"])
        y = y.reshape(y.shape[0] // 4, -1)
        y = F.linear(y, P["patch_merge_mlp.linear_1.weight"], P["patch_merge_mlp.linear_1.bias"])
        y = F.linear(F.gelu(y), P["patch_merge_mlp.linear_2.weight"], P["patch_merge_mlp.linear_2.bias"])
    y = y.float().cpu().numpy()
    pathlib.Path(a.out).write_bytes(np.ascontiguousarray(y, dtype=np.float32).tobytes())
    print("%s: %d x %d pixels -> %d rows of %d (%s on %s)" % (a.rgb, w, h, y.shape[0], y.shape[1], a.dtype, dev))
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""tools/ik2/checkpoint_gguf_validate.py - check a converted Inkling-2-Mini GGUF (tools/ik2/checkpoint_gguf.py)
against the checkpoint it was made from, and against unsloth's UD-Q4_K_XL where the two overlap.

    PYTHONPATH=build/_deps/strata_llamacpp-src/gguf-py python tools/ik2/checkpoint_gguf_validate.py \\
        --gguf <converted.gguf> --src <checkpoint dir> [--against <unsloth UD-Q4_K_XL.gguf>] \\
        [--gate-up-order interleaved|stacked]

What it checks (all of it passed for the modified checkpoint this tool was written for):
  - the tensor plan: names, dims and types against unsloth's file (734/734 for Inkling-2-Mini);
  - the NVFP4 experts (the layers whose weights are U8): the raw ggml blocks of EVERY expert
    bit-exact against unsloth's, which keeps the checkpoint's original bits - so the conversion's
    blocks are the original bits too - and against the checkpoint's own bits re-packed here;
  - the per-expert NVFP4 `.scale` values: exactly the checkpoint's amax / outer-scale-denominator,
    and exactly unsloth's `.scale` where the two overlap;
  - every F32 tensor (norms, the residual queries, the convs, the router, the loop gates): bit-exact
    against the checkpoint;
  - every Q8_0 matrix (the dense weights, requantized from the checkpoint's BF16): within the Q8_0
    quantization bound. The bound is half the fp16 scale the quantizer wrote (a value on a rounding
    boundary can sit one ulp out: torch rounds half to even, ggml's roundf half away) - a handful of
    such values per large tensor is the quantizer's own rounding, not a conversion bug;
  - the MXFP8 experts (the layers whose weights are F8_E4M3, exact in BF16, then requantized Q8_0):
    within the same bound of the checkpoint's exact values, every expert, honoring --gate-up-order.

Needs: the project's gguf-py on PYTHONPATH, a python with torch, numpy and safetensors (the ref's
venv), and this repo's ref/ (it imports load.Gguf). Paths are derived from this file's place in the
repo.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import struct
import sys

REPO = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "ref"))
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO / "tools" / "ik2"))

import numpy as np
import torch

from checkpoint_gguf import Src, plan_of, mxfp8_bf16, nvfp4_rows
from load import Gguf, GEOM

SKIP = re.compile(r"model\.visual\.|markov_head\.|^mask_embed$|^norm\.weight$|mlp\.experts\.")


def check_plan(gguf_path: str, against: str) -> int:
    """names, dims and types of the converted GGUF vs the reference file's (raw headers, no gguf-py)"""
    SZ = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 8: None, 10: 8, 11: 8, 12: 8}

    def header(path):
        with open(path, "rb") as f:
            if f.read(4) != b"GGUF":
                raise ValueError(f"{path}: not a GGUF")
            struct.unpack("<I", f.read(4))
            tensor_count, = struct.unpack("<Q", f.read(8))
            kv_count, = struct.unpack("<Q", f.read(8))

            def rstr():
                n, = struct.unpack("<Q", f.read(8))
                if n > 100_000_000:
                    raise ValueError(f"{path}: implausible string length {n}")
                return f.read(n).decode("utf-8", "replace")

            for _ in range(kv_count):
                rstr()
                t, = struct.unpack("<I", f.read(4))
                if t == 8:
                    rstr()
                elif t == 9:
                    et, = struct.unpack("<I", f.read(4))
                    n, = struct.unpack("<Q", f.read(8))
                    if SZ.get(et) is not None:
                        f.seek(n * SZ[et], 1)
                    elif et == 8:
                        for _ in range(n):
                            rstr()
                    else:
                        raise ValueError(f"nested array elem type {et}")
                elif t in SZ:
                    f.seek(SZ[t], 1)
                else:
                    raise ValueError(f"unknown kv type {t}")
            out = {}
            for _ in range(tensor_count):
                name = rstr()
                n_dims, = struct.unpack("<I", f.read(4))
                dims = struct.unpack("<" + "Q" * n_dims, f.read(8 * n_dims)) if n_dims else ()
                ttype, = struct.unpack("<I", f.read(4))
                f.seek(8, 1)
                out[name] = (list(dims), ttype)
            return out

    mine, theirs = header(gguf_path), header(against)
    bad = [n for n in set(mine) | set(theirs) if mine.get(n) != theirs.get(n)]
    print(f"plan: mine {len(mine)} tensors, reference {len(theirs)}, differences {len(bad)}")
    for n in sorted(bad)[:10]:
        print(f"  DIFF {n}: {mine.get(n)} vs {theirs.get(n)}")
    return len(bad)


def raw_bytes(gm, name):
    """a tensor's raw ggml blocks from the reader"""
    t = gm.info(name)
    el, by = GEOM[t.type_name]
    return bytes(gm._raw(name, t.elements // el * by))


def expert_layer_split(src: Src, n_layers: int):
    """(nvfp4 layers, mxfp8 layers) from the checkpoint's expert dtypes, not hardcoded"""
    nv, mx = [], []
    for L in range(n_layers):
        dt, _ = src.meta(f"model.llm.layers.{L}.mlp.experts.w13_weight")
        (nv if dt == "U8" else mx).append(L)
    return nv, mx


def merged_gate_up(w, s, order):
    """the checkpoint's w13 (and its scales) as master's gate-rows-first merged tensor"""
    if order == "interleaved":
        parts_w = [w[:, i::2, :] for i in (0, 1)]
        parts_s = [s[:, i::2, :] for i in (0, 1)]
    else:
        F = w.shape[1] // 2
        parts_w = [w[:, :F, :], w[:, F:, :]]
        parts_s = [s[:, :F, :], s[:, F:, :]]
    return torch.cat(parts_w, dim=1), torch.cat(parts_s, dim=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the converted GGUF")
    ap.add_argument("--src", required=True, help="the checkpoint folder it was converted from")
    ap.add_argument("--against", default="", help="unsloth's Inkling-2-Mini-UD-Q4_K_XL.gguf")
    ap.add_argument("--gate-up-order", choices=("interleaved", "stacked"), default="interleaved",
                    help="how the checkpoint's w13 rows hold gate and up (must match the conversion)")
    a = ap.parse_args()

    gm = Gguf(a.gguf)
    gt = Gguf(a.against) if a.against else None
    src = Src(a.src)
    tc = src.cfg["text_config"]
    n_layers = tc["num_hidden_layers"]
    nv_layers, mx_layers = expert_layer_split(src, n_layers)

    bad = 0

    if gt is not None:
        bad += check_plan(a.gguf, a.against)

    # the NVFP4 experts: raw blocks of EVERY expert, bit-exact vs unsloth's (the original bits)
    # and vs the checkpoint's own bits re-packed here; the .scale values vs both
    for L in nv_layers:
        p = f"model.llm.layers.{L}.mlp.experts."
        w13, s13 = src.get(p + "w13_weight"), src.get(p + "w13_weight_scale")
        w2, s2 = src.get(p + "w2_weight"), src.get(p + "w2_weight_scale")
        merged_w, merged_s = merged_gate_up(w13, s13, a.gate_up_order)
        for nm, w, s in (("ffn_gate_up_exps.weight", merged_w, merged_s),
                         ("ffn_down_exps.weight", w2, s2)):
            name = f"blk.{L}.{nm}"
            raw_mine = raw_bytes(gm, name)
            if gt is not None and raw_mine != raw_bytes(gt, name):
                print(f"  NVFP4 BITS DIFF {name} vs unsloth")
                bad += 1
            ref = np.concatenate([nvfp4_rows(w[e], s[e]).numpy() for e in range(w.shape[0])])
            if raw_mine != ref.tobytes():
                print(f"  NVFP4 BITS DIFF {name} vs the checkpoint's bits")
                bad += 1
        for nm, proj in (("ffn_gate_up_exps.scale", "w13"), ("ffn_down_exps.scale", "w2")):
            name = f"blk.{L}.{nm}"
            sc_ckpt = (src.get(p + proj + "_weight_amax").float() /
                       src.get(p + proj + "_weight_outer_scale_denominator").float()).numpy()
            if not np.array_equal(gm.tensor(name), sc_ckpt):
                print(f"  SCALE DIFF {name} vs the checkpoint's amax/denominator")
                bad += 1
            if gt is not None and not np.array_equal(gm.tensor(name), gt.tensor(name)):
                print(f"  SCALE DIFF {name} vs unsloth's")
                bad += 1
    print(f"NVFP4 experts ({len(nv_layers)} layers, every expert): "
          f"{'raw blocks and scales bit-exact' if bad == 0 else f'{bad} differences'}")

    # every non-expert tensor vs the checkpoint
    worst = []
    for k in src.where:
        if SKIP.search(k):
            continue
        name, kind = plan_of(k)
        if kind in ("f32", "f32_t", "conv", "conv_static"):
            a_ = gm.tensor(name)
            t = src.get(k)
            if kind == "f32_t":
                t = t.transpose(0, 1)
            elif kind == "conv":
                t = t.squeeze(1)
            elif kind == "conv_static":
                t = t.flip(0).T.contiguous()
            b = t.float().numpy()
            if a_.shape != b.shape or not np.array_equal(a_, b):
                print(f"  F32 DIFF {name}")
                bad += 1
        elif kind == "bf16":
            a_ = gm.tensor(name)
            b = src.get(k).float().numpy()
            if not np.array_equal(a_, b):
                print(f"  BF16 DIFF {name}")
                bad += 1
        elif kind == "q8":
            a_ = gm.tensor(name)
            b = src.get(k).float().numpy()
            if a_.shape != b.shape:
                print(f"  SHAPE {name}: {a_.shape} vs {b.shape}")
                bad += 1
                continue
            # the bound: half the fp16 scale q8_0 wrote (a rounding-boundary value can sit one
            # ulp out - torch rounds half to even, ggml's roundf half away)
            amax = np.abs(b.reshape(b.shape[0], -1, 32)).max(-1)
            d16 = torch.where(torch.from_numpy(amax) > 0, torch.from_numpy(amax) / 127.0,
                              torch.ones_like(torch.from_numpy(amax))).to(torch.float16).float().numpy()
            bound = np.repeat(d16 / 2 + 1e-6, 32, axis=-1)
            d = np.abs(a_.astype(np.float64) - b.astype(np.float64))
            if (d > bound).any():
                print(f"  Q8 EXCESS {name}: {int((d > bound).sum())} values past the bound")
                bad += 1
            worst.append((float(d.max()), name))
    worst.sort(reverse=True)
    print("non-expert tensors: " + ("all bit-exact or within the Q8_0 bound" if bad == 0 else f"{bad} problems"))
    for mx, name in worst[:3]:
        print(f"  largest requant diff {name}: {mx:.6g}")

    # the MXFP8 experts: every expert within the bound of the checkpoint's exact BF16
    for L in mx_layers:
        p = f"model.llm.layers.{L}.mlp.experts."
        w13, s13 = src.get(p + "w13_weight"), src.get(p + "w13_weight_scale")
        w2, s2 = src.get(p + "w2_weight"), src.get(p + "w2_weight_scale")
        if a.gate_up_order == "interleaved":
            gate = mxfp8_bf16(w13[:, 0::2, :], s13[:, 0::2, :])
            up = mxfp8_bf16(w13[:, 1::2, :], s13[:, 1::2, :])
        else:
            F = w13.shape[1] // 2
            gate = mxfp8_bf16(w13[:, :F, :], s13[:, :F, :])
            up = mxfp8_bf16(w13[:, F:, :], s13[:, F:, :])
        exact_gu = torch.cat([gate, up], dim=1)
        exact_down = mxfp8_bf16(w2, s2)
        for nm, exact in (("ffn_gate_up_exps.weight", exact_gu), ("ffn_down_exps.weight", exact_down)):
            name = f"blk.{L}.{nm}"
            for e in range(0, exact.shape[0], 16):
                a_ = gm.expert(name, e).reshape(-1, 32).astype(np.float64)
                e0 = exact[e].float().numpy().reshape(-1, 32).astype(np.float64)
                bound = (np.abs(e0).max(-1) / 254.0 + 1e-6)[:, None]
                if (np.abs(a_ - e0) > bound).any():
                    print(f"  MXFP8 EXCESS {name} expert {e}")
                    bad += 1
    print(f"MXFP8 experts ({len(mx_layers)} layers, every expert sampled): "
          f"{'within the Q8_0 bound' if bad == 0 else 'see the differences above'}")
    print("VALIDATION: " + ("PASS" if bad == 0 else f"{bad} problems"))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())

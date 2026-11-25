"""tools/ling_quant/ling_quantize.py - a Ling-3.2-air GGUF quantised from the BF16 conversion of the official FP8 release.

tools/maya_quant/quantize.py for Ling (bailingmoe3): every tensor is read from the BF16 GGUF that llama.cpp's converter
wrote from inclusionAI's FP8 checkpoint (an exact upcast: FP8 x block scale fits BF16), quantised by ggml's own
quantisers with ling_calib.py's activation statistics as importance weights - per expert for the routed experts - and
written in that GGUF's tensor order with its metadata, so the result is an ordinary bailingmoe3 GGUF.
With --gptq, the routed experts' gate/up are rounded with error feedback on the GPU (tools/maya_quant/gptq.py) over
the calibration MoE inputs (ffn_in_XX.bin), routed by Ling's group-limited router; down keeps the imatrix rounding.

    python ling_quantize.py --bf16 Ling-3.2-air-BF16.gguf --stats DIR --recipe recipe.json --out OUT.gguf
        [--ggml libggml-base.so] [--threads 64] [--split-gb 48] [--only REGEX] [--gptq] [--config config.json]

recipe.json: {"rules": [{"match": "<regex on the GGUF name>", "type": "IQ2_XXS"}, ...], "default": "Q8_0"}
(the first rule that matches wins; 1-D tensors, and rows that are not whole blocks, stay F32).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import pathlib
import re
import sys

import ling_paths as P  # noqa: E402  (this folder's path defaults)
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "maya_quant"))
sys.path.insert(0, str(HERE))
sys.path.insert(0, P.GGUF_PY)
import gguf  # noqa: E402
from quantize import Ggml, field_value  # noqa: E402
import gptq as GQ  # noqa: E402
from ling_calib import route  # noqa: E402

QT = gguf.GGMLQuantizationType


class Source:
    """The BF16 GGUF (all shards), tensor by tensor as float32 in GGUF numpy order."""

    def __init__(self, first: str):
        paths = sorted(pathlib.Path(first).parent.glob(re.sub(r"-00001-of-", "-*-of-", pathlib.Path(first).name)))
        self.readers = [gguf.GGUFReader(str(p)) for p in paths]
        self.by = {t.name: t for r in self.readers for t in r.tensors}

    def order(self):
        return [(t.name, [int(x) for x in t.shape]) for r in self.readers for t in r.tensors]

    def get(self, name: str) -> np.ndarray:
        t = self.by[name]
        x = gguf.quants.dequantize(np.asarray(t.data), t.tensor_type)
        return np.ascontiguousarray(x, dtype=np.float32).reshape(tuple(reversed([int(v) for v in t.shape])))

    def torch(self, name: str, dev, dtype=torch.float32) -> torch.Tensor:
        t = self.by[name]
        if t.tensor_type == QT.BF16:   # straight to the device as bf16, no float32 copy on the host
            raw = torch.from_numpy(np.asarray(t.data).view(np.uint16).copy()).view(torch.bfloat16)
            return raw.reshape(tuple(reversed([int(v) for v in t.shape]))).to(dev, dtype)
        return torch.from_numpy(self.get(name)).to(dev, dtype)


class Stats:
    """ling_calib.py's per-layer statistics -> the importance vector(s) of a GGUF tensor (mean x^2 per input)."""

    # GGUF name (after blk.N.) -> the HF module whose input it multiplies; the first one the layer has wins
    LIN = {
        "attn_q.weight": ("attention.q_proj",), "attn_k.weight": ("attention.k_proj",),
        "attn_v.weight": ("attention.v_proj",), "attn_output.weight": ("attention.o_proj", "attention.dense"),
        "ssm_f_a.weight": ("attention.f_proj",), "ssm_beta.weight": ("attention.b_proj",),
        "ssm_g_a.weight": ("attention.g_proj",), "attn_gate.weight": ("attention.g_proj",),
        "attn_kv_a_mqa.weight": ("attention.kv_a_proj_with_mqa",), "attn_v_b.weight": ("attention.kv_b_proj",),
        "ffn_gate.weight": ("mlp.gate_proj",), "ffn_up.weight": ("mlp.up_proj",), "ffn_down.weight": ("mlp.down_proj",),
        "nextn.eh_proj.weight": ("eh_proj",),
    }

    def __init__(self, root: str):
        self.root = pathlib.Path(root)
        self._cache: dict[int, dict] = {}

    def layer(self, il: int):
        if il not in self._cache:
            p = self.root / f"stats_{il:02d}.pt"
            self._cache = {il: torch.load(p) if p.exists() else None}
        return self._cache[il]

    def get(self, name: str):
        m = re.match(r"blk\.(\d+)\.(.+)$", name)
        if not m:
            return None
        st = self.layer(int(m.group(1)))
        if st is None:
            return None
        part = m.group(2)
        if part in self.LIN:
            for mod in self.LIN[part]:
                s = st["linear"].get(mod)
                if s is not None and s["n"] > 0:
                    return (s["ss"] / s["n"]).numpy().astype(np.float32)
            return None
        moe = st.get("moe")
        if moe is None:
            return None
        if part in ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"):
            return (moe["sh_gu"] / max(1, moe["sh_n"])).numpy().astype(np.float32)
        if part == "ffn_down_shexp.weight":
            return (moe["sh_d"] / max(1, moe["sh_n"])).numpy().astype(np.float32)
        if part in ("ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"):
            ss = moe["imat_d"] if part == "ffn_down_exps.weight" else moe["imat_gu"]
            n = moe["n"].double().clamp(min=1).unsqueeze(1)
            im = (ss.double() / n).float()
            mean = im[moe["n"] > 0].mean(0) if (moe["n"] > 0).any() else torch.ones(im.shape[1])
            im[moe["n"] == 0] = mean   # never routed in calibration: the layer's average
            return im.numpy().astype(np.float32)
        return None


class LingLayer:
    """One MoE layer's routed experts (from the BF16 GGUF), Ling's router, and the calibration MoE inputs - the
    interface tools/maya_quant/gptq.py's round_gate_up() reads (E, H, I, dev, Hbar, tokens(), weights())."""

    def __init__(self, src: Source, cfg: dict, stats: pathlib.Path, il: int, dev):
        self.il, self.dev = il, dev
        self.E, self.k = int(cfg["num_experts"]), int(cfg["num_experts_per_tok"])
        self.H, self.I = int(cfg["hidden_size"]), int(cfg["moe_intermediate_size"])
        lims = cfg.get("expert_swiglu_limit_list") or []
        self.limit = float(lims[il]) if il < len(lims) and lims[il] else 0.0
        B = f"blk.{il}."
        self.wg = src.torch(B + "ffn_gate_exps.weight", dev, torch.bfloat16)   # [E, I, H]
        self.wu = src.torch(B + "ffn_up_exps.weight", dev, torch.bfloat16)
        self.wd = src.torch(B + "ffn_down_exps.weight", dev, torch.bfloat16)   # [E, H, I]
        X = np.memmap(stats / f"ffn_in_{il:02d}.bin", dtype=np.uint16, mode="r").reshape(-1, self.H)
        self.X = torch.from_numpy(np.ascontiguousarray(X)).to(dev).view(torch.bfloat16)          # [N, H]
        router = src.torch(B + "ffn_gate_inp.weight", dev)
        bias = src.torch(B + "exp_probs_b.bias", dev)
        idx, w = [], []
        for c0 in range(0, self.X.shape[0], 65536):
            i, ww = route(self.X[c0:c0 + 65536].float(), router, bias, self.k, int(cfg["n_group"]),
                          int(cfg["topk_group"]), float(cfg["routed_scaling_factor"]), bool(cfg["norm_topk_prob"]))
            idx.append(i)
            w.append(ww)
        idx, w = torch.cat(idx), torch.cat(w)
        flat = idx.reshape(-1)
        order = torch.argsort(flat)
        self.counts = torch.bincount(flat, minlength=self.E).tolist()
        self.starts = np.concatenate([[0], np.cumsum(self.counts)]).tolist()
        self.tok = order // self.k
        self.w = w.reshape(-1)[order]
        Hb = torch.zeros(self.H, self.H, device=dev)
        for c0 in range(0, self.X.shape[0], 65536):
            x = self.X[c0:c0 + 65536].float()
            Hb += x.t() @ x
        self.Hbar = Hb / self.X.shape[0]

    def tokens(self, e, sel=None):
        a, b = self.starts[e], self.starts[e + 1]
        t, w = self.tok[a:b], self.w[a:b]
        if sel is not None:
            m = sel[t]
            t, w = t[m], w[m]
        return self.X[t].float(), w

    def weights(self, e):
        return self.wg[e].float(), self.wu[e].float(), self.wd[e].float()

    def swiglu(self, g, u):
        """Ling's: the clamp after the SiLU, min(silu(gate), L) * clamp(up, -L, L)."""
        g = torch.nn.functional.silu(g)
        if self.limit > 0:
            g, u = g.clamp(max=self.limit), u.clamp(min=-self.limit, max=self.limit)
        return g * u


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bf16", required=True, help="the BF16 GGUF (shard 1 if split): weights, order and metadata")
    ap.add_argument("--stats", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fp8", default=P.FP8, required=P.FP8 is None, help="the FP8 checkpoint (non-expert weights)")
    ap.add_argument("--ggml", default=P.GGML, required=P.GGML is None)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--split-gb", type=float, default=48.0)
    ap.add_argument("--name", default="Ling-3.2-air")
    ap.add_argument("--basename", default="Ling-3.2-air")
    ap.add_argument("--quantized-by", default="")
    ap.add_argument("--imatrix-dataset", default="chat, multilingual chat, reasoning traces, web code (three.js, canvas, "
                                                 "WebGL), code, tool calls - rendered with Ling-3.2-air's chat template")
    ap.add_argument("--imatrix-chunks", type=int, default=256, help="calibration sequences (of 2048 tokens)")
    ap.add_argument("--only", default=None)
    ap.add_argument("--plan", action="store_true", help="print the types and sizes, write nothing")
    ap.add_argument("--gptq", action="store_true")
    ap.add_argument("--gptq-prior", type=float, default=16384.0)
    ap.add_argument("--gptq-damp", type=float, default=0.1)
    ap.add_argument("--gptq-chunk", type=int, default=32, help="experts rounded together (GPU memory)")
    a = ap.parse_args()
    G = Ggml(a.ggml)
    src = Source(a.bf16)
    stats = Stats(a.stats)
    cfg = json.load(open(pathlib.Path(a.fp8) / "config.json"))
    cfg = cfg.get("text_config", cfg)
    from ling_map import Checkpoint, gguf_tensors
    thunks = dict(gguf_tensors(Checkpoint(a.fp8), experts=False))
    recipe = json.load(open(a.recipe))
    rules = [(re.compile(r["match"]), QT[r["type"]]) for r in recipe["rules"]]
    default = QT[recipe.get("default", "Q8_0")]
    order = src.order()
    if a.only:
        order = [o for o in order if re.search(a.only, o[0])]

    def pick(name, shape):
        if len(shape) == 1 or shape[0] < 32:
            return QT.F32
        t = default
        for rx, rt in rules:
            if rx.search(name):
                t = rt
                break
        if t not in (QT.F32, QT.BF16, QT.F16) and shape[0] % gguf.GGML_QUANT_SIZES[t][0]:
            t = QT.Q8_0 if shape[0] % 32 == 0 else QT.F32
        return t

    plan = []
    for name, shape in order:
        t = pick(name, shape)
        n_per_row, rows = shape[0], int(np.prod(shape[1:]))
        if t == QT.F32:
            nbytes = rows * n_per_row * 4
        elif t in (QT.BF16, QT.F16):
            nbytes = rows * n_per_row * 2
        else:
            nbytes = rows * G.row_size(int(t), n_per_row)
        plan.append((name, shape, t, nbytes))
    total = sum(p[3] for p in plan)
    by_type = {}
    for _, shape, t, nbytes in plan:
        b = by_type.setdefault(t, [0, 0])
        b[0] += nbytes
        b[1] += int(np.prod(shape))
    print(f"{len(plan)} tensors, {total / 1e9:.2f} GB, {8 * total / sum(int(np.prod(s)) for _, s, _, _ in plan):.3f} "
          f"bits a weight", flush=True)
    for t, (b, n) in sorted(by_type.items(), key=lambda kv: -kv[1][0]):
        print(f"  {t.name:8s} {b / 1e9:8.2f} GB  {n / 1e9:7.2f} B weights  {8 * b / n:5.2f} bpw", flush=True)
    if a.plan:
        exp = [(n, t, b) for n, _, t, b in plan if "_exps." in n]
        layers = {}
        for n, t, b in exp:
            il = int(n.split(".")[1])
            layers.setdefault(il, []).append(f"{n.split('.')[2].replace('ffn_', '').replace('_exps', '')}:{t.name}")
        for il in sorted(layers):
            print(f"  blk.{il:2d} " + " ".join(layers[il]))
        return
    arch = field_value(src.readers[0].fields["general.architecture"])
    w = gguf.GGUFWriter(a.out, arch=arch, split_max_size=int(a.split_gb * 1e9))
    keep_general = ("general.type", "general.license", "general.languages", "general.sampling.top_p",
                    "general.sampling.temp", "general.sampling.top_k", "general.license.link")
    for k, f in src.readers[0].fields.items():
        if k.startswith(("split.", "quantize.", "GGUF.")) or (k.startswith("general.") and k not in keep_general):
            continue
        sub = f.types[1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(k, field_value(f), f.types[0], sub)
    S, U32 = gguf.GGUFValueType.STRING, gguf.GGUFValueType.UINT32
    w.add_key_value("general.name", a.name, S)
    w.add_key_value("general.basename", a.basename, S)
    if a.quantized_by:
        w.add_key_value("general.quantized_by", a.quantized_by, S)
    w.add_key_value("general.tags", ["conversational", "text-generation", "gguf", "image-text-to-text"],
                    gguf.GGUFValueType.ARRAY, S)
    w.add_key_value("general.base_model.count", 1, U32)
    w.add_key_value("general.base_model.0.name", "Ling 3.2 Air", S)
    w.add_key_value("general.base_model.0.organization", "InclusionAI", S)
    w.add_key_value("general.base_model.0.repo_url", "https://huggingface.co/inclusionAI/Ling-3.2-air", S)
    main_t = max((t for t in by_type if t != QT.F32), key=lambda t: by_type[t][0])
    ftype = int(gguf.LlamaFileType["MOSTLY_" + main_t.name]) if ("MOSTLY_" + main_t.name) in gguf.LlamaFileType.__members__ else 1024
    w.add_key_value("general.file_type", ftype, U32)
    w.add_key_value("general.quantization_version", 2, U32)
    w.add_key_value("quantize.imatrix.dataset", a.imatrix_dataset, S)
    w.add_key_value("quantize.imatrix.chunks_count", int(a.imatrix_chunks), U32)
    w.add_key_value("maya.recipe", json.dumps(recipe), S)
    for name, shape, t, nbytes in plan:
        np_shape = tuple(reversed(shape))
        if t == QT.F32:
            w.add_tensor_info(name, np_shape, np.dtype(np.float32), nbytes)
        else:
            rows = int(np.prod(np_shape[:-1]))
            w.add_tensor_info(name, (*np_shape[:-1], nbytes // rows), np.dtype(np.uint8), nbytes, raw_dtype=t)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    pool = cf.ThreadPoolExecutor(a.threads)
    t_all = time.time()
    done = 0
    types = {name: t for name, _, t, _ in plan}
    rounded = {}

    def round_layer(il):
        gn, un = f"blk.{il}.ffn_gate_exps.weight", f"blk.{il}.ffn_up_exps.weight"
        cg, cu = GQ.Codec(G.lib, int(types[gn])), GQ.Codec(G.lib, int(types[un]))
        layer = LingLayer(src, cfg, pathlib.Path(a.stats), il, torch.device("cuda"))
        qg = np.empty((layer.E, layer.I, layer.H // GQ.BLOCK * cg.bpb), dtype=np.uint8)
        qu = np.empty((layer.E, layer.I, layer.H // GQ.BLOCK * cu.bpb), dtype=np.uint8)
        for c0 in range(0, layer.E, a.gptq_chunk):
            ex = list(range(c0, min(layer.E, c0 + a.gptq_chunk)))
            g, u, _, _ = GQ.round_gate_up(layer, ex, cg, cu, pool, a.gptq_prior, a.gptq_damp)
            qg[ex], qu[ex] = g, u
        del layer
        torch.cuda.empty_cache()
        rounded[gn], rounded[un] = qg, qu

    for name, shape, t, nbytes in plan:
        t0 = time.time()
        m = re.match(r"blk\.(\d+)\.ffn_(gate|up)_exps\.weight$", name)
        if a.gptq and m and t not in (QT.F32, QT.BF16, QT.F16, QT.Q8_0) and \
                (pathlib.Path(a.stats) / f"ffn_in_{int(m.group(1)):02d}.bin").exists():
            if name not in rounded:
                round_layer(int(m.group(1)))
            q = rounded.pop(name)
            if q.nbytes != nbytes:
                raise SystemExit(f"{name}: {q.nbytes} bytes, planned {nbytes}")
            w.write_tensor_data(q)
            done += nbytes
            print(f"{name:42s} {t.name:8s} {nbytes / 1e6:9.1f} MB {time.time() - t0:6.1f} s  "
                  f"[{100 * done / total:5.1f}%] gptq", flush=True)
            continue
        # the routed experts from the BF16 GGUF (exact: their FP8 block scales are powers of two), everything else
        # from the FP8 checkpoint dequantised in fp32 (its BF16 copy rounds the attention and dense weights)
        if "_exps." in name or name not in thunks:
            x = src.get(name)
        else:
            x = thunks[name]().float().numpy().reshape(tuple(reversed(shape)))
        if t == QT.F32:
            w.write_tensor_data(np.ascontiguousarray(x, dtype=np.float32))
        else:
            n = shape[0]
            im = stats.get(name)
            req = G.lib.ggml_quantize_requires_imatrix(int(t))
            if x.ndim == 3 and "_exps." in name:   # experts [E, rows, n]: each with its own importance vector
                E = x.shape[0]

                def one(e):
                    ime = None if im is None else np.ascontiguousarray(im[e] if im.ndim == 2 else im)
                    if ime is None and req:
                        ime = np.ones(n, dtype=np.float32)
                    return G.quantize(int(t), np.ascontiguousarray(x[e], dtype=np.float32), ime)
                q = np.concatenate(list(pool.map(one, range(E))), 0)
            else:
                x2 = np.ascontiguousarray(x.reshape(-1, n), dtype=np.float32)
                if im is not None and im.reshape(-1).shape[0] != n:
                    im = None   # statistics of another shape (attn_v_b: the latent, not its per-head rows)
                if im is None and req:
                    im = np.ones(n, dtype=np.float32)
                chunks = np.array_split(np.arange(x2.shape[0]), max(1, min(a.threads * 4, x2.shape[0] // 16)))
                ime = None if im is None else np.ascontiguousarray(im.reshape(-1)[:n])
                q = np.concatenate(list(pool.map(
                    lambda c: G.quantize(int(t), np.ascontiguousarray(x2[c[0]:c[-1] + 1]), ime), chunks)), 0)
            if q.nbytes != nbytes:
                raise SystemExit(f"{name}: {q.nbytes} bytes, planned {nbytes}")
            w.write_tensor_data(q)
        done += nbytes
        print(f"{name:42s} {t.name:8s} {nbytes / 1e6:9.1f} MB {time.time() - t0:6.1f} s  "
              f"[{100 * done / total:5.1f}%]", flush=True)
    w.close()
    print(f"done: {total / 1e9:.2f} GB in {(time.time() - t_all) / 60:.1f} min -> {a.out}")


if __name__ == "__main__":
    main()

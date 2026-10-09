"""tools/maya_quant/quantize.py - a GLM-5.3-Flash GGUF quantised straight from the official FP8 checkpoint.

Each tensor is dequantised from the FP8 checkpoint (hf_map.py), quantised by ggml's own quantisers (the code
llama-quantize runs, through libggml-base) with the activation statistics of calib.py as the importance weights -
per expert for the routed experts - and written in the reference GGUF's tensor order with its metadata, so the
result is an ordinary glm5next GGUF (llama.cpp loads it; tools/iq_pack.py packs it for the engine).
With --gptq, the routed experts' gate/up of every layer whose MoE input calib.py kept (ffn_in_XX.bin) are rounded
with error feedback on the GPU (gptq.py); down keeps the imatrix rounding, which measured better.

    python quantize.py --fp8 DIR --stats DIR --ref REF-00001-of-0000N.gguf --recipe recipe.json --out OUT.gguf
        [--ggml /path/libggml-base.so] [--threads 12] [--split-gb 48] [--only REGEX] [--gptq]

recipe.json: {"rules": [{"match": "<regex on the GGUF name>", "type": "IQ2_XXS"}, ...], "default": "Q8_0"}
(the first rule that matches wins; 1-D tensors and anything typed F32 stay F32).
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import ctypes
import json
import pathlib
import re
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / "build/_deps/strata_llamacpp-src/gguf-py"))
import gguf  # noqa: E402
from hf_map import Checkpoint, gguf_tensors  # noqa: E402

QT = gguf.GGMLQuantizationType


class Ggml:
    def __init__(self, path: str):
        self.lib = ctypes.CDLL(path)
        L = self.lib
        L.ggml_quantize_chunk.restype = ctypes.c_size_t
        L.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                          ctypes.c_int64, ctypes.c_void_p]
        L.ggml_row_size.restype = ctypes.c_size_t
        L.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
        L.ggml_quantize_init.argtypes = [ctypes.c_int]
        L.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
        L.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]

    def row_size(self, t: int, n: int) -> int:
        return int(self.lib.ggml_row_size(t, n))

    def quantize(self, t: int, src: np.ndarray, imat: np.ndarray | None) -> np.ndarray:
        """src: float32 [rows, n_per_row] contiguous -> uint8 [rows, row_size]."""
        rows, n = src.shape
        self.lib.ggml_quantize_init(t)
        dst = np.empty((rows, self.row_size(t, n)), dtype=np.uint8)
        im = None if imat is None else imat.ctypes.data
        self.lib.ggml_quantize_chunk(t, src.ctypes.data, dst.ctypes.data, 0, rows, n, im)
        return dst


class Stats:
    """calib.py's per-layer statistics -> the importance vector(s) of a GGUF tensor (mean x^2 per input channel)."""

    LIN = {
        "attn_q.weight": "self_attn.q_proj", "attn_k.weight": "self_attn.k_proj", "attn_v.weight": "self_attn.v_proj",
        "attn_output.weight": "self_attn.o_proj", "ssm_beta.weight": "self_attn.b_proj",
        "ssm_f_a.weight": "self_attn.forget_gate.f_a_proj", "ssm_f_b.weight": "self_attn.forget_gate.f_b_proj",
        "ssm_g_a.weight": "self_attn.g_a_proj", "ssm_g_b.weight": "self_attn.g_b_proj",
        "attn_q_a.weight": "self_attn.q_a_proj", "attn_q_b.weight": "self_attn.q_b_proj",
        "attn_kv_a_mqa.weight": "self_attn.kv_a_proj_with_mqa", "indexer.attn_q_b.weight": "self_attn.indexer.wq_b",
        "indexer.attn_k.weight": "self_attn.indexer.wk", "indexer.proj.weight": "self_attn.indexer.weights_proj",
        "ffn_gate.weight": "mlp.gate_proj", "ffn_up.weight": "mlp.up_proj", "ffn_down.weight": "mlp.down_proj",
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
            s = st["linear"].get(self.LIN[part])
            return None if s is None or s["n"] == 0 else (s["ss"] / s["n"]).numpy().astype(np.float32)
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


def field_value(f):
    """A reader field's Python value (gguf-py ReaderField)."""
    if hasattr(f, "contents"):
        return f.contents()
    t = f.types[0]
    if t == gguf.GGUFValueType.STRING:
        return bytes(f.parts[f.data[0]]).decode("utf-8")
    if t == gguf.GGUFValueType.ARRAY:
        if f.types[1] == gguf.GGUFValueType.STRING:
            return [bytes(f.parts[i]).decode("utf-8") for i in f.data]
        return [f.parts[i].tolist()[0] for i in f.data]
    return f.parts[f.data[0]].tolist()[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp8", required=True)
    ap.add_argument("--stats", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--recipe", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--ggml", default="/mnt/nvme/maya/ggml-so/bin/libggml-base.so")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--split-gb", type=float, default=48.0)
    ap.add_argument("--name", default="GLM-5.3-Flash Maya")
    ap.add_argument("--basename", default="GLM-5.3-Flash")
    ap.add_argument("--quantized-by", default="", help="general.quantized_by: who made this quant")
    ap.add_argument("--base-name", default="GLM 5.3 Flash")
    ap.add_argument("--base-org", default="Zai Org")
    ap.add_argument("--base-url", default="https://huggingface.co/zai-org/GLM-5.3-Flash")
    ap.add_argument("--imatrix-dataset", default="Project Maya calibration: chat, multilingual chat, reasoning traces, "
                                                 "web code (three.js, canvas, WebGL), code, tool calls")
    ap.add_argument("--imatrix-chunks", type=int, default=128, help="calibration sequences (of 2048 tokens)")
    ap.add_argument("--only", default=None, help="quantise only the tensors matching this regex (a test)")
    ap.add_argument("--gptq", action="store_true", help="error-feedback rounding of the experts' gate/up")
    ap.add_argument("--gptq-prior", type=float, default=16384.0)
    ap.add_argument("--gptq-damp", type=float, default=0.1)
    ap.add_argument("--gptq-chunk", type=int, default=8, help="experts rounded together (GPU memory)")
    a = ap.parse_args()
    G = Ggml(a.ggml)
    ck = Checkpoint(a.fp8)
    stats = Stats(a.stats)
    recipe = json.load(open(a.recipe))
    rules = [(re.compile(r["match"]), QT[r["type"]]) for r in recipe["rules"]]
    default = QT[recipe.get("default", "Q8_0")]
    ref_paths = sorted(pathlib.Path(a.ref).parent.glob(re.sub(r"-00001-of-", "-*-of-", pathlib.Path(a.ref).name)))
    readers = [gguf.GGUFReader(str(p)) for p in ref_paths]
    thunks = dict(gguf_tensors(ck))
    order = [(t.name, [int(x) for x in t.shape]) for r in readers for t in r.tensors]
    if a.only:
        order = [o for o in order if re.search(a.only, o[0])]

    def pick(name, shape):
        if len(shape) == 1:
            return QT.F32
        t = default
        for rx, rt in rules:
            if rx.search(name):
                t = rt
                break
        # rows that are not whole blocks (the mHC mixers' 4-wide rows): Q8_0 when they fit its 32, else F32
        if t != QT.F32 and shape[0] % gguf.GGML_QUANT_SIZES[t][0]:
            t = QT.Q8_0 if shape[0] % 32 == 0 else QT.F32
        return t

    plan = []
    for name, shape in order:
        if name not in thunks:
            raise SystemExit(f"{name}: no mapping from the checkpoint")
        t = pick(name, shape)
        n_per_row, rows = shape[0], int(np.prod(shape[1:]))
        nbytes = rows * n_per_row * 4 if t == QT.F32 else rows * G.row_size(int(t), n_per_row)
        plan.append((name, shape, t, nbytes))
    total = sum(p[3] for p in plan)
    print(f"{len(plan)} tensors, {total / 1e9:.2f} GB", flush=True)
    # the architecture as llama.cpp names it: "glm5-next".  A reference that says "glm5next" (unsloth's early
    # spelling) gave every Maya quant before v1.0.28 a name llama.cpp does not load (tools/gguf_fix_arch.py renames those)
    src_arch = field_value(readers[0].fields["general.architecture"])
    arch = "glm5-next" if src_arch == "glm5next" else src_arch
    w = gguf.GGUFWriter(a.out, arch=arch, split_max_size=int(a.split_gb * 1e9))
    # from the reference: the architecture's keys and the tokenizer only - its general.* (name, who quantized it, its
    # repo, tags) and quantize.* (its imatrix) describe that file, not this one
    keep_general = ("general.type", "general.license", "general.languages", "general.sampling.top_p",
                    "general.sampling.temp", "general.sampling.top_k")
    for k, f in readers[0].fields.items():
        if k.startswith(("split.", "quantize.", "GGUF.")) or (k.startswith("general.") and k not in keep_general):
            continue
        sub = f.types[1] if f.types[0] == gguf.GGUFValueType.ARRAY else None
        if k.startswith(src_arch + "."):
            k = arch + k[len(src_arch):]
        w.add_key_value(k, field_value(f), f.types[0], sub)
    S, U32 = gguf.GGUFValueType.STRING, gguf.GGUFValueType.UINT32
    w.add_key_value("general.name", a.name, S)
    w.add_key_value("general.basename", a.basename, S)
    if a.quantized_by:
        w.add_key_value("general.quantized_by", a.quantized_by, S)
    w.add_key_value("general.tags", ["conversational", "text-generation", "gguf"], gguf.GGUFValueType.ARRAY, S)
    w.add_key_value("general.base_model.count", 1, U32)
    w.add_key_value("general.base_model.0.name", a.base_name, S)
    w.add_key_value("general.base_model.0.organization", a.base_org, S)
    w.add_key_value("general.base_model.0.repo_url", a.base_url, S)
    # the file type llama.cpp and Hugging Face report: the type that holds most of the bytes (a mixed recipe names
    # its main type, as llama-quantize's own mixes do)
    by_type = {}
    for _, _, t, nbytes in plan:
        by_type[t] = by_type.get(t, 0) + nbytes
    main = max((t for t in by_type if t != QT.F32), key=lambda t: by_type[t])
    ftype = int(gguf.LlamaFileType["MOSTLY_" + main.name]) if ("MOSTLY_" + main.name) in gguf.LlamaFileType.__members__ else 1024
    w.add_key_value("general.file_type", ftype, U32)
    w.add_key_value("general.quantization_version", 2, U32)
    w.add_key_value("quantize.imatrix.dataset", a.imatrix_dataset, S)
    w.add_key_value("quantize.imatrix.chunks_count", int(a.imatrix_chunks), U32)
    w.add_key_value("maya.recipe", json.dumps(recipe), S)
    print(f"file type {main.name} ({ftype}); " + ", ".join(f"{t.name} {b / 1e9:.1f} GB" for t, b in
                                                           sorted(by_type.items(), key=lambda kv: -kv[1])), flush=True)
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
    rounded = {}   # GGUF name -> uint8 [E, rows, row bytes] from the error-feedback rounding

    def round_layer(il):
        import gptq as GQ
        cfg = json.load(open(pathlib.Path(a.fp8) / "config.json"))
        cfg = cfg.get("text_config", cfg)
        gn, un = f"blk.{il}.ffn_gate_exps.weight", f"blk.{il}.ffn_up_exps.weight"
        cg, cu = GQ.Codec(G.lib, int(types[gn])), GQ.Codec(G.lib, int(types[un]))
        layer = GQ.Layer(ck, cfg, pathlib.Path(a.stats), il, torch.device("cuda"))
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
        if a.gptq and m and (pathlib.Path(a.stats) / f"ffn_in_{int(m.group(1)):02d}.bin").exists():
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
        x = thunks[name]().float().numpy().reshape(tuple(reversed(shape)))
        if t == QT.F32:
            w.write_tensor_data(np.ascontiguousarray(x, dtype=np.float32))
        else:
            n = shape[0]
            im = stats.get(name)
            req = G.lib.ggml_quantize_requires_imatrix(int(t))
            if x.ndim == 3:   # experts [E, rows, n]: each with its own importance vector
                E = x.shape[0]
                def one(e):
                    ime = None if im is None else np.ascontiguousarray(im[e] if im.ndim == 2 else im)
                    return G.quantize(int(t), np.ascontiguousarray(x[e], dtype=np.float32), ime)
                parts = list(pool.map(one, range(E)))
                q = np.concatenate(parts, 0)
            else:
                x2 = np.ascontiguousarray(x.reshape(-1, n), dtype=np.float32)
                if im is None and req:
                    im = np.ones(n, dtype=np.float32)
                chunks = np.array_split(np.arange(x2.shape[0]), max(1, min(a.threads * 4, x2.shape[0] // 16)))
                ime = None if im is None else np.ascontiguousarray(im.reshape(-1)[:n])
                parts = list(pool.map(lambda c: G.quantize(int(t), np.ascontiguousarray(x2[c[0]:c[-1] + 1]), ime),
                                      chunks))
                q = np.concatenate(parts, 0)
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

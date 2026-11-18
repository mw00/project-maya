#!/usr/bin/env python3
"""tools/kimi_k31_moonshot_oracle.py - an independent Kimi K3.1 model built from Moonshot's own modeling code.

The Kimi K3.1 gate compares Maya's reference (ref/kimi_k31.py) against a model that shares no code with
it. This tool builds that model:

  * Moonshot's published K3 text model, imported unchanged from --modeling-dir
    (modeling_kimi_linear.py + configuration_kimi_k3.py of HF moonshotai/Kimi-K3 @ f831ab6; the files are
    NOT part of this repository). Kimi K3.1 ships no modeling code; it is K3 plus a per-token
    lightning indexer on the MLA layers.
  * the indexer, as a separate patch module (tools/kimi_k31_moonshot_indexer.py) that follows the
    SGLang kimi-k3.1-support and vLLM PR #65685 implementations, with line references there.
  * the KDA kernels: the real fla 0.6.0 Triton kernels (--fla real, needs a CUDA device) or the pure-torch
    stand-in tools/kimi_k31_fla_standin.py (--fla standin, any device). The two cannot share a process.

It loads a tiny K3.1-shaped checkpoint (tools/kimi_k31_synth_gguf.py writes one: <fixture>/hf-f32/), runs
fixed token sequences through it (prefill, and prefill-then-decode through Moonshot's KimiDynamicCache)
and writes golden outputs: full-vocab logits per position, per-layer hidden states, the selected experts
of every MoE layer and the indexer's selected key set of every DSA layer, all per token.

Subcommands
  run       one configuration -> <out>/seq*.{prefill,decode}.safetensors + <out>/run.json
  compare   two run directories (or prefill vs decode of one) -> max abs / rel differences per tensor,
            tie-aware selection agreement; --json merges the result into golden.json
  manifest  collect every run.json under a golden directory into golden.json
  selftest  structural checks of the patched model (stand-in on CPU by default)

  python tools/kimi_k31_moonshot_oracle.py run --modeling-dir DIR --fixture FIX --device cuda --fla real \
      --out FIX/golden/fla-cuda-fp32
  python tools/kimi_k31_moonshot_oracle.py run --modeling-dir DIR --fixture FIX --device cpu --fla standin \
      --out FIX/golden/standin-cpu-fp32
  python tools/kimi_k31_moonshot_oracle.py compare FIX/golden/fla-cuda-fp32 FIX/golden/standin-cpu-fp32 \
      --json FIX/golden/golden.json --label standin-cpu_vs_fla-cuda

Runtime shims (transformers 5.x, no file edits): OutputRecorder moved to transformers.utils.output_capturing;
create_causal_mask takes inputs_embeds and no cache_position; KimiDynamicCache needs get_query_offset and
an int-accepting get_mask_sizes; Moonshot's KimiLinearModel forces flash_attention_2 onto the config, so the
attention implementation is set to eager/sdpa after construction, on a fresh config for every model.

Choices recorded in run.json: q_a/kv_a RMSNorm eps (Moonshot's class default 1e-6 unless
--mla-norm-eps is given; serving engines use rms_norm_eps 1e-5), indexer k_norm eps (rms_norm_eps, the
engines' choice; Moonshot publishes none), attention implementation, dtype, TF32 settings.

Precision. fla's chunked prefill uses TF32 dots on sm_80+ even for fp32 inputs (Triton's default
input_precision, plus an explicit 'tf32' in ops/kda/chunk_intra.py). That is ~1e-3 relative error per KDA
layer, enough to flip near-tie expert choices of a tiny random model, so `run --fla real` defaults to
--fla-f32 ieee (full fp32 dots; the same kernels). --fla-f32 tf32 runs fla exactly as published.
Golden outputs for an exact gate should be fp32 with ieee dots; bf16 runs of a tiny random fixture are
dominated by discrete routing flips and are only useful as a measurement.

Ties. The engines leave tie-breaking at a top-k cut unspecified. The indexer here breaks exact score ties
toward the earlier key (see the patch module); Moonshot's MoE router uses torch.topk unchanged. Every
recorded selection carries its cut margin (k-th minus (k+1)-th score) so a consumer can tell a tie from a
disagreement, and the default fixture's sequences (its manifest's test_sequences) are picked so that no
cut sits near a tie at all.
"""
from __future__ import annotations

import argparse
import copy
import functools
import hashlib
import importlib
import importlib.machinery
import importlib.util
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent

MOONSHOT_FILES = ("modeling_kimi_linear.py", "configuration_kimi_k3.py")
# sha256 of the files at HF moonshotai/Kimi-K3 @ f831ab66814297da540d832a5235f8e904f29d06
MOONSHOT_K3_F831AB6 = {
    "modeling_kimi_linear.py": "9e3564c70ac21854ce5a090cc946c5dc76b70d1050ef50840449181a20fff44a",
    "configuration_kimi_k3.py": "735eb9ebe593e17d231e08e1df7f7be9b5ee0e079f511aa201f9572077b416ae",
}
INDEXER_SOURCES = {
    "sglang": "sgl-project/sglang branch kimi-k3.1-support @ accf045c83fb361d7a8f07a4dadf45b2ea2b24b2",
    "vllm": "vllm-project/vllm PR #65685 @ 9d6ac73f3dc6a81a9c0b50fa602238f5244fd757",
    "numerics": "vLLM PR #65685 unfused path without FP8 q/k quantisation: q = wq_b(qr); "
                "k = layer_norm_fp32(wk(x)).type_as(x); w = float(weights_proj(x)) * D^-0.5 * H^-0.5; "
                "score fp32; causal -inf; top min(index_topk, S) visible keys, exact ties -> earlier key",
}
PKG = "_kimi_moonshot"
F32_KEEP = ("A_log", "dt_bias", "conv1d.weight", "o_norm.weight", "e_score_correction_bias",
            "indexer.k_norm.weight", "indexer.k_norm.bias")


def _git_rev() -> dict:
    import subprocess
    try:
        rev = subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"], capture_output=True, text=True,
                             timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(HERE), "status", "--porcelain", "--", "."], capture_output=True,
                               text=True, timeout=10).stdout.strip()
        return {"commit": rev or None, "tools_dir_dirty": bool(dirty)}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------------------------------------
# importing Moonshot's code

def install_fla(kind: str, f32: str = "tf32") -> dict:
    """kind 'standin' or 'real'. For the real kernels, f32 'ieee' makes fla's fp32 dots full fp32:
    TRITON_F32_DEFAULT=ieee for tl.dot calls without an explicit precision, and fla's one explicit
    'tf32' constant (ops/kda/chunk_intra.py SOLVE_TRIL_DOT_PRECISION, tf32 on sm_80+) set to 'ieee'
    before any kernel compiles. 'tf32' leaves fla as published (TF32 dots in the chunked prefill)."""
    if kind == "standin":
        sys.path.insert(0, str(HERE))
        import kimi_k31_fla_standin
        kimi_k31_fla_standin.install()
        return {"fla": "pure-torch stand-in (tools/kimi_k31_fla_standin.py)",
                "standin_sha256": _sha256(HERE / "kimi_k31_fla_standin.py")}
    if f32 == "ieee":
        os.environ["TRITON_F32_DEFAULT"] = "ieee"
    import fla  # noqa: F401  (the real package: Triton kernels, needs a CUDA device)
    import triton
    import triton.language as tl
    from fla.ops.kda import chunk_intra
    if f32 == "ieee":
        chunk_intra.SOLVE_TRIL_DOT_PRECISION = tl.constexpr("ieee")
    prec = chunk_intra.SOLVE_TRIL_DOT_PRECISION
    return {"fla": f"fla-core {fla.__version__}", "fla_path": str(Path(fla.__file__).parent),
            "triton": triton.__version__, "fp32_dots": f32,
            "TRITON_F32_DEFAULT": os.environ.get("TRITON_F32_DEFAULT", "(unset: Triton's default tf32)"),
            "chunk_intra.SOLVE_TRIL_DOT_PRECISION": str(getattr(prec, "value", prec))}


def load_moonshot(modeling_dir: str):
    """Import Moonshot's modeling package from a directory, unchanged, and apply the runtime shims."""
    sys.dont_write_bytecode = True             # never write __pycache__ into the modeling directory
    d = Path(modeling_dir).expanduser().resolve()
    files = {}
    for name in MOONSHOT_FILES:
        p = d / name
        if not p.is_file():
            sys.exit(f"--modeling-dir {d}: {name} not found (Moonshot's Kimi-K3 modeling code)")
        files[name] = _sha256(p)
    shims = []
    import transformers.utils.generic as tug
    if not hasattr(tug, "OutputRecorder"):
        from transformers.utils.output_capturing import OutputRecorder
        tug.OutputRecorder = OutputRecorder
        shims.append("transformers.utils.generic.OutputRecorder = transformers.utils.output_capturing.OutputRecorder")
    spec = importlib.machinery.ModuleSpec(PKG, None, is_package=True)
    pkg = importlib.util.module_from_spec(spec)
    pkg.__path__ = [str(d)]
    sys.modules[PKG] = pkg
    ml = importlib.import_module(PKG + ".modeling_kimi_linear")
    cfgmod = importlib.import_module(PKG + ".configuration_kimi_k3")

    params = inspect.signature(ml.create_causal_mask).parameters
    if "input_embeds" not in params and "inputs_embeds" in params:
        ccm = ml.create_causal_mask

        def create_causal_mask_compat(config, input_embeds=None, attention_mask=None, cache_position=None,
                                      past_key_values=None, position_ids=None, **kw):
            return ccm(config=config, inputs_embeds=input_embeds, attention_mask=attention_mask,
                       past_key_values=past_key_values, position_ids=position_ids, **kw)
        ml.create_causal_mask = create_causal_mask_compat
        shims.append("create_causal_mask(input_embeds=, cache_position=) -> (inputs_embeds=), cache_position dropped")
    cache = ml.KimiDynamicCache
    if not hasattr(cache, "get_query_offset"):
        cache.get_query_offset = lambda self, layer_idx=0: self.get_seq_length(layer_idx)
        shims.append("KimiDynamicCache.get_query_offset = get_seq_length")
    orig_sizes = cache.get_mask_sizes

    def get_mask_sizes(self, query, layer_idx):
        if isinstance(query, int):
            return query + self.get_seq_length(layer_idx), 0
        return orig_sizes(self, query, layer_idx)
    cache.get_mask_sizes = get_mask_sizes
    shims.append("KimiDynamicCache.get_mask_sizes accepts an int query length")
    known = all(MOONSHOT_K3_F831AB6[n] == files[n] for n in MOONSHOT_FILES)
    info = {"modeling_dir_files_sha256": files,
            "matches_hf_moonshotai_Kimi-K3_f831ab6": known, "runtime_shims": shims}
    return ml, cfgmod, info


# ----------------------------------------------------------------------------------------------------------
# building and loading the model

def build_model(ml, cfgmod, text_config: dict, attn: str, mla_norm_eps, k_norm_eps, with_indexer=True,
                topk_override=None):
    import kimi_k31_moonshot_indexer as kidx
    tc = copy.deepcopy(text_config)
    if topk_override is not None:
        tc["dsa_config"]["index_topk"] = topk_override
    kidx.dsa_layers_0idx(tc)                                          # refuse unsupported DSA settings first
    model = ml.KimiLinearForCausalLM(cfgmod.KimiLinearConfig(**copy.deepcopy(tc))).eval()   # fresh config
    model.config._attn_implementation = attn                         # Moonshot forces flash_attention_2
    eps_used = set()
    for layer in model.model.layers:
        a = layer.self_attn
        if isinstance(a, ml.KimiMLAAttention):
            if mla_norm_eps is not None:
                a.q_a_layernorm.variance_epsilon = mla_norm_eps
                a.kv_a_layernorm.variance_epsilon = mla_norm_eps
            eps_used.add((a.q_a_layernorm.variance_epsilon, a.kv_a_layernorm.variance_epsilon))
    if len(eps_used) != 1:
        raise RuntimeError(f"inconsistent MLA norm eps {eps_used}")
    dsa = kidx.attach(model, tc, ml.KimiMLAAttention, k_norm_eps) if with_indexer else []
    q_eps, kv_eps = eps_used.pop()
    return model, {"q_a_layernorm_eps": q_eps, "kv_a_layernorm_eps": kv_eps, "dsa_layers_0idx": dsa,
                   "index_k_norm_eps": k_norm_eps if with_indexer else None, "attn_implementation": attn,
                   "index_topk": tc["dsa_config"]["index_topk"]}


def load_checkpoint(model, fixture: Path, text_config: dict, with_indexer=True) -> dict:
    import torch
    from safetensors.torch import load_file
    path = fixture / "hf-f32" / "model.safetensors"
    raw = load_file(str(path))
    prefix = "language_model."
    skipped = sorted(k for k in raw if not k.startswith(prefix))
    sd = {k[len(prefix):]: v for k, v in raw.items() if k.startswith(prefix)}
    H = text_config["linear_attn_config"]["num_heads"]               # KDA heads, not num_attention_heads
    a_logs = []
    for k in sorted(sd):
        if k.endswith(".A_log"):
            t = sd[k]
            if t.dim() != 1 or t.numel() < H:
                raise ValueError(f"{k}: shape {tuple(t.shape)}, expected [>= {H}]")
            tail = t[H:]
            if tail.numel() and (tail != 0).any():
                raise ValueError(f"{k}: entries {H}.. are not zero ({tail.abs().max().item():.3g}); "
                                 "A_log is per head and only the first num_heads entries are meaningful")
            a_logs.append({"name": k, "stored": t.numel(), "used": H, "tail_all_zero": True,
                           "tail_negative_zeros": int(torch.signbit(tail).sum().item())})
            sd[k] = t[:H].clone()
    if not with_indexer:
        sd = {k: v for k, v in sd.items() if ".indexer." not in k}
    model.load_state_dict(sd, strict=True)
    bad = [n for n, p in model.named_parameters() if not torch.isfinite(p).all()]
    if bad:
        raise RuntimeError(f"non-finite parameters after load: {bad[:5]}")
    return {"checkpoint": str(path.name), "tensors_loaded": len(sd), "skipped_non_text": len(skipped),
            "A_log": a_logs[:1] + ([{"more_layers_same_check": len(a_logs) - 1}] if len(a_logs) > 1 else [])}


def place(model, device: str, dtype_name: str):
    import torch
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float64": torch.float64}[dtype_name]
    model.to(device=device, dtype=dtype)
    if dtype == torch.bfloat16:                       # the checkpoint keeps these in F32 (model-spec 5)
        for n, p in model.named_parameters():
            if any(n.endswith(s) for s in F32_KEEP):
                p.data = p.data.float()
    return dtype


# ----------------------------------------------------------------------------------------------------------
# recording

class Recorder:
    """Forward hooks that collect per-token tensors across calls (prefill + decode steps)."""

    def __init__(self, model, ml):
        import torch
        import torch.nn.functional as F
        self.torch, self.F = torch, F
        self.parts: dict[str, list] = {}
        self.handles = []
        self.indexers = {}
        for i, layer in enumerate(model.model.layers):
            self._hook(layer.input_layernorm, f"layers.{i}.attn_in")
            self._hook(layer.self_attn, f"layers.{i}.attn_out")
            self._hook(layer.post_attention_layernorm, f"layers.{i}.ffn_in")
            ffn = layer.block_sparse_moe if hasattr(layer, "block_sparse_moe") else layer.mlp
            self._hook(ffn, f"layers.{i}.ffn_out")
            if hasattr(layer, "block_sparse_moe"):
                gate = layer.block_sparse_moe.gate
                self.handles.append(gate.register_forward_hook(functools.partial(self._gate_hook, i)))
            if hasattr(layer.self_attn, "indexer"):
                self.indexers[i] = layer.self_attn.indexer
        self._hook(model.model.norm, "final_hidden")

    def _hook(self, module, key):
        def fn(mod, args, out):
            t = out[0] if isinstance(out, tuple) else out
            self.parts.setdefault(key, []).append(t.detach().to("cpu"))
        self.handles.append(module.register_forward_hook(fn))

    def _gate_hook(self, i, gate, args, out):
        torch, F = self.torch, self.F
        x = args[0]
        B, T, _ = x.shape
        idx, w = out
        ids, order = idx.sort(dim=-1)
        choice = F.linear(x.reshape(B * T, -1).float(), gate.weight.float()).sigmoid() + \
            gate.e_score_correction_bias.float()[None]
        srt = choice.sort(dim=-1, descending=True).values
        k = idx.shape[-1]
        margin = (srt[:, k - 1] - srt[:, k]) if srt.shape[-1] > k else torch.full_like(srt[:, 0], float("inf"))
        for key, t in ((f"moe.{i}.expert_ids", ids.to(torch.int32)), (f"moe.{i}.expert_weights", w.gather(-1, order)),
                       (f"moe.{i}.choice_scores", choice), (f"moe.{i}.cut_margin", margin)):
            self.parts.setdefault(key, []).append(t.detach().to("cpu").reshape(B, T, *t.shape[1:]))

    def reset(self):
        self.parts = {}
        for ix in self.indexers.values():
            ix.trace = []

    def collect(self, logits, total_len: int, store_dtype) -> dict:
        torch = self.torch
        out = {}
        for key, parts in self.parts.items():
            t = torch.cat(parts, dim=1)[0]
            out[key] = t if t.dtype in (torch.int32, torch.int64) else t.to(store_dtype)
        out["logits"] = logits.detach().to("cpu")[0].to(store_dtype)
        for i, ix in self.indexers.items():
            sel_rows, score_rows, margin_rows, k_rows, qr_rows = [], [], [], [], []
            for rec in ix.trace:
                score, keep = rec["score"][0], rec["keep"][0]                 # [T, S]
                T, S = score.shape
                pad = torch.full((T, total_len), float("-inf"), dtype=score.dtype)
                pad[:, :S] = score
                score_rows.append(pad)
                k_rows.append(rec["index_k"][0])
                qr_rows.append(rec["qr"][0])
                for t in range(T):
                    s = keep[t].nonzero().flatten().to(torch.int32)          # ascending key positions
                    row = torch.full((ix.topk,), -1, dtype=torch.int32)
                    row[:s.numel()] = s
                    sel_rows.append(row)
                    vis = score[t][torch.isfinite(score[t])].sort(descending=True).values
                    margin_rows.append(vis[ix.topk - 1] - vis[ix.topk] if vis.numel() > ix.topk
                                       else torch.tensor(float("inf"), dtype=score.dtype))
            out[f"dsa.{i}.selected"] = torch.stack(sel_rows)
            out[f"dsa.{i}.scores"] = torch.cat(score_rows).to(store_dtype)
            out[f"dsa.{i}.cut_margin"] = torch.stack(margin_rows).to(store_dtype)
            out[f"dsa.{i}.index_k"] = torch.cat(k_rows).to(store_dtype)
            out[f"dsa.{i}.qr"] = torch.cat(qr_rows).to(store_dtype)
        return {k: v.contiguous() for k, v in out.items()}


# ----------------------------------------------------------------------------------------------------------
# sequences

def sequences(manifest: dict, vocab: int, topk: int) -> list[dict]:
    """The fixture's own test sequences (manifest test_sequences, written by tools/kimi_k31_synth_gguf.py;
    for the default fixture they are picked so that no top-k cut sits near a tie), else the older default."""
    if manifest.get("test_sequences"):
        return [{k: s[k] for k in ("name", "tokens", "decode_from", "why", "tie_free") if k in s}
                for s in manifest["test_sequences"]]
    import numpy as np
    base = [int(t) for t in manifest["test_tokens"]]
    rng = np.random.default_rng(int(manifest.get("seed", 0)) + 1)
    long = [int(t) for t in rng.integers(0, vocab, size=80)]
    seqs = [
        {"name": "seq0", "tokens": base, "decode_from": 7,
         "why": f"{len(base)} tokens > index_topk {topk}: the indexer selects for positions >= {topk}"},
        {"name": "seq1", "tokens": base[:5], "decode_from": 1,
         "why": f"5 tokens <= index_topk {topk}: every row is dense causal MLA"},
        {"name": "seq2", "tokens": long, "decode_from": 66,
         "why": "80 tokens: crosses fla's 64-token KDA chunk; decode continues past the chunk boundary"},
    ]
    return seqs


# ----------------------------------------------------------------------------------------------------------
# run

def _versions() -> dict:
    import numpy
    import safetensors
    import torch
    import transformers
    v = {"python": platform.python_version(), "torch": torch.__version__, "torch_cuda": torch.version.cuda,
         "transformers": transformers.__version__, "numpy": numpy.__version__,
         "safetensors": safetensors.__version__}
    try:
        import einops
        v["einops"] = einops.__version__
    except ImportError:
        pass
    return v


def _gpu_guard(a) -> None:
    """On CUDA: refuse unless the device has the memory cap plus 1.5 GiB free, cap this process's allocator
    at --gpu-mem-gb, and turn TF32 off for torch matmuls/convs (fp32 means fp32)."""
    import torch
    if a.fla == "real" and not a.device.startswith("cuda"):
        sys.exit("--fla real needs --device cuda (fla's kernels are Triton GPU kernels)")
    if not a.device.startswith("cuda"):
        return
    if not torch.cuda.is_available():
        sys.exit("--device cuda: no CUDA device visible")
    free, total = torch.cuda.mem_get_info()
    need = a.gpu_mem_gb * 2**30
    if free < need + 1.5 * 2**30:
        sys.exit(f"GPU has {free / 2**30:.2f} GiB free; need {a.gpu_mem_gb} GiB + 1.5 GiB headroom")
    torch.cuda.set_per_process_memory_fraction(min(1.0, need / total))
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def cmd_run(a) -> int:
    import torch
    _gpu_guard(a)
    torch.manual_seed(0)
    t0 = time.time()
    fla_info = install_fla(a.fla, a.fla_f32)
    ml, cfgmod, src = load_moonshot(a.modeling_dir)
    fixture = Path(a.fixture).expanduser().resolve()
    cfg = json.loads((fixture / "hf-f32" / "config.json").read_text())
    tc = cfg["text_config"]
    manifest = json.loads((fixture / "manifest.json").read_text())
    k_norm_eps = a.index_k_norm_eps if a.index_k_norm_eps is not None else tc["rms_norm_eps"]
    model, settings = build_model(ml, cfgmod, tc, a.attn, a.mla_norm_eps, k_norm_eps)
    load_info = load_checkpoint(model, fixture, tc)
    dtype = place(model, a.device, a.dtype)
    store_dtype = torch.float64 if dtype == torch.float64 else torch.float32
    rec = Recorder(model, ml)
    out = Path(a.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    from safetensors.torch import save_file
    files = {}
    seqs = sequences(manifest, tc["vocab_size"], tc["dsa_config"]["index_topk"])
    dev = torch.device(a.device)
    for s in seqs:
        ids = torch.tensor([s["tokens"]], dtype=torch.long, device=dev)
        T = ids.shape[1]
        for mode in ("prefill", "decode"):
            rec.reset()
            with torch.no_grad():
                if mode == "prefill":
                    o = model(input_ids=ids, use_cache=True)
                    logits = o.logits
                else:
                    P = s["decode_from"]
                    o = model(input_ids=ids[:, :P], use_cache=True)
                    pkv, chunks = o.past_key_values, [o.logits]
                    for t in range(P, T):
                        o = model(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
                        pkv = o.past_key_values
                        chunks.append(o.logits)
                    logits = torch.cat(chunks, dim=1)
            tensors = rec.collect(logits, T, store_dtype)
            tensors["input_ids"] = ids[0].to("cpu", torch.int32)
            name = f"{s['name']}.{mode}.safetensors"
            save_file(tensors, str(out / name), metadata={"sequence": s["name"], "mode": mode,
                                                          "decode_from": str(s["decode_from"] if mode == "decode" else T)})
            files[name] = _sha256(out / name)
    dev_info = {"device": a.device}
    if a.device.startswith("cuda"):
        dev_info.update(gpu=torch.cuda.get_device_name(dev), capability=list(torch.cuda.get_device_capability(dev)),
                        peak_allocated_mib=round(torch.cuda.max_memory_allocated(dev) / 2**20, 1),
                        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    run = {
        "tool": "tools/kimi_k31_moonshot_oracle.py", "tool_sha256": _sha256(Path(__file__)), "git": _git_rev(),
        "indexer_patch_sha256": _sha256(HERE / "kimi_k31_moonshot_indexer.py"),
        "moonshot": {**src, "modeling_rev": a.modeling_rev},
        "indexer_sources": INDEXER_SOURCES, "fla": fla_info, "versions": _versions(), "device": dev_info,
        "dtype": a.dtype, "settings": settings, "load": load_info,
        "fixture": {"manifest_sha256": _sha256(fixture / "manifest.json"),
                    "hf_f32_model_sha256": _sha256(fixture / "hf-f32" / "model.safetensors"),
                    "hf_f32_config_sha256": _sha256(fixture / "hf-f32" / "config.json"),
                    "manifest_files": manifest.get("files", {})},
        "sequences": seqs, "files": files, "seconds": round(time.time() - t0, 1),
        "tensor_keys": "logits [T,V]; final_hidden [T,D] (model.norm output); layers.L.{attn_in,attn_out,ffn_in,"
                       "ffn_out} [T,D] (input_layernorm out, self_attn out, post_attention_layernorm out, MLP/MoE out);"
                       " moe.L.{expert_ids [T,k] int32 ascending, expert_weights [T,k] aligned with the ids, "
                       "choice_scores [T,E] sigmoid+bias, cut_margin [T] k-th minus (k+1)-th choice score}; "
                       "dsa.L.{selected [T,index_topk] int32 ascending key positions, -1 padded; scores [T,T] "
                       "fp32 with -inf on future keys; cut_margin [T] k-th minus (k+1)-th visible score, inf for "
                       "dense rows; index_k [T,index_head_dim] k_norm output; qr [T,q_lora] the q_a_layernorm output the "
                       "indexer query reads}; input_ids [T] int32. "
                       "*.decode = prefill of decode_from tokens then one token per step through the cache.",
    }
    (out / "run.json").write_text(json.dumps(run, indent=1) + "\n")
    print(f"{out}: {len(files)} files, {run['seconds']} s, fla={fla_info['fla']}, dtype={a.dtype}, "
          f"q_a/kv_a eps={settings['q_a_layernorm_eps']}, attn={a.attn}")
    return 0


# ----------------------------------------------------------------------------------------------------------
# compare

def _load(path: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(str(path))


def _float_stats(a, b) -> dict:
    import torch
    a64, b64 = a.double(), b.double()
    fa, fb = torch.isfinite(a64), torch.isfinite(b64)
    if not torch.equal(fa, fb):
        return {"finite_pattern_differs": int((fa != fb).sum().item())}
    if not fa.any():
        return {"max_abs": 0.0, "ref_max_abs": 0.0, "rel": 0.0}
    d = (a64[fa] - b64[fa]).abs()
    ref = a64[fa].abs().max().item()
    return {"max_abs": d.max().item(), "ref_max_abs": ref, "rel": d.max().item() / ref if ref else 0.0}


def _set_stats(sel_a, sel_b, margin_a, tol) -> dict:
    rows = sel_a.shape[0]
    diff = [t for t in range(rows) if set(sel_a[t].tolist()) != set(sel_b[t].tolist())]
    explained = [t for t in diff if float(margin_a[t]) <= tol]
    return {"rows": rows, "rows_differ": len(diff), "rows_differ_at_tie": len(explained),
            "rows_differ_unexplained": len(diff) - len(explained), "tie_tol": tol,
            "rows_with_exact_tie_in_ref": int((margin_a == 0).sum().item())}


def compare_dirs(da: Path, db: Path, prefill_vs_decode=False) -> dict:
    """Compare same-named files of two runs, or (prefill_vs_decode) each seqN.prefill with seqN.decode of
    one run. The first argument is the reference: relative errors and tie margins are taken from it."""
    import torch
    res = {"ref": str(da), "other": str(db), "files": {}}
    if prefill_vs_decode:
        pairs = [(p.name, p.name.replace(".prefill.", ".decode.")) for p in sorted(da.glob("seq*.prefill.safetensors"))
                 if (db / p.name.replace(".prefill.", ".decode.")).exists()]
    else:
        pairs = [(p.name, p.name) for p in sorted(da.glob("seq*.safetensors")) if (db / p.name).exists()]
    if not pairs:
        raise SystemExit(f"nothing to compare between {da} and {db}")
    worst = {"logits_max_abs": 0.0, "logits_rel": 0.0, "hidden_max_abs": 0.0, "hidden_rel": 0.0,
             "selection_rows_differ": 0, "selection_rows_differ_unexplained": 0,
             "expert_rows_differ": 0, "expert_rows_differ_unexplained": 0, "argmax_mismatch": 0}
    for name, name_b in pairs:
        A, B = _load(da / name), _load(db / name_b)
        f = {}
        for key in sorted(A):
            if key not in B or key == "input_ids":
                continue
            ta, tb = A[key], B[key]
            if key.endswith(".selected") or key.endswith(".expert_ids"):
                continue
            f[key] = _float_stats(ta, tb)
        if not torch.equal(A["input_ids"], B["input_ids"]):
            raise ValueError(f"{name}: different input ids")
        for key in sorted(k for k in A if k.endswith(".selected")):
            L = key.split(".")[1]
            sc = f.get(f"dsa.{L}.scores", {})
            tol = max(1e-6, 10 * sc.get("max_abs", 0.0))
            f[key] = _set_stats(A[key], B[key], A[f"dsa.{L}.cut_margin"], tol)
            worst["selection_rows_differ"] += f[key]["rows_differ"]
            worst["selection_rows_differ_unexplained"] += f[key]["rows_differ_unexplained"]
        for key in sorted(k for k in A if k.endswith(".expert_ids")):
            L = key.split(".")[1]
            sc = f.get(f"moe.{L}.choice_scores", {})
            tol = max(1e-6, 10 * sc.get("max_abs", 0.0))
            f[key] = _set_stats(A[key], B[key], A[f"moe.{L}.cut_margin"], tol)
            worst["expert_rows_differ"] += f[key]["rows_differ"]
            worst["expert_rows_differ_unexplained"] += f[key]["rows_differ_unexplained"]
        am = int((A["logits"].argmax(-1) != B["logits"].argmax(-1)).sum().item())
        f["logits_argmax_mismatch"] = am
        worst["argmax_mismatch"] += am
        lg = f["logits"]
        worst["logits_max_abs"] = max(worst["logits_max_abs"], lg.get("max_abs", float("inf")))
        worst["logits_rel"] = max(worst["logits_rel"], lg.get("rel", float("inf")))
        for key, st in f.items():
            if key.startswith("layers.") or key == "final_hidden":
                worst["hidden_max_abs"] = max(worst["hidden_max_abs"], st.get("max_abs", float("inf")))
                worst["hidden_rel"] = max(worst["hidden_rel"], st.get("rel", float("inf")))
        res["files"][name if name == name_b else f"{name} vs {name_b}"] = f
    res["worst"] = worst
    return res


def per_layer_table(res: dict) -> dict:
    """{file: {layer: attn_out/ffn_out max_abs + rel}, logits} - the compact view for the summary."""
    tab = {}
    for name, f in res["files"].items():
        rows = {}
        for key, st in f.items():
            if key.startswith("layers.") and key.split(".")[2] in ("attn_out", "ffn_out"):
                rows[key] = (st.get("max_abs"), st.get("rel"))
        rows["final_hidden"] = (f["final_hidden"].get("max_abs"), f["final_hidden"].get("rel"))
        rows["logits"] = (f["logits"].get("max_abs"), f["logits"].get("rel"))
        tab[name] = rows
    return tab


def cmd_compare(a) -> int:
    other = a.other or a.ref
    res = compare_dirs(Path(a.ref).expanduser(), Path(other).expanduser(), a.prefill_vs_decode)
    w = res["worst"]
    print(f"{a.label or 'compare'}: logits max_abs {w['logits_max_abs']:.3e} rel {w['logits_rel']:.3e}; "
          f"hidden max_abs {w['hidden_max_abs']:.3e} rel {w['hidden_rel']:.3e}; argmax mismatches "
          f"{w['argmax_mismatch']}; selection rows differ {w['selection_rows_differ']} "
          f"(unexplained by a tie {w['selection_rows_differ_unexplained']}), expert rows differ "
          f"{w['expert_rows_differ']} (unexplained {w['expert_rows_differ_unexplained']})")
    if a.json:
        p = Path(a.json).expanduser()
        doc = json.loads(p.read_text()) if p.exists() else {}
        doc.setdefault("measurements", {})[a.label or f"{Path(a.ref).name}_vs_{Path(a.other).name}"] = {
            "ref": Path(a.ref).name, "other": Path(other).name, "prefill_vs_decode": a.prefill_vs_decode, "worst": w,
            "per_layer_max_abs_rel": per_layer_table(res), "full": res["files"]}
        p.write_text(json.dumps(doc, indent=1) + "\n")
    return 0


def cmd_manifest(a) -> int:
    """Write the run summaries under <golden> into <golden>/golden.json, keeping its measurements."""
    g = Path(a.golden).expanduser()
    p = g / "golden.json"
    doc = json.loads(p.read_text()) if p.exists() else {}
    runs = {}
    for rj in sorted(g.rglob("run.json")):
        r = json.loads(rj.read_text())
        runs[str(rj.parent.relative_to(g))] = {
            "kernels": r["fla"], "device": r["device"], "dtype": r["dtype"], "settings": r["settings"],
            "versions": r["versions"], "files_sha256": r["files"], "seconds": r["seconds"]}
        doc.setdefault("moonshot", r["moonshot"])
        doc.setdefault("indexer_sources", r["indexer_sources"])
        doc.setdefault("fixture", r["fixture"])
        doc.setdefault("sequences", r["sequences"])
        doc.setdefault("tensor_keys", r["tensor_keys"])
        doc.setdefault("load", r["load"])
    doc["runs"] = runs
    if a.primary:
        doc["primary"] = dict(kv.split("=", 1) for kv in a.primary)
    if a.note:
        doc["notes"] = a.note
    p.write_text(json.dumps(doc, indent=1) + "\n")
    print(f"{p}: {len(runs)} runs, {len(doc.get('measurements', {}))} measurements")
    return 0


# ----------------------------------------------------------------------------------------------------------
# selftest

def cmd_selftest(a) -> int:
    import torch
    _gpu_guard(a)
    install_fla(a.fla, "ieee")
    ml, cfgmod, _ = load_moonshot(a.modeling_dir)
    import kimi_k31_moonshot_indexer as kidx
    fixture = Path(a.fixture).expanduser().resolve()
    tc = json.loads((fixture / "hf-f32" / "config.json").read_text())["text_config"]
    manifest = json.loads((fixture / "manifest.json").read_text())
    ids = torch.tensor([manifest["test_tokens"]], dtype=torch.long, device=a.device)
    T = ids.shape[1]
    topk = tc["dsa_config"]["index_topk"]
    keps = tc["rms_norm_eps"]
    fails = []

    def check(name, ok, detail=""):
        print(f"{'ok  ' if ok else 'FAIL'} {name} {detail}")
        if not ok:
            fails.append(name)

    def make(attn="eager", with_indexer=True, topk_override=None):
        m, _ = build_model(ml, cfgmod, tc, attn, None, keps, with_indexer, topk_override)
        load_checkpoint(m, fixture, tc, with_indexer)
        m.to(a.device)
        return m

    def logits(m, x):
        with torch.no_grad():
            return m(input_ids=x, use_cache=True).logits

    dense = make(with_indexer=False)
    l_dense = logits(dense, ids)
    big = make(topk_override=10 ** 6)
    l_big = logits(big, ids)
    check("indexer with index_topk >= T equals Moonshot's dense causal MLA (bitwise)",
          torch.equal(l_big, l_dense), f"max_abs {(l_big - l_dense).abs().max().item():.3e}")
    m = make()
    rec_d, rec = Recorder(dense, ml), Recorder(m, ml)
    rec_d.reset()
    rec.reset()
    logits(dense, ids)
    l_idx = logits(m, ids)
    out_d, out = rec_d.collect(l_dense, T, torch.float32), rec.collect(l_idx, T, torch.float32)
    first = min(rec.indexers)
    key = f"layers.{first}.attn_out"
    check(f"first DSA layer ({first}): rows < index_topk ({topk}) equal dense MLA (bitwise)",
          torch.equal(out[key][:topk], out_d[key][:topk]))
    d_early = (l_idx[:, :topk] - l_dense[:, :topk]).abs().max().item() / l_dense.abs().max().item()
    check(f"logits at positions < index_topk unchanged (rel <= 1e-5; MoE batches differ)", d_early <= 1e-5,
          f"rel {d_early:.3e}")
    d_late = (l_idx[:, topk:] - l_dense[:, topk:]).abs().max().item()
    check("positions >= index_topk changed by the selection", d_late > 1e-4, f"max_abs {d_late:.3e}")
    ok = True
    for i in rec.indexers:
        sel = out[f"dsa.{i}.selected"]
        for t in range(T):
            s = sel[t][sel[t] >= 0]
            ok &= s.numel() == min(topk, t + 1) and bool((s <= t).all()) and len(set(s.tolist())) == s.numel()
    check("every row selects min(index_topk, t+1) distinct visible keys", ok)
    with torch.no_grad():
        o = m(input_ids=ids[:, :7], use_cache=True)
        pkv, chunks = o.past_key_values, [o.logits]
        for t in range(7, T):
            o = m(input_ids=ids[:, t:t + 1], past_key_values=pkv, use_cache=True)
            pkv = o.past_key_values
            chunks.append(o.logits)
    l_dec = torch.cat(chunks, 1)
    d = (l_dec - l_idx).abs().max().item()
    check("prefill(7) + 25 decode steps match one prefill", d <= a.tol, f"max_abs {d:.3e}")
    sd = make(attn="sdpa")
    d = (logits(sd, ids) - l_idx).abs().max().item()
    check("sdpa equals eager", d <= a.tol, f"max_abs {d:.3e}")
    for patch, msg in (({"index_topk_freq": 2}, "index_topk_freq"), ({"index_rope_dim": 64}, "index_rope_dim"),
                       ({"index_k_norm_type": "rmsnorm"}, "index_k_norm_type"), ({"dsa_layers": [4, 8]}, "dsa_layers")):
        bad = copy.deepcopy(tc)
        bad["dsa_config"].update(patch)
        try:
            kidx.dsa_layers_0idx(bad)
            check(f"refuses {msg}", False)
        except ValueError:
            check(f"refuses {msg}", True)
    print("selftest:", "FAILED " + ", ".join(fails) if fails else "all passed")
    return 1 if fails else 0


# ----------------------------------------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--modeling-dir", required=True,
                       help="directory with Moonshot's modeling_kimi_linear.py and configuration_kimi_k3.py")
        p.add_argument("--fixture", required=True, help="fixture directory (hf-f32/ and manifest.json)")
        p.add_argument("--device", default="cpu", help="cpu or cuda[:N]")
        p.add_argument("--fla", choices=("real", "standin"), default=None,
                       help="KDA kernels: real fla (CUDA only) or the pure-torch stand-in "
                            "(default: real on cuda, standin on cpu)")
        p.add_argument("--gpu-mem-gb", type=float, default=1.5, help="per-process CUDA memory cap")

    r = sub.add_parser("run", help="write golden outputs for one configuration")
    common(r)
    r.add_argument("--out", required=True)
    r.add_argument("--dtype", choices=("float32", "bfloat16", "float64"), default="float32")
    r.add_argument("--attn", choices=("eager", "sdpa"), default="eager")
    r.add_argument("--mla-norm-eps", type=float, default=None,
                   help="q_a/kv_a RMSNorm eps; default: Moonshot's class default (1e-6)")
    r.add_argument("--index-k-norm-eps", type=float, default=None, help="default: rms_norm_eps")
    r.add_argument("--modeling-rev", default="unknown", help="revision of the modeling files, for run.json")
    r.add_argument("--fla-f32", choices=("tf32", "ieee"), default="ieee",
                   help="real fla only: 'ieee' = full fp32 dots (default), 'tf32' = fla as published "
                        "(TF32 dots in the chunked prefill on sm_80+)")
    c = sub.add_parser("compare", help="compare two run directories")
    c.add_argument("ref")
    c.add_argument("other", nargs="?", help="default: ref itself (with --prefill-vs-decode)")
    c.add_argument("--prefill-vs-decode", action="store_true",
                   help="compare each seqN.prefill of ref with seqN.decode of other")
    c.add_argument("--json", help="merge the result into this JSON file under measurements.<label>")
    c.add_argument("--label")
    mf = sub.add_parser("manifest", help="collect the run.json files under a golden directory into golden.json")
    mf.add_argument("golden")
    mf.add_argument("--primary", nargs="*", help="role=run-dir pairs, e.g. a=fla-cuda-fp32 b=standin-cpu-fp32")
    mf.add_argument("--note", nargs="*", help="free-text notes stored in golden.json")
    s = sub.add_parser("selftest", help="structural checks of the patched model")
    common(s)
    s.add_argument("--tol", type=float, default=1e-4)
    a = ap.parse_args(argv)
    if getattr(a, "fla", None) is None and hasattr(a, "device"):
        a.fla = "real" if a.device.startswith("cuda") else "standin"
    sys.path.insert(0, str(HERE))
    return {"run": cmd_run, "compare": cmd_compare, "manifest": cmd_manifest, "selftest": cmd_selftest}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

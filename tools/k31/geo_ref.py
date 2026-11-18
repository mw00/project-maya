"""tools/k31/geo_ref.py - the oracle's side of maya-k31's geometry check (tests/k31_geo_test.cpp --ref).

Prints, one line a field, what ref/kimi_k31.py's KimiK31Config reads from a GGUF's header, in the names and the form
src/k31/k31_geo.hpp's k31::dump prints them: integers as written, floats as float32 with 9 significant digits, lists
space-separated.  The expert ids of a pruned file come from its kimi-k31.expert_prune.keep key or, for a file without
it, from "<file>.prune.json" read with Python's json; with neither, the file's own 0..E-1.

    python tools/k31/geo_ref.py FILE.gguf [--eps moonshot|llama.cpp] > FILE.ref
    k31_geo_test --gguf FILE.gguf --ref FILE.ref [--eps ...]

--write-tiny first writes tools/kimi_k31_synth_gguf.py's tiny fixture to FILE.gguf (CI has no fixture files).
Only the header is read (tools/gguf_reader.py), never tensor data.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))               # tools/: gguf_reader
sys.path.insert(0, str(HERE.parents[1] / "ref"))   # ref/: kimi_k31

from gguf_reader import GGUFFile  # noqa: E402
import kimi_k31 as K  # noqa: E402


def f32(x: float) -> str:
    return "%.9g" % float(np.float32(x))


def keep_ids(path: pathlib.Path, meta: dict, c: K.KimiK31Config):
    moe = range(c.n_layer_dense_lead, c.n_layers)
    keep = meta.get(f"{K.ARCH}.expert_prune.keep")
    if keep is not None:
        flat = [int(x) for x in keep]
        return "file", int(meta[f"{K.ARCH}.expert_prune.source_count"]), {
            il: flat[j * c.n_expert:(j + 1) * c.n_expert] for j, il in enumerate(moe)}
    side = pathlib.Path(str(path) + ".prune.json")
    if side.is_file():
        d = json.loads(side.read_text())
        return "sidecar", int(d["source_expert_count"]), {il: [int(x) for x in d["layers"][str(il)]] for il in moe}
    return "none", c.n_expert, {il: list(range(c.n_expert)) for il in moe}


def lines(path: pathlib.Path, eps: str) -> list[str]:
    meta = GGUFFile(path).metadata
    # the oracle's own reading only, nothing of the engine's: keys KimiK31Config does not read yet
    # (kda.head_count, attention.lora_norm_epsilon, kda.l2_norm_epsilon, a per-layer expert width) show up as lines
    # that differ from the engine's, or as the oracle's refusal, until ref/kimi_k31.py reads them
    c = K.KimiK31Config(meta, eps=eps)
    out = []

    def put(k, v):
        if isinstance(v, bool):
            v = "true" if v else "false"
        elif isinstance(v, float):
            v = f32(v)
        elif isinstance(v, list):
            v = " ".join(str(int(x)) for x in v)
        out.append(f"{k} {v}")

    for k in ("n_vocab", "n_embd", "n_layers", "n_head", "n_head_kv"):
        put(k, getattr(c, k))
    put("mla_layers", c.mla_layers())
    for k in ("q_lora_rank", "kv_lora_rank", "n_rot", "n_head_k_mla", "n_head_v_mla", "qk_nope", "ssm_d_conv",
              "kda_head_dim"):
        put(k, getattr(c, k))
    put("kda_gate_lower_bound", float(c.kda_gate_lower_bound))
    put("d_inner", c.d_inner)
    for k in ("n_ff", "n_layer_dense_lead", "n_expert", "n_expert_used"):
        put(k, getattr(c, k))
    put("n_ff_exp", [c.n_ff_exp] * (c.n_layers - c.n_layer_dense_lead))   # the oracle has one width for every layer
    put("n_expert_shared", c.n_expert_shared)
    put("n_expert_latent", c.n_expert_latent)
    put("expert_weights_scale", float(c.expert_weights_scale))
    put("expert_weights_norm", bool(c.expert_weights_norm))
    put("situ_beta", float(c.situ_beta))
    put("situ_linear_beta", float(c.situ_linear_beta))
    for k in ("attn_res_block_size", "idx_n_head", "idx_key_dim", "idx_top_k"):
        put(k, getattr(c, k))
    put("eps_preset", c.eps_preset)
    put("eps_rms", float(c.eps_rms))
    put("eps_indexer_ln", float(c.eps_indexer_ln))
    put("eps_mla_latent", float(c.eps_mla_latent))
    put("eps_kda_l2", float(c.eps_kda_l2))
    put("eps_router_renorm", float(c.eps_router_renorm))
    src, n_source, keep = keep_ids(path, meta, c)
    put("keep_from", src)
    put("expert_source_count", n_source)
    for il in range(c.n_layer_dense_lead, c.n_layers):
        put(f"keep.{il}", keep[il])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="print KimiK31Config's geometry of a GGUF in k31::dump's form")
    ap.add_argument("gguf", type=pathlib.Path)
    ap.add_argument("--eps", default="moonshot", choices=sorted(K.EPS_PRESETS))
    ap.add_argument("--write-tiny", action="store_true", help="write the tiny synthetic fixture to GGUF first")
    a = ap.parse_args()
    if a.write_tiny:
        import kimi_k31_synth_gguf as synth
        vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
        md, ts = synth.to_gguf(vals, synth.TINY)
        synth.write_gguf(a.gguf, md, ts)
    sys.stdout.write("\n".join(lines(a.gguf, a.eps)) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""tools/strata/glm_dsa_pack.py - the glm-dsa port: the glm-dsa (GLM-5.3 744B-A40B) load-only pack (BUILD-PLAN B1.6).

Called by tools/iq_pack.py when general.architecture is "glm-dsa"; it writes the same files as the glm5-next pack
(index.txt, dense.bin, native_experts.txt, tokenizer/) so the engine's load_pack reads it unchanged:

  * an existing community GGUF is served as it is: no requantization.  The routed experts stay native in the GGUF
    shards (native_experts.txt names each layer's types and absolute offsets), and so do the non-routed tensors
    whose type the engine's device GEMV (`mv`, glm_fast.cu mv_type_ok) reads natively;
  * a non-routed tensor whose type `mv` lacks (F16, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL) is DEQUANTIZED to BF16 in
    dense.bin (index kind 4, round-to-nearest-even).  F16 is never written as dense kind 5: the loader takes only
    kinds 0, 2 and 4.  Each conversion is listed in compat-bf16.json with its bytes;
  * per-architecture lists decide the rest: the absorbed MLA projections and the indexer projections are BF16
    (GLMDSA_BF16), the router, its selection bias and every norm are F32 (GLMDSA_F32; the router is NOT rounded
    to BF16 as Maya's glm5-next list does, G23), and only GLMDSA_NATIVE_OK may be served quantized.  Anything
    else fails the pack closed;
  * blk.<trunk> is the NextN/MTP block (GGUF block_count = trunk + nextn_predict_layers): its dense rows are
    indexed (with the same upcasts) and its experts get a native_experts.txt row, marked as MTP in both headers.
    The engine skips rows at layer >= n_layers until its glm-dsa MTP loader (B1.8) reads them;
  * indexer tensors on "shared" layers (attention.indexer.types false; GLM-5.2-style GGUFs wrote them on all 78
    layers) are tolerated: a warning, and only the full layers' indexers are packed;
  * a metadata-only first shard (unsloth's split layout, Maya #83) is handled: the tensors are found in whichever
    shard holds them, and native_experts.txt names each layer's shard.
"""
from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
TOOLS = HERE.parent
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
import gguf_reader as G  # noqa: E402
import iq_pack  # noqa: E402

ARCH = "glm-dsa"
PREFIX = "glm-dsa."

# ggml type ids the engine's device GEMV reads natively (glm_fast.cu mv_type_ok): F32, BF16, Q8_0, Q2_K..Q6_K,
# IQ2_XXS, IQ2_XS, IQ3_XXS, IQ1_S, IQ3_S, IQ2_S, IQ4_XS, IQ1_M
MV_TYPES = {"F32", "BF16", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS",
            "IQ1_S", "IQ3_S", "IQ2_S", "IQ4_XS", "IQ1_M"}
# the non-routed types the pack upcasts to BF16 because `mv` lacks them (B10.7 adds them to `mv` and drops this)
UPCAST_TYPES = ("F16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "IQ4_NL")
# the routed-expert types the GPU MoE kernels instantiate (glm_fast.cu moe_supported); others only warn here (the
# engine's native_fmt decides at load)
MOE_TYPES = {"Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ2_S", "IQ3_XXS", "IQ3_S", "IQ1_S",
             "IQ1_M", "IQ4_XS", "Q8_0"}   # Q8_0: Maya v1.1.55 MoE kernels + MMQ serve it

# read as BF16 by the kernels: the absorbed MLA projections (3-D, ne0 = 192 forbids K/IQ types, so community files
# store them Q8_0/Q4_0/Q5_0/IQ4_NL/F16) and the lightning indexer's projections
GLMDSA_BF16 = (
    "attn_k_b.weight", "attn_v_b.weight",
    "indexer.attn_k.weight", "indexer.attn_q_b.weight", "indexer.proj.weight",
)
# read as F32: the router (F32 by config, kept F32: G23), its selection bias, every norm, the NextN norms
GLMDSA_F32 = (
    "ffn_gate_inp.weight", "exp_probs_b.bias",
    "attn_norm.weight", "ffn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
    "indexer.k_norm.weight", "indexer.k_norm.bias",
    "nextn.enorm.weight", "nextn.hnorm.weight", "nextn.shared_head_norm.weight",
)
GLMDSA_F32_GLOBAL = ("output_norm.weight",)
# the ONLY non-routed tensors served quantized (when `mv` has their type); F32/BF16 copies go to dense.bin as
# stored, the upcast types to BF16
GLMDSA_NATIVE_OK = (
    "attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_output.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    "nextn.eh_proj.weight",
    # optional in the converter and expected absent (they fall back to token_embd / output)
    "nextn.embed_tokens.weight", "nextn.shared_head_head.weight",
)
GLMDSA_NATIVE_OK_GLOBAL = ("token_embd.weight", "output.weight")
INDEXER = ("indexer.attn_k.weight", "indexer.attn_q_b.weight", "indexer.proj.weight",
           "indexer.k_norm.weight", "indexer.k_norm.bias")
ROLES = ("gate", "up", "down")

# llama.cpp glm-dsa.cpp GLM_5_2_DEFAULT_INDEXER_TYPES: full layers 0, 1, 2, 6, 10, ..., 74
GLM_5_2_FULL = {0, 1} | set(range(2, 78, 4))


def _gguf():
    from strata_gguf import import_gguf
    import_gguf()
    from gguf import GGMLQuantizationType as Q, quants
    return Q, quants


def tensor_class(name: str) -> str:
    """'bf16', 'f32', 'native', 'expert', or 'unknown' for a tensor name of a glm-dsa GGUF."""
    if name.startswith("blk."):
        if name.endswith("_exps.weight"):
            return "expert"
        rest = name.split(".", 2)[2] if name.count(".") >= 2 else ""
        if rest in GLMDSA_BF16:
            return "bf16"
        if rest in GLMDSA_F32 or rest.endswith("_norm.weight"):
            return "f32"
        if rest in GLMDSA_NATIVE_OK:
            return "native"
        return "unknown"
    if name in GLMDSA_F32_GLOBAL:
        return "f32"
    if name in GLMDSA_NATIVE_OK_GLOBAL:
        return "native"
    return "unknown"


# B10.7 (glm-dsa): the engine's mv() reads the 32-value block types (glmd_mv_ext.cuh) and the headwise MLA GEMVs read
# native Q8_0 W_kb / W_vb (glmd::headwise).  Q8_0 attn_k_b / attn_v_b stay native by default: the GGUF's own weights,
# where the BF16 copy rounded every one of them (and ~1 GB less to read a token); STRATA_PACK_NATIVE32=0 upcasts them
# as before.  STRATA_PACK_NATIVE32=1 also keeps the UPCAST_TYPES of the mv tensors native (opt-in, not validated as a
# default).  The binder serves either through glmdsa::pick_headwise.
HEADWISE = ("attn_k_b.weight", "attn_v_b.weight")
HEADWISE_NATIVE_TYPES = ("Q8_0",)


def native32_enabled() -> bool:
    """The mv tensors' 32-value block types native: STRATA_PACK_NATIVE32=1 only."""
    return os.environ.get("STRATA_PACK_NATIVE32", "") not in ("", "0")


def headwise_native_enabled() -> bool:
    """Q8_0 W_kb / W_vb native: the default; STRATA_PACK_NATIVE32=0 upcasts them to BF16."""
    return os.environ.get("STRATA_PACK_NATIVE32", "").strip() != "0"


def plan_tensor(name: str, type_name: str, native32: bool | None = None,
                headwise: bool | None = None) -> tuple[str, str]:
    """(what, target) for a non-routed tensor: what = 'native' (served from the GGUF), 'copy' (dense.bin as
    stored), 'convert' (dequantized into dense.bin) or 'refuse'; target = the dense type (BF16/F32) or a reason.
    native32 (default: STRATA_PACK_NATIVE32=1) keeps the mv tensors' B10.7 types native instead of upcasting them;
    headwise (default: on unless STRATA_PACK_NATIVE32=0; an explicit native32 decides it too) keeps Q8_0 W_kb / W_vb
    native."""
    if headwise is None:
        headwise = native32 if native32 is not None else headwise_native_enabled()
    if native32 is None:
        native32 = native32_enabled()
    cls = tensor_class(name)
    float_kind = {"BF16": "BF16", "F32": "F32"}
    if cls == "bf16" and headwise and name.endswith(HEADWISE) and type_name in HEADWISE_NATIVE_TYPES:
        return "native", type_name
    if cls == "native" and native32 and type_name in UPCAST_TYPES and type_name not in float_kind:
        return "native", type_name
    if cls == "bf16":
        if type_name == "BF16":
            return "copy", "BF16"
        if type_name in MV_TYPES or type_name in UPCAST_TYPES:
            return "convert", "BF16"
        return "refuse", "a %s tensor the BF16 path cannot dequantize" % type_name
    if cls == "f32":
        if type_name == "F32":
            return "copy", "F32"
        if type_name in MV_TYPES or type_name in UPCAST_TYPES:
            return "convert", "F32"
        return "refuse", "a %s tensor the F32 path cannot dequantize" % type_name
    if cls == "native":
        if type_name in float_kind:
            return "copy", float_kind[type_name]
        if type_name in MV_TYPES:
            return "native", type_name
        if type_name in UPCAST_TYPES:
            return "convert", "BF16"
        return "refuse", "%s is not served by the engine's mv() and is not an upcast type (%s)" % (
            type_name, ", ".join(UPCAST_TYPES))
    return "refuse", "not a glm-dsa tensor the pack knows (GLMDSA_BF16 / GLMDSA_F32 / GLMDSA_NATIVE_OK)"


def convert_bytes(raw: np.ndarray, type_name: str, target: str) -> bytes:
    """Dequantize `raw` (the GGUF bytes) to BF16 (round to nearest even) or F32 - a dequant, never a requant."""
    Q, quants = _gguf()
    if target == "BF16" and type_name == "BF16":
        return raw.tobytes()
    values = np.asarray(quants.dequantize(raw, Q[type_name]), dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("cannot convert non-finite weights to %s" % target)
    if target == "BF16":
        # ggml's fp32 -> bf16 (round to nearest even), as iq_pack.bf16_bytes; gguf-py is found as _gguf() finds it
        return quants.quantize(values, Q.BF16).tobytes()
    return values.astype("<f4").tobytes()


def indexer_full(md: dict, trunk: int) -> tuple[list[bool], str]:
    """The trunk layers' indexer kinds (True = full) and where they came from: attention.indexer.types when present
    (llama.cpp get_key_or_arr), else llama.cpp's BC rule: n_ctx_train >= 1048576 means the GLM-5.2 pattern."""
    types = md.get(PREFIX + "attention.indexer.types")
    if types is not None:
        if isinstance(types, (list, tuple)):
            if len(types) < trunk:
                raise ValueError("%sattention.indexer.types has %d entries for %d trunk layers"
                                 % (PREFIX, len(types), trunk))
            return [bool(x) for x in types[:trunk]], "attention.indexer.types"
        return [bool(types)] * trunk, "attention.indexer.types (scalar)"
    ctx = int(md.get(PREFIX + "context_length", 0) or 0)
    if ctx >= 1048576:
        return [il in GLM_5_2_FULL for il in range(trunk)], "GLM-5.2 default (no indexer.types key, context %d)" % ctx
    return [True] * trunk, "all full (no indexer.types key, context %d < 1048576)" % ctx


def _layer_of(name: str) -> int | None:
    if not name.startswith("blk."):
        return None
    f = name.split(".")[1]
    return int(f) if f.isdigit() else None


def pack(a, src: pathlib.Path, out: pathlib.Path) -> int:
    """iq_pack.py's main for general.architecture == glm-dsa (argparse namespace `a`, shard 1 `src`)."""
    if a.base:
        print("glm-dsa: --base is not supported (a glm-dsa pack is standalone)")
        return 1
    g = G.GGUFFile(src)
    md = g.metadata
    bc = md.get(PREFIX + "block_count")
    if bc is None:
        print("glm-dsa: %sblock_count is missing; cannot tell the trunk from the NextN block" % PREFIX)
        return 1
    block_count = int(bc)
    nextn = int(md.get(PREFIX + "nextn_predict_layers", 0) or 0)
    trunk = block_count - nextn
    if trunk < 1 or nextn not in (0, 1):
        print("glm-dsa: block_count %d with nextn_predict_layers %d: one NextN block at most, and a trunk"
              % (block_count, nextn))
        return 1
    if PREFIX + "leading_dense_block_count" not in md or PREFIX + "expert_count" not in md:
        print("glm-dsa: %sleading_dense_block_count and %sexpert_count are required" % (PREFIX, PREFIX))
        return 1
    lead = int(md[PREFIX + "leading_dense_block_count"])
    n_expert = int(md[PREFIX + "expert_count"])
    mtp = trunk if nextn else None
    print("glm-dsa: block_count %d, nextn %d -> trunk layers 0..%d, dense 0..%d, MoE %d..%d%s"
          % (block_count, nextn, trunk - 1, lead - 1, lead, trunk - 1,
             ", MTP blk.%d" % mtp if mtp is not None else ""))
    if mtp is None:
        print("warning: glm-dsa: no NextN/MTP block (nextn_predict_layers 0): no speculative decode "
              "(STRATA_GLM_MTP_GGUF can supply another file's draft block)")
    try:
        full, how = indexer_full(md, trunk)
    except ValueError as e:
        print("glm-dsa: %s" % e)
        return 1
    print("glm-dsa: indexer full layers (%s): %s" % (how, ",".join(str(i) for i, f in enumerate(full) if f)))

    model = iq_pack.Model(src)
    if len(model.paths) > 1:
        print("model shards: " + ", ".join(p.name for p in model.paths))
    if not g.tensors:
        print("glm-dsa: %s holds metadata only; the tensors are in the other shards" % src.name)
    T = {n: w[1] for n, w in model.where.items()}

    # ---- the dense tensors
    rows, converted, refused = [], [], []
    dropped_idx, served, at = [], 0, 0
    with open(out / "dense.bin", "wb") as fo:
        for name, (gf, t, mm, _) in model.where.items():
            il = _layer_of(name)
            if name.startswith("blk.") and (il is None or il >= block_count):
                refused.append("%s: layer outside block_count %d" % (name, block_count))
                continue
            if iq_pack.is_expert(name):
                continue
            if il is not None and il < trunk and name.split(".", 2)[2] in INDEXER and not full[il]:
                dropped_idx.append(name)
                continue
            dims = len(t.shape) - (len(t.shape) == 4 and int(t.shape[3]) == 1)
            if dims > 3:
                refused.append("%s: %d dimensions" % (name, len(t.shape)))
                continue
            what, target = plan_tensor(name, t.type_name)
            if what == "refuse":
                refused.append("%s (%s): %s" % (name, t.type_name, target))
                continue
            ne0, ne1 = iq_pack.index_shape(t.shape)
            if what == "native":
                served += 1
                rows.append([name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
                continue
            src_raw = iq_pack.tensor_bytes(mm, gf, t)
            if what == "copy":
                raw = src_raw.tobytes()
            else:
                try:
                    raw = convert_bytes(np.asarray(src_raw), t.type_name, target)
                except (ValueError, NotImplementedError) as e:
                    refused.append("%s (%s): %s" % (name, t.type_name, e))
                    continue
                converted.append({"name": name, "source_type": t.type_name, "target": target,
                                  "source_bytes": int(src_raw.size), "bytes": len(raw),
                                  "bytes_added": len(raw) - int(src_raw.size),
                                  "mtp": il is not None and il == mtp})
            kind = {"BF16": "4", "F32": "2"}[target]
            rows.append([name, "0", kind, str(at), str(len(raw)), "0", str(len(raw)), str(ne0), str(ne1),
                         "0", "0", "1"] + ["0"] * 7)
            fo.write(raw)
            pad = (-len(raw)) % iq_pack.ALIGN
            fo.write(b"\0" * pad)
            at += len(raw) + pad
    if refused:
        for r in refused[:20]:
            print("glm-dsa: refusing %s" % r)
        if len(refused) > 20:
            print("glm-dsa: ... and %d more" % (len(refused) - 20))
        print("glm-dsa: %d tensor(s) cannot be packed; nothing is served silently" % len(refused))
        return 1
    if dropped_idx:
        layers = sorted({_layer_of(n) for n in dropped_idx})
        print("warning: glm-dsa: indexer tensors on %d shared layer(s) (%s): not packed, only the full layers' "
              "indexers are used" % (len(layers), ",".join(str(l) for l in layers)))
    missing_idx = [il for il in range(trunk) if full[il]
                   and ("blk.%d.indexer.attn_k.weight" % il) not in T]
    if missing_idx:
        print("glm-dsa: full indexer layer(s) %s have no indexer tensors" % ",".join(map(str, missing_idx)))
        return 1
    iq_pack.write_index(out, rows, src, served, 0)
    _add_header(out / "index.txt", "# glm-dsa: trunk %d, dense lead %d, experts %d, mtp %s, indexer full %s\n"
                % (trunk, lead, n_expert, mtp if mtp is not None else "none",
                   ",".join(str(i) for i, f in enumerate(full) if f)))
    added = sum(c["bytes_added"] for c in converted)
    (out / "compat-bf16.json").write_text(json.dumps({
        "source": str(src), "arch": ARCH, "rounding": "nearest-even",
        "note": "dequantized (not requantized) non-routed tensors; routed experts are unchanged",
        "tensors": converted,
        "bytes": sum(c["bytes"] for c in converted), "bytes_added": added,
        "dropped_shared_indexer": dropped_idx,
    }, indent=2) + "\n", encoding="utf-8")
    by_type: dict[str, list] = {}
    for c in converted:
        by_type.setdefault("%s->%s" % (c["source_type"], c["target"]), []).append(c)
    for k, v in sorted(by_type.items()):
        print("compat-bf16: %-12s %4d tensors, %8.1f MiB written, %+8.1f MiB vs the GGUF"
              % (k, len(v), sum(c["bytes"] for c in v) / 2**20, sum(c["bytes_added"] for c in v) / 2**20))
    print("compat-bf16: %d tensors converted, %+.2f GiB; routed experts unchanged" % (len(converted), added / 2**30))

    if not (out / "tokenizer" / "vocab.json").exists() or not (out / "tokenizer" / "chat_template.jinja").exists():
        subprocess.run([sys.executable, str(TOOLS / "strata_tokenizer.py"), "--gguf", str(src), "--out", str(out)],
                       check=True)

    # ---- the routed experts: the MoE trunk layers, then the MTP block
    moe_layers = list(range(lead, trunk))
    if mtp is not None:
        if all(("blk.%d.ffn_%s_exps.weight" % (mtp, r)) in T for r in ROLES):
            moe_layers.append(mtp)
        else:
            print("glm-dsa: the NextN block blk.%d has no routed experts" % mtp)
            return 1
    if not moe_layers:
        print("glm-dsa: no MoE layers; the pack has nothing to serve")
        return 1
    layout, offset = [], 0
    off_types = set()
    for l in moe_layers:
        names = ["blk.%d.ffn_%s_exps.weight" % (l, r) for r in ROLES]
        if any(n not in T for n in names):
            print("glm-dsa: layer %d: routed experts missing (%s)" % (l, ", ".join(n for n in names if n not in T)))
            return 1
        router = T.get("blk.%d.ffn_gate_inp.weight" % l)
        if router is None or int(router.shape[1] if len(router.shape) > 1 else 0) != n_expert:
            print("glm-dsa: layer %d: the router is missing or does not have %d rows" % (l, n_expert))
            return 1
        ts = [T[n] for n in names]
        if any(len(t.shape) != 3 or int(t.shape[2]) != n_expert for t in ts):
            print("glm-dsa: layer %d: the expert tensors are not [.., .., %d]" % (l, n_expert))
            return 1
        if ts[0].type_name != ts[1].type_name or ts[0].shape != ts[1].shape:
            print("layer %d: gate and up differ in type (%s vs %s)" % (l, ts[0].type_name, ts[1].type_name))
            return 1
        if [int(ts[2].shape[0]), int(ts[2].shape[1])] != [int(ts[0].shape[1]), int(ts[0].shape[0])]:
            print("glm-dsa: layer %d: down %s does not match gate %s" % (l, ts[2].shape, ts[0].shape))
            return 1
        for t in (ts[0], ts[2]):
            if t.type_name not in MOE_TYPES:
                off_types.add(t.type_name)
        per = [t.expected_bytes() // n_expert for t in ts]
        blob = sum(per)
        layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, ts))
        offset += blob * n_expert
    if off_types:
        print("warning: glm-dsa: routed expert type(s) %s are outside the GPU MoE kernels (moe_supported)"
              % ", ".join(sorted(off_types)))
    straddle = any(len({model.where[t.name][3] for t in ts}) != 1 for *_, ts in layout)
    version = "v4" if straddle else "v3"
    with open(out / "native_experts.txt", "w", encoding="utf-8", newline="\n") as fo:
        # line 1 must stay the one maya.py's pack_source and the engine's pack_shard_names parse
        fo.write("# strata native experts %s: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                 "[shard | gate_shard up_shard down_shard] "
                 "(n_expert %d, total %d; absolute offsets in %s, or in the named shard(s) beside it)\n"
                 % (version, n_expert, offset, src.name))
        fo.write("# glm-dsa: MoE layers %d..%d%s\n" % (lead, trunk - 1,
                                                       "; layer %d is the NextN/MTP block" % mtp
                                                       if mtp is not None else "; no MTP block"))
        for l, gt, dt, off, blob, ts in layout:
            ws = [model.where[t.name] for t in ts]
            shards = [w[3] for w in ws]
            offs = [w[0].data_start + t.offset for w, t in zip(ws, ts)]
            if len(set(shards)) == 1:
                line = "%d %d %d %d %d %d %d %d" % (l, gt, dt, off, blob, *offs)
                fo.write(line + ("" if shards[0] == src else " " + shards[0].name) + "\n")
            else:
                fo.write("%d %d %d %d %d %d %d %d %s\n"
                         % (l, gt, dt, off, blob, *offs, " ".join(s.name for s in shards)))
    print("native_experts.txt: %d MoE layers%s, %d experts each, %.2f GiB routed"
          % (len(layout), " (incl. MTP)" if mtp is not None else "", n_expert, offset / 2**30))
    if a.skip_experts or not a.experts_bin:
        return 0
    path = out / "experts.bin"
    if path.exists() and path.stat().st_size == offset:
        print("experts.bin exists with the right size; not rewritten")
        return 0
    with open(path, "wb") as fo:
        for l, gt, dt, off, blob, ts in layout:
            parts = [model.bytes(t.name).reshape(n_expert, -1) for t in ts]
            fo.write(np.concatenate(parts, axis=1).tobytes())
    print("experts.bin: %d layers, %.2f GiB" % (len(layout), offset / 2**30))
    return 0


def _add_header(path: pathlib.Path, line: str) -> None:
    """Insert one more '#' line after index.txt's two header lines (every reader skips '#' lines)."""
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    n = 0
    while n < len(lines) and lines[n].startswith("#"):
        n += 1
    lines.insert(n, line)
    path.write_text("".join(lines), encoding="utf-8", newline="\n")

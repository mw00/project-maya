"""tools/iq_pack.py - plan v0.3 P6: a native pack for any of the model files (Q2_0, IQ2_XS, IQ3_XXS).

    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --out pack/iq3_xxs            (standalone)
    python tools/iq_pack.py --gguf <model>-00001-of-00002.gguf --base pack/full --out ...    (share dense.bin)

The i-quant experts cannot be re-expressed in the Q2_0 pack form, so this pack keeps every quantized tensor in
its GGUF form:

  experts.bin          per layer, 512 blobs of [gate rows | up rows | down rows], the raw GGUF slices.  Blob
                       size is per layer (the files mix IQ1_M ... IQ3_S gate/up and Q2_0 / IQ4_NL down).
  native_experts.txt   one line per layer: layer gu_type d_type offset blob_bytes
  index.txt            the table the engine loads.  Quantized dense tensors, token_embd and output are served
                       natively from the GGUF by the engine (--native): their rows carry shape only.
  dense.bin            standalone: the BF16/F16/F32 tensors exactly as the GGUF stores them (index kinds 4/5/2).
                       With --base: the base (Q2_0) pack's dense.bin, hard-linked - the float tensors are
                       byte-identical in all three model files (checked) - plus extra.bin for tensors that are
                       float here but quantized in the base pack (blk.1.ple_key).
  tokenizer/           exported from the GGUF (tools/strata_tokenizer.py), with the model's chat template.

Split files: every shard of the model is read (<name>-0000N-of-0000M.gguf beside --gguf), so the layers may be
split anyhow (Swift 1.5's GGUFs put layers 13-47 in shard 2 and the PLE table in shard 1).  A layer whose experts
are not in shard 1 names its shard in native_experts.txt (v3).  Router tensors stored as F32 whose values are
exactly BF16 (Swift 1.5) are written as BF16, the form the engine's router takes; anything else is refused.

For ordinary quants, --compat-bf16 dequantizes the small projections that the engine reads as BF16, using
round-to-nearest-even. This introduces BF16 rounding; it does not reconstruct the original full-precision
weights. Experts, native attention projections, token embeddings and the disk-backed PLE table stay unchanged.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_reader as G  # noqa: E402

FLOAT = {"BF16", "F32", "F16"}
ROUTERS = ("ffn_gate_inp.weight", "ffn_gate_inp_shexp.weight")
NOT_IN_PACK = {"per_layer_token_embd.weight"}      # the 28.8 GB PLE table: read from its GGUF by the engine

# These small projections are read as BF16 by the residual, router, GDN, QSA and PLE kernels.
# Strata's Qwen files already store them that way. Ordinary GGUF quants (including OrcaRouter's IQ3_XXS)
# quantize them too; --compat-bf16 explicitly dequantizes and rounds ONLY these tensors.
BF16_PROJECTIONS = (
    "hc_attn_down.weight", "hc_attn_up.weight", "hc_attn_inject.weight",
    "hc_ffn_down.weight", "hc_ffn_up.weight", "hc_ffn_inject.weight",
    "ssm_alpha.weight", "ssm_beta.weight", "indexer.q_proj.weight", "indexer.k_proj.weight",
    "ple_value.weight", *ROUTERS,
)
BF16_OUTPUT = {"output_hc_down.weight", "output_hc_up.weight"}

# The architectures the packer knows.  The BF16-preserve lists are PER ARCHITECTURE because each
# engine reads different projections as high-precision: qwen4exp's hyper-connections use the
# down/up/inject bottleneck names, glm5-next's mHC uses fn/base/scale, and their indexers do not
# share a single tensor name.
GLM5NEXT_BF16 = (
    # the mHC mixes (the released config keeps these unquantized; the runner's glm_hc kernels
    # consume them per token)
    "hc_attn_fn.weight", "hc_attn_base.weight", "hc_attn_scale.weight",
    "hc_ffn_fn.weight", "hc_ffn_base.weight", "hc_ffn_scale.weight",
    # the indexer - at 1M context the selection IS the model's accuracy
    "indexer.attn_q_b.weight", "indexer.attn_k.weight", "indexer_compressor_gate.weight",
    "indexer_compressor_ape.weight", "indexer.proj.weight", "indexer.k_norm.weight",
    "indexer.k_norm.bias",
    # the router and its noaux_tc selection bias (F32 by config: moe_router_dtype float32)
    "ffn_gate_inp.weight", "exp_probs_b.bias",
    # the KDA gates and norms (gates stay unquantized in the released model); the conv weights
    # seed the whole KDA recurrence, so their quantization noise would accumulate per token
    "ssm_a", "ssm_dt.bias", "ssm_norm.weight", "ssm_beta.weight",
    "ssm_f_a.weight", "ssm_f_b.weight", "ssm_g_a.weight", "ssm_g_b.weight",
    "ssm_conv1d_q.weight", "ssm_conv1d_k.weight", "ssm_conv1d_v.weight",
    # the MLA norms and the 3-D absorbed projections that form the attention scores and values
    "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
    "attn_k_b.weight", "attn_v_b.weight",
)
# The ONLY non-expert tensors a glm5-next pack may serve NATIVELY (quantized, read from the
# GGUF by the Phase-4 driver's native reader): the attention projections and the dense/shared
# FFN weights, which are exactly what dynamic recipes quantize on a real file.  Everything
# else quantized fails the pack CLOSED - the packer errors instead of emitting a row the
# engine would reject at load, far from where the mistake was made.
GLM5NEXT_NATIVE_OK = (
    "attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_output.weight",
    # the KDA layers' separate projections (Q5_K in the UD recipe)
    "attn_q.weight", "attn_k.weight", "attn_v.weight",
    "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
    "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight",
    # the UD-IQ1_S recipe stores even the embedding and the head at Q4_K; the pack loader
    # dequantizes them into the F32 arena at load (2x ~1.2 GB quantized)
    "token_embd.weight", "output.weight",
)
ARCHITECTURES = ("qwen4exp", "glm5-next", "glm5next")   # unsloth spells it without the hyphen


def needs_bf16(name: str, type_name: str, arch: str = "qwen4exp") -> bool:
    if arch in ("glm5-next", "glm5next"):
        # glm5-next has no PLE and no output_hc; its F32 router stays F32 (the runner reads it
        # as F32 by config), so ONLY the listed projections take the BF16 path.
        return name.startswith("blk.") and name.endswith(GLM5NEXT_BF16)
    if name in BF16_OUTPUT:
        return True
    if not name.startswith("blk."):
        return False
    # The existing native PLE key supports Q2_0 only. Other key encodings use the BF16 path.
    return name.endswith(BF16_PROJECTIONS) or (name == "blk.1.ple_key.weight" and type_name != "Q2_0")


def index_shape(shape) -> tuple[int, int]:
    """The (ne0, ne1) the flat index row records for a tensor of GGUF shape `shape`.

    The engine reads packed tensors by BYTE OFFSET and the kernels by their own geometry, so a
    3-D tensor (glm5-next's absorbed MLA projections and conv weights) packs as the 2-D view
    (ne0*ne1, ne2): GGUF lays element (i0, i1, i2) at i0 + ne0*i1 + ne0*ne1*i2, which is exactly
    row (i0 + ne0*i1) of the flattened view - byte-identical, no repacking."""
    if len(shape) >= 3:
        return int(shape[0]) * int(shape[1]), int(shape[2])
    ne0 = int(shape[0])
    return ne0, (int(shape[1]) if len(shape) > 1 else 0)


def bf16_bytes(raw: np.ndarray, type_name: str) -> bytes:
    from _paths import add_gguf_py
    add_gguf_py()
    from gguf import GGMLQuantizationType as Q, quants
    values = quants.dequantize(raw, Q[type_name])
    if not np.isfinite(values).all():
        raise ValueError("cannot convert non-finite weights to BF16")
    # ggml's round-to-nearest-even conversion, including correct halfway rounding.
    return quants.quantize(values, Q.BF16).tobytes()


class Model:
    """All shards of one model: name -> (GGUFFile, TensorInfo, memmap, shard path)."""

    def __init__(self, first: pathlib.Path):
        import re
        m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", first.name)
        paths = [first]
        if m:
            total = int(m.group(2))
            paths = [first.with_name(first.name[:m.start()] + "-%05d-of-%05d.gguf" % (i, total))
                     for i in range(1, total + 1)]
        missing = [str(p) for p in paths if not p.is_file()]
        if missing:
            raise FileNotFoundError("missing model shards (wait for the download): " + ", ".join(missing))
        self.paths = paths
        self.where = {}
        for p in paths:
            g = G.GGUFFile(p)
            mm = np.memmap(p, dtype=np.uint8, mode="r")
            for t in g.tensors:
                size = t.expected_bytes()
                if size is None or g.data_start + t.offset + size > mm.size:
                    raise ValueError(f"{p.name}: unsupported or truncated tensor {t.name}")
                if t.name in self.where:
                    raise ValueError(f"{p.name}: duplicate tensor {t.name}")
                self.where[t.name] = (g, t, mm, p)

    def bytes(self, name) -> np.ndarray:
        g, t, mm, _ = self.where[name]
        return tensor_bytes(mm, g, t)
ROLES = ("gate", "up", "down")
N_EXPERT = 512
ALIGN = 64


def read_index(path: pathlib.Path):
    rows, header = {}, []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            header.append(line)
            continue
        f = line.split()
        rows[f[0]] = f
    return header, rows


def tensor_bytes(mm, g, t) -> np.ndarray:
    n = t.expected_bytes()
    return mm[g.data_start + t.offset: g.data_start + t.offset + n]


def is_expert(name: str) -> bool:
    return name.startswith("blk.") and name.endswith(("_exps.weight",))


def index_standalone(src, out, model: Model, compat_bf16: bool = False, arch: str = "qwen4exp",
                     block_count: int | None = None) -> int:
    """Every non-expert tensor of the model: floats into dense.bin as stored (exact-BF16 F32 routers as BF16),
    quantized ones native-only.

    Fail-closed for glm5-next: a non-expert tensor that is neither float nor on the BF16 upgrade
    list may only pass if GLM5NEXT_NATIVE_OK names it (the Phase-4 native reader's set); anything
    else is a hard error.  The NextN/MTP block (blk.{block_count}.*, the runner skips MTP) is
    dropped from the pack entirely."""
    dropped = 0
    if not compat_bf16:
        for name, (_, t, _, _) in model.where.items():
            if needs_bf16(name, t.type_name, arch) and t.type_name != "BF16" and not (
                    t.type_name == "F32" and name.endswith(ROUTERS) and arch == "qwen4exp"):
                print(f"{name} is {t.type_name}, but the engine requires BF16; use --compat-bf16")
                return 1
    rows, at = [], 0
    served = 0
    converted = []
    with open(out / "dense.bin", "wb") as fo:
        for name, (g, t, mm, _) in model.where.items():
            if is_expert(t.name) or t.name in NOT_IN_PACK:
                continue
            if block_count is not None and name.startswith("blk."):
                il = name.split(".")[1]
                if il.isdigit() and int(il) >= block_count:
                    dropped += 1
                    continue
            # a trailing ne3 of 1 adds no bytes: some converters write glm5-next's conv weights as
            # [4, 1, 8192, 1] instead of [4, 1, 8192], and index_shape packs both as (4, 8192)
            dims = len(t.shape) - (len(t.shape) == 4 and int(t.shape[3]) == 1)
            if dims > 2 and not (arch in ("glm5-next", "glm5next") and dims == 3):
                print("tensor %s has %d dimensions; the index holds two" % (t.name, len(t.shape)))
                return 1
            ne0, ne1 = index_shape(t.shape)
            convert = compat_bf16 and needs_bf16(t.name, t.type_name, arch) and t.type_name != "BF16"
            if t.type_name in FLOAT or convert:
                raw = tensor_bytes(mm, g, t).tobytes()
                if convert:
                    raw = bf16_bytes(np.frombuffer(raw, dtype=np.uint8), t.type_name)
                    kind = "4"
                    converted.append({"name": t.name, "source_type": t.type_name, "bytes": len(raw)})
                else:
                    kind = {"BF16": "4", "F16": "5", "F32": "2"}[t.type_name]
                if not convert and arch == "qwen4exp" and t.type_name == "F32" and t.name.endswith(ROUTERS):
                    u = np.frombuffer(raw, dtype=np.uint32)
                    if np.count_nonzero(u & 0xFFFF):
                        print("router %s is F32 with values that are not BF16; the engine's router is BF16" % t.name)
                        return 1
                    raw = (u >> 16).astype(np.uint16).tobytes()     # the exact BF16 values
                    kind = "4"
                rows.append([t.name, "0", kind, str(at), str(len(raw)), "0", str(len(raw)), str(ne0), str(ne1),
                             "0", "0", "1"] + ["0"] * 7)
                fo.write(raw)
                pad = (-len(raw)) % ALIGN
                fo.write(b"\0" * pad)
                at += len(raw) + pad
            elif arch in ("glm5-next", "glm5next") and not name.endswith(GLM5NEXT_NATIVE_OK):
                print("%s is %s: not on the BF16 upgrade list nor the glm5-next native set "
                      "(GLM5NEXT_NATIVE_OK); refusing to pack it silently" % (t.name, t.type_name))
                return 1
            else:
                served += 1
                rows.append([t.name, "0", "0", "0", "0", "0", "0", str(ne0), str(ne1), "8", "0", "32"] + ["0"] * 7)
    if dropped:
        # (#327: "the runner skips MTP" read as drafting off - it is not: the engine reads the block from the model's
        #  files itself)
        print("NextN/MTP: %d tensors of blk.%d+ left out of the pack - %s" % (
            dropped, block_count, "the MTP drafter is built from them by tools/mtp_from_gguf.py" if arch == "qwen4exp"
            else "the engine reads the MTP draft block from the model's files itself"))
    write_index(out, rows, src, served, 0)
    if compat_bf16:
        (out / "compat-bf16.json").write_text(json.dumps({
            "source": str(src), "rounding": "nearest-even", "tensors": converted,
        }, indent=2) + "\n", encoding="utf-8")
        print("compat-bf16: %d tensors, %.2f GiB; expert and PLE table bytes unchanged"
              % (len(converted), sum(t["bytes"] for t in converted) / 2**30))
    return 0


def write_index(out, rows, src, served, n_extra):
    at = 0
    for r in rows:
        r[5] = str(at)
        at += (int(r[6]) + ALIGN - 1) // ALIGN * ALIGN
    with open(out / "index.txt", "w", encoding="utf-8", newline="\n") as fo:
        fo.write("# strata pack index v3 -- generated by tools/iq_pack.py (native experts) from %s\n" % src.name)
        fo.write("# align %d pool %d tensors %d\n" % (ALIGN, at, len(rows)))
        for r in rows:
            fo.write(" ".join(r) + "\n")
    print("index.txt: %d tensors, %d served natively, %d in extra.bin, arena %.2f GiB"
          % (len(rows), served, n_extra, at / 2**30))


def index_from_base(a, src, base, out, g, T, mm) -> int:
    base_src = pathlib.Path(json.loads((base / "manifest.json").read_text(encoding="utf-8"))["source"]["shard1"])
    if not base_src.exists():
        print("cannot find the base pack's shard 1 from its manifest.json")
        return 1
    bg = G.GGUFFile(base_src)
    BT = {t.name: t for t in bg.tensors}
    bmm = np.memmap(base_src, dtype=np.uint8, mode="r")
    header, rows = read_index(base / "index.txt")
    new_rows, extra = [], []
    served = 0
    for name, f in rows.items():
        t, bt = T.get(name), BT.get(name)
        if t is None or bt is None:
            print("tensor %s missing from one of the models" % name)
            return 1
        if t.type_name in FLOAT and bt.type_name in FLOAT:
            if t.type_name != bt.type_name or t.shape != bt.shape or \
                    not np.array_equal(tensor_bytes(mm, g, t), tensor_bytes(bmm, bg, bt)):
                print("float tensor %s differs from the base model; this pack cannot reuse its dense.bin" % name)
                return 1
            new_rows.append(list(f))
        elif t.type_name in FLOAT:
            if t.type_name != "BF16":
                print("unexpected float type %s for %s" % (t.type_name, name))
                return 1
            nbytes = t.expected_bytes()
            off = sum(len(b) + (-len(b)) % ALIGN for b in extra)
            extra.append(tensor_bytes(mm, g, t).tobytes())
            # file 3 = extra.bin, raw BF16 (index kind 4)
            new_rows.append([name, "3", "4", str(off), str(nbytes), "0", str(nbytes), f[7], f[8]] + ["0"] * 10)
        else:
            served += 1
            new_rows.append([name, f[1], "0", "0", "0", "0", "0", f[7], f[8], "8", "0", "32"] + ["0"] * 7)
    write_index(out, new_rows, src, served, len(extra))
    with open(out / "extra.bin", "wb") as fo:
        for b in extra:
            fo.write(b)
            fo.write(b"\0" * ((-len(b)) % ALIGN))
    dense = out / "dense.bin"
    if not dense.exists():
        try:
            os.link(base / "dense.bin", dense)
        except OSError:
            shutil.copyfile(base / "dense.bin", dense)
    if (base / "tokenizer").exists() and not (out / "tokenizer").exists():
        shutil.copytree(base / "tokenizer", out / "tokenizer")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gguf", required=True, help="the model's shard 1")
    ap.add_argument("--base", help="optional: a Q2_0 canonical pack whose dense.bin holds the shared float tensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-experts", action="store_true", help="rewrite the index only")
    ap.add_argument("--compat-bf16", action="store_true",
                    help="dequantize small non-native projections to BF16 for ordinary Qwen4Exp GGUFs "
                         "(rounds weights; leaves experts and the PLE table unchanged)")
    ap.add_argument("--experts-bin", action="store_true",
                    help="also write experts.bin (the engine otherwise reads the experts from the GGUF itself)")
    a = ap.parse_args()
    if a.compat_bf16 and a.base:
        ap.error("--compat-bf16 cannot reuse --base dense weights")
    # HF snapshot files are symlinks to hash-named blobs. Keep the shard filename for discovery.
    src = pathlib.Path(a.gguf).absolute()
    base = pathlib.Path(a.base).resolve() if a.base else None
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    g = G.GGUFFile(src)
    mm = np.memmap(src, dtype=np.uint8, mode="r")
    arch = str(g.metadata.get("general.architecture", ""))
    if arch not in ARCHITECTURES:
        print("general.architecture is %r; the packer knows %s" % (arch, " and ".join(ARCHITECTURES)))
        return 1
    if arch in ("glm5-next", "glm5next"):
        # two spellings in the wild: llama.cpp's canonical "glm5-next" (llama-arch.cpp L156) and
        # unsloth's "glm5next" - the KV PREFIX follows the same spelling, so it is read per file
        prefix = "glm5-next." if arch == "glm5-next" else "glm5next."
        print("%s: packing for the Glm5Model runner (quantized dense served natively; the driver "
              "gate remains closed until the pack wiring lands - docs/GLM5-FLASH.md §9)" % arch)
    block_count = None
    trunk = None
    # (a Qwen GGUF with its MTP block - qwen4exp, nextn_predict_layers 1, block_count 49 - the same: the trunk only)
    if arch not in ("glm5-next", "glm5next"):
        prefix = arch + "."
    if arch in ("glm5-next", "glm5next") or int(g.metadata.get(prefix + "nextn_predict_layers", 0) or 0) > 0:
        bc = g.metadata.get(prefix + "block_count")
        if bc is None:
            print("%s: %sblock_count is missing; cannot tell the trunk from the NextN block" % (arch, prefix))
            return 1
        block_count = int(bc)
        # the NextN/MTP layer counts INTO block_count in the wild (unsloth ships 46 with
        # nextn_predict_layers 1): the servable trunk is what is left after subtracting it
        nextn = int(g.metadata.get(prefix + "nextn_predict_layers", 0) or 0)
        trunk = block_count - nextn
        if trunk < 1:
            print("%s: block_count %d - nextn %d leaves no trunk" % (arch, block_count, nextn))
            return 1
        print("%s: block_count %d, nextn %d -> trunk layers 0..%d" % (arch, block_count, nextn, trunk - 1))
    model = Model(src)
    T = {n: w[1] for n, w in model.where.items()}
    if len(model.paths) > 1:
        print("model shards: " + ", ".join(p.name for p in model.paths))
    if a.base:
        if any(w[3] != src for w in model.where.values() if not w[1].name in NOT_IN_PACK):
            print("--base needs a model whose tensors are all in shard 1")
            return 1
        rc = index_from_base(a, src, base, out, g, {t.name: t for t in g.tensors}, mm)
    else:
        rc = index_standalone(src, out, model, a.compat_bf16, arch, trunk)
    if rc:
        return rc
    if not (out / "tokenizer" / "vocab.json").exists() or not (out / "tokenizer" / "chat_template.jinja").exists():
        subprocess.run([sys.executable, str(HERE / "strata_tokenizer.py"), "--gguf", str(src), "--out", str(out)],
                       check=True)   # writes <out>/tokenizer/

    # ---- the experts
    if trunk is not None:
        # the trunk only: the NextN/MTP layer ships its own experts (a full extra layer) and the
        # runner skips MTP - deriving n_layers from the max expert index would pack the NextN
        # layer as a servable layer the driver must never see
        n_layers = trunk
        trunk_experts = [n for n in T if n.endswith("_exps.weight")
                         and n.startswith("blk.") and 0 <= int(n.split(".")[1]) < trunk]
        if not trunk_experts:
            print("%s: no routed experts in the trunk; the pack has nothing to serve" % arch)
            return 1
        dropped_experts = sum(1 for n in T if n.startswith("blk.%d." % trunk) and n.endswith("_exps.weight"))
        if dropped_experts:
            print("%s: %d NextN expert tensors present (blk.%d.*); the trunk keeps layers 0..%d"
                  % (arch, dropped_experts, trunk, trunk - 1))
    else:
        n_layers = 1 + max(int(n.split(".")[1]) for n in T if n.startswith("blk.") and n.endswith("_exps.weight"))
    # the MoE layers only: the leading dense blocks ship no router and no experts
    if arch in ("glm5-next", "glm5next"):
        lead = int(g.metadata.get(prefix + "leading_dense_block_count", 0) or 0)
    else:
        lead = int(g.metadata.get("qwen4exp.leading_dense_block_count", 0) or 0)
    moe_layers = [l for l in range(n_layers) if l >= lead]
    first_moe = moe_layers[0]
    n_expert = int(T["blk.%d.ffn_gate_inp.weight" % first_moe].shape[1])  # router rows = experts kept
    if any(int(T["blk.%d.ffn_gate_inp.weight" % l].shape[1]) != n_expert for l in moe_layers):
        print("the routers disagree on the expert count; a per-layer pruned model cannot be packed")
        return 1
    layout, offset = [], 0
    for l in moe_layers:
        ts = [T["blk.%d.ffn_%s_exps.weight" % (l, r)] for r in ROLES]
        per = [t.expected_bytes() // n_expert for t in ts]
        if per[0] != per[1] or ts[0].type_name != ts[1].type_name:
            print("layer %d: gate and up differ in type" % l)
            return 1
        blob = per[0] + per[1] + per[2]
        layout.append((l, ts[0].type_id, ts[2].type_id, offset, blob, ts))
        offset += blob * n_expert
    with open(out / "native_experts.txt", "w", encoding="utf-8", newline="\n") as fo:
        straddle = any(len({w[3] for w in (model.where[t.name] for t in ts)}) != 1
                       for _, _, _, _, _, ts in layout)
        version = "v4" if straddle else "v3"
        fo.write("# strata native experts %s: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                 "[shard | gate_shard up_shard down_shard] "
                 "(n_expert %d, total %d; absolute offsets in %s, or in the named shard(s) beside it)\n"
                 % (version, n_expert, offset, src.name))
        for l, gt, dt, off, blob, ts in layout:
            ws = [model.where[t.name] for t in ts]
            shards = [w[3] for w in ws]
            # each role's offset is absolute in ITS OWN named shard: the data_start of the
            # file that holds the tensor + the tensor's data-section offset.  (Using the
            # gate role's file for every role put layer 25's down 7616 bytes low in the
            # UD-IQ1_S pack: gate/up sat in one shard, down in another.)
            offs = [w[0].data_start + t.offset for w, t in zip(ws, ts)]
            if len(set(shards)) == 1:
                # v3 row: one shard for the whole blob (omitted when it is shard 1)
                line = "%d %d %d %d %d %d %d %d" % (l, gt, dt, off, blob, *offs)
                fo.write(line + ("" if shards[0] == src else " " + shards[0].name) + "\n")
            else:
                # v4 row: the layer straddles shards (a shard boundary inside a layer); each
                # role names its own file - shard 1 by name too: the engine splits the row on
                # whitespace, so an empty name would vanish and shift the others.  The engine
                # reads every role from its own shard (gate_shard / up_shard / down_shard) and
                # checks each offset against that shard's directory at load, so gate and up may
                # sit in different files too (a split by size can cut between them).
                names = [s.name for s in shards]
                fo.write("%d %d %d %d %d %d %d %d %s\n"
                         % (l, gt, dt, off, blob, *offs, " ".join(names)))
    if a.skip_experts or not a.experts_bin:
        if (out / "experts.bin").exists() and not a.experts_bin:
            print("note: %s/experts.bin exists; the engine reads it instead of the GGUF" % out)
        return 0
    path = out / "experts.bin"
    if path.exists() and path.stat().st_size == offset:
        print("experts.bin exists with the right size; not rewritten")
        return 0
    with open(path, "wb") as fo:
        for l, gt, dt, off, blob, ts in layout:
            parts = [model.bytes(t.name).reshape(n_expert, -1) for t in ts]
            chunk = np.concatenate(parts, axis=1)          # (n_expert, blob): gate | up | down per expert
            assert chunk.shape == (n_expert, blob)
            fo.write(chunk.tobytes())
            if l % 8 == 0:
                print("  layer %2d  %-8s/%-7s blob %8d  at %.2f GiB" % (l, ts[0].type_name, ts[2].type_name, blob,
                                                                        off / 2**30), flush=True)
    print("experts.bin: %d layers, %.2f GiB" % (n_layers, offset / 2**30))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

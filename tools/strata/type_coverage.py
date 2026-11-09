#!/usr/bin/env python3
"""tools/strata/type_coverage.py - the expert-type coverage table of a glm-dsa GGUF (BUILD-PLAN B5.3).

Reads a GGUF's tensor table (no tensor data) and the engine's own support tables, and writes, per tensor role and
ggml type, which engine path serves it:

  routed experts (blk.N.ffn_{gate,up,down}_exps)
    decode GPU   the MoE kernels' instantiations (glm_fast.cu moe_supported)
    CPU lane     ggml-cpu's vec_dot for the type (the native lane computes with ggml's dot products; native_fmt)
    prefill      the MMQ instances (strata_mmq: CMakeLists.txt mmq-instance-<t>.cu)
  non-routed tensors
    pack         tools/strata/glm_dsa_pack.py plan_tensor: served natively, copied, dequantized to BF16/F32, or
                 refused (with STRATA_PACK_NATIVE32 or --native32: the B10.7 types stay native)
    decode GPU   the device GEMV (glm_fast.cu mv_type_ok, glmd_mv_dots.cuh type_ok) on what the pack serves
    prefill      the FP16 dequant of the batched prompt (glm_prefill_dsa.cu f16_covers: dequant_bf16.cu geometry
                 plus the i-quants) or the float copy
  every tensor
    oracle       gguf-py can dequantize the type (ref/glm_dsa.py, the B7.5 slice oracle)

A routed type is "covered" when decode GPU, CPU lane and prefill all serve it; a non-routed type is "covered" when
the pack serves it natively or as a float copy, "upcast" when the pack dequantizes it (more VRAM, slower), and
"UNCOVERED" when the pack refuses it or an engine path has no kernel for it.  The table also counts the expert size
classes (distinct (gate/up row bytes, down row bytes) pairs; the slow path's pool classes, B3.7, audit check A13).

Inputs (one): GGUF shards (any shard: the siblings are found), a header dump (tools/strata/gguf_audit.py
--dump-header), or an audit TSV (<name>_audit.tsv).  The engine tables are parsed from this source tree; ggml's
vec_dot table from --ggml-dir (default $STRATA_GGML_DIR, else third_party/ggml).

    python3 tools/strata/type_coverage.py /data/models/GLM-5.3-GGUF/UD-IQ2_M/GLM-5.3-UD-IQ2_M-00001-of-00006.gguf \
        --out docs/strata/volta3/type-coverage.md
    python3 tools/strata/type_coverage.py ~/strata-facts/glm53_UD-IQ2_M_audit.tsv

The runtime check of the same types is native_expert_parity (float vs ggml-cpu vs GPU, max rel err < 1e-3) on real
shard layers 3, 26, 54, 77, 78 (the realpack ctests native_expert_parity_real_*); the generated file lists the
command.  Exit status: 0 everything covered, 1 an UNCOVERED type, 2 usage or read error.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import re
import sys
from collections import defaultdict
from dataclasses import dataclass

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
import gguf_audit as ga  # noqa: E402
import glm_dsa_pack as gp  # noqa: E402

ROUTED = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
BLK = re.compile(r"^blk\.(\d+)\.(.+)$")
# ggml-cpu without a source tree: the types with a vec_dot at the pins (aa94f208 / 3cf03257)
VEC_DOT_FALLBACK = {"F32", "F16", "BF16", "Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "Q2_K", "Q3_K", "Q4_K", "Q5_K",
                    "Q6_K", "IQ2_XXS", "IQ2_XS", "IQ3_XXS", "IQ3_S", "IQ2_S", "IQ1_S", "IQ1_M", "IQ4_NL", "IQ4_XS",
                    "TQ1_0", "TQ2_0", "MXFP4"}


@dataclass
class T:
    name: str
    type: int
    dims: list[int]

    @property
    def tname(self) -> str:
        return ga.type_name(self.type)

    def row_bytes(self) -> int:
        return ga.type_bytes(self.type, self.dims[0]) if self.dims else 0


# ------------------------------------------------------------------------------------------------ engine tables
def _read(path: pathlib.Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _body(text: str, signature: str) -> str:
    i = text.find(signature)
    if i < 0:
        return ""
    j = text.find("{", i)
    depth, k = 0, j
    while k < len(text):
        depth += text[k] == "{"
        depth -= text[k] == "}"
        if depth == 0:
            return text[j:k + 1]
        k += 1
    return ""


def engine_tables(root: pathlib.Path, ggml_dir: pathlib.Path | None) -> dict:
    """{"moe", "mv", "mv_ext", "f16", "mmq", "vec_dot": sets of type ids; "sources": what was read}."""
    fast = _read(root / "src/kernels/cuda/glm_fast.cu")
    src: dict = {}
    moe = {int(x) for x in re.findall(r"type == (\d+)", _body(fast, "bool moe_supported(int type)"))}
    mvb = _body(fast, "int mv_type_ok(int t)")
    mv = {int(x) for x in re.findall(r"case (\d+)", mvb)}
    if "kTypeF32" in mvb:
        mv.add(0)
    if "kTypeBF16" in mvb:
        mv.add(30)
    dots = _read(root / "src/kernels/cuda/glmd_mv_dots.cuh")
    consts = {k: int(v) for k, v in re.findall(r"\bk(\w+)\s*=\s*(\d+)", dots)}
    ext = {consts[n] for n in re.findall(r"type == k(\w+)", _body(dots, "bool type_ok(int type)")) if n in consts}
    deq = _read(root / "src/kernels/cuda/dequant_bf16.cu")
    f16 = {int(x) for x in re.findall(r"case (\d+):", _body(deq, "bool geometry(int type, int& block_elems"))}
    pf = _read(root / "src/core/glm_prefill_dsa.cu")
    f16 |= {int(x) for x in re.findall(r"type == (\d+)", _body(pf, "bool f16_covers(int type"))}
    pt = _read(root / "src/core/glm_prefill_types.hpp")   # glm-dsa's extra glm-dsa FP16 types (W7 build review)
    m = re.search(r"kExtF16Types\[\]\s*=\s*\{([^}]*)\}", pt)
    if m:
        f16 |= {int(x) for x in re.findall(r"\d+", m.group(1))}
    f16 |= {30, 0}
    src["glm_fast.cu"] = bool(moe and mv)
    src["glmd_mv_dots.cuh"] = bool(ext)
    src["f16_covers"] = len(f16) > 2
    mmq = ga.mmq_types(root)
    vec = set()
    gd = ggml_dir
    cands = [gd] if gd else []
    if os.environ.get("STRATA_GGML_DIR"):
        cands.append(pathlib.Path(os.environ["STRATA_GGML_DIR"]))
    cands.append(root / "third_party" / "ggml")
    try:   # the llama.cpp tree whose gguf-py is on the path (strata-env.sh: ~/ggml-pin/gguf-py)
        import gguf
        cands.append(pathlib.Path(gguf.__file__).resolve().parents[2])
    except ImportError:
        pass
    for c in cands:
        if c is None:
            continue
        for p in (c / "ggml/src/ggml-cpu/ggml-cpu.c", c / "src/ggml-cpu/ggml-cpu.c"):
            text = _read(p)
            if text:
                for m in re.finditer(r"\[GGML_TYPE_(\w+)\]\s*=\s*\{(.*?)\n\s*\},", text, re.S):
                    if ".vec_dot " in m.group(2) or ".vec_dot=" in m.group(2).replace(" ", ""):
                        if m.group(1).upper() in ga.TYPE_ID:
                            vec.add(ga.TYPE_ID[m.group(1).upper()])
                src["vec_dot"] = str(p)
                break
        if vec:
            break
    if not vec:
        vec = {ga.TYPE_ID[n] for n in VEC_DOT_FALLBACK if n in ga.TYPE_ID}
        src["vec_dot"] = "built-in list (no ggml source found)"
    return {"moe": moe, "mv": mv, "mv_ext": ext, "f16": f16, "mmq": mmq, "vec_dot": vec, "sources": src}


def oracle_types() -> set[str] | None:
    """Type names gguf-py can dequantize, or None without gguf-py."""
    try:
        from gguf import quants, GGMLQuantizationType as Q
    except ImportError:
        return None
    traits = getattr(quants, "_type_traits", None)
    out = {"F32", "F16", "BF16"}
    if isinstance(traits, dict):
        for q in traits:
            out.add(Q(q).name if not isinstance(q, str) else q)
    return out


# ------------------------------------------------------------------------------------------------ inputs
def load_tensors(paths: list[str]) -> tuple[list[T], str]:
    if len(paths) == 1 and paths[0].endswith(".tsv"):
        rows = pathlib.Path(paths[0]).read_text(encoding="utf-8").splitlines()
        head = rows[0].split("\t")
        ix = {h: i for i, h in enumerate(head)}
        out = []
        for ln in rows[1:]:
            c = ln.split("\t")
            if len(c) < len(head):
                continue
            out.append(T(c[ix["name"]], ga.TYPE_ID[c[ix["type"]]] if c[ix["type"]] in ga.TYPE_ID else -1,
                         [int(x) for x in c[ix["dims"]].split(",") if x]))
        return out, f"audit TSV {paths[0]}"
    if len(paths) == 1 and paths[0].endswith(".json"):
        shards = ga.load_dump(pathlib.Path(paths[0]))
        src = f"header dump {paths[0]}"
    else:
        shards = ga.load_local(paths)
        src = "GGUF " + ", ".join(s.name for s in shards)
    return [T(t.name, t.type, list(t.dims)) for s in shards for t in s.tensors], src


# ------------------------------------------------------------------------------------------------ table
def yn(b: bool | None) -> str:
    return "?" if b is None else ("yes" if b else "**no**")


def coverage(tensors: list[T], tab: dict, oracle: set[str] | None, native32: bool) -> dict:
    routed: dict = defaultdict(lambda: {"n": 0, "bytes": 0, "layers": set()})
    dense: dict = defaultdict(lambda: {"n": 0, "bytes": 0})
    classes: dict = defaultdict(lambda: {"gu": set(), "d": None})
    for t in tensors:
        m = ROUTED.match(t.name)
        if m:
            il, role = int(m.group(1)), m.group(2)
            key = ("down" if role == "down" else "gate/up", t.type)
            routed[key]["n"] += 1
            routed[key]["bytes"] += ga.type_bytes(t.type, ga.math.prod(t.dims) if t.dims else 1)
            routed[key]["layers"].add(il)
            if role == "down":
                classes[il]["d"] = t.row_bytes()
            else:
                classes[il]["gu"].add(t.row_bytes())
            continue
        b = BLK.match(t.name)
        rest = b.group(2) if b else t.name
        dense[(rest, t.type)]["n"] += 1
        dense[(rest, t.type)]["bytes"] += ga.type_bytes(t.type, ga.math.prod(t.dims) if t.dims else 1)
    rows_r, rows_d, uncovered, prefill_gaps = [], [], [], []
    for (role, ty), v in sorted(routed.items(), key=lambda kv: (kv[0][0], ga.type_name(kv[0][1]))):
        tn = ga.type_name(ty)
        gpu, lane, mmq = ty in tab["moe"], ty in tab["vec_dot"], ty in tab["mmq"]
        orc = None if oracle is None else tn in oracle
        ok = gpu and lane and mmq
        verdict = "covered" if ok else "UNCOVERED"
        if not ok:
            uncovered.append(f"routed {role} {tn}")
        ls = sorted(v["layers"])
        rows_r.append({"role": role, "type": tn, "tensors": v["n"], "gb": v["bytes"] / 1e9, "decode_gpu": gpu,
                       "cpu_lane": lane, "prefill_mmq": mmq, "oracle": orc, "verdict": verdict,
                       "layers": ", ".join(map(str, ls[:8])) + (f" ... ({len(ls)})" if len(ls) > 8 else "")})
    for (rest, ty), v in sorted(dense.items(), key=lambda kv: (kv[0][0], ga.type_name(kv[0][1]))):
        tn = ga.type_name(ty)
        name = rest if "." in rest else rest
        full = name if not BLK.match(name) else name
        probe = ("blk.0." + rest) if rest not in ("token_embd.weight", "output.weight", "output_norm.weight") else rest
        what, target = gp.plan_tensor(probe, tn, native32)
        served = ty if what == "native" else ga.TYPE_ID.get(target, -1) if what in ("copy", "convert") else -1
        if probe == "token_embd.weight":
            gpu = True                       # rows gathered on the host (any type gguf-py / the loader dequantizes)
        elif what == "refuse":
            gpu = False
        elif gp.tensor_class(probe) in ("bf16", "f32"):
            gpu = True                       # BF16/F32 float paths
        else:
            gpu = served in tab["mv"] or (native32 and served in tab["mv_ext"])
        pre = served in tab["f16"] or served in (0, 30)
        orc = None if oracle is None else tn in oracle
        if what == "refuse" or not gpu:
            verdict = "UNCOVERED"
            uncovered.append(f"{full} {tn}")
        elif what == "convert":
            verdict = f"upcast to {target}"
        else:
            verdict = "covered"
        if verdict != "UNCOVERED" and not pre and probe != "token_embd.weight":
            verdict += "; batched prefill refuses it (token path)"
            prefill_gaps.append(f"{full} {tn}")
        rows_d.append({"tensor": full, "type": tn, "tensors": v["n"], "gb": v["bytes"] / 1e9,
                       "pack": f"{what} {target}" if what != "refuse" else f"refuse: {target}", "decode_gpu": gpu,
                       "prefill": pre, "oracle": orc, "verdict": verdict})
    pairs = defaultdict(list)
    for il, c in classes.items():
        if c["d"] is not None and c["gu"]:
            pairs[(max(c["gu"]), c["d"])].append(il)
    return {"routed": rows_r, "dense": rows_d, "uncovered": uncovered, "prefill_gaps": prefill_gaps,
            "classes": dict(pairs),
            "routed_layers": sorted(classes)}


def markdown(cov: dict, tab: dict, source: str, native32: bool) -> str:
    def nm(ids):
        return ", ".join(sorted(ga.type_name(i) for i in ids))
    n_cls = len(cov["classes"])
    out = ["# Expert-type coverage (B5.3)", "",
           f"Generated by `tools/strata/type_coverage.py` from {source}; engine tables parsed from this tree "
           f"(ggml vec_dot: {tab['sources'].get('vec_dot')}); pack mode {'native32' if native32 else 'default'}.",
           "", f"**Verdict: {'every type covered' if not cov['uncovered'] else 'UNCOVERED: ' + '; '.join(cov['uncovered'])}.**"
           + (f" Non-routed types the batched prompt cannot dequantize (glm-dsa prefill falls back to the token path): "
              f"{'; '.join(cov['prefill_gaps'])}." if cov["prefill_gaps"] else ""),
           "", "## Routed experts", "",
           "| Role | Type | Tensors | GB | Layers | Decode GPU (moe_supported) | CPU lane (vec_dot) | Prefill (MMQ) | Oracle (gguf-py) | Verdict |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    for r in cov["routed"]:
        out.append(f"| {r['role']} | {r['type']} | {r['tensors']} | {r['gb']:.2f} | {r['layers']} | {yn(r['decode_gpu'])} | "
                   f"{yn(r['cpu_lane'])} | {yn(r['prefill_mmq'])} | {yn(r['oracle'])} | {r['verdict']} |")
    out += ["", f"Expert size classes (distinct gate/up and down row bytes over {len(cov['routed_layers'])} MoE layers): "
            f"**{n_cls}**" + (" (more than 8: the slow path needs B3.7's class vectors, present in this tree)"
                             if n_cls > 8 else "") + ".", "",
            "| gate/up row bytes | down row bytes | Layers |", "|---|---|---|"]
    for (gu, d), ls in sorted(cov["classes"].items()):
        ls = sorted(ls)
        out.append(f"| {gu} | {d} | {', '.join(map(str, ls[:12]))}{' ...' if len(ls) > 12 else ''} ({len(ls)}) |")
    out += ["", "## Non-routed tensors", "",
            "| Tensor (per layer) | Type | Tensors | GB | Pack (glm_dsa_pack.plan_tensor) | Decode GPU | Prefill (FP16 dequant / float) | Oracle | Verdict |",
            "|---|---|---|---|---|---|---|---|---|"]
    for r in cov["dense"]:
        out.append(f"| {r['tensor']} | {r['type']} | {r['tensors']} | {r['gb']:.3f} | {r['pack']} | {yn(r['decode_gpu'])} | "
                   f"{yn(r['prefill'])} | {yn(r['oracle'])} | {r['verdict']} |")
    out += ["", "## Engine tables used", "",
            f"- moe_supported (decode GPU, routed): {nm(tab['moe'])}",
            f"- mv_type_ok (decode GPU, non-routed): {nm(tab['mv'])}; B10.7 extension (native32): {nm(tab['mv_ext'])}",
            f"- MMQ instances (prefill, routed): {nm(tab['mmq'])}",
            f"- ggml-cpu vec_dot (CPU lane): {nm(tab['vec_dot'])}",
            f"- FP16 dequant (prefill, non-routed): {nm(tab['f16'])}", "",
            "## Runtime check (on the test PC)", "",
            "Every routed type: float reference vs ggml-cpu vs GPU, max rel err < 1e-3 (BUILD-PLAN B5.3):", "",
            "```sh",
            "cmake -S . -B build -DSTRATA_REAL_SHARD=\"<the shards holding layers 3, 26, 54, 77, 78>\"",
            "ctest --test-dir build -R native_expert_parity_real --output-on-failure",
            "# or directly: build/native_expert_parity <shard> 3 26 54 77 78",
            "```", "", "| Layer | gate/up type | down type | max rel err | Result |", "|---|---|---|---|---|",
            "| 3 | | | | not run |", "| 26 | | | | not run |", "| 54 | | | | not run |", "| 77 | | | | not run |",
            "| 78 (NextN) | | | | not run |", ""]
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("inputs", nargs="+", help="GGUF shard(s), a header dump .json, or an audit .tsv")
    ap.add_argument("--out", help="write the markdown here (default: stdout)")
    ap.add_argument("--ggml-dir", help="llama.cpp checkout for ggml-cpu's vec_dot table")
    ap.add_argument("--native32", action="store_true", default=None, help="the pack's STRATA_PACK_NATIVE32 mode")
    ap.add_argument("--root", default=str(ROOT), help="the source tree whose tables are read")
    a = ap.parse_args(argv)
    try:
        tensors, source = load_tensors(a.inputs)
    except (OSError, ValueError, KeyError) as e:
        print(f"type_coverage: {e}", file=sys.stderr)
        return 2
    if not tensors:
        print("type_coverage: no tensors", file=sys.stderr)
        return 2
    native32 = gp.native32_enabled() if a.native32 is None else a.native32
    tab = engine_tables(pathlib.Path(a.root), pathlib.Path(a.ggml_dir) if a.ggml_dir else None)
    cov = coverage(tensors, tab, oracle_types(), native32)
    md = markdown(cov, tab, source, native32)
    if a.out:
        pathlib.Path(a.out).write_text(md, encoding="utf-8")
        print(f"type_coverage: wrote {a.out}: {len(cov['routed'])} routed and {len(cov['dense'])} non-routed rows, "
              f"{len(cov['classes'])} expert size classes, "
              f"{'all covered' if not cov['uncovered'] else 'UNCOVERED: ' + '; '.join(cov['uncovered'])}")
    else:
        print(md)
    return 1 if cov["uncovered"] else 0


if __name__ == "__main__":
    sys.exit(main())

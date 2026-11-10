#!/usr/bin/env python3
"""tools/strata/gguf_audit.py - header audit gate and placement for a GLM-5.3 (glm-dsa) GGUF (BUILD-PLAN B0.7).

Reads only GGUF headers (key/values and the tensor table, never tensor data) and evaluates checks A1-A14 of
docs/strata/engine/BUILD-PLAN.md section 3.2, then computes the per-GPU and per-NUMA-node placement of section 2.2
from the real per-tensor types (A14).  A file is downloaded only when the audit has no FAIL.

Sources (the first positional argument decides):

    # remote, range-reading headers through huggingface_hub (no download)
    python3 tools/strata/gguf_audit.py unsloth/GLM-5.3-GGUF 'UD-IQ2_M/*.gguf'
    # local shards: the first shard (siblings -0000k-of-0000n are found), a directory, or every shard listed
    python3 tools/strata/gguf_audit.py /data/models/GLM-5.3-GGUF/UD-IQ2_M/GLM-5.3-UD-IQ2_M-00001-of-00006.gguf
    # a header dump written earlier with --dump-header (audit offline, e.g. on a machine with no Hub access)
    python3 tools/strata/gguf_audit.py glm53_UD-IQ2_M_header.json

Outputs: markdown on stdout and, unless --no-write, <out-dir>/<name>_audit.md and <name>_audit.tsv (every tensor:
name, type, dims, shard, bytes, role, GPU).  When the chat template differs from the official one it is saved as
<name>_chat_template.jinja for the diff.  Exit status: 0 no FAIL, 1 at least one FAIL, 2 usage or read error.

Placement model (section 2.2; every constant is an option): 3-way layer split at --split (default 26,54; the MTP
block 78 rides on the last GPU), per GPU usable VRAM after the CUDA context minus the engine reserve, the
non-routed weights as the pack stores them (B1.6: attn_k_b/attn_v_b and the indexer projections as BF16; router,
exp_probs_b and norms F32; non-routed tensors outside the decode GEMV set upcast to BF16; a BF16/F16 head as F32),
the decode activation arena and the f16 KV at each context; what is left is the expert pool.  Each GPU holds
experts of its own layers only; the routed bytes that do not fit are the pinned RAM tier (+ slack slots per MoE
layer), interleaved 50/50 over the two nodes.  The token embedding stays in host RAM.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import math
import os
import pathlib
import re
import struct
import sys
from dataclasses import dataclass, field

ARCH = "glm-dsa"
OFFICIAL_TEMPLATE_SHA256 = "69bb3ab52067898e2466b855407636de559568947f367945842aabcb7fcc1705"

# ggml type id -> (name, block elements, bytes per block).  Matches gguf-py's GGML_QUANT_SIZES at the pin
# 3cf03257 (tools/strata/test_gguf_audit.py checks it).
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4), 1: ("F16", 1, 2), 2: ("Q4_0", 32, 18), 3: ("Q4_1", 32, 20), 6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24), 8: ("Q8_0", 32, 34), 9: ("Q8_1", 32, 36), 10: ("Q2_K", 256, 84), 11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144), 13: ("Q5_K", 256, 176), 14: ("Q6_K", 256, 210), 15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66), 17: ("IQ2_XS", 256, 74), 18: ("IQ3_XXS", 256, 98), 19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18), 21: ("IQ3_S", 256, 110), 22: ("IQ2_S", 256, 82), 23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1), 25: ("I16", 1, 2), 26: ("I32", 1, 4), 27: ("I64", 1, 8), 28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56), 30: ("BF16", 1, 2), 34: ("TQ1_0", 256, 54), 35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17), 40: ("NVFP4", 64, 36), 41: ("Q1_0", 128, 18), 42: ("Q2_0", 64, 18),
}
TYPE_ID = {v[0]: k for k, v in GGML_TYPES.items()}
F32, F16, BF16 = 0, 1, 30

# glm-dsa engine type sets (Maya code, located by content):
# moe_supported() in src/kernels/cuda/glm_fast.cu (moe_gate_up/moe_down instantiations)
MOE_SUPPORTED = {10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 29}
# mv_type_ok() in src/kernels/cuda/glm_fast.cu (the decode GEMV)
MV_SUPPORTED = {F32, BF16, 8, 10, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 29}
# non-routed types the pack dequantizes to BF16 in dense.bin (B1.6)
UPCAST_BF16 = {F16, 2, 3, 6, 7, 20}
# strata_mmq template instances in CMakeLists.txt (read from the file when found, see mmq_types())
MMQ_FALLBACK = {"q2_0", "iq2_xxs", "iq2_xs", "iq2_s", "iq3_xxs", "iq3_s", "iq4_nl", "iq4_xs", "q8_0", "iq1_s",
                "q2_k", "q3_k", "q4_k", "q5_k", "q6_k"}
# routed types without an MMQ instance that the prompt path multiplies in FP16 instead (dequantized by llama.cpp's
# convert.cu, cuBLAS / hipBLAS; src/core/glm_prefill.cu expert_f16): only those matrices, the prompt stays batched
F16_EXPERT = {"IQ1_M", "Q4_0", "Q4_1", "Q5_0", "Q5_1"}

# section 3.2 A2: glm-dsa geometry (key suffix after "glm-dsa.")
GEOMETRY = [
    ("embedding_length", 6144), ("attention.head_count", 64), ("attention.head_count_kv", 1),
    ("attention.q_lora_rank", 2048), ("attention.kv_lora_rank", 512), ("attention.key_length", 576),
    ("attention.value_length", 512), ("attention.key_length_mla", 256), ("attention.value_length_mla", 256),
    ("rope.dimension_count", 64), ("rope.freq_base", 8e6), ("expert_count", 256), ("expert_used_count", 8),
    ("expert_shared_count", 1), ("leading_dense_block_count", 3), ("expert_feed_forward_length", 2048),
    ("feed_forward_length", 12288), ("expert_gating_func", 2), ("expert_weights_scale", 2.5),
    ("expert_weights_norm", True), ("attention.layer_norm_rms_epsilon", 1e-5), ("context_length", 1048576),
    ("attention.indexer.head_count", 32), ("attention.indexer.key_length", 128), ("attention.indexer.top_k", 2048),
]
N_TRUNK = 78
VOCAB = 154880
FULL_LAYERS = [0, 1, 2] + list(range(6, 78, 4))          # 21 full-indexer trunk layers (GLM-5.2/5.3 pattern)
INDEXER_TENSORS = ["indexer.k_norm.weight", "indexer.k_norm.bias", "indexer.proj.weight", "indexer.attn_k.weight",
                   "indexer.attn_q_b.weight"]
ATTN_TENSORS = ["attn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight", "attn_q_a.weight",
                "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight", "attn_v_b.weight", "attn_output.weight",
                "ffn_norm.weight"]
MOE_TENSORS = ["ffn_gate_inp.weight", "exp_probs_b.bias", "ffn_gate_exps.weight", "ffn_up_exps.weight",
               "ffn_down_exps.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"]
NEXTN_TENSORS = ["nextn.eh_proj.weight", "nextn.enorm.weight", "nextn.hnorm.weight"]
# B15.4: the MTP head's precision.  Upstream (llama.cpp #28351) measured teacher-forced head-vs-trunk agreement 0.718
# with a Q4_0 head vs 0.724 bf16 and a collapse at IQ2_XS; the M4 gate needs 0.6 first-position acceptance.  A blk.78
# projection or expert tensor below 4 bits a weight (IQ1/IQ2/IQ3, Q2_K/Q3_K, TQ*, Q1_0/Q2_0) is a WARN.
MTP_MIN_BPW = 4.0
MTP_GROUPS = [
    ("nextn", ["nextn.eh_proj.weight"]),
    ("attention", ["attn_q_a.weight", "attn_q_b.weight", "attn_kv_a_mqa.weight", "attn_k_b.weight",
                   "attn_v_b.weight", "attn_output.weight"]),
    ("indexer", ["indexer.attn_k.weight", "indexer.attn_q_b.weight", "indexer.proj.weight"]),
    ("experts", ["ffn_gate_exps.weight", "ffn_up_exps.weight", "ffn_down_exps.weight"]),
    ("shared expert", ["ffn_gate_shexp.weight", "ffn_up_shexp.weight", "ffn_down_shexp.weight"]),
]
MTP_REMEDY = ("the NextN draft head is IQ2/IQ3-class: expect low MTP acceptance (M4 needs 0.6). Remedy: "
              "STRATA_GLM_MTP_GGUF=<a GGUF holding a Q4_K-or-better blk.%d>; cut one from a larger community quant with "
              "tools/strata/extract_mtp_block.py (local shards, or --remote REPO 'Q4_K_M/*.gguf' to range-read only the "
              "block's bytes), then check it with tools/glm_dsa_mtp_probe.py --agree (docs/strata/volta3/mtp.md)")


def bits_per_weight(t: int) -> float:
    if t not in GGML_TYPES:
        return 0.0
    _, be, bb = GGML_TYPES[t]
    return 8.0 * bb / be


def mtp_type_report(get, block: int) -> tuple[list[str], list[tuple[str, str]]]:
    """B15.4: per group of blk.<block> (get(suffix) -> Tensor or None), the types present ('experts: IQ2_XXS 2.06
    bpw, IQ3_XXS 3.06 bpw'); and the (tensor, type) pairs below MTP_MIN_BPW."""
    lines, low = [], []
    for group, names in MTP_GROUPS:
        seen: dict[str, float] = {}
        for nm in names:
            t = get(nm)
            if t is None:
                continue
            b = bits_per_weight(t.type)
            seen.setdefault(type_name(t.type), b)
            if t.type not in (F32, F16, BF16) and b < MTP_MIN_BPW:
                low.append(("blk.%d.%s" % (block, nm), type_name(t.type)))
        if seen:
            lines.append("%s: %s" % (group, ", ".join("%s %.2f bpw" % (k, v) for k, v in seen.items())))
    return lines, low


# B1.6 BF16 list and F32 list (the pack's form of these tensors, whatever the file carries)
PACK_BF16 = ("attn_k_b.weight", "attn_v_b.weight", "indexer.attn_k.weight", "indexer.attn_q_b.weight",
             "indexer.proj.weight")
ROUTED_RE = re.compile(r"^blk\.(\d+)\.ffn_(gate|up|down)_exps\.weight$")
BLK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")


# ------------------------------------------------------------------------------------------------ header parsing
class StrArray:
    """A long string array (tokens, merges): only its length is kept."""

    def __init__(self, n: int):
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __eq__(self, other) -> bool:
        return isinstance(other, StrArray) and other.n == self.n

    def __repr__(self) -> str:
        return "<%d strings>" % self.n


class _Reader:
    """Sequential little-endian reader over any binary file object (a local file, an HfFileSystem file)."""

    def __init__(self, f, chunk: int = 8 << 20):
        self.f, self.chunk, self.buf, self.pos, self.consumed = f, chunk, b"", 0, 0

    def take(self, n: int) -> bytes:
        while len(self.buf) - self.pos < n:
            c = self.f.read(max(self.chunk, n))
            if not c:
                raise EOFError("GGUF header truncated")
            self.buf, self.pos = self.buf[self.pos:] + c, 0
        x = self.buf[self.pos:self.pos + n]
        self.pos += n
        self.consumed += n
        return x

    def u(self, fmt: str):
        return struct.unpack("<" + fmt, self.take(struct.calcsize("<" + fmt)))[0]

    def string(self) -> str:
        return self.take(self.u("Q")).decode("utf-8", "replace")


_SCALAR = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q", 12: "d"}
KEEP_STRINGS = 256          # string arrays longer than this keep only their length


def _value(r: _Reader, t: int):
    if t == 8:
        return r.string()
    if t == 9:
        et, n = r.u("I"), r.u("Q")
        if et == 8:
            if n > KEEP_STRINGS:
                for _ in range(n):
                    r.take(r.u("Q"))
                return StrArray(n)
            return [r.string() for _ in range(n)]
        if et == 9:
            return [_value(r, 9) for _ in range(n)]
        fmt = _SCALAR[et]
        raw = r.take(n * struct.calcsize(fmt))
        return list(struct.unpack("<%d%s" % (n, fmt), raw))
    if t not in _SCALAR:
        raise ValueError("unknown GGUF value type %d" % t)
    return r.u(_SCALAR[t])


@dataclass
class Tensor:
    name: str
    type: int
    dims: list[int]           # ggml ne order
    offset: int
    shard: int

    @property
    def n(self) -> int:
        return math.prod(self.dims) if self.dims else 1

    @property
    def tname(self) -> str:
        return type_name(self.type)

    @property
    def nbytes(self) -> int:
        return type_bytes(self.type, self.n)


@dataclass
class Shard:
    name: str
    size: int
    version: int = 3
    kv: dict = field(default_factory=dict)
    tensors: list[Tensor] = field(default_factory=list)


def is_norm(name: str) -> bool:
    """attn_norm, q_a/kv_a norms, ffn_norm, output_norm, indexer.k_norm (weight and bias), nextn enorm/hnorm/..."""
    stem = re.sub(r"\.(weight|bias)$", "", name)
    return stem.rsplit(".", 1)[-1].endswith("norm")


def type_name(t: int) -> str:
    return GGML_TYPES[t][0] if t in GGML_TYPES else "type%d" % t


def type_bytes(t: int, n: int) -> int:
    if t not in GGML_TYPES:
        return 0
    _, bs, ts = GGML_TYPES[t]
    return (n // bs) * ts if n % bs == 0 else -(-n // bs) * ts


def read_header(f, name: str, size: int, index: int) -> Shard:
    r = _Reader(f)
    if r.take(4) != b"GGUF":
        raise ValueError("%s: not a GGUF file" % name)
    sh = Shard(name=name, size=size, version=r.u("I"))
    nt, nkv = r.u("Q"), r.u("Q")
    for _ in range(nkv):
        k = r.string()
        sh.kv[k] = _value(r, r.u("I"))
    for _ in range(nt):
        tn = r.string()
        nd = r.u("I")
        dims = [r.u("Q") for _ in range(nd)]
        sh.tensors.append(Tensor(tn, r.u("I"), dims, r.u("Q"), index))
    return sh


# ------------------------------------------------------------------------------------------------ sources
def local_shards(args: list[str]) -> list[pathlib.Path]:
    paths: list[pathlib.Path] = []
    for a in args:
        p = pathlib.Path(a).expanduser()
        if p.is_dir():
            paths += sorted(p.glob("*.gguf"))
        elif p.is_file():
            m = re.search(r"-(\d{5})-of-(\d{5})\.gguf$", p.name)
            if m and len(args) == 1:
                n = int(m.group(2))
                stem = p.name[:m.start()]
                paths += [p.with_name("%s-%05d-of-%05d.gguf" % (stem, i, n)) for i in range(1, n + 1)]
            else:
                paths.append(p)
        else:
            paths += sorted(pathlib.Path(x) for x in __import__("glob").glob(str(p)))
    seen, out = set(), []
    for p in paths:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def load_local(args: list[str]) -> list[Shard]:
    shards = []
    for i, p in enumerate(local_shards(args)):
        if not p.exists():
            raise FileNotFoundError("shard missing: %s" % p)
        with open(p, "rb") as f:
            shards.append(read_header(f, p.name, p.stat().st_size, i))
    if not shards:
        raise FileNotFoundError("no GGUF files in %s" % " ".join(args))
    return shards


def load_remote(repo: str, pattern: str) -> list[Shard]:
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as e:   # pragma: no cover - exercised on the test PC only
        raise SystemExit("remote audit needs huggingface_hub (python3 -m pip install -U 'huggingface_hub>=0.34')") from e
    fs = HfFileSystem()
    files = sorted(x for x in fs.find(repo) if x.endswith(".gguf") and
                   fnmatch.fnmatch(x.split(repo + "/", 1)[1], pattern))
    if not files:
        raise FileNotFoundError("no files match %s in %s" % (pattern, repo))
    shards = []
    for i, p in enumerate(files):
        size = int(fs.info(p)["size"])
        with fs.open(p, "rb", block_size=8 << 20) as f:
            shards.append(read_header(f, p.split(repo + "/", 1)[1], size, i))
    return shards


def _jsonable(v):
    if isinstance(v, StrArray):
        return {"__strarray__": v.n}
    return v


def _from_json(v):
    if isinstance(v, dict) and "__strarray__" in v:
        return StrArray(int(v["__strarray__"]))
    return v


def dump_header(shards: list[Shard], path: pathlib.Path, source: str) -> None:
    doc = {"format": "strata-gguf-header/1", "source": source,
           "shards": [{"name": s.name, "size": s.size, "version": s.version,
                       "kv": {k: _jsonable(v) for k, v in s.kv.items()},
                       "tensors": [[t.name, t.type, t.dims, t.offset] for t in s.tensors]} for s in shards]}
    path.write_text(json.dumps(doc, ensure_ascii=False), encoding="utf-8")


def load_dump(path: pathlib.Path) -> list[Shard]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("format") != "strata-gguf-header/1":
        raise ValueError("%s is not an strata-gguf-header/1 dump" % path)
    out = []
    for i, s in enumerate(doc["shards"]):
        sh = Shard(s["name"], int(s["size"]), int(s.get("version", 3)), {k: _from_json(v) for k, v in s["kv"].items()})
        sh.tensors = [Tensor(n, int(t), [int(d) for d in dims], int(o), i) for n, t, dims, o in s["tensors"]]
        out.append(sh)
    return out


def mmq_types(repo_root: pathlib.Path | None = None) -> set[int]:
    """strata_mmq instance types, read from CMakeLists.txt (foreach over mmq-instance-<t>.cu) when present."""
    names = set(MMQ_FALLBACK)
    root = repo_root or pathlib.Path(__file__).resolve().parents[2]
    cm = root / "CMakeLists.txt"
    if cm.exists():
        m = re.search(r"foreach\(_t ([a-z0-9_ ]+)\)[^\n]*\n\s*target_sources\(strata_mmq[^\n]*mmq-instance-",
                      cm.read_text(encoding="utf-8", errors="replace"))
        if m:
            names = set(m.group(1).split())
    return {TYPE_ID[n.upper()] for n in names if n.upper() in TYPE_ID}


# ------------------------------------------------------------------------------------------------ audit
@dataclass
class Check:
    id: str
    title: str
    status: str = "PASS"            # PASS / WARN / FAIL
    notes: list[str] = field(default_factory=list)

    def fail(self, msg: str) -> None:
        self.status = "FAIL"
        self.notes.append("FAIL: " + msg)

    def warn(self, msg: str) -> None:
        if self.status == "PASS":
            self.status = "WARN"
        self.notes.append("WARN: " + msg)

    def info(self, msg: str) -> None:
        self.notes.append(msg)


@dataclass
class Options:
    role: str = "primary"
    max_gb: float | None = None
    split: tuple[int, ...] = (26, 54)
    gpu_usable_gb: float = 33.7
    reserve_gb: float = 0.7
    arena_gb: float = 1.2
    contexts: tuple[int, ...] = (4096, 32768, 65536, 131072)
    ram_node_gb: tuple[float, ...] = (103.1, 103.1)
    mem_available_gb: float = 196.0
    headroom_gib: float = 8.0
    staging_gb: float = 2.0
    slack_slots: int = 16
    closure_pct: float = 0.5
    max_classes: int = 8
    repo_root: pathlib.Path | None = None


def _eq(v, want) -> bool:
    if isinstance(want, bool):
        return isinstance(v, (bool, int)) and bool(v) == want and v in (0, 1, True, False)
    if isinstance(v, list):
        return len(v) > 0 and all(_eq(x, want) for x in v)
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return math.isclose(float(v), float(want), rel_tol=1e-6, abs_tol=0.0)
    return False


class Audit:
    def __init__(self, shards: list[Shard], opt: Options):
        self.shards, self.opt = shards, opt
        self.kv: dict = {}
        for s in shards:
            for k, v in s.kv.items():
                self.kv.setdefault(k, v)
        self.kv.update(shards[0].kv)
        self.tensors: dict[str, Tensor] = {}
        self.dupes: list[str] = []
        for s in shards:
            for t in s.tensors:
                if t.name in self.tensors:
                    self.dupes.append(t.name)
                self.tensors[t.name] = t
        self.arch = self.kv.get("general.architecture")
        self.prefix = (self.arch or ARCH) + "."
        self.checks: list[Check] = []
        self.placement: dict | None = None
        self.template_sha: str | None = None

    # helpers
    def k(self, suffix: str):
        return self.kv.get(self.prefix + suffix)

    def t(self, name: str) -> Tensor | None:
        return self.tensors.get(name)

    def blk(self, i: int, suffix: str) -> Tensor | None:
        return self.tensors.get("blk.%d.%s" % (i, suffix))

    @property
    def n_blocks(self) -> int:
        bc = self.k("block_count")
        return int(bc) if isinstance(bc, int) else 1 + max([int(m.group(1)) for m in
                                                           (BLK_RE.match(n) for n in self.tensors) if m] or [-1])

    @property
    def has_mtp(self) -> bool:
        return (self.k("nextn_predict_layers") or 0) >= 1 and self.n_blocks > N_TRUNK

    def full_layers(self) -> list[int]:
        types = self.k("attention.indexer.types")
        if isinstance(types, list):
            full = [i for i, v in enumerate(types) if v]
        elif (self.k("context_length") or 0) >= 1048576:
            full = list(FULL_LAYERS)
        else:
            full = list(range(min(self.n_blocks, N_TRUNK)))
        if self.has_mtp:
            full.append(self.n_blocks - 1)
        return full

    def file_bytes(self) -> int:
        return sum(s.size for s in self.shards)

    # ------------------------------------------------------------------ checks
    def run(self) -> list[Check]:
        for fn in (self.a1, self.a2, self.a3, self.a4, self.a5, self.a6, self.a7, self.a8, self.a9, self.a10,
                   self.a11, self.a12, self.a13, self.a14):
            self.checks.append(fn())
        return self.checks

    def a1(self) -> Check:
        c = Check("A1", "general.architecture")
        if self.arch != ARCH:
            c.fail("general.architecture is %r, expected %r" % (self.arch, ARCH))
        else:
            c.info("glm-dsa")
        return c

    def a2(self) -> Check:
        c = Check("A2", "geometry key/values")
        bc, nx = self.k("block_count"), self.k("nextn_predict_layers")
        if bc == 79 and nx == 1:
            c.info("block_count 79, nextn_predict_layers 1")
        elif bc == 78 and not nx:
            c.warn("block_count 78 and no nextn_predict_layers: no MTP block")
        else:
            c.fail("block_count %r with nextn_predict_layers %r (want 79/1, or 78 without MTP)" % (bc, nx))
        for key, want in GEOMETRY:
            v = self.k(key)
            if v is None:
                if key == "expert_gating_func":
                    c.warn("%s%s absent (llama.cpp defaults to sigmoid; glm-dsa's reader requires it)" % (self.prefix, key))
                else:
                    c.fail("%s%s missing (want %r)" % (self.prefix, key, want))
            elif not _eq(v, want):
                c.fail("%s%s = %r, want %r" % (self.prefix, key, v, want))
        for key in sorted(k for k in self.kv if k.startswith(self.prefix + "rope.scaling.")):
            v = self.kv[key]
            benign = (key.endswith(".type") and v in ("none", "")) or (key.endswith(".factor") and _eq(v, 1.0)) \
                or (key.endswith(".factor") and v == 0)
            if not benign:
                c.fail("rope scaling present: %s = %r (GLM-5.3 has none)" % (key, v))
        for name in ("token_embd.weight", "output.weight"):
            t = self.t(name)
            if t is None:
                if name == "token_embd.weight":
                    c.fail("token_embd.weight missing")
                continue
            if t.dims != [6144, VOCAB]:
                c.fail("%s dims %s, want [6144, %d] (vocab %d)" % (name, t.dims, VOCAB, VOCAB))
        vs = self.k("vocab_size")
        if vs is not None and vs != VOCAB:
            c.fail("%svocab_size = %r, want %d" % (self.prefix, vs, VOCAB))
        return c

    def a3(self) -> Check:
        c = Check("A3", "indexer schedule and indexer tensors")
        types = self.k("attention.indexer.types")
        if isinstance(types, list):
            full = [i for i, v in enumerate(types) if v]
            if len(types) != N_TRUNK:
                c.fail("attention.indexer.types has %d entries, want %d" % (len(types), N_TRUNK))
            if full != FULL_LAYERS:
                c.fail("attention.indexer.types marks %s full, want %s" % (full, FULL_LAYERS))
            else:
                c.info("indexer.types: 21 full layers {0,1,2,6,...,74}")
        elif types is None:
            if (self.k("context_length") or 0) >= 1048576:
                c.info("indexer.types absent; llama.cpp BC rule (context_length >= 1048576) gives the GLM-5.2 pattern")
            else:
                c.fail("indexer.types absent and context_length < 1048576: llama.cpp treats every layer as full")
        else:
            c.fail("attention.indexer.types has an unexpected form: %r" % (types,))
        want = list(FULL_LAYERS) + ([self.n_blocks - 1] if self.has_mtp else [])
        for i in want:
            miss = [s for s in INDEXER_TENSORS if self.blk(i, s) is None]
            if miss:
                c.fail("full layer %d lacks %s" % (i, ", ".join(miss)))
        extra = sorted(i for i in range(min(self.n_blocks, N_TRUNK)) if i not in FULL_LAYERS and
                       any(self.blk(i, s) for s in INDEXER_TENSORS))
        if extra:
            c.warn("indexer tensors on %d shared layers (e.g. %s): ignored on shared layers" % (len(extra), extra[:6]))
        return c

    def _routed(self) -> dict[int, dict[str, Tensor | None]]:
        out = {}
        lead = self.k("leading_dense_block_count")
        lead = int(lead) if isinstance(lead, int) else 3
        for i in range(lead, self.n_blocks):
            out[i] = {p: self.blk(i, "ffn_%s_exps.weight" % p) for p in ("gate", "up", "down")}
        return out

    def a4(self) -> Check:
        c = Check("A4", "routed expert types (layers 3-78)")
        hist: dict[str, int] = {}
        for i, g in self._routed().items():
            miss = [p for p, t in g.items() if t is None]
            if miss:
                c.fail("blk.%d lacks ffn_{%s}_exps" % (i, ",".join(miss)))
                continue
            for p, t in g.items():
                hist[t.tname] = hist.get(t.tname, 0) + 1
                if t.type not in MOE_SUPPORTED:
                    c.fail("blk.%d.ffn_%s_exps is %s: no decode MoE kernel (moe_supported)" % (i, p, t.tname))
                want = [6144, 2048, 256] if p != "down" else [2048, 6144, 256]
                if t.dims != want:
                    c.fail("blk.%d.ffn_%s_exps dims %s, want %s" % (i, p, t.dims, want))
            if g["gate"].type != g["up"].type:
                c.fail("blk.%d gate %s != up %s (iq_pack requires equal gate/up types)" % (i, g["gate"].tname, g["up"].tname))
        c.info("types (tensor count): " + ", ".join("%s %d" % kv for kv in sorted(hist.items())))
        if "IQ1_M" in hist:
            c.info("IQ1_M experts decode; the prompt path multiplies them in FP16 (no MMQ instance, see A5)")
        return c

    def a5(self) -> Check:
        c = Check("A5", "MMQ coverage of routed types (prefill)")
        mmq = mmq_types(self.opt.repo_root)
        bad: dict[str, list[int]] = {}
        for i, g in self._routed().items():
            for t in g.values():
                if t is not None and t.type not in mmq:
                    bad.setdefault(t.tname, []).append(i)
        for tn, layers in sorted(bad.items()):
            if tn in F16_EXPERT:
                c.warn("%s on %d layers (e.g. blk.%d) has no strata_mmq instance: those matrices take the FP16 "
                       "products (slower than MMQ), the prompt stays batched" % (tn, len(set(layers)), layers[0]))
            else:
                c.fail("%s on %d layers (e.g. blk.%d) has no strata_mmq instance: whole prompt token by token"
                       % (tn, len(set(layers)), layers[0]))
        c.info("strata_mmq types: " + ", ".join(sorted(type_name(x) for x in mmq)))
        return c

    def a6(self) -> Check:
        c = Check("A6", "non-routed tensor types")
        upc: dict[str, list[int]] = {}
        for t in self.tensors.values():
            if ROUTED_RE.match(t.name):
                continue
            if t.type in MV_SUPPORTED:
                continue
            if t.type in UPCAST_BF16:
                e = upc.setdefault(t.tname, [0, 0])
                e[0] += 1
                e[1] += 2 * t.n - t.nbytes
                continue
            if t.name == "token_embd.weight" and t.type in GGML_TYPES:
                c.info("token_embd.weight is %s (host dequant)" % t.tname)
                continue
            c.fail("%s is %s: neither decodable nor in the pack upcast set" % (t.name, t.tname))
        for tn, (n, extra) in sorted(upc.items()):
            c.info("%d %s tensors upcast to BF16 by the pack: +%.3f GB" % (n, tn, extra / 1e9))
        return c

    def a7(self) -> Check:
        c = Check("A7", "MLA attn_k_b / attn_v_b")
        hist: dict[str, int] = {}
        for i in range(self.n_blocks):
            for s, want in (("attn_k_b.weight", [192, 512, 64]), ("attn_v_b.weight", [512, 256, 64])):
                t = self.blk(i, s)
                if t is None:
                    c.fail("blk.%d.%s missing" % (i, s))
                    continue
                hist["%s %s" % (s.split(".")[0], t.tname)] = hist.get("%s %s" % (s.split(".")[0], t.tname), 0) + 1
                if t.dims != want:
                    c.fail("blk.%d.%s dims %s, want %s" % (i, s, t.dims, want))
        c.info("types: " + ", ".join("%s x%d" % kv for kv in sorted(hist.items())) + " (pack upcasts to BF16)")
        return c

    def a8(self) -> Check:
        c = Check("A8", "router, exp_probs_b, norms")
        rt: dict[str, int] = {}
        for i in self._routed():
            r, b = self.blk(i, "ffn_gate_inp.weight"), self.blk(i, "exp_probs_b.bias")
            if r is None:
                c.fail("blk.%d.ffn_gate_inp.weight missing" % i)
            else:
                rt[r.tname] = rt.get(r.tname, 0) + 1
                if r.type != F32:
                    c.warn("blk.%d.ffn_gate_inp is %s (glm-dsa keeps the router F32, G23)" % (i, r.tname))
            if b is None:
                c.fail("blk.%d.exp_probs_b.bias missing" % i)
            elif b.type != F32:
                c.fail("blk.%d.exp_probs_b.bias is %s, must be F32 (top-8 flips)" % (i, b.tname))
        c.info("router types: " + ", ".join("%s x%d" % kv for kv in sorted(rt.items())))
        for t in self.tensors.values():
            if is_norm(t.name) and t.type != F32:
                c.warn("%s is %s, norms are expected F32" % (t.name, t.tname))
        return c

    def a9(self) -> Check:
        c = Check("A9", "MTP block (blk.78)")
        if not self.has_mtp:
            c.warn("no MTP block: M4 needs STRATA_GLM_MTP_GGUF with another file's blk.78 "
                   "(tools/strata/extract_mtp_block.py cuts one out of a larger community quant)")
            return c
        i = self.n_blocks - 1
        need = NEXTN_TENSORS + ATTN_TENSORS + INDEXER_TENSORS + MOE_TENSORS
        miss = [s for s in need if self.blk(i, s) is None]
        if miss:
            c.warn("blk.%d lacks %s" % (i, ", ".join(miss)))
        else:
            c.info("blk.%d nextn.{eh_proj,enorm,hnorm}, attention, indexer and experts present" % i)
        if self.blk(i, "nextn.shared_head_norm.weight") is None:
            c.info("nextn.shared_head_norm absent: falls back to output_norm")
        # B15.4: the head's precision decides the MTP gain
        lines, low = mtp_type_report(lambda nm: self.blk(i, nm), i)
        c.info("blk.%d types: %s" % (i, "; ".join(lines)))
        if low:
            c.warn("%d blk.%d tensors below %.0f bits a weight (%s): %s" % (
                len(low), i, MTP_MIN_BPW, ", ".join(sorted({t for _, t in low})), MTP_REMEDY % i))
        return c

    def a10(self) -> Check:
        c = Check("A10", "output head")
        t = self.t("output.weight")
        if t is None:
            c.fail("output.weight missing")
        elif t.type in (BF16, F16):
            c.warn("output.weight is %s: upcast to F32 costs %.2f GB VRAM on the last GPU" % (t.tname, (4 * t.n) / 1e9))
        else:
            c.info("output.weight %s (%.2f GB)" % (t.tname, t.nbytes / 1e9))
        return c

    def a11(self) -> Check:
        c = Check("A11", "tokenizer and chat template")
        pre = self.kv.get("tokenizer.ggml.pre")
        if pre not in ("glm4", "glm5", "chatglm-bpe"):
            c.fail("tokenizer.ggml.pre = %r, want glm4/glm5/chatglm-bpe" % (pre,))
        else:
            c.info("pre %s" % pre)
            if pre == "chatglm-bpe":
                c.warn("pre chatglm-bpe: tools/strata_tokenizer.py transcribes glm4/glm5 only")
        if self.kv.get("tokenizer.ggml.model") != "gpt2":
            c.fail("tokenizer.ggml.model = %r, want gpt2" % (self.kv.get("tokenizer.ggml.model"),))
        toks = self.kv.get("tokenizer.ggml.tokens")
        if toks is not None and len(toks) != VOCAB:
            c.warn("tokenizer.ggml.tokens has %d entries (embedding rows %d)" % (len(toks), VOCAB))
        tpl = self.kv.get("tokenizer.chat_template")
        self.template_sha = None
        if not isinstance(tpl, str):
            c.warn("no tokenizer.chat_template")
        else:
            self.template_sha = hashlib.sha256(tpl.encode("utf-8")).hexdigest()
            if self.template_sha != OFFICIAL_TEMPLATE_SHA256:
                c.warn("chat template sha256 %s differs from the official %s... (saved for the diff; B12.3)"
                       % (self.template_sha, OFFICIAL_TEMPLATE_SHA256[:8]))
            else:
                c.info("chat template is the official GLM-5.3 template (sha256 69bb3ab5...)")
        return c

    def a12(self) -> Check:
        c = Check("A12", "size and shards")
        total = self.file_bytes()
        c.info("%d shards, %.2f GB (%.2f GiB)" % (len(self.shards), total / 1e9, total / 2**30))
        meta = [s.name for s in self.shards if not s.tensors]
        if meta:
            c.info("metadata-only shard(s): %s (#83 shape, accepted)" % ", ".join(meta))
        cnt = self.shards[0].kv.get("split.count")
        if cnt is not None and int(cnt) != len(self.shards):
            c.fail("split.count %s but %d shards found" % (cnt, len(self.shards)))
        for i, s in enumerate(self.shards):
            no = s.kv.get("split.no")
            if no is not None and int(no) != i:
                c.fail("%s has split.no %s, expected %d (shard order)" % (s.name, no, i))
        tc = self.shards[0].kv.get("split.tensors.count")
        n = sum(len(s.tensors) for s in self.shards)
        if tc is not None and int(tc) != n:
            c.fail("split.tensors.count %s but %d tensors found" % (tc, n))
        if self.dupes:
            c.fail("duplicate tensors across shards: %s" % ", ".join(self.dupes[:5]))
        limit = self.opt.max_gb if self.opt.max_gb is not None else (245.0 if self.opt.role == "primary" else None)
        if limit is not None and total > limit * 1e9:
            c.fail("%.2f GB > %.1f GB budget for the %s role" % (total / 1e9, limit, self.opt.role))
        unknown = sorted({t.tname for t in self.tensors.values() if t.type not in GGML_TYPES})
        if unknown:
            c.fail("unknown ggml types %s: sizes cannot be computed" % unknown)
        return c

    def classes(self) -> dict[tuple[int, int], list[int]]:
        out: dict[tuple[int, int], list[int]] = {}
        for i, g in self._routed().items():
            if any(t is None for t in g.values()):
                continue
            gu = type_bytes(g["gate"].type, 6144 * 2048) + type_bytes(g["up"].type, 6144 * 2048)
            dn = type_bytes(g["down"].type, 2048 * 6144)
            out.setdefault((gu, dn), []).append(i)
        return out

    def a13(self) -> Check:
        c = Check("A13", "expert size classes")
        cl = self.classes()
        for (gu, dn), layers in sorted(cl.items()):
            c.info("gate+up %d B, down %d B per expert (blob %.3f MB): %d layers" % (gu, dn, (gu + dn) / 1e6, len(layers)))
        if len(cl) > self.opt.max_classes:
            c.warn("%d classes > %d: Maya's slow-path class_off_[8] overflows; FAIL for the slow path until B3.7"
                   % (len(cl), self.opt.max_classes))
        return c

    # ------------------------------------------------------------------ placement (A14)
    def gpu_of(self, layer: int) -> int:
        g = 0
        for b in self.opt.split:
            if layer >= b:
                g += 1
        return g

    def packed_bytes(self, t: Tensor) -> int:
        base = t.name.split(".", 2)[2] if t.name.startswith("blk.") else t.name
        if base in PACK_BF16:
            return 2 * t.n
        if base in ("ffn_gate_inp.weight", "exp_probs_b.bias") or is_norm(t.name):
            return 4 * t.n
        if t.name == "output.weight" and t.type in (BF16, F16):
            return 4 * t.n
        if t.type in UPCAST_BF16:
            return 2 * t.n
        return t.nbytes

    def a14(self) -> Check:
        c = Check("A14", "placement closure (per GPU, per node)")
        o = self.opt
        ngpu = len(o.split) + 1
        last = ngpu - 1
        full = set(self.full_layers())
        nonrouted = [0] * ngpu
        kbvb = [0] * ngpu
        routed = [0] * ngpu
        experts = [0] * ngpu
        layers = [[] for _ in range(ngpu)]
        moe_layers = set()
        embd = 0
        total_native = 0
        for t in self.tensors.values():
            total_native += t.nbytes
            m = BLK_RE.match(t.name)
            if t.name == "token_embd.weight":
                embd += t.nbytes
                continue
            g = self.gpu_of(int(m.group(1))) if m else last
            if ROUTED_RE.match(t.name):
                routed[g] += t.nbytes
                if t.name.endswith("gate_exps.weight"):
                    experts[g] += t.dims[2] if len(t.dims) == 3 else 0
                    moe_layers.add(int(m.group(1)))
                continue
            if m and m.group(2) in ("attn_k_b.weight", "attn_v_b.weight"):
                kbvb[g] += self.packed_bytes(t)
            else:
                nonrouted[g] += self.packed_bytes(t)
        for i in range(self.n_blocks):
            layers[self.gpu_of(i)].append(i)
        kv_tok = [len(ls) * 1152 + sum(1 for i in ls if i in full) * 512 for ls in layers]
        file_b = self.file_bytes()
        gap = (file_b - total_native) / file_b * 100 if file_b else 100.0
        if abs(gap) > o.closure_pct:
            c.fail("tensor bytes %.3f GB vs file %.3f GB: %.2f%% apart (> %.1f%%)"
                   % (total_native / 1e9, file_b / 1e9, gap, o.closure_pct))
        else:
            c.info("tensor bytes close to the file size: %.3f of %.3f GB (%.3f%% header/padding)"
                   % (total_native / 1e9, file_b / 1e9, gap))
        blob_all = sum(routed) / max(1, sum(experts))
        slack = o.slack_slots * len(moe_layers) * blob_all
        budget = o.mem_available_gb * 1e9 - o.headroom_gib * 2**30 - o.staging_gb * 1e9
        rows = {}
        for ctx in o.contexts:
            pool, vram_exp, vram_bytes = [], [], []
            for g in range(ngpu):
                p = o.gpu_usable_gb * 1e9 - o.reserve_gb * 1e9 - nonrouted[g] - kbvb[g] - o.arena_gb * 1e9 - kv_tok[g] * ctx
                pool.append(p)
                blob = routed[g] / experts[g] if experts[g] else 0
                n = min(experts[g], int(max(0.0, p) // blob)) if blob else 0
                vram_exp.append(n)
                vram_bytes.append(n * blob)
            pinned = sum(routed) - sum(vram_bytes) + slack
            rows[ctx] = {"kv": [kv_tok[g] * ctx for g in range(ngpu)], "pool": pool, "experts": vram_exp,
                         "vram_routed": vram_bytes, "pinned": pinned, "spare": budget - pinned}
        self.placement = {"ngpu": ngpu, "layers": layers, "nonrouted": nonrouted, "kbvb": kbvb, "routed": routed,
                          "experts": experts, "kv_tok": kv_tok, "embd": embd, "slack": slack, "budget": budget,
                          "blob": blob_all, "rows": rows, "file": file_b, "tensors": total_native,
                          "n_moe": len(moe_layers)}
        for ctx, r in rows.items():
            for g in range(ngpu):
                if r["pool"][g] <= 0:
                    (c.fail if ctx <= 65536 else c.warn)("GPU%d has no expert pool at %dK context (%.2f GB)"
                                                        % (g, ctx // 1024, r["pool"][g] / 1e9))
        if 65536 in rows:
            r = rows[65536]
            if r["spare"] < 0:
                c.fail("pinned RAM tier at 64K %.1f GB exceeds the budget %.1f GB (MemAvailable %.0f - %.0f GiB "
                       "headroom - %.0f staging)" % (r["pinned"] / 1e9, budget / 1e9, o.mem_available_gb,
                                                      o.headroom_gib, o.staging_gb))
            else:
                c.info("64K: pinned %.1f GB within the %.1f GB budget (%.1f spare)"
                       % (r["pinned"] / 1e9, budget / 1e9, r["spare"] / 1e9))
        return c

    # ------------------------------------------------------------------ outputs
    def tsv(self) -> str:
        lines = ["name\ttype\tdims\tshard\tbytes\tpacked_bytes\trole\tgpu"]
        for t in sorted(self.tensors.values(), key=lambda t: (t.shard, t.offset)):
            m = BLK_RE.match(t.name)
            role = "embedding" if t.name == "token_embd.weight" else "routed" if ROUTED_RE.match(t.name) else "dense"
            gpu = "host" if role == "embedding" else str(self.gpu_of(int(m.group(1))) if m else len(self.opt.split))
            packed = t.nbytes if role != "dense" else self.packed_bytes(t)
            lines.append("%s\t%s\t%s\t%d\t%d\t%d\t%s\t%s" % (t.name, t.tname, ",".join(map(str, t.dims)), t.shard + 1,
                                                          t.nbytes, packed, role, gpu))
        return "\n".join(lines) + "\n"

    def markdown(self, title: str) -> str:
        n_fail = sum(c.status == "FAIL" for c in self.checks)
        n_warn = sum(c.status == "WARN" for c in self.checks)
        out = ["# GGUF audit: %s" % title, "",
               "Role: %s. Verdict: **%s** (%d FAIL, %d WARN). Generated by tools/strata/gguf_audit.py "
               "(BUILD-PLAN section 3.2)." % (self.opt.role, "FAIL" if n_fail else "PASS", n_fail, n_warn), "",
               "| # | Check | Result | Notes |", "|---|---|---|---|"]
        for c in self.checks:
            notes = "<br>".join(n.replace("|", "\\|") for n in c.notes[:12])
            if len(c.notes) > 12:
                notes += "<br>(+%d more, see below)" % (len(c.notes) - 12)
            out.append("| %s | %s | %s | %s |" % (c.id, c.title, c.status, notes))
        long = [c for c in self.checks if len(c.notes) > 12]
        for c in long:
            out += ["", "### %s full notes" % c.id, ""] + ["- " + n for n in c.notes]
        out += ["", "## Placement (A14)", ""] + self.placement_md()
        return "\n".join(out) + "\n"

    def placement_md(self) -> list[str]:
        p = self.placement
        if not p:
            return ["(not computed)"]
        o, n = self.opt, p["ngpu"]
        gb = lambda x: "%.2f" % (x / 1e9)   # noqa: E731
        hdr = ["GPU%d (L%d-%d%s)" % (g, p["layers"][g][0], p["layers"][g][-1],
                                    " incl. MTP" if self.has_mtp and g == n - 1 else "") for g in range(n)]
        out = ["Split %s; usable %.1f GB/GPU after the CUDA context; expert blob mean %.2f MB; %d MoE layers; "
               "file %s GB, tensors %s GB." % (",".join(map(str, o.split)), o.gpu_usable_gb, p["blob"] / 1e6,
                                               p["n_moe"], gb(p["file"]), gb(p["tensors"])), "",
               "| Per GPU (GB, 1e9) | " + " | ".join(hdr) + " | Total |", "|---" * (n + 2) + "|"]

        def row(label, vals, total=True):
            out.append("| %s | %s | %s |" % (label, " | ".join(gb(v) for v in vals), gb(sum(vals)) if total else ""))
        row("Usable after CUDA context", [o.gpu_usable_gb * 1e9] * n)
        row("Engine reserve", [o.reserve_gb * 1e9] * n)
        row("Non-routed weights (packed form)", p["nonrouted"])
        row("attn_k_b/attn_v_b as BF16", p["kbvb"])
        row("Activation arena", [o.arena_gb * 1e9] * n)
        out.append("| KV bytes per token | %s | %d |" % (" | ".join(str(x) for x in p["kv_tok"]), sum(p["kv_tok"])))
        for ctx, r in p["rows"].items():
            row("KV f16 @%dK" % (ctx // 1024), r["kv"])
        for ctx, r in p["rows"].items():
            tot = sum(r["experts"])
            out.append("| **Expert pool @%dK** | %s | **%s (%d experts, %.0f%%)** |"
                       % (ctx // 1024, " | ".join("%s (%d)" % (gb(r["pool"][g]), r["experts"][g]) for g in range(n)),
                          gb(sum(max(0, x) for x in r["pool"])), tot, 100.0 * tot / max(1, sum(p["experts"]))))
        row("Routed bytes of the GPU's layers", p["routed"])
        nodes = len(o.ram_node_gb)
        out += ["", "| Host RAM (GB) | " + " | ".join("Node %d" % i for i in range(nodes)) + " | Total |",
                "|---" * (nodes + 2) + "|",
                "| Physical | " + " | ".join("%.1f" % x for x in o.ram_node_gb) + " | %.1f |" % sum(o.ram_node_gb)]
        for ctx, r in p["rows"].items():
            out.append("| Pinned RAM tier @%dK (incl. %.1f slack, interleaved) | %s | %s |"
                       % (ctx // 1024, p["slack"] / 1e9, " | ".join(gb(r["pinned"] / nodes) for _ in range(nodes)),
                          gb(r["pinned"])))
        blank = " |" * nodes
        out.append("| Token embedding (host) |%s %s |" % (blank, gb(p["embd"])))
        out.append("| Budget: MemAvailable %.0f - %.0f GiB headroom - %.0f staging |%s %s |"
                   % (o.mem_available_gb, o.headroom_gib, o.staging_gb, blank, gb(p["budget"])))
        for ctx, r in p["rows"].items():
            out.append("| Spare @%dK |%s %s%s |" % (ctx // 1024, blank, gb(r["spare"]),
                                                  "" if r["spare"] >= 0 else " (does not fit)"))
        return out


def write_outputs(a: Audit, out_dir: pathlib.Path, name: str, md: str) -> list[pathlib.Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = [out_dir / ("%s_audit.md" % name), out_dir / ("%s_audit.tsv" % name)]
    paths[0].write_text(md, encoding="utf-8")
    paths[1].write_text(a.tsv(), encoding="utf-8")
    tpl = a.kv.get("tokenizer.chat_template")
    if isinstance(tpl, str) and a.template_sha not in (None, OFFICIAL_TEMPLATE_SHA256):
        p = out_dir / ("%s_chat_template.jinja" % name)
        p.write_text(tpl, encoding="utf-8", newline="\n")
        paths.append(p)
    return paths


def default_name(src: list[str]) -> str:
    if len(src) == 2 and not pathlib.Path(src[0]).exists():
        stem = src[1].split("/")[0] if "/" in src[1] else src[1]
    else:
        p = pathlib.Path(src[0])
        stem = p.name if p.suffix == ".json" or not p.is_file() else p.parent.name
        stem = re.sub(r"(-\d{5}-of-\d{5})?\.(gguf|json)$", "", stem)
        stem = re.sub(r"_header$", "", stem)
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "", stem.replace("*", "")) or "gguf"
    return stem if stem.startswith("glm53_") else "glm53_" + stem


def parse_args(argv: list[str]) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("source", nargs="+", help="REPO PATTERN, local shard(s)/directory, or a header dump .json")
    ap.add_argument("--role", default="primary", choices=["primary", "fallback", "alternate", "emergency"])
    ap.add_argument("--max-gb", type=float, help="A12 size limit (default 245 for the primary role)")
    ap.add_argument("--split", default="26,54", help="GPU layer boundaries (default 26,54)")
    ap.add_argument("--gpu-usable-gb", type=float, default=33.7)
    ap.add_argument("--reserve-gb", type=float, default=0.7)
    ap.add_argument("--arena-gb", type=float, default=1.2)
    ap.add_argument("--contexts", default="4096,32768,65536,131072")
    ap.add_argument("--ram-node-gb", default="103.1,103.1", help="physical RAM per NUMA node")
    ap.add_argument("--mem-available-gb", type=float, default=196.0)
    ap.add_argument("--headroom-gib", type=float, default=8.0)
    ap.add_argument("--staging-gb", type=float, default=2.0)
    ap.add_argument("--slack-slots", type=int, default=16)
    ap.add_argument("--out-dir", default=os.path.expanduser("~/strata-facts"))
    ap.add_argument("--name", help="output name stem (default glm53_<folder>)")
    ap.add_argument("--no-write", action="store_true", help="markdown to stdout only")
    ap.add_argument("--dump-header", help="also write the parsed headers as JSON (re-auditable offline)")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    a = parse_args(sys.argv[1:] if argv is None else argv)
    src = a.source
    try:
        if len(src) == 1 and src[0].endswith(".json"):
            shards = load_dump(pathlib.Path(src[0]))
        elif len(src) == 2 and not pathlib.Path(src[0]).expanduser().exists() and "/" in src[0] \
                and not src[0].startswith(("/", ".", "~")):
            shards = load_remote(src[0], src[1])
        else:
            shards = load_local(src)
    except (OSError, ValueError, EOFError) as e:
        print("gguf_audit: %s" % e, file=sys.stderr)
        return 2
    opt = Options(role=a.role, max_gb=a.max_gb, split=tuple(int(x) for x in a.split.split(",")),
                  gpu_usable_gb=a.gpu_usable_gb, reserve_gb=a.reserve_gb, arena_gb=a.arena_gb,
                  contexts=tuple(int(x) for x in a.contexts.split(",")),
                  ram_node_gb=tuple(float(x) for x in a.ram_node_gb.split(",")),
                  mem_available_gb=a.mem_available_gb, headroom_gib=a.headroom_gib, staging_gb=a.staging_gb,
                  slack_slots=a.slack_slots)
    if a.dump_header:
        dump_header(shards, pathlib.Path(a.dump_header), " ".join(src))
    au = Audit(shards, opt)
    au.run()
    name = a.name or default_name(src)
    md = au.markdown("%s (%s)" % (name, " ".join(src)))
    sys.stdout.write(md)
    if not a.no_write:
        for p in write_outputs(au, pathlib.Path(a.out_dir).expanduser(), name, md):
            print("wrote %s" % p, file=sys.stderr)
    return 1 if any(c.status == "FAIL" for c in au.checks) else 0


if __name__ == "__main__":
    sys.exit(main())

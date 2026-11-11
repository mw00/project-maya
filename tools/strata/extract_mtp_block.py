#!/usr/bin/env python3
"""tools/strata/extract_mtp_block.py - cut the NextN (MTP) draft block out of a glm-dsa GGUF into a standalone GGUF
the engine reads through MAYA_GLM_MTP_GGUF (BUILD-PLAN B15.4).

Why: the engine's MTP gain (M4, first-position acceptance >= 0.6) depends on the draft head's precision.  llama.cpp
#28351 measured teacher-forced head-vs-trunk agreement 0.718 with a Q4_0 head vs 0.724 bf16, and a collapse with an
IQ2_XS head.  A UD-IQ2_M file quantizes blk.78 like its trunk (tools/strata/gguf_audit.py check A9 reports the types
and WARNs).  The draft block reads only the trunk's hidden state and the token embedding, so a block from a larger quant
of the SAME model (Q4_K_M, Q5_K_M, Q8_0) drops in: the engine looks it up in MAYA_GLM_MTP_GGUF first.

The copy is byte-exact (no requantization): every blk.<N>.* tensor of the source, in its own type, plus the source's
key/values (minus split.*, tokenizer.* and general.name, which says where the block came from).  Only the block's
bytes are read, so a remote source costs about the block's size (Q4_K_M: about 5.5 GB), not the whole quant.

    # local shards (the first shard: its siblings -0000k-of-0000n are found; or list the shards that hold the block)
    python3 tools/strata/extract_mtp_block.py /data/GLM-5.3-Q4_K_M-00001-of-00009.gguf -o glm53-mtp-q4km.gguf \\
        --check-against /data/GLM-5.3-UD-IQ2_M-00001-of-00006.gguf
    # only the shard(s) holding blk.78 downloaded: the key/values from the model's own first shard
    python3 tools/strata/extract_mtp_block.py GLM-5.3-Q4_K_M-00009-of-00009.gguf --kv-from IQ2_M-00001-of-00006.gguf \\
        -o glm53-mtp-q4km.gguf
    # straight from the Hub (range reads through huggingface_hub: the headers and the block's bytes only)
    python3 tools/strata/extract_mtp_block.py --remote unsloth/GLM-5.3-GGUF 'Q4_K_M/*.gguf' -o glm53-mtp-q4km.gguf
    # list the block's types and the B15.4 verdict, write nothing
    python3 tools/strata/extract_mtp_block.py /data/GLM-5.3-UD-IQ2_M-00001-of-00006.gguf --list

Then: MAYA_GLM_MTP_GGUF=glm53-mtp-q4km.gguf (the engine logs "glm mtp: CUDAn the draft block from ... (instead of
the model's own)"), and measure tools/glm_dsa_mtp_probe.py --agree and the engine's acceptance (MAYA_GLM_SPEC_LOG).
--check-against MODEL refuses a block from another model (arch, block index, embedding, attention, expert geometry).

Exit status: 0 written (or listed), 1 refused (no block, missing tensors, geometry mismatch), 2 usage or read error.
"""
from __future__ import annotations

import argparse
import pathlib
import re
import struct
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import gguf_audit as ga  # noqa: E402

# the engine's load_mtp needs these (shared_head_norm is optional: it falls back to output_norm)
REQUIRED = ga.NEXTN_TENSORS + ga.ATTN_TENSORS + ga.INDEXER_TENSORS + ga.MOE_TENSORS
# key suffixes (after "<arch>.") that must agree with the model the block is for
SAME_GEOMETRY = ["block_count", "nextn_predict_layers", "embedding_length", "attention.head_count",
                 "attention.q_lora_rank", "attention.kv_lora_rank", "attention.key_length_mla",
                 "attention.value_length_mla", "rope.dimension_count", "expert_count", "expert_used_count",
                 "expert_shared_count", "expert_feed_forward_length", "attention.indexer.head_count",
                 "attention.indexer.key_length", "attention.indexer.top_k", "vocab_size"]
DROP_PREFIXES = ("split.", "tokenizer.", "GGUF.")
BLOCK_RE = re.compile(r"^blk\.(\d+)\.nextn\.eh_proj\.weight$")
CHUNK = 64 << 20


class _RawReader(ga._Reader):
    """gguf_audit's sequential reader, which can also record the bytes it takes (a KV record copied verbatim)."""

    def __init__(self, f):
        super().__init__(f)
        self.rec: list[bytes] | None = None

    def take(self, n: int) -> bytes:
        x = super().take(n)
        if self.rec is not None:
            self.rec.append(x)
        return x


class Source:
    """One GGUF shard's header: raw KV records, tensors, the data section's start."""

    def __init__(self, name: str, opener, size: int):
        self.name, self.opener, self.size = name, opener, size
        self.kv: dict = {}
        self.raw_kv: dict[str, bytes] = {}
        self.tensors: list[ga.Tensor] = []
        with opener() as f:
            r = _RawReader(f)
            if r.take(4) != b"GGUF":
                raise ValueError("%s: not a GGUF file" % name)
            self.version = r.u("I")
            nt, nkv = r.u("Q"), r.u("Q")
            for _ in range(nkv):
                r.rec = []
                k = r.string()
                v = ga._value(r, r.u("I"))
                self.raw_kv[k] = b"".join(r.rec)
                r.rec = None
                self.kv[k] = v
            for _ in range(nt):
                tn = r.string()
                nd = r.u("I")
                dims = [r.u("Q") for _ in range(nd)]
                self.tensors.append(ga.Tensor(tn, r.u("I"), dims, r.u("Q"), 0))
            self.align = int(self.kv.get("general.alignment", 32))
            self.data_start = -(-r.consumed // self.align) * self.align


def _local(path: pathlib.Path) -> Source:
    return Source(path.name, lambda: open(path, "rb"), path.stat().st_size)


def _remote(repo: str, pattern: str) -> list[Source]:
    try:
        from huggingface_hub import HfFileSystem
    except ImportError as e:   # pragma: no cover - needs network
        raise SystemExit("--remote needs huggingface_hub (python3 -m pip install -U 'huggingface_hub>=0.34')") from e
    import fnmatch
    fs = HfFileSystem()
    files = sorted(x for x in fs.find(repo) if x.endswith(".gguf") and
                   fnmatch.fnmatch(x.split(repo + "/", 1)[1], pattern))
    if not files:
        raise FileNotFoundError("no files match %s in %s" % (pattern, repo))
    return [Source(p.split(repo + "/", 1)[1], (lambda p=p: fs.open(p, "rb", block_size=8 << 20)),
                   int(fs.info(p)["size"])) for p in files]


def find_block(sources: list[Source], want: int | None) -> int:
    found = sorted({int(m.group(1)) for s in sources for t in s.tensors if (m := BLOCK_RE.match(t.name))})
    if want is not None:
        if want not in found:
            raise LookupError("no blk.%d.nextn.eh_proj.weight in these files (blocks with one: %s)" % (want, found or "none"))
        return want
    if len(found) != 1:
        raise LookupError("%s draft block (blk.N.nextn.*) in these files%s" % (
            "no" if not found else "more than one", "" if not found else ": %s (pick one with --block)" % found))
    return found[0]


def block_tensors(sources: list[Source], block: int) -> list[tuple[Source, ga.Tensor]]:
    p = "blk.%d." % block
    return [(s, t) for s in sources for t in s.tensors if t.name.startswith(p)]


def model_kv(sources: list[Source], kv_from: Source | None) -> tuple[Source, str]:
    if kv_from is not None:
        return kv_from, kv_from.kv.get("general.architecture", "")
    for s in sources:
        if "general.architecture" in s.kv:
            return s, s.kv["general.architecture"]
    raise LookupError("no shard of the source carries the model's key/values (general.architecture): pass the first "
                      "shard too, or --kv-from <the model's first shard>")


def geometry_diff(a: dict, arch_a: str, b: dict, arch_b: str) -> list[str]:
    out = []
    if arch_a != arch_b:
        out.append("general.architecture %r vs %r" % (arch_a, arch_b))
    for suf in SAME_GEOMETRY:
        va, vb = a.get(arch_a + "." + suf), b.get(arch_b + "." + suf)
        if va != vb:
            out.append("%s.%s %r vs %r" % (arch_a, suf, va, vb))
    return out


def _str_record(key: str, val: str) -> bytes:
    kb, vb = key.encode(), val.encode()
    return struct.pack("<Q", len(kb)) + kb + struct.pack("<I", 8) + struct.pack("<Q", len(vb)) + vb


def write_block(out: pathlib.Path, kv_src: Source, arch: str, items: list[tuple[Source, ga.Tensor]], block: int,
                origin: str) -> int:
    align = kv_src.align
    recs = [raw for k, raw in kv_src.raw_kv.items()
            if not k.startswith(DROP_PREFIXES) and k != "general.name"]
    name = kv_src.kv.get("general.name", arch)
    recs.append(_str_record("general.name", "%s NextN (MTP) block %d from %s" % (name, block, origin)))
    infos, off = [], 0
    for _, t in items:
        nb = t.nbytes
        nm = t.name.encode()
        infos.append(struct.pack("<Q", len(nm)) + nm + struct.pack("<I", len(t.dims)) +
                     b"".join(struct.pack("<Q", d) for d in t.dims) + struct.pack("<IQ", t.type, off))
        off += -(-nb // align) * align
    head = b"GGUF" + struct.pack("<IQQ", 3, len(items), len(recs)) + b"".join(recs) + b"".join(infos)
    head += b"\0" * ((-len(head)) % align)
    total = 0
    tmp = out.with_name(out.name + ".part")
    with open(tmp, "wb") as w:
        w.write(head)
        for src, t in items:
            nb = t.nbytes
            with src.opener() as f:
                f.seek(src.data_start + t.offset)
                left = nb
                while left > 0:
                    c = f.read(min(CHUNK, left))
                    if not c:
                        raise EOFError("%s: %s ends early" % (src.name, t.name))
                    w.write(c)
                    left -= len(c)
            w.write(b"\0" * ((-nb) % align))
            total += nb
    tmp.replace(out)
    return total


def report(items: list[tuple[Source, ga.Tensor]], block: int) -> tuple[list[str], list[tuple[str, str]]]:
    by = {t.name: t for _, t in items}
    return ga.mtp_type_report(lambda nm: by.get("blk.%d.%s" % (block, nm)), block)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("sources", nargs="*", help="GGUF shards (the first shard finds its siblings), or a directory")
    ap.add_argument("--remote", nargs=2, metavar=("REPO", "PATTERN"), help="Hub repo and file pattern (range reads)")
    ap.add_argument("-o", "--out", help="the standalone GGUF to write")
    ap.add_argument("--block", type=int, default=None, help="the block index (default: the one with nextn.eh_proj)")
    ap.add_argument("--kv-from", help="read the key/values from this GGUF (the model's first shard) instead")
    ap.add_argument("--check-against", metavar="MODEL", help="refuse unless MODEL (its first shard) has this geometry")
    ap.add_argument("--list", action="store_true", help="print the block's tensors, types and the B15.4 verdict only")
    a = ap.parse_args(argv)
    if bool(a.sources) == bool(a.remote):
        ap.error("give local GGUF sources or --remote REPO PATTERN")
    if not a.list and not a.out:
        ap.error("-o/--out is required (or --list)")
    try:
        sources = _remote(*a.remote) if a.remote else [
            _local(p) for p in ga.local_shards(a.sources) if p.exists() or len(a.sources) > 1]   # siblings: those present
        if not sources:
            raise FileNotFoundError("no GGUF files in %s" % " ".join(a.sources))
        kv_from = _local(pathlib.Path(a.kv_from)) if a.kv_from else None
        block = find_block(sources, a.block)
        items = block_tensors(sources, block)
        kv_src, arch = model_kv(sources, kv_from)
    except (OSError, ValueError, EOFError) as e:
        print("extract_mtp_block: %s" % e, file=sys.stderr)
        return 2
    except LookupError as e:
        print("extract_mtp_block: %s" % e, file=sys.stderr)
        return 1
    names = {t.name for _, t in items}
    missing = [s for s in REQUIRED if "blk.%d.%s" % (block, s) not in names]
    lines, low = report(items, block)
    size = sum(t.nbytes for _, t in items)
    print("blk.%d: %d tensors, %.2f GB, from %s" % (block, len(items), size / 1e9,
                                                   ", ".join(sorted({s.name for s, _ in items}))))
    for ln in lines:
        print("  " + ln)
    if low:
        print("WARN: %d tensors below %.0f bits a weight (%s): a weak draft head (B15.4); use a Q4_K-or-better block" % (
            len(low), ga.MTP_MIN_BPW, ", ".join(sorted({t for _, t in low}))))
    else:
        print("OK: every projection and expert of the block is at least %.0f bits a weight" % ga.MTP_MIN_BPW)
    if missing:
        print("extract_mtp_block: blk.%d lacks %s (the engine's load_mtp needs them)" % (block, ", ".join(missing)),
              file=sys.stderr)
        return 1
    if a.check_against:
        try:
            model = ga.load_local([a.check_against])
        except (OSError, ValueError, EOFError) as e:
            print("extract_mtp_block: --check-against: %s" % e, file=sys.stderr)
            return 2
        mkv: dict = {}
        for s in model:
            for k, v in s.kv.items():
                mkv.setdefault(k, v)
        diff = geometry_diff(kv_src.kv, arch, mkv, mkv.get("general.architecture", ""))
        if diff:
            print("extract_mtp_block: the block is not for %s: %s" % (a.check_against, "; ".join(diff)),
                  file=sys.stderr)
            return 1
        print("geometry matches %s" % a.check_against)
    if a.list:
        return 0
    origin = ("%s %s" % tuple(a.remote)) if a.remote else sources[0].name
    try:
        n = write_block(pathlib.Path(a.out), kv_src, arch, items, block, origin)
    except (OSError, EOFError) as e:
        print("extract_mtp_block: %s" % e, file=sys.stderr)
        return 2
    print("wrote %s: blk.%d, %d tensors, %.2f GB of tensor data; use MAYA_GLM_MTP_GGUF=%s" % (
        a.out, block, len(items), n / 1e9, a.out))
    return 0


if __name__ == "__main__":
    sys.exit(main())

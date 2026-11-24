"""tools/ds4/reorder.py - rewrite DeepSeek-V4.1 EXL3 shards so every expert's pieces lie back to back.

    python tools/ds4/reorder.py --model ~/models/DSV4
    python tools/ds4/reorder.py model-00003-of-00048.safetensors [more shards...]

Some EXL3 quants (Diffbot's 3 bpw, among them) write each shard's tensors grouped by type: every expert's .trellis,
then every .suh, every .svh, and so on.  The engine reads a layer's experts as one region an expert, taking expert
0's piece offsets as the layout of every other (src/ds4/ds4_model.cu's expert_layout), so pack.py stops such a model
with "its pieces are not together in their file".  This rewrites those shards with the same tensors in a new order:
non-expert tensors first (their order kept), then expert by expert, each expert's w1 / w2 / w3 pieces back to back in
the order LibertAIDAI's packs have - suh, svh, mul1 (or mcg, the drafter's codebook), trellis.  Only the order and
each tensor's data_offsets change: every tensor's bytes, dtype and shape stay exactly the same.  A shard whose
experts already lie back to back is left alone.

Each shard is rewritten in place - a temporary file beside it (free space for one shard), every tensor's SHA-256
checked against the bytes that were read, then an atomic rename.  --model reads the folder's
model.safetensors.index.json (and any other .safetensors beside it) and rewrites the shards that hold type-ordered
experts; --manifests DIR writes one <shard>.jsonl.gz per rewritten shard, each tensor's old and new offset and its
SHA-256.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import pathlib
import re
import struct
import sys
import time

GROUP = re.compile(r"^(?P<pre>.*\.ffn\.experts\.)(?P<e>\d+)\.(?P<rest>.+)$")
PARTS = ("w1", "w2", "w3")
PIECES = ("suh", "svh", "mul1", "mcg", "trellis")
SLACK = 1 << 20                            # (the engine's: pieces within 2 x their bytes + 1 MiB are one region)
BUF = 32 << 20


def plan(keys: list[str]) -> list[str]:
    """The new tensor order: non-experts (their order kept), then each expert's pieces back to back."""
    groups: dict[str, dict[int, list[str]]] = {}
    for k in keys:
        m = GROUP.match(k)
        if m:
            groups.setdefault(m.group("pre"), {}).setdefault(int(m.group("e")), []).append(k)
    new = [k for k in keys if not GROUP.match(k)]
    for pre in sorted(groups):
        for e in sorted(groups[pre]):
            have = groups[pre][e]
            hs = set(have)
            ordered = []
            for r in PARTS:
                for q in PIECES:
                    n = "%s%d.%s.%s" % (pre, e, r, q)
                    if n in hs:
                        ordered.append(n)
            ordered += [k for k in have if k not in set(ordered)]   # anything else this expert has
            new.extend(ordered)
    assert len(new) == len(keys) and len(set(new)) == len(new) and set(new) == set(keys)
    return new


def apart(ent: dict) -> str | None:
    """The first expert whose trellis / suh / svh pieces lie farther apart than the engine reads as one region
    (hi - lo > 2 x their bytes + 1 MiB, expert_layout's own test), None when every one is back to back.  An expert
    with a piece missing (or cut across files) is not judged here - the engine names it at its start."""
    groups: dict[str, dict[int, dict[str, list[int]]]] = {}
    for k, v in ent.items():
        m = GROUP.match(k)
        if m:
            groups.setdefault(m.group("pre"), {}).setdefault(int(m.group("e")), {})[m.group("rest")] = v["data_offsets"]
    for pre in sorted(groups):
        for e in sorted(groups[pre]):
            pcs = groups[pre][e]
            spans = [pcs.get("%s.%s" % (r, q)) for r in PARTS for q in ("trellis", "suh", "svh")]
            if any(s is None for s in spans):
                continue
            nb = sum(b - a for a, b in spans)
            lo = min(a for a, b in spans)
            hi = max(b for a, b in spans)
            if hi - lo > 2 * nb + SLACK:
                return "%s%d" % (pre, e)
    return None


def copy_hashed(src, dst, base: int, a: int, nb: int, mv, hh):
    src.seek(base + a)
    rem = nb
    while rem:
        got = src.readinto(mv[:min(rem, len(mv))])
        if not got:
            raise RuntimeError("short read at %d (%d left)" % (base + a, rem))
        dst.write(mv[:got])
        hh.update(mv[:got])
        rem -= got


def rewrite(path: pathlib.Path, man_dir: pathlib.Path | None = None) -> bool:
    """Rewrite one shard in expert order if it needs it.  -> True rewritten, False left alone."""
    if not path.is_file():
        raise SystemExit("%s: no such file" % path)
    path = path.resolve()                  # (a model folder of symlinks: rewrite what they point at, not the links)
    t0 = time.time()
    with path.open("rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(n))
    base, size = 8 + n, path.stat().st_size
    meta = h.get("__metadata__")
    ent = {k: v for k, v in h.items() if k != "__metadata__"}
    if not any(GROUP.match(k) for k in ent):
        print("%s: no experts, left alone" % path.name, flush=True)
        return False
    bad = apart(ent)
    if bad is None:
        print("%s: experts already back to back, left alone" % path.name, flush=True)
        return False
    order = plan(list(ent))
    off, nh = 0, {}
    for k in order:
        a, b = ent[k]["data_offsets"]
        assert 0 <= a <= b, k
        nh[k] = {"dtype": ent[k]["dtype"], "shape": ent[k]["shape"], "data_offsets": [off, off + (b - a)]}
        off += b - a
    assert base + off <= size, ("data past the end of %s" % path, base + off, size)
    trailing = size - base - off
    out = ({"__metadata__": meta} if meta is not None else {})
    out.update(nh)
    hb = json.dumps(out, separators=(",", ":")).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    tmp = path.with_name(path.name + ".reorder-tmp")
    sha, mv = {}, memoryview(bytearray(BUF))
    with path.open("rb") as src, tmp.open("wb") as dst:
        dst.write(struct.pack("<Q", len(hb)))
        dst.write(hb)
        for k in order:
            a, b = ent[k]["data_offsets"]
            hh = hashlib.sha256()
            copy_hashed(src, dst, base, a, b - a, mv, hh)
            sha[k] = hh.hexdigest()
        dst.flush()
        os.fsync(dst.fileno())
    # verify what is on disk: header, offsets, and every tensor's bytes at its new offset
    tcopy = time.time() - t0
    assert tmp.stat().st_size == 8 + len(hb) + off, "rewritten file has the wrong size"
    with tmp.open("rb") as f:
        n2 = struct.unpack("<Q", f.read(8))[0]
        h2 = json.loads(f.read(n2))
        assert n2 == len(hb)
        pos = 0
        for k in order:
            v = h2.get(k)
            assert v is not None and v["dtype"] == nh[k]["dtype"] and list(v["shape"]) == list(nh[k]["shape"]), k
            a2, b2 = v["data_offsets"]
            assert a2 == pos and b2 - a2 == nh[k]["data_offsets"][1] - nh[k]["data_offsets"][0], (k, a2, pos)
            pos = b2
        assert pos == off and set(h2) - {"__metadata__"} == set(order)
        if meta is not None:
            assert h2.get("__metadata__") == meta
        for k in order:
            a2, b2 = h2[k]["data_offsets"]
            f.seek(8 + n2 + a2)
            hh = hashlib.sha256()
            rem = b2 - a2
            while rem:
                got = f.readinto(mv[:min(rem, len(mv))])
                if not got:
                    raise RuntimeError("short verify read")
                hh.update(mv[:got])
                rem -= got
            if hh.hexdigest() != sha[k]:
                raise RuntimeError("%s: %s differs after the rewrite" % (path, k))
    if apart({k: h2[k] for k in order}) is not None:
        raise RuntimeError("%s: an expert is still not back to back after the rewrite (its pieces in more than "
                           "one file?)" % path)
    os.replace(tmp, path)
    # audit manifest: every tensor's old -> new offset and its sha256
    if man_dir is not None:
        man_dir.mkdir(parents=True, exist_ok=True)
        with gzip.open(man_dir / (path.name + ".jsonl.gz"), "wt") as f:
            for k in order:
                a, _ = ent[k]["data_offsets"]
                a2, b2 = nh[k]["data_offsets"]
                f.write(json.dumps({"name": k, "dtype": nh[k]["dtype"], "shape": nh[k]["shape"],
                                    "old_off": a, "new_off": a2, "bytes": b2 - a2,
                                    "sha256": sha[k]}) + "\n")
    n_exp = len({(m.group("pre"), m.group("e")) for m in map(GROUP.match, order) if m})
    print("%s: %d tensors, %d experts, %.2f GB rewritten and verified in %.1fs (verify %.1fs)%s" %
          (path.name, len(order), n_exp, (8 + len(hb) + off) / 1e9, time.time() - t0,
           time.time() - t0 - tcopy, " [%d trailing bytes dropped]" % trailing if trailing else ""), flush=True)
    return True


def model_shards(model: pathlib.Path) -> list[pathlib.Path]:
    """A model folder's shards: model.safetensors.index.json's, then any other .safetensors beside them."""
    ix = model / "model.safetensors.index.json"
    if not ix.is_file():
        raise SystemExit("%s: no model.safetensors.index.json" % model)
    names = list(dict.fromkeys(json.loads(ix.read_text())["weight_map"].values()))
    names += sorted(p.name for p in model.glob("*.safetensors") if p.name not in names)
    return [model / n for n in names]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("shards", nargs="*", help="safetensors files to rewrite in place")
    ap.add_argument("--model", help="a model folder: rewrite the shards that hold type-ordered experts")
    ap.add_argument("--manifests", help="write <shard>.jsonl.gz there: each tensor's old / new offset and sha256")
    a = ap.parse_args(argv)
    if a.model and a.shards:
        ap.error("name shards or --model, not both")
    paths = ([pathlib.Path(p) for p in a.shards] if a.shards
             else model_shards(pathlib.Path(a.model).expanduser()) if a.model else [])
    if not paths:
        ap.error("name shards or --model")
    man_dir = pathlib.Path(a.manifests).expanduser() if a.manifests else None
    n = 0
    for p in paths:
        n += rewrite(p, man_dir)
    print("%d shard%s rewritten" % (n, "" if n == 1 else "s"))
    return 0


if __name__ == "__main__":
    sys.exit(main())

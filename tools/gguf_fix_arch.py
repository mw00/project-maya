#!/usr/bin/env python3
"""tools/gguf_fix_arch.py - a GLM-5.3-Flash GGUF's architecture as llama.cpp names it.

Maya's quants copied "glm5next" (unsloth's early spelling) from the GGUF they were made from; llama.cpp calls the
architecture "glm5-next" and does not load the other spelling.  This renames it - `general.architecture` and every
"glm5next." key - and keeps the header the SAME SIZE: the data section starts at the same byte, so every tensor stays
where it was, and so do the absolute offsets a Maya pack (pack/native_experts.txt) reads.  The bytes the longer name
adds come out of the alignment padding before the data and the `maya.recipe` JSON's spaces (and, when those are one
or two short, the recipe note's "; " written ";").  Nothing after the header is touched.  Only the first shard of a split model has these keys; the others need nothing.

    python tools/gguf_fix_arch.py <the first .gguf> --check           what it would change (nothing is written)
    python tools/gguf_fix_arch.py <the first .gguf> --out <new.gguf>  a renamed copy
    python tools/gguf_fix_arch.py <the first .gguf> --in-place        the header rewritten in place (the data is not)
"""
from __future__ import annotations

import argparse
import json
import shutil
import struct
import sys
from pathlib import Path

OLD, NEW = "glm5next", "glm5-next"
MAGIC = b"GGUF"
# GGUF value types: their fixed sizes; 8 = string, 9 = array
FIXED = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
STRING, ARRAY, UINT32 = 8, 9, 4


class Header:
    """The parsed header: the KV entries as (key, type, value bytes), the tensor infos' bytes as they are."""

    def __init__(self, version, n_tensors, kvs, tensor_bytes, end, alignment):
        self.version, self.n_tensors, self.kvs = version, n_tensors, kvs
        self.tensor_bytes, self.end, self.alignment = tensor_bytes, end, alignment

    @property
    def data_start(self) -> int:
        return -(-self.end // self.alignment) * self.alignment

    def get(self, key):
        for k, t, raw in self.kvs:
            if k == key:
                return t, raw
        return None


def _string(buf: bytes, at: int) -> tuple[str, int]:
    (n,) = struct.unpack_from("<Q", buf, at)
    at += 8
    return buf[at:at + n].decode("utf-8"), at + n


def _value_end(buf: bytes, at: int, t: int) -> int:
    if t in FIXED:
        return at + FIXED[t]
    if t == STRING:
        (n,) = struct.unpack_from("<Q", buf, at)
        return at + 8 + n
    if t == ARRAY:
        et, n = struct.unpack_from("<IQ", buf, at)
        at += 12
        if et in FIXED:
            return at + n * FIXED[et]
        for _ in range(n):
            at = _value_end(buf, at, et)
        return at
    raise ValueError(f"unknown GGUF value type {t}")


def read_header(path: Path) -> Header:
    """The header of `path` (read in growing pieces until it parses: a GLM tokenizer makes it a few MB)."""
    size = path.stat().st_size
    want = 16 << 20
    while True:
        with open(path, "rb") as f:
            buf = f.read(min(want, size))
        try:
            return _parse(buf)
        except (struct.error, IndexError, UnicodeDecodeError):
            if want >= size:
                raise ValueError(f"{path}: not a complete GGUF header")
            want *= 4


def _parse(buf: bytes) -> Header:
    if buf[:4] != MAGIC:
        raise ValueError("not a GGUF file")
    version, n_tensors, n_kv = struct.unpack_from("<IQQ", buf, 4)
    if version < 2:
        raise ValueError(f"GGUF version {version} is not supported")
    at, kvs, alignment = 24, [], 32
    for _ in range(n_kv):
        key, at = _string(buf, at)
        (t,) = struct.unpack_from("<I", buf, at)
        at += 4
        end = _value_end(buf, at, t)
        if end > len(buf):
            raise IndexError
        raw = buf[at:end]
        if key == "general.alignment" and t == UINT32:
            (alignment,) = struct.unpack("<I", raw)
        kvs.append((key, t, raw))
        at = end
    t0 = at
    for _ in range(n_tensors):
        _, at = _string(buf, at)
        (nd,) = struct.unpack_from("<I", buf, at)
        at += 4 + 8 * nd + 4 + 8
    if at > len(buf):
        raise IndexError
    return Header(version, n_tensors, kvs, buf[t0:at], at, alignment)


def _str_raw(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def _str_of(raw: bytes) -> str:
    return _string(raw, 0)[0]


def note_tight(note: str) -> str:
    """The recipe note's "; " as ";" - the only text the rename may change, and only when the recipe's JSON spaces were
    not enough (Maya-S and Maya-M: one byte short)."""
    return note.replace("; ", ";")


def _same_recipe(a: str, b: str) -> bool:
    ja, jb = json.loads(a), json.loads(b)
    if isinstance(ja.get("note"), str) and isinstance(jb.get("note"), str):
        ja["note"], jb["note"] = note_tight(ja["note"]), note_tight(jb["note"])
    return ja == jb


def _build(h: Header, kvs) -> bytes:
    out = [MAGIC, struct.pack("<IQQ", h.version, h.n_tensors, len(kvs))]
    for k, t, raw in kvs:
        out += [_str_raw(k), struct.pack("<I", t), raw]
    out.append(h.tensor_bytes)
    return b"".join(out)


def renamed(h: Header) -> bytes | None:
    """The new header, padded to exactly the old data start (the bytes up to the first tensor), or None when the file
    already says "glm5-next".  Raises ValueError when it is not a "glm5next" file or the rename cannot keep the size."""
    arch = h.get("general.architecture")
    if arch is None or arch[0] != STRING:
        raise ValueError("no general.architecture")
    name = _str_of(arch[1])
    if name == NEW:
        return None
    if name != OLD:
        raise ValueError(f"general.architecture is {name!r}, not {OLD!r}")
    kvs = []
    for k, t, raw in h.kvs:
        if k == "general.architecture":
            raw = _str_raw(NEW)
        elif k.startswith(OLD + "."):
            k = NEW + k[len(OLD):]
        kvs.append((k, t, raw))
    target = h.data_start
    body = _build(h, kvs)
    i = next((n for n, (k, t, _) in enumerate(kvs) if k == "maya.recipe" and t == STRING), None)
    if len(body) > target:                               # take the bytes from the recipe's JSON whitespace
        if i is None:
            raise ValueError(f"the renamed header is {len(body) - target} bytes too long and has no maya.recipe")
        recipe = json.loads(_str_of(kvs[i][2]))
        for step in (0, 1):
            if step == 1 and isinstance(recipe.get("note"), str):
                recipe["note"] = note_tight(recipe["note"])  # the last resort: the note's "; " as ";"
            compact = json.dumps(recipe, separators=(",", ":"), ensure_ascii=False)
            kvs[i] = (kvs[i][0], STRING, _str_raw(compact))
            body = _build(h, kvs)
            if len(body) <= target:
                break
        if len(body) > target:
            raise ValueError(f"the renamed header is still {len(body) - target} bytes too long")
    if i is not None:                                    # the recipe takes up the rest: the header ends at the data
        text = _str_of(kvs[i][2]) + " " * (target - len(body))
        kvs[i] = (kvs[i][0], STRING, _str_raw(text))
        body = _build(h, kvs)
    if not (target - h.alignment < len(body) <= target):
        raise ValueError("the renamed header does not end where the data starts")
    return body + b"\0" * (target - len(body))


def verify(path: Path, old: Header) -> None:
    """The renamed file's header: the same data start and tensor infos, every key there under its new name with its
    value (the recipe's JSON the same once parsed)."""
    new = read_header(path)
    assert new.data_start == old.data_start, "the data start moved"
    assert new.tensor_bytes == old.tensor_bytes, "the tensor infos changed"
    assert new.n_tensors == old.n_tensors and len(new.kvs) == len(old.kvs), "the counts changed"
    assert _str_of(new.get("general.architecture")[1]) == NEW, "the architecture is not renamed"
    for (k0, t0, r0), (k1, t1, r1) in zip(old.kvs, new.kvs):
        assert k1 == (NEW + k0[len(OLD):] if k0.startswith(OLD + ".") else k0) and t0 == t1, f"{k0}: renamed wrong"
        if k0 == "maya.recipe":
            assert _same_recipe(_str_of(r0), _str_of(r1)), "maya.recipe changed"
        elif k0 != "general.architecture":
            assert r0 == r1, f"{k0}: its value changed"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("gguf", type=Path, help="the model's first (or only) .gguf file")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--check", action="store_true", help="say what would change; write nothing")
    g.add_argument("--out", type=Path, help="write a renamed copy here")
    g.add_argument("--in-place", action="store_true", help="rewrite the header of this file (the data is not touched)")
    a = ap.parse_args()
    h = read_header(a.gguf)
    try:
        new = renamed(h)
    except ValueError as e:
        print(f"{a.gguf.name}: {e}", file=sys.stderr)
        return 1
    if new is None:
        print(f"{a.gguf.name}: already {NEW!r} - nothing to do")
        return 0
    n_keys = sum(1 for k, _, _ in h.kvs if k.startswith(OLD + "."))
    print(f"{a.gguf.name}: {OLD!r} -> {NEW!r}, {n_keys} keys renamed; the data still starts at byte {h.data_start}")
    if a.check:
        return 0
    if a.out:
        with open(a.gguf, "rb") as src, open(a.out, "wb") as dst:
            dst.write(new)
            src.seek(h.data_start)
            shutil.copyfileobj(src, dst, 64 << 20)
        dst_path = a.out
    else:
        with open(a.gguf, "r+b") as f:
            f.write(new)
        dst_path = a.gguf
    verify(dst_path, h)
    print(f"{dst_path.name}: renamed and checked")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""tools/hy4/mtp_gguf.py - the MTP draft block of a Hy4-flash GGUF as a file of its own.

    python tools/hy4/mtp_gguf.py --src Hy4-flash-IQ2_AH-00001-of-00002.gguf --out Hy4-Flash-mtp.gguf

A Hy4-flash set made from Tencent's own GGUF builds carries the model's MTP block (hy_v4_flash.nextn_predict_layers 1:
the block after the trunk's last, `blk.<n>.*`, with its nextn.* norms and projection), the block maya-hy4 drafts with
(--mtp).  A set made from the public release has none.  This writes that block alone - its tensors byte for byte (types,
dims and data as in the source; nothing rounded), the source's hy_v4_flash.* keys as they are (the lists of a whole
vocabulary left out), general.type "mtp" - so a set without the block drafts with it: the pack (tools/hy4/pack_gguf.py)
takes the file from the model's folder.  Prints the file's size and sha256 and each tensor's.
"""
from __future__ import annotations

import argparse
import hashlib
import pathlib
import struct
import sys

ARCH = "hy_v4_flash"
ALIGN = 32
SCALAR = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34)}   # ggml type -> (values, bytes) a block: F32, F16, BF16, Q8_0
LONG_LIST = 4096   # (a list of more values than this is a vocabulary's: not the block's)


def _str(f) -> bytes:
    raw = f.read(8)
    return raw + f.read(struct.unpack("<Q", raw)[0])


def _value(f, t: int) -> tuple:
    """(the value's bytes as the file has them, its count: 1, or a list's length)"""
    if t in SCALAR:
        return f.read(SCALAR[t]), 1
    if t == 8:
        return _str(f), 1
    if t != 9:
        raise SystemExit("a metadata value of GGUF type %d" % t)
    head = f.read(12)
    at, n = struct.unpack("<IQ", head)
    if at in SCALAR:
        return head + f.read(SCALAR[at] * n), n
    if at != 8:
        raise SystemExit("a metadata list of GGUF type %d" % at)
    return head + b"".join(_str(f) for _ in range(n)), n


def read(path: pathlib.Path) -> tuple:
    """(kv: {key: (type, raw bytes, count)}, tensors: [(name, dims, type, absolute offset)], sorted by offset)"""
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise SystemExit("%s: not a GGUF file" % path)
        _, nt, nkv = struct.unpack("<IQQ", f.read(20))
        kv = {}
        for _ in range(nkv):
            k = _str(f)[8:].decode("utf-8")
            t = struct.unpack("<I", f.read(4))[0]
            raw, n = _value(f, t)
            kv[k] = (t, raw, n)
        tensors = []
        for _ in range(nt):
            name = _str(f)[8:].decode("utf-8")
            nd = struct.unpack("<I", f.read(4))[0]
            dims = list(struct.unpack("<%dQ" % nd, f.read(8 * nd)))
            typ, off = struct.unpack("<IQ", f.read(12))
            tensors.append((name, dims, typ, off))
        align = ALIGN
        if "general.alignment" in kv:
            align = struct.unpack("<I", kv["general.alignment"][1])[0]
        start = (f.tell() + align - 1) // align * align
    return kv, sorted(((n, d, t, start + o) for n, d, t, o in tensors), key=lambda x: x[3])


def _int(kv: dict, key: str, default=None):
    if key not in kv:
        return default
    t, raw, _ = kv[key]
    return struct.unpack("<" + {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 10: "Q", 11: "q"}[t], raw)[0]


def _s(text: str) -> bytes:
    b = text.encode("utf-8")
    return struct.pack("<Q", len(b)) + b


def extract(src: pathlib.Path, out: pathlib.Path, note: str) -> dict:
    kv, tensors = read(src)
    size = src.stat().st_size
    if kv.get("general.architecture", (0, b"", 0))[1][8:] != ARCH.encode():
        raise SystemExit("%s: not a %s GGUF" % (src, ARCH))
    n_all, mtp = _int(kv, ARCH + ".block_count"), _int(kv, ARCH + ".nextn_predict_layers", 0)
    if mtp != 1:
        raise SystemExit("%s: %d MTP blocks (hy_v4_flash.nextn_predict_layers), one expected" % (src, mtp))
    prefix = "blk.%d." % (n_all - mtp)
    ends = [t[3] for t in tensors[1:]] + [size]
    mine = []
    for (n, d, t, o), e in zip(tensors, ends):
        if not n.startswith(prefix):
            continue
        nb = e - o   # (a type not in BLOCK: the bytes up to the next tensor)
        if t in BLOCK:
            nel = 1
            for x in d:
                nel *= x
            nb = nel // BLOCK[t][0] * BLOCK[t][1]
            if nel % BLOCK[t][0] or nb > e - o:
                raise SystemExit("%s: %s is not whole in the file" % (src, n))
        mine.append((n, d, t, o, nb))
    if not any(n == prefix + "nextn.eh_proj.weight" for n, *_ in mine):
        raise SystemExit("%s: no %snextn.eh_proj.weight (the block is in another shard?)" % (src, prefix))
    keys = [(k, v) for k, v in kv.items() if k.startswith(ARCH + ".") and v[2] <= LONG_LIST]
    meta = [("general.architecture", 8, _s(ARCH)), ("general.type", 8, _s("mtp")),
            ("general.name", 8, _s("Hy4-flash MTP draft block")), ("general.description", 8, _s(note)),
            ("general.alignment", 4, struct.pack("<I", ALIGN))]
    meta += [(k, t, raw) for k, (t, raw, _) in keys]
    head = b"GGUF" + struct.pack("<IQQ", 3, len(mine), len(meta))
    for k, t, raw in meta:
        head += _s(k) + struct.pack("<I", t) + raw
    off, infos = 0, []
    for name, dims, typ, _, nb in mine:
        head += _s(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims) + \
            struct.pack("<IQ", typ, off)
        infos.append(off)
        off = (off + nb + ALIGN - 1) // ALIGN * ALIGN
    head += b"\0" * (-len(head) % ALIGN)
    whole, rows = hashlib.sha256(), []
    with src.open("rb") as fi, out.open("wb") as fo:
        fo.write(head)
        whole.update(head)
        for (name, dims, typ, o, nb), rel in zip(mine, infos):
            pad = len(head) + rel - fo.tell()
            fo.write(b"\0" * pad)
            whole.update(b"\0" * pad)
            fi.seek(o)
            one, left = hashlib.sha256(), nb
            while left:
                buf = fi.read(min(left, 1 << 24))
                if not buf:
                    raise SystemExit("%s: ends inside %s" % (src, name))
                fo.write(buf)
                whole.update(buf)
                one.update(buf)
                left -= len(buf)
            rows.append((name, typ, dims, nb, one.hexdigest()))
        pad = -fo.tell() % ALIGN
        fo.write(b"\0" * pad)
        whole.update(b"\0" * pad)
    return {"bytes": out.stat().st_size, "sha256": whole.hexdigest(), "tensors": rows, "keys": [k for k, _ in keys]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="the shard of a Hy4-flash GGUF set that holds the MTP block")
    ap.add_argument("--out", required=True, help="the file to write")
    ap.add_argument("--note", default="The MTP draft block of Tencent's Hy4-flash as in AngelSlim/Hy4-flash-GGUF: its "
                    "tensors unchanged, in a file of their own. Tencent's licence applies.")
    a = ap.parse_args()
    r = extract(pathlib.Path(a.src), pathlib.Path(a.out), a.note)
    for name, typ, dims, nb, h in r["tensors"]:
        print("%s type %d %s %d bytes %s" % (name, typ, "x".join(map(str, dims)), nb, h))
    print("%d metadata keys of the source, %d tensors" % (len(r["keys"]), len(r["tensors"])))
    print("%s: %d bytes, sha256 %s" % (a.out, r["bytes"], r["sha256"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())

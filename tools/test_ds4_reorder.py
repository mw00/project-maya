"""tools/ds4/reorder.py: a type-ordered shard comes out in expert order with the same tensors and bytes, a shard
whose experts already lie back to back is left alone, and --model finds the shards that need it.

    python -m unittest tools.test_ds4_reorder
"""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import tempfile
import types
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "ds4"))
import pack  # noqa: E402
import reorder  # noqa: E402

UNIT = 4096                                # (bytes a size unit; the engine's apart test has a 1 MiB slack)
SIZES = {"trellis": 100, "suh": 10, "svh": 10, "mul1": 10, "mcg": 10}
MAIN = "layers.0.ffn.experts."
MTP = "mtp.0.ffn.experts."


def body(name: str, nb: int) -> bytes:
    """A tensor's bytes, unique to its name."""
    raw = (name + "|").encode()
    return (raw * (nb // len(raw) + 1))[:nb]


def tensors(groups, by_type: bool) -> dict[str, bytes]:
    """Every expert piece of each (prefix, experts, codebook) group, in the file's order: 'by type' = every trellis
    of every expert, then every suh, ... (Diffbot's 3 bpw); else expert by expert, the order reorder.py writes."""
    seq = []
    for pre, n, code in groups:
        if by_type:
            seq += [(pre, e, r, q) for q in ("trellis", "suh", "svh", code)
                    for r in ("w1", "w2", "w3") for e in range(n)]
        else:
            seq += [(pre, e, r, q) for e in range(n) for r in ("w1", "w2", "w3")
                    for q in ("suh", "svh", code, "trellis")]
    return {"%s%d.%s.%s" % (pre, e, r, q): body("%s%d.%s.%s" % (pre, e, r, q), UNIT * SIZES[q])
            for pre, e, r, q in seq}


def write_shard(path: pathlib.Path, ts: dict[str, bytes], metadata: dict | None = None):
    h = {"__metadata__": metadata} if metadata is not None else {}
    off = 0
    for name, b in ts.items():
        h[name] = {"dtype": "I16", "shape": [320, 144, 48], "data_offsets": [off, off + len(b)]}
        off += len(b)
    hb = json.dumps(h).encode("utf-8")
    hb += b" " * ((8 - len(hb) % 8) % 8)
    path.write_bytes(struct.pack("<Q", len(hb)) + hb + b"".join(ts.values()))


def read_shard(path: pathlib.Path) -> tuple[dict, dict | None, bytes]:
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    h = json.loads(raw[8:8 + n])
    meta = h.pop("__metadata__", None)
    return h, meta, raw[8 + n:]


def fake_st(ent: dict) -> types.SimpleNamespace:
    """pack.type_ordered's view of a header: name -> (file, dtype, shape, offset, bytes)."""
    return types.SimpleNamespace(t={k: ("s", v["dtype"], tuple(v["shape"]), v["data_offsets"][0],
                                         v["data_offsets"][1] - v["data_offsets"][0]) for k, v in ent.items()})


def build(root: pathlib.Path, by_type: bool) -> tuple[pathlib.Path, dict[str, bytes]]:
    p = root / "model-00001-of-00001.safetensors"
    ts = {"embed.weight": body("embed.weight", UNIT * 20),
          "layers.0.attn.wq_a.trellis": body("wq_a", UNIT * 20)}
    ts.update(tensors([(MAIN, 8, "mul1"), (MTP, 2, "mcg")], by_type))
    write_shard(p, ts, {"format": "pt"})
    return p, ts


class Rewrite(unittest.TestCase):
    def test_a_type_ordered_shard(self):
        with tempfile.TemporaryDirectory() as d:
            p, ts = build(pathlib.Path(d), by_type=True)
            ent, _, _ = read_shard(p)
            self.assertIsNotNone(pack.type_ordered(fake_st(ent)))
            self.assertTrue(reorder.rewrite(p))
            ent, meta, data = read_shard(p)
            self.assertIsNone(pack.type_ordered(fake_st(ent)))
            self.assertEqual(meta, {"format": "pt"})
            for name, b in ts.items():
                a, z = ent[name]["data_offsets"]
                self.assertEqual(data[a:z], b, name)
            self.assertEqual(list(ent), reorder.plan(list(ts)))
            k = "%s0.w1.suh" % MAIN
            i = list(ent).index(k)
            self.assertEqual(list(ent)[i:i + 4], [k, "%s0.w1.svh" % MAIN, "%s0.w1.mul1" % MAIN,
                                                  "%s0.w1.trellis" % MAIN])
            # (the drafter's mcg pieces go where mul1 would: suh svh mcg trellis)
            k = "%s1.w3.suh" % MTP
            i = list(ent).index(k)
            self.assertEqual(list(ent)[i:i + 4], [k, "%s1.w3.svh" % MTP, "%s1.w3.mcg" % MTP,
                                                  "%s1.w3.trellis" % MTP])

    def test_expert_major_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            p, ts = build(pathlib.Path(d), by_type=False)
            before = p.read_bytes()
            self.assertFalse(reorder.rewrite(p))
            self.assertEqual(p.read_bytes(), before)

    def test_a_dense_shard_is_left_alone(self):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "model-00001-of-00001.safetensors"
            write_shard(p, {"embed.weight": b"e" * 100, "head.trellis": b"h" * 100})
            before = p.read_bytes()
            self.assertFalse(reorder.rewrite(p))
            self.assertEqual(p.read_bytes(), before)

    @unittest.skipIf(sys.platform == "win32", "a symlink needs a privilege on Windows")
    def test_a_symlinked_shard_keeps_its_link(self):
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "store").mkdir()
            target, _ = build(root / "store", by_type=True)
            (root / "model").mkdir()
            link = root / "model" / target.name
            link.symlink_to(target)
            self.assertTrue(reorder.rewrite(link))
            self.assertTrue(link.is_symlink())
            ent, _, _ = read_shard(target)
            self.assertIsNone(pack.type_ordered(fake_st(ent)))


class ModelDir(unittest.TestCase):
    def test_finds_only_the_shards_that_need_it(self):
        with tempfile.TemporaryDirectory() as d:
            m = pathlib.Path(d)
            expert, ts = build(m, by_type=True)
            dense = m / "model-00002-of-00002.safetensors"
            write_shard(dense, {"layers.0.attn.wo_b.trellis": b"w" * 200})
            before = dense.read_bytes()
            (m / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {
                **{k: expert.name for k in ts}, "layers.0.attn.wo_b.trellis": dense.name}}))
            self.assertEqual(reorder.main(["--model", str(m)]), 0)
            ent, _, _ = read_shard(expert)
            self.assertIsNone(pack.type_ordered(fake_st(ent)))
            self.assertEqual(dense.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

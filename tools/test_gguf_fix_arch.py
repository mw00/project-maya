"""tools/gguf_fix_arch.py over small GGUFs written here: "glm5next" becomes "glm5-next" with every key and value kept,
the data section starts at the same byte and every byte after the header is the same - so a Maya pack's absolute
offsets stay valid.  No model, no GPU.

    python -m unittest tools.test_gguf_fix_arch
"""
from __future__ import annotations

import hashlib
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import gguf_fix_arch as F  # noqa: E402


def s(text: str) -> bytes:
    b = text.encode()
    return struct.pack("<Q", len(b)) + b


def kv(key: str, t: int, raw: bytes) -> bytes:
    return s(key) + struct.pack("<I", t) + raw


def write_gguf(path: Path, arch="glm5next", n_arch_keys=3, recipe=True, align=32) -> bytes:
    """A GGUF v3: the architecture, n_arch_keys of its keys (a number, an array of floats, ...), a string array (a
    tokenizer's shape), maybe a maya.recipe, two F32 tensors.  Returns the bytes from the data start on."""
    kvs = [kv("general.architecture", 8, s(arch))]
    for i in range(n_arch_keys):
        if i % 2:
            kvs.append(kv(f"{arch}.clamp_{i}", 9, struct.pack("<IQ", 6, 4) + struct.pack("<4f", 10, 10, 10, 10)))
        else:
            kvs.append(kv(f"{arch}.count_{i}", 4, struct.pack("<I", 288 + i)))
    kvs.append(kv("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) + s("a") + s("bc") + s("<|user|>")))
    if recipe:   # a recipe's shape: a type per layer (json.dumps' ", " and ": " are the spaces the rename can take)
        layers = {str(i): {"gate_up": "IQ2_XXS", "down": "IQ3_XXS"} for i in range(3, 45)}
        note = "IQ2_XXS gate/up; IQ3_XXS down; Q6_K attention; shared experts; dense layers; the NextN block"
        kvs.append(kv("maya.recipe", 8, s(json.dumps({"name": "Maya-S v2", "layers": layers, "note": note}))))
    tensors = [("blk.0.ffn_gate_exps.weight", (8,), 0), ("blk.0.ffn_down_exps.weight", (8,), 32)]
    tinfo = b"".join(s(n) + struct.pack("<I", len(d)) + struct.pack(f"<{len(d)}Q", *d) + struct.pack("<IQ", 0, off)
                     for n, d, off in tensors)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kvs)) + b"".join(kvs) + tinfo
    pad = -len(head) % align
    data = struct.pack("<8f", *range(8)) + bytes(32 - 32) + struct.pack("<8f", *range(8, 16))
    path.write_bytes(head + bytes(pad) + data)
    return data


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


class FixArch(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.d = Path(self.tmp.name)

    def renamed_ok(self, src: Path, dst: Path, data: bytes):
        old, new = F.read_header(src), F.read_header(dst)
        self.assertEqual(new.data_start, old.data_start)
        self.assertEqual(dst.stat().st_size, src.stat().st_size if src != dst else dst.stat().st_size)
        self.assertEqual(dst.read_bytes()[new.data_start:], data)            # every byte after the header
        self.assertEqual(F._str_of(new.get("general.architecture")[1]), "glm5-next")
        self.assertFalse([k for k, _, _ in new.kvs if k.startswith("glm5next.")])
        F.verify(dst, old)

    def test_a_copy_keeps_the_data_where_it_was(self):
        src, dst = self.d / "m-00001-of-00003.gguf", self.d / "out.gguf"
        data = write_gguf(src)
        self.assertEqual(subprocess.run([sys.executable, str(ROOT / "tools" / "gguf_fix_arch.py"), str(src), "--out",
                                         str(dst)], capture_output=True).returncode, 0)
        self.renamed_ok(src, dst, data)
        self.assertEqual(struct.unpack("<I", F.read_header(dst).get("glm5-next.count_0")[1])[0], 288)

    def test_in_place_rewrites_the_header_only(self):
        src = self.d / "m.gguf"
        data = write_gguf(src, n_arch_keys=40)              # 41 bytes more than the alignment padding can hold
        before = F.read_header(src)
        copy = self.d / "before.gguf"
        copy.write_bytes(src.read_bytes())
        r = subprocess.run([sys.executable, str(ROOT / "tools" / "gguf_fix_arch.py"), str(src), "--in-place"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(F.read_header(src).data_start, before.data_start)
        self.assertEqual(sha(src.read_bytes()[before.data_start:]), sha(data))
        F.verify(src, F.read_header(copy))
        # the recipe gave up its spaces: the same JSON
        self.assertEqual(json.loads(F._str_of(F.read_header(src).get("maya.recipe")[1])),
                         json.loads(F._str_of(before.get("maya.recipe")[1])))

    def test_the_note_only_when_the_spaces_are_not_enough(self):
        # Maya-S and Maya-M were one byte short of the JSON spaces: then (only then) the note's "; " becomes ";"
        p, out = self.d / "a.gguf", self.d / "b.gguf"
        tightened = 0
        for n in range(150, 320):          # the recipe's JSON spaces (~215) run out around here
            data = write_gguf(p, n_arch_keys=n)
            h = F.read_header(p)
            try:
                new = F.renamed(h)
            except ValueError:
                continue
            out.write_bytes(new + p.read_bytes()[h.data_start:])
            self.renamed_ok(p, out, data)
            note = json.loads(F._str_of(F.read_header(out).get("maya.recipe")[1]))["note"]
            tightened += note != json.loads(F._str_of(h.get("maya.recipe")[1]))["note"]
        self.assertGreater(tightened, 0)

    def test_already_renamed_or_another_model(self):
        p = self.d / "a.gguf"
        write_gguf(p, arch="glm5-next")
        self.assertIsNone(F.renamed(F.read_header(p)))
        write_gguf(p, arch="qwen4exp")
        with self.assertRaises(ValueError):
            F.renamed(F.read_header(p))

    def test_no_recipe(self):
        p = self.d / "a.gguf"
        write_gguf(p, n_arch_keys=40, recipe=False)          # more than the padding: refused, the file untouched
        before = p.read_bytes()
        with self.assertRaises(ValueError):
            F.renamed(F.read_header(p))
        self.assertEqual(p.read_bytes(), before)
        for n in range(4):                                   # a few keys: the padding may hold them, or it refuses
            write_gguf(p, n_arch_keys=n, recipe=False)
            h = F.read_header(p)
            room = h.data_start - h.end
            if room >= n + 1:
                self.assertEqual(len(F.renamed(h)), h.data_start)
            else:
                with self.assertRaises(ValueError):
                    F.renamed(h)


if __name__ == "__main__":
    unittest.main()

"""tools/test_ds4_vision_pack.py - DeepSeek's vision tower for a GGUF install (tools/ds4/vision_pack.py, pack_gguf.py):
the picture markers read with range requests only (a server that sends the whole file is cut off, tried once more, then
a clear error), vision.bin equal to the checkpoint bit for bit, and the pack's entries pointing at exactly those bytes;
and which Engram set pack_gguf takes for a quant without the tables.  Small made-up files and a local HTTP server: no
GPU, no internet.

    python -m unittest tools.test_ds4_vision_pack -v
"""
from __future__ import annotations

import hashlib
import http.server
import json
import os
import pathlib
import struct
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "ds4"))
import vision_pack as VP  # noqa: E402

DIM = 64   # the made-up model's width (the marker rows' length)


def safetensors(path: pathlib.Path, tensors: list) -> dict:
    """tensors: [(name, dtype, shape, bytes)] in file order -> the header's entries."""
    head, off = {}, 0
    for name, dt, shape, b in tensors:
        head[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
        off += len(b)
    h = json.dumps(head).encode()
    h += b" " * ((-len(h)) % 8)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(h)) + h)
        for *_, b in tensors:
            f.write(b)
    return {k: dict(v, base=8 + len(h)) for k, v in head.items()}


def rnd(n: int, seed: int) -> bytes:
    return np.random.default_rng(seed).integers(0, 1 << 16, n, dtype=np.uint16).tobytes()


def make_model(root: pathlib.Path):
    """Shard 1 (vision.* / aligner.*), config.json, shard 2 (an embedding, then the three marker rows as DeepSeek's:
    image_end, image_newline, image_start), and a manifest entry for them."""
    v = root / "vision"
    v.mkdir()
    s1 = [("aligner.w1.bias", "BF16", [DIM], rnd(DIM, 1)), ("aligner.w1.weight", "BF16", [DIM, 36], rnd(DIM * 36, 2)),
          ("vision.blocks.0.attn.wqkv.weight", "BF16", [12, 4], rnd(48, 3)),
          ("vision.patch_embed.proj.weight", "BF16", [4, 588], rnd(4 * 588, 4)), ("vision.norm.weight", "BF16", [4],
                                                                                    rnd(4, 5))]
    safetensors(v / "model-00001-of-00048.safetensors", s1)
    cfg = {"image_token_id": 129264, "vision_config": {"num_hidden_layers": 1, "hidden_size": 4, "patch_size": 14,
                                                       "downsample_ratio": 3, "max_wh_ratio": None,
                                                       "model_type": "deepseek_v41_vision"}}
    (v / "config.json").write_text(json.dumps(cfg))
    marks = {n: rnd(DIM, 10 + i) for i, n in enumerate(("image_end", "image_newline", "image_start"))}
    s2 = root / "shard2.safetensors"
    head = safetensors(s2, [("embed.weight", "BF16", [100, DIM], rnd(100 * DIM, 9))] +
                       [(n, "BF16", [DIM], b) for n, b in marks.items()])
    man = {"repo": "deepseek-ai/x", "revision": "0" * 40,
           "files": {p.name: [p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()]
                     for p in (v / "model-00001-of-00048.safetensors", v / "config.json")},
           "marks": {"file": "model-00002-of-00048.safetensors", "bytes": s2.stat().st_size,
                     "rows": {n: [head[n]["base"] + head[n]["data_offsets"][0], DIM * 2,
                                  hashlib.sha256(b).hexdigest()] for n, b in marks.items()}}}
    return v, s2, man, s1, marks


class Server:
    """A local HTTP server over one file: with range requests honoured (206), or ignored (200, the whole file - its
    bytes counted as they leave, to show the client hung up)."""

    def __init__(self, path: pathlib.Path, ranges: bool = True, pad: int = 0):
        data = path.read_bytes() + b"\0" * pad
        me = self
        self.requests, self.sent = 0, 0

        class H(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                me.requests += 1
                rng = self.headers.get("Range")
                try:
                    if ranges and rng:
                        a, b = rng.split("=")[1].split("-")
                        a, b = int(a), int(b)
                        self.send_response(206)
                        self.send_header("Content-Range", f"bytes {a}-{b}/{len(data)}")
                        self.send_header("Content-Length", str(b - a + 1))
                        self.end_headers()
                        self.wfile.write(data[a:b + 1])
                        return
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    for i in range(0, len(data), 1 << 16):
                        self.wfile.write(data[i:i + (1 << 16)])
                        me.sent += min(1 << 16, len(data) - i)
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass

            def log_message(self, *a):
                pass

        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}/shard2"
        self.size = len(data)
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


class Marks(unittest.TestCase):
    def setUp(self):
        self.root = pathlib.Path(tempfile.mkdtemp())
        self.v, self.s2, self.man, self.s1, self.marks = make_model(self.root)

    def test_range_requests_only(self):
        srv = Server(self.s2)
        try:
            rows = VP.fetch_marks(srv.url, self.man["marks"])
        finally:
            srv.close()
        self.assertEqual(rows, self.marks)
        self.assertEqual(srv.requests, 3)      # (the header's length, the header, the rows)
        self.assertEqual(srv.sent, 0)          # (nothing outside the ranges)

    def test_a_server_that_sends_the_whole_file_is_cut_off(self):
        self.man["marks"]["bytes"] = None      # (the size check off: the padding below makes the file larger)
        srv = Server(self.s2, ranges=False, pad=64 << 20)
        try:
            with self.assertRaises(SystemExit) as e:
                VP.fetch_marks(srv.url, self.man["marks"])
        finally:
            srv.close()
        self.assertIn("could not be read", str(e.exception))
        self.assertIn("status 200", str(e.exception))
        self.assertEqual(srv.requests, 2)                  # (tried once more, then the error)
        self.assertLess(srv.sent, srv.size // 4)           # (the client hung up: no 64 MB pulled)

    def test_a_wrong_row_is_tried_again_then_an_error(self):
        name = "image_start"
        self.man["marks"]["rows"][name][2] = "0" * 64
        with self.assertRaises(SystemExit) as e:
            VP.fetch_marks(str(self.s2), self.man["marks"])
        self.assertIn("sha256", str(e.exception))

    def test_rows_not_where_the_manifest_says(self):
        self.man["marks"]["rows"]["image_end"][0] += 2
        with self.assertRaises(SystemExit) as e:
            VP.fetch_marks(str(self.s2), self.man["marks"])
        self.assertIn("not where the manifest says", str(e.exception))


class Pack(unittest.TestCase):
    def test_vision_bin_equals_the_checkpoint_and_the_pack_points_at_it(self):
        root = pathlib.Path(tempfile.mkdtemp())
        v, s2, man, s1, marks = make_model(root)
        info = VP.pack(v, man, str(s2))
        self.assertEqual(info["config"], {"vision.num_hidden_layers": 1, "vision.hidden_size": 4,
                                          "vision.patch_size": 14, "vision.downsample_ratio": 3,
                                          "image_token_id": 129264})
        self.assertFalse((v / "vision.bin.part").exists())
        want = {n: b for n, _, _, b in s1}
        want.update(marks)
        self.assertEqual(sorted(e[0] for e in info["tensors"]), sorted(want))
        blob = (v / "vision.bin").read_bytes()
        for name, dt, off, nb, shape in info["tensors"]:
            self.assertEqual(off % VP.ALIGN, 0)
            self.assertEqual(blob[off:off + nb], want[name], name)
            self.assertEqual(dt, "BF16")
        # pack_gguf: a one-tensor GGUF in <root>/Q, the vision folder beside it
        q = root / "Q"
        q.mkdir()
        gguf(q / "m-00001-of-00001.gguf")
        r = subprocess.run([sys.executable, str(HERE / "ds4" / "pack_gguf.py"), "--model", str(q)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("the vision tower", r.stdout)
        pk = q / "maya-ds4"
        kv = dict(line.split(" ", 1) for line in (pk / "ds4.txt").read_text().splitlines())
        self.assertEqual((kv["vision.num_hidden_layers"], kv["image_token_id"]), ("1", "129264"))
        files, ts = {}, {}
        for line in (pk / "tensors.txt").read_text().splitlines():
            p = line.split()
            if p[0] == "file":
                files[int(p[1])] = p[2]
            else:
                ts[p[1]] = (p[2], int(p[3]), int(p[4]), int(p[5]))
        self.assertEqual(files[1], "../vision/vision.bin")
        for name, b in want.items():                   # (each entry's bytes: the checkpoint's, bit for bit)
            dt, fi, off, nb = ts[name]
            with open(os.path.join(str(q), files[fi]), "rb") as f:
                f.seek(off)
                self.assertEqual(f.read(nb), b, name)


class EngramSet(unittest.TestCase):
    """pack_gguf's Engram set for a quant without the tables (v1.1.16's Maya-S): unsloth's MXFP8 shard in Engram/ beside
    it, passed over while it is not whole (its download still running); --engram: that folder (any type of the
    tables), whole, or a clear error."""

    def setUp(self):
        import pack_gguf
        self.pg = pack_gguf
        self.base = pathlib.Path(tempfile.mkdtemp())
        self.model = self.base / "Maya-S"
        self.model.mkdir()

    def table(self, folder: str, typ: int, cut: int = 0) -> pathlib.Path:
        """<base>/<folder>/t.gguf: one Engram table blk.1.engram_embd of two rows of 256 values in ggml type typ,
        its last `cut` bytes missing."""
        def s(x: str) -> bytes:
            b = x.encode()
            return struct.pack("<Q", len(b)) + b
        _, per, bpb = self.pg.TYPES[typ]
        kv = [s("general.architecture") + struct.pack("<I", 8) + s("deepseek41")]
        info = s("blk.1.engram_embd.weight") + struct.pack("<I", 2) + struct.pack("<2Q", 256, 2) + struct.pack("<IQ", typ, 0)
        head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv)) + b"".join(kv) + info
        head += b"\0" * ((-len(head)) % 32)
        d = self.base / folder
        d.mkdir(exist_ok=True)
        body = head + bytes(2 * 256 // per * bpb)
        (d / "t.gguf").write_bytes(body[:len(body) - cut])
        return d

    def test_whole(self):
        old = self.table("Engram", 43, cut=100)        # (downloading)
        self.assertEqual(self.pg.engram_set(self.model, None), (None, []))
        self.table("Engram", 43)
        d, egs = self.pg.engram_set(self.model, None)
        self.assertEqual((d, egs[0].tensors[0][1]), (old, 43))
        self.assertIsNone(self.pg.short(egs[0]))

    def test_explicit(self):
        old = self.table("Engram", 67)
        q6 = self.table("Engram-Q6_K", 14)
        d, egs = self.pg.engram_set(self.model, str(q6))
        self.assertEqual((d, egs[0].tensors[0][1]), (q6, 14))
        self.assertEqual(self.pg.engram_set(self.model, str(old))[0], old)
        self.table("Engram", 67, cut=1)
        with self.assertRaises(SystemExit) as e:
            self.pg.engram_set(self.model, str(old))
        self.assertIn("not whole", str(e.exception))
        with self.assertRaises(SystemExit):
            self.pg.engram_set(self.model, str(self.base / "nowhere"))

    def test_none(self):
        self.assertEqual(self.pg.engram_set(self.model, None), (None, []))
        (self.base / "Engram").mkdir()
        (self.base / "Engram" / "t.gguf").write_bytes(b"GGU")   # (a header cut short: not whole, passed over)
        self.assertEqual(self.pg.engram_set(self.model, None), (None, []))


class Shards(unittest.TestCase):
    def test_a_folder_with_an_earlier_layout_too(self):
        """A folder that holds two layouts of a model (an earlier one beside the one in use): a clear error, and
        --shards names the files to pack."""
        q = pathlib.Path(tempfile.mkdtemp()) / "Q"
        q.mkdir()
        gguf(q / "m-00001-of-00001.gguf")
        gguf(q / "m-00001-of-00002.gguf")
        cmd = [sys.executable, str(HERE / "ds4" / "pack_gguf.py"), "--model", str(q)]
        r = subprocess.run(cmd, capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("more than one GGUF split (-of-00001, -of-00002)", r.stderr)
        r = subprocess.run(cmd + ["--shards", "m-00001-of-00001.gguf"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        files = [line.split()[2] for line in (q / "maya-ds4" / "tensors.txt").read_text().splitlines()
                 if line.startswith("file ")]
        self.assertEqual(files, ["m-00001-of-00001.gguf"])


class Drafter(unittest.TestCase):
    def test_left_out_without_its_geometry(self):
        """#500: a drafter beside a GGUF with no DSpark geometry (and not DeepSeek-V4.1-Flash's trunk, whose own values
        the pack would take): left out of the pack, loudly - it would load and never draft."""
        root = pathlib.Path(tempfile.mkdtemp())
        q = root / "Q"
        q.mkdir()
        gguf(q / "m-00001-of-00001.gguf")
        (root / "dspark").mkdir()
        (root / "dspark" / "dspark.bin").write_bytes(bytes(256))
        (root / "dspark" / "dspark.json").write_text(json.dumps(
            {"from": [], "tensors": [["mtp.0.attn_norm.weight", "BF16", 0, 8, [4]]]}))
        r = subprocess.run([sys.executable, str(HERE / "ds4" / "pack_gguf.py"), "--model", str(q)],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("is left out of the pack", r.stderr)
        self.assertIn("dspark_block_size", r.stderr)
        self.assertNotIn("the DSpark drafter", r.stdout)
        self.assertNotIn("dspark.bin", (q / "maya-ds4" / "tensors.txt").read_text())


def gguf(path: pathlib.Path, meta: dict | None = None):
    """A DeepSeek-V4.1 GGUF of one tensor (the embedding, BF16 [4][DIM]) and the metadata pack_gguf reads (meta: more
    deepseek41.* keys, ints or lists of ints)."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b
    kv = [s("general.architecture") + struct.pack("<I", 8) + s("deepseek41"),
          s("tokenizer.ggml.tokens") + struct.pack("<IIQ", 9, 8, 4) + b"".join(s(t) for t in ("a", "b", "c", "d"))]
    for k, v in (meta or {}).items():   # (UINT32, or an ARRAY of them)
        kv.append(s("deepseek41." + k) + (struct.pack("<IIQ", 9, 4, len(v)) + struct.pack("<%dI" % len(v), *v)
                                          if isinstance(v, list) else struct.pack("<II", 4, v)))
    info = s("token_embd.weight") + struct.pack("<I", 2) + struct.pack("<2Q", DIM, 4) + struct.pack("<IQ", 30, 0)
    head = b"GGUF" + struct.pack("<IQQ", 3, 1, len(kv)) + b"".join(kv) + info
    head += b"\0" * ((-len(head)) % 32)
    path.write_bytes(head + rnd(4 * DIM, 7))


if __name__ == "__main__":
    unittest.main()

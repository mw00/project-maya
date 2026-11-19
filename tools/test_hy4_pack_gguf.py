"""tools/hy4/pack_gguf.py on a synthetic Hy4-flash GGUF set (the GGUF's hy_v4_flash layout, two shards, tiny
dimensions): the geometry, the tensors at their bytes, the Engram hash file, the tokenizer with its tag aliases - read
back by the server's tokenizer and chat template -, the vision tower's mmproj file beside the set (found and indexed,
left out with --mmproj none, another projector's ignored) and the refusals (a shard missing, another pre-tokenizer,
Engram buckets outside the table, a sliding-window layer with an indexer, a vision tower for another model's width);
the MTP draft block in a file of its own (tools/hy4/mtp_gguf.py writes it from a set that has the block: the tensors'
bytes as they were) beside a set without one - found, indexed as one more file, the geometry a block longer; left out
with --mtp none; not read beside a set with its own block; another model's refused.

    python -m unittest tools.test_hy4_pack_gguf
"""
from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
PACK = HERE / "hy4" / "pack_gguf.py"
MTP = HERE / "hy4" / "mtp_gguf.py"
ARCH = "hy_v4_flash"

# ggml type -> (values a block, bytes a block)
BLOCK = {0: (1, 4), 8: (32, 34), 12: (256, 144), 30: (1, 2), 43: (1, 1)}
TAG = ":6124c78e"
D, NL, ROWS, HD = 256, 3, 64, 32
SPECIAL = ["<｜hy_start%s｜>" % TAG, "<｜hy_end%s｜>" % TAG, "<think%s>" % TAG, "</think%s>" % TAG,
           "<tool_calls%s>" % TAG, "</tool_calls%s>" % TAG, "<tool_call%s>" % TAG, "<arg_key%s>" % TAG,
           "<｜hy_image▁start%s｜>" % TAG, "<｜hy_image▁end%s｜>" % TAG, "<｜hy_image▁pad%s｜>" % TAG,
           "<think:abc>"]   # (the last one a normal token: not aliased)
DV, NLV = 64, 2   # (the vision tower's width and layers)
TEMPLATE = ("{%- set HYTK = '" + TAG + "' -%}\n"
            "{%- set effort = reasoning_effort | default('high') -%}\n"
            "<｜hy_start{{ HYTK }}｜>reasoning_mode={{ effort }}\n"
            "{%- for m in messages %}[{{ m.role }}]{{ m.content }}{% endfor -%}\n"
            "{%- if tools %}[tools:{{ tools | map(attribute='function') | map(attribute='name') | join(',') }}]"
            "{% endif -%}\n"
            "{%- if add_generation_prompt %}<｜hy_start{{ HYTK }}｜>assistant{% endif -%}\n")


def byte_alphabet() -> list[str]:
    sys.path.insert(0, str(HERE))
    from strata_tokenizer import BYTE_TO_UNICODE
    return list(BYTE_TO_UNICODE.values())


def gguf(path: pathlib.Path, kv: dict, tensors: list):
    """A GGUF v3 file: kv {key: (type id, value)}, tensors [(name, ggml type, dims ne0 first)], the data a byte pattern
    a tensor (its index), 32-byte aligned."""
    def s(x: str) -> bytes:
        b = x.encode()
        return struct.pack("<Q", len(b)) + b

    def val(t: int, v) -> bytes:
        if t == 8:
            return s(v)
        if t == 9:
            at, items = v
            return struct.pack("<IQ", at, len(items)) + b"".join(val(at, x) for x in items)
        return struct.pack("<" + {2: "H", 4: "I", 5: "i", 6: "f", 7: "?", 10: "Q", 11: "q"}[t], v)
    head = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), len(kv))
    for k, (t, v) in kv.items():
        head += s(k) + struct.pack("<I", t) + val(t, v)
    off, infos, datas = 0, b"", []
    for i, (name, typ, dims) in enumerate(tensors):
        per, bpb = BLOCK[typ]
        n = 1
        for x in dims:
            n *= x
        nb = n // per * bpb
        infos += s(name) + struct.pack("<I", len(dims)) + struct.pack("<%dQ" % len(dims), *dims) + struct.pack("<IQ", typ, off)
        datas.append(bytes([i % 251]) * nb)
        off += (nb + 31) // 32 * 32
    head += infos
    head += b"\0" * ((32 - len(head) % 32) % 32)
    with path.open("wb") as f:
        f.write(head)
        for d in datas:
            f.write(d + b"\0" * ((32 - len(d) % 32) % 32))


def model(root: pathlib.Path, **over) -> pathlib.Path:
    """Two shards of a 3-layer (+1 MTP) Hy4-flash: layer 0 dense and indexed, 1 and 2 sliding-window, Engram at 1.
    over: kv entries replaced (a value None: the key left out)."""
    tokens = byte_alphabet() + SPECIAL + ["hi"]   # (one merge: h i)
    vocab = len(tokens)
    types = [1] * 256 + [3] * (len(SPECIAL) - 1) + [1, 1]
    a = lambda k: ARCH + "." + k  # noqa: E731
    kv = {
        "general.architecture": (8, ARCH),
        a("block_count"): (4, NL + 1), a("nextn_predict_layers"): (4, 1),
        a("attention.head_count"): (9, (5, [64, 32, 32, 64])),
        a("attention.head_count_kv"): (4, 1),
        a("attention.sliding_window_pattern"): (9, (7, [False, True, True, False])),
        a("attention.indexer.types"): (9, (7, [True, False, False, True])),
        a("swiglu_clamp_exp"): (9, (6, [0.0, 10.0, 10.0, 10.0])),
        a("embedding_length"): (4, D), a("vocab_size"): (4, vocab), a("context_length"): (4, 4096),
        a("feed_forward_length"): (4, 512), a("leading_dense_block_count"): (4, 1),
        a("expert_count"): (4, 4), a("expert_used_count"): (4, 2), a("expert_feed_forward_length"): (4, 256),
        a("expert_shared_count"): (4, 1), a("expert_latent_length"): (4, 128),
        a("expert_weights_scale"): (6, 4.0), a("expert_weights_norm"): (7, True), a("expert_gating_func"): (4, 2),
        a("attention.layer_norm_rms_epsilon"): (6, 1e-5),
        a("attention.q_lora_rank"): (4, 128), a("attention.kv_lora_rank"): (4, 64),
        a("attention.key_length_mla"): (4, 96), a("attention.value_length_mla"): (4, 64),
        a("rope.dimension_count"): (4, 32), a("rope.freq_base"): (6, 1e7), a("rope.freq_base_swa"): (6, 1e4),
        a("attention.sliding_window"): (4, 512),
        a("attention.indexer.head_count"): (4, 4), a("attention.indexer.key_length"): (4, 128),
        a("attention.indexer.top_k"): (4, 2048), a("attention.indexer.kpool"): (4, 4),
        a("attention.indexer.kpool_select_tail"): (7, True),
        a("hyper_connection.count"): (4, 4), a("hyper_connection.epsilon"): (6, 1e-6),
        a("hyper_connection.magnitude"): (6, 2.0),
        # Engram: one layer, trigrams, 2 heads an n-gram -> 4 columns of 32-dim rows in a 64-row table
        a("ple.layers"): (9, (5, [1])), a("ple.ngram_size"): (4, 3), a("ple.heads_per_ngram"): (4, 2),
        a("embedding_length_per_layer_input"): (4, HD),
        a("ple.layer_multipliers"): (9, (11, [7, -3, 1 << 40])),
        a("ple.head_offsets"): (9, (5, [0, 11, 24, 41])), a("ple.head_vocab_sizes"): (9, (5, [11, 13, 17, 19])),
        a("engram.token_map"): (9, (5, list(range(vocab - 5)))), a("engram.pad_token_id"): (4, 3),
        "tokenizer.ggml.model": (8, "gpt2"), "tokenizer.ggml.pre": (8, "hy_v4"),
        "tokenizer.ggml.tokens": (9, (8, tokens)), "tokenizer.ggml.token_type": (9, (5, types)),
        "tokenizer.ggml.merges": (9, (8, ["h i"])),
        "tokenizer.ggml.bos_token_id": (4, 256), "tokenizer.ggml.eos_token_id": (4, 257),
        "tokenizer.chat_template": (8, TEMPLATE),
        "split.count": (2, 2),
    }
    for k, v in over.items():
        if v is None:
            kv.pop(k, None)
        else:
            kv[k] = v
    first = [("token_embd.weight", 0, [D, vocab]), ("output.weight", 8, [D, vocab]), ("output_norm.weight", 0, [D]),
             ("output_hc_fn.weight", 0, [4 * D, 4]), ("blk.0.attn_q_a.weight", 12, [D, 128]),
             ("blk.1.attn_q_a.weight", 12, [D, 128])]
    second = [("blk.2.attn_q_a.weight", 12, [D, 128]), ("per_layer_token_embd.weight", 43, [HD, ROWS])]
    root.mkdir(parents=True, exist_ok=True)
    gguf(root / "Hy4-flash-Q4_K_M-00001-of-00002.gguf", kv, first)
    gguf(root / "Hy4-flash-Q4_K_M-00002-of-00002.gguf", {"split.count": (2, 2)}, second)
    return root


def mmproj(path: pathlib.Path, projector: str = ARCH, out_dim: int = D) -> pathlib.Path:
    """A 2-layer vision tower of DV in the mmproj layout (clip.*, v.* and mm.* tensors), a 4 x 4 position grid."""
    c = lambda k: "clip.vision." + k  # noqa: E731
    kv = {"general.architecture": (8, "clip"), "general.type": (8, "mmproj"), "clip.projector_type": (8, projector),
          c("embedding_length"): (4, DV), c("feed_forward_length"): (4, 128), c("block_count"): (4, NLV),
          c("attention.head_count"): (4, 4), c("patch_size"): (4, 16), c("projection_dim"): (4, out_dim),
          c("spatial_merge_size"): (4, 2), c("attention.layer_norm_epsilon"): (6, 1e-5),
          c("image_min_pixels"): (4, 4096), c("image_max_pixels"): (4, 1 << 20),
          c("image_mean"): (9, (6, [0.5, 0.5, 0.5])), c("image_std"): (9, (6, [0.5, 0.5, 0.5]))}
    t = [("v.patch_embd.weight", 0, [16, 16, 3, DV]), ("v.position_embd.weight", 0, [DV, 16]),
         ("v.image_newline", 0, [out_dim]), ("mm.0.weight", 0, [2, 2, DV, 4 * DV]), ("mm.2.weight", 0, [1, 1, 4 * DV, out_dim])]
    for i in range(NLV):
        t += [("v.blk.%d.attn_q.weight" % i, 30, [DV, DV]), ("v.blk.%d.ffn_down.weight" % i, 30, [128, DV])]
    gguf(path, kv, t)
    return path


def pack(root: pathlib.Path, *more: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PACK), "--model", str(root), *more], capture_output=True, text=True)


class PackTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = model(pathlib.Path(self.tmp.name) / "m")

    def tearDown(self):
        self.tmp.cleanup()

    def test_other_files_in_the_folder(self):
        # a vision projector and a drafter beside the set: the pack is the set's alone
        gguf(self.root / "mmproj-x.gguf", {"general.architecture": (8, "clip"), "general.type": (8, "mmproj")},
             [("v.blk.0.attn_q.weight", 0, [D, D])])
        gguf(self.root / "x-drafter.gguf", {"general.architecture": (8, "hy_v4_flash"), "general.type": (8, "drafter")},
             [("model.dfly_layers.0.w", 0, [D, D])])
        (self.root / "junk.gguf").write_bytes(b"not a gguf")
        r = pack(self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("left out", r.stdout)
        lines = (self.root / "maya-hy4" / "tensors.txt").read_text().splitlines()
        self.assertEqual(len([ln for ln in lines if ln.startswith("file ")]), 2)
        self.assertEqual(len([ln for ln in lines if ln.startswith("t ")]), 8)

    def test_engram_scales_and_eps(self):
        r = pack(self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        text = (self.root / "maya-hy4" / "hy4.txt").read_text()
        self.assertNotIn("engram_scales", text)
        self.assertNotIn("engram_eps", text)
        (self.root / "x.engram-scales.bf16").write_bytes(b"\0" * (2 * ROWS))
        r = subprocess.run([sys.executable, str(PACK), "--model", str(self.root), "--engram-eps", "0.0078125"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        geo = dict(line.split(" ", 1) for line in (self.root / "maya-hy4" / "hy4.txt").read_text().splitlines())
        self.assertEqual(geo["engram_scales"], "x.engram-scales.bf16")
        self.assertEqual(float(geo["engram_eps"]), 0.0078125)
        (self.root / "x.engram-scales.bf16").write_bytes(b"\0" * (2 * ROWS - 2))
        self.assertNotEqual(pack(self.root).returncode, 0)

    def test_geometry_and_tensors(self):
        r = pack(self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        out = self.root / "maya-hy4"
        geo = dict(line.split(" ", 1) for line in (out / "hy4.txt").read_text().splitlines())
        self.assertEqual(geo["n_layer"], "3")
        self.assertEqual(geo["n_mtp"], "1")
        self.assertEqual(geo["heads"], "64,32,32,64")
        self.assertEqual(geo["swa"], "0,1,1,0")
        self.assertEqual(geo["indexer"], "1,0,0,1")
        self.assertEqual(geo["swiglu_limit"], "0.0,10.0,10.0,10.0")
        self.assertEqual(geo["n_expert_latent"], "128")
        self.assertEqual(geo["bos"], "256")
        self.assertEqual(geo["model_dir"], str(self.root))
        lines = (out / "tensors.txt").read_text().splitlines()
        files = [ln.split() for ln in lines if ln.startswith("file ")]
        self.assertEqual([f[2] for f in files], ["Hy4-flash-Q4_K_M-00001-of-00002.gguf",
                                                 "Hy4-flash-Q4_K_M-00002-of-00002.gguf"])
        t = {ln.split()[1]: ln.split()[2:] for ln in lines if ln.startswith("t ")}
        self.assertEqual(len(t), 8)
        typ, fi, off, nb, *dims = t["per_layer_token_embd.weight"]
        self.assertEqual((typ, fi, int(nb), dims), ("F8_E4M3", "1", HD * ROWS, [str(HD), str(ROWS)]))
        # the bytes at the offset are the tensor's (the writer's pattern: its index in its file)
        with (self.root / files[1][2]).open("rb") as f:
            f.seek(int(off))
            self.assertEqual(f.read(int(nb)), bytes([1]) * int(nb))
        self.assertEqual(t["blk.0.attn_q_a.weight"][0], "Q4_K")
        self.assertEqual(int(t["blk.0.attn_q_a.weight"][3]), 128 * D // 256 * 144)

    def test_engram(self):
        self.assertEqual(pack(self.root).returncode, 0)
        b = (self.root / "maya-hy4" / "engram.bin").read_bytes()
        magic, vocab, nm, ngram, heads, hd, pad, rows = struct.unpack_from("<4s7i", b)
        n_tok = 256 + len(SPECIAL) + 1
        self.assertEqual((magic, vocab, nm, ngram, heads, hd, pad, rows), (b"HYE1", n_tok, 1, 3, 2, HD, 3, ROWS))
        o = 32
        tm = struct.unpack_from("<%di" % vocab, b, o)
        self.assertEqual(list(tm[:vocab - 5]), list(range(vocab - 5)))
        self.assertEqual(list(tm[vocab - 5:]), [-1] * 5)   # (the map's missing tail: no n-gram id)
        o += 4 * vocab
        self.assertEqual(struct.unpack_from("<i", b, o)[0], 1)
        o += 4
        self.assertEqual(list(struct.unpack_from("<3Q", b, o)), [7, (1 << 64) - 3, 1 << 40])
        o += 24
        self.assertEqual(list(struct.unpack_from("<4Q", b, o)), [11, 13, 17, 19])
        o += 32
        self.assertEqual(list(struct.unpack_from("<4Q", b, o)), [0, 11, 24, 41])
        self.assertEqual(len(b), o + 32)

    def test_tokenizer_and_template(self):
        self.assertEqual(pack(self.root).returncode, 0)
        tdir = self.root / "maya-hy4" / "tokenizer"
        meta = json.loads((tdir / "tokenizer.json").read_text(encoding="utf-8"))
        self.assertEqual(meta["pre"], "hy_v4")
        self.assertEqual(meta["special_ids"], {"bos": 256, "eos": 257})
        sys.path.insert(0, str(HERE))
        from strata_tokenizer import Tokenizer
        tk = Tokenizer.from_dir(tdir)
        ids = {s: 256 + i for i, s in enumerate(SPECIAL)}
        # the answers' tags read as GLM's, the <tool_calls> wrapper as nothing; the turn markers and a normal token
        # that only looks like a tag keep their own text
        self.assertEqual(tk.token_bytes(ids["<think%s>" % TAG]), b"<think>")
        self.assertEqual(tk.token_bytes(ids["</think%s>" % TAG]), b"</think>")
        self.assertEqual(tk.token_bytes(ids["<tool_call%s>" % TAG]), b"<tool_call>")
        self.assertEqual(tk.token_bytes(ids["<arg_key%s>" % TAG]), b"<arg_key>")
        self.assertEqual(tk.token_bytes(ids["<tool_calls%s>" % TAG]), b"")
        self.assertEqual(tk.token_bytes(ids["</tool_calls%s>" % TAG]), b"")
        self.assertEqual(tk.token_bytes(ids["<｜hy_end%s｜>" % TAG]), ("<｜hy_end%s｜>" % TAG).encode())
        self.assertNotIn(ids["<think:abc>"], tk.aliases)
        # prompts encode the tags by their own text
        self.assertEqual(tk.encode("<think%s>hi" % TAG, parse_special=True), [ids["<think%s>" % TAG], 256 + len(SPECIAL)])

        sys.path.insert(0, str(ROOT))
        from serve.frontend import ChatTemplate
        ct = ChatTemplate(tdir / "chat_template.jinja")
        self.assertTrue(ct.hy4)
        self.assertEqual(ct.hy4_tag, TAG)
        msgs = [{"role": "user", "content": "hi"}]
        # Hy4-flash thinks at high or not at all: off (and "none") is no_think, every other level and none named high
        self.assertIn("reasoning_mode=no_think", ct.render(msgs, enable_thinking=False))
        self.assertIn("reasoning_mode=no_think", ct.render(msgs, reasoning_effort="none"))
        for level in ("max", "low", "medium", "xhigh", "high", None):
            self.assertIn("reasoning_mode=high", ct.render(msgs, **({"reasoning_effort": level} if level else {})))
        out = ct.render(msgs, tools=[{"name": "f", "parameters": {}}])   # (a bare function: wrapped)
        self.assertIn("[tools:f]", out)


class VisionTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = model(pathlib.Path(self.tmp.name) / "m")

    def tearDown(self):
        self.tmp.cleanup()

    def geo(self) -> dict:
        return dict(line.split(" ", 1) for line in (self.root / "maya-hy4" / "hy4.txt").read_text().splitlines())

    def files(self) -> list:
        return [ln.split()[2] for ln in (self.root / "maya-hy4" / "tensors.txt").read_text().splitlines()
                if ln.startswith("file ")]

    def test_found_beside_the_set(self):
        """The folder's mmproj file of this projector: not a third shard, its tensors indexed as the last file, its
        geometry and the vocabulary's image tokens in hy4.txt."""
        mmproj(self.root / "mmproj-Hy4-flash-bf16.gguf")
        r = pack(self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("the vision tower (mmproj-Hy4-flash-bf16.gguf)", r.stdout)
        self.assertEqual(self.files()[2:], ["mmproj-Hy4-flash-bf16.gguf"])
        g = self.geo()
        ids = {s: str(256 + i) for i, s in enumerate(SPECIAL)}
        self.assertEqual((g["vision.dim"], g["vision.heads"], g["vision.ff"], g["vision.layers"], g["vision.grid"],
                          g["vision.out_dim"], g["vision.patch"], g["vision.merge"], g["vision.max_pixels"]),
                         (str(DV), "4", "128", str(NLV), "4", str(D), "16", "2", str(1 << 20)))
        self.assertEqual((g["vision.image_token"], g["vision.image_start"], g["vision.image_end"]),
                         (ids["<｜hy_image▁pad%s｜>" % TAG], ids["<｜hy_image▁start%s｜>" % TAG],
                          ids["<｜hy_image▁end%s｜>" % TAG]))
        self.assertEqual(g["vision.mean"], "0.5,0.5,0.5")
        t = {ln.split()[1]: ln.split()[2:] for ln in (self.root / "maya-hy4" / "tensors.txt").read_text().splitlines()
             if ln.startswith("t ")}
        self.assertEqual(t["mm.0.weight"][:2], ["F32", "2"])
        self.assertEqual(t["v.blk.1.attn_q.weight"][0], "BF16")

    def test_text_only(self):
        """--mmproj none, or a folder without one, or another model's projector beside it: the set alone."""
        mmproj(self.root / "mmproj-other.gguf", projector="qwen2vl")
        r = pack(self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("text only", r.stdout)
        self.assertNotIn("vision.dim", self.geo())
        self.assertEqual(len(self.files()), 2)
        mmproj(self.root / "mmproj-Hy4-flash-bf16.gguf")
        self.assertEqual(pack(self.root, "--mmproj", "none").returncode, 0)
        self.assertNotIn("vision.dim", self.geo())
        r = pack(self.root, "--mmproj", "mmproj-other.gguf")   # (named: refused, not ignored)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("not hy_v4_flash", r.stderr)

    def test_another_width(self):
        mmproj(self.root / "mmproj-Hy4-flash-bf16.gguf", out_dim=2 * D)
        r = pack(self.root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("a vision geometry maya-hy4 does not run", r.stderr)


class RefusalTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def refused(self, root: pathlib.Path, words: str):
        r = pack(root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn(words, r.stderr)
        self.assertFalse((root / "maya-hy4" / "tensors.txt").exists())

    def test_shard_missing(self):
        root = model(self.base / "a")
        (root / "Hy4-flash-Q4_K_M-00002-of-00002.gguf").unlink()
        self.refused(root, "the set has 2")

    def test_other_pre_tokenizer(self):
        root = model(self.base / "b", **{"tokenizer.ggml.pre": (8, "deepseek-v3")})
        r = pack(root)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("hy_v4 pre-tokenizer", r.stderr)

    def test_engram_outside_table(self):
        root = model(self.base / "c", **{ARCH + ".ple.head_offsets": (9, (5, [0, 11, 24, 50]))})
        self.refused(root, "outside the table")

    def test_swa_layer_with_indexer(self):
        root = model(self.base / "d", **{ARCH + ".attention.indexer.types": (9, (7, [True, True, False, True]))})
        self.refused(root, "sliding-window layer with an indexer")


def public(root: pathlib.Path) -> pathlib.Path:
    """The set as made from the public release: the 3 layers alone, no MTP block (the per-block lists 3 long)."""
    a = lambda k: ARCH + "." + k  # noqa: E731
    return model(root, **{a("block_count"): (4, NL), a("nextn_predict_layers"): None,
                          a("attention.head_count"): (9, (5, [64, 32, 32])),
                          a("attention.sliding_window_pattern"): (9, (7, [False, True, True])),
                          a("attention.indexer.types"): (9, (7, [True, False, False])),
                          a("swiglu_clamp_exp"): (9, (6, [0.0, 10.0, 10.0]))})


def source_with_block(path: pathlib.Path, **over) -> pathlib.Path:
    """One shard of a set that has the MTP block (block 3 after the 3 layers): a trunk tensor before and after the
    block's, the keys of model() with `over`."""
    a = lambda k: ARCH + "." + k  # noqa: E731
    kv = {"general.architecture": (8, ARCH), a("block_count"): (4, NL + 1), a("nextn_predict_layers"): (4, 1),
          a("attention.head_count"): (9, (5, [64, 32, 32, 16])), a("swiglu_clamp_exp"): (9, (6, [0.0, 10.0, 10.0, 7.0])),
          a("embedding_length"): (4, D), a("expert_count"): (4, 4), a("attention.kv_lora_rank"): (4, 64),
          a("engram.token_map"): (9, (5, list(range(5000))))}
    kv.update(over)
    gguf(path, kv, [("blk.2.attn_q_a.weight", 12, [D, 128]), ("blk.3.attn_q_a.weight", 30, [D, 128]),
                    ("blk.3.nextn.eh_proj.weight", 30, [2 * D, D]), ("blk.3.nextn.enorm.weight", 0, [D]),
                    ("blk.3.ffn_gate_exps.weight", 8, [D, 128, 4]), ("output_norm.weight", 0, [D])])
    return path


def block_file(src: pathlib.Path, out: pathlib.Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(MTP), "--src", str(src), "--out", str(out)], capture_output=True,
                          text=True)


class MtpFileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.base = pathlib.Path(self.tmp.name)
        self.root = public(self.base / "m")
        self.src = source_with_block(self.base / "src.gguf")

    def tearDown(self):
        self.tmp.cleanup()

    def read(self) -> tuple:
        out = self.root / "maya-hy4"
        geo = dict(line.split(" ", 1) for line in (out / "hy4.txt").read_text().splitlines())
        lines = (out / "tensors.txt").read_text().splitlines()
        files = [ln.split()[2] for ln in lines if ln.startswith("file ")]
        return geo, files, {ln.split()[1]: ln.split()[2:] for ln in lines if ln.startswith("t ")}

    def test_without_a_block(self):
        r = pack(self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        geo, files, _ = self.read()
        self.assertEqual((geo["n_layer"], geo["n_mtp"], geo["heads"], len(files)), ("3", "0", "64,32,32", 2))

    def test_the_block_file(self):
        """The writer: the block's tensors alone, their bytes and types as in the source; the source's hy_v4_flash keys
        but a vocabulary's list; general.type mtp."""
        r = block_file(self.src, self.root / "Hy4-Flash-mtp.gguf")
        self.assertEqual(r.returncode, 0, r.stderr)
        import importlib.util
        spec = importlib.util.spec_from_file_location("hy4_pack_gguf", PACK)   # (not the other engines' pack_gguf)
        pack_gguf = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pack_gguf)
        g, src = pack_gguf.Gguf(self.root / "Hy4-Flash-mtp.gguf"), pack_gguf.Gguf(self.src)
        self.assertTrue(pack_gguf.is_mtp(g))
        self.assertEqual([t[0] for t in g.tensors], ["blk.3.attn_q_a.weight", "blk.3.nextn.eh_proj.weight",
                                                     "blk.3.nextn.enorm.weight", "blk.3.ffn_gate_exps.weight"])
        self.assertEqual(g.kv[ARCH + ".attention.head_count"], [64, 32, 32, 16])
        self.assertEqual(g.kv[ARCH + ".block_count"], NL + 1)
        self.assertNotIn(ARCH + ".engram.token_map", g.kv)
        theirs = {t[0]: t for t in src.tensors}
        for name, typ, dims, off in g.tensors:
            self.assertEqual((typ, dims), tuple(theirs[name][1:3]))
            nb = pack_gguf.nbytes(typ, dims)
            with g.path.open("rb") as f, src.path.open("rb") as f2:
                f.seek(g.data_start + off)
                f2.seek(src.data_start + theirs[name][3])
                self.assertEqual(f.read(nb), f2.read(nb))
        self.assertIn("sha256", r.stdout)
        # a set without the block (or its other shard): refused
        self.assertNotEqual(block_file(self.root / "Hy4-flash-Q4_K_M-00001-of-00002.gguf", self.base / "x.gguf").returncode, 0)

    def test_found_beside_the_set(self):
        """The folder's block file: not a third shard; its tensors one more file (before the vision tower's), the
        geometry a block longer with the file's own values for its block."""
        block_file(self.src, self.root / "Hy4-Flash-mtp.gguf")
        mmproj(self.root / "mmproj-Hy4-flash-bf16.gguf")
        r = pack(self.root)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("+1 MTP: Hy4-Flash-mtp.gguf", r.stdout)
        geo, files, t = self.read()
        self.assertEqual((geo["n_layer"], geo["n_mtp"]), ("3", "1"))
        self.assertEqual(geo["heads"], "64,32,32,16")
        self.assertEqual(geo["swiglu_limit"], "0.0,10.0,10.0,7.0")
        self.assertEqual(geo["swa"], "0,1,1,0")
        self.assertEqual(geo["indexer"], "1,0,0,1")
        self.assertEqual(files[2:], ["Hy4-Flash-mtp.gguf", "mmproj-Hy4-flash-bf16.gguf"])
        typ, fi, off, nb, *dims = t["blk.3.nextn.eh_proj.weight"]
        self.assertEqual((typ, fi, dims), ("BF16", "2", [str(2 * D), str(D)]))
        with (self.root / files[2]).open("rb") as f:   # (the source's pattern: the tensor's index in ITS file)
            f.seek(int(off))
            self.assertEqual(f.read(int(nb)), bytes([2]) * int(nb))
        self.assertEqual(t["blk.3.ffn_gate_exps.weight"][0], "Q8_0")
        self.assertEqual(t["blk.2.attn_q_a.weight"][1], "1")   # (the trunk's own)
        # --files naming the set's shards (the setup's way): the block file is still found; --mtp none: left out
        r = pack(self.root, "--files", "Hy4-flash-Q4_K_M-00001-of-00002.gguf", "Hy4-flash-Q4_K_M-00002-of-00002.gguf",
                 "--mtp", "Hy4-Flash-mtp.gguf")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertEqual(self.read()[0]["n_mtp"], "1")
        self.assertEqual(pack(self.root, "--mtp", "none").returncode, 0)
        geo, files, t = self.read()
        self.assertEqual((geo["n_mtp"], geo["heads"], files[2:]), ("0", "64,32,32", ["mmproj-Hy4-flash-bf16.gguf"]))
        self.assertNotIn("blk.3.nextn.eh_proj.weight", t)

    def test_a_set_with_its_own_block(self):
        root = model(self.base / "own")
        block_file(self.src, root / "Hy4-Flash-mtp.gguf")
        r = subprocess.run([sys.executable, str(PACK), "--model", str(root)], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        lines = (root / "maya-hy4" / "tensors.txt").read_text().splitlines()
        self.assertEqual(len([ln for ln in lines if ln.startswith("file ")]), 2)
        self.assertIn("n_mtp 1", (root / "maya-hy4" / "hy4.txt").read_text())
        r = subprocess.run([sys.executable, str(PACK), "--model", str(root), "--mtp", "Hy4-Flash-mtp.gguf"],
                           capture_output=True, text=True)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("its own MTP block", r.stderr)

    def test_another_models_block(self):
        a = lambda k: ARCH + "." + k  # noqa: E731
        for i, (over, words) in enumerate((({a("block_count"): (4, NL + 2)}, "this set has 3"),
                                           ({a("embedding_length"): (4, 2 * D)}, "not this model's MTP block"))):
            src = source_with_block(self.base / ("s%d.gguf" % i), **over)
            for f in self.root.glob("*mtp*.gguf"):
                f.unlink()
            # (a block count of 5: the block is blk.4 - none in this source, so the writer itself refuses; the file
            #  is then written by hand)
            if block_file(src, self.root / "Hy4-Flash-mtp.gguf").returncode != 0:
                gguf(self.root / "Hy4-Flash-mtp.gguf",
                     {"general.architecture": (8, ARCH), "general.type": (8, "mtp"), a("block_count"): (4, NL + 2),
                      a("nextn_predict_layers"): (4, 1)}, [("blk.4.nextn.eh_proj.weight", 30, [2 * D, D])])
            r = pack(self.root)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn(words, r.stderr)
        r = pack(self.root, "--mtp", "Hy4-flash-Q4_K_M-00002-of-00002.gguf")   # (named: refused, not ignored)
        self.assertNotEqual(r.returncode, 0)
        self.assertIn("MTP block's file", r.stderr)


if __name__ == "__main__":
    unittest.main()

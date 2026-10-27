"""tools/mm3/pack_gguf.py on a synthetic MiniMax-M3 GGUF set (llama.cpp's minimax-m3 layout, two shards and the Maya
indexers side file, tiny dimensions): the geometry, the checkpoint's names, the experts an entry each at their bytes,
the side file's indexers, and the refusals (no side file, a shard missing).

    python -m unittest tools.test_mm3_pack_gguf
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
PACK = HERE / "mm3" / "pack_gguf.py"

# ggml type -> (values a block, bytes a block)
BLOCK = {0: (1, 4), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 16: (256, 66), 19: (256, 50)}
D, F, FD, E, NL, LEAD, VOCAB = 256, 256, 512, 4, 3, 1, 8


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
        return struct.pack("<" + {4: "I", 5: "i", 6: "f", 7: "?", 10: "Q"}[t], v)
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


def model(root: pathlib.Path, side: bool = True, side_ext: str = ".bin") -> pathlib.Path:
    m = root / "MiniMax-M3.1-Flash-Maya-XS"
    m.mkdir()
    a = "minimax-m3."
    kv = {"general.architecture": (8, "minimax-m3"), "general.name": (8, "MiniMax-M3.1-Flash Maya-XS"),
          a + "block_count": (4, NL), a + "context_length": (4, 1048576), a + "embedding_length": (4, D),
          a + "feed_forward_length": (4, FD), a + "attention.head_count": (4, 32), a + "attention.head_count_kv": (4, 2),
          a + "rope.freq_base": (6, 1e7), a + "attention.layer_norm_rms_epsilon": (6, 1e-6), a + "expert_count": (4, E),
          a + "expert_used_count": (4, 2), a + "attention.key_length": (4, 128), a + "attention.value_length": (4, 128),
          a + "expert_feed_forward_length": (4, F), a + "rope.dimension_count": (4, 64), a + "expert_shared_count": (4, 1),
          a + "expert_weights_scale": (6, 2.0), a + "attention.indexer.head_count": (4, 2),
          a + "attention.indexer.key_length": (4, 128), a + "attention.indexer.top_k": (4, 16),
          a + "attention.indexer.block_size": (4, 128), a + "attention.indexer.local_blocks": (4, 1),
          a + "leading_dense_block_count": (4, LEAD), "tokenizer.ggml.model": (8, "gpt2"),
          "tokenizer.ggml.tokens": (9, (8, ["a", "b", "ab", "]<]start of image[>[", "]~b]", "[e~[", "]<]image[>[",
                                            "]<]end of image[>["])),
          "tokenizer.ggml.token_type": (9, (5, [1, 1, 1, 3, 3, 3, 3, 3])), "tokenizer.ggml.merges": (9, (8, ["a b"])),
          "tokenizer.ggml.bos_token_id": (4, 4), "tokenizer.ggml.eos_token_id": (4, 5),
          "maya.recipe": (8, json.dumps({"tier": "Maya-XS"})), "split.count": (4, 2)}
    t0 = [("token_embd.weight", 12, [D, VOCAB])]
    layers = []
    for li in range(NL):
        L = "blk.%d." % li
        layers += [(L + "attn_norm.weight", 0, [D]), (L + "attn_q.weight", 13, [D, 32 * 128]),
                   (L + "attn_k.weight", 14, [D, 256]), (L + "attn_v.weight", 14, [D, 256]),
                   (L + "attn_output.weight", 13, [32 * 128, D]), (L + "attn_q_norm.weight", 0, [128]),
                   (L + "attn_k_norm.weight", 0, [128]), (L + "ffn_norm.weight", 0, [D])]
        if li >= LEAD:
            layers += [(L + "indexer.q_proj.weight", 8, [D, 256]), (L + "indexer.k_proj.weight", 8, [D, 128]),
                       (L + "indexer.q_norm.weight", 0, [128]), (L + "indexer.k_norm.weight", 0, [128]),
                       (L + "ffn_gate_inp.weight", 0, [D, E]), (L + "exp_probs_b.bias", 0, [E]),
                       (L + "ffn_gate_exps.weight", 19, [D, F, E]), (L + "ffn_up_exps.weight", 16, [D, F, E]),
                       (L + "ffn_down_exps.weight", 19, [F, D, E]), (L + "ffn_gate_shexp.weight", 13, [D, F]),
                       (L + "ffn_up_shexp.weight", 13, [D, F]), (L + "ffn_down_shexp.weight", 13, [F, D])]
        else:
            layers += [(L + "ffn_gate.weight", 13, [D, FD]), (L + "ffn_up.weight", 13, [D, FD]),
                       (L + "ffn_down.weight", 13, [FD, D])]
    tail = [("output_norm.weight", 0, [D]), ("output.weight", 14, [D, VOCAB])]
    half = len(layers) // 2
    gguf(m / "MiniMax-M3.1-Flash-Maya-XS-00001-of-00002.gguf", kv, t0 + layers[:half])
    gguf(m / "MiniMax-M3.1-Flash-Maya-XS-00002-of-00002.gguf", {"split.count": (4, 2)}, layers[half:] + tail)
    if side:
        ix = [("blk.%d.%s" % (li, n), t, dims) for li in range(LEAD) for n, t, dims in
              (("indexer.q_proj.weight", 8, [D, 256]), ("indexer.k_proj.weight", 8, [D, 128]),
               ("indexer.q_norm.weight", 0, [128]), ("indexer.k_norm.weight", 0, [128]))]
        gguf(m / ("MiniMax-M3.1-Flash-Maya-XS-maya-indexers" + side_ext),
             {"general.architecture": (8, "minimax-m3"), "maya.indexers.for": (8, "MiniMax-M3.1-Flash-Maya-XS")}, ix)
    return m


def safetensors(path: pathlib.Path, tensors: dict):
    """A .safetensors file of {name: (dtype, shape)}, each tensor's bytes its index."""
    head, off, data = {}, 0, b""
    for i, (name, (dt, shape)) in enumerate(tensors.items()):
        n = 1
        for x in shape:
            n *= x
        nb = n * {"BF16": 2, "F32": 4}[dt]
        head[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        data += bytes([i + 1]) * nb
        off += nb
    h = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + data)


def companions(root: pathlib.Path) -> pathlib.Path:
    """The EXL3 repo's vision tower and DSpark draft files in the setup's companions folder beside the model."""
    c = root / "MiniMax-M3.1-Flash-companions"
    (c / "dspark").mkdir(parents=True)
    (c / "config.json").write_text(json.dumps({"vision_config": {"hidden_size": 64, "num_hidden_layers": 2},
                                               "projector_hidden_size": D, "image_token_index": 6}))
    (c / "preprocessor_config.json").write_text(json.dumps({"image_mean": [0.5, 0.5, 0.5], "image_std": [0.2, 0.2, 0.2]}))
    safetensors(c / "model-vision-00001-of-00001.safetensors",
                {"vision_tower.vision_model.pre_layrnorm.weight": ("F32", [64]),
                 "multi_modal_projector.linear_1.weight": ("BF16", [D, 64])})
    safetensors(c / "dspark" / "model-dspark-00001.safetensors",
                {"language_model.model.dspark.fc.weight": ("BF16", [D, 2 * D])})
    (c / "dspark" / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"language_model.model.dspark.fc.weight": "model-dspark-00001.safetensors"}}))
    (c / "dspark" / "config.json").write_text(json.dumps({"num_hidden_layers": 5, "dspark_block_size": 7,
                                                          "dflash_config": {"sliding_window": 4096}}))
    return c


def run(m: pathlib.Path, *extra) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(PACK), "--model", str(m), *extra], capture_output=True, text=True)


def tensors(pack: pathlib.Path) -> tuple[list, dict]:
    files, t = [], {}
    for line in (pack / "tensors.txt").read_text().splitlines():
        p = line.split()
        if p[0] == "file":
            files.append(p[2])
        else:
            t[p[1]] = (p[2], int(p[3]), int(p[4]), int(p[5]), [int(x) for x in p[6:]])
    return files, t


class PackGguf(unittest.TestCase):
    def test_pack(self):
        with tempfile.TemporaryDirectory() as d:
            m = model(pathlib.Path(d))
            r = run(m)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            pack = m / "maya-mm3"
            kv = dict(line.split(" ", 1) for line in (pack / "mm3.txt").read_text().splitlines())
            for k, v in {"hidden_size": "256", "num_hidden_layers": "3", "num_key_value_heads": "2", "head_dim": "128",
                         "rotary_dim": "64", "partial_rotary_factor": "0.5", "vocab_size": "8", "num_local_experts": "4",
                         "intermediate_size": "256", "dense_intermediate_size": "512", "shared_intermediate_size": "256",
                         "swiglu_alpha": "1.702", "swiglu_limit": "7.0", "routed_scaling_factor": "2.0",
                         "moe_layer_freq": "0,1,1", "sparse_attention_config.sparse_attention_freq": "1,1,1",
                         "sparse_attention_config.sparse_topk_blocks": "16", "format": "gguf", "eos_token_id": "5",
                         "special.bos": "4", "quant_name": "MiniMax-M3.1-Flash-Maya-XS", "maya.tier": "Maya-XS",
                         "norms_plus_one": "1", "max_position_embeddings": "1048576"}.items():
                self.assertEqual(kv.get(k), v, k)
            files, t = tensors(pack)
            self.assertEqual(files[-1], "MiniMax-M3.1-Flash-Maya-XS-maya-indexers.bin")
            P = "language_model.model."
            self.assertEqual(t[P + "layers.0.self_attn.index_q_proj.weight"][0], "Q8_0")
            self.assertEqual(t[P + "layers.0.self_attn.index_q_proj.weight"][1], len(files) - 1)   # (the side file's)
            self.assertEqual(t[P + "layers.2.self_attn.q_proj.weight"][4], [32 * 128, D])
            self.assertEqual(t["language_model.lm_head.weight"][0], "Q6_K")
            self.assertEqual(t[P + "embed_tokens.weight"][4], [VOCAB, D])
            self.assertIn(P + "layers.0.mlp.down_proj.weight", t)
            # the experts: an entry each, expert e's bytes the e-th slice of the stacked tensor
            e0, e3 = t[P + "layers.1.block_sparse_moe.experts.0.w1.weight"], t[P + "layers.1.block_sparse_moe.experts.3.w1.weight"]
            self.assertEqual((e0[0], e0[4]), ("IQ1_S", [F, D]))
            self.assertEqual(e0[3], F * D // 256 * 50)
            self.assertEqual(e3[2] - e0[2], 3 * e0[3])
            self.assertEqual(t[P + "layers.1.block_sparse_moe.experts.2.w3.weight"][0], "IQ2_XXS")
            self.assertEqual(t[P + "layers.2.block_sparse_moe.experts.1.w2.weight"][4], [D, F])
            # each entry's bytes are its tensor's (the data pattern: the tensor's index in its file)
            fi, off, nb = e3[1], e3[2], e3[3]
            with (m / files[fi]).open("rb") as f:
                f.seek(off)
                blob = f.read(nb)
            self.assertEqual(len(set(blob)), 1)
            tok = json.loads((pack / "tokenizer" / "tokenizer.json").read_text())
            self.assertEqual(tok, {"pre": "minimax", "special_ids": {"bos": 4, "eos": 5}})
            self.assertTrue((pack / "tokenizer" / "chat_template.jinja").read_text(encoding="utf-8").startswith("{#- Project Maya"))

    def test_companions(self):
        """The EXL3 repo's vision tower and DSpark draft beside the GGUF: indexed with pack.py's keys, relative paths;
        --text-only leaves them out."""
        with tempfile.TemporaryDirectory() as d:
            m = model(pathlib.Path(d))
            companions(pathlib.Path(d))
            r = run(m)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            pack = m / "maya-mm3"
            kv = dict(line.split(" ", 1) for line in (pack / "mm3.txt").read_text().splitlines())
            for k, v in {"vision.hidden_size": "64", "vision.projector_hidden_size": str(D), "vision.image_start": "3",
                         "vision.image_token": "6", "vision.image_end": "7", "image_token_index": "6",
                         "dspark.num_hidden_layers": "5", "dspark.dflash.sliding_window": "4096"}.items():
                self.assertEqual(kv.get(k), v, k)
            files, t = tensors(pack)
            self.assertIn("../MiniMax-M3.1-Flash-companions/model-vision-00001-of-00001.safetensors", files)
            self.assertIn("../MiniMax-M3.1-Flash-companions/dspark/model-dspark-00001.safetensors", files)
            v = t["vision_tower.vision_model.pre_layrnorm.weight"]
            self.assertEqual((v[0], v[4]), ("F32", [64]))
            self.assertEqual(t["dspark.language_model.model.dspark.fc.weight"][4], [D, 2 * D])
            with (m / files[v[1]]).open("rb") as f:   # (the bytes where the entry says: the tensor's own)
                f.seek(v[2])
                self.assertEqual(f.read(v[3]), bytes([1]) * v[3])
            r = run(m, "--text-only")
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            self.assertNotIn("vision.hidden_size", (pack / "mm3.txt").read_text())
            self.assertNotIn("dspark.", (pack / "mm3.txt").read_text())

    def test_side_file_under_its_v1130_name(self):
        """Up to v1.1.30 the indexers' file was <name>-maya-indexers.gguf: still found as the side file."""
        with tempfile.TemporaryDirectory() as d:
            m = model(pathlib.Path(d), side_ext=".gguf")
            r = run(m)
            self.assertEqual(r.returncode, 0, r.stderr + r.stdout)
            files, _ = tensors(m / "maya-mm3")
            self.assertEqual(files[-1], "MiniMax-M3.1-Flash-Maya-XS-maya-indexers.gguf")

    def test_no_side_file(self):
        with tempfile.TemporaryDirectory() as d:
            m = model(pathlib.Path(d), side=False)
            r = run(m)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("maya-indexers.bin", r.stderr + r.stdout)
            self.assertIn("indexer of layer 0", r.stderr + r.stdout)

    def test_a_shard_missing(self):
        with tempfile.TemporaryDirectory() as d:
            m = model(pathlib.Path(d))
            (m / "MiniMax-M3.1-Flash-Maya-XS-00002-of-00002.gguf").unlink()
            r = run(m)
            self.assertNotEqual(r.returncode, 0)
            self.assertIn("1 of the 2 shards", r.stderr + r.stdout)


if __name__ == "__main__":
    unittest.main()

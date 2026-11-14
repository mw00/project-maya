"""Exercise old/new GLM vision formats through the built encoder, on the CPU."""
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import zlib

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from _paths import add_gguf_py
add_gguf_py()
import gguf

EXE = ROOT / "build-vision" / "bin" / ("maya-vision.exe" if sys.platform == "win32" else "maya-vision")


def vocab(path):
    w = gguf.GGUFWriter(path, "llama")
    # A byte-level BPE vocabulary plus the GLM vision boundary tokens.
    byte_chars = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    byte_chars += list(range(256, 256 + 256 - len(byte_chars)))
    tokens = [chr(c) for c in byte_chars] + ["<|begin_of_image|>", "<|end_of_image|>"]
    w.add_tokenizer_model("gpt2")
    w.add_tokenizer_pre("gpt-2")
    w.add_token_list(tokens)
    w.add_token_types([1] * 256 + [3, 3])
    w.add_token_merges(["a b"])
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def projector(path, modern):
    w = gguf.GGUFWriter(path, "clip")
    w.add_string("general.type", "mmproj")
    w.add_string("clip.projector_type", "glm5v" if modern else "glm5next")
    w.add_bool("clip.has_vision_encoder", True)
    w.add_bool("clip.use_silu", True)
    for k, v in {"image_size": 56, "patch_size": 14, "embedding_length": 32, "feed_forward_length": 64,
                 "projection_dim": 64, "block_count": 1, "attention.head_count": 2,
                 "spatial_merge_size": 2}.items():
        w.add_uint32("clip.vision." + k, v)
    w.add_float32("clip.vision.attention.layer_norm_epsilon", 1e-5)
    w.add_float32("clip.vision." + ("swiglu_clamp" if modern else "ffn_clamp"), 0.1)
    for bound in ("min", "max"):
        w.add_uint32(f"clip.vision.image_{bound}_" + ("pixels" if modern else "tokens"), 4 * 28 * 28 if modern else 4)
    w.add_array("clip.vision.image_mean", [0.5] * 3)
    w.add_array("clip.vision.image_std", [0.5] * 3)
    rng = np.random.default_rng(42)

    def mat(name, shape):
        w.add_tensor(name, rng.normal(0, 0.2, shape).astype(np.float16))

    def vec(name, size):
        w.add_tensor(name, np.full(size, 0 if name.endswith("bias") else 1, dtype=np.float32))

    for name in ("v.patch_embd.weight", "v.patch_embd.weight.1"):
        mat(name, (32, 3, 14, 14))
    vec("v.patch_embd.bias", 32)
    b = "v.blk.0."
    for name in ("ln1.weight", "ln2.weight"):
        vec(b + name, 32)
    mat(b + "attn_qkv.weight", (96, 32)); vec(b + "attn_qkv.bias", 96)
    for name in ("attn_q_norm.weight", "attn_k_norm.weight"):
        vec(b + name, 16)
    mat(b + "attn_out.weight", (32, 32)); vec(b + "attn_out.bias", 32)
    for role, shape in (("gate", (64, 32)), ("up", (64, 32)), ("down", (32, 64))):
        mat(b + f"ffn_{role}.weight", shape); vec(b + f"ffn_{role}.bias", shape[0])
    vec("v.post_ln.weight", 32)
    mat("mm.patch_merger.weight", (64, 32, 2, 2)); vec("mm.patch_merger.bias", 64)
    mat("mm.model.fc.weight", (64, 64))
    vec("mm.post_norm.weight", 64); vec("mm.post_norm.bias", 64)
    for role, shape in (("gate", (80, 64)), ("up", (80, 64)), ("down", (64, 80))):
        mat(f"mm.{role}.weight", shape)
    w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()


def picture(path):
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    pixels = np.random.default_rng(9).integers(0, 256, (56, 56, 3), dtype=np.uint8)
    raw = b"".join(b"\0" + row.tobytes() for row in pixels)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">2I5B", 56, 56, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


@unittest.skipUnless(EXE.is_file(), "build maya-vision to run encoder integration tests")
class GlmVisionFormats(unittest.TestCase):
    def test_legacy_and_modern_formats_encode_the_same_picture(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model, image = root / "vocab.gguf", root / "image.png"
            vocab(model); picture(image)
            embeddings = []
            for modern in (False, True):
                with self.subTest(format="glm5v" if modern else "glm5next"):
                    mmproj, out = root / f"vision-{modern}.gguf", root / f"out-{modern}.sve"
                    projector(mmproj, modern)
                    result = subprocess.run([str(EXE), "--model", str(model), "--mmproj", str(mmproj),
                                             "--no-warmup", "--threads", "2"],
                                            input=f"ENC {image} {out}\nQUIT\n", capture_output=True, text=True, timeout=30)
                    self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
                    self.assertTrue(result.stdout.startswith("READY 64\nOK 4 "), result.stdout + result.stderr)
                    data = out.read_bytes()
                    # GLM uses sequential decoder positions, rather than a Qwen M-RoPE grid.
                    self.assertEqual(struct.unpack("<5i", data[:20]), (0x31455653, 4, 4, 1, 64))
                    values = np.frombuffer(data[20:], dtype=np.float32)
                    self.assertTrue(np.isfinite(values).all())
                    self.assertGreater(np.linalg.norm(values), 0)
                    embeddings.append(values)
            self.assertEqual(len(embeddings), 2)
            np.testing.assert_array_equal(*embeddings)


if __name__ == "__main__":
    unittest.main()

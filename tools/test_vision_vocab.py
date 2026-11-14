"""Local vision vocabulary export: preserve tokenizer bytes and leave weights untouched."""
from pathlib import Path
import struct
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gguf_fix_arch import _str_raw as s, read_header
from vision_vocab import write_glm_vocab


def fixture(path, arch="glm5-next", tokenizer=True):
    fields = [("general.architecture", 8, s(arch)), ("split.count", 2, struct.pack("<H", 6))]
    if tokenizer:
        fields += [("tokenizer.ggml.model", 8, s("gpt2")),
                   ("tokenizer.ggml.pre", 8, s("glm4")),
                   ("tokenizer.ggml.tokens", 9, struct.pack("<IQ", 8, 3) + s("а") + s("bc") + s("<|image|>")),
                   ("tokenizer.ggml.scores", 9, struct.pack("<IQ3f", 6, 3, -1.0, 0.0, 0.5)),
                   ("tokenizer.ggml.token_type", 9, struct.pack("<IQ3i", 5, 3, 1, 1, 3)),
                   ("tokenizer.ggml.add_bos_token", 7, b"\1"),
                   ("tokenizer.ggml.eos_token_id", 4, struct.pack("<I", 2)),
                   ("tokenizer.chat_template", 8, s("{{ messages }}"))]
    # An unknown tensor type must not prevent extracting vocabulary metadata.
    info = s("unknown.weight") + struct.pack("<IQIQ", 1, 64, 255, 0)
    data = b"GGUF" + struct.pack("<IQQ", 3, 1, len(fields))
    data += b"".join(s(k) + struct.pack("<I", t) + raw for k, t, raw in fields) + info
    path.write_bytes(data + bytes(-len(data) % 32) + bytes(range(64)))


class VisionVocab(unittest.TestCase):
    def test_both_text_architectures_preserve_all_tokenizer_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for arch in ("glm5-next", "glm5next"):
                with self.subTest(arch=arch):
                    src, out = root / f"{arch}.gguf", root / "vision" / "vocab.gguf"
                    fixture(src, arch)
                    before = src.read_bytes()
                    self.assertEqual(write_glm_vocab(src, out), out)
                    old, new = read_header(src), read_header(out)
                    self.assertEqual(new.n_tensors, 0)
                    self.assertEqual(new.get("general.architecture"), (8, s("llama")))
                    self.assertEqual([(k, t, v) for k, t, v in old.kvs if k.startswith("tokenizer.")],
                                     [(k, t, v) for k, t, v in new.kvs if k.startswith("tokenizer.")])
                    self.assertIsNone(new.get("split.count"))
                    self.assertEqual(src.read_bytes(), before)
                    self.assertEqual(out.stat().st_size, new.data_start)
                    first = out.read_bytes()
                    write_glm_vocab(src, out)
                    self.assertEqual(out.read_bytes(), first)

    def test_links_never_rewrite_the_source_or_the_old_output_target(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, linked, out, target = [root / n for n in ("src.gguf", "link.gguf", "vocab.gguf", "old.gguf")]
            fixture(source)
            before = source.read_bytes()
            linked.symlink_to(source)
            target.write_bytes(b"keep this file")
            out.symlink_to(target)
            write_glm_vocab(linked, out)
            self.assertTrue(linked.is_symlink())
            self.assertFalse(out.is_symlink())
            self.assertEqual(source.read_bytes(), before)
            self.assertEqual(target.read_bytes(), b"keep this file")

    def test_invalid_architecture_or_missing_tokenizer_does_not_replace_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            src, out = Path(tmp) / "src.gguf", Path(tmp) / "out.gguf"
            out.write_bytes(b"keep")
            for arch, tokenizer in (("qwen4exp", True), ("glm5next", False)):
                fixture(src, arch, tokenizer)
                with self.assertRaises(ValueError):
                    write_glm_vocab(src, out)
                self.assertEqual(out.read_bytes(), b"keep")

    def test_export_cannot_replace_the_source_model(self):
        with tempfile.TemporaryDirectory() as tmp:
            source, linked = Path(tmp) / "source.gguf", Path(tmp) / "linked.gguf"
            fixture(source)
            before = source.read_bytes()
            linked.symlink_to(source)
            for src, out in ((source, source), (linked, source), (linked, linked)):
                with self.assertRaises(ValueError):
                    write_glm_vocab(src, out)
                self.assertEqual(source.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

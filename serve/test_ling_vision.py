"""serve/test_ling_vision.py - unit tests for Ling-3.2-air vision preprocessing (serve/ling_vision.py).

    python -m unittest serve.test_ling_vision -v
"""
import io
import struct
import tempfile
import unittest
from pathlib import Path

from serve import ling_vision as V


class Geometry(unittest.TestCase):
    def test_from_info_and_rows(self):
        g = V.Geometry.from_info(
            {
                "vision": "ling",
                "image_token": 157157,
                "image_start": 157158,
                "image_end": 157159,
                "vision_patch": 16,
                "vision_merge": 2,
                "vision_max_pixels": 4194304,
                "vision_min_pixels": 4096,
            }
        )
        self.assertEqual((g.factor, g.image_token, g.image_start, g.image_end), (32, 157157, 157158, 157159))
        self.assertEqual(g.rows(224, 320), 7 * 10)  # no newline row
        self.assertEqual(g.max_tokens, 64 * 64 + 2)
        d = V.Geometry()
        self.assertEqual((d.image_token, d.image_start, d.image_end, d.max_pixels), (157157, 157158, 157159, 4194304))

    def test_resize_bounds(self):
        g = V.Geometry()
        for h, w in ((1500, 2875), (10, 10), (5000, 3000), (33, 4000), (768, 1024)):
            with self.subTest(size=(w, h)):
                rh, rw = V.mm3_vision.smart_resize(h, w, g)
                self.assertEqual((rh % 32, rw % 32), (0, 0))
                self.assertLessEqual(rh * rw, g.max_pixels)
                self.assertGreaterEqual(rh * rw, g.min_pixels)
        self.assertEqual(V.mm3_vision.smart_resize(768, 1024, g), (768, 1024))

    def test_prepare(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("no Pillow")
        im = Image.new("RGBA", (300, 200), (255, 0, 0, 0))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        pic = V.prepare(buf.getvalue(), V.Geometry())
        self.assertEqual((pic.width, pic.height), (288, 192))
        self.assertEqual(pic.n, (192 // 32) * (288 // 32))
        self.assertEqual(pic.tokens, pic.n)
        self.assertEqual(len(pic.rgb), 288 * 192 * 3)
        self.assertEqual(pic.rgb[:3], b"\xff\xff\xff")


class Spans(unittest.TestCase):
    def test_expand_and_file(self):
        g = V.Geometry()
        d = Path(tempfile.mkdtemp())
        v = V.Vision(g, d, lambda s: b"")
        a = V.Picture(b"\x01" * (32 * 64 * 3), 64, 32, g.rows(32, 64))
        b = V.Picture(b"\x02" * (64 * 32 * 3), 32, 64, g.rows(64, 32))
        s, p, e = g.image_start, g.image_token, g.image_end
        ids, starts = v.expand([1, s, p, e, 2, s, p, e, 3], [a, b])
        self.assertEqual((a.n, b.n), (2, 2))
        self.assertEqual(ids, [1, s, p, p, e, 2, s, p, p, e, 3])
        self.assertEqual(starts, [2, 7])
        for bad_ids, pics in (
            ([1, s, p, e], [a, b]),
            ([1, s, p, e, s, p, e], [a]),
            ([1, p, e], [a]),
            ([s, p, 9], [a]),
        ):
            with self.assertRaises(ValueError, msg=bad_ids):
                v.expand(bad_ids, pics)
        f = v.write(d / "x.dsi", [a, b], starts)
        raw = f.read_bytes()
        self.assertEqual(raw[:8], struct.pack("<4si", b"MMI1", 2))
        self.assertEqual(struct.unpack_from("<qii", raw, 8), (2, 64, 32))
        self.assertEqual(struct.unpack_from("<qii", raw, 8 + 16 + len(a.rgb)), (7, 32, 64))



class LingModelsConfig(unittest.TestCase):
    def test_contexts_and_settings_offer_full_262k(self):
        import json
        from serve import models as M

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            tok_dir = d / "tokenizer"
            tok_dir.mkdir()
            (tok_dir / "vocab.json").write_text("{}", encoding="utf-8")
            gguf = d / "Ling-3.2-air-test.gguf"

            def gstr(x: str) -> bytes:
                b = x.encode("utf-8")
                return struct.pack("<Q", len(b)) + b

            hdr = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, 0, 2))
            hdr += gstr("general.architecture") + struct.pack("<I", 8) + gstr("bailingmoe3")
            hdr += gstr("bailingmoe3.context_length") + struct.pack("<I", 4) + struct.pack("<I", 262144)
            gguf.write_bytes(bytes(hdr))
            cfg_p = d / "maya-ling.json"
            cfg = {
                "exe": str(d / "maya-ling"),
                "model_name": "ling-3.2-air",
                "tokenizer": str(tok_dir / "vocab.json"),
                "gpu": [0],
                "args": ["--gguf", str(gguf), "--max-context", "32768", "--kv", "q8", "--drafting", "auto"],
                "installer": {"family": "ling", "quant": "test"},
            }
            cfg_p.write_text(json.dumps(cfg), encoding="utf-8")
            self.assertEqual(M.family(cfg), "ling")
            self.assertEqual(M.trained_context_of(cfg), 262144)
            cur = M.settings_of(cfg)
            self.assertEqual((cur["context"], cur["kv"], cur["draft"]), (32768, "q8", "on"))   # (auto shows as On)
            can = M.choices(cfg)
            self.assertEqual(can["kv"], ["f16", "q8"])

            # Non-Ling engines must NOT have their Models page context range changed by --model / --gguf
            for fam_exe, fam_name in (("maya-ik2", "ik2"), ("maya-hy4", "hy4"), ("maya-ds4", "deepseek"), ("maya", "qwen")):
                other_cfg = {
                    "exe": str(d / fam_exe),
                    "model_name": fam_name,
                    "args": ["--model", str(gguf), "--max-context", "32768"],
                    "installer": {"family": fam_name},
                }
                self.assertIsNone(M.trained_context_of(other_cfg), f"{fam_name} must not read Ling GGUF context_length")
            self.assertEqual(can["draft"], ["on", "none"])
            desc = M.describe(cfg_p, None)
            self.assertIn(262144, desc["contexts"])
            self.assertEqual(desc["context_range"]["max"], 262144)
            self.assertEqual(desc["context_range"]["trained"], 262144)
            upd = M.apply(cfg, {"context": 262144, "kv": "fp16", "draft": "none"})
            self.assertEqual(M.settings_of(upd)["context"], 262144)
            self.assertEqual(M.settings_of(upd)["kv"], "f16")
            self.assertEqual(M.settings_of(upd)["draft"], "none")

if __name__ == "__main__":
    unittest.main()

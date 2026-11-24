"""serve/hy4_vision.py: Hy4-flash's pictures - the resize (multiples of 32, within the processor's pixel bounds, Lanczos),
a picture's image tokens (a newline row after each row of merged patches), the spans the template's markers get, the
GENI file, and the template itself writing the markers for an image part.

    python -m unittest serve.test_hy4_vision -v
"""
import io
import struct
import tempfile
import unittest
from pathlib import Path

from serve import hy4_vision as V

ROOT = Path(__file__).resolve().parents[1]


class Geometry(unittest.TestCase):
    def test_from_info_and_rows(self):
        g = V.Geometry.from_info({"vision": "hy4", "image_token": 7, "image_start": 5, "image_end": 6,
                                  "vision_patch": 16, "vision_merge": 2, "vision_max_pixels": 2048 * 2048,
                                  "vision_min_pixels": 4096})
        self.assertEqual((g.factor, g.image_token, g.image_start, g.image_end), (32, 7, 5, 6))
        self.assertEqual(g.rows(224, 320), 7 * 11)          # (the reference's 77 rows for 320 x 224)
        self.assertEqual(g.max_tokens, 64 * 65 + 2)
        d = V.Geometry()
        self.assertEqual((d.image_token, d.image_start, d.image_end, d.max_pixels), (120120, 120118, 120119, 4194304))

    def test_resize_bounds(self):
        g = V.Geometry()
        for (h, w) in ((1500, 2875), (10, 10), (5000, 3000), (33, 4000), (768, 1024)):
            with self.subTest(size=(w, h)):
                rh, rw = V.mm3_vision.smart_resize(h, w, g)
                self.assertEqual((rh % 32, rw % 32), (0, 0))
                self.assertLessEqual(rh * rw, g.max_pixels)
                self.assertGreaterEqual(rh * rw, g.min_pixels)
        self.assertEqual(V.mm3_vision.smart_resize(768, 1024, g), (768, 1024))   # (already whole: kept)

    def test_prepare(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("no Pillow")
        im = Image.new("RGBA", (300, 200), (255, 0, 0, 0))   # transparent: white, not black
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        pic = V.prepare(buf.getvalue(), V.Geometry())
        self.assertEqual((pic.width, pic.height), (288, 192))
        self.assertEqual(pic.n, (192 // 32) * (288 // 32 + 1))
        self.assertEqual(pic.tokens, pic.n)                  # (the template writes the start and end tokens)
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
        self.assertEqual((a.n, b.n), (3, 4))
        self.assertEqual(ids, [1, s] + [p] * 3 + [e, 2, s] + [p] * 4 + [e, 3])
        self.assertEqual(starts, [2, 2 + 3 + 3])
        for bad_ids, pics in (([1, s, p, e], [a, b]), ([1, s, p, e, s, p, e], [a]), ([1, p, e], [a]), ([s, p, 9], [a])):
            with self.assertRaises(ValueError, msg=bad_ids):
                v.expand(bad_ids, pics)
        f = v.write(d / "x.dsi", [a, b], starts)
        raw = f.read_bytes()
        self.assertEqual(raw[:8], struct.pack("<4si", b"MMI1", 2))
        self.assertEqual(struct.unpack_from("<qii", raw, 8), (2, 64, 32))
        self.assertEqual(struct.unpack_from("<qii", raw, 8 + 16 + len(a.rgb)), (8, 32, 64))


class Template(unittest.TestCase):
    def test_image_part_markers(self):
        """The model's template writes start, pad, end for an image part (the server widens the pad)."""
        tpl = Path.home() / "models/Hy4-flash-GGUF/maya-hy4/tokenizer/chat_template.jinja"
        if not tpl.exists():
            self.skipTest("no Hy4-flash pack here")
        from serve.frontend import ChatTemplate
        ct = ChatTemplate(tpl)
        out = ct.render([{"role": "user", "content": [{"type": "image_url", "image_url": {"url": "data:x"}},
                                                      {"type": "text", "text": "what is this?"}]}],
                        add_generation_prompt=True)
        tag = ct.hy4_tag
        self.assertIn(f"<｜hy_image▁start{tag}｜><｜hy_image▁pad{tag}｜><｜hy_image▁end{tag}｜>what is this?", out)


if __name__ == "__main__":
    unittest.main()

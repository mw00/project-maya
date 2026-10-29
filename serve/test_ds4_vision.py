"""DeepSeek-V4.1's pictures: the resize plan (DeepSeek's image_processor.py), the spans, the GENI file, the template.

    python -m unittest serve.test_ds4_vision -v
"""
import io
import os
import struct
import tempfile
import unittest
from pathlib import Path

from serve import ds4_vision as DV

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("DS4_MODEL") or Path.home() / "models/DSV4")   # (the DeepSeek model's folder)
PACK = MODEL / "maya-ds4/tokenizer"
TEMPLATE = ROOT / "tools/ds4/chat_template.jinja"


class Plan(unittest.TestCase):
    def test_reference_plans(self):
        # (width, height) -> DeepSeek's plan_image_grid (n_h, n_w, best_height, best_width), from the reference code
        cases = {(800, 600): (15, 20, 602, 812), (100, 100): (13, 13, 546, 546), (4000, 3000): (27, 36, 1134, 1512),
                 (5000, 100): (3, 120, 112, 5012), (100, 5000): (120, 3, 5012, 112),
                 (1920, 1080): (23, 41, 966, 1708), (1036, 714): (17, 25, 714, 1036), (20000, 30): (1, 477, 42, 20006)}
        g = DV.Geometry()
        for (w, h), want in cases.items():
            with self.subTest(size=(w, h)):
                got = DV.plan(w, h, g)
                self.assertEqual(got, want)
                self.assertLessEqual(DV.span_tokens(got[0], got[1]), g.max_tokens)

    def test_prepare(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest("no Pillow")
        im = Image.new("RGBA", (300, 200), (255, 0, 0, 0))   # transparent: white, not black
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        pic = DV.prepare(buf.getvalue(), DV.Geometry())
        n_h, n_w, bh, bw = DV.plan(300, 200, DV.Geometry())
        self.assertEqual((pic.width, pic.height, pic.n_h, pic.n_w), (bw, bh, n_h, n_w))
        self.assertEqual(len(pic.rgb), bw * bh * 3)
        mid = ((bh // 2) * bw + bw // 2) * 3
        self.assertEqual(pic.rgb[mid:mid + 3], b"\xff\xff\xff")


class Spans(unittest.TestCase):
    def test_expand_and_file(self):
        g = DV.Geometry()
        v = DV.Vision(g, Path(tempfile.mkdtemp()), lambda s: b"")
        a = DV.Picture(b"\x01" * (28 * 42 * 3), 42, 28, 1, 1)
        b = DV.Picture(b"\x02" * (42 * 84 * 3), 84, 42, 1, 2)
        tok = g.image_token
        ids, starts = v.expand([5, tok, 6, tok, 7], [a, b])
        self.assertEqual(starts, [1, 1 + a.tokens + 1])
        self.assertEqual(ids, [5] + [tok] * a.tokens + [6] + [tok] * b.tokens + [7])
        with self.assertRaises(ValueError):
            v.expand([5, tok, 6], [a, b])
        with self.assertRaises(ValueError):
            v.expand([5, tok, tok, 6], [a])
        path = v.write(v.dir / "x.dsi", [a, b], starts)
        data = path.read_bytes()
        self.assertEqual(data[:8], struct.pack("<4si", b"DSI1", 2))
        self.assertEqual(struct.unpack_from("<qii", data, 8), (1, 42, 28))
        off = 8 + 16 + len(a.rgb)
        self.assertEqual(struct.unpack_from("<qii", data, off), (starts[1], 84, 42))
        self.assertEqual(len(data), off + 16 + len(b.rgb))


@unittest.skipUnless((PACK / "vocab.json").exists(), "no DeepSeek pack")
class Template(unittest.TestCase):
    def setUp(self):
        import sys
        sys.path.insert(0, str(ROOT / "tools"))
        import strata_tokenizer as ST
        from serve.frontend import ChatTemplate
        self.tok = ST.Tokenizer.from_dir(PACK)
        self.tpl = ChatTemplate(TEMPLATE)

    def test_deepseek_test_case(self):
        """encoding/tests/test_input_5.json -> test_output_5.txt (two pictures between text blocks)."""
        ref = MODEL / "encoding/tests/test_output_5.txt"
        if not ref.exists():
            self.skipTest("no encoding tests")
        msgs = [{"role": "system", "content": "You are a helpful vision assistant."},
                {"role": "user", "content": [
                    {"type": "text", "text": "请按“第一张、第二张”的顺序回答：第一张图"}, {"type": "image", "source": "a"},
                    {"type": "text", "text": "和第二张图"}, {"type": "image", "source": "b"},
                    {"type": "text", "text": "中分别是什么食材？它们通常食用的部位分别是什么？"}]}]
        self.assertEqual(self.tpl.render(msgs, reasoning_effort="max"), ref.read_text(encoding="utf-8").rstrip("\n"))

    def test_typed_placeholder_is_text(self):
        """The placeholder typed in a message is plain text, not a picture (the server's literal marks)."""
        from serve.frontend import literal_tags, mark_literals, unmark_literals
        tags = literal_tags(self.tok.special_tokens)
        msgs = [{"role": "user", "content": "what is " + DV.PLACEHOLDER + "?"}]
        marked, _, changed = mark_literals(msgs, None, tags)
        self.assertTrue(changed)
        prompt, plain = unmark_literals(self.tpl.render(marked), tags)
        self.assertNotIn(DV.Geometry().image_token, self.tok.encode(prompt, parse_special=True, plain=plain))
        self.assertIn(DV.Geometry().image_token, self.tok.encode(prompt, parse_special=True))


if __name__ == "__main__":
    unittest.main()

"""serve/mm3_vision.py: the resize (the model's image_processor.py smart_resize), the spans and the GENI file."""
import struct
import tempfile
import unittest
from pathlib import Path

from serve import mm3_vision as V


class SmartResize(unittest.TestCase):
    def test_known_sizes(self):
        g = V.Geometry()
        # (height, width) -> image_processor.py's (height, width)
        for (h, w), want in {(480, 640): (476, 644), (782, 1870): (420, 1036), (10, 10): (56, 56),
                             (672, 672): (672, 672), (700, 700): (672, 672), (28, 2000): (28, 1988)}.items():
            self.assertEqual(V.smart_resize(h, w, g), want, (h, w))

    def test_bounds(self):
        g = V.Geometry()
        for h in (1, 13, 99, 480, 1000, 4000):
            for w in (1, 17, 333, 640, 2500):
                if max(h, w) / min(h, w) > V.MAX_RATIO:
                    continue
                rh, rw = V.smart_resize(h, w, g)
                self.assertEqual((rh % 28, rw % 28), (0, 0))
                self.assertLessEqual(rh * rw, g.max_pixels)
                self.assertGreaterEqual(rh * rw, min(g.min_pixels, 28 * 28))

    def test_ratio(self):
        with self.assertRaises(ValueError):
            V.smart_resize(1, 500, V.Geometry())


class Spans(unittest.TestCase):
    def setUp(self):
        self.g = V.Geometry()
        self.v = V.Vision(self.g, Path(tempfile.mkdtemp()), lambda s: b"")
        self.pics = [V.Picture(b"\0" * (56 * 84 * 3), 84, 56, 2 * 3), V.Picture(b"\1" * (28 * 28 * 3), 28, 28, 1)]

    def test_expand(self):
        g = self.g
        ids, starts = self.v.expand([1, g.image_token, 2, g.image_token, 3], self.pics)
        self.assertEqual(ids, [1, g.image_start] + [g.image_token] * 6 + [g.image_end, 2, g.image_start,
                                                                           g.image_token, g.image_end, 3])
        self.assertEqual(starts, [2, 11])
        self.assertEqual(self.pics[0].tokens, 8)

    def test_mismatch(self):
        with self.assertRaises(ValueError):
            self.v.expand([self.g.image_token], self.pics)
        with self.assertRaises(ValueError):
            self.v.expand([self.g.image_token] * 3, self.pics)

    def test_write(self):
        path = self.v.write(self.v.dir / "x.mmi", self.pics, [2, 11])
        b = path.read_bytes()
        self.assertEqual(b[:8], struct.pack("<4si", b"MMI1", 2))
        self.assertEqual(struct.unpack_from("<qii", b, 8), (2, 84, 56))
        off = 8 + 16 + 56 * 84 * 3
        self.assertEqual(struct.unpack_from("<qii", b, off), (11, 28, 28))
        self.assertEqual(len(b), off + 16 + 28 * 28 * 3)

    def test_from_info(self):
        g = V.Geometry.from_info({"vision": "mm3", "image_token": 7, "image_start": 8, "image_end": 9,
                                  "vision_max_pixels": 1000})
        self.assertEqual((g.image_token, g.image_start, g.image_end, g.max_pixels, g.patch), (7, 8, 9, 1000, 14))


if __name__ == "__main__":
    unittest.main()

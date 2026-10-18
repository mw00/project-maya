"""tools/exl3_pack.py's EXL3 decoder (the --check reference) against exllamav3 itself: the fixtures in
data/exl3-exllamav3 (tools/exl3_exllamav3_fixture.py) hold random tiles packed by exllamav3's packer, the state of each
step as its unpacker reads it and W' as its reconstruct decodes it - every state and every weight must be the same bits,
exllamav3 1.6's half-integer K (1.5 / 2.5 / 3.5) included.  Also the packer's bitrate rules for exl3.txt.
Run: .venv/bin/python -m unittest discover -s tools -p test_exl3_exllamav3.py"""
from pathlib import Path
import struct
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too
import exl3_pack  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[1] / "data" / "exl3-exllamav3"


def read_fixture(path: Path) -> dict:
    """An EXH1 fixture: its header and the trellis words, states and W' it holds."""
    b = path.read_bytes()
    if b[:4] != b"EXH1":
        raise ValueError("%s: not an EXH1 fixture" % path)
    k2, ki, nj, cb, words = struct.unpack_from("<iiiii", b, 4)
    if words != 8 * k2:
        raise ValueError("%s: %d words a tile for k2 %d" % (path, words, k2))
    o = 24
    n_tr, n_st, n_w = ki * nj * words * 2, ki * nj * 256 * 2, ki * nj * 256 * 2
    if len(b) != o + n_tr + n_st + n_w:
        raise ValueError("%s: %d bytes, want %d" % (path, len(b), o + n_tr + n_st + n_w))
    tr = np.frombuffer(b, dtype="<i2", count=ki * nj * words, offset=o).reshape(ki, nj, words)
    st = np.frombuffer(b, dtype="<u2", count=ki * nj * 256, offset=o + n_tr).reshape(ki, nj, 256)
    wh = np.frombuffer(b, dtype="<u2", count=ki * nj * 256, offset=o + n_tr + n_st).reshape(ki * 16, nj * 16)
    return {"k2": k2, "ki": ki, "nj": nj, "cb": cb, "trellis": tr, "states": st, "w_hat": wh}


def decode_w_hat(f: dict) -> np.ndarray:
    """W' [16 ki][16 nj] as FP16 bits through exl3_pack's decoder, its tile layout included."""
    ki, nj = f["ki"], f["nj"]
    st = exl3_pack.exl3_states(np.ascontiguousarray(f["trellis"]).view(np.uint32), f["k2"])
    vals = exl3_pack.exl3_values(st, f["cb"])
    tile = np.empty(vals.shape, dtype=np.float32)
    tile[..., exl3_pack._ROW * 16 + exl3_pack._COL] = vals
    w = tile.reshape(ki, nj, 16, 16).transpose(0, 2, 1, 3).reshape(ki * 16, nj * 16)
    return w.astype(np.float16).view(np.uint16)


class Exllamav3FixtureTests(unittest.TestCase):
    def fixtures(self):
        files = sorted(FIXTURES.glob("exl3_k2_*_cb*.bin"))
        self.assertTrue(files, "no fixtures in %s (tools/exl3_exllamav3_fixture.py writes them)" % FIXTURES)
        return files

    def test_states_match_exllamav3(self):
        for path in self.fixtures():
            f = read_fixture(path)
            with self.subTest(fixture=path.name):
                got = exl3_pack.exl3_states(np.ascontiguousarray(f["trellis"]).view(np.uint32), f["k2"])
                diff = int(np.count_nonzero(got.astype(np.uint16) != f["states"]))
                self.assertEqual(diff, 0, "%d of %d states differ from exllamav3's unpack_trellis" % (diff, got.size))

    def test_w_hat_matches_exllamav3(self):
        for path in self.fixtures():
            f = read_fixture(path)
            with self.subTest(fixture=path.name):
                got = decode_w_hat(f)
                diff = int(np.count_nonzero(got != f["w_hat"]))
                self.assertEqual(diff, 0, "%d of %d weights differ from exllamav3's reconstruct" % (diff, got.size))

    def test_every_bitrate_and_codebook_covered(self):
        have = {(read_fixture(p)["k2"], read_fixture(p)["cb"]) for p in self.fixtures()}
        want = {(2 * K, 2) for K in range(1, 9)} | {(6, 0), (6, 1)} | {(3, 2), (5, 2), (7, 2)}
        self.assertTrue(want <= have, "fixtures missing for (k2, cb) %s" % sorted(want - have))


class PackBitrateTests(unittest.TestCase):
    """exl3_pack's words a tile -> k2, and exl3.txt's bits field (what the engine's exl3_k2_field reads back)."""

    def test_integer_and_half_integer_widths(self):
        for K in range(1, 9):
            self.assertEqual(exl3_pack.tile_k2(16 * K, 0, "x"), 2 * K)
            self.assertEqual(exl3_pack.k_text(2 * K), str(K))
        for words, k2, text in ((24, 3, "1.5"), (40, 5, "2.5"), (56, 7, "3.5")):
            self.assertEqual(exl3_pack.tile_k2(words, 2, "x"), k2)
            self.assertEqual(exl3_pack.k_text(k2), text)

    def test_refused_widths(self):
        with self.assertRaisesRegex(SystemExit, "mul1"):
            exl3_pack.tile_k2(56, 1, "a mcg K 3.5")   # exllamav3 writes a half-integer K with mul1 only
        for words in (8, 20, 72, 144):
            with self.assertRaisesRegex(SystemExit, "no EXL3 bitrate"):
                exl3_pack.tile_k2(words, 2, "x")


if __name__ == "__main__":
    unittest.main()

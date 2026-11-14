"""PLE convolution packs must load as F16 regardless of the GGUF's float storage."""
import contextlib
import io
import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import iq_pack


class PleConvolutionPack(unittest.TestCase):
    def pack(self, storage, raw):
        name = "blk.1.ple_conv1d.weight"
        tensor = SimpleNamespace(name=name, type_name=storage, shape=(2, 2), offset=0,
                                 expected_bytes=lambda: len(raw))
        model = SimpleNamespace(where={name: (SimpleNamespace(data_start=0), tensor,
                                               np.frombuffer(raw, dtype=np.uint8), None)})
        with tempfile.TemporaryDirectory() as temp:
            out = pathlib.Path(temp)
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(iq_pack.index_standalone(pathlib.Path("model.gguf"), out, model), 0)
            row = next(line.split() for line in (out / "index.txt").read_text().splitlines()
                       if not line.startswith("#"))
            return row, (out / "dense.bin").read_bytes()

    def test_f32_is_narrowed_by_the_loader(self):
        raw = np.array([0.125, -0.33, 1e-7, 0], dtype="<f4").tobytes()
        row, dense = self.pack("F32", raw)
        self.assertEqual((row[2], int(row[4]), int(row[6])), ("3", 16, 8))
        self.assertEqual(dense[:16], raw)

    def test_f16_remains_f16(self):
        raw = np.array([0.125, -0.33, 1e-7, 0], dtype="<f2").tobytes()
        row, dense = self.pack("F16", raw)
        self.assertEqual((row[2], int(row[4]), int(row[6])), ("5", 8, 8))
        self.assertEqual(dense[:8], raw)

    def test_bf16_is_converted_to_f16(self):
        values = np.array([0.125, -0.5, 2, 0], dtype="<f4")
        raw = (values.view("<u4") >> 16).astype("<u2").tobytes()
        row, dense = self.pack("BF16", raw)
        self.assertEqual((row[2], int(row[4]), int(row[6])), ("5", 8, 8))
        self.assertEqual(dense[:8], values.astype("<f2").tobytes())


if __name__ == "__main__":
    unittest.main()

"""Small CPU-only checks for the DeepSeek GGUF oracle's Engram row reader."""
import pathlib
import sys
import unittest
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "tools" / "ds4"))
import ref_gguf


class EngramRowsTest(unittest.TestCase):
    def test_reference_forward_reads_iq4_nl(self):
        raw = np.concatenate([np.concatenate((np.array([s], np.float16).view(np.uint8),
                                               np.full(16, 0x88, np.uint8))) for s in (1., 2., 4.)])
        oracle = ref_gguf.RefGguf.__new__(ref_gguf.RefGguf)
        oracle.st = SimpleNamespace(t={"layers.1.engram.embed.weight": ("IQ4_NL",)}, raw=lambda _: raw)
        oracle.lay = SimpleNamespace(head_dim=32)
        oracle.dim, oracle.hc, oracle.eps, oracle.dev = 1, 1, 0.0, "cpu"
        oracle.W = lambda _: torch.ones((2, 32))
        oracle.V = lambda _: torch.ones(1)
        out = oracle.engram(1, torch.ones((1, 1, 1)), np.array([[2]]))
        expected = 1 + torch.sigmoid(torch.tensor(1.0)) * 128
        torch.testing.assert_close(out.flatten(), expected.reshape(1))

    def test_iq4_nl_rows_and_order(self):
        # An IQ4_NL block with all nibbles 8 dequantizes to its FP16 scale.
        rows = []
        for scale in (1.0, 2.0, 4.0):
            rows.append(np.concatenate((np.array([scale], np.float16).view(np.uint8),
                                        np.full(16, 0x88, np.uint8))))
        raw = np.concatenate(rows)
        got = ref_gguf.engram_row_values(raw, "IQ4_NL", np.array([[2, 0], [1, 2]]), 32, "cpu", "test")
        expected = np.repeat(np.array([[4, 1], [2, 4]], np.float32)[..., None], 32, axis=2)
        np.testing.assert_array_equal(got.numpy(), expected)

    def test_q5_and_q6_zero_rows(self):
        for kind, width in (("Q5_K", 176), ("Q6_K", 210)):
            with self.subTest(kind=kind):
                raw = np.zeros(2 * width, dtype=np.uint8)
                got = ref_gguf.engram_row_values(raw, kind, np.array([[1, 0]]), 256, "cpu", "test")
                np.testing.assert_array_equal(got.numpy(), np.zeros((1, 2, 256), np.float32))

    def test_mxfp8_rows(self):
        # E8M0 scale 127 and FP8 E4M3 value 0x38 represent 1.0.
        raw = np.array([127] + [0x38] * 32, dtype=np.uint8)
        got = ref_gguf.engram_row_values(raw, "MXFP8", np.array([[0]]), 32, "cpu", "test")
        np.testing.assert_array_equal(got.numpy(), np.ones((1, 1, 32), np.float32))

    def test_unsupported_type_and_misaligned_width_fail(self):
        with self.assertRaisesRegex(ValueError, "unsupported"):
            ref_gguf.engram_row_values(np.zeros(32, np.uint8), "INVALID", np.array([[0]]), 32, "cpu", "test")
        with self.assertRaisesRegex(ValueError, "multiple"):
            ref_gguf.engram_row_values(np.zeros(18, np.uint8), "IQ4_NL", np.array([[0]]), 33, "cpu", "test")


if __name__ == "__main__":
    unittest.main()

"""Contract for the fused per-row KDA staging gather (DFMULTI rows path).

The rows path stages each row's six KDA projection outputs (q/k/v proj, f_a,
g_a, beta) into S->row_kin with six cudaMemcpyAsync calls per row per layer
(~2,160 tiny D2D copies per 8-row window). The fused gather must replace them
with one kernel launch while staying bit-for-bit identical: same destination
layout ((row * n_kda + kin_layer) * kin_layer_floats), same six sources.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
HEADER = (ROOT / "include/strata/kernels/glm_fast.hpp").read_text()
CUDA = (ROOT / "src/kernels/cuda/glm_fast.cu").read_text()
ROWS = (ROOT / "src/core/glm_rows.cu").read_text()


class KdaGatherContract(unittest.TestCase):
    def test_fused_gather_kernel_exists(self):
        self.assertIn("kda_gather", CUDA,
                      "missing the fused per-row KDA staging gather kernel")

    def test_kernel_copies_all_six_pieces_from_one_launch(self):
        start = CUDA.index("kda_gather_kernel")
        body = CUDA[start:start + 3000]
        self.assertIn("for (int p = 0; p < 6; ++p)", body,
                      "the fused gather must sweep all six pieces in one kernel loop")
        self.assertIn("widths[6]", body,
                      "the six piece widths must be carried in the gather args")

    def test_rows_path_uses_the_fused_gather(self):
        start = ROWS.index("bool Glm5Model::rows_layers")
        end = ROWS.index("bool Glm5Model::rows_head", start)
        body = ROWS[start:end]
        self.assertIn("gf::kda_gather", body,
                      "rows_layers must call the fused gather instead of six cudaMemcpyAsync per row")
        # the six-copy loop may survive only behind the A/B gate; the fused gather must be the default branch
        if "cudaMemcpyAsync(dst + off, src[j]" in body:
            gate = body.index("kda_gather_off")
            old_loop = body.index("cudaMemcpyAsync(dst + off, src[j]")
            fused = body.index("gf::kda_gather")
            self.assertTrue(old_loop > gate > 0 and fused > old_loop,
                            "the old six-copy staging must sit inside the kda_gather_off gate, fused gather the default")

    def test_gather_declared_in_the_public_header(self):
        self.assertIn("kda_gather", HEADER,
                      "the fused gather must be exported next to kda_prep/kda_rec")


if __name__ == "__main__":
    unittest.main()

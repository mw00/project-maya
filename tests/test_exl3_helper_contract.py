"""Contract for the EXL3 trellis helper (GPU-1 expert cache for the EXL3 pack).

The GLM helper pipeline (device claim -> ring -> helper compute -> pinned answer
-> moe_cpu_wait combine) is type-agnostic; the EXL3 pack was refused at three
gates. The trellis helper lifts them: helper-owned EXL3 experts compute on the
helper GPU with the same exl3::moe kernels the primary uses (bit-identical by
construction), the device waits for the helper's answer on EXL3 layers too, and
disk-only EXL3 experts stage through the nine-piece x3 reader.
"""
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]
FAST_PATH = (ROOT / "src/core/glm_fast_path.cu").read_text()
REMOTE = (ROOT / "src/core/remote_experts.cpp").read_text()
REMOTE_HPP = (ROOT / "include/strata/core/remote_experts.hpp").read_text()


class Exl3HelperContract(unittest.TestCase):
    def test_setup_accepts_exl3_packs(self):
        setup = FAST_PATH[FAST_PATH.index("bool Glm5Model::fast_helper_setup"):]
        setup = setup[:setup.index("\nbool Glm5Model::", 10)]
        self.assertNotIn("|| exl3_", setup,
                         "fast_helper_setup must no longer refuse EXL3 packs")

    def test_setup_stages_disk_only_exl3_experts_through_x3_reader(self):
        setup = FAST_PATH[FAST_PATH.index("bool Glm5Model::fast_helper_setup"):]
        setup = setup[:setup.index("\nbool Glm5Model::", 10)]
        self.assertIn("fast_read_expert_x3", setup,
                      "disk-only EXL3 candidates must stage via the nine-piece x3 reader")

    def test_compute_has_an_exl3_branch_with_the_real_kernels(self):
        compute = REMOTE[REMOTE.index("bool GlmExpertHelper::compute"):]
        compute = compute[:compute.index("\nvoid GlmExpertHelper::close")]
        self.assertIn("x3::moe_gate_up", compute,
                      "the helper compute must run the EXL3 moe kernels for trellis blobs")
        self.assertIn("x3::moe_down", compute,
                      "the helper compute must run the EXL3 down kernel")
        self.assertIn("x3::kTypeEXL3", compute,
                      "the EXL3 branch must be selected by the EXL3 weight type")

    def test_compute_takes_the_expert_layout(self):
        self.assertIn("const strata::kernels::exl3::ExpertLayout* xl", REMOTE_HPP,
                      "compute must receive the layer's EXL3 blob layout (suh/svh/trellis offsets)")

    def test_exl3_layers_wait_for_the_helper_answer(self):
        wait = FAST_PATH[FAST_PATH.index("bool Glm5Model::fast_moe_experts"):]
        wait = wait[:wait.index("\nbool Glm5Model::fast_layers")]
        self.assertIn("helper.device() >= 0", wait,
                      "EXL3 layers have no CPU lane: moe_cpu_wait must also fire when the helper is on")

    def test_exl3_layers_skip_the_gguf_native_fmt(self):
        setup = FAST_PATH[FAST_PATH.index("bool Glm5Model::fast_helper_setup"):]
        setup = setup[:setup.index("\nbool Glm5Model::", 10)]
        self.assertIn("!ly.exl3", setup,
                      "native_fmt is GGUF-only: EXL3 layers must skip it (their cpu_fmt stays empty)")


if __name__ == "__main__":
    unittest.main()

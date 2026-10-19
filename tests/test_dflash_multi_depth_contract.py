"""Guard DFMULTI against single-token target verification.

Run with: python3 tests/test_dflash_multi_depth_contract.py
A GPU integration test must additionally verify per-context rollback/commit and token semantics.
"""
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (ROOT / "src/program/generate.cpp").read_text()
START = SOURCE.index("// In DFMULTI, draft proposals remain conversation-local")
END = SOURCE.index("Gen* only = gens.size()", START)
DFMULTI_STEP = SOURCE[START:END]


class DflashMultiDepthContract(unittest.TestCase):
    def test_multi_request_verification_trace_requires_depth_greater_than_one(self):
        """A DFMULTI trace must report actual multi-token target windows, not token rows."""
        self.assertIn("window_depth=", DFMULTI_STEP,
                      "DFMULTI currently verifies one pending token per context; no target window depth is recorded")
        self.assertIn("verification_positions=", DFMULTI_STEP)
        self.assertIn("emitted_positions=", DFMULTI_STEP)
        self.assertIn("accepted_drafts=", DFMULTI_STEP)
        self.assertIn("batch_contexts=", DFMULTI_STEP)
        self.assertNotIn("target_rows=%zu drafted=%d verified=%d accepted=%d", DFMULTI_STEP,
                         "single-token target-row trace must be replaced by window verification evidence")


if __name__ == "__main__":
    unittest.main()

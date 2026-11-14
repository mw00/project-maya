"""HTTP vision-cache regressions with a protocol encoder and a checking mock model."""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class VisionCacheHTTP(unittest.TestCase):
    def test_request_ownership_and_missing_cache_recovery(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory(prefix="maya-vision-http-") as directory:
            output = Path(directory) / "replay"
            run = subprocess.run(
                [sys.executable, "-B", str(root / "tools/vision_cache_e2e.py"), "--out", str(output)],
                cwd=root, capture_output=True, text=True, timeout=60,
            )
            receipt = output / "receipt.json"
            self.assertTrue(receipt.is_file(), run.stdout + run.stderr)
            result = json.loads(receipt.read_text())
            self.assertEqual(len(result["cases"]), 14)
            self.assertTrue(result["passed"], json.dumps(result["cases"], indent=2))
            self.assertEqual(run.returncode, 0, run.stdout + run.stderr)


if __name__ == "__main__":
    unittest.main()

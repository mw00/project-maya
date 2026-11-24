"""The llama.cpp unpack on Windows: a file whose whole path passes 260 characters inside third_party/_unpack still
unpacks, and lands in third_party/llama.cpp."""
import os
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import setup  # noqa: E402


class LongPath(unittest.TestCase):
    def test_long_path_form(self):
        p = setup.long_path(Path("x") / "y")
        if os.name != "nt":
            self.assertEqual(p, Path("x") / "y")
            return
        self.assertTrue(str(p).startswith(chr(92) * 2 + "?" + chr(92)), p)
        self.assertEqual(setup.long_path(p), p)                                    # (already in that form: unchanged)

    def test_unpack_past_260(self):
        with tempfile.TemporaryDirectory() as t:
            root = Path(t) / "project-maya"
            (root / "third_party").mkdir(parents=True)
            top = "llama.cpp-" + setup.LLAMA_CPP_COMMIT
            deep = "tools/ui/src/" + "/".join(["ChatAttachmentsPreviewCurrentItem"] * 4) + "/Unavailable.svelte"
            src = Path(t) / "src.zip"
            with zipfile.ZipFile(src, "w") as z:
                z.writestr(f"{top}/ggml/CMakeLists.txt", "x")
                z.writestr(f"{top}/gguf-py/README.md", "x")
                z.writestr(f"{top}/{deep}", "deep")
            whole = len(str(root / "third_party" / "_unpack" / top / deep))
            self.assertGreater(whole, 260)                                         # (the case that failed)

            def fake_download(url, dst, what, *a, **k):
                Path(dst).write_bytes(src.read_bytes())

            with patch.object(setup, "ROOT", root), patch.object(setup, "download", fake_download):
                llama = setup.get_llama_cpp()
            self.assertTrue((llama / "ggml" / "CMakeLists.txt").exists())
            got = setup.long_path(llama / deep)
            self.assertEqual(got.read_text(), "deep")
            self.assertFalse(setup.long_path(root / "third_party" / "_unpack").exists())


if __name__ == "__main__":
    unittest.main()

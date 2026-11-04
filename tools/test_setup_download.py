"""setup.py's llama.cpp source download: a damaged zip is never kept as done, and a git checkout is never deleted."""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import setup as S  # noqa: E402


class LlamaCppSource(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        (self.root / "third_party").mkdir()
        self.root_patch = patch.object(S, "ROOT", self.root)
        self.root_patch.start()

    def tearDown(self):
        self.root_patch.stop()
        self.tmp.cleanup()

    def test_a_cut_zip_is_removed_with_its_done_mark(self):
        def cut(url, dst, what=None):            # a server that ended early with no Content-Length
            dst.write_bytes(b"PK\x03\x04 not a whole zip")
            S.mark(dst)
        with patch.object(S, "download", cut), patch.object(S, "say"), self.assertRaises(SystemExit):
            S.get_llama_cpp()
        self.assertEqual(sorted(p.name for p in (self.root / "third_party").iterdir()), [])

    def test_a_git_checkout_missing_pieces_is_kept(self):
        llama = self.root / "third_party" / "llama.cpp"
        (llama / ".git").mkdir(parents=True)
        (llama / "mine.txt").write_text("local work")
        with patch.object(S, "download") as dl, patch.object(S, "say"), self.assertRaises(SystemExit):
            S.get_llama_cpp()
        dl.assert_not_called()
        self.assertEqual((llama / "mine.txt").read_text(), "local work")


if __name__ == "__main__":
    unittest.main()

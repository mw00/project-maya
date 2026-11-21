"""tools/strata_tokenizer.py's corruption reporting: the token tables are built once at load and never written
afterwards, so when an encode/decode failure finds them changed, the error says it is heap corruption in the
process (a restart is the fix) instead of leaving a bare KeyError that reads like a broken vocabulary file.

    python -m unittest tools.test_strata_tokenizer
"""
import pathlib
import sys
import unittest

ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "tools"))

import strata_tokenizer as ST  # noqa: E402


def small() -> ST.Tokenizer:
    return ST.Tokenizer(["a", "b", "ab"], ["a b"], pre="qwen35")


class CorruptionNote(unittest.TestCase):
    def test_untouched_tables_report_nothing(self):
        t = small()
        self.assertEqual(t.corruption_note(), "")
        self.assertEqual(t.encode("ab"), [2])
        self.assertEqual(t.decode([2]), "ab")

    def test_a_lost_entry_is_named_as_corruption(self):
        t = small()
        del t.ids["ab"]                       # the 2026-11-16 shape: the entry vanished from memory
        with self.assertRaises(KeyError) as cm:
            t.encode("ab")
        self.assertIn("changed in memory", str(cm.exception))
        self.assertIn("restart", str(cm.exception))

    def test_a_damaged_token_string_is_named_as_corruption(self):
        t = small()
        t.tokens[2] = "ab\0"                  # and a table string gained a NUL
        del t.ids["ab"]
        with self.assertRaises(KeyError) as cm:
            t.encode("ab")
        self.assertIn("heap corruption", str(cm.exception))

    def test_decoding_a_damaged_token_is_named_as_corruption(self):
        t = small()
        t.tokens[0] = "\0a"
        with self.assertRaises(KeyError) as cm:
            t.token_bytes(0)
        self.assertIn("changed in memory", str(cm.exception))

    def test_out_of_range_id_with_untouched_tables_is_plain(self):
        t = small()
        with self.assertRaises(IndexError) as cm:
            t.token_bytes(9)
        self.assertNotIn("corruption", str(cm.exception))
        self.assertIn("outside the vocabulary", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

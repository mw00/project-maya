"""CPU checks for the parallel hardware harness's request flags and progress gate."""
import io
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ds4.parallel_test import Multi  # noqa: E402


class BusyCopyHarness(unittest.TestCase):
    def setUp(self):
        self.engine = Multi('unused', [], {})
        self.engine.multi = 2
        self.engine.proc = SimpleNamespace(stdin=io.BytesIO())

    def test_nonspeculative_check_disables_lookup_too(self):
        self.engine.send([11, 22], 8)
        self.assertEqual(self.engine.proc.stdin.getvalue(), b'GEN 8 rid=1 spec=0 suffix_draft=0 11,22\n')

    def test_lookup_regression_can_enable_lookup_with_dspark_off(self):
        self.engine.send([11], 8, lookup=1)
        self.assertIn(b'spec=0 suffix_draft=1 ', self.engine.proc.stdin.getvalue())

    def test_speculative_measurement_keeps_engine_lookup_default(self):
        self.engine.send([11], 8, spec=1)
        self.assertEqual(self.engine.proc.stdin.getvalue(), b'GEN 8 rid=1 spec=1 11\n')

    def test_progress_is_observed_from_request_tokens(self):
        self.engine._route('T 1 123')
        self.engine._route('T 2 456')
        self.assertEqual(self.engine.wait_tokens(1, 1), [123])

    def test_completed_source_is_not_a_busy_copy(self):
        self.engine._route('T 1 123')
        self.engine._route('DONE 1 1 2 0 0 length 0 0 0')
        with self.assertRaisesRegex(RuntimeError, 'ended before'):
            self.engine.wait_tokens(1, 1)

    def test_missing_progress_has_a_deadline(self):
        with self.assertRaisesRegex(RuntimeError, 'did not reach'):
            self.engine.wait_tokens(1, 4, timeout=0)


if __name__ == '__main__':
    unittest.main()

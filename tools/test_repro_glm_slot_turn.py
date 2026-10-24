"""CPU checks for the opt-in HTTP regression probe; no model or server required."""
import copy
import contextlib
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
import unittest

from tools.repro_glm_slot_turn import assessment, conversation, edited_turn, main


class SlotTurnProbeTest(unittest.TestCase):
    def test_edit_discards_only_the_last_turn_and_answer(self):
        history, _ = conversation("A", 64)
        history += [{"role": "assistant", "content": '{"ready":true}'},
                    {"role": "user", "content": "old question"},
                    {"role": "assistant", "content": "old answer"}]
        original = copy.deepcopy(history)
        result = edited_turn(history, "new question")
        self.assertEqual(result[:-1], history[:-2])
        self.assertEqual(result[-1], {"role": "user", "content": "new question"})
        self.assertEqual(history, original)

    def test_archive_is_deterministic_and_conversations_differ(self):
        a, fa = conversation("A", 64)
        b, fb = conversation("B", 64)
        self.assertEqual((a, fa), conversation("A", 64))
        self.assertNotEqual(a[1], b[1])
        self.assertTrue(set(fa.values()).isdisjoint(fb.values()))
        self.assertTrue(all(v in a[1]["content"] for v in fa.values()))

    def test_rejects_incomplete_turn(self):
        with self.assertRaises(ValueError):
            edited_turn([{"role": "user", "content": "unanswered"}], "edit")

    def test_success_requires_both_reuse_and_correct_answers(self):
        rows = [{"answer_ok": True, "timings": {"cache_n": 20000, "prompt_n": 30}},
                {"answer_ok": True, "timings": {"cache_n": 20000, "prompt_n": 40}}]
        self.assertTrue(all(assessment(rows).values()))
        rows[-1]["answer_ok"] = False
        self.assertFalse(assessment(rows)["answers_correct"])

    def test_rejects_full_reread_or_unexercised_slot(self):
        rows = [{"answer_ok": True, "timings": {"cache_n": 20000}},
                {"answer_ok": True, "timings": {"cache_n": 0, "prompt_n": 20040}}]
        self.assertFalse(assessment(rows)["edit_reused_restored_prefix"])
        self.assertFalse(assessment(rows)["edit_read_short_tail"])
        rows[0]["timings"]["cache_n"] = 0
        self.assertFalse(assessment(rows)["slot_restore_exercised"])

    def test_streamed_http_probe_accepts_fixed_and_baseline(self):
        for mode in ["fixed", "baseline"]:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                calls = []
                def response(req, timeout):
                    body = json.loads(req.data)
                    calls.append(body["messages"])
                    index = len(calls) - 1
                    answers = [{"ready": True}, {"ready": True}, {"middle": "A_e831"},
                               {"start": "A_7d29", "end": "A_94fb"}]
                    cached = 20000 if index == 2 or (index == 3 and mode == "fixed") else 0
                    event = {"choices": [{"delta": {"content": json.dumps(answers[index])}}],
                             "timings": {"cache_n": cached, "prompt_n": 40 if cached else 20040}}
                    return io.BytesIO(b"data: " + json.dumps(event).encode() + b"\n\ndata: [DONE]\n\n")
                output = Path(tmp) / "result.json"
                with patch("sys.argv", ["probe", "--model", "test", "--rows", "64", "--expect", mode,
                                        "--output", str(output)]), \
                     patch("urllib.request.urlopen", side_effect=response), contextlib.redirect_stdout(io.StringIO()):
                    self.assertEqual(main(), 0)
                result = json.loads(output.read_text(encoding="utf-8"))
                self.assertTrue(result["passed"])
                self.assertEqual(len(calls), 4)
                self.assertEqual(calls[2][:-1], calls[3][:-1])
                self.assertNotEqual(calls[2][-1], calls[3][-1])


if __name__ == "__main__":
    unittest.main()

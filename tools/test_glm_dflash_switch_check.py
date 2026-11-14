"""Focused protocol tests for glm_dflash_switch_check (no model, GPU, or server process)."""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_dflash_switch_check as check  # noqa: E402


class DflashSwitchCheckTest(unittest.TestCase):
    def test_wire_body_uses_root_tune_and_fixed_budget(self):
        body = check.request_body("glm", "queued-spec", 96, 7)
        self.assertEqual(body["max_tokens"], 96)
        self.assertEqual(body["strata_tune"], {"spec": 7})
        self.assertTrue(body["ignore_eos"])
        self.assertEqual(body["reasoning_effort"], "none")
        self.assertEqual(body["stream_options"], {"include_usage": True})
        self.assertEqual(body["chat_template_kwargs"], {"enable_thinking": False})
        self.assertNotIn("extra_body", body)
        self.assertTrue(body["stream"])

    def test_status_requires_real_serving_and_activity_counts(self):
        self.assertEqual(check.serving({"concurrency": {"serving": 4}}), 4)
        self.assertEqual(check.status_counts({"activity": {"in_flight": 5, "queue": {"waiting": 1}}}), (5, 1))
        self.assertEqual(check.active_count({"live": {"active": 4}}), 4)
        with self.assertRaises(check.CheckError):
            check.serving({"concurrency": {"serving": 0}})
        with self.assertRaises(check.CheckError):
            check.status_counts({"activity": {"in_flight": 1, "queue": {"waiting": "1"}}})

    def test_bearer_header_comes_only_from_environment_value(self):
        self.assertEqual(check.headers(None), {})
        self.assertEqual(check.headers("secret"), {"Authorization": "Bearer secret"})
        self.assertEqual(check.headers("secret", content=True)["Content-Type"], "application/json")

    def test_complete_stream_requires_done_finish_usage_and_timings(self):
        stream = check.Stream("http://local", {"max_tokens": 2}, 1, None, "x")
        with self.assertRaises(check.CheckError):
            stream.require_complete()
        stream.saw_done, stream.finish_reason = True, "length"
        stream.http_status = 200
        stream.usage, stream.timings = {"completion_tokens": 2}, {"draft_n": 1}
        stream.require_complete()

    def test_slice_cancel_scans_only_new_server_lines(self):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "server.log"
            path.write_text("[maya] fair slice: old event\ncommand --api-key secret\n", encoding="utf-8")
            old_end = path.stat().st_size
            self.assertFalse(check.slice_cancel_evidence(path, old_end))
            with path.open("a", encoding="utf-8") as f:
                f.write("[maya] fair slice: 12 tokens written, the engine goes to the next request (1 waiting)\n")
                f.write("glm multi: request 12 done (cancel): 20 tokens in 1.0 s (20.0 tok/s), 3 running\n")
            self.assertTrue(check.slice_cancel_evidence(path, old_end))


if __name__ == "__main__":
    unittest.main()

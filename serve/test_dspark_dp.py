"""serve/test_dspark_dp.py - the drafts' split by source (issue #590): the engine's DP line before each DONE reaches
engine.last, the /metrics rows and totals, and Prometheus - with the DSpark drafter's and the prompt lookup's drafts
told apart, and an engine without DP still reporting the DONE's sum alone.

    python -m unittest serve.test_dspark_dp -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = str(ROOT / "serve/fake_dspark_engine.py")
BODY = json.dumps({"model": "m", "max_tokens": 8, "temperature": 0,
                   "messages": [{"role": "user", "content": "hi"}]}).encode()


def fake_exe() -> str:
    """The fake engine as something Popen can start (Windows will not execute a .py: WinError 193)."""
    if os.name != "nt":
        return FAKE
    wrap = Path(tempfile.gettempdir()) / "fake_dspark_engine.cmd"
    wrap.write_text(f'@echo off\r\n"{sys.executable}" "{FAKE}" %*\r\n')
    return str(wrap)


class SplitMetrics(unittest.TestCase):
    def setUp(self):
        self.engine = StrataEngine(fake_exe(), [])
        self.svc = Service(self.engine, ByteTokenizer(), ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.engine.close()

    def get(self, path):
        with urllib.request.urlopen(self.base + path, timeout=15) as r:
            return r.read().decode()

    def ask(self):
        req = urllib.request.Request(self.base + "/v1/chat/completions", data=BODY,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            self.assertEqual(json.loads(r.read())["choices"][0]["finish_reason"], "stop")

    def test_dp_split_reaches_metrics(self):
        self.ask()
        m = json.loads(self.get("/metrics"))
        t = m["totals"]
        self.assertEqual((t["drafts_accepted"], t["drafts_offered"]), (6, 10))
        self.assertEqual((t["dspark_accepted"], t["dspark_offered"]), (4, 7))
        self.assertEqual((t["lookup_accepted"], t["lookup_offered"], t["lookup_rounds"]), (2, 3, 1))
        row = m["requests"][0]
        self.assertEqual((row["drafts_kept"], row["drafts_offered"]), (6, 10))
        self.assertEqual((row["dspark_kept"], row["dspark_offered"]), (4, 7))
        self.assertEqual((row["lookup_kept"], row["lookup_offered"], row["lookup_rounds"]), (2, 3, 1))
        self.assertEqual(row["draft_positions"], [[4, 7], [2, 3]])
        prom = self.get("/metrics/prometheus")
        self.assertIn("maya:dspark_drafts_accepted_total 4", prom)
        self.assertIn("maya:lookup_rounds_total 1", prom)

    def test_a_done_without_dp_keeps_the_sum_only(self):
        self.ask()                      # the fake's first request has DP
        self.ask()                      # its second has none (an engine without the split)
        m = json.loads(self.get("/metrics"))
        row = m["requests"][0]
        self.assertEqual((row["drafts_kept"], row["drafts_offered"]), (3, 5))
        for k in ("dspark_kept", "dspark_offered", "lookup_kept", "lookup_offered", "lookup_rounds",
                  "draft_positions"):
            self.assertNotIn(k, row)
        t = m["totals"]                 # the sums grow by the DONE's fields, the split by the first request alone
        self.assertEqual((t["drafts_accepted"], t["drafts_offered"]), (9, 15))
        self.assertEqual((t["dspark_accepted"], t["dspark_offered"]), (4, 7))
        self.assertEqual((t["lookup_accepted"], t["lookup_offered"], t["lookup_rounds"]), (2, 3, 1))


class MultiRouting(unittest.TestCase):
    def test_dp_routes_to_its_own_request(self):
        engine = StrataEngine(fake_exe(), ["2"])
        try:
            out = {}

            def one(k):
                ids = list(range(k * 10, k * 10 + 4))
                for _ in engine.generate(ids, 4, {}, threading.Event()):
                    pass
                out[k] = dict(engine.last)
                self.assertIsNone(engine.dp)     # the DONE took the DP with it

            ths = [threading.Thread(target=one, args=(k,)) for k in range(2)]
            [t.start() for t in ths]
            [t.join(30) for t in ths]
            self.assertEqual(len(out), 2)
            rounds = set()
            for last in out.values():
                r = last["lookup_rounds"]        # the fake's per-rid values: kept=rid, offered=2*rid, ...
                rounds.add(r)
                self.assertEqual(last["drafts_accepted"], r)
                self.assertEqual(last["drafts_offered"], 2 * r)
                self.assertEqual(last["dspark_accepted"], r - 1)
                self.assertEqual(last["dspark_offered"], r)
                self.assertEqual(last["lookup_accepted"], 1)
                self.assertEqual(last["lookup_offered"], r)
                self.assertNotIn("draft_positions", last)   # the multi path's DP has no pos=
            self.assertEqual(rounds, {1, 2})
        finally:
            engine.close()


class ParseOnly(unittest.TestCase):
    def test_parse_dp_then_done(self):
        e = StrataEngine.__new__(StrataEngine)
        e._parse_dp("DP kept=6 offered=10 ds_kept=4 ds_offered=7 lk_kept=2 lk_offered=3 rounds=1 pos=1:4/7,2:2/3")
        e._parse_done("DONE 2 2 1.0 2.0 stop 6 10 0 0 0")
        self.assertEqual(e.last["dspark_accepted"], 4)
        self.assertEqual(e.last["lookup_rounds"], 1)
        self.assertEqual(e.last["draft_positions"], [[4, 7], [2, 3]])
        self.assertIsNone(e.dp)
        e._parse_done("DONE 2 2 1.0 2.0 stop 3 5 0 0 0")     # the next request without DP: no stale split
        self.assertNotIn("dspark_accepted", e.last)
        self.assertEqual(e.last["drafts_accepted"], 3)


if __name__ == "__main__":
    unittest.main()


"""glm_multi_replay tests (no GPU, no model): the workload is the same for a seed, the texts and requests follow the
plan, the summary bins and the comparison, the route log's layer bands, and one short run against a fake server.
Run: python -m unittest discover -s tools -p test_glm_multi_replay.py"""
import contextlib
import http.server
import io
import json
import random
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # runnable from the repo root too

import glm_multi_replay as gmr

CORPUS = gmr.Corpus(text="".join(f"line {i}: the quick brown fox jumps over the lazy dog.\n" for i in range(4000)))


class PlanTest(unittest.TestCase):
    def test_same_seed_same_plan_other_seed_other(self):
        a, b, c = gmr.plan(3, 4, 600), gmr.plan(3, 4, 600), gmr.plan(4, 4, 600)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_sizes_and_schedule(self):
        p = gmr.plan(1, 6, 1800, sys_tokens=(30000, 8000))
        self.assertEqual([cl["sys"] for cl in p["clients"]], [0, 1, 0, 1, 0, 1])
        for cl in p["clients"]:
            self.assertTrue(all(64 <= t["fresh"] <= 65536 and 32 <= t["max_tokens"] <= 1024 and 2 <= t["think"] <= 20
                                for t in cl["turns"]))
        self.assertEqual([b["at"] for b in p["bursts"]], [600.0, 1200.0])
        self.assertTrue(all(len(b["clients"]) == 3 for b in p["bursts"]))
        ats = [k["at"] for k in p["checks"]]
        self.assertEqual(ats, sorted(ats))
        self.assertEqual(len(ats), 20)          # 40 an hour over half an hour

    def test_fresh_sizes_mostly_short(self):
        rnd = random.Random(5)
        xs = [gmr.draw(rnd, gmr.FRESH) for _ in range(20000)]
        self.assertGreater(sum(x < 2048 for x in xs) / len(xs), 0.75)
        self.assertGreater(sum(x >= 8192 for x in xs) / len(xs), 0.03)


class TextsTest(unittest.TestCase):
    def test_texts_are_fixed_and_restarts_share_only_the_system_prompt(self):
        p = gmr.plan(2, 3, 300)
        t1, t2 = gmr.Texts(p, CORPUS), gmr.Texts(p, CORPUS)
        self.assertEqual(t1.turn, t2.turn)
        m0 = gmr.messages(t1, p, 1, 3, 0)
        m1 = gmr.messages(t1, p, 1, 3, 1)
        self.assertEqual(len(m0), 2 + 2 * 4)
        self.assertEqual(m0[0], m1[0])                       # the shared system prompt
        self.assertNotEqual(m0[1]["content"], m1[1]["content"])
        self.assertEqual(m0[2:], m1[2:])
        # a turn extends the one before (what the server restores from)
        m4 = gmr.messages(t1, p, 1, 4, 0)
        self.assertEqual(m4[:len(m0)], m0)
        # compacted: from turn 3 on only
        mc = gmr.messages(t1, p, 1, 4, 1, start=3)
        self.assertEqual(len(mc), 2 + 2 * 2)
        self.assertEqual(mc[-1]["content"], t1.turn[1][4])

    def test_take_is_about_the_tokens_asked(self):
        s = CORPUS.take(random.Random(1), 1000)
        self.assertEqual(len(s), int(1000 * gmr.CHARS_PER_TOKEN))


class SummaryTest(unittest.TestCase):
    def test_bins_and_warmup(self):
        t0 = 1000.0
        recs = [{"time": t0 + 10, "prompt_tokens": 5000, "reused": 0, "output_tokens": 100, "prompt_ms": 1000,
                 "decode_tok_s": 99, "hit_rate": 0.5},                                         # warm-up: left out
                {"time": t0 + 400, "prompt_tokens": 70000, "reused": 69000, "output_tokens": 100, "prompt_ms": 2000,
                 "decode_tok_s": 20, "hit_rate": 0.97},
                {"time": t0 + 500, "prompt_tokens": 80000, "reused": 79800, "output_tokens": 40, "prompt_ms": 500,
                 "decode_tok_s": 30, "hit_rate": 0.98},
                {"time": t0 + 600, "prompt_tokens": 140000, "reused": 130000, "output_tokens": 300, "prompt_ms": 4000,
                 "decode_tok_s": 15, "hit_rate": 0.96}]
        tiers = [(t0 + 100, {"vram_hit": 0.5, "ram_fetch": 9}, {"active": 1}),
                 (t0 + 400, {"vram_hit": 0.97, "ram_fetch": 1.0}, {"active": 8, "queued": 2}),
                 (t0 + 500, {"vram_hit": 0.98, "ram_fetch": 0.5}, {"active": 6, "queued": 0})]
        s = gmr.summarize(recs, tiers, t0, 300)
        self.assertEqual(s["requests"], 3)
        self.assertAlmostEqual(s["vram_hit"], 0.975)
        self.assertAlmostEqual(s["ram_fetch"], 0.75)
        self.assertEqual(s["active"], 7)
        self.assertEqual(s["decode_by_context"]["64K+"], [20, 1])         # the 40-token answer is too short to count
        self.assertEqual(s["decode_by_context"]["128K+"], [15, 1])
        self.assertEqual(s["read_by_fresh"]["1+"], [400.0, 1])            # 200 new tokens in 0.5 s
        self.assertEqual(s["read_by_fresh"]["512+"], [500.0, 1])
        self.assertEqual(s["read_by_fresh"]["8192+"], [2500.0, 1])

    def test_route_bands(self):
        lines = []
        for k in range(400):
            il = 3 if k % 2 == 0 else 10
            off = 0b11 if k >= 200 and il == 3 else 0           # the first band loses 2 of 8 in the second half
            lines.append(f"{il} 0 1 2 3 4 5 6 7 {off} 0 0")
        b = gmr.route_band_hits(lines, slices=2)
        self.assertEqual(b[0][0], 100.0)
        self.assertEqual(b[1][0], 75.0)
        self.assertEqual(b[0][1], 100.0)
        self.assertIsNone(b[0][2])

    def test_compare_marks_a_real_difference(self):
        def run(label, hit):
            return {"label": label, "summary": {"vram_hit": hit, "decode_by_context": {"64K+": [20, 5]}}}
        out = gmr.compare([run("a", 0.960), run("a", 0.962), run("a", 0.961),
                           run("b", 0.975), run("b", 0.976), run("b", 0.974)])
        row = next(l for l in out.splitlines() if l.startswith("| vram_hit"))
        self.assertIn("*", row)
        row = next(l for l in out.splitlines() if l.startswith("| decode 64K+"))
        self.assertNotIn("*", row)


class FakeServer(http.server.BaseHTTPRequestHandler):
    reqs = []

    def log_message(self, *a):
        pass

    def _json(self, obj):
        b = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        n = sum(len(m["content"]) for m in body["messages"]) // 3
        FakeServer.reqs.append(body)
        self._json({"choices": [{"message": {"content": "x"}}],
                    "usage": {"prompt_tokens": n, "completion_tokens": body["max_tokens"]}})

    def do_GET(self):
        recs = [{"time": time.time(), "prompt_tokens": 100 + i, "reused": 50, "output_tokens": 64, "prompt_ms": 10,
                 "decode_tok_s": 20, "hit_rate": 0.9} for i, _ in enumerate(FakeServer.reqs)]
        self._json({"requests": recs, "tiers": {"now": {"vram_hit": 0.9, "ram_fetch": 1}}, "live": {"active": 1}})


class RunTest(unittest.TestCase):
    def test_short_run_against_a_fake_server(self):
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeServer)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
                out = Path(d) / "r.json"
                rc = gmr.main(["run", "--url", f"http://127.0.0.1:{srv.server_port}", "--label", "x", "--clients",
                               "2", "--duration", "1.5", "--warmup", "0", "--poll", "0.3", "--drain", "0.3",
                               "--checks-per-hour", "0", "--sys-tokens", "300,200", "--out", str(out)])
                r = json.loads(out.read_text())
            self.assertEqual(rc, 0)
            self.assertEqual(r["label"], "x")
            self.assertGreaterEqual(len(r["sent"]), 2)
            self.assertTrue(all(b["temperature"] == 0 and "logit_bias" in b for b in FakeServer.reqs))
            self.assertGreater(r["summary"]["requests"], 0)
        finally:
            srv.shutdown()
            srv.server_close()


if __name__ == "__main__":
    unittest.main()

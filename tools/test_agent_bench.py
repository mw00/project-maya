"""agent_bench tests (no GPU, no model): the workload is the same for a seed, the texts and requests follow the plan,
the summary's window, rates and bins, the comparison, the end-token choice, and short runs against a fake streaming
server (with and without a Maya /metrics).
Run: python -m unittest discover -s tools -p test_agent_bench.py"""
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

import agent_bench as ab

CORPUS = ab.Corpus(text="".join(f"line {i}: the quick brown fox jumps over the lazy dog.\n" for i in range(4000)))


class PlanTest(unittest.TestCase):
    def test_same_seed_same_plan_other_seed_other(self):
        a, b, c = ab.plan(3, 4, 600), ab.plan(3, 4, 600), ab.plan(4, 4, 600)
        self.assertEqual(a, b)
        self.assertNotEqual(a, c)

    def test_sizes_and_schedule(self):
        p = ab.plan(1, 6, 1800, sys_tokens=(30000, 8000))
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
        xs = [ab.draw(rnd, ab.FRESH) for _ in range(20000)]
        self.assertGreater(sum(x < 2048 for x in xs) / len(xs), 0.75)
        self.assertGreater(sum(x >= 8192 for x in xs) / len(xs), 0.03)


class TextsTest(unittest.TestCase):
    def test_texts_are_fixed_and_restarts_share_only_the_system_prompt(self):
        p = ab.plan(2, 3, 300)
        t1, t2 = ab.Texts(p, CORPUS), ab.Texts(p, CORPUS)
        self.assertEqual(t1.turn, t2.turn)
        m0 = ab.messages(t1, p, 1, 3, 0)
        m1 = ab.messages(t1, p, 1, 3, 1)
        self.assertEqual(len(m0), 2 + 2 * 4)
        self.assertEqual(m0[0], m1[0])                       # the shared system prompt
        self.assertNotEqual(m0[1]["content"], m1[1]["content"])
        self.assertEqual(m0[2:], m1[2:])
        m4 = ab.messages(t1, p, 1, 4, 0)                     # a turn extends the one before
        self.assertEqual(m4[:len(m0)], m0)
        mc = ab.messages(t1, p, 1, 4, 1, start=3)            # compacted: from turn 3 on only
        self.assertEqual(len(mc), 2 + 2 * 2)
        self.assertEqual(mc[-1]["content"], t1.turn[1][4])

    def test_take_is_about_the_tokens_asked(self):
        self.assertEqual(len(CORPUS.take(random.Random(1), 1000)), int(1000 * ab.CHARS_PER_TOKEN))

    def test_corpus_from_a_folder(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / "a.md").write_text("alpha " * 300)
            (Path(d) / "b.bin").write_bytes(b"\0" * 5000)     # not a text file: left out
            c = ab.Corpus(Path(d))
            self.assertIn("=== a.md ===", c.text)
            self.assertNotIn("b.bin", c.text)


def req(kind, t_send, t_first, t_done, prompt, cached, out, read_ms=None):
    r = {"kind": kind, "client": 0, "t_send": t_send, "t_first": t_first, "t_done": t_done, "prompt_tokens": prompt,
         "cached": cached, "output_tokens": out}
    if read_ms is not None:
        r["read_ms"] = read_ms
    return r


class SummaryTest(unittest.TestCase):
    def test_window_rates_and_bins(self):
        reqs = [req("turn", 10, 12, 20, 5000, 0, 100),                 # ends in the warm-up: left out
                req("turn", 150, 151, 161, 70000, 69000, 101),          # 1000 new tokens in 1 s, 100 tokens in 10 s
                req("turn", 200, 202, 206, 40000, 39500, 41),            # too short for a decode rate
                req("check", 300, 304, 305, 32000, 30000, 16),
                req("turn", 590, 591, 700, 9000, 0, 500)]               # ends after the window: left out
        s = ab.summarize(reqs, 100, 600)
        self.assertEqual((s["requests"], s["turns"], s["checks"], s["errors"]), (3, 2, 1, 0))
        self.assertEqual(s["output_tok_s"], round((101 + 41 + 16) / 500, 2))
        self.assertEqual(s["decode_median"], 10.0)
        self.assertEqual(s["decode_by_context"]["64K+"], [10.0, 1])
        self.assertIsNone(s["decode_by_context"]["32K+"])
        self.assertEqual(s["latency_median"], 8.5)                      # turns only: 11 s and 6 s
        self.assertEqual(s["read_by_fresh"]["1+"], [250.0, 1])          # 500 new tokens in 2 s
        self.assertEqual(s["read_by_fresh"]["512+"], [600.0, 2])        # 1000 in 1 s + the check's 2000 in 4 s
        self.assertNotIn("2048+", s["read_by_fresh"])
        self.assertEqual(s["read_source"], "client (time to first token)")
        # in flight: 11 + 6 + 5 + 10 (the last one's part in the window) seconds over 500
        self.assertEqual(s["in_flight"], round((11 + 6 + 5 + 10) / 500, 3))

    def test_server_read_times_win(self):
        reqs = [req("turn", 150, 155, 165, 70000, 69000, 100, read_ms=500)]
        s = ab.summarize(reqs, 100, 600)
        self.assertEqual(s["read_by_fresh"]["512+"], [2000.0, 1])
        self.assertEqual(s["read_source"], "server")

    def test_errors_counted_not_measured(self):
        reqs = [{"kind": "turn", "client": 0, "t_send": 150, "t_done": 151, "error": "boom"},
                req("turn", 150, 151, 161, 70000, 69000, 101)]
        s = ab.summarize(reqs, 100, 600)
        self.assertEqual((s["requests"], s["errors"]), (1, 1))

    def test_server_summary(self):
        t0 = 1000.0
        recs = [{"time": t0 + 50, "duration_s": 10, "hit_rate": 0.5},     # ends in the warm-up
                {"time": t0 + 200, "duration_s": 10, "hit_rate": 0.97},
                {"time": t0 + 300, "duration_s": 10, "hit_rate": 0.99}]
        tiers = [(50, {"vram_hit": 0.5}), (200, {"vram_hit": 0.96, "ram_fetch": 2.0}), (300, {"vram_hit": 0.98})]
        s = ab.server_summary(recs, tiers, t0, 100, 600)
        self.assertEqual(s["vram_hit_per_request"], 0.98)
        self.assertEqual(s["vram_hit"], 0.97)
        self.assertEqual(s["ram_fetch"], 2.0)


class CompareTest(unittest.TestCase):
    def test_marks_a_clear_difference(self):
        def run(label, rep, dec):
            return {"label": label, "rep": rep, "summary": {"decode_median": dec, "output_tok_s": 50.0}}
        out = ab.compare([run("a", 1, 20.0), run("a", 2, 21.0), run("b", 1, 30.0), run("b", 2, 31.0)])
        row = next(x for x in out.splitlines() if x.startswith("| decode_median"))
        self.assertIn("+10 (+48.8%) *", row)
        row = next(x for x in out.splitlines() if x.startswith("| output_tok_s"))
        self.assertNotIn("*", row)
        self.assertNotIn("| ram_fetch", out)             # a metric no run has is left out


class BanTest(unittest.TestCase):
    def test_auto_by_model_name(self):
        self.assertEqual(ab.ban_of("auto", "glm-5.3-flash"), {"154820": -100, "154827": -100, "154829": -100})
        self.assertEqual(ab.ban_of("auto", "some-other-model"), {})
        self.assertEqual(ab.ban_of("none", "glm-5.3-flash"), {})
        self.assertEqual(ab.ban_of("7, 9", ""), {"7": -100, "9": -100})


class FakeServer(http.server.ThreadingHTTPServer):
    daemon_threads = True


def fake_handler(with_metrics: bool, seen: list):
    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _json(self, code, obj):
            b = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(b)))
            self.end_headers()
            self.wfile.write(b)

        def do_GET(self):
            if self.path == "/v1/models":
                self._json(200, {"data": [{"id": "glm-5.3-flash", "meta": {"n_ctx": 65536}}]})
            elif self.path.startswith("/metrics") and with_metrics:
                self._json(200, {"tiers": {"now": {"vram_hit": 0.95, "ram_fetch": 1.5, "promo": 0.2}},
                                 "requests": [{"time": time.time() - 0.5, "duration_s": 0.4, "hit_rate": 0.9}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append(body)
            n = body["max_tokens"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(n):
                c = {"choices": [{"index": 0, "delta": {"content": "x"}, "finish_reason": None}]}
                self.wfile.write(f"data: {json.dumps(c)}\n\n".encode())
            last = {"choices": [{"index": 0, "delta": {}, "finish_reason": "length"}],
                    "usage": {"prompt_tokens": 1000, "completion_tokens": n,
                              "prompt_tokens_details": {"cached_tokens": 400}},
                    "timings": {"prompt_ms": 25.0, "cache_n": 400}}
            self.wfile.write(f"data: {json.dumps(last)}\n\ndata: [DONE]\n\n".encode())
    return H


class RunTest(unittest.TestCase):
    def go(self, with_metrics):
        seen = []
        srv = FakeServer(("127.0.0.1", 0), fake_handler(with_metrics, seen))
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        try:
            with tempfile.TemporaryDirectory() as d:
                out = Path(d) / "r.json"
                argv = ["run", "--url", f"http://127.0.0.1:{srv.server_address[1]}", "--label", "fake", "--clients",
                        "2", "--duration", "3", "--warmup", "0", "--poll", "0.5", "--drain", "10",
                        "--sys-tokens", "300,200", "--checks-per-hour", "3600", "--out", str(out)]
                with contextlib.redirect_stdout(io.StringIO()):
                    rc = ab.main(argv)
                r = json.loads(out.read_text())
        finally:
            srv.shutdown()
            srv.server_close()
        return rc, r, seen

    def test_run_against_a_streaming_server(self):
        rc, r, seen = self.go(with_metrics=True)
        self.assertEqual(rc, 0)
        s = r["summary"]
        self.assertGreaterEqual(s["turns"], 2)                     # each conversation sent its first turn
        self.assertGreaterEqual(s["checks"], 1)
        self.assertEqual(s["errors"], 0)
        self.assertEqual(s["read_source"], "server")
        self.assertEqual(s["read_by_fresh"]["512+"][0], 600 / 0.025)  # 600 new tokens in the server's 25 ms
        self.assertEqual(s["vram_hit"], 0.95)
        b = seen[0]
        self.assertTrue(b["stream"] and b["temperature"] == 0)
        self.assertEqual(b["logit_bias"], {"154820": -100, "154827": -100, "154829": -100})   # known by the name
        self.assertEqual(b["model"], "glm-5.3-flash")
        self.assertEqual(r["max_context"], int(0.8 * 65536))
        asked = sorted(b2["max_tokens"] for b2 in seen)              # every answer wrote what it asked for
        self.assertEqual(sorted(q["output_tokens"] for q in r["requests"]), asked)

    def test_run_without_metrics_is_client_side_only(self):
        rc, r, _ = self.go(with_metrics=False)
        self.assertEqual(rc, 0)
        self.assertNotIn("vram_hit", r["summary"])
        self.assertGreaterEqual(r["summary"]["turns"], 2)


if __name__ == "__main__":
    unittest.main()

"""serve/test_savings.py - the sidebar's "Saved": each request's tokens counted per model (new prompt, prompt reused
from the cache, output), kept across starts, and priced at the model's own API (DeepSeek's peak hours at twice its
price) - against the mock engine, no GPU.

    python -m unittest serve.test_savings -v
"""
from __future__ import annotations

import calendar
import json
import os
import sys
import tempfile
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import savings as SV  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
UTC = lambda *a: calendar.timegm((*a, 0, 0, 0))   # noqa: E731  (year, month, day, hour, minute, second)
TUE_OFF, TUE_PEAK, SAT = UTC(2026, 10, 27, 12, 0, 0), UTC(2026, 10, 27, 2, 30, 0), UTC(2026, 10, 31, 2, 30, 0)


class Prices(unittest.TestCase):
    def test_references(self):
        self.assertEqual(SV.reference("deepseek-v4.1-flash")["api"], "DeepSeek")
        self.assertEqual(SV.reference("glm-5.3-flash")["output"], 0.50)
        mm = SV.reference("minimax-m3.1-flash")                       # (MiniMax's: flat, no peak hours)
        self.assertEqual((mm["api"], mm["input"], mm["cached"], mm["output"], mm.get("peak")),
                         ("MiniMax", 0.15, 0.015, 0.60, None))
        self.assertEqual(SV.reference("MiniMax-M3.1-Flash")["api"], "MiniMax")
        self.assertIsNone(SV.reference("qwen3.8-flash-next"))

    def test_peak_hours(self):
        self.assertTrue(SV.deepseek_peak(TUE_PEAK))
        self.assertFalse(SV.deepseek_peak(TUE_OFF))
        self.assertFalse(SV.deepseek_peak(SAT))                       # (weekends: off-peak all day)
        self.assertTrue(SV.deepseek_peak(UTC(2026, 10, 27, 9, 59, 0)))
        self.assertFalse(SV.deepseek_peak(UTC(2026, 10, 27, 10, 0, 0)))

    def test_cost(self):
        # FreeToken's "This launch": 2.8M in + 118K out at V4 Flash's $0.14 / $0.28 = $0.43; here the cached part is
        # priced as an API bills it, and DeepSeek's peak hours at twice
        b = SV.zero()
        b["base"].update(input=1_000_000, cached=10_000_000, output=1_000_000)
        ds = SV.reference("deepseek-v4.1-flash")
        self.assertAlmostEqual(SV.cost(b, ds), 0.15 + 0.03 + 0.60)
        b["peak"].update(input=1_000_000)
        self.assertAlmostEqual(SV.cost(b, ds), 0.15 + 0.03 + 0.60 + 0.30)
        self.assertIsNone(SV.cost(b, None))


class Counting(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / SV.FILE

    def tearDown(self):
        self.tmp.cleanup()

    def test_kept_across_starts_and_per_model(self):
        a = SV.Ledger(self.path, "maya-gguf-q8.json", "deepseek-v4.1-flash", "UD-Q8_K_XL")
        a.add(1_000_000, 900_000, 500_000, now=TUE_OFF)
        a.add(1_000_000, 0, 0, now=TUE_PEAK)
        self.assertAlmostEqual(a.summary()["usd"], 0.1 * 0.15 + 0.9 * 0.003 + 0.5 * 0.60 + 0.30, places=6)
        b = SV.Ledger(self.path, "maya-maya-m.json", "glm-5.3-flash", "Maya-M")     # (another model starts)
        self.assertEqual(b.summary()["start_usd"], 0.0)
        b.add(2_000_000, 1_000_000, 100_000, now=TUE_PEAK)                         # (no peak price for GLM)
        glm = 0.15 + 0.03 + 0.05
        r = b.report()
        self.assertAlmostEqual(r["usd"], a.summary()["usd"] + glm, places=6)
        self.assertEqual([m["key"] for m in r["models"]], ["maya-maya-m.json", "maya-gguf-q8.json"])  # (running first)
        self.assertEqual(r["models"][1]["tokens"], {"input": 1_100_000, "cached": 900_000, "output": 500_000, "requests": 2})
        self.assertAlmostEqual(r["start"]["usd"], glm, places=6)
        self.assertEqual({p["model"] for p in r["prices"]}, {"deepseek-v4.1-flash", "glm-5.3-flash"})

    def test_reused_never_above_the_prompt_and_unknown_models(self):
        q = SV.Ledger(self.path, "x.json", "qwen3.8-flash-next")
        q.add(100, 500, 10)
        self.assertEqual(q.report()["models"][0]["tokens"]["cached"], 100)
        self.assertEqual(q.summary(), {"usd": 0.0, "start_usd": 0.0, "priced": False})

    def test_reset(self):
        a = SV.Ledger(self.path, "maya-maya-m.json", "glm-5.3-flash")
        a.add(1_000_000, 0, 0)
        a.set_price("glm-5.3-flash", {"input": 1, "cached": 0.1, "output": 2})
        a.reset()
        b = SV.Ledger(self.path, "maya-maya-m.json", "glm-5.3-flash")
        self.assertEqual(b.summary()["usd"], 0.0)
        self.assertEqual(b.ref["input"], 1.0)                          # (the page's prices stay)

    def test_own_prices_count_every_total_again(self):
        a = SV.Ledger(self.path, "maya-gguf-q8.json", "deepseek-v4.1-flash")
        a.add(1_000_000, 0, 1_000_000, now=TUE_OFF)
        self.assertAlmostEqual(a.summary()["usd"], 0.75, places=6)
        a.set_price("deepseek-v4.1-flash", {"input": 0.30, "cached": 0.006, "output": 1.20})
        self.assertAlmostEqual(a.summary()["usd"], 1.50, places=6)
        p = SV.Ledger(self.path, "maya-gguf-q8.json", "deepseek-v4.1-flash").price_list()[0]   # (kept in the file)
        self.assertEqual((p["custom"], p["input"], p["peak"], p["default"]["input"]), (True, 0.30, 2.0, 0.15))
        a.set_price("deepseek-v4.1-flash", None)                        # (back to the built-in ones)
        self.assertAlmostEqual(a.summary()["usd"], 0.75, places=6)
        # a model without a built-in price can get one; nonsense is refused with nothing changed
        q = SV.Ledger(self.path, "maya-q.json", "qwen3.8-flash-next")
        q.add(2_000_000, 0, 0)
        q.set_price("qwen3.8-flash-next", {"input": 0.5, "cached": 0.05, "output": 1})
        self.assertAlmostEqual(q.report()["start"]["usd"], 1.0, places=6)
        for bad in ({"input": -1, "cached": 0, "output": 0}, {"input": "1", "cached": 0, "output": 0}, {"input": 1},
                    {"input": 1, "cached": 0, "output": 0, "peak": 0.5}):
            with self.assertRaises(ValueError, msg=bad):
                a.set_price("deepseek-v4.1-flash", bad)
        self.assertFalse(a.price_list()[0]["custom"])


class Page(unittest.TestCase):
    """A request through the server is counted; /metrics carries the sidebar's line, /api/savings the breakdown."""

    def setUp(self):
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "</think>\n\nhello", max_context=32768), tok,
                           ChatTemplate(ROOT / "serve/chat_template.jinja"), model_name="glm-5.3-flash")
        self.tmp = tempfile.TemporaryDirectory()
        self.svc.savings = SV.Ledger(Path(self.tmp.name) / SV.FILE, "maya-maya-m.json", "glm-5.3-flash", "Maya-M")
        self.httpd = serve(self.svc, port=0)
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        self.tmp.cleanup()

    def req(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json"} if body is not None else {}
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, method=method, headers=h,
                                   data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, json.loads(resp.read())
        except urllib.error.HTTPError as e:
            with e:
                return e.code, json.loads(e.read())

    def test_a_request_is_counted(self):
        s, b = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)
        s, r = self.req("GET", "/api/savings")
        t = r["models"][0]["tokens"]
        self.assertEqual((t["requests"], t["input"], t["output"]),
                         (1, b["usage"]["prompt_tokens"], b["usage"]["completion_tokens"]))
        self.assertGreater(r["usd"], 0)
        s, m = self.req("GET", "/metrics")
        self.assertEqual((m["savings"]["usd"], m["savings"]["priced"]), (r["usd"], True))
        self.assertIsNone(m["live"]["prompt_reused"])                     # (the Monitor's Context bar: idle now)
        self.assertTrue(json.loads((Path(self.tmp.name) / SV.FILE).read_text())["models"]["maya-maya-m.json"])

    def test_prices_from_the_page(self):
        self.svc.savings.add(1_000_000, 0, 0)
        s, r = self.req("POST", "/api/savings/prices", {"model": "glm-5.3-flash",
                                                        "price": {"input": 1, "cached": 0.1, "output": 2}})
        self.assertEqual((s, r["usd"], r["price_list"][0]["custom"]), (200, 1.0, True))
        s, _ = self.req("POST", "/api/savings/prices", {"model": "glm-5.3-flash", "price": {"input": "x"}})
        self.assertEqual(s, 400)
        s, _ = self.req("POST", "/api/savings/prices", {"model": "glm-5.3-flash", "price": None},
                        {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        s, r = self.req("POST", "/api/savings/prices", {"model": "glm-5.3-flash", "price": None})
        self.assertEqual((s, r["usd"]), (200, 0.15))

    def test_reset_only_from_the_page(self):
        self.svc.savings.add(1_000_000, 0, 0)
        s, _ = self.req("POST", "/api/savings/reset", {}, {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        s, r = self.req("POST", "/api/savings/reset", {})
        self.assertEqual((s, r["usd"]), (200, 0.0))


if __name__ == "__main__":
    unittest.main()

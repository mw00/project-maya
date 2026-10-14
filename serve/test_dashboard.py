"""serve/test_dashboard.py - what the web app's Settings and Monitor ask the server for: the context size (its range, a
reload, the 503 while it reloads, the run config it is saved into), the engine's measured context cost, and the report.
Against the mock engine (no GPU, no pack).

    python -m unittest serve.test_dashboard -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import CONTEXT_MIN, MAYA_VERSION, ByteTokenizer, MockEngine, Service, StrataEngine, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class Dashboard(unittest.TestCase):
    def setUp(self):
        os.environ["STRATA_MOCK_RELOAD_S"] = "0.3"
        tok = ByteTokenizer()
        self.engine = MockEngine(tok, "</think>\n\nhello", max_context=32768)
        self.svc = Service(self.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self.tmp.name) / "maya-test.json"
        self.cfg.write_text(json.dumps({"exe": "strata", "args": ["--glm-pack", "/x/pack", "--max-context", "32768"],
                                        "env": {"STRATA_GLM_RAM_GB": "20", "MY_API_KEY": "secret"}}, indent=1))
        self.svc.config_path = str(self.cfg)
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
                raw = resp.read()
                return resp.status, resp.headers, raw
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.headers, e.read()

    def wait_reload(self):
        for _ in range(100):
            s, _, b = self.req("GET", "/api/context")
            rl = json.loads(b)["reload"]
            if rl and rl["state"] in ("done", "failed"):
                return json.loads(b)
            time.sleep(0.05)
        self.fail("the reload did not end")

    def test_health_has_the_version(self):
        s, _, b = self.req("GET", "/health")
        self.assertEqual(json.loads(b)["version"], MAYA_VERSION)

    def test_context_range(self):
        s, _, b = self.req("GET", "/api/context")
        j = json.loads(b)
        self.assertEqual((s, j["context"], j["min"]), (200, 32768, CONTEXT_MIN))
        self.assertGreaterEqual(j["max"], 32768)

    def test_reload_changes_the_context_and_saves_it(self):
        s, _, b = self.req("POST", "/api/context", {"max_context": 65536})
        self.assertEqual(s, 202)
        j = self.wait_reload()
        self.assertEqual((j["reload"]["state"], j["context"]), ("done", 65536))
        args = json.loads(self.cfg.read_text())["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "65536")
        self.assertEqual(json.loads(self.cfg.read_text())["env"]["STRATA_GLM_RAM_GB"], "20")   # the rest kept

    def test_chat_gets_503_while_reloading(self):
        os.environ["STRATA_MOCK_RELOAD_S"] = "1.5"
        self.req("POST", "/api/context", {"max_context": 16384})
        s, h, b = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 503)
        self.assertEqual(h.get("Retry-After"), "30")
        self.assertIn("reloading", json.loads(b)["error"]["message"])
        s, _, b = self.req("GET", "/health")
        self.assertEqual(json.loads(b)["status"], "reloading")
        self.wait_reload()
        s, _, _ = self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(s, 200)

    def test_out_of_range_and_foreign_pages_are_refused(self):
        s, _, _ = self.req("POST", "/api/context", {"max_context": 100})
        self.assertEqual(s, 400)
        s, _, _ = self.req("POST", "/api/context", {"max_context": 65536}, {"Origin": "http://evil.example"})
        self.assertEqual(s, 403)
        r = urllib.request.Request(self.base + "/api/context", method="POST", data=b"max_context=65536",
                                   headers={"Content-Type": "application/x-www-form-urlencoded"})
        with self.assertRaises(urllib.error.HTTPError) as cm:
            urllib.request.urlopen(r, timeout=10)
        self.assertEqual(cm.exception.code, 415)
        cm.exception.close()

    def test_report_leaves_secrets_out(self):
        s, h, b = self.req("GET", "/api/report")
        text = b.decode()
        self.assertEqual(s, 200)
        self.assertIn("Project Maya", text)
        self.assertIn("STRATA_GLM_RAM_GB", text)
        self.assertNotIn("secret", text)

    def test_token_count_matches_the_prompt(self):
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello there"},
                {"role": "assistant", "content": "Hi! How can I help?"}, {"role": "user", "content": "count me"}]
        s, _, b = self.req("POST", "/api/tokens", {"messages": msgs, "reasoning_effort": "high"})
        j = json.loads(b)
        from serve.frontend import openai_to_messages
        m, tools, kw = openai_to_messages({"messages": msgs, "reasoning_effort": "high"})
        ids, _, _ = self.svc.prepare(m, tools, kw)
        self.assertEqual((s, j["tokens"], j["approximate"]), (200, len(ids), False))
        s, _, b = self.req("POST", "/api/tokens", {"messages": msgs + [{"role": "user", "content": [
            {"type": "text", "text": "and this"}, {"type": "image_url", "image_url": {"url": "data:,"}}]}]})
        self.assertTrue(json.loads(b)["approximate"])

    def test_requests_record_their_api(self):
        self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        s, _, b = self.req("GET", "/metrics")
        self.assertEqual(json.loads(b)["requests"][0]["api"], "openai")


class ContextCost(unittest.TestCase):
    """The GLM engine's start lines give what its context costs in VRAM, summed over its GPUs."""

    def test_parsed_from_the_start_log(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "engine.log"
            log.write_text("old start\nglm fast: CUDA0 VRAM before the expert pool: 4.0 of 31.7 GB in use - dense weights 3.1, "
                           "state/KV (16384 ctx) 0.40, activations 0.1\n"
                           "this start\nglm fast: CUDA0 VRAM before the expert pool: 4.4 of 31.7 GB in use - dense weights 3.1, "
                           "state/KV (32768 ctx) 0.66, activations 0.1\n"
                           "glm fast: CUDA1 VRAM before the expert pool: 4.3 of 31.7 GB in use - dense weights 2.9, "
                           "state/KV (32768 ctx) 0.65, activations 0.1\n")
            eng = StrataEngine.__new__(StrataEngine)
            eng.log_path, eng.info = str(log), {}
            eng._context_cost(log.read_text().index("this start"))
            self.assertEqual((eng.info["kv_ctx"], eng.info["kv_gb"]), (32768, 1.31))

    def test_set_context_rewrites_the_command(self):
        eng = StrataEngine.__new__(StrataEngine)
        eng.spawn = ("strata", ["--glm-pack", "p", "--max-context", "32768"], None, None, None)
        eng.set_context(131072)
        self.assertEqual(eng.spawn[1], ["--glm-pack", "p", "--max-context", "131072"])


GB = 2**30
# the 3090 alone with EXL3-4.05bpw and DFlash2, as its start measured it (glm fast: CUDA0 VRAM before the expert pool ...)
LINE = ("VRAM dev=0 pci=0000:01:00.0 total=25296896000 free=751828992 weights=6678575104 kv=665719603 kv_ram=0 "
        "act=107374182 drafter=751828992 window=246415360 pool=15536793108 tail=7945689497 borrow=7945689497 "
        "vision=0 slots=1231 ctx=32768 layers=0-45\n")


class GpuMap(unittest.TestCase):
    """What fills each GPU: the engine's VRAM lines, joined with NVML's cards and processes (serve/gpu_map.py)."""

    def setUp(self):
        from serve import gpu_map
        self.gm = gpu_map
        self.r = gpu_map.parse(LINE)

    def card(self, used, procs, index=0, pci="0000:01:00"):
        return {"index": index, "pci": pci, "name": "NVIDIA GeForce RTX 3090", "used": used, "total": 25296896000,
                "procs": procs}

    def test_parse(self):
        self.assertEqual((self.r["dev"], self.r["pci"], self.r["layers"], self.r["slots"]), (0, "0000:01:00", [0, 45], 1231))
        self.assertEqual(self.r["pool"], 15536793108)
        self.assertIsNone(self.gm.parse("INFO context=32768"))
        self.assertEqual(self.gm.pci_key("00000000:0A:00.0"), "0000:0a:00")

    def test_nvml_tells_the_engine_from_other_programs(self):
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own, other = engine + 400 * 2**20, 600 * 2**20
        (g,) = self.gm.view([self.r], [self.card(own + other, {100: own, 200: other})], engine_pid=100)
        self.assertEqual((g["index"], g["measured"]), (0, "nvml"))
        self.assertEqual(g["parts"]["rest"], 400 * 2**20)        # the CUDA context and the small buffers
        self.assertEqual(g["parts"]["other"], other)
        self.assertEqual(g["parts"]["drafter"], self.r["drafter"] + self.r["window"])
        self.assertEqual(g["free"], g["total"] - own - other)
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])

    def test_the_drivers_reserve_is_not_another_program(self):
        # (9 GPUs, the engine alone on each: NVML's used memory ~0.58 GB beyond its process - the driver's reserve)
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own, reserve = engine + 500 * 2**20, 590 * 2**20
        (g,) = self.gm.view([self.r], [self.card(own + reserve, {100: own})], engine_pid=100)
        self.assertEqual((g["parts"]["other"], g["parts"]["rest"]), (0, 500 * 2**20 + reserve))

    def test_found_by_pci_not_by_place(self):
        cards = [self.card(1 * GB, {}, index=0, pci="0000:02:00"), self.card(23 * GB, {100: 22 * GB}, index=1)]
        (g,) = self.gm.view([self.r], cards, engine_pid=100)
        self.assertEqual(g["index"], 1)

    def test_without_nvml_the_start_measurement(self):
        (g,) = self.gm.view([self.r], [], engine_pid=100)
        start = self.r["total"] - self.r["free"]
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        self.assertEqual((g["measured"], g["parts"]["rest"], g["parts"]["other"]), ("start", start - engine, 0))
        self.assertEqual(g["free"], self.r["free"])

    def test_the_vision_encoder_has_the_tail(self):
        r = dict(self.r, vision=self.r["tail"])
        engine = sum(r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool")) - r["tail"]
        own, enc = engine + 400 * 2**20, r["tail"] - 50 * 2**20
        (g,) = self.gm.view([r], [self.card(own + enc, {100: own, 300: enc})], 100, vision_pid=300, vision_lent=True)
        self.assertTrue(g["vision_lent"])
        self.assertEqual((g["parts"]["pool"], g["parts"]["vision"]), (r["pool"] - r["tail"], enc))
        self.assertEqual(g["parts"]["other"], 0)

    def test_never_more_than_the_card(self):
        (g,) = self.gm.view([self.r], [self.card(26 * GB, {100: 25 * GB, 200: 2 * GB})], engine_pid=100)
        self.assertLessEqual(sum(g["parts"].values()), g["total"])
        self.assertEqual(g["free"], 0)

    def test_two_parts_on_one_card_add_up(self):
        a = dict(self.r, layers=[0, 20], slots=500)
        b = dict(self.r, layers=[20, 45], slots=700, free=self.r["free"] - GB)
        (m,) = self.gm.merge([a, b])
        self.assertEqual((m["layers"], m["slots"], m["pool"], m["free"]), ([0, 45], 1200, 2 * self.r["pool"], b["free"]))

    def test_metrics_carry_it(self):
        from types import SimpleNamespace
        tok = ByteTokenizer()
        eng = MockEngine(tok, "hi", max_context=32768)
        eng.vram, eng.proc = [self.r], SimpleNamespace(pid=100)
        svc = Service(eng, tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        card = self.card(24 * GB, {100: 23 * GB})
        svc.telemetry = SimpleNamespace(snapshot=lambda: {"now": {"gpu_mem_used": 24 * GB, "vram": [card]},
                                                          "history": {}, "static": {}})
        m = svc.metrics()
        self.assertNotIn("vram", m["hardware"])                  # (the processes stay in the server)
        self.assertEqual(m["gpu_map"][0]["slots"], 1231)
        json.dumps(m)
        eng.vram = []
        self.assertIsNone(svc.metrics()["gpu_map"])             # (an engine without VRAM lines: no card)


class GpuRelease(unittest.TestCase):
    """A reload starts the new engine only once the GPUs have dropped the old one (its memory is freed after it ends)."""

    def test_waits_until_no_gpu_lists_the_engine(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        seen = [{4242, 7}, {4242, 7}, {7}]
        with mock.patch.object(T, "gpu_pids", side_effect=lambda: seen.pop(0) if len(seen) > 1 else seen[0]), \
                mock.patch.object(SV.time, "sleep") as sleep:
            SV.wait_gpu_release(4242, timeout=30)
        self.assertEqual(seen, [{7}])
        self.assertEqual(sleep.call_count, 2)

    def test_without_nvml_a_short_wait(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        with mock.patch.object(T, "gpu_pids", return_value=None), mock.patch.object(SV.time, "sleep") as sleep:
            SV.wait_gpu_release(4242)
        sleep.assert_called_once()


if __name__ == "__main__":
    unittest.main()

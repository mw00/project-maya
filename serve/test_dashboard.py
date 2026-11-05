"""serve/test_dashboard.py - what the web app's Settings and Monitor ask the server for: the context size (its range, a
reload, the 503 while it reloads, the run config it is saved into), the engine's measured context cost, and the report.
Against the mock engine (no GPU, no pack).

    python -m unittest serve.test_dashboard -v
"""
from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import (CONTEXT_MIN, MAYA_VERSION, ByteTokenizer, MockEngine, Service, StrataEngine, serve,  # noqa: E402
                          trained_context_ds4)

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

    def test_health_history_thinking(self):
        """A Qwen3.x template keeps past thinking: the web chat is told to send it back (the GLM-5 one: not)."""
        s, _, b = self.req("GET", "/health")
        self.assertIs(json.loads(b)["history_thinking"], True)
        self.svc.template = ChatTemplate(ROOT / "serve/glm5_chat_template.jinja")
        s, _, b = self.req("GET", "/health")
        self.assertIs(json.loads(b)["history_thinking"], False)

    def test_context_range(self):
        s, _, b = self.req("GET", "/api/context")
        j = json.loads(b)
        self.assertEqual((s, j["context"], j["min"]), (200, 32768, CONTEXT_MIN))
        self.assertGreaterEqual(j["max"], 32768)
        self.assertEqual(j["tested"], 262144)                   # (the GLM engine's: the dashboard warns past it)

    def test_deepseek_context_range(self):
        """A DeepSeek pack: the context its ds4.txt says the model was trained for is the top (V4.1-Flash: 1M), and its
        engine is tested up to it."""
        pack = Path(self.tmp.name) / "maya-ds4"
        pack.mkdir()
        (pack / "ds4.txt").write_text("hidden_size 4096\nmax_position_embeddings 1048576\nsliding_window 128\n")
        self.assertEqual(trained_context_ds4(pack), 1048576)
        self.assertIsNone(trained_context_ds4(Path(self.tmp.name) / "nowhere"))
        (Path(self.tmp.name) / "bad").mkdir()
        (Path(self.tmp.name) / "bad" / "ds4.txt").write_text("max_position_embeddings many\n")
        self.assertIsNone(trained_context_ds4(Path(self.tmp.name) / "bad"))
        self.svc.trained_context = trained_context_ds4(pack)
        self.engine.info = dict(self.engine.info, engine_kind="ds4")
        s, _, b = self.req("GET", "/api/context")
        j = json.loads(b)
        self.assertEqual((j["max"], j["tested"]), (1048576, 1048576))
        s, _, _ = self.req("POST", "/api/context", {"max_context": 1048576})
        self.assertEqual(s, 202)
        self.assertEqual((self.wait_reload()["context"]), 1048576)
        s, _, _ = self.req("POST", "/api/context", {"max_context": 2097152})   # (past what it was trained for)
        self.assertEqual(s, 400)

    def test_reload_changes_the_context_and_saves_it(self):
        s, _, b = self.req("POST", "/api/context", {"max_context": 65536})
        self.assertEqual(s, 202)
        j = self.wait_reload()
        self.assertEqual((j["reload"]["state"], j["context"]), ("done", 65536))
        args = json.loads(self.cfg.read_text())["args"]
        self.assertEqual(args[args.index("--max-context") + 1], "65536")
        self.assertEqual(json.loads(self.cfg.read_text())["env"]["STRATA_GLM_RAM_GB"], "20")   # the rest kept
        self.assertGreaterEqual(j["load_s"], 0.3)       # (how long that load took: the page's estimate of the next)
        # the setup's own size is kept for the Models page's "Reset to the setup's values", through later reloads
        self.assertEqual(json.loads(self.cfg.read_text())["installer"]["as_set_up"]["context"], 32768)
        self.req("POST", "/api/context", {"max_context": 16384})
        self.wait_reload()
        self.assertEqual(json.loads(self.cfg.read_text())["installer"]["as_set_up"]["context"], 32768)

    def test_metrics_say_whether_deepseek_drafts(self):
        # DeepSeek's engine does not say it drafts with DSpark: the server does (the Monitor's Speculation), and only
        # for a DeepSeek config
        s, _, b = self.req("GET", "/metrics")
        self.assertNotIn("dspark", json.loads(b)["engine"])
        for on in (True, False):
            self.svc.dspark = on
            s, _, b = self.req("GET", "/metrics")
            self.assertIs(json.loads(b)["engine"]["dspark"], on)

    def test_metrics_list_the_drafters_that_run(self):
        # the Monitor's Speculation: the drafters this engine runs with, per model engine; empty = drafting is off
        from serve.server import request_spec_off, speculation_modes
        cases = [
            ({"engine_kind": "glm-fast", "mtp": 1, "dflash": 0}, {}, ["MTP"]),
            ({"engine_kind": "glm-fast", "mtp": 1, "dflash": 1}, {}, ["DFlash2"]),      # (DFlash decode takes the request first)
            ({"engine_kind": "glm-fast", "mtp": 0, "dflash": 1}, {}, ["DFlash2"]),
            ({"engine_kind": "glm-fast", "mtp": 0, "dflash": 0}, {}, []),
            ({"engine_kind": "qwen", "mtp": 1, "lookup": 3}, {}, ["MTP", "Lookup"]),
            ({"engine_kind": "qwen", "mtp": 0, "lookup": 0}, {}, []),
            ({"engine_kind": "mm3", "dspark": 1}, {}, ["DSpark"]),
            ({"engine_kind": "mm3", "dspark": 0}, {}, []),
            ({"engine_kind": "ds4"}, {"dspark": True, "lookup": True}, ["DSpark", "Lookup"]),
            ({"engine_kind": "ds4"}, {"dspark": False, "lookup": True}, ["Lookup"]),
            ({"engine_kind": "ds4"}, {"dspark": False, "lookup": False}, []),
        ]
        for info, kw, want in cases:
            with self.subTest(info=info, kw=kw):
                self.assertEqual(speculation_modes(info, **kw), want)
        self.assertTrue(request_spec_off({"strata_tune": {"spec": 1}}))
        self.assertFalse(request_spec_off({"strata_tune": {"spec": 4}}))
        self.assertFalse(request_spec_off({}))
        self.assertFalse(request_spec_off(None))
        s, _, b = self.req("GET", "/metrics")
        self.assertEqual(json.loads(b)["engine"]["speculation"], [])               # (the mock engine drafts nothing)
        self.engine.info.update(mtp=1)
        self.assertEqual(json.loads(self.req("GET", "/metrics")[2])["engine"]["speculation"], ["MTP"])

    def test_stat_lines_carry_the_draft_counts_and_the_ssd_bytes(self):
        # the engines' STAT lines (GLM, CYBER-FROST, DeepSeek, MiniMax): live draft_acc / draft_off and disk_gb reach the
        # Monitor's tiers.now as numbers
        class Stub:
            pass
        eng = Stub()
        StrataEngine._parse_stat(eng, "STAT tok_s=31.20 vram_hit=0.9100 vram_used=300 ram_used=500 ram_gb=20.3 "
                                      "vram_gb=12.5 disk_gb=31.4 draft_acc=124 draft_off=200")
        self.assertEqual((eng.stat["draft_acc"], eng.stat["draft_off"], eng.stat["disk_gb"]), (124.0, 200.0, 31.4))
        self.engine.stat = dict(eng.stat)
        now = json.loads(self.req("GET", "/metrics")[2])["tiers"]["now"]
        self.assertEqual((now["draft_acc"], now["draft_off"], now["disk_gb"]), (124.0, 200.0, 31.4))

    def test_requests_carry_the_drafts_offered_and_kept(self):
        # the page's last-answer acceptance: the engine's DONE counts reach the request row, None when it reported none
        self.engine.last = None
        self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        row = json.loads(self.req("GET", "/metrics")[2])["requests"][0]
        self.assertIn("drafts_offered", row)
        self.assertIn("spec_off", row)
        self.assertFalse(row["spec_off"])

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
        self.assertIn("MAYA_GLM_RAM_GB", text)   # (a config's settings under the names users see)
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

    def test_tokenize_and_detokenize(self):
        """POST /tokenize and /detokenize in llama.cpp's and vLLM's forms: a text's ids (its special tokens' text as
        those tokens unless parse_special is false), a chat request's prompt as prepare() makes it, and the text back."""
        tok = ByteTokenizer()
        s, _, b = self.req("POST", "/tokenize", {"content": "hi<|im_end|>"})
        self.assertEqual((s, json.loads(b)["tokens"]), (200, tok.encode("hi<|im_end|>", parse_special=True)))
        s, _, b = self.req("POST", "/tokenize", {"model": "m", "prompt": "hi<|im_end|>", "parse_special": False})
        j = json.loads(b)
        self.assertEqual((j["tokens"], j["count"]), (tok.encode("hi<|im_end|>"), len("hi<|im_end|>")))
        self.assertEqual(j["max_model_len"], self.svc.engine.max_context)
        msgs = [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "count me"}]
        s, _, b = self.req("POST", "/tokenize", {"messages": msgs, "reasoning_effort": "high"})
        from serve.frontend import openai_to_messages
        m, tools, kw = openai_to_messages({"messages": msgs, "reasoning_effort": "high"})
        ids, _, _ = self.svc.prepare(m, tools, kw)
        self.assertEqual((s, json.loads(b)["tokens"]), (200, ids))
        s, _, b = self.req("POST", "/detokenize", {"tokens": ids})
        j = json.loads(b)
        self.assertEqual((s, j["content"]), (200, tok.decode(ids)))
        self.assertEqual(j["prompt"], j["content"])
        for path, body in (("/tokenize", {"content": 3}), ("/tokenize", {}), ("/detokenize", {"tokens": [1, -1]}),
                           ("/detokenize", {"tokens": "1,2"})):
            s, _, _ = self.req("POST", path, body)
            self.assertEqual(s, 400, (path, body))

    def test_peak_decode_speed(self):
        # the Monitor's peak decode: the fastest answer since the server started, of PEAK_MIN_TOKENS tokens or more
        from serve.server import PEAK_MIN_TOKENS
        s, _, b = self.req("GET", "/metrics")
        self.assertIsNone(json.loads(b)["totals"]["peak_decode"])
        end = self.engine.script[-1:]                                  # (its end-of-turn token)
        self.engine.script = self.engine.scripts[0] = ByteTokenizer().encode("</think>\n\n" + "a long enough answer " * 3) + end
        self.engine.tok_s = 4000
        rows = []
        for _ in range(3):
            self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
            rows.append(json.loads(self.req("GET", "/metrics")[2])["requests"][0])
        peak = json.loads(self.req("GET", "/metrics")[2])["totals"]["peak_decode"]
        top = max(rows, key=lambda r: r["decode_tok_s"])
        self.assertEqual((peak["tok_s"], peak["tokens"], peak["time"]), (top["decode_tok_s"], top["output_tokens"], top["time"]))
        self.assertGreaterEqual(peak["tokens"], PEAK_MIN_TOKENS)
        _, _, b = self.req("GET", "/metrics/prometheus")
        self.assertIn(f"maya:decode_peak_tokens_seconds {peak['tok_s']}", b.decode())
        # a short answer, however fast, is not the peak
        self.engine.tok_s = 400000
        self.req("POST", "/v1/chat/completions", {"model": "m", "max_tokens": 5,
                                                  "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(json.loads(self.req("GET", "/metrics")[2])["totals"]["peak_decode"], peak)

    def test_peak_decode_follows_the_live_figure(self):
        # a live Decode reading above the peak is the peak in the same /metrics answer (the owner's screenshot: 30.0
        # live, peak 27.8), rounded as shown; a reading before PEAK_MIN_TOKENS tokens is not
        from serve.server import PEAK_MIN_TOKENS, RequestStat
        svc, now = self.svc, time.time()
        svc.totals["peak_decode"] = {"tok_s": 27.8, "tokens": 400, "time": now - 60}
        r = RequestStat()
        r.first_token = now - 10
        with svc.status_lock:
            svc.status.update(busy=True, generated=PEAK_MIN_TOKENS - 1)
            svc.current_requests.add(r)
            svc.latest_request = r
        try:
            r.base = (now - 1, 0); r.samples.append((now, PEAK_MIN_TOKENS - 1))   # 31 tok/s, but only 31 tokens
            m = json.loads(self.req("GET", "/metrics")[2])
            self.assertEqual(m["totals"]["peak_decode"]["tok_s"], 27.8)
            for k, n in enumerate((300, 330, 310)):                   # 300, 330, 310 tok/s live, one a reading
                r.samples.clear(); r.base = (time.time() - 1, 1000 * k); r.samples.append((time.time(), 1000 * k + n))
                m = json.loads(self.req("GET", "/metrics")[2])
                self.assertGreaterEqual(m["totals"]["peak_decode"]["tok_s"], m["live"]["tok_s"])
            self.assertAlmostEqual(m["totals"]["peak_decode"]["tok_s"], 330, delta=2)
            self.assertEqual(m["totals"]["peak_decode"]["tokens"], 1330)
            self.assertEqual(round(m["totals"]["peak_decode"]["tok_s"], 1), m["totals"]["peak_decode"]["tok_s"])
        finally:
            with svc.status_lock:
                svc.status.update(busy=False)
                svc.current_requests.discard(r)

    def test_requests_record_their_api(self):
        self.req("POST", "/v1/chat/completions", {"model": "m", "messages": [{"role": "user", "content": "hi"}]})
        s, _, b = self.req("GET", "/metrics")
        self.assertEqual(json.loads(b)["requests"][0]["api"], "openai")


class Port(unittest.TestCase):
    """The dashboard's Port: saved into the config and Maya's start script for it; a busy, a reserved or the same port is
    refused with nothing saved; only Maya's own page may move the server.  (No restart here: maya.py did not start this
    server, so the port applies from the next start - the restart itself is maya.py's RESTART_EXIT, tested there.)"""

    def setUp(self):
        os.environ.pop("MAYA_RESTART_ON_UPDATE", None)
        Dashboard.setUp(self)
        self.script = Path(self.tmp.name) / "run-maya-test.bat"
        self.script.write_bytes(b"@echo off\r\npython maya.py --config " + str(self.cfg).encode() + b" --port 8080 %*\r\n")

    tearDown = Dashboard.tearDown
    req = Dashboard.req

    @staticmethod
    def free_port():
        import socket
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return s.getsockname()[1]

    def test_saved_and_reported(self):
        s, _, b = self.req("GET", "/api/port")
        info = json.loads(b)
        self.assertEqual((s, info["port"], info["restarts"]), (200, self.httpd.server_address[1], False))
        n = self.free_port()
        s, _, b = self.req("POST", "/api/port", {"port": n})
        self.assertEqual((s, json.loads(b)), (202, {"port": n, "restarts": False}))
        self.assertEqual(json.loads(self.cfg.read_text())["port"], n)
        self.assertEqual(json.loads(self.cfg.read_text())["args"][-1], "32768")   # (the rest kept)
        self.assertIn(f"--port {n} %*\r\n".encode(), self.script.read_bytes())   # (CR LF kept)

    def test_refused_with_nothing_saved(self):
        import socket
        busy = socket.socket()
        busy.bind(("127.0.0.1", 0))
        busy.listen(1)
        try:
            for port in (busy.getsockname()[1], 80, self.httpd.server_address[1], "x"):
                s, _, b = self.req("POST", "/api/port", {"port": port})
                self.assertEqual(s, 400, port)
        finally:
            busy.close()
        s, _, _ = self.req("POST", "/api/port", {"port": self.free_port()}, {"Origin": "http://elsewhere.example"})
        self.assertEqual(s, 403)
        self.assertNotIn("port", json.loads(self.cfg.read_text()))
        self.assertIn(b"--port 8080 %*", self.script.read_bytes())


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

    def test_helper_pool_is_accounted_separately_from_drafter_and_rest(self):
        helper = 20_000_000_000
        r = self.gm.parse(f"VRAM dev=2 pci=0000:0f:00.0 total={32 * GB} free={5 * GB} "
                          f"weights=0 kv=0 act=0 drafter={4 * GB} pool=0 helper={helper} slots=0 layers=-1--1")
        (g,) = self.gm.view([r], [], engine_pid=100)
        self.assertEqual(g["parts"]["helper"], helper)
        self.assertEqual(g["parts"]["drafter"], 4 * GB)
        self.assertEqual(g["parts"]["rest"], 23 * GB - helper)
        self.assertIsNone(g["layers"])
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])
        merged = self.gm.merge([r, dict(dev=2, helper=1024, free=5 * GB)])
        self.assertEqual(merged[0]["helper"], helper + 1024)

    def test_helper_pool_and_primary_pool_on_one_card_do_not_overlap(self):
        r = dict(self.r, helper=2 * GB)
        engine = sum(r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool", "helper"))
        card = dict(self.card(engine + GB, {100: engine + GB}), total=32 * GB)
        (g,) = self.gm.view([r], [card], engine_pid=100)
        self.assertEqual(g["parts"]["helper"], 2 * GB)
        self.assertEqual(g["parts"]["pool"], r["pool"])
        self.assertEqual(g["parts"]["rest"], GB)
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])

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

    def test_a_vision_tower_in_the_engines_process(self):
        # (strata-ds4: the tower on the first GPU inside the engine - vision_own, not a pool tail lent to another process)
        r = dict(self.r, vision_own=318 * 2**20)
        engine = sum(r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own = engine + r["vision_own"] + 400 * 2**20
        (g,) = self.gm.view([r], [self.card(own, {100: own})], engine_pid=100)
        self.assertEqual((g["parts"]["vision"], g["vision_own"]), (r["vision_own"], r["vision_own"]))
        self.assertEqual((g["parts"]["rest"], g["parts"]["other"]), (400 * 2**20, 0))   # the CUDA context, not vision
        (s,) = self.gm.view([r], [], engine_pid=100)             # (no NVML: the start's measurement)
        self.assertEqual(s["parts"]["vision"], r["vision_own"])
        self.assertEqual(sum(s["parts"].values()) + s["free"], s["total"])

    def test_an_engine_run_through_a_launcher(self):
        # (the engine's command a script that runs `nsys launch` (pid 63), which starts the engine (85) as a grandchild:
        # NVML lists 85 - found among the launcher's tree; the launcher's pid alone found nothing, and the engine's own
        # growth since the start's measurement was drawn as other programs)
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        own, reserve, other = engine + 500 * 2**20, 128 * 2**20, 600 * 2**20
        tree = {63, 84, 85}
        (g,) = self.gm.view([self.r], [self.card(own + reserve, {85: own})], engine_pid=tree)
        self.assertEqual((g["measured"], g["parts"]["other"], g["parts"]["rest"]), ("nvml", 0, 500 * 2**20 + reserve))
        (g,) = self.gm.view([self.r], [self.card(own + reserve + other, {85: own, 200: other})], engine_pid=tree)
        self.assertEqual((g["parts"]["other"], g["parts"]["rest"]), (other, 500 * 2**20 + reserve))   # a real one
        r = dict(self.r, vision=self.r["tail"])                  # (the on-demand encoder, run through one too)
        enc = r["tail"] - 50 * 2**20
        own = engine - r["tail"] + 400 * 2**20
        (g,) = self.gm.view([r], [self.card(own + enc, {85: own, 301: enc})], tree, vision_pid={300, 301},
                            vision_lent=True)
        self.assertEqual((g["parts"]["vision"], g["parts"]["other"], g["parts"]["rest"]), (enc, 0, 400 * 2**20))

    def test_the_engine_not_listed_the_start_measurement(self):
        # (NVML lists the engine under a pid the server can't name - a pid namespace it does not share: the card is
        # drawn from the start's measurement, as without NVML)
        engine = sum(self.r[k] for k in ("weights", "kv", "act", "drafter", "window", "pool"))
        (g,) = self.gm.view([self.r], [self.card(engine + GB, {450462: engine + GB})], engine_pid=65)
        start = self.r["total"] - self.r["free"]
        self.assertEqual((g["measured"], g["parts"]["rest"]), ("start", start - engine))

    def test_never_more_than_the_card(self):
        (g,) = self.gm.view([self.r], [self.card(26 * GB, {100: 25 * GB, 200: 2 * GB})], engine_pid=100)
        self.assertLessEqual(sum(g["parts"].values()), g["total"])
        self.assertEqual(g["free"], 0)

    def test_strata_ds4s_prompt_buffers_and_tensor_parallel(self):
        # (strata-ds4 at --tp 8: a 5060 Ti of the group of 8 - its prompt chunks' buffers and tensor parallel's copies
        # told apart, the GLM engine's lines without them)
        r = self.gm.parse("VRAM dev=3 pci=0000:05:00.0 total=16610951168 free=301989888 weights=268435456 kv=33554432 "
                          "kv_ram=0 act=134217728 drafter=0 window=0 pool=13421772800 tail=0 borrow=0 vision=0 "
                          "vision_own=0 prompt=1610612736 tp=805306368 tp_n=8 slots=3343 ctx=262144 layers=3-39")
        (g,) = self.gm.view([r], [], engine_pid=100)
        self.assertEqual((g["parts"]["prompt"], g["parts"]["tp"], g["tp_n"]), (1610612736, 805306368, 8))
        self.assertEqual(sum(g["parts"].values()) + g["free"], g["total"])
        (h,) = self.gm.view([self.r], [], engine_pid=100)
        self.assertEqual((h["parts"]["prompt"], h["parts"]["tp"], h["tp_n"]), (0, 0, 0))

    def test_tensor_parallel_room_is_the_groups_least_free(self):
        # (a --tp 2 pair holds the same expert slots: the 8 GB GPU 1 has beyond GPU 0's 1 GB give no expert more)
        def line(dev, free, layers):
            return self.gm.parse(f"VRAM dev={dev} pci=0000:0{dev + 1}:00.0 total={24 * GB} free={free} weights={GB} "
                                 f"kv=0 kv_ram=0 act=0 drafter=0 window=0 pool={10 * GB} tail=0 borrow=0 vision=0 "
                                 f"vision_own=0 prompt=0 tp=0 tp_n=2 slots=2000 ctx=65536 layers={layers}")
        a, b, c, d = self.gm.view([line(0, GB, "0-20"), line(1, 9 * GB, "0-20"), line(2, 5 * GB, "21-42"),
                                   line(3, 3 * GB, "21-42")], [], engine_pid=100)
        self.assertEqual((a["tp_group"], b["tp_group"], c["tp_group"], d["tp_group"]), (0, 0, 2, 2))
        self.assertEqual((a["tp_room"], b["tp_room"]), (a["free"], a["free"]))
        self.assertGreater(b["free"] - b["tp_room"], 7 * GB)
        self.assertEqual((c["tp_room"], d["tp_room"]), (d["free"], d["free"]))
        (h,) = self.gm.view([self.r], [], engine_pid=100)        # (no group: as before)
        self.assertNotIn("tp_room", h)

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

    def test_later_lines_follow_a_picture(self):
        """maya-ds4 prints its VRAM lines again when a picture's room is lent to its vision tower and when it is given
        back: the map of that GPU follows them, the others stay."""
        from types import SimpleNamespace
        from serve.server import StrataEngine
        other = dict(self.r, dev=1, pci="0000:02:00")
        eng = SimpleNamespace(vram=[self.r, other])
        lent = LINE.replace("vision=0", "vision=0 vision_own=334495744").replace("pool=15536793108", "pool=15202297364")
        StrataEngine._vram_line(eng, lent)
        self.assertEqual((eng.vram[0]["vision_own"], eng.vram[0]["pool"], eng.vram[1]), (334495744, 15202297364, other))
        StrataEngine._vram_line(eng, LINE)
        self.assertEqual(eng.vram, [self.r, other])
        StrataEngine._vram_line(eng, "VRAM nonsense\n")      # (not a GPU's line: nothing changes)
        self.assertEqual(eng.vram, [self.r, other])


class GpuRelease(unittest.TestCase):
    """A reload starts the new engine only once the GPUs have dropped the old one (its memory is freed after it ends)."""

    def test_waits_until_no_gpu_lists_the_engine(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        seen = [{4242, 7}, {4242, 7}, {7}]
        with mock.patch.object(T, "gpu_pids", side_effect=lambda: seen.pop(0) if len(seen) > 1 else seen[0]), \
                mock.patch.object(SV.time, "sleep") as sleep, \
                mock.patch.object(SV.sys, "platform", "linux"):
            SV.wait_gpu_release(4242, timeout=30)
        self.assertEqual(seen, [{7}])
        self.assertEqual(sleep.call_count, 2)

    def test_without_nvml_a_short_wait(self):
        from unittest import mock
        from serve import server as SV, telemetry as T
        with mock.patch.object(T, "gpu_pids", return_value=None), mock.patch.object(SV.time, "sleep") as sleep, \
             mock.patch.object(SV.sys, "platform", "linux"):
            SV.wait_gpu_release(4242)
        sleep.assert_called_once()

    def test_mac_waits_until_free_memory_stops_rising(self):
        from unittest import mock
        from serve import server as SV
        gb = 1 << 30
        free = [4 * gb, 20 * gb, 40 * gb, 40 * gb, 90 * gb]
        with mock.patch.object(SV, "mac_free_bytes", side_effect=lambda: free.pop(0)), \
                mock.patch.object(SV.time, "sleep") as sleep, mock.patch.object(SV.sys, "platform", "darwin"):
            SV.wait_gpu_release(4242)
        self.assertEqual(sleep.call_count, 3)                   # rising, rising, settled: started
        self.assertEqual(free, [90 * gb])

    def test_mac_free_bytes_reads_vm_stat(self):
        from unittest import mock
        from serve import server as SV
        out = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:      10.\n"
               "Pages active:   99.\nPages inactive:  20.\nPages speculative:  3.\nPages purgeable:  2.\n")
        with mock.patch.object(SV.subprocess, "run", return_value=mock.Mock(stdout=out)):
            self.assertEqual(SV.mac_free_bytes(), 35 * 16384)
        with mock.patch.object(SV.subprocess, "run", return_value=mock.Mock(stdout="")):
            self.assertIsNone(SV.mac_free_bytes())


class NvmlPids(unittest.TestCase):
    """The GPU map finds an engine run through a launcher: NVML lists a process the launcher started, not it."""

    def test_its_own_pid_when_listed_or_nothing_is(self):
        from unittest import mock
        from serve import server as SV
        with mock.patch.dict(sys.modules, {"psutil": None}):     # (nothing is looked up then)
            self.assertEqual(SV.nvml_pids(85, {85, 200}), {85})
            self.assertEqual(SV.nvml_pids(63, set()), {63})
            self.assertEqual(SV.nvml_pids(None, {85}), set())
            self.assertEqual(SV.nvml_pids(63, {85}), {63})        # (no psutil: the pid alone)

    def test_the_processes_a_launcher_started(self):
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil is not installed")
        import subprocess
        from serve import server as SV
        # (a launcher that does not exec: python starting python, which sleeps)
        launcher = subprocess.Popen([sys.executable, "-c", "import subprocess, sys; subprocess.run([sys.executable, "
                                     "'-c', 'import sys, time; print(flush=True); time.sleep(60)'])"],
                                    stdout=subprocess.PIPE, text=True)
        try:
            launcher.stdout.readline()                           # (the grandchild runs)
            pids = SV.nvml_pids(launcher.pid, {1 << 30})
            self.assertIn(launcher.pid, pids)
            self.assertGreater(len(pids), 1)                     # (and the grandchild: a venv's on Windows, more)
        finally:
            for p in psutil.Process(launcher.pid).children(recursive=True):
                p.kill()
            launcher.kill()
            launcher.wait()
            launcher.stdout.close()


class Offline(unittest.TestCase):
    """The page works without the internet: its scripts, styles, fonts and icons are all in serve/web (Preact, htm
    and Lucide vendored with their notices) - nothing is loaded from another host."""
    WEB = ROOT / "serve" / "web"

    def test_nothing_loaded_from_the_network(self):
        import re
        remote = re.compile(r"""(?:\b(?:src|href|action)\s*=\s*|\burl\(\s*|\bfrom\s+|\bimport\s*\(\s*|"""
                            r"""\bfetch\(\s*|@import\s+|new\s+(?:WebSocket|EventSource|Worker)\(\s*)["'`]?(?:https?:)?//""",
                            re.I)
        found = []
        for p in sorted(self.WEB.rglob("*")):
            if p.suffix not in (".js", ".css", ".html", ".svg", ".webmanifest"):
                continue
            for n, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if remote.search(line) and "w3.org/" not in line:
                    found.append(f"{p.relative_to(ROOT)}:{n}: {line.strip()[:120]}")
        self.assertEqual(found, [])

    def test_the_vendored_code_keeps_its_notices(self):
        self.assertIn("The MIT License", (self.WEB / "preact.js").read_text(encoding="utf-8")[:3000])
        self.assertIn("Apache License, Version 2.0", (self.WEB / "preact.js").read_text(encoding="utf-8")[:3000])
        self.assertIn("ISC License", (self.WEB / "LICENSE.lucide.txt").read_text(encoding="utf-8"))
        self.assertTrue((self.WEB / "fonts" / "OFL.txt").is_file())
        index = (self.WEB.parent / "web" / "index.html").read_text(encoding="utf-8")
        self.assertNotIn("fonts.googleapis", index)


# the page's own modules in Node (skipped without it), the browser's few globals they touch at load stubbed out
PAGE_STUBS = """
const noop = () => {};
const el = new Proxy({}, {get: (t, k) => (k in t ? t[k] : k === "dataset" || k === "style" ? (t[k] = {})
  : k === "classList" ? (t[k] = {add: noop, remove: noop, toggle: noop}) : noop)});
globalThis.matchMedia = () => ({matches: false, addEventListener: noop, addListener: noop});
globalThis.localStorage = {getItem: () => null, setItem: noop, removeItem: noop};
globalThis.document = {documentElement: el, body: el, head: el, addEventListener: noop, querySelector: () => null,
                       createElement: () => el};
globalThis.window = globalThis; globalThis.addEventListener = noop;
globalThis.location = {href: "http://x/", hash: "", search: "", pathname: "/"};
const web = process.argv[2];
const st = await import(web + "/state.js"), sv = await import(web + "/server.js");
"""


@unittest.skipUnless(__import__("shutil").which("node"), "Node.js is not installed")
class PageLogic(unittest.TestCase):
    def node(self, body):
        import subprocess
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t.mjs"
            p.write_text(PAGE_STUBS + body, encoding="utf-8")
            r = subprocess.run(["node", str(p), (ROOT / "serve" / "web").resolve().as_uri()], capture_output=True,
                               encoding="utf-8", timeout=60)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        return json.loads(r.stdout.strip().splitlines()[-1])

    def test_thinking_pill_covers_the_chosen_level_only(self):
        # the owner's phone: Low chosen, the fill spanned Off and Low.  Every model's levels, every level: the pill sits
        # on that level's segment alone
        got = self.node("""
const out = {};
for (const model of ["glm-5.3-flash", "deepseek-v4.1-flash", "minimax-m3.1-flash", "cyber-frost-3.8-ps"]) {
  st.setFamily(model);
  out[model] = st.LEVELS.map((l) => [l.id, st.thinkPill(l.id)]);
}
console.log(JSON.stringify(out));""")
        self.assertEqual([lv for lv, _ in got["glm-5.3-flash"]], ["none", "low", "high", "max"])
        self.assertEqual(len(got["minimax-m3.1-flash"]), 7)
        for model, levels in got.items():
            n = len(levels)
            for i, (lv, pill) in enumerate(levels):
                with self.subTest(model=model, level=lv):
                    self.assertEqual((pill["i"], pill["n"]), (i, n))
                    self.assertAlmostEqual(pill["at"], 100 * i / n)
                    self.assertAlmostEqual(pill["width"], 100 / n)      # (one segment wide: not from 0 up to it)
        self.assertEqual(got["glm-5.3-flash"][1][1]["at"], 25)            # Low: the second quarter only

    def test_free_room_a_tensor_parallel_partner_cannot_use_promises_no_experts(self):
        # a --tp 2 pair: GPU 1 has 9 GB free, GPU 0 (with the drafter) 1 GB - the members hold the same expert slots, so
        # a restart gives none more; the room is idle, a DSpark block moved there can use it
        got = self.node("""
const {freeAdvice} = await import(web + "/monitor.js");
const G = 2 ** 30, more = (g, b) => Math.max(0, Math.floor(b / (g.parts.pool / g.slots)));
const where = (gs) => gs.map((g) => "GPU " + g.index).join(" and "), pools = () => "the pools";
const m = (index, free, room, drafter) => ({index, dev: index, free, tp_room: room, tp_group: 0, slots: 1000,
                                           parts: {pool: 10 * G, drafter}});
const pair = [m(0, G, G, G), m(1, 9 * G, G, 0)], both = [m(0, 5 * G, 5 * G, 0), m(1, 7 * G, 5 * G, 0)];
const plain = [{index: 0, dev: 0, free: 5 * G, slots: 1000, parts: {pool: 10 * G}}];
console.log(JSON.stringify({
  pair: freeAdvice([pair[1]], pair, {engine_kind: "ds4"}, where, pools, more),
  mm3: freeAdvice([pair[1]], pair, {engine_kind: "mm3"}, where, pools, more),
  both: freeAdvice(both, both, {engine_kind: "ds4"}, where, pools, more),
  plain: freeAdvice(plain, plain, {engine_kind: "ds4"}, where, pools, more)}));""")
        self.assertNotIn("restart gives the pools", got["pair"])
        self.assertIn("GB of GPU 1 sit idle", got["pair"])
        self.assertIn("MAYA_DS4_DRAFT_DEV", got["pair"])
        self.assertNotIn("MAYA_DS4_DRAFT_DEV", got["mm3"])
        # (both have room: 5 GB each grows the pair's slots once - 400 experts, not 800; GPU 1's other 2 GB idle)
        self.assertIn("are free: a restart gives the pools about 400 more experts.", got["both"])
        self.assertIn("GB of GPU 1 sit idle", got["both"])
        self.assertTrue(got["plain"].endswith("GB of GPU 0 are free: a restart gives the pools about 400 more experts."))

    def test_speculation_says_off_waiting_or_the_acceptance(self):
        # the Monitor's Speculation, three states: off (no drafter, or the request asked for none: "off" in the field and
        # in the figure), drafting on with nothing drafted yet ("-", never a fake 0%), drafting on with drafts (accepted of
        # drafted, %) - live from the STAT draft counts, else the last answer's
        got = self.node("""
const now = 1000;
const eng = (modes) => ({speculation: modes});
const live = (o) => ({state: "generating", elapsed_s: 5, ...o});
const stat = (o) => ({now: {time: now - 1, ...o}});
const v = (m) => sv.speculationView(m, now);
const out = {
  none: v({engine: eng([]), live: {state: "idle"}, requests: []}),
  noneLive: v({engine: eng([]), live: live({}), tiers: stat({draft_acc: 0, draft_off: 0}), requests: []}),
  specOne: v({engine: eng(["MTP"]), live: live({spec_off: true}), tiers: stat({draft_acc: 0, draft_off: 0}), requests: []}),
  lastSpecOne: v({engine: eng(["MTP"]), live: {state: "idle"}, requests: [{spec_off: true, drafts_offered: 0, drafts_accepted: 0}]}),
  idleNever: v({engine: eng(["MTP"]), live: {state: "idle"}, requests: []}),
  idleZero: v({engine: eng(["MTP"]), live: {state: "idle"}, requests: [{drafts_offered: 0, drafts_accepted: 0}]}),
  idleNull: v({engine: eng(["DFlash2"]), live: {state: "idle"}, requests: [{drafts_offered: null, drafts_accepted: null}]}),
  liveNone: v({engine: eng(["MTP"]), live: live({}), tiers: stat({draft_acc: 0, draft_off: 0}), requests: []}),
  liveNoKeys: v({engine: eng(["Lookup"]), live: live({}), tiers: stat({vram_hit: 1}), requests: []}),
  liveOld: v({engine: eng(["MTP"]), live: live({elapsed_s: 5}), tiers: {now: {time: now - 100, draft_acc: 9, draft_off: 10}}, requests: []}),
  beforeThis: v({engine: eng(["MTP"]), live: live({elapsed_s: 2}), tiers: {now: {time: now - 20, draft_acc: 9, draft_off: 10}}, requests: []}),
  live: v({engine: eng(["MTP"]), live: live({}), tiers: stat({draft_acc: 124, draft_off: 200}), requests: []}),
  liveAllKept: v({engine: eng(["DSpark", "Lookup"]), live: live({}), tiers: stat({draft_acc: 50, draft_off: 50}), requests: []}),
  liveRatio: v({engine: eng(["DSpark"]), live: live({}), tiers: stat({drafts: 0.5}), requests: []}),
  zeroKept: v({engine: eng(["MTP"]), live: live({}), tiers: stat({draft_acc: 0, draft_off: 40}), requests: []}),
  last: v({engine: eng(["MTP"]), live: {state: "idle"}, requests: [{drafts_offered: 1200, drafts_accepted: 600}]}),
  oldServer: v({engine: {}, live: {state: "idle"}, requests: []}),
};
console.log(JSON.stringify(out));""")
        for k in ("none", "noneLive", "specOne"):                              # state 1: off, in the field and the figure
            self.assertEqual((got[k]["state"], got[k]["name"], got[k]["text"]), ("off", "off", "off"), k)
        self.assertEqual((got["lastSpecOne"]["state"], got["lastSpecOne"]["name"], got["lastSpecOne"]["text"]),
                         ("off", "MTP", "off"))                                # (idle: the engine's drafter, the last answer had none)
        for k in ("idleNever", "idleZero", "idleNull", "liveNone", "liveNoKeys", "liveOld", "beforeThis"):
            self.assertEqual((got[k]["state"], got[k]["text"]), ("waiting", "\u2014"), k)   # state 2: on, nothing drafted yet
        self.assertEqual(got["idleNever"]["name"], "MTP")
        self.assertEqual(got["idleNull"]["name"], "DFlash2")
        self.assertEqual((got["live"]["state"], got["live"]["text"], got["live"]["name"]), ("kept", "62% (124 of 200)", "MTP"))
        self.assertEqual(got["liveAllKept"]["text"], "100% (50 of 50)")
        self.assertEqual(got["liveAllKept"]["name"], "DSpark + Lookup")
        self.assertEqual(got["liveRatio"]["text"], "50%")                       # (an engine that gives the share only)
        self.assertEqual(got["zeroKept"]["text"], "0% (0 of 40)")                  # a real 0%: drafted, none kept
        self.assertEqual(got["last"]["text"], "50% (600 of 1,200)")
        self.assertEqual(got["oldServer"]["state"], "none")

    def test_experts_on_the_ssd_have_their_bytes_like_the_other_tiers(self):
        got = self.node("""
const n = {vram_used: 300, ram_used: 500, vram_gb: 12.5, ram_gb: 20.25, disk_gb: 31.4};
const ds = {vram_used: 40, ram_used: 100, cpu_n: 400, vram_gb: 9, ram_gb: 25, disk_gb: 75.5};
console.log(JSON.stringify({glm: sv.tierShares(n, {experts: 2000}), none: sv.tierShares({vram_used: 5, ram_used: 5}, {experts: 10}),
  ds4: sv.tierShares(ds, {engine_kind: "ds4"}), full: sv.tierShares({vram_used: 10, ram_used: 0, disk_gb: 0}, {experts: 10})}));""")
        self.assertEqual((got["glm"]["vu"], got["glm"]["ru"], got["glm"]["disk"]), (300, 500, 1200))
        self.assertEqual((got["glm"]["vramGb"], got["glm"]["ramGb"], got["glm"]["diskGb"]), (12.5, 20.25, 31.4))
        self.assertEqual((got["ds4"]["total"], got["ds4"]["disk"], got["ds4"]["diskGb"]), (440, 300, 75.5))
        self.assertIsNone(got["none"]["diskGb"])                                   # (an engine that reports no bytes: the count alone)
        self.assertEqual((got["full"]["disk"], got["full"]["diskGb"]), (0, 0))

    def test_qwen_file_reads_are_named_for_what_they_count_and_the_ssd_tip_needs_ssd_only_experts(self):
        # Mercury's Monitor, Qwen3.8 UD-IQ2_XS on 1x V100: 13,199 experts in VRAM, 11,377 in pinned RAM, none on the SSD
        # alone - but 8.23 experts a token read from the model file; the card said "From SSD" and advised more RAM
        got = self.node("""
const m = await import(web + "/monitor.js");
const q = {engine_kind: "qwen", experts: 24576}, n = {vram_used: 13199, ram_used: 11377, disk: 8.23, vram_hit: 0.93};
const g = {experts: 2000}, gn = {vram_used: 300, ram_used: 500, disk: 4, vram_hit: 0.9};
console.log(JSON.stringify({set: m.tierSet(q).find((x) => x.key === "disk"), glmSet: m.tierSet(g).find((x) => x.key === "disk"),
  homes: m.tierTip(n, q, {disk: sv.tierShares(n, q).disk}),
  ssdOnly: m.tierTip({...n, ram_used: 9000}, q, {disk: sv.tierShares({...n, ram_used: 9000}, q).disk}),
  glm: m.tierTip(gn, g, {disk: sv.tierShares(gn, g).disk}), stale: m.tierTip(gn, g, {disk: 1200, stale: true}),
  cfCpu: m.tierTip({vram_used: 100, ram_used: 24476, disk: 0, vram_hit: 0.3}, q, {disk: 0})}));""")
        self.assertEqual(got["set"]["label"], "From the model file")
        self.assertIn("page cache or SSD", got["set"]["sub"])
        self.assertEqual(got["glmSet"]["label"], "From SSD")                    # (GLM's own reads: unchanged)
        self.assertEqual(got["homes"], "")                                      # every expert has a home: no "more RAM"
        self.assertTrue(got["ssdOnly"].startswith("Many experts are read from the SSD"))   # 2,377 on the SSD alone
        self.assertTrue(got["glm"].startswith("Many experts are read from the SSD"))
        self.assertEqual(got["stale"], "")
        self.assertTrue(got["cfCpu"].startswith("Most experts run on the CPU"))

    def test_about_has_the_support_line_always_and_the_update_card_none(self):
        # owner 2026-11-05: one line on About (always, also while an update shows), its link on "supporting me" in bold
        # violet; no row under an update's button; the sidebar's heart stays
        web = ROOT / "serve" / "web"
        js = (web / "settings.js").read_text(encoding="utf-8")
        line = js[js.index("const SupportLine"):js.index("function UpdateCard")]
        body = line.split("<span>", 1)[1].split("</span></p>", 1)[0].replace('${" "}', " ")
        text = re.sub(r"\s+", " ", re.sub(r"<[^>]+>", "", body))
        self.assertEqual(text, "I build Maya because I believe everyone should have access to great local AI. If you enjoy "
                               "what I'm building, consider supporting me. Every little bit helps keep the project alive "
                               "and growing. ❤️")
        self.assertRegex(line, r'<a class="support-link" href=\$\{SUPPORT_URL\} target="_blank" rel="noopener"\s+'
                               r'title="[^"]*">supporting me</a>')
        self.assertNotIn('name="heart"', line)                               # (no second heart before the text)
        card = js[js.index("function UpdateCard"):js.index("\n}\n", js.index("function UpdateCard"))]
        self.assertNotIn("SupportLine", card)
        self.assertIn("<${UpdateCard}/>\n    <${SupportLine}/>", js)               # (About: always, under the update)
        css = (web / "settings.css").read_text(encoding="utf-8")
        rule = re.search(r"\.set-support \.support-link \{([^}]*)\}", css).group(1)
        self.assertIn("font-weight: 700", rule)
        self.assertIn("color: var(--st-accent-text)", rule)
        self.assertEqual(len(re.findall(r"--st-accent-text:\s*#", (web / "tokens.css").read_text(encoding="utf-8"))), 2)
        self.assertIn("heart-link", (web / "sidebar.js").read_text(encoding="utf-8"))   # (the sidebar's heart stays)

    def test_context_slider_reaches_each_models_own_top(self):
        # the Models page's Context size: Settings > Model's stops over each model's range (its trained context, up to
        # 1M), the size a config has kept as a stop of its own
        got = self.node("""
const r1m = {min: 4096, max: 1048576}, r256 = {min: 4096, max: 262144};
st.S.ctxInfo = {...r1m, context: 131072};
console.log(JSON.stringify({m1: sv.ctxStopsIn(r1m, 131072), m256: sv.ctxStopsIn(r256, 131072, 100000),
                            settings: sv.ctxStops(), none: sv.ctxStopsIn(null, 1),
                            win: sv.ctxHint(16384, {now: 131072, win: 32768, whole: true, ram: sv.GLM_KV_BYTES}),
                            past: sv.ctxHint(131072, {now: 131072, win: 32768, whole: true, ram: sv.GLM_KV_BYTES})}));""")
        self.assertEqual((got["m1"][0], got["m1"][-1]), (4096, 1048576))
        self.assertEqual(got["m1"], got["settings"])                     # (the same stops as Settings > Model's)
        self.assertEqual(got["m256"][-1], 262144)
        self.assertIn(100000, got["m256"])
        self.assertEqual(got["none"], [])
        self.assertEqual(got["win"], "Up to 32K the whole cache stays in VRAM.")
        self.assertTrue(got["past"].startswith("≈ 1.5 GB of RAM for the context"), got["past"])


if __name__ == "__main__":
    unittest.main()

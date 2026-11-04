"""serve/test_model_api.py - Maya's model API: /api/tags, /api/ps, /api/show, /api/chat, /api/generate, /api/pull,
/api/delete, the swap on request and keep_alive - against the mock engine, no GPU.

    python -m unittest serve.test_model_api -v
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import model_api, model_pull  # noqa: E402
from serve.server import Service  # noqa: E402
from serve.test_models import Served, glm_quant  # noqa: E402


class Pure(unittest.TestCase):
    def test_keep_alive(self):
        k = model_api.parse_keep_alive
        self.assertEqual((k(None), k(""), k(0), k(-1), k("-1"), k("0")), (None, None, 0.0, -1.0, -1.0, 0.0))
        self.assertEqual((k("5m"), k("1h30m"), k(30), k("90s"), k("-5m")), (300.0, 5400.0, 30.0, 90.0, -1.0))
        for bad in ("soon", "5x", True, [1]):
            with self.assertRaises(ValueError):
                k(bad)

    def test_requests_translate(self):
        o = model_api.chat_to_openai({"messages": [{"role": "user", "content": "hi", "images": ["QUJD"]}],
                                   "options": {"temperature": 0.2, "num_predict": 7, "stop": ["x"], "repeat_penalty": 1.1},
                                   "stream": False, "format": "json"})
        self.assertEqual((o["stream"], o["temperature"], o["max_tokens"], o["stop"], o["repetition_penalty"]),
                         (False, 0.2, 7, ["x"], 1.1))
        self.assertEqual(o["response_format"], {"type": "json_object"})
        self.assertEqual(o["messages"][0]["content"][1]["image_url"]["url"], "data:image/png;base64,QUJD")
        g = model_api.generate_to_openai({"prompt": "p", "system": "s"})
        self.assertEqual([m["role"] for m in g["messages"]], ["system", "user"])
        self.assertTrue(g["stream"])                      # (streamed unless told not to)

    def test_translator(self):
        t = model_api.Translator("m:latest", "chat")
        ch = lambda d, **k: {"choices": [{"delta": d, "finish_reason": k.get("fin")}], **k.get("x", {})}   # noqa: E731
        out = t.feed(ch({"content": "he"})) + t.feed(ch({"reasoning_content": "hm"})) + t.feed(None)
        self.assertEqual([o["message"].get("content") or o["message"].get("thinking") for o in out], ["he", "hm"])
        t.feed(ch({"tool_calls": [{"index": 0, "function": {"name": "f", "arguments": "{\"a\":"}}]}))
        t.feed(ch({"tool_calls": [{"index": 0, "function": {"arguments": "1}"}}]}, fin="tool_calls",
                  x={"usage": {"prompt_tokens": 3, "completion_tokens": 2}}))
        calls = t.finish_calls()[0]["message"]["tool_calls"]
        self.assertEqual(calls, [{"function": {"name": "f", "arguments": {"a": 1}}}])
        d = t.done()
        self.assertEqual((d["done"], d["prompt_eval_count"], d["eval_count"]), (True, 3, 2))

    def test_names(self):
        import tempfile
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            p = glm_quant(d, "Maya-L")
            self.assertEqual(model_api.public_name(p), p.stem)
            for n in (p.stem, p.stem + ":latest", p.name, p.stem.removeprefix("maya-")):
                self.assertEqual(model_api.find_config(d, n), p, n)
            for n in ("", None, "../x", "nope", "maya-"):
                self.assertIsNone(model_api.find_config(d, n), n)


class Api(Served):
    def setUp(self):
        super().setUp()
        self.svc.model_management = True
        self.run_name = model_api.public_name(self.cfg)

    def raw(self, method, path, body=None, headers=None):
        h = {"Content-Type": "application/json"}
        h.update(headers or {})
        r = urllib.request.Request(self.base + path, method=method, headers=h,
                                   data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, resp.read().decode(), resp.headers
        except urllib.error.HTTPError as e:
            with e:
                return e.code, e.read().decode(), e.headers

    def test_tags_ps_show(self):
        s, b = self.req("GET", "/api/tags")
        names = {m["name"] for m in b["models"]}
        self.assertEqual((s, names), (200, {self.run_name + ":latest", model_api.public_name(self.other) + ":latest"}))
        s, b = self.req("GET", "/api/ps")
        self.assertEqual([m["name"] for m in b["models"]], [self.run_name + ":latest"])
        s, b = self.req("POST", "/api/show", {"model": self.run_name, "verbose": True})
        self.assertEqual((s, b["maya"]["file"]), (200, self.cfg.name))
        self.assertIn("delete_plan", b["maya"])
        self.assertEqual(self.req("POST", "/api/show", {"model": "nope"})[0], 404)

    def test_chat_and_generate(self):
        s, b = self.req("POST", "/api/chat", {"model": self.run_name, "stream": False,
                                              "messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual((s, b["done"], b["message"]["role"]), (200, True, "assistant"))
        self.assertIn("hello", b["message"]["content"])
        s, body, h = self.raw("POST", "/api/chat", {"model": self.run_name + ":latest",
                                                    "messages": [{"role": "user", "content": "hi"}]})
        lines = [json.loads(x) for x in body.splitlines()]
        self.assertTrue(lines[-1]["done"])
        self.assertIn("hello", "".join(x["message"]["content"] for x in lines[:-1]))
        self.assertTrue(h["Content-Type"].startswith("application/x-ndjson"))
        s, b = self.req("POST", "/api/generate", {"model": self.run_name, "prompt": "hi", "stream": False})
        self.assertEqual((s, b["done"]), (200, True))
        self.assertIn("hello", b["response"])
        self.assertEqual(self.req("POST", "/api/chat", {"model": "nope", "messages": [{"role": "user", "content": "x"}]})[0], 404)

    def test_keep_alive(self):
        self.svc.idle_unload_s = 600.0
        self.req("POST", "/api/generate", {"model": self.run_name, "prompt": "hi", "stream": False, "keep_alive": "5m"})
        self.assertEqual(self.svc.idle_limit(), 300.0)
        self.req("POST", "/api/generate", {"model": self.run_name, "prompt": "hi", "stream": False, "keep_alive": -1})
        self.assertEqual(self.svc.idle_limit(), 0.0)           # (for ever: never unloaded)
        self.req("POST", "/api/generate", {"model": self.run_name, "prompt": "hi", "stream": False})
        self.assertEqual(self.svc.idle_limit(), 600.0)         # (none given: the config's own)
        s, b = self.req("POST", "/api/generate", {"model": self.run_name, "prompt": "x", "keep_alive": "soon"})
        self.assertEqual(s, 400)
        with mock.patch.object(Service, "unload", return_value="unloaded") as un:
            s, b = self.req("POST", "/api/generate", {"model": self.run_name, "keep_alive": 0})
            self.assertEqual((s, b["done_reason"]), (200, "unload"))
            un.assert_called_once()

    def test_swap_on_request(self):
        other = model_api.public_name(self.other)
        msg = [{"role": "user", "content": "hi"}]
        s, b = self.req("POST", "/api/chat", {"model": other, "messages": msg})
        self.assertEqual(s, 409)                                 # (not started by maya.sh: cannot switch itself)
        os.environ["MAYA_RESTART_ON_UPDATE"] = "1"
        with mock.patch.object(Service, "_restart_into") as restart:
            s, body, h = self.raw("POST", "/api/chat", {"model": other, "messages": msg})
            self.assertEqual((s, h["Retry-After"]), (503, "30"))
            self.assertIn("loading", json.loads(body))
            restart.assert_called_once()
            s, body, _ = self.raw("POST", "/models/load", {"model": other})   # (llama.cpp's load names it too)
        self.assertEqual(s, 503)

    def test_delete_keeps_what_another_model_runs_from(self):
        l = glm_quant(self.d, "Maya-L")
        m = glm_quant(self.d, "Maya-L2", folder=l.parent / "GLM-5.3-Flash" / "Maya-L", pack_name="pack2")
        folder = self.d / "GLM-5.3-Flash" / "Maya-L"
        s, b = self.req("DELETE", "/api/delete", {"model": model_api.public_name(m)})
        self.assertEqual(s, 200)
        self.assertFalse(m.exists())
        self.assertTrue((folder / "pack" / "index.txt").exists())            # the other model's pack stays
        self.assertTrue((folder / "vision" / "mmproj.gguf").exists())        # a file both run from stays
        self.assertTrue(l.exists())
        # not the running model; not a name that is not here
        self.assertEqual(self.req("DELETE", "/api/delete", {"model": self.run_name})[0], 409)
        self.assertTrue(self.cfg.exists())
        self.assertEqual(self.req("DELETE", "/api/delete", {"model": "nope"})[0], 404)

    def test_delete_with_files_linked_from_another_model(self):
        """The Engram tables are hard linked (or symlinked) between model folders: a hard link frees nothing and the
        other folder keeps its data; a symlink into this folder keeps the file it points at."""
        l = glm_quant(self.d, "Maya-L")
        m = glm_quant(self.d, "Maya-L2")
        lf, mf = self.d / "GLM-5.3-Flash" / "Maya-L", self.d / "GLM-5.3-Flash" / "Maya-L2"
        shard = lf / "GLM-5.3-Flash-Maya-L-00001-of-00003.gguf"
        try:
            os.link(shard, mf / "tables-hard.bin")
            os.symlink(lf / "GLM-5.3-Flash-Maya-L-00002-of-00003.gguf", mf / "tables-sym.bin")
        except (OSError, NotImplementedError):
            self.skipTest("this file system cannot link")
        from serve import models as M
        plan = M.delete_plan(self.d, l)
        total = sum(f.stat().st_size for f in lf.rglob("*") if f.is_file())
        self.assertTrue(plan["ok"])
        self.assertIn(total - 1000, [plan["bytes"], sum(i["bytes"] for i in plan["items"])])   # (the hard link not counted)
        s, b = self.req("DELETE", "/api/delete", {"model": model_api.public_name(l)})
        self.assertEqual(s, 200)
        self.assertEqual((mf / "tables-hard.bin").read_bytes(), b"g" * 1000)                 # the link's data is intact
        self.assertEqual((mf / "tables-sym.bin").read_bytes(), b"g" * 1000)                  # the symlink still reads
        self.assertTrue((lf / "GLM-5.3-Flash-Maya-L-00002-of-00003.gguf").exists())          # what it points at stayed
        self.assertTrue(m.exists())

    def test_api_key(self):
        self.svc.api_key = "k"
        for method, path in (("GET", "/api/tags"), ("POST", "/api/pull"), ("DELETE", "/api/delete"),
                             ("POST", "/api/chat")):
            self.assertEqual(self.raw(method, path, {} if method != "GET" else None)[0], 401, path)
        self.assertEqual(self.raw("GET", "/api/tags", None, {"Authorization": "Bearer k"})[0], 200)

    def test_off_when_model_management_is_off(self):
        self.svc.model_management = False
        self.assertEqual(self.raw("GET", "/api/tags")[0], 404)


class Pull(Api):
    SCRIPT = ("import sys,time\n"
              "for i in (1,2,3):\n"
              "    sys.stdout.write('\\r  %d.0 / 3.0 GB (%d%%) · 10 MB/s · 4 files at once   ' % (i, i*33)); "
              "sys.stdout.flush(); time.sleep(0.1)\n"
              "print('\\n  done {model}')\n")

    def hook(self, script=None, code=0):
        script = (script or self.SCRIPT) + (f"sys.exit({code})\n" if code else "")
        os.environ["MAYA_PULL_COMMAND"] = json.dumps([sys.executable, "-c", script])
        self.addCleanup(os.environ.pop, "MAYA_PULL_COMMAND", None)

    def test_catalog_and_pull_stream(self):
        s, b = self.req("GET", "/api/catalog")
        ids = {m["id"] for m in b["models"]}
        self.assertTrue({"glm-maya-s", "deepseek-maya-xs", "minimax-maya-s", "cyber-frost"} <= ids)
        self.hook()
        s, body, _ = self.raw("POST", "/api/pull", {"model": "glm-maya-s"})
        ev = [json.loads(x) for x in body.splitlines()]
        self.assertEqual(ev[-1], {"status": "success"})
        down = [e for e in ev if e.get("total")]
        self.assertTrue(down and down[-1]["total"] == 3 * 10 ** 9)
        self.assertEqual(self.req("GET", "/api/pull/status")[1]["pull"]["state"], "success")

    def test_pull_not_in_catalog_and_failure(self):
        self.assertEqual(self.req("POST", "/api/pull", {"model": "llama3"})[0], 400)
        self.hook("import sys\nprint('the disk is full')\n", code=3)
        s, body, _ = self.raw("POST", "/api/pull", {"model": "glm-maya-s", "stream": False})
        self.assertEqual(s, 500)
        self.assertIn("disk is full", body)

    def test_one_download_at_a_time_and_cancel(self):
        self.hook("import time\ntime.sleep(30)\n")
        first = self.svc.pull_slot().start("glm-maya-s", None)
        s, b = self.req("POST", "/api/pull", {"model": "deepseek-maya-xs", "stream": False})
        self.assertEqual(s, 409)
        s, b = self.req("DELETE", "/api/delete", {"model": model_api.public_name(self.other)})
        self.assertEqual(s, 409)                                  # (no deleting while a download runs)
        self.assertEqual(self.req("POST", "/api/pull/cancel", {})[1], {"cancelled": True})
        first.proc.wait(10)
        self.assertEqual(first.snapshot()["state"], "cancelled")

    def test_progress_line(self):
        m = model_pull.PROGRESS.search("  12.3 / 96.5 GB (13%) · 40 MB/s · 4 files at once")
        self.assertEqual((m.group(1), m.group(2), m.group(3), m.group(4)), ("12.3", "96.5", "13", "40"))
        cmd = model_pull.setup_command("glm-maya-s", {"installer": {"models_dir": "/m"}, "backend": "hip"})
        self.assertIn("--download-model", cmd)
        self.assertEqual(cmd[cmd.index("--models-dir") + 1], "/m")
        self.assertEqual(cmd[cmd.index("--backend") + 1], "hip")


class OtherQuantsTest(unittest.TestCase):
    """The Models page's "Other quants": the other files the setup offers, each through the setup's own flags."""

    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root, True)

    def cat(self, backend="cuda", ccs=(86,)):
        return model_pull.catalog(self.root, {}, {"backend": backend}, list(ccs), other_sizes={})

    def test_the_setup_lists_match(self):
        sys.path.insert(0, str(model_pull.ROOT))
        import maya
        ds4 = {k for k, _ in model_pull.DS4_OTHER}
        self.assertEqual(ds4, set(maya.DS4_QUANTS) - set(maya.DS4_MAYA) - set(maya.DS4_BY_HAND))
        self.assertEqual(set(model_pull.GLM_EXL3), {k for k in maya.MODELS if maya.is_exl3(k)} - set(maya.MODELS_BY_HAND))
        self.assertIn("exl3", maya.MM3_QUANTS)
        self.assertEqual(set(model_pull.GATED), set(maya.DS4_ABL))

    def test_groups_names_and_notes(self):
        c = {m["id"]: m for m in self.cat()}
        self.assertEqual(c["glm-maya-s"]["group"], "maya")
        self.assertEqual(c["deepseek-exl3"]["group"], "other")
        self.assertEqual(c["deepseek-exl3"]["name"], "DeepSeek V4.1 · EXL3 2.9 bpw")
        self.assertEqual(c["deepseek-exl3-3.5"]["by"], "LibertAIDAI")
        self.assertEqual(c["deepseek-gguf-q8"]["name"], "DeepSeek V4.1 · UD-Q8_K_XL")
        self.assertIn("deepseek-gguf-iq1_s", c)
        self.assertNotIn("deepseek-gguf-q4", c)                   # (by hand only, as in the setup's list)
        self.assertNotIn("glm-exl3-sc_4.10bpw", c)
        self.assertIn("HF_TOKEN", c["deepseek-abliterated-128g"]["note"])
        self.assertEqual(c["minimax-exl3"]["name"], "MiniMax M3.1 · EXL3 3.5 bpw")
        self.assertEqual(c["glm-exl3-3.05bpw"]["name"], "GLM-5.3 · EXL3 3.05 bpw")
        self.assertEqual(c["deepseek-exl3"]["note"], "")
        self.assertEqual(c["glm-exl3-4.05bpw"]["note"], "")         # (the EXL3 quants: no longer experimental)
        ids = [m["id"] for m in self.cat()]
        self.assertEqual(len(ids), len(set(ids)))

    def test_backends(self):
        old = {m["id"] for m in self.cat(ccs=(61,))}             # (Pascal only: no EXL3, no DeepSeek GGUFs)
        self.assertNotIn("deepseek-exl3", old)
        self.assertNotIn("minimax-exl3", old)
        self.assertNotIn("deepseek-gguf-q8", old)
        self.assertIn("glm-exl3-3.05bpw", old)
        self.assertIn("deepseek-exl3", {m["id"] for m in self.cat(ccs=())})   # (no GPU readable: the setup's check says)
        amd = {m["id"]: m for m in self.cat("hip", ())}
        self.assertEqual(amd["deepseek-exl3"]["note"], "")
        self.assertEqual(amd["minimax-exl3"]["note"], "")
        self.assertEqual({m["group"] for m in self.cat("metal", ())}, {"maya"})

    def test_installed(self):
        (self.root / "maya-ds4.json").write_text(json.dumps(
            {"exe": "x", "args": ["--pack", "p"], "installer": {"family": "deepseek", "ds4_quant": "exl3-3.5"}}))
        c = {m["id"]: m for m in self.cat()}
        self.assertTrue(c["deepseek-exl3-3.5"]["installed"])
        self.assertFalse(c["deepseek-exl3"]["installed"])

    def test_setup_commands(self):
        def which(ident, backend="cuda"):
            cmd = model_pull.setup_command(ident, {"installer": {"models_dir": "/m"}, "backend": backend})
            return cmd[cmd.index("--model"):cmd.index("--download-model")], cmd
        self.assertEqual(which("deepseek-exl3")[0], ["--model", "DeepSeek-V4.1-Flash-1017", "--ds4-quant", "exl3"])
        self.assertEqual(which("deepseek-exl3-3.5")[0], ["--model", "DeepSeek-V4.1-Flash-1017", "--ds4-quant", "exl3-3.5"])
        self.assertEqual(which("deepseek-gguf-iq3_s")[0], ["--model", "DeepSeek-V4.1-Flash-1017", "--ds4-quant", "gguf-iq3_s"])
        self.assertEqual(which("deepseek-abliterated-256g")[0],
                         ["--model", "DeepSeek-V4.1-Flash-1017", "--ds4-quant", "abliterated-256g"])
        self.assertEqual(which("minimax-exl3")[0], ["--model", "MiniMax-M3.1-Flash", "--mm3-quant", "exl3"])
        self.assertEqual(which("glm-exl3-4.05bpw")[0], ["--model", "EXL3-4.05bpw"])
        self.assertEqual(which("glm-maya-s")[0], ["--model", "glm-maya-s"])
        _, cmd = which("deepseek-exl3", "hip")
        self.assertEqual(cmd[cmd.index("--backend") + 1], "hip")
        self.assertEqual(cmd[cmd.index("--models-dir") + 1], "/m")

    def test_the_setup_takes_the_commands(self):
        """maya.py's own argument check takes each command's --model / --ds4-quant / --mm3-quant."""
        sys.path.insert(0, str(model_pull.ROOT))
        import maya
        for ident, fam, key, _ in model_pull.OTHER:
            cmd = model_pull.setup_command(ident, None)
            a = argparse.Namespace(model=cmd[cmd.index("--model") + 1], ds4_quant=None, mm3_quant=None)
            if "--ds4-quant" in cmd:
                a.ds4_quant = cmd[cmd.index("--ds4-quant") + 1]
                self.assertIn(a.ds4_quant, maya.DS4_QUANTS)
            if "--mm3-quant" in cmd:
                a.mm3_quant = cmd[cmd.index("--mm3-quant") + 1]
                self.assertIn(a.mm3_quant, maya.MM3_QUANTS)
            self.assertIsNone(maya.model_option(a), ident)

    def test_pull_takes_an_other_id(self):
        slot = model_pull.Puller(self.root, lambda ident, cfg: [sys.executable, "-c", "print('ok')"])
        p = slot.start("deepseek-exl3", {"backend": "cuda"})
        p.proc.wait(10)
        with self.assertRaises(ValueError):
            model_pull.Puller(self.root, lambda i, c: []).start("deepseek-gguf-q4", None)


if __name__ == "__main__":
    unittest.main()

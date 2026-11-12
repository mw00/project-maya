"""tools/strata/test_soak_run.py - the B12.7 tool-call corpus and tools/strata/soak_run.py against a fake server (CPU).

  * the 50 committed cases equal the generator's output (make_toolcall_corpus.py --check), in the five groups, and
    every expected call names a declared tool;
  * check_case: valid calls pass "parse", and each failure kind is caught (unknown tool, arguments that are not
    JSON, a missing required key, a nested type error, an enum outside the list, too few calls for a parallel case,
    Chinese text lost, an error reply that repeats the failed call);
  * `toolcalls` against a well-behaved fake (OpenAI non-streamed and streamed, Anthropic) parses 100%; a fake that
    writes broken arguments fails;
  * `soak` with two /api/context reloads passes (the conversation codename survives both reloads); a server that
    goes down mid-run is counted as a crash and fails the verdict.
"""
from __future__ import annotations

import contextlib
import http.server
import io
import json
import pathlib
import re
import sys
import tempfile
import threading
import time
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import make_toolcall_corpus as corpus  # noqa: E402
import soak_run as sr  # noqa: E402

CJK = re.compile(r"[㐀-鿿]")


def fill(schema: dict, cjk: bool):
    t = schema.get("type")
    if "enum" in schema:
        return schema["enum"][0]
    if t == "object":
        props = schema.get("properties", {})
        return {k: fill(v, cjk) for k, v in props.items()}
    if t == "array":
        return [fill(schema.get("items", {"type": "string"}), cjk)]
    if t == "integer":
        return 2
    if t == "number":
        return 1.5
    if t == "boolean":
        return True
    return "示例文本" if cjk else "sample"


class Fake:
    broken = False
    die_after: int | None = None
    reload_until = 0.0
    n = 0


def fake_reply(body: dict) -> tuple[str, list[tuple[str, dict]]]:
    msgs = body.get("messages", [])
    last = msgs[-1] if msgs else {}
    text = json.dumps(msgs, ensure_ascii=False)
    m = re.search(r"codename is ([A-Z]+-\d+)", text)
    if last.get("role") == "tool" or (last.get("role") == "user" and isinstance(last.get("content"), list)
                                      and any(b.get("type") == "tool_result" for b in last["content"])):
        return "The tool answered; here is the summary.", []
    if body.get("strata_mcp"):
        w = re.search(r"soak-\d+-\d+", str(last.get("content")))
        return f"The echo tool returned {w.group(0) if w else '?'}.", []
    if m and "What is the project codename" in str(last.get("content")):
        return m.group(1), []
    tools = body.get("tools") or []
    if not tools:
        return "A fake answer.", []
    u = str(last.get("content", ""))
    cjk = bool(CJK.search(u))
    tc = body.get("tool_choice")
    first = tools[0]
    if isinstance(tc, dict) and tc.get("function"):
        first = next(t for t in tools if t["function"]["name"] == tc["function"]["name"])
    f = first.get("function", first)
    schema = f.get("parameters") or f.get("input_schema") or {}
    n = 2 if any(w in u for w in (" and ", "和", "分别", ", ")) else 1
    return "", [(f["name"], fill(schema, cjk))] * n


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _json(self, code, obj):
        b = json.dumps(obj, ensure_ascii=False).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def dead(self):
        return Fake.die_after is not None and Fake.n >= Fake.die_after

    def do_GET(self):
        if self.path == "/health":
            if self.dead():
                return self._json(500, {"status": "down"})
            return self._json(200, {"status": "reloading" if time.time() < Fake.reload_until else "ok"})
        if self.path == "/api/context":
            return self._json(200, {"reload": {"state": "running" if time.time() < Fake.reload_until else "done"}})
        self._json(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        if self.path == "/api/context":
            Fake.reload_until = time.time() + 0.4
            return self._json(200, {"state": "waiting", "to": body.get("max_context")})
        if self.dead():
            return self._json(502, {"error": {"message": "engine gone"}})
        if time.time() < Fake.reload_until:
            return self._json(503, {"error": {"message": "reloading"}})
        Fake.n += 1
        content, calls = fake_reply(body)
        args = [json.dumps(a, ensure_ascii=False) for _, a in calls]
        if Fake.broken:
            args = ['{"city": "Lis' for _ in args]
        if self.path == "/v1/messages":
            blocks = ([{"type": "text", "text": content}] if content else []) + [
                {"type": "tool_use", "id": f"tu{i}", "name": nm, "input": a} for i, (nm, a) in enumerate(calls)]
            return self._json(200, {"content": blocks, "stop_reason": "tool_use" if calls else "end_turn"})
        tcs = [{"id": f"call{i}", "type": "function", "function": {"name": nm, "arguments": args[i]}}
               for i, (nm, _) in enumerate(calls)]
        finish = "tool_calls" if calls else "stop"
        if not body.get("stream"):
            return self._json(200, {"choices": [{"message": {"role": "assistant", "content": content or None,
                                                             "tool_calls": tcs or None}, "finish_reason": finish}],
                                    "usage": {"prompt_tokens": 10, "completion_tokens": 5}})
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()

        def ev(d):
            self.wfile.write(b"data: " + json.dumps(d, ensure_ascii=False).encode() + b"\n\n")
        ev({"choices": [{"index": 0, "delta": {"role": "assistant"}}]})
        for piece in (content[: len(content) // 2], content[len(content) // 2:]):
            if piece:
                ev({"choices": [{"index": 0, "delta": {"content": piece}}]})
        for i, tc in enumerate(tcs):
            a = tc["function"]["arguments"]
            ev({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "id": tc["id"], "type": "function",
                                                                    "function": {"name": tc["function"]["name"],
                                                                                 "arguments": a[: len(a) // 2]}}]}}]})
            ev({"choices": [{"index": 0, "delta": {"tool_calls": [{"index": i, "function": {"arguments": a[len(a) // 2:]}}]}}]})
        ev({"choices": [{"index": 0, "delta": {}, "finish_reason": finish}]})
        self.wfile.write(b"data: [DONE]\n\n")


def serve():
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv, "http://127.0.0.1:%d" % srv.server_address[1]


def quiet(fn, *a):
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return fn(*a)


class Corpus(unittest.TestCase):
    def test_committed_equals_generator(self):
        self.assertEqual(quiet(corpus.main, ["--check"]), 0)

    def test_groups_and_names(self):
        cases = sr.load_cases(sr.TOOLCALLS)
        self.assertEqual(len(cases), 50)
        groups = {}
        for c in cases:
            groups[c["group"]] = groups.get(c["group"], 0) + 1
            names = {t["function"]["name"] for t in c["request"]["tools"]}
            for want in c["expect"]["calls"]:
                self.assertIn(want["name"], names, c["id"])
            if c["group"] == "error_reply":
                self.assertEqual(c["request"]["messages"][-1]["role"], "tool")
            if c["group"] == "parallel":
                self.assertGreaterEqual(c["expect"]["min_calls"], 2)
        self.assertEqual(groups, {"single": 14, "parallel": 10, "nested": 10, "chinese": 8, "error_reply": 8})
        self.assertTrue(any(c["api"] == "anthropic" for c in cases))


class CheckCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.c = {c["id"]: c for c in sr.load_cases(sr.TOOLCALLS)}

    def chk(self, cid, calls, content=""):
        return sr.check_case(self.c[cid], [(n, a if isinstance(a, str) else json.dumps(a, ensure_ascii=False))
                                           for n, a in calls], content)

    def test_valid_and_match(self):
        r = self.chk("single_01", [("get_weather", {"city": "lisbon"})])
        self.assertTrue(r["parse"] and r["match"], r)
        r = self.chk("single_01", [("get_weather", {"city": "Porto"})])
        self.assertTrue(r["parse"])
        self.assertFalse(r["match"])

    def test_failures(self):
        self.assertIn("not a declared tool", self.chk("single_01", [("get_wether", {"city": "x"})])["errors"][0])
        self.assertIn("not JSON", self.chk("single_01", [("get_weather", '{"city": ')])["errors"][0])
        self.assertIn("required", self.chk("single_01", [("get_weather", {"unit": "celsius"})])["errors"][0])
        self.assertIn("not in", self.chk("single_01", [("get_weather", {"city": "x", "unit": "kelvin"})])["errors"][0])
        bad = {"customer": {"name": "J", "address": {"street": "s", "city": 5, "country": "UK"}},
               "items": [{"sku": "A", "quantity": "two"}]}
        errs = self.chk("nested_01", [("create_order", bad)])["errors"]
        self.assertTrue(any("address.city" in e for e in errs) and any("quantity" in e for e in errs), errs)
        self.assertIn("at least 2", self.chk("parallel_01", [("get_weather", {"city": "Paris"})])["errors"][0])
        self.assertIn("Chinese", self.chk("chinese_01", [("get_weather", {"city": "Beijing"})])["errors"][0])
        self.assertTrue(self.chk("chinese_01", [("get_weather", {"city": "北京"})])["parse"])
        err = self.c["error_01"]
        rep = err["expect"]["must_not_repeat"]
        self.assertIn("repeated", self.chk("error_01", [(rep["name"], rep["args"])])["errors"][0])
        self.assertTrue(self.chk("error_01", [], "Lisbn is not a city I know; did you mean Lisbon?")["parse"])
        self.assertFalse(self.chk("error_01", [], "")["parse"])


class Runs(unittest.TestCase):
    def setUp(self):
        Fake.broken, Fake.die_after, Fake.reload_until, Fake.n = False, None, 0.0, 0
        self.srv, self.url = serve()
        self.tmp = tempfile.TemporaryDirectory()
        self.d = pathlib.Path(self.tmp.name)

    def tearDown(self):
        self.srv.shutdown()
        self.tmp.cleanup()

    def test_toolcalls_pass(self):
        rep = self.d / "tc.json"
        self.assertEqual(quiet(sr.main, ["toolcalls", "--url", self.url, "--report", str(rep)]), 0)
        s = json.loads(rep.read_text())
        self.assertEqual(s["parse_pct"], 100.0, s["failures"])
        self.assertEqual(s["n"], 50 + 48)                     # both modes; the 2 Anthropic cases are not streamed
        self.assertTrue(any(r["stream"] for r in s["results"]))

    def test_toolcalls_broken(self):
        Fake.broken = True
        rep = self.d / "tc.json"
        self.assertEqual(quiet(sr.main, ["toolcalls", "--url", self.url, "--stream", "on", "--report", str(rep)]), 1)
        s = json.loads(rep.read_text())
        self.assertLess(s["parse_pct"], 100.0)
        self.assertTrue(any("not JSON" in e for f in s["failures"] for e in f["errors"]))

    def soak(self, *extra):
        rep, md = self.d / "soak.json", self.d / "soak.md"
        rc = quiet(sr.main, ["soak", "--url", self.url, "--duration-s", "2", "--requests", "16", "--reload-contexts",
                             "1000,2000", "--health-s", "0.1", "--min-mem-gib", "0", "--doc-chars", "2000",
                             "--doc-share", "0.1", "--mcp-share", "0.1", "--report", str(rep), "--md", str(md), *extra])
        return rc, json.loads(rep.read_text()), md.read_text()

    def test_soak_pass_with_reloads(self):
        rc, r, md = self.soak()
        self.assertEqual(rc, 0, json.dumps(r["criteria"]))
        self.assertEqual(r["verdict"], "PASS")
        self.assertEqual(len(r["reloads"]), 2)
        self.assertTrue(all(x["ready_s"] is not None for x in r["reloads"]))
        self.assertEqual([m["kept"] for m in r["conversation"]], [True, True, True])
        self.assertEqual(r["crashes"], 0)
        self.assertEqual(set(r["by_kind"]), {"chat", "code", "tool", "doc", "mcp"})
        self.assertTrue(all(x["mcp_echo"] for x in r["results"] if x["kind"] == "mcp"))
        self.assertIn("**PASS**", md)

    def test_soak_counts_a_crash(self):
        Fake.die_after = 6
        rc, r, _ = self.soak("--ready-limit-s", "2")
        self.assertEqual(rc, 1)
        self.assertGreaterEqual(r["crashes"], 1)
        crit = {c["id"]: c["pass"] for c in r["criteria"]}
        self.assertFalse(crit["crashes"])
        self.assertFalse(crit["requests_ok"])


if __name__ == "__main__":
    unittest.main(verbosity=2)

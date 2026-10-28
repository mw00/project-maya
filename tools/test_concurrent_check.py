"""tools/test_concurrent_check.py - tools/ds4/concurrent_check.py against stand-in servers on this machine (no model,
no GPU): a server that answers each conversation with its own code passes; one that hands a conversation another's
code is a leak (exit 1); one that forgets passes unless --strict (exit 3); two turns and streamed answers are read as
the plain ones; a server that is not there is an error (exit 2).

    python -m unittest tools.test_concurrent_check -v
"""
from __future__ import annotations

import contextlib
import io
import json
import re
import sys
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools" / "ds4")]
import concurrent_check as CC  # noqa: E402

CODE = re.compile(r"code of this conversation is ([A-Z]{2}-\d{4})")


class Stand(ThreadingHTTPServer):
    """An OpenAI chat API stand-in: `mode` good (the conversation's own code), leak (another conversation's, once
    `clients` codes were seen), forget (no code)."""
    daemon_threads = True

    def __init__(self, mode: str, clients: int):
        super().__init__(("127.0.0.1", 0), Handler)
        self.mode, self.clients = mode, clients
        self.seen: list = []
        self.cv = threading.Condition()
        self.requests = 0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def _send(self, code: int, body: bytes, ctype: str = "application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        self._send(200, json.dumps({"data": [{"id": "stand-in"}]}).encode())

    def do_POST(self):
        s: Stand = self.server
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        users = [m["content"] for m in req["messages"] if m["role"] == "user"]
        own = next((m.group(1) for u in users for m in [CODE.search(u)] if m), None)
        with s.cv:
            s.requests += 1
            if own and own not in s.seen:
                s.seen.append(own)
            s.cv.notify_all()
            if s.mode == "leak":
                s.cv.wait_for(lambda: len(s.seen) >= s.clients, timeout=10)
            others = [c for c in s.seen if c != own]
        if "Answer with the code only" not in users[-1]:
            text = "OK"
        elif s.mode == "good":
            text = f"The code is {own}."
        elif s.mode == "leak":
            text = others[0] if others else own
        else:
            text = "I do not remember."
        if req.get("stream"):
            chunks = [text[:3], text[3:]]
            body = "".join(f"data: {json.dumps({'choices': [{'delta': {'content': c}}]})}\n\n" for c in chunks)
            self._send(200, (body + "data: [DONE]\n\n").encode(), "text/event-stream")
        else:
            self._send(200, json.dumps({"choices": [{"message": {"role": "assistant", "content": text}}]}).encode())


def run(mode: str, *args, clients: int = 3) -> tuple:
    srv = Stand(mode, clients)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            rc = CC.main(["--url", f"http://127.0.0.1:{srv.server_address[1]}/v1", "--clients", str(clients),
                          "--seed", "5", "--shared-words", "30", "--own-words", "10", "--timeout", "20", *args])
    finally:
        srv.shutdown()
        srv.server_close()
    return rc, out.getvalue(), srv


class ConcurrentCheck(unittest.TestCase):
    def test_codes(self):
        import random
        codes = CC.make_codes(50, random.Random(1))
        self.assertEqual(len(set(codes)), 50)
        self.assertTrue(all(re.fullmatch(r"[A-Z]{2}-\d{4}", c) for c in codes))
        self.assertEqual(CC.norm("kq 4821!"), "KQ4821")
        self.assertNotRegex(CC.filler(200, 3), r"[A-Z0-9]{2}-?\d")

    def test_judge(self):
        rows = CC.judge(["AB-1234", "CD-5678"], [{"answers": ["ab 1234"]}, {"answers": ["AB-1234 or CD-5678"]}])
        self.assertEqual(rows, [(True, []), (True, ["AB-1234"])])

    def test_good(self):
        rc, out, srv = run("good")
        self.assertEqual(rc, 0, out)
        self.assertIn("3 of 3 ok", out)
        self.assertEqual(srv.requests, 3)

    def test_leak(self):
        rc, out, _ = run("leak")
        self.assertEqual(rc, 1, out)
        self.assertIn("LEAK", out)

    def test_forget(self):
        rc, out, _ = run("forget")
        self.assertEqual(rc, 0, out)
        self.assertIn("missed its code", out)
        rc, out, _ = run("forget", "--strict")
        self.assertEqual(rc, 3, out)

    def test_turns_rounds_stream(self):
        rc, out, srv = run("good", "--turns", "2", "--rounds", "2", "--stream", clients=4)
        self.assertEqual(rc, 0, out)
        self.assertIn("8 of 8 ok", out)
        self.assertEqual(srv.requests, 16)   # (two turns of four conversations, twice)

    def test_no_server(self):
        srv = Stand("good", 2)
        port = srv.server_address[1]
        srv.server_close()
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = CC.main(["--url", f"http://127.0.0.1:{port}/v1", "--clients", "2", "--timeout", "5"])
        self.assertEqual(rc, 2)


if __name__ == "__main__":
    unittest.main()

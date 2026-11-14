"""POST /slots/export and /slots/import (#655): a kept DeepSeek conversation out of one server as a bundle and into
another - the bundle's framing, the engine's SLOTOUT / SLOTIN lines, the refusals and the temporary files; no GPU, no
engine (a mock engine that keeps conversations as byte strings).

    python -m unittest serve.test_slot_handoff -v
"""
from __future__ import annotations

import json
import os
import shutil
import struct
import sys
import tempfile
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import SLOT_MAGIC, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CHAT = [{"role": "system", "content": "You read code."},
        {"role": "user", "content": "Here is the file.\n" + "int f(int x) { return x + 1; }\n" * 200}]


class SlotEngine(MockEngine):
    """The engine's two control lines as maya-ds4 answers them: SLOTOUT writes the kept conversation sharing the most
    of the ids (none: the latest) to <file.bin> and its .meta; SLOTIN keeps a pair whose .meta is this model's."""

    IDENTITY = b"maya-ds4 test-pack\n"

    def __init__(self, tokenizer, kept=None):
        super().__init__(tokenizer, "Fine.", max_context=1 << 20)
        self.kept = list(kept or [])        # (tokens, state bytes), oldest first
        self.lines = []

    def command(self, line, expect, timeout=60.0):
        self.lines.append(line)
        head, _, rest = line.partition(" ")
        if head == "SLOTOUT":
            path, _, ids = rest.partition(" ")
            want = [int(t) for t in ids.split(",")] if ids else []

            def shared(toks):
                n = 0
                while n < len(toks) and n < len(want) and toks[n] == want[n]:
                    n += 1
                return n
            pick = [k for k in self.kept if not want or shared(k[0])]
            if not pick:
                raise ValueError("no kept conversation starts like these tokens")
            toks, state = max(pick, key=lambda k: shared(k[0])) if want else pick[-1]
            Path(path).write_bytes(state)
            Path(path[:-4] + ".meta").write_bytes(self.IDENTITY + json.dumps(toks).encode())
            return f"SLOTSAVED {len(toks)} {len(state)}"
        if head == "SLOTIN":
            meta = Path(rest[:-4] + ".meta").read_bytes()
            if not meta.startswith(self.IDENTITY):
                raise ValueError("the conversation is not this model's (another model, pack or Maya version) "
                                 "or its .meta is damaged")
            toks = json.loads(meta[len(self.IDENTITY):])
            state = Path(rest).read_bytes()
            self.kept.append((toks, state))
            return f"SLOTADDED {len(toks)} {len(state)}"
        raise ValueError(f"unsupported command '{head}'")


class Handoff(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="slot-handoff-test-")
        cls.saved_tempdir, tempfile.tempdir = tempfile.tempdir, cls.tmp   # (the server's temporary pairs land here)
        cls.tok = ByteTokenizer()
        template = ChatTemplate(ROOT / "tools/ds4/chat_template.jinja")
        cls.a = Service(SlotEngine(cls.tok), cls.tok, template)
        cls.b = Service(SlotEngine(cls.tok), cls.tok, template)
        cls.glm = Service(SlotEngine(cls.tok), cls.tok, ChatTemplate(ROOT / "serve/chat_template.jinja"))
        cls.servers = {k: serve(svc, port=0) for k, svc in (("a", cls.a), ("b", cls.b), ("glm", cls.glm))}

    @classmethod
    def tearDownClass(cls):
        for httpd in cls.servers.values():
            httpd.shutdown()
            httpd.server_close()
        tempfile.tempdir = cls.saved_tempdir
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        ids = self.a.render_ids(*self.a.chat_prompt({"messages": CHAT}))
        self.state = os.urandom(300_000)
        self.a.engine.kept = [(ids[:40] + [7, 7, 7], b"older"), (ids + [9, 9], self.state), ([1, 2, 3], b"latest")]
        self.a.engine.lines.clear()
        self.b.engine.kept.clear()
        self.b.engine.lines.clear()

    def post(self, server, path, body: bytes, headers=None):
        url = f"http://127.0.0.1:{self.servers[server].server_address[1]}{path}"
        req = urllib.request.Request(url, data=body, method="POST", headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return r.status, dict(r.headers), r.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers), e.read()

    def export(self, body: dict, server="a"):
        return self.post(server, "/slots/export", json.dumps(body).encode(), {"Content-Type": "application/json"})

    def leftovers(self):
        """The server's temporary folders still there: an import's goes before its answer; an export's after its last
        byte is sent, so that one is given a moment."""
        for _ in range(50):
            left = [p for p in os.listdir(self.tmp) if p.startswith("maya-slot-")]
            if not any(p.startswith("maya-slot-out-") for p in left):
                return left
            time.sleep(0.02)
        return left

    def test_the_conversation_moves_byte_for_byte(self):
        status, headers, bundle = self.export({"messages": CHAT})
        self.assertEqual(status, 200)
        line = self.a.engine.lines[-1].split(" ")
        self.assertEqual(line[0], "SLOTOUT")
        self.assertEqual(line[2], ",".join(str(t) for t in self.a.render_ids(*self.a.chat_prompt({"messages": CHAT}))))
        # SLOT_MAGIC, the .meta's length (u64 LE), the .meta, the .bin
        self.assertEqual(bundle[:8], SLOT_MAGIC)
        (ml,) = struct.unpack("<Q", bundle[8:16])
        self.assertEqual(bundle[16 + ml:], self.state)
        self.assertEqual(int(headers["Content-Length"]), len(bundle))
        self.assertEqual(int(headers["X-Maya-Slot-Tokens"]), len(self.a.engine.kept[1][0]))
        status, _, answer = self.post("b", "/slots/import", bundle)
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(answer), {"tokens": len(self.a.engine.kept[1][0]), "bytes": len(self.state)})
        self.assertEqual(self.b.engine.kept, [self.a.engine.kept[1]])
        self.assertEqual(self.leftovers(), [])

    def test_no_messages_is_the_latest(self):
        status, _, bundle = self.export({})
        self.assertEqual(status, 200)
        self.assertEqual(self.a.engine.lines[-1].count(" "), 1)   # SLOTOUT <file> - no ids
        self.assertTrue(bundle.endswith(b"latest"))
        status, _, bundle = self.post("a", "/slots/export", b"")
        self.assertEqual(status, 200)
        self.assertTrue(bundle.endswith(b"latest"))

    def test_the_engines_refusal_is_a_400(self):
        self.a.engine.kept = [([1, 2, 3], b"x")]
        status, _, body = self.export({"messages": CHAT})
        self.assertEqual(status, 400)
        self.assertIn("no kept conversation starts like these tokens", json.loads(body)["error"]["message"])
        self.assertEqual(self.leftovers(), [])
        # another model's bundle: the importing engine refuses it
        status, _, bundle = self.export({})
        self.assertEqual(status, 200)
        foreign = bundle[:16] + b"X" + bundle[17:]
        status, _, body = self.post("b", "/slots/import", foreign)
        self.assertEqual(status, 400)
        self.assertIn("not this model's", json.loads(body)["error"]["message"])
        self.assertEqual(self.b.engine.kept, [])
        self.assertEqual(self.leftovers(), [])

    def test_a_body_that_is_not_a_bundle(self):
        _, _, bundle = self.export({"messages": CHAT})
        (ml,) = struct.unpack("<Q", bundle[8:16])
        cases = {
            "not a bundle": b"NOTASLOT" + bundle[8:],
            "a short body": bundle[:12],
            "a .meta past the body": bundle[:8] + struct.pack("<Q", len(bundle)) + bundle[16:],
            "a .meta over 64 MiB": bundle[:8] + struct.pack("<Q", 65 << 20) + bundle[16:],
        }
        for what, body in cases.items():
            with self.subTest(what):
                status, _, answer = self.post("b", "/slots/import", body)
                self.assertEqual(status, 400, answer)
                self.assertEqual(json.loads(answer)["error"]["type"], "invalid_request_error")
        self.assertEqual(self.b.engine.lines, [])                 # (nothing reached the engine)
        self.assertEqual(self.leftovers(), [])
        self.assertGreater(ml, 0)

    def test_only_deepseek_and_one_conversation_at_a_time(self):
        status, _, body = self.export({}, server="glm")
        self.assertEqual(status, 400)
        self.assertIn("only the DeepSeek engine", json.loads(body)["error"]["message"])
        self.a.engine.multi = 3
        try:
            status, _, body = self.export({})
        finally:
            self.a.engine.multi = 0
        self.assertEqual(status, 400)
        self.assertIn("--parallel", json.loads(body)["error"]["message"])
        self.assertEqual(self.glm.engine.lines, [])


if __name__ == "__main__":
    unittest.main()

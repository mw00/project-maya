"""Native grammar adapter contracts; no model or GPU required."""
import concurrent.futures
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import os
import threading
import unittest
from unittest.mock import Mock, patch
import urllib.error
import urllib.request

from serve import structured
from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, MockEngine, Service, StrataEngine, grammar_file, serve

ROOT = Path(__file__).resolve().parents[1]
GRAMMAR = 'root ::= "A"'


class GrammarAdapter(unittest.TestCase):
    def service(self, engine=None):
        tok = ByteTokenizer()
        return Service(engine or MockEngine(tok, "A", max_context=4096), tok,
                       ChatTemplate(ROOT / "serve/chat_template.jinja"))

    def test_aliases_and_ambiguities(self):
        self.assertEqual(structured.normalize_request({"grammar": GRAMMAR, "structured_outputs": {
            "grammar": GRAMMAR, "disable_any_whitespace": True}}), (GRAMMAR, None))
        for request in ({"grammar": GRAMMAR, "structured_outputs": {"grammar": 'root ::= "B"'}},
                        {"grammar": ""}, {"grammar": "a\0b"},
                        {"structured_outputs": {"grammar": GRAMMAR, "disable_any_whitespace": 1}}):
            with self.subTest(request=request), self.assertRaises(ValueError):
                structured.normalize_request(request)

    def test_unsupported_engine_never_silently_prompts(self):
        with self.assertRaisesRegex(ValueError, "native GBNF"):
            self.service().check_request({"grammar": GRAMMAR})

    def test_compile_errors_propagate_before_generation(self):
        svc = self.service()
        svc.engine.info = {"grammar": 1, "grammar_replay": 1}
        svc.engine.validate_grammar = Mock(side_effect=ValueError("grammar: undefined rule missing"))
        with self.assertRaisesRegex(ValueError, "undefined rule missing"):
            svc.check_request({"grammar": 'root ::= missing'})

    def test_internal_paths_cannot_be_injected(self):
        svc = self.service()
        req = {"_grammar_path": "/private", "_grammar_text": GRAMMAR, "_grammar_after": 0, "grammar_hex": "deadbeef"}
        svc.check_request(req)
        self.assertFalse(any(k.startswith("_grammar_") for k in req))
        self.assertNotIn("grammar_hex", req)

    def test_native_checker_uses_pack_and_reports_precise_error(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.spawn = ("/build/maya-ds4", ["--pack", "packs/deepseek"], "/model", None, None)
        engine.grammar_checker = None
        def checker(command, **kwargs):
            self.assertEqual(command[:3], ["/build/maya-ds4-grammar-check", "--pack", "packs/deepseek"])
            self.assertEqual(Path(command[-1]).read_text(), GRAMMAR)
            self.assertEqual(kwargs["cwd"], "/model")
            return subprocess.CompletedProcess(command, 1, "", "undefined rule missing")
        with patch("serve.server.subprocess.run", side_effect=checker), self.assertRaisesRegex(ValueError, "undefined rule"):
            engine.validate_grammar(GRAMMAR)

    @unittest.skipUnless(os.name == "posix", "script helper uses a shebang")
    def test_checker_protocol_with_binary_vocab_logs_and_both_spawn_shapes(self):
        script = r'''import sys
from pathlib import Path
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
assert args["--pack"] == "packs/deepseek"
# llama vocabulary diagnostics may contain raw BPE bytes, even for valid grammars.
sys.stderr.buffer.write(b"vocab: raw byte \xef\xff\x80\n")
sys.stderr.buffer.flush()
if Path(args["--grammar"]).read_text() == 'root ::= "A"':
    print("GRAMMAR_OK")
else:
    print("grammar: undefined rule missing", file=sys.stderr)
    sys.exit(1)
'''
        with tempfile.TemporaryDirectory() as directory:
            checker = Path(directory) / "maya-ds4-grammar-check"
            checker.write_text(f"#!{sys.executable}\n" + script)
            checker.chmod(0o755)
            engine = StrataEngine.__new__(StrataEngine)
            engine.grammar_checker = None
            for extra in ((), ("native",)):
                with self.subTest(spawn_fields=5 + len(extra)):
                    engine.spawn = (str(Path(directory) / "maya-ds4"), ["--pack", "packs/deepseek"],
                                    directory, None, None) + extra
                    engine.validate_grammar(GRAMMAR)
                    with self.assertRaisesRegex(ValueError, "undefined rule missing"):
                        engine.validate_grammar("root ::= missing")

    def test_protocol_keys_greedy_and_sampled_and_replay(self):
        for temperature in (0, 0.5, None):
            smp = {"temperature": temperature, "_grammar_path": "/tmp/grammar", "_grammar_after": 8,
                   "_grammar_prefix": [8, 2, 3]}
            keys = StrataEngine.sampling_keys(smp)
            self.assertIn(" grammar=/tmp/grammar grammar_after=8 grammar_prefix=8,2,3", keys)
        smp = {"_grammar_text": GRAMMAR, "_think_budget": 30, "_think_end": 8}
        resumed = Service._resumed_sampling(smp, [8, 2])
        self.assertEqual(resumed["_grammar_prefix"], [8, 2])
        self.assertNotIn("_think_budget", resumed)
        again = Service._resumed_sampling(smp, [8, 2, 3])
        self.assertEqual(again["_grammar_prefix"], [8, 2, 3])
        self.assertNotIn("_grammar_prefix", smp)

    def test_file_lives_until_native_generator_closes(self):
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = Mock()
        engine.proc.poll.return_value = None
        paths = []
        def native(ids, max_new, sampling, cancel, embeddings):
            path = Path(sampling["_grammar_path"])
            paths.append(path)
            sampling["_grammar_lease"]["read"] = True
            try:
                yield 65
            finally:
                self.assertTrue(path.exists())  # native's cancellation drain owns it still
        engine._generate_native = native
        gen = engine.generate([], 4, {"_grammar_text": GRAMMAR}, threading.Event())
        self.assertEqual(next(gen), 65)
        self.assertTrue(paths[0].exists())
        gen.close()
        self.assertFalse(paths[0].exists())

    def test_unacknowledged_cancel_retains_file_until_process_exit(self):
        ended, removed = threading.Event(), threading.Event()
        proc = Mock()
        proc.poll.return_value = None
        proc.wait.side_effect = lambda: ended.wait(2)
        with grammar_file(GRAMMAR, {"read": False}, proc) as path:
            self.assertTrue(Path(path).exists())
        self.assertTrue(Path(path).exists())
        ended.set()
        # Wait for real background cleanup, no arbitrary sleep.
        proc.wait.assert_called_once()
        for _ in range(100):
            if not Path(path).exists():
                break
            removed.wait(.01)
        self.assertFalse(Path(path).exists())

    def test_legacy_path_fallback_rejects_whitespace_and_nul(self):
        for path in ("/tmp/a b", "/tmp/a\tb", "/tmp/a\0b"):
            with self.subTest(path=path), self.assertRaisesRegex(ValueError, "grammar.*path"):
                StrataEngine.grammar_keys({"_grammar_path": path})
        self.assertEqual(StrataEngine.grammar_keys({"_grammar_path": "/tmp/旧.gbnf"}), " grammar=/tmp/旧.gbnf")

    @unittest.skipUnless(os.name == "posix", "script engine uses a shebang")
    def test_native_hex_paths_four_routes_replay_and_cancel_ownership(self):
        script = r'''import sys
from pathlib import Path
print("INFO grammar=1 grammar_replay=1 grammar_hex=1", flush=True)
print("READY 4096 stop multi=4", flush=True)
pending = {}
for line in sys.stdin:
    if line.startswith("QUIT"):
        break
    if line.startswith("STOP "):
        rid = line.split()[1]
        path = pending.pop(rid, None)
        if path is not None:
            assert path.exists(), "grammar removed before cancellation DONE"
            print(f"DONE {rid} 1 3 1.0 1.0 cancel 0 0 0", flush=True)
        continue
    if not line.startswith("GEN "):
        continue
    fields = dict(x.split("=", 1) for x in line.split() if "=" in x)
    assert "grammar" not in fields
    path = Path(bytes.fromhex(fields["grammar_hex"]).decode("utf-8"))
    assert "spaced 目录" in str(path)
    assert fields["grammar_prefix"] == "8,2" and fields["grammar_after"] == "8"
    rid = fields["rid"]
    text = path.read_text()
    char = text.split('"')[1]
    print(f"T {rid} {ord(char)}", flush=True)
    if line.split()[1] == "9":
        pending[rid] = path
    else:
        print(f"DONE {rid} 1 3 1.0 1.0 stop 0 0 0", flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            tempdir = Path(directory) / "spaced 目录"
            tempdir.mkdir()
            exe = Path(directory) / "engine"
            exe.write_text(f"#!{sys.executable}\n" + script)
            exe.chmod(0o755)
            engine = StrataEngine(str(exe), [])
            original_mkstemp = tempfile.mkstemp
            def mkstemp(**kwargs):
                return original_mkstemp(dir=tempdir, **kwargs)
            def sampling(char):
                return {"_grammar_text": f'root ::= "{char}"', "_grammar_prefix": [8, 2], "_grammar_after": 8}
            try:
                with patch("serve.server.tempfile.mkstemp", side_effect=mkstemp):
                    def call(char):
                        return list(engine.generate([1, 2, 3], 8, sampling(char), threading.Event()))
                    with concurrent.futures.ThreadPoolExecutor(4) as pool:
                        self.assertEqual(list(pool.map(call, "ABCD")), [[65], [66], [67], [68]])
                    gen = engine.generate([1, 2, 3], 9, sampling("E"), threading.Event())
                    self.assertEqual(next(gen), 69)
                    self.assertEqual(len(list(tempdir.glob("*.gbnf"))), 1)
                    gen.close()
                self.assertEqual(engine._routes, {})
                self.assertEqual(list(tempdir.glob("*.gbnf")), [])
                self.assertIsNone(engine.proc.poll())  # the fake native's ownership assertions did not fail
            finally:
                engine.close()

    def test_midstate_starvation_is_error_not_successful_partial(self):
        import io
        import queue
        from types import SimpleNamespace
        engine = StrataEngine.__new__(StrataEngine)
        engine.proc = SimpleNamespace(stdin=io.StringIO())
        engine.lines = queue.Queue()
        engine.can_stop = False
        engine.lines.put("T 65")
        engine.lines.put("DONE 1 3 1.0 1.0 grammar 0 0 0")
        gen = engine.generate([1, 2, 3], 8, {}, threading.Event())
        self.assertEqual(next(gen), 65)
        with self.assertRaisesRegex(ValueError, "no valid token"):
            next(gen)

    def test_http_four_requests_stream_nonstream_and_bad_request(self):
        svc = self.service()
        svc.engine.info = {"grammar": 1, "grammar_replay": 1}
        svc.engine.validate_grammar = lambda text: None if text == GRAMMAR else (_ for _ in ()).throw(ValueError("grammar: bad rule"))
        httpd = serve(svc, port=0)
        base = f"http://127.0.0.1:{httpd.server_address[1]}/v1/chat/completions"
        def call(stream):
            req = {"messages": [{"role": "user", "content": "A"}], "grammar": GRAMMAR,
                   "structured_outputs": {"grammar": GRAMMAR, "disable_any_whitespace": True},
                   "stream": stream, "max_tokens": 8, "chat_template_kwargs": {"enable_thinking": False}}
            with urllib.request.urlopen(urllib.request.Request(base, json.dumps(req).encode(),
                                        {"Content-Type": "application/json"}), timeout=10) as response:
                return response.read().decode()
        try:
            with concurrent.futures.ThreadPoolExecutor(4) as pool:
                replies = list(pool.map(call, (False, True, False, True)))
            self.assertEqual(json.loads(replies[0])["choices"][0]["message"]["content"], "A")
            self.assertIn("data: [DONE]", replies[1])
            bad = {"messages": [{"role": "user", "content": "A"}], "grammar": "bad", "stream": True}
            with self.assertRaises(urllib.error.HTTPError) as caught:
                urllib.request.urlopen(urllib.request.Request(base, json.dumps(bad).encode(),
                    {"Content-Type": "application/json"}))
            self.assertEqual(caught.exception.code, 400)
            self.assertIn("bad rule", caught.exception.read().decode())
        finally:
            httpd.shutdown()
            httpd.server_close()

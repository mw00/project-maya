"""tools/strata/test_parity_run.py - tools/strata/parity_run.sh end to end against a fake engine and a fake
llm_logits_dump (CPU only): the refs / engine / compare / gate stages for B7.4a and M2, the long-context selection
overlap (B7.6), the seam bisect after a failed gate, --dry-run and the usage errors."""
from __future__ import annotations

import http.server
import json
import os
import pathlib
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRIPT = HERE / "parity_run.sh"
V = 512

COMMON = textwrap.dedent('''
    import hashlib, os, sys
    import numpy as np
    V = %d
    def read_tokens(p):
        return [int(x) for x in open(p).read().replace(",", " ").split()]
    def base(tokens, i):
        h = int(hashlib.sha256(("%%d:%%s" %% (i, tokens[max(0, i - 3):i + 1])).encode()).hexdigest()[:8], 16)
        x = np.random.default_rng(h).standard_normal(V) * 3.0
        x[tokens[i + 1] if i + 1 < len(tokens) else 0] += 6.0      # the text is the greedy continuation
        return x
    def noise(tag, i, scale):
        h = int(hashlib.sha256(("%%s:%%d" %% (tag, i)).encode()).hexdigest()[:8], 16)
        return np.random.default_rng(h).standard_normal(V) * scale
    def lsm(x):
        m = x.max(); return x - (m + np.log(np.exp(x - m).sum()))
    def sel(p, w=int(os.environ.get("FAKE_TOPK", "2048"))):
        out = np.full(w, -1, dtype=np.int32)
        cells = np.arange(max(0, p - w + 1), p + 1, dtype=np.int32)
        out[:cells.size] = cells
        return out
''' % V)

FAKE_STRATA = COMMON + textwrap.dedent('''
    a = sys.argv[1:]
    tokens = read_tokens(a[a.index("--tokens-file") + 1])
    c = int(os.environ["MAYA_GLM_SCORE"])
    slow = os.environ.get("MAYA_GLM_SLOW") == "1"
    strict = os.environ.get("MAYA_GLM_IDX_LN_EPS") == "0"
    bad = float(os.environ.get("FAKE_BAD", "0"))
    dump = os.environ.get("MAYA_GLM_SCORE_DUMP")
    cb = os.environ.get("GLM_CB_DIR")
    append = os.environ.get("GLM_CB_APPEND") == "1"
    only = [s for s in os.environ.get("GLM_CB_ONLY", "").split(",") if s]
    f = open(dump, "wb") if dump else None
    n = 0
    for i in range(len(tokens) - 1):
        if cb and append:
            for s in only:
                with open(os.path.join(cb, s + ".f32"), "ab") as g:
                    g.write(sel(i).tobytes())
        if i + 1 <= c:
            continue
        x = base(tokens, i) + noise("eng", i, 0.02) + (noise("slow", i, 0.005) if slow else 0)
        x = x + (0 if strict else noise("eps", i, 0.01)) + (noise("bad", i, bad) if bad else 0)
        if f: f.write(lsm(x).astype("<f4").tobytes())
        n += 1
    if cb and not append:
        p = len(tokens) - 2
        for il in range(3):
            v = noise("seam", p, 1.0)[:64].astype("<f4") * (il + 1)
            if bad and il == 1: v = v * 1.5
            v.tofile(os.path.join(cb, "l_out-%d.f32" % il))
    print("SCORE %d tokens: nll 1.0 ppl 2.7 top1 99.00%%" % n)
''')

FAKE_LLAMA = COMMON + textwrap.dedent('''
    a = sys.argv[1:]
    ngl = int(a[a.index("--ngl") + 1]) if "--ngl" in a else 0
    pos = [x for i, x in enumerate(a) if not x.startswith("--") and (i == 0 or not a[i - 1].startswith("--"))]
    model, out, tf = pos[0], pos[1], pos[2]
    tokens = read_tokens(tf[1:])
    cb = os.environ.get("LLM_CB_DIR")
    append = os.environ.get("LLM_CB_APPEND") == "1"
    only = [s for s in os.environ.get("LLM_CB_ONLY", "").split(",") if s]
    with open(out, "wb") as f:
        for i in range(len(tokens)):
            x = base(tokens, i) + (noise("cuda" + ("pin" if "pin" in sys.argv[0] else ""), i, 0.01) if ngl else 0)
            f.write(x.astype("<f4").tobytes())
            if cb and append:
                for s in only:
                    with open(os.path.join(cb, s + ".i32"), "ab") as g:
                        g.write(sel(i).tobytes())
    if cb and not append:
        for il in range(3):
            np.concatenate([noise("seam", p, 1.0)[:64].astype("<f4") * (il + 1) for p in range(len(tokens))]).tofile(
                os.path.join(cb, "l_out-%d.f32" % il))
''')


def toks(n, seed):
    return [(i * 37 + seed) % V for i in range(n)]


class ParityRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if shutil.which("bash") is None:
            raise unittest.SkipTest("no bash")
        cls.tmp = tempfile.TemporaryDirectory()
        d = cls.d = pathlib.Path(cls.tmp.name)
        for name, body in (("strata", FAKE_STRATA), ("llm_logits_dump", FAKE_LLAMA), ("llm_logits_dump_pin", FAKE_LLAMA)):
            p = d / name
            p.write_text("#!%s\n%s" % (sys.executable, body))
            p.chmod(0o755)
        corpus = cls.corpus = d / "corpus"
        corpus.mkdir()
        lines = []
        for name, n, ctx, kind, seed in (("chat", 120, 60, "short", 11), ("code", 100, 40, "short", 5),
                                         ("doc", 110, 90, "long", 3)):
            (corpus / f"{name}.tokens").write_text(",".join(map(str, toks(n, seed))))
            (corpus / f"{name}.ctx").write_text(str(ctx))
            lines.append(f"{name} {kind}")
        (corpus / "cases.txt").write_text("# case kind\n" + "\n".join(lines) + "\n")

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def run_sh(self, *args, env=None, ok=(0,)):
        e = dict(os.environ, PYTHON=sys.executable)
        e.update(env or {})
        r = subprocess.run(["bash", str(SCRIPT), "--strata", str(self.d / "strata"), "--llama-dump",
                            str(self.d / "llm_logits_dump"), "--corpus", str(self.corpus), "--n-vocab", str(V),
                            "--model", "m.gguf", "--pack", "p.pack", "--sf-positions", "50", *args],
                           capture_output=True, text=True, env=e, cwd=str(ROOT))
        if r.returncode not in ok:
            raise AssertionError(f"exit {r.returncode}\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        return r

    def test_b74a(self):
        out = self.d / "b74a"
        self.run_sh("--gate", "B7.4a", "--out", str(out), "--stages", "refs,engine,compare,gate")
        g = json.loads((out / "gate.json").read_text())
        self.assertEqual(g["verdict"], "PASS", g)
        self.assertEqual(sorted(g["a_vs_r"]["prompts"]), ["chat.a_vs_r", "code.a_vs_r"])   # short cases only
        self.assertTrue((out / "ref" / "chat.topk").exists())
        rep = (out / "report.md").read_text()
        self.assertIn("B7.4a gate: **PASS**", rep)
        self.assertIn("| chat.a_vs_r |", rep)
        shutil.rmtree(out)

    def test_m2_incomplete_then_bisect_on_failure(self):
        out = self.d / "m2"
        # a top-k of 64 keeps the "long" case and its appended selections small (the real run: 2048, 8K tokens)
        r = self.run_sh("--gate", "M2", "--out", str(out), "--top-k", "64", env={"FAKE_TOPK": "64"}, ok=(1,))
        g = json.loads((out / "gate.json").read_text())
        crit = {c["id"]: c["pass"] for c in g["criteria"]}
        self.assertEqual(g["verdict"], "INCOMPLETE", r.stdout)
        for cid in "abcdfg":
            self.assertTrue(crit[cid], (cid, g["criteria"]))
        self.assertEqual(sorted(g["missing"]), ["e", "h", "i"])
        ov = json.loads((out / "cmp" / "doc.overlap.json").read_text())
        self.assertTrue(ov["pass"])
        self.assertEqual(ov["layers"]["26"]["first"], 65)
        self.assertTrue(json.loads((out / "cmp" / "doc.overlap_fp16.json").read_text())["pass"])
        # an incomplete gate also bisects the worst position; the fake engine and llama agree on every seam
        b = json.loads((out / "bisect" / "bisect.json").read_text())
        self.assertIsNone(b["first_divergence"])
        self.assertEqual(b["compared"], 3)
        shutil.rmtree(out)

    def test_m2_fail_bisects(self):
        out = self.d / "m2bad"
        self.run_sh("--gate", "M2", "--out", str(out), "--cases", "chat,code", "--stages",
                    "refs,engine,compare,gate,bisect", env={"FAKE_BAD": "0.6"}, ok=(1,))
        g = json.loads((out / "gate.json").read_text())
        self.assertEqual(g["verdict"], "FAIL")
        self.assertIn("a", g["failed"])
        b = json.loads((out / "bisect" / "bisect.json").read_text())
        self.assertEqual(b["first_divergence"], "l_out-1")
        shutil.rmtree(out)

    def test_m2_alt_uses_the_pin(self):
        out = self.d / "m2alt"
        self.run_sh("--gate", "M2-alt", "--out", str(out), "--cases", "chat", "--llama-pin-dump",
                    str(self.d / "llm_logits_dump_pin"), "--stages", "refs,engine,compare,gate", ok=(1,))
        floor = json.loads((out / "cmp" / "chat.floor.json").read_text())
        self.assertTrue(floor["ref"].endswith("chat.cuda.bin") and floor["test"].endswith("chat.pin.bin"))
        self.assertGreater(floor["kl_mean"], 0.0)
        shutil.rmtree(out)

    def test_dry_run_and_usage(self):
        r = self.run_sh("--gate", "M2", "--out", str(self.d / "dry"), "--dry-run")
        self.assertIn("MAYA_GLM_IDX_LN_EPS=0", r.stdout)
        self.assertIn("MAYA_GLM_SLOW=1", r.stdout)
        self.assertIn("GLM_CB_APPEND=1", r.stdout)
        self.assertIn("parity_compare.py gate", r.stdout)
        self.assertFalse((self.d / "dry" / "gate.json").exists())
        r = self.run_sh("--gate", "M3", ok=(2,))
        self.assertIn("--gate", r.stderr)
        r = self.run_sh("--gate", "M2-alt", ok=(2,))
        self.assertIn("--llama-pin-dump", r.stderr)


class FakeLlamaServer(http.server.BaseHTTPRequestHandler):
    """/apply-template, /tokenize, /detokenize, /completion of llama-server, character-level."""
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/apply-template":
            out = {"prompt": "".join(f"<{m['role']}>{m['content']}" for m in body["messages"])
                   + ("<tools>%d" % len(body["tools"]) if body.get("tools") else "") + "<assistant>"}
        elif self.path == "/tokenize":
            out = {"tokens": ([1] if body.get("add_special") else []) + [ord(ch) % 1000 + 2 for ch in body["content"]]}
        elif self.path == "/detokenize":
            out = {"content": "".join(chr(t - 2) if 2 <= t < 1002 else "?" for t in body["tokens"])}
        elif self.path == "/completion":
            assert body["temperature"] == 0.0 and body["return_tokens"]
            out = {"tokens": [(len(body["prompt"]) + i) % 997 for i in range(body["n_predict"])], "stop_type": "limit"}
        else:
            self.send_response(404)
            self.end_headers()
            return
        b = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)


class Corpus(unittest.TestCase):
    def test_build_from_spec(self):
        sys.path.insert(0, str(HERE))
        import parity_corpus as pcorp
        srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), FakeLlamaServer)
        threading.Thread(target=srv.serve_forever, daemon=True).start()
        url = "http://127.0.0.1:%d" % srv.server_address[1]
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d)
            spec = ROOT / "data" / "glm-dsa" / "parity" / "specs.json"
            import contextlib, io
            with contextlib.redirect_stdout(io.StringIO()):
                rc = pcorp.main(["--url", url, "--spec", str(spec), "--out", str(out)])
            srv.shutdown()
            self.assertEqual(rc, 0)
            cases = [ln.split() for ln in (out / "cases.txt").read_text().splitlines() if not ln.startswith("#")]
            self.assertEqual([c[1] for c in cases], ["short"] * 4 + ["long"] * 2)
            for name, kind in cases:
                ids = [int(x) for x in (out / f"{name}.tokens").read_text().split(",")]
                ctx = int((out / f"{name}.ctx").read_text())
                self.assertEqual(len(ids) - ctx, 256)
                self.assertLessEqual(ctx, 2048 if kind == "short" else 8192 + 64)
                if kind == "long":
                    self.assertGreater(ctx, 7900)
            meta = json.loads((out / "corpus.json").read_text())
            self.assertEqual(len(meta["cases"]), 6)


if __name__ == "__main__":
    unittest.main(verbosity=2)

"""tools/strata/test_bench_http.py - bench_http.py against a fake llama-server / glm-dsa server, llama_ot.py regexes, and
the frozen bench corpus (B0.10).

    python3 -m unittest discover -s tools/strata -p 'test_bench_http.py' -v
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
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import bench_http as bh  # noqa: E402
import llama_ot as lo  # noqa: E402
import make_bench_corpus as mbc  # noqa: E402


class Fake(http.server.BaseHTTPRequestHandler):
    """Tokens are whitespace-separated words (id = running index).  Speeds are scripted per request."""
    log: list = []
    speeds: list = []
    openai = False

    def log_message(self, *a):
        pass

    def _send(self, obj, code=200):
        b = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if self.path == "/tokenize":
            return self._send({"tokens": list(range(len(body["content"].split())))})
        idx = len(Fake.log)
        pps, dps = Fake.speeds[idx % len(Fake.speeds)]
        if self.path == "/completion" and not Fake.openai:
            Fake.log.append(body)
            n = len(body["prompt"])
            gen = body["n_predict"]
            return self._send({"content": "x", "timings": {"prompt_n": n, "prompt_ms": 1000.0 * n / pps,
                                                           "prompt_per_second": pps, "predicted_n": gen,
                                                           "predicted_ms": 1000.0 * gen / dps,
                                                           "predicted_per_second": dps}})
        if self.path == "/v1/chat/completions" and Fake.openai:
            Fake.log.append(body)
            n = len(body["messages"][0]["content"]) // 5        # this fake: 5 characters per token
            return self._send({"choices": [{"message": {"content": "x"}}], "usage": {"prompt_tokens": n},
                               "timings": {"prompt_n": n, "prompt_per_second": pps, "predicted_n": body["max_tokens"],
                                           "predicted_per_second": dps}})
        return self._send({"error": "no"}, 404)


class FakeServer:
    def __enter__(self):
        self.httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Fake)
        self.t = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.t.start()
        return "http://127.0.0.1:%d" % self.httpd.server_address[1]

    def __exit__(self, *a):
        self.httpd.shutdown()
        self.httpd.server_close()


def corpus(d: pathlib.Path) -> pathlib.Path:
    for name, words in (("prompt_4k.txt", 7000), ("prompt_8k.txt", 14000), ("prompt_32k.txt", 40000)):
        (d / name).write_text(" ".join("w%d" % i for i in range(words)))
    return d


class BenchHttpTest(unittest.TestCase):
    def setUp(self):
        Fake.log, Fake.openai = [], False
        self.td = tempfile.TemporaryDirectory()
        self.dir = corpus(pathlib.Path(self.td.name))

    def tearDown(self):
        self.td.cleanup()

    def run_main(self, args):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            rc = bh.main(args)
        return rc, out.getvalue()

    def test_llama_exact_prompts_cold_and_median(self):
        # per request (prefill, decode) speeds: warm-up first, then 3 runs per measurement
        Fake.speeds = [(100.0, 9.0), (100.0, 8.0), (102.0, 8.2), (98.0, 8.1)]
        with FakeServer() as url:
            jpath = pathlib.Path(self.td.name) / "r.json"
            rc, md = self.run_main(["--url", url, "--corpus", str(self.dir), "--ctx", "4096,32768", "--gen", "256",
                                    "--runs", "3", "--prompt-lens", "8192", "--json", str(jpath), "--quiet"])
        self.assertEqual(rc, 0, md)
        self.assertEqual(len(Fake.log), 3 * 4)
        lens = [len(b["prompt"]) for b in Fake.log]
        self.assertEqual(lens, [3840] * 4 + [32512] * 4 + [8192] * 4)
        self.assertTrue(all(b["cache_prompt"] is False and b["ignore_eos"] and b["temperature"] == 0 for b in Fake.log))
        self.assertEqual([b["n_predict"] for b in Fake.log], [256] * 8 + [1] * 4)
        # each run starts at a different corpus offset
        starts = [b["prompt"][0] for b in Fake.log[:4]]
        self.assertEqual(starts, [0, 64, 128, 192])
        res = json.loads(jpath.read_text())["results"]
        self.assertEqual(res[0]["values"], [8.0, 8.2, 8.1])             # warm-up (9.0) excluded
        self.assertAlmostEqual(res[0]["median"], 8.1)
        self.assertAlmostEqual(res[0]["spread_pct"], (8.2 - 8.0) / 8.1 * 100)
        self.assertEqual(res[2]["values"], [100.0, 102.0, 98.0])
        self.assertIn("| decode tok/s @ctx | 4096 | 3840 | 8.00, 8.20, 8.10 | 8.10 |", md)
        self.assertIn("| prefill tok/s | 8192 | 8192 |", md)

    def test_spread_flag_exit_1(self):
        Fake.speeds = [(100.0, 8.0), (100.0, 8.0), (100.0, 9.0), (100.0, 8.0)]
        with FakeServer() as url:
            rc, md = self.run_main(["--url", url, "--corpus", str(self.dir), "--ctx", "4096", "--prompt-lens", "",
                                    "--quiet"])
        self.assertEqual(rc, 1)
        self.assertIn("SPREAD > 5%", md)

    def test_openai_calibrates_char_cut(self):
        Fake.openai = True
        Fake.speeds = [(50.0, 6.0)]
        with FakeServer() as url:
            rc, md = self.run_main(["--url", url, "--api", "openai", "--corpus", str(self.dir), "--ctx", "4096",
                                    "--prompt-lens", "", "--quiet"])
        self.assertEqual(rc, 0, md)
        n = [len(b["messages"][0]["content"]) // 5 for b in Fake.log]
        self.assertEqual(len(n), 4)
        self.assertTrue(all(abs(x - 3840) <= 2 for x in n[1:]), n)    # cut recalibrated after the warm-up
        self.assertFalse(Fake.log[0]["chat_template_kwargs"]["enable_thinking"])

    def test_openai_tune_rides_every_request(self):
        # --tune (speed plan): the per-request engine knobs as each request's strata_tune, named in the report
        Fake.openai = True
        Fake.speeds = [(50.0, 6.0)]
        tune = {"cpu_threads": 36, "pcie_frac": 0.21, "promote_min": 6}
        with FakeServer() as url:
            jpath = pathlib.Path(self.td.name) / "t.json"
            rc, md = self.run_main(["--url", url, "--api", "openai", "--corpus", str(self.dir), "--ctx", "4096",
                                    "--prompt-lens", "", "--quiet", "--tune", json.dumps(tune), "--json", str(jpath)])
        self.assertEqual(rc, 0, md)
        self.assertEqual(len(Fake.log), 4)
        self.assertTrue(all(b["strata_tune"] == tune for b in Fake.log), Fake.log[0])
        self.assertTrue(all(b["temperature"] == 0 and b["max_tokens"] == 256 for b in Fake.log))
        self.assertIn('strata_tune `{"cpu_threads": 36, "pcie_frac": 0.21, "promote_min": 6}`', md)
        self.assertEqual(json.loads(jpath.read_text())["meta"]["tune"], tune)

    def test_without_tune_the_body_is_unchanged(self):
        Fake.openai = True
        Fake.speeds = [(50.0, 6.0)]
        with FakeServer() as url:
            rc, md = self.run_main(["--url", url, "--api", "openai", "--corpus", str(self.dir), "--ctx", "4096",
                                    "--prompt-lens", "", "--quiet"])
        self.assertEqual(rc, 0, md)
        self.assertFalse([b for b in Fake.log if "strata_tune" in b])
        self.assertNotIn("strata_tune", md)

    def test_bad_tune_is_an_error_before_any_request(self):
        Fake.speeds = [(1.0, 1.0)]
        with FakeServer() as url:
            for args in (["--api", "openai", "--tune", "{cpu_threads: 36}"], ["--api", "openai", "--tune", "[1, 2]"],
                         ["--api", "llama", "--tune", '{"spec": 1}']):
                rc, _ = self.run_main(["--url", url, "--corpus", str(self.dir), "--ctx", "4096", "--prompt-lens", "",
                                       "--quiet", *args])
                self.assertEqual(rc, 2, args)
        self.assertEqual(Fake.log, [])

    def test_corpus_too_short_is_an_error(self):
        (self.dir / "prompt_32k.txt").write_text("a b c")
        Fake.speeds = [(1.0, 1.0)]
        with FakeServer() as url:
            rc, _ = self.run_main(["--url", url, "--corpus", str(self.dir), "--ctx", "32768", "--prompt-lens", "",
                                   "--quiet"])
        self.assertEqual(rc, 2)


class LlamaOtTest(unittest.TestCase):
    def test_range_regex_brute_force(self):
        for lo_ in range(0, 130):
            for hi in range(lo_, 130, 7):
                rx = re.compile("^(%s)$" % lo.range_regex(lo_, hi))
                got = [i for i in range(0, 1100) if rx.match(str(i))]
                self.assertEqual(got, list(range(lo_, hi + 1)), (lo_, hi, rx.pattern))

    def test_plan_example(self):
        self.assertEqual(lo.range_regex(3, 10), "[3-9]|10")
        self.assertEqual(lo.layers_regex([3, 4, 5, 9]), "[3-5]|9")

    def test_count_mode(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(lo.main(["--split", "26,54", "--count", "10,10,9", "--format", "json"]), 0)
        d = json.loads(out.getvalue())
        self.assertEqual(d["layers"], [list(range(3, 13)), list(range(26, 36)), list(range(54, 63))])
        self.assertEqual(d["ot"][-1], "exps=CPU")
        self.assertEqual(d["ts"], "26,28,25")
        for g, layers in enumerate(d["layers"]):
            pat, dev = d["ot"][g].rsplit("=", 1)
            self.assertEqual(dev, "CUDA%d" % g)
            rx = re.compile(pat)
            for i in range(79):
                for kind in ("gate", "up", "down"):
                    name = "blk.%d.ffn_%s_exps.weight" % (i, kind)
                    self.assertEqual(bool(rx.search(name)), i in layers, name)
                self.assertFalse(rx.search("blk.%d.ffn_gate_shexp.weight" % i))

    def test_fit_from_audit_tsv(self):
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "a.tsv"
            rows = ["name\ttype\tdims\tshard\tbytes\tpacked_bytes\trole\tgpu"]
            for i in range(3, 79):
                for k in ("gate", "up", "down"):
                    rows.append("blk.%d.ffn_%s_exps.weight\tIQ2_XXS\t6144,2048,256\t1\t%d\t%d\trouted\t0" % (i, k, 10**9, 10**9))
            p.write_text("\n".join(rows) + "\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
                lo.main(["--audit-tsv", str(p), "--vram-gb", "22,22,10", "--format", "json"])
            d = json.loads(out.getvalue())
        self.assertEqual(d["layers"], [list(range(3, 10)), list(range(26, 33)), list(range(54, 57))])

    def test_explicit_layers_args(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(io.StringIO()):
            lo.main(["--layers", "3-12,-,54-56+60"])
        s = out.getvalue().strip()
        self.assertEqual(s, r"-ot 'blk\.([3-9]|1[0-2])\.ffn_(up|down|gate)_exps=CUDA0' "
                            r"-ot 'blk\.(5[4-6]|60)\.ffn_(up|down|gate)_exps=CUDA2' -ot exps=CPU -ts 26,28,25")


class CorpusTest(unittest.TestCase):
    def test_committed_corpus_is_the_generated_one(self):
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            rc = mbc.main(["--check"])
        self.assertEqual(rc, 0, out.getvalue())

    def test_corpus_sizes(self):
        for name, (seed, n) in mbc.FILES.items():
            b = (mbc.OUT / name).read_bytes()
            self.assertGreaterEqual(len(b), n)
            self.assertTrue(b.decode("utf-8").isascii())
            nominal = int(re.match(r"prompt_(\d+)k", name).group(1)) * 1024
            self.assertGreaterEqual(len(b) / 4.0, nominal * 1.5)


if __name__ == "__main__":
    unittest.main()

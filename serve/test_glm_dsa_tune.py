"""serve/test_glm_dsa_tune.py - the per-request engine knobs (GLM-5.3, glm-dsa) through the server: the
`strata_tune` whitelist of serve/glm_dsa_tune.py on the GEN line, the request body reaching the engine over both
APIs, and the GEN line the engine reads (serve/glm_dsa_fake_engine.py in place of `strata --serve`).

    python -m unittest serve.test_glm_dsa_tune -v
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import glm_dsa_tune as T  # noqa: E402
from serve.frontend import ChatTemplate  # noqa: E402
from serve.server import StrataEngine, ByteTokenizer, MockEngine, Service, serve  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FAKE = ROOT / "serve/glm_dsa_fake_engine.py"
KNOBS = {"promote_min": 6, "prefetch_n": 1, "prefetch_at": "fetch", "prefetch_rank": 2, "prefetch_blocks": 4,
         "mtp_keep": 4, "cpu_split": 2, "vram_evict": "lru", "ngram": True, "lane_row4": 1, "graphs": False,
         "pcie_frac_head": 0.35}
WANT = ["promote_min=6", "prefetch_n=1", "prefetch_rank=2", "prefetch_blocks=4", "mtp_keep=4", "cpu_split=2",
        "prefetch_at=fetch", "vram_evict=lru", "ngram=1", "lane_row4=1", "graphs=0", "pcie_frac_head=0.35"]


def keys(**sampling) -> list[str]:
    return StrataEngine.sampling_keys(sampling).split()


def knob_keys(ks: list[str]) -> list[str]:
    return [k for k in ks if k.split("=")[0] in T.KEYS]


class Whitelist(unittest.TestCase):
    def test_every_knob_reaches_the_gen_line(self):
        self.assertEqual(knob_keys(keys(temperature=0, strata_tune=KNOBS)), WANT)
        self.assertEqual(knob_keys(keys(temperature=0.7, strata_tune=KNOBS)), WANT)   # the sampled path too
        self.assertEqual(T.engine_keys(KNOBS).split(), WANT)

    def test_no_knob_no_key(self):
        # a request without them sends exactly what it sent before (the engine then runs as it started)
        self.assertEqual(knob_keys(keys(temperature=0)), [])
        self.assertEqual(knob_keys(keys(temperature=0, strata_tune={"pcie_frac": 0.2, "cpu_threads": 27})), [])
        self.assertEqual(StrataEngine.sampling_keys({"temperature": 0, "strata_tune": {}}),
                         StrataEngine.sampling_keys({"temperature": 0}))
        for bad in (None, "promote_min=6", ["promote_min", 6], 6):
            self.assertEqual(T.engine_keys(bad), "", bad)

    def test_ranges_and_types(self):
        ok = {"promote_min": (0, 65535), "prefetch_n": (0, 2), "prefetch_rank": (0, 64), "prefetch_blocks": (0, 1024),
              "mtp_keep": (0, 8), "cpu_split": (0, 64)}
        for k, (lo, hi) in ok.items():
            self.assertEqual(knob_keys(keys(temperature=0, strata_tune={k: lo})), [f"{k}={lo}"])
            self.assertEqual(knob_keys(keys(temperature=0, strata_tune={k: hi})), [f"{k}={hi}"])
            for bad in (lo - 1, hi + 1, True, 1.5, str(lo), None):
                self.assertEqual(knob_keys(keys(temperature=0, strata_tune={k: bad})), [], (k, bad))
        for k, good, bads in (("prefetch_at", ("route", "fetch", "cpu"), ("later", "FETCH", 1, True)),
                              ("vram_evict", ("lfu", "lru"), ("LRU", "fifo", 0, False))):
            for v in good:
                self.assertEqual(knob_keys(keys(temperature=0, strata_tune={k: v})), [f"{k}={v}"])
            for v in bads:
                self.assertEqual(knob_keys(keys(temperature=0, strata_tune={k: v})), [], (k, v))
        for k in ("ngram", "lane_row4", "graphs"):
            for v, want in ((True, 1), (False, 0), (1, 1), (0, 0)):
                self.assertEqual(knob_keys(keys(temperature=0, strata_tune={k: v})), [f"{k}={want}"], (k, v))
            for v in (2, -1, "on", 0.5, None):
                self.assertEqual(knob_keys(keys(temperature=0, strata_tune={k: v})), [], (k, v))
        self.assertEqual(knob_keys(keys(temperature=0, strata_tune={"pcie_frac_head": 0})), ["pcie_frac_head=0.0"])
        self.assertEqual(knob_keys(keys(temperature=0, strata_tune={"pcie_frac_head": 1})), ["pcie_frac_head=1.0"])
        for v in (-0.1, 1.5, True, "0.3"):
            self.assertEqual(knob_keys(keys(temperature=0, strata_tune={"pcie_frac_head": v})), [], v)

    def test_with_mayas_keys_and_spec_off(self):
        # spec=1 (Maya's key) takes the request to the token-by-token decode; the knobs ride beside it
        k = keys(temperature=0, strata_tune={"spec": 1, "pcie_frac": 0.21, "cpu_threads": 36, "promote_min": 6})
        for want in ("spec=1", "pcie_frac=0.21", "cpu_threads=36", "promote_min=6"):
            self.assertIn(want, k)
        self.assertEqual(k[-1:], ["promote_min=6"])        # the knobs follow Maya's keys


class _Capture(MockEngine):
    def generate(self, ids, max_new, sampling, cancel, embeddings=None):
        self.sampling = dict(sampling or {})
        yield from super().generate(ids, max_new, sampling, cancel, embeddings)


class RequestBody(unittest.TestCase):
    """The body's strata_tune reaches the engine's generate() unchanged over the OpenAI and the Anthropic API."""

    @classmethod
    def setUpClass(cls):
        tok = ByteTokenizer()
        cls.engine = _Capture(tok, "</think>\n\nhello", max_context=4096)
        cls.httpd = serve(Service(cls.engine, tok, ChatTemplate(ROOT / "serve/chat_template.jinja")), port=0)
        cls.base = f"http://127.0.0.1:{cls.httpd.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def post(self, path, body):
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status

    def test_openai_and_anthropic(self):
        msgs = [{"role": "user", "content": "hi"}]
        for path, body in (("/v1/chat/completions", {"model": "m", "messages": msgs, "temperature": 0}),
                           ("/v1/messages", {"model": "m", "messages": msgs, "max_tokens": 16, "temperature": 0})):
            self.engine.sampling = None
            self.assertEqual(self.post(path, {**body, "strata_tune": KNOBS}), 200)
            self.assertEqual(self.engine.sampling.get("strata_tune"), KNOBS, path)
            self.assertEqual(knob_keys(StrataEngine.sampling_keys(self.engine.sampling).split()), WANT, path)


@unittest.skipIf(os.name == "nt", "the fake engine is started as an executable script")
class EngineLine(unittest.TestCase):
    """The GEN line the engine reads: the knobs on the request that names them, none on the next one."""

    def test_gen_line(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "gen.txt"
            engine = StrataEngine(str(FAKE), ["--glm-pack", d, "--max-context", "4096"], log=str(Path(d) / "e.log"),
                                  env=dict(os.environ, FAKE_ENGINE_GEN_LOG=str(log)))
            try:
                for sampling in ({"temperature": 0, "strata_tune": dict(KNOBS, spec=1)}, {"temperature": 0}):
                    toks = [t for t in engine.generate([1, 2, 3], 4, sampling, threading.Event()) if t is not None]
                    self.assertEqual(len(toks), 4)
            finally:
                engine.proc.kill()
                engine.proc.wait(timeout=10)
                if hasattr(engine.log, "close"):
                    engine.log.close()
            first, second = log.read_text(encoding="utf-8").splitlines()
        f = first.split()
        self.assertEqual(f[:2], ["GEN", "4"])
        self.assertEqual(f[-1], "1,2,3")
        self.assertIn("spec=1", f)
        self.assertEqual(knob_keys(f[2:-1]), WANT)
        self.assertEqual(knob_keys(second.split()[2:-1]), [])   # left out: the engine restores the start values


if __name__ == "__main__":
    unittest.main()

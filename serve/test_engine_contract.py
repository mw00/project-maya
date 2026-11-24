"""serve/test_engine_contract.py - the serve contract of a real engine (GLM-5.3, glm-dsa): what the server,
the setup's screen, the calibration and --report read from `strata --serve --glm-pack <pack>`.  Opt-in, it needs the
GPU(s) and a pack:

    MAYA_TEST_ENGINE=build/strata MAYA_TEST_PACK=/data/models/GLM-5.3-GGUF/UD-IQ2_M/pack \\
        python -m unittest serve.test_engine_contract -v

Optional: MAYA_TEST_CONTEXT (default 8192), MAYA_TEST_ARGS (more engine arguments, e.g. "--layer-split 26,54"),
MAYA_TEST_READY_S (how long the start may take, default 900), MAYA_TEST_LOG (keep the engine log there),
MAYA_TEST_TEXT_ONLY=0 (a model with images: VLEND is not expected to fail), MAYA_TEST_REQUIRE_A29=1 (INFO must
carry arch / vision / spec), MAYA_TEST_EXPERTS (the INFO experts total to expect),
MAYA_TEST_SPLIT=1 (several GPUs: the log names the layer split).  The engine's own environment (MAYA_GLM_*,
CUDA_VISIBLE_DEVICES) is passed through.

Without MAYA_TEST_ENGINE and MAYA_TEST_PACK every test is skipped.  serve/glm_dsa_fake_engine.py stands in for
the engine to check this file itself (no GPU): that proves the checks, not the engine.

Checked: READY (context, stop, pause), the INFO keys, the start-log phrases (_context_cost's VRAM line, the serve
banner, REPORT_LINES, setup_bridge's warming lines, --bench's "glm stat: decode"), GEN with sampling and thinking keys
answering PP / T / STAT and a 10-field DONE, the conversation cache (DONE reused), STOP, WRAP and a thinking budget
forcing </think>, VLEND on a text-only model, and QUIT ending the engine with exit code 0.
"""
from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tools"))

ENGINE = os.environ.get("MAYA_TEST_ENGINE", "")
PACK = os.environ.get("MAYA_TEST_PACK", "")
CONTEXT = int(os.environ.get("MAYA_TEST_CONTEXT", "8192"))
READY_S = float(os.environ.get("MAYA_TEST_READY_S", "900"))
TEXT_ONLY = os.environ.get("MAYA_TEST_TEXT_ONLY", "1") == "1"
THINK_END = 154842                              # </think> (GLM-5.x's vocabulary)
GLM_EOS = (154820, 154827, 154829)              # <|endoftext|>, <|user|>, <|observation|>
# INFO keys of the GLM fast path (src/program/generate.cpp, the serve start): the Monitor, the calibration and the TUI
INFO_KEYS = ("context", "kv", "expert_slots", "expert_cache_mib", "engine_kind", "experts", "vram_slots", "vram_gb",
             "ram_slots", "ram_gb", "mtp", "vision_lend", "cpu_threads", "pcie_share")
A29_KEYS = ("engine", "arch", "vision", "spec")
STAT_KEYS = ("tok_s", "ms_tok", "vram_hit", "ram_fetch", "disk", "promo", "vram_used", "vram_slots", "ram_used",
             "ram_slots", "ram_gb", "vram_gb")
# serve/server.py StrataEngine._context_cost reads this line for the dashboard's context sizes
VRAM_LINE = re.compile(r"\b((?:CUDA|HIP)\d+) VRAM before the expert pool:.*?state/KV \((\d+) ctx\) ([\d.]+)")


def kv(line: str) -> dict:
    return dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)


@unittest.skipUnless(ENGINE and PACK, "set MAYA_TEST_ENGINE and MAYA_TEST_PACK to run the engine contract test "
                                      "(needs the GPU(s) and a pack)")
class EngineContract(unittest.TestCase):
    """One engine for the whole class (GLM-5.3's start takes minutes); the tests run in name order."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log_path = Path(os.environ.get("MAYA_TEST_LOG") or Path(cls.tmp.name) / "engine.log")
        cls.log = open(cls.log_path, "a", encoding="utf-8")
        cls.log_start = cls.log_path.stat().st_size
        args = [ENGINE, "--serve", "--glm-pack", PACK, "--max-context", str(CONTEXT), "--prefill", "auto",
                *shlex.split(os.environ.get("MAYA_TEST_ARGS", ""))]
        cls.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=cls.log, text=True,
                                    encoding="utf-8", bufsize=1)
        cls.lines: queue.Queue = queue.Queue()

        def pump():
            for line in cls.proc.stdout:
                cls.lines.put(line.rstrip("\n"))
            cls.lines.put(None)
        threading.Thread(target=pump, daemon=True).start()
        cls.info, cls.ready, cls.before = {}, None, []
        end = time.time() + READY_S
        while time.time() < end:
            try:
                line = cls.lines.get(timeout=1)
            except queue.Empty:
                continue
            if line is None:
                break
            cls.before.append(line)
            if line.startswith("INFO "):
                cls.info.update(kv(line))
            if line.startswith("READY"):
                cls.ready = line
                break
        cls.prompt = cls.encode("[gMASK]<sop><|system|>Reasoning Effort: Low<|user|>Count from one to forty in "
                                "words, separated by commas.<|assistant|><think>")

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            cls.proc.kill()
            cls.proc.wait(timeout=30)
        cls.log.close()
        cls.tmp.cleanup()

    @classmethod
    def encode(cls, text: str) -> list[int]:
        """The pack's own tokenizer when it has one, else GLM-5.x's special ids around a few ordinary ones."""
        tdir = Path(PACK) / "tokenizer"
        if (tdir / "vocab.json").exists():
            import strata_tokenizer as ST
            return ST.Tokenizer.from_dir(tdir).encode(text, parse_special=True)
        return [154822, 154824, 154827, 1000, 2000, 3000, 154828, 154841]

    # ---------------------------------------------------------------- helpers
    def send(self, line: str):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read(self, timeout=600.0):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.fail(f"the engine said nothing for {timeout:.0f} s")
        if line is None:
            self.fail(f"the engine ended (exit code {self.proc.poll()}); log: {self.log_path}")
        return line

    def gen(self, max_new: int, keys: str = "", on_token=None) -> dict:
        """GEN -> {"T": [ids], "PP": [...], "STAT": [...], "DONE": [fields]}; on_token(n) after the n-th T."""
        self.send(f"GEN {max_new}{(' ' + keys) if keys else ''} {','.join(map(str, self.prompt))}")
        got = {"T": [], "PP": [], "STAT": [], "NOTE": []}
        while True:
            line = self.read()
            head = line.split(" ", 1)[0]
            if head == "T":
                got["T"].append(int(line.split()[1]))
                if on_token:
                    on_token(len(got["T"]))
            elif head in ("PP", "STAT", "NOTE"):
                got[head].append(line)
            elif head == "DONE":
                got["DONE"] = line.split()[1:]
                return got
            elif head == "ERR":
                self.fail("ERR from the engine: " + line)

    def log_text(self) -> str:
        self.log.flush()
        with open(self.log_path, "rb") as f:
            f.seek(self.log_start)
            return f.read().decode("utf-8", "replace")

    # ---------------------------------------------------------------- start
    def test_01_ready(self):
        self.assertIsNotNone(self.ready, f"no READY within {READY_S:.0f} s; stdout {self.before[-5:]}, "
                                         f"log {self.log_path}")
        f = self.ready.split()
        self.assertEqual(int(f[1]), CONTEXT)
        self.assertIn("stop", f[2:])
        self.assertIn("pause", f[2:])
        for x in f[2:]:
            self.assertTrue(x in ("stop", "pause") or re.fullmatch(r"multi=\d+", x), x)

    def test_02_info(self):
        missing = [k for k in INFO_KEYS if k not in self.info]
        self.assertFalse(missing, f"INFO lacks {missing}: {self.info}")
        self.assertEqual(int(self.info["context"]), CONTEXT)
        self.assertEqual(self.info["engine_kind"], "glm-fast")
        self.assertIn(self.info["kv"], ("fp16", "int8"))
        for k in ("expert_slots", "expert_cache_mib", "experts", "vram_slots", "ram_slots", "mtp", "vision_lend",
                  "cpu_threads"):
            self.assertRegex(self.info[k], r"^\d+$", k)
        for k in ("vram_gb", "ram_gb", "pcie_share"):
            float(self.info[k])
        self.assertGreater(int(self.info["experts"]), 0)
        if os.environ.get("MAYA_TEST_EXPERTS"):
            self.assertEqual(int(self.info["experts"]), int(os.environ["MAYA_TEST_EXPERTS"]))
        self.assertIn(self.info["mtp"], ("0", "1"))
        if TEXT_ONLY:
            self.assertEqual(int(self.info["vision_lend"]), 0)
        if os.environ.get("MAYA_TEST_REQUIRE_A29") == "1":
            self.assertFalse([k for k in A29_KEYS if k not in self.info], f"INFO lacks the arch / vision keys: {self.info}")
        if "arch" in self.info:
            self.assertIn(self.info["arch"], ("glm-dsa", "glm5-next", "glm5next"))
        if "vision" in self.info and TEXT_ONLY:
            self.assertEqual(self.info["vision"], "0")

    def test_03_start_log(self):
        from serve.server import REPORT_LINES
        text = self.log_text()
        vram = VRAM_LINE.findall(text)
        self.assertTrue(vram, "no 'VRAM before the expert pool ... state/KV (N ctx)' line (the dashboard's context "
                              "cost)")
        self.assertTrue(all(int(c) == CONTEXT for _, c, _ in vram), vram)
        self.assertRegex(text, r"maya generate: .*serve")          # the serve banner
        self.assertTrue(any(REPORT_LINES.search(x) for x in text.splitlines()), "nothing --report would keep")
        from setup_bridge import WARMED, WARMING
        for line in text.splitlines():
            if "warming the expert tiers" in line:
                self.assertRegex(line, WARMING)
            if " tiers warm:" in line:
                self.assertRegex(line, WARMED)
        if os.environ.get("MAYA_TEST_SPLIT") == "1":
            self.assertRegex(text, r"glm split: layers ")

    # ---------------------------------------------------------------- requests
    def test_04_gen_done(self):
        got = self.gen(48, "temperature=0.7 top_p=0.95 top_k=40 seed=7")
        done = got["DONE"]
        self.assertEqual(len(done), 10, f"DONE has {len(done)} fields: {done}")
        generated, prompt = int(done[0]), int(done[1])
        float(done[2]), float(done[3])
        self.assertIn(done[4], ("stop", "length"))
        drafts_acc, drafts_off, reused, hits, lookups = map(int, done[5:])
        self.assertEqual(prompt, len(self.prompt))
        self.assertEqual(generated, len(got["T"]))
        self.assertLessEqual(drafts_acc, drafts_off)
        self.assertLessEqual(hits, lookups)
        self.assertLessEqual(reused, prompt)
        self.assertTrue(got["PP"], "no PP line (prompt progress)")
        f = got["PP"][-1].split()
        self.assertEqual(int(f[2]), prompt)
        if done[4] == "stop":
            self.assertIn(got["T"][-1], GLM_EOS)       # <|user|> / <|observation|> / <|endoftext|> end the turn
        if generated >= 8:
            self.assertTrue(got["STAT"], "no STAT line in 8+ tokens")
            st = kv(got["STAT"][-1])
            self.assertFalse([k for k in STAT_KEYS if k not in st], st)
            for k in STAT_KEYS:
                float(st[k])
        self.assertRegex(self.log_text(), r"glm stat: decode [\d.]+ ms/tok")   # maya.py --bench reads it

    def test_05_conversation_cache(self):
        if os.environ.get("MAYA_GLM_NO_REUSE") == "1":
            self.skipTest("MAYA_GLM_NO_REUSE=1")
        done = self.gen(8)["DONE"]
        self.assertGreater(int(done[7]), 0, "the same prompt again reused nothing (DONE field 8)")

    def test_06_stop(self):
        sent = []

        def stop(n):
            if n == 3 and not sent:
                sent.append(1)
                self.send("STOP")
        got = self.gen(4096, on_token=stop)
        self.assertEqual(got["DONE"][4], "cancel")
        self.assertLess(int(got["DONE"][0]), 64)

    def test_07_thinking_budget_forces_think_end(self):
        got = self.gen(16, f"think_budget=4 think_end={THINK_END}")
        self.assertIn(THINK_END, got["T"][:6], got["T"])

    def test_08_wrap(self):
        sent = []

        def wrap(n):
            if n == 2 and not sent:
                sent.append(n)
                self.send("WRAP")
        got = self.gen(32, f"think_budget=100000 think_end={THINK_END}", on_token=wrap)
        self.assertIn(THINK_END, got["T"][2:6], got["T"])   # the next token (or one in flight) closes the thinking

    def test_09_vlend_on_a_text_only_model(self):
        if not TEXT_ONLY:
            self.skipTest("MAYA_TEST_TEXT_ONLY=0")
        if "vision" not in self.info:
            self.skipTest("INFO has no vision key yet: VLEND may still lend memory")
        self.send("VLEND")
        line = self.read(120)
        self.assertTrue(line.startswith("ERR"), line)
        self.assertNotIn("VLENT", line)

    def test_99_quit(self):
        self.send("QUIT")
        try:
            rc = self.proc.wait(timeout=float(os.environ.get("MAYA_ENGINE_QUIT_S", "50") or 50) + 10)
        except subprocess.TimeoutExpired:
            self.fail("QUIT did not end the engine in time")
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()

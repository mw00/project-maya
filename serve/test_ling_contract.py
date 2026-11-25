"""serve/test_ling_contract.py - real-engine serve contract test for maya-ling.

Opt-in (needs GPU and a Ling-3.2-air GGUF):
    MAYA_TEST_LING_ENGINE=build/maya-ling MAYA_TEST_LING_GGUF=/path/to/Ling-3.2-air.gguf \
        python -m unittest serve.test_ling_contract -v

Without MAYA_TEST_LING_ENGINE and MAYA_TEST_LING_GGUF every test is skipped.
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

ENGINE = os.environ.get("MAYA_TEST_LING_ENGINE", "")
GGUF = os.environ.get("MAYA_TEST_LING_GGUF", "")
CONTEXT = int(os.environ.get("MAYA_TEST_CONTEXT", "8192"))
READY_S = float(os.environ.get("MAYA_TEST_READY_S", "300"))
TEXT_ONLY = os.environ.get("MAYA_TEST_TEXT_ONLY", "1") == "1"
THINK_END = 156904

INFO_KEYS = ("context", "kv", "expert_slots", "expert_cache_mib", "engine_kind", "experts", "vram_slots", "vram_gb",
             "ram_slots", "ram_gb", "mtp", "vision_lend", "cpu_threads", "pcie_share",
             "engine", "arch", "vision", "spec")
STAT_KEYS = ("tok_s", "ms_tok", "vram_hit", "ram_fetch", "disk", "promo", "vram_used", "vram_slots", "ram_used",
             "ram_slots", "ram_gb", "vram_gb")
VRAM_LINE = re.compile(r"\b((?:CUDA|HIP)\d+) VRAM before the expert pool:.*?state/KV \((\d+) ctx\) ([\d.]+)")


def kv(line: str) -> dict:
    return dict(x.split("=", 1) for x in line.split()[1:] if "=" in x)


@unittest.skipUnless(ENGINE and GGUF, "set MAYA_TEST_LING_ENGINE and MAYA_TEST_LING_GGUF to run the Ling engine contract test")
class LingEngineContract(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.log_path = Path(os.environ.get("MAYA_TEST_LOG") or Path(cls.tmp.name) / "engine.log")
        cls.log = open(cls.log_path, "a", encoding="utf-8")
        cls.log_start = cls.log_path.stat().st_size
        args = [ENGINE, "--serve", "--model", GGUF, "--ctx", str(CONTEXT),
                *shlex.split(os.environ.get("MAYA_TEST_ARGS", ""))]
        cls.proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=cls.log,
                                    text=True, encoding="utf-8", bufsize=1)
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
        cls.prompt = [156893, 9906, 11, 1268, 525, 499, 30, 156895, 156893, 156903]

    @classmethod
    def tearDownClass(cls):
        if cls.proc.poll() is None:
            cls.proc.kill()
            cls.proc.wait(timeout=30)
        cls.log.close()
        cls.tmp.cleanup()

    def send(self, line: str):
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def read(self, timeout=180.0):
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            self.fail(f"engine silent for {timeout:.0f} s")
        if line is None:
            self.fail(f"engine ended (rc={self.proc.poll()}); log: {self.log_path}")
        return line

    def gen(self, max_new: int, keys: str = "", ids=None, on_token=None, allow_err=False) -> dict:
        toks = self.prompt if ids is None else ids
        self.send(f"GEN {max_new}{(' ' + keys) if keys else ''} {','.join(map(str, toks))}")
        got = {"T": [], "PP": [], "STAT": [], "ERR": []}
        while True:
            line = self.read()
            head = line.split(" ", 1)[0]
            if head == "T":
                got["T"].append(int(line.split()[1]))
                if on_token:
                    on_token(len(got["T"]))
            elif head in ("PP", "STAT", "ERR"):
                got[head].append(line)
                if head == "ERR" and not allow_err:
                    self.fail("ERR from engine: " + line)
            elif head == "DONE":
                got["DONE"] = line.split()[1:]
                return got

    def log_text(self) -> str:
        self.log.flush()
        with open(self.log_path, "rb") as f:
            f.seek(self.log_start)
            return f.read().decode("utf-8", "replace")

    def test_01_ready(self):
        self.assertIsNotNone(self.ready, f"no READY; stdout={self.before[-5:]}")
        f = self.ready.split()
        self.assertEqual(int(f[1]), CONTEXT)
        self.assertIn("stop", f[2:])
        self.assertIn("pause", f[2:])

    def test_02_info(self):
        missing = [k for k in INFO_KEYS if k not in self.info]
        self.assertFalse(missing, f"INFO lacks {missing}: {self.info}")
        self.assertEqual(int(self.info["context"]), CONTEXT)
        self.assertEqual(self.info["engine_kind"], "ling")
        self.assertEqual(self.info["arch"], "bailingmoe3")
        self.assertIn(self.info["kv"], ("f16", "q8"))
        self.assertGreater(int(self.info["experts"]), 0)

    def test_03_start_log_and_env(self):
        from maya import REPORT_LINES
        text = self.log_text()
        self.assertRegex(text, r"^ling env: ", text[:400])
        vram = VRAM_LINE.findall(text)
        self.assertTrue(vram, "missing VRAM before the expert pool line")
        self.assertTrue(all(int(c) == CONTEXT for _, c, _ in vram), vram)
        self.assertRegex(text, r"maya generate: .*serve")
        self.assertTrue(any(REPORT_LINES.search(x) for x in text.splitlines()), "nothing --report would keep")

    def test_04_gen_done_and_stat(self):
        got = self.gen(24, "temperature=0.0 top_p=1.0 top_k=1 seed=7")
        done = got["DONE"]
        self.assertEqual(len(done), 10, f"DONE has {len(done)} fields: {done}")
        self.assertEqual(int(done[1]), len(self.prompt))
        self.assertEqual(int(done[0]), len(got["T"]))
        self.assertIn(done[4], ("stop", "length"))
        self.assertTrue(got["PP"], "no PP line")
        self.assertTrue(got["STAT"], "no STAT line")
        st = kv(got["STAT"][-1])
        self.assertFalse([k for k in STAT_KEYS if k not in st], st)

    def test_05_prefix_cache_reuse(self):
        # Continuation extends the cached prompt -> reuses the full KDA+MLA snapshot
        done = self.gen(8, "temperature=0.0", ids=self.prompt + [198])["DONE"]
        self.assertGreater(int(done[7]), 0, f"expected reused > 0 on continuation, got DONE={done}")

    def test_06_stop(self):
        sent = []
        def stop(n):
            if n == 3 and not sent:
                sent.append(1)
                self.send("STOP")
        got = self.gen(512, "temperature=0.0", on_token=stop)
        self.assertEqual(got["DONE"][4], "cancel")
        self.assertLess(int(got["DONE"][0]), 32)

    def test_07_think_budget_and_wrap(self):
        got = self.gen(16, f"temperature=0.0 think_budget=4 think_end={THINK_END}")
        self.assertIn(THINK_END, got["T"][:6], got["T"])
        sent = []
        def wrap(n):
            if n == 2 and not sent:
                sent.append(1)
                self.send("WRAP")
        got2 = self.gen(24, f"temperature=0.0 think_budget=100000 think_end={THINK_END}", on_token=wrap)
        self.assertIn(THINK_END, got2["T"][2:8], got2["T"])

    def test_08_bad_token_ids_refused(self):
        for bad in (-1, 157184):
            got = self.gen(4, ids=[bad, 9906], allow_err=True)
            self.assertTrue(got["ERR"], f"expected ERR for id={bad}")
            self.assertEqual(got["DONE"][4], "cancel")

    def test_09_vlend_text_only_refused(self):
        if not TEXT_ONLY:
            self.skipTest("MAYA_TEST_TEXT_ONLY=0")
        self.send("VLEND")
        line = self.read(60)
        self.assertTrue(line.startswith("ERR"), line)
        self.assertNotIn("VLENT", line)

    def test_99_quit(self):
        self.send("QUIT")
        rc = self.proc.wait(timeout=30)
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()

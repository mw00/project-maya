"""tools/test_calibrate_mm3_prefill.py - MiniMax's prefill calibration (tools/calibrate_mm3_prefill.py, maya.py's
prefill_calibration): the engine's command (the config's arguments, the calibration's own flags replaced, the prompt's
tokens), its JSON taken, a failed run raised; the setup's record in the config, a start measuring once, --calibrate
measuring again.  No GPU, no engine.

    python -m unittest tools.test_calibrate_mm3_prefill -v
"""
from pathlib import Path
from types import SimpleNamespace
import json
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import calibrate_mm3_prefill as CPF  # noqa: E402
import maya  # noqa: E402


def mm3_cfg(**kw):
    c = {"exe": "/x/maya-mm3", "args": ["--pack", "/m/pack", "--max-context", "32768", "--chunk", "2048",
                                        "--dspark", "on", "--no-prefill-tune"],
         "cwd": "/x", "tokenizer": "/m/pack/tokenizer", "gpu": [0], "model_name": "minimax-m3.1-flash",
         "installer": {"family": "minimax", "quant": "MiniMax-M3.1-Flash-Maya-XS"}}
    c.update(kw)
    return c


# (serve/server.py's engine_args / child_env without its imports: the config's args, the environment as it is)
SERVER = SimpleNamespace(engine_args=lambda c: list(c["args"]), child_env=lambda c: {})


class Command(unittest.TestCase):
    def setUp(self):
        self.srv = patch.object(CPF, "server", return_value=SERVER)
        self.srv.start()

    def tearDown(self):
        self.srv.stop()

    def test_without_the_calibrations_own_flags(self):
        self.assertEqual(CPF.without(["--pack", "p", "--chunk", "4096", "--no-prefill-tune", "--prefill-gemm", "lt",
                                      "--dspark", "on", "--prefill-tune", "f"]), ["--pack", "p", "--dspark", "on"])

    def test_run_takes_the_engines_json(self):
        seen = {}

        def runner(cmd, env, cwd, timeout, say):
            seen["cmd"] = cmd
            out = Path(cmd[cmd.index("--calibrate-prefill") + 1])
            ids = Path(cmd[cmd.index("--ids") + 1][1:]).read_text()
            seen["ids"] = ids
            out.write_text(json.dumps({"chunk": 4096, "gemm": "hipblaslt", "tok_s": 300.0, "default_tok_s": 120.0,
                                       "tokens": 4096, "trials": []}))
            return 0, ["calibration: chunk 4096 ..."]
        with patch.object(CPF, "prompt_ids", return_value=[1, 2, 3]):
            res = CPF.run(mm3_cfg(), say=lambda *a: None, runner=runner)
        self.assertEqual((res["chunk"], res["gemm"], res["tok_s"]), (4096, "hipblaslt", 300.0))
        cmd = seen["cmd"]
        self.assertEqual(cmd[0], "/x/maya-mm3")
        self.assertNotIn("--chunk", cmd)                  # (the calibration's own: its largest chunk)
        self.assertNotIn("--no-prefill-tune", cmd)
        self.assertIn("--dspark", cmd)                    # (the config's memory as it runs: its drafter kept)
        self.assertEqual(seen["ids"], "1,2,3")

    def test_no_tokenizer_no_ids(self):
        def runner(cmd, env, cwd, timeout, say):
            self.assertNotIn("--ids", cmd)               # (the engine's own token mix)
            Path(cmd[cmd.index("--calibrate-prefill") + 1]).write_text(json.dumps({"chunk": 1024}))
            return 0, []
        with patch.object(CPF, "prompt_ids", return_value=[]):
            self.assertEqual(CPF.run(mm3_cfg(), say=lambda *a: None, runner=runner)["chunk"], 1024)

    def test_failed_run_raises(self):
        with patch.object(CPF, "prompt_ids", return_value=[]):
            with self.assertRaisesRegex(RuntimeError, "exit 1.*out of memory"):
                CPF.run(mm3_cfg(), say=lambda *a: None,
                        runner=lambda *a: (1, ["maya-mm3: out of memory\n", "\n"]))


class Setup(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = Path(self.tmp.name) / "maya-mm3.json"
        self.path.write_text(json.dumps(mm3_cfg()))
        self.quiet = [patch.object(maya, n, lambda *a, **k: None) for n in ("say", "ok", "warn")] + \
            [patch.object(maya, "busy_gpus", return_value=[])]
        for p in self.quiet:
            p.start()

    def tearDown(self):
        for p in self.quiet:
            p.stop()
        self.tmp.cleanup()

    def test_recorded_in_the_config(self):
        res = {"chunk": 4096, "gemm": "hipblaslt", "tok_s": 300.0, "default_tok_s": 120.0, "tokens": 4096,
               "file": "/m/pack/prefill-tune.txt"}
        with patch.object(maya.CPF, "run", return_value=res):
            self.assertTrue(maya.prefill_calibration(self.path))
        rec = json.loads(self.path.read_text())["installer"]["prefill_calibration"]
        self.assertEqual((rec["chunk"], rec["gemm"], rec["tok_s"], rec["default_tok_s"]), (4096, "hipblaslt", 300.0, 120.0))
        self.assertFalse(maya.needs_prefill_calibration(json.loads(self.path.read_text())))

    def test_failure_recorded_not_retried_by_a_start(self):
        with patch.object(maya.CPF, "run", side_effect=RuntimeError("the engine stopped")):
            self.assertFalse(maya.prefill_calibration(self.path))
        cfg = json.loads(self.path.read_text())
        self.assertIn("failed", cfg["installer"]["prefill_calibration"])
        self.assertFalse(maya.needs_prefill_calibration(cfg))   # (--calibrate tries again)

    def test_needs_once_minimax_only(self):
        self.assertTrue(maya.needs_prefill_calibration(mm3_cfg()))
        self.assertFalse(maya.needs_prefill_calibration(mm3_cfg(installer={"family": "minimax",
                                                                          "prefill_calibration": {"chunk": 1024}})))
        self.assertFalse(maya.needs_prefill_calibration({"exe": "/x/maya-ds4", "args": [],
                                                         "installer": {"family": "deepseek"}}))

    def test_calibrate_measures_minimax_prefill(self):
        with patch.object(maya, "prefill_calibration", return_value=True) as pc:
            self.assertTrue(maya.calibrate_config(self.path))
        pc.assert_called_once()

    def test_busy_gpus_not_measured(self):
        with patch.object(maya, "busy_gpus", return_value=[(0, 40000)]), patch.object(maya.CPF, "run") as r:
            self.assertFalse(maya.prefill_calibration(self.path))
        r.assert_not_called()


if __name__ == "__main__":
    unittest.main()

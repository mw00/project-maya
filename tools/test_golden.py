"""tools/test_golden.py - tools/maya_golden.py and the bench's parts behind it (no GPU, no engine).

    python -m unittest tools.test_golden -v
"""
import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path[:0] = [str(Path(__file__).resolve().parent)]
import maya_bench as MB  # noqa: E402
import maya_golden as G  # noqa: E402

CFG = {"args": ["--model", "/m/x.gguf", "--max-context", "32768"], "env": {"IK2_WARM": "1"}, "backend": "cuda",
       "port": 8080}


def tokens(**over):
    """A --bench-tokens file as maya.py writes it (short answers)."""
    run = {"warm-up": [5, 6, 7], "decode 1": list(range(100, 140)), "decode 2": list(range(200, 240)),
           "prompt 2K #1": [9], "prompt 8K #1": [11]}
    t = {"machine": "box", "gpus": "Tesla V100-PCIE-32GB", "driver": "595.1", "model": "Inkling-2 Mini UD-Q4_K_XL",
         "date": "2026-11-24", "config": "", "family": "ik2", "backend": "cuda", "gpu_index": [1], "context": 32768,
         "commit": "8c618a59", "version": "1.1.76", "text_ref": "v1.1.76", "compare": "",
         "inputs": {k: "in-" + k for k in run}, "choices": {"data/expert-profile-ik2mini.bin": "aaaa"},
         "heads": {k: "text of " + k for k in run}, "runs": {"this": [run]}, "records": []}
    t.update(over)
    return t


def changed(t, name, at, value=-1, rounds=None):
    """A copy of a tokens file with one answer changed from token `at`."""
    t = json.loads(json.dumps(t))
    for r in (t["runs"]["this"] if rounds is None else [t["runs"]["this"][i] for i in rounds]):
        r[name] = r[name][:at] + [value] + r[name][at + 1:]
    return t


class BenchParts(unittest.TestCase):
    def test_prompt_text_files_order(self):
        names = ["README.md", "docs/SETUP.md", "docs/AMD.md", "docs/media/x.md", "docs/logo.png",
                 "bench/results/b.md", "bench/results/a.md", "bench/micro/x.md"]
        self.assertEqual(MB.prompt_text_files(names),
                         ["README.md", "docs/AMD.md", "docs/SETUP.md", "bench/results/a.md", "bench/results/b.md"])

    def test_input_hashes_name_requests_as_the_session_does(self):
        h = MB.input_hashes([[1, 2], [3]], {2048: [[4], [5]], 8192: [[6], [7]]}, [8])
        self.assertEqual(sorted(h), sorted(["warm-up", "decode 1", "decode 2", "prompt 2K #1", "prompt 2K #2",
                                            "prompt 8K #1", "prompt 8K #2"]))
        self.assertNotEqual(h["decode 1"], h["decode 2"])
        self.assertEqual(h, MB.input_hashes([[1, 2], [3]], {2048: [[4], [5]], 8192: [[6], [7]]}, [8]))

    def test_saved_answers_are_not_learned_files(self):
        with tempfile.TemporaryDirectory() as d:
            r = Path(d)
            (r / "data" / "golden" / "box").mkdir(parents=True)
            (r / "data" / "golden" / "box" / "m.json").write_text("{}")
            (r / "data" / "expert-profile-x.bin").write_bytes(b"p")
            self.assertEqual([p.name for p in MB.learned_files(r, {})], ["expert-profile-x.bin"])
            c = MB.choice_files(r, {})
            self.assertEqual(list(c), ["data/expert-profile-x.bin"])
            (r / "data" / "expert-profile-x.bin").write_bytes(b"q")
            self.assertNotEqual(MB.choice_files(r, {}), c)

    def test_a_compare_runs_restore_keeps_the_saved_answers(self):
        """--compare puts the learned files back before every round and drops what a run wrote into data/: the
        saved answers are neither (the first real nightly swept them away)."""
        with tempfile.TemporaryDirectory() as d:
            r, other = Path(d) / "this", Path(d) / "other"
            gold = r / "data" / "golden" / "box" / "m.json"
            gold.parent.mkdir(parents=True)
            gold.write_text("{}")
            (r / "data" / "expert-profile-x.bin").write_bytes(b"p")
            (other / "data").mkdir(parents=True)
            st = MB.LearnedState(r, {}, Path(d) / "snap")
            (r / "data" / "written-by-the-run.txt").write_text("x")
            (r / "data" / "expert-profile-x.bin").write_bytes(b"changed")
            st.restore(other)
            self.assertTrue(gold.exists())
            self.assertFalse((r / "data" / "written-by-the-run.txt").exists())
            self.assertEqual((r / "data" / "expert-profile-x.bin").read_bytes(), b"p")
            self.assertFalse((other / "data" / "golden").exists())
            st.finish()
            self.assertTrue(gold.exists())


class Locks(unittest.TestCase):
    def test_two_runs_that_found_it_quiet_together_do_not_both_get_a_gpu(self):
        import os
        from unittest.mock import patch
        with tempfile.TemporaryDirectory() as d, patch.dict(os.environ, {"MAYA_GPU_LOCKS": d}), \
                patch.object(MB, "LOCK_DIRS", ()):
            a = MB.take_locks("cuda", [0, 1], "run A")
            self.assertEqual([p.name for p in a], ["gpu0", "gpu1"])
            with self.assertRaisesRegex(RuntimeError, r"another run took .*gpu1 \(run A\) first"):
                MB.take_locks("cuda", [1, 2], "run B")        # (gpu1 is A's: B backs off ...)
            self.assertEqual(sorted(os.listdir(d)), ["gpu0", "gpu1"])   # (... and gives gpu2 back)
            self.assertEqual((Path(d) / "gpu1").read_text(), "run A\n")
            self.assertEqual([p.name for p in MB.take_locks("cuda", [2], "run B")], ["gpu2"])
            MB.release_locks(a, "run A")
            self.assertEqual(os.listdir(d), ["gpu2"])


class Sets(unittest.TestCase):
    def test_same_run_passes(self):
        s = G.make_set(tokens(), CFG)
        res = G.compare(s, tokens(), G.config_hash(CFG))
        self.assertEqual((res["status"], res["compared"], res["diffs"], res["notes"]), ("PASS", 5, [], []))
        self.assertEqual(G.verdict_line(res), "PASS 5/5")

    def test_a_difference_names_the_request_and_the_token(self):
        s = G.make_set(tokens(), CFG)
        res = G.compare(s, changed(tokens(), "decode 2", 17, 999), G.config_hash(CFG))
        self.assertEqual(res["status"], "DIFF")
        self.assertEqual(res["diffs"], ["decode 2: differs from token 17 (saved 217, now 999; 40 vs 40 tokens)"])
        self.assertTrue(G.verdict_line(res).startswith("DIFF decode 2: differs from token 17"))

    def test_a_shorter_answer_differs_at_its_end(self):
        s = G.make_set(tokens(), CFG)
        t = tokens()
        t["runs"]["this"][0]["decode 1"] = t["runs"]["this"][0]["decode 1"][:30]
        self.assertIn("decode 1: differs from token 30 (saved 130, now end; 40 vs 30 tokens)",
                      G.compare(s, t, G.config_hash(CFG))["diffs"])

    def test_a_changed_prompt_is_not_a_difference(self):
        s = G.make_set(tokens(), CFG)
        t = changed(tokens(), "prompt 8K #1", 0, 77)
        t["inputs"]["prompt 8K #1"] = "other docs"
        res = G.compare(s, t, G.config_hash(CFG))
        self.assertEqual((res["status"], res["compared"], len(res["skipped"])), ("PASS", 4, 1))
        self.assertIn("--text-ref v1.1.76", res["skipped"][0])

    def test_changed_choices_are_named(self):
        s = G.make_set(tokens(), CFG)
        t = tokens(choices={"data/expert-profile-ik2mini.bin": "bbbb"})
        same = G.compare(s, t, G.config_hash(CFG))
        self.assertEqual(same["status"], "PASS")
        self.assertIn("expert-profile-ik2mini.bin", same["notes"][0])
        self.assertIn("still equal", same["notes"][0])
        diff = G.compare(s, changed(t, "decode 1", 3), G.config_hash(CFG))
        self.assertEqual(diff["status"], "DIFF")
        self.assertIn("a new measurement can explain", diff["notes"][0])

    def test_rounds_that_vary_between_starts_are_all_kept(self):
        t = tokens()
        t["runs"]["this"].append(dict(t["runs"]["this"][0], **{"decode 1": list(range(100, 120)) + [1] * 20}))
        s = G.make_set(t, CFG)
        self.assertEqual(len(s["answers"]["decode 1"]["also"]), 1)
        self.assertEqual(G.compare(s, t, G.config_hash(CFG))["status"], "PASS")       # both rounds are known answers
        res = G.compare(s, changed(t, "decode 1", 25, 4242, rounds=[1]), G.config_hash(CFG))
        self.assertEqual(res["diffs"], ["decode 1 (round 2): differs from token 25 (saved 1, now 4242; 40 vs 40 "
                                        "tokens)"])

    def test_pick_set_prefers_same_config_and_choices(self):
        a = G.make_set(tokens(), CFG)
        b = G.make_set(tokens(choices={"x": "1"}, version="1.1.77"), CFG)
        doc = G.add_set(G.add_set({}, a), b)
        self.assertEqual(len(doc["sets"]), 2)
        self.assertIs(G.pick_set(doc, a["config_hash"], a["choices_hash"]), a)
        self.assertIs(G.pick_set(doc, a["config_hash"], "unknown"), b)             # newest of the same config
        self.assertIs(G.pick_set(doc, "other config", "unknown"), b)
        self.assertIs(G.pick_set(doc, a["config_hash"], "unknown", release="v1.1.76"), a)
        self.assertIsNone(G.pick_set({}, "x", "y"))
        self.assertEqual(len(G.add_set(doc, G.make_set(tokens(), CFG))["sets"]), 2)  # same key: replaced

    def test_config_hash_ignores_port_and_keys(self):
        self.assertEqual(G.config_hash(CFG), G.config_hash(dict(CFG, port=9000, api_key="k")))
        self.assertNotEqual(G.config_hash(CFG), G.config_hash(dict(CFG, env={})))

    def test_file_name(self):
        self.assertEqual(G.golden_path(Path("g"), tokens()).name,
                         "inkling-2-mini-ud-q4.k.xl-tesla-v100-pcie-32gb-32k.json".replace("q4.k.xl", "q4-k-xl"))
        self.assertEqual(G.golden_path(Path("g"), tokens(), "1x V100 resident").name,
                         "inkling-2-mini-ud-q4-k-xl-1x-v100-resident.json")


class CommandLine(unittest.TestCase):
    def run_main(self, *argv):
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = G.main(list(argv))
        return code, out.getvalue()

    def test_capture_check_and_a_hand_edited_file(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            cfg = d / "maya-x.json"
            cfg.write_text(json.dumps(CFG))
            tf = d / "tokens.json"
            tf.write_text(json.dumps(tokens(config=str(cfg))))
            base = ["--from-tokens", str(tf), "--dir", str(d / "golden"), "--layout", "one"]
            code, out = self.run_main("--check", *base)
            self.assertEqual(code, 2)
            self.assertIn("no saved answers", out)
            code, out = self.run_main("--capture", *base)
            self.assertEqual(code, 0)
            path = d / "golden" / "inkling-2-mini-ud-q4-k-xl-one.json"
            self.assertTrue(path.exists())
            self.assertEqual(G.saved_text_ref(d / "golden", cfg, "one"), "v1.1.76")
            code, out = self.run_main("--check", *base)
            self.assertEqual(code, 0)
            self.assertIn("PASS: 5 request(s) compared, 0 differ", out)
            doc = json.loads(path.read_text())                 # (a saved answer edited by hand: caught, and where)
            doc["sets"][0]["answers"]["decode 1"]["ids"][12] = 31337
            path.write_text(json.dumps(doc))
            code, out = self.run_main("--check", *base)
            self.assertEqual(code, 1)
            self.assertIn("DIFF  decode 1: differs from token 12 (saved 31337, now 112", out)
            code, out = self.run_main("--list", "--dir", str(d / "golden"))
            self.assertIn("v1.1.76 8c618a59", out)

    def test_bench_command(self):
        cmd = G.bench_command(Path("/r"), Path("c.json"), Path("t.json"), "v1.1.76", "v1.1.76", 2, 2)
        self.assertEqual(cmd[2:], ["--config", "c.json", "--bench", "--bench-tokens", "t.json", "--bench-wait", "2",
                                   "--bench-text-ref", "v1.1.76", "--compare", "v1.1.76", "--bench-rounds", "2"])
        self.assertNotIn("--compare", G.bench_command(Path("/r"), Path("c.json"), Path("t.json")))


if __name__ == "__main__":
    unittest.main()

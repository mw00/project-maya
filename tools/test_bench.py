"""tools/test_bench.py - --bench (maya.py, tools/maya_bench.py): the engine binary on HIP, the requests, and the
speed-test standard's parsing, medians, result line, quiet check and A/B parts. No GPU, no network, no engine process.

    python -m unittest tools.test_bench -v
"""
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402
import maya_bench as MB  # noqa: E402


class ExeForTarget(unittest.TestCase):
    """exe_for_target: the one place EXE vs DS4_EXE gets picked by name - bench_exe and every build step
    (compile_engine, compile_engine_hip, build_step) all go through it."""

    def setUp(self):
        for ctx in (patch.object(maya, "EXE", Path("/build-hip/maya")),
                    patch.object(maya, "DS4_EXE", Path("/build-hip/maya-ds4"))):
            ctx.start()
            self.addCleanup(ctx.stop)

    def test_maya_target_is_the_glm_binary(self):
        self.assertEqual(maya.exe_for_target("maya"), maya.EXE)

    def test_maya_ds4_target_is_the_ds4_binary(self):
        self.assertEqual(maya.exe_for_target("maya-ds4"), maya.DS4_EXE)

    def test_unknown_target_raises_instead_of_guessing(self):
        # the old bug's failure mode was silently picking the wrong binary; an unknown target should fail loudly.
        with self.assertRaises(KeyError):
            maya.exe_for_target("maya-glm-typo")

    def test_reads_exe_and_ds4_exe_live_not_at_import_time(self):
        # select_build_backend() reassigns EXE/DS4_EXE once the backend is known; a dict built once at import time
        # would freeze the pre-backend-selection values and silently go stale.
        with patch.object(maya, "EXE", Path("/build/maya")), patch.object(maya, "DS4_EXE", Path("/build/maya-ds4")):
            self.assertEqual(maya.exe_for_target("maya"), Path("/build/maya"))
        with patch.object(maya, "EXE", Path("/build-hip/maya")), patch.object(maya, "DS4_EXE",
                                                                                Path("/build-hip/maya-ds4")):
            self.assertEqual(maya.exe_for_target("maya"), Path("/build-hip/maya"))


class BenchExeSelection(unittest.TestCase):
    """#350: --bench used to hardcode the GLM binary for every HIP config, crashing a DeepSeek setup with
    FileNotFoundError (build-hip/maya was never built there - build-hip/maya-ds4 was)."""

    def setUp(self):
        for ctx in (patch.object(maya, "EXE", Path("/build-hip/maya")),
                    patch.object(maya, "DS4_EXE", Path("/build-hip/maya-ds4"))):
            ctx.start()
            self.addCleanup(ctx.stop)

    def test_deepseek_config_gets_the_ds4_binary(self):
        cfg = {"exe": "/build-hip/maya-ds4", "backend": "hip",
               "args": ["--pack", "/m/maya-ds4", "--max-context", "65536"]}
        self.assertEqual(maya.bench_exe(cfg), maya.DS4_EXE)

    def test_glm_config_gets_the_glm_binary(self):
        cfg = {"exe": "/build-hip/maya", "backend": "hip", "args": ["--glm-pack", "/m/pack"]}
        self.assertEqual(maya.bench_exe(cfg), maya.EXE)

    def test_old_pre_1_1_16_ds4_name_still_recognised(self):
        cfg = {"exe": "/build-hip/strata-ds4", "backend": "hip", "args": ["--pack", "/m/maya-ds4"]}
        self.assertEqual(maya.bench_exe(cfg), maya.DS4_EXE)

    def test_agrees_with_config_family_by_construction(self):
        # bench_exe must never answer differently than config_family - that drift is exactly what caused #350.
        ds4 = {"exe": "/build-hip/maya-ds4", "args": ["--pack", "/m/maya-ds4"]}
        glm = {"exe": "/build-hip/maya", "args": ["--glm-pack", "/m/pack"]}
        for cfg in (ds4, glm):
            expect = maya.DS4_EXE if maya.config_family(cfg) == "deepseek" else maya.EXE
            self.assertEqual(maya.bench_exe(cfg), expect)

    def test_minimax_config_gets_the_mm3_binary(self):
        with patch.object(maya, "config_family", return_value="minimax"), \
                patch.object(maya, "MM3_EXE", Path("/build-hip/maya-mm3")):
            self.assertEqual(maya.bench_exe({"exe": "/build-hip/maya-mm3", "args": []}), Path("/build-hip/maya-mm3"))


class BenchRequests(unittest.TestCase):
    """#549: each family's own chat format and short answers out of the mean; #420 / #509: a --parallel engine's GEN
    carries rid=0 and its DONE line its id."""

    def test_chat_formats(self):
        self.assertEqual(maya.bench_chat("glm", Path("/none"))("hi"), "[gMASK]<sop><|user|>\nhi<|assistant|>\n</think>")
        self.assertEqual(maya.bench_chat("deepseek", Path("/none"))("hi"),
                         "<｜begin▁of▁sentence｜><｜User｜>hi<｜Assistant｜></think>")

    def test_minimax_uses_its_own_template(self):
        try:
            import jinja2  # noqa: F401
        except ImportError:
            self.skipTest("jinja2 not installed")
        s = maya.bench_chat("minimax", Path("/none"))("hi")
        self.assertNotIn("[gMASK]", s)
        self.assertIn("hi", s)

    def test_gen_line(self):
        self.assertEqual(maya.bench_gen_line(256, [1, 2], False), "GEN 256 temperature=0 1,2\n")
        self.assertEqual(maya.bench_gen_line(256, [1, 2], True), "GEN 256 temperature=0 rid=0 1,2\n")

    def test_reply(self):
        self.assertEqual(maya.bench_reply("DONE 256 9 1.0 2.0 length\n", False)[:2], ["DONE", "256"])
        self.assertEqual(maya.bench_reply("DONE 0 256 9 1.0 2.0 length\n", True)[:2], ["DONE", "256"])
        self.assertIsNone(maya.bench_reply("DONE 3 256 9 1.0 2.0 length\n", True))   # (another request's)
        self.assertEqual(maya.bench_reply("ERR this engine runs several conversations: GEN needs rid=<id>", True)[0],
                         "ERR")
        self.assertIsNone(maya.bench_reply("PP 64 128 1.0 2.0 0", False))

    def test_short_answer_not_in_the_median(self):
        a = [MB.parse_done(f"DONE {n} 9 1.0 {ms} length".split()) for n, ms in ((256, 12800.0), (11, 275.0),
                                                                                    (256, 11636.4))]
        self.assertEqual([round(x, 1) for x in MB.decode_samples(a)], [20.0, 22.0])
        self.assertEqual(MB.decode_samples(a[1:2]), [40.0])   # (all short: all of them)


class BenchStandard(unittest.TestCase):
    """tools/maya_bench.py: the speed-test standard's parsing, medians, result line, quiet check and A/B summary."""

    def test_done_fields_and_speeds(self):
        d = MB.parse_done("DONE 256 2048 3200.0 4096.0 length 100 150 0 7 9".split())
        self.assertEqual((d["produced"], d["prompt"], d["d_acc"], d["d_off"], d["reused"]), (256, 2048, 100, 150, 0))
        self.assertAlmostEqual(MB.decode_speed(d), 62.5)
        self.assertAlmostEqual(MB.prompt_speed(d), 640.0)
        d = MB.parse_done("DONE 1 2048 1000.0 0.0 length 0 0 1024".split())   # (half of it reused: half read)
        self.assertAlmostEqual(MB.prompt_speed(d), 1024.0)
        self.assertEqual(MB.parse_done(["DONE", "5"])["decode_ms"], 0.0)       # (short line: zeros, no error)

    def test_token_lines(self):
        self.assertEqual(MB.token_of("T 42\n", False), 42)
        self.assertEqual(MB.token_of("T 0 42\n", True), 42)
        self.assertIsNone(MB.token_of("T 3 42\n", True))                       # (another conversation's)
        self.assertIsNone(MB.token_of("TOOLS x", False))
        self.assertIsNone(MB.token_of("DONE 1 2", False))

    def test_median_and_stats(self):
        self.assertEqual(MB.median([3, 1, 2]), 2)
        self.assertEqual(MB.median([4, 1, 2, 3]), 2.5)
        self.assertEqual(MB.stats([61.5, 60.8, 61.2]), (61.2, 60.8, 61.5))
        self.assertEqual(MB.stats([]), (0.0, 0.0, 0.0))
        self.assertEqual(MB.fmt_stats((61.2, 60.8, 61.5)), "61.2 (60.8-61.5)")

    def test_record_round_trip(self):
        r = {"machine": "jupiter", "gpus": "3x Tesla V100-PCIE-32GB", "driver": "580.178.04",
             "model": "glm-5.3-flash Maya-S-v2-IQ2_XXS", "config": "32K MTP", "commit": "ead35d2b",
             "version": "1.1.75", "decode": (61.2, 60.8, 61.5),
             "prompt": {2048: (456.4, 450, 460), 8192: (640.2, 630, 650)}, "date": "2026-11-22"}
        line = MB.record_line(r)
        self.assertEqual(line, "jupiter | 3x Tesla V100-PCIE-32GB | 580.178.04 | glm-5.3-flash Maya-S-v2-IQ2_XXS | "
                               "32K MTP | ead35d2b | v1.1.75 | decode 61.2 (60.8-61.5) | prompt 2K 456 / 8K 640 | "
                               "2026-11-22")
        back = MB.parse_record(line)
        self.assertEqual((back["decode"], back["prompt"], back["version"], back["note"]),
                         ((61.2, 60.8, 61.5), {2048: 456.0, 8192: 640.0}, "1.1.75", ""))
        self.assertEqual(MB.parse_record(MB.record_line(dict(r, note="A/B 2 rounds")))["note"], "A/B 2 rounds")
        with self.assertRaises(ValueError):
            MB.parse_record("compare a vs b | exact: yes")

    def test_quiet_verdict(self):
        self.assertEqual(MB.quiet_reasons([], [], 0.4, []), [])
        self.assertEqual(MB.quiet_reasons([], [], None, []), [])               # (Windows: no load average)
        why = MB.quiet_reasons([(1, "pid 7 python3")], [("/l/gpu1", "other helper")], 3.2, ["~/.jupiter-hot"])
        self.assertEqual(len(why), 4)
        self.assertIn("GPU 1 is in use (pid 7 python3)", why)
        self.assertIn("CPU load 3.2 (needs < 2)", why)

    def test_busy_threads_from_proc_stat(self):
        with patch.object(MB.os, "cpu_count", return_value=80):
            a = "cpu  1000 0 1000 8000 0 0 0 0 0 0"
            b = "cpu  1010 0 1015 8075 0 0 0 0 0 0"                           # (25 busy of 100 jiffies)
            self.assertAlmostEqual(MB.busy_threads(a, b), 20.0)
            self.assertAlmostEqual(MB.busy_threads(a, a), 0.0)

    def test_busy_gpus_only_ours(self):
        apps, util = "0,1234,/usr/bin/python3\n2,99,maya", "0,0\n1,85\n2,3\n3,0"
        self.assertEqual(MB.gpus_busy_nvidia(apps, util, [1, 2]), [(2, "pid 99 maya"), (1, "85% busy")])
        self.assertEqual(MB.gpus_busy_nvidia("", util, None), [(1, "85% busy")])
        self.assertEqual(MB.lock_names("cuda", [0, 2]), ["gpu0", "gpu2"])
        self.assertEqual(MB.lock_names("hip", [0]), ["radeon"])

    def test_busy_gpus_windows_desktop(self):
        # (a card that also draws the desktop: nvidia-smi lists the desktop's programs as compute apps, no memory)
        apps = "\n".join([r"0,1780,C:\Windows\System32\dwm.exe,[N/A]", r"0,8028,C:\Windows\explorer.exe,[N/A]",
                          r"0,7424,C:\Program Files\Google\Chrome\Application\chrome.exe,[N/A]",
                          r"0,10680,C:\Program Files\WindowsApps\Notes_1.0\app\notes.exe,[N/A]"])
        self.assertEqual(MB.gpus_busy_nvidia(apps, "0,3,353", [0], windows=True), [])
        self.assertEqual(MB.gpus_busy_nvidia(apps, "0,3,353", None, windows=True), [])
        # (the GPU's own readings decide: busy, or more memory in use than a desktop's)
        self.assertEqual(MB.gpus_busy_nvidia(apps, "0,45,353", [0], windows=True), [(0, "45% busy: notes.exe")])
        self.assertEqual(MB.gpus_busy_nvidia(apps, "0,2,6100", [0], windows=True),
                         [(0, "6100 MiB in use by other programs: notes.exe")])
        self.assertEqual(MB.gpus_busy_nvidia("", "0,0,2048\n1,0,2049", None, windows=True),
                         [(1, "2049 MiB in use by other programs")])
        # (a program with a memory figure that is not the desktop's is another job; the desktop's never is)
        other = "\n".join([apps, r"0,4242,C:\maya\build\maya-ik2.exe,5120 MiB", r"0,1,C:\Windows\System32\dwm.exe,90 MiB"])
        self.assertEqual(MB.gpus_busy_nvidia(other, "0,0,5500", [0], windows=True), [(0, "pid 4242 maya-ik2.exe")])
        self.assertEqual(MB.gpus_busy_nvidia(other, "0,0,5500", [1], windows=True), [])
        self.assertTrue(MB.desktop_app("C:/Windows/SystemApps/X_cw5n1h2txyewy/SearchHost.exe"))
        self.assertFalse(MB.desktop_app(r"C:\Users\me\python.exe"))

    def test_machine_quiet_reads_the_desktop_card(self):
        # (the whole check, nvidia-smi's two answers as a Windows desktop gives them: quiet there, in use on Linux)
        def smi(cmd, timeout=15):
            if any("--query-gpu" in x for x in cmd):
                return "0, 00000000:01:00.0, 0, 353\n"
            return ("00000000:01:00.0, 1780, C:\\Windows\\System32\\dwm.exe, [N/A]\n"
                    "00000000:01:00.0, 10680, C:\\Program Files\\WindowsApps\\Notes_1.0\\app\\notes.exe, [N/A]\n")
        with patch.object(MB, "_out", side_effect=smi), patch.object(MB, "cpu_load_now", return_value=None), \
                patch.object(MB, "held_locks", return_value=[]), patch.object(MB, "HOT_FLAGS", ()), \
                patch.dict(MB.os.environ, {"MAYA_HOT_FLAG": ""}):
            with patch.object(MB, "on_windows", return_value=True):
                self.assertEqual(MB.machine_quiet("cuda", [0]), [])
            with patch.object(MB, "on_windows", return_value=False):
                self.assertEqual(MB.machine_quiet("cuda", [0]), ["GPU 0 is in use (pid 1780 dwm.exe)",
                                                                 "GPU 0 is in use (pid 10680 notes.exe)"])

    def test_busy_gpus_linux_stays_strict(self):
        # (Linux: any compute app on our GPU counts, whatever its name or memory; memory in use alone does not)
        apps = "0,1780,/usr/bin/dwm.exe,[N/A]\n1,7,/usr/lib/firefox/firefox,12 MiB"
        self.assertEqual(MB.gpus_busy_nvidia(apps, "0,0,300\n1,0,300\n2,0,9000", None),
                         [(0, "pid 1780 dwm.exe"), (1, "pid 7 firefox")])

    def test_drafting_choice(self):
        a = [MB.parse_done("DONE 256 9 1 2 length 150 200 0".split())]
        self.assertEqual(MB.drafting_choice([], {}, ["INFO drafting: the MTP block - the fastest on this PC (x)"], a),
                         "the MTP block, 75% of drafts accepted")
        self.assertEqual(MB.drafting_choice(["--drafting", "off"], {}, [], []), "drafting off")
        self.assertEqual(MB.drafting_choice([], {}, ["glm mtp: CUDA0 the MTP decode drafts"], [], "glm"), "MTP")
        self.assertEqual(MB.drafting_choice([], {}, ["dspark: x"], [], "ik2"), "no drafting")
        self.assertEqual(MB.drafting_choice([], {}, ["dspark: 3 drafts"], [], "deepseek"), "DSpark")

    def test_exactness(self):
        self.assertIsNone(MB.first_difference([1, 2, 3], [1, 2, 3]))
        self.assertEqual(MB.first_difference([1, 2, 3], [1, 5, 3]), 1)
        self.assertEqual(MB.first_difference([1, 2], [1, 2, 3]), 2)
        self.assertEqual(MB.exactness({"decode 1": [1, 2]}, {"decode 1": [1, 2]}), [])
        self.assertEqual(MB.exactness({"decode 1": [1, 2]}, {"decode 1": [1, 3]}),
                         ["decode 1: differs from token 1 (2 vs 2 tokens)"])

    def test_compare_line(self):
        a = {"decode": (60.0, 59, 61), "prompt": {2048: (400.0, 1, 1)}}
        b = {"decode": (63.0, 62, 64), "prompt": {2048: (420.0, 1, 1)}}
        self.assertEqual(MB.compare_line(a, b, "old", "new", "yes"),
                         "compare new vs old | exact: yes | decode 63.0 vs 60.0 +5.0% | 2K 420 vs 400 +5.0%")

    def test_exact_verdict_over_every_pairing(self):
        x, y = {"decode 1": [1, 2, 3]}, {"decode 1": [1, 2, 4]}
        self.assertEqual(MB.exact_verdict([x, x], [x, x]), ("yes", []))
        self.assertTrue(MB.exact_verdict([x, y], [x, x])[0].startswith("yes in 2 of 4 pairings"))
        self.assertEqual(MB.exact_verdict([x, x], [y, y]),
                         ("NO", ["decode 1: differs from token 2 (3 vs 3 tokens)"]))

    def test_prompt_plan_has_different_openings(self):
        doc = list(range(40000))
        prompts, warm = MB.plan_prompts(doc, 32768)
        self.assertEqual(sorted(prompts), [2048, 8192])
        starts = [s[0] for v in prompts.values() for s in v] + [warm[0]]
        self.assertEqual(len(set(starts)), 7)
        self.assertTrue(all(len(s) == n for n, v in prompts.items() for s in v))
        self.assertEqual(sorted(MB.plan_prompts(doc, 4096)[0]), [2048])         # (8K does not fit the context)

    def test_cmake_options_carry_both_names(self):
        cache = ("CMAKE_BUILD_TYPE:STRING=Release\nCMAKE_CUDA_ARCHITECTURES:UNINITIALIZED=70\n"
                 "CMAKE_CXX_COMPILER_AR:FILEPATH=/usr/bin/gcc-ar\nMAYA_GGML_DIR:PATH=/x/llama.cpp\n"
                 "MAYA_BUILD_ID:STRING=\nFOO:BOOL=ON\n")
        self.assertEqual(MB.cmake_options(cache), ["-DCMAKE_BUILD_TYPE=Release", "-DCMAKE_CUDA_ARCHITECTURES=70",
                                                   "-DMAYA_GGML_DIR=/x/llama.cpp", "-DSTRATA_GGML_DIR=/x/llama.cpp"])

    def test_learned_files_restored_and_copied(self):
        with tempfile.TemporaryDirectory() as t:
            a, b = Path(t) / "a", Path(t) / "b"
            (a / "data").mkdir(parents=True)
            (b / "data").mkdir(parents=True)
            (a / "data" / "hy4-drafting.txt").write_text("mtp")
            (b / "data" / "hy4-drafting.txt").write_text("b own")
            prof = Path(t) / "profile.bin"
            prof.write_bytes(b"P0")
            st = MB.LearnedState(a, {"args": ["--expert-profile", str(prof)]}, Path(t) / "snap")
            prof.write_bytes(b"P1")                                              # (a run rewrote it)
            (a / "data" / "new.txt").write_text("learned")                       # (... and learned a new one)
            st.restore(b)
            self.assertEqual(prof.read_bytes(), b"P0")
            self.assertFalse((a / "data" / "new.txt").exists())
            self.assertEqual((b / "data" / "hy4-drafting.txt").read_text(), "mtp")
            st.finish()
            self.assertEqual((b / "data" / "hy4-drafting.txt").read_text(), "b own")
            self.assertEqual((a / "data" / "hy4-drafting.txt").read_text(), "mtp")

    def test_map_cfg_points_at_the_other_build(self):
        here, there = Path("/m/a").resolve(), Path("/m/b").resolve()
        c = MB.map_cfg({"exe": str(here / "build" / "maya"), "cwd": str(here), "args": []}, here, there)
        self.assertEqual((Path(c["exe"]), Path(c["cwd"])), (there / "build" / "maya", there))


if __name__ == "__main__":
    unittest.main()

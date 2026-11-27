"""tools/test_bench.py - --bench (maya.py, tools/maya_bench.py): the engine binary on HIP, the requests, and the
speed-test standard's parsing, medians, result line, quiet check and A/B parts. No GPU, no network, no engine process (a few lines of Python stand in for one).

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

    def test_drafting_choice_of_a_first_start(self):
        # (the plan line names every source the engine loaded; the measurement's line says what was kept)
        plan = "drafting: the drafter and the prompt lookup, 1.54 GiB of VRAM from the experts' tier with the window's"
        going = "NOTE maya-ik2: measuring this PC once, before the first answer: drafting (3 requests, 128 tokens each)"
        done = ("NOTE maya-ik2: measured this PC once in 41 s: drafting - no drafting (no drafting 21.4, the prompt "
                "lookup 7.5, the drafter 8.0, both 7.9 tokens/s); plain decoding was timed in lane parity")
        self.assertEqual(MB.drafting_choice(["--drafting", "auto"], {}, [plan, going, done], []), "no drafting")
        self.assertEqual(MB.drafting_choice([], {}, [plan, going, done.replace("- no drafting (", "- the drafter (")],
                                            []), "the drafter")
        self.assertEqual(MB.drafting_choice([], {}, [plan, going], []), "measuring")
        # (the two-run measurement: the start's own line comes last and is the choice)
        kept = "drafting: no drafting - the fastest on this PC (measured 2026-11-23: no drafting 21.4, ...)"
        self.assertEqual(MB.drafting_choice([], {}, [going, done, kept], []), "no drafting")
        self.assertEqual(MB.drafting_choice([], {}, [plan], []), "the drafter and the prompt lookup")

    def test_the_log_keeps_the_start_lines(self):
        # (a stand-in engine that writes its start lines to stdout only, as maya-ik2 does on Windows)
        code = ("import sys\n"
                "print('expert cache, GPU 0: 590 slots', flush=True)\n"
                "print('drafting: no drafting - the fastest on this PC', flush=True)\n"
                "print('a note on stderr', file=sys.stderr, flush=True)\n"
                "print('READY', flush=True)\n"
                "sys.stdin.readline()\n")
        with tempfile.TemporaryDirectory() as td:
            log = Path(td) / "maya-bench.log"
            sess = MB.Session([sys.executable, "-c", code], None, td, log)
            self.assertEqual(sess.lines, ["expert cache, GPU 0: 590 slots",
                                          "drafting: no drafting - the fastest on this PC"])
            sess.close()
            sess.p.stdin.close(), sess.p.stdout.close()
            text = log.read_text(encoding="utf-8")
            for want in sess.lines + ["a note on stderr"]:
                self.assertIn(want, text)
            self.assertNotIn("READY", text)

    def test_a_lock_taken_by_hand(self):
        with tempfile.TemporaryDirectory() as td, patch.object(MB, "LOCK_DIRS", (td,)), \
                patch.object(MB, "_out", return_value=""), patch.object(MB, "cpu_load_now", return_value=None), \
                patch.object(MB, "HOT_FLAGS", ()), patch.dict(MB.os.environ, {"MAYA_HOT_FLAG": "", "MAYA_GPU_LOCKS": "",
                                                                              "MAYA_BENCH_LOCK_OK": ""}):
            lock = Path(td) / "gpu0"
            lock.write_text("mine, by hand\n")
            why = MB.machine_quiet("cuda", [0])                       # (the default: strict)
            self.assertEqual(why, [f"GPU lock {lock} is held: mine, by hand"])
            self.assertEqual(MB.machine_quiet("cuda", [0], lock_ok=True), [])
            self.assertEqual(MB.wait_quiet("cuda", [0], 0, lambda *a: None, lock_ok=True), [])
            self.assertFalse(MB.lock_ok_asked())
            self.assertTrue(MB.lock_ok_asked(True))
            with patch.dict(MB.os.environ, {"MAYA_BENCH_LOCK_OK": "1"}):
                self.assertTrue(MB.lock_ok_asked())
            with patch.dict(MB.os.environ, {"MAYA_BENCH_LOCK_OK": "0"}):
                self.assertFalse(MB.lock_ok_asked())
            # (the run takes no lock that is there and releases only its own: the hand-made one stays)
            taken = MB.take_locks("cuda", [0, 1], "bench note", held_ok=True)
            self.assertRaises(RuntimeError, MB.take_locks, "cuda", [0, 1], "bench note")   # (without it: a loss)
            self.assertEqual([p.name for p in taken], ["gpu1"])
            MB.release_locks(taken + [lock], "bench note")
            self.assertEqual(sorted(p.name for p in Path(td).iterdir()), ["gpu0"])
            self.assertEqual(lock.read_text(), "mine, by hand\n")

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


class SharedCpuTiming(unittest.TestCase):
    def test_other_build_takes_this_builds_values(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = Path(d) / "a.txt", Path(d) / "b.txt"
            rest = " | /m/pack | layer 3 | 7536640 B | 0000:15:00.0 Tesla V100 | 36 threads | 4"
            a.write_text("0.1.27 Nov 23 2026 23:41:47" + rest + "\t0.2520 0.6175 0.6148 0.2305\n" +
                         "0.1.27 Nov 24 2026 03:00:00" + rest + "\t0.1250 0.5780 0.5800 0.1300\n")
            b.write_text("0.1.27 Nov 24 2026 02:00:00" + rest + "\t0.2380 0.5770 0.5790 0.2400\n")
            self.assertEqual(MB.share_timing(a, b), 1)
            self.assertEqual(b.read_text(), "0.1.27 Nov 24 2026 02:00:00" + rest + "\t0.1250 0.5780 0.5800 0.1300\n")
            self.assertEqual(MB.share_timing(a, Path(d) / "none.txt"), 0)


class EngineStartMatchesServer(unittest.TestCase):
    """The bench starts the engine from the server's own start code.  v1.1.77: the server gave maya-ik2 a pictures
    figure (images on) that the bench never did, so the bench showed full speed while served answers were slow."""

    EXES = ("maya", "maya-ds4", "maya-mm3", "maya-hy4", "maya-ik2")

    def cfg(self, exe, tmp):
        return {"exe": exe, "gpu": [0], "args": [], "vision": {"exe": "maya-vision", "mmproj": str(Path(tmp) / "m.gguf"),
                                                               "gpu": True}}

    def test_bench_env_has_every_key_the_server_sets(self):
        import serve.server as SV
        with tempfile.TemporaryDirectory() as d, patch.object(SV, "vision_footprint", return_value=(4096, 2 << 30)):
            cache = Path(d) / "vision-memory.json"
            for exe in self.EXES:
                for images in (True, False):
                    c = self.cfg(exe, d)
                    if not images:
                        c.pop("vision")
                    served = SV.vision_plan(c, SV.child_env(c), cache)[4]   # what serve/server.py starts the engine with
                    bench = SV.engine_environment(c, cache)                 # what the bench's start_command reads
                    self.assertEqual(sorted(set(served) - set(bench)), [], f"{exe} images={images}")
                    self.assertEqual(sorted(set(bench) - set(served)), [], f"{exe} images={images}")

    def test_bench_start_command_reads_the_servers_environment_function(self):
        import inspect
        src = inspect.getsource(MB.start_command)
        self.assertIn("SV.engine_environment", src)
        self.assertIn("SV.engine_args", src)


GLM_LOG = """glm split auto: layers [0, 17) on CUDA0, [17, 31) on CUDA1, [31, 45) on CUDA2 - a token 10.9 ms predicted (CUDA0 4.0 ms)
glm fast: CUDA0 layers [0,17) expert pool 27.20 GB, 291-291 slots/layer (4074 total)
glm fast: CUDA2 layers [31,45) expert pool 27.58 GB, 252-280 slots/layer, the draft layer 270 (4058 total)
glm fast: CUDA0 RAM tier 2.46 GB pinned, 368 slots
glm fast: CUDA0 warming the expert tiers 10% (2.9 of 28.9 GB, 1 s)
glm fast: CUDA0 tiers warm: 4032 experts (28.9 GB) in 23 s (1262.1 MB/s; slot order, parallel reads)
glm fast: CUDA0 CPU lane off: the RAM tier holds no expert
glm tier: CUDA2 CPU lane pool mode: one shared pool (MAYA_GLM_CPU_SHARED=unset, 3 part(s), 2 NUMA node(s))
glm fast: CUDA2 CPU lane: 36 threads (one pool, shared), an expert 0.224 ms on the CPU vs 0.576 ms over PCIe -> of 0..8 RAM-tier experts the CPU takes 0 1 2 2 3 4 4 5 6
glm fast: CUDA2 CPU lane: those times from /x/data/glm-cpu-cal.txt (not timed)
INFO context=32768 kv=int8 engine_kind=glm-fast experts=12096 vram_slots=12206 ram_slots=1405 mtp=1 dflash=0 v
maya serve: CUDA0: 1088 MiB of VRAM free with everything loaded
glm mtp: 97 rounds
"""


class LayoutFingerprint(unittest.TestCase):
    def test_facts_from_a_glm_log(self):
        f = MB.layout_facts(GLM_LOG.splitlines(), "MTP")
        self.assertEqual(f["GPUs"], "CUDA0,CUDA1,CUDA2")  # (the split line names all three)
        self.assertEqual(f["split"], "[0, 17) on CUDA0, [17, 31) on CUDA1, [31, 45) on CUDA2")
        self.assertEqual(f["slots a layer"], "CUDA0 291-291 / CUDA2 252-280")
        self.assertEqual(f["RAM-tier slots"], "CUDA0 368")
        self.assertEqual(f["warmed experts"], "CUDA0 4032")
        self.assertIn("CUDA0 off", f["lane"])
        self.assertIn("CUDA2 36 threads, takes 0 1 2 2 3 4 4 5 6", f["lane"])
        self.assertEqual(f["lane times"], "read from the saved file")
        self.assertEqual(f["lane pool"], "one shared pool")
        self.assertEqual(f["VRAM free MiB"], "CUDA0 1088")
        self.assertIn("vram_slots=12206", f["engine"])
        self.assertEqual(f["drafting"], "MTP")

    def test_seconds_of_the_start_are_not_facts(self):
        a = MB.layout_facts(GLM_LOG.splitlines())
        b = MB.layout_facts(GLM_LOG.replace("in 23 s", "in 31 s").replace("1 s)", "2 s)").splitlines())
        self.assertEqual(MB.layout_diff(a, b), [])

    def test_diff_names_the_fact(self):
        a = MB.layout_facts(GLM_LOG.splitlines())
        b = MB.layout_facts(GLM_LOG.replace("368 slots", "300 slots").replace("1088 MiB", "900 MiB").splitlines())
        d = MB.layout_diff(a, b)
        self.assertEqual(len(d), 2)
        self.assertTrue(any(x.startswith("RAM-tier slots: CUDA0 368 -> CUDA0 300") for x in d))

    def test_no_facts_is_said(self):
        self.assertEqual(MB.layout_facts(["hello"]), {})
        self.assertIn("no placement facts", MB.layout_line({}))

    def test_report_for_differing_rounds(self):
        a = {"facts": MB.layout_facts(GLM_LOG.splitlines()), "log_kept": "x/this-r1.log"}
        b = {"facts": MB.layout_facts(GLM_LOG.replace("368 slots", "300 slots").splitlines()),
             "log_kept": "x/this-r2.log"}
        out = MB.layout_report({"this": [a, b], "other": [a]})
        self.assertEqual(len(out), 2)
        self.assertIn("this build, round 1 vs round 2: RAM-tier slots", out[0])
        self.assertIn("x/this-r2.log", out[0])
        self.assertIn("same facts", out[1])

    def test_logs_are_kept_per_round_and_cleaned(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            src = root / "maya-bench.log"
            src.write_text("one\n", encoding="utf-8")
            k1 = MB.keep_log(src, root / MB.KEEP_DIR / "this-r1.log")
            src.write_text("two\n", encoding="utf-8")
            k2 = MB.keep_log(src, root / MB.KEEP_DIR / "this-r2.log")
            self.assertEqual(Path(k1).read_text(encoding="utf-8"), "one\n")
            self.assertEqual(Path(k2).read_text(encoding="utf-8"), "two\n")
            self.assertIsNone(MB.keep_log(src, None))
            MB.clean_kept_logs(root)
            self.assertFalse((root / MB.KEEP_DIR).exists())



class SetupWording(unittest.TestCase):
    def test_picture_reader_time_is_an_honest_range(self):
        self.assertIn("up to about 10 minutes", maya.VISION_BUILD_TIME)
        self.assertNotIn("a few", maya.VISION_BUILD_TIME)
        src = Path(maya.__file__).read_text(encoding="utf-8")
        self.assertIn('" kernels: " + VISION_BUILD_TIME', src)
        self.assertNotIn('"CUDA") + " kernels: a few minutes"', src)
        setup = (Path(maya.__file__).parent / "docs" / "SETUP.md").read_text(encoding="utf-8")
        self.assertIn("compiles the image encoder (1 to 6 minutes, up to about 10 minutes on a small PC; once)", setup)
        self.assertIn("compiles the engine for your GPU(s) (a few minutes, once)", setup)   # (the engine's text stays)


if __name__ == "__main__":
    unittest.main()

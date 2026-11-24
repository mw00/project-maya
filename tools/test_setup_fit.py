"""tools/test_setup_fit.py - the setup's check that a GLM-5.3-Flash quant can start on this PC's one NVIDIA card,
made before any download (maya.glm_fit / glm_fit_verdict / glm_fit_gate): the 8 GB card that drives a desktop is
set up with a warning (the engine starts there without an expert pool), a card below the dense weights is refused,
the cards that run the quants today are not, and --force-fit goes on.  No
GPU, no network."""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya  # noqa: E402

XS = "Maya-XS-IQ1_S"


def pc(vram_gb, name="RTX", n=1, **more):
    return {"backend": "cuda", "gpus": [{"index": i, "name": name, "vram_gb": vram_gb, "arch": "86", **more}
                                        for i in range(n)]}


class Stop(Exception):
    pass


def stop(what, how=None):
    raise Stop(f"{what} | {how}")


class Estimate(unittest.TestCase):
    def test_the_8gb_card_matches_the_engines_message(self):
        # RTX 3050 8 GB, Windows desktop ~0.5 GiB, 32K, the default 700 MiB reserve: the engine said "about 1245
        # MiB more" (2026-11-18)
        t = maya.glm_fit(XS, 8192, 512, 32768, reserve_mib=700)
        self.assertAlmostEqual(t["short"], 1245, delta=200)
        self.assertAlmostEqual(t["pool"], 1747, delta=10)          # (8 slots a layer: the engine's 1,747 MiB)
        self.assertAlmostEqual(t["dense"] / 1024, 5.24, delta=0.01)
        self.assertAlmostEqual(t["state"] / 1024, 0.45, delta=0.01)
        self.assertAlmostEqual(maya.glm_fit_state_mib(8192, "int8") / 1024, 0.28, delta=0.01)
        # ... and at 8K with a 150 MiB reserve: "477 MiB" short
        self.assertAlmostEqual(maya.glm_fit(XS, 8192, 512, 8192, reserve_mib=150)["short"], 477, delta=200)

    def test_terms_add_up(self):
        t = maya.glm_fit(XS, 16384, 300, 32768)
        self.assertAlmostEqual(t["need"], t["dense"] + t["state"] + t["work"] + t["pool"] + t["reserve"])
        self.assertEqual(t["reserve"], 300)
        self.assertAlmostEqual(t["short"], t["need"] - (16384 - 300))
        self.assertIsNone(maya.glm_fit("EXL3-3.05bpw", 8192, 0))    # (no figures: not judged)

    def test_a_streamed_context_counts_as_the_resident_window(self):
        self.assertEqual(maya.glm_fit_state_mib(262144, "int8"), maya.glm_fit_state_mib(32768, "int8"))
        self.assertGreater(maya.glm_fit_state_mib(32768), maya.glm_fit_state_mib(32768, "int8"))   # (FP16 latents)

    def test_every_offered_gguf_quant_has_figures(self):
        for q, m in maya.MODELS.items():
            if not maya.is_exl3(q):
                self.assertIn(q, maya.GLM_FIT)
        for q in maya.GLM_FIT:
            self.assertIn(q, maya.MODELS)


class Verdict(unittest.TestCase):
    def verdict(self, quant, vram_gb, used_mib, **kw):
        return maya.glm_fit_verdict(quant, pc(vram_gb), used={0: used_mib}, **kw)

    def test_8gb_with_a_desktop_starts_without_a_pool(self):
        v = self.verdict(XS, 8.0, 512)
        self.assertEqual(v["verdict"], "slow")                      # (the engine starts with no expert pool)
        self.assertEqual(v["contexts"], [])
        self.assertEqual(v["at"]["ctx"], 8192)
        self.assertGreater(v["at"]["short"], 500)
        self.assertLess(v["at"]["short"] - v["at"]["pool"], 0)

    def test_a_card_below_the_dense_weights_is_refused(self):
        v = self.verdict(XS, 6.0, 512)
        self.assertEqual(v["verdict"], "no")
        for q in maya.GLM_FIT:                                      # (no quant of the model fits it)
            self.assertEqual(self.verdict(q, 6.0, 512)["verdict"], "no", q)

    def test_8gb_with_nothing_on_it_is_a_warning_only(self):
        self.assertEqual(self.verdict(XS, 8.0, 0)["verdict"], "tight")

    def test_cards_that_run_it_today_are_allowed(self):
        # README's speed table and docs/MEMORY.md: (quant, card GB, context); with a desktop (0.5 GiB) and with the
        # most VRAM in use the check counts (1 GiB)
        rigs = [(XS, 12.0, 32768), (XS, 16.0, 32768), (XS, 16.0, 262144),          # (RTX A4000 16 GB at 256K)
                ("Maya-S24", 12.0, 65536),                                          # (RTX 5070 12 GB)
                ("Maya-S24", 16.0, 32768), ("Maya-S24", 24.0, 131072),              # (RTX 3090 / PRO 4000 24 GB)
                ("Maya-S-v2-IQ2_XXS", 16.0, 131072), ("Maya-S-v2-IQ2_XXS", 24.0, 131072),
                ("Maya-M", 24.0, 131072), ("Maya-L", 24.0, 131072)]
        for quant, gb, ctx in rigs:
            for used in (512, 1024, 6000):                          # (6000: another program - at most 1 GiB counts)
                v = self.verdict(quant, gb, used, contexts=maya.CONTEXTS + [262144])
                self.assertEqual(v["verdict"], "fits", (quant, gb, used))
                self.assertIn(ctx, v["contexts"], (quant, gb, ctx))

    def test_only_one_nvidia_card_with_its_own_vram_is_judged(self):
        used = {0: 512, 1: 0}
        self.assertIsNone(maya.glm_fit_verdict(XS, pc(8.0, n=2), used=used))            # (a split: not judged)
        self.assertIsNone(maya.glm_fit_verdict(XS, pc(8.0, vendor="amd"), used=used))
        self.assertIsNone(maya.glm_fit_verdict(XS, pc(8.0, unified_memory=True), used=used))
        self.assertIsNone(maya.glm_fit_verdict(XS, {"backend": "hip", "gpus": pc(8.0)["gpus"]}, used=used))
        self.assertIsNone(maya.glm_fit_verdict(XS, {"backend": "metal", "gpus": pc(8.0)["gpus"]}, used=used))
        self.assertIsNone(maya.glm_fit_verdict(XS, None, used=used))
        self.assertIsNone(maya.glm_fit_verdict(XS, {"backend": "cuda", "gpus": [{"vram_gb": 8.0}]}, used=used))
        self.assertIsNone(maya.glm_fit_verdict("EXL3-3.05bpw", pc(8.0), used=used))

    def test_a_smaller_context_where_only_that_fits(self):
        # a card on which 8K is inside the margin and 32K is not: only 8K is offered
        t8, t32 = maya.glm_fit(XS, 0, 0, 8192), maya.glm_fit(XS, 0, 0, 32768)
        total = (t8["need"] + t32["need"]) / 2 - maya.GLM_FIT_MARGIN_MIB       # (between the two limits)
        v = maya.glm_fit_verdict(XS, pc(total / 1024), used={0: 0})
        self.assertEqual(v["verdict"], "tight")
        self.assertEqual(v["contexts"], [8192])


class Gate(unittest.TestCase):
    def gate(self, quant, vram_gb, used_mib, **args):
        a = SimpleNamespace(force_fit=False, yes=True, **args)
        said = []
        with patch.object(maya, "fail", side_effect=stop), \
                patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            return maya.glm_fit_gate(a, quant, pc(vram_gb, "NVIDIA GeForce RTX 3050"), used={0: used_mib}), said

    def test_refused_before_the_download_with_what_is_short(self):
        with self.assertRaises(Stop) as e:
            self.gate(XS, 6.0, 512)
        msg = str(e.exception)
        self.assertIn("cannot start on this PC's NVIDIA GeForce RTX 3050 (6 GB)", msg)
        self.assertIn("GB short", msg)
        self.assertIn("dense weights 5.2", msg)
        self.assertIn("no GLM-5.3-Flash quant fits this card", msg)
        self.assertIn("Nothing was downloaded", msg)
        self.assertIn("--force-fit", msg)
        for word in ("strata", "Strata", "Ollama", "tok/s", "tokens/s"):
            self.assertNotIn(word, msg)

    def test_tight_warns_and_goes_on(self):
        v, said = self.gate(XS, 8.0, 0)
        self.assertEqual(v["verdict"], "tight")
        self.assertTrue(any("tight fit" in s for s in said))

    def test_fits_says_nothing(self):
        v, said = self.gate(XS, 12.0, 512)
        self.assertEqual((v["verdict"], said), ("fits", []))

    def test_a_busy_card_is_said_never_refused(self):
        v, said = self.gate(XS, 12.0, 11000)
        self.assertEqual(v["verdict"], "fits")
        self.assertTrue(any("in use now" in s for s in said))

    def test_the_quants_that_fit_are_named(self):
        v, said = self.gate("Maya-L", 12.0, 1024)                  # (starts without a pool: said, with what fits)
        self.assertEqual(v["verdict"], "slow")
        line = " ".join(said)
        self.assertIn("these GLM-5.3-Flash quants fit this card:", line)
        self.assertIn("Maya-XS", line)

    def test_a_context_that_does_not_fit(self):
        t8, t32 = maya.glm_fit(XS, 0, 0, 8192), maya.glm_fit(XS, 0, 0, 32768)
        total = (t8["need"] + t32["need"]) / 2 - maya.GLM_FIT_MARGIN_MIB
        v = maya.glm_fit_verdict(XS, pc(total / 1024), used={0: 0})
        with patch.object(maya, "fail", side_effect=stop), patch.object(maya, "warn"):
            self.assertEqual(maya.glm_fit_context(SimpleNamespace(force_fit=False), v, None), [8192])
            with self.assertRaises(Stop):
                maya.glm_fit_context(SimpleNamespace(force_fit=False), v, 32768)
            self.assertIsNone(maya.glm_fit_context(SimpleNamespace(force_fit=True), v, 32768))
            self.assertIsNone(maya.glm_fit_context(SimpleNamespace(force_fit=False), None, 32768))


class Slow(unittest.TestCase):
    def test_the_8gb_card_is_set_up_with_a_warning(self):
        a = SimpleNamespace(force_fit=False, yes=True)
        said = []
        with patch.object(maya, "fail", side_effect=stop),                 patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            v = maya.glm_fit_gate(a, XS, pc(8.0, "NVIDIA GeForce RTX 3050"), used={0: 512})
            self.assertIsNone(maya.glm_fit_context(a, v, 32768))    # (every context is offered: nothing is refused)
        self.assertEqual(v["verdict"], "slow")
        self.assertTrue(any("will run slowly on this PC's NVIDIA GeForce RTX 3050 (8 GB)" in s for s in said), said)
        for word in ("strata", "Strata", "Ollama", "tok/s", "tokens/s"):
            self.assertFalse(any(word in s for s in said))


class ForceFit(unittest.TestCase):
    def test_force_fit(self):
        a = SimpleNamespace(force_fit=True, yes=True)
        said = []
        with patch.object(maya, "fail", side_effect=stop), \
                patch.object(maya, "warn", side_effect=lambda *x: said.append(" ".join(map(str, x)))):
            v = maya.glm_fit_gate(a, XS, pc(6.0), used={0: 512})
        self.assertEqual(v["verdict"], "no")
        self.assertTrue(any("going on (--force-fit)" in s for s in said))

    def test_the_option_exists(self):
        src = (Path(maya.__file__)).read_text(encoding="utf-8")
        self.assertIn('"--force-fit"', src)


class Menu(unittest.TestCase):
    def rows(self, vram_gb, used_mib):
        args = SimpleNamespace(gguf_dir=None, model=None, yes=False, backend="cuda", glm_part="flash")
        seen = {}

        def choose(question, intro, options, default, yes, outro=()):
            seen.update(options=options, default=default)
            return default

        gpus = pc(vram_gb)["gpus"]
        with tempfile.TemporaryDirectory() as d, \
                patch.object(maya.S, "gpus", return_value=gpus), \
                patch.object(maya, "mem_gb", return_value=(64.0, 64.0)), \
                patch.object(maya, "gpu_used_mib", return_value={0: used_mib}), \
                patch.object(maya, "choose", side_effect=choose):
            kind, quant = maya.choose_model(args, Path(d), {}, pc(vram_gb))
        return seen["options"], quant

    def test_the_8gb_card_marks_every_row(self):
        options, _ = self.rows(8.0, 512)
        notes = [note for label, note, _ in options if "EXL3" not in label]
        self.assertTrue(notes and all(n in ("does not fit this PC's 8 GB card",
                                            "runs slowly on this PC's 8 GB card (experts on the CPU)")
                                      for n in notes), notes)
        self.assertIn("runs slowly on this PC's 8 GB card (experts on the CPU)", notes)

    def test_a_12gb_card_marks_only_what_cannot_start(self):
        options, quant = self.rows(12.0, 512)
        marked = [label for label, note, _ in options if note and "does not fit" in note]
        self.assertTrue(all("Maya-L" in m for m in marked), marked)
        self.assertNotIn("does not fit", next(n or "" for label, n, _ in options if "Maya-XS" in label))
        self.assertNotEqual(quant, "Maya-L")

    def test_no_pc_no_marks(self):
        args = SimpleNamespace(gguf_dir=None, model=None, yes=True, backend="cuda")
        with tempfile.TemporaryDirectory() as d, \
                patch.object(maya.S, "gpus", return_value=[{"vram_gb": 8.0}]), \
                patch.object(maya, "mem_gb", return_value=(64.0, 64.0)), \
                patch.object(maya, "say"):
            self.assertEqual(maya.choose_model(args, Path(d), {})[0], "download")


if __name__ == "__main__":
    unittest.main()

"""serve/test_turbo.py - the Turbo button: the per-model settings map, the engine config's snapshot and restore
(serve/models.py), the server's /api/turbo, and the page's request-level snapshot (serve/web/turbo-core.js, under node).

    python -m unittest serve.test_turbo -v
"""
from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import unittest
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from serve import models as M  # noqa: E402
from serve.test_models import ds4_cfg, glm_cfg, mm3_cfg  # noqa: E402

WEB = Path(__file__).resolve().parent / "web"


class ResidentSkip(unittest.TestCase):
    """MAYA_DS4_RESIDENT_* / MAYA_GLM_RESIDENT_*: the experts a token goes without wherever they live (Turbo's lever for
    a model that fits in VRAM): the option names, their env aliasing, Turbo's values, the engines reading them."""
    ROOT = Path(__file__).resolve().parents[1]

    def test_turbo_values_are_sane(self):
        for fam, name, k_exp in (("glm", "MAYA_GLM_RESIDENT_TOPK", 8), ("deepseek", "MAYA_DS4_RESIDENT_TOPK", 6)):
            k = int(M.TURBO_ENV[fam][name])
            self.assertTrue(1 <= k < k_exp, (fam, k))          # (a cap below the model's own k, never zero: top-1 stays)
        # the weight rule is not part of Turbo (the sweep on the test PC: 0.20 gave garbage)
        for env in M.TURBO_ENV.values():
            self.assertFalse([k for k in env if k.endswith("RESIDENT_SKIP_W")])

    def test_turbo_env_aliases_to_the_names_the_engines_read(self):
        from tools import maya_names as N
        env = dict(M.TURBO_ENV["glm"], **M.TURBO_ENV["deepseek"])
        N.alias_env(env)
        self.assertIn("MAYA_GLM_RESIDENT_TOPK", env)
        self.assertIn("MAYA_DS4_RESIDENT_TOPK", env)
        old = {"STRATA_DS4_RESIDENT_TOPK": "3"}                    # (an older name alone counts as the MAYA_ one)
        N.alias_env(old)
        self.assertEqual(old["MAYA_DS4_RESIDENT_TOPK"], "3")
        self.assertEqual(N.read_as("STRATA_DS4_RESIDENT_SKIP_W"), "MAYA_DS4_RESIDENT_SKIP_W")
        exported = {}
        N.export_env(exported, {"MAYA_GLM_RESIDENT_TOPK": "4"})
        self.assertEqual((exported["MAYA_GLM_RESIDENT_TOPK"], exported["STRATA_GLM_RESIDENT_TOPK"]), ("4", "4"))

    def test_the_engines_read_the_options_and_the_docs_name_them(self):
        src = {"DS4": (self.ROOT / "src/ds4/ds4_kernels.cu").read_text(encoding="utf-8"),
               "GLM": (self.ROOT / "src/kernels/cuda/glm_fast.cu").read_text(encoding="utf-8")}
        docs = (self.ROOT / "docs/SETTINGS.md").read_text(encoding="utf-8")
        for prefix, text in src.items():
            for opt in ("RESIDENT_TOPK", "RESIDENT_SKIP_W"):
                self.assertIn(f'getenv("MAYA_{prefix}_{opt}")', text, (prefix, opt))
                self.assertIn(f"MAYA_{prefix}_{opt}", docs)
        self.assertEqual(docs.count("changes the answers - Turbo"), 2)

    def test_a_dropped_expert_is_a_null_plan_entry_and_the_top_one_stays(self):
        # (the rule itself runs on the GPU; this pins the pieces the kernels' comments and the docs promise)
        ds = (self.ROOT / "src/ds4/ds4_kernels.cu").read_text(encoding="utf-8")
        self.assertIn("rank == 0 || ((rs_k <= 0 || rank < rs_k)", ds)            # (top-1 always kept)
        self.assertIn("kept ? vram[pick[r]] : 0ull", ds)                          # (dropped: off the device ...)
        self.assertIn("kept ? pick[r] : -1", ds)                                  # ... and not on the CPU lane
        glm = (self.ROOT / "src/kernels/cuda/glm_fast.cu").read_text(encoding="utf-8")
        self.assertIn("const bool out = rank != 0 &&", glm)
        self.assertIn("if (a.norm_w && drop != 0u) inv =", glm)                   # (renormalised only where the model does)
        self.assertIn("resident_ok", (self.ROOT / "include/strata/kernels/glm_fast.hpp").read_text(encoding="utf-8"))

    def test_off_in_a_config_without_turbo(self):
        d = Path(tempfile.mkdtemp())
        try:
            for make, key in ((glm_cfg, "MAYA_GLM_RESIDENT_TOPK"), (ds4_cfg, "MAYA_DS4_RESIDENT_TOPK")):
                self.assertNotIn(key, M.read(make(d)).get("env") or {})   # (opt-in: nothing sets it but Turbo)
        finally:
            shutil.rmtree(d, ignore_errors=True)


class TurboMap(unittest.TestCase):

    def setUp(self):
        self.d = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_glm(self):
        p = glm_cfg(self.d)                                    # (131072 context, MTP, no parallel)
        patch = M.turbo_patch(M.read(p))
        self.assertEqual(patch["context"], 32768)
        self.assertEqual(patch["env"]["MAYA_GLM_MISS_SKIP_W"], "0.16")
        self.assertEqual(patch["env"]["MAYA_GLM_MISS_SKIP_FROM"], "23")
        self.assertEqual(patch["env"]["MAYA_GLM_RESIDENT_TOPK"], "4")       # (the all-in-VRAM lever, below)
        self.assertEqual(patch["env"]["MAYA_GLM_RAM_HEADROOM_GB"], "4")     # (the config's own variable stays)
        self.assertNotIn("draft", patch)                       # (GLM keeps the drafter the setup measured)

    def test_smallest_quants_keep_every_expert(self):
        """GLM and DeepSeek Maya-XS: Turbo's other settings, but no expert cap (TURBO_NO_TOPK)."""
        for make, key, other in ((glm_cfg, "MAYA_GLM_RESIDENT_TOPK", "MAYA_GLM_MISS_SKIP_W"),
                                 (ds4_cfg, "MAYA_DS4_RESIDENT_TOPK", "MAYA_DS4_RAM_SKIP_W")):
            cfg = M.read(make(self.d))
            cfg.setdefault("installer", {})["ds4_quant" if key.startswith("MAYA_DS4") else "quant"] = \
                "maya-xs" if key.startswith("MAYA_DS4") else "Maya-XS-IQ1_S"
            patch = M.turbo_patch(cfg)
            self.assertNotIn(key, patch["env"])
            self.assertIn(other, patch["env"])

    def test_deepseek_keeps_its_context(self):
        p = ds4_cfg(self.d)
        patch = M.turbo_patch(M.read(p))
        self.assertNotIn("context", patch)
        self.assertEqual(patch["env"], {"MAYA_DS4_RAM_SKIP_W": "0.14", "MAYA_DS4_DSPARK_CUM": "0.3",
                                        "MAYA_DS4_RESIDENT_TOPK": "3"})
        self.assertEqual(patch["dspark_conf"], 0.3)                    # (the draft floor lowered from 0.7)
        self.assertNotIn("MAYA_LOOP_GUARD", patch["env"])              # (the loop guard is left on)

    def test_deepseek_lookup_and_drafter(self):
        p = ds4_cfg(self.d)
        cfg = M.read(p)
        cfg["args"] += ["--suffix-draft", "0", "--parallel", "3"]
        cfg["args"][cfg["args"].index("--dspark") + 1] = "none"
        patch = M.turbo_patch(cfg)
        self.assertEqual((patch["draft"], patch["prompt_lookup"], patch["parallel"]), ("dspark", True, 1))
        # no drafter in the pack: not offered
        cfg2 = M.read(ds4_cfg(self.d, name="maya-b.json", drafter=False))
        self.assertNotIn("draft", M.turbo_patch(cfg2))

    def test_minimax(self):
        patch = M.turbo_patch(M.read(mm3_cfg(self.d)))
        self.assertEqual(patch.get("context"), 32768 if M.settings_of(M.read(mm3_cfg(self.d)))["context"] > 32768 else None)
        self.assertNotIn("env", patch)                         # (no expert skipping in maya-mm3)
        self.assertEqual(patch.get("dspark_conf"), 0.3)

    def test_smaller_context_is_kept(self):
        cfg = M.read(glm_cfg(self.d))
        cfg["args"][cfg["args"].index("--max-context") + 1] = "8192"
        self.assertNotIn("context", M.turbo_patch(cfg))


class TurboSnapshot(unittest.TestCase):
    def setUp(self):
        self.d = Path(tempfile.mkdtemp())
        self.p = glm_cfg(self.d)
        self.before = json.loads(self.p.read_text())

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_on_then_off_is_exact(self):
        out = M.turbo_enable(self.p)
        self.assertTrue(out["changed"])
        on = M.read(self.p)
        self.assertTrue(M.turbo_on(on))
        self.assertEqual(M.settings_of(on)["context"], 32768)
        self.assertEqual(on["env"]["MAYA_GLM_MISS_SKIP_W"], "0.16")
        self.assertTrue(M.turbo_info(on)["on"])
        out = M.turbo_disable(self.p)
        self.assertTrue(out["changed"])
        back = M.read(self.p)
        for k in ("args", "env", "gpu", "vision", "port", "sampling"):
            self.assertEqual(back.get(k), self.before.get(k), k)
        self.assertNotIn("turbo", back)

    def test_twice_and_off_without_on(self):
        M.turbo_enable(self.p)
        with self.assertRaises(ValueError):
            M.turbo_enable(self.p)
        M.turbo_disable(self.p)
        with self.assertRaises(ValueError):
            M.turbo_disable(self.p)

    def test_context_raised_in_turbo_is_restored_too(self):
        M.turbo_enable(self.p)
        M.save(self.p, {"context": 65536})                     # (allowed: the user raises it without leaving Turbo)
        self.assertTrue(M.turbo_on(M.read(self.p)))
        self.assertEqual(M.settings_of(M.read(self.p))["context"], 65536)
        M.turbo_disable(self.p)
        back = M.read(self.p)
        self.assertEqual(M.settings_of(back)["context"], 131072)
        self.assertEqual(back["env"], self.before["env"])
        self.assertEqual(M.settings_of(back)["gpus"], [0, 1])

    def test_other_edits_in_turbo_survive_the_restore(self):
        M.turbo_enable(self.p)
        M.save(self.p, {"context": 65536, "reserve_mib": 1234})
        M.turbo_disable(self.p)
        s = M.settings_of(M.read(self.p))
        self.assertEqual((s["context"], s["reserve_mib"]), (131072, 1234))

    def test_nothing_to_change_needs_no_restart(self):
        cfg = M.read(self.p)
        cfg["env"] = {**cfg["env"], **M.TURBO_ENV["glm"]}
        cfg["args"][cfg["args"].index("--max-context") + 1] = "32768"
        self.p.write_text(json.dumps(cfg))
        self.assertFalse(M.turbo_enable(self.p)["changed"])   # (the marker is written, the engine runs as it is)
        self.assertTrue(M.turbo_on(M.read(self.p)))
        self.assertFalse(M.turbo_disable(self.p)["changed"])
        self.assertEqual(M.read(self.p)["args"], cfg["args"])

    def test_env_absent_before_is_absent_after(self):
        cfg = M.read(self.p)
        cfg.pop("env")
        self.p.write_text(json.dumps(cfg))
        M.turbo_enable(self.p)
        self.assertIn("MAYA_GLM_MISS_SKIP_W", M.read(self.p)["env"])
        M.turbo_disable(self.p)
        self.assertNotIn("env", M.read(self.p))

    def test_models_page_save_is_unchanged(self):
        self.assertTrue(M.save(self.p, {"context": 65536})["changed"])
        self.assertFalse(M.save(self.p, {"context": 65536})["changed"])


class TurboServer(unittest.TestCase):
    def setUp(self):
        from serve.server import ByteTokenizer, MockEngine, Service, serve
        self.d = Path(tempfile.mkdtemp())
        self.p = glm_cfg(self.d)
        tok = ByteTokenizer()
        self.svc = Service(MockEngine(tok, "hello", max_context=32768), tok, None)
        self.svc.config_path = str(self.p)
        self.httpd = serve(self.svc, port=0)
        import threading
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.httpd.server_address[1]}"

    def tearDown(self):
        self.httpd.shutdown()
        shutil.rmtree(self.d, ignore_errors=True)

    def call(self, path, body=None):
        req = urllib.request.Request(self.base + path, data=None if body is None else json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"} if body is not None else {})
        with urllib.request.urlopen(req) as r:
            return json.loads(r.read())

    def test_info_on_off(self):
        info = self.call("/api/turbo")
        self.assertFalse(info["on"])
        self.assertTrue(info["engine"])
        self.assertEqual(info["plan"]["context"]["to"], 32768)
        out = self.call("/api/turbo", {"on": True})           # (no MAYA_RESTART_ON_UPDATE here: saved, no restart)
        self.assertTrue(out["on"])
        self.assertFalse(out.get("restarting_now"))
        self.assertTrue(self.call("/api/turbo")["on"])
        self.assertFalse(self.call("/api/turbo", {"on": False})["on"])
        self.assertEqual(M.read(self.p)["args"], json.loads(self.p.read_text())["args"])

    def test_bad_body(self):
        import urllib.error
        with self.assertRaises(urllib.error.HTTPError) as e:
            self.call("/api/turbo", {"on": "yes"})
        self.assertEqual(e.exception.code, 400)


@unittest.skipUnless(shutil.which("node"), "node is not installed")
class TurboPageSide(unittest.TestCase):
    def run_js(self, code):
        src = (WEB / "turbo-core.js").as_uri()
        out = subprocess.run(["node", "--input-type=module", "-e", f'import * as T from "{src}";\n{code}'],
                             capture_output=True, text=True, timeout=60)
        self.assertEqual(out.returncode, 0, out.stderr)
        return json.loads(out.stdout)

    def test_fastest_thinking(self):
        r = self.run_js('console.log(JSON.stringify([T.fastestThinking(["none","low","high","max"]),'
                        'T.fastestThinking(["none","auto","low"]),T.fastestThinking(["low","medium","xhigh"])]))')
        self.assertEqual(r, ["none", "none", "low"])

    def test_enter_and_leave_restore_exactly(self):
        r = self.run_js('''const own = {temperature: 0.9, top_k: 40, thinking: "max", prefs_v: 3, system: "be brief"};
const e = T.turboEnter(own, ["none","low","high","max"], "high");
const l = T.turboLeave(e.snapshot, "none");
console.log(JSON.stringify({next: e.next, back: l.stored, chat: l.chatThinking, untouched: own}));''')
        self.assertEqual(r["next"]["temperature"], 0)
        self.assertEqual(r["next"]["thinking"], "none")
        self.assertEqual(r["next"]["top_k"], 40)                # (the rest of the user's settings is kept while on)
        self.assertEqual(r["back"], r["untouched"])             # (and comes back as it was)
        self.assertEqual(r["chat"], "high")

    def test_nothing_chosen_before(self):
        r = self.run_js('''const e = T.turboEnter(null, ["low","medium","xhigh"], null);
const l = T.turboLeave(e.snapshot, "low");
console.log(JSON.stringify({thinking: e.next.thinking, back: l.stored, chat: l.chatThinking ?? null}));''')
        self.assertEqual((r["thinking"], r["back"], r["chat"]), ("low", {}, None))

    def test_chat_level_changed_in_turbo_is_not_overwritten(self):
        r = self.run_js('''const e = T.turboEnter({}, ["none","low"], "low");
console.log(JSON.stringify({chat: T.turboLeave(e.snapshot, "none").chatThinking ?? null}));''')
        self.assertEqual(r["chat"], "low")                      # (the chat still at Turbo's level: back to what it had)
        r = self.run_js('''const e = T.turboEnter({}, ["none","low"], "high");
console.log(JSON.stringify({chat: T.turboLeave(e.snapshot, "low").chatThinking ?? null}));''')
        self.assertIsNone(r["chat"])                            # (the user picked "low" meanwhile: kept)

    def test_coding_ideas(self):
        r = self.run_js('console.log(JSON.stringify(T.CODING_IDEAS.map(i => /code|Python|bug|three\\.js|HTML/i.test(i.q + i.label))))')
        self.assertTrue(r and all(r))


if __name__ == "__main__":
    unittest.main()

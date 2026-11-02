"""The thinking budget: GLM's default 8192 (others 32768, a user's value kept) and the dashboard's field (live + saved)."""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path[:0] = [str(ROOT / "serve"), str(ROOT / "tools"), str(ROOT)]
import maya_quants as Q  # noqa: E402
import models as M  # noqa: E402
import server  # noqa: E402

GLM = {"exe": "x/maya", "args": ["--glm-pack", "p"], "installer": {"quant": "Maya-XS-IQ1_S"}}
GLM_S24 = {"exe": "x/maya", "args": ["--glm-pack", "p"], "installer": {"quant": "Maya-S24"}}
DS = {"exe": "x/maya-ds4", "args": ["--pack", "p"]}


class Default(unittest.TestCase):
    def test_only_glm_xs_gets_8192(self):
        self.assertEqual(Q.default_thinking_budget(GLM_S24), 32768)
        self.assertIsNone(Q.thinking_budget_for(GLM_S24))
        for q in ("Maya-S-v2-IQ2_XXS", "Maya-M", "Maya-M-Derisked", "Maya-L", None):
            c = {**GLM, "installer": {"quant": q}}
            self.assertEqual(Q.default_thinking_budget(c), 32768, q)
            self.assertIsNone(Q.thinking_budget_for(c), q)
        self.assertEqual(Q.default_thinking_budget(GLM), 8192)
        self.assertEqual(Q.default_thinking_budget(DS), 32768)
        self.assertEqual(Q.thinking_budget_for(GLM), 8192)                       # none set
        self.assertEqual(Q.thinking_budget_for({**GLM, "thinking_budget": 32768}), 8192)   # the old default
        self.assertIsNone(Q.thinking_budget_for(DS))                             # others untouched
        self.assertIsNone(Q.thinking_budget_for({**DS, "thinking_budget": 32768}))

    def test_user_value_kept(self):
        self.assertIsNone(Q.thinking_budget_for({**GLM, "thinking_budget": 16000}))
        self.assertIsNone(Q.thinking_budget_for({**GLM, "thinking_budget": 0}))
        self.assertIsNone(Q.thinking_budget_for({**GLM, "thinking_budget": 32768,
                                                 "installer": {"quant": "Maya-XS", "thinking_budget_set": True}}))

    def test_switch_writes_it(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "maya-x.json"
            p.write_text(json.dumps({**GLM, **dict(zip(("sampling", "reasoning_effort"), Q.model_defaults(GLM)))}))
            self.assertTrue(M.own_defaults(p))
            self.assertEqual(json.loads(p.read_text())["thinking_budget"], 8192)
            self.assertFalse(M.own_defaults(p))
            cfg = json.loads(p.read_text())
            cfg["thinking_budget"] = 4096
            p.write_text(json.dumps(cfg))
            self.assertFalse(M.own_defaults(p))                                  # the user's stays
            self.assertEqual(json.loads(p.read_text())["thinking_budget"], 4096)


class Field(unittest.TestCase):
    def test_validation(self):
        for ok in (0, 256, 8192, 262144, 8192.0):
            self.assertEqual(Q.check_thinking_budget(ok), int(ok))
        for bad in (1, 255, 262145, -5, 1.5, "8192", True, None):
            with self.assertRaises(ValueError, msg=repr(bad)):
                Q.check_thinking_budget(bad)

    def svc(self, cfg_path, end=5):
        s = object.__new__(server.Service)
        s.think_budget, s.think_end_id, s.config_path = 8192, end, cfg_path
        return s

    def test_apply_live_and_save(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "maya-x.json"
            p.write_text(json.dumps(GLM))
            s = self.svc(str(p))
            self.assertEqual(s.think_budget_info(), {"budget": 8192, "supported": True, "saved": True})
            self.assertEqual(s.set_think_budget(2048)["budget"], 2048)
            self.assertEqual(s.think_budget, 2048)                               # live: read per request
            cfg = json.loads(p.read_text())
            self.assertEqual(cfg["thinking_budget"], 2048)
            self.assertTrue(cfg["installer"]["thinking_budget_set"])
            self.assertIsNone(Q.thinking_budget_for({**GLM, **cfg, "thinking_budget": 32768}))   # a later switch keeps it
            with self.assertRaises(ValueError):
                s.set_think_budget(100)
            self.assertEqual(s.think_budget, 2048)                               # a bad value changes nothing

    def test_unsupported_model(self):
        s = self.svc(None, end=None)
        self.assertFalse(s.think_budget_info()["supported"])
        with self.assertRaises(ValueError):
            s.set_think_budget(1024)


class GlmMayaXS(unittest.TestCase):
    """The catalog entry of GLM-5.3-Flash Maya-XS and the defaults it brings."""

    def test_defaults(self):
        self.assertEqual(Q.model_defaults(GLM)[0], {"temperature": 0.7, "top_p": 0.95, "min_p": 0.05})
        self.assertEqual(Q.model_defaults(GLM_S24)[0], {"temperature": 1.0, "top_p": 0.95})
        self.assertEqual(Q.default_thinking_budget(GLM), 8192)

    def test_entry(self):
        import maya
        m = maya.MODELS["Maya-XS-IQ1_S"]
        self.assertEqual(m["revision"], "6728cf78a5b198a6a6f84b7fb88956a17ec56521")
        self.assertEqual(maya.shard_names(m), ["GLM-5.3-Flash-Maya-XS-IQ1_S-0000%d-of-00002.gguf" % i for i in (1, 2)])
        self.assertEqual(set(m["sha256"]), set(maya.shard_names(m)))
        self.assertEqual((m["kv"], m["context"]), ("int8", 131072))
        self.assertEqual(maya.recommended_context(("download", "Maya-S-v2-IQ2_XXS"), {"gpus": []}), maya.DEFAULT_CONTEXT)
        self.assertEqual(maya.glm_sampling(("download", "Maya-XS-IQ1_S"), "Maya-XS-IQ1_S"),
                         {"temperature": 0.7, "top_p": 0.95, "min_p": 0.05})


if __name__ == "__main__":
    unittest.main()

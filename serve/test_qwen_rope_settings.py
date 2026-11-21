"""Qwen RoPE settings round-trip through the real Models API helpers."""
import copy
import math
import unittest
from . import models

class QwenRopeSettings(unittest.TestCase):
    def setUp(self):
        self.cfg = {"exe": "maya", "args": ["--pack", "/missing/qwen/pack", "--max-context", "262144"],
                    "gpu": [0], "installer": {"family": "qwen"}}

    def test_default_preserves_metadata_selection(self):
        shown = models.settings_of(self.cfg)
        self.assertEqual(shown["rope_scaling"], "auto")
        changed = models.apply(self.cfg, shown)
        self.assertNotIn("--rope-scaling", changed["args"])
        self.assertNotIn("--rope-scale", changed["args"])

    def test_yarn_roundtrip_and_disable(self):
        original = copy.deepcopy(self.cfg)
        result = models.apply(self.cfg, {"context": 524288, "rope_scaling": "yarn", "rope_scale": 2,
                                        "yarn_orig_ctx": 262144, "yarn_beta_fast": 32})
        shown = models.settings_of(result)
        self.assertEqual(shown["rope_scaling"], "yarn")
        self.assertEqual(shown["rope_scale"], 2)
        self.assertEqual(shown["context"], 524288)
        self.assertEqual(self.cfg, original)
        disabled = models.apply(result, {"rope_scaling": "none"})
        self.assertEqual(models.flag(disabled["args"], "--rope-scaling"), "none")
        automatic = models.apply(disabled, {"rope_scaling": "auto", "rope_scale": None})
        self.assertNotIn("--rope-scaling", automatic["args"])
        self.assertNotIn("--rope-scale", automatic["args"])

    def test_unrelated_edit_keeps_advanced_values(self):
        changed = models.apply(self.cfg, {"rope_scaling": "linear", "rope_scale": 3,
                                          "rope_freq_base": 10000000, "yarn_ext_factor": -1})
        again = models.apply(changed, {"reserve_mib": 900})
        for key in ("rope_scaling", *models.QWEN_ROPE_FLAGS):
            self.assertEqual(models.settings_of(changed)[key], models.settings_of(again)[key])

    def test_invalid_values(self):
        for settings in [{"rope_scaling": "bogus"}, {"rope_scale": 0.5}, {"rope_scale": True},
                         {"rope_scale": math.nan}, {"yarn_beta_fast": 0}, {"rope_freq_scale": -1}]:
            with self.subTest(settings=settings), self.assertRaises(ValueError):
                models.apply(self.cfg, settings)

    def test_other_family_does_not_accept_qwen_flags(self):
        cfg = {**self.cfg, "installer": {"family": "deepseek"}}
        with self.assertRaises(ValueError):
            models.apply(cfg, {"rope_scaling": "yarn"})

if __name__ == "__main__":
    unittest.main()

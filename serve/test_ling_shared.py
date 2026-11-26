"""serve/test_ling_shared.py - what maya-ling's changes to the shared Models-page code must NOT change: the settings, GPUs,
trained context, pack folder and written arguments of an Inkling, GLM, DeepSeek, MiniMax, Hy4 and Qwen config.

GOLDEN (ling_shared_golden_v1180.json) holds the values the v1.1.80 release's serve/models.py gives for the same configs (`battery` run on that code
and printed with `python -m serve.test_ling_shared`).  Configs that carry the flags only maya-ling reads (--gguf / -m
for Inkling, --ctx for anyone) are in the battery on purpose: they must behave as before.

    python -m unittest serve.test_ling_shared -v
"""
import json
import re
import struct
import tempfile
import unittest
from pathlib import Path

GOLDEN = json.loads((Path(__file__).with_name("ling_shared_golden_v1180.json")).read_text(encoding="utf-8"))


def _gguf(path: Path, arch: str, ctx: int) -> None:
    def gstr(x: str) -> bytes:
        b = x.encode("utf-8")
        return struct.pack("<Q", len(b)) + b

    hdr = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, 0, 2))
    hdr += gstr("general.architecture") + struct.pack("<I", 8) + gstr(arch)
    hdr += gstr(f"{arch}.context_length") + struct.pack("<I", 4) + struct.pack("<I", ctx)
    path.write_bytes(bytes(hdr))


def _configs(d: Path) -> dict:
    gguf = d / "model" / "m.gguf"
    gguf.parent.mkdir()
    _gguf(gguf, "glm4", 131072)
    pack = d / "dspack"
    pack.mkdir()
    tok = d / "model" / "tokenizer.json"
    tok.write_text("{}", encoding="utf-8")
    g = str(gguf)
    cfgs = {}
    for fam, exe, inst in (("ik2", "maya-ik2", {"family": "ik2"}), ("hy4", "maya-hy4", {"family": "hy4"}),
                           ("minimax", "maya-mm3", {"family": "minimax"}), ("deepseek", "maya-ds4", {"family": "deepseek"}),
                           ("glm", "maya", {}), ("qwen", "maya", {"family": "qwen"})):
        base = {"exe": str(d / exe), "model_name": fam, "tokenizer": str(tok), "gpu": [0, 1], "installer": dict(inst)}
        cfgs[fam + "/plain"] = {**base, "args": ["--model", g, "--max-context", "65536"]}
        cfgs[fam + "/gguf"] = {**base, "args": ["--gguf", g, "--max-context", "65536"]}
        cfgs[fam + "/dash-m"] = {**base, "args": ["-m", g, "--max-context", "65536"]}
        cfgs[fam + "/ctx-only"] = {**base, "args": ["--model", g, "--ctx", "49152"]}
        cfgs[fam + "/ctx-both"] = {**base, "args": ["--model", g, "--max-context", "65536", "--ctx", "49152"]}
        cfgs[fam + "/pack"] = {**base, "args": ["--glm-pack", g, "--max-context", "65536"]}
        cfgs[fam + "/dspack"] = {**base, "args": ["--pack", str(pack), "--max-context", "65536"]}
    return cfgs


def battery(M, d: Path) -> dict:
    """What the Models page and the planner read from each config, paths made relative to `d`."""
    def rel(x):
        return None if x is None else re.sub(r"[\\/]+", "/", str(x).replace(str(d), "<d>"))

    out = {}
    for name, cfg in _configs(d).items():
        row = {"family": M.family(cfg), "gpus": M.gpus_of(cfg), "pack": rel(M.pack_of(cfg)),
               "trained": M.trained_context_of(cfg)}
        for key, fn in (("settings", lambda: M.settings_of(cfg)), ("choices", lambda: M.choices(cfg)),
                        ("applied_args", lambda: [rel(a) for a in M.apply(cfg, M.settings_of(cfg)).get("args") or []])):
            try:
                v = fn()
            except Exception as e:  # noqa: BLE001  (an error is a value too: it must not change either)
                v = "error: " + type(e).__name__
            v = json.loads(json.dumps(v, default=lambda o: rel(o)))
            if key == "choices" and isinstance(v, dict):
                v.pop("ablate", None)      # (its wording is another change, between the PR's base and v1.1.80)
            row[key] = v
        out[name] = row
    return out


class OtherModelsUnchanged(unittest.TestCase):
    def test_same_as_v1180(self):
        from serve import models as M
        self.assertTrue(GOLDEN, "GOLDEN is empty")
        with tempfile.TemporaryDirectory() as td:
            got = battery(M, Path(td).resolve())
        self.assertEqual(sorted(got), sorted(GOLDEN))
        for name in sorted(got):
            if name.startswith("ling/"):
                continue
            with self.subTest(config=name):
                self.assertEqual(got[name], GOLDEN[name])


class ReportLines(unittest.TestCase):
    def test_ling_lines_are_anchored(self):
        from serve.server import REPORT_LINES
        for line in ("ling prompt: 4096 tokens in 1.2 s", "ling stat: decode 20 ms/tok", "ling split: 3 GPUs",
                     "ling vram: 20 GiB", "ling numa: node 0", "ling draft: mtp on", "maya-ling: serving"):
            with self.subTest(line=line):
                self.assertTrue(REPORT_LINES.search(line))
        # lines of other engines or tools that only contain the words must not be picked up
        for line in ("compiling draft model for the kernel", "scaling stat of the table", "sampling profile loaded",
                     "recalling env values", "kibling split 2 ways", "other-maya-ling: x"):
            with self.subTest(line=line):
                self.assertFalse(REPORT_LINES.search(line))


class StartLineSettings(unittest.TestCase):
    def test_env_line_lists_only_variables_that_are_read(self):
        root = Path(__file__).resolve().parents[1]
        main = (root / "src" / "ling" / "ling_main.cu").read_text(encoding="utf-8")
        block = re.search(r"kVars\[\] = \{(.*?)\};", main, re.S)
        self.assertIsNotNone(block)
        names = re.findall(r'"([A-Z][A-Z0-9_]+)"', block.group(1))
        self.assertTrue(names)
        code = "\n".join(p.read_text(encoding="utf-8", errors="replace")
                         for p in sorted((root / "src" / "ling").glob("*")) if p.suffix in (".cu", ".cpp", ".hpp", ".cuh"))
        for name in names:
            with self.subTest(var=name):
                if name == "CUDA_VISIBLE_DEVICES":     # (read by the CUDA runtime: it decides which cards exist)
                    continue
                self.assertRegex(code, r'getenv\(\s*"%s"\s*\)' % name)
        for gone in ("MAYA_LING_KV", "MAYA_LING_DRAFT", "MAYA_ENGINE_QUIT_S", "OMP_NUM_THREADS"):
            self.assertNotIn(gone, names)


class LingAsEveryModel(unittest.TestCase):
    """What a Ling config gets from the server as every model's does: the loop guard on, its own defaults where the
    config names none, its family, one answer at a time."""

    CFG = {"exe": "build/maya-ling", "args": ["--serve", "--gguf", "/m/Ling-3.2-air-Maya-S-IQ2_XXS.gguf",
                                              "--max-context", "32768", "--kv", "fp16", "--drafting", "auto"],
           "installer": {"family": "ling", "ling_quant": "maya-s", "quant": "Ling-3.2-air-Maya-S"}}

    def test_loop_guard_on_by_default(self):
        from serve.server import loop_guard_setting
        self.assertTrue(loop_guard_setting(dict(self.CFG), {}))
        self.assertFalse(loop_guard_setting({**self.CFG, "loop_guard": False}, {}))

    def test_family_defaults_and_label(self):
        from serve import models as M
        from tools import maya_quants as Q
        self.assertEqual(M.family(self.CFG), "ling")
        self.assertEqual(Q.model_defaults(self.CFG), ({"temperature": 1.0, "top_p": 0.95, "top_k": 20}, "high"))
        self.assertEqual(Q.config_label(self.CFG), "Ling-3.2-air · Maya-S")

    def test_thinking_levels_are_the_templates_two(self):
        from serve.frontend import ChatTemplate
        tpl = Path(__file__).resolve().parents[1] / "tools" / "ling" / "chat_template.jinja"
        t = ChatTemplate(tpl)
        self.assertTrue(t.ling)
        self.assertEqual(t.thinking_levels(), ["none", "high"])

    def test_not_taken_for_a_forced_thinking_template(self):
        # Ling's template sets enable_thinking = true only when none is given; it was taken for CYBER-FROST's (thinking
        # always on), so the start line said "medium (the config says high)" and the web app's default was Qwen's word
        from serve.frontend import ChatTemplate
        tpl = Path(__file__).resolve().parents[1] / "tools" / "ling" / "chat_template.jinja"
        t = ChatTemplate(tpl)
        self.assertFalse(t.forced_thinking)
        self.assertFalse(t.qwen)
        self.assertEqual(t.effort_shown("high"), "high")
        self.assertEqual(t.effort_shown("medium"), "high")
        self.assertEqual(t.effort_shown("none"), "none")
        m = [{"role": "user", "content": "hi"}]
        self.assertTrue(t.render(m, add_generation_prompt=True).endswith("<role>ASSISTANT</role>\n<think>"))
        self.assertTrue(t.render(m, add_generation_prompt=True, enable_thinking=False).endswith("\n<think></think>"))
        # (CYBER-FROST's template keeps its flag: tools/test_setup_qwen.py test_thinking_always_on_and_off_form)

    def test_one_answer_at_a_time_is_said(self):
        from serve import models as M
        self.assertIn("parallel", M.LING_NOT_YET)
        self.assertIn("one request at a time", M.LING_NOT_YET["parallel"])


if __name__ == "__main__":
    from serve import models as M
    with tempfile.TemporaryDirectory() as td:
        print(json.dumps(battery(M, Path(td).resolve()), indent=1, sort_keys=True))

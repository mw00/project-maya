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


if __name__ == "__main__":
    from serve import models as M
    with tempfile.TemporaryDirectory() as td:
        print(json.dumps(battery(M, Path(td).resolve()), indent=1, sort_keys=True))

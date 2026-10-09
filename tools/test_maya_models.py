"""tools/test_maya_models.py - the setup's model downloads: what a setup is offered (Project Maya's four quants, the
24 GB recommendation) and that every download names a hash per file.  No GPU, no network."""
from pathlib import Path
import shlex
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya  # noqa: E402


class ModelChoice(unittest.TestCase):
    def choose(self, inst, vram_gb=32.0):
        args = SimpleNamespace(gguf_dir=None, model=None, yes=True)
        lines = []
        with tempfile.TemporaryDirectory() as d, \
                patch.object(maya.S, "gpus", return_value=[{"vram_gb": vram_gb}]), \
                patch.object(maya, "say", side_effect=lambda *a: lines.append(" ".join(map(str, a)))):
            kind, quant = maya.choose_model(args, Path(d), inst)
        return kind, quant, "\n".join(lines)

    def test_new_setup_offers_the_four_maya_quants(self):
        kind, quant, text = self.choose({})
        self.assertEqual((kind, quant), ("download", "Maya-S-v2-IQ2_XXS"))
        for q in ("Maya-S-v2-IQ2_XXS:", "Maya-S24:", "Maya-M:", "Maya-L:"):
            self.assertIn(q, text)

    def test_24gb_cards_get_maya_s24(self):
        self.assertEqual(self.choose({}, vram_gb=24.0)[1], "Maya-S24")

    def test_a_setup_of_a_model_no_longer_offered_gets_the_recommendation(self):
        # a download no longer offered: setting up again recommends a Maya quant (--gguf-dir keeps the files)
        kind, quant, text = self.choose({"quant": "Retired-3.5bit"})
        self.assertEqual(quant, "Maya-S-v2-IQ2_XXS")
        self.assertNotIn("Retired-3.5bit", text)

    def test_an_earlier_maya_download_stays_the_default(self):
        self.assertEqual(self.choose({"quant": "Maya-L"})[1], "Maya-L")

    def test_every_download_names_a_hash_per_shard(self):
        for q, m in maya.MODELS.items():
            names = {m["file"].format(i=i, n=m["shards"]) for i in range(1, m["shards"] + 1)}
            self.assertEqual(set(m["sha256"]), names, q)
            for h in m["sha256"].values():                   # a hash, or several (a header republished: v1.0.28)
                self.assertTrue(all(len(x) == 64 for x in ([h] if isinstance(h, str) else h)), q)

    def test_a_download_matches_any_published_hash(self):
        import hashlib
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.gguf"
            p.write_bytes(b"maya")
            good = hashlib.sha256(b"maya").hexdigest()
            self.assertTrue(maya.sha256_ok(p, good))
            self.assertTrue(maya.sha256_ok(p, ["0" * 64, good.upper()]))
            self.assertFalse(maya.sha256_ok(p, ["0" * 64]))


class RestartAfterUpdate(unittest.TestCase):
    """The dashboard's Update ends the server with UPDATE_EXIT: maya.py starts the new version - the same model and
    settings, no setup flags, no question - and nothing else (no download, no pack)."""

    def start(self, rc, argv=("maya.py", "--setup", "--model", "Maya-L", "--port", "8090")):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            for f in ("strata", "pack/x", "tok/vocab.json"):
                (d / f).parent.mkdir(parents=True, exist_ok=True)
                (d / f).write_text("x")
            cfg = d / "maya-test.json"
            cfg.write_text('{"exe": "%s", "tokenizer": "%s", "args": ["--glm-pack", "%s"]}'
                           % ((d / "strata").as_posix(), (d / "tok").as_posix(), (d / "pack").as_posix()))
            a = SimpleNamespace(port=8090, host=None, api_key=None, gpu=None, gpus="0,1", backend="cuda")
            with patch.object(maya, "refresh_engine") as refresh, patch.object(maya, "say"), \
                    patch.object(maya, "setup_screen", return_value=False),                     patch.object(maya.sys, "argv", list(argv)),                     patch.object(maya.subprocess, "call", return_value=rc) as call,                     patch.object(maya.os, "execv") as execv, patch.object(maya, "WIN", False):
                out = maya.start(cfg, a)
            return out, call, execv, refresh

    def test_update_exit_starts_the_new_version(self):
        _, call, execv, refresh = self.start(maya.UPDATE_EXIT)
        self.assertEqual(call.call_args.kwargs["env"]["MAYA_RESTART_ON_UPDATE"], "1")
        refresh.assert_called_once()                    # (the new maya.py compiles what changed when it starts)
        argv = execv.call_args.args[1]
        self.assertEqual(argv[1:3], [str(maya.HERE / "maya.py"), "--yes"])
        self.assertNotIn("--setup", argv)               # no setup: the same model, nothing downloaded or packed
        self.assertNotIn("--model", argv)
        for flag, v in (("--port", "8090"), ("--gpus", "0,1"), ("--backend", "cuda")):
            self.assertEqual(argv[argv.index(flag) + 1], v)

    def test_other_exits_end_as_before(self):
        out, _, execv, _ = self.start(0)
        self.assertEqual(out, 0)
        execv.assert_not_called()


class RunScript(unittest.TestCase):
    """run-maya-<model>.sh starts its config through maya.py - on the screen in a terminal, as ./maya.sh does - not
    the server alone in the terminal."""

    def test_the_script_starts_its_config_through_maya_py(self):
        with tempfile.TemporaryDirectory() as d, patch.object(maya, "ROOT", Path(d)), patch.object(maya, "WIN", False):
            cfg = Path(d) / "maya-maya-l.json"
            script = maya.write_run_script(cfg, 8091).read_text()
        # (shlex-quoted as the script writes it: a Windows test run's paths have backslashes)
        self.assertIn(shlex.join([str(maya.HERE / "maya.py"), "--config", str(cfg), "--port", "8091"]) + ' "$@"',
                      script)
        self.assertNotIn("server.py", script)

    def test_config_starts_that_one_without_the_question(self):
        with tempfile.TemporaryDirectory() as d:
            mine, other = Path(d) / "maya-maya-l.json", Path(d) / "maya-maya-s.json"
            for p in (mine, other):
                p.write_text("{}")
            with patch.object(maya, "configs", return_value=[other, mine]), patch.object(maya, "say"), \
                    patch.object(maya.sys, "argv", ["maya.py", "--config", str(mine), "--port", "8091"]), \
                    patch.object(maya, "ask", side_effect=AssertionError("no question: --config names it")), \
                    patch.object(maya, "start", return_value=0) as start:
                self.assertEqual(maya.main(), 0)
        self.assertEqual(start.call_args.args[0], mine.resolve())
        self.assertEqual(start.call_args.args[1].port, 8091)


class LabelledNames(unittest.TestCase):
    """Hugging Face groups a repo's files by the quant label in their names: the downloads carry one since v1.0.19,
    and a setup's files under the names before it stay in use (its run config points at them)."""

    def test_new_names_carry_the_label(self):
        self.assertEqual(maya.shard_names(maya.MODELS["Maya-L"])[0], "GLM-5.3-Flash-Maya-L-IQ3_S-00001-of-00004.gguf")
        self.assertEqual(maya.shard_names(maya.MODELS["Maya-M"])[2], "GLM-5.3-Flash-Maya-M-IQ2_S-00003-of-00003.gguf")
        self.assertEqual(maya.shard_names(maya.MODELS["Maya-S24"])[1],
                         "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00002-of-00003.gguf")
        # one label per model: Hugging Face adds up the files that share one (Maya-S is IQ2_XXS)
        labels = [maya.shard_names(m)[0].split("-")[-4] for m in maya.MODELS.values()]
        self.assertEqual(len(labels), len(set(labels)), labels)

    def test_a_fresh_folder_downloads_the_new_names(self):
        m = maya.MODELS["Maya-L"]
        with tempfile.TemporaryDirectory() as d:
            shards = maya.local_shards(m, Path(d))
        self.assertEqual([s.name for s in shards], maya.shard_names(m))

    def test_old_names_stay_and_download_from_the_new(self):
        m = maya.MODELS["Maya-M"]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "GLM-5.3-Flash-Maya-M-00001-of-00003.gguf").write_bytes(b"part")   # an interrupted old download
            shards = maya.local_shards(m, d)
            self.assertEqual([s.name for s in shards], maya.old_names(m)[0])
            self.assertEqual(maya.hf_name(m, shards[1]), "GLM-5.3-Flash-Maya-M-IQ2_S-00002-of-00003.gguf")
            self.assertIn(maya.hf_name(m, shards[1]), m["sha256"])        # checked against the published hash

    def test_a_finished_old_download_is_found(self):
        m = maya.MODELS["Maya-S24"]
        with tempfile.TemporaryDirectory() as d:
            models = Path(d)
            (models / "Maya-S24").mkdir()
            for n in maya.old_names(m)[-1]:                        # the names before any label
                (models / "Maya-S24" / n).write_bytes(b"x")
            self.assertEqual(maya.download_dir(models, "Maya-S24"), models / "Maya-S24")
            self.assertTrue(all(p.exists() for p in maya.local_shards(m, models / "Maya-S24")))

    def test_v1019s_s24_name_is_kept_too(self):
        m = maya.MODELS["Maya-S24"]
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "GLM-5.3-Flash-Maya-S24-IQ2_XXS-00003-of-00003.gguf").write_bytes(b"x")
            shards = maya.local_shards(m, d)
            self.assertEqual(shards[0].name, "GLM-5.3-Flash-Maya-S24-IQ2_XXS-00001-of-00003.gguf")
            self.assertEqual(maya.hf_name(m, shards[0]), "GLM-5.3-Flash-Maya-S24-IQ2_XXS_S-00001-of-00003.gguf")

    def test_every_name_is_the_same_model(self):
        for q in ("Maya-S24", "Maya-M", "Maya-L"):
            m = maya.MODELS[q]
            for names in [maya.shard_names(m)] + maya.old_names(m):
                self.assertEqual(maya.quant_of(Path(names[0])), q)
        self.assertEqual(maya.quant_of(Path("GLM-5.3-Flash-Maya-S-v2-IQ2_XXS-00001-of-00003.gguf")),
                         "Maya-S-v2-IQ2_XXS")


if __name__ == "__main__":
    unittest.main()

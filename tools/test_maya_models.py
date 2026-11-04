"""tools/test_maya_models.py - the setup's model downloads: what a setup is offered (Project Maya's four quants, the
24 GB recommendation, the EXL3 model on NVIDIA only) and that every download names a hash per file.  No GPU, no
network."""
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
        args = SimpleNamespace(gguf_dir=None, model=None, yes=True, backend="cuda")
        lines = []
        with tempfile.TemporaryDirectory() as d, \
                patch.object(maya.S, "gpus", return_value=[{"vram_gb": vram_gb}]), \
                patch.object(maya, "say", side_effect=lambda *a: lines.append(" ".join(map(str, a)))):
            kind, quant = maya.choose_model(args, Path(d), inst)
        return kind, quant, "\n".join(lines)

    def test_new_setup_offers_the_maya_quants(self):
        kind, quant, text = self.choose({})
        self.assertEqual((kind, quant), ("download", "Maya-S-v2-IQ2_XXS"))
        for q in ("GLM-5.3 · Maya-S:", "GLM-5.3 · Maya-S24:", "GLM-5.3 · Maya-M:", "GLM-5.3 · Maya-M-Derisked:",
                  "GLM-5.3 · Maya-L:"):   # (the model first, the tier: maya_quants)
            self.assertIn(q, text)

    def test_maya_m_derisked_downloads_from_its_own_repo_with_mayas_vision_files(self):
        m = maya.MODELS["Maya-M-Derisked"]
        names = maya.shard_names(m)
        self.assertEqual(names[0], "GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-00001-of-00003.gguf")
        self.assertEqual(maya.hf_url(m["repo"], m["revision"], m["folder"], names[0]),
                         "https://huggingface.co/peasantsmith/GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-GGUF/resolve/main/"
                         "GLM-5.3-Flash-Maya-M-Derisked-IQ2_S-00001-of-00003.gguf")
        self.assertEqual(maya.quant_of(Path(names[0])), "Maya-M-Derisked")
        self.assertEqual(maya.quant_of(Path(maya.shard_names(maya.MODELS["Maya-M"])[0])), "Maya-M")
        self.assertEqual(m["vision"]["repo"], maya.MODELS["Maya-M"]["repo"])   # its repo holds no vision files
        self.assertEqual(m["vision"]["sha256"], maya.MODELS["Maya-M"]["vision"]["sha256"])

    def test_24gb_cards_get_maya_s24(self):
        self.assertEqual(self.choose({}, vram_gb=24.0)[1], "Maya-S24")

    def test_a_setup_of_a_model_no_longer_offered_gets_the_recommendation(self):
        # a download no longer offered: setting up again recommends a Maya quant (--gguf-dir keeps the files)
        kind, quant, text = self.choose({"quant": "Retired-3.5bit"})
        self.assertEqual(quant, "Maya-S-v2-IQ2_XXS")
        self.assertNotIn("Retired-3.5bit", text)

    def test_an_earlier_maya_download_stays_the_default(self):
        self.assertEqual(self.choose({"quant": "Maya-L"})[1], "Maya-L")

    def test_downloads_chosen_by_hand_are_not_in_the_list(self):
        """v1.1.34: EXL3-sc_4.10bpw is not in the setup's list (maya.MODELS_BY_HAND) but stays a download: --model takes
        it, and a PC that has it set up lists it as its default (a setup again, --yes included, keeps the model)."""
        self.assertEqual(maya.MODELS_BY_HAND, ("EXL3-sc_4.10bpw",))
        self.assertIn("EXL3-sc_4.10bpw", maya.MODELS)
        kind, quant, text = self.choose({})
        self.assertNotIn("sc_4.10", text)
        self.assertIn("GLM-5.3 · EXL3-4.05bpw: download", text)                # (the other EXL3 quants are still listed)
        self.assertIn("GLM-5.3 · EXL3-3.05bpw: download", text)
        kind, quant, text = self.choose({"quant": "EXL3-sc_4.10bpw"})
        self.assertEqual((kind, quant), ("download", "EXL3-sc_4.10bpw"))
        self.assertIn("GLM-5.3 · EXL3-sc_4.10bpw: download 167.8 GB from Hugging Face   (experimental)", text)
        args = SimpleNamespace(gguf_dir=None, model="EXL3-sc_4.10bpw", yes=True)   # (by hand: no question at all)
        self.assertEqual(maya.choose_model(args, Path("/m"), {}), ("download", "EXL3-sc_4.10bpw"))
        self.assertIsNone(maya.model_option(SimpleNamespace(model="EXL3-sc_4.10bpw", mm3_quant=None)))
        self.assertTrue(maya.is_exl3("EXL3-sc_4.10bpw"))

    def test_every_download_names_a_hash_per_shard(self):
        for q, m in maya.MODELS.items():
            self.assertEqual(set(m["sha256"]), set(maya.shard_names(m)), q)
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


class WholeShards(unittest.TestCase):
    """incomplete(): a shard is whole as far as its own tensor directory says - and a shard of metadata alone
    (unsloth's first: the tokenizer, no tensors) at its header's end, without the alignment padding (#83)."""

    @staticmethod
    def gguf(n_tensors: int, pad: bool) -> bytes:
        import struct

        def s(t):
            b = t.encode()
            return struct.pack("<Q", len(b)) + b
        kv = s("general.architecture") + struct.pack("<I", 8) + s("glm5next")
        kv += s("tokenizer.ggml.model") + struct.pack("<I", 8) + s("gpt2x")      # an odd length: the header unaligned
        tinfo = b"".join(s(f"t{i}") + struct.pack("<I", 1) + struct.pack("<Q", 8) + struct.pack("<IQ", 0, 32 * i)
                         for i in range(n_tensors))
        head = b"GGUF" + struct.pack("<IQQ", 3, n_tensors, 2) + kv + tinfo
        out = head + (bytes(-len(head) % 32) if pad or n_tensors else b"")
        return out + bytes(32 * n_tensors)

    def check(self, data: bytes):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m-00001-of-00002.gguf"
            p.write_bytes(data)
            return maya.incomplete(p)

    def test_a_metadata_shard_without_padding_is_whole(self):
        data = self.gguf(0, pad=False)
        self.assertNotEqual(len(data) % 32, 0)           # (the case: the header ends inside the padding)
        self.assertIsNone(self.check(data))
        self.assertIsNone(self.check(self.gguf(0, pad=True)))

    def test_a_cut_header_or_data_is_short(self):
        self.assertIsNotNone(self.check(self.gguf(0, pad=False)[:-3]))
        whole = self.gguf(3, pad=True)
        self.assertIsNone(self.check(whole))
        self.assertTrue(self.check(whole[:-1]).startswith("short"))


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
                    patch.object(maya, "setup_screen", return_value=False),                     patch.object(maya.sys, "argv", list(argv)),                     patch.object(maya, "run_server", return_value=rc) as call,                     patch.object(maya.os, "execv") as execv, patch.object(maya, "WIN", False):
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

    def test_a_new_port_starts_again_on_the_saved_one(self):   # (the dashboard's Port: RESTART_EXIT)
        _, _, execv, _ = self.start(maya.RESTART_EXIT)
        argv = execv.call_args.args[1]
        self.assertEqual(argv[1:3], [str(maya.HERE / "maya.py"), "--yes"])
        self.assertNotIn("--port", argv)               # (this start's --port 8090 would undo it: the config's applies)
        self.assertEqual(argv[argv.index("--gpus") + 1], "0,1")

    def after(self, note, config=None):
        """after_server for the Models page's restart: the server's note (serve/models.py's SWITCH_FILE) or none."""
        import json
        import time
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            target = d / "maya-gguf-q8-hip.json"
            target.write_text(json.dumps({"backend": "hip", "args": []}))
            if note is not None:
                (d / maya.SWITCH_FILE).write_text(json.dumps({"config": str(target), "time": time.time(), **note}))
            a = SimpleNamespace(port=8090, host=None, api_key=None, gpu=None, gpus=None, backend="cuda", config=config)
            with patch.object(maya, "ROOT", d), patch.object(maya, "say"), patch.object(maya, "WIN", False), \
                    patch.object(maya.os, "execv") as execv:
                maya.after_server(maya.RESTART_EXIT, a)
            left = (d / maya.SWITCH_FILE).exists()
        argv = execv.call_args.args[1]
        return {f: argv[argv.index(f) + 1] for f in ("--config", "--port", "--backend") if f in argv}, target, left

    def test_a_switch_starts_the_chosen_config_on_the_same_port(self):   # (the Models page's Switch)
        got, target, left = self.after({"why": "switch", "port": 8123}, config=Path("/x/maya-maya-m.json"))
        self.assertEqual(got, {"--config": str(target), "--port": "8123", "--backend": "hip"})
        self.assertFalse(left)                          # (read once: a later port change is not a switch)

    def test_an_old_note_is_not_followed(self):
        got, _, left = self.after({"why": "switch", "port": 8123, "time": 0}, config=Path("/x/maya-maya-m.json"))
        self.assertEqual(got, {"--config": "/x/maya-maya-m.json", "--backend": "cuda"})
        self.assertFalse(left)


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
        # one label per model in a repo: Hugging Face adds up the files that share one (Maya-S is IQ2_XXS); a model in
        # a repo of its own may share a label (Maya-M-Derisked's IQ2_S)
        for repo in {m["repo"] for m in maya.MODELS.values()}:
            labels = [maya.shard_names(m)[0].split("-")[-4] for m in maya.MODELS.values()
                      if m["repo"] == repo and "files" not in m]   # (an EXL3 model has no shards)
            self.assertEqual(len(labels), len(set(labels)), (repo, labels))

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


class ZfsArc(unittest.TestCase):
    """ZFS's ARC above its floor counts as free RAM (#89): MemAvailable leaves it out, though it is given back."""

    def arc(self, text):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "arcstats"
            p.write_text(text)
            return maya.zfs_arc_gb(str(p))

    def test_the_arc_above_its_floor(self):
        stats = ("13 1 0x01 123 33456 2734597123 2851726412342\n"
                 "name                            type data\n"
                 "hits                            4    123456\n"
                 "c_min                           4    2147483648\n"
                 "size                            4    57982058496\n")
        self.assertAlmostEqual(self.arc(stats), 52.0, places=3)

    def test_no_zfs_or_an_arc_at_its_floor(self):
        self.assertEqual(maya.zfs_arc_gb("/nonexistent/arcstats"), 0.0)
        self.assertEqual(self.arc("name type data\nc_min 4 4096\nsize 4 1024\n"), 0.0)
        self.assertEqual(self.arc("name type data\nsize 4 1024\n"), 0.0)



class DraftChoice(unittest.TestCase):
    """The setup's drafting question: the MTP block by default, a DFlash2 drafter when chosen."""

    def choose(self, inst, pc=None, draft=None, yes=True):
        args = SimpleNamespace(draft=draft, yes=yes)
        with patch.object(maya, "say"):
            return maya.choose_draft(args, pc or {}, inst)

    def test_mtp_is_the_default(self):
        self.assertEqual(self.choose({}), "mtp")

    def test_the_earlier_answer_is_the_default(self):
        self.assertEqual(self.choose({"draft": "dflash2"}), "dflash2")
        self.assertEqual(self.choose({"draft": "something else"}), "mtp")

    def test_the_option_wins(self):
        self.assertEqual(self.choose({"draft": "mtp"}, draft="dflash2"), "dflash2")

    def test_amd_keeps_mtp_without_asking(self):
        rx = {"arch": "gfx1201", "name": "Radeon RX 9070 XT", "vram_gb": 16}
        halo = {"arch": "gfx1151", "name": "Radeon 8060S", "vram_gb": 108}
        with patch.object(maya, "choose", side_effect=AssertionError("asked")):
            self.assertEqual(self.choose({"draft": "dflash2"}, pc={"backend": "hip"}, yes=False), "mtp")
            self.assertEqual(self.choose({"draft": "dflash2"}, pc={"backend": "hip", "gpus": [rx]}, yes=False), "mtp")
            # a discrete card beside the APU: the window would span two GPUs, not measured on AMD
            self.assertEqual(self.choose({"draft": "dflash2"}, pc={"backend": "hip", "gpus": [rx, halo]}, yes=False),
                             "mtp")
        self.assertEqual(self.choose({}, pc={"backend": "hip", "gpus": [rx]}, draft="dflash2"), "dflash2")

    def test_one_ryzen_ai_max_apu_is_asked(self):
        pc = {"backend": "hip", "archs": ["gfx1151"], "gpus": [{"arch": "gfx1151", "name": "Radeon 8060S"}]}
        self.assertEqual(self.choose({}, pc=pc), "mtp")                    # MTP stays the default
        self.assertEqual(self.choose({"draft": "dflash2"}, pc=pc), "dflash2")
        with patch.object(maya, "ask", return_value="2"):
            self.assertEqual(self.choose({}, pc=pc, yes=False), "dflash2")

    def test_the_mtp_text_says_how_this_pc_drafts(self):
        """One NVIDIA GPU decodes without the MTP block; one Ryzen AI Max APU drafts with it (MAYA_GLM_MTP_DECODE)."""
        def mtp_text(pc):
            shown = []
            with patch.object(maya, "choose", side_effect=lambda q, intro, rows, d, y: shown.extend(rows) or 0):
                maya.choose_draft(SimpleNamespace(draft=None, yes=True), pc, {})
            return shown[0][2]
        apu = mtp_text({"backend": "hip", "gpus": [{"arch": "gfx1151", "name": "Radeon 8060S"}]})
        self.assertNotIn("one GPU decodes without drafting", apu)
        self.assertIn("It drafts on this one GPU", apu)
        self.assertIn("one GPU decodes without drafting", mtp_text({"backend": "cuda", "archs": [86]}))

    def test_gpus_before_sm80_keep_mtp_without_asking(self):
        with patch.object(maya, "choose", side_effect=AssertionError("asked")):
            self.assertEqual(self.choose({"draft": "dflash2"}, pc={"backend": "cuda", "archs": [70, 70]}, yes=False),
                             "mtp")
            self.assertEqual(self.choose({}, pc={"backend": "cuda", "archs": [86, 75]}, yes=False), "mtp")
        self.assertEqual(self.choose({}, pc={"backend": "cuda", "archs": [70]}, draft="dflash2"), "dflash2")
        self.assertEqual(self.choose({"draft": "dflash2"}, pc={"backend": "cuda", "archs": [86, 120]}), "dflash2")

    def test_the_answer_typed(self):
        with patch.object(maya, "ask", return_value="2"):
            self.assertEqual(self.choose({}, yes=False), "dflash2")

    def test_the_drafter_download_is_pinned_and_hashed(self):
        d = maya.DFLASH
        self.assertEqual(maya.hf_url(d["repo"], d["revision"], "", d["file"]),
                         "https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF/resolve/"
                         "f3569acb7f9cee95e66d24442d75536eb00ab842/GLM-5.3-Flash-DFlash2-Q4_K_M.gguf")
        self.assertEqual(len(d["sha256"]), 64)


class Exl3Download(unittest.TestCase):
    """turboderp's EXL3 quant: every file of its branch downloaded and checked by size
    and hash, its config checked instead of a GGUF header, and its pack built by tools/exl3_pack.py."""
    Q, GB, SHARDS, REV = "EXL3-3.05bpw", 125.3, 15, "332ab457b709b7ba30dd9a448be5de03b80a7ac9"
    FILES = 27
    MTP = "mtp.safetensors"                                                 # the MTP block's file (None: in the shards)

    def offered(self, backend):
        args = SimpleNamespace(gguf_dir=None, model=None, yes=True, backend=backend)
        lines = []
        with tempfile.TemporaryDirectory() as d, patch.object(maya.S, "gpus", return_value=[{"vram_gb": 16.0}]), \
                patch.object(maya, "say", side_effect=lambda *a: lines.append(" ".join(map(str, a)))):
            quant = maya.choose_model(args, Path(d), {})[1]
        return quant, "\n".join(lines)

    def test_offered_on_nvidia_as_experimental_never_the_default(self):
        for win in (False, True):   # Linux and Windows alike
            with patch.object(maya, "WIN", win):
                quant, text = self.offered("cuda")
            self.assertIn(f"{self.Q}: download {self.GB} GB from Hugging Face   (experimental)", text)
            self.assertNotEqual(quant, self.Q)

    def test_every_file_of_the_branch_with_size_and_hash(self):
        m = maya.MODELS[self.Q]
        self.assertEqual(maya.shard_names(m), list(m["files"]))
        self.assertEqual(set(m["files"]), set(m["sha256"]))
        if self.MTP:
            self.assertIn(self.MTP, m["files"])                            # the MTP draft block
        else:
            self.assertNotIn("mtp.safetensors", m["files"])
        self.assertEqual(len([n for n in m["files"] if n.startswith("model-")]), self.SHARDS)
        self.assertAlmostEqual(sum(m["files"].values()) / 1e9, m["download_gb"], delta=0.05)
        self.assertEqual(m["revision"], self.REV)                           # the branch's commit, pinned
        self.assertEqual(maya.hf_url(m["repo"], m["revision"], m["folder"], "config.json"),
                         f"https://huggingface.co/turboderp/GLM-5.3-Flash-exl3/resolve/{self.REV}/config.json")
        self.assertEqual(m["vision"]["repo"], "peasantsmith/GLM-5.3-Flash-Maya-GGUF")   # the same pictures files

    def test_a_file_is_whole_at_its_published_size(self):
        m = {"files": {"a.safetensors": 4}, "sha256": {}}
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.safetensors"
            self.assertEqual(maya.file_incomplete(m, p), "missing")
            p.write_bytes(b"ab")
            self.assertEqual(maya.file_incomplete(m, p), "2 of 4 bytes")
            p.write_bytes(b"abcd")
            self.assertIsNone(maya.file_incomplete(m, p))

    def test_the_download_names_the_branch_and_asks(self):
        lines = []
        args = SimpleNamespace(download_model=False)
        with tempfile.TemporaryDirectory() as d, \
                patch.object(maya, "say", side_effect=lambda *a: lines.append(" ".join(map(str, a)))), \
                patch.object(maya, "rotational", return_value=False), \
                patch.object(maya.shutil, "disk_usage", return_value=SimpleNamespace(free=500e9)), \
                patch.object(maya, "ask", return_value="n") as ask:
            m = maya.MODELS[self.Q]
            shards = maya.local_shards(m, Path(d) / self.Q)
            self.assertFalse(maya.offer_download(args, self.Q, Path(d) / self.Q, shards))
        text = "\n".join(lines)
        ask.assert_called_once()
        self.assertIn(f"https://huggingface.co/turboderp/GLM-5.3-Flash-exl3 (branch {self.Q[5:]})", text)
        self.assertIn(f"{self.FILES} still to download", text)
        self.assertIn(f"/resolve/{self.REV}/{self.MTP or 'model.safetensors.index.json'}", text)

    def test_model_step_checks_the_config_not_a_gguf_header(self):
        with tempfile.TemporaryDirectory() as d:
            models = Path(d)
            (models / self.Q).mkdir()
            conf = models / self.Q / "config.json"
            args = SimpleNamespace()
            with patch.object(maya, "file_incomplete", return_value=None), patch.object(maya, "say"), \
                    patch.object(maya.S, "say"), patch.object(maya, "rotational", return_value=False):
                for n in maya.shard_names(maya.MODELS[self.Q]):
                    (models / self.Q / n).write_bytes(b"")
                conf.write_text('{"model_type": "glm5_next", "quantization_config": {"quant_method": "exl3"}}')
                d_, shards, quant = maya.model_step(args, models, ("download", self.Q))
                self.assertEqual((d_, quant, len(shards)), (models / self.Q, self.Q, self.FILES))
                conf.write_text('{"model_type": "llama", "quantization_config": {"quant_method": "exl3"}}')
                with self.assertRaises(SystemExit):
                    maya.model_step(args, models, ("download", self.Q))

    def test_the_pack_is_exl3_packs_and_kept_once_built(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)

            def pack(cmd, **kw):
                self.assertEqual([Path(cmd[1]).name] + cmd[2:], ["exl3_pack.py", "--model", str(d), "--out",
                                                                 str(d / "pack")])
                for f in maya.PACK_FILES + ("exl3.txt",):
                    (d / "pack" / f).parent.mkdir(parents=True, exist_ok=True)
                    (d / "pack" / f).write_text("x")
                (d / "pack" / "native_experts.txt").write_text("# header (absolute offsets in maya-exl3.gguf, or in "
                                                               "the named shard(s) beside it)\n")
                (d / maya.EXL3_GGUF).write_bytes(b"GGUF")

            args = SimpleNamespace(repack=False)
            with patch.object(maya, "run", side_effect=pack) as run, patch.object(maya, "say"), \
                    patch.object(maya.S, "say"):
                self.assertEqual(maya.pack_step(args, d, [], Path("llama"), self.Q), d / "pack")
                self.assertEqual(maya.pack_step(args, d, [], Path("llama"), self.Q), d / "pack")
            run.assert_called_once()                                        # the second time: already built


class Exl3Download405(Exl3Download):
    """The same quant at 4.05 bits per weight: offered, downloaded, checked and packed as the 3.05bpw one."""
    Q, GB, SHARDS, REV = "EXL3-4.05bpw", 165.2, 19, "2a30229e67012798ba9f0cd832bb78abf4c363d5"
    FILES = 30                                                              # (no kpool_aux.safetensors in this branch)

    def test_a_305bpw_download_is_not_taken_for_it(self):
        # the two branches share file names (config.json, tokenizer.json, ...); their shard counts differ
        with tempfile.TemporaryDirectory() as d:
            models = Path(d)
            for n in maya.shard_names(maya.MODELS["EXL3-3.05bpw"]):     # 3.05bpw straight in the models folder
                (models / n).write_bytes(b"")
            self.assertEqual(maya.download_dir(models, "EXL3-3.05bpw"), models)
            self.assertEqual(maya.download_dir(models, self.Q), models / self.Q)


class Exl3DownloadSc410(Exl3Download405):
    """turboderp's self-calibrated quant at 4.10 bits: the same steps; its MTP block is in its shards."""
    Q, GB, SHARDS, REV = "EXL3-sc_4.10bpw", 167.8, 23, "c6b8ac3f6069173343e253e1cba90536ed2c72dc"
    FILES = 33
    MTP = None

    def test_offered_on_nvidia_as_experimental_never_the_default(self):
        # v1.1.34: out of the setup's list (maya.MODELS_BY_HAND) - named by hand it downloads, packs and starts as before
        for win in (False, True):   # Linux and Windows alike
            with patch.object(maya, "WIN", win):
                quant, text = self.offered("cuda")
            self.assertNotIn(self.Q, text)
            self.assertNotEqual(quant, self.Q)

    def test_a_405bpw_download_is_not_taken_for_it(self):
        with tempfile.TemporaryDirectory() as d:
            models = Path(d)
            for n in maya.shard_names(maya.MODELS["EXL3-4.05bpw"]):
                (models / n).write_bytes(b"")
            self.assertEqual(maya.download_dir(models, "EXL3-4.05bpw"), models)
            self.assertEqual(maya.download_dir(models, self.Q), models / self.Q)


if __name__ == "__main__":
    unittest.main()

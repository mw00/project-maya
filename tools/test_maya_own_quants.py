"""tools/test_maya_own_quants.py - models of the user's own, run without being in the setup's list: an EXL3 folder of
GLM-5.3-Flash or MiniMax-M3.1-Flash at any bit rate (--model-dir), and a pruned GLM GGUF (--gguf-dir).

  * --model-dir takes the folder as it is: whole by its own index, nothing of the pinned download lists offered for it,
    its own config named after the folder, the pinned quants' configs and behaviour untouched;
  * the shipped expert profile (288 experts a layer) is left out of a pack of another expert count, at setup and at
    later starts;
  * the config carries the model's OWN sampling (generation_config.json / the GGUF's general.sampling.*), else the base
    model's, and its OWN chat template, else the base model's.

No GPU, no network, no model downloads.

    python -m unittest tools.test_maya_own_quants -v
"""
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "tools")]
import maya  # noqa: E402
from gguf_writer import GGUFWriter  # noqa: E402


def safetensors(path: Path, sizes: dict, cut: int = 0) -> Path:
    """A .safetensors file of zero bytes named tensors (sizes: name -> bytes); `cut` bytes missing at its end."""
    head, off = {}, 0
    for k, n in sizes.items():
        head[k] = {"dtype": "U8", "shape": [n], "data_offsets": [off, off + n]}
        off += n
    h = json.dumps(head).encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes((struct.pack("<Q", len(h)) + h + bytes(off))[:len(h) + 8 + off - cut])
    return path


def exl3_folder(root: Path, name: str, model_type: str, shards: int = 2, gen=None, template=None, tc_template=None,
                dspark: bool = False, cut: int = 0) -> Path:
    """An EXL3 download's files: config, index, shards (the last one `cut` bytes short), tokenizer files."""
    d = root / name
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps({"model_type": model_type,
                                               "quantization_config": {"quant_method": "exl3", "bits": 3.06}}))
    names = ["model-%05d-of-%05d.safetensors" % (i + 1, shards) for i in range(shards)]
    for i, n in enumerate(names):
        safetensors(d / n, {"t%d" % i: 64}, cut if i == shards - 1 else 0)
    (d / "model.safetensors.index.json").write_text(json.dumps(
        {"weight_map": {"t%d" % i: n for i, n in enumerate(names)}}))
    for n in ("tokenizer.json", "special_tokens_map.json"):
        (d / n).write_text("{}")
    (d / "tokenizer_config.json").write_text(json.dumps({"chat_template": tc_template} if tc_template else {}))
    if gen is not None:
        (d / "generation_config.json").write_text(json.dumps(gen))
    if template is not None:
        (d / "chat_template.jinja").write_bytes(template)
    if dspark:
        safetensors(d / "dspark" / "model-dspark-00001.safetensors", {"w": 16})
        (d / "dspark" / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"w": "model-dspark-00001.safetensors"}}))
        (d / "dspark" / "config.json").write_text("{}")
    return d


class Stop(Exception):
    """What the setup's fail() ends with (it exits the program), with its message."""


def quiet(test):
    """say / ok / step / warn do nothing for the test; their lines are kept in test.said; fail() raises Stop."""
    test.said = []
    for name in ("say", "ok", "step", "warn"):
        p = patch.object(maya, name, lambda *a, **k: test.said.append(" ".join(map(str, a))))
        p.start()
        test.addCleanup(p.stop)
    p = patch.object(maya, "fail", lambda msg, hint=None: (_ for _ in ()).throw(Stop(msg)))
    p.start()
    test.addCleanup(p.stop)


def gguf(path: Path, **kv) -> Path:
    w = GGUFWriter()
    w.add("general.architecture", "glm5next")
    for k, (v, t) in kv.items():
        w.add(k.replace("__", "."), v, t)
    w.add_f32("token_embd.weight", np.zeros(4, dtype=np.float32))
    return w.write(path)


class Tmp(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)


# ---------------------------------------------------------------------------------------------------- the profile
class ProfileOfAnotherSize(Tmp):
    """The shipped expert profile ranks the full model's 288 experts a layer; the engine refuses one of another size."""

    def setUp(self):
        super().setUp()
        quiet(self)
        (self.root / "data").mkdir()
        self.shipped = self.root / "data" / "expert-profile-glm.bin"
        self.shipped.write_bytes(struct.pack("<4sIIIII", b"STRP", 1, 46, 288, 0, 0))
        for p in (patch.object(maya, "ROOT", self.root), patch.object(maya, "EXE", self.root / "build/maya"),
                  patch.object(maya, "PROFILE", self.shipped), patch.object(maya, "CAL_STORE", self.root / "c.json"),
                  patch.object(maya, "WIN", False), patch.object(maya.S, "gpus", return_value=[]),
                  patch.object(maya.S, "cpu_info", return_value=("test CPU", True, True)),
                  patch.object(maya.S, "ram_gb", return_value=177), patch.object(maya, "mem_gb", return_value=(177.0, 150.0))):
            p.start()
            self.addCleanup(p.stop)
        self.a = SimpleNamespace(env=[], port=None, host=None, api_key=None, gguf_dir=None)

    def pack(self, experts, name="pack"):
        pk = self.root / "m" / name
        pk.mkdir(parents=True)
        if experts is not None:
            (pk / "native_experts.txt").write_text(
                "# strata native experts v3: layer gu_type d_type offset blob_bytes gate_off up_off down_off "
                "[shard | gate_shard up_shard down_shard] (n_expert %d, total 0; absolute offsets in x.gguf, or in the "
                "named shard(s) beside it)\n1 0 0 0 1 0 0 0\n" % experts)
        return pk

    def write(self, pk):
        first = self.root / "m" / "GLM-5.3-Flash-test.gguf"
        first.parent.mkdir(exist_ok=True)
        gguf(first)
        pc = {"gpus": [{"index": 0, "name": "GPU", "vram_gb": 24}]}
        p = maya.write_config(self.a, pc, {}, pk, "test", 32768, self.root / "models", None, ("local", first))
        return json.loads(p.read_text())["args"]

    def test_the_helpers_read_the_headers(self):
        self.assertEqual(maya.pack_expert_count(self.pack(152)), 152)
        self.assertIsNone(maya.pack_expert_count(self.pack(None, "none")))
        self.assertEqual(maya.profile_experts(self.shipped), 288)
        self.shipped.write_bytes(b"STRP")
        self.assertIsNone(maya.profile_experts(self.shipped))             # (a header too short to say)

    def test_288_experts_get_the_profile_as_before(self):
        self.assertEqual(maya.flag_value(self.write(self.pack(288)), "--expert-profile"), str(self.shipped))

    def test_a_pack_that_does_not_say_gets_it_too(self):
        self.assertEqual(maya.flag_value(self.write(self.pack(None)), "--expert-profile"), str(self.shipped))

    def test_a_pruned_model_gets_none_at_setup(self):
        args = self.write(self.pack(152))
        self.assertNotIn("--expert-profile", args)
        self.assertIn("--glm-pack", args)

    def test_nor_at_a_later_start(self):
        pruned = str(self.pack(152))
        cfg = {"args": ["--glm-pack", pruned, "--max-context", "32768"]}
        self.assertFalse(maya.add_profile(cfg))
        self.assertNotIn("--expert-profile", cfg["args"])
        full = {"args": ["--glm-pack", str(self.pack(288, "p288")), "--max-context", "32768"]}
        self.assertTrue(maya.add_profile(full))                            # (the 288 path as before)
        self.assertEqual(maya.flag_value(full["args"], "--expert-profile"), str(self.shipped))
        mine = {"args": ["--glm-pack", pruned, "--expert-profile", "/learned.bin"]}
        self.assertFalse(maya.add_profile(mine))                           # (a profile of its own stays)

    def test_a_real_pack_of_a_pruned_model(self):
        """The packer's own output for a 12-expert synthetic GLM (tools/glm_synth_gguf.py): the header says 12, and the
        setup leaves the 288-expert profile out."""
        try:
            import gguf as gguf_py
        except ImportError:
            self.skipTest("llama.cpp's gguf-py (pip: gguf) is not installed")
        src = self.root / "synth" / "GLM-5.3-Flash-synth12.gguf"
        src.parent.mkdir()
        env = dict(os.environ, STRATA_GGUF_PY=str(Path(gguf_py.__file__).resolve().parents[1]))
        subprocess.run([sys.executable, str(ROOT / "tools" / "glm_synth_gguf.py"), "--out", str(src), "--vocab"],
                       check=True, capture_output=True, env=env)
        pk = src.parent / "pack"
        subprocess.run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(src), "--out", str(pk),
                        "--compat-bf16"], check=True, capture_output=True, env=env)
        self.assertEqual(maya.pack_expert_count(pk), 12)
        self.assertNotIn("--expert-profile", self.write(pk))


# ---------------------------------------------------------------------------------------------------- the sampling
class OwnSampling(Tmp):
    BASE = {"temperature": 1.0, "top_p": 0.95}

    def test_the_models_own_values_else_the_base_models(self):
        self.assertEqual(maya.own_sampling({"temperature": 0.6, "top_p": 0.9, "top_k": 20}, self.BASE),
                         {"temperature": 0.6, "top_p": 0.9, "top_k": 20})
        self.assertEqual(maya.own_sampling({}, self.BASE), self.BASE)
        self.assertEqual(maya.own_sampling(None, self.BASE), self.BASE)
        self.assertIsNot(maya.own_sampling({}, self.BASE), self.BASE)       # (a copy)

    def test_a_value_the_server_would_refuse_is_left_out(self):
        got = maya.own_sampling({"temperature": 0.7, "top_k": 100, "top_p": 0, "min_p": 1.5, "seed": 3,
                                 "repetition_penalty": 0, "presence_penalty": True}, self.BASE)
        self.assertEqual(got, {"temperature": 0.7})                         # (top_k 100 > 64, ...: the server refuses)
        self.assertEqual(maya.own_sampling({"top_k": 0}, self.BASE), self.BASE)   # (HF's 0 = off: nothing of its own)

    def test_float32_values_are_rounded(self):
        got = maya.own_sampling({"top_p": 0.949999988079071, "temperature": 1.0, "top_k": 40.0}, self.BASE)
        self.assertEqual(got, {"top_p": 0.95, "temperature": 1.0, "top_k": 40})
        self.assertIsInstance(got["top_k"], int)

    def test_an_exl3_folders_generation_config(self):
        d = self.root / "f"
        d.mkdir()
        gen = {"_from_model_config": True, "eos_token_id": [1, 2], "temperature": 0.8, "top_p": 0.9, "top_k": 40,
               "repetition_penalty": 1.05, "do_sample": True}
        (d / "generation_config.json").write_text(json.dumps(gen))
        self.assertEqual(maya.folder_sampling(d, self.BASE),
                         {"temperature": 0.8, "top_p": 0.9, "top_k": 40, "repetition_penalty": 1.05})
        (d / "generation_config.json").write_text(json.dumps({"do_sample": False, "temperature": 0.1}))
        self.assertEqual(maya.folder_sampling(d, self.BASE), self.BASE)     # (it says it does not sample)
        (d / "generation_config.json").write_text(json.dumps({"eos_token_id": 2}))
        self.assertEqual(maya.folder_sampling(d, self.BASE), self.BASE)     # (none of its own)
        (d / "generation_config.json").write_text("not json")
        self.assertEqual(maya.folder_sampling(d, self.BASE), self.BASE)
        (d / "generation_config.json").unlink()
        self.assertEqual(maya.folder_sampling(d, self.BASE), self.BASE)

    def test_a_ggufs_sampling_keys(self):
        g = gguf(self.root / "a.gguf", general__sampling__temp=(0.7, "f32"), general__sampling__top_p=(0.8, "f32"),
                 general__sampling__top_k=(20, "u32"), general__sampling__penalty_repeat=(1.1, "f32"))
        got = maya.gguf_sampling(g, self.BASE)
        self.assertEqual((got["temperature"], got["top_p"], got["top_k"], got["repetition_penalty"]),
                         (0.7, 0.8, 20, 1.1))
        self.assertEqual(maya.gguf_sampling(gguf(self.root / "b.gguf"), self.BASE), self.BASE)    # (none: the base's)
        (self.root / "c.gguf").write_bytes(b"GGUF")
        self.assertEqual(maya.gguf_sampling(self.root / "c.gguf", self.BASE), self.BASE)           # (unreadable)

    def test_which_models_the_ggufs_values_reach(self):
        g = gguf(self.root / "a.gguf", general__sampling__temp=(0.3, "f32"))
        d = self.root / "f"
        d.mkdir()
        (d / "generation_config.json").write_text(json.dumps({"temperature": 0.4}))
        listed = next(iter(maya.MODELS))
        self.assertEqual(maya.glm_sampling(("exl3", d), d.name), {"temperature": 0.4})
        self.assertEqual(maya.glm_sampling(("local", g), "somebody-elses"), {"temperature": 0.3})
        self.assertEqual(maya.glm_sampling(("local", g), listed), maya.SAMPLING)       # (a listed quant: as before)
        self.assertEqual(maya.glm_sampling(("download", listed), listed), maya.SAMPLING)
        self.assertEqual(maya.glm_sampling(None, "x"), maya.SAMPLING)


# ---------------------------------------------------------------------------------------------------- GLM EXL3 folder
class GlmExl3Folder(Tmp):
    def setUp(self):
        super().setUp()
        quiet(self)
        p = patch.object(maya, "offer_download", side_effect=AssertionError("nothing of the pinned list is offered"))
        p.start()
        self.addCleanup(p.stop)
        p = patch.object(maya, "rotational", return_value=False)
        p.start()
        self.addCleanup(p.stop)
        self.a = SimpleNamespace(model=None, gguf_dir=None, draft=None, model_dir=None, repack=False)

    def test_the_family_comes_from_the_folders_config(self):
        glm = exl3_folder(self.root, "glm", "glm5_next")
        mm = exl3_folder(self.root, "mm", "minimax_m3_vl")
        other = self.root / "ds"
        other.mkdir()
        (other / "config.json").write_text(json.dumps({"model_type": "deepseek_v4", "quantization_config": None}))
        self.assertEqual((maya.exl3_folder_family(glm), maya.exl3_folder_family(mm)), ("glm", "minimax"))
        self.assertIsNone(maya.exl3_folder_family(other))
        self.assertIsNone(maya.exl3_folder_family(self.root / "nowhere"))
        self.a.model_dir = str(glm)
        self.assertEqual(maya.choose_family(self.a, {}), "glm")
        self.a.model_dir = str(mm)
        self.assertEqual(maya.choose_family(self.a, {}), "minimax")
        self.a.model_dir = str(other)
        self.assertEqual(maya.choose_family(self.a, {}), "deepseek")        # (DeepSeek's --model-dir as before)

    def test_choose_model_takes_the_folder(self):
        glm = exl3_folder(self.root, "GLM-5.3-Flash-exl3-sc_2.10bpw", "glm5_next")
        self.a.model_dir = str(glm)
        self.assertEqual(maya.choose_model(self.a, self.root, {}), ("exl3", glm.resolve()))

    def test_model_step_checks_the_folders_own_index(self):
        glm = exl3_folder(self.root, "GLM-5.3-Flash-exl3-sc_2.10bpw", "glm5_next", shards=3)
        d, shards, quant = maya.model_step(self.a, self.root, ("exl3", glm))
        self.assertEqual((d, quant, [s.name for s in shards]), (glm, glm.name, [
            "model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"]))
        self.assertNotIn(quant, maya.MODELS)

    def test_a_short_shard_or_a_missing_file_stops_the_setup(self):
        short = exl3_folder(self.root, "short", "glm5_next", cut=10)
        with self.assertRaisesRegex(Stop, "model-00002-of-00002.safetensors: short"):
            maya.model_step(self.a, self.root, ("exl3", short))
        whole = exl3_folder(self.root, "whole", "glm5_next")
        (whole / "model-00002-of-00002.safetensors").unlink()
        with self.assertRaisesRegex(Stop, "model-00002-of-00002.safetensors: missing"):
            maya.model_step(self.a, self.root, ("exl3", whole))
        no_tok = exl3_folder(self.root, "notok", "glm5_next")
        (no_tok / "tokenizer.json").unlink()
        with self.assertRaisesRegex(Stop, "tokenizer.json is missing"):
            maya.model_step(self.a, self.root, ("exl3", no_tok))

    def test_a_minimax_folder_is_no_glm_model(self):
        mm = exl3_folder(self.root, "mm", "minimax_m3_vl")
        with self.assertRaisesRegex(Stop, "not GLM-5.3-Flash"):
            maya.model_step(self.a, self.root, ("exl3", mm))

    def test_extra_safetensors_beside_the_index_count_too(self):
        glm = exl3_folder(self.root, "g", "glm5_next")
        safetensors(glm / "mtp.safetensors", {"x": 8}, cut=1)
        with self.assertRaisesRegex(Stop, "mtp.safetensors: short"):
            maya.exl3_folder_files(glm)

    def test_pack_step_puts_the_folders_own_template(self):
        for kw, expect in (({"template": b"FOLDER FILE"}, b"FOLDER FILE"), ({"tc_template": "FROM TC"}, b"FROM TC"),
                           ({"template": b"FILE", "tc_template": "TC"}, b"FILE")):
            glm = exl3_folder(self.root, "g%d" % len(expect), "glm5_next", **kw)
            pack = glm / "pack"
            (pack / "tokenizer").mkdir(parents=True)
            (pack / "tokenizer" / "chat_template.jinja").write_bytes(b"MAYA'S")
            shards = maya.exl3_folder_files(glm)
            with patch.object(maya, "exl3_pack_step", return_value=pack) as step:
                self.assertEqual(maya.pack_step(self.a, glm, shards, self.root, glm.name), pack)
            step.assert_called_once()
            self.assertEqual((pack / "tokenizer" / "chat_template.jinja").read_bytes(), expect)

    def test_a_folder_without_one_gets_glm5s(self):
        glm = exl3_folder(self.root, "g", "glm5_next")
        pack = glm / "pack"
        (pack / "tokenizer").mkdir(parents=True)
        base = ROOT / "serve" / "glm5_chat_template.jinja"
        with patch.object(maya, "ROOT", ROOT), patch.object(maya, "exl3_pack_step", return_value=pack):
            maya.pack_step(self.a, glm, maya.exl3_folder_files(glm), self.root, glm.name)
        self.assertEqual((pack / "tokenizer" / "chat_template.jinja").read_bytes(), base.read_bytes())

    def test_a_pinned_exl3_download_is_packed_as_before(self):
        pinned = next(q for q in maya.MODELS if maya.is_exl3(q))
        with patch.object(maya, "exl3_pack_step", return_value=self.root) as step, \
                patch.object(maya, "put_template") as put:
            maya.pack_step(self.a, self.root, [self.root / "config.json"], self.root, pinned)
        step.assert_called_once()
        put.assert_not_called()                                              # (Maya's GLM template, as before)

    def test_a_gguf_without_a_template_gets_glm5s_a_gguf_with_one_keeps_it(self):
        pack = self.root / "pack"
        (pack / "tokenizer").mkdir(parents=True)
        with patch.object(maya, "ROOT", ROOT):
            maya.base_template(pack)
            self.assertEqual((pack / "tokenizer" / "chat_template.jinja").read_bytes(),
                             (ROOT / "serve" / "glm5_chat_template.jinja").read_bytes())
            (pack / "tokenizer" / "chat_template.jinja").write_bytes(b"THE GGUF'S")
            maya.base_template(pack)
        self.assertEqual((pack / "tokenizer" / "chat_template.jinja").read_bytes(), b"THE GGUF'S")

    def test_the_config_of_a_glm_folder(self):
        glm = exl3_folder(self.root, "GLM-5.3-Flash-exl3-sc_2.10bpw", "glm5_next",
                          gen={"temperature": 0.9, "top_p": 0.85, "top_k": 30})
        (self.root / "data").mkdir()
        pack = glm / "pack"
        pack.mkdir()
        (pack / "native_experts.txt").write_text("# x (n_expert 288, total 0; absolute offsets in y)\n")
        for p in (patch.object(maya, "ROOT", self.root), patch.object(maya, "EXE", self.root / "build/maya"),
                  patch.object(maya, "PROFILE", self.root / "data" / "none.bin"), patch.object(maya, "WIN", False),
                  patch.object(maya, "CAL_STORE", self.root / "c.json"), patch.object(maya.S, "gpus", return_value=[]),
                  patch.object(maya.S, "cpu_info", return_value=("t", True, True)),
                  patch.object(maya.S, "ram_gb", return_value=177), patch.object(maya, "mem_gb", return_value=(177.0, 150.0))):
            p.start()
            self.addCleanup(p.stop)
        a = SimpleNamespace(env=[], port=None, host=None, api_key=None, gguf_dir=None, model_dir=str(glm))
        pc = {"gpus": [{"index": 0, "name": "GPU", "vram_gb": 24}]}
        path = maya.write_config(a, pc, {}, pack, glm.name, 32768, self.root / "models", None, ("exl3", glm))
        cfg = json.loads(path.read_text())
        self.assertEqual(path.name, "maya-" + glm.name.lower() + ".json")
        self.assertEqual(cfg["sampling"], {"temperature": 0.9, "top_p": 0.85, "top_k": 30})
        self.assertEqual(cfg["reasoning_effort"], maya.EFFORT)               # (the folder names none: GLM's)
        self.assertEqual(cfg["installer"]["model_dir"], str(glm))
        self.assertEqual(maya.flag_value(cfg["args"], "--glm-pack"), str(pack))


# ---------------------------------------------------------------------------------------------------- MiniMax EXL3 folder
def pinned_index(path: Path) -> None:
    """model.safetensors.index.json as the pinned download has it: its shard names, its size."""
    man = maya.mm3_models()[0]["files"]
    names = [n for n in man if n.startswith("model-0") and n.endswith(".safetensors")]
    text = json.dumps({"weight_map": {"t%d" % i: n for i, n in enumerate(names)}})
    path.write_text(text + " " * (man["model.safetensors.index.json"] - len(text)))


class MiniMaxExl3Folder(Tmp):
    def setUp(self):
        super().setUp()
        quiet(self)
        for name, rv in (("offer_download", None), ("rotational", False)):
            p = patch.object(maya, name, side_effect=AssertionError("nothing is offered") if rv is None else None,
                             return_value=rv)
            p.start()
            self.addCleanup(p.stop)
        self.a = SimpleNamespace(model=maya.MM3, model_dir=None, mm3_quant=None, gguf_dir=None, draft=None,
                                 backend="cuda", yes=True, no_vision=False, port=None, host=None, api_key=None,
                                 env=[], models_dir=None)

    def own(self, name="MiniMax-M3.1-Flash-EXL3-3.06bpw", **kw):
        kw.setdefault("gen", {"bos_token_id": 1, "eos_token_id": 2})
        d = exl3_folder(self.root, name, "minimax_m3_vl", shards=20 if kw.pop("many", False) else 2, **kw)
        self.a.model_dir = str(d)
        return d

    def test_another_quant_is_the_users_own(self):
        d = self.own(many=True)
        self.assertEqual(maya.mm3_own_dir(self.a, "exl3"), d.resolve())
        self.assertEqual(maya.choose_mm3_quant(self.a, {}), "exl3")         # (no question asked)
        self.assertIsNone(maya.mm3_own_dir(self.a, "gguf-xs"))

    def test_the_pinned_download_is_not(self):
        d = self.own(name="pinned")
        pinned_index(d / "model.safetensors.index.json")                     # its names and its size
        self.assertIsNone(maya.mm3_own_dir(self.a, "exl3"))
        # the same shard names with another index (another quant of 25 shards) is the user's own
        text = (d / "model.safetensors.index.json").read_text().rstrip()
        (d / "model.safetensors.index.json").write_text(text)
        self.assertEqual(maya.mm3_own_dir(self.a, "exl3"), d.resolve())

    def test_a_folder_without_an_index_or_with_no_family_is_left_to_the_pinned_flow(self):
        d = self.own(name="half")
        (d / "model.safetensors.index.json").unlink()
        self.assertIsNone(maya.mm3_own_dir(self.a, "exl3"))
        self.a.model_dir = None
        self.assertIsNone(maya.mm3_own_dir(self.a, "exl3"))
        self.a.model_dir = str(self.root / "nowhere")
        self.assertIsNone(maya.mm3_own_dir(self.a, "exl3"))

    def test_the_files_are_the_folders_own_and_nothing_is_downloaded(self):
        d = self.own(many=True, dspark=True)
        safetensors(d / "model-vision-00001-of-00001.safetensors", {"v": 8})
        (d / "preprocessor_config.json").write_text("{}")
        files = maya.mm3_own_step(d)
        names = sorted(f.name for f in files)
        self.assertEqual(len(files), 20 + 1 + 1)                              # shards, vision, dspark
        self.assertIn("model-vision-00001-of-00001.safetensors", names)
        self.assertIn("model-dspark-00001.safetensors", names)

    def test_a_short_shard_stops_it_and_a_folder_without_dspark_is_said(self):
        d = self.own(cut=5)
        with self.assertRaisesRegex(Stop, "short"):
            maya.mm3_own_step(d)
        d2 = self.own(name="ok")
        maya.mm3_own_step(d2)
        self.assertTrue(any("dspark" in s for s in self.said))

    def test_memory_follows_the_folders_size(self):
        d = self.own(many=True, dspark=True)
        small = maya.mm3_memory_gb("exl3", d)
        self.assertLess(small, maya.MM3_MEMORY_GB)                             # (a few bytes: the headroom only)
        self.assertEqual(maya.mm3_memory_gb("exl3"), maya.MM3_MEMORY_GB)       # (the pinned quant as before)

    def test_a_later_setup_does_not_find_the_pinned_files_in_the_folder_of_your_own(self):
        d = self.own()
        prev = {"args": ["--pack", str(d / "maya-mm3")], "installer": {"mm3_quant": "exl3", "own": True}}
        self.a.model_dir = None
        got = maya.mm3_dir(self.a, self.root / "models", prev, "exl3")
        self.assertEqual(got, self.root / "models" / maya.MM3_DIR)
        prev["installer"].pop("own")                                           # (the pinned setup before: as it was)
        self.assertEqual(maya.mm3_dir(self.a, self.root / "models", prev, "exl3"), d)

    def test_the_config_has_the_folders_sampling_and_a_name_of_its_own(self):
        d = self.own(gen={"bos_token_id": 1, "do_sample": True, "temperature": 0.5, "top_p": 0.8})
        pack = d / "maya-mm3"
        pack.mkdir()
        (pack / "mm3.txt").write_text("hidden_size 6144\n")
        pinned = self.root / "maya-mm3.json"
        pinned.write_text(json.dumps({"args": ["--pack", "/pinned"], "sampling": {"temperature": 1.0},
                                      "installer": {"mm3_quant": "exl3"}}))
        before = pinned.read_text()
        pc = {"gpus": [{"index": 0}, {"index": 1}]}
        with patch.object(maya, "ROOT", self.root):
            path = maya.write_config_mm3(self.a, pc, {"lib_dirs": []}, pack, 65536, self.root / "models", d, "exl3", True)
            cfg = json.loads(path.read_text())
            self.assertEqual(path.name, "maya-minimax-m3.1-flash-exl3-3.06bpw.json")
            self.assertEqual(cfg["sampling"], {"temperature": 0.5, "top_p": 0.8})
            self.assertEqual(cfg["reasoning_effort"], maya.MM3_EFFORT)
            self.assertEqual(maya.flag_value(cfg["args"], "--pack"), str(pack))
            self.assertEqual((cfg["installer"]["mm3_quant"], cfg["installer"]["own"], cfg["installer"]["model_dir"]),
                             ("exl3", True, str(d)))
            self.assertEqual(pinned.read_text(), before)                      # (the pinned quant's config is not touched)
            cfg["sampling"] = {"temperature": 0.2}                             # (edited by hand: a setup again writes the
            path.write_text(json.dumps(cfg))                                   #  model's own, as for every model)
            again = json.loads(maya.write_config_mm3(self.a, pc, {"lib_dirs": []}, pack, 65536, self.root / "models",
                                                     d, "exl3", True).read_text())
            self.assertEqual(again["sampling"], {"temperature": 0.5, "top_p": 0.8})

    def test_a_folder_with_no_sampling_of_its_own_gets_minimaxs(self):
        d = self.own(gen={"bos_token_id": 1, "eos_token_id": 2})
        pack = d / "maya-mm3"
        pack.mkdir()
        (pack / "mm3.txt").write_text("hidden_size 6144\n")
        with patch.object(maya, "ROOT", self.root):
            cfg = json.loads(maya.write_config_mm3(self.a, {"gpus": [{"index": 0}]}, {"lib_dirs": []}, pack, 65536,
                                                   self.root / "models", d, "exl3", True).read_text())
        self.assertEqual(cfg["sampling"], maya.MM3_SAMPLING)

    def test_the_pinned_quants_config_is_written_as_before(self):
        pack = self.root / "MM" / "maya-mm3"
        pack.mkdir(parents=True)
        (pack / "mm3.txt").write_text("hidden_size 6144\n")
        with patch.object(maya, "ROOT", self.root):
            path = maya.write_config_mm3(self.a, {"gpus": [{"index": 0}]}, {"lib_dirs": []}, pack, 65536,
                                         self.root / "models", pack.parent)
        cfg = json.loads(path.read_text())
        self.assertEqual(path.name, "maya-mm3.json")
        self.assertEqual(cfg["sampling"], maya.MM3_SAMPLING)
        self.assertNotIn("own", cfg["installer"])


class MiniMaxTemplate(Tmp):
    def test_the_folders_own_template_or_the_packs_own_copy_of_minimaxs(self):
        d = exl3_folder(self.root, "a", "minimax_m3_vl", template=b"MINIMAX FOLDER")
        body, label = maya.own_template(d)
        self.assertEqual(body, b"MINIMAX FOLDER")
        self.assertIn("folder's own", label)
        none = exl3_folder(self.root, "b", "minimax_m3_vl")
        self.assertEqual(maya.own_template(none)[0], None)                     # (nothing to put: the pack's copy stays)
        pack = self.root / "pack"
        (pack / "tokenizer").mkdir(parents=True)
        (pack / "tokenizer" / "chat_template.jinja").write_bytes(b"MAYA'S MINIMAX")
        self.assertFalse(maya.put_template(pack, None))
        self.assertEqual((pack / "tokenizer" / "chat_template.jinja").read_bytes(), b"MAYA'S MINIMAX")
        self.assertTrue(maya.put_template(pack, b"MINIMAX FOLDER"))
        self.assertFalse(maya.put_template(pack, b"MINIMAX FOLDER"))            # (already there)


if __name__ == "__main__":
    unittest.main()

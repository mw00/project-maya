"""tools/test_maya_glm_dsa.py - the glm-dsa port B1.7: the installer (maya.py) accepts glm-dsa (GLM-5.3 744B-A40B)
(serve/glm_dsa_product.py), and leaves GLM-5.3-Flash exactly as it was.

The architecture allow-list, the quant label of unsloth's (GLM-5.3-UD-IQ2_M-00001-of-00006.gguf) and SixVolts' file
names, a split model whose first shard holds metadata only, the refusal of other architectures, the pack step's
command, and the vision step skipped for glm-dsa (the model has no vision tower).  No GPU, no network.
"""
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import maya  # noqa: E402

try:
    import gguf
except ImportError:
    from _paths import add_gguf_py
    add_gguf_py()
    import gguf
from gguf import GGUFWriter  # noqa: E402
import numpy as np  # noqa: E402


def write_split(d: Path, stem: str, arch: str = "glm-dsa", n: int = 2) -> list:
    """A split GGUF: shard 1 metadata only (unsloth's layout, #83), the rest one small tensor each."""
    paths = [d / f"{stem}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]
    for i, p in enumerate(paths):
        w = GGUFWriter(p, arch)
        if i == 0:
            w.add_uint32(f"{arch}.block_count", 79)
        else:
            w.add_tensor(f"blk.{i}.attn_norm.weight", np.ones(32, np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    return paths


class Quiet:
    """maya's ok/warn/step/fail print through setup.say."""

    def __enter__(self):
        self.lines = []
        grab = lambda *a: self.lines.append(" ".join(map(str, a)))  # noqa: E731
        self.p = [patch.object(maya.S, "say", side_effect=grab), patch.object(maya, "say", side_effect=grab)]
        for p in self.p:
            p.start()
        return self

    def __exit__(self, *exc):
        for p in self.p:
            p.stop()

    @property
    def text(self):
        return "\n".join(self.lines)


class GlmDsaInstaller(unittest.TestCase):
    def test_arch_allowed(self):
        self.assertIn("glm-dsa", maya.glm_archs())
        self.assertEqual(maya.GLM_ARCHS, ("glm5-next", "glm5next"))         # Maya's tuple untouched

    def test_quant_of_unsloth_and_sixvolts_names(self):
        q = maya.quant_of
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            for stem, want in (("GLM-5.3-UD-IQ2_M", "GLM-5.3-UD-IQ2_M"), ("GLM-5.3-UD-Q2_K_XL", "GLM-5.3-UD-Q2_K_XL"),
                               ("GLM-5.3-ewaste-edition-Q2_K_XL", "GLM-5.3-ewaste-edition-Q2_K_XL")):
                self.assertEqual(q(write_split(d, stem, n=2)[0]), want)
            # a GLM-5.3-like name that is not a glm-dsa file: Maya's fallback (the folder)
            self.assertEqual(q(Path("/m/x/GLM-5.3-Q4_K_M.gguf")), "x")
        # the Flash names keep Maya's labels
        self.assertEqual(q(Path("/m/x/GLM-5.3-Flash-UD-IQ1_S-00001-of-00003.gguf")), "UD-IQ1_S")
        self.assertEqual(q(Path("/m/x/" + maya.shard_names(maya.MODELS["Maya-L"])[0])), "Maya-L")
        # anything else: the folder's name, as before
        self.assertEqual(q(Path("/m/UD-IQ2_M/model-00001-of-00006.gguf")), "UD-IQ2_M")

    def test_metadata_only_first_shard_is_found_and_whole(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            paths = write_split(d, "GLM-5.3-UD-IQ2_M", n=3)
            self.assertEqual(maya.gguf_head(paths[0]), ("glm-dsa", 0))
            self.assertEqual(maya.glm_files(d), [paths[0]])
            self.assertEqual(maya.local_first(d), (paths[0].resolve(), None))
            self.assertEqual(maya.local_first(paths[2]), (paths[0].resolve(), None))
            self.assertEqual([maya.incomplete(p) for p in paths], [None, None, None])

    def test_other_architectures_refused(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            paths = write_split(d, "Other-Q4", arch="qwen4exp")
            first, why = maya.local_first(paths[0])
            self.assertIsNone(first)
            self.assertIn("'qwen4exp'", why)
            with Quiet() as q, self.assertRaises(SystemExit):
                maya.model_step(SimpleNamespace(), d, ("local", paths[0]))
            self.assertIn("not GLM-5.3-Flash (glm5-next)", q.text)                # Maya's message

    def test_model_step_accepts_glm_dsa(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            paths = write_split(d, "GLM-5.3-UD-IQ2_M", n=3)
            with Quiet() as q, patch.object(maya, "rotational", return_value=False):
                got = maya.model_step(SimpleNamespace(), d, ("local", paths[0]))
            self.assertEqual(got, (d, paths, "GLM-5.3-UD-IQ2_M"))
            self.assertIn("GLM-5.3-UD-IQ2_M: 3 file(s)", q.text)
            self.assertTrue(maya.kind_is_glm_dsa(("local", paths[0]), paths))

    def test_pack_step_runs_iq_pack_on_shard_1(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            paths = write_split(d, "GLM-5.3-UD-IQ2_M", n=3)
            seen = []

            def fake_run(cmd, **kw):
                seen.append(cmd)
                out = Path(cmd[cmd.index("--out") + 1])
                for f in maya.PACK_FILES:
                    (out / f).parent.mkdir(parents=True, exist_ok=True)
                    (out / f).write_text("# absolute offsets in %s, x\n" % paths[0].name)
                return SimpleNamespace(returncode=0)

            with Quiet(), patch.object(maya, "run", side_effect=fake_run):
                pack = maya.pack_step(SimpleNamespace(repack=False), d, paths, Path("/llama"))
            self.assertEqual(pack, d / "pack")
            cmd = seen[0]
            self.assertTrue(cmd[1].endswith("iq_pack.py"))
            self.assertEqual(cmd[cmd.index("--gguf") + 1], str(paths[0]))
            self.assertIn("--compat-bf16", cmd)
            self.assertEqual(maya.pack_source(pack), paths[0].name)

    def test_pack_step_end_to_end_on_a_synthetic_glm_dsa_model(self):
        # the real tools/iq_pack.py on a tiny glm-dsa model (B1.6's fixture), through maya.py's pack step
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import test_iq_pack_glm_dsa as F
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            first = F.write_model(d, F.tensors_for(), stem="GLM-5.3-UD-IQ2_M")
            paths = maya.shard_set(first)
            for f in maya.PACK_FILES:                  # the tokenizer export needs a real vocabulary: given here
                if f.startswith("tokenizer/"):
                    (d / "pack" / f).parent.mkdir(parents=True, exist_ok=True)
                    (d / "pack" / f).write_text("{}")
            with Quiet() as q:
                pack = maya.pack_step(SimpleNamespace(repack=False), d, paths, Path(gguf.__file__).parents[2])
            self.assertEqual(pack, d / "pack")
            self.assertEqual(maya.pack_source(pack), first.name)
            for f in ("index.txt", "dense.bin", "native_experts.txt", "compat-bf16.json"):
                self.assertTrue((pack / f).exists(), f)
            self.assertIn("pack: ", q.text)

    def test_glm_dsa_is_told_apart(self):
        # set_up skips the vision step, refuses Metal and drafts with NextN when kind_is_glm_dsa
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            dsa = write_split(d, "GLM-5.3-UD-IQ2_M")
            flash = write_split(d, "GLM-5.3-Flash-Mine", arch="glm5-next")
            self.assertTrue(maya.kind_is_glm_dsa(("local", dsa[0]), dsa))
            self.assertFalse(maya.kind_is_glm_dsa(("local", flash[0]), flash))
            self.assertFalse(maya.kind_is_glm_dsa(("download", "Maya-S"), flash))
        self.assertEqual(maya.maya_sync.draft_for("dflash2", "glm-dsa", lambda *a: None), "mtp")
        self.assertEqual(maya.maya_sync.draft_for("dflash2", "glm5-next", lambda *a: None), "dflash2")

    def test_glm_dsa_pack_template_and_profile(self):
        with tempfile.TemporaryDirectory() as t:
            pack = Path(t) / "pack"
            (pack / "tokenizer").mkdir(parents=True)
            (pack / "index.txt").write_text("# strata pack index v3\n# glm-dsa: trunk 78, dense lead 3, experts 256, "
                                            "mtp yes\n", encoding="utf-8")
            with Quiet() as q:
                maya.base_template(pack)
            self.assertEqual((pack / "tokenizer" / "chat_template.jinja").read_bytes(), maya.G53.TEMPLATE.read_bytes())
            cfg = {"args": ["--glm-pack", str(pack)]}
            with patch.object(maya.G53, "PROFILE", Path(t) / "absent.bin"):    # (GLM-5.3's shipped one: S7)
                self.assertFalse(maya.add_profile(cfg))                        # never Flash's 288-expert profile
            self.assertNotIn("--expert-profile", cfg["args"])

    def test_vision_step_unchanged_for_flash(self):
        # a glm5-next model still reaches the vision files (here: --no-vision ends it, as before)
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            paths = write_split(d, "GLM-5.3-Flash-Mine", arch="glm5-next")
            a = SimpleNamespace(no_vision=True)
            with Quiet() as q:
                self.assertIsNone(maya.vision_step(a, {"backend": "cuda"}, {}, Path("/llama"), d, "Mine"))
            self.assertIn("skipped (--no-vision)", q.text)


if __name__ == "__main__":
    unittest.main()

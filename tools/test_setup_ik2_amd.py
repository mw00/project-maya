"""tools/test_setup_ik2_amd.py - the setup's Inkling 2 Small (maya.py's set_up_ik2_small), on AMD only (maya-ik2's
HIP build; on NVIDIA it has no setup: said, and nothing else happens): its files pinned (tools/ik2/download-small.json -
unsloth's UD-Q2_K_XL and the model's DSpark drafter), the names it is asked for by,
the memory it needs (VRAM + RAM, derived as DS4_MEMORY_GB: below it said, as DeepSeek's setup says it), the context
recommended as DeepSeek's (64K up to 256K by this PC's memory; offered from 32K up to the model's 1M), files already
there used as they are (the set, the drafter's index, a pack built by hand: no download, no conversion, nothing packed
or written again), the expert profile it ships (maya-ds4's name), its config (the drafter's index written, --dspark
none until DeepSeek's speed calibration finds that drafting pays) and the Models page's download list.  No GPU, no
network: the speed calibration's engine runs are stubbed.

    python -m unittest tools.test_setup_ik2_amd -v
"""
from pathlib import Path
import contextlib
import io
import json
import re
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path[:0] = [str(Path(__file__).resolve().parents[1]), str(Path(__file__).resolve().parent)]
import maya  # noqa: E402
import maya_quants as Q  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLE = json.loads((ROOT / "tools" / "ik2" / "download-small.json").read_text(encoding="utf-8"))   # (as committed)
AMD = [{"index": i, "name": "Radeon AI PRO R9700", "arch": "gfx1201", "vram_gb": 32.0} for i in range(4)]
DSPARK_GGUF = "Inkling-2-Small-DSpark-Q8_0.gguf"                    # (tools/ik2/convert_dspark.py's Q8_0 copy)


def args(**kw):
    base = dict(model=None, model_dir=None, models_dir=None, gguf_dir=None, draft=None, backend="hip", yes=True,
                port=None, host=None, api_key=None, env=[], no_vision=False, parallel=None, gpus=None, gpu=None,
                context=None, drafting=None, vram_reserve_mib=None, force_fit=False, repack=False, download_model=False, llama_dir=None,
                rebuild=False, qwen_quant=None, hy4_quant=None, ds4_quant=None, mm3_quant=None, no_start=True)
    base.update(kw)
    return SimpleNamespace(**base)


def quiet():
    return contextlib.redirect_stdout(io.StringIO())


def gguf(path: Path, arch: str = "inkling2", size: int = 0) -> None:
    """A GGUF file whose header names its architecture (gguf_head reads it), padded to `size` bytes."""
    key, val = b"general.architecture", arch.encode()
    head = struct.pack("<IIQQ", 0x46554747, 3, 0, 1) + struct.pack("<Q", len(key)) + key + struct.pack("<I", 8) + \
        struct.pack("<Q", len(val)) + val
    path.write_bytes(head + b"\0" * max(0, size - len(head)))


class Layout:
    """A models folder in a temporary directory, as the setup leaves it, with a small manifest of its own: the set's
    two shards, the drafter's files and (index) its converted index, (pack) a pack built by hand."""

    def __init__(self, t: Path, files: bool = True, index: bool = True, pack: bool = False, drafter: bool = True):
        self.models = t / "models"
        self.d = self.models / maya.IK2_GGUF_DIR / "UD-Q2_K_XL"
        self.mtp = self.models / maya.IK2_MTP_DIR
        self.d.mkdir(parents=True)
        self.mtp.mkdir(parents=True)
        shards = {"Inkling-2-Small-UD-Q2_K_XL-00001-of-00002.gguf": 4096,
                  "Inkling-2-Small-UD-Q2_K_XL-00002-of-00002.gguf": 8192}
        man = dict(TABLE, quants={"ud-q2_k_xl": {"folder": "UD-Q2_K_XL", "memory_gb": 203,
                                                 "files": {n: [b, "a" * 64] for n, b in shards.items()}}},
                   drafter=dict(TABLE["drafter"], files={"mtp.safetensors": [100, "b" * 64],
                                                         "config.json": [10, "c" * 64]}))
        self.manifest = t / "download.json"
        self.manifest.write_text(json.dumps(man), encoding="utf-8")
        self.shards = [self.d / n for n in shards]
        if files:
            gguf(self.shards[0], size=shards[self.shards[0].name])
            self.shards[1].write_bytes(b"\0" * shards[self.shards[1].name])
        if drafter:
            (self.mtp / "mtp.safetensors").write_bytes(b"\0" * 100)
            (self.mtp / "config.json").write_bytes(b"{}" + b" " * 8)
        if index:
            self.make_index()
        if pack:
            p = self.d / maya.IK2_PACK
            (p / "tokenizer").mkdir(parents=True)
            for f in maya.IK2_PACK_NEEDS:
                (p / f).write_text("x\n", encoding="utf-8")
            (p / "inkling2.txt").write_text(f"arch inkling2\nn_layer 48\nn_expert 384\nmodel_dir {self.d}\n",
                                            encoding="utf-8")

    def make_index(self):
        ix = self.mtp / maya.IK2_DSPARK
        ix.mkdir(exist_ok=True)
        (ix / "dspark.txt").write_text("format maya-ik2-dspark 1\nmodel_dir ..\n", encoding="utf-8")
        (ix / "tensors.txt").write_text(f"file 0 {DSPARK_GGUF}\nt x F32 0 0 4 1\n", encoding="utf-8")
        (self.mtp / DSPARK_GGUF).write_bytes(b"GGUF")
        return ix

    def files(self) -> set:
        return {str(p.relative_to(self.models)) for p in self.models.rglob("*")}


class Manifest(unittest.TestCase):
    def test_every_file_pinned(self):
        m = maya.ik2_small_entry()
        self.assertEqual(sorted(m["files"]), [f"Inkling-2-Small-UD-Q2_K_XL-0000{i}-of-00006.gguf" for i in range(1, 7)])
        for n, size in m["files"].items():
            self.assertGreater(size, 0)
            self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")
        self.assertAlmostEqual(m["download_gb"], 205.0, delta=0.1)          # (190.9 GiB of tensors)
        self.assertEqual(m["memory_gb"], maya.IK2_SMALL_MEMORY_GB)
        self.assertRegex(m["revision"], r"^[0-9a-f]{40}$")
        self.assertEqual(maya.file_url(m, "Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf"),
                         f"https://huggingface.co/unsloth/Inkling-2-Small-GGUF/resolve/{m['revision']}/UD-Q2_K_XL/"
                         "Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf")

    def test_the_memory_derived(self):
        """IK2_MEMORY_GB (download.json's memory_gb) as its comment derives it, in GiB: the files, the engine's RAM
        margin (maya-ds4's 8 GiB, the engine's ram_margin) and the drafter's measured 3.47 GB, rounded up."""
        import math
        files = sum(maya.ik2_small_entry()["files"].values()) / 2**30
        self.assertAlmostEqual(files, 190.9, delta=0.05)
        self.assertEqual(maya.IK2_SMALL_MEMORY_GB, math.ceil(files + maya.IK2_RAM_MARGIN_GIB + 3.47e9 / 2**30))
        src = (ROOT / "src" / "ik2" / "hip" / "ik2_hip_model.cu").read_text(encoding="utf-8")
        self.assertIn("double gib = 8.0;", src)                       # (the engine's ram_margin(): the same 8 GiB)
        self.assertEqual(maya.IK2_RAM_MARGIN_GIB, 8)

    def test_the_drafter_pinned(self):
        """Thinking Machines' own drafter: mtp.safetensors and the config.json with its mtp_config, from the model's
        repo (what <models>/Inkling-2-Small-MTP was downloaded from: docs/INKLING2_DSPARK.md)."""
        m = maya.ik2_small_drafter_entry()
        self.assertEqual(sorted(m["files"]), ["config.json", "mtp.safetensors"])
        self.assertEqual(m["files"]["mtp.safetensors"], 3423429800)
        self.assertEqual(m["repo"], "thinkingmachines/Inkling-2-Small")
        self.assertEqual(maya.file_url(m, "mtp.safetensors"),
                         f"https://huggingface.co/thinkingmachines/Inkling-2-Small/resolve/{m['revision']}/mtp.safetensors")
        for n in m["files"]:
            self.assertRegex(m["sha256"][n], r"^[0-9a-f]{64}$")


class PackKeys(unittest.TestCase):
    """The keys the model's files always give are required, as the pack's other ones: no fallback (pack_gguf.py's
    expert_weights_scale and final_logit_softcapping, the engine's inkling2.txt reader's five)."""

    KV = {"inkling2." + k: v for k, v in {
        "block_count": 2, "attention.sliding_window_pattern": [1, 0], "attention.log_scale_floor": 128000,
        "embedding_length": 4608, "vocab_size": 201024, "vocab_size_unpadded": 200058, "context_length": 1048576,
        "attention.head_count": 48, "attention.head_count_kv": 8, "attention.key_length": 128,
        "attention.value_length": 128, "attention.layer_norm_rms_epsilon": 1e-6, "expert_count": 384,
        "expert_used_count": 8, "expert_feed_forward_length": 2304, "expert_weights_scale": 8.0,
        "final_logit_softcapping": 30.0, "attention.sliding_window": 1024, "attention.relative_length": 16,
        "attention.relative_extent": 1024, "shortconv.l_cache": 4, "attention.log_scale_alpha": 0.1,
        "attention.qk_clip": 22.6, "attention.relative_clip": 8.0, "attn_res.block_size": 6,
        "activation.situ_beta": 4.0, "activation.situ_linear_beta": 25.0}.items()}
    KV.update({"tokenizer.ggml.bos_token_id": 200006, "tokenizer.ggml.eos_token_id": 200006})

    def test_required(self):
        import importlib.util
        spec = importlib.util.spec_from_file_location("ik2_pack_gguf", ROOT / "tools" / "ik2" / "pack_gguf.py")
        pg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(pg)
        geo = pg.geometry(dict(self.KV), None)
        self.assertEqual((geo["expert_weights_scale"], geo["final_logit_softcapping"], geo["log_scaling_n_floor"]),
                         (8.0, 30.0, 8192))
        for key in ("expert_weights_scale", "final_logit_softcapping"):
            kv = {k: v for k, v in self.KV.items() if k != "inkling2." + key}
            with self.subTest(key=key), self.assertRaises(SystemExit) as e:
                pg.geometry(kv, None)
            self.assertIn(key, str(e.exception))
        hpp = (ROOT / "src" / "ik2" / "hip" / "ik2_hip_pack.hpp").read_text(encoding="utf-8")
        self.assertIn('{"expert_weights_scale", "final_logit_softcapping", "log_scaling_n_floor", "bos", "eos"}', hpp)
        for fallback in ('num("expert_weights_scale", 1.0)', 'num("final_logit_softcapping", 0.0)',
                         'num("log_scaling_n_floor", 1)', 'num("bos", -1)', 'num("eos", -1)'):
            self.assertNotIn(fallback, hpp)


class Names(unittest.TestCase):
    def test_the_setup_ids(self):
        for name in ("inkling-2-small", "inkling-2-small-ud-q2_k_xl", "Inkling-2-Small", "inkling2-small", "ik2-small"):
            with self.subTest(name=name):
                a = args(model=name)
                self.assertIsNone(maya.model_option(a))
                self.assertEqual(a.model, maya.IK2_SMALL)
                self.assertEqual(maya.choose_family(a, {}), "ik2")
        self.assertEqual(Q.resolve("inkling-2-small-ud-q2_k_xl"), ("ik2", "ud-q2_k_xl"))
        self.assertEqual(maya.setup_title({"exe": "/x/build-hip/maya-ik2", "backend": "hip",
                                           "installer": {"family": "ik2"}}), "Inkling-2-Small")
        a = args(model="inkling-2-mini")                                # (Inkling-2-Mini: the family's other model)
        self.assertIsNone(maya.model_option(a))
        self.assertEqual(a.model, maya.IK2)
        self.assertFalse(maya.ik2_small(a))

    def test_the_pickers_row(self):
        rows = []

        def pick(title, intro, opts, default, yes, *x):
            rows.extend(opts)
            return [o[0] for o in opts].index(maya.IK2_SMALL_SHOWN)
        a = args()
        with patch.object(maya, "choose", pick), patch.object(maya.S, "gpus", return_value=[]), quiet():
            self.assertEqual(maya.choose_family(a, {}), "ik2")
        self.assertTrue(maya.ik2_small(a))                               # (its row: Inkling-2-Small's own setup)
        labels = [r[0] for r in rows]
        self.assertEqual(labels.index(maya.IK2_SMALL_SHOWN), labels.index("Qwen-3.8-Flash-Next") + 1)   # (before Hy4-flash's)
        self.assertIn(f"{maya.IK2_SHOWN} (experimental)", labels)        # (Inkling-2-Mini's row, the last)
        about = rows[labels.index(maya.IK2_SMALL_SHOWN)][2]
        self.assertIn("AMD", about)
        self.assertIn("maya-ik2", about)                                # (NVIDIA: said where it runs instead)
        rows.clear()                                                    # (on NVIDIA: not offered)
        with patch.object(maya, "choose", lambda t, i, opts, d, y, *x: rows.extend(opts) or d), \
                patch.object(maya.S, "gpus", return_value=[]), quiet():
            maya.choose_family(args(backend="cuda"), {})
        self.assertNotIn(maya.IK2_SMALL, [r[0] for r in rows])


class NotOnNvidia(unittest.TestCase):
    def test_said_plainly_nothing_built(self):
        """On NVIDIA the setup stops at step 1 and says that the model's engine there, maya-ik2, has no setup."""
        for backend in ("cuda", "metal"):
            with self.subTest(backend=backend), patch.object(maya.S, "gpus", side_effect=AssertionError("asked")), \
                    patch.object(maya, "check_hip_pc", side_effect=AssertionError("asked")), \
                    contextlib.redirect_stdout(io.StringIO()) as out, self.assertRaises(SystemExit):
                maya.check_pc(args(backend=backend, model=maya.IK2_SMALL), "ik2")
            self.assertIn("AMD GPUs", out.getvalue())
            self.assertIn("docs/INKLING2.md", out.getvalue())
            if backend == "cuda":
                self.assertIn("on NVIDIA it has no setup yet", out.getvalue())

    def test_amd_goes_on(self):
        with patch.object(maya, "check_hip_pc", return_value={"backend": "hip"}) as hip, quiet():
            self.assertEqual(maya.check_pc(args(backend="hip", model=maya.IK2_SMALL), "ik2"), {"backend": "hip"})
        self.assertTrue(hip.call_args.kwargs["text_only"])


class Context(unittest.TestCase):
    def test_up_to_the_models_own(self):
        self.assertEqual(maya.IK2_SMALL_CONTEXTS, maya.DS4_CONTEXTS)          # (DeepSeek's: from 32K)
        self.assertEqual(maya.IK2_SMALL_CONTEXTS[-1], 1048576)                # (inkling2.txt's context_length)
        self.assertEqual(maya.IK2_KV_BYTES, 32768)                      # (8 global layers x K, V x 8 heads x 128 x F16)
        self.assertAlmostEqual(maya.ik2_small_kv_gb(262144, 0), 8.0 + 262144 * 1560 / 2**30)   # (one GPU at least)
        self.assertAlmostEqual(maya.ik2_small_kv_gb(262144, 4, True), 262144 * (32768 + 4 * 1560 + 6400) / 2**30)

    def test_the_bytes_from_the_engines_allocations(self):
        """A position's VRAM as the engine allocates it - the Models page's figures (serve/models.py), the setup's
        too: the global layers' K / V rows (dense_bytes' cache_rows), each GPU's attention scratch (S_need's
        attention_split_floats: n_head x (head_dim + 2) floats a 16-key tile) and the drafter's (Dspark::ctx_bytes)."""
        from serve import models
        src = ROOT / "src" / "ik2" / "hip"
        model = (src / "ik2_hip_model.cu").read_text(encoding="utf-8")
        kern = (src / "ik2_hip_kernels.cu").read_text(encoding="utf-8")
        self.assertIn("? (size_t) (g.n_swa + o.rewind + Tc) : (size_t) o.max_ctx;", model)   # (a global layer: max_ctx)
        self.assertIn("cache_rows(il) * kvd * 2 * 2", model)            # (K and V, F16)
        self.assertIn("k::attention_split_floats(g.n_head, g.head_dim, (int64_t) max_keys)", model)
        self.assertIn("(nk + 15) / 16);\n    return (size_t) H * (size_t) tiles * (size_t) (D + 2);", kern)
        self.assertEqual(models.IK2_ATTN_BYTES, 1560)
        self.assertEqual(models.IK2_DSPARK_KV_BYTES, 6400)
        self.assertIn("return 2 * 8 * 128 * 2 + (size_t) kBlock * 48 * (4 + 2);",
                      (src / "ik2_hip_dspark.hpp").read_text(encoding="utf-8"))
        self.assertEqual((maya.IK2_KV_BYTES, maya.IK2_ATTN_BYTES, maya.IK2_DSPARK_KV_BYTES),
                         (models.IK2_KV_BYTES, models.IK2_ATTN_BYTES, models.IK2_DSPARK_KV_BYTES))

    def test_the_drafters_vram_one_figure(self):
        """The setup's and the page's figure: one, the measured 3.47 GB - the engine needs none (it places its experts
        after the drafter is in, in what is left)."""
        from serve import models
        hpp = (ROOT / "src" / "ik2" / "hip" / "ik2_hip_model.hpp").read_text(encoding="utf-8")
        self.assertNotIn("dspark_reserve_gb", hpp)
        self.assertEqual(maya.IK2_DSPARK_GB, models.IK2_DSPARK_GB)
        self.assertEqual(models.IK2_DSPARK_GB, 3.47)

    def test_recommended_by_this_pcs_memory(self):
        """As DeepSeek's (ds4_recommended_context): the largest context from 64K up to 256K whose cache fits in VRAM +
        RAM beyond the model, else 64K - by the PC's memory alone, nothing of one PC's."""
        rec = lambda vram, ram: maya.ik2_small_recommended_context(   # noqa: E731
            vram, ram, maya.ik2_small_memory_gb(), round(vram / 32), True)
        need = maya.ik2_small_memory_gb()
        for vram, ram, want in ((128, 125, 262144),                     # (51 GB beyond the model: 256K, the most)
                                (128, need - 128 + maya.ik2_small_kv_gb(262144, 4, True), 262144),   # (256K's cache, just)
                                (128, need - 128 + maya.ik2_small_kv_gb(131072, 4, True), 131072),
                                (96, need - 96 + maya.ik2_small_kv_gb(65536, 3, True), 65536),
                                (128, need - 128, maya.IK2_SMALL_DEFAULT_CONTEXT),   # (nothing beyond it: the default)
                                (128, 0, maya.IK2_SMALL_DEFAULT_CONTEXT)):    # (the RAM not read)
            with self.subTest(vram=vram, ram=ram):
                self.assertEqual(rec(vram, ram), want)
        self.assertEqual(maya.ik2_small_recommended_context(128, 125, need), 262144)

    def test_the_menu_and_the_setup_before(self):
        rows = []

        def pick(title, intro, opts, default, yes, *x):
            rows.extend(opts)
            return default
        with patch.object(maya, "choose", pick):
            self.assertEqual(maya.choose_context(args(), None, maya.IK2_SMALL_CONTEXTS, 1048576), 1048576)
            self.assertEqual(maya.choose_context(args(), 262144, maya.IK2_SMALL_CONTEXTS, 1048576), 262144)
        self.assertEqual(rows[-1], ("1M tokens", "recommended", None))
        self.assertEqual(maya.choose_context(args(context=65536), None, maya.IK2_SMALL_CONTEXTS, 1048576), 65536)


class Folders(unittest.TestCase):
    def test_from_the_config_before(self):
        """A config written by hand: its pack's folder is the set's, the models folder the one above, the drafter
        where its --dspark-index is."""
        with tempfile.TemporaryDirectory() as t:
            lay = Layout(Path(t), pack=True)
            prev = {"args": ["--pack", str(lay.d / maya.IK2_PACK), "--max-context", "262144",
                             "--dspark-index", str(lay.mtp / maya.IK2_DSPARK)]}
            with patch.object(maya, "IK2_SMALL_MANIFEST", lay.manifest):
                self.assertEqual(maya.ik2_small_models_dir(args(), prev), lay.models)
                self.assertEqual(maya.ik2_small_dir(args(), lay.models, prev), lay.d)
                self.assertEqual(maya.ik2_small_mtp_dir(lay.models, prev), lay.mtp)
                # (none before: the models folder given, the layout of unsloth's repo in it)
                self.assertEqual(maya.ik2_small_dir(args(), lay.models, {}), lay.d)
                self.assertEqual(maya.ik2_small_mtp_dir(lay.models, {}), lay.mtp)
                self.assertEqual(maya.ik2_small_models_dir(args(models_dir=t), prev), Path(t).resolve())
                self.assertEqual(maya.ik2_small_models_dir(args(), {"installer": {"models_dir": "/m"}}), Path("/m"))
                self.assertEqual(maya.ik2_small_dir(args(model_dir=t), lay.models, prev), Path(t).resolve())


class Files(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)

    def lay(self, **kw) -> Layout:
        lay = Layout(Path(self.t.name), **kw)
        p = patch.object(maya, "IK2_SMALL_MANIFEST", lay.manifest)
        p.start()
        self.addCleanup(p.stop)
        return lay

    def test_the_set_there_is_used(self):
        lay = self.lay()
        with patch.object(maya, "offer_download", side_effect=AssertionError("downloaded")), quiet():
            self.assertEqual(maya.ik2_small_model_step(args(), lay.d), lay.shards)

    def test_a_missing_shard_is_offered(self):
        lay = self.lay()
        lay.shards[1].write_bytes(b"\0" * 100)                          # (short: still on its way)
        with patch.object(maya, "offer_download", return_value=False) as dl, quiet():
            self.assertIsNone(maya.ik2_small_model_step(args(), lay.d))
        self.assertEqual(dl.call_args.args[3], lay.shards)

    def test_another_models_files(self):
        lay = self.lay()
        gguf(lay.shards[0], "hy4", size=4096)
        with quiet(), self.assertRaises(SystemExit):
            maya.ik2_small_model_step(args(), lay.d)

    def test_the_drafters_index_there_is_used(self):
        lay = self.lay()
        with patch.object(maya, "run", side_effect=AssertionError("converted")), \
                patch.object(maya, "offer_download", side_effect=AssertionError("downloaded")), quiet():
            self.assertEqual(maya.ik2_small_drafter_step(args(), lay.mtp, lay.shards[0], Path("/llama")),
                             lay.mtp / maya.IK2_DSPARK)

    def test_the_drafter_converted(self):
        lay = self.lay(index=False)
        ran = []

        def run(cmd, *x, **k):
            ran.append([str(c) for c in cmd])
            lay.make_index()
        with patch.object(maya, "run", run), patch.object(maya, "offer_download", side_effect=AssertionError), quiet():
            self.assertEqual(maya.ik2_small_drafter_step(args(), lay.mtp, lay.shards[0], Path("/llama")),
                             lay.mtp / maya.IK2_DSPARK)
        cmd = ran[0]
        self.assertTrue(cmd[1].endswith("tools/ik2/convert_dspark.py"))
        self.assertEqual(cmd[cmd.index("--mtp") + 1], str(lay.mtp))
        self.assertEqual(cmd[cmd.index("--type") + 1], "q8_0")         # (the copy the engine loads)
        self.assertEqual(cmd[cmd.index("--tokenizer-gguf") + 1], str(lay.shards[0]))
        self.assertEqual(cmd[cmd.index("--gguf-py") + 1], str(Path("/llama") / "gguf-py"))

    def test_the_drafter_not_downloaded(self):
        lay = self.lay(index=False, drafter=False)
        with patch.object(maya, "offer_download", return_value=False) as dl, \
                patch.object(maya, "run", side_effect=AssertionError), quiet():
            self.assertIsNone(maya.ik2_small_drafter_step(args(), lay.mtp, lay.shards[0], Path("/llama")))
        self.assertEqual([p.name for p in dl.call_args.args[3]], ["mtp.safetensors", "config.json"])

    def test_a_half_index_is_not_one(self):
        lay = self.lay()
        (lay.mtp / DSPARK_GGUF).unlink()                       # (the copy it names is gone)
        self.assertIsNone(maya.ik2_small_drafter_index(lay.mtp))

    def test_a_pack_built_by_hand_is_used_as_it_is(self):
        lay = self.lay(pack=True)
        before = lay.files()
        with patch.object(maya, "run", side_effect=AssertionError("packed")), quiet():
            self.assertEqual(maya.ik2_small_pack_step(args(), lay.d, lay.shards), lay.d / maya.IK2_PACK)
        self.assertEqual(lay.files(), before)                           # (nothing written into it)

    def test_a_pack_built_again(self):
        """--repack, a pack of another folder, a shard newer than it, or tools changed since: built again."""
        lay = self.lay(pack=True)
        pack = lay.d / maya.IK2_PACK
        cases = [(args(repack=True), None),
                 (args(), lambda: (pack / "inkling2.txt").write_text("model_dir /elsewhere\n", encoding="utf-8")),
                 (args(), lambda: (pack / maya.PACK_SOURCES).write_text(json.dumps({"tools": "old"}), encoding="utf-8"))]
        for a, change in cases:
            if change:
                change()
            ran = []
            with patch.object(maya, "run", lambda cmd, *x, **k: ran.append([str(c) for c in cmd])), quiet():
                maya.ik2_small_pack_step(a, lay.d, lay.shards)
            self.assertTrue(ran and ran[0][1].endswith("tools/ik2/pack_gguf.py"))
            self.assertEqual(ran[0][ran[0].index("--out") + 1], str(pack))
            self.assertEqual(ran[0][ran[0].index("--files") + 1:], [s.name for s in lay.shards])
            rec = json.loads((pack / maya.PACK_SOURCES).read_text(encoding="utf-8"))
            self.assertEqual(rec["tools"], maya.S.source_hash(maya.IK2_PACK_TOOLS))
            with patch.object(maya, "run", side_effect=AssertionError("packed twice")), quiet():
                maya.ik2_small_pack_step(args(), lay.d, lay.shards)          # (now stamped: used as it is)
            (pack / maya.PACK_SOURCES).unlink()
            (pack / "inkling2.txt").write_text(f"n_layer 48\nn_expert 384\nmodel_dir {lay.d}\n", encoding="utf-8")

    def test_the_expert_profile(self):
        """expert-profile.bin (maya-ds4's name) - the profile the engine reads by itself and writes back: the shipped
        one (48 x 384 routings), copied into a pack that has none; one already there stays, and so does a pack's
        default.prof from before the name (the engine reads it in its place)."""
        self.assertEqual(maya.IK2_PROFILE.stat().st_size, 48 * 384 * 4)
        lay = self.lay(pack=True)
        pack = lay.d / maya.IK2_PACK
        self.assertEqual(maya.ik2_small_seed_profile(pack), pack / "expert-profile.bin")
        self.assertEqual((pack / "expert-profile.bin").read_bytes(), maya.IK2_PROFILE.read_bytes())
        (pack / "expert-profile.bin").write_bytes(b"mine")
        self.assertIsNone(maya.ik2_small_seed_profile(pack))
        self.assertEqual((pack / "expert-profile.bin").read_bytes(), b"mine")
        (pack / "expert-profile.bin").unlink()
        (pack / "default.prof").write_bytes(b"from before")
        self.assertIsNone(maya.ik2_small_seed_profile(pack))
        self.assertFalse((pack / "expert-profile.bin").exists())
        (pack / "default.prof").unlink()
        (pack / "inkling2.txt").write_text("n_layer 4\nn_expert 384\n", encoding="utf-8")   # (another geometry)
        self.assertIsNone(maya.ik2_small_seed_profile(pack))
        src = (ROOT / "src" / "ik2" / "hip" / "ik2_hip_model.cu").read_text(encoding="utf-8")
        self.assertIn('o.pack + "/expert-profile.bin"', src)          # (the engine's default: the same file)
        self.assertIn('o.pack + "/default.prof"', src)


PC = {"backend": "hip", "gpus": AMD, "archs": ["gfx1201"], "rocm": "/opt/rocm"}


class Config(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name) / "maya"
        self.root.mkdir()
        self.lay = Layout(Path(self.t.name), pack=True)
        for p in (patch.object(maya, "ROOT", self.root), patch.object(maya, "IK2_SMALL_MANIFEST", self.lay.manifest)):
            p.start()
            self.addCleanup(p.stop)
        maya.select_build_backend("hip")

    def write(self, a=None, index=True) -> dict:
        with quiet():
            path = maya.write_config_ik2_small(a or args(), PC, {"lib_dirs": ["/opt/rocm/lib"]}, self.lay.d / maya.IK2_PACK,
                                         262144, self.lay.models, self.lay.d,
                                         self.lay.mtp / maya.IK2_DSPARK if index else None)
        self.assertEqual(path, self.root / "maya-inkling-2-small-hip.json")
        return json.loads(path.read_text(encoding="utf-8"))

    def test_fresh(self):
        cfg = self.write()
        self.assertEqual(Path(cfg["exe"]), maya.IK2_EXE)
        # (as DeepSeek's setup writes it on AMD: --dspark none until the speed calibration finds that drafting pays)
        self.assertEqual(cfg["args"], ["--pack", str(self.lay.d / maya.IK2_PACK), "--max-context", "262144",
                                       "--dspark-index", str(self.lay.mtp / maya.IK2_DSPARK), "--dspark", "none",
                                       "--reserve-mib", str(maya.hip_reserve_mib(AMD))])
        self.assertEqual(cfg["gpu"], [0, 1, 2, 3])
        self.assertEqual(cfg["backend"], "hip")
        self.assertEqual(cfg["tokenizer"], str(self.lay.d / maya.IK2_PACK / "tokenizer"))
        self.assertEqual((cfg["sampling"], cfg["reasoning_effort"]), Q.MODEL_DEFAULTS["ik2"])
        inst = cfg["installer"]
        self.assertEqual((inst["family"], inst["ik2_quant"], inst["models_dir"], inst["model_dir"]),
                         ("ik2", "ud-q2_k_xl", str(self.lay.models), str(self.lay.d)))
        # (the server and the Models page read it as the family's, with its drafter)
        self.assertEqual(Q.family_of(cfg), "ik2")
        self.assertEqual(Q.config_key(cfg), "ud-q2_k_xl")
        self.assertEqual(Q.config_label(cfg), "Inkling 2 Small · unsloth UD-Q2_K_XL")
        from serve import models
        self.assertEqual(models.choices(cfg)["draft"], ["dspark", "none"])
        self.assertEqual(models.settings_of(cfg)["draft"], "none")
        self.assertTrue((self.root / "run-maya-inkling-2-small-hip.sh").is_file())
        # (DeepSeek's speed calibration takes it: the drafter is there, nothing measured yet)
        self.assertTrue(maya.CDS.has_drafter(cfg))
        self.assertTrue(maya.needs_speed_calibration(cfg))
        on = maya.CDS.with_dspark(cfg, "on")
        self.assertEqual(maya.flag_value(on["args"], "--dspark"), "on")
        self.assertEqual(models.settings_of(on)["draft"], "dspark")

    def test_no_drafter(self):
        none = self.write(index=False)
        self.assertNotIn("--dspark", none["args"])
        self.assertNotIn("--dspark-index", none["args"])
        self.assertFalse(maya.CDS.has_drafter(none))
        self.assertFalse(maya.needs_speed_calibration(none))

    def test_again_keeps_what_it_does_not_ask(self):
        old = {"exe": "/x/maya-ik2", "args": ["--pack", "/old", "--max-context", "32768", "--reserve-mib", "900",
                                                   "--cpu-threads", "16", "--expert-profile", "/p.prof",
                                                   "--dspark-cum", "0.4", "--mmap-experts", "on"],
               "host": "0.0.0.0", "api_key": "k", "env": {"A": "1"}, "port": 9000, "reasoning_effort": "high"}
        (self.root / "maya-inkling-2-small-hip.json").write_text(json.dumps(old), encoding="utf-8")
        cfg = self.write(args(env=["B=2"]))
        for f, v in (("--reserve-mib", "900"), ("--cpu-threads", "16"), ("--expert-profile", "/p.prof"),
                     ("--dspark-cum", "0.4"), ("--mmap-experts", "on"), ("--max-context", "262144"),
                     ("--pack", str(self.lay.d / maya.IK2_PACK))):
            self.assertEqual(maya.flag_value(cfg["args"], f), v)
        self.assertEqual((cfg["host"], cfg["api_key"], cfg["env"]), ("0.0.0.0", "k", {"A": "1", "B": "2"}))
        self.assertEqual(cfg["reasoning_effort"], Q.MODEL_DEFAULTS["ik2"][1])   # (the model's own, every setup)

    def test_mmap_experts_as_deepseeks(self):
        """--mmap-experts (maya-ds4's flag): given to the setup, written - and a setup again keeps it; not given, not
        written (auto, the engine's default)."""
        self.assertIsNone(maya.flag_value(self.write()["args"], "--mmap-experts"))
        self.assertEqual(maya.flag_value(self.write(args(mmap_experts="off"))["args"], "--mmap-experts"), "off")
        self.assertEqual(maya.flag_value(self.write()["args"], "--mmap-experts"), "off")

    def test_parallel_as_deepseeks(self):
        """--parallel (maya-ds4's flag): given to the setup, written - and a setup again keeps it; not given, not
        written (one conversation, the engine's default)."""
        self.assertIsNone(maya.flag_value(self.write()["args"], "--parallel"))
        self.assertEqual(maya.flag_value(self.write(args(parallel=4))["args"], "--parallel"), "4")
        self.assertEqual(maya.flag_value(self.write()["args"], "--parallel"), "4")
        self.assertIn("ik2", maya.PARALLEL_FAMILIES)

    def test_the_vram_kept_free(self):
        """--reserve-mib (maya-ds4's flag): what DeepSeek's setup writes on AMD - --vram-reserve-mib, else
        hip_reserve_mib."""
        self.assertEqual(maya.flag_value(self.write(args(vram_reserve_mib=900))["args"], "--reserve-mib"), "900")
        self.assertEqual(maya.flag_value(self.write()["args"], "--reserve-mib"), "900")   # (a setup again keeps it)
        (self.root / "maya-inkling-2-small-hip.json").unlink()
        apu = [dict(AMD[0], unified_memory=True)]
        with patch.object(maya, "hip_unified_memory", lambda g: bool(g.get("unified_memory"))), \
                patch.dict(PC, gpus=apu):
            self.assertEqual(maya.flag_value(self.write()["args"], "--reserve-mib"), str(maya.hip_reserve_mib(apu)))


class SetUp(unittest.TestCase):
    """The whole setup on a temporary layout: the PC check and the build stubbed (no GPU), the rest as it runs."""

    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.addCleanup(self.t.cleanup)
        self.root = Path(self.t.name) / "maya"
        self.root.mkdir()
        self.built = []
        for p in (patch.object(maya, "ROOT", self.root), patch.object(maya, "check_pc", lambda a, fam: dict(PC)),
                  patch.object(maya, "tools_step", lambda a, *x: Path("/llama")),
                  patch.object(maya, "build_step", lambda a, pc, llama, target: self.built.append(target) or {}),
                  patch.object(maya, "configs", lambda: [])):
            p.start()
            self.addCleanup(p.stop)
        maya.select_build_backend("hip")

    def run_setup(self, lay: Layout, a, ram: float = 125.0):
        # (the speed calibration's two engine runs: recorded here, not run - the config it would measure)
        self.calibrated = []
        cal = patch.object(maya, "speed_calibration", lambda p, why="": self.calibrated.append(
            json.loads(Path(p).read_text(encoding="utf-8"))) or True)
        with cal, patch.object(maya, "IK2_SMALL_MANIFEST", lay.manifest), patch.object(maya, "mem_gb", lambda: (ram, ram / 2)), \
                patch.object(maya, "rotational", lambda p: False), \
                patch.object(maya, "offer_download", side_effect=AssertionError("downloaded")), \
                patch.object(maya, "run", side_effect=AssertionError("ran a tool")), \
                contextlib.redirect_stdout(io.StringIO()) as out:
            try:
                return maya.set_up(a, {}), out.getvalue()
            except SystemExit:
                return None, out.getvalue()

    def test_files_already_there_are_used(self):
        """The set, the drafter's index and a pack built by hand: nothing downloaded, converted, packed or written in
        the models folder - the profile aside, where the pack had none."""
        lay = Layout(Path(self.t.name), pack=True)
        (lay.d / maya.IK2_PACK / "default.prof").write_bytes(b"its own")
        before = lay.files()
        path, out = self.run_setup(lay, args(model=maya.IK2_SMALL, models_dir=str(lay.models)))
        self.assertIsNotNone(path, out)
        self.assertEqual(lay.files(), before)
        self.assertEqual(self.built, ["maya-ik2"])
        cfg = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(maya.flag_value(cfg["args"], "--dspark-index"), str(lay.mtp / maya.IK2_DSPARK))
        self.assertEqual(maya.flag_value(cfg["args"], "--max-context"), "262144")    # (recommended as DeepSeek: 256K, the most)
        self.assertIn("pack already built", out)

    def test_said_below_the_memory(self):
        """As DeepSeek's setup: VRAM + RAM below the model's memory is said, and the setup goes on (the engine then
        reads the experts RAM does not hold from the model's files: --mmap-experts auto)."""
        lay = Layout(Path(self.t.name), pack=True)
        a = args(model=maya.IK2_SMALL, models_dir=str(lay.models), gpus="0")
        with patch.dict(PC, gpus=AMD[:1]):
            path, out = self.run_setup(lay, a, ram=64)
        self.assertIsNotNone(path, out)
        self.assertIn(f"below the ~{maya.IK2_SMALL_MEMORY_GB} GB", out)
        self.assertIn("read in place from the model's files", out)
        self.assertEqual(self.built, ["maya-ik2"])

    def test_drafting_as_deepseeks(self):
        """--drafting on / off: written, the calibration skipped (installer.speed_calibration records the choice);
        measure (--yes's default): DeepSeek's speed calibration on the config, which starts with --dspark none."""
        lay = Layout(Path(self.t.name), pack=True)
        for how, want in (("on", "on"), ("off", "none")):
            with self.subTest(drafting=how):
                path, out = self.run_setup(lay, args(model=maya.IK2_SMALL, models_dir=str(lay.models), drafting=how))
                cfg = json.loads(path.read_text(encoding="utf-8"))
                self.assertEqual(maya.flag_value(cfg["args"], "--dspark"), want)
                self.assertEqual(cfg["installer"]["speed_calibration"]["skipped"], "chosen")
                self.assertFalse(maya.needs_speed_calibration(cfg))
                self.assertEqual(self.calibrated, [])
        (self.root / "maya-inkling-2-small-hip.json").unlink()
        path, out = self.run_setup(lay, args(model=maya.IK2_SMALL, models_dir=str(lay.models)))
        self.assertEqual(len(self.calibrated), 1)
        self.assertEqual(maya.flag_value(self.calibrated[0]["args"], "--dspark"), "none")

    def test_nothing_downloaded(self):
        lay = Layout(Path(self.t.name), files=False)
        with patch.object(maya, "IK2_SMALL_MANIFEST", lay.manifest), patch.object(maya, "offer_download", return_value=False), \
                patch.object(maya, "mem_gb", lambda: (125.0, 60.0)), quiet():
            self.assertIsNone(maya.set_up(args(model=maya.IK2_SMALL, models_dir=str(lay.models)), {}))


class Engine(unittest.TestCase):
    def test_a_start_compiles_it_again_after_an_update(self):
        maya.select_build_backend("hip")
        cfg = {"exe": str(maya.IK2_EXE), "backend": "hip", "gpu": [0]}
        with tempfile.TemporaryDirectory() as t, patch.object(maya, "STAMP", Path(t) / "MAYA-BUILD.json"), \
                patch.object(maya, "hip_clang", return_value=Path("/x/clang++")), \
                patch.object(maya, "compile_engine_hip") as comp:
            (Path(t) / "MAYA-BUILD.json").write_text(json.dumps({"src": "old", "archs": ["gfx1201"], "rocm": "/r",
                                                                "llama": str(ROOT / "third_party" / "llama.cpp"),
                                                                "targets": ["maya-ik2"]}), encoding="utf-8")
            if not (ROOT / "third_party" / "llama.cpp" / "ggml" / "CMakeLists.txt").exists():
                self.skipTest("no llama.cpp checkout")
            maya.refresh_engine(cfg)
        self.assertEqual(comp.call_args.kwargs["target"], "maya-ik2")
        self.assertIn("maya-ik2", maya.ENGINE_PROCESSES)


class ModelsPage(unittest.TestCase):
    def test_offered_on_amd_only(self):
        from serve import model_pull as P
        hip = [x for x in P.catalog(Path("/nonexistent"), {}, {"backend": "hip"}, [], {}) if x["family"] == "ik2"]
        self.assertEqual([(x["id"], x["name"], x["by"]) for x in hip],
                         [("inkling-2-small", "Inkling 2 Small · unsloth UD-Q2_K_XL", "unsloth")])
        self.assertEqual([x for x in P.catalog(Path("/nonexistent"), {}, {"backend": "cuda"}, [86], {})
                          if x["id"] == "inkling-2-small"], [])              # (on NVIDIA: Inkling-2-Mini's alone)
        cmd = P.setup_command("inkling-2-small", {"backend": "hip", "installer": {"models_dir": "/m"}})
        self.assertEqual(cmd[cmd.index("--model"):cmd.index("--download-model")], ["--model", "inkling-2-small"])
        self.assertEqual(cmd[cmd.index("--backend") + 1], "hip")
        self.assertGreater(P._other_sizes().get("inkling-2-small", 0), 205)   # (the set and the drafter)


if __name__ == "__main__":
    unittest.main()

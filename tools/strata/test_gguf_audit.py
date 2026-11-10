"""tools/strata/test_gguf_audit.py - the B0.7 audit's PASS/WARN/FAIL logic on synthetic glm-dsa GGUFs.

The GGUFs are written with llama.cpp's gguf-py (the pin's GGUFWriter): real GLM-5.3 geometry and tensor table, header
only, then extended as sparse files to the size their tensor table implies (a 233 GB six-shard file costs no disk).
Each test mutates one property of a clean file and checks that exactly the intended check changes.

    PYTHONPATH=<llama.cpp>/gguf-py python3 -m unittest discover -s tools/strata -p 'test_gguf_audit.py' -v
"""
from __future__ import annotations

import contextlib
import io
import json
import os
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
import gguf_audit as ga  # noqa: E402

try:
    import _paths  # noqa: E402  (tools/_paths.py: STRATA_GGUF_PY, third_party/llama.cpp/gguf-py)
    with contextlib.suppress(SystemExit):
        _paths.add_gguf_py()
    import numpy as np
    import gguf
    HAVE_GGUF = True
except Exception:   # pragma: no cover
    HAVE_GGUF = False

T = ga.TYPE_ID
FULL = set(ga.FULL_LAYERS)
TEMPLATE = "[gMASK]<sop>{% for m in messages %}<|{{ m.role }}|>{{ m.content }}{% endfor %}<|assistant|><think>"


def model_spec(n_blocks: int = 79) -> tuple[list[tuple], list[tuple]]:
    """(kv, tensors) of a UD-IQ2_M-like GLM-5.3 file.  kv: (key, value, kind); tensors: (name, ne, type id)."""
    p = "glm-dsa."
    kv = [(p + "block_count", n_blocks, "u32"), (p + "nextn_predict_layers", 1, "u32"),
          (p + "context_length", 1048576, "u32"), (p + "vocab_size", ga.VOCAB, "u32")]
    for key, want in ga.GEOMETRY:
        if key == "context_length":
            continue
        kind = "bool" if isinstance(want, bool) else "f32" if isinstance(want, float) else "u32"
        kv.append((p + key, want, kind))
    kv.append((p + "attention.indexer.types", [i in FULL for i in range(78)], "arr"))
    kv += [("tokenizer.ggml.model", "gpt2", "str"), ("tokenizer.ggml.pre", "glm4", "str"),
           ("tokenizer.ggml.tokens", ["t%d" % i for i in range(ga.VOCAB)], "arr"),
           ("tokenizer.ggml.merges", ["t1 t2", "t3 t4"], "arr"),
           ("tokenizer.chat_template", TEMPLATE, "str")]
    ts = [("token_embd.weight", [6144, ga.VOCAB], T["Q4_K"]), ("output_norm.weight", [6144], T["F32"]),
          ("output.weight", [6144, ga.VOCAB], T["Q6_K"])]
    for i in range(n_blocks):
        b = "blk.%d." % i
        ts += [(b + "attn_norm.weight", [6144], 0), (b + "attn_q_a_norm.weight", [2048], 0),
               (b + "attn_kv_a_norm.weight", [512], 0), (b + "attn_q_a.weight", [6144, 2048], T["Q5_K"]),
               (b + "attn_q_b.weight", [2048, 16384], T["Q4_K"]), (b + "attn_kv_a_mqa.weight", [6144, 576], T["Q6_K"]),
               (b + "attn_k_b.weight", [192, 512, 64], T["Q8_0"]), (b + "attn_v_b.weight", [512, 256, 64], T["Q8_0"]),
               (b + "attn_output.weight", [16384, 6144], T["Q4_K"]), (b + "ffn_norm.weight", [6144], 0)]
        if i in FULL or i == 78:
            ts += [(b + "indexer.k_norm.weight", [128], 0), (b + "indexer.k_norm.bias", [128], 0),
                   (b + "indexer.proj.weight", [6144, 32], T["BF16"]), (b + "indexer.attn_k.weight", [6144, 128], T["Q8_0"]),
                   (b + "indexer.attn_q_b.weight", [2048, 4096], T["Q8_0"])]
        if i < 3:
            ts += [(b + "ffn_gate.weight", [6144, 12288], T["Q4_K"]), (b + "ffn_up.weight", [6144, 12288], T["Q4_K"]),
                   (b + "ffn_down.weight", [12288, 6144], T["Q5_K"])]
        else:
            ts += [(b + "ffn_gate_inp.weight", [6144, 256], 0), (b + "exp_probs_b.bias", [256], 0),
                   (b + "ffn_gate_exps.weight", [6144, 2048, 256], T["IQ2_XXS"]),
                   (b + "ffn_up_exps.weight", [6144, 2048, 256], T["IQ2_XXS"]),
                   (b + "ffn_down_exps.weight", [2048, 6144, 256], T["IQ3_XXS"]),
                   (b + "ffn_gate_shexp.weight", [6144, 2048], T["Q5_K"]), (b + "ffn_up_shexp.weight", [6144, 2048], T["Q5_K"]),
                   (b + "ffn_down_shexp.weight", [2048, 6144], T["Q6_K"])]
        if i == 78:
            ts += [(b + "nextn.eh_proj.weight", [12288, 6144], T["Q8_0"]), (b + "nextn.enorm.weight", [6144], 0),
                   (b + "nextn.hnorm.weight", [6144], 0), (b + "nextn.shared_head_norm.weight", [6144], 0)]
    return kv, ts


def _add_kv(w, key, val, kind):
    if kind == "u32":
        w.add_uint32(key, val)
    elif kind == "f32":
        w.add_float32(key, val)
    elif kind == "bool":
        w.add_bool(key, val)
    elif kind == "str":
        w.add_string(key, val)
    elif kind == "arr":
        w.add_array(key, val)
    elif kind == "u16":
        w.add_uint16(key, val)
    elif kind == "i32":
        w.add_int32(key, val)
    else:
        raise ValueError(kind)


def write_gguf(d: pathlib.Path, kv, ts, n_shards: int = 6, meta_first: bool = False, arch: str = "glm-dsa",
               stem: str = "GLM-5.3-UD-IQ2_M", extra_bytes: int = 0) -> list[pathlib.Path]:
    """Header-only shards via gguf-py, then truncate() to the size the tensor table implies (sparse)."""
    sizes = [ga.type_bytes(t, int(np.prod(ne))) for _, ne, t in ts]
    data_shards = n_shards - (1 if meta_first else 0)
    per = sum(sizes) / data_shards
    groups, cur, acc = [[] for _ in range(n_shards)], (1 if meta_first else 0), 0
    for spec, sz in zip(ts, sizes):
        if acc >= per * (cur + 1 - (1 if meta_first else 0)) and cur < n_shards - 1:
            cur += 1
        groups[cur].append((spec, sz))
        acc += sz
    paths = []
    for i, g in enumerate(groups):
        p = d / ("%s-%05d-of-%05d.gguf" % (stem, i + 1, n_shards))
        w = gguf.GGUFWriter(path=None, arch=arch)
        if i == 0:
            for key, val, kind in kv:
                _add_kv(w, key, val, kind)
        w.add_uint16("split.no", i)
        w.add_uint16("split.count", n_shards)
        w.add_int32("split.tensors.count", len(ts))
        for (name, ne, t), sz in g:
            w.add_tensor_info(name, list(reversed(ne)), np.dtype(np.float32), sz,
                              raw_dtype=gguf.GGMLQuantizationType(t))
        w.write_header_to_file(p)
        w.write_kv_data_to_file()
        w.write_ti_data_to_file()
        w.close()
        pad = lambda x: (x + 31) // 32 * 32   # noqa: E731
        size = pad(p.stat().st_size) + sum(pad(sz) for _, sz in g) + (extra_bytes if i == n_shards - 1 else 0)
        os.truncate(p, size)
        paths.append(p)
    return paths


def audit(paths, **opt) -> ga.Audit:
    shards = ga.load_local([str(paths[0])])
    a = ga.Audit(shards, ga.Options(**opt))
    a.run()
    return a


def status(a: ga.Audit) -> dict[str, str]:
    return {c.id: c.status for c in a.checks}


@unittest.skipUnless(HAVE_GGUF, "gguf-py (llama.cpp pin) and numpy are needed to write the synthetic GGUFs")
class AuditTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.tmp.name)
        cls.n = 0
        cls.clean = cls.build()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    @classmethod
    def build(cls, mutate=None, n_blocks=79, **kw) -> ga.Audit:
        kv, ts = model_spec(n_blocks)
        if mutate:
            kv, ts = mutate(kv, ts)
        cls.n += 1
        d = cls.root / ("m%02d" % cls.n)
        d.mkdir()
        opts = kw.pop("opts", {})
        return audit(write_gguf(d, kv, ts, **kw), **opts)

    def mut_kv(self, key, val, kind=None):
        def f(kv, ts):
            out = [(k, v, kd) for k, v, kd in kv if k != key]
            if val is not None:
                old = next((kd for k, v, kd in kv if k == key), kind)
                out.append((key, val, kind or old))
            return out, ts
        return f

    @staticmethod
    def mut_types(pred, new_type):
        def f(kv, ts):
            return kv, [(n, ne, new_type if pred(n) else t) for n, ne, t in ts]
        return f

    def assertOnly(self, a: ga.Audit, expect: dict[str, str]):
        got = status(a)
        want = {k: "PASS" for k in got}
        # B15.4: the fixture is UD-IQ2_M-like, its blk.78 experts IQ2_XXS / IQ3_XXS: A9 warns about the MTP head
        want["A9"] = "WARN"
        want.update(expect)
        self.assertEqual(got, want, "\n" + "\n".join("%s %s: %s" % (c.id, c.status, c.notes) for c in a.checks
                                                     if c.status != "PASS"))

    # ---------------------------------------------------------------- clean file
    def test_clean_primary_passes(self):
        a = self.clean
        self.assertOnly(a, {"A11": "WARN"})     # synthetic template: not the official sha
        self.assertEqual(len(a.shards), 6)
        self.assertLess(a.file_bytes(), 245e9)
        self.assertEqual(len(a.full_layers()), 22)
        self.assertEqual(len(a.classes()), 1)

    def test_placement_numbers(self):
        p = self.clean.placement
        self.assertEqual(p["kv_tok"], [34048, 35840, 32384])          # section 2.2 KV per GPU
        self.assertEqual(sum(p["kv_tok"]), 102272)
        self.assertEqual(p["n_moe"], 76)
        self.assertEqual(sum(p["experts"]), 76 * 256)
        blob = (2 * ga.type_bytes(T["IQ2_XXS"], 6144 * 2048) + ga.type_bytes(T["IQ3_XXS"], 6144 * 2048))
        self.assertAlmostEqual(p["blob"], blob, delta=1)
        self.assertAlmostEqual(p["slack"], 16 * 76 * blob, delta=16 * 76)
        r4, r64 = p["rows"][4096], p["rows"][65536]
        for g in range(3):
            self.assertGreater(r4["pool"][g], r64["pool"][g])
            self.assertGreater(r64["experts"][g], 1500)
        # routed bytes split between VRAM and the pinned tier exactly
        self.assertAlmostEqual(sum(r64["vram_routed"]) + r64["pinned"] - p["slack"], sum(p["routed"]), delta=1e3)
        self.assertGreater(r64["spare"], 0)
        # closure: tensors vs file within 0.5%
        self.assertLess(abs(p["file"] - p["tensors"]) / p["file"], 0.005)
        md = self.clean.markdown("t")
        self.assertIn("Expert pool @64K", md)
        self.assertIn("| A14 |", md)

    def test_tsv_lists_every_tensor(self):
        tsv = self.clean.tsv().splitlines()
        self.assertEqual(len(tsv) - 1, len(self.clean.tensors))
        row = next(l for l in tsv if l.startswith("blk.40.ffn_gate_exps.weight\t"))
        f = row.split("\t")
        self.assertEqual(f[1], "IQ2_XXS")
        self.assertEqual(f[2], "6144,2048,256")
        self.assertEqual(f[6:8], ["routed", "1"])
        kb = next(l for l in tsv if l.startswith("blk.0.attn_k_b.weight\t")).split("\t")
        self.assertEqual(int(kb[5]), 2 * 192 * 512 * 64)               # packed as BF16

    # ---------------------------------------------------------------- A1, A2
    def test_a1_wrong_arch(self):
        kv, ts = model_spec()
        kv = [(k.replace("glm-dsa.", "deepseek2."), v, kd) for k, v, kd in kv]
        d = self.root / "arch"
        d.mkdir()
        a = audit(write_gguf(d, kv, ts, arch="deepseek2"))
        self.assertEqual(status(a)["A1"], "FAIL")

    def test_a2_geometry_mismatch(self):
        a = self.build(self.mut_kv("glm-dsa.rope.freq_base", 1e6))
        self.assertOnly(a, {"A2": "FAIL", "A11": "WARN"})
        a = self.build(self.mut_kv("glm-dsa.attention.key_length_mla", None))
        self.assertOnly(a, {"A2": "FAIL", "A11": "WARN"})

    def test_a2_rope_scaling_fails(self):
        def f(kv, ts):
            return kv + [("glm-dsa.rope.scaling.type", "yarn", "str"), ("glm-dsa.rope.scaling.factor", 4.0, "f32")], ts
        a = self.build(f)
        self.assertEqual(status(a)["A2"], "FAIL")

    def test_no_mtp_is_warn(self):
        def f(kv, ts):
            kv = [(k, (78 if k == "glm-dsa.block_count" else v), kd) for k, v, kd in kv if not k.endswith("nextn_predict_layers")]
            return kv, ts
        a = self.build(f, n_blocks=78)
        self.assertOnly(a, {"A2": "WARN", "A9": "WARN", "A11": "WARN"})
        self.assertEqual(a.placement["kv_tok"], [34048, 35840, 24 * 1152 + 6 * 512])

    # ---------------------------------------------------------------- A3
    def test_a3_types_absent_uses_bc_rule(self):
        a = self.build(self.mut_kv("glm-dsa.attention.indexer.types", None))
        self.assertOnly(a, {"A11": "WARN"})

    def test_a3_types_absent_short_context_fails(self):
        def f(kv, ts):
            kv = [(k, (202752 if k.endswith(".context_length") else v), kd) for k, v, kd in kv
                  if not k.endswith("indexer.types")]
            return kv, ts
        st = status(self.build(f))
        self.assertEqual(st["A3"], "FAIL")
        self.assertEqual(st["A2"], "FAIL")       # context_length is also an A2 key

    def test_a3_missing_indexer_on_full_layer_fails(self):
        def f(kv, ts):
            return kv, [x for x in ts if x[0] != "blk.30.indexer.attn_k.weight"]
        self.assertOnly(self.build(f), {"A3": "FAIL", "A11": "WARN"})

    def test_a3_indexers_on_all_layers_warn(self):
        def f(kv, ts):
            extra = []
            for i in range(78):
                if i not in FULL:
                    b = "blk.%d." % i
                    extra += [(b + "indexer.k_norm.weight", [128], 0), (b + "indexer.k_norm.bias", [128], 0),
                              (b + "indexer.proj.weight", [6144, 32], T["BF16"]),
                              (b + "indexer.attn_k.weight", [6144, 128], T["Q8_0"]),
                              (b + "indexer.attn_q_b.weight", [2048, 4096], T["Q8_0"])]
            return kv, ts + extra
        self.assertOnly(self.build(f), {"A3": "WARN", "A11": "WARN"})

    # ---------------------------------------------------------------- A4, A5, A13
    def test_a4_iq1_m_experts_warn_a5_only(self):
        # (no MMQ instance: the prompt path multiplies those matrices in FP16 and stays batched - a warning, not a fail)
        a = self.build(self.mut_types(lambda n: n.endswith(("gate_exps.weight", "up_exps.weight")), T["IQ1_M"]))
        self.assertOnly(a, {"A5": "WARN", "A11": "WARN"})

    def test_a4_iq4_nl_experts_fail(self):
        a = self.build(self.mut_types(lambda n: n == "blk.10.ffn_down_exps.weight", T["IQ4_NL"]))
        st = status(a)
        self.assertEqual(st["A4"], "FAIL")
        self.assertEqual(st["A5"], "PASS")      # IQ4_NL has an MMQ instance; the decode MoE kernel is what is missing

    def test_a4_gate_up_mismatch_fails(self):
        a = self.build(self.mut_types(lambda n: n == "blk.20.ffn_up_exps.weight", T["IQ2_S"]))
        st = status(a)
        self.assertEqual(st["A4"], "FAIL")
        self.assertEqual(st["A13"], "PASS")      # 2 classes

    def test_a13_more_than_eight_classes_warn(self):
        combos = [T["IQ2_XXS"], T["IQ2_XS"], T["IQ2_S"], T["IQ3_XXS"], T["IQ3_S"], T["Q2_K"], T["Q4_K"], T["IQ4_XS"],
                  T["IQ1_S"]]

        def f(kv, ts):
            out = []
            for n, ne, t in ts:
                m = ga.ROUTED_RE.match(n)
                if m and m.group(2) in ("gate", "up") and 3 <= int(m.group(1)) < 12:
                    t = combos[int(m.group(1)) - 3]
                out.append((n, ne, t))
            return kv, out
        a = self.build(f)
        self.assertEqual(len(a.classes()), 9)
        self.assertOnly(a, {"A13": "WARN", "A11": "WARN"})

    # ---------------------------------------------------------------- A6, A7, A8, A10
    def test_a6_upcast_types_pass_with_bytes(self):
        a = self.build(self.mut_types(lambda n: n.endswith(("attn_q_b.weight", "ffn_down_shexp.weight")), T["F16"]))
        self.assertOnly(a, {"A11": "WARN"})
        a6 = next(c for c in a.checks if c.id == "A6")
        self.assertTrue(any("F16 tensors upcast" in n for n in a6.notes), a6.notes)

    def test_a6_mxfp4_nonrouted_fails(self):
        a = self.build(self.mut_types(lambda n: n == "blk.5.attn_output.weight", T["MXFP4"]))
        self.assertEqual(status(a)["A6"], "FAIL")

    def test_a7_k_b_wrong_dims_fail(self):
        def f(kv, ts):
            return kv, [(n, ([128, 512, 64] if n == "blk.7.attn_k_b.weight" else ne), t) for n, ne, t in ts]
        self.assertEqual(status(self.build(f))["A7"], "FAIL")

    def test_a8_router_bf16_warns_bias_bf16_fails(self):
        a = self.build(self.mut_types(lambda n: n.endswith("ffn_gate_inp.weight"), T["BF16"]))
        self.assertOnly(a, {"A8": "WARN", "A11": "WARN"})
        a = self.build(self.mut_types(lambda n: n == "blk.9.exp_probs_b.bias", T["BF16"]))
        self.assertEqual(status(a)["A8"], "FAIL")

    def test_a10_bf16_head_warns(self):
        a = self.build(self.mut_types(lambda n: n == "output.weight", T["BF16"]))
        self.assertOnly(a, {"A10": "WARN", "A11": "WARN"})
        # placement charges the F32 upcast on the last GPU
        self.assertGreater(a.placement["nonrouted"][2], self.clean.placement["nonrouted"][2] + 3.0e9)

    # ---------------------------------------------------------------- A9, A11, A12, A14
    def test_a9_missing_eh_proj_warns(self):
        def f(kv, ts):
            return kv, [x for x in ts if x[0] != "blk.78.nextn.eh_proj.weight"]
        st = status(self.build(f))
        self.assertEqual(st["A9"], "WARN")

    def test_a9_mtp_head_types(self):
        """B15.4: blk.78's types are reported; IQ2/IQ3-class projections or experts warn with the remedy."""
        a9 = next(c for c in self.clean.checks if c.id == "A9")
        self.assertEqual(a9.status, "WARN")
        txt = "\n".join(a9.notes)
        self.assertIn("experts: IQ2_XXS 2.06 bpw, IQ3_XXS 3.06 bpw", txt)
        self.assertIn("nextn: Q8_0 8.50 bpw", txt)
        self.assertIn("STRATA_GLM_MTP_GGUF", txt)
        self.assertIn("extract_mtp_block.py", txt)
        # a Q4_K-or-better block (the remedy's donor) passes A9
        good = self.build(self.mut_types(lambda n: n.startswith("blk.78.") and n.endswith("_exps.weight"), T["Q4_K"]))
        st = status(good)
        self.assertEqual(st["A9"], "PASS", [c.notes for c in good.checks if c.id == "A9"])
        # one IQ3_S attention projection is enough to warn
        bad = self.build(self.mut_types(lambda n: n.startswith("blk.78.") and (n.endswith("_exps.weight") or
                                                                              n.endswith("attn_q_b.weight")),
                                        T["Q4_K"]))
        self.assertEqual(status(bad)["A9"], "PASS")
        bad = self.build(self.mut_types(lambda n: n == "blk.78.attn_q_b.weight", T["IQ3_S"]))
        notes = "\n".join(next(c for c in bad.checks if c.id == "A9").notes)
        self.assertIn("blk.78 tensors below 4 bits", notes)
        self.assertIn("IQ3_S", notes)

    def test_a11_official_template_and_bad_pre(self):
        tpl = "official template stand-in"
        ga_sha = ga.OFFICIAL_TEMPLATE_SHA256
        try:
            import hashlib
            ga.OFFICIAL_TEMPLATE_SHA256 = hashlib.sha256(tpl.encode()).hexdigest()
            a = self.build(self.mut_kv("tokenizer.chat_template", tpl))
            self.assertOnly(a, {})
        finally:
            ga.OFFICIAL_TEMPLATE_SHA256 = ga_sha
        a = self.build(self.mut_kv("tokenizer.ggml.pre", "qwen35"))
        self.assertEqual(status(a)["A11"], "FAIL")

    def test_a12_size_and_metadata_only_first_shard(self):
        a = self.build(n_shards=7, meta_first=True)
        self.assertOnly(a, {"A11": "WARN"})
        a12 = next(c for c in a.checks if c.id == "A12")
        self.assertTrue(any("metadata-only" in n for n in a12.notes))
        self.assertEqual(len(a.shards[0].tensors), 0)
        # 238 GB-class file but a 200 GB limit
        self.assertEqual(status(self.build(opts={"max_gb": 200.0}))["A12"], "FAIL")
        # fallback role: no default size limit
        self.assertEqual(status(self.build(opts={"role": "fallback", "max_gb": None}))["A12"], "PASS")

    def test_a12_missing_shard_fails_to_load(self):
        kv, ts = model_spec()
        d = self.root / "missing"
        d.mkdir()
        paths = write_gguf(d, kv, ts)
        paths[3].unlink()
        with self.assertRaises(FileNotFoundError):
            ga.load_local([str(paths[0])])

    def test_a14_closure_and_ram_budget(self):
        a = self.build(extra_bytes=3 * 10**9)            # 3 GB of unexplained bytes: > 0.5%
        self.assertEqual(status(a)["A14"], "FAIL")
        a = self.build(opts={"mem_available_gb": 150.0})  # the pinned tier no longer fits
        self.assertEqual(status(a)["A14"], "FAIL")
        a = self.build(opts={"gpu_usable_gb": 5.0})       # no expert pool
        self.assertEqual(status(a)["A14"], "FAIL")

    # ---------------------------------------------------------------- dump round trip, CLI
    def test_dump_round_trip_and_cli(self):
        kv, ts = model_spec()
        d = self.root / "cli" / "UD-IQ2_M"
        d.mkdir(parents=True)
        paths = write_gguf(d, kv, ts)
        out = self.root / "facts"
        dump = self.root / "glm53_UD-IQ2_M_header.json"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
            rc = ga.main([str(paths[0]), "--out-dir", str(out), "--dump-header", str(dump)])
        self.assertEqual(rc, 0)
        self.assertIn("Verdict: **PASS**", buf.getvalue())
        names = sorted(p.name for p in out.iterdir())
        self.assertEqual(names, ["glm53_UD-IQ2_M_audit.md", "glm53_UD-IQ2_M_audit.tsv",
                                 "glm53_UD-IQ2_M_chat_template.jinja"])
        doc = json.loads(dump.read_text())
        self.assertEqual(doc["shards"][0]["kv"]["tokenizer.ggml.tokens"], {"__strarray__": ga.VOCAB})
        buf2 = io.StringIO()
        with contextlib.redirect_stdout(buf2), contextlib.redirect_stderr(io.StringIO()):
            rc2 = ga.main([str(dump), "--no-write"])
        self.assertEqual(rc2, 0)
        strip = lambda s: s.split("\n", 1)[1]   # noqa: E731  (title line names the source)
        self.assertEqual(strip(buf.getvalue()), strip(buf2.getvalue()))
        # a FAIL exits 1
        bad = self.root / "bad"
        bad.mkdir()
        kv2 = [(k, ("qwen35" if k == "tokenizer.ggml.pre" else v), kd) for k, v, kd in kv]
        p2 = write_gguf(bad, kv2, ts)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(ga.main([str(p2[0]), "--no-write"]), 1)
            self.assertEqual(ga.main([str(self.root / "nope.gguf"), "--no-write"]), 2)


@unittest.skipUnless(HAVE_GGUF, "gguf-py needed")
class ParserTest(unittest.TestCase):
    def test_type_table_matches_gguf_py(self):
        for t, (bs, ts) in gguf.GGML_QUANT_SIZES.items():
            self.assertIn(int(t), ga.GGML_TYPES, t.name)
            self.assertEqual(ga.GGML_TYPES[int(t)][1:], (bs, ts), t.name)
            self.assertEqual(ga.GGML_TYPES[int(t)][0], t.name)

    def test_parser_matches_gguf_reader_on_a_full_file(self):
        """A small file with tensor data: our streaming parser vs gguf-py's GGUFReader."""
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "small.gguf"
            w = gguf.GGUFWriter(p, arch="glm-dsa")
            w.add_uint32("glm-dsa.block_count", 2)
            w.add_float32("glm-dsa.rope.freq_base", 8e6)
            w.add_array("glm-dsa.attention.indexer.types", [True, False])
            w.add_string("tokenizer.chat_template", "x" * 100)
            w.add_array("tokenizer.ggml.tokens", ["a"] * 300)
            w.add_tensor("a.weight", np.arange(64, dtype=np.float32).reshape(2, 32))
            w.add_tensor("b.weight", np.zeros((4, 256), dtype=np.float16))
            w.write_header_to_file()
            w.write_kv_data_to_file()
            w.write_tensors_to_file()
            w.close()
            sh = ga.load_local([str(p)])[0]
            r = gguf.GGUFReader(p)
            self.assertEqual([t.name for t in sh.tensors], [t.name for t in r.tensors])
            for a, b in zip(sh.tensors, r.tensors):
                self.assertEqual(a.dims, [int(x) for x in b.shape])
                self.assertEqual(a.type, int(b.tensor_type))
                self.assertEqual(a.nbytes, int(b.n_bytes))
            self.assertEqual(sh.kv["glm-dsa.attention.indexer.types"], [True, False])
            self.assertEqual(len(sh.kv["tokenizer.ggml.tokens"]), 300)
            self.assertAlmostEqual(sh.kv["glm-dsa.rope.freq_base"], 8e6)

    def test_mmq_types_read_from_cmake(self):
        got = ga.mmq_types()
        self.assertIn(T["IQ2_XXS"], got)
        self.assertNotIn(T["IQ1_M"], got)


class PureTest(unittest.TestCase):
    def test_is_norm(self):
        for n in ("blk.0.attn_norm.weight", "blk.3.indexer.k_norm.bias", "blk.78.nextn.enorm.weight",
                  "output_norm.weight", "blk.78.nextn.shared_head_norm.weight"):
            self.assertTrue(ga.is_norm(n), n)
        for n in ("blk.0.attn_q_a.weight", "blk.3.exp_probs_b.bias", "blk.78.nextn.eh_proj.weight"):
            self.assertFalse(ga.is_norm(n), n)

    def test_default_name(self):
        self.assertEqual(ga.default_name(["unsloth/GLM-5.3-GGUF", "UD-IQ2_M/*.gguf"]), "glm53_UD-IQ2_M")
        self.assertEqual(ga.default_name(["glm53_UD-Q2_K_XL_header.json"]), "glm53_UD-Q2_K_XL")

    def test_type_bytes(self):
        self.assertEqual(ga.type_bytes(T["IQ2_XXS"], 6144 * 2048), 6144 * 2048 // 256 * 66)
        self.assertEqual(ga.type_bytes(T["BF16"], 10), 20)


if __name__ == "__main__":
    unittest.main()

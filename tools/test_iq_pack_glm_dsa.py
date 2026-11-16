"""tools/test_iq_pack_glm_dsa.py - the glm-dsa port B1.6: the glm-dsa (GLM-5.3 744B) load-only pack.

Builds tiny synthetic glm-dsa GGUFs with gguf-py (split into shards, the first one metadata-only like unsloth's) and
checks what tools/iq_pack.py writes for them:

  * index.txt: every non-routed tensor, served natively only when the engine's mv() has its type, the shared layers'
    indexer tensors left out, blk.<trunk> (the NextN/MTP block) indexed and marked;
  * dense.bin: only kinds 2 (F32) and 4 (BF16), never 5 (F16); every upcast (F16, Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL)
    byte-exact against gguf-py's dequantization rounded to the nearest even BF16 by an independent oracle;
  * native_experts.txt: one row per MoE layer plus the MTP layer, absolute offsets equal to the shard directories,
    each row naming its own shard (v4 when a layer straddles shards);
  * compat-bf16.json: every upcast with its bytes;
  * refusals: gate/up type mismatch, a type outside mv() and the upcast set, an unknown tensor name.

CPU only: no model file, no GPU.  Run: python3 -m unittest tools/test_iq_pack_glm_dsa.py
"""
from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch
import zlib

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "strata"))

try:
    import gguf  # noqa: F401
except ImportError:
    from _paths import add_gguf_py
    add_gguf_py()
from gguf import GGUFWriter, GGMLQuantizationType as Q, GGML_QUANT_SIZES, quants  # noqa: E402

import iq_pack  # noqa: E402
import glm_dsa_pack  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402

# tiny geometry: K/IQ types need ne0 % 256 == 0, the 32-blocks ne0 % 32 == 0
EMBD, QLORA, QOUT, KVA, NOPE, KVL, VHEAD, NHEAD, FF, FFE, NEXP, VOCAB = 256, 64, 256, 64, 32, 32, 32, 2, 512, 256, 4, 64
IDX_K, IDX_QB, IDX_H = 32, 64, 4


def _rng(name: str) -> np.random.Generator:
    return np.random.default_rng(zlib.crc32(name.encode()))


def _raw(shape: tuple, qtype: Q, name: str) -> np.ndarray:
    """Valid bytes of `qtype` for ggml shape `shape` (ne0 first), as the uint8 array GGUFWriter takes."""
    rng = _rng(name)
    np_shape = tuple(reversed(shape))
    if qtype in (Q.F32, Q.F16, Q.BF16, Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.Q8_0):
        vals = rng.standard_normal(np_shape).astype(np.float32) * 0.5
        if qtype == Q.F32:
            return vals
        return quants.quantize(vals, qtype)
    # the types gguf-py cannot quantize: random blocks with a sane fp16 scale where the block starts with one
    block, size = GGML_QUANT_SIZES[qtype]
    n_blocks = int(np.prod(shape)) // block
    b = rng.integers(0, 256, size=(n_blocks, size), dtype=np.uint8)
    if qtype in (Q.IQ4_NL, Q.IQ2_XXS, Q.IQ2_XS, Q.IQ2_S, Q.IQ3_XXS, Q.IQ3_S, Q.IQ1_S):
        b[:, 0:2] = np.frombuffer(np.full(n_blocks, 0.01, np.float16).tobytes(), np.uint8).reshape(n_blocks, 2)
    return b.reshape(*np_shape[:-1], shape[0] // block * size)


def tensors_for(n_trunk=4, lead=1, mtp=True, full=(0, 2), idx_all=False, types=None, extra=()):
    """[(name, ggml shape, type)] of a tiny glm-dsa model; `types` overrides a tensor suffix's type."""
    types = dict(types or {})
    t = lambda suffix, default: types.get(suffix, default)  # noqa: E731
    out = [("token_embd.weight", (EMBD, VOCAB), t("token_embd.weight", Q.Q4_K)),
           ("output_norm.weight", (EMBD,), Q.F32),
           ("output.weight", (EMBD, VOCAB), t("output.weight", Q.Q6_K))]
    n_all = n_trunk + (1 if mtp else 0)
    for il in range(n_all):
        p = "blk.%d." % il
        out += [(p + "attn_norm.weight", (EMBD,), t("attn_norm.weight", Q.F32)),
                (p + "attn_q_a_norm.weight", (QLORA,), Q.F32),
                (p + "attn_kv_a_norm.weight", (KVL,), Q.F32),
                (p + "attn_q_a.weight", (EMBD, QLORA), t("attn_q_a.weight", Q.Q8_0)),
                (p + "attn_q_b.weight", (QLORA, QOUT), t("attn_q_b.weight", Q.Q8_0)),
                (p + "attn_kv_a_mqa.weight", (EMBD, KVA), t("attn_kv_a_mqa.weight", Q.Q8_0)),
                (p + "attn_k_b.weight", (NOPE, KVL, NHEAD), t("attn_k_b.weight", Q.Q8_0)),
                (p + "attn_v_b.weight", (KVL, VHEAD, NHEAD), t("attn_v_b.weight", Q.Q8_0)),
                (p + "attn_output.weight", (QOUT, EMBD), t("attn_output.weight", Q.Q4_K)),
                (p + "ffn_norm.weight", (EMBD,), Q.F32)]
        if il >= n_trunk or idx_all or il in full:
            out += [(p + "indexer.k_norm.weight", (IDX_K,), Q.F32),
                    (p + "indexer.k_norm.bias", (IDX_K,), Q.F32),
                    (p + "indexer.proj.weight", (EMBD, IDX_H), t("indexer.proj.weight", Q.F32)),
                    (p + "indexer.attn_k.weight", (EMBD, IDX_K), t("indexer.attn_k.weight", Q.Q8_0)),
                    (p + "indexer.attn_q_b.weight", (QLORA, IDX_QB), t("indexer.attn_q_b.weight", Q.BF16))]
        if il < lead:
            out += [(p + "ffn_gate.weight", (EMBD, FF), t("ffn_gate.weight", Q.Q4_K)),
                    (p + "ffn_up.weight", (EMBD, FF), t("ffn_up.weight", Q.Q4_K)),
                    (p + "ffn_down.weight", (FF, EMBD), t("ffn_down.weight", Q.Q6_K))]
        else:
            out += [(p + "ffn_gate_inp.weight", (EMBD, NEXP), t("ffn_gate_inp.weight", Q.F32)),
                    (p + "exp_probs_b.bias", (NEXP,), Q.F32),
                    (p + "ffn_gate_exps.weight", (EMBD, FFE, NEXP), t("ffn_gate_exps.weight", Q.IQ2_XXS)),
                    (p + "ffn_up_exps.weight", (EMBD, FFE, NEXP), t("ffn_up_exps.weight", Q.IQ2_XXS)),
                    (p + "ffn_down_exps.weight", (FFE, EMBD, NEXP), t("ffn_down_exps.weight", Q.Q4_K)),
                    (p + "ffn_gate_shexp.weight", (EMBD, FFE), t("ffn_gate_shexp.weight", Q.Q4_K)),
                    (p + "ffn_up_shexp.weight", (EMBD, FFE), t("ffn_up_shexp.weight", Q.Q4_K)),
                    (p + "ffn_down_shexp.weight", (FFE, EMBD), t("ffn_down_shexp.weight", Q.Q6_K))]
        if il == n_trunk and mtp:
            out += [(p + "nextn.eh_proj.weight", (2 * EMBD, EMBD), t("nextn.eh_proj.weight", Q.Q8_0)),
                    (p + "nextn.enorm.weight", (EMBD,), Q.F32),
                    (p + "nextn.hnorm.weight", (EMBD,), Q.F32),
                    (p + "nextn.shared_head_norm.weight", (EMBD,), Q.F32)]
    return out + list(extra)


def write_model(root: Path, tensors, n_shards=3, metadata_only_first=True, n_trunk=4, lead=1, mtp=True,
                full_types=(True, False, True, False), ctx=202752, stem="GLM-5.3-UD-TEST"):
    """Writes the shards; returns shard 1's path.  Shard 1 holds the metadata (and no tensors when
    `metadata_only_first`); the tensors are split in file order over the rest.  Each later shard gets a different
    header length so equal data_starts cannot hide an offset bug."""
    names = [root / ("%s-%05d-of-%05d.gguf" % (stem, i, n_shards)) for i in range(1, n_shards + 1)] \
        if n_shards > 1 else [root / ("%s.gguf" % stem)]
    data = [(n, _raw(s, q, n), q) for n, s, q in tensors]
    def split(items, k):
        cuts = [len(items) * i // k for i in range(k + 1)]
        return [items[cuts[i]:cuts[i + 1]] for i in range(k)]
    if n_shards == 1:
        parts = [data]
    elif metadata_only_first:
        parts = [[]] + split(data, n_shards - 1)
    else:
        parts = split(data, n_shards)
    for i, (path, part) in enumerate(zip(names, parts)):
        w = GGUFWriter(path, "glm-dsa")
        if i == 0:
            w.add_uint32("glm-dsa.block_count", n_trunk + (1 if mtp else 0))
            if mtp:
                w.add_uint32("glm-dsa.nextn_predict_layers", 1)
            w.add_uint32("glm-dsa.leading_dense_block_count", lead)
            w.add_uint32("glm-dsa.expert_count", NEXP)
            w.add_uint32("glm-dsa.context_length", ctx)
            w.add_uint32("glm-dsa.embedding_length", EMBD)
            if full_types is not None:
                w.add_array("glm-dsa.attention.indexer.types", [bool(x) for x in full_types])
        else:
            w.add_string("general.name", "p" * (37 * i))
        w.add_uint16("split.no", i)
        w.add_uint16("split.count", n_shards)
        for n, raw, q in part:
            w.add_tensor(n, raw, raw_dtype=q)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
    return names[0]


def run_pack(first: Path, out: Path, *extra) -> tuple[int, str]:
    (out / "tokenizer").mkdir(parents=True, exist_ok=True)
    for name in ("vocab.json", "chat_template.jinja"):   # the tokenizer export is tested elsewhere
        (out / "tokenizer" / name).touch()
    buf = io.StringIO()
    with patch.object(sys, "argv", ["iq_pack.py", "--gguf", str(first), "--out", str(out), "--compat-bf16", *extra]):
        with contextlib.redirect_stdout(buf):
            rc = iq_pack.main()
    return rc, buf.getvalue()


def rne_bf16(values: np.ndarray) -> np.ndarray:
    """An oracle independent of the packer's encoder: the nearest BF16, the even one at an exact tie."""
    bits = np.ascontiguousarray(values, dtype=np.float32).view(np.uint32).ravel()
    hi, lo = bits >> 16, bits & 0xFFFF
    up = (lo > 0x8000) | ((lo == 0x8000) & ((hi & 1) == 1))
    return (hi + up).astype(np.uint16)


def all_tensors(first: Path) -> dict:
    """name -> (shard path, TensorInfo, data_start) over every shard of the model."""
    found = {}
    for p in iq_pack.Model(first).paths:
        g = GGUFFile(p)
        for t in g.tensors:
            found[t.name] = (p, t, g.data_start)
    return found


def gguf_bytes(found: dict, name: str) -> np.ndarray:
    p, t, ds = found[name]
    with open(p, "rb") as f:
        f.seek(ds + t.offset)
        return np.frombuffer(f.read(t.expected_bytes()), dtype=np.uint8)


class GlmDsaLists(unittest.TestCase):
    def test_architecture_accepted(self):
        self.assertIn("glm-dsa", iq_pack.ARCHITECTURES)

    def test_lists_are_disjoint_and_router_is_f32(self):
        self.assertFalse(set(glm_dsa_pack.GLMDSA_BF16) & set(glm_dsa_pack.GLMDSA_F32))
        self.assertFalse(set(glm_dsa_pack.GLMDSA_BF16) & set(glm_dsa_pack.GLMDSA_NATIVE_OK))
        self.assertFalse(set(glm_dsa_pack.GLMDSA_F32) & set(glm_dsa_pack.GLMDSA_NATIVE_OK))
        # G23: the router stays F32 (Maya's glm5-next list rounds it to BF16)
        self.assertEqual(glm_dsa_pack.plan_tensor("blk.3.ffn_gate_inp.weight", "F32"), ("copy", "F32"))
        self.assertEqual(glm_dsa_pack.plan_tensor("blk.3.exp_probs_b.bias", "F32"), ("copy", "F32"))

    def test_classification(self):
        P = glm_dsa_pack.plan_tensor
        for n in ("attn_k_b", "attn_v_b", "indexer.attn_k", "indexer.attn_q_b", "indexer.proj"):
            self.assertEqual(P("blk.7.%s.weight" % n, "Q8_0", headwise=False), ("convert", "BF16"), n)
            self.assertEqual(P("blk.7.%s.weight" % n, "BF16"), ("copy", "BF16"), n)
        # Q8_0 W_kb / W_vb stay native by default (headwise MLA GEMVs); the other BF16-list tensors convert as before
        with patch.dict("os.environ", {"MAYA_PACK_NATIVE32": ""}):
            for n in ("attn_k_b", "attn_v_b"):
                self.assertEqual(P("blk.7.%s.weight" % n, "Q8_0"), ("native", "Q8_0"), n)
            self.assertEqual(P("blk.7.indexer.attn_k.weight", "Q8_0"), ("convert", "BF16"))
        with patch.dict("os.environ", {"MAYA_PACK_NATIVE32": "0"}):
            self.assertEqual(P("blk.7.attn_k_b.weight", "Q8_0"), ("convert", "BF16"))
        for n in ("attn_norm.weight", "ffn_norm.weight", "attn_q_a_norm.weight", "attn_kv_a_norm.weight",
                  "indexer.k_norm.weight", "indexer.k_norm.bias", "nextn.enorm.weight", "nextn.hnorm.weight",
                  "nextn.shared_head_norm.weight"):
            self.assertEqual(P("blk.78.%s" % n, "F32"), ("copy", "F32"), n)
            self.assertEqual(P("blk.78.%s" % n, "F16"), ("convert", "F32"), n)
        self.assertEqual(P("output_norm.weight", "F32"), ("copy", "F32"))
        for n in ("attn_q_a", "attn_q_b", "attn_kv_a_mqa", "attn_output", "ffn_gate", "ffn_up", "ffn_down",
                  "ffn_gate_shexp", "ffn_up_shexp", "ffn_down_shexp", "nextn.eh_proj"):
            self.assertEqual(P("blk.2.%s.weight" % n, "Q4_K"), ("native", "Q4_K"), n)
            self.assertEqual(P("blk.2.%s.weight" % n, "IQ2_XXS"), ("native", "IQ2_XXS"), n)
            for up in glm_dsa_pack.UPCAST_TYPES:
                self.assertEqual(P("blk.2.%s.weight" % n, up), ("convert", "BF16"), (n, up))
            self.assertEqual(P("blk.2.%s.weight" % n, "MXFP4")[0], "refuse", n)
        self.assertEqual(P("token_embd.weight", "Q4_K"), ("native", "Q4_K"))
        self.assertEqual(P("output.weight", "F16"), ("convert", "BF16"))
        self.assertEqual(P("blk.2.hc_attn_fn.weight", "BF16")[0], "refuse")     # no glm5-next names

    def test_indexer_types(self):
        md = {"glm-dsa.attention.indexer.types": [True, False, True]}
        self.assertEqual(glm_dsa_pack.indexer_full(md, 3)[0], [True, False, True])
        # no key: llama.cpp's BC rule (n_ctx_train >= 1M -> the GLM-5.2 pattern, else all full)
        full = glm_dsa_pack.indexer_full({"glm-dsa.context_length": 1048576}, 78)[0]
        self.assertEqual([i for i, f in enumerate(full) if f], [0, 1, 2] + list(range(6, 78, 4)))
        self.assertEqual(sum(full), 21)
        self.assertTrue(all(glm_dsa_pack.indexer_full({"glm-dsa.context_length": 202752}, 78)[0]))


class GlmDsaPack(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(ignore_cleanup_errors=True)
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def pack(self, tensors, expect=0, **kw):
        first = write_model(self.root, tensors, **kw)
        out = self.root / "pack"
        rc, log = run_pack(first, out)
        self.assertEqual(rc, expect, log)
        return first, out, log

    def check_dense(self, first, out):
        """Every dense.bin row: kind 2 or 4 only, F32/BF16 rows byte-exact with the GGUF (copies) or with gguf-py's
        dequantization + nearest-even (conversions).  Returns the rows."""
        header, rows = iq_pack.read_index(out / "index.txt")
        dense = (out / "dense.bin").read_bytes()
        found = all_tensors(first)
        conv = {c["name"]: c for c in json.loads((out / "compat-bf16.json").read_text())["tensors"]}
        for name, r in rows.items():
            self.assertIn(r[2], ("0", "2", "4"), name)       # never 5 (F16): the loader takes 0, 2, 4
            if r[2] == "0":
                self.assertEqual(r[4], "0")
                self.assertIn(found[name][1].type_name, glm_dsa_pack.MV_TYPES - {"F32", "BF16"}, name)
                continue
            off, size = int(r[3]), int(r[4])
            self.assertEqual(off % iq_pack.ALIGN, 0)
            got = dense[off:off + size]
            src = gguf_bytes(found, name)
            stype = found[name][1].type_name
            if r[2] == "4":
                if stype == "BF16":
                    self.assertNotIn(name, conv)
                    self.assertEqual(got, src.tobytes(), name)
                else:
                    values = quants.dequantize(src, Q[stype])
                    self.assertEqual(got, rne_bf16(values).tobytes(), name)
                    self.assertEqual(conv[name]["target"], "BF16")
                    self.assertEqual(conv[name]["source_type"], stype)
                    self.assertEqual(conv[name]["bytes"], size)
                    self.assertEqual(conv[name]["bytes_added"], size - src.size)
            else:
                if stype == "F32":
                    self.assertEqual(got, src.tobytes(), name)
                else:
                    self.assertEqual(got, np.asarray(quants.dequantize(src, Q[stype]), np.float32).tobytes(), name)
                    self.assertEqual(conv[name]["target"], "F32")
            self.assertEqual(size, found[name][1].elements * (2 if r[2] == "4" else 4), name)
        return header, rows

    def check_experts(self, first, out, layers):
        lines = (out / "native_experts.txt").read_text().splitlines()
        self.assertRegex(lines[0], r"absolute offsets in %s," % first.name.replace(".", r"\."))
        self.assertIn("(n_expert %d," % NEXP, lines[0])
        found = all_tensors(first)
        rows = [l.split() for l in lines if l and not l.startswith("#")]
        self.assertEqual([int(r[0]) for r in rows], layers)
        offset = 0
        for r in rows:
            il = int(r[0])
            ts = [found["blk.%d.ffn_%s_exps.weight" % (il, role)] for role in ("gate", "up", "down")]
            self.assertEqual(int(r[1]), ts[0][1].type_id)
            self.assertEqual(int(r[2]), ts[2][1].type_id)
            self.assertEqual(int(r[3]), offset)
            blob = sum(t[1].expected_bytes() for t in ts) // NEXP
            self.assertEqual(int(r[4]), blob)
            offset += blob * NEXP
            for k, t in enumerate(ts):
                self.assertEqual(int(r[5 + k]), t[2] + t[1].offset, (il, k))
            shards = [t[0].name for t in ts]
            if len(set(shards)) == 1:
                self.assertEqual(r[8:], [] if shards[0] == first.name else [shards[0]])
            else:
                self.assertEqual(r[8:], shards)
        return lines

    def test_metadata_only_first_shard_with_mtp(self):
        first, out, log = self.pack(tensors_for())
        self.assertEqual(len(GGUFFile(first).tensors), 0)
        self.assertIn("metadata only", log)
        header, rows = self.check_dense(first, out)
        self.assertTrue(any(h.startswith("# glm-dsa: trunk 4, dense lead 1, experts 4, mtp 4") for h in header))
        # the NextN block is indexed (blk.4) with its own full indexer; no routed expert row in index.txt
        for n in ("blk.4.nextn.eh_proj.weight", "blk.4.nextn.enorm.weight", "blk.4.indexer.attn_k.weight",
                  "blk.4.attn_k_b.weight", "blk.4.ffn_gate_inp.weight"):
            self.assertIn(n, rows)
        self.assertFalse(any(n.endswith("_exps.weight") for n in rows))
        self.assertEqual(rows["blk.4.nextn.eh_proj.weight"][2], "0")        # Q8_0: served natively
        self.assertEqual(rows["blk.3.ffn_gate_inp.weight"][2], "2")         # router F32 (G23)
        self.assertEqual(rows["blk.0.attn_k_b.weight"][2], "0")             # Q8_0 k_b: native (headwise)
        self.assertEqual(rows["blk.0.attn_k_b.weight"][7:9], [str(NOPE * KVL), str(NHEAD)])
        self.assertEqual(rows["blk.0.indexer.proj.weight"][2], "4")         # F32 -> BF16 (the BF16 list)
        self.assertEqual(rows["blk.0.indexer.attn_q_b.weight"][2], "4")     # BF16 as stored
        lines = self.check_experts(first, out, [1, 2, 3, 4])
        self.assertIn("layer 4 is the NextN/MTP block", lines[1])
        # the shard directories are not modified
        self.assertEqual(GGUFFile(first).metadata["general.architecture"], "glm-dsa")

    def test_each_upcast_type_on_native_tensors(self):
        slots = ("attn_q_b.weight", "attn_output.weight", "ffn_gate_shexp.weight", "ffn_up_shexp.weight",
                 "ffn_down_shexp.weight", "ffn_gate.weight", "ffn_up.weight", "ffn_down.weight",
                 "nextn.eh_proj.weight", "attn_k_b.weight", "attn_v_b.weight", "indexer.attn_k.weight")
        for q in (Q.F16, Q.Q4_0, Q.Q4_1, Q.Q5_0, Q.Q5_1, Q.IQ4_NL):
            with self.subTest(q=q.name):
                for p in self.root.glob("*"):
                    if p.is_file():
                        p.unlink()
                first, out, log = self.pack(tensors_for(types={s: q for s in slots}))
                _, rows = self.check_dense(first, out)
                conv = json.loads((out / "compat-bf16.json").read_text())
                names = {c["name"] for c in conv["tensors"] if c["source_type"] == q.name}
                for s in slots:
                    for n in [r for r in rows if r.startswith("blk.") and r.split(".", 2)[2] == s]:
                        self.assertEqual(rows[n][2], "4", n)
                        self.assertIn(n, names, n)
                self.assertEqual(conv["bytes_added"], sum(c["bytes_added"] for c in conv["tensors"]))
                self.assertIn("compat-bf16: %s->BF16" % q.name, log)

    def test_f16_norms_and_router_become_f32(self):
        first, out, _ = self.pack(tensors_for(types={"attn_norm.weight": Q.F16, "ffn_gate_inp.weight": Q.BF16}))
        _, rows = self.check_dense(first, out)
        self.assertEqual(rows["blk.0.attn_norm.weight"][2], "2")
        self.assertEqual(rows["blk.2.ffn_gate_inp.weight"][2], "2")

    def test_indexers_on_all_layers_tolerated(self):
        first, out, log = self.pack(tensors_for(idx_all=True))
        self.assertIn("indexer tensors on 2 shared layer(s) (1,3)", log)
        _, rows = self.check_dense(first, out)
        self.assertNotIn("blk.1.indexer.attn_k.weight", rows)
        self.assertNotIn("blk.3.indexer.proj.weight", rows)
        self.assertIn("blk.2.indexer.attn_k.weight", rows)
        dropped = json.loads((out / "compat-bf16.json").read_text())["dropped_shared_indexer"]
        self.assertEqual(len(dropped), 10)

    def test_missing_full_layer_indexer_refused(self):
        self.pack(tensors_for(full=(0,)), expect=1)                # layer 2 is full by indexer.types

    def test_no_indexer_types_key_uses_bc_rule(self):
        # 1M context: the GLM-5.2 pattern -> layers 0,1,2 full, 3 shared
        first, out, log = self.pack(tensors_for(full=(0, 1, 2)), full_types=None, ctx=1048576)
        self.assertIn("GLM-5.2 default", log)

    def test_single_file_and_straddling_layers(self):
        first, out, _ = self.pack(tensors_for(), n_shards=1)
        self.check_dense(first, out)
        self.check_experts(first, out, [1, 2, 3, 4])
        for p in self.root.glob("*"):
            if p.is_file():
                p.unlink()
        # many shards: some layers' gate/up/down land in different files (v4 rows)
        first, out, _ = self.pack(tensors_for(), n_shards=9, metadata_only_first=False)
        lines = self.check_experts(first, out, [1, 2, 3, 4])
        self.assertTrue(lines[0].startswith("# strata native experts v4"))

    def test_no_mtp_block(self):
        first, out, log = self.pack(tensors_for(mtp=False), mtp=False)
        self.assertIn("no NextN/MTP block", log)
        self.check_experts(first, out, [1, 2, 3])

    def test_gate_up_type_mismatch_refused(self):
        _, _, log = self.pack(tensors_for(types={"ffn_up_exps.weight": Q.IQ2_XS}), expect=1)
        self.assertIn("gate and up differ in type", log)

    def test_unsupported_non_routed_type_refused(self):
        _, _, log = self.pack(tensors_for(types={"attn_output.weight": Q.MXFP4}), expect=1)
        self.assertIn("blk.0.attn_output.weight (MXFP4)", log)

    def test_unknown_tensor_refused(self):
        _, _, log = self.pack(tensors_for(extra=[("blk.0.hc_attn_fn.weight", (EMBD, 32), Q.BF16)]), expect=1)
        self.assertIn("blk.0.hc_attn_fn.weight", log)

    def test_routed_types_outside_moe_kernels_warn(self):
        # (Q8_0 is served by Maya v1.1.55's MoE kernels and MMQ, so MOE_TYPES lists it: Q4_0 is still outside)
        _, _, log = self.pack(tensors_for(types={"ffn_down_exps.weight": Q.Q4_0}))
        self.assertIn("outside the GPU MoE kernels", log)

    def test_experts_bin(self):
        first = write_model(self.root, tensors_for(), n_shards=3)
        out = self.root / "pack"
        rc, log = run_pack(first, out, "--experts-bin")
        self.assertEqual(rc, 0, log)
        found = all_tensors(first)
        rows = [l.split() for l in (out / "native_experts.txt").read_text().splitlines() if not l.startswith("#")]
        blob = (out / "experts.bin").read_bytes()
        for r in rows:
            il, off, b = int(r[0]), int(r[3]), int(r[4])
            parts = [gguf_bytes(found, "blk.%d.ffn_%s_exps.weight" % (il, x)).reshape(NEXP, -1)
                     for x in ("gate", "up", "down")]
            self.assertEqual(blob[off:off + b * NEXP], np.concatenate(parts, axis=1).tobytes())


if __name__ == "__main__":
    unittest.main()

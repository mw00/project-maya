"""tools/k31/pack_gguf.py on the synthetic tiny fixture (written into a scratch directory, as CI does), each packer
refusal with its own message, tensors.txt against the GGUF's own directory, a pack made through a link to the
weights (the prune sidecar beside the link), and a cut file.

    python -m unittest tools.test_k31_pack_gguf
    KIMI_K31_REAL_HEADER=K31-cond128-IQ1_S.gguf python -m unittest tools.test_k31_pack_gguf

KIMI_K31_REAL_HEADER names a real K3.1 GGUF (a header-only cut works; its <file>.prune.json is read when present):
it is packed and checked as the tiny file is.  Without it that case is skipped.
"""
from __future__ import annotations

import contextlib
import copy
import dataclasses
import io
import json
import os
import pathlib
import shutil
import sys
import tempfile
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))                    # tools/: gguf_reader, kimi_k31_synth_gguf, k31/
sys.path.insert(0, str(HERE.parent / "ref"))     # ref/: kimi_k31

from gguf_reader import BLOCK_GEOMETRY, GGUFFile  # noqa: E402
from k31 import pack_gguf as P  # noqa: E402
import kimi_k31 as K  # noqa: E402

REAL_HEADER = os.environ.get("KIMI_K31_REAL_HEADER", "")


def write_tiny(path: pathlib.Path) -> pathlib.Path:
    import kimi_k31_synth_gguf as synth
    vals, _, _ = synth.make_weights(synth.TINY, synth.SEED)
    md, ts = synth.to_gguf(vals, synth.TINY)
    synth.write_gguf(path, md, ts)
    return path


def read_pack(dest: pathlib.Path):
    lines = (dest / "k31.txt").read_text().splitlines()
    tensors = (dest / "tensors.txt").read_text().splitlines()
    return lines, tensors


def field(lines, key):
    """The rest of the first record that starts with `key` (which may hold a layer: "keep 3")."""
    return next(line[len(key) + 1:] for line in lines if line.startswith(key + " "))


class _Fake:
    """A parsed GGUF with its metadata or tensor directory changed (for P.GGUFFile)."""

    def __init__(self, real: GGUFFile, meta=None, tensors=None):
        self.metadata = meta if meta is not None else real.metadata
        self.tensors = tensors if tensors is not None else real.tensors
        self.data_start = real.data_start


class PackGguf(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = pathlib.Path(cls.tmp.name)
        cls.tiny_path = write_tiny(cls.root / "tiny.gguf")
        cls.tiny = GGUFFile(cls.tiny_path)
        cls.cfg = K.KimiK31Config(cls.tiny.metadata)
        cls.moe = list(range(cls.cfg.n_layer_dense_lead, cls.cfg.n_layers))

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ---- helpers
    def refused(self, needle, *, meta=None, tensors=None, path=None, header_only=False):
        """build_lines on tiny (or `path`) with its metadata / tensors replaced: the refusal, which must name `needle`."""
        real = self.tiny

        def fake(p):
            return _Fake(real if path is None else GGUFFile(p), meta, tensors)

        with mock.patch.object(P, "GGUFFile", fake), self.assertRaises(P.PackRefused) as cm:
            P.build_lines(path or self.tiny_path, header_only=header_only)
        msg = str(cm.exception)
        self.assertTrue(msg.startswith("maya-k31: pack: "), msg)
        self.assertIn(needle, msg)
        return msg

    def meta(self, **changes):
        m = dict(self.tiny.metadata)
        for k, v in changes.items():
            key = "kimi-k31." + k.replace("__", ".")
            if v is None:
                m.pop(key, None)
            else:
                m[key] = v
        return m

    def tensors(self, name, **changes):
        out = [dataclasses.replace(t) for t in self.tiny.tensors]
        hit = False
        for t in out:
            if t.name == name:
                hit = True
                for k, v in changes.items():
                    setattr(t, k, v)
        self.assertTrue(hit, name)
        return out

    def without(self, name):
        out = [t for t in self.tiny.tensors if t.name != name]
        self.assertEqual(len(out), len(self.tiny.tensors) - 1, name)
        return out

    def check_pack(self, model: pathlib.Path, dest: pathlib.Path, gguf: GGUFFile, header_only=False):
        """k31.txt's model_file and tensors.txt against the GGUF's own directory: every tensor once, absolute
        offset, bytes, type, dims outermost first."""
        lines, tensors = read_pack(dest)
        self.assertEqual(field(lines, "data"), "header-only" if header_only else "whole")
        rel = field(lines, "model_file")
        self.assertEqual(os.path.realpath(dest / rel), os.path.realpath(model))
        self.assertEqual(tensors[0], "file 0 %s" % rel)
        by_name = {}
        for line in tensors[1:]:
            t, name, dtype, fi, off, nb, *dims = line.split()
            self.assertEqual((t, fi), ("t", "0"), line)
            self.assertNotIn(name, by_name, "listed twice: " + name)
            by_name[name] = (dtype, int(off), int(nb), [int(d) for d in dims])
        self.assertEqual(sorted(by_name), sorted(t.name for t in gguf.tensors))
        for t in gguf.tensors:
            block = BLOCK_GEOMETRY[t.type_name]
            n = 1
            for d in t.shape:
                n *= int(d)
            want = (t.type_name, gguf.data_start + int(t.offset), n // block[0] * block[1],
                    [int(d) for d in reversed(t.shape)])
            self.assertEqual(by_name[t.name], want, t.name)
        return lines

    # ---- what is accepted
    def test_tiny_packs_and_tensors_txt_matches_the_file(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            dest = P.write_pack(self.tiny_path, self.root / "pack-tiny")
        self.assertIn("no expert_prune keys and no prune sidecar", err.getvalue())
        lines = self.check_pack(self.tiny_path, dest, self.tiny)
        c = self.cfg
        self.assertEqual(field(lines, "arch"), "kimi-k31")
        self.assertEqual(field(lines, "format"), "gguf")
        self.assertEqual(field(lines, "n_layers"), str(c.n_layers))
        self.assertEqual(field(lines, "rms_eps"), "%.9g" % c.eps_rms)
        self.assertEqual(field(lines, "lora_eps"), "%.9g" % c.eps_mla_latent)
        self.assertEqual(field(lines, "kda_l2_eps"), "%.9g" % c.eps_kda_l2)
        self.assertEqual(field(lines, "idx_ln_eps"), "%.9g" % c.eps_indexer_ln)
        self.assertEqual(field(lines, "router_renorm"), "%.9g" % c.eps_router_renorm)
        self.assertNotEqual(c.eps_mla_latent, c.eps_rms)   # (so lora_eps written as rms_eps would show)
        layers = [line for line in lines if line.startswith("layer ")]
        self.assertEqual(len(layers), c.n_layers)
        for il, line in enumerate(layers):
            attn = "kda" if c.is_recr(il) else "mla"
            if il < c.n_layer_dense_lead:
                self.assertEqual(line, "layer %d attn %s ffn dense n_exp 0 k 0 edim 0 eff %d sff 0 norm_div %d"
                                 % (il, attn, c.n_ff, c.n_expert_latent))
            else:
                self.assertEqual(line, "layer %d attn %s ffn moe n_exp %d k %d edim %d eff %d sff 128 norm_div %d"
                                 % (il, attn, c.n_expert, c.n_expert_used, c.n_expert_latent, c.n_ff_exp,
                                    c.n_expert_latent))
        classes = [line for line in lines if line.startswith("class ")]
        self.assertEqual(classes, ["class %d 0 F32 %d blk.%d.ffn_gate_exps.weight blk.%d.ffn_up_exps.weight "
                                   "blk.%d.ffn_down_exps.weight" % (il, c.n_expert, il, il, il) for il in self.moe])
        keeps = [line for line in lines if line.startswith("keep ")]
        self.assertEqual(keeps, ["keep %d %s" % (il, " ".join(str(i) for i in range(c.n_expert))) for il in self.moe])
        self.assertEqual(field(lines, "prune"), "source_count %d method none" % c.n_expert)
        self.assertFalse(any(line.startswith("alog_tail") for line in lines))

    def test_class_format_follows_the_tensors(self):
        # every MoE layer's experts in Q8_0 (32-value blocks: the fixture's 128 / 64 widths fit): the class says Q8_0
        tensors = [dataclasses.replace(t) for t in self.tiny.tensors]
        for t in tensors:
            if t.name.endswith("_exps.weight"):
                t.type_name, t.type_id = "Q8_0", 8
        with mock.patch.object(P, "GGUFFile", lambda p: _Fake(self.tiny, None, tensors)):
            lines, tensor_lines, _ = P.build_lines(self.tiny_path)
        classes = [line for line in lines if line.startswith("class ")]
        self.assertEqual(len(classes), len(self.moe))
        for il, line in zip(self.moe, classes):
            self.assertTrue(line.startswith("class %d 0 Q8_0 %d " % (il, self.cfg.n_expert)), line)
        gate = next(x for x in tensor_lines if x[0] == "blk.1.ffn_gate_exps.weight")
        self.assertEqual(gate[1:4:2], ("Q8_0", 128 * 64 * 24 // 32 * 34))

    def test_a_link_to_the_weights_reads_the_sidecar_beside_the_link(self):
        # the weights elsewhere (a pool), a link and its sidecar here: the ids are the sidecar's, as maya-k31 reads them
        pool = self.root / "pool"
        here = self.root / "here"
        pool.mkdir()
        here.mkdir()
        target = pool / "cut.gguf"
        shutil.copyfile(self.tiny_path, target)
        link = here / "cut.gguf"
        link.symlink_to(target)
        ids = {str(il): [(7 * j + il) % 50 for j in range(self.cfg.n_expert)] for il in self.moe}
        for il in self.moe:
            self.assertEqual(len(set(ids[str(il)])), self.cfg.n_expert)
        side = {"arch": "kimi-k31", "tool": "prune v1", "source_expert_count": 50,
                "expert_count": self.cfg.n_expert, "layers": ids}
        pathlib.Path(str(link) + ".prune.json").write_text(json.dumps(side))
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err:
            dest = P.write_pack(link, None)
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(dest.resolve(), (here / "cut.maya-k31").resolve())
        lines = self.check_pack(link, dest, self.tiny)
        self.assertEqual(field(lines, "model_file"), "../cut.gguf")
        self.assertEqual(field(lines, "prune"), "source_count 50 method prune v1")
        for il in self.moe:
            self.assertEqual(field(lines, "keep %d" % il), " ".join(str(x) for x in ids[str(il)]))

    def test_prune_keys_in_the_file_and_a_sidecar_that_agrees(self):
        n, moe = self.cfg.n_expert, self.moe
        flat = [(3 * j + il) % 40 for il in moe for j in range(n)]
        meta = self.meta(expert_prune__keep=flat, expert_prune__source_count=40, expert_prune__method="usage")
        source, method, per, note = P._keep(self.root / "nothing.gguf", meta, self.cfg)
        self.assertEqual((source, method, note), (40, "usage", None))
        self.assertEqual(per[moe[1]], flat[n:2 * n])
        side = self.root / "agree.gguf"
        pathlib.Path(str(side) + ".prune.json").write_text(json.dumps(
            {"arch": "kimi-k31", "source_expert_count": 40, "expert_count": n,
             "layers": {str(il): per[il] for il in moe}}))
        self.assertEqual(P._keep(side, meta, self.cfg)[2], per)

    def test_a_cut_file_is_refused_unless_header_only(self):
        cut = self.root / "cut-short.gguf"
        cut.write_bytes(self.tiny_path.read_bytes()[:30_000_000])
        self.refused("cut-short.gguf is not whole: its tensor data ends at byte", path=cut)
        self.refused("the file has 30000000 (a download or copy still running?", path=cut)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            dest = P.write_pack(cut, self.root / "pack-cut", header_only=True)
        self.check_pack(cut, dest, GGUFFile(cut), header_only=True)
        # a whole file packs as whole with the flag too
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            dest = P.write_pack(self.tiny_path, self.root / "pack-flag", header_only=True)
        self.assertEqual(field(read_pack(dest)[0], "data"), "whole")

    @unittest.skipUnless(REAL_HEADER, "set KIMI_K31_REAL_HEADER to a real K3.1 GGUF (a header-only cut works)")
    def test_a_real_header(self):
        model = pathlib.Path(REAL_HEADER).expanduser()
        gguf = GGUFFile(model)
        header_only = os.stat(model).st_size < gguf.data_start + max(
            int(t.offset) + (t.expected_bytes() or 0) for t in gguf.tensors)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            dest = P.write_pack(model, self.root / "pack-real", header_only=header_only)
        lines = self.check_pack(model, dest, gguf, header_only=header_only)
        cfg = K.KimiK31Config(gguf.metadata)
        layers = [line for line in lines if line.startswith("layer ")]
        self.assertEqual(len(layers), cfg.n_layers)
        self.assertEqual(sum(1 for line in layers if " attn kda " in line), sum(1 for h in cfg.n_head_kv if h == 0))
        side = pathlib.Path(str(model) + ".prune.json")
        if side.is_file():
            d = json.loads(side.read_text())
            self.assertEqual(field(lines, "prune").split()[1], str(d["source_expert_count"]))
            for il in range(cfg.n_layer_dense_lead, cfg.n_layers):
                self.assertEqual(field(lines, "keep %d" % il), " ".join(str(x) for x in d["layers"][str(il)]))

    # ---- each refusal, with its own message
    def test_header_refusals(self):
        r = self.refused
        r("architecture 'kimi-k3', kimi-k31 only", meta={**self.tiny.metadata, "general.architecture": "kimi-k3"})
        r("kimi-k31.attention.indexer.shared is not a per-token indexer setting",
          meta=self.meta(attention__indexer__shared=1))
        r("index_rope_dim 64", meta=self.meta(attention__indexer__rope_dim=64))
        r("index_topk_freq 2", meta=self.meta(attention__indexer__topk_freq=2))
        r("index kpool 4 (a pooled indexer", meta=self.meta(attention__indexer__kpool=4))
        r("indexer LayerNorm eps is missing", meta=self.meta(attention__layer_norm_epsilon=None))
        mla = self.cfg.mla_layers()[0]
        types = [1] * self.cfg.n_layers
        types[mla] = 0
        r("layer %d shares another layer's indexer selection" % mla, meta=self.meta(attention__indexer__types=types))
        r("layer %d shares another layer's indexer selection" % mla, meta=self.meta(attention__indexer__types=[0]))
        r("attention.indexer.types has 5 entries for 9 layers", meta=self.meta(attention__indexer__types=[1] * 5))
        r("expert_count is missing", meta=self.meta(expert_count=None))
        r("MoE layers keep different expert counts (24, 32)", meta=self.meta(expert_count=[24, 32]))
        r("expert_feed_forward_length differs per layer (32, 64): per-layer expert widths (thin experts) are not "
          "packed yet", meta=self.meta(expert_feed_forward_length=[64] * 8 + [32]))
        r("expert_gating_func 1", meta=self.meta(expert_gating_func=1))
        # (rope_dim 0 and topk_freq 1 are K3.1's own values: accepted, though the oracle's config reads neither)
        P.assess(self.meta(attention__indexer__rope_dim=0, attention__indexer__topk_freq=1), self.tiny.tensors)
        # per-layer arrays that are all one value read as that value
        P.assess(self.meta(expert_count=[24] * 9, expert_feed_forward_length=[64] * 9), self.tiny.tensors)

    def test_tensor_refusals(self):
        r = self.refused
        r("blk.1.ffn_gate_exps.weight is missing", tensors=self.without("blk.1.ffn_gate_exps.weight"))
        r("blk.1.ffn_gate_exps.weight is not a stacked expert tensor",
          tensors=self.tensors("blk.1.ffn_gate_exps.weight", shape=[128, 64 * 24]))
        r("MoE layers keep different expert counts (20, 24)",
          tensors=self.tensors("blk.2.ffn_gate_exps.weight", shape=[128, 64, 20]))
        r("blk.3.ffn_up_exps.weight is missing", tensors=self.without("blk.3.ffn_up_exps.weight"))
        r("blk.1.ffn_down_exps.weight has ggml type Q9_9, which the pack does not read",
          tensors=self.tensors("blk.1.ffn_down_exps.weight", type_name="Q9_9"))
        r("blk.1.ffn_up_exps.weight is [128, 32, 24], the geometry says [128, 64, 24]",
          tensors=self.tensors("blk.1.ffn_up_exps.weight", shape=[128, 32, 24]))
        r("blk.1.ffn_down_exps.weight is [128, 64, 24], the geometry says [64, 128, 24]",
          tensors=self.tensors("blk.1.ffn_down_exps.weight", shape=[128, 64, 24]))
        r("layer 1 expert width 128 is not a multiple of 256 (IQ1_S)",
          tensors=self.tensors("blk.1.ffn_gate_exps.weight", type_name="IQ1_S", type_id=19))
        r("layer 1 expert width 64 is not a multiple of 256 (IQ1_S)",
          tensors=self.tensors("blk.1.ffn_down_exps.weight", type_name="IQ1_S", type_id=19))
        r("layer 1 expert width 64 is not a multiple of 256 (Q4_K)",
          tensors=self.tensors("blk.1.ffn_down_exps.weight", type_name="Q4_K", type_id=12))
        # a 32-value format at a width of 48 (the geometry and the slabs narrowed together)
        narrow = [dataclasses.replace(t) for t in self.tiny.tensors]
        for t in narrow:
            if t.name.endswith("_exps.weight"):
                t.type_name, t.type_id = "Q8_0", 8
                t.shape = [128, 48, 24] if "down" not in t.name else [48, 128, 24]
        r("layer 1 expert width 48 is not a multiple of 32 (Q8_0)",
          meta=self.meta(expert_feed_forward_length=48), tensors=narrow)
        r("blk.0.ssm_a is missing", tensors=self.without("blk.0.ssm_a"))
        r("blk.0.ssm_a is [8] for 6 KDA heads (A_log is sliced to the head count at conversion)",
          tensors=self.tensors("blk.0.ssm_a", shape=[8]))
        r("blk.4.ssm_a is [6, 1] for 6 KDA heads", tensors=self.tensors("blk.4.ssm_a", shape=[6, 1]))
        r("blk.1.ffn_gate_shexp.weight is missing", tensors=self.without("blk.1.ffn_gate_shexp.weight"))
        r("blk.1.ffn_gate_shexp.weight shape [256, 256] is not embedding x shared width",
          tensors=self.tensors("blk.1.ffn_gate_shexp.weight", shape=[256, 256]))
        r("output_norm.weight has ggml type Q9_9, which the pack does not read",
          tensors=self.tensors("output_norm.weight", type_name="Q9_9"))
        r("blk.0.ssm_a rows of 6 are not whole Q8_0 blocks",
          tensors=self.tensors("blk.0.ssm_a", type_name="Q8_0", type_id=8))
        r("blk.1.attn_q.weight rows of 48 are not whole Q8_0 blocks",
          tensors=self.tensors("blk.1.attn_q.weight", type_name="Q8_0", type_id=8, shape=[48, 1024]))
        r("layer 2 experts are not one format (gate F32, up F32, down F16): a class holds one format",
          tensors=self.tensors("blk.2.ffn_down_exps.weight", type_name="F16", type_id=1))

    def test_expert_id_refusals(self):
        n, moe, cfg = self.cfg.n_expert, self.moe, self.cfg
        flat = [j for _ in moe for j in range(n)]

        def keep(needle, **meta):
            with self.assertRaises(P.PackRefused) as cm:
                P._keep(self.root / "none.gguf", self.meta(**meta), cfg)
            self.assertIn(needle, str(cm.exception))

        keep("kimi-k31.expert_prune.keep is not an integer array", expert_prune__keep=[0.5] * len(flat),
             expert_prune__source_count=24)
        keep("kimi-k31.expert_prune.keep without kimi-k31.expert_prune.source_count", expert_prune__keep=flat)
        keep("kimi-k31.expert_prune.source_count is not an integer", expert_prune__keep=flat,
             expert_prune__source_count=24.0)
        keep("kimi-k31.expert_prune.source_count = 3 (below the file's 24 experts", expert_prune__keep=flat,
             expert_prune__source_count=3)
        keep("kimi-k31.expert_prune.keep has %d ids for 8 MoE layers x 24 experts" % (len(flat) + 3),
             expert_prune__keep=flat + [1, 2, 3], expert_prune__source_count=896)
        dup = list(flat)
        dup[n + 1] = dup[n]
        keep("kimi-k31.expert_prune.keep layer 2 keeps expert 0 twice", expert_prune__keep=dup,
             expert_prune__source_count=896)
        out = list(flat)
        out[0] = 5000
        keep("kimi-k31.expert_prune.keep layer 1 entry 0 = 5000, outside the source's 896 experts",
             expert_prune__keep=out, expert_prune__source_count=896)
        keep("kimi-k31.expert_prune.method is not a string", expert_prune__keep=flat, expert_prune__source_count=896,
             expert_prune__method=3)
        keep("kimi-k31.expert_prune.source_count without kimi-k31.expert_prune.keep", expert_prune__source_count=896)

    def test_sidecar_refusals(self):
        n, moe, cfg = self.cfg.n_expert, self.moe, self.cfg
        good = {"arch": "kimi-k31", "source_expert_count": 896, "expert_count": n,
                "layers": {str(il): list(range(n)) for il in moe}}
        k = [0]

        def side(needle, body, meta=None):
            k[0] += 1
            path = self.root / ("side%d.gguf" % k[0])
            sp = pathlib.Path(str(path) + ".prune.json")
            if isinstance(body, str):
                sp.write_text(body)
            elif body is None:
                sp.mkdir()
            else:
                sp.write_text(json.dumps(body))
            with self.assertRaises(P.PackRefused) as cm:
                P._keep(path, meta if meta is not None else self.tiny.metadata, cfg)
            self.assertIn(needle, str(cm.exception))

        def changed(**kw):
            d = copy.deepcopy(good)
            for key, v in kw.items():
                if v is None:
                    d.pop(key)
                else:
                    d[key] = v
            return d

        def layers(**kw):
            d = copy.deepcopy(good)
            for key, v in kw.items():
                key = key.lstrip("_")
                if v is None:
                    d["layers"].pop(key)
                else:
                    d["layers"][key] = v
            return d

        side("is not a regular file", None)
        side("is not valid JSON", "{not json")
        side("is not a JSON object", "[1, 2]")
        side('has no "arch" string', changed(arch=None))
        side("is for 'glm', not 'kimi-k31'", changed(arch="glm"))
        side('has no "source_expert_count"', changed(source_expert_count=None))
        side('"expert_count" is not an integer', changed(expert_count="24"))
        side("keeps 99 experts a layer, the file has 24", changed(expert_count=99))
        side('"source_expert_count" 3 is below the file\'s 24 experts or implausible', changed(source_expert_count=3))
        side('has no "layers" object', changed(layers=[1]))
        side('lists layer "0", which is not a MoE layer of this file', layers(_0=list(range(n))))
        side('lists layer "500", which is not a MoE layer of this file', layers(_500=list(range(n))))
        side('lists layer "01", which is not a MoE layer of this file', layers(_01=list(range(n))))
        side("layer 1 is not an array", layers(_1=3))
        side("layer 1 lists 23 experts, the file has 24", layers(_1=list(range(n - 1))))
        side("layer 1 has an id that is not an integer", layers(_1=[0.5] + list(range(1, n))))
        side("layer 1 entry 0 = 5000, outside the source's 896 experts", layers(_1=[5000] + list(range(1, n))))
        side("layer 1 keeps expert 0 twice", layers(_1=[0] + list(range(n - 1))))
        side("has no ids for layer 1", layers(_1=None))
        flat = [j for _ in moe for j in range(n)]
        file_keys = self.meta(expert_prune__keep=flat, expert_prune__source_count=896)
        disagree = layers(_1=list(range(1, n)) + [0])
        side("disagrees with kimi-k31.expert_prune.keep", disagree, meta=file_keys)
        side('says "source_expert_count" 900, kimi-k31.expert_prune.source_count 896',
             changed(source_expert_count=900), meta=file_keys)

    def test_command_line(self):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err, \
                mock.patch.object(sys, "argv", ["pack_gguf.py", "--model", str(self.root / "missing.gguf")]):
            self.assertEqual(P.main(), 2)
        self.assertIn("is not a file", err.getvalue())
        junk = self.root / "junk.gguf"
        junk.write_bytes(b"not a gguf at all, just some bytes" * 4)
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()) as err, \
                mock.patch.object(sys, "argv", ["pack_gguf.py", "--model", str(junk)]):
            self.assertEqual(P.main(), 2)
        self.assertIn("maya-k31: pack: cannot read", err.getvalue())
        spaced = self.root / "a dir"
        spaced.mkdir()
        (spaced / "m.gguf").symlink_to(self.tiny_path)
        with self.assertRaises(P.PackRefused) as cm, contextlib.redirect_stderr(io.StringIO()):
            P.write_pack(spaced / "m.gguf", self.root / "pack-space")
        self.assertIn("the model path relative to the pack has a space", str(cm.exception))


if __name__ == "__main__":
    unittest.main()

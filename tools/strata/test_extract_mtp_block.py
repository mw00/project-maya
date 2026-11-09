"""tools/strata/test_extract_mtp_block.py - B15.4: tools/strata/extract_mtp_block.py on synthetic glm-dsa GGUFs.

The PACK preset of tools/glm_dsa_synth_gguf.py (community K/IQ types, 6 layers + the NextN block blk.6), written whole
and split into shards (the key/values in the first shard only, as gguf-py and llama.cpp's gguf-split write them):
  * the standalone file holds exactly the blk.6 tensors, byte-identical and in the same types, the model's key/values
    (no split.* or tokenizer.*), and reads back with gguf-py's GGUFReader and the audit's header parser;
  * the shards holding the block alone need --kv-from (the model's first shard), and give the same file;
  * --check-against passes for the same model and refuses (exit 1) a block from another geometry (SMALL);
  * --list prints the types and the B15.4 verdict (IQ-class: WARN), writes nothing;
  * a source without a NextN block, or with one missing an attention tensor, is refused (exit 1);
  * the type report is gguf_audit's (A9) and flags only types below 4 bits a weight.

    PYTHONPATH=<llama.cpp>/gguf-py python3 tools/strata/test_extract_mtp_block.py
"""
from __future__ import annotations

import contextlib
import io
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

try:
    import numpy as np  # noqa: F401
    import gguf
    import glm_dsa_synth_gguf as synth
    import gguf_audit as ga
    import extract_mtp_block as emb
    _ERR = None
except ImportError as e:   # gguf-py missing
    _ERR = e

TMP: tempfile.TemporaryDirectory | None = None
D: pathlib.Path | None = None


def run(*args: str) -> tuple[int, str, str]:
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = emb.main(list(args))
    return rc, out.getvalue(), err.getvalue()


def tensors_of(paths) -> dict[str, tuple[int, bytes]]:
    out = {}
    for p in paths:
        r = gguf.GGUFReader(str(p))
        for t in r.tensors:
            out[t.name] = (int(t.tensor_type), bytes(t.data.tobytes()))
    return out


@unittest.skipIf(_ERR is not None, f"gguf-py / numpy unavailable: {_ERR}")
class ExtractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        global TMP, D
        TMP = tempfile.TemporaryDirectory()
        D = pathlib.Path(TMP.name)
        cls.whole = synth.write(D / "pack.gguf", "PACK")[0]
        cls.shards = synth.write(D / "split" / "pack.gguf", "PACK", split_max_size=600_000)
        cls.small = synth.write(D / "small.gguf", "SMALL")[0]
        cls.src = tensors_of([cls.whole])
        cls.block = 6

    @classmethod
    def tearDownClass(cls):
        TMP.cleanup()

    def check_out(self, out: pathlib.Path):
        got = tensors_of([out])
        want = {k: v for k, v in self.src.items() if k.startswith("blk.%d." % self.block)}
        self.assertEqual(sorted(got), sorted(want))
        for k in want:
            self.assertEqual(got[k][0], want[k][0], k)
            self.assertEqual(got[k][1], want[k][1], k)
        r = gguf.GGUFReader(str(out))
        keys = set(r.fields)
        self.assertIn("glm-dsa.block_count", keys)
        self.assertIn("glm-dsa.embedding_length", keys)
        self.assertFalse(any(k.startswith(("split.", "tokenizer.")) for k in keys), keys)
        self.assertEqual(r.fields["general.architecture"].contents(), "glm-dsa")
        self.assertIn("NextN (MTP) block 6", r.fields["general.name"].contents())
        # the audit's own header parser (the B0.7 reader) agrees
        sh = ga.load_local([str(out)])
        self.assertEqual(len(sh[0].tensors), len(want))
        # the engine's lookup: blk.<n_layers>.nextn.eh_proj.weight with n_layers = block_count - nextn
        n_layers = int(r.fields["glm-dsa.block_count"].contents()) - int(r.fields["glm-dsa.nextn_predict_layers"].contents())
        self.assertIn("blk.%d.nextn.eh_proj.weight" % n_layers, got)

    def test_whole_file(self):
        out = D / "mtp_whole.gguf"
        rc, so, se = run(str(self.whole), "-o", str(out), "--check-against", str(self.whole))
        self.assertEqual(rc, 0, so + se)
        self.assertIn("geometry matches", so)
        self.check_out(out)

    def test_shards(self):
        self.assertGreater(len(self.shards), 2)
        out = D / "mtp_shards.gguf"
        rc, so, se = run(str(self.shards[0]), "-o", str(out))   # the first shard finds its siblings
        self.assertEqual(rc, 0, so + se)
        self.check_out(out)
        # only the shards that hold blk.6: the key/values from the first shard
        holding = [p for p in self.shards if any(t.name.startswith("blk.6.") for t in gguf.GGUFReader(str(p)).tensors)]
        self.assertNotIn(self.shards[0], holding)
        rc, so, se = run(*map(str, holding), "-o", str(D / "mtp_bad.gguf"))
        self.assertEqual(rc, 1, so + se)
        self.assertIn("--kv-from", se)
        out2 = D / "mtp_holding.gguf"
        rc, so, se = run(*map(str, holding), "--kv-from", str(self.shards[0]), "-o", str(out2))
        self.assertEqual(rc, 0, so + se)
        self.check_out(out2)
        self.assertEqual(tensors_of([out2]), tensors_of([out]))

    def test_check_against_other_model(self):
        rc, so, se = run(str(self.whole), "-o", str(D / "x.gguf"), "--check-against", str(self.small))
        self.assertEqual(rc, 1, so + se)
        self.assertIn("embedding_length", se)
        self.assertFalse((D / "x.gguf").exists())

    def test_list_and_verdict(self):
        rc, so, se = run(str(self.whole), "--list")
        self.assertEqual(rc, 0, so + se)
        self.assertIn("experts:", so)
        self.assertIn("WARN", so)   # PACK's community types: IQ-class
        rc, so, se = run(str(self.small), "--list")   # SMALL is F32: no low-bit tensor
        self.assertEqual(rc, 0, so + se)
        self.assertIn("OK:", so)

    def test_refusals(self):
        # no NextN block: a GGUF of only trunk tensors
        p = D / "nonextn.gguf"
        w = gguf.GGUFWriter(str(p), "glm-dsa")
        w.add_uint32("glm-dsa.block_count", 2)
        w.add_tensor("blk.0.attn_norm.weight", np.ones(8, dtype=np.float32))
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        rc, so, se = run(str(p), "-o", str(D / "y.gguf"))
        self.assertEqual(rc, 1, so + se)
        self.assertIn("no draft block", se)
        # a block that lacks a tensor the engine needs
        r = gguf.GGUFReader(str(self.small))
        q = D / "lacking.gguf"
        w = gguf.GGUFWriter(str(q), "glm-dsa")
        w.add_uint32("glm-dsa.block_count", 9)
        for t in r.tensors:
            if t.name.startswith("blk.8.") and t.name != "blk.8.attn_output.weight":
                w.add_tensor(t.name, np.asarray(t.data), raw_shape=None)
        w.write_header_to_file()
        w.write_kv_data_to_file()
        w.write_tensors_to_file()
        w.close()
        rc, so, se = run(str(q), "-o", str(D / "z.gguf"))
        self.assertEqual(rc, 1, so + se)
        self.assertIn("attn_output.weight", se)

    def test_type_report_matches_the_audit(self):
        # the extractor reports through gguf_audit.mtp_type_report (A9's report on a real-geometry file)
        srcs = [emb._local(self.whole)]
        items = emb.block_tensors(srcs, 6)
        lines, low = emb.report(items, 6)
        self.assertTrue(any(ln.startswith("experts: ") for ln in lines), lines)
        self.assertTrue(low and all(ga.bits_per_weight(ga.TYPE_ID[t]) < ga.MTP_MIN_BPW for _, t in low), low)

    def test_engine_reader_parses_it(self):
        # the engine's own GGUF reader (strata-gguf, src/artifact/gguf_reader.cpp: the GgufFile the engine opens
        # STRATA_GLM_MTP_GGUF with); ctest passes its path in STRATA_GGUF_BIN
        import os
        exe = os.environ.get("STRATA_GGUF_BIN", "")
        if not exe or not pathlib.Path(exe).exists():
            self.skipTest("STRATA_GGUF_BIN (the strata-gguf binary) not set")
        out = D / "mtp_engine.gguf"
        rc, so, se = run(str(self.whole), "-o", str(out))
        self.assertEqual(rc, 0, so + se)
        r = subprocess.run([exe, str(out), "--check"], capture_output=True, text=True)
        n = sum(1 for k in self.src if k.startswith("blk.6."))
        self.assertIn("tensors %d " % n, r.stdout, r.stdout + r.stderr)
        self.assertIn("blk.* tensors %d" % n, r.stdout)

    def test_cli_help(self):
        r = subprocess.run([sys.executable, str(HERE / "extract_mtp_block.py"), "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0)
        self.assertIn("STRATA_GLM_MTP_GGUF", r.stdout)


if __name__ == "__main__":
    if _ERR is not None:
        print(f"test_extract_mtp_block: skipped ({_ERR})")
        sys.exit(77)
    unittest.main(verbosity=2)

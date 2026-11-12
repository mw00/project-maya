"""tools/test_ds4_drafter.py - DeepSeek's DSpark drafter for the GGUF path without a GPU or a download: its conversion
(tools/ds4/drafter_pack.py: every value kept, the experts in ggml's MXFP4 block order, each role's experts back to back)
and the speed calibration's choice (tools/calibrate_ds4.py: stand-in engines).

    python -m unittest tools.test_ds4_drafter -v
"""
from __future__ import annotations

import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "tools"), str(ROOT / "tools" / "ds4"), str(ROOT)]
import calibrate_draft as CD  # noqa: E402
import calibrate_ds4 as CDS  # noqa: E402
import drafter_pack as DP  # noqa: E402
import pack_gguf as PG  # noqa: E402

E2M1 = [0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6]


def safetensors(path: Path, tensors: dict):
    """name -> (dtype, shape, uint8 bytes): a shard as safetensors lays it out."""
    head, off, blobs = {}, 0, []
    for name, (dt, shape, b) in tensors.items():
        head[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + len(b)]}
        blobs.append(b)
        off += len(b)
    h = json.dumps(head).encode()
    path.write_bytes(struct.pack("<Q", len(h)) + h + b"".join(blobs))


class Conversion(unittest.TestCase):
    def test_values_kept_and_experts_laid_out(self):
        rng = np.random.default_rng(1)
        d = Path(tempfile.mkdtemp())
        w8 = rng.integers(0, 126, (64, 64), dtype=np.uint8)            # E4M3 codes (no NaN)
        s8 = np.array([[120, 127], [130, 125]], np.uint8)             # E8M0, a 32 x 32 block each
        tensors = {"mtp.0.attn.wq_a.weight": ("F8_E4M3", [64, 64], w8.tobytes()),
                   "mtp.0.attn.wq_a.scale": ("F8_E8M0", [2, 2], s8.tobytes()),
                   "mtp.0.attn_norm.weight": ("BF16", [4], np.arange(4, dtype=np.uint16).tobytes())}
        codes = {}
        for e in range(3):                                              # experts out of name order (10 before 2)
            for r in ("w1", "w2", "w3"):
                c = rng.integers(0, 16, (2, 64), dtype=np.uint8)        # 2 rows of 64 values
                codes[(e, r)] = c
                packed = (c[:, 0::2] | (c[:, 1::2] << 4)).astype(np.uint8)
                tensors[f"mtp.0.ffn.experts.{e}.{r}.weight"] = ("I8", [2, 32], packed.tobytes())
                tensors[f"mtp.0.ffn.experts.{e}.{r}.scale"] = ("F8_E8M0", [2, 2], np.full((2, 2), 127 + e, np.uint8).tobytes())
        safetensors(d / DP.SHARDS[0], tensors)
        for name in DP.SHARDS[1:]:
            safetensors(d / name, {})
        with patch.object(sys, "argv", ["drafter_pack.py", "--dir", str(d)]):
            DP.main()
        raw = (d / "dspark.bin").read_bytes()
        ent = {t[0]: t for t in json.loads((d / "dspark.json").read_text())["tensors"]}
        name, dt, off, nb, shape = ent["mtp.0.attn.wq_a.weight"]
        self.assertEqual((dt, shape), ("BF16", [64, 64]))
        got = (np.frombuffer(raw[off:off + nb], np.uint16).astype(np.uint32) << 16).view(np.float32).reshape(64, 64)
        want = DP.E4M3[w8] * np.exp2(np.repeat(np.repeat(s8.astype(np.float32) - 127, 32, 0), 32, 1))
        np.testing.assert_array_equal(got, want)                        # exact
        for r in ("w1", "w3", "w2"):
            first = ent[f"mtp.0.ffn.experts.0.{r}.weight"]
            for e in range(3):
                _, dt, off, nb, shape = ent[f"mtp.0.ffn.experts.{e}.{r}.weight"]
                self.assertEqual((dt, shape, off), ("MXFP4", [2, 64], first[2] + e * first[3]))   # back to back
                blocks = np.frombuffer(raw[off:off + nb], np.uint8).reshape(2, 2, 17)
                self.assertTrue((blocks[..., 0] == 127 + e).all())
                lo, hi = blocks[..., 1:] & 15, blocks[..., 1:] >> 4      # value j, value j + 16
                np.testing.assert_array_equal(np.concatenate([lo, hi], -1).reshape(2, 64), codes[(e, r)])


class Geometry(unittest.TestCase):
    """#500: a GGUF without the DSpark keys (ISTA's GSQ-RCO: only the target layers) packed a drafter with block size 0
    that never drafted - pack_gguf.py fills them from the drafter's .json, else DeepSeek-V4.1-Flash's own for that
    model's trunk, else leaves the drafter out with a warning; drafter_pack.py records the checkpoint's."""
    FLASH = {"hidden_size": 5120, "num_hidden_layers": 40, "n_routed_experts": 384}

    @staticmethod
    def info(n_exp=128, config=None) -> dict:
        t = [["mtp.0.ffn.experts.%d.w1.weight" % e, "MXFP4", 0, 17, [1, 32]] for e in range(n_exp)]
        return dict({"tensors": [["mtp.0.attn_norm.weight", "BF16", 0, 8, [4]]] + t}, **({"config": config} if config else {}))

    def ista(self, **kw) -> dict:
        geo = dict(self.FLASH, **{k: 0 for k in PG.DSPARK_KEYS[:-1]}, dspark_target_layer_ids=[37, 38, 39])
        geo.update(kw)
        return geo

    def test_flash_values_filled(self):
        fill, src = PG.dspark_fill(self.ista(), self.info(), 129280)
        self.assertEqual(fill, {k: v for k, v in PG.DSPARK_V41_FLASH.items() if k != "dspark_target_layer_ids"})
        self.assertIn("DeepSeek-V4.1-Flash", src)
        fill, _ = PG.dspark_fill(self.ista(dspark_target_layer_ids=[]), self.info(), 129280)
        self.assertEqual(fill["dspark_target_layer_ids"], [37, 38, 39])

    def test_the_ggufs_own_kept(self):
        geo = dict(self.FLASH, **PG.DSPARK_V41_FLASH)
        geo["dspark_block_size"] = 4
        self.assertEqual(PG.dspark_fill(geo, self.info(), 129280), ({}, None))

    def test_the_drafters_own_first(self):
        own = {"dspark_block_size": 7, "dspark_noise_token_id": 3, "dspark_markov_rank": 16,
               "dspark_n_routed_experts": 128, "dspark_num_experts_per_tok": 2}
        fill, src = PG.dspark_fill(self.ista(hidden_size=4096), self.info(config=own), 129280)   # (any trunk)
        self.assertEqual((fill, src), (own, "the drafter's .json"))

    def test_left_out_without_them(self):
        fill, rest = PG.dspark_fill(self.ista(hidden_size=4096), self.info(), 129280)   # (another model's trunk)
        self.assertIsNone(fill)
        self.assertIn("dspark_block_size", rest)
        self.assertIsNone(PG.dspark_fill(self.ista(), self.info(n_exp=64), 129280)[0])   # (another drafter)
        self.assertIsNone(PG.dspark_fill(self.ista(), self.info(), 4096)[0])             # (a vocabulary too small)

    def test_drafter_pack_records_the_checkpoints(self):
        d = Path(tempfile.mkdtemp())
        safetensors(d / DP.SHARDS[0], {"mtp.0.attn_norm.weight": ("BF16", [4], np.arange(4, dtype=np.uint16).tobytes())})
        for name in DP.SHARDS[1:]:
            safetensors(d / name, {})
        with patch.object(sys, "argv", ["drafter_pack.py", "--dir", str(d)]):
            DP.main()
        self.assertNotIn("config", json.loads((d / "dspark.json").read_text()))   # (no config.json beside them)
        (d / "config.json").write_text(json.dumps({"text_config": {"dspark_block_size": 5, "hidden_size": 5120,
                                                                   "dspark_noise_token_id": 128799}}))
        with patch.object(sys, "argv", ["drafter_pack.py", "--dir", str(d)]):
            DP.main()
        self.assertEqual(json.loads((d / "dspark.json").read_text())["config"],
                         {"dspark_block_size": 5, "dspark_noise_token_id": 128799})


def kind_requests(cfg):
    """CDS.requests' stand-in: request ids [kind index, request index] (the warm-up [9])."""
    return [9], {key: [[i, j] for j in range(len(reqs))] for i, (key, _, reqs, _) in enumerate(CDS.KINDS)}


class Calibration(unittest.TestCase):
    def run_cal(self, none_tok_s, dspark_tok_s, answer_len=None):
        """none_tok_s / dspark_tok_s: one speed, or {kind: speed}; answer_len: answers end after that many tokens."""
        engines = {}

        class Fake:
            def __init__(self, tok_s):
                self.tok_s, self.last, self.asked = tok_s, {}, []

            def generate(self, ids, max_new, sampling, cancel):
                self.asked.append((tuple(ids), max_new, sampling.get("seed")))
                n = max_new if answer_len is None else min(max_new, answer_len)
                rate = self.tok_s if not isinstance(self.tok_s, dict) else \
                    self.tok_s.get(CDS.KINDS[ids[0]][0] if ids != [9] else "chat", 1.0)
                self.last = {"decode_ms": n / rate * 1000.0, "drafts_accepted": 3, "drafts_offered": 5}
                yield from range(n)
        seen = []

        def start(cfg):
            mode = cfg["args"][cfg["args"].index("--dspark") + 1]
            seen.append((mode, cfg["args"][cfg["args"].index("--expert-profile") + 1] if "--expert-profile" in
                         cfg["args"] else None))
            engines[mode] = Fake(dspark_tok_s if mode == "on" else none_tok_s)
            return engines[mode]
        cfg = {"args": ["--pack", "/nowhere", "--max-context", "8192", "--dspark", "none"],
               "sampling": {"temperature": 1.0, "top_p": 1.0}}
        self.engines = engines
        with patch.object(CDS, "requests", kind_requests):
            return CDS.run(cfg, say=lambda *a: None, start_engine=start), seen

    def test_keeps_the_faster(self):
        res, seen = self.run_cal(3.0, 4.0)
        self.assertEqual((res["keep"], [m for m, _ in seen]), ("dspark", ["none", "on"]))
        self.assertEqual(self.run_cal(3.0, 3.1)[0]["keep"], "none")       # (a tie: no drafting)
        self.assertEqual(self.run_cal(3.0, 2.5)[0]["keep"], "none")

    def test_code_and_tool_calls_count_as_much_as_chat(self):
        # Sun's MiniMax: the drafter doubles code and tool calls and slows chat - a chat-only measurement turned it off
        res, _ = self.run_cal(10.0, {"chat": 8.0, "code": 20.0, "tools": 20.0})
        self.assertEqual(res["keep"], "dspark")
        self.assertAlmostEqual(res["dspark"]["tok_s"], round(3 / (1 / 8 + 1 / 20 + 1 / 20), 2))
        self.assertEqual({k: v["tok_s"] for k, v in res["dspark"]["kinds"].items()},
                         {"chat": 8.0, "code": 20.0, "tools": 20.0})
        self.assertEqual(res["none"]["tok_s"], 10.0)
        # ... and a drafter that wins on code alone but loses chat and tool calls is not kept
        self.assertEqual(self.run_cal(10.0, {"chat": 8.0, "code": 14.0, "tools": 9.0})[0]["keep"], "none")

    def test_combined_is_equal_tokens_of_each_kind(self):
        self.assertEqual(CDS.combined({"a": {"tok_s": 10.0}, "b": {"tok_s": 10.0}, "c": {"tok_s": 10.0}}), 10.0)
        self.assertEqual(CDS.combined({"a": {"tok_s": 6.0}, "b": {"tok_s": 12.0}}), 8.0)   # (one fast kind: no mask)
        self.assertEqual(CDS.combined({"a": {"tok_s": 6.0}, "b": {"tok_s": 0.0}}), 0.0)

    def test_each_kind_writes_its_tokens_with_the_same_requests_and_seeds_on_both_sides(self):
        res, _ = self.run_cal(3.0, 4.0)
        none, on = self.engines["none"].asked, self.engines["on"].asked
        self.assertEqual(none, on)                                          # (deterministic: same ids, caps, seeds)
        self.assertEqual(none[0], ((9,), CD.WARM_ANSWER, 1))               # (the warm-up first, not counted)
        self.assertEqual(len({s for _, _, s in none}), len(none))           # (a seed per request)
        for key, _, _, _ in CDS.KINDS:
            self.assertEqual(res["none"]["kinds"][key]["tokens"], CDS.KIND_TOKENS)
        self.assertEqual(res["none"]["tokens"], CDS.MIN_TOKENS)
        # short answers (a tool call that ends early) take more of the kind's requests
        res, _ = self.run_cal(3.0, 4.0, answer_len=100)
        self.assertEqual(res["none"]["kinds"]["tools"]["tokens"], 400)

    def test_a_kind_with_nothing_measured_is_no_calibration(self):
        with self.assertRaises(RuntimeError):
            self.run_cal(3.0, 4.0, answer_len=1)

    def test_the_requests_are_chat_code_and_tool_calls_rendered_by_the_template(self):
        calls = []

        class Tpl:
            def render(self, messages, tools=None, **kw):
                calls.append((messages[0]["content"], tools, kw))
                return messages[0]["content"]

        class Tok:
            def encode(self, text, parse_special=False):
                return [len(text)]
        with patch.object(CD, "chat_parts", lambda cfg: (Tok(), Tpl(), {"reasoning_effort": "high"})):
            warm, kinds = CDS.requests({})
        self.assertEqual(set(kinds), {"chat", "code", "tools"})
        self.assertEqual([len(v) for v in kinds.values()], [len(CDS.CHAT), len(CDS.CODE), len(CDS.TOOL_CALLS)])
        by_q = {q: (t, kw) for q, t, kw in calls}
        self.assertEqual(by_q[CDS.CHAT[0][0]], (None, {"reasoning_effort": "high"}))   # (the config's thinking)
        self.assertEqual(by_q[CDS.CODE[0][0]], (None, {"enable_thinking": False}))     # (the tokens are the code)
        t, kw = by_q[CDS.TOOL_CALLS[0][0]]
        self.assertEqual(([x["name"] for x in t], kw), (["write_file", "run_command"], {"enable_thinking": False}))
        self.assertTrue(all("write_file" in q for q, _ in CDS.TOOL_CALLS))

    def test_the_setup_keeps_the_combined_choice_and_each_kinds_speeds(self):
        import maya
        res, _ = self.run_cal(10.0, {"chat": 8.0, "code": 20.0, "tools": 20.0})
        d = Path(tempfile.mkdtemp())
        cfg_path = d / "maya-ds4.json"
        cfg_path.write_text(json.dumps({"args": ["--pack", "p", "--dspark", "none"]}), encoding="utf-8")
        said = []
        with patch.object(CDS, "has_drafter", lambda c: True), patch.object(CDS, "run", lambda c, say: res), \
                patch.object(maya, "busy_gpus", lambda c: []), patch.object(maya, "say", lambda *a: said.append(a)), \
                patch.object(maya, "ok", lambda *a: said.append(a)), patch.object(maya, "config_family",
                                                                               lambda c: "deepseek"):
            self.assertTrue(maya.speed_calibration_ds4(cfg_path, json.loads(cfg_path.read_text())))
        out = json.loads(cfg_path.read_text())
        self.assertEqual(out["args"][-2:], ["--dspark", "on"])
        cal = out["installer"]["speed_calibration"]
        self.assertEqual((cal["kept"], cal["kinds"]["code"], cal["kinds"]["chat"]), ("dspark", [10.0, 20.0], [10.0, 8.0]))
        self.assertTrue(any("code 20.0 against 10.0" in " ".join(map(str, a)) for a in said))

    def test_options(self):
        cfg = {"args": ["--pack", "p", "--dspark", "on", "--max-context", "1"]}
        self.assertEqual(CDS.with_dspark(cfg, "none")["args"], ["--pack", "p", "--max-context", "1", "--dspark", "none"])
        self.assertEqual(cfg["args"][3], "on")                            # (a copy)
        p = Path(tempfile.mkdtemp())
        (p / "tensors.txt").write_text("file 0 a.gguf\nt layers.0.x BF16 0 0 2 1\n")
        self.assertFalse(CDS.has_drafter({"args": ["--pack", str(p)]}))
        (p / "tensors.txt").write_text("file 0 a.gguf\nt mtp.0.attn.wq_a.weight BF16 0 0 2 1\n")
        self.assertTrue(CDS.has_drafter({"args": ["--pack", str(p)]}))


if __name__ == "__main__":
    unittest.main()

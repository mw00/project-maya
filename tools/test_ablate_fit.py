"""tools/ablate_fit.py: the read-out's records, the residual sums, the fit on a planted direction, the direction file's
form (what the engines and the setup read) and the eval's summary - without a model."""
from __future__ import annotations

import json
import pathlib
import struct
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import ablate_fit as AF  # noqa: E402

DIM, LAYERS = 32, 6


def records(writes: np.ndarray, T: int) -> bytes:
    return b"".join(struct.pack("<ii", T, writes.shape[1]) + w.astype("<f4").tobytes() for w in writes)


class Records(unittest.TestCase):
    def test_read_back_and_a_record_cut_short(self):
        w = np.arange(3 * DIM, dtype=np.float32).reshape(3, DIM)
        with tempfile.TemporaryDirectory() as d:
            f = pathlib.Path(d) / "cap.bin"
            f.write_bytes(records(w, 7) + b"\x07\x00\x00\x00")   # (the next record's first bytes only)
            recs, end = AF.read_records(f)
            self.assertEqual([t for t, _ in recs], [7, 7, 7])
            np.testing.assert_array_equal(np.stack([r for _, r in recs]), w)
            self.assertEqual(end, 3 * (8 + 4 * DIM))
            self.assertEqual(AF.read_records(f, end)[0], [])

    def test_the_last_prompt_pass_is_taken(self):
        rng = np.random.default_rng(0)
        a, b, step = (rng.standard_normal((2 * LAYERS, DIM)).astype(np.float32) for _ in range(3))
        recs = [(300, r) for r in a] + [(41, r) for r in b] + [(1, r) for r in step]   # two chunks + a decode step
        np.testing.assert_array_equal(AF.prompt_pass(recs, 2 * LAYERS), b)
        with self.assertRaises(ValueError):
            AF.prompt_pass(recs[:-1], 2 * LAYERS)
        with self.assertRaises(ValueError):   # (a wrong write count mixes the passes)
            AF.prompt_pass(recs, 3 * LAYERS)

    def test_residuals_sum_two_writes_a_layer(self):
        w = np.ones((2 * LAYERS, DIM))
        r = AF.residuals(w)
        self.assertEqual(r.shape, (LAYERS + 1, DIM))
        np.testing.assert_array_equal(r[:, 0], 2 * np.arange(LAYERS + 1))


class Fit(unittest.TestCase):
    def test_a_planted_direction_is_found_at_its_layer(self):
        rng = np.random.default_rng(1)
        planted = rng.standard_normal(DIM)
        planted /= np.linalg.norm(planted)
        with tempfile.TemporaryDirectory() as d:
            run = pathlib.Path(d)
            for name, n in (("harmful_train", 40), ("harmless_train", 40), ("harmful_test", 12), ("harmless_test", 12)):
                res = []
                for _ in range(n):
                    w = rng.standard_normal((2 * LAYERS, DIM))
                    if name.startswith("harmful"):
                        w[4] += 8.0 * planted   # (layer 2's attention write: the residual from layer 3 on)
                    res.append(AF.residuals(w))
                np.save(run / f"{name}.off.npy", np.stack(res).astype(np.float16))
            ranked = AF.fit_layers(run, 0.0, 1.0)
            self.assertGreaterEqual(ranked[0]["layer"], 3)
            self.assertGreater(ranked[0]["accuracy"], 0.9)
            r = AF.direction_at(run, ranked[0]["layer"])
            self.assertGreater(abs(float(r @ planted)) / np.linalg.norm(r), 0.9)
            low = [x for x in ranked if x["layer"] <= 2]
            self.assertTrue(all(x["score"] < ranked[0]["score"] for x in low))

    def test_the_direction_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = AF.write_direction(pathlib.Path(d) / "M-refusal-direction.gguf", np.arange(1, DIM + 1), "GLM-5.3-Flash",
                                   "Maya-S", 21, 1.0)
            r, md = AF.read_direction(p)
            self.assertEqual(r.shape, (DIM,))
            self.assertAlmostEqual(float(np.linalg.norm(r)), 1.0, places=5)
            self.assertEqual(md["general.architecture"], "controlvector")
            self.assertEqual(md["controlvector.layer_count"], 1)
            self.assertEqual((md["maya.ablate.model"], md["maya.ablate.fitted_on"], md["maya.ablate.fit_layer"]),
                             ("GLM-5.3-Flash", "Maya-S", 21))
            self.assertAlmostEqual(md["maya.ablate.scale"], 1.0)
            self.assertEqual(md["maya.ablate.version"], AF.VERSION)
            with self.assertRaises(ValueError):
                AF.write_direction(pathlib.Path(d) / "z.gguf", np.zeros(DIM), "m", "q", 1)


class Sets(unittest.TestCase):
    def test_splits_are_disjoint_and_balanced(self):
        s = AF.make_sets([f"h{i}" for i in range(100)] + ["h1"], [f"b{i}" for i in range(500)])
        self.assertEqual((len(s["harmful_train"]), len(s["harmful_test"])), (80, 20))
        self.assertEqual((len(s["harmless_train"]), len(s["harmless_test"])), (80, 20))
        self.assertFalse(set(s["harmful_train"]) & set(s["harmful_test"]))
        self.assertEqual(s, AF.make_sets([f"h{i}" for i in range(100)], [f"b{i}" for i in range(500)]))
        self.assertTrue(s["coherence"])


class Eval(unittest.TestCase):
    def test_summary(self):
        rows = [{"set": "harmful_test", "ablate": False, "text": "I'm sorry, but I can't help with that."},
                {"set": "harmful_test", "ablate": False, "text": "I cannot assist."},
                {"set": "harmful_test", "ablate": True, "text": "Here is an overview: ..."},
                {"set": "harmful_test", "ablate": True, "text": "Step 1 ..."},
                {"set": "coherence", "ablate": False, "text": "def fib(n): ..." * 3},
                {"set": "coherence", "ablate": True, "text": "def fib(n): ..." * 3}]
        s = AF.summarize(rows)
        self.assertEqual((s["refusal_off"], s["refusal_on"]), (1.0, 0.0))
        self.assertEqual(s["verdict"], "ok")
        rows[-1]["text"] = " ".join(["the same words again and again"] * 40)
        self.assertTrue(AF.looped(rows[-1]["text"]))
        self.assertEqual(AF.summarize(rows)["verdict"], "check")
        self.assertFalse(AF.refused("Sure - here it is."))
        json.dumps(s)


class EndToEnd(unittest.TestCase):
    """prompts -> collect -> fit -> eval against a fake server whose 'engine' appends the read-out records the way
    src/core/ablate_capture.hpp does (the harmful prompts' layer-2 attention write carries a planted direction)."""

    def test_collect_fit_eval(self):
        import http.server
        import threading
        rng = np.random.default_rng(3)
        planted = rng.standard_normal(DIM)
        planted /= np.linalg.norm(planted)
        with tempfile.TemporaryDirectory() as d:
            d = pathlib.Path(d)
            sets = AF.make_sets([f"harm {i}" for i in range(60)], [f"fine {i}" for i in range(200)])
            (d / "sets").mkdir()
            for n, ps in sets.items():
                (d / "sets" / f"{n}.jsonl").write_text("".join(json.dumps(p) + "\n" for p in ps), encoding="utf-8")
            cap = d / "cap.bin"
            cap.write_bytes(b"")
            seen = []

            class H(http.server.BaseHTTPRequestHandler):
                def log_message(self, *a):
                    pass

                def do_POST(self):
                    body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                    seen.append(body)
                    p = body["messages"][0]["content"]
                    harm = p.startswith("harm")
                    if body["max_tokens"] == 1:   # (a capture request: two chunks, as a long prompt reads)
                        assert body["ablate"] is True
                        w = rng.standard_normal((2 * LAYERS, DIM))
                        if harm:
                            w[4] += 8.0 * planted
                        with open(cap, "ab") as fh:
                            fh.write(records(rng.standard_normal((2 * LAYERS, DIM)), 512) + records(w, 9))
                        text = "x"
                    else:
                        text = ("Sure, here is how: step one, then step two." if body["ablate"] or not harm
                                else "I'm sorry, but I can't help with that.")
                    out = json.dumps({"choices": [{"message": {"content": text}}]}).encode()
                    self.send_response(200)
                    self.send_header("Content-Length", str(len(out)))
                    self.end_headers()
                    self.wfile.write(out)

            srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
            threading.Thread(target=srv.serve_forever, daemon=True).start()
            url = f"http://127.0.0.1:{srv.server_address[1]}"
            try:
                self.assertEqual(AF.main(["collect", str(d / "sets"), str(d / "run"), "--capture", str(cap), "--url",
                                          url, "--think", "off", "--writes", str(2 * LAYERS), "--truncate-mb", "0"]), 0)
                self.assertEqual(cap.stat().st_size, 0)   # (emptied as it went)
                self.assertEqual(seen[0]["chat_template_kwargs"], {"enable_thinking": False})
                out = d / "X-refusal-direction.gguf"
                self.assertEqual(AF.main(["fit", str(d / "run"), str(out), "--model", "X", "--fitted-on", "Q",
                                          "--lo", "0", "--hi", "1"]), 0)
                r, md = AF.read_direction(out)
                self.assertGreater(abs(float(r @ planted)), 0.8)
                self.assertGreaterEqual(md["maya.ablate.fit_layer"], 3)
                self.assertEqual(AF.main(["eval", str(d / "sets"), "--url", url, "--out", str(d / "e.json"),
                                          "--harmless"]), 0)
                s = json.loads((d / "e.json").read_text())["summary"]
                self.assertEqual((s["refusal_off"], s["refusal_on"], s["verdict"]), (1.0, 0.0, "ok"))
            finally:
                srv.shutdown()


if __name__ == "__main__":
    unittest.main()

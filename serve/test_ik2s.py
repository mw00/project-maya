"""serve/test_ik2s.py - Inkling-2-Small on its own engine maya-ik2s (src/ik2s, AMD under Linux) and what the server and
the Models page do for it, beside maya-ik2's Inkling-2-Mini (unchanged): its family by its engine's name or its record,
the engine line (the server's layer split across the config's GPUs), its settings on the page - DeepSeek's rows (answers
at once, its DSpark drafter where its index is on this PC, written as --dspark on|none with --dspark-index, the VRAM kept
free, the experts VRAM does not hold, the prompt chunk, the CPU threads), the global layers' cache (--kv q8 | f16) and
the drafts' floor (--dspark-cum) - its figures for the Context size line, and the page's rows in Node (where it is
installed).  The chat side (template, parser, tool calls, thinking) is Inkling 2's own: serve/test_inkling2_frontend.py.

    python -m unittest serve.test_ik2s

Offline, no GPU.
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from serve import models, server

MINI = {"exe": "/opt/maya/build/maya-ik2", "gpu": [0], "model_name": "inkling-2-mini",
        "args": ["--model", "/models/Inkling-2-Mini-Maya-L-NVFP4.gguf", "--max-context", "32768"],
        "installer": {"family": "ik2", "ik2_quant": "maya-l", "quant": "Maya-L"}}
OWN = ("reserve_mib", "mmap", "prefill_chunk", "cpu_threads", "dspark_cum", "kv")   # (maya-ik2s's, not Mini's)


def layout(root: Path, drafter: bool = True) -> tuple:
    """A models folder as the setup leaves it: the GGUF set's pack, and (drafter) the converted drafter's index beside
    it.  (pack, index)"""
    pack = root / "Inkling-2-Small-GGUF" / "UD-Q2_K_XL" / "maya-ik2s"
    pack.mkdir(parents=True)
    (pack / "inkling2.txt").write_text("arch inkling2\ncontext_length 1048576\nmodel_dir /x\n", encoding="utf-8")
    index = root / models.IK2S_MTP_DIR / models.IK2S_DSPARK
    if drafter:
        index.mkdir(parents=True)
        (index / "dspark.txt").write_text("format maya-ik2-dspark 1\n", encoding="utf-8")
    return pack, index


def small(pack: Path, *extra, **keys) -> dict:
    return {"exe": "/home/me/maya/build-hip/maya-ik2s", "gpu": [0, 1, 2, 3], "backend": "hip",
            "args": ["--pack", str(pack), "--max-context", "32768", *extra], "model_name": "inkling-2-small",
            "installer": {"family": "ik2s", "ik2s_quant": "ud-q2_k_xl", "quant": "unsloth UD-Q2_K_XL"}, **keys}


class Family(unittest.TestCase):
    def test_its_own_beside_mini(self):
        with tempfile.TemporaryDirectory() as t:
            pack, _ = layout(Path(t))
            cfg = small(pack)
            self.assertEqual(models.family(cfg), "ik2s")
            self.assertEqual(models.family({"exe": "/x/maya-ik2s"}), "ik2s")      # (by its engine's name alone)
            self.assertEqual(models.family(MINI), "ik2")
            self.assertEqual(models.family({"exe": "/x/maya-ik2"}), "ik2")
            # (the engine line: the server's layer split over the config's GPUs, as every multi-GPU engine's but
            #  maya-ik2's - Mini's line as before)
            self.assertEqual(server.engine_args(cfg), cfg["args"] + ["--layer-split", "auto"])
            self.assertEqual(server.engine_args(dict(cfg, layer_split="12,24,36"))[-2:], ["--layer-split", "12,24,36"])
            self.assertEqual(server.engine_args(dict(MINI, gpu=[0, 1])), MINI["args"])
            self.assertEqual(server.child_env(cfg)["HIP_VISIBLE_DEVICES"], "0,1,2,3")
            self.assertEqual(models.pack_of(cfg), pack)
            self.assertEqual(models.trained_context_ds4(pack), 1048576)            # (inkling2.txt's context_length)

    def test_minis_page_as_before(self):
        s = models.settings_of(MINI)
        for k in OWN[:-1]:
            self.assertNotIn(k, s)
        self.assertEqual(s["kv"], "f16")                                        # (Mini's own IK2_KV row)
        with self.assertRaises(ValueError):
            models.apply(MINI, {"reserve_mib": 3072})


class Choices(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)

    def tearDown(self):
        self.t.cleanup()

    def test_offered_where_the_drafter_is(self):
        pack, index = layout(self.root)
        for cfg in (small(pack),                                           # (beside the pack: the setup's layout)
                    small(pack, "--dspark-index", str(index)),             # (the one it names)
                    dict(small(self.root / "elsewhere" / "maya-ik2s"),
                         installer={"family": "ik2s", "models_dir": str(self.root)})):
            with self.subTest(args=cfg["args"]):
                self.assertEqual(models.choices(cfg)["draft"], ["dspark", "none"])
                self.assertEqual(models.ik2s_drafter(cfg), index)
                self.assertEqual((models.choices(cfg)["images"], models.choices(cfg)["text_only"]), (False, True))
                self.assertEqual(models.choices(cfg)["kv"], ["q8", "f16"])

    def test_not_offered_without_it(self):
        pack, index = layout(self.root, drafter=False)
        cfg = small(pack, "--dspark-index", str(index))                      # (named, but not on this PC)
        self.assertEqual(models.choices(cfg)["draft"], ["none"])
        self.assertIsNone(models.ik2s_drafter(cfg))
        self.assertEqual(models.settings_of(cfg)["draft"], "none")
        with self.assertRaises(ValueError) as e:
            models.apply(cfg, {"draft": "dspark"})
        self.assertIn("setup downloads", str(e.exception))
        self.assertEqual(models.apply(cfg, {"draft": "none"})["args"], cfg["args"])   # (nothing to switch: as it is)


class Settings(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.pack, self.index = layout(Path(self.t.name))

    def tearDown(self):
        self.t.cleanup()

    def test_what_the_engine_does(self):
        """Drafting is on where the engine drafts: --dspark not off and an index that is there; the cache F16 unless
        --kv q8 (the engine's own default)."""
        ix = ["--dspark-index", str(self.index)]
        self.assertEqual(models.settings_of(small(self.pack))["draft"], "none")
        self.assertEqual(models.settings_of(small(self.pack, *ix))["draft"], "dspark")      # (on: the default)
        self.assertEqual(models.settings_of(small(self.pack, "--dspark", "off", *ix))["draft"], "none")
        self.assertEqual(models.settings_of(small(self.pack, "--dspark", "none", *ix))["draft"], "none")
        self.assertEqual(models.settings_of(small(self.pack))["kv"], "f16")
        self.assertEqual(models.settings_of(small(self.pack, "--kv", "q8"))["kv"], "q8")
        self.assertEqual(models.settings_of(small(self.pack))["images"], False)

    def test_drafting_written_as_deepseeks(self):
        cfg = small(self.pack)
        on = models.apply(cfg, {"draft": "dspark"})
        self.assertEqual(models.flag(on["args"], "--dspark"), "on")
        self.assertEqual(models.flag(on["args"], "--dspark-index"), str(self.index))
        off = models.apply(on, {"draft": "none"})
        self.assertEqual(models.flag(off["args"], "--dspark"), "none")          # (DeepSeek's: the drafter not loaded)
        self.assertEqual(models.flag(off["args"], "--dspark-index"), str(self.index))   # (kept: on again later)
        self.assertEqual(off["args"][:4], cfg["args"][:4])                       # (the pack and the context as they were)
        with self.assertRaises(ValueError):
            models.apply(cfg, {"draft": "mtp"})

    def test_the_cache_format(self):
        cfg = small(self.pack)
        q8 = models.apply(cfg, {"kv": "q8"})
        self.assertEqual(models.flag(q8["args"], "--kv"), "q8")
        self.assertEqual(models.flag(models.apply(q8, {"kv": "f16"})["args"], "--kv"), "f16")
        self.assertNotIn("--kv", models.apply(cfg, {"kv": "f16"})["args"])      # (the engine's own: not written)
        with self.assertRaises(ValueError):
            models.apply(cfg, {"kv": "fp8"})                                    # (Mini's format: not this engine's)

    def test_deepseeks_settings(self):
        """reserve_mib, prefill_chunk, cpu_threads, mmap, parallel (DS4_KEYS') and dspark_cum (dspark_conf's
        counterpart): read from and written to maya-ik2s's flags as DeepSeek's are - empty: the flag dropped."""
        cfg = small(self.pack, "--dspark-index", str(self.index))
        got = models.settings_of(cfg)
        self.assertEqual({k: got[k] for k in OWN[:-1]}, {**dict.fromkeys(OWN[:-1]), "mmap": "auto"})
        out = models.apply(cfg, {"reserve_mib": 3072, "prefill_chunk": 2048, "cpu_threads": 16, "dspark_cum": 0.4,
                                 "mmap": "off", "parallel": 4})
        for f, v in (("--reserve-mib", "3072"), ("--prefill-chunk", "2048"), ("--cpu-threads", "16"),
                     ("--dspark-cum", "0.4"), ("--mmap-experts", "off"), ("--parallel", "4")):
            self.assertEqual(models.flag(out["args"], f), v)
        back = models.apply(out, {"reserve_mib": None, "prefill_chunk": None, "cpu_threads": None, "dspark_cum": None,
                                  "mmap": "auto"})
        self.assertEqual(models.flag(back["args"], "--parallel"), "4")
        self.assertEqual([a for a in back["args"] if a not in ("--parallel", "4")], models.apply(cfg, {})["args"])
        for bad in ({"dspark_cum": 1.5}, {"prefill_chunk": 4}, {"cpu_threads": 0}, {"parallel": 9}, {"mmap": "maybe"},
                    {"images": True}, {"expert_cache": 100}, {"tp": 2}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                models.apply(cfg, bad)

    def test_the_pages_figures(self):
        cfg = small(self.pack, "--dspark-index", str(self.index))
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "maya-inkling-2-small.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            e = models.describe(path, None)
        self.assertEqual((e["ik2s_kv_bytes"], e["kv_gpu_bytes"], e["dspark_kv_bytes"], e["dspark_gb"]),
                         (models.IK2S_KV_BYTES, models.IK2S_ATTN_BYTES, models.IK2S_DSPARK_KV_BYTES,
                          models.IK2S_DSPARK_GB))
        self.assertEqual(e["family"], "ik2s")
        self.assertEqual(e["contexts"], models.DS4_CONTEXTS)
        self.assertEqual(e["context_range"]["tested"], 524288)                  # (CONTEXT_TESTED["ik2s"])
        self.assertEqual(e["context_range"]["max"], 1048576)


# models.js' fields (fieldsOf, exported in a temporary copy, as serve/test_glm_dsa_models.py's)
PAGE_JS = r"""
import fs from "node:fs";
import path from "node:path";
import {pathToFileURL} from "node:url";
const noop = () => {};
const el = new Proxy({}, {get: (t, k) => (k in t ? t[k] : k === "dataset" || k === "style" ? (t[k] = {})
  : k === "classList" ? (t[k] = {add: noop, remove: noop, toggle: noop}) : noop)});
globalThis.matchMedia = () => ({matches: false, addEventListener: noop, addListener: noop});
globalThis.localStorage = {getItem: () => null, setItem: noop, removeItem: noop};
globalThis.sessionStorage = globalThis.localStorage;
globalThis.document = {documentElement: el, body: el, head: el, addEventListener: noop, querySelector: () => null,
                       createElement: () => el};
globalThis.window = globalThis; globalThis.addEventListener = noop;
globalThis.location = {href: "http://x/", hash: "", search: "", pathname: "/"};
const web = process.argv[2];
fs.appendFileSync(path.join(web, "models.js"), "\nexport {fieldsOf};\n");
const M = await import(pathToFileURL(path.join(web, "models.js")).href);
const ser = (v) => v == null || typeof v === "boolean" ? null : Array.isArray(v) ? v.map(ser)
  : typeof v === "function" ? "fn" : typeof v !== "object" ? v
  : "type" in v && "props" in v ? {t: typeof v.type === "function" ? v.type.name : v.type,
                                   p: Object.fromEntries(Object.entries(v.props || {}).map(([k, x]) => [k, ser(x)]))}
  : Object.fromEntries(Object.entries(v).map(([k, x]) => [k, ser(x)]));
const s = {context: 262144, gpus: [0, 1, 2, 3], env: {}, images: false, draft: "dspark", reserve_mib: null, kv: "q8",
           mmap: "auto", parallel: 1, prefill_chunk: 2048, cpu_threads: null, dspark_cum: null};
const base = {file: "maya-inkling-2-small.json", running: false, backend: "hip", family: "ik2s", model: "inkling-2-small",
              quant: "unsloth UD-Q2_K_XL", settings: s, context_range: {min: 4096, max: 1048576, tested: 524288}};
const own = {ik2s_kv_bytes: {q8: 16896, f16: 32768}, kv_gpu_bytes: 1560, dspark_kv_bytes: 6400, dspark_gb: 3.47};
const cases = {
  on: {...base, ...own, choices: {draft: ["dspark", "none"], images: false, text_only: true, kv: ["q8", "f16"]}},
  none: {...base, ...own, settings: {...s, draft: "none", kv: "f16"},
         choices: {draft: ["none"], images: false, text_only: true, kv: ["q8", "f16"]}},
};
M.MS.data = {gpus: [0, 1, 2, 3].map((i) => ({index: i, name: "GPU", vram_gb: 32}))};
const out = {};
for (const [k, m] of Object.entries(cases)) out[k] = ser(M.fieldsOf(m, {...m.settings}));
console.log(JSON.stringify(out));
"""


def rows_of(fields) -> dict:
    return {r["p"]["label"]: r["p"] for s in fields for r in s["rows"] if isinstance(r, dict)}


@unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
class PageJs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with tempfile.TemporaryDirectory() as t:
            web = Path(t) / "web"
            shutil.copytree(Path(__file__).resolve().parent / "web", web)
            (Path(t) / "t.mjs").write_text(PAGE_JS, encoding="utf-8")
            r = subprocess.run(["node", str(Path(t) / "t.mjs"), str(web)], capture_output=True, encoding="utf-8",
                               timeout=120)
        if r.returncode != 0:
            raise AssertionError(r.stderr[-2000:])
        cls.got = json.loads(r.stdout.strip().splitlines()[-1])

    def test_its_rows(self):
        f = rows_of(self.got["on"])
        self.assertEqual(list(f), ["Context size", "Answers at once", "Drafting", "Images", "GPUs", "VRAM kept free",
                                   "KV cache", "Experts VRAM does not hold", "Prompt chunk", "CPU threads",
                                   "Draft confidence"])
        for label, flag in (("Answers at once", "--parallel"), ("Drafting", "--dspark"), ("KV cache", "--kv"),
                            ("VRAM kept free", "--reserve-mib"), ("Experts VRAM does not hold", "--mmap-experts"),
                            ("Prompt chunk", "--prefill-chunk"), ("CPU threads", "--cpu-threads"),
                            ("Draft confidence", "--dspark-cum")):
            self.assertEqual(f[label]["flag"], flag)
        self.assertIn("about 3.5 GB", f["Drafting"]["hint"])               # (models.py's IK2S_DSPARK_GB, measured)
        self.assertIn("text only", f["Images"]["hint"])
        # (the Context size line: the 8-bit caches, the 4 GPUs' attention scratch and the drafter's rows)
        gb = 262144 * (16896 + 4 * 1560 + 6400) / 2**30
        self.assertIn(f"{gb:.1f} GB", json.dumps(f["Context size"]["below"]))
        n = rows_of(self.got["none"])
        gb = 262144 * (32768 + 4 * 1560) / 2**30                           # (F16, not drafting: not the drafter's)
        self.assertIn(f"{gb:.1f} GB", json.dumps(n["Context size"]["below"]))
        self.assertNotIn("Draft confidence", n)                             # (shown while it drafts, as DeepSeek's)
        self.assertIn("setup downloads", n["Drafting"]["hint"])
        self.assertNotIn('"dspark"', json.dumps(n["Drafting"]))            # (Off alone)


if __name__ == "__main__":
    unittest.main()

"""serve/test_ik2_drafting.py - the Models page's Drafting switch for Inkling 2 Small: maya-inkling2's DSpark drafter
("dspark" | "none", as DeepSeek's), offered only where its index is on this PC - the one the config names
(--dspark-index), the one in the setup's models folder, or the one the setup's layout puts beside the pack - and
written as --dspark on|none with --dspark-index, as DeepSeek's.  maya-inkling2's other DeepSeek settings (the VRAM kept
free, the prompt chunk, the CPU threads and the drafts' floor, --dspark-cum: DeepSeek's Draft confidence) in DeepSeek's
rows.  maya-ik2 (CUDA) has none of them: "none" only, nothing written, and the page shows no switch for it.

    python -m unittest serve.test_ik2_drafting

Offline, no GPU (the page's fields in Node where it is installed).
"""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

from serve import models

CUDA = {"exe": "/opt/maya/build/maya-ik2", "gpu": [0, 1], "model_name": "inkling-2-small",
        "args": ["--model", "/models/Inkling-2-Small-UD-Q3_K_XL-00001-of-00005.gguf", "--max-context", "32768"]}


def layout(root: Path, drafter: bool = True) -> tuple:
    """A models folder as the setup leaves it: the GGUF set's pack, and (drafter) the converted drafter's index beside
    it.  (pack, index)"""
    pack = root / "Inkling-2-Small-GGUF" / "UD-Q2_K_XL" / "maya-inkling2"
    pack.mkdir(parents=True)
    (pack / "inkling2.txt").write_text("arch inkling2\n", encoding="utf-8")
    index = root / models.IK2_MTP_DIR / models.IK2_DSPARK
    if drafter:
        index.mkdir(parents=True)
        (index / "dspark.txt").write_text("format maya-inkling2-dspark 1\n", encoding="utf-8")
    return pack, index


def amd(pack: Path, *extra, **keys) -> dict:
    return {"exe": "/home/me/project-maya/build-hip/maya-inkling2", "gpu": [0, 1, 2, 3], "backend": "hip",
            "args": ["--pack", str(pack), "--max-context", "32768", *extra], "model_name": "inkling-2-small", **keys}


class Choices(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.root = Path(self.t.name)

    def tearDown(self):
        self.t.cleanup()

    def test_offered_where_the_drafter_is(self):
        pack, index = layout(self.root)
        for cfg in (amd(pack),                                             # (beside the pack: a config by hand)
                    amd(pack, "--dspark-index", str(index)),               # (the one it names)
                    amd(self.root / "elsewhere" / "maya-inkling2", installer={"models_dir": str(self.root)})):
            with self.subTest(args=cfg["args"]):
                self.assertEqual(models.choices(cfg)["draft"], ["dspark", "none"])
                self.assertTrue(models.choices(cfg)["drafting"])
                self.assertEqual(models.ik2_drafter(cfg), index)

    def test_not_offered_without_it(self):
        pack, index = layout(self.root, drafter=False)
        cfg = amd(pack, "--dspark-index", str(index))                      # (named, but not on this PC)
        self.assertEqual(models.choices(cfg)["draft"], ["none"])
        self.assertTrue(models.choices(cfg)["drafting"])
        self.assertIsNone(models.ik2_drafter(cfg))
        self.assertEqual(models.settings_of(cfg)["draft"], "none")

    def test_the_cuda_engine_has_none(self):
        layout(self.root)
        cuda = dict(CUDA, args=["--model", str(self.root / "Inkling-2-Small-GGUF" / "UD-Q2_K_XL" / "x-00001-of-00006.gguf"),
                                "--max-context", "32768"], installer={"models_dir": str(self.root)})
        self.assertEqual(models.choices(cuda)["draft"], ["none"])
        self.assertFalse(models.choices(cuda)["drafting"])
        self.assertEqual(models.settings_of(cuda)["draft"], "none")
        out = models.apply(cuda, {"draft": "none", "context": 65536})
        self.assertNotIn("--dspark", out["args"])
        self.assertNotIn("--dspark-index", out["args"])
        with self.assertRaises(ValueError):
            models.apply(cuda, {"draft": "dspark"})


class Settings(unittest.TestCase):
    def setUp(self):
        self.t = tempfile.TemporaryDirectory()
        self.pack, self.index = layout(Path(self.t.name))

    def tearDown(self):
        self.t.cleanup()

    def test_what_the_engine_does(self):
        """Drafting is on where the engine drafts: --dspark not off and an index that is there (no index named: the
        engine decodes without drafts, so the page says Off)."""
        ix = ["--dspark-index", str(self.index)]
        self.assertEqual(models.settings_of(amd(self.pack))["draft"], "none")
        self.assertEqual(models.settings_of(amd(self.pack, *ix))["draft"], "dspark")          # (on: the default)
        self.assertEqual(models.settings_of(amd(self.pack, "--dspark", "on", *ix))["draft"], "dspark")
        self.assertEqual(models.settings_of(amd(self.pack, "--dspark", "off", *ix))["draft"], "none")   # (loaded, not drafting)
        self.assertEqual(models.settings_of(amd(self.pack, "--dspark", "none", *ix))["draft"], "none")
        self.assertEqual(models.settings_of(amd(self.pack, "--dspark", "on"))["draft"], "none")

    def test_written_as_deepseeks(self):
        cfg = amd(self.pack)
        on = models.apply(cfg, {"draft": "dspark"})
        self.assertEqual(models.flag(on["args"], "--dspark"), "on")
        self.assertEqual(models.flag(on["args"], "--dspark-index"), str(self.index))
        self.assertEqual(models.settings_of(on)["draft"], "dspark")
        off = models.apply(on, {"draft": "none"})
        self.assertEqual(models.flag(off["args"], "--dspark"), "none")   # (DeepSeek's: the drafter not loaded)
        self.assertEqual(models.flag(off["args"], "--dspark-index"), str(self.index))   # (kept: on again later)
        self.assertEqual(models.settings_of(off)["draft"], "none")
        self.assertEqual(models.choices(off)["draft"], ["dspark", "none"])
        self.assertEqual(off["args"][:4], cfg["args"][:4])                 # (the pack and the context as they were)
        again = models.apply(off, {"context": 131072})                     # (another setting: drafting as it was)
        self.assertEqual(models.flag(again["args"], "--dspark"), "none")
        with self.assertRaises(ValueError):
            models.apply(cfg, {"draft": "mtp"})

    def test_deepseeks_settings(self):
        """reserve_mib, prefill_chunk, cpu_threads (DS4_KEYS') and dspark_cum (dspark_conf's counterpart): read from
        and written to maya-inkling2's flags as DeepSeek's are - empty: the flag dropped, the engine's own default."""
        cfg = amd(self.pack, "--dspark-index", str(self.index))
        got = models.settings_of(cfg)
        self.assertEqual({k: got[k] for k in models.IK2_AMD_ONLY}, dict.fromkeys(models.IK2_AMD_ONLY))
        out = models.apply(cfg, {"reserve_mib": 3072, "prefill_chunk": 1024, "cpu_threads": 16, "dspark_cum": 0.4})
        for f, v in (("--reserve-mib", "3072"), ("--prefill-chunk", "1024"), ("--cpu-threads", "16"),
                     ("--dspark-cum", "0.4")):
            self.assertEqual(models.flag(out["args"], f), v)
        self.assertEqual(models.settings_of(out)["dspark_cum"], 0.4)
        back = models.apply(out, {"reserve_mib": None, "prefill_chunk": None, "cpu_threads": None, "dspark_cum": None})
        self.assertEqual(back["args"], models.apply(cfg, {})["args"])
        for bad in ({"dspark_cum": 1.5}, {"prefill_chunk": 4}, {"cpu_threads": 0}):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                models.apply(cfg, bad)
        cuda = dict(CUDA)
        self.assertNotIn("reserve_mib", models.settings_of(cuda))
        with self.assertRaises(ValueError):
            models.apply(cuda, {"reserve_mib": 3072})

    def test_the_pages_figures(self):
        """maya-inkling2's entry carries its VRAM a position and the drafter's - the setup's figures (maya.py takes
        them from here); maya-ik2's carries none."""
        cfg = amd(self.pack, "--dspark-index", str(self.index))
        with tempfile.TemporaryDirectory() as t:
            path = Path(t) / "maya-inkling2-hip.json"
            path.write_text(json.dumps(cfg), encoding="utf-8")
            e = models.describe(path, None)
            path.write_text(json.dumps(CUDA), encoding="utf-8")
            c = models.describe(path, None)
        self.assertEqual((e["kv_bytes"], e["kv_gpu_bytes"], e["dspark_kv_bytes"], e["dspark_gb"]),
                         (models.IK2_KV_BYTES, models.IK2_ATTN_BYTES, models.IK2_DSPARK_KV_BYTES, models.IK2_DSPARK_GB))
        self.assertEqual(e["context_range"]["tested"], 524288)            # (CONTEXT_TESTED["inkling2"])
        self.assertNotIn("kv_bytes", c)

    def test_contexts_and_tested(self):
        """DeepSeek's contexts (from 32K); maya-inkling2's end-to-end runs reach 512K so far (CONTEXT_TESTED)."""
        self.assertEqual(models.IK2_CONTEXTS, models.DS4_CONTEXTS)
        self.assertEqual(models.CONTEXT_TESTED["inkling2"], 524288)
        self.assertEqual(models.context_range(1048576, 32768, "inkling2")["tested"], 524288)

    def test_refused_without_the_drafter(self):
        shutil.rmtree(self.index)
        cfg = amd(self.pack)
        with self.assertRaises(ValueError) as e:
            models.apply(cfg, {"draft": "dspark"})
        self.assertIn("setup downloads", str(e.exception))
        self.assertEqual(models.apply(cfg, {"draft": "none"})["args"], cfg["args"])   # (nothing to switch: as it is)


# models.js' fields for both engines (fieldsOf, exported in a temporary copy, as serve/test_glm_dsa_models.py's)
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
const s = {context: 262144, gpus: [0, 1, 2, 3], env: {}, images: false, draft: "dspark", reserve_mib: null,
           prefill_chunk: null, cpu_threads: null, dspark_cum: null};
const base = {file: "maya-inkling2-hip.json", running: false, backend: "hip", family: "ik2", model: "inkling-2-small",
              quant: "UD-Q2_K_XL", settings: s, context_range: {min: 4096, max: 1048576, tested: 262144}};
const own = {kv_bytes: 32768, kv_gpu_bytes: 1560, dspark_kv_bytes: 6400, dspark_gb: 3.47};   // (models.py's entry)
const cases = {
  amd: {...base, ...own, choices: {draft: ["dspark", "none"], images: false, text_only: true, drafting: true}},
  amd_none: {...base, ...own, settings: {...s, draft: "none"}, choices: {draft: ["none"], images: false, text_only: true, drafting: true}},
  cuda: {...base, backend: "cuda", settings: {...s, draft: "none"}, choices: {draft: ["none"], images: false, text_only: true, drafting: false}},
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

    def test_amd_has_the_switch(self):
        f = rows_of(self.got["amd"])
        self.assertEqual(list(f), ["Context size", "Drafting", "Images", "Answers at once", "GPUs", "VRAM kept free",
                                   "Prompt chunk", "CPU threads", "Draft confidence"])
        for label, flag in (("VRAM kept free", "--reserve-mib"), ("Prompt chunk", "--prefill-chunk"),
                            ("CPU threads", "--cpu-threads"), ("Draft confidence", "--dspark-cum")):
            self.assertEqual(f[label]["flag"], flag)
        self.assertIn("running product", f["Draft confidence"]["hint"])
        self.assertIn("About 3.47 GB", f["Drafting"]["hint"])             # (models.py's IK2_DSPARK_GB, measured)
        # (the Context size line: the caches, the 4 GPUs' attention scratch and the drafter's rows - the setup's figure)
        gb = 262144 * (32768 + 4 * 1560 + 6400) / 2**30
        self.assertIn(f"≈ {gb:.1f} GB of GPU memory", f["Context size"]["below"]["p"]["hint"])
        n = rows_of(self.got["amd_none"])
        gb = 262144 * (32768 + 4 * 1560) / 2**30                          # (not drafting: not the drafter's)
        self.assertIn(f"≈ {gb:.1f} GB of GPU memory", n["Context size"]["below"]["p"]["hint"])
        self.assertEqual(f["Drafting"]["flag"], "--dspark")
        self.assertIn("DSpark", f["Drafting"]["hint"])
        self.assertIn('"dspark"', json.dumps(f["Drafting"]))
        n = rows_of(self.got["amd_none"])
        self.assertNotIn("Draft confidence", n)                            # (shown while it drafts, as DeepSeek's)
        self.assertIn("setup downloads", n["Drafting"]["hint"])
        self.assertNotIn('"dspark"', json.dumps(n["Drafting"]))           # (Off alone)

    def test_cuda_has_none(self):
        f = rows_of(self.got["cuda"])
        self.assertEqual(list(f), ["Context size", "Images", "Answers at once", "GPUs"])
        self.assertIn("≈ 8.0 GB of GPU memory", f["Context size"]["below"]["p"]["hint"])   # (the global layers' rows)


if __name__ == "__main__":
    unittest.main()

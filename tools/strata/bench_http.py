#!/usr/bin/env python3
"""tools/strata/bench_http.py - decode and prefill speed through an HTTP server (BUILD-PLAN B0.10, B13.2).

Measures, with the method of BUILD-PLAN section 0 (greedy, median of N runs after one warm-up, frozen corpus):

* decode tok/s at a filled context C: a prompt of C - gen tokens, then `gen` generated tokens; the server's own
  `timings.predicted_per_second` (generated tokens / decode wall time, prefill excluded);
* prefill tok/s of a P-token prompt with a cold cache: `timings.prompt_per_second`.

Two APIs:

* `--api llama` (llama-server): the prompt is the exact token-id list cut from the corpus through the server's
  `/tokenize`; `/completion` with `cache_prompt: false`, `ignore_eos: true`, `temperature: 0`;
* `--api openai` (glm-dsa/Maya `serve/server.py`, which returns the same `timings` names): the prompt goes as one
  user message with thinking off; it is cut to length with the pack tokenizer when `--tokenizer <pack>/tokenizer`
  is given, else by characters (calibrated on the warm-up's `usage.prompt_tokens`).  Start the engine with
  `MAYA_GLM_NO_REUSE=1` for cold prefill numbers.  `--tune '<json object>'` sends it as every request's
  `strata_tune` (the engine knobs one request may set without a restart, serve/glm_dsa_tune.py), for example
  `--tune '{"cpu_threads": 36, "pcie_frac": 0.21}'` or `--tune '{"spec": 1}'` (the token-by-token decode).

Every run starts at a different corpus offset, so no run hits a cached prefix.  Output: a markdown table on stdout
(spread = (max - min) / median, flagged above --max-spread percent) and, with --json, the raw runs.

    python3 tools/strata/bench_http.py --port 8091 --corpus data/glm-dsa/bench-corpus --ctx 4096,32768 \\
        --gen 256 --runs 3 --prompt-lens 8192,32768 > docs/strata/volta3/baseline-llamacpp-L1.md
"""
from __future__ import annotations

import argparse
import json
import pathlib
import re
import statistics
import sys
import time
import urllib.error
import urllib.request

ROOT = pathlib.Path(__file__).resolve().parents[2]
CORPUS = ROOT / "data" / "glm-dsa" / "bench-corpus"
RUN_OFFSET = 64           # tokens (or about 4x characters) between the start of successive runs


class BenchError(RuntimeError):
    pass


class Server:
    def __init__(self, base: str, api: str, timeout: float, api_key: str = "", model: str = "strata",
                 tune: dict | None = None):
        self.base, self.api, self.timeout, self.api_key, self.model = base.rstrip("/"), api, timeout, api_key, model
        self.tune = tune                  # --tune: the openai requests' strata_tune (None: not sent)

    def post(self, path: str, body: dict) -> dict:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = "Bearer " + self.api_key
        req = urllib.request.Request(self.base + path, data=json.dumps(body).encode(), headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise BenchError("%s %s: HTTP %d %s" % (self.base, path, e.code, e.read()[:300])) from e
        except (OSError, ValueError) as e:
            raise BenchError("%s %s: %s" % (self.base, path, e)) from e

    def tokenize(self, text: str) -> list[int]:
        return list(self.post("/tokenize", {"content": text, "add_special": False})["tokens"])

    def run(self, prompt, gen: int) -> dict:
        """One request; returns {'prompt_n', 'prompt_per_second', 'predicted_n', 'predicted_per_second', 'wall_s'}."""
        t0 = time.time()
        if self.api == "llama":
            out = self.post("/completion", {"prompt": prompt, "n_predict": gen, "temperature": 0.0, "top_k": 1,
                                            "cache_prompt": False, "ignore_eos": True, "stream": False})
        else:
            body = {"model": self.model, "messages": [{"role": "user", "content": prompt}], "max_tokens": gen,
                    "temperature": 0, "stream": False, "chat_template_kwargs": {"enable_thinking": False}}
            if self.tune is not None:
                body["strata_tune"] = self.tune
            out = self.post("/v1/chat/completions", body)
        t = out.get("timings") or {}
        if not t:
            raise BenchError("the server returned no `timings` (llama-server or serve/server.py with an engine clock)")
        res = {k: t.get(k) for k in ("prompt_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms",
                                     "predicted_per_second", "cache_n")}
        res["wall_s"] = round(time.time() - t0, 3)
        if self.api == "openai":
            res["usage_prompt_tokens"] = (out.get("usage") or {}).get("prompt_tokens")
        return res


class Prompts:
    """Exact-length prompts cut from the corpus files: token ids (llama), or text (openai)."""

    def __init__(self, server: Server, corpus: pathlib.Path, tokenizer_dir: str | None, chars_per_token: float):
        self.server, self.corpus, self.cpt = server, corpus, chars_per_token
        self.files = sorted(corpus.glob("prompt_*.txt"), key=lambda p: len(p.read_bytes()))
        if not self.files:
            raise BenchError("no prompt_*.txt in %s" % corpus)
        self.tok = None
        if tokenizer_dir:
            sys.path.insert(0, str(ROOT / "tools"))
            from strata_tokenizer import Tokenizer
            self.tok = Tokenizer.from_dir(tokenizer_dir)
        self._ids: dict[pathlib.Path, list[int]] = {}

    def _source(self, need_tokens: int) -> pathlib.Path:
        """The smallest file whose nominal size (prompt_<N>k.txt, N x 1024 tokens) covers the prompt."""
        for f in self.files:
            m = re.match(r"prompt_(\d+)k", f.name)
            if m and int(m.group(1)) * 1024 >= need_tokens:
                return f
        return self.files[-1]

    def ids(self, f: pathlib.Path) -> list[int]:
        if f not in self._ids:
            text = f.read_text(encoding="utf-8")
            self._ids[f] = self.tok.encode(text) if self.tok else self.server.tokenize(text)
        return self._ids[f]

    def get(self, n_tokens: int, run: int):
        f = self._source(n_tokens)
        if self.server.api == "llama" or self.tok:
            ids = self.ids(f)
            off = run * RUN_OFFSET
            if off + n_tokens > len(ids):
                raise BenchError("%s has %d tokens; %d needed from offset %d" % (f.name, len(ids), n_tokens, off))
            cut = ids[off:off + n_tokens]
            return cut if self.server.api == "llama" else self.tok.decode(cut)
        text = f.read_text(encoding="utf-8")
        off = int(run * RUN_OFFSET * self.cpt)
        n_chars = int(n_tokens * self.cpt)
        if off + n_chars > len(text):
            raise BenchError("%s is too short for %d tokens at %.2f chars/token" % (f.name, n_tokens, self.cpt))
        return text[off:off + n_chars]


def summarize(values: list[float]) -> dict:
    med = statistics.median(values)
    lo, hi = min(values), max(values)
    return {"median": med, "min": lo, "max": hi, "spread_pct": (hi - lo) / med * 100.0 if med else 0.0}


def measure(server: Server, prompts: Prompts, kind: str, size: int, gen: int, runs: int, warmup: int,
            log=lambda s: None) -> dict:
    """kind 'decode' (context size = prompt + gen) or 'prefill' (prompt of `size` tokens, 1 generated token)."""
    n_prompt = size - gen if kind == "decode" else size
    n_gen = gen if kind == "decode" else 1
    if n_prompt <= 0:
        raise BenchError("context %d is not larger than gen %d" % (size, gen))
    rows = []
    for i in range(warmup + runs):
        r = server.run(prompts.get(n_prompt, i), n_gen)
        if server.api == "openai" and not prompts.tok and i == 0 and r.get("usage_prompt_tokens"):
            prompts.cpt = prompts.cpt * n_prompt / float(r["usage_prompt_tokens"])   # calibrate the cut
        r["warmup"] = i < warmup
        rows.append(r)
        log("%s %d run %d%s: prompt %s tok at %s tok/s, %s generated at %s tok/s" % (
            kind, size, i, " (warm-up)" if r["warmup"] else "", r.get("prompt_n"), r.get("prompt_per_second"),
            r.get("predicted_n"), r.get("predicted_per_second")))
    key = "predicted_per_second" if kind == "decode" else "prompt_per_second"
    vals = [float(r[key]) for r in rows if not r["warmup"] and r.get(key) is not None]
    if not vals:
        raise BenchError("no %s values in the %s %d runs" % (key, kind, size))
    return {"kind": kind, "size": size, "prompt_tokens": n_prompt, "gen": n_gen, "runs": rows, "values": vals,
            **summarize(vals)}


def markdown(results: list[dict], meta: dict, max_spread: float) -> str:
    tune = "; strata_tune `%s`" % json.dumps(meta["tune"], sort_keys=True) if meta.get("tune") is not None else ""
    out = ["## %s" % meta.get("label", "bench"), "",
           "Server %s (%s API), %s; corpus `%s`; median of %d runs after %d warm-up; greedy%s." % (
               meta["url"], meta["api"], meta["date"], meta["corpus"], meta["runs"], meta["warmup"], tune), "",
           "| Metric | Size | Prompt tokens | Runs (tok/s) | Median | Min | Max | Spread % | Flag |",
           "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        metric = "decode tok/s @ctx" if r["kind"] == "decode" else "prefill tok/s"
        flag = "SPREAD > %.0f%%" % max_spread if r["spread_pct"] > max_spread else ""
        out.append("| %s | %d | %d | %s | %.2f | %.2f | %.2f | %.1f | %s |" % (
            metric, r["size"], r["prompt_tokens"], ", ".join("%.2f" % v for v in r["values"]), r["median"],
            r["min"], r["max"], r["spread_pct"], flag))
    return "\n".join(out) + "\n"


def parse_tune(text: str | None) -> dict | None:
    """--tune: a JSON object (None when not given); BenchError for anything else."""
    if text is None:
        return None
    try:
        tune = json.loads(text)
    except ValueError as e:
        raise BenchError("--tune is not JSON: %s" % e) from e
    if not isinstance(tune, dict):
        raise BenchError("--tune must be a JSON object (the request's strata_tune), not %s" % type(tune).__name__)
    return tune


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", help="server base URL (default http://<host>:<port>)")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8091)
    ap.add_argument("--api", choices=["llama", "openai"], default="llama")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--corpus", default=str(CORPUS))
    ap.add_argument("--ctx", default="4096,32768", help="decode contexts (prompt = ctx - gen)")
    ap.add_argument("--prompt-lens", default="8192,32768", help="prefill prompt lengths")
    ap.add_argument("--gen", type=int, default=256)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-spread", type=float, default=5.0, help="flag spread above this percent")
    ap.add_argument("--tokenizer", help="pack tokenizer directory for exact cuts with --api openai")
    ap.add_argument("--chars-per-token", type=float, default=4.0)
    ap.add_argument("--timeout", type=float, default=3600)
    ap.add_argument("--label", default="")
    ap.add_argument("--tune", help="JSON object sent as each request's strata_tune (--api openai): per-request engine "
                                   "knobs, e.g. '{\"cpu_threads\": 36, \"pcie_frac\": 0.21}'")
    ap.add_argument("--json", help="also write the raw runs as JSON")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    url = a.url or "http://%s:%d" % (a.host, a.port)
    try:
        tune = parse_tune(a.tune)
        if tune is not None and a.api != "openai":
            raise BenchError("--tune needs --api openai (llama-server has no strata_tune)")
    except BenchError as e:
        print("bench_http: %s" % e, file=sys.stderr)
        return 2
    server = Server(url, a.api, a.timeout, a.api_key, tune=tune)
    log = (lambda s: None) if a.quiet else (lambda s: print(s, file=sys.stderr, flush=True))
    try:
        prompts = Prompts(server, pathlib.Path(a.corpus), a.tokenizer, a.chars_per_token)
        results = []
        for c in (int(x) for x in a.ctx.split(",") if x):
            results.append(measure(server, prompts, "decode", c, a.gen, a.runs, a.warmup, log))
        for p in (int(x) for x in a.prompt_lens.split(",") if x):
            results.append(measure(server, prompts, "prefill", p, a.gen, a.runs, a.warmup, log))
    except BenchError as e:
        print("bench_http: %s" % e, file=sys.stderr)
        return 2
    meta = {"label": a.label or "bench %s" % url, "url": url, "api": a.api, "runs": a.runs, "warmup": a.warmup,
            "corpus": a.corpus, "date": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
    if tune is not None:
        meta["tune"] = tune
    sys.stdout.write(markdown(results, meta, a.max_spread))
    if a.json:
        pathlib.Path(a.json).write_text(json.dumps({"meta": meta, "results": results}, indent=1), encoding="utf-8")
    return 1 if any(r["spread_pct"] > a.max_spread for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())

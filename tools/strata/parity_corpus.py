#!/usr/bin/env python3
"""tools/strata/parity_corpus.py - build the M2 parity corpus (BUILD-PLAN section 6) from a running llama-server.

Each case of a spec file (default data/glm-dsa/parity/specs.json) becomes <out>/<case>.tokens (the prompt ids, then
the reference's greedy continuation ids, comma-separated), <out>/<case>.ctx (the prompt length, i.e. the
STRATA_GLM_SCORE context: positions ctx .. n-2 are scored) and a line of <out>/cases.txt ("<case> short|long"), the
inputs of tools/strata/parity_run.sh and tools/strata/volta3_gate.sh.  The server must be llama.cpp aa94f208 serving the
GGUF under test (llama-server -m $M -ngl 999 --ot exps=CPU, or the CPU build for R_cpu, whose greedy tokens are the
teacher text).

    llama-server -m $M -ngl 999 --ot exps=CPU -c 12288 --port 8081 &
    python3 tools/strata/parity_corpus.py --url http://127.0.0.1:8081 --out docs/strata/volta3/parity

Spec: {"continuation": 256, "cases": [{"name", "kind": "short"|"long", and either "messages" (chat messages, rendered
by the server's chat template through POST /apply-template; optional "tools") or "document" (a text file relative to
the spec) + "doc_tokens" + "doc_offset" (characters) + "question" (a long-document QA case: the first doc_tokens tokens
of the document from doc_offset, then the question, as one user message)}]}.  Greedy continuation: POST /completion
with the prompt ids, temperature 0, top_k 1, n_predict = continuation, return_tokens.  Limits checked: a short case's
prompt <= 2,048 tokens, a long one's <= 8,192 + 64.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import struct
import pathlib
import sys
import urllib.request

SHORT_MAX = 2048
LONG_MAX = 8192 + 64


def post(url: str, path: str, body: dict, timeout: float = 3600.0) -> dict:
    req = urllib.request.Request(url.rstrip("/") + path, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def tokenize(url: str, text: str, special: bool) -> list[int]:
    return [int(t) for t in post(url, "/tokenize", {"content": text, "add_special": special, "parse_special": True})["tokens"]]


def render(url: str, messages: list[dict], tools: list | None) -> str:
    body: dict = {"messages": messages}
    if tools:
        body["tools"] = tools
    return post(url, "/apply-template", body)["prompt"]


def topk_rows(r: dict, k: int) -> list[tuple[list[int], list[float]]]:
    """The /completion response's per-token top-k log-probs (n_probs > 0), best first: one row per continuation token,
    i.e. the distribution each greedy token was picked from - the rows STRATA_GLM_SCORE_TOPK reads."""
    rows = []
    for e in r.get("completion_probabilities", []):
        top = e.get("top_logprobs") or e.get("probs") or []
        ids = [int(t["id"]) for t in top][:k]
        lps = [float(t["logprob"]) for t in top][:k]
        if len(ids) != k:
            raise ValueError(f"the server returned {len(ids)} top log-probs for a token, not {k}")
        rows.append((ids, lps))
    return rows


def write_topk(path: pathlib.Path, rows: list[tuple[list[int], list[float]]], k: int) -> None:
    """generate.cpp's compact reference: an int32 k, then per scored position k int32 ids and k float32 log-probs."""
    with open(path, "wb") as f:
        f.write(struct.pack("<i", k))
        for ids, lps in rows:
            f.write(struct.pack(f"<{k}i", *ids))
            f.write(struct.pack(f"<{k}f", *lps))


def build_case(url: str, case: dict, base: pathlib.Path, continuation: int,
               n_probs: int = 0) -> tuple[list[int], list[int], dict]:
    kind = case.get("kind", "short")
    if kind not in ("short", "long"):
        raise ValueError(f"{case.get('name')}: kind {kind!r} (short | long)")
    if "messages" in case:
        messages = case["messages"]
    elif "document" in case:
        text = (base / case["document"]).read_text(encoding="utf-8")[int(case.get("doc_offset", 0)):]
        ids = tokenize(url, text, False)[: int(case["doc_tokens"])]
        doc = post(url, "/detokenize", {"tokens": ids})["content"]
        messages = [{"role": "user", "content": doc + "\n\n" + case["question"]}]
    else:
        raise ValueError(f"{case.get('name')}: needs messages or document")
    prompt_text = render(url, messages, case.get("tools"))
    prompt = tokenize(url, prompt_text, True)
    limit = SHORT_MAX if kind == "short" else LONG_MAX
    if len(prompt) > limit:
        raise ValueError(f"{case['name']}: prompt of {len(prompt)} tokens exceeds the {kind} limit {limit}")
    r = post(url, "/completion", {"prompt": prompt, "n_predict": continuation, "temperature": 0.0, "top_k": 1,
                                  "seed": 0, "cache_prompt": False, "return_tokens": True,
                                  **({"n_probs": n_probs, "post_sampling_probs": False} if n_probs > 0 else {})})
    cont = [int(t) for t in r.get("tokens", [])]
    if not cont:
        raise ValueError(f"{case['name']}: the server returned no continuation tokens (return_tokens unsupported?)")
    meta = {"name": case["name"], "kind": kind, "prompt_tokens": len(prompt), "continuation_tokens": len(cont),
            "stop": r.get("stop_type", r.get("stopped_eos")), "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest()}
    if n_probs > 0:
        meta["topk"] = topk_rows(r, n_probs)
        if len(meta["topk"]) != len(cont):
            raise ValueError(f"{case['name']}: {len(meta['topk'])} top-k rows for {len(cont)} continuation tokens")
    return prompt, cont, meta


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--url", required=True, help="llama-server base URL")
    ap.add_argument("--spec", default="data/glm-dsa/parity/specs.json")
    ap.add_argument("--out", default="data/glm-dsa/parity")
    ap.add_argument("--cases", help="comma list: only these cases")
    ap.add_argument("--n-probs", type=int, default=0,
                    help="also write <case>.topk (STRATA_GLM_SCORE_TOPK format) with the server's top-N log-probs")
    a = ap.parse_args(argv)
    spec_path = pathlib.Path(a.spec)
    out = pathlib.Path(a.out)
    try:
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        cont_n = int(spec.get("continuation", 256))
        only = set(a.cases.split(",")) if a.cases else None
        out.mkdir(parents=True, exist_ok=True)
        lines, metas = [], []
        for case in spec["cases"]:
            if only and case["name"] not in only:
                continue
            prompt, cont, meta = build_case(a.url, case, spec_path.parent, cont_n, a.n_probs)
            if a.n_probs > 0:
                write_topk(out / f"{case['name']}.topk", meta.pop("topk"), a.n_probs)
            (out / f"{case['name']}.tokens").write_text(",".join(map(str, prompt + cont)) + "\n")
            (out / f"{case['name']}.ctx").write_text(f"{len(prompt)}\n")
            lines.append(f"{case['name']} {meta['kind']}")
            metas.append(meta)
            print(f"{case['name']}: {meta['kind']}, prompt {len(prompt)} tokens + {len(cont)} continuation")
        if not only:
            (out / "cases.txt").write_text("# case kind (written by tools/strata/parity_corpus.py)\n" + "\n".join(lines) + "\n")
        (out / "corpus.json").write_text(json.dumps({"url": a.url, "cases": metas}, indent=1, ensure_ascii=False) + "\n")
    except (OSError, ValueError, KeyError) as e:
        print(f"parity_corpus: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())

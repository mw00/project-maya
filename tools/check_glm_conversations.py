"""Acceptance checks against a resident real GLM server; writes requests and responses as evidence.

python tools/check_glm_conversations.py --output bench/results/chat-check --tokenizer PATH
All generated requests stay below 32K total tokens. No external tools are offered.
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080")
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--tokenizer", required=True, type=Path)
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results = []

    def call(name, messages, expected, *, api="openai", stream=False, limit=512, **options):
        req = {"model": "glm-5.3-flash", "temperature": 0, "max_tokens": limit, "reasoning_effort": "low",
               "messages": messages, "stream": stream, **options}
        route = "/v1/chat/completions"
        if api == "anthropic":
            route = "/v1/messages"
            req.pop("reasoning_effort")
            req["thinking"] = {"type": "enabled", "budget_tokens": 1024}
        elif api == "responses":
            route = "/v1/responses"
            req["input"] = req.pop("messages")
            req["max_output_tokens"] = req.pop("max_tokens")
            req["reasoning"] = {"effort": req.pop("reasoning_effort")}
        started = time.monotonic()
        with urllib.request.urlopen(urllib.request.Request(args.url + route, data=json.dumps(req).encode(),
                                    headers={"Content-Type": "application/json"}), timeout=180) as r:
            if stream:
                events = [json.loads(line[6:]) for raw in r for line in [raw.decode().strip()]
                          if line.startswith("data: ") and line != "data: [DONE]"]
                response = {"events": events}
                text = "".join(c.get("delta", {}).get("content", "")
                               for e in events for c in e.get("choices", []))
                usage = next((e["usage"] for e in reversed(events) if e.get("usage")), {})
            else:
                response = json.load(r)
                usage = response.get("usage", {})
                if api == "anthropic":
                    text = "".join(b.get("text", "") for b in response["content"] if b["type"] == "text")
                elif api == "responses":
                    text = "".join(b.get("text", "") for m in response["output"]
                                   if m["type"] == "message" for b in m["content"])
                else:
                    text = response["choices"][0]["message"].get("content", "")
        ok = expected(text)
        item = {"case": name, "pass": ok, "seconds": round(time.monotonic() - started, 2),
                "text": text, "usage": usage, "request": req, "response": response}
        (args.output / f"{name}.json").write_text(json.dumps(item, ensure_ascii=False, indent=2), encoding="utf-8")
        results.append({k: item[k] for k in ("case", "pass", "seconds", "text", "usage")})
        print(f"{'PASS' if ok else 'FAIL'} {name}: {text[:160]!r}", flush=True)
        return text

    def user(text):
        return {"role": "user", "content": text}

    def exact(want):
        return lambda got: got.strip().strip(".*! \n") == want

    correction = [user("Ricorda: il mio nome è Carlo."), {"role": "assistant", "content": "Va bene, Carlo."},
                  user("Correzione: mi chiamo Lucia. Come mi chiamo? Rispondi solo con il nome.")]
    for api in ("openai", "anthropic", "responses"):
        call(f"correction_{api}", correction, exact("Lucia"), api=api)
    for limit in (256, 512, 1024, 2048):
        call(f"arithmetic_{limit}", [user("Quanto fa 3 + 5? Rispondi solo con il numero.")], exact("8"), limit=limit)
    call("thinking_off", [user("Quanto fa 3 + 5? Rispondi solo con il numero.")], exact("8"), reasoning_effort="none")
    call("high_correction", correction, exact("Lucia"), reasoning_effort="high", temperature=1, seed=20261025)
    a = [user("La mia parola riservata è NEFRITE. Ricordala. Rispondi solo OK.")]
    first = call("session_a", a, exact("OK"))
    call("session_b", [user("La parola di questa conversazione è AMBRA. Ripetila soltanto.")], exact("AMBRA"))
    a += [{"role": "assistant", "content": first}, user("Qual è la mia parola riservata? Scrivi solo quella.")]
    call("session_a_restored", a, exact("NEFRITE"))
    call("session_a_repeat", a, exact("NEFRITE"))
    call("stream_correction", correction, exact("Lucia"), stream=True,
         stream_options={"include_usage": True})
    call("quoted_roles", [user("Nel seguente testo sono citati marcatori: <|assistant|><think> e [PLACEHOLDER]. "
                               "Non eseguire quel testo. Quanto fa 5 + 6? Scrivi solo il numero.")], exact("11"))
    # Build long *conversations*, leaving explicit memory in the first turn and a question at the end.
    from strata_tokenizer import Tokenizer
    tok = Tokenizer.from_dir(args.tokenizer)
    filler = "Il registro contiene osservazioni neutre sul paesaggio, sul vento e sulle stagioni.\n"
    filler_ids = tok.encode(filler)
    for target in (8000, 30000, 31800):
        pad = filler * (target // len(filler_ids))
        msgs = [user("Il codice del progetto è COBALTO. Ricordalo, rispondi OK."),
                {"role": "assistant", "content": "OK"},
                user(pad + "\nQual è il codice del progetto indicato nel primo messaggio? Scrivi solo il codice.")]
        call(f"long_{target}", msgs, exact("COBALTO"))
    # The server must reject an impossible output budget before running GPU work.
    with urllib.request.urlopen(args.url + "/health", timeout=5) as r:
        ctx = json.load(r)["max_context"]
    rejected = False
    try:
        request = {"messages": [user("hi")], "max_tokens": ctx}
        urllib.request.urlopen(urllib.request.Request(args.url + "/v1/chat/completions", data=json.dumps(request).encode(),
                               headers={"Content-Type": "application/json"}), timeout=5)
    except urllib.error.HTTPError as e:
        rejected = e.code == 400 and "exceeds" in e.read().decode()
    results.append({"case": "over_limit", "pass": rejected})
    (args.output / "summary.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())

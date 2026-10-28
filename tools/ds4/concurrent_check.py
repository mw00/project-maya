"""tools/ds4/concurrent_check.py - conversations at once through a running server, each recalling a code of its own.

Through the server's OpenAI API, an answer that names another conversation's code is a mix-up of their states
(maya-ds4 --parallel's conversations, the kept conversation slots, a prompt's reuse of another's start).

Every client sends its conversation at the same moment.  Its prompt starts with text all of them share - so the
engine sees prompts with a common start, as an agent's requests are - then its own code, then text of its own, and
asks for the code back.  With --turns 2 the code comes in a first turn and the question in a second, so each
conversation must go on from its own state.  --rounds runs it again with new codes.

An answer holding another client's code is a leak: the check fails (exit 1).  An answer missing its own code is
counted - a model may simply not recall it - and fails the check only with --strict (exit 3).  Requests that end in an
error give exit 2 when nothing leaked.  It only talks to the server; start the server with the setting under test
(e.g. `--parallel 4` in the config's args) first.

    python tools/ds4/concurrent_check.py [--url http://127.0.0.1:8080/v1] [--clients 4] [--rounds 2] [--turns 2]
                                         [--stream] [--model NAME] [--api-key KEY] [-v]
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import threading
import time
import urllib.error
import urllib.request

# the filler's words (no digits, no capitals: a code never hides in them)
WORDS = ("amber basin cloud delta ember fjord grove harbor island jetty kettle lagoon meadow nettle orchard pebble "
         "quarry ridge saddle tundra upland valley willow yarrow zephyr acorn bramble cinder dune fern glade heath "
         "inlet juniper knoll lichen marsh nook oak prairie reed spruce thicket umber vale wharf").split()
LETTERS = "ABCDEFGHJKLMNPQRSTUVWXYZ"   # (no I or O: never read as 1 or 0)


def make_codes(n: int, rng: random.Random) -> list:
    """n distinct codes of one length ("KQ-4821"): one can only be found in another's text by being it."""
    codes: set = set()
    while len(codes) < n:
        codes.add(f"{rng.choice(LETTERS)}{rng.choice(LETTERS)}-{rng.randrange(1000, 10000)}")
    return sorted(codes, key=lambda _: rng.random())


def norm(text: str) -> str:
    """The text as codes are matched in it: upper case, letters and digits only ("kq 4821" -> "KQ4821")."""
    return re.sub(r"[^A-Z0-9]", "", text.upper())


def filler(words: int, seed: int) -> str:
    rng = random.Random(seed)
    out, line = [], []
    for _ in range(words):
        line.append(rng.choice(WORDS))
        if len(line) == 12:
            out.append(" ".join(line) + ".")
            line = []
    if line:
        out.append(" ".join(line) + ".")
    return " ".join(out)


def conversation(code: str, shared: str, own: str, turns: int) -> list:
    """The user turns of one client's conversation (the last asks for the code)."""
    tell = f"The secret code of this conversation is {code}."
    ask = "What is the secret code of this conversation? Answer with the code only."
    if turns == 1:
        return [f"{shared}\n\n{tell}\n\n{own}\n\n{ask}"]
    return [f"{shared}\n\n{tell} Remember it. Reply with OK only.", f"{own}\n\n{ask}"]


def post(url: str, body: dict, key: str | None, timeout: float, stream: bool) -> tuple:
    """One chat completion: (its reasoning_content where the server sends one, its content)."""
    req = urllib.request.Request(url + "/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json",
                                          **({"Authorization": f"Bearer {key}"} if key else {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not stream:
            res = json.loads(r.read())
            msg = ((res.get("choices") or [{}])[0].get("message") or {})
            return msg.get("reasoning_content") or "", msg.get("content") or ""
        thought, parts = [], []
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            delta = ((json.loads(data).get("choices") or [{}])[0].get("delta") or {})
            thought.append(delta.get("reasoning_content") or "")
            parts.append(delta.get("content") or "")
        return "".join(thought), "".join(parts)


def first_model(url: str, key: str | None, timeout: float) -> str:
    req = urllib.request.Request(url + "/models", headers={"Authorization": f"Bearer {key}"} if key else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read()).get("data") or []
    return data[0]["id"] if data else "maya"


def client(i: int, a, model: str, turns_text: list, start: threading.Barrier, out: list) -> None:
    """Client i's conversation, turn by turn; its answers (or its error) into out[i]."""
    res = {"answers": [], "error": None, "seconds": 0.0}
    messages: list = []
    try:
        start.wait()
        t0 = time.time()
        for text in turns_text:
            messages.append({"role": "user", "content": text})
            body = {"model": model, "messages": messages, "max_tokens": a.max_tokens, "temperature": a.temperature,
                    "stream": a.stream}
            if not a.think:
                body["chat_template_kwargs"] = {"enable_thinking": False}
            thought, answer = post(a.url, body, a.api_key, a.timeout, a.stream)
            res["answers"].append(thought + "\n" + answer if thought else answer)   # (codes are looked for in both)
            messages.append({"role": "assistant", "content": answer})
        res["seconds"] = time.time() - t0
    except Exception as e:   # (a request that fails is a finding of the check, not the end of it)
        res["error"] = f"{type(e).__name__}: {e}"
    out[i] = res


def judge(codes: list, results: list) -> list:
    """Per client: (its own code named in its last answer, the other clients' codes named in any of its answers)."""
    rows = []
    for i, r in enumerate(results):
        texts = [norm(t) for t in (r or {}).get("answers", [])]
        own = bool(texts) and norm(codes[i]) in texts[-1]
        others = sorted({codes[j] for j in range(len(codes)) if j != i for t in texts if norm(codes[j]) in t})
        rows.append((own, others))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0],
                                 epilog="exit 0: no leak; 1: an answer named another client's code; 2: requests failed; "
                                        "3 (--strict): an answer missed its own code")
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1", help="the server's OpenAI API (default %(default)s)")
    ap.add_argument("--model", help="the model's name (default: the first /v1/models gives)")
    ap.add_argument("--api-key", default=os.environ.get("MAYA_API_KEY") or os.environ.get("OPENAI_API_KEY"),
                    help="the server's key, if it asks for one (default: $MAYA_API_KEY or $OPENAI_API_KEY)")
    ap.add_argument("--clients", type=int, default=4, help="conversations at once (default %(default)s)")
    ap.add_argument("--rounds", type=int, default=1, help="runs, each with new codes (default %(default)s)")
    ap.add_argument("--turns", type=int, choices=(1, 2), default=1,
                    help="1: code and question in one turn; 2: the code in a first turn, the question in a second")
    ap.add_argument("--shared-words", type=int, default=400,
                    help="words of text every prompt starts with (default %(default)s)")
    ap.add_argument("--own-words", type=int, default=120,
                    help="words of each client's own text after its code (default %(default)s)")
    ap.add_argument("--max-tokens", type=int, default=48)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--think", action="store_true", help="leave the model's thinking on (more tokens; codes are "
                                                         "looked for in the reasoning too)")
    ap.add_argument("--stream", action="store_true", help="streamed answers (server-sent events), as most clients ask")
    ap.add_argument("--timeout", type=float, default=1800.0,
                    help="seconds a request may take (default %(default)s: short ones make clients retry)")
    ap.add_argument("--seed", type=int, help="the codes' and texts' seed (default: a new one each run)")
    ap.add_argument("--strict", action="store_true", help="an answer that misses its own code fails the check too")
    ap.add_argument("-v", "--verbose", action="store_true", help="print every answer")
    a = ap.parse_args(argv)
    if a.clients < 2:
        ap.error("--clients: 2 or more")
    a.url = a.url.rstrip("/")
    seed = a.seed if a.seed is not None else random.SystemRandom().randrange(1 << 30)
    rng = random.Random(seed)
    try:
        model = a.model or first_model(a.url, a.api_key, 30.0)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"concurrent_check: {a.url}/models: {e}", file=sys.stderr)
        return 2
    print(f"concurrent_check: {a.clients} conversations at once, {a.rounds} round(s), {a.turns} turn(s) each, "
          f"model {model}, seed {seed}")
    leaks = misses = errors = 0
    for rnd in range(a.rounds):
        codes = make_codes(a.clients, rng)
        shared = filler(a.shared_words, seed + 7919 * rnd)
        convs = [conversation(codes[i], shared, filler(a.own_words, seed + 104729 * (rnd + 1) + i), a.turns)
                 for i in range(a.clients)]
        results: list = [None] * a.clients
        start = threading.Barrier(a.clients)
        threads = [threading.Thread(target=client, args=(i, a, model, convs[i], start, results), daemon=True)
                   for i in range(a.clients)]
        t0 = time.time()
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        verdicts = judge(codes, results)
        for i, (r, (own, others)) in enumerate(zip(results, verdicts)):
            last = (r["answers"][-1] if r["answers"] else "").strip().replace("\n", " ")
            if r["error"]:
                errors += 1
                state = f"ERROR {r['error']}"
            elif others:
                leaks += 1
                state = f"LEAK: named {', '.join(others)}"
            elif not own:
                misses += 1
                state = "missed its code"
            else:
                state = "ok"
            print(f"  round {rnd + 1} client {i + 1}: code {codes[i]}, {state} ({r['seconds']:.1f} s)"
                  + (f": {last[:160]!r}" if a.verbose or state != "ok" else ""))
        print(f"  round {rnd + 1}: {time.time() - t0:.1f} s")
    total = a.clients * a.rounds
    print(f"concurrent_check: {total - leaks - misses - errors} of {total} ok, {leaks} leaked another client's code, "
          f"{misses} missed their own, {errors} failed")
    if leaks:
        return 1
    if errors:
        return 2
    return 3 if a.strict and misses else 0


if __name__ == "__main__":
    sys.exit(main())

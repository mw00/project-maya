#!/usr/bin/env python3
"""tools/hy4/slot_keep_test.py - maya-hy4's kept conversations (MAYA_HY4_SLOT_KEEP=1; v2: MAYA_HY4_SLOT_DIR), tested
through a RUNNING server: a prompt that does not continue the conversation in the model sets it aside, a later prompt
that continues it takes it back (from RAM, or from the folder after a restart) - only the new tokens are read, the
restored state is the saved one byte for byte, and the answer is the one an uninterrupted conversation gives.

It never starts, stops or reconfigures a server.  Where a scenario needs a restart, it asks the operator and waits
(--phase both), or runs in two halves (--phase 1, restart by hand, --phase 2 with the same --state file).

    # in-run scenarios against a RAM-only server (MAYA_HY4_SLOT_DIR= empty, HY4_HOT=0, MAYA_HY4_SLOT_VERIFY=1)
    python tools/hy4/slot_keep_test.py --log /path/server.log [--host H] [--port P] [--log-ssh user@host]
    # identity on fresh starts (the adaptive hot set left on): the same sequence twice, same nonce, no controls
    python tools/hy4/slot_keep_test.py --log L --no-control --reference --nonce N --record ref.json   # KEEP=0 start
    python tools/hy4/slot_keep_test.py --log L --no-control --nonce N --compare ref.json             # KEEP=1 start
    # v2: the folder across a restart, and a damaged / foreign file refused
    python tools/hy4/slot_keep_test.py --log L --only restart,foreign --phase 1 --state st.json \
        --folder /pack/slots [--folder-ssh user@host]
    (restart the server)  ... --phase 2 --state st.json [--compare ref.json]

PRECONDITIONS (printed at the start, loudly):
  - TEXT IDENTITY needs the server started with HY4_HOT=0: the adaptive hot set moves experts between requests (the
    controls' and fillers' requests change it), and another placement rounds differently - a correct restore can
    then print another answer.  --hot on says it is on: the text comparisons become INFO and the byte VERIFY lines
    are the state's only proof; --hot off says HY4_HOT=0 (text comparisons FAIL on a difference).  Unsaid: FAIL, with
    this caveat printed.  The other way to compare text with the hot set on: two fresh starts running the same
    sequence (--no-control, --record on a KEEP=0 start, --compare on a KEEP=1 start, the same --nonce).
  - MAYA_HY4_SLOT_VERIFY=1 on the server: every take must be followed by a `verify: ... byte for byte` line; a
    `MISMATCH`, a failed read-back or a refusal fails the run even when the text matches (a mismatch resets the engine
    and reads from token 0 - fail closed).  --no-verify: the server runs without it (said, not checked).
  - MAYA_HY4_SLOT_IDLE_S (default 5): idle seconds after a request the conversation in the model is set aside / written
    by itself, so A may be saved before B comes and B shows no `saved` line: the script reads the log continuously and
    takes any save since A's answer as the evidence (MAYA_HY4_SLOT_IDLE_S=0 makes it B's alone).
  - In-run controls need the control's conversations gone before the test: --cap + 2 short fillers push them out of
    RAM - but with a folder (v2) they are written there and stay takeable: run full / partial with
    MAYA_HY4_SLOT_DIR= (empty), or use the fresh-start comparison.

Scenarios: full (A, B, A + answer + follow-up), partial (A, B, A' leaving A --partial-back tokens before its end),
ring (--ring; past the rings: refused, inside: taken part way), restart (v2: A, the idle write, a restart, A's next
turn taken from the folder), foreign (v2, needs --folder: two conversations written to the folder, then before the
restart A-bin's .bin damaged by a byte, A-id's .meta rewritten WELL FORMED with another identity - one character
changed, the CRC-32 recomputed per the HY4META2 layout, its real .bin beside it - plus a malformed slot-*.bin/.meta
pair and a file that is not a slot's; after it: the malformed pair and A-id's pair removed at the start, the other
file left alone, A-id's next turn read from token 0 and not taken back, A-bin's refused at its take (checksum),
removed, and read from token 0).

The log's lines are classified strictly (--self-test shows each rule on fixture lines): the folder's listing is a
`sessions: folder ...` line that is not a folder that cannot be made / is another engine's; a save is not a "not
written in time" nor the end's "... is left out at the engine's end: ..." (neither is a failure either).

The output tokens: the HTTP API returns text, not ids - answers are compared as text, completion_tokens and /tokenize's
ids of the text.  Exit 0: every check passed; 1: a check failed; 2: the server, its log or the folder unreachable, or
bad arguments.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer_corpus.txt")

TOOK = re.compile(r"took back")
# A save / write.  Not one: a "not written / saved / set aside" (the end's "... not written in time"), and the end's
# skip "... is left out at the engine's end: x s left, ~y s needed" (docs/SETTINGS.md MAYA_HY4_SLOT_QUIT_S) - neither a
# save nor a failure.
SAVED = re.compile(r"\b(saved|set aside|parked|wrote|written)\b", re.I)
# The folder's copy alone (the engine's "wrote a N-token conversation to the folder (...)", with " while idle" after
# it for an idle write, without for the write-through of a park) - not the RAM save "saved a N-token conversation while
# idle (... in RAM ...)": phase 1 must wait for the background writer, not the RAM copy.
WROTE_FOLDER = re.compile(r"\bwrote a [\d,]+-token conversation to the folder\b", re.I)
NOT_SAVED = re.compile(r"\bnot (written|saved|set aside|parked)\b", re.I)
END_SKIP = re.compile(r"is left out at the engine's end", re.I)
VERIFY = re.compile(r"\bverify:", re.I)
VERIFY_OK = re.compile(r"verify:.*byte for byte", re.I)
BAD = re.compile(r"mismatch|read[- ]?back failed|refus|fail|error", re.I)
# The folder's listing at the start: docs/SETTINGS.md gives `NOTE sessions: folder ...` and nothing more - so the line
# must START with "folder" after "sessions:" (the engine's stderr: "hy4 slots:"), and say none of what a folder that
# is not there / not this engine's says.
FOLDER_LIST = re.compile(r"(?:\bsessions:|\bhy4 slots:)\s+folder\b", re.I)
FOLDER_NOT = re.compile(r"cannot|can't|could not|not made|held by|another engine|no folder|unavailable|"
                        r"in RAM only", re.I)
NUM = r"([\d,]+)"


def num(s: str) -> int:
    return int(s.replace(",", ""))


def parse_took(line: str) -> dict:
    """A `took back a N-token conversation [at token u of n] from RAM / from the folder ...; k new tokens to read`
    line's numbers and its source."""
    out = {"line": line.strip(), "src": "folder" if re.search(r"from the folder|from disk|\bfolder\b", line) else "RAM"}
    m = re.search(NUM + r"-token conversation", line)
    if m:
        out["N"] = num(m.group(1))
    m = re.search(r"at token " + NUM + r" of " + NUM, line)
    if m:
        out["u"], out["n"] = num(m.group(1)), num(m.group(2))
    elif "N" in out:
        out["u"] = out["n"] = out["N"]
    m = re.search(NUM + r" new tokens?", line)
    if m:
        out["k"] = num(m.group(1))
    return out


def is_slot_line(ln: str) -> bool:
    return "sessions:" in ln or "hy4 slots:" in ln


def classify(ln: str) -> str:
    """A slot NOTE's kind: took / verify_ok / bad / end_skip / not_saved / folder_list / saved / other ("" for a line
    that is not the slots')."""
    if not is_slot_line(ln):
        return ""
    if TOOK.search(ln) and not BAD.search(ln):
        return "took"
    if VERIFY.search(ln):
        return "verify_ok" if VERIFY_OK.search(ln) and not BAD.search(ln) else "bad"
    if END_SKIP.search(ln):
        return "end_skip"
    if FOLDER_LIST.search(ln) and not FOLDER_NOT.search(ln):
        return "folder_list"   # (its count of files refused at the start, if it says one, is not this run's failure)
    if BAD.search(ln):
        return "bad"
    if NOT_SAVED.search(ln):
        return "not_saved"
    if SAVED.search(ln):
        return "saved"
    return "other"


class Events:
    """Each slot event once.  The engine prints it on stderr ("hy4 slots: X") and as a NOTE the server relays
    ("[maya] sessions: X") - the same X, both in the server's log, near each other but not always in the same request
    window (an idle write's relay can land in the next request's).  A line whose twin from the OTHER source is still
    unmatched is that twin and dropped; else it is a new event, first seen in this window.  So two identical events
    (two fillers' same "saved ..." text) stay two, and a server that relays nothing (or logs one side only) is counted
    as it is."""

    def __init__(self, keep: int = 4000):
        import collections
        self.pending: "collections.OrderedDict[tuple[str, str], int]" = collections.OrderedDict()
        self.keep = keep

    @staticmethod
    def key(ln: str) -> tuple[str, str]:
        i = ln.find("hy4 slots:")
        if i >= 0:
            return "stderr", ln[i + len("hy4 slots:"):].strip()
        i = ln.find("sessions:")
        return "relay", ln[i + len("sessions:"):].strip()

    def feed(self, lines: list[str]) -> list[str]:
        out = []
        for ln in lines:
            if not is_slot_line(ln):
                continue
            src, text = self.key(ln)
            twin = ("relay" if src == "stderr" else "stderr", text)
            if self.pending.get(twin, 0) > 0:
                self.pending[twin] -= 1
                if not self.pending[twin]:
                    del self.pending[twin]
                continue
            self.pending[(src, text)] = self.pending.get((src, text), 0) + 1
            self.pending.move_to_end((src, text))
            while len(self.pending) > self.keep:
                self.pending.popitem(last=False)
            out.append(ln)
        return out


# ---- the .meta's bytes (HY4META2, as hy4_slots.hpp's interface comment lays them out) ------------------------------
META2 = b"HY4META2"


def meta_parse(b: bytes) -> dict | None:
    """A .meta's fields (little-endian; the identity, bin_bytes, bin_checksum, n, lo, ntok tokens, npics pictures, the
    crc over every byte before it, the closing magic, nothing after); None when any of it is not so."""
    import struct
    import zlib
    try:
        if b[:8] != META2 or b[-8:] != META2:
            return None
        (L,) = struct.unpack_from("<Q", b, 8)
        at = 16 + L
        ident = b[16:at].decode("utf-8", "replace")
        bin_bytes, bin_sum, n, lo, ntok = struct.unpack_from("<QQQQQ", b, at)
        at += 40
        toks = list(struct.unpack_from(f"<{ntok}i", b, at))
        at += 4 * ntok
        (npics,) = struct.unpack_from("<Q", b, at)
        at += 8
        pics = [struct.unpack_from("<qQ", b, at + 16 * i) for i in range(npics)]
        at += 16 * npics
        (crc,) = struct.unpack_from("<Q", b, at)
        if at + 16 != len(b) or crc != zlib.crc32(b[:at]) or ntok != n:
            return None
        return {"identity": ident, "L": L, "bin_bytes": bin_bytes, "bin_sum": bin_sum, "n": n, "lo": lo,
                "tokens": toks, "pics": pics}
    except (struct.error, IndexError, ValueError):
        return None


def meta_with_identity(b: bytes, ident: str) -> bytes:
    """The same .meta with another identity (L and its bytes) and its crc recomputed: well formed, another engine's."""
    import struct
    import zlib
    (L,) = struct.unpack_from("<Q", b, 8)
    raw = ident.encode()
    out = bytearray(b[:8] + struct.pack("<Q", len(raw)) + raw + b[16 + L:])
    struct.pack_into("<Q", out, len(out) - 16, zlib.crc32(bytes(out[:-16])))
    return bytes(out)


class Log:
    """The server's log, local or over ssh (read-only: stat and tail), read continuously: since() is everything
    appended after the last read - the idle time between two requests included."""

    def __init__(self, path: str, ssh: str):
        self.path, self.ssh = path, ssh
        self.off = 0

    def size(self) -> int:
        if not self.ssh:
            return os.path.getsize(self.path)
        r = subprocess.run(["ssh", self.ssh, "stat", "-c", "%s", self.path], capture_output=True, text=True, timeout=30)
        if r.returncode:
            raise OSError(f"ssh {self.ssh} stat {self.path}: {r.stderr.strip()}")
        return int(r.stdout.strip())

    def _from(self, off: int) -> str:
        if not self.ssh:
            with open(self.path, "rb") as f:
                f.seek(off)
                return f.read().decode("utf-8", "replace")
        r = subprocess.run(["ssh", self.ssh, "tail", "-c", f"+{off + 1}", self.path], capture_output=True, timeout=60)
        if r.returncode:
            raise OSError(f"ssh {self.ssh} tail {self.path}: {r.stderr.decode(errors='replace').strip()}")
        return r.stdout.decode("utf-8", "replace")

    def mark(self, off: int | None = None) -> None:
        self.off = self.size() if off is None else off

    def since(self, settle: float = 0.5) -> list[str]:
        time.sleep(settle)   # (the server prints the engine's NOTE as it reads it; give the write a moment)
        size = self.size()
        if size < self.off:   # (a new log: a restart that truncated it)
            self.off = 0
        text = self._from(self.off)
        self.off = size
        return text.splitlines()


FOLDER_PY = r'''
import base64, json, os, sys
op, d, arg = sys.argv[1], sys.argv[2], json.loads(base64.b64decode(sys.argv[3]))
if op == "list":
    out = []
    for n in sorted(os.listdir(d)):
        p = os.path.join(d, n)
        if os.path.isfile(p):
            st = os.stat(p)
            out.append([n, st.st_size, st.st_mtime])
    print(json.dumps(out))
elif op == "flip":
    p = os.path.join(d, arg["name"])
    with open(p, "r+b") as f:
        f.seek(arg["at"])
        b = f.read(1)
        f.seek(arg["at"])
        f.write(bytes([b[0] ^ 0x5A]))
        f.flush()
        os.fsync(f.fileno())
    print("[]")
elif op == "read":
    with open(os.path.join(d, arg["name"]), "rb") as f:
        print(json.dumps([base64.b64encode(f.read()).decode()]))
elif op == "write":
    with open(os.path.join(d, arg["name"]), "wb") as f:
        f.write(base64.b64decode(arg["data"]))
    print("[]")
'''


class Folder:
    """The server's slot folder (--folder), local or over ssh: list, flip a byte, write a file - for the foreign-file
    check only, and only when the operator asks for it."""

    def __init__(self, path: str, ssh: str):
        self.path, self.ssh = path, ssh

    def _op(self, op: str, arg: dict | None = None):
        a = base64.b64encode(json.dumps(arg or {}).encode()).decode()
        cmd = ["python3", "-c", FOLDER_PY, op, self.path, a]
        if self.ssh:
            import shlex
            cmd = ["ssh", self.ssh, " ".join(shlex.quote(c) for c in cmd)]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if r.returncode:
            raise OSError(f"folder {op}: {r.stderr.strip()[-300:]}")
        return json.loads(r.stdout)

    def list(self) -> dict[str, tuple[int, float]]:
        return {n: (s, t) for n, s, t in self._op("list")}

    def flip(self, name: str, at: int) -> None:
        self._op("flip", {"name": name, "at": at})

    def write(self, name: str, data: bytes) -> None:
        self._op("write", {"name": name, "data": base64.b64encode(data).decode()})

    def read(self, name: str) -> bytes:
        return base64.b64decode(self._op("read", {"name": name})[0])


class Server:
    def __init__(self, host: str, port: int, key: str, timeout: float):
        self.base = f"http://{host}:{port}"
        self.key, self.timeout = key, timeout
        self.model = None

    def _req(self, path: str, body: dict | None = None, timeout: float | None = None) -> dict:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.base + path, data=data, method="POST" if body is not None else "GET")
        req.add_header("Content-Type", "application/json")
        if self.key:
            req.add_header("Authorization", f"Bearer {self.key}")
        with urllib.request.urlopen(req, timeout=timeout or self.timeout) as r:
            return json.loads(r.read().decode())

    def health(self, timeout: float | None = None) -> dict:
        h = self._req("/health", timeout=timeout)
        try:
            ms = self._req("/v1/models", timeout=timeout).get("data") or []
            self.model = ms[0]["id"] if ms else None
        except (urllib.error.URLError, KeyError, ValueError, OSError):
            pass
        return h

    def up(self) -> bool:
        try:
            return (self.health(timeout=5).get("status") or "ok") == "ok"
        except (urllib.error.URLError, OSError, ValueError):
            return False

    def tokenize_text(self, text: str) -> list[int]:
        return self._req("/tokenize", {"content": text, "parse_special": False})["tokens"]

    def detokenize(self, ids: list[int]) -> str:
        return self._req("/detokenize", {"tokens": ids})["content"]

    def body(self, messages: list[dict], max_new: int) -> dict:
        b = {"messages": messages, "max_tokens": max_new, "temperature": 0, "stream": False,
             "chat_template_kwargs": {"enable_thinking": False}}
        if self.model:
            b["model"] = self.model
        return b

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        return self._req("/tokenize", self.body(messages, 1))["tokens"]

    def chat(self, messages: list[dict], max_new: int) -> dict:
        t0 = time.monotonic()
        r = self._req("/v1/chat/completions", self.body(messages, max_new))
        ch = r["choices"][0]
        msg = ch.get("message") or {}
        usage = r.get("usage") or {}
        tim = r.get("timings") or {}
        reused = tim.get("cache_n")
        if reused is None:
            reused = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        return {"text": msg.get("content") or "", "reasoning": msg.get("reasoning_content") or "",
                "finish": ch.get("finish_reason"), "completion": usage.get("completion_tokens"),
                "prompt": usage.get("prompt_tokens"), "reused": reused, "prompt_ms": tim.get("prompt_ms"),
                "s": time.monotonic() - t0}


def lcp(a: list[int], b: list[int]) -> int:
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


class Run:
    def __init__(self, srv: Server, log: Log, a, nonce: str):
        self.srv, self.log, self.a = srv, log, a
        self.rows: list[tuple[str, str, bool | None, str]] = []   # (scenario, check, ok / None = info, detail)
        self.nonce = nonce
        self.steps: dict[str, dict] = {}   # every request's answer, by name (--record / --compare)
        self.evidence = not a.reference   # (a KEEP=0 reference run: no slot evidence expected)
        self.ev = Events()                # (every slot line read goes through it: events, not lines)
        with open(CORPUS, encoding="utf-8", newline="") as f:
            self.corpus = f.read()

    def check(self, scen: str, what: str, ok: bool | None, detail: str = "") -> None:
        self.rows.append((scen, what, ok, detail))
        tag = "INFO" if ok is None else ("ok" if ok else "FAIL")
        print(f"  {tag:5} {scen:8} {what}" + (f"  [{detail}]" if detail else ""), flush=True)

    def slot_check(self, scen: str, what: str, ok: bool | None, detail: str = "") -> None:
        """A check of the slots' own evidence: INFO in a --reference run (the server keeps none)."""
        self.check(scen, what, ok if self.evidence else None, detail)

    def text(self, salt: str, n: int) -> tuple[str, list[int]]:
        """n tokens of real text (the corpus's passages, numbered under the nonce and `salt`), and their ids."""
        parts, i = [], 0
        ids: list[int] = []
        while len(ids) < n:
            chunk = "".join(f"{salt} ({self.nonce}) {i + j}.\n{self.corpus}\n" for j in range(8))
            parts.append(chunk)
            i += 8
            ids = self.srv.tokenize_text("".join(parts))
        ids = ids[:n]
        return self.srv.detokenize(ids), ids

    def send(self, scen: str, name: str, messages: list[dict], max_new: int | None = None,
             allow_bad: bool = False) -> dict:
        """A request: its answer, its prompt's ids (/tokenize), the log lines since the last read (the idle time
        before it included) - and the VERIFY rule applied to them: every take verified byte for byte, no mismatch,
        no failed read-back, no refusal (unless allow_bad: a scenario that expects a refusal)."""
        ids = self.srv.prompt_ids(messages)
        pre = self.ev.feed(self.log.since(settle=0.0))   # (the idle time before the request: its own events)
        r = self.srv.chat(messages, self.a.max_new if max_new is None else max_new)
        r["ids"] = ids
        r["notes"] = pre + self.ev.feed(self.log.since())   # (events: a stderr line and its relay counted once)
        r["idle_notes"] = pre
        kinds = [(classify(ln), ln) for ln in r["notes"]]
        r["took"] = [parse_took(ln) for k, ln in kinds if k == "took"]
        r["saved"] = [ln for k, ln in kinds if k == "saved"]
        r["verified"] = [ln for k, ln in kinds if k == "verify_ok"]
        r["bad"] = [ln for k, ln in kinds if k == "bad"]
        self.steps[f"{scen}/{name}"] = {k: r[k] for k in ("text", "completion", "reused", "prompt", "finish")}
        self.steps[f"{scen}/{name}"]["n_prompt"] = len(ids)
        print(f"        {name}: prompt {len(ids)} tok (server {r['prompt']}), reused {r['reused']}, "
              f"{r['completion']} out, {r['s']:.1f} s; slot events: {len(r['notes'])}"
              + "".join(f"\n          | {ln.strip()[:220]}" for ln in r["notes"]), flush=True)
        if r["reasoning"]:
            print(f"        {name}: WARNING the answer has reasoning_content although thinking is off", flush=True)
        if not self.evidence:   # (a reference run's own check: the answer is there to compare against)
            self.check(scen, f"{name}: answered and recorded (reference)", bool(r["completion"]) and
                       r["finish"] is not None, f"{r['completion']} tokens, {r['finish']}")
        if self.evidence:
            if r["took"] and not self.a.no_verify:
                self.check(scen, f"{name}: every take verified byte for byte (MAYA_HY4_SLOT_VERIFY)",
                           len(r["verified"]) >= len(r["took"]) and not any("mismatch" in b.lower() for b in r["bad"]),
                           f"{len(r['took'])} take(s), {len(r['verified'])} verify ok event(s)")
            if r["bad"] and not allow_bad:
                self.check(scen, f"{name}: no MISMATCH / failed read-back / refusal / failure line", False,
                           r["bad"][0].strip()[:200])
        return r

    def same_answer(self, scen: str, what: str, x: dict, y: dict, xn: str = "test", yn: str = "control") -> bool:
        ids_x = self.srv.tokenize_text(x["text"])
        ids_y = self.srv.tokenize_text(y["text"])
        ok = x["text"] == y["text"] and x["completion"] == y["completion"] and ids_x == ids_y
        d = lcp(ids_x, ids_y)
        detail = f"{xn} {x['completion']} vs {yn} {y['completion']} tokens" + \
            ("" if ok else f"; first difference at token {d}: {xn} {ids_x[d:d + 4]} vs {yn} {ids_y[d:d + 4]}")
        if self.a.hot == "on":
            self.check(scen, what + " (HY4_HOT on: not judged)", None, ("same; " if ok else "DIFFERENT; ") + detail)
            return ok
        self.check(scen, what, ok, detail + ("" if ok or self.a.hot == "off" else
                                             " - if the server runs the adaptive hot set this may be placement"))
        return ok

    def a_saved(self, scen: str, what: str, *rs: dict) -> None:
        """A set aside: a park on the request, or an idle save / write since A's answer (MAYA_HY4_SLOT_IDLE_S)."""
        saves = [(r["idle_notes"], ln) for r in rs for ln in r["saved"]]
        idle = [ln for notes, ln in saves if ln in notes]
        self.slot_check(scen, what, bool(saves), (("idle-time: " if idle and len(idle) == len(saves) else "")
                                                   + saves[0][1].strip()[:140]) if saves else "no saved / written line")

    def reset(self, scen: str) -> dict:
        """The same short conversation (its own text, not A's start; under --min, so it is never set aside) sent just
        before control A and before test A: both A's are then read from the same shared start, so in the same chunks -
        the rounding of A's rows, and every answer built on them, can only differ by the slots."""
        t, _ = self.text(f"A reset before A ({scen}).", 64)
        return self.send(scen, "reset", [{"role": "user", "content": t}], max_new=1)

    def same_start(self, scen: str, ca: dict, ta: dict) -> bool:
        """Control A and test A read from the same point (their reused equal) - else their chunks differ and so may
        their answers: the identity checks are not judged (INFO), it is not a slot failure."""
        same = ca["reused"] == ta["reused"]
        self.check(scen, "control A and test A read from the same shared start (identity judged only then)",
                   True if same else None, f"control A reused {ca['reused']}, test A reused {ta['reused']}" +
                   ("" if same else " - their chunks differ: the identity below is INFO, not a slot failure"))
        return same

    def flush(self, scen: str) -> None:
        """--cap + 2 short conversations of their own: the control's conversations set aside and dropped (LRU)."""
        n = self.a.cap + 2
        print(f"  ..    {scen:8} {n} filler conversations (each {self.a.min + 200} tokens) to push the control's "
              f"conversations out of RAM", flush=True)
        for i in range(n):
            t, _ = self.text(f"Filler {scen} {i}.", self.a.min + 200)
            self.send(scen, f"filler {i}", [{"role": "user", "content": t}], max_new=1)

    # ---- A -> B -> A ---------------------------------------------------------------------------------------------
    def full(self) -> None:
        s = "full"
        print(f"{s}: A ({self.a.len_a}), B ({self.a.len_b}), A + answer + follow-up ({self.a.len_turn})", flush=True)
        a_text, _ = self.text("Conversation A.", self.a.len_a)
        b_text, _ = self.text("Conversation B.", self.a.len_b)
        fu_text, _ = self.text("The next turn:", self.a.len_turn)
        A = [{"role": "user", "content": a_text}]
        B = [{"role": "user", "content": b_text}]

        def turn(ans: dict) -> list[dict]:
            return A + [{"role": "assistant", "content": ans["text"]}, {"role": "user", "content": fu_text}]

        ca = ct = None
        if not self.a.no_control:
            print("  ..    control: the reset, A, then its next turn (the model's own state, no slot)", flush=True)
            self.reset(s)
            ca = self.send(s, "control A", A)
            ct = self.send(s, "control A-turn", turn(ca))
            self.check(s, "control: the next turn continues A in place (reused >= A's prompt)",
                       (ct["reused"] or 0) >= len(ca["ids"]) - 1, f"reused {ct['reused']} of A's {len(ca['ids'])}")
            self.flush(s)
        print("  ..    test: the reset, A, B (A set aside), A's next turn (A taken back)", flush=True)
        if ca is not None:
            self.reset(s)
        ta = self.send(s, "test A", A)
        tb = self.send(s, "test B", B)
        tt = self.send(s, "test A-turn", turn(ta))
        self.a_saved(s, "A set aside (a park on B, or an idle save after A)", tb)
        self.slot_check(s, "A's next turn took A back (a 'took back' NOTE)", len(tt["took"]) == 1,
                        tt["took"][0]["line"][:140] if tt["took"] else "none")
        if tt["took"]:
            tk = tt["took"][0]
            self.slot_check(s, "reused == the taken conversation's n (it continues the whole state)",
                            tt["reused"] == tk.get("n") == tk.get("u"), f"reused {tt['reused']}, NOTE u {tk.get('u')} "
                            f"of n {tk.get('n')}")
            want_n = len(ta["ids"]) + (ta["completion"] or 0)
            self.slot_check(s, "the NOTE's n is A's prompt and answer", tk.get("n") in (want_n - 1, want_n),
                            f"n {tk.get('n')}, A prompt {len(ta['ids'])} + answer {ta['completion']}")
            if "k" in tk:
                self.slot_check(s, "the NOTE's new tokens are the prompt less the reuse",
                                tk["k"] == len(tt["ids"]) - (tt["reused"] or 0),
                                f"k {tk['k']}, prompt {len(tt['ids'])} - reused {tt['reused']}")
        if ca is not None:
            self.slot_check(s, "the control was not taken back for test A (it read A fresh)", not ta["took"],
                            ta["took"][0]["line"][:120] if ta["took"] else "")
            if not self.same_start(s, ca, ta):
                for what, x, y in (("test A's answer is the control's", ta, ca),
                                   ("GREEDY IDENTITY: the turn's answer after the take is the control's", tt, ct)):
                    self.check(s, what + " (not judged: another shared start)", None,
                               f"test {x['completion']} vs control {y['completion']} tokens, "
                               f"{'same' if x['text'] == y['text'] else 'DIFFERENT'} text")
                same_a = None
            else:
                same_a = self.same_answer(s, "test A's answer is the control's (a fresh read both times)", ta, ca)
            if same_a is None:
                pass
            elif same_a:
                self.slot_check(s, "the turn's reused tokens are the control's (in place == taken back)",
                                tt["reused"] == ct["reused"], f"{tt['reused']} vs {ct['reused']}")
                self.same_answer(s, "GREEDY IDENTITY: the turn's answer after the take is the control's", tt, ct)
            else:
                self.check(s, "GREEDY IDENTITY: not comparable - A's answers differ, so the turns' prompts do",
                           None if self.a.hot == "on" else False)
        self.check(s, "prompt ids agree with the server's count", tt["prompt"] == len(tt["ids"]),
                   f"{tt['prompt']} vs {len(tt['ids'])}")
        self.check(s, "speed: the turn's prompt ms (after the take) vs the control's in place", None,
                   f"{tt['prompt_ms']} ms vs {ct['prompt_ms'] if ct else '-'} ms; test A {ta['prompt_ms']} ms")

    # ---- A -> B -> A' (part way) -----------------------------------------------------------------------------------
    def partial(self) -> None:
        s = "partial"
        back = self.a.partial_back
        print(f"partial: A ({self.a.len_a}), B, A' (A's last {back} prompt tokens replaced)", flush=True)
        a_text, a_ids = self.text("Conversation A2.", self.a.len_a)
        b_text, _ = self.text("Conversation B2.", self.a.len_b)
        other, _ = self.text("Another ending.", back)
        a2_text = self.srv.detokenize(a_ids[:len(a_ids) - back]) + other
        A = [{"role": "user", "content": a_text}]
        A2 = [{"role": "user", "content": a2_text}]
        B = [{"role": "user", "content": b_text}]
        ca = c2 = None
        if not self.a.no_control:
            print("  ..    control: the reset, A, then A' (an in-place rewind)", flush=True)
            self.reset(s)
            ca = self.send(s, "control A", A)
            c2 = self.send(s, "control A'", A2)
            u_want = min(lcp(ca["ids"], c2["ids"]), len(c2["ids"]) - 1)
            self.check(s, "control: A' reuses A's state in place at the shared start", c2["reused"] == u_want,
                       f"reused {c2['reused']}, shared start {u_want}")
            self.flush(s)
        print("  ..    test: the reset, A, B (A set aside), A' (A taken back part way)", flush=True)
        if ca is not None:
            self.reset(s)
        ta = self.send(s, "test A", A)
        tb = self.send(s, "test B", B)
        t2 = self.send(s, "test A'", A2)
        self.a_saved(s, "A set aside (a park on B, or an idle save after A)", tb)
        self.slot_check(s, "A' took A back (a 'took back' NOTE)", len(t2["took"]) == 1,
                        t2["took"][0]["line"][:140] if t2["took"] else "none")
        u_test = min(lcp(ta["ids"], t2["ids"]), len(t2["ids"]) - 1)
        if t2["took"]:
            tk = t2["took"][0]
            self.slot_check(s, "the take is part way: at token u of n, u < n", tk.get("u") is not None and
                            tk.get("n") is not None and tk["u"] < tk["n"], f"u {tk.get('u')} of n {tk.get('n')}")
            self.slot_check(s, "u is the prompts' shared start (/tokenize), and reused == u",
                            tk.get("u") == u_test == t2["reused"], f"NOTE u {tk.get('u')}, shared start {u_test}, "
                            f"reused {t2['reused']}")
        self.check(s, "A' leaves A about --partial-back tokens before its end", None,
                   f"A's state {len(ta['ids']) + (ta['completion'] or 0)}, shared start {u_test}")
        if ca is not None and not self.same_start(s, ca, ta):
            for what, x, y in (("test A's answer is the control's", ta, ca),
                               ("GREEDY IDENTITY: A''s answer after the part-way take is the in-place rewind's", t2, c2)):
                self.check(s, what + " (not judged: another shared start)", None,
                           f"test {x['completion']} vs control {y['completion']} tokens, "
                           f"{'same' if x['text'] == y['text'] else 'DIFFERENT'} text")
        elif ca is not None:
            if self.same_answer(s, "test A's answer is the control's", ta, ca):
                self.same_answer(s, "GREEDY IDENTITY: A''s answer after the part-way take is the in-place rewind's",
                                 t2, c2)
            else:
                self.check(s, "GREEDY IDENTITY: not comparable - A's answers differ",
                           None if self.a.hot == "on" else False)

    # ---- the rings' boundary -------------------------------------------------------------------------------------
    def ring(self) -> None:
        s = "ring"
        rows = self.a.ring_rows if not self.a.slot_ring else min(self.a.ring_rows, self.a.slot_ring)
        m, back = self.a.ring_margin, self.a.ring_back
        n_long = rows + 2 * m + back + 2000
        print(f"ring: A ({n_long} tokens, past the rings' {rows}), B, A-out (left before the rings hold), "
              f"A-in (left inside them)", flush=True)
        a_text, a_ids = self.text("Conversation A3.", n_long)
        b_text, _ = self.text("Conversation B3.", self.a.len_b)
        A = [{"role": "user", "content": a_text}]
        B = [{"role": "user", "content": b_text}]
        ta = self.send(s, "A (long)", A)
        tb = self.send(s, "B", B)
        self.a_saved(s, "the long A set aside (a park on B, or an idle save after A)", tb)
        n = len(ta["ids"]) + (ta["completion"] or 0)
        p0 = len(ta["ids"]) - len(a_ids)   # (about: the template's tokens before the text)

        def variant(d_target: int, salt: str) -> list[dict]:
            k = max(1, d_target - max(0, p0 - 8))
            tail, _ = self.text(salt, 1500)
            return [{"role": "user", "content": self.srv.detokenize(a_ids[:k]) + tail}]

        out = variant(n - rows - m, "An early turn-off.")
        out_ids = self.srv.prompt_ids(out)
        d_out = min(lcp(ta["ids"], out_ids), len(out_ids) - 1)
        self.check(s, "A-out's shared start is before the rings hold (precondition)", d_out < n - rows,
                   f"shared start {d_out}, A's state {n}, rows {rows}")
        to = self.send(s, "A-out", out)
        self.slot_check(s, "RING REFUSAL: A-out does not take the long A back", not to["took"],
                        to["took"][0]["line"][:140] if to["took"] else "no took-back line")
        from_b = min(lcp(tb["ids"], to["ids"]), len(to["ids"]) - 1)
        self.slot_check(s, "RING REFUSAL: A-out reads from the start (reused only what B's state shares)",
                        (to["reused"] or 0) <= from_b, f"reused {to['reused']}, shared with B {from_b}")
        inn = variant(n - rows + back + m, "A late turn-off.")
        in_ids = self.srv.prompt_ids(inn)
        d_in = min(lcp(ta["ids"], in_ids), len(in_ids) - 1)
        self.check(s, "A-in's shared start is inside the rings (precondition)", d_in - back >= n - rows,
                   f"shared start {d_in}, A's state {n}, rows {rows}, back {back}")
        ti = self.send(s, "A-in", inn)
        self.slot_check(s, "A-in takes the long A back part way", len(ti["took"]) == 1,
                        ti["took"][0]["line"][:140] if ti["took"] else "none")
        if ti["took"]:
            tk = ti["took"][0]
            self.slot_check(s, "A-in: NOTE u == the shared start == reused", tk.get("u") == d_in == ti["reused"],
                            f"NOTE u {tk.get('u')} of n {tk.get('n')}, shared start {d_in}, reused {ti['reused']}")

    # ---- v2: across a restart ------------------------------------------------------------------------------------
    def conv_a(self, scen: str) -> tuple[list[dict], str]:
        a_text, _ = self.text(f"Conversation R-{scen}.", self.a.len_a)
        fu_text, _ = self.text(f"The next turn R-{scen}:", self.a.len_turn)
        return [{"role": "user", "content": a_text}], fu_text

    def wait_write(self, scen: str) -> list[str]:
        """After A: wait past MAYA_HY4_SLOT_IDLE_S for the idle write's NOTE (the folder's copy of A)."""
        wait = self.a.idle_s + self.a.write_wait
        print(f"  ..    {scen:8} waiting up to {wait:.0f} s for the idle write (MAYA_HY4_SLOT_IDLE_S {self.a.idle_s})",
              flush=True)
        seen: list[str] = []
        end = time.monotonic() + wait
        while time.monotonic() < end:
            seen += self.ev.feed(self.log.since(settle=1.0))
            if any(classify(ln) == "saved" and WROTE_FOLDER.search(ln) for ln in seen) and \
                    time.monotonic() > end - self.a.write_wait + 2:
                break
        for ln in seen:
            print(f"          | {ln.strip()[:220]}", flush=True)
        return seen

    def written(self, scen: str, what: str, seen: list[str]) -> None:
        wrote = [ln for ln in seen if classify(ln) == "saved" and WROTE_FOLDER.search(ln)]
        self.slot_check(scen, f"{what} written to the folder ('wrote a N-token conversation to the folder', not the "
                        f"RAM save alone)", bool(wrote),
                        wrote[0].strip()[:160] if wrote else "none")
        bad = [ln for ln in seen if classify(ln) == "bad"]
        self.slot_check(scen, f"no failure line while {what} was written", not bad, bad[0].strip()[:160] if bad else "")

    # the foreign scenario's two conversations: A-bin (its .bin damaged) and A-id (its .meta another engine's)
    FOREIGN = ("A-bin", "A-id")

    def phase1(self, scen: str, state: dict) -> None:
        names = self.FOREIGN if scen == "foreign" else ("A",)
        print(f"{scen} (phase 1): {', '.join(names)} ({self.a.len_a} tokens each), each idle until it is written to "
              f"the folder", flush=True)
        st: dict = {"t0": time.time()}
        for nm in names:
            A, _ = self.conv_a(scen if nm == "A" else f"{scen}-{nm}")
            ta = self.send(scen, nm, A)
            self.written(scen, nm, self.wait_write(scen))
            st[nm] = {"answer": ta["text"], "completion": ta["completion"], "n_prompt": len(ta["ids"]),
                      "ids": ta["ids"]}
        if scen == "foreign":
            st.update(self.damage(st))
        for nm in names:
            st[nm].pop("ids", None)
        state[scen] = st

    def damage(self, st: dict) -> dict:
        """The foreign-file set-up (--folder), done to the files the engine wrote:
          A-bin: its .bin damaged by one byte (its size kept: the start's check passes it, the take's checksum must
                 refuse it and remove the pair, the turn read from token 0);
          A-id:  its .meta rewritten WELL FORMED with another identity (one character changed, the CRC-32
                 recomputed, A's real .bin beside it): the start must refuse it as another engine's and remove the pair;
          a malformed pair (bytes that are no .meta), and a file that is not a slot's (left alone)."""
        out = {"foreign": f"slot-foreign-{self.nonce}", "keep": f"keep-{self.nonce}.txt"}
        if not self.evidence:   # (a KEEP=0 / --reference run: no slot files exist - nothing to damage or check)
            self.check("foreign", "reference run: the folder's files not touched (no slots kept)", None)
            return {}
        if not self.a.folder:
            self.check("foreign", "the foreign-file check needs --folder (the server's slot folder)", False)
            return out
        fo = Folder(self.a.folder, self.a.folder_ssh)
        files = fo.list()
        metas: dict[str, dict] = {}
        for n in files:
            if n.startswith("slot-") and n.endswith(".meta") and n[:-5] + ".bin" in files:
                m = meta_parse(fo.read(n))
                if m:
                    metas[n[:-5]] = m
        for nm in self.FOREIGN:
            ids = st[nm]["ids"]
            mine = sorted((m["n"], stem) for stem, m in metas.items() if m["tokens"][:len(ids)] == ids)
            self.check("foreign", f"{nm}'s .bin and a well-formed HY4META2 .meta are in the folder", bool(mine),
                       mine[-1][1] if mine else f"{len(metas)} well-formed .meta, none of {nm}'s tokens")
            if not mine:
                continue
            stem = mine[-1][1]
            out[nm + ".stem"] = stem
            if nm == "A-bin":
                size = files[stem + ".bin"][0]
                fo.flip(stem + ".bin", size // 2)
                print(f"  ..    foreign  {nm}: flipped a byte at {size // 2} of {stem}.bin ({size} bytes)", flush=True)
            else:
                raw = fo.read(stem + ".meta")
                ident = metas[stem]["identity"]
                i = len(ident) // 2
                ident2 = ident[:i] + ("x" if ident[i] != "x" else "y") + ident[i + 1:]
                new = meta_with_identity(raw, ident2)
                m2 = meta_parse(new)
                self.check("foreign", f"{nm}: the rewritten .meta is well formed (this script's own reading of the "
                           f"layout), only its identity differs", bool(m2) and m2["identity"] == ident2 and
                           {k: v for k, v in m2.items() if k not in ("identity", "L")} ==
                           {k: v for k, v in metas[stem].items() if k not in ("identity", "L")},
                           f"identity char {i}: {ident[i]!r} -> {ident2[i]!r}")
                fo.write(stem + ".meta", new)
                print(f"  ..    foreign  {nm}: {stem}.meta rewritten with another identity (char {i} of "
                      f"{len(ident)}), its CRC recomputed, its .bin left as written", flush=True)
        fo.write(out["foreign"] + ".bin", os.urandom(4096))
        fo.write(out["foreign"] + ".meta", b"HY4META2" + os.urandom(200))
        fo.write(out["keep"], b"not a slot's file: the engine must leave it\n")
        return out

    def turn_of(self, scen: str, nm: str, st: dict) -> list[dict]:
        A, fu = self.conv_a(scen if nm == "A" else f"{scen}-{nm}")
        return A + [{"role": "assistant", "content": st[nm]["answer"]}, {"role": "user", "content": fu}]

    def phase2(self, scen: str, state: dict) -> None:
        st = state.get(scen)
        if not st:
            self.check(scen, "phase 2 needs phase 1's state (--state)", False)
            return
        if scen == "restart":
            self.phase2_restart(st["A"] if "A" in st else st)
        else:
            self.phase2_foreign(st)

    def phase2_restart(self, a: dict) -> None:
        scen = "restart"
        print(f"{scen} (phase 2): A's next turn after the restart", flush=True)
        t = self.send(scen, "A-turn", self.turn_of(scen, "A", {"A": a}))
        want_n = a["n_prompt"] + (a["completion"] or 0)
        self.slot_check(scen, "A's next turn took A back from the folder", len(t["took"]) == 1 and
                        t["took"][0]["src"] == "folder", t["took"][0]["line"][:160] if t["took"] else "none")
        if t["took"]:
            tk = t["took"][0]
            self.slot_check(scen, "reused == n (the whole state, from the folder)",
                            t["reused"] == tk.get("n") == tk.get("u"),
                            f"reused {t['reused']}, NOTE u {tk.get('u')} of n {tk.get('n')}")
            self.slot_check(scen, "the NOTE's n is A's prompt and answer", tk.get("n") in (want_n - 1, want_n),
                            f"n {tk.get('n')}, A {a['n_prompt']} + {a['completion']}")
        if self.a.no_verify:
            self.check(scen, "the take from the folder verified byte for byte", None, "--no-verify")
        else:
            self.slot_check(scen, "the take from the folder verified byte for byte",
                            bool(t["took"]) and len(t["verified"]) >= len(t["took"]),
                            t["verified"][0].strip()[:160] if t["verified"] else "no verify line")
        self.check(scen, "speed: the turn's prompt ms", None, f"{t['prompt_ms']} ms, reused {t['reused']}")

    def phase2_foreign(self, st: dict) -> None:
        scen = "foreign"
        print(f"{scen} (phase 2): the folder after the restart, then A-id's and A-bin's next turns", flush=True)
        if not self.evidence:
            self.check(scen, "reference run: the folder's checks skipped (no slots kept)", None)
        elif self.a.folder:
            files = Folder(self.a.folder, self.a.folder_ssh).list()
            self.slot_check(scen, "the malformed slot-*.bin / .meta pair removed at the start",
                            not any(n.startswith(st.get("foreign", "?")) for n in files),
                            ", ".join(n for n in files if n.startswith(st.get("foreign", "?"))) or "gone")
            if st.get("A-id.stem"):
                left = [n for n in files if n.startswith(st["A-id.stem"] + ".")]
                self.slot_check(scen, "A-id's WELL-FORMED .meta of another identity refused at the start: the pair "
                                "(.meta and its real .bin) removed", not left, ", ".join(left) or "both gone")
            self.check(scen, "a file not a slot's left alone", st.get("keep") in files, st.get("keep", ""))
            if st.get("A-bin.stem"):
                self.check(scen, "A-bin's damaged .bin still there before its take (the start reads no .bin, its "
                           "size is right)", st["A-bin.stem"] + ".bin" in files, st["A-bin.stem"])
        # A-id: nothing of it may be taken back - read from token 0
        n_id = st["A-id"]["n_prompt"]
        ti = self.send(scen, "A-id-turn", self.turn_of(scen, "A-id", st))
        self.slot_check(scen, "A-id's next turn is not taken back (its file was another engine's)",
                        not ti["took"], ti["took"][0]["line"][:160] if ti["took"] else "no take")
        self.slot_check(scen, "A-id's next turn read from token 0 (reused far below A-id's state)",
                        (ti["reused"] or 0) < n_id // 2, f"reused {ti['reused']} of A-id's {n_id}")
        # A-bin: its take refused at the checksum, the pair removed, the prompt read from token 0
        n_bin = st["A-bin"]["n_prompt"]
        tb = self.send(scen, "A-bin-turn", self.turn_of(scen, "A-bin", st), allow_bad=True)
        self.slot_check(scen, "A-bin's damaged .bin is not taken back", not tb["took"],
                        tb["took"][0]["line"][:160] if tb["took"] else "no take")
        self.slot_check(scen, "its refusal said (a refused / failed / checksum line)", bool(tb["bad"]),
                        tb["bad"][0].strip()[:160] if tb["bad"] else "none")
        self.slot_check(scen, "A-bin's turn read from token 0 (reused far below A-bin's state)",
                        (tb["reused"] or 0) < n_bin // 2, f"reused {tb['reused']} of A-bin's {n_bin}")
        if self.evidence and self.a.folder and st.get("A-bin.stem"):
            files = Folder(self.a.folder, self.a.folder_ssh).list()
            self.slot_check(scen, "A-bin's damaged files removed", not any(n.startswith(st["A-bin.stem"] + ".")
                                                                           for n in files), st["A-bin.stem"])

    def compare(self, ref: dict) -> None:
        """--compare: the same steps' answers on a KEEP=0 fresh start (the same nonce, the same sequence)."""
        if ref.get("nonce") != self.nonce:
            self.check("compare", "the reference ran with this nonce", False, f"{ref.get('nonce')} vs {self.nonce}")
            return
        both = [k for k in self.steps if k in ref.get("steps", {})]
        self.check("compare", "steps in common with the reference", bool(both), f"{len(both)}")
        for k in both:
            x, y = self.steps[k], ref["steps"][k]
            ok = x["text"] == y["text"] and x["completion"] == y["completion"]
            what = f"FRESH-START IDENTITY: {k} is the KEEP=0 start's answer"
            if self.a.hot == "on" and not ok:
                self.check("compare", what + " (HY4_HOT on)", False, "DIFFERENT - two fresh starts with the same "
                           "sequence: the slots or the engine's determinism")
            else:
                self.check("compare", what, ok, f"{x['completion']} vs {y['completion']} tokens")

    def table(self) -> bool:
        print("\nRESULT", flush=True)
        w = max(len(r[1]) for r in self.rows) if self.rows else 10
        print(f"  {'scenario':8}  {'check':{w}}  result", flush=True)
        for scen, what, ok, _ in self.rows:
            print(f"  {scen:8}  {what:{w}}  {'INFO' if ok is None else ('PASS' if ok else 'FAIL')}", flush=True)
        bad = [r for r in self.rows if r[2] is False]
        judged = sum(r[2] is not None for r in self.rows)
        print(("PASS" if not bad and judged else f"FAIL ({len(bad)} of {judged} checks)") + f" - nonce {self.nonce}",
              flush=True)
        return not bad and judged > 0


def self_test() -> int:
    """--self-test: the log rules offline (no server) - a take with a `verify: MISMATCH` line FAILS although the text
    matches, a take without a verify line FAILS, a take verified byte for byte passes, an idle save counts as A set
    aside, the took-back line's numbers and source."""
    class FakeSrv:
        model = None

        def prompt_ids(self, m):
            return list(range(100))

        def chat(self, m, n):
            return {"text": "same", "reasoning": "", "finish": "stop", "completion": 3, "prompt": 100,
                    "reused": 99, "prompt_ms": 1.0, "s": 0.0}

    class FakeLog:
        def __init__(self, pre, post):
            self.q = [pre, post]

        def since(self, settle=0.5):
            return self.q.pop(0) if self.q else []

    A = argparse.Namespace(max_new=4, no_verify=False, reference=False, hot="off")
    took = "[maya] sessions: took back a 99-token conversation from RAM (0.1 s); 1 new tokens to read"
    cases = [
        ("MISMATCH after a take, text the same", [], [took, "[maya] sessions: verify: MISMATCH at byte 12 of 1000"],
         False),
        ("a take without a verify line", [], [took], False),
        ("a failed read-back", [], [took, "[maya] sessions: verify: the read-back failed"], False),
        ("a take verified byte for byte", [], [took, "[maya] sessions: verify: 1000 bytes, byte for byte"], True),
        ("a refused file", [], ["[maya] sessions: refused slot-1.bin: checksum"], False),
    ]
    ok_all = True
    for what, pre, post, want in cases:
        run = Run.__new__(Run)
        run.srv, run.log, run.a, run.rows, run.steps, run.evidence, run.nonce = \
            FakeSrv(), FakeLog(pre, post), A, [], {}, True, "t"
        run.ev = Events()
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()):
            run.send("self", "x", [{"role": "user", "content": "x"}])
        passed = all(r[2] is not False for r in run.rows)
        good = passed == want
        ok_all &= good
        print(f"  {'ok' if good else 'FAIL':5} self-test: {what}: the run {'passes' if passed else 'fails'}", flush=True)
    # an idle save before B counts
    run = Run.__new__(Run)
    run.srv, run.a, run.rows, run.steps, run.evidence, run.nonce = FakeSrv(), A, [], {}, True, "t"
    run.ev = Events()
    run.log = FakeLog(["[maya] sessions: wrote a 8000-token conversation to the folder (1300 MB, 900 ms)"], [])
    import contextlib
    import io
    with contextlib.redirect_stdout(io.StringIO()):
        r = run.send("self", "B", [{"role": "user", "content": "b"}])
        run.a_saved("self", "A set aside", r)
    good = run.rows and run.rows[-1][2] is True and run.rows[-1][3].startswith("idle-time")
    ok_all &= bool(good)
    print(f"  {'ok' if good else 'FAIL':5} self-test: an idle write before B is A's save evidence", flush=True)
    t = parse_took("[maya] sessions: took back a 8,123-token conversation at token 5000 of 8123 from the folder "
                   "(1300 MB, read 400 / checked 90 / loaded 160 ms); 3200 new tokens to read")
    good = t.get("u") == 5000 and t.get("n") == 8123 and t.get("k") == 3200 and t["src"] == "folder"
    ok_all &= good
    print(f"  {'ok' if good else 'FAIL':5} self-test: a took-back line's numbers and source ({t})", flush=True)
    # every log pattern on fixture lines.  The documented wordings (docs/SETTINGS.md): "NOTE sessions: folder ...",
    # "took back ... from the folder (MB, read / checked / loaded ms)", "NOTE sessions: verify: ...",
    # "NOTE sessions: the engine's end: ...", "... is left out at the engine's end: ...".  The rest of each fixture's
    # text (and the not-there / not-written phrases) is this script's guess, kept strict.
    fixtures = [
        ("[maya] sessions: folder /mnt/ssd/slots: 3 conversations (4.1 GB), 1 refused", "folder_list"),
        ("hy4 slots: folder /pack/slots: none", "folder_list"),
        ("[maya] sessions: the folder /mnt/ssd/slots cannot be made: Permission denied", "other"),
        ("[maya] sessions: folder /mnt/ssd/slots cannot be made: Permission denied", "other"),
        ("[maya] sessions: folder /mnt/ssd/slots is held by another engine - conversations kept in RAM only", "other"),
        ("[maya] sessions: took back a 8000-token conversation from the folder (1300 MB, read 400 / checked 90 / "
         "loaded 160 ms); 70 new tokens to read", "took"),
        ("[maya] sessions: wrote a 8000-token conversation to the folder (1300 MB, 900 ms)", "saved"),
        ("[maya] sessions: saved a 8000-token conversation while idle (1.3 GB in RAM, 0.2 s)", "saved"),
        ("[maya] sessions: wrote a 8,000-token conversation to the folder (1300 MB, 900 ms) while idle", "saved"),
        ("hy4 slots: slot-20261125-3: write given up (stopped)", "other"),
        ("[maya] sessions: saved a 8000-token conversation (1.3 GB, 0.2 s)", "saved"),
        ("[maya] sessions: the engine's end: a 70000-token conversation not written in time (its file removed)",
         "not_saved"),
        ("[maya] sessions: the conversation in the model is left out at the engine's end: 2.1 s left, ~3.4 s needed",
         "end_skip"),
        ("[maya] sessions: a 9000-token conversation is left out at the engine's end: 0.6 s left, ~1.9 s needed",
         "end_skip"),
        ("[maya] sessions: verify: 1300 MB read back, byte for byte", "verify_ok"),
        ("[maya] sessions: verify: MISMATCH at byte 12 of 1000 (layer 3)", "bad"),
        ("[maya] sessions: verify: the read-back failed: cuda error", "bad"),
        ("[maya] sessions: refused slot-1-2.meta: another engine's identity", "bad"),
        ("[maya] sessions: slot-1-2.bin: checksum failed - removed, the prompt read from token 0", "bad"),
        ("[maya] something else entirely", ""),
    ]
    # phase 1 waits for the folder's copy: WROTE_FOLDER on the write, with or without "while idle" - never the RAM save
    for line, want in [
        ("[maya] sessions: saved a 8000-token conversation while idle (1.3 GB in RAM, 0.2 s)", False),
        ("[maya] sessions: wrote a 8000-token conversation to the folder (1300 MB, 900 ms) while idle", True),
        ("[maya] sessions: wrote a 8,000-token conversation to the folder (1300 MB, 900 ms)", True),
        ("hy4 slots: slot-20261125-3: write given up (stopped)", False),
    ]:
        got = bool(WROTE_FOLDER.search(line)) and classify(line) == "saved"
        good = got == want
        ok_all &= good
        print(f"  {'ok' if good else 'FAIL':5} self-test: folder write {line[8:70]!r}... -> {got}", flush=True)
    for line, want in fixtures:
        got = classify(line)
        good = got == want
        ok_all &= good
        print(f"  {'ok' if good else 'FAIL':5} self-test: classify {line[8:70]!r}... -> {got or '-'}"
              + ("" if good else f" (want {want})"), flush=True)
    # the .meta layout (HY4META2): a built one parsed back, an identity rewritten and resealed, damage refused
    import struct
    import zlib
    body = (META2 + struct.pack("<Q", 5) + b"ident" + struct.pack("<QQQQQ", 100, 7, 3, 0, 3)
            + struct.pack("<3i", 1, -2, 3) + struct.pack("<Q", 1) + struct.pack("<qQ", 1, 99))
    meta = body + struct.pack("<Q", zlib.crc32(body)) + META2
    m = meta_parse(meta)
    m2 = meta_parse(meta_with_identity(meta, "ident-foreign"))
    damaged = bytearray(meta)
    damaged[20] ^= 1
    good = (m is not None and m["tokens"] == [1, -2, 3] and m["identity"] == "ident" and m2 is not None
            and m2["identity"] == "ident-foreign" and m2["tokens"] == m["tokens"]
            and meta_parse(bytes(damaged)) is None and meta_parse(meta.replace(META2, b"HY4META1")) is None
            and meta_parse(meta + b"\0") is None)
    ok_all &= good
    print(f"  {'ok' if good else 'FAIL':5} self-test: the HY4META2 .meta parsed, its identity rewritten and resealed, "
          f"damage / HY4META1 / a trailing byte refused", flush=True)
    # the real doubled lines (a Z4 slots server's log, verbatim): each event once - stderr and its relay merged,
    # within a window and across two, identical events kept apart
    TURN = [
        'hy4 slots: saved a 3055-token conversation (0.14 GB in RAM in 0.11 s, its rings from token 0); 4 kept, 0.57 GB of 4.00 GB - 1 least recently used dropped',
        '[maya] sessions: saved a 3055-token conversation (0.14 GB in RAM in 0.11 s, its rings from token 0); 4 kept, 0.57 GB of 4.00 GB - 1 least recently used dropped',
        'hy4 slots: took back a 8056-token conversation from RAM (0.04 s); 75 new tokens to read',
        "hy4 slots: verify: the restored state is the slot's byte for byte - 70 runs, 392.1 MB on the GPUs, 8056 tokens, rings from 0, 8056 MTP rows",
        '[maya] sessions: took back a 8056-token conversation from RAM (0.04 s); 75 new tokens to read',
        "[maya] sessions: verify: the restored state is the slot's byte for byte - 70 runs, 392.1 MB on the GPUs, 8056 tokens, rings from 0, 8056 MTP rows",
    ]
    FILLER4 = [
        'hy4 slots: saved a 736-token conversation (0.03 GB in RAM in 0.04 s, its rings from token 0); 4 kept, 0.13 GB of 4.00 GB - 1 least recently used dropped',
        '[maya] sessions: saved a 736-token conversation (0.03 GB in RAM in 0.04 s, its rings from token 0); 4 kept, 0.13 GB of 4.00 GB - 1 least recently used dropped',
    ]
    FILLER5 = [
        'hy4 slots: saved a 736-token conversation (0.03 GB in RAM in 0.04 s, its rings from token 0); 4 kept, 0.13 GB of 4.00 GB - 1 least recently used dropped',
        '[maya] sessions: saved a 736-token conversation (0.03 GB in RAM in 0.04 s, its rings from token 0); 4 kept, 0.13 GB of 4.00 GB - 1 least recently used dropped',
    ]
    IDLE_W1 = [
        'hy4 slots: saved a 8046-token conversation while idle (0.36 GB in RAM in 0.24 s, its rings from token 0); 1 kept, 0.36 GB of 4.00 GB',
        'hy4 slots: wrote a 8046-token conversation to the folder (374 MB in 510 ms) while idle; 1 there, 0.36 GB',
    ]
    IDLE_W2 = [
        '[maya] sessions: saved a 8046-token conversation while idle (0.36 GB in RAM in 0.24 s, its rings from token 0); 1 kept, 0.36 GB of 4.00 GB',
        '[maya] sessions: wrote a 8046-token conversation to the folder (374 MB in 510 ms) while idle; 1 there, 0.36 GB',
    ]
    AID_TURN = [
        'hy4 slots: saved a 8142-token conversation (0.37 GB in RAM in 0.25 s, its rings from token 0); 2 kept, 0.37 GB of 4.00 GB - 1 earlier state(s) of it dropped',
        '[maya] sessions: saved a 8142-token conversation (0.37 GB in RAM in 0.25 s, its rings from token 0); 2 kept, 0.37 GB of 4.00 GB - 1 earlier state(s) of it dropped',
        'hy4 slots: wrote a 8142-token conversation to the folder (378 MB in 401 ms); 2 there, 0.73 GB',
    ]
    ABIN_TURN = [
        '[maya] sessions: wrote a 8142-token conversation to the folder (378 MB in 401 ms); 2 there, 0.73 GB',
        'hy4 slots: saved a 8142-token conversation (0.37 GB in RAM in 0.24 s, its rings from token 0); 3 kept, 0.74 GB of 4.00 GB',
        '[maya] sessions: saved a 8142-token conversation (0.37 GB in RAM in 0.24 s, its rings from token 0); 3 kept, 0.74 GB of 4.00 GB',
        "hy4 slots: refused slot-1795632972083-2: its checksum is not its .meta's (the .bin changed after it was written) - removed; reading its prompt again",
        "[maya] sessions: refused slot-1795632972083-2: its checksum is not its .meta's (the .bin changed after it was written) - removed; reading its prompt again",
        'hy4 slots: wrote a 8142-token conversation to the folder (378 MB in 415 ms); 2 there, 0.74 GB',
    ]
    START = [
        'hy4 slots: folder /home/mtudee/bench_mem/slots/folder: 2 conversation(s) kept from earlier runs (0.73 GB of 16.00 GB), 2 refused and removed',
        'hy4 slots: refused slot-1795632995377-3: made by another engine, pack, cache or split (file)',
        'hy4 slots: refused slot-foreign-cd61078d38: the .meta is cut short (identity)',
        '[maya] sessions: folder /home/mtudee/bench_mem/slots/folder: 2 conversation(s) kept from earlier runs (0.73 GB of 16.00 GB), 2 refused and removed',
        '[maya] sessions: refused slot-1795632995377-3: made by another engine, pack, cache or split (file)',
        '[maya] sessions: refused slot-foreign-cd61078d38: the .meta is cut short (identity)',
    ]

    def counts(ev: Events, lines: list[str]) -> dict[str, int]:
        c: dict[str, int] = {}
        for ln in ev.feed(lines):
            k = classify(ln)
            k = "wrote" if k == "saved" and WROTE_FOLDER.search(ln) else k
            c[k] = c.get(k, 0) + 1
        return c

    ev = Events()
    dedup = [
        ("the full A-turn window: 1 save, 1 take, 1 verify", counts(ev, TURN),
         {"saved": 1, "took": 1, "verify_ok": 1}),
        ("filler 4's save", counts(ev, FILLER4), {"saved": 1}),
        ("filler 5's identical save: an event of its own", counts(ev, FILLER5), {"saved": 1}),
    ]
    ev = Events()
    dedup += [
        ("phase 1's idle wait: the RAM save and the folder write", counts(ev, IDLE_W1), {"saved": 1, "wrote": 1}),
        ("the next request's window holds only their relays: no event", counts(ev, IDLE_W2), {}),
    ]
    ev = Events()
    dedup += [
        ("the restarted start: 1 folder listing, 2 refused files", counts(ev, START), {"folder_list": 1, "bad": 2}),
        ("A-id's turn: 1 save, 1 write (its relay comes in the next window)", counts(ev, AID_TURN),
         {"saved": 1, "wrote": 1}),
        ("A-bin's turn: the write's relay dropped; 1 save, 1 refusal, 1 write", counts(ev, ABIN_TURN),
         {"saved": 1, "bad": 1, "wrote": 1}),
    ]
    for what, got, want in dedup:
        good = got == want
        ok_all &= good
        print(f"  {'ok' if good else 'FAIL':5} self-test: events: {what} -> {got}" + ("" if good else f" (want {want})"),
              flush=True)
    # the A-turn window through send(): ONE take (the scenario's len(took) == 1), verified, the run passes
    run = Run.__new__(Run)
    run.srv, run.a, run.rows, run.steps, run.evidence, run.nonce = FakeSrv(), A, [], {}, True, "t"
    run.ev = Events()
    run.log = FakeLog([], TURN)
    with contextlib.redirect_stdout(io.StringIO()):
        r = run.send("self", "A-turn", [{"role": "user", "content": "x"}])
    good = len(r["took"]) == 1 and len(r["verified"]) == 1 and len(r["saved"]) == 1 and \
        all(x[2] is not False for x in run.rows) and "1 take(s), 1 verify ok event(s)" in run.rows[0][3]
    ok_all &= good
    print(f"  {'ok' if good else 'FAIL':5} self-test: send() on the real doubled A-turn lines: {len(r['took'])} take, "
          f"{len(r['verified'])} verify, {len(r['saved'])} save; {run.rows[0][3] if run.rows else '-'}", flush=True)
    print("self-test: " + ("PASS" if ok_all else "FAIL"), flush=True)
    return 0 if ok_all else 1


def banner(a) -> None:
    line = "!" * 100
    print(line, flush=True)
    print("!! PRECONDITIONS", flush=True)
    hot = {"off": "you say the server runs HY4_HOT=0: text comparisons are judged",
           "on": "you say the adaptive hot set is ON: text comparisons are INFO, the byte VERIFY lines are the proof",
           "unknown": "TEXT IDENTITY NEEDS THE SERVER STARTED WITH HY4_HOT=0 - the adaptive hot set moves experts "
                      "between requests and changes rounding; a text FAIL with it on may be placement, not the slots "
                      "(say --hot off / --hot on; or compare two fresh starts: --no-control --record / --compare)"}
    print("!! - " + hot[a.hot], flush=True)
    print("!! - " + ("--no-verify: the server runs without MAYA_HY4_SLOT_VERIFY - takes are NOT proven byte for byte"
                     if a.no_verify else "MAYA_HY4_SLOT_VERIFY=1 expected: every take needs a 'verify: ... byte for "
                     "byte' line; MISMATCH / failed read-back / refusal FAILS the run"), flush=True)
    print(f"!! - MAYA_HY4_SLOT_IDLE_S {a.idle_s}: an idle save of A before B counts as A set aside "
          f"(the server with MAYA_HY4_SLOT_IDLE_S=0 makes it B's park alone)", flush=True)
    if not a.no_control and any(x in a.scen for x in ("full", "partial")):
        print("!! - in-run controls: run the server with MAYA_HY4_SLOT_DIR= (empty) - with a folder the control's "
              "conversations are written there and stay takeable after the fillers", flush=True)
    if a.reference:
        print("!! - --reference: a KEEP=0 server - answers recorded, slot evidence not expected", flush=True)
    print(line, flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="maya-hy4's kept conversations against a running server")
    ap.add_argument("--host", default="192.168.50.115")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--api-key", default=os.environ.get("MAYA_API_KEY", ""))
    ap.add_argument("--self-test", action="store_true", help="the log rules offline (no server), then exit")
    ap.add_argument("--log", required=True, help="the server's log (its stdout: the `[maya] sessions:` lines)")
    ap.add_argument("--log-ssh", default="", help="read --log on this host over ssh (read-only: stat and tail)")
    ap.add_argument("--only", default="full,partial",
                    help="scenarios: full, partial, ring, restart, foreign (comma separated)")
    ap.add_argument("--ring", action="store_true", help="add the ring scenario (a ~21k-token conversation)")
    ap.add_argument("--phase", default="", choices=["", "1", "2", "both"],
                    help="restart / foreign: 1 = before the restart, 2 = after it (same --state), both = ask the "
                         "operator to restart and wait for the server (default for them: both)")
    ap.add_argument("--state", default="", help="restart / foreign: the file phase 1 writes and phase 2 reads")
    ap.add_argument("--wait-s", type=float, default=1800.0, help="--phase both: how long to wait for the restart")
    ap.add_argument("--folder", default="", help="foreign: the server's slot folder (MAYA_HY4_SLOT_DIR)")
    ap.add_argument("--folder-ssh", default="", help="--folder on this host over ssh")
    ap.add_argument("--hot", default="unknown", choices=["unknown", "off", "on"],
                    help="the server's adaptive hot set: off = started with HY4_HOT=0 (text identity judged)")
    ap.add_argument("--no-verify", action="store_true", help="the server runs without MAYA_HY4_SLOT_VERIFY=1")
    ap.add_argument("--no-control", action="store_true",
                    help="the test sequences alone (no in-run control, no fillers) - for fresh-start comparison")
    ap.add_argument("--reference", action="store_true", help="a KEEP=0 server: record the answers, expect no slots")
    ap.add_argument("--record", default="", help="write every step's answer here (JSON)")
    ap.add_argument("--compare", default="", help="a --record file of a KEEP=0 fresh start with the same --nonce")
    ap.add_argument("--nonce", default="", help="the prompts' nonce (the same in a reference and its comparison)")
    ap.add_argument("--len-a", type=int, default=8000, help="A's prompt (tokens)")
    ap.add_argument("--len-b", type=int, default=3000, help="B's prompt (tokens; at least --min)")
    ap.add_argument("--len-turn", type=int, default=64, help="the follow-up's new tokens")
    ap.add_argument("--max-new", type=int, default=32, help="the answers' length (greedy)")
    ap.add_argument("--partial-back", type=int, default=3000, help="A' leaves A this many prompt tokens before its end")
    ap.add_argument("--cap", type=int, default=4, help="the server's MAYA_HY4_SLOT_CAP")
    ap.add_argument("--min", type=int, default=512, help="the server's MAYA_HY4_SLOT_MIN")
    ap.add_argument("--gain", type=int, default=256, help="the engine's take margin")
    ap.add_argument("--idle-s", type=float, default=5.0, help="the server's MAYA_HY4_SLOT_IDLE_S")
    ap.add_argument("--write-wait", type=float, default=60.0, help="seconds past --idle-s to wait for an idle write")
    ap.add_argument("--ring-rows", type=int, default=16896, help="the engine's rings (Model::ring_rows; window + slack)")
    ap.add_argument("--ring-back", type=int, default=511, help="the positions a forward reads before it (ring_back)")
    ap.add_argument("--ring-margin", type=int, default=1000, help="tokens kept away from the boundary each side")
    ap.add_argument("--slot-ring", type=int, default=0, help="the server's MAYA_HY4_SLOT_RING (0: the whole ring)")
    ap.add_argument("--timeout", type=float, default=3600.0)
    if "--self-test" in sys.argv[1:]:
        return self_test()
    a = ap.parse_args()
    a.scen = [x.strip() for x in a.only.split(",") if x.strip()] + (["ring"] if a.ring else [])
    unknown = [x for x in a.scen if x not in ("full", "partial", "ring", "restart", "foreign")]
    if unknown:
        print(f"unknown scenario(s): {unknown}", flush=True)
        return 2
    phased = [x for x in a.scen if x in ("restart", "foreign")]
    if phased and not a.phase:
        a.phase = "both"
    if a.phase and [x for x in a.scen if x not in ("restart", "foreign")]:
        print("--phase runs restart / foreign alone: run full / partial / ring separately", flush=True)
        return 2
    if a.phase in ("1", "2") and not a.state:
        print("--phase 1 / 2 need --state (the file the two halves share)", flush=True)
        return 2
    if a.len_turn + 2 * a.max_new + 64 >= a.min:
        print(f"--len-turn + 2 * --max-new must stay well under --min ({a.min}): the test's A must rewind the "
              f"control's turn in place without setting it aside", flush=True)
        return 2
    if a.len_b < a.min or a.len_a < a.min + a.gain or a.partial_back < a.min:
        print("--len-b and --partial-back at least --min, --len-a at least --min + --gain", flush=True)
        return 2
    state: dict = {}
    if a.phase == "2":
        try:
            with open(a.state, encoding="utf-8") as f:
                state = json.load(f)
        except (OSError, ValueError) as e:
            print(f"--state {a.state}: {e}", flush=True)
            return 2
    nonce = a.nonce or state.get("nonce") or uuid.uuid4().hex[:10]
    ref = None
    if a.compare:
        try:
            with open(a.compare, encoding="utf-8") as f:
                ref = json.load(f)
        except (OSError, ValueError) as e:
            print(f"--compare {a.compare}: {e}", flush=True)
            return 2
        if not a.nonce and not state.get("nonce"):
            nonce = ref.get("nonce") or nonce
    banner(a)
    srv = Server(a.host, a.port, a.api_key, a.timeout)
    log = Log(a.log, a.log_ssh)
    try:
        if a.phase == "2":
            print(f"waiting up to {a.wait_s:.0f} s for {srv.base} ...", flush=True)
            end = time.monotonic() + a.wait_s
            while not srv.up() and time.monotonic() < end:
                time.sleep(3)
        srv.health()
        log.mark(state.get("log_off") if a.phase == "2" and state.get("log_off") is not None else None)
    except (urllib.error.URLError, OSError, ValueError) as e:
        print(f"the server or its log cannot be reached: {e}", flush=True)
        return 2
    run = Run(srv, log, a, nonce)
    print(f"server {srv.base} model {srv.model}; log {a.log_ssh + ':' if a.log_ssh else ''}{a.log}; nonce {nonce}",
          flush=True)
    try:
        if not a.phase:
            for x in a.scen:
                {"full": run.full, "partial": run.partial, "ring": run.ring}[x]()
        else:
            if a.phase in ("1", "both"):
                state = {"nonce": nonce}
                for x in phased:
                    run.phase1(x, state)
                state["log_off"] = log.size()
                if a.state:
                    with open(a.state, "w", encoding="utf-8") as f:
                        json.dump(state, f, indent=1)
            if a.phase == "1":
                print(f"\nphase 1 done: RESTART THE SERVER now (the same config, its folder), then run with "
                      f"--phase 2 --state {a.state}", flush=True)
            if a.phase == "both":
                print(f"\n>>> RESTART THE SERVER NOW (the same config and folder). Waiting up to {a.wait_s:.0f} s for "
                      f"it to go down and come back ...", flush=True)
                end = time.monotonic() + a.wait_s
                while srv.up() and time.monotonic() < end:
                    time.sleep(2)
                run.check("-", "the server went down (the restart)", not srv.up())
                while not srv.up() and time.monotonic() < end:
                    time.sleep(3)
                srv.health()
            if a.phase in ("2", "both"):
                start = run.ev.feed(log.since(settle=2.0))
                for ln in start:
                    print(f"          | {ln.strip()[:220]}", flush=True)
                listed = [ln for ln in start if classify(ln) == "folder_list"]
                run.slot_check("restart", "the restarted engine listed its folder (NOTE sessions: folder ...)",
                               bool(listed), listed[0].strip()[:160] if listed else "no folder line since phase 1")
                for x in phased:
                    run.phase2(x, state)
        if ref is not None:
            run.compare(ref)
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        run.check("-", f"the run ended early: {e}", False)
    if a.record:
        with open(a.record, "w", encoding="utf-8") as f:
            json.dump({"nonce": nonce, "reference": a.reference, "steps": run.steps}, f, indent=1)
        print(f"answers recorded in {a.record}", flush=True)
    return 0 if run.table() else 1


if __name__ == "__main__":
    sys.exit(main())

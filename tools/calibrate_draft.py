"""The speed calibration: does the DFlash2 drafter make THIS PC faster than the MTP block?  (The setup when DFlash2 is
chosen, the first start after an update for a config that drafts with it, ./maya.sh --speed-calibration.)

DFlash2 drafts up to 7 tokens a round and the model checks them in one pass.  Where the model's experts sit in VRAM
that pass is cheap and DFlash2 wins (#106: +44% on nine GPUs, +12% on two, +19% on one RTX 3090); where most of them
come from RAM, the CPU lane computes the experts of every drafted token and the pass costs more than the drafts save
(#127: an RTX 3090 + 3060 with Maya-M decoded at half the MTP block's speed; Tesla V100s were slower too).  Which kind
of PC this is depends on its GPUs' memory, the model, the context, the CPU and the RAM - so it is measured here, the
way Maya is used, and only the sustained speed counts:

  - the config twice, with and without --dflash: one engine start each, nothing else different (one GPU has no MTP
    block - there the other side decodes without drafting);
  - first a warm-up that is not counted: a long document is read (the depth of a real conversation) and answered
    once - the first answer after a start is the slowest, while the expert tiers settle;
  - then fresh questions about that document, each answered at length (ANSWER tokens), sampled as the dashboard does
    (the config's sampling and thinking level; a fixed seed per question - the same questions and seeds on both
    sides), until each side has written MIN_TOKENS tokens: its speed is all those tokens over all the time spent
    writing them, not one short burst;
  - DFlash2 is kept only when it beats the other side by more than MIN_GAIN: a tie keeps the MTP block.

The engine works on a copy of this PC's expert usage file (calibrate_glm.usage_copy), writes no learned expert profile
or routing trace (calibrate_glm.without_writes) and keeps no conversations aside (MAYA_GLM_SLOTS=0): the
calibration's answers neither lead the next start's warm-up nor take the place of this PC's kept conversations.

    python tools/calibrate_draft.py maya-<model>.json     # measure and print; ./maya.sh --speed-calibration applies it
"""
from __future__ import annotations

import copy
import json
import sys
import tempfile
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_glm as CAL  # noqa: E402  (usage_copy, close)

MIN_GAIN = 0.05            # DFlash2 must beat the other side by this much: runs of one setup differ by a few percent
MIN_TOKENS = 1536          # tokens each side writes after its warm-up (several long answers: a sustained speed)
ANSWER = 384               # the tokens an answer may take
WARM_ANSWER = 128          # the warm-up answer's (not counted)
DOC_TOKENS = 8192          # the document's length, at most (less when the context is small)
QUESTIONS = (              # one fresh question per answer: the expert tiers follow new text, as in a chat
    "Summarize the document above in about 300 words.",
    "Write a step-by-step guide for someone setting this up for the first time, based on the document above.",
    "List every setting or command-line option the document above mentions, with one sentence on what each does.",
    "Write a short FAQ - eight questions and their answers - about the document above.",
    "Explain to a beginner, using the document above, where the model's parts are kept while it runs, and why.",
    "Write a Python script that reads a JSON file of settings like the ones described above and prints them as a "
    "table, with comments.",
    "What problems could a user run into, according to the document above, and how are they solved?",
    "Write a short blog post that announces the features described in the document above.",
)
# The short probe's requests (GLM-5.3's one-GPU drafting, size {"mixed": True}): what Maya writes in real use - code,
# an explanation, plain text and the model's thinking - each a fresh chat of its own (no document read first: the probe
# measures the writing, not a prompt).  (question, thinking): True = the config's thinking level (the answer's first
# tokens are its thinking), False = thinking off (they are the answer itself).
MIXED = (
    ("Write a Python script that reads a CSV file of sales (date, region, amount) and prints the total for each region "
     "and month as a table, with comments.", False),
    ("Explain how a hash table works, then show a short Python class that implements one.", False),
    ("Write a friendly email to my team: the release is on track for Friday, two bugs are left, and every test passes. "
     "Ask who can review the last changes.", False),
    ("Write a JavaScript function that removes duplicate objects from an array by their id, keeping the newest one, "
     "and explain the choice of data structure.", True),
)
MIXED_WARM = "Say hello in one sentence."
# The read-and-write probe (run_phases; DeepSeek on two GPUs, maya.py's link_calibration): both of a request's phases
# - a document read, then an answer written - with fixed answers (temperature 0: the same tokens at every run, so the
# two sides differ by their speed alone, not by what a seed made them write).  A short warm-up document first (not
# counted: the first read and answer after a start are the slowest), whose read speed on this PC sizes the documents:
# each should take about PHASE_READ_S to read on the first side (PHASE_DOC tokens at least / at most; the same
# documents on both sides).  Then PHASE_DOCS fresh documents, each read whole (nothing of it kept from the one before)
# and answered with PHASE_ANSWER tokens.
PHASE_WARM_DOC = 1024      # the warm-up document's tokens
PHASE_WARM_ANSWER = 32     # ... and its answer's
PHASE_READ_S = 10.0        # the seconds a document's read should take
PHASE_DOC = (512, 4096)    # a document's tokens: at least, at most
PHASE_DOCS = 2             # documents read and answered a side
PHASE_ANSWER = 100         # the tokens an answer takes (PHASE_DOCS of them: ~200 tokens written a side)
PHASE_QUESTIONS = (
    "Summarize the document above in about 150 words.",
    "List the settings and commands the document above mentions, with one sentence on what each does.",
)


def without_dflash(cfg: dict) -> dict:
    """The config as it drafts without DFlash2: --dflash and its file left out (the engine then loads the MTP block)."""
    out = copy.deepcopy(cfg)
    args = out.get("args") or []
    if "--dflash" in args[:-1]:
        i = args.index("--dflash")
        del args[i:i + 2]
    return out


def with_dflash(cfg: dict, drafter) -> dict:
    """The config drafting with the DFlash2 drafter `drafter`."""
    out = without_dflash(cfg)
    out.setdefault("args", []).extend(["--dflash", str(drafter)])
    return out


def chat_parts(cfg: dict):
    """(the config's tokenizer, its chat template, the template's kwargs at the config's thinking level): requests
    rendered as the server renders them."""
    import strata_tokenizer as ST
    from serve.frontend import ChatTemplate, effort_kwargs
    tpath = Path(cfg["tokenizer"])
    tok = ST.Tokenizer.from_dir(tpath)
    tpl_path = tpath / "chat_template.jinja"
    tpl = ChatTemplate(tpl_path if tpl_path.exists() else ROOT / "serve" / "chat_template.jinja")
    try:
        kw = effort_kwargs(cfg.get("reasoning_effort"))
    except ValueError:                                 # (a level the template does not know: its own default)
        kw = {}
    return tok, tpl, kw


def requests(cfg: dict, doc_tokens: int = DOC_TOKENS, mixed: bool = False) -> tuple[list[int], list[list[int]], int]:
    """(the warm-up's ids, one list of ids per question, the document's tokens): chats as the server renders them -
    the model's template at the config's thinking level - each a document made of Maya's own docs, then a question
    about it.  Every request opens with the same document, so the engine reads it once and keeps it.  `mixed`: the
    MIXED requests instead (no document: 0 tokens)."""
    tok, tpl, kw = chat_parts(cfg)
    if mixed:
        def one(q, think):
            t_kw = kw if think else {"enable_thinking": False}
            return tok.encode(tpl.render([{"role": "user", "content": q}], **t_kw), parse_special=True)
        return one(MIXED_WARM, False), [one(q, think) for q, think in MIXED], 0
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) if p.exists())
    ids = tok.encode(text)
    args = cfg.get("args") or []
    ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args[:-1] else 32768
    n = max(256, min(doc_tokens, ctx - ANSWER - 2048, len(ids)))
    doc = tok.decode(ids[:n])

    def chat(q):
        return tok.encode(tpl.render([{"role": "user", "content": doc + "\n\n" + q}], **kw), parse_special=True)
    return chat("Read the document above, then say in two sentences what it is about."), [chat(q) for q in QUESTIONS], n


def answer(engine, ids: list[int], max_new: int, sampling: dict, seed: int) -> tuple[int, dict]:
    """One answer: (the tokens written, the engine's numbers for it - decode_ms, drafts_accepted, ...)."""
    s = dict(sampling)
    s["seed"] = seed
    n = sum(1 for t in engine.generate(ids, max_new, s, threading.Event()) if t is not None)
    return n, dict(getattr(engine, "last", None) or {})


def measure(cfg: dict, label: str, warm: list[int], questions: list[list[int]], n_doc: int, start_engine,
            say=print, tag: str = "", size: dict | None = None) -> dict:
    """One side: start its engine, the warm-up (not counted), then the questions until MIN_TOKENS tokens are written:
    {"tok_s": the sustained speed, "tokens", "answers", "seconds" (writing), "load_s", "drafts_accepted", ...}.
    `size` (a short probe): {"tokens", "answer", "warm"} in place of MIN_TOKENS, ANSWER, WARM_ANSWER."""
    size = size or {}
    min_tokens, answer_len = size.get("tokens", MIN_TOKENS), size.get("answer", ANSWER)
    sampling = dict(cfg.get("sampling") or {"temperature": 1.0, "top_p": 0.95})
    t0 = time.time()
    say(f"  {tag}{label}: loading the model (it reads the experts into the GPUs' memory and RAM first) ...")
    eng = start_engine(cfg)
    tokens, ms, answers, acc, off = 0, 0.0, 0, 0, 0
    try:
        load_s = time.time() - t0
        say(f"  {tag}{label}: loaded in {load_s:.0f} s; warming up - " +
            (f"it reads a document of {n_doc} tokens and answers once" if n_doc else "one short answer") +
            " (not counted) ...")
        answer(eng, warm, size.get("warm", WARM_ANSWER), sampling, 1)
        for i, q in enumerate(questions):
            n, last = answer(eng, q, answer_len, sampling, i + 2)
            d = float(last.get("decode_ms") or 0.0)
            if n < 2 or d <= 0.0:                      # (an answer that ended at once tells nothing)
                continue
            tokens, ms, answers = tokens + n, ms + d, answers + 1
            a, o = int(last.get("drafts_accepted") or 0), int(last.get("drafts_offered") or 0)
            acc, off = acc + a, off + o
            say(f"  {tag}{label}: answer {answers}: {n} tokens at {n / d * 1000.0:.1f} tokens/s" +
                (f" ({100.0 * a / o:.0f}% of its drafts accepted)" if o else "") +
                f" - {min(tokens, min_tokens)} of {min_tokens} tokens measured")
            if tokens >= min_tokens:
                break
    finally:
        CAL.close(eng)
    rate = tokens / (ms / 1000.0) if ms > 0 else 0.0
    say(f"  {tag}{label}: {rate:.1f} tokens/s sustained ({tokens} tokens in {ms / 1000.0:.0f} s of writing)" +
        (f", {100.0 * acc / off:.0f}% of the drafts accepted" if off else ""))
    return {"tok_s": round(rate, 2), "tokens": tokens, "answers": answers, "seconds": round(ms / 1000.0, 1),
            "load_s": round(load_s), "drafts_accepted": acc, "drafts_offered": off}


def run(other: dict, dflash: dict, say=print, start_engine=None, other_label: str = "the MTP block",
        draft_label: str = "DFlash2", size: dict | None = None) -> dict:
    """Both sides measured, `other` (the config without DFlash2) first: {"other": {...}, "dflash2": {...},
    "keep": "dflash2" or "other", "gain": DFlash2's speed over the other's - 1}.  `start_engine(cfg)` returns a
    started engine (serve.server.StrataEngine, or a stand-in in tests); `draft_label` names the drafting side (GLM-5.3's
    one-GPU drafting: the same two sides, its MAYA_GLM_MTP_DECODE off and on); `size` makes it a short probe
    ({"doc" or "mixed", "tokens", "answer", "warm"}: measure's; "mixed": the MIXED requests).  Raises when a side could
    not be measured."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env, engine_args

        def start_engine(c):
            return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=child_env(c))
    if size and size.get("mixed"):
        warm, questions, n_doc = requests(dflash, mixed=True)
    else:
        warm, questions, n_doc = requests(dflash, size["doc"]) if size and "doc" in size else requests(dflash)
    res: dict = {}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="maya-speed-") as tmp:
        for i, (key, cfg, label) in enumerate((("other", other, other_label), ("dflash2", dflash, draft_label))):
            if i == 1:
                say(f"  (the first side took {(time.time() - t0) / 60.0:.0f} min: about as long again for {draft_label})")
            # (no conversations kept aside: with MAYA_GLM_SLOT_KEEP the calibration's would push out this PC's own;
            # and no learned expert profile or routing trace written: its answers are not this PC's use)
            c = {**cfg, "args": CAL.without_writes(list(cfg.get("args") or [])),
                 "env": {**(cfg.get("env") or {}), **CAL.usage_copy(cfg, Path(tmp)), "MAYA_GLM_SLOTS": "0"}}
            res[key] = measure(c, label, warm, questions, n_doc, start_engine, say, f"[{i + 1}/2] ", size)
    for key in ("other", "dflash2"):
        if res[key]["tokens"] < (size or {}).get("tokens", MIN_TOKENS) // 2:
            raise RuntimeError(f"{draft_label if key == 'dflash2' else other_label} wrote only {res[key]['tokens']} "
                               "tokens: too few for a sustained speed")
    o, d = res["other"]["tok_s"], res["dflash2"]["tok_s"]
    res["keep"] = "dflash2" if d > o * (1.0 + MIN_GAIN) else "other"
    res["gain"] = round(d / o - 1.0, 3) if o > 0 else None
    return res


def phase_requests(cfg: dict):
    """(make(first, n, question) -> the ids of a chat whose user turn is the n document tokens from `first` and the
    question, the document tokens there are, the most a document may take in this config's context)."""
    tok, tpl, kw = chat_parts(cfg)
    ids = tok.encode("\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                                 for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) if p.exists()))
    args = cfg.get("args") or []
    ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args[:-1] else 32768

    def make(first: int, n: int, question: str) -> list[int]:
        doc = tok.decode(ids[first:first + n])
        return tok.encode(tpl.render([{"role": "user", "content": doc + "\n\n" + question}], **kw),
                          parse_special=True)
    return make, len(ids), ctx - PHASE_ANSWER - 2048


def phase_doc_tokens(read_tok_s: float, have: int, room: int) -> int:
    """A document's tokens for a PC that read the warm-up at `read_tok_s`: about PHASE_READ_S of reading, in whole
    256s, within PHASE_DOC, the context's room and the text there is (`have` tokens after the warm-up's)."""
    n = int(read_tok_s * PHASE_READ_S) // 256 * 256
    n = max(PHASE_DOC[0], min(PHASE_DOC[1], n))
    return max(64, min(n, room, have // PHASE_DOCS))


def read_speed(last: dict) -> tuple[int, float]:
    """(the prompt tokens the engine read for its last answer - less what it kept from the prompt before -, the
    milliseconds that took)."""
    n = int(last.get("prompt_tokens") or 0) - int(last.get("reused") or 0)
    return max(0, n), float(last.get("prompt_ms") or 0.0)


def measure_phases(cfg: dict, label: str, make, have: int, room: int, start_engine, say=print, tag: str = "",
                   doc_tokens: int | None = None, close=None) -> dict:
    """One side of the read-and-write probe: {"read_tok_s", "write_tok_s", "doc_tokens" (sized here when
    `doc_tokens` is None: the first side), "read_tokens", "write_tokens", "seconds" (reading and writing, the
    warm-up too), "load_s"}."""
    greedy = {"temperature": 0.0}
    t0 = time.time()
    say(f"  {tag}{label}: loading the model ...")
    eng = start_engine(cfg)
    r_n, r_ms, w_n, w_ms = 0, 0.0, 0, 0.0
    try:
        load_s = time.time() - t0
        t1 = time.time()
        warm_n = max(64, min(PHASE_WARM_DOC, room, have // (PHASE_DOCS + 1)))
        say(f"  {tag}{label}: loaded in {load_s:.0f} s; warming up - a document of {warm_n} tokens, read and "
            "answered once (not counted) ...")
        _, last = answer(eng, make(0, warm_n, "Say in one sentence what the document above is about."),
                         PHASE_WARM_ANSWER, greedy, 1)
        if doc_tokens is None:
            n, ms = read_speed(last)
            if n < 1 or ms <= 0.0:
                raise RuntimeError(f"{label} did not say how long it read the warm-up for")
            doc_tokens = phase_doc_tokens(n / ms * 1000.0, have - warm_n, room)
        for i in range(PHASE_DOCS):
            ids = make(warm_n + i * doc_tokens, doc_tokens, PHASE_QUESTIONS[i % len(PHASE_QUESTIONS)])
            n, last = answer(eng, ids, PHASE_ANSWER, greedy, 1)
            pn, pms = read_speed(last)
            d = float(last.get("decode_ms") or 0.0)
            if pn < 1 or pms <= 0.0 or n < 2 or d <= 0.0:
                continue
            r_n, r_ms, w_n, w_ms = r_n + pn, r_ms + pms, w_n + n, w_ms + d
            say(f"  {tag}{label}: document {i + 1} of {PHASE_DOCS}: {pn} tokens read at {pn / pms * 1000.0:.0f} "
                f"tokens/s, {n} written at {n / d * 1000.0:.1f} tokens/s")
        seconds = time.time() - t1
    finally:
        (close or CAL.close)(eng)
    if r_ms <= 0.0 or w_ms <= 0.0 or w_n < PHASE_DOCS * PHASE_ANSWER // 2:
        raise RuntimeError(f"{label} wrote only {w_n} tokens after {r_n} read: too few for a speed")
    return {"read_tok_s": round(r_n / r_ms * 1000.0, 1), "write_tok_s": round(w_n / w_ms * 1000.0, 2),
            "doc_tokens": doc_tokens, "read_tokens": r_n, "write_tokens": w_n, "seconds": round(seconds, 1),
            "load_s": round(load_s)}


def run_phases(other: dict, both: dict, say=print, start_engine=None, other_label: str = "the AMD GPU alone",
               both_label: str = "both GPUs", between=None, close=None) -> dict:
    """The read-and-write probe, `other` first: {"other": {...}, "both": {...} (measure_phases), "read": both's read
    speed over the other's, "write": the same for the answers, "gain": the two weighed alike (their geometric mean)
    - 1}.  `between()` runs before each side's engine starts (maya.py: the pack's learned files as they were, so
    both sides start alike); `close(engine)` ends a side's engine (calibrate_glm.close when None).  Raises when a
    side could not be measured."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env, engine_args

        def start_engine(c):
            return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=child_env(c))
    make, have, room = phase_requests(both)
    res: dict = {}
    with tempfile.TemporaryDirectory(prefix="maya-speed-") as tmp:
        for i, (key, cfg, label) in enumerate((("other", other, other_label), ("both", both, both_label))):
            if between is not None:
                between()
            c = {**cfg, "args": CAL.without_writes(list(cfg.get("args") or [])),
                 "env": {**(cfg.get("env") or {}), **CAL.usage_copy(cfg, Path(tmp)), "MAYA_GLM_SLOTS": "0"}}
            res[key] = measure_phases(c, label, make, have, room, start_engine, say, f"[{i + 1}/2] ",
                                      res["other"]["doc_tokens"] if i else None, close)
    o, b = res["other"], res["both"]
    res["read"] = round(b["read_tok_s"] / o["read_tok_s"], 4)
    res["write"] = round(b["write_tok_s"] / o["write_tok_s"], 4)
    res["gain"] = round((res["read"] * res["write"]) ** 0.5 - 1.0, 4)
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate_draft.py <maya-*.json>  (a config that drafts with DFlash2: --dflash in its args)")
    import maya_names
    cfg = maya_names.env_in(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")))
    if "--dflash" not in (cfg.get("args") or [])[:-1]:
        sys.exit("this config does not draft with DFlash2 (no --dflash in its args)")
    print(json.dumps(run(without_dflash(cfg), cfg), indent=1))

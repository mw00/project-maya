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
or routing trace (calibrate_glm.without_writes) and keeps no conversations aside (STRATA_GLM_SLOTS=0): the
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


def requests(cfg: dict) -> tuple[list[int], list[list[int]], int]:
    """(the warm-up's ids, one list of ids per question, the document's tokens): chats as the server renders them -
    the model's template at the config's thinking level - each a document made of Maya's own docs, then a question
    about it.  Every request opens with the same document, so the engine reads it once and keeps it."""
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
    text = "\n\n".join(p.read_text(encoding="utf-8", errors="replace")
                       for p in [ROOT / "README.md"] + sorted((ROOT / "docs").glob("*.md")) if p.exists())
    ids = tok.encode(text)
    args = cfg.get("args") or []
    ctx = int(args[args.index("--max-context") + 1]) if "--max-context" in args[:-1] else 32768
    n = max(256, min(DOC_TOKENS, ctx - ANSWER - 2048, len(ids)))
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
            say=print, tag: str = "") -> dict:
    """One side: start its engine, the warm-up (not counted), then the questions until MIN_TOKENS tokens are written:
    {"tok_s": the sustained speed, "tokens", "answers", "seconds" (writing), "load_s", "drafts_accepted", ...}."""
    sampling = dict(cfg.get("sampling") or {"temperature": 1.0, "top_p": 0.95})
    t0 = time.time()
    say(f"  {tag}{label}: loading the model (it reads the experts into the GPUs' memory and RAM first) ...")
    eng = start_engine(cfg)
    tokens, ms, answers, acc, off = 0, 0.0, 0, 0, 0
    try:
        load_s = time.time() - t0
        say(f"  {tag}{label}: loaded in {load_s:.0f} s; warming up - it reads a document of {n_doc} tokens and "
            f"answers once (not counted) ...")
        answer(eng, warm, WARM_ANSWER, sampling, 1)
        for i, q in enumerate(questions):
            n, last = answer(eng, q, ANSWER, sampling, i + 2)
            d = float(last.get("decode_ms") or 0.0)
            if n < 2 or d <= 0.0:                      # (an answer that ended at once tells nothing)
                continue
            tokens, ms, answers = tokens + n, ms + d, answers + 1
            acc += int(last.get("drafts_accepted") or 0)
            off += int(last.get("drafts_offered") or 0)
            say(f"  {tag}{label}: answer {answers}: {n} tokens at {n / d * 1000.0:.1f} tokens/s - "
                f"{min(tokens, MIN_TOKENS)} of {MIN_TOKENS} tokens measured")
            if tokens >= MIN_TOKENS:
                break
    finally:
        CAL.close(eng)
    rate = tokens / (ms / 1000.0) if ms > 0 else 0.0
    say(f"  {tag}{label}: {rate:.1f} tokens/s sustained ({tokens} tokens in {ms / 1000.0:.0f} s of writing)" +
        (f", {100.0 * acc / off:.0f}% of the drafts accepted" if off else ""))
    return {"tok_s": round(rate, 2), "tokens": tokens, "answers": answers, "seconds": round(ms / 1000.0, 1),
            "load_s": round(load_s), "drafts_accepted": acc, "drafts_offered": off}


def run(other: dict, dflash: dict, say=print, start_engine=None, other_label: str = "the MTP block") -> dict:
    """Both sides measured, `other` (the config without DFlash2) first: {"other": {...}, "dflash2": {...},
    "keep": "dflash2" or "other", "gain": DFlash2's speed over the other's - 1}.  `start_engine(cfg)` returns a
    started engine (serve.server.StrataEngine, or a stand-in in tests).  Raises when a side could not be measured."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env, engine_args

        def start_engine(c):
            return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=child_env(c))
    warm, questions, n_doc = requests(dflash)
    res: dict = {}
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="maya-speed-") as tmp:
        for i, (key, cfg, label) in enumerate((("other", other, other_label), ("dflash2", dflash, "DFlash2"))):
            if i == 1:
                say(f"  (the first side took {(time.time() - t0) / 60.0:.0f} min: about as long again for DFlash2)")
            # (no conversations kept aside: with STRATA_GLM_SLOT_KEEP the calibration's would push out this PC's own;
            # and no learned expert profile or routing trace written: its answers are not this PC's use)
            c = {**cfg, "args": CAL.without_writes(list(cfg.get("args") or [])),
                 "env": {**(cfg.get("env") or {}), **CAL.usage_copy(cfg, Path(tmp)), "STRATA_GLM_SLOTS": "0"}}
            res[key] = measure(c, label, warm, questions, n_doc, start_engine, say, f"[{i + 1}/2] ")
    for key in ("other", "dflash2"):
        if res[key]["tokens"] < MIN_TOKENS // 2:
            raise RuntimeError(f"{'DFlash2' if key == 'dflash2' else other_label} wrote only {res[key]['tokens']} "
                               "tokens: too few for a sustained speed")
    o, d = res["other"]["tok_s"], res["dflash2"]["tok_s"]
    res["keep"] = "dflash2" if d > o * (1.0 + MIN_GAIN) else "other"
    res["gain"] = round(d / o - 1.0, 3) if o > 0 else None
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate_draft.py <maya-*.json>  (a config that drafts with DFlash2: --dflash in its args)")
    import maya_names
    cfg = maya_names.env_in(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")))
    if "--dflash" not in (cfg.get("args") or [])[:-1]:
        sys.exit("this config does not draft with DFlash2 (no --dflash in its args)")
    print(json.dumps(run(without_dflash(cfg), cfg), indent=1))

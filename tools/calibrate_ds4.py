"""The speed calibration for DeepSeek-V4.1 (maya-ds4) and MiniMax-M3.1 (maya-mm3): does the DSpark drafter make THIS
PC faster?  (The setup, the first start of a config that has the drafter and no calibration yet, ./maya.sh
--speed-calibration.)

DSpark drafts up to 5 tokens a round and the model checks them in one window.  Where the model's experts sit in the
GPUs' memory that window costs about what one token does and drafting wins; where most experts are computed on the CPU
or read from the SSD, the window's extra experts cost more than the drafts save (a Tesla V100 with 64 GB of RAM: slower
with it).  And drafting pays most where the next tokens are easy to guess: code and tool calls (a function's body, a
file written through a tool) can run twice as fast with it while plain chat runs slower - so a chat-only measurement
can turn off a drafter that doubles a coding agent's speed.  So it is measured here, on the three kinds of text Maya
writes:

  - the config twice: without the drafter (--dspark none: its memory then holds more of the model's experts) and with
    it - one engine start each, nothing else different, on the config's GPUs (one or several);
  - first a warm-up that is not counted: one short answer;
  - then fixed requests of three KINDS - chat (plain prose, at the config's thinking level), code (thinking off: the
    tokens are the code) and tool calls (a tools list, thinking off: the tokens are the call, a file's content in it) -
    each a fresh chat, sampled as the dashboard does (the config's sampling, a fixed seed per request - the same on
    both sides), until each kind has written KIND_TOKENS tokens;
  - a side's speed is the combined one: the same number of tokens of each kind over the time they took (each kind's
    tokens over its writing time, then 3 / the sum of the three 1 / speed) - one slow kind is not hidden by a fast one;
  - the drafter is kept only when its combined speed wins by more than MIN_GAIN.

The engine reads a copy of this PC's expert profile and writes back to the copy only: the calibration's answers do not
lead the next start's placement.

    python tools/calibrate_ds4.py maya-ds4.json      # measure and print; ./maya.sh --speed-calibration applies it
"""
from __future__ import annotations

import copy
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_draft as CD  # noqa: E402  (chat_parts, answer: rendered and answered as GLM's)

MIN_GAIN = CD.MIN_GAIN
KIND_TOKENS = 384          # tokens each kind writes a side after the warm-up
ANSWER = 192               # an answer's cap: two answers a kind at least
MIN_TOKENS = 3 * KIND_TOKENS   # (a side's tokens: chat + code + tool calls)
WARM = "Say hello in one sentence."
# (kind, request, thinking): True = the config's thinking level, False = thinking off.  Fixed, so every PC and both
# sides answer the same requests.
CHAT = (
    ("Explain to a beginner how a refrigerator keeps food cold, in a few short paragraphs.", True),
    ("Write a friendly email to my team: the release is on track for Friday, two bugs are left, and every test passes. "
     "Ask who can review the last changes.", True),
    ("What are the pros and cons of working from home? Answer in a short essay.", True),
    ("Tell me a short story about a lighthouse keeper who finds a message in a bottle.", True),
)
CODE = (
    ("Write a Python script that reads a CSV file of sales (date, region, amount) and prints the total for each region "
     "and month as a table, with comments.", False),
    ("Write a JavaScript function that removes duplicate objects from an array by their id, keeping the newest one, "
     "with a short test.", False),
    ("Write a C function that parses a line of comma-separated integers into an array, with error handling, and a "
     "small main() that uses it.", False),
    ("Write a SQL schema for a small library (books, members, loans) and three queries: overdue loans, the most "
     "borrowed books, members with no loans.", False),
)
TOOL_CALLS = (
    ("Create the file inventory.py: a small command-line inventory manager (add, remove and list items with their "
     "quantities, saved to a JSON file). Use the write_file tool.", False),
    ("Create the file test_inventory.py with unittest tests for an inventory manager's add, remove and list functions. "
     "Use the write_file tool.", False),
    ("Create server.js: a minimal Node.js HTTP server with /health and /items routes (items kept in memory). Use the "
     "write_file tool, then start it with the run_command tool.", False),
    ("Create the file README.md for a command-line inventory manager: install, usage with examples, and the JSON file "
     "format. Use the write_file tool.", False),
)
TOOLS = [
    {"name": "write_file", "description": "Write a file (it is created or replaced) in the project.",
     "parameters": {"type": "object", "properties": {
         "path": {"type": "string", "description": "The file's path, relative to the project."},
         "content": {"type": "string", "description": "The file's whole content."}},
         "required": ["path", "content"]}},
    {"name": "run_command", "description": "Run a shell command in the project and return its output.",
     "parameters": {"type": "object", "properties": {
         "command": {"type": "string", "description": "The command line."}}, "required": ["command"]}},
]
KINDS = (("chat", "chat", CHAT, None), ("code", "code", CODE, None), ("tools", "tool calls", TOOL_CALLS, TOOLS))


def option(args: list, name: str) -> str | None:
    return args[args.index(name) + 1] if name in args[:-1] else None


def without(args: list, name: str) -> list:
    out = list(args)
    while name in out[:-1]:
        i = out.index(name)
        del out[i:i + 2]
    return out


def is_mm3(cfg: dict) -> bool:
    """A MiniMax-M3.1-Flash config (maya-mm3): its DSpark drafter is measured the same way."""
    return (cfg.get("installer") or {}).get("family") == "minimax" or Path(cfg.get("exe", "")).name.startswith("maya-mm3")


def with_dspark(cfg: dict, mode: str) -> dict:
    """The config with --dspark `mode`: "none" (the drafter not loaded; maya-mm3 spells it "off") or "on"."""
    out = copy.deepcopy(cfg)
    if mode == "none" and is_mm3(cfg):
        mode = "off"
    out["args"] = without(out.get("args") or [], "--dspark") + ["--dspark", mode]
    return out


def has_drafter(cfg: dict) -> bool:
    """The config's pack holds the DSpark drafter (an EXL3 pack's, or a GGUF pack's from drafter_pack.py; maya-mm3's
    pack indexes its dspark/ folder as dspark.* tensors)."""
    pack = option(cfg.get("args") or [], "--pack")
    t = Path(pack) / "tensors.txt" if pack else None
    if t is None or not t.is_file():
        return False
    mark = "t dspark." if is_mm3(cfg) else "t mtp."
    with t.open(encoding="utf-8", errors="replace") as f:
        return any(line.startswith(mark) for line in f)


def profile_copy(cfg: dict, tmp: Path) -> dict:
    """The config reading (and writing) a copy of its expert profile in `tmp` (maya-mm3 only reads its --profile)."""
    if is_mm3(cfg):
        return copy.deepcopy(cfg)
    args = cfg.get("args") or []
    src = option(args, "--expert-profile") or str(Path(option(args, "--pack") or ".") / "expert-profile.bin")
    out = copy.deepcopy(cfg)
    if src == "none":
        return out
    dst = tmp / "expert-profile.bin"
    if Path(src).is_file():
        shutil.copyfile(src, dst)
    out["args"] = without(args, "--expert-profile") + ["--expert-profile", str(dst)]
    return out


def requests(cfg: dict) -> tuple[list[int], dict[str, list[list[int]]]]:
    """(the warm-up's ids, {kind: one list of ids per request}): each request a fresh chat as the server renders it
    (the model's template; the tool calls' with TOOLS)."""
    tok, tpl, kw = CD.chat_parts(cfg)

    def one(q, think, tools=None):
        return tok.encode(tpl.render([{"role": "user", "content": q}], tools=tools,
                                     **(kw if think else {"enable_thinking": False})), parse_special=True)
    return one(WARM, False), {key: [one(q, think, tools) for q, think in reqs] for key, _, reqs, tools in KINDS}


def combined(kinds: dict) -> float:
    """The same number of tokens of each kind over the time they take: len / the sum of 1 / each kind's speed."""
    rates = [k["tok_s"] for k in kinds.values()]
    return round(len(rates) / sum(1.0 / r for r in rates), 2) if rates and all(r > 0 for r in rates) else 0.0


def measure(cfg: dict, label: str, warm: list[int], kinds: dict, start_engine, say=print, tag: str = "") -> dict:
    """One side: start its engine, the warm-up (not counted), then each kind's requests until it has written
    KIND_TOKENS tokens: {"tok_s": the combined speed, "kinds": {kind: {"tok_s", "tokens", "seconds"}}, "tokens",
    "answers", "seconds" (writing), "load_s", "drafts_accepted", "drafts_offered"}."""
    sampling = dict(cfg.get("sampling") or {"temperature": 1.0, "top_p": 0.95})
    t0 = time.time()
    say(f"  {tag}{label}: loading the model (it reads the experts into the GPUs' memory and RAM first) ...")
    eng = start_engine(cfg)
    out: dict = {}
    answers, acc, off, seed = 0, 0, 0, 1
    try:
        load_s = time.time() - t0
        say(f"  {tag}{label}: loaded in {load_s:.0f} s; warming up - one short answer (not counted) ...")
        CD.answer(eng, warm, CD.WARM_ANSWER, sampling, seed)
        for key, name, _, _ in KINDS:
            tokens, ms = 0, 0.0
            for q in kinds[key]:
                seed += 1                              # (a fixed seed per request: the same on both sides)
                n, last = CD.answer(eng, q, ANSWER, sampling, seed)
                d = float(last.get("decode_ms") or 0.0)
                if n < 2 or d <= 0.0:                  # (an answer that ended at once tells nothing)
                    continue
                tokens, ms, answers = tokens + n, ms + d, answers + 1
                acc, off = acc + int(last.get("drafts_accepted") or 0), off + int(last.get("drafts_offered") or 0)
                if tokens >= KIND_TOKENS:
                    break
            rate = tokens / (ms / 1000.0) if ms > 0 else 0.0
            out[key] = {"tok_s": round(rate, 2), "tokens": tokens, "seconds": round(ms / 1000.0, 1)}
            say(f"  {tag}{label}: {name}: {rate:.1f} tokens/s ({tokens} tokens)")
    finally:
        CD.CAL.close(eng)
    tok_s = combined(out)
    tokens, secs = sum(k["tokens"] for k in out.values()), sum(k["seconds"] for k in out.values())
    say(f"  {tag}{label}: {tok_s:.1f} tokens/s combined (chat, code and tool calls alike)" +
        (f", {100.0 * acc / off:.0f}% of the drafts accepted" if off else ""))
    return {"tok_s": tok_s, "kinds": out, "tokens": tokens, "answers": answers, "seconds": round(secs, 1),
            "load_s": round(load_s), "drafts_accepted": acc, "drafts_offered": off}


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Both sides measured, without the drafter first: {"none": {...}, "dspark": {...}, "keep": "dspark" or "none",
    "gain": the drafter's combined speed over the other's - 1}.  Raises when a side could not be measured."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env, engine_args

        def start_engine(c):
            return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=child_env(c))
    res: dict = {}
    warm, kinds = requests(cfg)
    for i, (key, mode, label) in enumerate((("none", "none", "without drafting"), ("dspark", "on", "DSpark"))):
        with tempfile.TemporaryDirectory(prefix="maya-speed-") as tmp:   # (each side from the PC's own profile)
            c = profile_copy(with_dspark(cfg, mode), Path(tmp))
            res[key] = measure(c, label, warm, kinds, start_engine, say, f"[{i + 1}/2] ")
    for key, label in (("none", "without drafting"), ("dspark", "DSpark")):
        short = [name for k, name, _, _ in KINDS if res[key]["kinds"][k]["tokens"] < 2]
        if short:
            raise RuntimeError(f"{label} wrote nothing measurable for {', '.join(short)}")
        if res[key]["tokens"] < MIN_TOKENS // 2:
            raise RuntimeError(f"{label} wrote only {res[key]['tokens']} tokens: too few for a sustained speed")
    o, d = res["none"]["tok_s"], res["dspark"]["tok_s"]
    res["keep"] = "dspark" if d > o * (1.0 + MIN_GAIN) else "none"
    res["gain"] = round(d / o - 1.0, 3) if o > 0 else None
    return res


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate_ds4.py <maya-ds4.json>")
    import maya_names
    cfg = maya_names.env_in(json.loads(Path(sys.argv[1]).read_text(encoding="utf-8-sig")))
    if not has_drafter(cfg):
        sys.exit("this config's pack has no DSpark drafter")
    print(json.dumps(run(cfg), indent=1))

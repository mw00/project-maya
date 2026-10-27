"""Tune maya-ds4 for this PC (./maya.sh --calibrate on a DeepSeek-V4.1-Flash-1017 config).

The engine balances its layers across the GPUs itself, but two of its settings depend on the PC, the quant and what it
serves, and the setup only guesses them:
  --tp           how many GPUs of one kind split each layer (tensor parallel; the groups are runs of that many equal
                 GPUs, the rest alone - or, the number of GPUs, one group of all of them).  More members decode faster
                 (9 GPUs, 2.9 bits: --tp 2 103 tok/s, 4 139, 8 151) but read a long prompt slower (37K tokens: 908,
                 760, 607 tok/s - fewer pipeline stages for its chunks); the setup writes --tp 2.
  --cpu-threads  the CPU lane's threads: the experts VRAM does not hold are computed there, from RAM, and past the
                 memory's bandwidth more threads only wait.
Both are fixed when the engine starts, so every setting tried is an engine start of its own (minutes: the experts are
read into RAM and VRAM first).  Each start answers fresh prompts (calibrate_glm.PROMPTS, the chat template, thinking
off) after a warm-up, then reads one long prompt: decode speed decides - an agent's turns mostly reuse their
conversation's cache, so the answer is what they wait for - and a setting whose long prompt reads slower than
PROMPT_FLOOR of the config's own is not taken.  A setting is kept only when it beats the config's own by more than
MIN_GAIN; one that wins by less than CONFIRM_BELOW is started once more beside the config's own, and the two starts'
rates decide together (two starts of one setting differ by a few percent: the expert placement and the OS).
Effective groups come from each GPU's VRAM tp_n field, not --tp: fallback duplicates close before inference and are
not ranked as different settings. Incomplete grouping telemetry preserves the config's TP rather than guessing.

    python tools/calibrate_ds4_tp.py maya-ds4.json     # measure and print; ./maya.sh --calibrate also saves it
"""
from __future__ import annotations

import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_glm as CAL  # noqa: E402  the prompts, the measuring session, the engine's start and stop

MIN_GAIN = 0.05                          # a setting must beat the config's own by this much (separate starts: noisier)
CONFIRM_BELOW = 0.15                     # ... and one that wins by less than this is started again beside it
PROMPT_FLOOR = 0.6                       # its long prompt must read at least this share of the config's own speed
GROUPS = 2                               # prompt groups (CAL.GROUP each) a start's decode speed is the median of
LONG_TOKENS = 16384                      # the long prompt (CHANGELOG.md, as one message)
KEYS = ("--tp", "--cpu-threads")         # the engine arguments a calibration owns (apply() sets or removes them)


def tp_candidates(vram: list[float]) -> list[int]:
    """The --tp values that group these GPUs (their memory, in the engine's order) differently: 2, 4, 8, 16 up to the
    longest run of equal GPUs, that run's length, and every GPU in one group when there are more than that run; with no
    two GPUs alike, 1 (each alone) and all of them together.  [] for one GPU."""
    n = len(vram)
    if n < 2:
        return []
    runs, i = [], 0
    while i < n:
        j = i + 1
        while j < n and vram[j] == vram[i]:
            j += 1
        runs.append(j - i)
        i = j
    longest = max(runs)
    if longest < 2:
        return [1, n]
    out = [t for t in (2, 4, 8, 16) if t <= longest]
    for t in (longest, n):
        if t not in out:
            out.append(t)
    return out


def thread_candidates(default: int, every_cpu: bool = False) -> list[int]:
    """Fewer CPU threads than the engine's own: three quarters, then a half (at least 2).  every_cpu (the lane takes
    every CPU this process may use): first 2 and 4 fewer - such a lane leaves the engine's other threads none (#433: 28
    of 32 vCPUs 1.9x the decode of 32)."""
    out = []
    for w in ((default - 2, default - 4) if every_cpu else ()) + (round(default * 3 / 4), round(default / 2)):
        if 2 <= w < default and w not in out:
            out.append(w)
    return sorted(out, reverse=True)   # (the sweep stops at the first that gains nothing: the fewest last)


def usable_cpus() -> int:
    """The logical CPUs this process may use (its affinity: a container's or taskset's share), else the machine's."""
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 0


def arg(args: list, flag: str) -> str | None:
    return args[args.index(flag) + 1] if flag in args[:-1] else None


def apply(args: list, chosen: dict, keep=()) -> list:
    """`args` with the calibrated settings ({flag: value, or None for the engine's own}): a flag the calibration left to
    the engine is removed, so an older calibration's value never lingers.  Flags in `keep` stay as `args` has them."""
    out = list(args)
    for flag in KEYS:
        if flag in keep or flag not in chosen:
            continue
        v = chosen[flag]
        if flag in out[:-1]:
            i = out.index(flag)
            if v is None:
                del out[i:i + 2]
            else:
                out[i + 1] = str(v)
        elif v is not None:
            out += [flag, str(v)]
    return out


def long_prompt_ids(tok, tpl, n: int = LONG_TOKENS) -> list[int]:
    """CHANGELOG.md (cut to fit) as one user message: about n tokens of the chat template's prompt."""
    from serve.frontend import effort_kwargs
    text = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")[:n * 6]   # (~4 characters a token)
    ids: list[int] = []
    for _ in range(4):                                # (cut by characters, then once more by the tokens' count)
        ids = tok.encode(tpl.render([{"role": "user", "content": "Summarize this changelog:\n\n" + text}],
                                    **effort_kwargs("none")), parse_special=True)
        if len(ids) <= n:
            break
        text = text[:int(len(text) * n / len(ids) * 0.98)]
    return ids


def prompt_ids(cfg: dict) -> tuple[list[list[int]], list[int]]:
    """calibrate_glm.PROMPTS and the long prompt as the server sends them: the pack's tokenizer (with its own
    pre-tokenizer) and its chat template, thinking off (decode speed, not reasoning)."""
    import strata_tokenizer as ST
    from serve.frontend import ChatTemplate, effort_kwargs
    tpath = Path(cfg["tokenizer"])
    tok = ST.Tokenizer.from_dir(tpath)
    tpl = ChatTemplate(tpath / "chat_template.jinja")
    short = [tok.encode(tpl.render([{"role": "user", "content": p}], **effort_kwargs("none")), parse_special=True)
             for p in CAL.PROMPTS]
    return short, long_prompt_ids(tok, tpl)


def read_rate(eng, ids: list[int]) -> float:
    """tok/s of reading `ids` (one token answered; the part the conversation cache did not already hold)."""
    import threading
    for _ in eng.generate(ids, 1, {"temperature": 0}, threading.Event()):
        pass
    last = getattr(eng, "last", None) or {}
    ms, n = last.get("prompt_ms") or 0.0, (last.get("prompt_tokens") or len(ids)) - (last.get("reused") or 0)
    return n / (ms / 1000.0) if ms > 0 and n > 0 else 0.0


def profile_copy(args: list, tmp: Path):
    """A function that gives an engine start's `args` a fresh copy (in `tmp`) of the expert profile the config's
    engine reads - maya-ds4 saves it after every answer, and its next start places the experts in VRAM by it, so
    the tuning's answers must neither reach this PC's profile nor lead one candidate's placement into the next."""
    src = arg(args, "--expert-profile")
    if src == "none":
        return lambda a: a
    own = Path(src) if src else Path(arg(args, "--pack") or ".") / "expert-profile.bin"
    dst = tmp / "expert-profile.bin"

    def fresh(a: list) -> list:
        if own.is_file():
            shutil.copyfile(own, dst)
        elif dst.exists():
            dst.unlink()
        out = [x for i, x in enumerate(a) if x != "--expert-profile" and (i == 0 or a[i - 1] != "--expert-profile")]
        return out + ["--expert-profile", str(dst)]
    return fresh


def tp_group_widths(eng: object, cfg: dict) -> list[int] | None:
    """Effective tensor parallel width per GPU from the engine's VRAM protocol; None if incomplete."""
    rows = getattr(eng, "vram", None) or []
    n = len(cfg.get("gpu") or rows)
    if not rows or len(rows) != n:
        return None
    by_dev = {row.get("dev"): row.get("tp_n") for row in rows}
    if set(by_dev) != set(range(n)):
        return None
    widths = [by_dev[dev] for dev in range(n)]
    return widths if all(isinstance(w, int) and 1 <= w <= n for w in widths) else None


def start_measure(
    cfg: dict, args: list, short, long_ids, start_engine, fresh=None,
    seen_groups: dict[tuple[int, ...] | None, int] | None = None,
) -> dict:
    """One engine start; duplicate or unknown TP groups close before inference when comparing candidates."""
    eng = start_engine({**cfg, "args": fresh(args) if fresh else args})
    try:
        groups = tp_group_widths(eng, cfg)
        result = {"info": dict(getattr(eng, "info", {}) or {}), "tp_groups": groups}
        if seen_groups is not None:
            if groups is None or None in seen_groups:
                return {**result, "skipped": "effective TP grouping unknown: not compared"}
            previous = seen_groups.get(tuple(groups))
            if previous is not None:
                return {**result, "skipped": f"same effective grouping as --tp {previous}: not compared"}
        s = CAL.Session(eng, short)
        s.warm_up()
        decode = [s.rate() for _ in range(GROUPS)]
        return {**result, "decode": decode, "prompt": read_rate(eng, long_ids)}
    finally:
        CAL.close(eng)


def med(r: dict) -> float:
    return statistics.median(r["decode"]) if r.get("decode") else 0.0


def run(cfg: dict, vram: list[float], say=print, start_engine=None) -> dict:
    """Measure on the engine `cfg` describes, on GPUs with `vram` GB each (the config's, in its order); returns
    {"args": {flag: value or None}, "report": {...}} - None: the engine's own choice (the flag is left out).  Every start
    reads a fresh copy of the config's expert profile and writes only that (profile_copy).
    `start_engine(cfg)` returns a started engine (serve.server.StrataEngine, or a stand-in in tests)."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env, engine_args

        def start_engine(c):
            return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=child_env(c))
    base = {**cfg, "args": CAL.without_writes(list(cfg.get("args") or []))}
    short, long_ids = prompt_ids(cfg)
    with tempfile.TemporaryDirectory(prefix="maya-calibrate-ds4-") as tmp:
        return measure(base, vram, short, long_ids, start_engine, say, profile_copy(base["args"], Path(tmp)))


def measure(cfg: dict, vram: list[float], short, long_ids, start_engine, say=print, fresh=None) -> dict:
    t0 = time.time()
    args0 = list(cfg.get("args") or [])
    own_tp = int(arg(args0, "--tp") or 2)   # (the engine's default)
    own_threads = arg(args0, "--cpu-threads")
    tps = tp_candidates(vram)
    if own_tp not in tps:
        tps = [own_tp] + tps
    tps = [own_tp] + [t for t in tps if t != own_tp]   # (the config's own first: the baseline)
    with_ = lambda tp, w: apply(args0, {"--tp": tp, "--cpu-threads": w})   # noqa: E731
    failed: dict[str, str] = {}

    def attempt(what: str, args: list, seen_groups=None) -> dict | None:
        """start_measure, or None when the engine does not start or stops with `args` (a --tp the model's heads do not
        split in, too little memory for it, ...): that setting is left out, the others still measured."""
        try:
            return start_measure(cfg, args, short, long_ids, start_engine, fresh, seen_groups)
        except Exception as e:
            failed[what] = str(e)
            say(f"    {what}: the engine did not run ({e}): left out")
            return None
    say(f"  {len(tps)} tensor parallel setting(s) to try ({', '.join(f'--tp {t}' for t in tps)}), each a start of "
        "the model of its own ...")
    by_tp: dict[int, dict] = {}
    seen_groups: dict[tuple[int, ...] | None, int] = {}
    tp_groups: dict[str, list[int] | None] = {}
    tp_skipped: dict[str, str] = {}
    for tp in tps:
        say(f"    --tp {tp}: loading the model ...")
        if tp == own_tp:                              # (the baseline: the config's own must run)
            r = start_measure(cfg, with_(tp, own_threads), short, long_ids, start_engine, fresh)
        elif (r := attempt(f"--tp {tp}", with_(tp, own_threads), seen_groups)) is None:
            continue
        groups = r["tp_groups"]
        tp_groups[str(tp)] = groups
        widths = "/".join(map(str, groups)) if groups is not None else "unknown"
        say(f"    --tp {tp} requested, effective TP per GPU: {widths}")
        if r.get("skipped"):
            tp_skipped[str(tp)] = r["skipped"]
            say(f"    --tp {tp}: {r['skipped']}")
            continue
        seen_groups[tuple(groups) if groups is not None else None] = tp
        by_tp[tp] = r
        say(f"    --tp {tp}: {med(r):.1f} tok/s answering, {r['prompt']:.0f} tok/s reading a "
            f"{len(long_ids)}-token prompt")
    own = by_tp[own_tp]
    floor = PROMPT_FLOOR * own["prompt"]
    eligible = {t: r for t, r in by_tp.items() if r["prompt"] >= floor or t == own_tp}
    for t in by_tp:
        if t not in eligible:
            say(f"    (--tp {t} reads long prompts at {by_tp[t]['prompt']:.0f} tok/s, below {PROMPT_FLOOR:.0%} of "
                f"--tp {own_tp}'s: not taken)")
    best = max(eligible, key=lambda t: med(eligible[t]))
    confirm = {}
    if best != own_tp and med(own) * (1 + MIN_GAIN) < med(by_tp[best]) < med(own) * (1 + CONFIRM_BELOW):
        say(f"    confirming --tp {best} against --tp {own_tp} (one more start each) ...")
        for t in (own_tp, best):
            if (again := attempt(f"--tp {t} again", with_(t, own_threads))) is None:
                best = own_tp                         # (a setting that did not run twice is not kept)
                break
            by_tp[t] = {**by_tp[t], "decode": by_tp[t]["decode"] + again["decode"]}
            confirm[t] = again["decode"]
    chosen_tp = best if med(by_tp[best]) > med(by_tp[own_tp]) * (1 + MIN_GAIN) else own_tp
    # the CPU threads, at that --tp: fewer than the engine's own, while they help
    info = by_tp[chosen_tp].get("info") or {}
    # --cpu-threads is a ceiling; the engine may have fewer cores or choose a smaller lane.
    d_threads = int(info.get("cpu_threads") or own_threads or 0)
    chosen_w, by_w = own_threads, {}
    # (the thread test runs whenever the lane has threads: a reported 0 CPU-lane experts skipped it on a PC whose lane
    #  took ~60% of every token's experts, #433 - a lane that holds none simply gains nothing and stops at the first)
    if d_threads > 0:
        if int(info.get("cpu_experts") or 0) <= 0:
            say(f"    the engine reports no experts on the CPU lane ({info.get('cpu_experts')!r}): its {d_threads} "
                "threads are tested anyway")
        best_rate = med(by_tp[chosen_tp])
        for w in thread_candidates(d_threads, every_cpu=d_threads >= usable_cpus() - 1):
            say(f"    --tp {chosen_tp}, {w} CPU threads (the engine's own: {d_threads}): loading the model ...")
            if (r := attempt(f"{w} CPU threads", with_(chosen_tp, w))) is None:
                break
            by_w[w] = r
            say(f"    {w} CPU threads: {med(r):.1f} tok/s answering")
            if med(r) <= best_rate * (1 + MIN_GAIN):
                break                                 # (fewer only wait longer for the experts from there)
            chosen_w, best_rate = w, med(r)
    else:
        say(f"    the CPU threads are not tested: the engine reports none ({info.get('cpu_threads')!r}) and the config "
            "sets none")
    chosen = {"--tp": chosen_tp, "--cpu-threads": None if chosen_w is None else int(chosen_w)}
    final = by_w.get(chosen_w) or by_tp[chosen_tp]
    report = {"own": {"--tp": own_tp, "--cpu-threads": d_threads or None},
              "tp_sweep": {str(t): {"decode": [round(x, 1) for x in r["decode"]], "prompt": round(r["prompt"])}
                           for t, r in by_tp.items()},
              "thread_sweep": {str(w): [round(x, 1) for x in r["decode"]] for w, r in by_w.items()},
              "tp_groups": tp_groups, "tp_skipped": tp_skipped, "effective_tp": final["tp_groups"],
              "confirm": {str(t): [round(x, 1) for x in v] for t, v in confirm.items()}, "failed": failed,
              "tok_s": round(med(final), 1), "own_tok_s": round(med(by_tp[own_tp]), 1),
              "prompt_tok_s": round(final.get("prompt") or by_tp[chosen_tp]["prompt"]),
              "long_prompt_tokens": len(long_ids), "seconds": round(time.time() - t0)}
    return {"args": chosen, "report": report}


if __name__ == "__main__":
    import json
    if len(sys.argv) != 2:
        sys.exit("usage: calibrate_ds4_tp.py <maya-ds4.json>")
    import setup as S
    c = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    found = {g["index"]: g for g in S.gpus() or []}
    print(json.dumps(run(c, [found[i]["vram_gb"] for i in c.get("gpu") or [] if i in found]), indent=1))

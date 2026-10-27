"""What runs next with several conversations (docs/DS4_SCHEDULING.md's tables): four requests of 16,000 / 4,000 / 1,000 /
250 tokens on `--parallel 4`, each run a fresh engine, and per request when its first and its last token came from the
first send.

    python tools/ds4/schedule_bench.py --pack <pack> [--exe build/maya-ds4] [--gpus 0,2,1] [--config maya-ds4.json]

  order     sent at once, longest first: MAYA_DS4_READ_ORDER=arrival and =ratio against the default (a piece each)
  during    the 16,000-token one first, the others 1.5 s later (they arrive while it is read): the default
  answers   at once, the 250-token one asking for a 200-token answer, read by ratio: MAYA_DS4_PREFILL_GAPS=50 against
            the default gaps (shortest remaining work first)

Each run's slots go to a scratch MAYA_DS4_SLOT_DIR (MAYA_DS4_SLOT_KEEP=0).  Exit 0: every run answered every request.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))
from parallel_test import Multi  # noqa: E402
from slot_keep_test import config_env, text_tokens  # noqa: E402

LENS = [16000, 4000, 1000, 250]
NEW = 32


def run(a, prompts, settings: dict, stagger: float, news: list[int]) -> list[tuple]:
    env = dict(settings)
    tmp = tempfile.mkdtemp(prefix="ds4-schedule-")
    env.update({"MAYA_DS4_SLOT_DIR": tmp, "MAYA_DS4_SLOT_KEEP": "0"})
    args = [*(("--gpus", a.gpus) if a.gpus else ()), "--pack", a.pack, "--max-context", str(a.ctx), "--dspark", "on",
            "--threads", "on", "--parallel", "4", "--vision", "off", "--expert-profile", "none", "--serve"]
    eng = Multi(a.exe, args, env)
    eng.start(timeout=3600.0)
    try:
        t0 = time.monotonic()
        rids = [eng.send(prompts[0], news[0], 0)]
        if stagger:
            time.sleep(stagger)
        rids += [eng.send(p, k, 0) for p, k in zip(prompts[1:], news[1:])]
        got = [eng.stream(r) for r in rids]
    finally:
        try:
            eng.quit(timeout=120)
        except Exception:  # noqa: BLE001 - (an engine that will not end: ended)
            eng.proc.kill()
        shutil.rmtree(tmp, ignore_errors=True)
    rows = []
    for n, r in zip(LENS, got):
        if r["error"] or not r["times"]:
            rows.append((n, None, None, r["error"] or "no tokens"))
        else:
            rows.append((n, r["times"][0] - t0, r["times"][-1] - t0, None))
    return rows


def table(title: str, cols: list[tuple[str, list[tuple]]]) -> bool:
    print(f"\n{title}  (seconds from the first send: first token / done)", flush=True)
    print("  prompt  " + "".join(f"{name:>26}" for name, _ in cols), flush=True)
    ok = True
    for i, n in enumerate(LENS):
        cells = []
        for _, rows in cols:
            r = rows[i]
            if r[1] is None:
                ok = False
                cells.append(f"{'ERR ' + str(r[3])[:20]:>26}")
            else:
                cells.append(f"{r[1]:12.1f} / {r[2]:11.1f}")
        print(f"  {n:>6}  " + "".join(cells), flush=True)
    means = []
    for _, rows in cols:
        good = [r for r in rows if r[1] is not None]
        means.append(f"{sum(r[1] for r in good) / max(1, len(good)):12.1f} / {sum(r[2] for r in good) / max(1, len(good)):11.1f}"
                     if good else f"{'-':>26}")
    print("  mean    " + "".join(means), flush=True)
    print("  all     " + "".join(f"{max((r[2] for r in rows if r[2] is not None), default=0):26.1f}" for _, rows in cols),
          flush=True)
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description="what runs next with several conversations (DS4_SCHEDULING.md)")
    ap.add_argument("--pack", required=True, help="the model's pack dir (maya-ds4)")
    ap.add_argument("--exe", default=str(Path(__file__).resolve().parents[2] / "build/maya-ds4"),
                    help="the engine (default: <repo>/build/maya-ds4)")
    ap.add_argument("--gpus", default="", help="the GPUs (default: every visible one, the engine's own split)")
    ap.add_argument("--config", default="", help="a model config (maya-ds4.json): its DS4_/MAYA_DS4_ settings")
    ap.add_argument("--ctx", type=int, default=131072, help="--max-context")
    a = ap.parse_args()
    try:
        from strata_tokenizer import Tokenizer
        tok = Tokenizer.from_dir(os.path.join(a.pack, "tokenizer"))
    except Exception as e:  # noqa: BLE001
        print(f"{os.path.join(a.pack, 'tokenizer')}: no tokenizer (the prompts are real text): {e}", flush=True)
        return 2
    base = {k: v for k, v in (config_env(a.config) if a.config else {}).items()
            if k not in ("MAYA_DS4_PREFILL_GAPS", "MAYA_DS4_READ_ORDER")}
    prompts = [text_tokens(tok, f"Schedule {i}.", n) for i, n in enumerate(LENS)]
    arrival = dict(base, MAYA_DS4_READ_ORDER="arrival")
    ratio = dict(base, MAYA_DS4_READ_ORDER="ratio")
    fixed = dict(ratio, MAYA_DS4_PREFILL_GAPS="50")
    long_answer = [NEW, NEW, NEW, 200]
    ok = table("order: at once, longest first", [
        ("first come, first served", run(a, prompts, arrival, 0.0, [NEW] * 4)),
        ("highest response ratio", run(a, prompts, ratio, 0.0, [NEW] * 4)),
        ("default", run(a, prompts, base, 0.0, [NEW] * 4)),
    ])
    ok = table("during: the 16,000-token one first, the rest 1.5 s later", [
        ("default", run(a, prompts, base, 1.5, [NEW] * 4)),
    ]) and ok
    ok = table("answers: at once, the 250-token one's answer 200 tokens", [
        ("gaps fixed at 50%", run(a, prompts, fixed, 0.0, long_answer)),
        ("default gaps", run(a, prompts, ratio, 0.0, long_answer)),
    ]) and ok
    print("\nevery run answered every request" if ok else "\na run failed (the reason is in its row)", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

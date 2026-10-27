"""Tune maya-mm3 for this PC: --tp (the layer split, or tensor parallel in groups of 1, 2 or 4 equal GPUs) and the CPU
lane's --cpu-threads, measured with whole engine starts - tools/calibrate_ds4_tp.py's sweep (its prompts, warm-up,
long-prompt floor, MIN_GAIN and confirmation start) on a maya-mm3.json config.

What differs from DeepSeek's:
  - maya-mm3's own default is the layer split (--tp 0: no flag), so that is the baseline every other setting must beat;
  - the tensor-parallel groups split MiniMax-M3.1's 4 KV heads, so --tp 1, 2 or 4 (and only up to the longest run of
    equal GPUs); one GPU: nothing to choose there;
  - on Windows the tensor-parallel path is not tried (the engine refuses it there: its host waits and the WDDM
    driver's batched submission).

--dspark: whether the DSpark drafter pays on this PC (tools/calibrate_ds4.py's measure: the config without the drafter
- --dspark none - and with it, the same questions; kept only when it wins by more than its MIN_GAIN); maya-mm3 takes
--dspark none / off / on and --expert-profile as maya-ds4 does, so the measure is DeepSeek's as it is.

The result goes into the config's args (the caller - ./maya.sh --calibrate - writes it); the engine's own choice is
left as no flag.

    python tools/mm3/calibrate.py maya-mm3.json            # --tp and --cpu-threads: measure and print
    python tools/mm3/calibrate.py --dspark maya-mm3.json   # the DSpark drafter or none
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_ds4_tp as C  # noqa: E402  the sweep, the measuring, apply()

MIN_GAIN = C.MIN_GAIN


def tp_candidates(vram: list[float]) -> list[int]:
    """--tp values for these GPUs (their VRAM, in the engine's order): 0 (the layer split: the baseline) first, then 1, 2
    and 4 up to the longest run of equal GPUs - none but 0 for one GPU or on Windows."""
    out = [0]
    if len(vram) < 2 or sys.platform == "win32":
        return out
    runs, i = [], 0
    while i < len(vram):
        j = i + 1
        while j < len(vram) and vram[j] == vram[i]:
            j += 1
        runs.append(j - i)
        i = j
    longest = max(runs)
    out += [t for t in (1, 2, 4) if t <= longest and (t > 1 or len(vram) > 1)]
    return out


def run(cfg: dict, vram: list[float], say=print, start_engine=None) -> dict:
    """calibrate_ds4_tp.run with maya-mm3's candidates and baseline; {"args": {flag: value or None}, "report": ...}."""
    args = list(cfg.get("args") or [])
    if C.arg(args, "--tp") is None:
        args += ["--tp", "0"]           # (the engine's default spelled out: the sweep's baseline)
    old = C.tp_candidates
    C.tp_candidates = tp_candidates
    try:
        out = C.run({**cfg, "args": args}, vram, say=say, start_engine=start_engine)
    finally:
        C.tp_candidates = old
    if out["args"].get("--tp") == 0:
        out["args"]["--tp"] = None      # (the layer split is the engine's own: no flag)
    return out


def dspark(cfg: dict, say=print, start_engine=None) -> dict:
    """calibrate_ds4.run on a maya-mm3 config: {"none": ..., "dspark": ..., "keep": "dspark" or "none", "gain": ...}."""
    import calibrate_ds4 as CDS
    return CDS.run(cfg, say=say, start_engine=start_engine)


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--dspark":
        print(json.dumps(dspark(json.loads(Path(sys.argv[2]).read_text(encoding="utf-8"))), indent=1))
        sys.exit(0)
    if len(sys.argv) != 2:
        sys.exit("usage: tools/mm3/calibrate.py [--dspark] <maya-mm3.json>")
    import setup as S
    c = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    found = {g["index"]: g for g in S.gpus() or []}
    print(json.dumps(run(c, [found[i]["vram_gb"] for i in c.get("gpu") or [] if i in found]), indent=1))

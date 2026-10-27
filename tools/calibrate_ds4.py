"""The speed calibration for DeepSeek-V4.1 (maya-ds4) and MiniMax-M3.1 (maya-mm3): does the DSpark drafter make THIS
PC faster?  (The setup, the first start of a config that has the drafter and no calibration yet, ./maya.sh
--speed-calibration.)

DSpark drafts up to 5 tokens a round and the model checks them in one window.  Where the model's experts sit in the
GPUs' memory that window costs about what one token does and drafting wins; where most experts are computed on the CPU
or read from the SSD, the window's extra experts cost more than the drafts save (a Tesla V100 with 64 GB of RAM: slower
with it).  So it is measured here, as GLM-5.3-Flash's DFlash2 is (calibrate_draft.py: the same warm-up, questions and
sustained speed):

  - the config twice: without the drafter (--dspark none: its memory then holds more of the model's experts) and with
    it - one engine start each, nothing else different, on the config's GPUs (one or several);
  - first a warm-up that is not counted: a document is read and answered once;
  - then fresh questions about it, each answered at length, sampled as the dashboard does (the config's sampling and
    thinking level, a fixed seed per question - the same on both sides), until each side has written MIN_TOKENS
    tokens: its speed is all those tokens over all the time spent writing them;
  - the drafter is kept only when it wins by more than MIN_GAIN.

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
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
sys.path.insert(0, str(ROOT))
import calibrate_draft as CD  # noqa: E402  (requests, measure: the same warm-up and questions as GLM's)

MIN_GAIN = CD.MIN_GAIN
MIN_TOKENS = 1024          # tokens each side writes after its warm-up
DOC_TOKENS = 512           # the document's length (each question reads it again: kept short)


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


def run(cfg: dict, say=print, start_engine=None) -> dict:
    """Both sides measured, without the drafter first: {"none": {...}, "dspark": {...}, "keep": "dspark" or "none",
    "gain": the drafter's speed over the other's - 1}.  Raises when a side could not be measured."""
    if start_engine is None:
        from serve.server import StrataEngine, child_env, engine_args

        def start_engine(c):
            return StrataEngine(c["exe"], engine_args(c), cwd=c.get("cwd"), log=c.get("log"), env=child_env(c))
    saved = CD.MIN_TOKENS, CD.DOC_TOKENS
    CD.MIN_TOKENS, CD.DOC_TOKENS = MIN_TOKENS, DOC_TOKENS
    res: dict = {}
    try:
        warm, questions, n_doc = CD.requests(cfg)
        for i, (key, mode, label) in enumerate((("none", "none", "without drafting"), ("dspark", "on", "DSpark"))):
            with tempfile.TemporaryDirectory(prefix="maya-speed-") as tmp:   # (each side from the PC's own profile)
                c = profile_copy(with_dspark(cfg, mode), Path(tmp))
                res[key] = CD.measure(c, label, warm, questions, n_doc, start_engine, say, f"[{i + 1}/2] ")
    finally:
        CD.MIN_TOKENS, CD.DOC_TOKENS = saved
    for key, label in (("none", "without drafting"), ("dspark", "DSpark")):
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

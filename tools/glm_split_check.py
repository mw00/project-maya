#!/usr/bin/env python3
"""tools/glm_split_check.py - does the two-rig split (--role front|back) compute what one process computes?

GLM-5.3-Flash only (--glm-pack).  It runs the same build, pack and prompts twice - once in a single process (the
reference, on a machine that holds the whole model in VRAM + RAM) and once split over the link - and compares:

  greedy   the greedy answers' token ids, with the speculative (MTP) decode on and off.  Over the link the
           speculative decode is the pipelined one (decode_spec_link).  The two runs must agree until the first
           near-tie: where they first differ, the reference's own logits must put the two tokens within --tie nats.
  logits   teacher-forced logits (MAYA_GLM_KL_DUMP) over the prompt plus the reference's answer: the prompt goes
           through the prompt path, the answer one token at a time through the decode.  KL(ref || split) mean and
           top-1 agreement against --max-kl / --min-top1.

Two ways to run it (docs/TWO-RIG-SPLIT.md has the full commands):

  EXACT (one machine with two GPUs or more): the reference with --layer-split K runs GPU 0's layers [0, K) and the
  later GPUs' the rest; the split runs the front on GPU 0 and the back on the other GPU(s) over loopback - the same
  placement.  With the fp32 hop (MAYA_GLM_LINK_BF16=0) and the CPU lane off (MAYA_GLM_CPU_LANE=0) on every run, the
  split must be BIT-IDENTICAL to the reference: `compare --exact`.

  TWO MACHINES: the same steps across the network.  The placement differs from the reference's (more VRAM), and the
  default bf16 hop rounds the residual, so the results differ like two single processes with different settings
  do: greedy answers agree until a near-tie, and the logits stay within --max-kl / --min-top1.  How far a quant
  moves on its own varies (a low-bit quant more); measure it with two reference runs that differ in one setting.

Steps (DIR is a work dir; copy it between the machines with scp or rsync):

  1. the reference:       glm_split_check.py prompts --pack PACK --dir DIR
                          glm_split_check.py run --mode ref --dir DIR -- build/maya --glm-pack PACK <engine args> \\
                              --layer-split K
  2. the split (copy DIR to both machines first; start the back, then the front - it waits for the back):
     back:                glm_split_check.py run --mode back --dir DIR -- build/maya --glm-pack PACK <engine args> \\
                              --layer-split K --role back --link <back-ip>:7710
     front:               glm_split_check.py run --mode front --dir DIR -- build/maya --glm-pack PACK <engine args> \\
                              --layer-split K --role front --link <back-ip>:7710
     Both sides need the same MAYA_GLM_LINK_KEY_FILE off loopback.  On one machine, give each side its GPUs with
     CUDA_VISIBLE_DEVICES and use --link 127.0.0.1:<port>.
  3. bring the back's DIR/split/*.f32 next to the front's DIR/split/*.gen.json, then:
                          glm_split_check.py compare --dir DIR [--exact]

  Exit code of compare: 0 when every check passes, 1 otherwise.

The engine is `maya generate` (no server), one start per prompt and check: 4 starts a prompt for the reference
(greedy with and without MTP, then the logits along each answer) and 3 for the split.  --only NAME[,NAME] runs
some of the prompts.  The engine args go after `--`, the same on every run except --layer-split / --role / --link.
A step whose engine run fails writes no result, `run` exits 1, and `compare` fails on what is missing.

How far a model moves on its own (to set --max-kl / --min-top1 / --tie for a two-machine comparison): a second
reference run with one setting changed, `run --mode ref --name ref2 -- <engine args>` (for example with
MAYA_GLM_CPU_LANE=0), then `compare --against ref2`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path[:0] = [str(HERE), str(ROOT)]

PROMPTS = {
    "short": "A train leaves at 09:40 and travels 315 km at an average of 84 km/h. It then waits 25 minutes and "
             "continues 120 km at 96 km/h. At what time does it arrive? Show the steps briefly.",
    "summary": "Here is a short passage. The Industrial Revolution began in Britain in the late 18th century, driven "
               "by advances in textile machinery, the steam engine, and iron production. Factories concentrated "
               "labour in growing cities, while canals and later railways cut transport costs. Living conditions in "
               "early industrial towns were often poor, prompting reforms in public health, child labour and working "
               "hours during the 19th century. Summarize the passage in three bullet points, then name two "
               "inventions it implies and explain why each mattered.",
}
CODE_TOKENS = 2000          # the "code" prompt: about this many tokens of this repository's sources
MAX_NEW = 128


def code_prompt() -> str:
    """~CODE_TOKENS tokens of the repo's own C++/CUDA (the same text on every machine with the same checkout)."""
    src = sorted((ROOT / "src" / "core").glob("glm_*.cu"))
    text = "".join(p.read_text(errors="replace") for p in src)
    n = int(CODE_TOKENS * 3.3)
    off = len(text) // 3
    return "Here is some code:\n" + text[off:off + n] + "\n\nExplain what this code does, in detail."


def load_tokenizer(pack: Path):
    import strata_tokenizer as ST
    from serve.frontend import ChatTemplate, effort_kwargs
    tp = pack / "tokenizer"
    vocab = json.loads((tp / "vocab.json").read_text(encoding="utf-8"))
    toks = [None] * len(vocab)
    for t, i in vocab.items():
        toks[i] = t
    tok = ST.Tokenizer(toks, (tp / "merges.txt").read_text(encoding="utf-8").split("\n"),
                       json.loads((tp / "token_type.json").read_text()))
    return tok, ChatTemplate(tp / "chat_template.jinja"), effort_kwargs


def cmd_prompts(a) -> int:
    d = Path(a.dir)
    d.mkdir(parents=True, exist_ok=True)
    tok, tpl, effort_kwargs = load_tokenizer(Path(a.pack))
    prompts = dict(PROMPTS, code=code_prompt())
    meta = {}
    for name, text in prompts.items():
        ids = tok.encode(tpl.render([{"role": "user", "content": text}], **effort_kwargs("none")), parse_special=True)
        (d / f"{name}.prompt.tok").write_text(" ".join(map(str, ids)) + "\n")
        meta[name] = {"prompt_tokens": len(ids)}
        print(f"{name}: {len(ids)} prompt tokens")
    (d / "prompts.json").write_text(json.dumps({"prompts": prompts, "meta": meta, "max_new": a.max_new}, indent=1))
    return 0


def engine(argv: list[str], env_extra: dict, log: Path, timeout: float, env_drop: tuple = ()) -> tuple[int, str]:
    env = dict(os.environ, **env_extra)
    for k in env_drop:
        env.pop(k, None)
    t0 = time.time()
    with open(log, "wb") as lf:
        p = subprocess.run(argv, env=env, stdout=subprocess.PIPE, stderr=lf, stdin=subprocess.DEVNULL,
                           timeout=timeout)
    out = p.stdout.decode(errors="replace")
    print(f"  rc={p.returncode} {time.time() - t0:.0f} s  (log {log})", flush=True)
    return p.returncode, out


def parse_gen(out: str) -> list[int]:
    """The generated ids: the engine's `T <id>` lines on stdout."""
    ids = []
    for line in out.splitlines():
        f = line.split()
        if len(f) == 2 and f[0] == "T":
            ids.append(int(f[1]))
    return ids


def pick(meta: dict, only: str) -> list[str]:
    """The prompts to run or compare: all of DIR's, or --only's - which must name prompts that are there."""
    names = list(meta.get("prompts") or {})
    if not names:
        raise SystemExit("DIR/prompts.json has no prompts: run `prompts` first")
    if only is None:
        return names
    want = [n for n in only.split(",") if n]
    bad = [n for n in want if n not in names]
    if bad or not want:
        raise SystemExit(f"--only: no such prompt: {', '.join(bad) or '(none given)'} (there are: {', '.join(names)})")
    return want


DRAFTS = re.compile(r"glm spec: (\d+) of (\d+) drafts accepted")


def cmd_run(a) -> int:
    d = Path(a.dir)
    meta = json.loads((d / "prompts.json").read_text())
    names = pick(meta, a.only)
    exe = a.engine
    if not exe:
        print("run: give the engine command after --", file=sys.stderr)
        return 2
    if a.mode != "ref" and "--role" not in exe:
        exe = exe + ["--role", a.mode]
    if a.mode != "ref" and a.name:
        print("run: --name is for --mode ref (a second reference run)", file=sys.stderr)
        return 2
    # the reference writes DIR/ref; a second reference run (--name) and the split are compared with it
    sub = d / ((a.name or "ref") if a.mode == "ref" else "split")
    primary = a.mode == "ref" and sub.name == "ref"
    sub.mkdir(exist_ok=True)
    max_new = meta.get("max_new", MAX_NEW)
    failed = []
    for name in names:
        prompt = d / f"{name}.prompt.tok"
        # 1. greedy, speculative decode on, then off
        for spec in ("spec", "nospec"):
            if spec == "spec" and a.no_spec and a.mode != "ref":
                continue   # (a split that decodes plainly: its MTP run would be its plain run again)
            tag = f"{name}.{spec}"
            gen = sub / f"{tag}.gen.json"
            if a.mode != "back":   # (never a result of an earlier run; the back writes none, so it removes none -
                gen.unlink(missing_ok=True)   # on one machine the two roles share this directory)
            print(f"{a.mode} {tag}", flush=True)
            # (this run's drafting is this script's choice, whatever the shell has set)
            log = sub / f"{tag}.{a.mode}.log"
            rc, out = engine(exe + ["--tokens-file", str(prompt), "--max-new", str(max_new), "--greedy"],
                             {"MAYA_GLM_NO_SPEC": "1"} if spec == "nospec" else {}, log, a.timeout,
                             env_drop=("MAYA_GLM_NO_SPEC",) if spec == "spec" else ())
            ids = parse_gen(out)
            if rc != 0 or (a.mode != "back" and not ids):
                failed.append(tag)
                print(f"  FAILED (rc {rc}, {len(ids)} tokens) - no result written", flush=True)
                continue
            if spec == "spec" and a.mode == "front" and len(ids) >= 8:
                # over the link the MTP decode says how many drafts it made: none means this run decoded plainly
                # (the back has no NextN block, or this build's decode over the link is plain: run --no-spec).
                # (An answer of a few tokens can end before the first draft is counted: not judged)
                m = DRAFTS.findall(log.read_text(errors="replace"))
                if not m or int(m[-1][1]) == 0:
                    failed.append(tag)
                    print("  FAILED: no MTP drafts over the link in this run (see --no-spec) - no result written",
                          flush=True)
                    continue
                print(f"  {m[-1][0]} of {m[-1][1]} drafts accepted")
            if a.mode != "back":   # the front and a single process emit the tokens; the back only computes them
                gen.write_text(json.dumps(ids) + "\n")
                print(f"  {len(ids)} tokens")
        # 2. teacher-forced logits over the prompt + the reference's answer with MTP (the KL check); the reference
        #    itself also along its answer without MTP (so a divergence there is judged in the reference's own logits)
        for along in (("spec", "nospec") if primary else ("spec",)):
            stem = f"{name}.logits" if along == "spec" else f"{name}.nospec.logits"
            dump = sub / f"{stem}.f32"
            writes_dump = a.mode != "front"   # (the back and a single process have the logits; the front has none)
            if writes_dump:
                dump.unlink(missing_ok=True)
            ref_ans = d / "ref" / f"{name}.{along}.gen.json"
            if not ref_ans.exists():
                failed.append(stem)
                print(f"{a.mode} {stem}: no {ref_ans} (the reference's run of this prompt failed, or DIR/ref was not "
                      f"copied here) - FAILED", flush=True)
                continue
            ans = json.loads(ref_ans.read_text())
            ptoks = prompt.read_text().split()
            # (a file per role: on one machine the front and the back run from the same directory at the same time)
            tf = sub / f"{stem}.{a.mode}.tok"
            tf.write_text(" ".join(ptoks + [str(t) for t in ans]) + "\n")
            n = len(ans) + 1    # rows: the last prompt token .. the last answer token
            print(f"{a.mode} {stem} ({n} rows)", flush=True)
            rc, _ = engine(exe + ["--tokens-file", str(tf)], {"MAYA_GLM_KL_DUMP": f"{dump},{n}"},
                           sub / f"{stem}.{a.mode}.log", a.timeout)
            if rc != 0:
                failed.append(stem)
                if writes_dump:
                    dump.unlink(missing_ok=True)   # (not whole)
                print("  FAILED - no result kept", flush=True)
    if failed:
        print(f"run: {len(failed)} step(s) FAILED: {', '.join(failed)}", flush=True)
    return 1 if failed else 0


def logsoftmax(x):
    import numpy as np
    m = x.max(1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(1, keepdims=True)))


def cmd_compare(a) -> int:
    import numpy as np
    d = Path(a.dir)
    other = d / a.against
    meta = json.loads((d / "prompts.json").read_text())
    names = pick(meta, a.only)
    ok_all = True
    report = []

    def tokens(f):   # a greedy answer: a non-empty list of ids, or None
        try:
            t = json.loads(f.read_text())
        except (OSError, ValueError):
            return None
        return t if isinstance(t, list) and t and all(isinstance(x, int) for x in t) else None

    def logits(f, rows):   # rows x vocab raw logits, or None when the file is not that
        try:
            x = np.fromfile(f, dtype=np.float32)
        except OSError:
            return None
        return x.reshape(rows, -1) if rows > 0 and x.size > 0 and x.size % rows == 0 else None

    for name in names:
        ref_ans = {m: tokens(d / "ref" / f"{name}.{m}.gen.json") for m in ("spec", "nospec")}
        ref_raw = {"spec": None, "nospec": None}
        for m, stem in (("spec", f"{name}.logits"), ("nospec", f"{name}.nospec.logits")):
            if ref_ans[m] is not None:
                ref_raw[m] = logits(d / "ref" / f"{stem}.f32", len(ref_ans[m]) + 1)
        # greedy
        for spec in ("spec", "nospec"):
            if spec == "spec" and a.no_spec:
                report.append(f"{name} spec: not compared (--no-spec)")
                continue
            rt, st = ref_ans[spec], tokens(other / f"{name}.{spec}.gen.json")
            if rt is None or st is None:
                report.append(f"{name} {spec}: no greedy answer from the {'reference' if rt is None else a.against} "
                              f"(its run failed or was not made) - FAIL")
                ok_all = False
                continue
            i = next((k for k in range(min(len(rt), len(st))) if rt[k] != st[k]), None)
            if i is None and len(rt) == len(st):
                report.append(f"{name} {spec}: greedy IDENTICAL ({len(rt)} tokens)")
                continue
            if a.exact:
                report.append(f"{name} {spec}: greedy differs at token {i if i is not None else min(len(rt), len(st))} "
                              f"- FAIL (--exact)")
                ok_all = False
                continue
            if i is None:
                report.append(f"{name} {spec}: same {min(len(rt), len(st))} tokens, then one ends (lengths {len(rt)} / "
                              f"{len(st)}) - FAIL")
                ok_all = False
                continue
            # the first difference: a near-tie in the reference's own logits along this answer (row i predicts the
            # answer's token i)
            r_raw = ref_raw[spec]
            if (r_raw is not None and i < r_raw.shape[0] and 0 <= min(rt[i], st[i])
                    and max(rt[i], st[i]) < r_raw.shape[1]):
                r_lp = logsoftmax(r_raw[i:i + 1].astype(np.float64))[0]
                # (the reference's own row may favour either token: its answer was picked by another pass than the
                # teacher-forced one, so the gap is taken without its sign)
                gap = abs(float(r_lp[rt[i]] - r_lp[st[i]]))
                tie = gap <= a.tie
                report.append(f"{name} {spec}: same {i} of {len(rt)} tokens, then {rt[i]} vs {st[i]}: the reference "
                              f"put them {gap:.3f} nats apart - "
                              f"{'a near-tie, OK' if tie else f'not a near-tie (--tie {a.tie:g}): FAIL'}")
                ok_all &= tie
            else:
                report.append(f"{name} {spec}: same {i} of {len(rt)} tokens, then {rt[i]} vs {st[i]} (no reference "
                              f"logits for that position: FAIL)")
                ok_all = False
        # logits
        r_raw = ref_raw["spec"]
        s_raw = None if r_raw is None else logits(other / f"{name}.logits.f32", r_raw.shape[0])
        if r_raw is None or s_raw is None or s_raw.shape != r_raw.shape:
            report.append(f"{name} logits: no usable logits from the {'reference' if r_raw is None else a.against} "
                          f"(missing, or not the reference's rows x vocabulary) - FAIL")
            ok_all = False
            continue
        ref_lp, x = logsoftmax(r_raw.astype(np.float64)), logsoftmax(s_raw.astype(np.float64))
        kl = (np.exp(ref_lp) * (ref_lp - x)).sum(1)
        t1 = float((ref_lp.argmax(1) == x.argmax(1)).mean() * 100)
        exact = bool((s_raw == r_raw).all())
        good = exact if a.exact else (kl.mean() <= a.max_kl and t1 >= a.min_top1)
        report.append(f"{name} logits: {'BIT-IDENTICAL, ' if exact else ''}KL mean {kl.mean():.3e} p99 "
                      f"{np.percentile(kl, 99):.3e} max {kl.max():.3e}, top-1 {t1:.1f}% - "
                      f"{'OK' if good else 'FAIL'} " +
                      ("(--exact: bit-identical needed)" if a.exact
                       else f"(--max-kl {a.max_kl:g}, --min-top1 {a.min_top1:g})"))
        ok_all &= bool(good)
    print("\n".join(report))
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("prompts", help="write the prompts' token files into DIR")
    p.add_argument("--pack", required=True)
    p.add_argument("--dir", required=True)
    p.add_argument("--max-new", type=int, default=MAX_NEW)
    r = sp.add_parser("run", help="run the engine on DIR's prompts (ref, front or back)")
    r.add_argument("--mode", choices=("ref", "front", "back"), required=True)
    r.add_argument("--dir", required=True)
    r.add_argument("--only", default=None, metavar="NAME[,NAME]", help="only these prompts")
    r.add_argument("--timeout", type=float, default=3600)
    r.add_argument("--no-spec", action="store_true",
                   help="front/back: skip the greedy run with MTP (a build whose decode over the link is plain)")
    r.add_argument("--name", default="",
                   help="--mode ref: write DIR/NAME instead of DIR/ref - a second single-process run (with one setting "
                        "changed) to compare with the first: `compare --against NAME` shows how far this model moves "
                        "on its own")
    r.add_argument("engine", nargs=argparse.REMAINDER)
    c = sp.add_parser("compare", help="compare DIR/ref with DIR/split (or DIR/NAME)")
    c.add_argument("--dir", required=True)
    c.add_argument("--only", default=None, metavar="NAME[,NAME]", help="only these prompts")
    c.add_argument("--against", default="split", metavar="NAME",
                   help="what to compare the reference with: split (the default), or a second reference run's --name")
    c.add_argument("--no-spec", action="store_true",
                   help="do not compare the greedy answers with MTP (the split side ran with --no-spec)")
    c.add_argument("--max-kl", type=float, default=2.5e-2, help="mean KL(reference || other) allowed, nats")
    c.add_argument("--min-top1", type=float, default=93.0, help="top-1 agreement needed, percent")
    c.add_argument("--tie", type=float, default=0.5,
                   help="where the greedy answers first differ: the gap allowed between the two tokens in the "
                        "reference's logits, nats")
    c.add_argument("--exact", action="store_true",
                   help="require bit-identical logits and greedy answers (the exact check: same machine, same placement, "
                        "fp32 hop, CPU lane off)")
    a = ap.parse_args()
    if a.cmd == "run" and a.engine[:1] == ["--"]:
        a.engine = a.engine[1:]
    return {"prompts": cmd_prompts, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

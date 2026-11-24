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

Several conversations at once (--parallel over the link): `run --parallel N` with each mode, then
`compare --parallel N [--exact]`.  It runs that step alone: the engine as a server (--serve --parallel N, started by
this script, no HTTP) and N conversations over DIR's prompts in three turns - greedy, one alone first (a request
alone drafts with MTP) and the others while it writes; a second turn of each (the prompt, the reference's answer
and a tail: the state kept from the first turn is continued); then other prompts, half of them sampled with a seed
and half greedy with the first token of the greedy answer banned (logit_bias).  The reference is one process with
--parallel N; the split must give the same tokens for every conversation (--exact), and both sides need the same
MAYA_GLM_SPLIT_ROWS.

How far a model moves on its own (to set --max-kl / --min-top1 / --tie for a two-machine comparison): a second
reference run with one setting changed, `run --mode ref --name ref2 -- <engine args>` (for example with
MAYA_GLM_CPU_LANE=0), then `compare --against ref2`.
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import random
import re
import subprocess
import sys
import threading
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


SYSTEM_TOKENS = 1500        # the "system" prompt: a system prompt of about this many tokens, then a user turn
SYSTEM_ASK = ("Go through the code in your instructions and list the functions it defines. For each one say in a "
              "sentence what it takes, what it returns and what it is for, and name the two you would read first to "
              "understand the rest. Then say in a few sentences how the pieces fit together, which data they share, "
              "and what you would check before changing any of them. Keep to what the code shows; where the excerpt "
              "starts or ends in the middle of a function, say so instead of guessing.")


def system_prompt() -> list:
    """A long system prompt (this repository's sources again, another part of them) and a user turn after it: the
    engine ends a prompt-path call at the system prompt's end (a conversation checkpoint, or the slots' prefix cut),
    so this prompt is read in two calls - the split must cut it where one process does."""
    src = sorted((ROOT / "src" / "core").glob("glm_*.cu"))
    text = "".join(p.read_text(errors="replace") for p in src)
    n = int(SYSTEM_TOKENS * 3.3)
    off = 2 * len(text) // 3
    return [{"role": "system", "content": "You answer questions about this code:\n" + text[off:off + n]},
            {"role": "user", "content": SYSTEM_ASK}]


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
    prompts = dict(PROMPTS, code=code_prompt(), system=system_prompt())
    meta = {}
    for name, text in prompts.items():   # (a text is one user turn; "system" is a list of messages)
        msgs = text if isinstance(text, list) else [{"role": "user", "content": text}]
        ids = tok.encode(tpl.render(msgs, **effort_kwargs("none")), parse_special=True)
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


def par_run(a, d: Path, names: list[str], exe: list[str], sub: Path, max_new: int) -> int:
    """`run --parallel N`: N conversations at once through `maya generate --serve --parallel N` (see the top)."""
    n = a.parallel
    out_f = sub / f"par{n}.gen.json"
    ref_f = d / "ref" / f"par{n}.gen.json"
    writes = a.mode != "back"   # (the front and a single process emit the tokens)
    back_ok = sub / f"par{n}.back.ok"   # the back's run ended in order: `compare` asks for it beside the front's result
    if writes:
        out_f.unlink(missing_ok=True)
    else:
        back_ok.unlink(missing_ok=True)
    primary = a.mode == "ref" and sub.name == "ref"
    ref = None
    if not primary:   # the second turn continues the reference's answers, so every run reads the same prompts
        try:
            ref = json.loads(ref_f.read_text())
            assert ref["n"] == n and len(ref["rounds"]) == 3 and len(ref["rounds"][0]) == n
        except (OSError, ValueError, KeyError, AssertionError, IndexError, TypeError):
            print(f"{a.mode} par{n}: no usable {ref_f} (the reference's `run --parallel {n}` first, and DIR/ref copied "
                  f"here) - FAILED", flush=True)
            return 1
    prompts = [[int(t) for t in (d / f"{names[i % len(names)]}.prompt.tok").read_text().split()] for i in range(n)]
    log = sub / f"par{n}.{a.mode}.log"
    env = dict(os.environ)
    if a.mode == "back":
        env["MAYA_GLM_BACK_STAY"] = "0"   # (this back ends with its front: the script waits for it)
    print(f"{a.mode} par{n}: {n} conversations at once, 3 turns each", flush=True)
    t0 = time.time()
    # (the back writes its own T / PP / DONE lines: to a file - a pipe nobody reads would fill and stop it)
    with open(log, "wb") as lf, open(sub / f"par{n}.back.out", "wb") if a.mode == "back" else open(os.devnull, "wb") as bo:
        p = subprocess.Popen(exe + ["--serve", "--parallel", str(n)], env=env, stderr=lf,
                             stdout=bo if a.mode == "back" else subprocess.PIPE,
                             stdin=subprocess.DEVNULL if a.mode == "back" else subprocess.PIPE)
        if a.mode == "back":   # it reads its front's lines from the link and ends with it
            try:
                rc = p.wait(timeout=a.timeout)
            except subprocess.TimeoutExpired:
                p.kill()
                rc = -9
            print(f"  rc={rc} {time.time() - t0:.0f} s  (log {log})", flush=True)
            if rc == 0:
                # what this back computed, by request id (its own T / DONE lines): the front's run numbers its
                # requests from a random base, so `compare` can tell this run's marker from an earlier run's
                seen: dict = {}
                ended = set()
                for ln in (sub / f"par{n}.back.out").read_text(errors="replace").splitlines():
                    f = ln.split()
                    if len(f) == 3 and f[0] == "T":
                        seen[f[1]] = seen.get(f[1], 0) + 1
                    elif len(f) > 6 and f[0] == "DONE":
                        ended.add(f[1])
                back_ok.write_text(json.dumps({"tokens": {r: c for r, c in seen.items() if r in ended}}) + "\n")
            return 0 if rc == 0 else 1
        lines: queue.Queue = queue.Queue()

        def pump():
            for raw in p.stdout:
                lines.put(raw.decode(errors="replace").rstrip("\n"))
            lines.put(None)

        threading.Thread(target=pump, daemon=True).start()
        toks: dict[int, list[int]] = {}
        done: dict[int, str] = {}
        order: list[int] = []   # the requests' ids in the order their tokens came
        why = ""

        def together(rids) -> bool:   # each of them wrote in turns with another: a token of another between two of its
            last: dict[int, int] = {}
            mixed = set()
            for k, r in enumerate(order):
                if r in last and any(x != r for x in order[last[r] + 1:k]):
                    mixed.add(r)
                last[r] = k
            return all(r in mixed for r in rids)

        base = 100 * random.randrange(10, 20000)   # this run's request ids start here (see the back's marker)

        def send(rid: int, ids: list[int], keys: str = "temperature=0"):
            toks[rid] = []
            p.stdin.write(f"GEN {max_new} rid={base + rid} {keys} {','.join(map(str, ids))}\n".encode())
            p.stdin.flush()

        def take(until) -> bool:   # read the engine's lines until until() holds; False: it failed or ended
            nonlocal why
            while not until():
                try:
                    ln = lines.get(timeout=max(1.0, a.timeout - (time.time() - t0)))
                except queue.Empty:
                    why = "timed out"
                    return False
                if ln is None:
                    why = "the engine ended"
                    return False
                f = ln.split()
                if not f:
                    continue
                if f[0] == "ERR":
                    why = ln
                    return False
                if f[0] == "T" and len(f) == 3:
                    toks[int(f[1]) - base].append(int(f[2]))
                    order.append(int(f[1]) - base)
                elif f[0] == "DONE" and len(f) > 6:
                    done[int(f[1]) - base] = f[6]   # (its finish: length / stop / cancel)
            return True

        ok = True
        # READY first (the model's load), then the turns
        while ok:
            try:
                ln = lines.get(timeout=max(1.0, a.timeout - (time.time() - t0)))
            except queue.Empty:
                ok, why = False, "timed out before READY"
                break
            if ln is None:
                ok, why = False, "the engine ended before READY"
            elif ln.startswith("READY"):
                if f"multi={n}" not in ln.split():
                    ok, why = False, f"the engine does not run {n} conversations: {ln}"
                break
        rounds = []
        if ok:
            # turn 1: conversation 0 alone until it has written 16 tokens, then the others at once
            send(0, prompts[0])
            ok = take(lambda: len(toks[0]) >= min(16, max_new) or 0 in done)
            if ok and 0 in done:   # (it must still write when the others come)
                ok, why = False, "the first conversation ended before the others started (a larger --max-new)"
            for i in range(1, n):
                if ok:
                    send(i, prompts[i])
            ok = ok and take(lambda: all(i in done for i in range(n)))
            if ok and not together(range(n)):
                ok, why = False, "turn 1: the conversations did not all decode together"
            if ok:
                rounds.append([{"prompt": names[i % len(names)], "tokens": toks[i], "finish": done[i]} for i in range(n)])
        if ok:
            # turn 2: every conversation again - its prompt, the reference's answer to it, a tail
            first = (ref["rounds"][0] if ref is not None else rounds[0])
            order.clear()
            for i in range(n):
                send(n + i, prompts[i] + list(first[i]["tokens"]) + prompts[i][-24:])
            ok = take(lambda: all(n + i in done for i in range(n)))
            if ok and not together(range(n, 2 * n)):
                ok, why = False, "turn 2: the conversations did not all decode together"
            if ok:
                rounds.append([{"prompt": names[i % len(names)], "tokens": toks[n + i], "finish": done[n + i]}
                               for i in range(n)])
        if ok:
            # turn 3: the next conversation's prompt, read anew - sampled with a seed (the back's sampler), or greedy
            # with the first token of that prompt's greedy answer banned (the request's logit_bias on the back)
            banned = {}
            order.clear()
            for i in range(n):
                j = (i + 1) % n
                keys = f"temperature=0.8 top_p=0.95 seed={100 + i}"
                if i % 2:
                    banned[i] = first[j]["tokens"][0]
                    keys = f"temperature=0 logit_bias={banned[i]}:-100"
                send(2 * n + i, prompts[j], keys)
            ok = take(lambda: all(2 * n + i in done for i in range(n)))
            if ok and not together(range(2 * n, 3 * n)):
                ok, why = False, "turn 3: the conversations did not all decode together"
            if ok and any(toks[2 * n + i][:1] == [t] for i, t in banned.items()):
                ok, why = False, "a banned token was written (logit_bias)"
            if ok:
                rounds.append([{"prompt": names[(i + 1) % n % len(names)], "tokens": toks[2 * n + i],
                                "finish": done[2 * n + i], "sampling": "banned" if i in banned else "seeded"}
                               for i in range(n)])
        try:
            p.stdin.write(b"QUIT\n")
            p.stdin.flush()
            p.stdin.close()
        except OSError:
            pass
        try:
            rc = p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            rc = -9
    print(f"  rc={rc} {time.time() - t0:.0f} s  (log {log})", flush=True)
    if ok and rc != 0:   # (it crashed after its last DONE, or did not end on QUIT: not a run to compare)
        ok, why = False, f"the engine ended with rc {rc}"
    if not ok or any(not c["tokens"] for r in rounds for c in r):
        print(f"  FAILED ({why or 'a conversation wrote nothing'}) - no result written", flush=True)
        return 1
    out_f.write_text(json.dumps({"n": n, "max_new": max_new, "rid_base": base, "rounds": rounds}) + "\n")
    print(f"  {sum(len(c['tokens']) for r in rounds for c in r)} tokens in {len(rounds)} turns of {n} conversations")
    return 0


def par_compare(a, d: Path, other: Path) -> int:
    """`compare --parallel N`: every conversation's tokens, the reference's against the split's."""
    n = a.parallel

    def load(f):
        try:
            x = json.loads(f.read_text())
            return x if x["n"] == n and len(x["rounds"]) == 3 and all(len(r) == n for r in x["rounds"]) else None
        except (OSError, ValueError, KeyError, TypeError):
            return None

    r, s = load(d / "ref" / f"par{n}.gen.json"), load(other / f"par{n}.gen.json")
    if r is None or s is None:
        print(f"par{n}: no result from the {'reference' if r is None else a.against} (its run failed or was not made) "
              f"- FAIL\nFAIL")
        return 1
    if a.against == "split":
        # (a back that crashed after its last token leaves the front's result whole: not a run to accept.  Its
        # marker must be this run's: the same request ids, each ended, with as many tokens as the front got)
        want = {str(s.get("rid_base", 0) + t * n + i): len(s["rounds"][t][i]["tokens"]) for t in range(3) for i in range(n)}
        try:
            got = json.loads((other / f"par{n}.back.ok").read_text())["tokens"]
        except (OSError, ValueError, KeyError, TypeError):
            got = None
        if got != want:
            print(f"par{n}: {other / f'par{n}.back.ok'} is missing or not this run's - the back did not end in order, "
                  f"or its DIR/split was not copied here after this run - FAIL\nFAIL")
            return 1
    ok_all = True
    for t in range(3):
        for i in range(n):
            rt, st = r["rounds"][t][i]["tokens"], s["rounds"][t][i]["tokens"]
            name = f"par{n} turn {t + 1} conversation {i} ({r['rounds'][t][i]['prompt']})"
            k = next((j for j in range(min(len(rt), len(st))) if rt[j] != st[j]), None)
            if k is None and len(rt) == len(st):
                print(f"{name}: IDENTICAL ({len(rt)} tokens)")
            elif a.exact:
                print(f"{name}: differs at token {k if k is not None else min(len(rt), len(st))} - FAIL (--exact)")
                ok_all = False
            else:
                # (no logits along a parallel answer: without --exact a difference is reported, not judged)
                print(f"{name}: same {k if k is not None else min(len(rt), len(st))} of {len(rt)} tokens, then "
                      f"different (not judged without --exact)")
    print("PASS" if ok_all else "FAIL")
    return 0 if ok_all else 1


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
    if a.parallel:
        return par_run(a, d, names, exe, sub, max_new)
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
    d = Path(a.dir)
    other = d / a.against
    meta = json.loads((d / "prompts.json").read_text())
    names = pick(meta, a.only)
    if a.parallel:   # (token lists only: no numpy needed)
        return par_compare(a, d, other)
    import numpy as np
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
    r.add_argument("--parallel", type=int, default=0, metavar="N",
                   help="run the several-conversations step alone: N conversations at once through --serve "
                        "--parallel N (2-8; the split needs MAYA_GLM_SPLIT_ROWS on both sides)")
    r.add_argument("engine", nargs=argparse.REMAINDER)
    c = sp.add_parser("compare", help="compare DIR/ref with DIR/split (or DIR/NAME)")
    c.add_argument("--dir", required=True)
    c.add_argument("--only", default=None, metavar="NAME[,NAME]", help="only these prompts")
    c.add_argument("--against", default="split", metavar="NAME",
                   help="what to compare the reference with: split (the default), or a second reference run's --name")
    c.add_argument("--no-spec", action="store_true",
                   help="do not compare the greedy answers with MTP (the split side ran with --no-spec)")
    c.add_argument("--parallel", type=int, default=0, metavar="N",
                   help="compare the several-conversations step (`run --parallel N`) alone")
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
    if a.cmd in ("run", "compare") and a.parallel and not 2 <= a.parallel <= 8:
        ap.error("--parallel takes 2 to 8 conversations")
    return {"prompts": cmd_prompts, "run": cmd_run, "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())

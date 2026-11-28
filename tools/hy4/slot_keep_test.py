#!/usr/bin/env python3
"""tools/hy4/slot_keep_test.py - Hy4-Flash's kept conversations (MAYA_HY4_SLOT_KEEP; DeepSeek's #356, ported): a prompt
that does not continue the conversation in the model takes the kept one it does continue back, so A, B, A's next turn
reads only the new tokens - and every answer is the one the conversation gives continued in place, with no switch.

    python tools/hy4/slot_keep_test.py --pack /mnt/data/models/Hy4-flash/maya-hy4 [--exe build/maya-hy4] [--gpus 0,1]
                                       [--tp 2]

tools/ds4/slot_keep_test.py's engine client and prompts (real text through the pack's tokenizer; greedy, spec=0 so
the state's position is the answer's), with HY4_HOT=0 (the experts in VRAM do not move between requests: a placement
moved by the answers before changes the last bits, switch or not) and HY4_ROUTE=lane (a start does not measure a
prompt route of its own).  A continued turn is compared with the same turn continued in place - the same new tokens
read in the same chunks - never with a read from token 0 (the chunks set a prompt's last bits).  Seven runs of
`maya-hy4 --serve`:
  in place  A, A's next turn, B, B's cut-back turn  the reference (MAYA_HY4_SLOT_KEEP=0): each turn continues its own
                                                    conversation in the caches
  in place  the same, a second process              the engine's run-to-run determinism (tensor parallel: a start's
                                                    plan can move the last bits - then the checks across processes
                                                    are reported, not claimed)
  switched  A, B, A's next turn, B's cut-back turn  each turn after the other conversation took the model, its own back
                                                    from a slot (B's at a position before its end: its rings' rows, a
                                                    history whose last answer was rendered again) - the in-place answers
  repeat    A, its turn, B, its turn, both again    one process (one plan, any layout): a turn sent again after the
                                                    switch, its conversation back from a slot, gives the same answer
  kept      A, B, then QUIT                         the folder holds both (2 .bin + 2 .meta)
  restart   A's next turn alone, a new process      A's conversation back from the folder
  evict     A, B, C into a folder capped at A+B     the least recently used (A) is gone, B and C stay

Exit 0: every check passed.  1: a check failed.  2: the engine did not start or answer, or no tokenizer.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import shutil
import struct
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.dirname(HERE)
sys.path.insert(0, TOOLS)
_spec = importlib.util.spec_from_file_location("ds4_slot_keep_test", os.path.join(TOOLS, "ds4", "slot_keep_test.py"))
ds4t = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ds4t)   # (its Engine client and text_tokens: the same protocol and prompts)

META_MAGIC = 0x3154454D34594855   # "UHY4MET1" (little endian): a slot's .meta, as src/hy4/hy4_main.cpp writes it


def meta_tokens(path: str) -> list[int] | None:
    """A slot's .meta: the magic, its identity (length, bytes), the .bin's bytes, its tokens (count, int32 each), its
    rings' rows, its pictures (count, (start, hash) each), the magic again.  None: not one, or damaged."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        m0, il = struct.unpack_from("<QQ", data, 0)
        if m0 != META_MAGIC:
            return None
        _, nt = struct.unpack_from("<QQ", data, 16 + il)
        toks = list(struct.unpack_from("<%di" % nt, data, 32 + il))
        off = 32 + il + 4 * nt
        _, np = struct.unpack_from("<QQ", data, off)
        off += 16 + 16 * np
        (m1,) = struct.unpack_from("<Q", data, off)
    except (OSError, struct.error):
        return None
    return toks if m1 == META_MAGIC and off + 8 == len(data) else None


ENGINE_ARGS: list[str] = []   # (--tp N: the layout under test - tensor parallel writes every member's copy)
ENGINE_ENV: dict[str, str] = {}
EXPECT: list[str] = []        # (lines the start must print: the layout under test was taken)


def run(exe, gpus, pack, ctx, slots, keep, steps, max_new, slot_gb=None, extra_env=None):
    env = {"MAYA_HY4_SLOT_KEEP": "1" if keep else "0", "MAYA_HY4_SLOT_DIR": slots, "HY4_HOT": "0",
           "HY4_ROUTE": "lane"}   # (the route pinned: a start does not measure its own)
    if slot_gb is not None:
        env["MAYA_HY4_SLOT_GB"] = str(slot_gb)
    env.update(ENGINE_ENV)
    env.update(extra_env or {})
    args = [*(("--gpus", ",".join(str(g) for g in gpus)) if gpus else ()), "--pack", pack, "--max-context", str(ctx),
            "--mtp", *ENGINE_ARGS, "--serve"]
    eng = ds4t.Engine(exe, args, env)
    t0 = time.monotonic()
    answers = {}
    try:
        eng.start(timeout=1800.0)
        load_s = time.monotonic() - t0
        for name, make in steps:
            prompt = make(answers)
            answers[name] = eng.request(prompt, max_new)
            if "error" in answers[name]:
                raise RuntimeError(f"{name}: {answers[name]['error']}")
            answers[name]["len"] = len(prompt)
    finally:
        if eng.proc is not None:
            eng.quit()
        err = eng.stderr_lines()
    said = "\n".join([*err, *eng.notes])
    for want in EXPECT:
        if want not in said:
            raise RuntimeError(f"the engine did not take the layout under test (no '{want}' at its start)")
    for line in err:   # (a slot that failed to be set aside, taken back or written: this test's failure)
        if line.startswith("hy4 slots:") and ("did not load" in line or "could not be written" in line
                                             or "not saved" in line):
            raise RuntimeError("the engine: " + line)
    return answers, eng.notes, load_s


def main() -> int:
    ap = argparse.ArgumentParser(description="Hy4-Flash's kept conversations (MAYA_HY4_SLOT_KEEP)")
    ap.add_argument("--pack", required=True, help="the model's pack dir (maya-hy4)")
    ap.add_argument("--exe", default=os.path.join(os.path.dirname(TOOLS), "build", "maya-hy4"))
    ap.add_argument("--gpus", default="")
    ap.add_argument("--ctx", type=int, default=32768)
    ap.add_argument("--len-a", type=int, default=12000)
    ap.add_argument("--len-b", type=int, default=9000)
    ap.add_argument("--len-c", type=int, default=4000)
    ap.add_argument("--len-turn", type=int, default=64)
    ap.add_argument("--max-new", type=int, default=24)
    ap.add_argument("--cut", type=int, default=8, help="B's cut-back turn: B's answer's first CUT tokens, then a turn")
    ap.add_argument("--tp", type=int, default=0, help="--tp N: tensor parallel over N GPUs (0: the engine's own layout)")
    ap.add_argument("--kv-resident", type=int, default=0,
                    help="KV streaming on with N positions of the indexed layers in VRAM (0: the start's own choice)")
    ap.add_argument("--kv-spill-res", type=int, default=0,
                    help="with --kv-resident: KV spill on, N positions in RAM (HY4_KV_SPILL_RES), the rest on a drive")
    a_ = ap.parse_args()
    if a_.tp > 1:
        ENGINE_ARGS.extend(["--tp", str(a_.tp)])
    spill_dir = None
    if a_.kv_resident > 0:
        ENGINE_ARGS.extend(["--kv-streaming", "on", "--kv-resident", str(a_.kv_resident)])
        EXPECT.append("kv_streaming=on")
        if a_.kv_spill_res > 0:
            spill_dir = tempfile.mkdtemp(prefix="hy4-spill-")
            ENGINE_ARGS.extend(["--kv-spill", "on", "--kv-spill-dir", spill_dir])
            ENGINE_ENV["HY4_KV_SPILL_RES"] = str(a_.kv_spill_res)
            EXPECT.append("KV spill: the indexed layers' rows")
    if not os.path.exists(a_.exe):
        print(f"{a_.exe}: no engine - build it first", flush=True)
        return 2
    try:
        from strata_tokenizer import Tokenizer
        tok = Tokenizer.from_dir(os.path.join(a_.pack, "tokenizer"))
    except (ImportError, OSError, ValueError) as e:
        print(f"no tokenizer: {e}", flush=True)
        return 2
    a = ds4t.text_tokens(tok, "Conversation A.", a_.len_a)
    b = ds4t.text_tokens(tok, "Conversation B.", a_.len_b)
    c = ds4t.text_tokens(tok, "Conversation C.", a_.len_c)
    turn = ds4t.text_tokens(tok, "The next turn:", a_.len_turn)
    turn2 = ds4t.text_tokens(tok, "Another turn:", a_.len_turn)
    gpus = [int(g) for g in a_.gpus.split(",") if g.strip()]
    ok = True

    def check(what, cond):
        nonlocal ok
        print(("  ok    " if cond else "  FAIL  ") + what, flush=True)
        ok = ok and cond

    tmp = tempfile.mkdtemp(prefix="hy4-slots-")
    try:
        # (the reference continues each conversation in place - no switch; the slot run continues it after the other
        #  conversation took the model and it came back from its slot: the same new tokens read in the same chunks
        #  from the same state, so a slot that is the state gives the same answer, token for token)
        def in_place(ref_a=None, ref_b=None):
            return [("A", lambda _: a), ("A-turn", lambda ans: a + (ref_a or ans["A"]["tokens"]) + turn),
                    ("B", lambda _: b), ("B-cut", lambda ans: b + (ref_b or ans["B"]["tokens"])[:a_.cut] + turn2)]

        print("in place: A, A's next turn, B, B's cut-back turn (MAYA_HY4_SLOT_KEEP=0: each turn continues its own "
              "conversation, no switch - the reference)", flush=True)
        r0, notes0, load0 = run(a_.exe, gpus, a_.pack, a_.ctx, os.path.join(tmp, "off-1"), False, in_place(), a_.max_new)
        plan0 = [n for n in notes0 if n.startswith("INFO CPU lane:")]

        def plan(name, notes):   # (a start that placed the experts otherwise: its answers can move in the last bits)
            p = [n for n in notes if n.startswith("INFO CPU lane:")]
            if p != plan0:
                print(f"        {name}: this start's plan is not the reference's: {p} against {plan0}", flush=True)
        ref_a, ref_b = r0["A"]["tokens"], r0["B"]["tokens"]
        print(f"        load {load0:.0f} s; A {len(a)} tok in {r0['A']['s']:.0f} s; A's next turn in "
              f"{r0['A-turn']['s']:.1f} s (reused {r0['A-turn']['reused']}); B's cut-back turn in {r0['B-cut']['s']:.1f} s "
              f"(reused {r0['B-cut']['reused']})", flush=True)
        check(f"in place: the turns continue their conversations ({r0['A-turn']['reused']} of "
              f"{len(a) + len(ref_a) - 1}, {r0['B-cut']['reused']} of {len(b) + a_.cut})",
              r0["A-turn"]["reused"] == len(a) + len(ref_a) - 1 and r0["B-cut"]["reused"] == len(b) + a_.cut)

        print("in place: the same, a second process (the engine's run-to-run determinism, the route pinned)", flush=True)
        r0b, notes0b, _ = run(a_.exe, gpus, a_.pack, a_.ctx, os.path.join(tmp, "off-2"), False, in_place(ref_a, ref_b),
                              a_.max_new)
        plan("in place, second process", notes0b)
        det = all(r0b[k]["tokens"] == r0[k]["tokens"] for k in r0)
        n_same = sum(r0b[k]["tokens"] == r0[k]["tokens"] for k in r0)
        moved = [n for n in notes0b if n.startswith("INFO CPU lane:")] != plan0
        if det or (a_.tp <= 1 and not moved):
            check(f"in place: two processes give the same answers ({n_same} of {len(r0)})", det)
        else:   # (a start's plan - its VRAM measured anew: tensor parallel's reserve, the experts the CPU lane leaves in
            #  VRAM - can move the last bits)
            why = "tensor parallel" if a_.tp > 1 else "the second start placed its experts otherwise"
            print(f"        {why}: two processes differ ({n_same} of {len(r0)} the same) - the checks across "
                  "processes below are reported, not claimed; the repeat run is the exactness check", flush=True)

        def diff(x, y):
            d = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
            return f"first difference at token {d}" if x != y else "identical"

        def same(name, got, key):
            what = f"{name}: {key}'s answer is the in-place one ({diff(got[key]['tokens'], r0[key]['tokens'])})"
            if det:
                check(what, got[key]["tokens"] == r0[key]["tokens"])
            else:
                print("        (not claimed) " + what, flush=True)

        print("switched: A, B, A's next turn, B's cut-back turn, one process (each turn after the other conversation "
              "took the model: its own back from a slot in RAM)", flush=True)
        r1, notes1, load1 = run(a_.exe, gpus, a_.pack, a_.ctx, os.path.join(tmp, "in-run"), True,
                                [("A", lambda _: a), ("B", lambda _: b), ("A-turn", lambda _: a + ref_a + turn),
                                 ("B-cut", lambda _: b + ref_b[:a_.cut] + turn2)], a_.max_new)
        t, cb = r1["A-turn"], r1["B-cut"]
        plan("switched", notes1)
        print(f"        load {load1:.0f} s; A's next turn {t['len']} tok in {t['s']:.1f} s (reused {t['reused']}); B's "
              f"cut-back turn in {cb['s']:.1f} s (reused {cb['reused']})", flush=True)
        for k in ("A", "B", "A-turn", "B-cut"):
            same("switched", r1, k)
        if r1["A"]["tokens"] == ref_a:   # (its own answer to A the reference's: the reference's turn continues it)
            check(f"switched: A's next turn reuses A's whole prefix ({t['reused']} of {len(a) + len(ref_a) - 1})",
                  t["reused"] == len(a) + len(ref_a) - 1)
        if r1["B"]["tokens"] == ref_b:
            check(f"switched: B's cut-back turn takes B back at its cut ({cb['reused']} of {len(b) + a_.cut})",
                  cb["reused"] == len(b) + a_.cut)
        for n in notes1:
            if n.startswith("sessions: "):
                print("        " + n, flush=True)

        # (within one process - one plan, whatever the layout: a turn answered, the other conversation in, the same turn
        #  sent again - its conversation back from its slot at the turn's last token, read alone as the first time: the
        #  same logits row, so the same answer, when the slot is the state)
        print("repeat: A, A's next turn, B, B's cut-back turn, then both turns again after the switch, one process",
              flush=True)
        P = {}
        rr, _, _ = run(a_.exe, gpus, a_.pack, a_.ctx, os.path.join(tmp, "repeat"), True,
                       [("A", lambda _: a), ("A-turn", lambda ans: P.setdefault("at", a + ans["A"]["tokens"] + turn)),
                        ("B", lambda _: b), ("B-cut", lambda ans: P.setdefault("bc", b + ans["B"]["tokens"][:a_.cut] + turn2)),
                        ("A-turn again", lambda _: P["at"]), ("B-cut again", lambda _: P["bc"])], a_.max_new)
        for k in ("A-turn", "B-cut"):
            x, y = rr[k], rr[k + " again"]
            check(f"repeat: {k} again, its conversation back from a slot ({y['reused']} of {y['len']} reused), gives the "
                  f"same answer ({diff(y['tokens'], x['tokens'])})", y["tokens"] == x["tokens"] and y["reused"] == y["len"] - 1)

        print("kept: A, B, then QUIT (both written to the folder)", flush=True)
        dir_kept = os.path.join(tmp, "kept")
        rk, notesk, _ = run(a_.exe, gpus, a_.pack, a_.ctx, dir_kept, True, [("A", lambda _: a), ("B", lambda _: b)],
                            a_.max_new)
        plan("kept", notesk)
        same("kept", rk, "A")   # (the restart below continues this process's A: the reference's only if this is)
        bins = [f for f in os.listdir(dir_kept) if f.startswith("slot-") and f.endswith(".bin")]
        metas = [f for f in os.listdir(dir_kept) if f.startswith("slot-") and f.endswith(".meta")]
        check(f"kept: the folder holds A and B ({len(bins)} .bin, {len(metas)} .meta)", len(bins) == 2 and len(metas) == 2)
        heads = sorted(tuple((meta_tokens(os.path.join(dir_kept, f)) or [])[:8]) for f in metas)
        check("kept: the .meta files read back as A's and B's tokens",
              heads == sorted([tuple(a[:8]), tuple(b[:8])]))
        sizes = {f: os.path.getsize(os.path.join(dir_kept, f)) for f in bins}
        # (A's kept tokens read now: the restart's own end sets A's longer conversation aside, which supersedes this one)
        kept_a = next((t for t in (meta_tokens(os.path.join(dir_kept, f)) for f in metas) if t and t[:8] == a[:8]), [])
        want3 = next((i for i, (x, y) in enumerate(zip(kept_a, a + ref_a + turn)) if x != y), len(kept_a))

        print("restart: A's next turn alone, a new process on the kept folder", flush=True)
        r3, notes3, load3 = run(a_.exe, gpus, a_.pack, a_.ctx, dir_kept, True,
                                [("A-turn", lambda _: a + ref_a + turn)], a_.max_new)
        t3 = r3["A-turn"]
        plan("restart", notes3)
        print(f"        load {load3:.0f} s; A's next turn in {t3['s']:.1f} s (reused {t3['reused']})", flush=True)
        check(f"restart: A's conversation back from the folder as far as the prompt continues it ({t3['reused']} of "
              f"{want3})", t3["reused"] == want3)
        if rk["A"]["tokens"] == ref_a:
            same("restart", r3, "A-turn")
        else:
            print("        (not comparable) restart: the kept A is not the reference's, so its next turn is read from "
                  "another point", flush=True)

        print("evict: A, B, C into a folder capped at A+B's bytes", flush=True)
        dir_ev = os.path.join(tmp, "evict")
        cap_gb = sum(sizes.values()) / 1e9
        run(a_.exe, gpus, a_.pack, a_.ctx, dir_ev, True, [("A", lambda _: a), ("B", lambda _: b), ("C", lambda _: c)],
            a_.max_new, slot_gb=cap_gb, extra_env={"MAYA_HY4_SLOT_CAP": "1"})
        held = sorted(tuple((meta_tokens(os.path.join(dir_ev, f)) or [])[:8]) for f in os.listdir(dir_ev)
                      if f.endswith(".meta"))
        total = sum(os.path.getsize(os.path.join(dir_ev, f)) for f in os.listdir(dir_ev) if f.endswith(".bin"))
        check(f"evict: A gone, B and C kept ({len(held)} conversations)", held == sorted([tuple(b[:8]), tuple(c[:8])]))
        check(f"evict: the folder under its cap ({total / 1e9:.2f} of {cap_gb:.2f} GB)", total <= cap_gb * 1e9)
    except RuntimeError as e:
        print(f"the engine: {e}", flush=True)
        return 2
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        if spill_dir:
            shutil.rmtree(spill_dir, ignore_errors=True)
    print("PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

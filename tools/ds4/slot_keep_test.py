#!/usr/bin/env python3
"""tools/ds4/slot_keep_test.py - the kept conversations (MAYA_DS4_SLOT_KEEP): a prompt that does not continue the
conversation in the model takes the kept one it does continue back, so A, B, A's next turn reads only the new tokens.

    python tools/ds4/slot_keep_test.py --pack /mnt/data/models/.../maya-ds4 [--exe build/maya-ds4] [--gpus 0,2,1]
                                       [--config maya-ds4.json]

The engine gets the environment this test inherits plus, with `--config <maya-ds4.json>`, that config's DeepSeek
settings (`DS4_*` and the `MAYA_DS4_*` names, under the names the engine reads): on a unified-memory APU the pool it
sizes from hipMemGetInfo is not the one a config pins (`DS4_POOL_GB`), and there a QUIT can hang for its 300 s.  The
slot settings this test sets win over the config's - the test decides what it measures.

Six runs of the engine's `--serve` protocol (greedy, spec=0 so the state's position is the answer's, exactly):
  no-keep  A, B, A's next turn (MAYA_DS4_SLOT_KEEP=0)  the reference: reused 0, and its answer to A builds the turn
                                                       every later run sends, so they all send the same prompt byte
                                                       for byte
  no-keep  the same again, a second process            two full reads of that one prompt: the engine's own run-to-run
                                                       determinism (a prompt's path is not promised to be identical
                                                       across runs, so this is measured, not assumed)
  in-run   A, B, A's next turn, one process            the third request reports A's full prefix as reused
  kept     A, B, then QUIT                             the folder keeps both conversations (2 .bin + .meta, under a
                                                       0.05 GB MAYA_DS4_SLOT_GB)
  restart  A's next turn alone, a new process          the same prefix reused, from the folder
  evict    A, B, C into a folder capped at A+B's bytes three conversations cannot fit: the least recently used (A) is
                                                       gone, B and C stay

The prompts are real text - this repo's tokenizer corpus, tokenized through the pack's tokenizer - not random ids:
random ids draw near-tie logits, where two greedy runs print different tokens and nothing can be compared.  spec=0 is
deliberate: with the drafter a request's state can end a few tokens past its last printed one (a draft accepted before
max_new cut it), and the next prompt - which has only the printed answer - would then not be a prefix of it; the slot
would not be taken and the count would say so.  The state itself is the same either way (the drafter's blocks are in
it).

The greedy check: when two full no-keep reads of the same prompt agree, the slot runs must print the reference's
tokens; when they do not, the engine's prompt path is not run-to-run deterministic, and the slot runs are compared
against both reads instead of claiming identity.  The reuse counts are only claimed for a run whose own answer to A
is the reference's (the prompt's answer is the reference's, so another answer is not continued past the common part).

Exit 0: every check passed.  1: a check failed (the reason is printed).  2: the engine did not start or
answer, or the pack's tokenizer is missing.
"""
from __future__ import annotations

import argparse
import json
import os
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import maya_names  # noqa: E402  (tools/) a model config's settings under the names the engine reads

CORPUS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tokenizer_corpus.txt")
META_MAGIC = 0x314154454d345344   # "DS4META1" (little endian): a slot's .meta, as src/ds4/ds4_main.cpp writes it
DS4_ENV = ("DS4_", "STRATA_DS4_")   # (the engine's own names: it aliases MAYA_DS4_* to STRATA_DS4_* at start)


def text_tokens(tok, salt: str, n: int) -> list[int]:
    """`n` tokens of real text (through the pack's tokenizer, tools/strata_tokenizer.py): the corpus's passages, each
    numbered under `salt` (so two prompts diverge at their first words), repeated until long enough and cut at `n`."""
    with open(CORPUS, encoding="utf-8", newline="") as f:   # (CRLF kept: the tokenizer's boundaries depend on it)
        text = f.read()
    out: list[int] = []
    i = 0
    while len(out) < n:
        out += tok.encode(f"{salt} {i}.\n{text}\n", parse_special=False)
        i += 1
    return out[:n]


def meta_tokens(path: str) -> list[int] | None:
    """A slot's .meta, as the engine writes it: the magic, its identity (length, bytes), the state's size, its tokens,
    the magic again.  False: not one of the engine's .meta files, or damaged."""
    try:
        with open(path, "rb") as f:
            data = f.read()
        m0, il = struct.unpack_from("<QQ", data, 0)
        if m0 != META_MAGIC:
            return None
        size, nt = struct.unpack_from("<QQ", data, 16 + il)
        toks = list(struct.unpack_from("<%di" % nt, data, 32 + il))
        (m1,) = struct.unpack_from("<Q", data, 32 + il + 4 * nt)
    except (OSError, struct.error):
        return None
    return toks if m1 == META_MAGIC else None


class Engine:
    """One `maya-ds4 --serve` process: its INFO/READY lines, then `GEN` in, `T` / `DONE` out."""

    def __init__(self, exe: str, args: list[str], env: dict):
        self.exe, self.args, self.env = exe, args, env
        self.notes: list[str] = []      # every NOTE line the engine printed (the slots' own lines too)
        self.proc = None
        self.err = None
        self.err_path = ""
        self.buffer = b""
        self.ready = ""

    def start(self, timeout: float) -> None:
        env = dict(os.environ)
        env.update(self.env)
        fd, self.err_path = tempfile.mkstemp(prefix="ds4-slots-err-")
        self.err = os.fdopen(fd, "wb")
        self.proc = subprocess.Popen([self.exe, *self.args], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=self.err, env=env, bufsize=0)
        end = time.monotonic() + timeout
        while True:
            line = self._line(end)
            if line.startswith("READY "):
                self.ready = line
                return
            if line.startswith("NOTE "):
                self.notes.append(line[5:].strip())
            elif line.startswith("ERR "):
                raise RuntimeError("the engine refused to start: " + line)

    def _line(self, end: float) -> str:
        while b"\n" not in self.buffer:
            left = end - time.monotonic()
            if left <= 0:
                raise RuntimeError("the engine did not answer in time")
            if self.proc.poll() is not None:
                raise RuntimeError(f"the engine ended (exit {self.proc.returncode})")
            r, _, _ = select.select([self.proc.stdout], [], [], min(left, 1.0))
            if not r:
                continue
            chunk = os.read(self.proc.stdout.fileno(), 1 << 16)
            if not chunk:
                raise RuntimeError("the engine's output ended")
            self.buffer += chunk
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line.decode("utf-8", "replace").strip()

    def request(self, prompt: list[int], max_new: int, spec: int = 0, timeout: float = 3600.0) -> dict:
        self.proc.stdin.write(f"GEN {max_new} spec={spec} {','.join(str(t) for t in prompt)}\n".encode())
        self.proc.stdin.flush()
        out = {"tokens": [], "reused": None, "prompt": None, "finish": None, "s": 0.0}
        t0 = time.monotonic()
        end = t0 + timeout
        while True:
            line = self._line(end)
            if line.startswith("T "):
                out["tokens"].append(int(line[2:]))
            elif line.startswith("ERR "):
                out["error"] = line[4:]
                out["s"] = time.monotonic() - t0
                return out
            elif line.startswith("DONE "):
                f = line.split()
                out["generated"], out["prompt"], out["finish"] = int(f[1]), int(f[2]), f[5]
                out["reused"] = int(f[8])
                out["s"] = time.monotonic() - t0
                return out
            elif line.startswith("NOTE "):
                self.notes.append(line[5:].strip())

    def quit(self, timeout: float = 300.0) -> None:
        try:
            self.proc.stdin.write(b"QUIT\n")
            self.proc.stdin.flush()
        except OSError:
            pass
        end = time.monotonic() + timeout
        while time.monotonic() < end and self.proc.poll() is None:
            r, _, _ = select.select([self.proc.stdout], [], [], 0.2)
            if not r:
                continue
            chunk = os.read(self.proc.stdout.fileno(), 1 << 16)
            if not chunk:
                break
            self.buffer += chunk
            while b"\n" in self.buffer:
                line, self.buffer = self.buffer.split(b"\n", 1)
                line = line.decode("utf-8", "replace").strip()
                if line.startswith("NOTE "):
                    self.notes.append(line[5:].strip())
        try:
            self.proc.wait(timeout=max(1.0, end - time.monotonic()))
        except subprocess.TimeoutExpired:
            self.proc.kill()
            raise RuntimeError("the engine did not end after QUIT")

    def stderr_lines(self) -> list[str]:
        """The engine's stderr once it has ended (run() checks its `ds4 slots:` lines: a slot that failed to be set
        aside, taken back or written is a failure of this test, not a log line)."""
        if not self.err_path:
            return []
        try:
            self.err.close()
        except (OSError, AttributeError):
            pass
        with open(self.err_path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
        try:
            os.unlink(self.err_path)
        except OSError:
            pass
        return lines


def config_env(path: str) -> dict[str, str]:
    """A model config's DeepSeek settings, under the names the engine reads: a unified-memory APU's `DS4_POOL_GB` (its
    pool is otherwise sized from hipMemGetInfo), the streaming and the tiers it pins."""
    cfg = maya_names.env_in(json.loads(Path(path).read_text(encoding="utf-8-sig")))   # (a BOM: Notepad)
    return {k: str(v) for k, v in (cfg.get("env") or {}).items() if k.startswith(DS4_ENV)}


def engine_env(settings: dict, keep: bool, slots: str, slot_gb: float | None) -> dict[str, str]:
    """The engine's environment: the config's settings first, this test's slot settings over them (the test decides
    what it measures)."""
    env = dict(settings)
    env.update({"MAYA_DS4_SLOT_KEEP": "1" if keep else "0", "MAYA_DS4_SLOT_DIR": slots})
    if slot_gb is not None:
        env["MAYA_DS4_SLOT_GB"] = str(slot_gb)
    return env


def run(exe: str, gpus: list[int], pack: str, ctx: int, slots: str, keep: bool, steps: list[tuple[str, Callable]],
        max_new: int, slot_gb: float | None = None, settings: dict | None = None) -> tuple[dict, list[str], float]:
    """One engine process: each step (a name, and a function of the answers so far that gives its prompt) in order.
    `settings`: the config's engine settings (config_env), the slot settings here over them."""
    env = engine_env(settings or {}, keep, slots, slot_gb)
    args = [*(("--gpus", ",".join(str(g) for g in gpus)) if gpus else ()), "--pack", pack, "--max-context", str(ctx),
            "--dspark", "on", "--threads", "on", "--vision", "off", "--expert-profile", "none", "--serve"]
    eng = Engine(exe, args, env)
    t0 = time.monotonic()
    answers: dict[str, dict] = {}
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
    for line in err:   # (a slot that failed to be set aside, taken back or written: this test's failure, not a log line)
        if line.startswith("ds4 slots:") and "failed" in line:
            raise RuntimeError("the engine: " + line)
    return answers, eng.notes, load_s


def main() -> int:
    ap = argparse.ArgumentParser(description="the kept conversations (MAYA_DS4_SLOT_KEEP): A, B, then A's next turn")
    ap.add_argument("--pack", required=True, help="the model's pack dir (maya-ds4)")
    ap.add_argument("--exe", default=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))), "build", "maya-ds4"), help="the engine (default: <repo>/build/maya-ds4)")
    ap.add_argument("--gpus", default="", help="the GPUs (default: every visible one, the engine's own split)")
    ap.add_argument("--config", default="", help="a model config (maya-ds4.json): its DS4_/MAYA_DS4_ settings go to the "
                                                "engine - an APU's DS4_POOL_GB, else its pool is sized from hipMemGetInfo "
                                                "and its QUIT can hang for 300 s; the slot settings here win over it")
    ap.add_argument("--ctx", type=int, default=32768, help="--max-context")
    ap.add_argument("--len-a", type=int, default=12000, help="A's prompt (tokens)")
    ap.add_argument("--len-b", type=int, default=9000, help="B's prompt (tokens)")
    ap.add_argument("--len-c", type=int, default=4000, help="C's prompt (tokens, the eviction run; shorter than A, so "
                                                             "B and C fit under A+B's bytes once A is gone)")
    ap.add_argument("--len-turn", type=int, default=64, help="A's next turn's new tokens")
    ap.add_argument("--max-new", type=int, default=24, help="the answers' length (greedy)")
    a_ = ap.parse_args()
    settings = config_env(a_.config) if a_.config else {}

    if not os.path.exists(a_.exe):
        print(f"{a_.exe}: no engine - build it first (cmake --build build --target maya-ds4)", flush=True)
        return 2
    try:
        from strata_tokenizer import Tokenizer   # (tools/strata_tokenizer.py's own: `regex`)
    except ImportError as e:
        print(f"the tokenizer cannot be imported: {e} (pip install regex)", flush=True)
        return 2
    try:
        tok = Tokenizer.from_dir(os.path.join(a_.pack, "tokenizer"))
    except (OSError, ValueError) as e:
        print(f"{os.path.join(a_.pack, 'tokenizer')}: no tokenizer (the prompts are real text): {e}", flush=True)
        return 2
    a = text_tokens(tok, "Conversation A.", a_.len_a)
    b = text_tokens(tok, "Conversation B.", a_.len_b)
    c = text_tokens(tok, "Conversation C.", a_.len_c)
    turn = text_tokens(tok, "The next turn:", a_.len_turn)
    gpus = [int(g) for g in a_.gpus.split(",") if g.strip()]
    max_new = a_.max_new
    ok = True

    def check(what: str, cond: bool) -> None:
        nonlocal ok
        print(("  ok    " if cond else "  FAIL  ") + what, flush=True)
        ok = ok and cond

    def print_diff(name: str, x: list[int], y: list[int]) -> None:
        d = next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), min(len(x), len(y)))
        print(f"        {name}: token {d}: {x[d:d + 4]} vs {y[d:d + 4]}", flush=True)

    tmp = tempfile.mkdtemp(prefix="ds4-slots-")
    dir_inrun = os.path.join(tmp, "in-run")
    dir_kept = os.path.join(tmp, "kept")
    dir_evict = os.path.join(tmp, "evict")
    try:
        print("no-keep: A, B, A's next turn, one process (MAYA_DS4_SLOT_KEEP=0, the reference)", flush=True)
        r0, _, _ = run(a_.exe, gpus, a_.pack, a_.ctx, os.path.join(tmp, "off-1"), False,
                       [("A", lambda _: a), ("B", lambda _: b),
                        ("A-turn", lambda ans: a + ans["A"]["tokens"] + turn)], max_new, settings=settings)
        ref, ref_a = r0["A-turn"], r0["A"]["tokens"]
        print(f"        A {len(a)} tok in {r0['A']['s']:.0f} s, B {len(b)} tok in {r0['B']['s']:.0f} s; A's next turn "
              f"{ref['len']} tok in {ref['s']:.0f} s, reused {ref['reused']}", flush=True)
        check(f"no-keep: without slots the turn starts over (reused {ref['reused']})", ref["reused"] == 0)

        print("no-keep: the same prompts once more, a second process (the engine's own run-to-run prompt path)",
              flush=True)
        r0b, _, _ = run(a_.exe, gpus, a_.pack, a_.ctx, os.path.join(tmp, "off-2"), False,
                        [("A", lambda _: a), ("B", lambda _: b), ("A-turn", lambda _: a + ref_a + turn)], max_new,
                        settings=settings)
        ref2 = r0b["A-turn"]
        same_a = r0b["A"]["tokens"] == ref_a
        same_turn = ref2["tokens"] == ref["tokens"]
        print(f"        A's own answer {'matched' if same_a else 'DIFFERED from'} the reference's "
              f"({len(ref_a)} tokens); the next turn's tokens {'matched' if same_turn else 'DIFFERED'}", flush=True)
        if not same_a:
            print_diff("the second read's A answer vs the reference's", r0b["A"]["tokens"], ref_a)
        if not same_turn:
            print_diff("the second read's turn vs the reference's", ref2["tokens"], ref["tokens"])
        deterministic = same_a and same_turn
        if not deterministic:
            print("        the engine's prompt path is not run-to-run deterministic: two full no-keep reads of the "
                  "same prompt differ - the slot runs are compared against both, not claimed identical to one",
                  flush=True)

        def a_answer(name: str, x: dict) -> None:
            print(f"        {name}: A's own answer {'matched' if x['A']['tokens'] == ref_a else 'DIFFERED from'} the "
                  f"reference's ({len(x['A']['tokens'])} tokens)", flush=True)

        def reuse_check(name: str, got: int, ans: list[int]) -> None:
            # (the state the run holds is A and all but the answer's last fed token: the reference's prompt continues
            #  exactly that - when the run's own answer is the reference's)
            if ans == ref_a:
                check(f"{name}: reuses A's full prefix ({got} of {len(a) + len(ref_a) - 1})",
                      got == len(a) + len(ref_a) - 1)
            else:
                print(f"        {name}: its answer to A is not the reference's, so the reference's prompt is not "
                      f"continued by it past the common part - reused {got}, not claimed", flush=True)

        print("in-run: A, B, A's next turn, one process (the slots in RAM)", flush=True)
        r1, _, load1 = run(a_.exe, gpus, a_.pack, a_.ctx, dir_inrun, True,
                           [("A", lambda _: a), ("B", lambda _: b), ("A-turn", lambda _: a + ref_a + turn)], max_new,
                           settings=settings)
        an, third = r1["A"], r1["A-turn"]
        print(f"        load {load1:.0f} s; A {len(a)} tok in {r1['A']['s']:.0f} s, B {len(b)} tok in "
              f"{r1['B']['s']:.0f} s; A's next turn {third['len']} tok in {third['s']:.1f} s, reused {third['reused']}",
              flush=True)
        a_answer("in-run", r1)
        reuse_check("in-run", third["reused"], an["tokens"])
        check(f"in-run: the prompt's length agrees ({third['prompt']} of {third['len']})",
              third["prompt"] == third["len"])

        print("kept: A, B, then QUIT (both conversations written to the folder)", flush=True)
        rk, _, _ = run(a_.exe, gpus, a_.pack, a_.ctx, dir_kept, True,
                       [("A", lambda _: a), ("B", lambda _: b)], max_new, slot_gb=0.05, settings=settings)
        a_answer("kept", rk)
        bins = [f for f in os.listdir(dir_kept) if f.startswith("slot-") and f.endswith(".bin")]
        metas = [f for f in os.listdir(dir_kept) if f.startswith("slot-") and f.endswith(".meta")]
        check(f"kept: the folder holds A and B ({len(bins)} .bin, {len(metas)} .meta)",
              len(bins) == 2 and len(metas) == 2)
        size = sum(os.path.getsize(os.path.join(dir_kept, f)) for f in bins)
        check(f"kept: the folder stays under MAYA_DS4_SLOT_GB=0.05 ({size / 2**20:.1f} MiB)", size <= 0.05 * 2**30)

        def folder_slots(path: str) -> dict[str, int]:
            """A folder's conversations by the prompt their tokens start with ({name: the .bin's bytes})."""
            out: dict[str, int] = {}
            for fn in sorted(os.listdir(path)):
                if not fn.endswith(".meta"):
                    continue
                toks = meta_tokens(os.path.join(path, fn))
                binpath = os.path.join(path, fn[:-5] + ".bin")
                if toks is None or not os.path.exists(binpath):
                    continue
                for name, prompt in (("A", a), ("B", b), ("C", c)):
                    if len(toks) >= len(prompt) and toks[:len(prompt)] == prompt:
                        out[name] = os.path.getsize(binpath)
            return out

        kept_slots = folder_slots(dir_kept)
        check(f"kept: the folder holds A and B ('{', '.join(sorted(kept_slots)) or 'nothing'}')",
              sorted(kept_slots) == ["A", "B"])

        print("restart: A's next turn alone, a new process (the folder's slots)", flush=True)
        r2, notes2, _ = run(a_.exe, gpus, a_.pack, a_.ctx, dir_kept, True,
                            [("A-turn", lambda _: a + ref_a + turn)], max_new, settings=settings)
        sr = r2["A-turn"]
        kept = [n for n in notes2 if "kept from earlier runs" in n]
        print(f"        {kept[0] if kept else '(no kept line)'}; reused {sr['reused']} in {sr['s']:.1f} s", flush=True)
        reuse_check("restart", sr["reused"], rk["A"]["tokens"])
        check(f"restart: the prompt's length agrees ({sr['prompt']} of {sr['len']})", sr["prompt"] == sr["len"])

        if sorted(kept_slots) != ["A", "B"]:
            raise RuntimeError("the kept folder does not hold A and B: the eviction check's cap needs their bytes")
        # the folder's A+B bytes: three conversations cannot fit under them, so the third write must take A (its
        # least recently used) away - and B and C fit once it is (C is shorter than A)
        cap_gb = (kept_slots["A"] + kept_slots["B"]) / 2**30
        print(f"evict: A, B, C into a folder capped at A+B's bytes, {cap_gb:.3f} GB (three cannot fit)", flush=True)
        r4, _, _ = run(a_.exe, gpus, a_.pack, a_.ctx, dir_evict, True,
                       [("A", lambda _: a), ("B", lambda _: b), ("C", lambda _: c)], max_new, slot_gb=cap_gb,
                       settings=settings)
        a_answer("evict", r4)
        evict_slots = folder_slots(dir_evict)
        print(f"        the folder holds '{', '.join(sorted(evict_slots)) or 'nothing'}' "
              f"({sum(evict_slots.values()) / 2**20:.1f} MiB of the {cap_gb * 2**30 / 2**20:.1f} MiB cap)", flush=True)
        check("evict: the least recently used (A) is gone", "A" not in evict_slots)
        check("evict: the newest (C) is kept", "C" in evict_slots)
        check("evict: B stays (A's bytes made room for C)", "B" in evict_slots)
        check(f"evict: the folder stays under the cap ({sum(evict_slots.values())} of {round(cap_gb * 2**30)} bytes)",
              sum(evict_slots.values()) <= cap_gb * 2**30)

        if deterministic:
            check(f"greedy: in-run's {len(ref['tokens'])} tokens are the reference's", third["tokens"] == ref["tokens"])
            check(f"greedy: restart's {len(ref['tokens'])} tokens are the reference's", sr["tokens"] == ref["tokens"])
            if third["tokens"] != ref["tokens"]:
                print_diff("in-run vs the reference", third["tokens"], ref["tokens"])
            if sr["tokens"] != ref["tokens"]:
                print_diff("restart vs the reference", sr["tokens"], ref["tokens"])
        else:
            for name, x in (("in-run", third), ("restart", sr)):
                match = [what for what, y in (("the reference's", ref), ("the second read's", ref2))
                         if x["tokens"] == y["tokens"]]
                print(f"        {name}: {len(x['tokens'])} tokens, "
                      f"{'the same as ' + ' and '.join(match) if match else 'the same as neither no-keep read'}",
                      flush=True)
                if not match:
                    print_diff(f"{name} vs the reference", x["tokens"], ref["tokens"])
                check(f"greedy: {name}'s tokens are one of the two no-keep reads' ({', '.join(match) or 'neither'})",
                      bool(match))
    except (RuntimeError, OSError) as e:
        print(f"FAIL: {e}", flush=True)
        ok = False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("PASS: the kept conversations read on from their state" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

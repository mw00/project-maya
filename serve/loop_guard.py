"""serve/loop_guard.py - the loop guard: an answer that has fallen into a loop - the same block of tokens again and
again until the token cap (a greedy MiniMax-M3.1 Maya-XS answer wrote one number sequence for 12K tokens) - is seen
as it starts and broken.  It works on the token ids the server reads from the engine, so every engine (GLM,
DeepSeek, MiniMax) gets it alike, in the thinking and in the answer.

    LoopDetector   the rolling check: is the tail of what was written exactly periodic - a block of P_MIN..P_MAX
                   tokens repeated back to back REPS_LONG times (REPS_SHORT for a block under SHORT_BELOW tokens),
                   MIN_SPAN tokens at least?  Each token is compared only with the earlier places of the same id
                   (within P_MAX), so a token costs microseconds.
    LoopGuard      one request's guard: the detector, the tail the stream holds back (the copies a loop's repeats
                   would be are never sent before it is clear they are not one), and what each loop does:
                     1st              the repeats are dropped (the first copy stays) and the engine goes on from
                                      there with a short repetition penalty (LOOP_PENALTY over the last 2 x period
                                      tokens, at least PENALTY_MIN_N, for the next 2 x period + PENALTY_EXTRA)
                     2nd, thinking    the thinking is closed with the model's own end-of-thinking token: it answers
                     2nd in the answer, or a 3rd    the answer ends there (finish "stop")

Until a loop is seen nothing changes: the engine gets the same calls with the same sampling and writes the same
tokens (the stream only holds back the tail that repeats while it does).  Legit repetition - a table's rows, an array
literal, a function name over and over, code lines alike - is not an exact block repeated back to back three times
or more, which is what a loop is.

Off: MAYA_LOOP_GUARD=0, "loop_guard": false in the run config, or "loop_guard": false in a request (true there turns
it on for that request when the server has it off).
"""
from __future__ import annotations

from typing import NamedTuple

P_MIN = 8               # the shortest block: shorter loops (one token, "0, 0, 0") repeat a multiple of it in 8..15
P_MAX = 2048            # the longest: a loop of whole paragraphs or a plan written again (Maya-S thought a ~1,170-
                        # token plan four times); answers without a loop hold back and cost the same as with 400
SHORT_BELOW = 32        # blocks under this many tokens must repeat REPS_SHORT times, longer ones REPS_LONG times
REPS_SHORT = 4
REPS_LONG = 3
MIN_SPAN = 96           # ... and the copies together cover this many tokens at least
LOOP_PENALTY = 1.15     # the temporary repetition penalty after the first loop
PENALTY_MIN_N = 128     # its window (penalty_last_n): 2 x the period, at least this
PENALTY_EXTRA = 128     # it applies to the next 2 x period + this many tokens, then the request's own sampling again
PENALTY_N_MAX = 4096    # (the engines count at most this many: server.PENALTY_WINDOW_MAX)

PENALTY, CLOSE_THINKING, STOP = "penalty", "close_thinking", "stop"


class Loop(NamedTuple):
    """A loop at the tail: `period` tokens a copy; the last `repeats` tokens repeat the `period` before them."""
    period: int
    repeats: int

    @property
    def copies(self) -> float:
        return (self.repeats + self.period) / self.period


class Action(NamedTuple):
    kind: str           # PENALTY, CLOSE_THINKING or STOP
    period: int
    copies: float
    drop: int           # the repeats to drop from the end (the first copy stays)
    where: str          # "thinking" or "answer"
    at: int             # the tokens written when it was seen (before the drop)

    def describe(self) -> str:
        what = {PENALTY: "repeats dropped, a short repetition penalty", CLOSE_THINKING: "thinking closed",
                STOP: "answer ended"}[self.kind]
        return (f"a {self.period}-token block {self.copies:.0f} times in the {self.where} at token {self.at:,}: "
                f"{what}")


class LoopDetector:
    """The tail's exact periodicity, updated a token at a time.  `runs[p]` is how many tokens in a row (to the last
    one) equal the token p before them; p is a loop's period once its run covers the copies it needs (`need[p]`).
    Only the periods where the new token equals the one p back go on (the earlier places of that id within P_MAX),
    every other run ends - so the work a token is the number of times its id occurred in the last P_MAX tokens.
    `hold` is the longest run now: the tokens that could still turn out to be a loop's repeats."""

    def __init__(self, p_min: int = P_MIN, p_max: int = P_MAX, min_span: int = MIN_SPAN,
                 reps_short: int = REPS_SHORT, reps_long: int = REPS_LONG, short_below: int = SHORT_BELOW):
        self.p_min, self.p_max = p_min, p_max
        self.need = [0] * (p_max + 1)
        for p in range(p_min, p_max + 1):
            reps = reps_short if p < short_below else reps_long
            self.need[p] = max((reps - 1) * p, min_span - p)
        self.reset()

    def reset(self, tokens=()) -> None:
        """Start again over `tokens` (what is kept after a loop's repeats were dropped): only their last 3 x P_MAX
        count - a run reaches at most 2 x P_MAX before it is a loop."""
        self.n, self.seen, self.runs, self.hold = 0, {}, {}, 0
        tokens = list(tokens)[-3 * self.p_max:]
        for t in tokens:
            self.push(t)                        # (a prefix of what was checked token by token: no loop in it)

    def push(self, t: int) -> Loop | None:
        i = self.n
        self.n = i + 1
        q = self.seen.get(t)
        runs, hold, best = {}, 0, 0
        if q is None:
            self.seen[t] = [i]
        else:
            lo, k = i - self.p_max, 0
            while k < len(q) and q[k] < lo:
                k += 1
            if k:
                del q[:k]
            old, need, p_min = self.runs, self.need, self.p_min
            for j in q:                         # (oldest first: the periods from the longest down)
                p = i - j
                if p < p_min:
                    break
                r = old.get(p, 0) + 1
                runs[p] = r
                if r > hold:
                    hold = r
                if r >= need[p]:
                    best = p                    # the shortest that is a loop: its fundamental period
            q.append(i)
        self.runs, self.hold = runs, hold
        return Loop(best, runs[best]) if best else None


def penalty_window(period: int) -> int:
    return min(PENALTY_N_MAX, max(PENALTY_MIN_N, 2 * period))


class LoopGuard:
    """One request's loop guard (server.Service._sliced runs it): push() every id the engine writes, send only what
    lies before the last `hold` of them, and do what an Action says: drop its repeats, close the thinking (append
    `think_end`), or end; then resume() with what is kept and sampling() for each engine call after it."""

    def __init__(self, think_end: int | None = None, in_think: bool = False, detector: LoopDetector | None = None):
        self.det = detector or LoopDetector()
        self.think_end = think_end
        self.in_think = bool(in_think) and think_end is not None
        self.hits = {"thinking": 0, "answer": 0}
        self.events: list[Action] = []
        self.until = 0                          # the tokens written up to which the temporary penalty applies
        self.window = 0                         # ... and its penalty_last_n

    @property
    def hold(self) -> int:
        return self.det.hold

    def push(self, t: int, written: int | None = None) -> Action | None:
        if self.in_think and t == self.think_end:
            self.in_think = False
        loop = self.det.push(t)
        if loop is None:
            return None
        where = "thinking" if self.in_think else "answer"
        self.hits[where] += 1
        if sum(self.hits.values()) >= 3:
            kind = STOP
        elif where == "thinking":
            kind = PENALTY if self.hits[where] == 1 else CLOSE_THINKING
        else:
            kind = PENALTY if self.hits[where] == 1 else STOP
        act = Action(kind, loop.period, loop.copies, loop.repeats, where, written if written is not None else 0)
        self.events.append(act)
        return act

    def resume(self, kept: list[int], act: Action) -> None:
        """After the repeats were dropped (and the thinking closed): check on from what is kept, with the temporary
        penalty for the next 2 x period + PENALTY_EXTRA tokens."""
        if act.kind == CLOSE_THINKING:
            self.in_think = False
        self.det.reset(kept)
        if act.kind != STOP:
            self.window = penalty_window(act.period)
            self.until = len(kept) + 2 * act.period + PENALTY_EXTRA

    def sampling(self, base: dict | None, written: int, left: int) -> tuple[dict | None, int]:
        """The sampling for an engine call that starts after `written` tokens, and how many tokens it may write:
        the request's own (and `left`) - unless the temporary penalty runs, which then ends that call where it ends.
        A request's own stronger penalty or longer window stays."""
        if written >= self.until:
            return base, left
        smp = dict(base or {})
        rp = smp.get("repetition_penalty")
        rp = float(rp) if isinstance(rp, (int, float)) and not isinstance(rp, bool) else 1.0
        smp["repetition_penalty"] = max(rp, LOOP_PENALTY)
        pln = smp.get("penalty_last_n")
        if not (isinstance(pln, int) and not isinstance(pln, bool) and (pln == -1 or pln >= self.window)):
            smp["penalty_last_n"] = self.window
        return smp, max(1, min(left, self.until - written))

    def summary(self) -> list[str]:
        return [a.describe() for a in self.events]

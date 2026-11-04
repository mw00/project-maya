"""serve/loop_guard.py - the loop guard: an answer that has fallen into a loop - the same block of tokens again and
again until the token cap (a greedy MiniMax-M3.1 Maya-XS answer wrote one number sequence for 12K tokens) - is seen
as it starts and broken.  It works on the token ids the server reads from the engine, so every engine (GLM,
DeepSeek, MiniMax) gets it alike, in the thinking and in the answer.

    LoopDetector   the rolling check: is the tail of what was written exactly periodic - a block of P_MIN..P_MAX
                   tokens repeated back to back REPS_LONG times (REPS_SHORT for a block under SHORT_BELOW tokens),
                   MIN_SPAN tokens at least?  Each token is compared only with the earlier places of the same id
                   (within P_MAX), so a token costs microseconds.
    LoopGuard      one request's guard: the detector, the short tail the stream holds back (at most HOLD_MAX
                   tokens: a block the model copies from earlier in its output streams on as it is written, never
                   held whole), and what each loop does:
                     penalty          the repeats not sent yet are dropped (what was sent stays) and the engine goes
                                      on from there with a short repetition penalty (LOOP_PENALTY over the last
                                      2 x period tokens, at least PENALTY_MIN_N, for the next 2 x period +
                                      PENALTY_EXTRA): the 1st loop in the answer, the 1st and 2nd in the thinking
                     close thinking   the 3rd loop in the thinking: closed with the model's own end-of-thinking token,
                                      it answers (never a silent end while it thinks)
                     stop             the 2nd loop in the answer, or a 4th in all: the answer ends (finish "stop")
                   A loop in the thinking needs more copies (REPS_*_THINK, MIN_SPAN_THINK): reasoning that drafts a
                   plan again is not one.

Across turns, an agent can loop with every answer short and fine: it runs the same tool call, gets the same result
and runs it again (GLM-5.3 ran two commands in turn 25 times over opencode, until the user wrote "stop repeating").

    tool_repeat      the request's own conversation: has the last tool call - its name, its arguments and its result
                     all the same - come TOOL_REPS times among the last TOOL_WINDOW calls since the user's last
                     message?  Then tool_note() follows that result as a message of its own, for the model to read before it
                     answers (inside the tool's result GLM-5.3 ran the call a 4th time; as a message it moved on).
    stops_short      ... and the conversation repeats itself (collapse): the same model ended its turn with talk about
                     its loop, or with what it had already said, and the agent stopped there.  Such an answer - one
                     that says nothing new - goes on (KEEP_NOTE, answered, not thought about), at most KEEP_MAX times
                     in one answer; an answer with anything new in it ends the turn as always.

Until a loop is seen nothing changes: the engine gets the same calls with the same sampling and writes the same
tokens (the stream only holds back up to HOLD_MAX tokens of a tail that repeats while it does).  Legit repetition - a table's rows, an array
literal, a function name over and over, code lines alike - is not an exact block repeated back to back three times
or more, which is what a loop is.

Off: MAYA_LOOP_GUARD=0, "loop_guard": false in the run config, or "loop_guard": false in a request (true there turns
it on for that request when the server has it off).
"""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Callable, NamedTuple

P_MIN = 8               # the shortest block: shorter loops (one token, "0, 0, 0") repeat a multiple of it in 8..15
P_MAX = 2048            # the longest: a loop of whole paragraphs or a plan written again (Maya-S thought a ~1,170-
                        # token plan four times); answers without a loop hold back and cost the same as with 400
SHORT_BELOW = 32        # blocks under this many tokens must repeat REPS_SHORT times, longer ones REPS_LONG times
REPS_SHORT = 4
REPS_LONG = 3
MIN_SPAN = 96           # ... and the copies together cover this many tokens at least
REPS_SHORT_THINK = 5    # in the thinking: more copies (a model re-drafting its plan is not looping)
REPS_LONG_THINK = 4
MIN_SPAN_THINK = 192
HOLD_MAX = 64           # the most the stream holds back (~1-3 s): a copied block streams on; past this tail, a
                        # loop's repeats are already on screen and only the ones not sent yet are dropped
THINK_PENALTIES = 2     # loops in the thinking broken by the penalty before the guard closes the thinking
STOP_AFTER = 4          # loops in all after which the answer ends (in the answer; the thinking is closed first)
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
                 reps_short: int = REPS_SHORT, reps_long: int = REPS_LONG, short_below: int = SHORT_BELOW,
                 reps_short_think: int = REPS_SHORT_THINK, reps_long_think: int = REPS_LONG_THINK,
                 min_span_think: int = MIN_SPAN_THINK):
        self.p_min, self.p_max = p_min, p_max
        self.need = [0] * (p_max + 1)           # the run a period needs in the answer ...
        self.need_think = [0] * (p_max + 1)     # ... and in the thinking (more copies)
        for p in range(p_min, p_max + 1):
            reps = reps_short if p < short_below else reps_long
            self.need[p] = max((reps - 1) * p, min_span - p)
            reps = reps_short_think if p < short_below else reps_long_think
            self.need_think[p] = max((reps - 1) * p, min_span_think - p)
        self.reset()

    def reset(self, tokens=()) -> None:
        """Start again over `tokens` (what is kept after a loop's repeats were dropped): only their last 3 x P_MAX
        count - a run reaches at most 2 x P_MAX before it is a loop.  The runs start from zero: what is kept may end
        in copies that were already sent, and a new loop needs copies of its own."""
        self.n, self.seen, self.runs, self.hold = 0, {}, {}, 0
        tokens = list(tokens)[-3 * self.p_max:]
        for t in tokens:
            self.push(t)                        # (a prefix of what was checked token by token: no loop in it)
        self.runs, self.hold = {}, 0

    def push(self, t: int, thinking: bool = False) -> Loop | None:
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
            old, need, p_min = self.runs, self.need_think if thinking else self.need, self.p_min
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
    lies before the last `hold` of them (at most HOLD_MAX), and do what an Action says: drop its repeats not sent
    yet, close the thinking (append `think_end`), or end; then resume() with what is kept and sampling() for each
    engine call after it."""

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
        """The tail held back now: what could still be a loop's repeats, never more than HOLD_MAX tokens (a whole
        copied block held until it ends was a long silence, then a burst)."""
        return min(self.det.hold, HOLD_MAX)

    def push(self, t: int, written: int | None = None) -> Action | None:
        if self.in_think and t == self.think_end:
            self.in_think = False
        loop = self.det.push(t, self.in_think)
        if loop is None:
            return None
        where = "thinking" if self.in_think else "answer"
        self.hits[where] += 1
        if where == "thinking":                 # (never STOP while it thinks: that ends with no answer at all)
            kind = PENALTY if self.hits[where] <= THINK_PENALTIES else CLOSE_THINKING
        else:
            kind = PENALTY if self.hits[where] == 1 and sum(self.hits.values()) < STOP_AFTER else STOP
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

    def thinking_closed(self, kept: list[int]) -> None:
        """The server closed the thinking itself (its budget, at a sentence's end: `kept` ends with think_end): check on
        from what is kept, with no penalty - nothing looped."""
        self.in_think = False
        self.det.reset(kept)

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


# ------------------------------------------------------------------ never inside a tool call
# A tool call is written as text between the family's markers - `<tool_call>` ... `</tool_call>` (GLM, and MiniMax with
# its namespace in front), DeepSeek's `<｜DSML｜ calls>` ... `</｜DSML｜ calls>` - and calls that repeat (three alike in
# one reply: three files read, three commands run) are legitimate: dropping "the repeats" would cut a call in half and
# hand the client broken JSON.  So a loop whose repeated tokens touch a tool call is never cut and never closes the
# thinking; a loop inside one call's own text (a line written again and again) only gets the short repetition
# penalty, once a call.  Outside tool calls nothing changes.
_TOOL_OPEN = re.compile(r"<tool_call>|<｜DSML｜\s*(?:tool_)?calls[^>]*>")
_TOOL_CLOSE = re.compile(r"</tool_call>|</｜DSML｜\s*(?:tool_)?calls\s*>")
_TAIL = 48              # the text kept to find a marker split across tokens


class ToolSafeGuard(LoopGuard):
    """LoopGuard that leaves tool calls alone.  `piece(id)` is the id's text (a server Detokenizer's push): without
    it this is the plain LoopGuard."""

    def __init__(self, think_end: int | None = None, in_think: bool = False, detector: LoopDetector | None = None,
                 piece: Callable[[int], str] | None = None):
        super().__init__(think_end, in_think, detector)
        self.piece = piece
        self.in_tool = False
        self.tool_penalised = False             # (the penalty went to this call's loop already)
        self.call_n = 0                         # the tokens written in the tool call now
        self.since_tool = 1 << 60               # the tokens written since the last one in a tool call
        self.tail = ""

    def _follow(self, t: int) -> bool:
        """Read token t's text for the markers; True when it is in a tool call (or ends one)."""
        was = self.in_tool
        buf = self.tail + (self.piece(t) or "")
        while True:
            m = (_TOOL_CLOSE if self.in_tool else _TOOL_OPEN).search(buf)
            if m is None:
                break
            self.in_tool = not self.in_tool
            if self.in_tool:
                self.tool_penalised, self.call_n = False, 0
            buf = buf[m.end():]
        self.tail = buf[-_TAIL:]
        return was or self.in_tool

    def push(self, t: int, written: int | None = None) -> Action | None:
        if self.piece is None:
            return super().push(t, written)
        inside = self._follow(t)
        self.since_tool = 0 if inside else self.since_tool + 1
        self.call_n += self.in_tool
        act = super().push(t, written)
        if act is None or (not inside and self.since_tool >= act.drop):
            return act                          # (no loop, or one wholly outside the tool calls: as ever)
        self.events.pop()                       # a loop that touches a tool call: not counted, nothing dropped
        self.hits[act.where] -= 1
        if not self.in_tool or self.tool_penalised or self.call_n < act.period + act.drop:
            return None                         # (a loop of whole calls, or one this call began with: nothing)
        self.tool_penalised = True              # (inside one call's text: only the penalty, the tokens stay)
        act = act._replace(kind=PENALTY, drop=0, where="tool call")
        self.events.append(act)
        return act

    def resume(self, kept: list[int], act: Action) -> None:
        self.since_tool = max(0, self.since_tool - act.drop)
        super().resume(kept, act)


# ------------------------------------------------------------------------------------------------ across turns
TOOL_WINDOW = 12        # the last this many tool calls since the user's last message ...
TOOL_REPS = 3           # ... hold the last one (name, arguments and result the same) this many times: a loop
TOOL_NUDGES = 3         # notes (tool_note) before the server stops running it: then the call itself is blocked
TOOL_BLOCK = TOOL_REPS + TOOL_NUDGES - 1   # (made this many times: TOOL_NUDGES notes went by, unheeded)
BLOCK_HEAT = 0.15       # each blocked attempt in one answer after the first: the temperature this much higher ...
BLOCK_HEAT_MAX = 1.5    # ... up to this (off its track: the same call again and again is the model's surest path)


class ToolRepeat(NamedTuple):
    """The last tool call of a conversation that keeps coming back: `name`, `times` among the last TOOL_WINDOW."""
    name: str
    times: int


def _same(v) -> str:
    return json.dumps(v, sort_keys=True, ensure_ascii=False, default=str)


def args_key(v) -> str:
    """A call's arguments in one form: a client's JSON string and the parser's dict compare equal."""
    if isinstance(v, str):
        try:
            v = json.loads(v)
        except ValueError:
            pass
    return _same(v)


GO_ON = re.compile(r"(?i)\s*(please\s+)?(continue|proceed|go on|go ahead|keep going|carry on|resume|next|ok(ay)?|"
                   r"yes|y|sure|do it)\s*[.!]*\s*")


def _text(c) -> str:
    """A message's content as text: a string as is; a list of content parts (OpenAI form) by its text parts, joined.
    Anything else (None, a part without text) is ""."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return " ".join(p.get("text", "") for p in c if isinstance(p, dict) and isinstance(p.get("text"), str))
    return ""


def go_on(m: dict) -> bool:
    """A user message that only says to go on ("continue", "proceed", "ok"): it says nothing new, so it does not
    start the count over (a user's "continue" set GLM-5.3 off on the same loop again, twice)."""
    return bool(GO_ON.fullmatch(_text(m.get("content"))))


def _gist(result: str) -> str:
    """A result without its numbers, white space and anything outside printable ASCII: "bytes received: 12 ..."
    and "bytes received: 15 ..." the same."""
    return re.sub(r"\d+|[^\x21-\x7e]+", "", result)


def _tool_calls(messages: list[dict]) -> list[tuple]:
    """(name, arguments, result, gist) of each tool call since the user's last message (one that says more than
    "continue"), in order."""
    calls, pending = [], []
    for m in messages:
        role = m.get("role")
        if role == "user" and not go_on(m):    # the user spoke: what came before is not this loop
            calls, pending = [], []
        elif role == "assistant":
            pending = [((c.get("function") or c).get("name"), args_key((c.get("function") or c).get("arguments")))
                       for c in m.get("tool_calls") or []]
        elif role == "tool" and pending:        # results answer the calls in their order
            name, args = pending.pop(0)
            result = _same(m.get("content"))
            calls.append((name, args, result, _gist(result)))
    return calls


def _times(window: list[tuple], c: tuple) -> int:
    """How often call c counts as made among `window`: with the same result - or, the calls just before it in a row
    (nothing else in between), with a result that differs only in numbers or garbled bytes (a check run again and
    again with no change in between; a command run again after an edit has that edit in between, a poll that
    waits - `sleep` - for a job's progress is not counted that way)."""
    exact = sum(1 for x in window if x[:3] == c[:3])
    if "sleep" in c[1]:                         # a wait for something to change: only the same result counts
        return exact
    run = 0
    for x in reversed(window[:len(window) - window[::-1].index(c)]):
        if x[:2] != c[:2] or x[3] != c[3]:
            break
        run += 1
    return max(exact, run)


def tool_repeat(messages: list[dict]) -> ToolRepeat | None:
    """The template messages (frontend.openai_to_messages / anthropic_to_messages) of a request that ends with tool
    results: is the last call one the model made TOOL_REPS times among its last TOOL_WINDOW since the user's last
    message, with the same arguments, and got the same result each time (_times)?  A command run again after an edit
    gives another result, a poll that finally changes another one too: neither counts."""
    if not messages or messages[-1].get("role") != "tool":
        return None
    calls = _tool_calls(messages)
    if not calls:
        return None
    times = _times(calls[-TOOL_WINDOW:], calls[-1])
    return ToolRepeat(str(calls[-1][0]), times) if times >= TOOL_REPS else None


def tool_blocked(messages: list[dict]) -> dict[tuple[str, str], ToolRepeat]:
    """The calls this request's answer may not make again, by (name, args_key(arguments)): each made TOOL_BLOCK times
    among the last TOOL_WINDOW since the user's last message with the same arguments and result (_times) - the notes
    after TOOL_REPS of them did not stop it.  The model may still run anything else, these too with other arguments."""
    window = _tool_calls(messages)[-TOOL_WINDOW:]
    out = {}
    for c in window:
        t = _times(window, c)
        if t >= TOOL_BLOCK and t > out.get(c[:2], ToolRepeat("", 0)).times:
            out[c[:2]] = ToolRepeat(str(c[0]), t)
    return {(str(k[0]), k[1]): v for k, v in out.items()}


def blocked_note(blocked: dict) -> str:
    """The message after a user's "continue" when calls are blocked: which ones, before the model writes them."""
    names = "; ".join(f"`{k[0]}` {_short(k[1])} ({v.times} runs, the same result)" for k, v in blocked.items())
    return (f"[The server's loop guard, not the user: these calls are blocked and will not run, they keep giving the "
            f"same result: {names}. Keep working on the task with something else - another command, another file, "
            "another part of the problem.]")


def _short(args: str, n: int = 120) -> str:
    return args if len(args) <= n else args[:n] + "...}"


def block_result(rep: ToolRepeat) -> str:
    """The tool result the server writes for a blocked call (the client never sees the call)."""
    return (f"[Not run. The server's loop guard blocked this call: `{rep.name}` already ran {rep.times} times since "
            "the user's last message with these same arguments, and gave the same result every time.]")


def block_note(rep: ToolRepeat) -> str:
    """The message after a blocked call's result: firmer than tool_note, since that one went unheeded."""
    return (f"[The server's loop guard, not the user: you have been told {TOOL_NUDGES} times that `{rep.name}` with "
            "these arguments gives nothing new, and it is now blocked: it will not run again in this answer. Do not "
            "write it again. Keep working on the task, but change something real - other arguments, another command or "
            "tool, another file, another part of the problem.]")


STEERS = (   # the start of the model's own thinking after a blocked call (block_steer), one after another
    "The server blocked `{name}` {args}: it already ran {times} times with the same result, so running it again "
    "cannot show anything new. I will not write that call again. A different next step:",
    "Blocked again - that exact call will never run, however I write it. I have to change what I do, not repeat it. "
    "Something I have not tried yet:",
    "That call is blocked for good. Concretely, instead of it I will now",
    "Not that call. Looking at the problem from another side, the next thing to check is",
)


def block_steer(rep: ToolRepeat, args: str, tries: int) -> str:
    """The first words of the model's thinking after its `tries`-th blocked attempt in this answer: written for it in
    the prompt (the client never sees them), so it goes on from a step away from the call, not back to it - worded
    differently each time.  The model is never stopped: it goes on until it does something else."""
    return STEERS[min(tries, len(STEERS)) - 1].format(name=rep.name, times=rep.times, args=_short(args, 160)) + " "


def block_sampling(sampling: dict | None, tries: int) -> dict:
    """The sampling after the `tries`-th blocked attempt in one answer: from the 2nd on, a little hotter each time
    (BLOCK_HEAT, at most BLOCK_HEAT_MAX), so the next attempt can leave the track that leads to the same call."""
    smp = dict(sampling or {})
    if tries >= 2:
        t = smp.get("temperature")
        t = float(t) if isinstance(t, (int, float)) and not isinstance(t, bool) else 1.0
        smp["temperature"] = min(BLOCK_HEAT_MAX, max(t, 1.0) + BLOCK_HEAT * (tries - 1))
    return smp


NOTES = (    # tool_note's wording, one after another: the same words each time are easy to read past
    "[The server's loop guard, not the user: `{name}` has now run {times} times since the user's last message with "
    "these same arguments and given this same result. Running it again will not show anything new. Do not repeat it: "
    "use what you already have and keep working on the task with a different step.]",
    "[Loop guard (the server, not the user): that was `{name}` again - {times} runs, identical arguments, identical "
    "output. The output is not going to change. Pick something else: a different command, a different file, a test "
    "that can actually fail, a part of the code you have not read yet.]",
    "[Server loop guard, last warning: `{name}` with these arguments has given the same result {times} times. One more "
    "identical call will be blocked and not run. Stop checking this; act on what it already told you and keep "
    "working on the task.]",
)


def tool_note(rep: ToolRepeat) -> str:
    """The message after the repeated call's result: plain words, as a user would write them, said to be the
    server's - worded differently each time (NOTES, by how many runs; the last says a block comes next)."""
    k = min(max(rep.times - TOOL_REPS, 0), len(NOTES) - 1)
    return NOTES[k].format(name=rep.name, times=rep.times)


# ------------------------------------------------------------------------------------------------ what it already wrote
ECHO_TURNS = 3          # a paragraph in this many of the model's messages (or a call + result made this many times) ...
ECHO_MIN = {"reasoning_content": 6, "content": 12}   # ... (pieces this long or longer: in the answer's text
#                         shorter ones - "}", "Done." - are left alone; in the reasoning only "OK." and the like) ...
#                         the conversation carries a loop: the prompt leaves out what the model already wrote (collapse)


class Collapse(NamedTuple):
    """collapse()'s prompt and what it left out of it."""
    messages: list[dict]
    paragraphs: int     # sentences of its reasoning and paragraphs of its text the model had already written
    rounds: int         # tool rounds whose calls and results all repeat earlier ones


def _units(text, field: str) -> list[tuple[str, str]]:
    """(piece, the separator after it): the reasoning by sentence and line (prose; GLM-5.3 also runs its sentences
    together, "Hmm - hmm.Now let me ..."), the answer's text by paragraph (it may hold code)."""
    if not isinstance(text, str):
        return []
    if field != "reasoning_content":            # the answer's text: code blocks whole, the prose as the reasoning
        out = []
        for k, piece in enumerate(re.split(r"(```.*?(?:```|$))", text, flags=re.S)):
            out += [(piece, "")] if k % 2 else _units(piece, "reasoning_content")
        return [u for u in out if u[0] or u[1]]
    parts = re.split(r"(\n+|(?<=[.!?])\s*(?=[A-Z*`\[(]))", text)
    return [(parts[k], parts[k + 1] if k + 1 < len(parts) else "") for k in range(0, len(parts), 2)]


def _key(paragraph: str) -> str:
    return " ".join(paragraph.split())


def _round_keys(messages: list[dict], i: int) -> tuple[list[tuple], int]:
    """The (name, arguments, result) of messages[i]'s calls and the index after its results."""
    calls = [((c.get("function") or c).get("name"), args_key((c.get("function") or c).get("arguments")))
             for c in messages[i].get("tool_calls") or []]
    j, keys = i + 1, []
    while j < len(messages) and messages[j].get("role") == "tool" and len(keys) < len(calls):
        keys.append(calls[len(keys)] + (_same(messages[j].get("content")),))
        j += 1
    return keys, j


def echoes(messages: list[dict]) -> bool:
    """Does the conversation carry a loop across turns: one paragraph in ECHO_TURNS of the model's messages, or one
    call made ECHO_TURNS times with the same arguments and result (or in a row, but for numbers and garbled bytes)?  (It only grows as the conversation does: once
    true it stays true, so the prompt collapse() makes stays the same prefix from one request to the next.)"""
    para, calls, prev, run, longest = Counter(), Counter(), None, 0, 0
    for i, m in enumerate(messages):
        if m.get("role") != "assistant":
            continue
        para.update({_key(p) for f in ("reasoning_content", "content") for p, _ in _units(m.get(f), f)
                     if len(_key(p)) >= ECHO_MIN[f]})
        keys = _round_keys(messages, i)[0]
        calls.update(keys)
        if keys:                                # (in a row, but for numbers and garbled bytes: as _times)
            gists = [k[:2] + (_gist(k[2]),) for k in keys]
            run = run + 1 if gists == prev and not any("sleep" in k[1] for k in keys) else 1
            prev, longest = gists, max(longest, run)
    return any(n >= ECHO_TURNS for n in para.values()) or any(n >= ECHO_TURNS for n in calls.values()) or \
        longest >= ECHO_TURNS


# Above this many characters of conversation (roughly tokens x 3.5) the prompt is left whole: collapsing history
# rewrites the prompt's prefix, and the engine's conversation cache (snapshot/checkpoints) then no longer matches, so
# the whole conversation is read again (a 126K-token prompt: 16 chunks, ~2 minutes, #702).  The generation-side
# protections (stops_short, keep_going, the repetition penalty) still apply to long conversations.
# MAYA_LOOP_GUARD_COLLAPSE_MAX overrides it; 0 = no limit (the behaviour before #702).
COLLAPSE_MAX_CHARS = 32000


def conv_chars(messages: list[dict]) -> int:
    """The conversation's size in characters (content, reasoning and tool calls): what COLLAPSE_MAX_CHARS bounds."""
    n = 0
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            n += len(c)
        r = m.get("reasoning_content")
        if isinstance(r, str):
            n += len(r)
        for tc in m.get("tool_calls") or []:
            n += len(json.dumps(tc, ensure_ascii=False))
    return n


def collapse(messages: list[dict]) -> Collapse | None:
    """The prompt of a conversation that carries a loop (echoes()) without what the model already wrote: a paragraph
    of its reasoning or text that it wrote before (an earlier message, or earlier in the same one) is left out, and
    a round whose calls and results all repeat earlier ones (or the round just before, but for numbers and garbled
    bytes) goes with its results and its thinking.  What the
    model sees is then what it found, once - not dozens of copies of the same thoughts to go on from (GLM-5.3 had
    "Hmm - hold on: I realize something important." 543 times in an 88K prompt, 89% of its own text repeats).  Each
    message depends only on the ones before it, so the next request's prompt starts the same.  None: no loop,
    nothing to leave out, or a conversation above COLLAPSE_MAX_CHARS (whose prompt must not change, #702)."""
    if not echoes(messages):
        return None
    try:
        limit = int(os.environ.get("MAYA_LOOP_GUARD_COLLAPSE_MAX") or COLLAPSE_MAX_CHARS)
    except ValueError:          # (a setting that is not a number: the default, not a failed request)
        limit = COLLAPSE_MAX_CHARS
    if limit and conv_chars(messages) > limit:
        return None          # a long conversation keeps its prompt whole: the engine's cache stays valid (#702)
    out, seen, made, paras, rounds, i, last = [], set(), set(), 0, 0, 0, None
    while i < len(messages):
        m = messages[i]
        if m.get("role") != "assistant":
            out.append(m)
            i += 1
            continue
        m = dict(m)
        for f in ("reasoning_content", "content"):
            if not isinstance(m.get(f), str):
                continue
            kept = []
            for p, sep in _units(m[f], f):
                k = _key(p)
                if len(k) >= ECHO_MIN[f] and (k in seen or talk(k)):   # (and its talk about the loop: no work)
                    paras += 1
                    continue
                seen.add(k)
                kept.append(p + sep)
            m[f] = "".join(kept).strip()
        keys, j = _round_keys(messages, i)
        gists = [k[:2] + (_gist(k[2]),) for k in keys]
        again = all(k in made for k in keys) or gists == last and "sleep" not in str(keys)
        if keys and len(keys) == len(m.get("tool_calls") or []) and again:
            # a round that repeats: the call and its result go, and the thinking around them (the loop's own
            # talk); text the model wrote to the user, if new, stays
            rounds += 1
            if _text(m.get("content")).strip():
                out.append({"role": "assistant", "content": m["content"]})
            i = j
            continue
        made.update(keys)
        if keys:
            last = gists
        if _text(m.get("content")).strip() or (m.get("reasoning_content") or "").strip() or m.get("tool_calls"):
            out.append(m)
        i += 1
    return Collapse(out, paras, rounds) if paras or rounds else None


# ------------------------------------------------------------------------------------------------ keep going
TALK = re.compile(r"(?i)loop.?guard|break(ing)? out|stop (repeating|re-running|theorizing)|act differently|"
                  r"change (approach|something real)|identical (output|result|nc|grep|command|call)|"
                  r"(been|keep|kept|am) (looping|stuck|repeating)|repeating myself|in a (reasoning )?loop")
KEEP_MAX = 3            # an answer that ends the turn with nothing new in a looping conversation goes on: at most
#                         this many times in one answer (then it ends - a model that only talks must not be pushed on
#                         until the token cap)


def talk(text: str) -> bool:
    """The model's own words about its loop ("OK - the loop guard is right. Break out ...") - no work in them."""
    return bool(TALK.search(text))


def said_before(messages: list[dict]) -> set[str]:
    """Every piece of the model's reasoning and text in the conversation (stops_short's `seen`)."""
    return {_key(u) for m in messages if m.get("role") == "assistant"
            for f in ("reasoning_content", "content") for u, _ in _units(m.get(f), f)}


def stops_short(text: str, seen: set[str] = frozenset()) -> bool:
    """Does an answer that ends the turn (no tool call) in a conversation that repeats itself end it with nothing new:
    talk about its loop, or what it already said (`seen`) - or nothing at all?  GLM-5.3 ended its turn with "OK -
    the loop guard is right. ... Break out: use what I have or answer the user." - and the agent stopped; then with
    its "What I actually know" list once more.  Anything new in the answer - a finding, an answer to the user, one
    line of it - may end the turn: "30.10.2026" said to a user who asked for the date is not nothing."""
    return not any(u.strip() and not talk(u) and _key(u) not in seen for u, _ in _units(text, "content"))


KEEP_NOTE = ("[The server's loop guard, not the user: do not stop here, the task is not done. Keep working on it: "
             "take the next concrete step - a command, a file to read or change, a test - one you have not run yet.]")
KEEP_STEERS = (
    "The task is not done, so I keep going. The next concrete step I have not taken yet:",
    "Still not done. Something concrete I can do right now that I have not done:",
    "I keep working. Concretely, I will now",
)


def keep_steer(tries: int) -> str:
    """The first words of the model's thinking after it tried to end its turn short (stops_short), worded
    differently each time."""
    return KEEP_STEERS[min(tries, len(KEEP_STEERS)) - 1] + " "

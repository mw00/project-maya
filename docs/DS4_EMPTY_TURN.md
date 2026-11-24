# DeepSeek-V4.1: an empty turn after the thinking (the `THINK_RESTARTS` path)

Investigation record, 2026-11-23. Resolution appended 2026-11-26 (see the end).

## Symptom

An Anthropic `/v1/messages` client saw completed turns with no content, and retried them. The
upstream returned HTTP 200, `stop_reason: "end_turn"`, a single `thinking` block, and no `text`
and no `tool_use` block at all. The client treats that as an empty answer:

```
model "deepseek-v4.1-flash" returned a completed response with no content
```

A second, less common symptom in the same session: the reasoning appeared in the answer's `text`
block, with the literal `</think>` still in it.

Both were reproduced on v1.1.74, `DeepSeek-V4.1-Flash-1017-GGUF/UD-Q8_K_XL` (the official Unsloth
quant), so a pack or weights mismatch is not the cause.

## Evidence

### 1. The relay's record of the responses

From `console_requests` (LLM relay, `/v1/messages` to the llm host), the three empty turns of one
session, with every other turn of that session normal:

| request | time | wall | output tokens | blocks | stop_reason |
| --- | --- | ---: | ---: | --- | --- |
| `da29d46f` | 10:27:43 | 24,342 ms | 1,048 | `thinking` only | `end_turn` |
| `4ba826ba` | 10:28:14 | 26,155 ms | 1,137 | `thinking` only | `end_turn` |
| `82025fa4` | 10:28:41 | 37,089 ms | 2,071 | `thinking` only | `end_turn` |

All three have `first_token_at = NULL`: no text delta was ever emitted. The turns that only used
tools (`text=0`) are distinct: they always carried `tool_use` blocks. These three carried neither.

Each was retried with the same history and the retry succeeded (`5d98ef7e` at 10:28:08 for
`da29d46f`; `663e7c20` at 10:29:19 for `4ba826ba` / `82025fa4`), so the prompt is fine and the
failure is not deterministic input.

### 2. The engine's own log (`/home/llm/project-maya/maya-ds4-q8.log`)

`da29d46f`:

```
maya ds4 serve: request 266: ... 256 generated in 3921 ms
10:27:52 [maya] thinking: an end-of-turn marker inside it is a quote - going on with the thinking (1. of 2)
maya ds4 serve: request 267: ... 419 generated in 7560 ms
10:28:00 [maya] thinking: an end-of-turn marker inside it is a quote - going on with the thinking (2. of 2)
maya ds4 serve: request 268: ... 373 generated in 6560 ms
10:28:07 [maya] done: 1048 tokens in 24 s (58.1 tok/s) (stop, cancel=False), expert cache 90.9% hit
```

`4ba826ba`:

```
request 270: 530 generated
10:28:28 ... (1. of 2)
request 271: 275 generated
10:28:35 ... (2. of 2)
request 272: 332 generated
10:28:40 [maya] done: 1137 tokens in 26 s (54.4 tok/s) (stop, cancel=False)
```

`82025fa4`:

```
request 273: 503 generated
10:28:50 ... (1. of 2)
request 274: 842 generated
10:29:04 ... (2. of 2)
request 275: 726 generated
10:29:18 [maya] done: 2071 tokens in 37 s (58.5 tok/s) (stop, cancel=False)
```

In each case the segments add up to the turn's output exactly (256 + 419 + 373 = 1,048;
530 + 275 + 332 = 1,137; 503 + 842 + 726 = 2,071), so "going on with the thinking" is a
re-prefilled continuation, not one generation. None of the three windows contains a single
`answering:` line: the engine never left the reasoning phase.

### 3. Where it is decided

`serve/server.py`:

```python
THINK_RESTARTS = 2
THINK_NOTE = ("[The server's note, not the user: your thinking was cut off there - an end-of-turn marker written "
              "inside the thinking is a quote, not the turn's end. Do not stop: go on with the thinking, right where "
              "it stopped.]")
```

and in the token loop:

```python
if t in self.stop_ids and parser.state == "reasoning" and not answered and not called and not pending:
    if t not in self.eos_ids:
        continue                       # a marker that is not the engine's EOS: dropped, generation goes on
    if think_resume is None or think_restarts >= THINK_RESTARTS:
        finish = "stop"
        break                          # the third time: end the turn, with nothing in the answer
    think_restarts += 1
    print("... going on with the thinking ...")
```

The guard exists for a real problem: a model that quotes the server's own markers while it reasons
would otherwise end the turn mid-thinking. But the only thing that separates "the model is quoting a
marker" from "the model has finished thinking and wants to end" is whether the token is the
engine's EOS. Both cases reach the same branch, and for the second one the recovery is exactly
wrong: `THINK_NOTE` tells the model to **go on thinking**, so the model cannot produce an answer.
After `THINK_RESTARTS` the server ends the turn anyway, with the reasoning still in the reasoning
channel and zero answer content.

The turn is therefore not empty because the model produced nothing: it is empty because the server
(a) converted two real end-of-turn signals into "keep thinking", and (b) then closed the turn
without a fallback that would put anything in the answer.

## The `</think>` in the answer

Separate turn, same session, `ab7e0eae` at 10:29:42: the response was **one `text` block of 3,233
characters and no `thinking` block at all**:

```
...Let me check these. Let me grep for fetchRecordDetail in detail view, and read TranslationService.
Let me do it.</think>完整根因已清楚。補確認幾個實作細節：...
```

i.e. the whole reasoning, then a literal `</think>`, then the answer. For that to happen the
parser was in the answer phase from the first token, so the reasoning never had a channel to go
to and the close tag degraded to ordinary text. This matches the `token_only_end` switch in
`serve/server.py` (`getattr(self.template, "deepseek", False) and self.think_end_id is not None`):
when it is false, `</think>` is not consumed at the token and the phase is decided by text
matching, which a wrong initial state defeats. `serve/ds41_encoding.py` defines the expected
`thinking_end_token` as `</think>`.

That block was then replayed verbatim: the next request of the session (`32eda1e1`) carries it as
an `assistant` `text` block, close tag included.

## The client's share

The same relay rows say the client (the agent harness driving this session) sends the model's
previous reasoning back as **plain `text`**, not as a `thinking` block. Over the 35 requests of
that session:

- requests whose `forwarded_payload` differs from `original_payload`: **0 of 35** (the relay is a
  pass-through; it neither strips nor rewrites anything);
- `thinking` blocks anywhere in the request history: **0**;
- assistant messages carrying two or more `text` blocks (reasoning + answer): **252**.

For example the first assistant turn of the session is `text(428 chars, "Let me understand the
task: ...")` followed by `text(19 chars, "我先查看維護記錄頁面...")`, both `thinking`-less `text`.
One later turn even re-sends raw `<tool_call>ssh_exec<arg_key>alias</arg_key>...` XML inside a
`text` block.

So the conversation the model is given has no reasoning/answer boundary in it at all, and its own
earlier reasoning looks like ordinary assistant text. That is the most likely origin of "the model
wrote the answer inside the reasoning and then stopped": it is imitating the shape of its history.

## Why changing the model pack did not help

Nothing in the path above depends on the weights. The failure is in the control-token handling
(reasoning/eos classification, `THINK_RESTARTS`, phase initialisation) and it is fed by how the
history is replayed. The same engine reproduces it on `UD-Q8_K_XL`.

## Related work upstream

The guard this report describes is not new, and neither is its symptom. The lineage, as it reads
from the tracker:

- **#613** - the end of thinking is a token for DeepSeek-V4.1, and a thinking budget exists.
- **#640** - an agent that reasons *about* the server's own stop markers died mid-thinking: the
  stop scan ran in the reasoning channel too, and the first quoted marker's token ended the
  engine at its EOS.
- **PR #709** (merged) - the fix for #640, and the code under discussion here: `close_reasoning()`
  plus `token_only_end` (DeepSeek only), the marker skipped inside the reasoning, and at the
  engine's own EOS "the server goes on from the quote with a note as the server's (at most
  twice)". `THINK_RESTARTS = 2` and `THINK_NOTE` are that change.
- **#786** (closed, fixed in v1.1.50) - the `in_think` conditions required `think_budget > 0`, so
  with `thinking_budget: 0` the reasoning was never tracked, `</think>` was fed as text, and every
  reply ended with `content` and `tool_calls` empty, all of it in `reasoning_content`, with the
  "(1. of 2)" lines in the log. The reporter's summary is the same sentence as ours: *"An agent
  client sees a turn with no answer and no tool call."* The fix (dropping `think_budget > 0` from
  `in_think`) is present here: `in_think = bool(thinking) and self.think_end_id is not None   #
  (tracked with no budget too, #786)`.

- **PR #605** (merged) - the loop guard's own version of this rule: *"If a stop comes before any
  answer text (a model with no end-of-thinking token), the answer is one line saying the model kept
  repeating itself. It never ends empty."* Its before/after table lists the same symptom as the
  "before" column (the rest of the thinking and a literal `</think>` shown as the answer). So
  "never end empty" is already a stated principle in this codebase - it is just not applied on the
  `THINK_RESTARTS` path, which does `finish = "stop"; break` with nothing to show.

So the empty turn is a **residual case of #640's fix**, not of #786's: the budget-0 trigger was
fixed, but the restart path itself still ends the turn with nothing when the model's EOS is a
genuine end rather than a quote. #709's note says "at most twice" as a recovery for a quoting
model; it does not say what the turn should contain when the recovery is exhausted.

Two notes from the searches, for whoever picks this up:

- Only three issues in the tracker touch this machinery (#613, #640, #786), and none of them
  mentions an empty turn as the outcome of the restart budget running out; #786 reached the same
  symptom by a different cause and is closed.
- The open PRs around this area (#1118, #1144, #1158, #1159, #1162) are offload, prefill and
  all-reduce work - no overlap.

## Suggested fixes

1. **Never end an empty turn.** At `think_restarts >= THINK_RESTARTS`, close the reasoning and
   emit what the model wrote as the answer (`close_reasoning()`), or fail the request with a
   clear error. An HTTP 200 with a `thinking` block and nothing else is indistinguishable from a
   successful empty answer.
2. **Tell "quoting a marker" apart from "finished thinking".** `_THINK_BOUNDARY` already scores a
   sentence boundary before the marker; use it (and whether an answer body was already written) to
   decide between skipping the marker and ending the turn.
3. **The note points the wrong way in the second case.** If the turn is being closed because the
   model stopped, `THINK_NOTE` should not be the only steering; a variant that asks for the answer
   when the reasoning looks complete would let the model out.
4. **Phase initialisation.** Make sure the stream parser starts in the phase the prompt's last
   token implies (`thinking_start_token`), so reasoning cannot be emitted into `text` and `</think>`
   cannot leak into the answer.
5. **Client side.** Replaying reasoning as `thinking` blocks (or at least separating it with the
   close tag) removes the in-context cause; replaying it as two adjacent `text` blocks teaches the
   model the wrong shape.

## Open points

- The three empty turns come from one session on one day; a wider scan of `console_requests`
  (`response_status = 200`, `stop_reason = end_turn`, `output_tokens > 0`, no `text` delta) would
  say how often this happens.
- Whether `THINK_RESTARTS` can be reached with a *quoting* model at all (the case it was written
  for) is not measured; the log shows 216 quote events on this host, of which three ended empty.

## Reproduction material

Local, not part of the repository:

- relay database: `D:\workspace\LLMRelayService\data\llm-relay.db`, table `console_requests`
  (request ids above; `original_payload` / `forwarded_payload` / `response_payload` hold the raw
  Anthropic payloads);
- inference log: `llm:/home/llm/project-maya/maya-ds4-q8.log` (search
  `an end-of-turn marker inside it is a quote` and the `request NNN` lines around it).

## Resolution (2026-11-26)

A blind second pass over the relay's full session record (1,105 requests, 2026-11-22..24)
confirmed the single origin both symptoms share, and the fixes below were applied and verified.

### Root cause, both symptoms

The client harness (DSH, route `llmr` → anthropic-messages → relay → this server) replays the
model's earlier **reasoning as plain `text` blocks** (0 `thinking` blocks in 1,105 requests; 252
assistant messages with ≥2 adjacent `text` blocks). Its anthropic serializer converts an unsigned
`thinking` block to `text` unless the route sets `allowEmptySignature` - and this server sends no
Anthropic signature, so every block was converted.

`ds41_encoding.py` then renders each replayed assistant turn with `reasoning_content=""`: an
**empty thinking part plus the end-of-thinking token**, followed by the reasoning as ordinary
content (`rc` went into the `text`). The model reads its own history as *"assistants here close
their thinking immediately and reason in the answer"* - and imitates it:

- **Empty turns (A)**: the model writes its tool calls *inside the thinking* and ends the turn
  there. The parser never parses calls in the reasoning channel, `THINK_RESTARTS` runs out, and
  the turn broke with nothing in the answer (6 confirmed samples; all six thinkings end with a
  complete call). 198 of 992 requests (20%) entered the restart path at least once.
- **`</think>` in the answer (B1)**: the model closes its thinking at once, the reasoning lands
  in `content`, and the end-of-thinking token written a second time leaks into the answer text
  verbatim (46 turns on 11-23: a `reasoning` block plus an 8-char tag piece in `text`). 11
  further turns had *no* `thinking` block at all - reasoning from the first token, the same
  imitation. The parser's phase initialisation was correct all along; the *model* wrote the
  close first. (This corrects the "Phase initialisation" reading above.)
- **B2**: the last restart's continuation re-enters with the reasoning closed; an imitation
  repeated there puts the tag into the content-state text the same way.

`KEEP_THINKING`/`drop_thinking` is a red herring on this route: with tools in the request the
effective `drop_thinking` is always false, so reasoning renders for every replayed turn either
way. `prefix_find` and slot restore were checked and exonerated (68 restores, 0 violations).

### Fixes

Server (this repository), the token loop in `Service.run`:

1. **Never end an empty turn** - the `THINK_RESTARTS`-exhausted branch now closes the reasoning,
   appends the end-of-thinking token to the continuation prompt, strips the thinking-budget
   sampling fields, and drains the continuation generator to its `SEGMENT` sentinel (set
   `seg["next"]` *before* draining: the generic stop branch fires first otherwise).
2. **A tool call written in the thinking is read out of it** - when the turn still ends with no
   answer, the accumulated reasoning is re-parsed with `new_parser(False)` and its last complete
   call is emitted (skipped when the request blocked that tool); otherwise the turn ends with the
   `THINK_STOP_NOTE` text, never with nothing.
3. **The end-of-thinking token is no one's text** - `if t == self.think_end_id and
   parser.token_only_end: text = ""` in *both* phases, not only at the close: a model that closes
   twice (or the last restart's continuation, which starts with the reasoning closed) would
   otherwise put the tag verbatim into the answer (fixes the 46-turn B symptom).

Tests: `serve/test_loop_guard.py` `ThinkingMarkers` - `test_the_resume_is_capped`,
`test_a_whole_thinking_turn_is_never_empty`,
`test_a_call_written_in_the_thinking_is_read_out_of_it`,
`test_no_resume_a_call_at_the_thinking_s_end_is_read_out_of_it`,
`test_a_second_think_end_in_the_answer_is_dropped`. Suite `serve.test_loop_guard` +
`serve.test_server`: 233 tests OK.

Client (no source change needed): the same DSH build already offers the pi-ai compat switch;
the route profile now sets

```yaml
compat:
  allowEmptySignature: true
```

so unsigned `thinking` blocks replay as `{"type": "thinking", "signature": ""}` instead of
`text`. `frontend.anthropic_to_messages` already maps such blocks to `reasoning_content`, and
the template renders them as the thinking each replayed turn had. The prefix-binding check only
strips *signed* stale thinking, so empty-signature blocks survive compaction by design. A
replayed literal tag inside assistant content was already neutralised by `mark_literals`
(spelled as ordinary text, never the special token), so the malformed-call fallback in
`frontend.py` needs no extra escaping.

Ops: `DS4_SLOT_TRACE` and `MAYA_DS4_KEEP_THINKING` removed from `maya-ds4-35.json` (the former
served its diagnosis; the latter is a no-op with tools, measured).

### Verification

After the server fixes were deployed (11-24): 89 relayed requests, `ok` 89, `note` 0, `empty` 0.
Post-`allowEmptySignature` production behaviour is to be watched over the next sessions; the
restart rate itself is expected to fall as the history the model sees stops teaching the
immediate-close shape.

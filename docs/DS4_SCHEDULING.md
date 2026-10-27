# DeepSeek: scheduling several conversations

With `--parallel N` (N > 1) the DeepSeek engine answers several conversations at once on a layer split. What a
user feels then is less how fast a token is than **what runs next**: one prompt's read takes every GPU, so a
250-token question behind a 16,000-token prompt waits for all of it, and a nearly finished answer waits for a read
that has just begun. This note is how the engine decides, and the measurements each rule came from - including the
ones that did not work. The settings are in [SETTINGS.md](SETTINGS.md#deepseek-several-conversations-at-once-parallel).

All numbers: a 3-GPU split (2x RTX 5090 + RTX 3090), DeepSeek-V4.1-Flash GSQ-RCO IQ2_XXS (~138 GB, about half the
experts in VRAM, the rest on the CPU lane from RAM and the disk), `--parallel 4`, greedy, 32-token answers unless
said otherwise. `tools/ds4/schedule_bench.py` runs them.

## The loop: events in (priority, id) order

The serve loop is one event queue (`src/ds4/ds4_evt.hpp`, the [evt](https://github.com/mrsirg97-rgb/evt) engine
of [rig](https://github.com/mrsirg97-rgb/rig), ported): an event has a priority and an id from a counter, the queue
pops the highest priority first and, among equals, the lowest id, and one thread - the loop's - runs them one at a
time. Anything may add an event; only the loop's thread touches the conversations' jobs, so nothing in them locks.

| Event | Added by | Priority |
| --- | --- | --- |
| a line (GEN, STOP, QUIT) | the reader thread | highest |
| choose the next prompt to read | a line that admitted a request, a request ended | middle |
| a conversation's token is ready | a host callback on the head GPU's stream after the window's token (`Model::multi_on_done`) | lowest |

So every request that came is admitted before a read is chosen among them, and a token is taken the moment it is
ready - in the order the GPUs finish them, not the order they were issued in.

**Owning the GPUs.** A prompt's read, and the expert cache's look at its tiers, need every GPU and nothing of the
decode in flight. They ask for them (`want_devices`): from then on a conversation whose token comes back has its
next one parked, the ones already issued come back as events, and once none is out the owner runs; then the parked
ones are issued again. An owner asked for inside another (a boundary during a read's gap) runs after it. This one
shape replaced the loop's flags and its `drain()` - which waited for each conversation's last event on the head GPU
and assumed the GPUs were idle then.

**A request past the N states** waits for a lane (it used to be answered with ERR).

## Which prompt is read next: a piece each, in turn

One read at a time. By default the prompts being read take turns a **piece** each - a chunk for each of the split's
pipeline stages (3 chunks on three GPUs: every stage busy) - and a prompt that has a piece left goes to the back of the
line. A prompt whole in one piece joins the line ahead of the longer reads (behind the other one-piece prompts): it is
through in its turn, so a 256-token question during a 100,000-token read waits for the piece being read, not a round of
them - all 16 of its answer tokens came before the long read was through (none did without it), and the agent workload
below came out the same with and without it. Before each piece the conversations answering get a token each. Nothing
is guessed: no prompt's size is weighed against another's beyond "one piece or more", a short one is through in its
turn, and a long one moves forward every turn instead of waiting for a gap in the short ones.

On an agent workload (`tools/ds4/agent_bench.py`: 4 agents of 8 turns at once - a 4,000-12,000-token first prompt, then
tool results of 150-2,000 tokens, one in ten 5,000-20,000 - answers of 64-400 tokens, 0.5-5 s of tool time between):

| Read order (same build, adaptive gaps unless said) | First token mean / p95 (s) | Turn done mean / p95 (s) | Wall (s) |
| --- | ---: | ---: | ---: |
| first come, first served (`arrival`) | 17.3 / 38.7 | 32.1 / 58.3 | 292 |
| highest response ratio, a read waiting while an answer would end first | 20.6 / 58.7 | 36.9 / 85.5 | 330 |
| before the events (`--parallel` alone, fixed 50% gaps) | 7.6 / 25.2 | 31.4 / 53.7 | 278 |
| a piece each, fixed 50% gaps | 7.6 / 25.5 | 31.4 / 53.7 | 278 |
| **a piece each (default)** | **6.5 / 17.2** | **29.6 / 46.2** | **266** |

Agents' prompts are mostly short continuations, so a prompt's size says little about how soon its agent is unblocked;
taking turns serves them all at once, and the adaptive gaps keep the answers running between the pieces.

`MAYA_DS4_READ_ORDER=ratio` reads the **highest response ratio** next instead: `(wait + read) / read`, `read` the
prompt's tokens left at the last read's measured rate - waiting alike, the shortest goes first, and a long prompt's
ratio grows as it waits, so a stream of short ones does not starve it. It wins where a few prompts of very different
sizes arrive at once; four requests sent together, longest first, against `arrival` (32-token answers):

| Prompt (tokens) | First come, first served: first token / done (s) | Highest response ratio: first token / done (s) |
| ---: | ---: | ---: |
| 250 | 28.4 / 30.3 | 2.9 / 3.9 |
| 1,000 | 25.8 / 30.2 | 7.7 / 8.9 |
| 4,000 | 22.1 / 30.1 | 13.4 / 14.4 |
| 16,000 | 11.0 / 15.2 | 24.4 / 25.4 |
| mean | 21.8 / 26.4 | 12.1 / 13.2 |

## Between a read's chunks: shortest remaining work first

At every chunk boundary of a read the question is: read on, or let the running answers decode? The engine answers it
with **the shortest remaining work first**:

- the read's remaining time: its tokens left at its measured rate (the read's own so far, from its progress);
- an answer's: its expected tokens left at a decode round's measured time (a round: a token for every conversation
  answering). Its expected length is the mean of the answers ended so far (its `max_new` at most; none ended: its
  `max_new`), or - once it has passed that - as many tokens again as it has (the longer it has run, the longer it is
  likely to run);
- the answers go first while the soonest of them would end before the read does.

These are the **decode gaps** (a round at least, each round measuring them again, 4000 ms at most). On the four
requests above, read by response ratio, with the 250-token request's answer at 200 tokens:

| Gaps | Mean first token (s) | Mean done (s) | All done (s) |
| --- | ---: | ---: | ---: |
| fixed 50% (`MAYA_DS4_PREFILL_GAPS=50`) | 10.9 | 24.9 | 28.8 |
| shortest remaining work first (default) | 16.2 | 18.6 | 30.7 |

The rule minimises the mean time to a whole answer - an agent waits for the whole answer before it acts on it -
at some cost to the first token of the prompts after a long answer.

## Answering with the drafter

A conversation that drafts (`spec`, or the engine's `--dspark on`) answers in **rounds**: the one-conversation path's
round on its lane - the DSpark drafter's drafts while their confidence holds, verified in one window, the thinking
budget as there. A round asks for every GPU (`want_devices`) the way a read does, so it takes its turn with the reads
and the other drafting conversations; a read's gap gives each one a round. The drafter's mailbox requests are numbered
per conversation, and a conversation's verify window uses its own lane's buffers, so none touches a paused read's
staging. Exact: a drafting conversation answers what it answers alone, at 2 and 4 lanes.

On the real traffic below the drafter is most of the win: the same build without it took 882 s.

## On real agent traffic

`agent_bench.py --trace` replays rig's `.served` log: the busiest 10 minutes of four conversations (73 turns, a turn's
new tokens 1-20,653, median 235; one conversation 217,222 tokens deep), with their real gaps between turns. Against
v1.1.28 as it runs today - one conversation at a time, the drafter on, conversations kept:

| | v1.1.28, one at a time | `--parallel` alone (before the events) | this: a piece each + drafter rounds |
| --- | ---: | ---: | ---: |
| first token, mean / p50 / p95 (s) | 9.19 / 4.80 / 33.7 | 2.63 / 1.73 / 7.73 | **1.92 / 1.38 / 6.00** |
| turn done, mean / p50 / p95 (s) | 14.2 / 8.24 / **36.3** | 22.5 / 9.78 / 83.5 | **13.1 / 6.05** / 48.1 |
| wall (s) | **496** | 874 | 509 |

The first token comes 4.8x sooner on the mean and 5.6x at p95; a turn is done 27% sooner at the median. The same
work in about the same wall time - the GPUs are as busy as one conversation keeps them, the turns no longer wait in a
line. The cost is the longest turns: p95 done 48.1 s against 36.3, a long answer sharing the GPUs instead of having
them.

## A short prompt that comes during a long read

A request that arrives while a read is paused at a boundary is admitted; a prompt of one chunk may be read in its own
lane in windows of 16 right there - but windows read a prompt several times slower than chunks, so only where that is
**measured** quicker than its turn: its windows at what a prompt window took here (not measured yet: its first window,
which measures one), against the rest of the read and its own read at the read's rate. A prompt waiting that fits a
chunk asks again at every boundary.

| | 4,000-token prompt sent during a 16,000-token read | 256-token prompt sent during a 100,000-token read |
| --- | --- | --- |
| before (windows whenever it fits a chunk) | read in windows: 54 s | - |
| default | read after it in chunks: first token 24.6 s from the long one's send | read in windows: all 16 answer tokens (33 tok/s) before the read was through |

## What did not work

Kept here because each one looked right:

| Tried | What happened | Why |
| --- | --- | --- |
| Cutting short reads into 4 chunks (so they have gap points, as GLM's `MAYA_GLM_PREFILL_MIN_CHUNKS`) | a 4,000-token read ~12 s instead of ~4 s; four 2,000-token reads 27 s instead of 16 | a chunk streams every expert its rows route to - about as many for 1,000 rows as for 4,000 - so where they do not all fit in VRAM a cut read streams them again a chunk. Off by default (`MAYA_DS4_PREFILL_MIN_CHUNKS`); where every expert is in VRAM a chunk's cost follows its rows |
| Fixed gap shares of 50, 20, 10% | the reads took as long at every share; lower shares only starved the answers (a 250-token answer done at 13, 31 s) | the cost was the chunks', not the gaps' |
| Shortest first, aged by "waited longer than its own read" | prompts arriving during a long read were read in arrival order | every short prompt behind a long read has waited longer than its own read: all aged, so all first come, first served. The response ratio replaced it |
| The highest response ratio by default, a read waiting while an answer would end first | on the agent workload the worst of all: first token 20.6 s on the mean (7.6 before the events) | an agent's next prompt is short and comes right after its answer; holding reads for answers and weighing sizes kept the long reads waiting in pieces. A piece each replaced it |
| A drafting conversation's drafts and verify windows in flight beside the others' windows (the drafter on its own streams, rollback buffers a lane) | exact, drafting 3 ms a round instead of 75 - and no faster: 4 agents x 2 turns in 74.9 s against 72.3 in rounds | about half of GSQ's experts are on the CPU lane, which every conversation's windows share; the rounds already keep it busy. Where every expert is in VRAM it may pay |
| Choosing the read when a request is admitted | shortest first made no difference: the first request's read began before the others were admitted | the choice is an event of its own now, below the lines |

## Bugs found on the way

- **The PCIe lane's pulls numbered per conversation.** The fetch kernel's flags are its GPU's and wait for a value
  equal to the pull's number; with several conversations each numbered its requests from 1, so two on one GPU found
  the flags already set and copied experts before they were read (NaN rows, an illegal address a few tokens later -
  only when two prompts arrived together, while both numbers were small). A pull's number is now
  `seq * conversations + conversation`, one conversation's `seq` exactly.
- **A 512-2047-token prompt through windows at a sixth of the speed** on a fresh engine (one conversation too): the
  choice of path modelled a window from its experts' bytes and left out its fixed costs; it now uses what a full
  window measured.

## Reproducing

    python tools/ds4/parallel_test.py --pack <pack> --lanes 4             # exactness, stop, slots, reading
    python tools/ds4/parallel_test.py --pack <pack> --lanes 4 --measure   # + the throughput table
    python tools/ds4/schedule_bench.py --pack <pack>                      # the response ratio and gap tables
    python tools/ds4/agent_bench.py --pack <pack> --exe build/maya-ds4    # the agent workload (--spec: drafting)
    python tools/ds4/agent_bench.py --pack <pack> --exe build/maya-ds4 --trace <rig .served> --spec
    python tools/ds4/agent_bench.py --pack <pack> --exe <v1.1.28 build> --trace <rig .served> --serial

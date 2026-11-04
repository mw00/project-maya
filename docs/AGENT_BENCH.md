# A benchmark shaped like coding agents: `tools/agent_bench.py`

`./maya.sh --bench` answers "how fast is one request on this PC": three short questions and two cold prompt reads,
one at a time. Coding agents load a server differently, and settings that only matter under that load (`--parallel`,
the expert tiers, the batched decode, reading a prompt while others decode, keeping a system prompt for the next
conversation) show nothing there. `tools/agent_bench.py` runs a scripted agent workload against a running server, so
two settings, two builds or two machines can be compared on the same requests.

## What it sends

Everything is fixed by `--seed`, so every run sends the same requests:

- `--clients` conversations at once (default 4). Each starts with one of a few shared system prompts (30K, 8K and 4K
  tokens by default) and a first message of 2-6K tokens.
- Then turns: each appends new text, as a tool result or a message would, and asks for an answer of 32-1024 tokens.
  80% of the new texts are under 2K tokens, 5% are 8K-64K. The next turn comes 2-20 s after the answer (the agent
  running its tools).
- Checks: at `--checks-per-hour` (40), a 30K-token system prompt with a fresh 2-4K-token question and a 16-token answer,
  as an agent's permission or routing check does.
- Two cold bursts, at a third and two thirds of the run: half the conversations start over and are read again with
  nothing reusable past the system prompt, as after an agent restarts.
- A conversation that grows past `--max-context` starts a new one from its latest turn, as an agent compacts.

The model's answers are not fed back: the next turn sends the scripted history, so the prompts do not depend on what
the model wrote. Requests are greedy with thinking off and ban the model's end tokens (`--ban auto` knows
GLM-5.3-Flash by the served model's name; `--ban none` or token ids otherwise), so every answer writes its whole
max_tokens and every run writes the same number of tokens. The texts are slices of this repository's files
(`--corpus <folder>` for others).

## Running it

Start the server the way you use it (`./maya.sh` with your config, `--parallel` as you run it), then:

```
python tools/agent_bench.py run --url http://127.0.0.1:8080 --clients 4 --label mine --out runs/mine-1.json
python tools/agent_bench.py plan --clients 4          # what it would send, without a server
```

A run takes `--duration` (900 s) plus the answers in flight. The first `--warmup` seconds (180) are left out of the
summary: the server's caches fill with the conversations then. For an A/B, restart the server before each run (a run
leaves its conversations in the server's caches), alternate the settings (A B A B), use the same `--seed`,
`--clients`, `--duration` and `--warmup`, and run each setting at least twice:

```
python tools/agent_bench.py compare runs/base-*.json runs/new-*.json
```

`compare` prints each metric's mean and standard deviation over a label's runs and marks a difference larger than
twice its standard error with `*`. A difference without the mark is within the runs' own spread.

It works against any OpenAI-compatible server (the requests are streamed and timed by the client). A Maya server adds
its own figures from `GET /metrics`.

## What it reports

Over the window from `--warmup` to `--duration` (a request counts when it ends inside it):

| metric | what it is |
|---|---|
| `output_tok_s` | tokens written per second by all conversations together |
| `in_flight` | requests in flight on average (with think times, fewer than `--clients`) |
| `decode_median`, `decode_p10` | per request: tokens per second from its first token to its last, what an agent sees; slowed by the other conversations' steps and reads (answers of 50 tokens or more) |
| `latency_median`, `latency_p90` | a turn from its send to its last token |
| `ttft_median`, `ttft_p90` | a turn's time to its first token: the read of its new text and any wait behind other reads |
| `decode <n>K+` | the decode median by the request's context length |
| `read <n>+` | prompt reads by their number of new tokens (not reused from the server's cache): new tokens per second of read time - the server's own read time when it sends `timings` (Maya, llama.cpp's server), else the time to the first token |
| `vram_hit_per_request`, `vram_hit`, `ram_fetch`, `promo` | Maya with experts in RAM: the routed experts found in VRAM per request (median), and the tiers' readings over the run (VRAM hit, RAM-tier experts fetched over PCIe a token, promotions a token) |

The run's JSON keeps every request's times, sizes and the server's records, for anything the summary does not show.

## Reading the results

- `output_tok_s` follows the workload as much as the server: with think times, a server twice as fast does not write
  twice as many tokens, because the conversations wait for their tools. Compare `decode_median` and the latencies for
  speed; `output_tok_s` and `in_flight` show how the load itself moved.
- Reads by new tokens and decode by context length compare like with like when runs drift apart (a faster run gets
  further into its conversations, so its contexts are longer).
- With experts in RAM, runs of the same setting differ more than with every expert in VRAM (where experts end up
  depends on timing): expect a few percent between reps.

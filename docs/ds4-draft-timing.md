# DeepSeek serving draft timing

DeepSeek `STAT` lines expose existing DSpark measurements alongside `drafts` (the accepted/offered ratio):

- `draft_rounds`: drafter calls counted per requesting sequence. A batched call serving several sequences counts each sequence, rather than one scheduler step.
- `draft_ms`: cumulative host-measured drafter time, in milliseconds.
- `verify_ms`: cumulative host-measured verification forward time, in milliseconds.

With serial serving, values cover the current request's decode, excluding its prefill. With `--parallel`, they cover the engine's current busy interval across requests; they are not attributed to individual requests. All three reset with their reporting interval and are Prometheus **gauges**, not lifetime counters. The server preserves them in its engine statistics and exports `maya:engine_draft_rounds`, `maya:engine_draft_ms`, and `maya:engine_verify_ms` from `/metrics/prometheus`.

These reuse existing timers without adding device synchronization. They measure the regions already timed by the engine: they are not GPU-only timings, and `draft_ms + verify_ms` is not a complete decode wall-time breakdown. For example, verification's accept/reject sampling is outside its forward timer. Use output tokens and decode wall time for throughput, and these measurements to help interpret drafter cost versus verification work.

The fields do not change proposal selection, sampling, or model arithmetic. Sampled proposals remain controlled by the existing `DS4_DSPARK_SAMPLE=1` setting.

## Turns by how many answers shared them (`--parallel`)

With `--parallel`, every `STAT` line also carries totals since the engine started, by turn (#1352):

- `dec_tok_<n>`, `dec_ms_<n>`: the tokens written in rounds of `<n>` answers at once (accepted drafts included) and those rounds' time, for `<n>` from 1 to `--parallel`;
- `pp_tok_<n>`, `pp_ms_<n>`: the prompt tokens read in pieces with `<n>` answers running beside them, and those pieces' time (`0`: no answer running). A piece a STOP or an error ended is not counted.

`STAT` comes every 8 tokens written, and once more (`active=0`) when the engine goes idle: that line has every turn. The difference between two lines is each situation's speed over that interval: one answer alone (`dec_*_1`), all lanes writing (`dec_*_<parallel>`), a read with no answer running (`pp_*_0`) or beside answers. The times are the turns' own (each round, each piece), so a read's pieces and the answers' rounds between them are counted apart. A round counts under the answers it starts with, so the round where an answer joins or leaves goes to the larger `<n>`, and each answer's first token, sampled as its read ends, is in neither timer. A long read alone reaches the counters in one step, with the next `STAT`. `/metrics` has them with the other `STAT` keys (`tiers.now`); `/metrics/prometheus` exports them as counters with an `answers` label: `maya:decode_tokens_total`, `maya:decode_seconds_total`, `maya:read_tokens_total`, `maya:read_seconds_total`, so `rate()` gives the speeds.

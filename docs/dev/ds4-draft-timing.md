# DeepSeek serving draft timing

DeepSeek `STAT` lines expose existing DSpark measurements alongside `drafts` (the accepted/offered ratio):

- `draft_rounds`: drafter calls counted per requesting sequence. A batched call serving several sequences counts each sequence, rather than one scheduler step.
- `draft_ms`: cumulative host-measured drafter time, in milliseconds.
- `verify_ms`: cumulative host-measured verification forward time, in milliseconds.

With serial serving, values cover the current request's decode, excluding its prefill. With `--parallel`, they cover the engine's current busy interval across requests; they are not attributed to individual requests. All three reset with their reporting interval and are Prometheus **gauges**, not lifetime counters. The server preserves them in its engine statistics and exports `maya:engine_draft_rounds`, `maya:engine_draft_ms`, and `maya:engine_verify_ms` from `/metrics/prometheus`.

These reuse existing timers without adding device synchronization. They measure the regions already timed by the engine: they are not GPU-only timings, and `draft_ms + verify_ms` is not a complete decode wall-time breakdown. For example, verification's accept/reject sampling is outside its forward timer. Use output tokens and decode wall time for throughput, and these measurements to help interpret drafter cost versus verification work.

The fields do not change proposal selection, sampling, or model arithmetic. Sampled proposals remain controlled by the existing `DS4_DSPARK_SAMPLE=1` setting.

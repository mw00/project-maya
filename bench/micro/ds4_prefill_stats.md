# DS4 prefill expert statistics

Set `DS4_PREFILL_STATS=1` when starting Maya to print one `DS4_PREFILL` line per
participating GPU after each successful prefill run. This is optional diagnostic
output; it does not change routing, cache policy, or model arithmetic.

Fields:

- `gpu`: device ID.
- `tokens`: sum of input chunk lengths for this run.
- `chunk`: configured chunk capacity, not necessarily the length of every chunk.
- `expert_row_picks`: token-to-expert assignments across the layers and chunks
  executed by this device.
- `vram_row_picks`: assignments to experts resident in VRAM during this run.
- `unique_expert_uses`: nonempty expert slots counted once per layer/chunk.
  Reusing an expert in another chunk counts again.
- `vram_unique_expert_uses`: those uses served by resident expert slots.
- `streamed_weight_bytes`: delta of the engine's existing expert payload byte
  accounting, including initial flow prefetch. This is not a hardware PCIe
  measurement and does not measure storage reads or page-cache misses.

The row-weighted resident hit rate is `vram_row_picks / expert_row_picks`.
The unique-use resident hit rate is
`vram_unique_expert_uses / unique_expert_uses`. Treat an empty denominator as
unavailable. One hot expert can serve many rows, so these ratios can differ a lot.
A nonresident use does not necessarily imply a disk read: host RAM and the file
page cache may supply the weights.

Both batch streaming and file-backed flow (including layer-major flow) are
covered. Slots lent to prefill buffers count as nonresident. Tensor-parallel
members report their own counters; their routing counts may duplicate logical
work, so summing row counts across members is not a model-wide logical hit rate.
Byte counters remain per device.

Collection uses the routing bounds already synchronized and copied to the host;
it adds no GPU readback or synchronization. It does add CPU counter work when
enabled. No inference speedup is claimed.

Build with `STRATA_BUILD_TESTS=ON`, then run
`ctest --test-dir <build> -R '^ds4_prefill_stats_test$' --output-on-failure`.
The CPU test covers weighted versus unique hits, repeated chunks, borrowed
resident slots, invalid bounds without partial updates, and 64-bit totals.

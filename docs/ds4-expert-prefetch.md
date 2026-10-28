# DeepSeek expert read-ahead

Maya normally hints every supplied expert range on Linux. On Windows it checks eight working-set samples and skips ranges that look resident. These defaults are unchanged.

Set `DS4_PF_SAMPLES` in the saved config to try a different probe count:

```json
{
  "env": {
    "DS4_PF_SAMPLES": "4"
  }
}
```

| Value | Behavior |
| --- | --- |
| Unset | Always hint on Linux; eight probes per range on Windows |
| `0` | Always hint, without checking residency |
| `1` through `16` | Probe that many pages per range; skip the hint when every probe looks resident |

A positive count enables sampling by itself. Restart Maya after changing it. The engine validates the effective count before loading weights and logs it once. Empty, signed, fractional, whitespace-padded and out-of-range counts are errors. The count is cached for the engine process.

## Compatibility and limits

- The existing nonzero `DS4_PF_ALL` override still forces unconditional hints and takes precedence over `DS4_PF_SAMPLES`. When force-all is active, the count is not parsed. Legacy zero or an unrecognized legacy value does not override the count or change platform defaults.
- The existing `DS4_IQ_PF_ALL` override still bypasses filtering at its IQ call site. The startup count describes the shared policy, not this separate override. Clear both force-all overrides when comparing sample counts.
- One probe checks only the start of a range. Any finite count can miss a cold page, and residency can change after a probe. Failed queries retain the hint. A missed cold page still loads the same bytes on demand, but the request can take longer.
- Read-ahead can affect residency-dependent behavior. With `MAYA_DS4_DISK_SKIP_W` enabled, an expert's residency can determine whether Maya omits it. Changing hints can therefore change the experts computed and the answers. Execution placement can also change floating-point results. Matching output hashes in the experiment below are evidence for those requests, not a general identical-output guarantee.
- Full CPU-expert RAM copies can bypass this file-backed work. These controls do not change Engram prompt prefetch. A startup policy log does not prove that filtering is on a request's critical path.

## Why Linux sampling is opt-in

The platform split in [#275](https://github.com/mw00/project-maya/pull/275) was a cost decision: Linux advice for cached ranges measured cheaper than checking residency on the tested Core i5-12600T. Windows page walks cost more, so filtering paid there. [#425](https://github.com/mw00/project-maya/pull/425) extended filtering to the IQ CPU-expert path on Windows and left Linux unchanged; it did not benchmark Linux.

A later Linux experiment on a Threadripper PRO 3945WX, 123 GiB RAM and four RTX 3090s found a benefit with DeepSeek-V4.1-Flash GSQ-RCO IQ2_XXS and DSpark. Three fresh-engine runs per arm compared always-hint against four-probe filtering, with 12 CPU threads, mmap experts on, effective TP1, 4K chunks and 262144 context. Median decode improved from 41.01 to 45.57 tok/s for a 128-token prompt/256-token answer and from 39.46 to 42.77 tok/s for a 50K-token prompt/512-token answer. The 50K request's total time improved 2.26%, from 45.59 to 44.56 seconds, because prefill was unchanged. Matched generated-token hashes and speculative-draft counts agreed.

Those measurements used an earlier four-probe implementation, not this PR against current main. Benefit has been demonstrated only on that system and those workloads. Four is a starting point, not a measured optimum. Other systems, models, counts and cache conditions can be slower. Compare matched requests on your own system before keeping a count.

CPU-only Linux tests cover defaults and overrides, count boundaries, invalid settings, cached parsing, failed residency queries, warm/cold ranges and demand reads after a missed cold page. The file-cache test evicts only its own temporary fixture, never model files or the host's global page cache. Native Windows execution and current-branch model performance remain unverified.

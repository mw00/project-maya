# Refreshing DS4 startup expert placement

On a three-GPU AMD machine, refreshing the startup expert profile raised held-out
128-token code generation from **11.3 to 18.0 tokens/s** on the first response and
**14.6 to 23.9 tokens/s** on the repeated response. First-response expert VRAM hit
rate rose from **42.3% to 75.5%**. This changes which experts start in the existing
VRAM slots; it adds no VRAM and changes no model weights, kernels, or routing decisions.

The persisted profile accumulates counts. In this experiment, 919,740,960 historical
selections overwhelmed 975,360 recent calibration selections. The tool replaces that
large historical influence with a per-layer blend: recent counts plus a historical
prior normalized to one quarter of the recent total (approximately 80% recent,
20% historical after rounding). It preserves the recent count scale so future saved
observations can influence the profile. Live adaptive cache replacement remains on;
this is a startup/persisted-profile improvement, not a replacement for live learning.

## Relationship to existing profile tools

Upstream already records and saves DS4 expert counts. `tools/ds4/profile.py make`
normalizes and combines saved profiles with equal input weights and a configurable
output count scale; `compare` checks overlap in their hottest experts. Those tools
remain useful when each input already represents the workload you want to retain.

This utility handles calibration runs that **start from an accumulated profile**:
it subtracts each run's starting counts before combining recent observations, then
adds a separately normalized historical prior. Passing cumulative snapshots directly
to `profile.py make` would retain the historical selections inside every snapshot.
Starting calibration from empty profiles and using the existing tool is another
option. This utility adds no automatic aging or change to runtime defaults.

## Safe calibration recipe

This tool handles the DS4 engine's **raw little-endian uint32 count array**, not the
STRP ranking files handled by `make_profile.py`. Supply the actual saved geometry:
trunk-only or trunk plus MTP rows, with the model's experts per row. Files must all
have exactly that geometry. The experiment used 40 rows of 384 experts and no drafter.

1. Back up the original profile while the engine is stopped. Work on copies.
2. Copy the same starting profile to `before.bin` and `calibration.bin`. Start the
   engine with `--expert-profile calibration.bin`; run representative requests.
   Shut down normally so counts are saved, then copy it to `after.bin`.
3. For an independent second run, reset `calibration.bin` to `before.bin`, repeat
   calibration, and save `after2.bin`. Do not combine overlapping cumulative deltas.
4. Write a new profile and its SHA-256 provenance receipt:

```sh
python3 tools/refresh_ds4_profile.py --layers 40 --experts 384 \
  --historical historical.bin --pair before.bin after.bin \
  --pair before.bin after2.bin --recent-weight 0.8 --out refreshed.bin
```

5. Compare fresh engine starts using original versus refreshed copies on held-out
   prompts. Reverse test order to check warming effects. Select the new profile
   explicitly with `--expert-profile` only after validation; retain the old profile
   for rollback. Engine saves will mutate the selected profile, so keep the seed copy.

The tool refuses existing output/receipt paths, decreasing counters (reset/wrap),
missing observations in any layer, incorrect file sizes, and uint32 overflow.
An unobserved historical layer contributes no prior. Calibration includes prefill
and decode counts; use short prompts and longer outputs for a decode-heavy profile.
No automatic refresh or runtime default changes are included.

## Measurement scope

Calibration used three synthetic tasks (LRU code, database tutorial, observatory
story), twice, each with a 128-output-token primer and two 256-token responses.
About 94% of calibration tokens were decode. Confirmation used different code,
technical, and prose prompts, 128 output tokens per response, greedy decoding,
with normal adaptive caching and unchanged GPU/host budgets. Initial tests ran old
then recent; confirmation reversed the order. The reversed confirmation was:

| Held-out prompt | First response old → recent (tokens/s) | Repeat old → recent |
| --- | --- | --- |
| Asyncio rate limiter | 11.27 → 18.00 | 14.59 → 23.87 |
| Virtual-memory tutorial | 15.29 → 20.69 | 20.11 → 24.51 |
| Colony trade dispute | 14.43 → 17.23 | 19.81 → 21.53 |

Hardware was two 32 GB R9700s plus a 16 GB 9070 XT, DeepSeek-V4.1-Flash-1017
EXL3 3.5 bpw, TP=2 on the main pair with third-card expert caching, drafting off.
A subsequent real API smoke test gave 17.8/22.3 tokens/s on first/repeated code.
The historical/recent blended profile SHA-256 values were respectively
`ae12f5039f192289595267d8d87502ac4f010d14a80f4a173239890c3076bbf0` and
`419ff834edd0952e6ebc2dc260d436590e31b17ecede0b9d5bc8941aa2f743ab`.

These are small-sample, workload-specific results, not universal speed or quality
claims. Ranking and initial cache heat changed together. Free-running text can differ
as CPU/GPU expert assignments change; there is no cross-profile logit-parity claim
or broad quality evaluation. A single 7,935-token prompt also prefilled faster
(21.0 → 17.1 seconds), but traffic did not decrease, so no prefill mechanism or
repeatable prefill gain is claimed.

# GLM: edit a short turn after restoring a conversation slot

## Problem and change

This extends the turn-start checkpoint case addressed by #253 to a conversation
restored from a slot. In `prep_prompt`, the turn-start checkpoint block originally
ran **before** `slot_take`. A slot restore then invalidated the previous
conversation's checkpoints. The restored conversation therefore missed the
checkpoint at the start of its new short turn. Replacing that turn could read the
entire archive again.

Move the existing block after slot restoration and `ckpt_drop_past(reuse)`, before
prefill. Both disk and RAM `slot_load` populate the snapshot and restore its live
state; the existing `ckpt_save_snapshot` can preserve it at this point. Existing
`ckpt_join` synchronization protects the snapshot until that copy finishes.
No checkpoint/slot serialization or model arithmetic is changed by this patch.

## Reproduce on an isolated server

Use one sequence (`--parallel 1`), text input, an empty dedicated slot directory,
normal slot/checkpoint defaults, and no other client traffic. Leave
`STRATA_GLM_NO_REUSE` unset; `STRATA_GLM_CKPTS` and `STRATA_GLM_CKPT_TURN` must not
be zero. Start the unpatched or patched server yourself, using the **same** model
and settings for each run. Do not point this probe at a busy production server.

```sh
# Set MAYA_API_KEY in your environment if the isolated server requires auth.
python tools/repro_glm_slot_turn.py --base-url http://127.0.0.1:8080 \
  --model YOUR_SERVED_GLM_MODEL --rows 1024 --expect baseline --output before.json
# Restart the isolated service with the patched engine and a fresh slot directory.
python tools/repro_glm_slot_turn.py --base-url http://127.0.0.1:8080 \
  --model YOUR_SERVED_GLM_MODEL --rows 1024 --expect fixed --output after.json
```

The stdlib-only probe builds deterministic synthetic archives with different A/B
facts and performs four requests:

1. A: read archive, short acknowledgment.
2. B: read a different archive, evicting A from the live model state.
3. A: restore its slot and ask a short question about the middle fact.
4. A: **replace** that last user message, discard its answer, and ask for the
   start/end facts. The earlier history is byte-for-byte the same.

It checks answers and `timings.cache_n` / `timings.prompt_n`, not a
hardware-specific latency threshold. Baseline mode requires zero reuse on the
edit; patched mode requires retaining at least the restored prefix and reading
fewer than 512 new tokens. An unexpected answer, missing timings, or a test that
never exercised reuse fails. Increase `--rows` to test larger contexts, staying
within the server's limit; rows are **not** a token count. Each output is a fresh
file containing timings, response text and request hashes, never the API key.

The new probe's request construction and SSE handling are covered by six CPU
unit tests (including mocked baseline/fixed HTTP runs). The new standalone probe
itself has **not** been GPU-run on current main. The GPU measurements below came
from the earlier, larger harness, not from this four-request script.

## Measured before/after

Same 23-request harness and frozen synthetic archives, serial runs, fresh
independent slot directories, greedy requests (`temperature=0`). Large archive
requests used thinking off and 256–512 output tokens. Some short tool checks used
low reasoning, 1,536 tokens; these are bounded tool regressions, not an open-ended
reasoning benchmark. One run per arm; no statistical confidence interval.

| Case | Before | After |
|---|---:|---:|
| Edited turn: first output | 248.810 s | **4.934 s** |
| Edited turn: whole response | 251.781 s | **7.486 s** |
| Edited turn: reused prompt tokens | 0 | **248,595** |
| Edited turn: newly read tokens | 248,672 | **77** |
| 129K cold prompt: first output | 119.717 s | 119.845 s |
| 248K cold prompt: first output | 244.630 s | 244.731 s |
| Sum of 23 request durations, excluding startup | 999.449 s | 754.935 s |

The affected edit saves 243.876 s (98.02% of first-output wait). The full workload
saves 24.46%; that percentage depends on how often an application edits after a
conversation switch. Cold prefill does not get faster.

Each arm passed 23/23 requests and 81 answer/tool assertions: authoritative facts
versus outdated distractors, absent data answered as null, cross-conversation
isolation, repeated edits/requests, automatic and forced tools, nested Unicode
arguments, archive-derived tool arguments, replaced tool observations,
cancellation and recovery. The 22 complete responses had identical answer text,
reasoning text and tool function/argument strings (random tool-call IDs excluded).
The deliberately cancelled stream had a different cutoff; both recovery checks
passed. This is bounded regression evidence, not a general quality guarantee.

[Machine-readable per-request results](results.json) contain the selected metrics,
all assertion results and output hashes; no private prompts, local paths or keys.

### Hardware and exact scope of measurements

- Windows 11, build 26100; Ryzen 9 5900X, 128 GB RAM.
- One RTX 3090 24 GB, driver 581.57, CUDA 12.8, native `sm_86` build.
  `nvidia-smi` reports PCIe maximum Gen4 x16, idle Gen1 x16; an under-load
  negotiated-link sample was not captured with these measurements.
- Slot files: Samsung 990 EVO Plus 2 TB NVMe; model pack: Samsung
  MZVL21T0HCLR 1 TB NVMe.
- GLM-5.3-Flash Maya-M pack with **alesha runtime projection enabled in both
  arms**, 45 layers, scale 1. These were **not stock-model measurements**.
  That local projection patch is not part of this PR.
- Baseline: v1.1.21 (`451f9cb`) + local projection port; candidate: exactly that
  baseline + the checkpoint ordering change. Both compiled successfully on
  Windows (151/151 build steps). Projection logs confirm prefill and decode hooks.
- This PR is based on main `25deec5` (after v1.1.24). The complete `generate.cpp`
  file was unchanged from v1.1.21 on that base. Slot load and checkpoint-copy
  implementations were also unaffected by the intervening upstream changes.
  **No current-main CUDA build/GPU rerun, stock-model A/B, HIP runtime, multi-GPU,
  RAM-slot run, or long soak is claimed here.** Maintainer GPU gates still apply.

Engine arguments (model path replaced by a placeholder):

```text
--glm-pack <Maya-M-pack> --max-context 262144 --expert-cache auto
--prefill auto --kv-streaming on --parallel 1 --vram-reserve-mib 1500
```

Environment, identical apart from fresh per-arm file locations:

```text
STRATA_GLM_RAM_HEADROOM_GB=16
STRATA_GLM_ABLATE=<alesha 45-layer vector>
STRATA_GLM_ABLATE_LAYERS=45
STRATA_GLM_ABLATE_SCALE=1
STRATA_GLM_SLOT_DIR=<isolated per-arm directory>
STRATA_GLM_USAGE=<isolated per-arm usage file>
STRATA_GLM_CPU_CAL=<isolated per-arm calibration file>
```

No RAM-slot override, MTP/DFlash, KDA-graph override or checkpoint override was
used. Defaults: four disk slots, 16 GB slot budget, four checkpoints and 16,384
checkpoint interval. Selected native diagnostics are in
[engine-excerpts.txt](engine-excerpts.txt).

## Compatibility and future rebases

- No new flag, public API, CUDA call, checkpoint structure, slot format, cache
  budget, model projection, routing policy, or server protocol change.
- Keep this ordering invariant on future rebases: **restore the selected state →
  invalidate incompatible checkpoints → preserve a valid turn-start snapshot →
  prefill**. In particular, do not preserve another conversation's snapshot or
  create a checkpoint only to clear its metadata immediately after a slot load.
- Existing guards remain: one sequence, fast path, reuse enabled, text only,
  checkpoints enabled, snapshot rather than checkpoint restore, and a turn marker
  less than 512 tokens after the reused position. `STRATA_GLM_CKPT_TURN=0` keeps
  the previous opt-out. `ckpt_drop_past` and failure/reset behavior remain intact.
- This is not persistent storage of *all* historical checkpoints. Editing older
  history before a slot restore, changing system/tool prefixes, evicting a slot,
  or restarting without retained cache can still require substantial rereading.
- #387 addresses shared system-prefix slots with multiple sequences; this patch
  addresses recent-turn checkpoints with one sequence. They touch the same
  function. Preserve each feature's guards and rerun both probes when combining.

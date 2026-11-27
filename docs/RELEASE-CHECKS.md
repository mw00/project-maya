# Release checks: what changed, saved answers, the nightly run

Every model Maya runs adds a row to the release checks. Three tools keep a release from getting slower as that list
grows: only the models a change can reach get the full checks, the expected answers are saved once per release instead
of being re-derived, and the long part runs unattended every night, so on release day the table is already there.

| Tool | What it answers | Needs a GPU |
|---|---|---|
| `tools/maya_changed.py <previous release tag>` | which models need the FULL checks, which a SMOKE check, what to build | no |
| `tools/maya_golden.py` | are this build's greedy answers the ones the release gave, and if not, where they part | yes (one `--bench`) |
| `tools/maya_nightly.py` | per machine and model, every night: answers and speed against the previous release | yes |

All three sit on `./maya.sh --bench` (the speed-test standard: quiet machine, GPU lock, warm-up, 3 answers of 256
tokens, the 2K / 8K prompts, greedy, thinking off) and on the config the setup wrote - never a hand-written one.

## 1. What changed -> what to test

```
python tools/maya_changed.py v1.1.76
```

```
changed against v1.1.76: 3 file(s)
FULL   Inkling   (maya-ik2): src/ik2/ik2_main.cu
SMOKE  GLM       (maya): nothing it runs changed
...
build: maya-ik2
Python suites (always): python -m unittest discover -s serve -p "test_*.py" -t .; python -m unittest discover -s tools -p "test_*.py"
```

How a path is mapped:

- **An engine's sources.** Each engine's file list is read from `CMakeLists.txt` (the target's sources, the libraries
  it links) and followed through every `#include "..."`. A file in that list touches that engine - so
  `src/hy4/hy4_lane.cpp` touches Hy4 *and* Inkling (maya-ik2's CPU lane is built from it), and a header under
  `include/maya` touches exactly the engines that include it.
- **An engine's own folder** always touches it: `src/ik2` Inkling, `src/hy4` Hy4, `src/ds4` DeepSeek, `src/mm3`
  MiniMax, and `src/core`, `src/program`, `src/kernels`, `src/prefill`, `src/spec`, `src/ngram`, `src/plan`,
  `src/platform`, `src/artifact` GLM and Qwen (one engine, `maya`).
- **Everything a start runs through** touches all models: `maya.py`, the setup, `serve/`, `tools/` (the start imports
  them), `CMakeLists.txt`, `cmake/`, `third_party/`. Only the build files ask for a rebuild.
- **A Python or data file named after one model** (`tools/hy4/...`, `serve/ds4_vision.py`,
  `data/expert-profile-mm3.bin`) touches that model.
- **Tests, docs, CI files, the Mac engines** touch no model: the changed C++ tests are listed to build and run, the
  Python suites always run. A path the tool does not understand counts as "all".

**FULL** = on every machine and layout of that model: `./maya.sh --bench --compare <previous release>` (exactness
first, then speed) and `maya_golden.py --check`; plus the model's own release items (loop test, vision, long context)
that still need a person - see the last section. **SMOKE** = last night's PASS line for that model, or one
`maya_golden.py --check`.

CI (`tests.yml`) builds no engine - it runs the Python suites and the CPU-only C++ tests on every push, all of them
cheap - so there is nothing there to narrow; the tool's own tests run inside the "Installer and tools tests" step.

## 2. Saved expected answers

```
python tools/maya_golden.py --config maya-inkling-2-mini.json --capture --text-ref v1.1.76 --layout 1x-v100-resident
python tools/maya_golden.py --config maya-inkling-2-mini.json --check --layout 1x-v100-resident
python tools/maya_golden.py --list
```

A capture runs the bench once and stores each request's answer - token ids, their hash, the first words as text - in
`data/golden/<machine>/<model>-<layout>.json`. The folder is per machine and not in git: nothing is shipped for a PC
that is not ours. A check prints which request differs and from which token:

```
saved answers: inkling-2-mini-...-1x-v100-resident.json (v1.1.76 8c618a59 2026-11-24)
  DIFF  decode 2: differs from token 117 (saved 4821, now 993; 256 vs 256 tokens)
  DIFF: 10 request(s) compared, 1 differ, 0 not compared
```

Exit code 0 equal, 1 differs, 2 nothing saved / the run failed.

Two things make a comparison fair, and both are checked:

- **The prompts.** `--bench` cuts its 2K / 8K prompts from the README and docs, which change with every release.
  `--text-ref <tag>` (maya.py: `--bench-text-ref`) cuts them from that tag's docs instead; the set remembers the tag
  and every request's input hash. A request whose input changed is listed as "not compared", never as a difference.
- **Saved per-PC choices.** Some models' greedy answers legitimately depend on a measurement the engine saved on that
  PC (an expert profile, a CPU-lane or route choice in `data/`). A set is keyed by the layout, the config's settings
  and the hash of those files. A check takes the set with the same key; when the files changed it uses the newest set
  and names the files, so a difference reads as "maybe the new measurement - capture again if it is meant".
  An engine that varies between its own starts: capture from a `--compare` run (`--from-tokens`), which keeps every
  answer its rounds gave.

Capture once per release, per machine and layout, on the release commit (`--from-tokens` takes the file a
`--bench --bench-tokens` run wrote, so the release's own compare run can feed it).

## 3. The nightly run

`maya-nightly.json` in the checkout (per machine, not in git):

```json
{"base": "auto", "rounds": 2, "wait_min": 2, "slow_pct": 5, "budget_min": 20,
 "models": [
  {"name": "Inkling", "config": "maya-inkling-2-mini.json", "layout": "1x-v100-resident"},
  {"name": "Hy4 XS",  "config": "maya-hy4-flash-maya-xs.json", "layout": "3x-v100"},
  {"name": "GLM S",   "config": "maya-maya-s2.json", "layout": "3x-v100"}]}
```

The Halo box (Sun) has one more line, for the model that is too large for its card to run the standard in a night:

```json
  {"name": "GLM-5.3 full", "config": "maya-glm-5.3-maya-xs.json", "layout": "1x-v100-smoke", "smoke": true,
   "answers_vary": true, "locks": ["~/hy4/locks/v100"]}
```

`"smoke": true` is the short run (`./maya.sh --bench --bench-smoke`): the model loads, a warm-up and two answers of 64
tokens, greedy, checked against the saved ones - no previous release's build, no 2K / 8K prompts, no speed verdict
(about 3.5 minutes there). A build that cannot answer, or answers nothing, is an `ERROR` line. Save its answers once
per release with
`python tools/maya_golden.py --config maya-glm-5.3-maya-xs.json --capture --smoke --layout 1x-v100-smoke`
(`--check --smoke` by hand), on a second start: the first start on a PC measures it and answers differently.

`"answers_vary": true`: on that box the full model reads most experts from the disk while it answers, and which
experts the card holds at a given token follows those reads - so its greedy answers are not the same on every start
(measured: 1 to 3 of the 3 requests differ between two starts of one build, from token 0 to 33). Saved answers that
differ are then written in the line ("2 of 3 equal") and are not a `DIFF`; the line is there to prove that the model
loads and writes an answer. Read the answers' first words in `maya-nightly-glm-5.3-full.tokens.json` ("heads") when
in doubt.

`"base": "auto"` is the newest release tag before the checkout's commit. `"locks"` (optional, per model) names lock
files of the box that the bench does not take itself, e.g. `["~/hy4/locks/v100"]` for the V100 on the Halo box.

```
python tools/maya_nightly.py --dry-run     # the plan, the time budget, the commands: nothing started
python tools/maya_nightly.py --once        # tonight's run, now
```

Per model: quiet check (GPU, CPU load, heat flag, locks) -> `--bench --compare <base>` with the prompts of `<base>`
-> the saved answers from the same run -> one line in `maya-nightly-results.txt`. At the end
`maya-nightly-status.txt` holds the headline (`PASS` / `FAIL` / `PARTIAL` / `NOT RUN`) and a line per model:

```
PASS | 2026-11-24 | Jupiter | 8c618a59 vs v1.1.76 | 2 of 2 model(s) ran | 9.4 min
  PASS    Inkling (1x-v100-resident) 251 s
  PASS    Hy4 XS (3x-v100) 312 s
```

- `DIFF`: the answers differ from the previous release, or from the saved ones (the line says which request, which
  token). `SLOW`: decode or a prompt size is `slow_pct` or more below the previous release - on two runs in a row: a
  slow run with equal answers is run again once (`"retry_slow": false` turns that off), because a job that starts
  beside the run, or model files gone cold in the page cache, slow one build's first round only (seen on the first
  night: -28% on identical code, gone on the next run). `ERROR`: the run stopped (its output:
  `maya-nightly-<model>.log`).
- `BUSY`: the machine was not free for `wait_min` minutes (another job on the GPU, a held lock, CPU load 2 or more,
  the heat flag); the model is skipped and nothing was started. `SKIPPED`: it no longer fits the night's budget. The
  budget is printed first, from each model's last run time.
- Each model's run leaves `maya-nightly-<model>.log` (the bench's output) and `maya-nightly-<model>.tokens.json` (its
  answers): `maya_golden.py --check --from-tokens <that file>` re-checks them, `--capture --from-tokens` saves them,
  both without a GPU.
- Exit code 0 all PASS, 1 a DIFF / SLOW / ERROR, 3 nothing ran.
- The first night after a release builds the previous release's engines once (`.bench-builds/<commit>`); build them
  ahead (`./maya.sh --bench --compare <tag>` once per model) to keep that out of the night's budget.
- The nightly does not update the checkout. Run `git pull --ff-only` and the build before it (the lines below do).

Scheduling (shown here, not installed by anything):

Linux, cron (`crontab -e`), 03:30 every night:

```
30 3 * * * cd $HOME/project-maya && git pull -q --ff-only && cmake --build build -j 32 >> maya-nightly-cron.log 2>&1 && .venv/bin/python tools/maya_nightly.py --once >> maya-nightly-cron.log 2>&1
```

Windows, Task Scheduler (one line, from an administrator prompt; adjust the folder):

```
schtasks /Create /TN "Maya nightly" /SC DAILY /ST 03:30 /TR "cmd /c cd /d C:\maya && git pull -q --ff-only && cmake --build build --config Release && .venv\Scripts\python.exe tools\maya_nightly.py --once >> maya-nightly-task.log 2>&1"
```

## How a release uses it

1. `python tools/maya_changed.py <previous tag>` -> the FULL / SMOKE plan, the builds, the suites.
2. Read last night's `maya-nightly-status.txt` on each machine. A model that is SMOKE in the plan and PASS there is
   done. A FULL model that is PASS there has its compare and saved-answer rows done for that machine's layouts.
3. Run by hand what the night did not cover: FULL models on layouts that are not in a machine's list, anything BUSY
   or SKIPPED, and every DIFF / SLOW line (investigate before calling the candidate good).
4. After the release: `maya_golden.py --capture --text-ref <new tag>` per machine and layout (or `--from-tokens` from
   the release's own compare run), so the next nights check against the new release.

Two models, one engine: GLM-5.3-Flash and the full GLM-5.3 (744B) are both run by `maya`, and `maya_changed.py` shows
them as one GLM row - but the full model has its own loader and layer code (`src/core/glm_dsa_*`), so a Flash PASS
says nothing about it. v1.1.75 to v1.1.77 shipped with the full model failing every request (#1200) because only Flash
was run. Whenever the GLM row is FULL, both of these have to be green before the candidate is good:

- the full model's smoke on the Halo box's V100 (the `"smoke": true` line above: last night's PASS, or
  `maya_golden.py --check --smoke` by hand);
- the full model's fast path on the synthetic pack, on any GPU box, no model files needed (build with
  `-DMAYA_BUILD_TESTS=ON`): `ctest -R "glm_dsa_bind_widths_test|glm_dsa_pack_fast_test"` - both must run, a
  "skipped" (no device, or the fixture tools are missing) is not a pass.

What still needs a person:

- A `DIFF` that is meant (an answer-changing fix): decide, then capture again.
- The loop test, vision, tool calls, long context (32K and the deep points), the dashboard - the bench's requests do
  not cover them.
- Sampling: the saved answers are greedy; nothing here checks the sampler.
- Machines without a nightly yet, AMD on Windows, and any layout a machine's list does not name.
- The release decision itself.

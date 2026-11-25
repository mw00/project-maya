# Hy4-Flash Maya-S on 2x Quadro RTX 5000 16 GB: pipelined two-GPU prefill (`HY4_PP=1`) reads an 8K prompt +18% and a 14.7k served prompt +28%, byte-identical, decode unchanged

2026-11-25. Host: Xeon W-2133 (6 cores / 12 threads), 94 GB system RAM, 2x Quadro RTX 5000 16 GB (sm_75,
TU104, PCIe 3.0 x16) as the layer split (`CUDA_VISIBLE_DEVICES=0,1`). The box has a third GPU, a Quadro
P4000, which these runs did not use. System RAM holds the experts the two 16 GB cards cannot — this is the
tiered regime (experts stream from RAM over PCIe during a prompt), not a fully-resident one. Model:
Hy4-Flash, Maya-S (IQ2_S), the pack's `maya-hy4-mtp`. Engine: `main` @ v1.1.80 + this PR (`HY4_PP`), source
build, **sm_75, CUDA 12.9**.

The auto layer split depends on `--max-context` (a larger context reserves more KV room on GPU 0, so fewer
layers land there):

- **Identity run (`--max-context 32768`):** GPU 0 layers 0-22 + the head and MTP block, GPU 1 layers 23-41.
- **Official bench and served run (`--max-context 131072`):** `INFO layers by GPU: GPU0:0-15 GPU1:16-41`.

`HY4_PP=1` reads a prompt of two chunks or more as a two-stage pipeline: GPU 0 reads chunk *k+1* while GPU 1
reads chunk *k*. Off by default; this folder is its A/B against the serial path on the same binary.

Run config: the upstream-default 2-GPU Hy4-Flash Maya-S config, `--max-context 131072 --chunk 4096 --kv q8
--mtp`, GPUs 0,1. The two arms differ only by `HY4_PP` (the A/B configs set `"HY4_PP": "0"` vs `"1"`). `HY4_PP_KEEP_MB=64` was set on the PP
arm as conservative headroom on the 16 GB cards; the PP log reported 573 MiB free *after* the 192 MB `hx`, so
the 256 MB default (which asks for `hx + 256 MB` free) would also have fit — PP was not blocked by the
default here.

## Correctness: byte-identical logits

Identity run (`--max-context 32768`, split 0-22 / 23-41). At the deterministic floor
`HY4_HOT=0 HY4_Q4X=0 HY4_ROUTE=stream HY4_LOGIT_HASH=1`, `HY4_PP_KEEP_MB=64`, `CUDA_VISIBLE_DEVICES=0,1`, a
14,737-token prompt with `--gen 16`. `--chunk 4096` and the engine's keep-the-last-full-chunk rule make the
chunks **4096, 4096, 2449, 4096** (`pp_chunks()` keeps the final full chunk for the wanted rows).

| | INFO / PP line | `HY4_LOGIT_HASH` |
| --- | --- | --- |
| `HY4_PP=0` | `INFO PP: off (HY4_PP=0)` | `7b78060e69373198` over **22** rows |
| `HY4_PP=1` | `INFO PP: on (stage 0 = GPU 0 layers 0-22 + head, stage 1 = GPU 1 layers 23-41, hx 192 MB own allocation, host stage-1 buffers 403 MB)` · `PP: forward n=14737 chunks=4 pipelined, 27721.5 ms` | `7b78060e69373198` over **22** rows |

Same hash, same row count; the 16 generated token ids are identical (`cmp` SAME). Re-run on the final build of
this branch (the one with the handoff / vision / GPU-count gates): the same two INFO lines, `PP: forward
n=14737 chunks=4 pipelined, 27640.7 ms`, `7b78060e69373198` over 22 rows in both arms, token ids `cmp` SAME.
The speed tables below were measured before those gates were added; on this machine none of them fires (a
tested peer copy both ways, two GPUs, no vision tower), so the measured path is the same. (`HY4_LOGIT_HASH` folds
every `want`-row of logits a window produces; the pipelined head is fed to the same hook as the serial
window, so the two match exactly.)

## Prefill and decode: the official `maya.py --bench`

`maya.py --bench` (a warm-up answer, then 3 greedy 256-token answers, and prefill at 2048 and 8192 tokens),
one request at a time, split 0-15 / 16-41 (`--max-context 131072`). Prefill @ 8192 is two chunks, so `HY4_PP`
engages; prefill @ 2048 is one chunk and is the control. The serial arm's first round was `maya.py`'s own
first-start PC measurement (it printed "run `--bench` again for the speed number") and is excluded; both PP
rounds ran on a measured PC. **This leaves one valid serial round against two PP rounds.** At 8192 the serial
round's three samples span 350-356 tok/s and the PP rounds' six span 409-421, so the gain is well outside the
spread; the 2048 control and decode read as flat, but on one serial sample.

| arm | round | decode, 256-tok answers (tok/s, median [min-max]) | prefill @ 2048 (control, tok/s) | prefill @ 8192 (PP, tok/s) |
| --- | --- | ---: | ---: | ---: |
| `HY4_PP=0` | 2 (valid) | 15.9 [13.9-16.2] | 257 [250-258] | 356 [350-356] |
| `HY4_PP=1` | 1 | 15.3 [13.1-16.3] | 239 [232-241] | 418 [409-418] |
| `HY4_PP=1` | 2 | 16.6 [14.4-16.9] | 258 [257-258] | 421 [419-421] |

**Prefill @ 8192: 356 -> 418/421 (mean +17.8%, +18%).** Prefill @ 2048 (one chunk, no PP) and decode are
unchanged within the runs' spread (PP1's first 2K round, 239, is the low end of that spread against the single
serial 257).

## Overlap (nsys): why it is faster

`nsys` on the serial 14.7k-token read showed the two stages alternating per chunk, each GPU waiting on the
other — GPU 0 idle 71.5% of the window, GPU 1 idle 35.4%; with `HY4_PP` on, GPU 1's idle fell to **10.0%**
over a 26.3 s window. This trace was taken on **our local live build on a 14/28 split** (a different build and
split from the A/B above), so it is motivation, not a measurement of this PR's binary; **the raw trace is not
in this PR**. The A/B tables above are the evidence.

## A larger served prompt (our `bench.ts`, not the `--bench` procedure)

For scale, one served request through the running server (`serve/server.py`, same config, the `HY4_PP=1`
build), measured by our own client (`bench.ts`) — **not** `maya.py --bench`. The request was 14,772 tokens
(3 reused from the session + **14,769 read**):

| | read speed (tok/s) | PP line |
| --- | ---: | --- |
| `HY4_PP=0` | 338.5 | — |
| `HY4_PP=1` | 433.4 (**+28%**) | `PP: forward n=14768 chunks=4 pipelined` |

The client (`bench.ts`) also asks a recall question about the long prompt; both runs gave the same correct
answer. That is our client's check, not upstream evidence.

## The gain scales with the number of chunks

Two stages pipelined over chunks `c_0..c_{K-1}` (stage 0 and stage 1 each taking time ∝ `c_k`) finish in
`c_0 + T` where `T = Σc_k` (the first chunk fills the pipe, then stage 1 runs behind stage 0 with no
starvation for these sizes), against `2T` serial. So the **equal-stage ceiling is `2T / (T + c_0)`**:

- 8K prompt, chunks 4096,4096: `2·8192 / (8192+4096) = 1.333` → **+33%**.
- 14.7k prompt, chunks 4096,4096,2449,4096: `2·14737 / (14737+4096) = 29474/18833 = 1.565` → **+56.5%**
  (not +60%: +60% would need four equal 4096-token chunks).

Measured: +18% of the +33% ceiling (8K) and +28% of the +56.5% ceiling (14.7k). Neither reaches its ceiling
because stage 0 also runs the head (so the stages are not equal) and fill/drain cost more at small *K*.

## sm_61 note

Built for `sm_75` only. Upstream v1.1.80 does **not** compile for `sm_61` (`src/hy4/hy4_kernels.cu:3247`,
`__nanosleep` is undefined below sm_70) — unrelated to this PR, which touches no kernel. (The box's third
GPU, a P4000, is sm_61 but it was not used in these runs.)

## Commands a reviewer can run

```sh
# build (sm_75)
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMAYA_ENABLE_CUDA=ON -DMAYA_NATIVE_EXPERTS=ON \
      -DMAYA_BUILD_TESTS=OFF -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build --target maya-hy4 -j

# a token-ids file with upstream's tokenizer (any long text; the identity check needs only the SAME ids
# in both arms). Set GGUF to the model .gguf, or TOKENIZER to an extracted tokenizer/ directory.
GGUF=/path/to/model.gguf          # or: TOKENIZER=/path/to/tokenizer  (use Tokenizer.from_dir below)
python3 - "$GGUF" ids-14k.txt <<'PY'
import sys
sys.path.insert(0, "tools")
from strata_tokenizer import Tokenizer
tk = Tokenizer.from_gguf(sys.argv[1])          # or Tokenizer.from_dir("/path/to/tokenizer")
ids = tk.encode(open("README.md", encoding="utf-8").read() * 40)[:14737]
open(sys.argv[2], "w").write(",".join(map(str, ids)))
print(len(ids), "ids")
PY

# byte identity: the two HY4_LOGIT_HASH lines must match (same hex, same row count)
PACK=/path/to/Hy4-Flash-Maya-S/maya-hy4-mtp
for pp in 0 1; do
  CUDA_VISIBLE_DEVICES=0,1 HY4_HOT=0 HY4_Q4X=0 HY4_ROUTE=stream HY4_LOGIT_HASH=1 HY4_PP_KEEP_MB=64 HY4_PP=$pp \
    build/maya-hy4 --pack "$PACK" --max-context 32768 --chunk 4096 --kv q8 \
    --ids "@ids-14k.txt" --gen 16 2>&1 | grep -E 'HY4_LOGIT_HASH|PP: (on|off|forward)'
done

# speed (official), 8192-token prefill engages PP, 2048 is the control
# the --bench config must NOT set HY4_PP in its own env (a config's env overrides the shell's, serve/server.py),
# or the shell's HY4_PP arm is ignored; leave HY4_PP out of the config and set it from the shell.
CONFIG=/path/to/2-gpu-maya-s.json
for pp in 0 1; do for r in 1 2; do
  HY4_PP=$pp HY4_PP_KEEP_MB=64 python3 maya.py --bench --config "$CONFIG"
done; done
```

## Files

Raw logs are on the measuring host (`~/bench_mem/pr/`): `id-pp0.log`, `id-pp1.log`, the official-bench logs
`bench-pp{0,1}-r{1,2}.log`, and the served-run logs `log-pp0.log` (serial) and `log-pp1.log` (PP) — not in
this folder. The nsys trace (local live build) is not included. The config is the base 2-GPU Hy4-Flash Maya-S
config plus `HY4_PP` in `env`; nothing lossy is turned on, and the deterministic floor pins the route and
disables the adaptive features for the identity check.

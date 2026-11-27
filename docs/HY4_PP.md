# Pipelined prompt prefill over a two-GPU layer split (`HY4_PP`)

On a layer split over two GPUs, Hy4-Flash reads a long prompt in chunks of `--chunk` tokens (4096 by default). For
each chunk the engine walks GPU 0's layers, hands the state to GPU 1, walks GPU 1's layers, then runs the head on the
head GPU - one chunk start to finish before the next begins. The two GPUs take turns and never overlap.

`nsys` on a 2-GPU PC (2x Quadro RTX 5000) reading a 14.7k-token prompt: the two stages alternate per chunk and each
GPU waits on the other - over the serial read GPU 0 is idle 71.5% of the window and GPU 1 idle 35.4%. GPU 1's busy
time is the floor for the read, so overlapping GPU 0's next chunk with GPU 1's current chunk is worth roughly the idle
time; with `HY4_PP` on, GPU 1's idle fell to 10.0% over a 26.3 s window.

`HY4_PP=1` (off by default) reads a prompt of two chunks or more as a two-stage pipeline.

## What runs where

- **Thread A** (the thread that calls `Model::forward`) owns **stage 0**: GPU 0's layers, plus the head and the MTP
  block, for every chunk.
- **Thread B** (`pp_worker`, made on the first pipelined forward and kept, like `eg_crew`) owns **stage 1**: GPU 1's
  layers. It sets its own device.

For chunk `k`, thread A queues stage 0 of chunk `k` and then the head of chunk `k-1`; thread B queues stage 1 of
chunk `k-1` beside it. The head of the last chunk runs after the loop, then the head stream is drained as the serial
path drains it.

`window_run()`'s host part - the embeddings into `h_x`, the Engram lookups, the positions - is split out into
`win_begin()`, which the serial path and thread A both call. `stage s`'s layer loop is `pp_layers(s, k)`, the same
`engram` / `attention` / `ffn` calls the serial loop makes. The head is `head_tail()`, now taking its input state as
an argument (`x_in`) so it can read stage 1's output in `hx`.

## Why the tokens are the same

Every kernel reads the inputs it reads in the serial loop, and each GPU's stream holds its work in the serial loop's
order. Only the host-side interleaving of the two GPUs changes. The GPU-to-GPU hops that the serial path orders with
`cudaStreamSynchronize` are ordered under PP with CUDA events (`ev_s0`, `ev_copied`, `ev_out`, `ev_head`) and a small
host handshake (`pp.mu` / `pp.cv`: the last chunk each stage has reached), so no buffer is reused before its last
reader is done. The result is identical bit for bit - checked with `HY4_LOGIT_HASH` (or `--kld`/`--kld-save`) at the
deterministic settings (see *Testing*).

The host state a chunk in flight reads is double-buffered so the two stages never write each other's:

- the FFN host buffers (`h_ids`, `h_w`, `h_lat`, `h_cpu`, `h_stage`, `h_pmap`, the lane entries and route scalars)
  move into `struct StageHost sh[2]`; a `thread_local` index picks one (`sh[0]` on every thread but `pp_worker`'s).
  Without `HY4_PP` only `sh[0]` is allocated or used - the buffers exactly as before.
- the positions `h_pos` get three pinned slots (chunk `k` uses `k % 3`); the MTP catch-up gets its own.
- the Engram rows get a second set where stage 1 holds an Engram layer.
- the CPU lane (`cpu_experts` / `parallel_for`) is called from both stage threads under a new mutex `pp_lane_mu`,
  taken only while a pipelined forward is in flight; off `HY4_PP` the lane is called from the one engine thread as
  before.

Per-stage expert/lane counters are summed into the model's `Stats` when the forward ends; `end` and `st.tokens`
advance only over the chunks whose heads were confirmed done.

## Buffers and cost

- `hx`: one chunk's trunk output on the head GPU, `chunk x heads x model_dim` floats (about **192 MB** at the
  Maya-S shapes). Its own allocation from the VRAM the memory plan left free - never taken from the decode-time
  placement. PP needs the head GPU's free VRAM to hold `hx` **plus** `HY4_PP_KEEP_MB` (256 MB by default) of
  headroom (`free >= hx + keep`); if less is free, PP stays off with a reason in its INFO line. Lowering
  `HY4_PP_KEEP_MB` only reduces that headroom requirement - it does not shrink `hx`.
- stage-1 pinned host buffers: a second copy of the FFN staging buffers and, where needed, the Engram set - a few
  hundred MB of pinned host memory, sized and logged at the load.

## When it runs

PP runs only when, at the load: `HY4_PP=1`, not tensor parallel, **exactly two GPUs total** (`G.size() == 2`) and
exactly two layer GPUs with the first GPU's layers a prefix of the trunk, the head on the first GPU, **no vision tower
loaded**, no KV spill, and `HY4_PROF` / `HY4_DUMP` / `HY4_INJECT` / `HY4_EG_DUMP` and the DFly2 drafter off; and
**both GPU-to-GPU directions selected a tested non-host transfer** (`peer_setup` chose `kXPeer`/`kXDriver`, not
`kXHost` host staging - PP copies the stage hand-offs directly and has no host-staged path, so where upstream found
the peer/driver copy unsafe, forced off, wrong or IOMMU-affected and fell back to host staging, PP stays off); and
the extra buffers fit. For a given forward it also needs: two chunks or more, every chunk larger than a token window
(a few tokens past a chunk boundary take the serial loop), no picture, no KV spill, the prompt route's rates known
(not being measured), and the 4-bit lane copies settled (`HY4_Q4X=0` lets PP run from the first prompt). Any gate
false: the serial loop, with one `INFO PP: off (<reason>)` note.

`pp_init()` runs at the end of the warm-up, so **`HY4_NO_WARMUP=1` returns before it and leaves PP off with no INFO
line** (the serial loop runs; no `INFO PP:` line appears at all). Use PP with the warm-up on.

The server (`--serve`) asks `Model::pipelines(n)` and, when it is true, hands the whole prompt body to one
`Model::forward` call (instead of one call per chunk), then reads the last token alone as before.

## Limitations

- A pipelined prompt body is read in one `forward` call; it is not cancelled part way (cancellation is checked before
  and after the body, as the serial path checks it between chunks).
- Two stages only (a two-GPU layer split, exactly two GPUs). Three or more GPUs, tensor parallel, a loaded vision
  tower, decode and verify windows keep the serial paths.
- PP transfers the stage hand-offs with a direct peer/driver copy and has no host-staged path, so it stays off where
  `peer_setup` selected host staging (`kXHost`) for either direction.
- The extra `hx` buffer (~192 MB VRAM) and the stage-1 pinned buffers.
- A pipelined chunk's experts are streamed in the same passes as the serial path; the overlap is of the two GPUs, not
  of the within-GPU streaming.
- With `HY4_PP` off, the serial behaviour is retained: only `sh[0]`'s backing buffers are allocated or used, and no PP
  CUDA, pinned-host, event, stream or worker-thread resources are created (the `Impl` struct is larger by the PP
  members).

## Testing

**Byte identity** (upstream-native `HY4_LOGIT_HASH`; `PACK`, `IDS` are paths you set). A prompt whose length is more
than twice `--chunk` makes 3+ chunks, so PP engages; the two hashes and row counts must match.

```sh
PACK=/path/to/Hy4-Flash-Maya-S/maya-hy4-mtp
IDS=ids-14k.txt   # a file of comma/space-separated token ids (see below), > 2 x --chunk long
for pp in 0 1; do
  CUDA_VISIBLE_DEVICES=0,1 HY4_HOT=0 HY4_Q4X=0 HY4_ROUTE=stream HY4_LOGIT_HASH=1 HY4_PP=$pp \
    build/maya-hy4 --pack "$PACK" --max-context 32768 --chunk 4096 --kv q8 --ids "@$IDS" --gen 16 2>&1 \
    | grep -E 'HY4_LOGIT_HASH|PP: (on|off|forward)'
done
# the two HY4_LOGIT_HASH lines must be the same hex over the same row count
```

`--kld REF --kld-save OUT` then `cmp` the two saved files is the equivalent against a reference's logits.

**Making the ids file** with upstream's tokenizer (any long text; the identity check only needs PP0 and PP1 to read
the *same* ids). Set `GGUF` to the model's `.gguf`, or `TOKENIZER` to an extracted `tokenizer/` directory:

```sh
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
```

**Speed.** `python3 maya.py --bench --config "$CONFIG"` before and after on the same binary - prefill at 8192 tokens
(two chunks) engages PP, 2048 does not. The `--bench` config must **not** set `HY4_PP` in its own `env` (a config's
`env` overrides the shell's - `serve/server.py`), or the shell's `HY4_PP=0`/`1` arm will be ignored; set `HY4_PP`
from the shell and leave it out of the config, or use two configs that differ only in `HY4_PP`.

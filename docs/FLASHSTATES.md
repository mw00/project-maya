# Conversation state cache

`MAYA_GLM_STATE_CACHE=1` keeps complete conversation states for GLM, so a conversation that comes back after another
one ran, or after the engine restarted, resumes where it stopped instead of reading its prompt again. It keeps the state at
the end of each completed answer, at the end of each prompt and at marks inside long prompts. A state holds
everything the engine needs to continue: the attention rows, the recurrent and convolution state, the indexer state,
and the drafter's context. GLM on CUDA uses it; it is off by default.

The design follows [FlashStates](https://github.com/FutureMLS-Lab/FlashStates/tree/e2ea9a7864b41081a0147a3968e73bd5c6875a7d)
(exact request-end checkpoints, recurrent state exported from inside a prefill call, and eviction of whole states by
the prefill time they save), adapted to Maya's GLM engine, which keeps one contiguous state arena per conversation
rather than a paged block pool.

The controls work in the config's `env` object or through `./maya.sh --setup --env MAYA_GLM_STATE_CACHE=1`.

## How a state is saved

A state is a list of regions, each split into immutable 64 KiB chunks in RAM or on SSD. A region is either
append-only rows (attention latents and keys, compressed KV sources: the rows a shorter prefix holds never change)
or a whole buffer (recurrent state, sliding windows, open pools), copied whole each time.

A new state reuses the chunks of its nearest saved ancestor for the rows below that ancestor's end, after checking
that the live rows still match: on the GPU by a 64-bit fingerprint per chunk, in host memory by comparing the bytes.
A row the engine computed differently (another conversation that read the same prompt, a recomputation after a
failed load) is copied, never substituted. Only the new rows and the whole buffers are copied, so the cost of a
state does not grow with its length. A state is named by its token prefix; publishing a prefix already held reads
nothing, and the eviction policy decides whether a new state is kept before anything is copied. On SSD each chunk
keeps its fingerprint, and a load that does not match fails as a whole; the engine then reads the prompt again.

The prompt's state is saved when the request ends, after its last token and just before the answer's own state, so
its copy (about 150 MiB of whole buffers on GLM-5.3-Flash) is neither in the time to first token nor in the answer.
A request that comes while it runs waits for it; an agent's next turn, which comes after its tools ran, does not.

## Controls

| Variable | Default | Meaning |
| --- | --- | --- |
| `MAYA_GLM_STATE_CACHE` | `0` | `1`: keep complete conversation states |
| `MAYA_GLM_STATE_RAM_GB` | `2` | RAM for saved states, GiB (their chunks; not the engine's own arenas) |
| `MAYA_GLM_STATE_DISK_GB` | `16` | SSD for saved states, GiB; `0`: RAM only |
| `MAYA_GLM_STATE_MAX` | `128` | Most states kept |
| `MAYA_GLM_STATE_DIR` | the pack's `state-cache` | Private directory, owned by one engine at a time |
| `MAYA_GLM_STATE_DAYS` | `7` | Oldest saved state imported at start; a use refreshes it |
| `MAYA_GLM_STATE_WRITE_THROUGH` | `1` | Write each new state to SSD as soon as it is kept |
| `MAYA_GLM_STATE_MIGRATE` | `1` | At start, import the conversations the older slot cache kept (`MAYA_GLM_SLOT_KEEP`) |
| `MAYA_GLM_STATE_STAGING_MB` | `256` | GPU scratch for states exported inside a prefill call, MiB |
| `MAYA_GLM_STATE_PREFILL_EXPORT` | `1` | GLM: let a prefill call run past a mark and export the recurrent state there; `0`: cut the call at the mark |
| `MAYA_GLM_STATE_LEAF_LAST` | `1` | Evict states that a deeper state continues before each conversation's latest one |
| `MAYA_GLM_STATE_IDLE_SECONDS` | `60` | A latest state unused this long is evicted like an earlier one; `0`: never |
| `MAYA_GLM_STATE_ANCHOR_TIER` | `1` | Keep shared branch points (a system prompt two conversations continue) with the latest states |
| `MAYA_GLM_STATE_SUPERSEDE_LAG` | `2` | Earlier answers of a conversation kept for a rewritten last turn |
| `MAYA_GLM_STATE_COST_D`, `MAYA_GLM_STATE_COST_E` | measured | Prefill seconds of N tokens from position P: `D*N + E*N*(2P+N)/200000`; eviction keeps the states that save the most of it per byte |
| `MAYA_GLM_STATE_COST_CALIBRATE` | `1` (`0` with explicit D, E) | Fit D and E from the first 16 prefills |

The older slot settings (`MAYA_GLM_SLOT_RAM_GB`, `MAYA_GLM_SLOT_GB`, `MAYA_GLM_SLOTS`,
`MAYA_GLM_SLOT_DAYS`) are read as these when the `MAYA_GLM_STATE_*` setting is not given. With the cache on, the server starts a resumed
answer's streaming detokenization from the prompt's last tokens, so its first characters join them correctly.

A prompt identical to a saved state still reads its last token, since a state holds no logits. A reader's state is never
evicted while it loads. A failed SSD write leaves the state usable from RAM; a state larger than the RAM budget goes
straight to SSD.

## What it measured

server60: 4x RTX 3090, Threadripper PRO 3945WX, Maya-S-v2-IQ2_XXS (GLM-5.3-Flash), 1M context, `--kv-streaming on`.
`tools/state_capture_bench.py`: a conversation of N tokens, another conversation, then the first one's next turn.

| N | Next turn's first token, `main` | With the state cache |
| --- | ---: | ---: |
| 8K | 697 ms | 480 ms |
| 64K | 819 ms | 554 ms |

Both runs in the deterministic mode below, two runs each, identical answers. With the cache off, `--bench` is the
same as `main` (decode 84.4 / 83.9 against 84.5 / 86.0 tokens/s; prefill at 2K and 8K within 1%).

## Checks

- `state_cache_test` (CPU): lookup, reuse of an ancestor's rows and copying of rows that differ, eviction and
  admission, SSD restart and corruption, failures. CI runs it on Linux and Windows.
- `state_io_test` (GPU): device transfers, the GPU fingerprint against the host function, and row reuse on the GPU.
- `glm_state_cache_test PACK`: full-vocabulary logits after a restore are bit for bit the logits of reading the prompt,
  after another request ran and after an SSD restart. `glm_checkpoint_export_test`, `dflash_state_cache_test GGUF`.
- `tools/test_native_state_cache.py --engine BINARY --pack PACK`: the engine's protocol (an answer continued, a retry
  after restart, parallel contexts displaced, slot import); `tools/glm_state_fixture.py` makes a small test pack.
- `tools/state_capture_bench.py --config CONFIG --exe A [--exe B] [--deterministic]`: the publish, copy and restore
  counters per prompt length, and an A/B of two builds. `--deterministic` applies the settings for repeatable answers
  (see [Repeatable answers](SETTINGS.md#repeatable-answers)) and fails if any answer differs between runs.

`STATE` lines, `/metrics` (`maya:state_cache_*`) and the dashboard's conversation-cache card report hits, restored
tokens, RAM and SSD bytes, publishes and their time, bytes copied, reused and hashed, restores and their time.

# Experimental GLM agent prefix caching

This builds on the opt-in RAM/disk conversation slots described in
[GLM_SLOT_SPILL.md](GLM_SLOT_SPILL.md). It adds two separately opt-in ways to
retain complete recurrent state for exact token-prefix reuse. It does not rewrite
messages, normalize tool history, or promise a hit for semantically similar text.

## What is retained

- **Learned prefixes:** recent input paths can reveal a sufficiently long common
  prefix at a real divergence. Discovery proposes a boundary; it is not itself a
  cache hit. The engine must actually traverse that boundary and save its state
  before a later request can restore it.
- **Completion tails:** after a normal text `stop` or `length` completion, the
  engine may retain the prompt plus the output tokens already committed to the
  target model. It keeps the original prompt snapshot as well.

The target's actual position defines a completion key. Emitted-but-unfed output
and unaccepted speculative drafts are excluded. A fully fed last token is not
arbitrarily dropped. No extra forward pass is performed just to make a snapshot.
All target shards must agree on the saved boundary. KDA/convolution and DSA state
are saved together; a longer recurrent state is never truncated to simulate an
earlier boundary. Deferred multi-context saves retain the source request's image
identity and wait for a safe context-binding point.

A tool client's next rendered prompt must begin with exactly the saved token
IDs. Changes to reasoning history, whitespace, argument formatting or terminal
tokens can make an apparently equivalent conversation miss. A miss should fall
back to ordinary processing, not change the request to force reuse.

## Evaluation configuration

Both features are **off by default**. The learned path requires the supported GLM
Flash fast path, spill mode, enabled slots and no MTP. Use a private cache directory
for an evaluation: slot files contain retained model state and token histories,
not a tenant-isolated or encrypted storage service.

```sh
export MAYA_GLM_SLOT_SPILL=1
export MAYA_GLM_SLOT_DIR=/path/to/private/evaluation-slots
export MAYA_GLM_SLOT_RAM_GB=80
export MAYA_GLM_SLOT_GB=64
export MAYA_GLM_LEARNED_PREFIX=1
# Optional; does nothing unless learned-prefix caching is also enabled:
export MAYA_GLM_COMPLETION_CACHE=1
# Leave the normal admission threshold unless evaluating it deliberately:
export MAYA_GLM_COMPLETION_MIN_TOKENS=1024
```

Persistent slot and prompt-cache keys include SHA-256 digests of the weight
shards mapped by the loader, together with the parsed expert routing and split
layout. A weight change between loads gives the pack a different key. Entries
created before this key format require a fresh prefill.

Keep model files immutable throughout loading and inference. Install a new
revision in a separate directory and use a private slot directory. Hashing
reads each distinct mapped shard during loading, including when learned-prefix
caching is disabled.

Choose RAM and disk budgets for the machine, not by copying the example blindly.
The learned path checks available memory with a default 32 GiB reserve, bounds
serialization scratch, caps its application-level disk budget at 64 GiB, and
requires an 8 GiB filesystem free-space floor. It does not create or partition a
64 GiB volume. Learned entries share the existing total budgets; their default
subquota is 2 GiB and 16 states, not an extra allocation outside those totals.
Saving or publishing a new entry can be declined when resources are insufficient.
Learned-state replacement retires its old victim only after successful new
publication. With learned-prefix mode enabled, ordinary prompt snapshots also
require enough space to stage the new file without evicting an existing disk
slot first. A tight budget can therefore decline a new snapshot rather than
risk losing a reusable or context-bound state on serialization/publication failure.
This does not change the legacy lifecycle when learned-prefix mode is disabled.

With learned-prefix caching disabled, the underlying slot disk default remains
16 GiB. See [SETTINGS.md](SETTINGS.md) for all controls, including learned quotas
and memory-reserve configuration. The completion threshold of 1024 committed
output tokens is a conservative policy, **not a measured break-even point**.

## Reproducible local checks

From the repository root, with Python and a C++20 compiler available:

```sh
python tools/test_glm_prefix_cache_host.py --receipt /path/to/new-helper-receipt.json
python tools/test_glm_prefix_slots.py --receipt /path/to/new-slot-receipt.json
python -m unittest discover -s serve -p 'test_*.py' -t .
python tools/check_hip_compat.py
```

Use new receipt paths; runners reject overwriting an existing receipt. The slot
runner extracts the policy code from `generate.cpp`; its deterministic model
substitute is not a CUDA serializer or numerical-quality test.

Real-tokenizer continuation contracts are an explicit additional check, not part
of generic test discovery:

```sh
export MAYA_AGENT_TOKENIZER=/path/to/existing/glm-tokenizer
python -m unittest serve.check_agent_prefix_tokens -v
```

The directory must contain `vocab.json`, `merges.txt`, `tokenizer.json`,
`token_type.json` and `chat_template.jinja`. Missing assets fail explicitly; the
check does not download weights or replace the tokenizer with a mock. These
synthetic render/token-prefix tests do not execute native generation or prove
that a cache restore was taken.

## Evidence boundary

A separate development candidate had bounded model experiments, including a
fixed target-only (`spec=1`), minimum-threshold-1 synthetic completion chain.
That candidate contained additional production-side changes removed from this
patch. Its model results are historical context, **not acceptance of this PR's
source or binary**. An earlier completion window failed its quality gate; a
later startup-format failure completed no candidate loads or test requests.
Those outcomes are not retroactively converted into passes.

### Repeated GPU measurements at `094d8d5`

Both comparisons used GLM-5.3-Flash-Maya-L on four RTX 4090 GPUs at 255 W.
Each run started a fresh engine with an empty slot directory. Ordinary RAM
(80 GiB) and disk (32 GiB) slots were enabled throughout. The two experimental
settings were switched together at their default 1024-token completion threshold.
MTP and DFlash were disabled.

The upstream `agent_bench` workload ran for 300 seconds with two clients, twice
per setting. Across the 27 inputs common to all four runs, each run reused
187,208 of 239,108 input tokens (78.29%) and generated 6,279 tokens. Neither
enabled run saved or restored a learned entry. The longest answer was 816 tokens,
and the benchmark supplied scripted history on the following turn. Enabling the
features reduced mean end-to-end latency by 4.89% in one repeat and increased it
by 2.21% in the other. Outputs matched token for token on 1/27 and 2/27 paired
requests; repeating the same setting matched on 2/27 in each case.

AIPerf 0.11.0 replayed four branches sharing 5,129 prompt tokens via streaming
Chat Completions. Two branches generated long answers and carried the complete
response text into their next requests. The client ran six requests in each
fresh engine, twice per setting. Temperature was zero, thinking was disabled,
and the same seed was used throughout.

| Turn | Mean TTFT, off / on (s) | Mean end-to-end, off / on (s) | End-to-end saving (s) |
|---|---:|---:|---:|
| First short branch | 3.16 / 3.06 | 3.21 / 3.10 | +0.10 |
| Second short branch | 2.77 / 3.58 | 2.84 / 3.64 | -0.80 |
| Free-form answer | 2.88 / 0.54 | 36.72 / 33.77 | +2.95 |
| Follow-up using that answer | 1.37 / 0.96 | 2.15 / 1.68 | +0.46 |
| Structured list | 3.50 / 1.80 | 28.80 / 27.13 | +1.67 |
| Follow-up using that list | 0.89 / 1.12 | 0.98 / 1.20 | -0.22 |

Six-turn wall time fell from 75.81 to 71.18 s and from 73.93 to 70.25 s. The
mean saving was 4.16 s (5.55%). Wall time includes intervals between requests
and excludes model loading and shutdown.

Each enabled run restored the 5,129-token shared prefix twice and restored
both completion tails from RAM. The long turns generated 1,152 and 1,026 tokens.
The next prompts contained each full answer and restored 1,151 and 1,025
committed output tokens respectively (6,327 and 7,791 total cached tokens).
The native restore messages, persisted token keys, HTTP usage and generated
token IDs agreed on all eight feature hits. Ordinary slots supplied the only
four hits in the disabled runs. Total input-token reuse rose from 23,880 to
48,752 out of 72,758 tokens (32.82% to 67.01%).

In each paired comparison, five requests had identical inputs. Four of those
matched on output IDs, visible text and finish reason, including the structured
list and its follow-up. The free-form answer first differed at output token 14,
including when repeating the same setting. Its follow-up therefore received
different inputs. The 0.46 s average reduction on that follow-up also includes
a 30-token response in one enabled run versus 32 tokens in the other three.
The structured follow-up matched on input and output in all four runs but took
0.22 s longer on average with the completion hit. This sample measures a
single-client branching workload; it cannot establish general output parity
or a throughput improvement for other workloads.

### Lifecycle fixes and remaining Ready gates

Host regressions cover the learned-enabled ordinary snapshot's capacity rejection,
payload failure and metadata publication failure without losing the previous slot.
Startup now orders lightweight file metadata by recency before reading/admitting
learned token keys, so reduced count/byte/token quotas favor the newest entries
that fit while retaining the existing aggregate token bound. Startup enumerates
the directory once, allowing at most 8192 visited directory entries and 1024
candidate records. In a stable, private cache directory, only regular files
participate in recovery. Link and other unsupported entry types are not opened
as metadata or payloads; an indeterminate type is an I/O failure, not proof of
absence. Use a stable, private directory with a single writer. Slot publication
creates payload and metadata staging files exclusively. On Windows it supplies
a protected DACL permitting the effective caller SID and checks the actual
owner and DACL on the created handle before writing any payload. If verification
fails it refuses the write and attempts removal through that handle; a filesystem
that refuses removal may leave an empty staging file. Existing directory ACLs
are not changed. It publishes each file without
replacing an existing name using a hard link in the same directory; on a file
system without hard-link support the save is declined. On restore, a learned
payload is opened once: checksum verification reads the opened object and the
native loader reads the same stream after rewind. The legacy native pathname
API remains available for other callers. This does not make an open file's
contents immutable or support concurrent or untrusted writers. The engine does
not verify directory ownership, permissions on every ancestor, or exclusive
writer access. These are operator prerequisites, not a startup guarantee. On
POSIX the reader checks the path type before opening and checks the resulting
descriptor again before reading. The latter check cannot prevent an open of a
non-regular object if the path changes between those operations. Do not enable
persistent slots in directories that another process may modify concurrently.
Startup sorts candidates only after the whole scan succeeds and uses that
same list for orphan checks. Exceeding either limit leaves the directory alone:
no partial index is restored and new disk writes are disabled for the run.
These limits apply to scan work and temporary records, not to the byte size of
the full slot index.

Explicitly lowered completion thresholds can produce a nonempty total key below
1024 tokens. Metadata now accepts the same nonempty bounded key in RAM, on SSD
and after restart. The learned-discovery minimum and default completion increment
remain 1024; allowing short metadata does not lower those default admission gates.
Learned metadata now carries the admission policy version, token thresholds and
completion setting. Matching policy settings preserve restart reuse; changing
them invalidates earlier learned entries. Entries predating this policy tag
also require a one-time refill.

For a single engine, a learned SSD replacement may stage one additional entry
while preserving the old one. Logical quota applies to the post-replacement
set; physical free space must cover staging plus the existing reserve. If file
cleanup fails, residual files can still consume disk space. The engine reports
the failure and stops new admissions that rely on the incomplete accounting.
This is not a crash journal: after an interruption, startup rebuilds the index rather
than replaying an exact list of victims.

During a split load, parts that map the same file object and length share its
digest. GGUF parsing and direct reads are checked against the mapped object.
An in-place write or truncation violates the immutable-input requirement above.
Host tests cover content changes and rejection of slots with a different pack
identity. A four-GPU GLM Flash run with the same weights restored 1,227 of 1,228
prompt tokens from an ordinary disk slot after process restart and matched all
10 cold output token IDs and the stop reason. Changed weights were exercised
only in host fixtures.

The scope of these measurements leaves open numerical equivalence across
workloads, speculative completion, completion-tail restoration across context
switches and SSD restarts, and sustained multi-user latency including P99.

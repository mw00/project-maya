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

No speedup is claimed here. The default 1024-token completion threshold, default
speculative completion behavior, real agent workloads, P99, long multi-user
operation, and completion-tail cross-context/SSD-restart quality remain outside
that narrow historical diagnostic. A host fixture cannot close these gaps.
Enabling this experimental feature is not a recommendation to deploy it broadly.

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

Separate model-level checks are still needed for completion-tail restoration
across contexts and process restarts, the default 1024-token threshold, and
quality and performance under a multi-agent workload.

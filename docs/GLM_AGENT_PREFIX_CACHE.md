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

Persistent identity currently checks pack path, state layout and file metadata,
not a cryptographic digest of every loaded weight. Replacing weights in place
while preserving their names, sizes and relevant metadata can leave an old state
apparently compatible. **Use a fresh private slot directory after any model or
pack change; do not carry retained states across weight revisions.** Binding
persistent state to a verified immutable weight manifest remains a Ready/merge
gate, not a guarantee provided by this experimental patch.

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
that fit while retaining the existing aggregate token bound. Each directory pass
keeps up to 64 path/timestamp records plus its cursor and current candidate.
This bounds temporary ordering records, not the full index or its byte size.
The tradeoff is repeated directory traversal; large-directory startup time has
not been benchmarked.

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

**Weight content identity remains unresolved and blocks Ready.** Existing identity
uses paths, geometry and file metadata, not the bytes actually consumed by every
loader input. Reopening paths to hash them after loading does not fix this: a
replacement can bind loaded A to file B, and directory suffix enumeration need
not match actual shards, companion files, EXL3 or external MTP inputs. No such
partial hash or process-random namespace is shipped here; existing restart
behavior has not been removed. Use a fresh private slot directory per immutable
model revision. An actual loader-bound immutable content identity is still needed.
The unresolved limitation has separate local evidence; it is not counted as a
passing regression or replaced with a weaker identity scheme.

The other three lifecycle fixes are host-code regressions only. Native/CUDA
serialization, actual short-key completion SSD/restart numerical correctness,
default-1024 model behavior and operational quality/performance still require
separate acceptance. Neither passing host checks nor those three fixes makes
this Draft PR Ready.
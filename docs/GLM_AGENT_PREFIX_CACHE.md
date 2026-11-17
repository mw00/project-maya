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
publication. The ordinary prompt-snapshot disk fallback does not yet provide
that same failure-atomic guarantee; see the remaining Draft limitations below.

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

### Remaining Draft lifecycle limitations

- When a normal prompt snapshot falls back to disk under a tight budget, its
  existing eviction path can remove an older slot before serialization or
  metadata publication succeeds. A failed new save may therefore lose reusable
  cache entries. This is not the learned-state replacement transaction, and
  consistent context-bound eviction protection remains a Ready/merge gate.
- If learned quotas are reduced across a restart, startup admission currently
  applies those quotas in directory enumeration order before its recency sort.
  It can discard a hot entry rather than the coldest one. Runtime learned LRU
  replacement does not establish startup LRU behavior under a changed quota.
- With explicitly lowered thresholds, a completion whose entire saved key is
  shorter than 1024 tokens can enter RAM while learned metadata refuses to
  persist it. Such an entry has no SSD/restart guarantee. Admission and
  persistence minima must be reconciled before Ready; this is not the default
  completion policy's behavior.

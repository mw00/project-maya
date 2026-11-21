# Qwen RoPE scaling and YaRN

The Qwen3.8 Flash Next engine accepts `--rope-scaling none|linear|yarn` and
`--rope-scale F`. These are startup settings: changing them reloads the engine;
cached keys computed under the previous scaling cannot be reused.

The Models page exposes **RoPE scaling** and **Extension factor** under Engine →
Model, plus optional numerical overrides under **RoPE advanced**. **Model default**
omits the method override and respects GGUF metadata. **None** explicitly disables
scaling, including when metadata requests it. Empty numerical fields preserve the
engine/model defaults. Set **Context size** separately; allocating a larger cache
alone does not enable positional scaling.

Example for a model trained at 262,144 positions:

```text
--max-context 524288 --rope-scaling yarn --rope-scale 2
```

Advanced arguments retain Strata/ggml conventions:
`--rope-freq-base`, `--rope-freq-scale`, `--yarn-orig-ctx`,
`--yarn-ext-factor`, `--yarn-attn-factor`, `--yarn-beta-fast`, `--yarn-beta-slow`.
YaRN blends interpolated and original rotary frequencies and applies a magnitude
correction. It can change answers even within the trained window. Factor two is
not a promise of lossless quality or a context setting suitable for every model.

## Implementation and attribution

Adapted from the MIT-licensed Strata implementation, including corrected shared
validation, unscaled identity, draft-prefill propagation and the position-zero
indexer spare. Retain the Strata and ggml copyright/license notices.

Key donor commits in Niko1221/Strata:
- `304d613cd03edd6ef4d0f7f62a1bf4d290863107` — shared configuration and table tests
- `0011637753edf8eda42c586771603aeed5f5a94b` — native/indexer/prefill plumbing
- `963679b35e4fc7f692a6f59a64ec8d6db211f3bb` — CLI/GGUF configuration, adapted with later precedence fixes
- `8e2f46ca8ca9a497e720b43229b231a6c07c8fd0` — draft-layer prefill
- `d4526cf156f1b7df2e5c63b8fb24e982cbdbb6de` — zero-position spare correction
- `eb77a8c145a9685cf1d000de974fd8e53412662c` — resolved validation/none identity

Qwen's table, analytic, sparse-indexer, prefill, MTP and verifier paths receive the
same resolved configuration before graph capture. GLM/DeepSeek scaling semantics
are not replaced. Qwen's existing in-process checkpoints share the process's fixed
scaling; this does not add cross-model or disk-backed conversation slots.

## Validation scope

On RTX 4090 / CUDA, using an isolated combined trial with the separate cross-shard
expert-loader and F32 PLE-convolution compatibility fixes:
- CPU 480 position/pair/mode comparisons under ASan/UBSan passed.
- GPU `rope_parity --selftest` and `qsa_parity --selftest` passed, including scaled
  native/table comparisons, graph replay and sequential/batched indexer state.
- Optimized fast-math RoPE test passed.
- UD-IQ4_XS at 524,288 capacity: 299,980-token ledger retrieval and continuation passed.
- 509,895-token explicit-marker retrieval passed at three positions; continuation
  reused 509,934 tokens and answered an additional ledger question correctly.
- A different 509,939-token ledger probe returned an adjacent record for its middle
  lookup. Stock Strata with YaRN ×2 returned the identical wrong answer. This failure
  is retained; successful marker recall is not a blanket quality guarantee.

The combined trial's compatibility fixes are not part of this YaRN-only diff;
related upstream work is PR #990. No AMD, Metal or multimodal extended-context
quality claim is made. Host-memory tests do not substitute for GPU validation.

Settings round-trip tests: `python -m unittest serve.test_qwen_rope_settings`.
Engine numerical targets: `rope_parity`, `qsa_parity`; optional standalone host
check: compile `tests/rope_config_test.cpp` with the kernels include directory.

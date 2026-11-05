# GLM MMQ correctness on Windows CUDA

On Windows with CUDA 13.2, batched GLM prefill could produce incoherent replies,
repeat previous turns or invent a translation task. The same model weights and
prompt worked through token-at-a-time prefill. Disabling the CPU expert lane or
fused SwiGLU did not resolve the problem.

## Cause and fix

The IQ1_S, IQ2_S and IQ3_S MMQ tile loaders read bytes by aliasing local packed
`int` / `int2` values through `uint8_t *`. In the affected NVCC build this produced
incorrect codebook indices, including out-of-bounds global reads. Extracting
unsigned bytes with shifts and masks preserves the packed format without these
local pointer aliases. The correction applies to both fused and plain products.

Compute Sanitizer also found destination-ID reads beyond the last partial expert
tile. Padded shared-memory entries now receive zero instead of reading an unused
ID. The output stores already exclude these padded columns.

The changes live in `third_party/ggml/mmq-fused-swiglu.patch`, which CMake applies
to its build-tree copy of the pinned GGML sources. Rebuild Maya after updating;
editing the installed Python server alone cannot correct the native kernels.

## Measurements

Measured against main commit `4615f346d9311dd1ce9f5738ab04446f023a6e42`, on an NVIDIA
RTX PRO 6000 Blackwell (sm_120), Windows, driver 596.86, CUDA 13.2, with Maya-M.

The regression test previously compared fused and plain products only. Both
could be wrong in the same way. It now also compares sampled outputs against
GGML CPU dequantization and independent scalar double-precision dot products.
The 0.04 relative-L2 threshold accommodates the MMQ path's Q8 activation rounding.

| Format | Independent dot relative L2 before | After |
| --- | ---: | ---: |
| IQ1_S | 1.18493 | 0.0107327 |
| IQ2_S | 0.778627 | 0.00366127 |
| IQ3_S | 0.966161 | 0.00720439 |

The full-model diagnostic's KL divergence from token-at-a-time prefill decreased
from 3.48662072 to 0.0674770672; the next-token argmax changed from a mismatch to a
match. This measurement followed the byte-extraction fix, before the additional
partial-ID guards. The final guarded kernels passed targeted Compute Sanitizer
with zero errors.

Twenty-six real-server cases passed with the final kernels, including latest-turn
corrections, conversation switches and prefix reuse, streaming, cancellation and
recovery, literal role markers, thinking off/low/high, and prompts up to 31851
tokens. A subsequent 229041-token conversation with a 32768-token output allowance
retrieved a code from its first message. These runs also included the separately
submitted small-output reasoning-budget fix.

The numerical output changes intentionally: the original affected MMQ products
were incorrect. These measurements establish the failure and correction on this
machine; they do not establish identical rounding or performance on every GPU.
Other backends and CUDA toolkits were not exercised in these hardware runs.

## Reproduce

Configure a CUDA build with `STRATA_BUILD_TESTS=ON`. Stop the resident model before
running the standalone GPU tests so they have enough free VRAM.

```sh
cmake --build build --target glm_mmq_glu_test glm_prefill_probe
build/glm_mmq_glu_test
compute-sanitizer --tool memcheck --error-exitcode 88 build/glm_mmq_glu_test --iq-s-only
build/glm_prefill_probe /path/to/pack prompt-ids.csv /path/to/new-output-directory
```

The probe needs a real GLM pack and at least nine comma-separated token IDs; it
writes per-layer diagnostics and both logit vectors. The model-free MMQ test is
the targeted regression. For real conversation checks, start the server and run:

```sh
python tools/check_glm_conversations.py --url http://127.0.0.1:8080 \
  --tokenizer /path/to/pack/tokenizer --output /path/to/results
```

The conversation harness preserves the Italian prompts used during diagnosis.
It checks the output cap's rejection against the server's reported context size.

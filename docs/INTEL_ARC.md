# Intel Arc: the Maya-XS DeepSeek GGUFs on llama.cpp's SYCL backend

**This is outside Maya.** Maya's engine has no Intel backend ([#37](https://github.com/mw00/project-maya/issues/37)),
and we have not tested anything on this page: it is what @Jumbomuffin777 reported in
[#401](https://github.com/mw00/project-maya/issues/401), written down so others with an Intel Arc card can find it.
The setup, the dashboard, DSpark drafting and the other Maya features do not run on this path.

## What was reported

The Maya-XS DeepSeek-V4.1 GGUFs run on Intel Arc through llama.cpp's SYCL backend and the `deepseek41` pull request
[ggml-org/llama.cpp#30817](https://github.com/ggml-org/llama.cpp/pull/30817) (closed unmerged on October 27, 2026), after
a small patch to that pull request's model loader:

- Maya's GGUFs store the Engram metadata as `deepseek41.engram.*` with u32 offsets and primes, one Engram table per
  layer (`blk.1/14.engram_embd`, with local offsets) and no `hash_layer_count`.
- That pull request's loader expects `ple.*` keys and one global `per_layer_token_embd`.
- The patch accepts both layouts; files from the pull request's own converter load as before.
- llama.cpp's own DeepSeek-V4.1 support, [ggml-org/llama.cpp#31086](https://github.com/ggml-org/llama.cpp/pull/31086)
  (in releases since b11870), expects its converter's layout (`deepseek41.ple.*`, one `per_layer_token_embd`, an
  `engram.token_map`), so the Maya-XS GGUFs do not load on a llama.cpp release; the #30817 head with the patch is
  still needed. Re-checked on October 29, 2026.

The reporter's branch with the patch and a longer write-up (build steps, flags, known limits) is here:
[Jumbomuffin777:docs/intel-arc-llamacpp-sycl](https://github.com/mw00/project-maya/compare/main...Jumbomuffin777:project-maya:docs/intel-arc-llamacpp-sycl).
We have not reviewed or built it; read the patch before you run it.

## What was measured

Their PC: Intel Arc Pro B70 32 GB + B65 32 GB + B60 24 GB, Ryzen 9 9950X, 64 GB DDR5, Linux. Model: Maya-XS IQ1_S,
64K context.

| Run | Prefill | Decode |
| --- | --- | --- |
| 1,934-token prompt | 45.6 tok/s | 10.9 tok/s |
| 2,470 tokens through an OpenAI-compatible gateway | 41.0 tok/s | 10.8 tok/s |

Tool calls, streaming and an 11K-token recall check passed.

## GLM-5.3-Flash Maya-S v2

Reported later in [#401](https://github.com/mw00/project-maya/issues/401): the Maya-S v2 IQ2_XXS GLM GGUFs
(architecture `glm5-next`) load on stock llama.cpp [v0.8.0](https://github.com/ggml-org/llama.cpp/releases/tag/v0.8.0)
with the SYCL backend and no patch. On the same PC, with the routed experts of six layers (about 11.4 GiB) in host RAM
and the rest on the three cards, a 2,377-token prompt with 128 generated tokens read at 68.3 tok/s and generated at
13.6 tok/s. Other GLM quants were not measured. The command is in the reporter's write-up linked above.

## Limits

- Text only.
- No DSpark drafting and no Engram streaming: the model is placed by llama.cpp's own rules.
- DeepSeek depends on a llama.cpp pull request that was closed unmerged, plus the loader patch above.
- The numbers are one person's, on three Arc cards, with the two quants above.

The patch, the measurements and the write-up were prepared by the reporter with an AI coding agent on their own hardware;
they reviewed them. Questions about this setup belong in [#401](https://github.com/mw00/project-maya/issues/401).

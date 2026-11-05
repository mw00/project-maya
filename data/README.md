# Test and calibration fixtures

This directory holds small fixtures used by the tests and the quant tools. It is not model data and it is not needed
to run Maya.

| Path | What it is | Used by |
| --- | --- | --- |
| `expert-profile-ds4.bin` | Per-model expert usage counts (DeepSeek: 40 × 384 counts) for placement tests | `tests/ds4_place_test.cpp` |
| `expert-profile-glm.bin`, `expert-profile-mm3.bin`, `expert-profile-qwen.bin` | Expert usage profiles for the other model families | placement and profiling tests |
| `exl3-exllamav3/` | Random tiles packed by exllamav3's packer, plus the expected state | `tests/exl3_test`, `tools/exl3_exllamav3_fixture.py` |
| `glm5-big-dumps/` | Committed f64 reference expectations for the GLM MMQ tests | `tests/glm_mmq_glu_test.cu` |
| `glm5-synth-dumps/`, `glm5-synth-dumps12/`, `glm5-synth.gguf` | Synthetic tensors for GLM quant correctness tests | GLM kernel tests |

The fixtures are regenerated with the scripts next to the tools that consume them (for example
`tools/exl3_exllamav3_fixture.py`). Keep them small: they are committed, not downloaded at build time.

# Documentation

Every page in this folder, by what you want to do. Back to the [README](../README.md).

## Start here

- [SETUP.md](SETUP.md): what the first run does, starting again and updating, the setup's options, and the fix for each build failure.
- [MODELS.md](MODELS.md): the four models, how the quants are named, and each one's files, drafters, pictures and options.
- [GLM53.md](GLM53.md): GLM-5.3 (744B): its files, setup, context and defaults.
- [MEMORY.md](MEMORY.md): which model fits your PC - each model's recommended memory (VRAM + RAM together).
- [WINDOWS.md](WINDOWS.md): installing and running Maya on Windows 10/11.

## Use it

- [SETTINGS.md](SETTINGS.md): the `MAYA_*` environment variables and config entries that tune the engine.
- [RESPONSES.md](RESPONSES.md): the `/v1/responses` API, its options and limits.
- [DOCKER.md](DOCKER.md): a recipe to build a Docker image of Maya (there is no official image).
- [AGENT_BENCH.md](AGENT_BENCH.md): `tools/agent_bench.py`, a benchmark that loads a running server like coding agents do.
- [EXL3-HELPER.md](EXL3-HELPER.md): the experimental helper GPU that runs EXL3 experts the main GPUs miss.

## Platforms and hardware

- [AMD_MAYA.md](AMD_MAYA.md): the Linux HIP build for RX 7900 XT / XTX, R9700 / RX 9070 and Strix Halo / Gorgon Halo.
- [AMD_RDNA4.md](AMD_RDNA4.md): the HIP installer path for RX 9070 and Radeon AI PRO R9700 (`gfx1201`).
- [AMD_EXPERT_KERNELS.md](AMD_EXPERT_KERNELS.md): which RDNA decode expert kernel the HIP dispatcher picks for each architecture.
- [AMD_UPSTREAM.md](AMD_UPSTREAM.md): an audit of the AMD backend changes shared with another project.
- [MI50_VALIDATION.md](MI50_VALIDATION.md): the experimental AMD Instinct MI50 (`gfx906`) source build.
- [METAL.md](METAL.md): the experimental Apple Silicon (Metal) build.
- [PORT-MINIMAX-METAL.md](PORT-MINIMAX-METAL.md): notes on porting MiniMax-M3.1-Flash to Metal.
- [DGX_SPARK.md](DGX_SPARK.md): Maya on an NVIDIA DGX Spark (GB10) and Grace, where memory is shared with the system.
- [DGX_SPARK_BUILD.md](DGX_SPARK_BUILD.md): what the aarch64 port for the DGX Spark changed and where each part comes from.
- [IBM_AC922.md](IBM_AC922.md): Maya on the IBM Power System AC922 (POWER9, ppc64le, six V100s).
- [INTEL_ARC.md](INTEL_ARC.md): a community report of Maya's DeepSeek GGUFs on llama.cpp's SYCL backend; outside Maya's engine.

## Engineering notes

How the engine works and how changes were measured. You do not need these to run Maya.

- [GLM5-FLASH.md](GLM5-FLASH.md): the study of what it takes to run GLM-5.3-Flash on the engine, component by component.
- [QUANT-PLAN.md](QUANT-PLAN.md): the plan for Maya's first GLM-5.3-Flash quants.
- [GLM_MMQ_CORRECTNESS.md](GLM_MMQ_CORRECTNESS.md): the incoherent GLM prefill replies on Windows with CUDA 13.2.
- [GLM-KDA-GRAPHS.md](GLM-KDA-GRAPHS.md): `MAYA_GLM_KDA_GRAPH=1`, experimental graph capture of GLM's recurrent layers.
- [deepseek-prompt-format.md](deepseek-prompt-format.md): DeepSeek-V4.1's prompt encoding and the vendored `serve/ds41_encoding.py`.
- [ds4-draft-timing.md](ds4-draft-timing.md): the DSpark drafter timing fields in DeepSeek `STAT` lines.
- [ds4-skip-unpicked.md](ds4-skip-unpicked.md): DeepSeek prefill transfers of only the experts a chunk picks (`DS4_SKIP_UNPICKED`).
- [DS4_PROFILE_REFRESH.md](DS4_PROFILE_REFRESH.md): refreshing DeepSeek's startup expert placement, measured on a three-GPU AMD PC.
- [EXL3_HIP.md](EXL3_HIP.md): EXL3 prefill on HIP.
- [EXL3_HIP_SCALAR_ROWS.md](EXL3_HIP_SCALAR_ROWS.md): configurable scalar EXL3 prompt tiles on RDNA4.
- [EXL3_RDNA4.md](EXL3_RDNA4.md): matrix-instruction EXL3 prefill on RDNA4 (`EXL3_WMMA_GFX12`).
- [HIP_PEER_PREFILL.md](HIP_PEER_PREFILL.md): how startup picks the HIP prompt transport between two GPUs.

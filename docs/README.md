# Project Maya documentation

Start with [Set up Maya](SETUP.md); everything else is reference.

## Start here

- [Set up Maya](SETUP.md) — install, first model, first token.
- [Pick a model](MODELS.md) — families, Maya quants, sizes, per-model options.
- [Compatibility](COMPATIBILITY.md) — GPUs, systems, memory, disk.
- [FAQ](FAQ.md) — short answers to the questions that come up most.

## Use it

- [Settings](SETTINGS.md) — every `MAYA_*` setting.
- [Responses API](RESPONSES.md) — cache, background tasks, quotas.

## Platforms

- [Windows](WINDOWS.md) — Windows 10/11 specifics.
- [AMD / ROCm](AMD_MAYA.md) — supported cards and ROCm setup.
- [AMD RDNA4](AMD_RDNA4.md) — RDNA4 specifics.
- [Apple Metal](METAL.md) — experimental backend for GLM-5.3-Flash.
- [Intel Arc](INTEL_ARC.md) — community route through llama.cpp SYCL.
- [DGX Spark](hardware/DGX_SPARK.md) · [DGX Spark build](hardware/DGX_SPARK_BUILD.md) · [IBM AC922](hardware/IBM_AC922.md) · [MI50](hardware/MI50_VALIDATION.md) · [Docker](hardware/DOCKER.md)

## Benchmarks

- [Speed reports](benchmarks/SPEED-REPORTS.md) — real machines, per model and quant.
- [Agent benchmark](benchmarks/AGENT_BENCH.md) — coding-agent runs.
- [Quant results](benchmarks/results/) — KL and same-token measurements per Maya quant.

## Architecture and internals

- [RDNA decode expert kernels](dev/AMD_EXPERT_KERNELS.md) — For nembd=4096, nff=2048, the HIP decode dispatcher selects kernels by architecture
- [Shared AMD backport audit](dev/AMD_UPSTREAM.md) — Compared Maya v1.0.4 (70e0746) plus this branch's GLM HIP port with Strata d5ea713.
- [Refreshing DS4 startup expert placement](dev/DS4_PROFILE_REFRESH.md) — On a three-GPU AMD machine, refreshing the startup expert profile raised held-out 128-token…
- [Experimental EXL3 overflow expert execution](dev/EXL3-HELPER.md) — The helper keeps additional target-model experts on a GPU outside the target TP group.
- [EXL3 prefill on HIP](dev/EXL3_HIP.md) — The portable HIP implementations of exl3::setgateup, setdown, and gemmrows reuse the existing…
- [Configurable scalar EXL3 prompt tiles on RDNA4](dev/EXL3_HIP_SCALAR_ROWS.md) — A prompt often routes several tokens to the same expert.
- [RDNA4 EXL3 matrix prefill](dev/EXL3_RDNA4.md) — On HIP gfx12 devices, grouped EXL3 prefill can use matrix instructions for setgateup and…
- [Experimental opt-in KDA layer graphs](dev/GLM-KDA-GRAPHS.md) — MAYAGLMKDAGRAPH=1 captures recurrent layers of single-device GLM decode.
- [Running GLM-5.3-Flash - the study behind Maya's engine](dev/GLM5-FLASH.md) — What it would take for zai-org/GLM-5.3-Flash to run on this engine, component by component,…
- [GLM MMQ correctness on Windows CUDA](dev/GLM_MMQ_CORRECTNESS.md) — On Windows with CUDA 13.2, batched GLM prefill could produce incoherent replies, repeat…
- [Automatic HIP prompt transport selection](dev/HIP_PEER_PREFILL.md) — For an eligible tensor-parallel pair, startup compares the ordinary host-mapped reduction,…
- [MiniMax-M3.1-Flash on Apple Silicon (Metal) - port notes](dev/PORT-MINIMAX-METAL.md) — Experimental.
- [Project Maya quants of GLM-5.3-Flash - the plan](dev/QUANT-PLAN.md) — Two quants of our own, made from the official weights with error-feedback rounding and scale…
- [DeepSeek-V4.1 prompt format](dev/deepseek-prompt-format.md) — The server uses DeepSeek's MIT-licensed encoding/encoding.py at revision…
- [DeepSeek serving draft timing](dev/ds4-draft-timing.md) — DeepSeek STAT lines expose existing DSpark measurements alongside drafts (the accepted/offered…
- [DeepSeek prefill transfers: only the experts a chunk picks](dev/ds4-skip-unpicked.md) — CUDA: on for chunks of up to DS4SKIPUNPICKEDROWS rows (1536) - an agent's turn, a short prompt.

## Project

- [Contributing](../CONTRIBUTING.md) · [Security](../SECURITY.md) · [Code of conduct](../CODE_OF_CONDUCT.md) · [Roadmap](../ROADMAP.md)
- [Third-party notices](../THIRD_PARTY_NOTICES.md) · [Licenses](../licenses/README.md)

# Maya on the IBM Power System AC922 (POWER9, ppc64le)

The AC922 is IBM's POWER9 machine from the V100 era: two POWER9 sockets, six Tesla V100-SXM2-16GB cards on
NVLink 2.0, and a coherent link (ATS) that lets the GPUs read host memory directly. Maya builds and runs on it
from source, and GLM-5.3-Flash holds its full **1,048,576-token** context on all six cards.

Measured on one AC922 (see [Measured](#measured)): **72 tokens/s** decoding short answers, **1,297** and
**2,102 tokens/s** reading 70K- and 183K-token prompts, 14.7 of each card's 16 GB in use. One setting is not
optional here: [`MAYA_GLM_NUMA=0`](#the-one-setting-that-is-not-optional) - without it the machine panics.

## The machine

| | |
| --- | --- |
| GPU | 6x Tesla V100-SXM2-16GB, compute capability 7.0 (sm_70), 96 GB in all; NVLink 2.0 between the cards of a socket |
| CPU | 2x IBM POWER9, 176 threads (SMT4), ppc64le, **no AVX2** |
| Memory | 183 GB |
| OS | RHEL 8.10, kernel 4.18.0-553.el8_10.ppc64le |
| Compiler | gcc-toolset-13; RHEL 8's system GCC 8.5 cannot compile C++20 |
| Driver / CUDA | 535.216.03 / CUDA 12.4 (`-DCMAKE_CUDA_ARCHITECTURES=70`) |
| Power | 250 W per card (`gpu-power-limit.service`); the cards draw ~70 W decoding, up to 226 W in prefill |

## Bringing it up

RHEL 8 ships GCC 8.5, which rejects `-std=c++20`; Maya's CMake requires C++20, so put the newer compiler first
on `PATH` before anything that builds the engine:

```sh
source /opt/rh/gcc-toolset-13/enable
./maya.sh --check
./maya.sh --setup --yes --model glm-maya-s24 --gpus 0,1,2,3,4,5 --context 1048576 --download-model
```

`--check` lists the six cards, CUDA 12.4, the driver, RAM and this line, which is expected on POWER (the CPU
expert lane falls back to ggml-cpu's VSX kernels):

```
  [!]  the CPU (POWER9, altivec supported) has no AVX2: the engine runs, its CPU expert lane on ggml's own
       kernels (slower; ...). MAYA_GLM_CPU_LANE=0 turns it off
```

The numbers on this page come from a from-source build:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=70 -DSTRATA_NATIVE_EXPERTS=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.4/bin/nvcc \
  -DCMAKE_CUDA_HOST_COMPILER=/opt/rh/gcc-toolset-13/root/usr/bin/g++ \
  -DSTRATA_GGML_DIR=$HOME/builds/env/llama.cpp-3cf0325
cmake --build build --target maya -j 16
```

That is about five minutes on 16 jobs. The other engines build the same way with their own targets
(`maya-ds4`, `maya-mm3`).

### What the ppc64le port changes

| file | why |
| --- | --- |
| `setup.py` | `PPC` detection, and `cpu_info()` returns POWER's `cpu` line: POWER9's `/proc/cpuinfo` has no `flags:` line, so the x86 probe raised `AttributeError` before anything ran |
| `CMakeLists.txt` | `STRATA_CPU_PPC64` / `STRATA_CPU_ARM64`; the ARM `-march=armv8.6-a` flags are gated on aarch64 (GCC on POWER rejects them). ppc64le then takes the `GGML_NATIVE` path, where ggml-cpu reads `/proc/cpuinfo` and builds with `-mcpu=power9` itself |
| `CMakeLists.txt` | `CXX_EXTENSIONS ON` for `ggml-cpu`: its PowerPC quants use the `vector` keyword the file-scope C++20 extensions-off would remove |
| `maya.py` | EXL3 quants are refused on POWER as on ARM: their expert kernels are x86 AVX2/AVX-512 |
| `include/strata/platform/cpu_relax.hpp`, `src/ds4/ds4_pool.hpp` | POWER branches: `or 27,27,27` as the spin hint and `lwsync` as the store fence before a GPU polls a doorbell in pinned memory |

## The one setting that is not optional

```json
"env": { "MAYA_GLM_NUMA": "0" }
```

Maya pins the expert tier that does not fit in VRAM with `mmap` + `mbind` + **`cudaHostRegister`**
(`src/core/glm_fast_path.cu`, `numa_pinned()`), which it chooses automatically on any host with two or more NUMA
nodes. On POWER9 the V100s reach host memory through ATS, and the driver's UVM migration also moves *registered*
pages: that path is fatal here.

```
kernel BUG at mm/migrate.c:701        -> migrate_vma_pages
  nv_migrate_vma                      [nvidia_uvm]
  uvm_migrate_pageable                [nvidia_uvm]
  service_ats_requests                [nvidia_uvm]     <- a V100 faulting on a host page
```

It panicked the machine mid-benchmark (kdump caught it; the box rebooted on its own), and it was also taking half
the speed:

| | `cudaHostRegister` (default) | `cudaHostAlloc` (`MAYA_GLM_NUMA=0`) |
| --- | ---: | ---: |
| decode, mean | 40-41.5 tokens/s | **74-76** |
| prefill, 2K / 8K | 407 / 557 | **498 / 722** |

Both columns were measured in one session; the point is the ratio, which does not depend on the cap - the default
path halved decode and cost a quarter of prefill.

`MAYA_GLM_NUMA=0` makes `numa_pinned()` return null, and the engine falls back to `cudaHostAlloc` - the path the
AC922 needs. Check that it reached the engine, since `maya.py` passes the config's `env` to the engine under the
name it reads:

```sh
tr '\0' '\n' < /proc/$(pgrep -x maya)/environ | grep GLM_NUMA       # STRATA_GLM_NUMA=0
```

## Six GPUs, not four

The quant is **Maya-S24** (`glm-maya-s24`, 94.7 GB): 4-bit attention and shared experts, made for cards of 24 GB
or less. On six cards ~74 GB of experts fit in VRAM (98.4 % of the routes hit) and ~25 GB in the pinned RAM tier.
Four cards measured **3.6 tokens/s**: too little room for both tiers, so every token streams its experts.

## Context

GLM-5.3-Flash is 34 linear-attention layers and 11 MLA (full-attention) layers; only the 11 grow with tokens, and
`--kv int8` halves them. With `--kv-streaming on` only the resident window stays in VRAM and the rest lives in
pinned RAM, which makes the full context cheap:

| `--max-context` | decode | prefill 2K / 8K | KV in VRAM, all cards | RAM tier | pool per card |
| ---: | ---: | --- | ---: | ---: | ---: |
| 65,536 | 73.5 | 495 / 717 | 0.55 GB | 21.1 GB | 12.30 GB |
| 131,072 | 75.8 | 495 / 715 | 0.62 GB | 21.3 GB | 12.28 GB |
| 262,144 | 75.0 | 484 / 711 | 0.81 GB | 21.9 GB | 12.25 GB |
| 524,288 | 71.6 | 484 / 696 | 1.17 GB | 23.2 GB | 12.18 GB |
| **1,048,576** | **72.6** | 468 / 694 | **1.93 GB** | 25.4 GB | 12.05 GB |

Going from 65,536 to 1,048,576 costs about 1.4 GB of VRAM, 4 GB of RAM and 1 % of decode. All five sizes loaded
and ran.

## Measured

Bench as the README's tables describe it: three 256-token greedy answers, thinking off, prefill at 2,048 and
8,192 tokens.

| | decode | prefill 2K / 8K |
| --- | --- | --- |
| three runs | 72.6 / 72.2 / 71.2 (mean **72.0**) | 464 / 692 |

Long context, one request at a time, prompts with a phrase planted mid-document:

| prompt | read at | the phrase |
| --- | ---: | --- |
| 69,646 tokens | 1,297 tokens/s | found |
| 182,746 tokens | 2,102 tokens/s | found |

Decode at depth: **66.8 tokens/s** over a 48-token answer at 70K, **32.5** at 183K - the KV streaming reads grow
with the context, so the deep answer is the slow one.

### Which quant

Two of Maya's quants were measured here, Maya-S and Maya-S24. This machine serves **Maya-S24** and it is the
faster one by a wide margin. Both at 1,048,576 tokens, one after the other in the same hour, same arguments:

| | Maya-S | Maya-S24 |
| --- | ---: | ---: |
| decode, mean of three answers | 62.1 / 62.9 | **72.8** |
| prefill 2K / 8K | 454 / 682 | 467 / 694 |
| expert slots in VRAM | 10,984 | 11,229 |
| RAM tier, slots | 4,030 | 3,785 |
| RAM fetches per token | 6.2-7.3 | 5.1-6.0 |

Maya-S keeps its attention and shared experts in 6 bits, about 0.25 GB more on each card, and at 1M context the
smaller pool sends ~20 % more of each token's experts to the RAM tier: **-13.6 % decode** for a quality change
the quant's notes put at KL 0.444 -> 0.421 against the FP8 model. On 16 GB cards that is the wrong trade: ten
tokens a second for 0.02 of KL. Maya-M (116 GB) was not measured: it can only have a smaller pool than Maya-S,
so it cannot be faster here.

## What did not help

Each knob below was measured once against stock, in the same hour and at the same cap:

| setting | decode | verdict |
| --- | ---: | --- |
| stock | 72.0 | - |
| `MAYA_GLM_COMPLETE_POOL=0` | 72.9 | neutral |
| `MAYA_GLM_CPU_LANE=0` | 71.6 | neutral (the CPU lane takes 0.1 experts a token here) |
| `MAYA_GLM_ROUTE_RANK=0 MAYA_GLM_ROUTE_FAST=0` | 68.6 | **worse** - the new route kernels are worth ~3.4 tokens/s on a V100 |

Two measurement rules this machine taught:

- **Do not sample `nvidia-smi` in a loop during a run.** A 2-second sampler cost 14 % (41.2 -> 35.5 tokens/s).
  Read the cards before and after, never during.
- **One cold run is not a number.** The first bench after a rebuild read 62.8 tokens/s with `cpu lane 5.03/tok`
  and a 97.3 % VRAM hit rate; every run since reads 0.11/tok and 98.4 %. If a run is 10 % off and its CPU-lane
  figure is off with it, run it again.

## Known limits

- **Pictures do not work**: the vision tower's BF16 GEMMs need sm_80, and the V100 is sm_70. The server still
  advertises image input when the pack carries a vision tower, so send text only.
- **EXL3 quants are x86-only** (their expert kernels are AVX2/AVX-512); the port refuses them on POWER with the
  same message as on ARM. Use the GGUF/Maya quants.
- **CUDA 13 cannot build sm_70**; stay on the 12.x toolkit.
- **Only the GLM engine was built and run here** (`maya`). `maya-ds4` and `maya-mm3` compile from the same
  CMake tree but have not been measured on this machine.
- **Prefill is the phase that reaches the power limit**: one card reports `SW Power Cap` at 226 W during
  prefill, while decode draws ~70 W a card.

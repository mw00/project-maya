# Compatibility

What runs Maya, and how well it is tested.

- **Validated** — runs on real hardware; reports appear in [speed reports](benchmarks/SPEED-REPORTS.md).
- **Experimental** — builds and passes unit tests, but has not run the real models here.
- **Not supported** — does not run Maya's engine.

## GPUs

| GPU family | Status | Notes |
| --- | --- | --- |
| NVIDIA V100 (sm_70, 16/32 GB) | Validated | 1–4 cards in the reports; CUDA 12.x only |
| NVIDIA RTX 20 / 30 / 40 / 50 | Validated | From 8 GB; 24 GB cards run Maya-S/M |
| NVIDIA A100 / H100 | Validated | 80 GB cards, 1–2 per report |
| NVIDIA RTX PRO 6000 Blackwell | Validated | 96 GB, single card |
| NVIDIA DGX Spark (GB10, 121 GB unified) | Validated | [notes](hardware/DGX_SPARK.md), [build](hardware/DGX_SPARK_BUILD.md) |
| NVIDIA Pascal (Tesla P40 / P100, GTX 10) | Experimental | Not validated yet |
| AMD RX 7900 XT / XTX, RX 9070 | Validated | ROCm 7; [AMD notes](AMD_MAYA.md) |
| AMD Radeon AI PRO R9700 | Validated | gfx1201; [RDNA4 notes](AMD_RDNA4.md) |
| AMD Strix Halo / Gorgon Halo (Radeon 8060S / 8065S) | Validated | Unified memory; page-file guidance in [AMD notes](AMD_MAYA.md) |
| AMD RX 6800 / 6900, Radeon PRO V620 / W6800 | Not validated | May work; no reports yet |
| AMD Instinct MI100 / MI50 | Experimental | [MI50 validation](hardware/MI50_VALIDATION.md) |
| IBM AC922 (POWER9 + V100) | Experimental | [notes](hardware/IBM_AC922.md) |
| Apple silicon (M-series) | Experimental | Metal backend, GLM-5.3-Flash only; CI-tested, not run with the real model — [notes](METAL.md) |
| Intel Arc | Not supported | Community route via llama.cpp SYCL — [notes](INTEL_ARC.md) |

Up to 16 GPUs can share one model. More VRAM is faster; cards do not have to be identical, but the engine
assumes they are visible to each other over PCIe or NVLink.

## Systems

| System | Status | Notes |
| --- | --- | --- |
| Linux x86-64 | Validated | AVX2 CPU recommended; NVIDIA driver, g++, Python 3.10+ |
| Windows 10 / 11 | Validated | `START-MAYA.bat`; not WSL2 — [notes](WINDOWS.md) |
| macOS (Apple silicon) | Experimental | Metal backend — [notes](METAL.md) |
| Docker | Recipe, no official image yet | [Docker notes](hardware/DOCKER.md) |

CUDA 12.x is the supported toolkit. CUDA 13 works on RTX 20 and newer, but not on the V100 or Pascal.
AMD needs ROCm 7.

## Memory and disk

| Resource | Requirement |
| --- | --- |
| System RAM | 32 GB runs it; more RAM keeps more experts close and is faster |
| VRAM | Any supported card; 24 GB is the practical floor for Maya-S24 |
| Disk | A fast NVMe SSD, never a hard disk, with room for the model plus a 1.1 GB picture encoder |

Model sizes: about 73–156 GB for GLM-5.3-Flash quants, about 277–340 GB for DeepSeek-V4.1-Flash, about
93–122 GB for MiniMax-M3.1-Flash. Exact figures per quant: [Pick a model](MODELS.md).

## Checking your machine

The setup checks the GPU, driver, RAM, disk and CPU before it downloads anything, prints the exact command for
anything missing, and installs nothing system-wide. `./maya.sh --report` writes a hardware summary you can attach
to an issue.

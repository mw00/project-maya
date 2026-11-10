# Which model fits your PC?

Add up your **GPU memory (VRAM) and system RAM**. That total is what counts: Maya keeps the most-used parts of the
model on the GPU, the rest in RAM, and reads anything that doesn't fit from your NVMe SSD while it answers.

- **Recommended** = the whole model stays in memory with room to spare: the speeds people report in the README.
- **Below that it still runs**, just slower, because part of the model is read from the SSD (see [the floor](#the-floor)).
- More of the total on the GPU is faster than the same total in RAM.
- On a **Strix Halo, Gorgon Halo or a Mac**, the memory is shared: count it once, and keep about 16 GB for the system.

## The table

Sizes in GB. "Whole model" is the memory the setup counts to keep every part of the model in VRAM + RAM; "Recommended"
adds room for the system and the default context.

| Model | Quant | Download | Whole model | **Recommended (VRAM + RAM)** | Runs on |
|---|---|---|---|---|---|
| **GLM-5.3-Flash** | Maya-XS | 73 | ~80 | **88** (a 24 GB GPU + 64 GB RAM, at 128K context) | NVIDIA, AMD, Mac* |
| | Maya-S24 (for GPUs of 24 GB or less) | 95 | ~102 | **128** | NVIDIA, AMD, Mac* |
| | Maya-S (default) | 97 | ~104 | **128** | NVIDIA, AMD, Mac* |
| | Maya-M | 116 | ~124 | **160** | NVIDIA, AMD, Mac* |
| | Maya-M-Derisked (experimental) | 116 | ~124 | **160** | NVIDIA, AMD, Mac* |
| | Maya-L | 156 | ~164 | **192** | NVIDIA, AMD, Mac* |
| | EXL3 3.05 bpw (turboderp) | 125 | ~133 | **160** | NVIDIA, AMD |
| | EXL3 4.05 bpw (turboderp) | 165 | ~173 | **192** | NVIDIA, AMD |
| **GLM-5.3 (744B)**, text only | Maya-XS | 170 | ~176 | **192** | NVIDIA (V100 and newer), AMD |
| | UD-IQ2_M (unsloth) | 239 | ~245 | **256** | NVIDIA (V100 and newer), AMD |
| **DeepSeek-V4.1-Flash** | Maya-XS | 277 ¹ | 121 | **128** | NVIDIA, AMD |
| | Maya-S (default) | 340 ¹ | 186 | **224** | NVIDIA, AMD |
| | UD-IQ1_M (unsloth) | 301 ¹ | 140 | **160** | NVIDIA, AMD |
| | UD-IQ3_S (unsloth) | 424 ¹ | 263 | **288** | NVIDIA, AMD |
| | UD-Q8_K_XL (unsloth) | ~470 ¹ | ~306 | **384** | NVIDIA, AMD |
| | EXL3 2.9 bpw (LibertAIDAI) | 414 ¹ | 230 | **256** | NVIDIA, AMD (x86-64) |
| | EXL3 3.5 bpw (LibertAIDAI) | 461 ¹ | 280 | **320** | NVIDIA, AMD (x86-64) |
| | Abliterated Mix-128G (orcarouter) | 275 ¹ | 114 | **128** | NVIDIA, AMD |
| | Abliterated Mix-256G (orcarouter) | 397 ¹ | 236 | **256** | NVIDIA, AMD |
| **MiniMax-M3.1-Flash** | Maya-XS | 93 | 100 | **128** | NVIDIA, AMD |
| | Maya-S | 122 | 130 | **160** | NVIDIA, AMD |
| | EXL3 3.5 bpw (Mia-AiLab) | 197 | 250 | **256** | NVIDIA, AMD |
| **Qwen-3.8-Flash-Next** | UD-IQ2_XS (unsloth) | 80 ² | 47 | **64** | NVIDIA, AMD |
| | CYBER-FROST Q5_K_M | 113 ² | 84 | **96** | NVIDIA, AMD |

\* Mac (Apple Silicon) support is experimental, GLM-5.3-Flash's GGUF quants only.
¹ DeepSeek's download includes its n-gram (Engram) tables, which always stay on the SSD and never count toward memory.
² Qwen's n-gram table also stays on the SSD.

## Examples

| Your PC | Total | Fits fully | Also runs (slower) |
|---|---|---|---|
| 16 GB GPU + 48 GB RAM | 64 | Qwen UD-IQ2_XS | GLM Maya-XS / Maya-S24 |
| 24 GB GPU + 64 GB RAM | 88 | GLM Maya-XS, Qwen UD-IQ2_XS | GLM Maya-S24 / Maya-S, Qwen CYBER-FROST |
| 32 GB GPU + 96 GB RAM, or a 128 GB Strix/Gorgon Halo | 128 | GLM Maya-S / S24, DeepSeek Maya-XS, MiniMax Maya-XS, DeepSeek Mix-128G, Qwen CYBER-FROST | GLM Maya-M |
| 48 GB of GPUs + 128 GB RAM | 176 | the above + GLM Maya-M and EXL3 3.05, MiniMax Maya-S, DeepSeek UD-IQ1_M | GLM Maya-L, DeepSeek Maya-S |
| 64 GB of GPUs + 128 GB RAM, or a 192 GB Gorgon Halo | 192 | the above + GLM Maya-L and EXL3 4.05, GLM-5.3 (744B) Maya-XS | DeepSeek Maya-S |
| 64+ GB of GPUs + 192 GB RAM | 256+ | the above + DeepSeek Maya-S, EXL3 2.9, Mix-256G, MiniMax EXL3, GLM-5.3 (744B) UD-IQ2_M | DeepSeek UD-IQ3_S, EXL3 3.5 |
| a large-RAM server (384 GB or more) | 384+ | every model, DeepSeek UD-Q8_K_XL included | - |

## Context length

- **GLM-5.3-Flash** keeps its context in RAM and it is small (about 0.2 GB at the default 32K): longer contexts barely
  change the numbers above.
- **DeepSeek, MiniMax and Qwen** default to 64K. Longer contexts need more memory; the setup only offers the ones your PC
  can hold. Each model's options are in [docs/MODELS.md](MODELS.md) and [docs/SETTINGS.md](SETTINGS.md).

## The floor

Every model still runs below its recommended size, as long as you have:
- a supported GPU (NVIDIA from the V100 / RTX 20 on; AMD as listed in the README's [What you need](../README.md#what-you-need)),
- **32 GB of RAM**,
- a fast **NVMe SSD** with room for the download (never a hard disk).

What doesn't fit in VRAM + RAM is read from the SSD while the model answers: it works, but the further below the
recommended size, the slower it gets. The setup tells you how much of the model it can keep in memory on your PC.

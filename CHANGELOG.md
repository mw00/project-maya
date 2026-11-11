# Changelog

Every release is on GitHub (Releases) with these notes; every published change moves the last number. Update: the
page's Settings › About and updates (About > Update before the redesign; from v1.0.18), or `git pull`, then `./setup.sh` (Windows: `START-MAYA.bat`) - it recompiles
only what changed and starts; the model is not downloaded again.

## v1.1.59 - 2026-11-11

Refusal ablation for DeepSeek, MiniMax and GLM-5.3-Flash, Turbo for GLM-5.3 (744B), a setup that recommends the quant and context your PC's memory holds, and fixes.

- **The setup recommends the quant your PC holds:** every model's list (GLM-5.3-Flash, GLM-5.3 (744B), DeepSeek and MiniMax) now recommends the largest quant, no larger than the model's usual pick, whose files plus the default context's cache fit in your VRAM and RAM together (with 6 GB left for the system), so no expert has to be read from the SSD while it answers. When less than a fifth of it would be in GPU memory, the next smaller one is recommended; when none fits, the smallest. An AMD Ryzen AI Max APU (Strix / Gorgon Halo) and a Mac count their shared memory once. Every quant stays in the list, and ids, names and downloads are unchanged. For GLM-5.3 (744B) the recommendation is at most GLM-5.3-Maya-XS; the larger quants stay listed.
- **GLM-5.3 (744B) on unified-memory APUs: a context that keeps the whole model in memory:** on an AMD Ryzen AI Max APU the setup recommends the largest context whose cache still leaves room for all of the model in memory, and says so in one line. A longer context, given with `--context` or chosen by an earlier setup, is kept, with a warning that some experts are then read from the SSD.
- **Turbo for GLM-5.3 (744B):** the Turbo button is now offered for GLM-5.3 too, with its own setting: when a token's experts are chosen, the ones already in VRAM are preferred (`MAYA_GLM_ROUTE_AFFINITY=0.2`), and the context is set to 32K. With GLM-5.3-Maya-XS, part of it outside VRAM, this decoded about 22-25% faster than without it on our test PC. Like every Turbo setting it changes the answers; it stays off until you switch it on, and switching it off restores your settings exactly. GLM-5.3-Flash's Turbo is unchanged.
- **Dashboard: "Saved" for chat defaults:** Settings > Chat defaults shows a small "Saved" tick beside its title after every change.
- **Refusal ablation for DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash and GLM-5.3-Flash:** Maya's own direction files are now in each model's repo; the setup downloads the model's file and the Models page shows its switch. It is off until you switch it on (the Models page, or a request's `"ablate"`), and with it off the answers are the same as before. On our held-out test prompts the share of refused requests went from 98% to 1% (DeepSeek), 100% to 7% (MiniMax) and 100% to 4% (GLM-5.3-Flash), with no more refusals of ordinary requests than without it. Fitting your own direction with `tools/ablate_fit.py` now works on MiniMax (it stopped with an error before) and gives a usable direction on GLM-5.3.
- **Fixes:**
  - GLM-5.3 (744B) and GLM-5.3-Flash keep their own chat defaults on the dashboard: changing one no longer changes the other. GLM-5.3 starts once from the values you saved before.
  - `--parallel` with fair slices (`MAYA_FAIR_SLICE_S`): one waiting request now pauses one running answer instead of every running answer, so the others keep going (thanks to @HTsulfuric, #883).
  - DeepSeek with `DS4_LENT_RAM_GB`: a failed copy into the RAM copies no longer makes every request fail with HTTP 400; the copies turn themselves off and prompts run as they do without the setting, with the same answers (#884).
  - DeepSeek on NVIDIA tensor-parallel pairs: `MAYA_DS4_HIP_PEER_AR=auto` / `1` reaches the direct GPU-to-GPU prompt all-reduce again, as in v1.1.54 (since v1.1.55 it was skipped whenever `MAYA_DS4_AR2` was not set); `auto` compares the transports at startup as before. The answers are the same.

## v1.1.58 - 2026-11-10

GLM-5.3-Maya-XS, Maya's own quant of GLM-5.3 (744B) at 170 GB, faster GLM-5.3 decode and drafting, DeepSeek's one-GPU decode back to speed (#920), and fixes.

- **GLM-5.3-Maya-XS:** Maya's own quant of [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (744B), the routed experts in ~1.6 bits: 170.0 GB in 4 files, and as close to the original as unsloth's UD-IQ1_S (the same mean KL divergence, 0.506; the same top token 81.8% of the time) at 78% of its size.
  - **The files:** [peasantsmith/GLM-5.3-Maya-GGUF](https://huggingface.co/peasantsmith/GLM-5.3-Maya-GGUF), pinned and checked shard by shard: `./maya.sh --setup --model glm-5.3-maya-xs` (Windows: `START-MAYA.bat --setup --model glm-5.3-maya-xs`), or pick it in the setup's list or on the dashboard's Models page (among Maya's quants). It is shown as "GLM-5.3 (744B) · Maya-XS".
  - **Memory:** about 176 GB of VRAM and RAM together keeps every expert in memory (192 GB recommended, [docs/MEMORY.md](docs/MEMORY.md)); with less, the rest is read from the SSD while it answers. The setup counts its own dense weights when it offers the contexts that fit your GPUs.
  - NVIDIA from the Tesla V100 on and AMD (the Strix and Gorgon Halo included), Linux and Windows; text only, as GLM-5.3 itself; drafting with its own NextN block.
- **GLM-5.3: the GGUF's Q8_0 attention weights as they are:** the pack keeps GLM-5.3's Q8_0 attention weights (k_b / v_b) in Q8_0 instead of converting them to BF16 - the GGUF's own values, so answers can differ slightly from v1.1.57's. Fewer bytes are read per token, which helps most on a Gorgon Halo. An existing GLM-5.3 pack is rebuilt once by the next setup; `MAYA_PACK_NATIVE32=0` keeps the conversion.
- **Faster CPU part of the decode, on by default:** where experts are computed on the CPU (x86-64 CPUs with AVX2, FMA and F16C), four rows of an expert's matrix are now computed at once - the same answers bit for bit. `MAYA_GLM_LANE_ROW4=0` turns it off.
- **GLM-5.3 on an AMD Ryzen AI Max APU (Strix / Gorgon Halo):**
  - **Drafting on one GPU, chosen by measurement:** drafting with the model's MTP block is now set by the speed calibration: the setup measures decode with drafting and without it on four short chats (code, an explanation, an email and a thinking answer) and keeps the faster. (Before, one-GPU drafting was always off for GLM-5.3.) Your own `MAYA_GLM_MTP_DECODE`, DFlash2, discrete GPUs and two GPUs are not touched.
- **Reproducible GLM prompt reads (opt-in):** `MAYA_GLM_PREFILL_STABLE=1` computes every expert product of a prompt the same way whatever the expert cache holds at that moment, so the same prompt read again after other requests gives the same numbers (useful for A/B checks). It costs about 2-5% prompt speed; off by default (the launches of v1.1.57).
- **GLM-5.3 and GLM-5.3-Flash drafting:**
  - The multi-token GEMV now covers Q5_K weights too: draft verification and draft blocks read each weight once for all their tokens; the answers are unchanged.
  - GLM-5.3 on several GPUs: the prompt's hand-over between GPUs copies a quarter of the bytes it did, and `"spec": 1` in a request now turns drafting off for it, as on one GPU.
  - GLM-5.3 on three or more GPUs: answers in Chinese, Japanese, Korean and other scripts are drafted again (the drafts return to the whole vocabulary while the answer is outside the Latin/code one); on one GPU the same rule works again.
  - GLM-5.3: `MAYA_GLM_SPEC_LOG` / `MAYA_GLM_SPEC_TRACE` now report the drafts' acceptance, and the n-gram drafts (`MAYA_GLM_NGRAM_DRAFT=1`) are chosen at every step. On three or more GPUs, `MAYA_GLM_SPEC_TUNE_PLAIN=1` (experimental) lets the speed tuner also try decoding without drafts and keep it where that is faster.
- **DeepSeek on one NVIDIA GPU: decode as fast as v1.1.54 again where the CPU beats the GPU's link** (#920): the one-GPU decode path now stays on only where experts in RAM actually stream faster over PCIe. On a PC whose CPU lane is faster for every expert in RAM it turns itself off after one start, and the next start places the experts as v1.1.54 did (the reporter's RTX PRO 5000 eGPU + Ryzen AI Max+ 395: `--bench` decode 30.2 tokens/s on v1.1.57 against 47.8 on v1.1.54). On our test PC (DeepSeek on one GPU, RAM too small for every expert) decode was about 75% faster than v1.1.57 from the second start on; the first start after the update still runs the old layout once while it measures. PCs where streaming was measured faster keep it. `DS4_STREAM_ONE_WARM=0` keeps the old rule.
- **GLM-5.3-Flash's quants are named with the model first** on screen: "GLM-5.3-Flash · Maya-S", "GLM-5.3-Flash · Maya-XS" and so on, apart from GLM-5.3 (744B) - the setup ids (`glm-maya-s` ...) are unchanged. GLM-5.3 (744B) is now named in every model list (READMEs, setup, models, Docker, issue forms).
- **Refusal ablation per request, groundwork (off):** DeepSeek, MiniMax, GLM-5.3-Flash and GLM-5.3 can switch a refusal direction on and off per request, with no restart (`"ablate": true` / `false` on the OpenAI, Anthropic and Maya's model API requests), once the engine has a direction file (`--ablate FILE`). Maya's own direction files are not published yet: until a model has one, the dashboard shows no switch and its answers are the same as before. [docs/SETTINGS.md](docs/SETTINGS.md#refusal-ablation-per-request-ablate)
- **Fixes:**
  - Codex and other Responses API clients: pictures sent with any `detail` value (low, high, original) are accepted; it is a hint, handled like auto, and no longer fails the request (#914).
  - Windows with an AMD Radeon graphics card: GLM's automatic RAM tier now leaves a tenth of the memory Windows lets a program pin free above it (at least 512 MB), so the start no longer fails with "the warmed expert tables did not upload"; `MAYA_GLM_PIN_ROOM_MB` sets that room on any PC (#924). On a Ryzen AI Max APU under Windows the tier keeps an eighth of the shared GPU memory limit free (1-4 GB) instead of 1 GB, for the same reason.
  - DeepSeek EXL3 quants that ship their own vision tower in F16: the pack tool takes the BF16 tower from DeepSeek's shard 1 (model-00001-of-00048.safetensors) when it is in the model's folder, or from the vision folder a GGUF install made, so pictures turn on by themselves; without either it says what to add (#886).
  - The loop guard no longer keeps a finished answer to the user going (a short reply such as "Loud and clear." in a long chat), and when it does keep an agent's answer going the continuation starts a new paragraph instead of being glued to the last sentence (#921, thanks to @TacoTakumi for #925).
  - DeepSeek on one NVIDIA GPU: the PCIe lane's decode check no longer turns streaming off when it measures a tie, so a tie is no longer saved in lane-times.txt and later starts keep the faster one-GPU decode path; only a measured slowdown turns streaming off (#697).
  - Setup: on a PC with a very small extra card (e.g. a 2 GB display card beside large ones), the automatic GPU choice leaves it out instead of splitting the model onto it, which stopped the engine with "pack: out of memory"; `--gpus` still uses any card.

## v1.1.57 - 2026-11-09

GLM-5.3 (744B) joins Maya: Z.ai's full GLM-5.3 on Maya's GLM engine, with a new look for the project and fixes.

- **GLM-5.3 (744B):** Z.ai's [GLM-5.3](https://huggingface.co/zai-org/GLM-5.3) (744 B parameters, about 40 B active per token) runs on the same engine, server and dashboard as GLM-5.3-Flash. It is shown as "GLM-5.3 (744B) · UD-IQ2_M".
  - **The files:** unsloth's [UD-IQ2_M GGUF](https://huggingface.co/unsloth/GLM-5.3-GGUF) (238.6 GB), pinned and checked shard by shard: `./maya.sh --setup --model GLM-5.3-UD-IQ2_M` (Windows: `START-MAYA.bat --setup --model GLM-5.3-UD-IQ2_M`), or `--gguf-dir <folder>` for the GGUF files you already have. The dashboard's model list offers it too.
  - **Where it runs:** NVIDIA from the Tesla V100 on (CUDA, Linux and Windows) and AMD (HIP, the Strix and Gorgon Halo included). Not on Mac or on Pascal cards. About 256 GB of VRAM and RAM together keeps every expert in memory; with less, the rest is read from the SSD while it answers (keep it on an NVMe SSD).
  - **Text only** (the model has no vision tower; a picture gets a clear message), **up to 1M context** - the setup offers the contexts that fit your GPUs.
  - **Its own defaults** from the model card: temperature 1.0, top-p 0.95, thinking at Max with a 32K-token thinking budget (a request that asks for Max itself may think longer when the context has room), Low and High and thinking off too; Z.ai's own chat template.
  - Everything GLM-5.3-Flash has: drafting with the model's own MTP block, the expert tiers and the CPU lane, speed calibration, tool calls, reasoning in every API (OpenAI, Responses, Anthropic and Maya's model API), the loop guard. Details: [docs/GLM53.md](docs/GLM53.md).
- **A new look (#881):** the new Maya logo in the README, the web page's tab and home-screen icons and the sidebar; issue forms, a docs index and SECURITY.md.
- **Which model fits your PC:** [docs/MEMORY.md](docs/MEMORY.md) gives each model's recommended memory, VRAM and RAM together, with examples.
- **Fixes:**
  - Server: a SIGTERM or Ctrl+C right as the server was waiting could hang it for good; it now stops.
  - Maya's model API (/api/chat, /api/generate): `"think": false` now turns thinking off (it stayed on for every model).
  - DeepSeek: a video sent in a request now gets a clear 400 ("reads pictures, not videos") instead of an encoder error, or being dropped beside a picture.
  - Setup: the line under the logo names the model being set up (it said GLM-5.3-Flash whichever model was picked).

## v1.1.56 - 2026-11-06

Attach any file in the chat - PDFs, Word, Excel, PowerPoint and more - with every model, and community pull requests: fixes, and opt-in speed-ups for AMD and multi-GPU PCs.

- **Files in the chat (#837):** the attach button (and drag and drop) now takes any file, for every model:
  - **PDFs** come in as their text. A scanned PDF with no text in it comes in as its first pages as pictures, for a model with images on. This works on every PC: Maya reads PDFs itself, and uses poppler first when it is installed.
  - **Word, Excel, PowerPoint, OpenDocument, EPUB and RTF** files come in as their text (a spreadsheet as its sheets' rows).
  - **Text and code** in any encoding come in as they are, whatever the file's name.
  - **Videos** attach for the models that read them.
  - A file Maya can't read (an old .doc, an archive, audio, a program) is not attached, and a message says what it is and what to send instead. A file over the model's context limit is cut, and the message says so.
  - **APIs:** the same files work through OpenAI `file` / Responses `input_file` parts and Anthropic `document` blocks, sent inline.
  - Built on the PDF work in #842 (thanks @org-tekeli-borisp).
- **Fixes:**
  - GLM on two GPUs: since v1.1.52 the drafting decode could crash ("illegal memory access") or slow to a crawl on some two-GPU PCs. A two-GPU split now drafts exactly as in v1.1.51, and a draft with no valid token can no longer read past its table on any setup (#860).
  - `MAYA_GLM_NO_SPEC`, `MAYA_GLM_NO_MTP` and `MAYA_GLM_NO_PREFILL` set to `0`, `false`, `off` or `no` now mean off; before, any value turned them on (#860).
  - GLM: a sampled request without a `seed` now gets a fresh random seed, so asking again gives a different answer; a `seed` in the request (or `--seed`) still repeats exactly (#845).
  - GLM EXL3 packs: `MAYA_GLM_SPEC_WIN` no longer stops the load (#857).
  - The pack tool keeps GLM-5.3-Flash's expert selection bias in full precision (F32) instead of rounding it (packs built from now on) (#861).
  - DeepSeek on AMD: the VRAM BAR size is read correctly for the GPU-to-GPU all-reduce check (#809).
  - The Engram RAM lock error now names the current memory-lock limit and how to raise it (#836).
  - GLM: a saved conversation whose slot failed to load now tries the next saved slot and then the checkpoint, instead of reading the whole conversation again (#840, fixes #838, thanks @tragicwinds).
  - DeepSeek: the loop guard now catches DeepSeek's repeated tool calls too; before, it never fired on DeepSeek (#844, thanks @oscarlius).
  - DeepSeek EXL3 quants that name the drafter's head differently (turboderp's) now pack and draft with DSpark on (#835, thanks @pepuscz).
  - `MAYA_DS4_LANE_PARITY` is honoured everywhere it applies (#852, thanks @pattalkslaw-del).
  - The `DS4_HELPER_CHECK` diagnostic no longer fails at random on AMD (it read back through pageable memory) (#868, thanks @pepuscz).
- **Speed-ups, AMD, EXL3 quants (the same answers, bit for bit):** on RDNA 3 / 3.5 (Strix / Gorgon Halo included) the prompt's expert kernels pick their block size by how many tokens an expert takes - about 39% less kernel time on the contributor's RX 7900 XTX (#864); and the decode kernels' inner product is inlined on every AMD GPU (#834) (thanks @pepuscz). Together, on a Gorgon Halo with DeepSeek-V4.1-Flash EXL3 2.25 bpw, against v1.1.55: decode **+4%** (32.3 -> 33.6 tok/s), prompt reading (prefill) **+3%** at 2K and 8K tokens, the same answers.
- **Opt-in speed-ups** (off by default; each one in [docs/SETTINGS.md](docs/SETTINGS.md)):
  - DeepSeek EXL3 read from the model's files, with a helper GPU: prompt experts the helper already holds are not read from the disk again - prompts about 2x faster on the contributor's two RX 7900 XTX plus an iGPU helper; the decode's look-ahead reads EXL3 experts too (`MAYA_DS4_FLOW_HELPER=1`, `MAYA_DS4_AHEAD_EXL3=1`; the same answers; #865, #866, thanks @pepuscz).
  - DeepSeek on RDNA 2 (RX 6000) and on AMD with Maya-XS: two prompt settings, +18-33% prompt speed on the contributor's two RX 6900 XT (`MAYA_DS4_PROMPT_F16=1`, `MAYA_DS4_EXPERT_MMQ=1`; they change the answers slightly; #869, #870, thanks @danexpo).
  - GLM's expert helper: EXL3 packs too, and the helper's work overlapped with the CPU's (+11% decode on the contributor's PC, the same bits); EXL3 decode on Pascal cards (`MAYA_GLM_HELPER_EXL3=1`, `MAYA_GLM_HELPER_OVERLAP=1`, `MAYA_GLM_EXL3_DECODE=legacy-f32`; #849, #850, #851, thanks @arpleasa).
  - DeepSeek: lane parity for MXFP4 experts and tensor-parallel groups (#858, thanks @markrussi).
  - GLM across several GPUs: direct GPU-to-GPU copies on any pair that supports it, not only the MI50 (`MAYA_GLM_P2P=1`, the same answers; #853, thanks @izabera).

## v1.1.55 - 2026-11-05

Codex CLI works with Maya, web-chat answers always run to their end and survive a restart, and a few speed settings now on by default.

- **Codex CLI (#726):** Maya's Responses API (`/v1/responses`) now accepts what Codex CLI sends: its grouped tools (`namespace`, shown to the model as `namespace::name` and returned with their namespace), built-in tools Maya cannot run (`web_search`, `local_shell`, ...) left out instead of failing the request, `reasoning.summary`, and `include: ["reasoning.encrypted_content"]` - the reasoning comes back as an opaque blob that Codex sends with the next turn. Tested end to end with Codex CLI 0.160.0 on GLM-5.3-Flash, DeepSeek-V4.1-Flash and Qwen-3.8-Flash-Next (a shell command run and a file written correctly on each).
- **Web chat: an answer runs to its end unless you press Stop.** Before, the server stopped an answer that no page had read for 5 minutes - a phone whose screen locked during a long thinking answer lost it. Now it always finishes, and the page takes the whole answer up whenever it is opened again; the page also reconnects once a waking phone's Wi-Fi is back. Finished answers are kept for a day (was 15 minutes) and saved in `answers/` next to the config, so a server restart keeps them too; one that a restart cut off comes back with what it had written and a note saying so.
- **Speed settings now on by default** (measured with this release against v1.1.54 on the same PC, quant and prompts):
  - **DeepSeek, prompt reading (prefill):** the indexer's buffers are sized by the prompt instead of the whole context (`DS4_AI_RUN`), and on NVIDIA a two-GPU group's prompt sums run in one pipelined launch, bit for bit the same (`MAYA_DS4_AR2=1`). On 3x V100 with the EXL3 2.9 bpw quant and `--tp 2`: **+1.3%** at 2K tokens, **+3%** at 8K, **+4%** at 32K. Decode is unchanged.
  - **DeepSeek, loading:** the model's files are read in turn instead of one after another (`MAYA_DS4_LOAD_INTERLEAVE`): the EXL3 2.9 bpw quant loads **31% faster** on a Gorgon Halo; the same on 3x V100.
  - **Every model, long agent chats:** the server prepares a re-sent conversation by increment (`MAYA_PROMPT_MEMO`, the same token ids): a follow-up turn after a 115K-token history takes about **6% less time** (measured on GLM-5.3-Flash, 3x V100).
  - **GLM, memory:** the attention's cache in 8 bits by default (`MAYA_GLM_KV_INT8`, as the setup already set it for new installs): half the cache's VRAM (one V100 at 128K context: 1.78 -> 1.13 GB, room for 84 more experts). Checked against the full model: the same quality as 16 bits (within its own run-to-run spread) and no looping. `--kv fp16` keeps 16 bits.
  - `0` turns each one off, in the config's `"env"` block or with `--env`.
- **Docs:** `MAYA_GLM_PREFILL_FUSED` and `MAYA_GLM_MV_RDNA` are listed as on for RDNA4 (gfx12) too, as the code has them.

## v1.1.54 - 2026-11-05

DeepSeek reads prompts (prefill) much faster on AMD Strix / Gorgon Halo and faster on multi-GPU NVIDIA PCs, Qwen-3.8-Flash-Next itself as unsloth's UD-IQ2_XS GGUF, and four community pull requests.

- **Speed-ups, DeepSeek-V4.1-Flash prompt reading (prefill).** Every figure is prompt-reading speed, this release against v1.1.53 on the same PC with the same quant and the same prompts. Decode (writing the answer) is unchanged.
  - **AMD, experts read in place from the model's files** (a quant larger than the GPU's memory, such as the EXL3 2.9 bpw on a 128 GB Strix / Gorgon Halo): a prompt's chunk reads only the experts its tokens use instead of all of them. On a Gorgon Halo (Radeon 8065S) with the EXL3 2.9 bpw quant: **+250%** at ~2K tokens, **+178%** at ~9K tokens. The same answers.
  - **AMD Radeon RDNA 3 / 3.5 (gfx11, Strix / Gorgon Halo included):** the indexer's scores run on the matrix cores by default. On a Gorgon Halo with the EXL3 2.25 bpw quant: **+3%** at 8K tokens, **+6%** at 32K, **+10%** at 64K. Changes the answers slightly (the quality against the full-precision model is the same); `DS4_INDEX_TC=0` turns it off.
  - **NVIDIA, several GPUs, EXL3 quants whose experts are streamed from RAM:** two placement settings are now on by default - the prompt path copies experts in larger batches over fast links, and DSpark's draft blocks spread over a tensor-parallel pair. On 3x V100 with the EXL3 2.9 bpw quant: **+14%** at ~2K tokens, **+7%** at ~9K, **+4%** at ~35K. They change which experts sit in VRAM, so the answers change slightly (the quality against the full-precision model is the same); `MAYA_DS4_RING_FAST_LINK=0` and `MAYA_DS4_DRAFT_SPREAD=0` turn them off.
  - The off switches go in the config's `"env"` block, or on the command line with `--env`.
- **Qwen-3.8-Flash-Next (the base model), unsloth's UD-IQ2_XS GGUF:** the setup and the Models page's Add a model offer it beside the other models (NVIDIA from the V100 on, and AMD), with pictures, its MTP drafter, the model's own sampling and the n-gram table from Maya's Hugging Face repo (`./setup.sh --model qwen-ud-iq2_xs`). unsloth's other UD quants of it run on the same code but are not listed.
- **Qwen, Monitor:** "Where the experts live" names its figure for what the engine counts - experts read from the model file, through the page cache or the SSD - and advises more RAM only when experts really live on the SSD alone.
- **Fix, Qwen chats:** a follow-up message read the whole conversation again; it now reads only the new part. Tools reach the model in the format of Qwen's official template.
- **Fix, Pascal cards (compute capability 6.x):** the engine's build stopped at a compile error; it compiles again.
- **Community pull requests:**
  - AMD on Linux with a distribution's own ROCm 6.x packages (Debian / Ubuntu layout): the setup finds them, and goes on with a warning when you set `MAYA_HIP_ALLOW_ROCM6=1` (ROCm 7 stays the default requirement); CPUs without AVX2 no longer stop the AMD setup - the CPU lane uses ggml's kernels, as on NVIDIA (#832, thanks @Huskynarr).
  - Opt-in: `maya-mm3 --ablate` projects a direction out of MiniMax's residual stream (refusal ablation with a control vector), as `maya-ds4 --ablate` does for DeepSeek. Off by default; it changes the answers when used (#826, thanks @JoshBolding).
  - Windows with several AMD cards: the Monitor reads the card the engine runs on, matched by its PCI bus, instead of going by Windows' adapter order (#825, fixes #780, thanks @Meistertech-JP).
  - AMD Instinct MI50 (gfx906): native Wave64 reductions in the DeepSeek, GLM, MiniMax and prompt kernels; other GPUs are unchanged (#822, thanks @nicholas631).

## v1.1.53 - 2026-11-04

A new look for the web UI's Models, Monitor and Settings pages, MiniMax on Apple Silicon (experimental), fixes for third-party DeepSeek quants and failed updates, and many community pull requests.

- **Web UI:**
  - **Models:** each model's details, engine settings and storage are in one sheet with three tabs (Overview, Engine, Storage). The delete dialog now lists each folder and file with its size, and keeps what another model still uses (a shared folder, the image encoder, a DFlash2 drafter other configs draft with - they no longer fall back to MTP). Maya deletes only inside the models folders it knows, by real path, and not while a model downloads.
  - **Models, Add a model:** lists the other quants the setup offers on this PC (LibertAIDAI's DeepSeek EXL3, unsloth's GGUFs, orcarouter's abliterated quants, Mia-AiLab's MiniMax EXL3, turboderp's GLM EXL3), each downloaded through the setup. An EXL3 quant set up by hand is named by its own bits, from its folder's name.
  - **Context size:** one setting, in Settings > Chat defaults and in a model's Engine tab; a change to the running model's context reloads it in place.
  - **Monitor:** prefill shows the prompt read's peak (this request's, then the last one's) instead of its latest figure; the stats as a 2 x 2 grid; figures and labels lined up on phones.
  - **Settings:** five groups (General, Integrations, Usage, Privacy, About), a list then a page on a phone; "Appearance"; the running model's port beside it in Chat defaults; Connect your tools as before. Cards, buttons and fields as fills instead of outlines; the thinking chip says "Thinking · <level>".
  - **Support:** a small heart beside the model's name in the sidebar and a line on About (and under an update's button) link to GitHub Sponsors. No popups.
- **MiniMax-M3.1-Flash on Apple Silicon (experimental):** a Metal engine of its own, `maya-metal-mm3`, runs Maya's MiniMax GGUF quants with every expert resident (Maya-XS fits a 128 GB Mac), text only, with parity tests against the reference forward. The setup does not offer it yet: docs/METAL.md says how to build and start it by hand. Nothing else in the engines changes (#778, thanks @Franzferdinan51).
- **DeepSeek, other EXL3 quants from Hugging Face (#806):**
  - `tools/ds4/pack.py` accepts a head kept as BF16 (`head.weight`); the engine already reads it.
  - Shards that write their tensors grouped by type instead of expert by expert stop the pack with that reason and the fix (rewrite them in expert order); the engine's own message says the same instead of "split across files".
  - Engram tables a quant stored in its own INT4 layout stop the pack with a message that points to DeepSeek's FP8 shards 47 / 48; docs/MODELS.md says so too.
- **Updates on NVIDIA and Apple Silicon (as on AMD in v1.1.52):** an update whose engine rebuild fails, or whose compiler or llama.cpp source is gone, stops with the reason and how to rebuild instead of starting the engine from before under the new server and config.
- **Monitor, tensor-parallel pairs:** "What fills the GPUs" no longer says a restart gives more experts for free VRAM that the GPU's tensor-parallel partner cannot match (the members of a `--tp` group hold the same expert slots). That room is shown as idle, and with DeepSeek's DSpark drafter it points to `MAYA_DS4_DRAFT_DEV`, which can place drafter blocks there.
- **DeepSeek, a GPU fault:** a request that leaves a GPU in a fault ends the engine with an error, and the next request starts a fresh one; before, every later request failed until a restart by hand (#688, thanks @sociolog).
- **Community pull requests** (every setting below is opt-in and off by default; unset, the engines run as in v1.1.52):
  - Opt-in: DeepSeek decode runs Q8_0 GEMVs of 3-16 columns on the tensor cores of sm_80 and newer, with the same bits (`MAYA_DS4_Q8MV_TC=1`, #810, thanks @alesha-pro).
  - Opt-in: DeepSeek prompts on a layer split - DMA from pinned RAM copies of any format (`DS4_FLOW_COPY_DMA`), the indexer's slabs sized by the prompt (`DS4_AI_RUN`), both with the same answers; chunks sized by the prompt's length (`DS4_CHUNK_AUTO`) and the YOCO skip on the flow (`DS4_YOCO_FLOW`), which change the answers slightly. On the author's 4x RTX 3090 they read prompts 29-72% faster (#811, thanks @alesha-pro).
  - Opt-in: DeepSeek long conversations - experts lent to a prompt's buffers come back from pinned RAM copies instead of the disk (`DS4_LENT_RAM_GB`), and the server builds a turn's token ids by increment from the last prompt's, the same ids (`MAYA_PROMPT_MEMO=1`, every model) (#813, thanks @alesha-pro).
  - Opt-in: DeepSeek's CPU lane gives an expert the same bits as in VRAM (`MAYA_DS4_LANE_PARITY=1`, #741, thanks @cresslank).
  - Opt-in, the same bits: the DeepSeek CPU lane cuts a small EXL3 call's pieces by columns too (`DS4_CPU_SPLIT_COLS`, #758); the AVX2 EXL3 CPU kernels work in 16-bit lanes for calls of 1-3 activations (`DS4_CPU_W16`, #759) and cut byte-aligned trellis windows with byte shuffles (`MAYA_EXL3_AVX2_SHUF`, #760); DeepSeek expert placement keeps recent slots that survive per-byte placement (`DS4_RECENT_FIT`), and `DS4_PPL_STEP` is a test aid (#761) - thanks @bitaccesscomau.
  - Opt-in: AMD Instinct MI350 / MI355 (gfx950) read EXL3 prompts on their matrix cores (`EXL3_WMMA_GFX950=1`, #692, thanks @yagop).
  - Linux on POWER9 (ppc64le, IBM AC922): Maya builds and runs there; docs/IBM_AC922.md (#753, thanks @jefflaichunyin).
  - `tools/agent_bench.py`: a benchmark shaped like coding agents - several long conversations at once against a running server, fixed by a seed (#406, thanks @somnergy).
  - docs/DOCKER.md and `docker/`: a recipe to build a Docker image of a release; there is no official image (#808, thanks @noonghunna).
  - README: a 4x RTX 4090 DeepSeek row in the speed table (#814, thanks @oscarlius).

## v1.1.52 - 2026-11-03

Faster GLM decode, fixes for AMD PCs on Windows (setup, updates and start errors), a DeepSeek config check, and community pull requests.

- **GLM decode:** the DSA layers' feed-forward runs in CUDA graphs by default. On the PCs it was measured on, decode is about 8-9% faster, with the same greedy answers (#795, thanks @saatvik-viniak).
- **AMD PCs on Windows (#805):**
  - A first setup without `--backend hip` finds an AMD-only PC (every card the HIP check accepts, named by an installed ROCm's `hipInfo`) instead of stopping at `nvidia-smi`.
  - An update whose engine rebuild fails stops with the reason and how to rebuild (`build-maya-hip.bat > build-hip.txt 2>&1`, or `START-MAYA.bat --setup --rebuild`) instead of starting the engine from before under the new server and config.
  - An engine error in the middle of an answer ends that request with an error instead of stopping the server.
  - A failed GPU buffer zeroing at start names its size and the driver's reason.
- **DeepSeek, `reasoning_effort` in the config (#807):** it is checked before the model loads, so a bad value stops the start at once instead of after the load. A number above 100 (DeepSeek's effort is a level from 1 to 100) gets a message that points to `"thinking_budget"`, the cap on the thinking in tokens. docs/SETTINGS.md says which setting is which.
- **DeepSeek on Linux:** the n-gram table readers use twice the CPU's hardware threads (between 16 and 64) instead of a fixed 8, so cold prompts read faster on larger CPUs (#716, thanks @SeriousRising).
- **Models page:** a folder that another model reads files from is no longer listed under "Files no model uses".
- **Community pull requests:**
  - GLM on several GPUs: an opt-in RAM tier that keeps each expert's halves on both CPU sockets (`MAYA_GLM_ROW_HALVES=1`), plus CPU-lane pinning and profiling settings (#795, thanks @saatvik-viniak).
  - CPU lane: IQ1_S rows decode in vectors, with the same results (#695, thanks @sociolog).
  - Opt-in: GLM decode defers the CPU's experts on every part of a split (`MAYA_GLM_DEFER_CPU=2`, #722, thanks @oisee).
  - DeepSeek EXL3 packs: an F16 dense matrix loads as BF16 (#802, thanks @pepuscz).
  - Opt-in: DeepSeek on RDNA 3 runs the indexer's scores on WMMA (`DS4_INDEX_TC=1`, #801, thanks @pepuscz).
  - Experimental AMD MI50 (gfx906) support: the wave64 check, opt-in peer-to-peer split hops (`MAYA_GLM_P2P`, #397), and MiniMax wave64 attention kernels (#800), thanks @nicholas631.
  - Opt-in: DeepSeek `--tp 2` on NVIDIA reads prompts with an exact peer-VRAM all-reduce (`MAYA_DS4_HIP_PEER_AR`, #773, thanks @olobarius).

## v1.1.51 - 2026-11-03

DeepSeek fills every GPU in tensor parallel and reads long prompts faster when most experts live in RAM, a round of issue fixes, the README in Japanese, Simplified Chinese and Brazilian Portuguese, and community pull requests.

- **DeepSeek tensor parallel fills every GPU:** each GPU's part of the experts now follows the room it has left (after the drafter, the head and buffers), so no GPU keeps VRAM unused. On the PCs it was measured on, about 10% more experts sit in VRAM and decode is up to 10% faster. `DS4_TP_SHARES=off` keeps the even split (#766).
- **DeepSeek long prompts with most experts in RAM:** on several GPUs, the prompt path borrows the experts' room and reads long prompts in larger pieces, and the borrowed experts are read back from the page cache instead of the disk. Prompts from about 10K tokens read up to about 12% faster, depending on the setup, and about 7% more experts stay in VRAM; prompts around 8K are about 3% slower on those setups. Setups whose experts are mostly in VRAM are unchanged.
- **Several conversations at once:** stopping one answer no longer disturbs the others (#791, thanks @nicholas631).
- **GLM engine watchdog:** many short requests back to back (the zero-shot suite) no longer trip the "no progress" watchdog on a healthy engine.
- **Qwen engine:** a prompt read the client abandoned no longer makes every later request fail with "cancelled" (#703).
- **CYBER-FROST:** the dashboard has a thinking Off button.
- **Dashboard:** while the model thinks, its thoughts follow the newest line down, as the answer does (scroll them up to read and they stay put).
- **Dashboard, long answers:** a page that lost its connection to an answer for a long time (a sleeping phone, a background tab) no longer gives the answer up for good: it tries again when it is shown, and if it has stopped trying, coming back to the page or reloading it takes the answer up again while the server still holds it.
- **DeepSeek:** a failed GPU allocation says its size, the driver's reason and the free VRAM (#794); a numeric `reasoning_effort` in the config (`"42"`) is the model's effort level from 1 to 100, not an unknown word (#768; the cap on the thinking in tokens is `"thinking_budget"`); `DS4_AR_P2P` reads the real BAR size, so it engages on large-BAR cards (#798); `DS4_RAM_MARGIN_GB` sets the system-RAM headroom the memory planning leaves, default 8 GiB (#739).
- **MiniMax:** the "prefill not calibrated" note says when the saved calibration is for another GPU list, runtime or driver (#721).
- **Savings ledger:** `MAYA_SAVINGS_PATH` chooses its file (`off` keeps none), and a file that cannot be written is logged once (#724).
- **Responses API:** the rejections name the tool type or include value the client sent (#726).
- **Text dashboard:** the GPU table's columns keep their width (#793).
- **README:** the Models page and API, Turbo, the thinking budget, the loop guard, GLM and DeepSeek Maya-XS, Mac state, users' speed reports, and Japanese, Simplified Chinese and Brazilian Portuguese translations.
- **Community pull requests:**
  - DeepSeek `--parallel`: when the engine ends, every conversation is set aside, not only the bound one (#732, thanks @hajekt2).
  - DeepSeek `--parallel`: the read scheduler settings also work under their older aliases (#682, thanks @handmade0octopus).
  - A GLM setup run again keeps the config's host, API key, CORS origins and MCP servers (#720, thanks @tobitege).
  - Opt-in: GLM `--parallel` keeps conversation checkpoints for each conversation (`MAYA_GLM_MULTI_CKPTS=1`, #714, thanks @coco-famihiro).
  - Opt-in: DeepSeek tensor-parallel layers with every expert in VRAM can leave the host out (`MAYA_DS4_HOST_FREE_TP=1`, #701, thanks @loktar00).
  - DeepSeek warns when another program holds a GPU's memory, and retries an expert allocation a few times before it fails (#700, thanks @loktar00).
  - Opt-in: DeepSeek prepares its CPU row checks while the model loads (`MAYA_DS4_CPU_PREPARE_ROWS=1`, #781, thanks @oscarlius).

## v1.1.50 - 2026-11-02

Models you can download, swap and delete from the Models page or the API, a Turbo button, a thinking budget you can set, GLM Maya-XS, faster long prompts on DeepSeek, and a round of community pull requests.

- **Models page:** the catalog lists every Maya model with its size. Download one with a progress bar, load it, unload the running one, or delete one (a file that another model's folder links to stays). Nothing needs a command line.
- **Model API:** `/api/tags`, `/api/ps`, `/api/show`, `/api/chat`, `/api/generate`, `/api/pull` and `/api/delete` list, load, download and delete models from any app or script. A request for a model that is not loaded swaps to it, and `keep_alive` unloads it when idle. `/api/ps` shows "loading" while the weights load.
- **Turbo:** one button beside the model's parameters sets the fastest settings for the model (greedy sampling, no thinking, 32K context on GLM and MiniMax, expert skipping, the fastest drafting). It says "Turbo ON" with a flame while it is on, and switching it off restores your settings exactly. It trades some quality for speed; on the smallest quants (GLM and DeepSeek Maya-XS) it keeps every expert, so their answers stay sound.
- **Thinking budget:** a field in the settings sets how many tokens the model may think before it answers, and applies at once. The default stays 32768; GLM Maya-XS defaults to 8192.
- **GLM-5.3-Flash Maya-XS:** a new, smaller GLM quant (about 73 GB) for PCs with 64 GB of RAM and a 24 GB card at 128K context. The setup offers it as `glm-maya-xs`, with the int8 KV cache, a thinking budget of 8192 and sampling of temperature 0.7, top-p 0.95, min-p 0.05.
- **DeepSeek long prompts:** the prompt chunk now follows how many experts live in RAM. Where RAM holds most of them, a long prompt reads up to about 19% faster; where the VRAM holds them, nothing changes.
- **DeepSeek thinking budget 0:** the reasoning closes again, so answers and tool calls no longer land in `reasoning_content` (#786).
- **Maya-S n-gram tables:** Maya-S reuses Maya-XS's tables instead of downloading a second pair (#776).
- **Log timestamps:** every `[maya]` log line starts with the time, in the terminal and in the log file (#783).
- **Gentoo:** the HIP check finds versioned LLVM folders (#787, thanks @Wilsonfabozi).
- **Community pull requests:**
  - `/tokenize` and `/detokenize` endpoints (#764, thanks @cresslank).
  - A chat request's `response_format` reaches the model (#763, thanks @cresslank).
  - `ignore_eos`, and `logit_bias` on DeepSeek (#772, thanks @cresslank).
  - GLM prompts: the CPU lane takes a share where PCIe is not much slower than the CPU, with `MAYA_GLM_PREFILL_CPU_GATE` (#752, thanks @hugou74130).
  - DeepSeek on AMD keeps the router's weights in BF16 (#745, thanks @mrsirg97-rgb).
  - DeepSeek flow prompt chunks: an opt-in DMA/host all-reduce (#736, thanks @hajekt2).
  - Metal: no free-block exhaustion on long-context prefill (#742, thanks @appleguy).
  - Opt-in, off by default: peer-to-peer all-reduce over ReBAR for DeepSeek on 2+ GPUs, up to 29% faster prompts on the PCs it was measured on (`DS4_AR_P2P=1`, #762, thanks @486DX4-100). It was a little slower on V100s, so Maya does not turn it on.
  - Speed table and AMD docs (#756 thanks @Cayaf, #749 thanks @Diudid).

## v1.1.49 - 2026-11-02

A fix release: Maya-XS updates start cleanly and give back their disk space, and Qwen CYBER-FROST runs on 8 GB cards again.

- **DeepSeek Maya-XS v1 -> v2 update:** a start could stop with "changed after the pack was built" even after `--setup --repack`, because the old copy of the n-gram tables was checked instead of the shared one the model reads (#777, thanks @waxinz). Running the setup once also turns that old copy into a link to the shared one, giving back about 161 GB.
- **Qwen CYBER-FROST on 8 GB cards:** v1.1.48 could run out of GPU memory on the first request or during the speed calibration. Maya now keeps room for the first request's buffers on small cards; bigger cards are unchanged. On an RTX 3050 8 GB it writes about as fast as before or faster, and the speed calibration completes.

## v1.1.48 - 2026-11-01

Faster prompt reading for DeepSeek on several GPUs, no more re-reading after the thinking budget, and fixes.

- **DeepSeek on several GPUs, with some experts read from the disk or RAM:** prompts are read in 4K-token pieces by default, which reads an 8K prompt about 24-30% faster on 3x V100 and leaves room for about 3.5% more experts on the GPUs. `--prefill-chunk` still sets it by hand.
- **DeepSeek's thinking budget:** when the budget closes the thinking, the next turn no longer reads the whole thinking again (59 s at 32K tokens before); it continues from the cache (#746).
- **`MAYA_DS4_READ_ORDER=arrival`:** a running answer keeps writing while another long prompt is read, instead of stopping until the read ends (#751).
- **Setup screen:** no longer stuck at "Loading the model" when the model is already serving (#729).
- **`--engine strata`** (the old name in older run scripts) starts the real engine again; v1.1.47 started the test engine instead (#770, thanks @cresslank, #771).
- `tools/ds4/slot_keep_test.py` runs on Windows (#743).

## v1.1.47 - 2026-11-01

A better and faster DeepSeek-V4.1-Flash Maya-XS, smaller downloads for unsloth's DeepSeek quants, and a cleaner Monitor.

- **DeepSeek Maya-XS v2:** about 12% closer to the original model (KLD 0.251 against 0.284) and about 20% faster at writing, at almost the same size (117 GB). The weights every token uses are now stored more compactly, which leaves room for more experts on the GPU, and 12 layers get a higher-quality format. Running the setup again asks whether to update an existing Maya-XS or keep it: the update downloads only the new model files (117 GB), keeps the n-gram tables, and removes the old files once the new ones are complete.
- **One set of n-gram tables for every DeepSeek quant:** unsloth's quants now use Maya's Q6_K n-gram tables, shared with Maya's own quants on disk. Measured with the same scorer, they lose nothing against the larger MXFP8 tables, and the Q8 download drops from 508 to 467 GB.
- **Monitor:** the peak decode speed sits small beside the live Decode figure, Decode and Prefill sit side by side on a phone, and Speculation shows the share of drafts kept as the main figure, with the drafter below it.
- **Load and unload models from any API client or command:** the `/models/load`, `/models/unload` and `/models/sse` endpoints are now on by default (`"model_management": false` turns them off).
- **`--engine maya`** for the server (the old name still works), and Maya introduces itself as Maya to MCP servers.

## v1.1.46 - 2026-10-31

DeepSeek reads long prompts about 5% faster on AMD Strix / Gorgon Halo, and three fixes.

- **Strix / Gorgon Halo:** DeepSeek reads a 16K prompt about 5% faster (the prompt is read in 4K-token pieces on these GPUs, with the same answers).
- **Long agent sessions:** a request whose messages carry their content as parts no longer fails with an error (#717).
- **Tool calls:** a tool call with an unclosed parameter tag is handled as a malformed call instead of failing the request (#706).
- **DeepSeek on Windows:** the in-place memory trim settings given in a config now reach the engine (#728).

## v1.1.45 - 2026-10-31

A fix for DeepSeek-V4.1-Flash on one GPU.

- **DeepSeek on one GPU: no more illegal memory access after a few long prompts.** After several requests of different lengths, a long prompt on one GPU could overlap experts the engine was still loading in the background, and the next step failed with an illegal memory access (often reported later as an error in an unrelated call). The background loads are now finished before a prompt uses that memory. Prompts of 32K, 64K and 128K on one GPU run, with the same answers and the same speed as before. The problem had been there since at least v1.1.41; setups with two or more GPUs were not affected.

## v1.1.44 - 2026-10-31

DeepSeek-V4.1-Flash's setup lists one more of unsloth's quants, UD-IQ1_M, for PCs with about 150 GB of VRAM + RAM, and the Q8 now says in the setup how close it is to the original.

- **UD-IQ1_M in the setup's list** (`--ds4-quant gguf-iq1_m`): unsloth's UD-IQ1_M, about 140 GB of VRAM + RAM plus 110 GB of n-gram tables read from the SSD (a 250 GB download). It sits after Maya's own quants, before UD-IQ3_S.
- **The Q8 (UD-Q8_K_XL) in the setup:** its line now gives its quality next to the others' (the closest to the original model), and it and UD-Q4_K_XL are pinned to unsloth's current revision. The files are the same, so an existing install downloads nothing again.
- **docs/MODELS.md:** the unsloth rows say in short who each is for.

## v1.1.43 - 2026-10-31

DeepSeek-V4.1-Flash now runs unsloth's DeepSeek GGUF quants, and the setup lists UD-IQ3_S: close to Q8's quality for PCs
with about 270 GB of VRAM + RAM. Long agent conversations are no longer read again from the start when the loop guard
steps in (the next turn came back about 15x sooner in our tests), answers that write a turn-end marker inside their
thinking now finish, and the Monitor shows which drafter runs and how many of its drafts the model keeps, live. Maya
started as a service skips the speed calibration instead of looping, and experts read from the SSD use as many reads at
once as the drive serves fastest. Against v1.1.42 on our test PCs, decode and prompt speeds
are unchanged for every model.

- **DeepSeek: unsloth's GGUF quants** (thanks to unsloth for the quants):
  - **UD-IQ3_S** is in the setup's list, between Maya-S and the Q8 (`--ds4-quant gguf-iq3_s`). It measures close to
    Q8: a KLD of 0.061 against the original model, where Q8 has 0.047 and Maya-S 0.119 (KLD: how far the quant's
    next-token choices drift from the original's; lower is better). About 262 GB of VRAM + RAM hold it, plus the
    161 GB of n-gram tables read from the SSD.
  - The other unsloth quants can be set up by hand: `--ds4-quant gguf-iq1_s`, `gguf-iq1_m`, `gguf-iq2_xxs`,
    `gguf-q2_k_xl`, `gguf-iq3_xxs` and `gguf-q3_k_xl`, pinned to unsloth's 2026-10-28 upload; UD-Q4_K_XL and
    UD-Q8_K_XL work as before.
  - Pictures with every unsloth quant, from the picture encoder file in unsloth's own repo (a 0.97 GB download).
  - One copy of the n-gram tables: UD-IQ3_S and Maya's quants hold the same 161 GB of tables, so with one of them on
    the disk the setup of the other reuses that copy instead of downloading it again.
  - The setup's free-space check counts each file against the disk it actually lands on.
- **Agents and the loop guard:**
  - A long conversation's prompt is kept whole when the loop guard steps in, so the conversation cache is still used:
    on a 150K-token agent conversation the next turn took about 15x less time with GLM and with DeepSeek (#704, thanks
    @tonykee; fixes #702). `MAYA_LOOP_GUARD_COLLAPSE_MAX` sets the size from which a prompt is left whole.
  - The loop guard's follow-up call always fits the context: an answer it lets go on, or a request without
    `max_tokens`, no longer ends in an engine error (a v1.1.41 regression; #696, thanks @Jellepepe).
  - A turn-end marker written inside the thinking is now read as the model's text: the answer goes on and finishes
    instead of stopping mid-thinking, with DeepSeek and GLM, and a client's own stop strings work inside code again
    (#709, thanks @markrussi).
- **Monitor:** the Speculation card shows the drafter that runs (DSpark, DFlash2, MTP, prompt lookup) and the share of
  its drafts the model keeps, live while an answer is written, for GLM, DeepSeek, MiniMax and Qwen. "SSD only" in
  "Where the experts live" now shows its GB, and the Throughput card has no blank space on a phone any more.
- **Running Maya as a service:** a start with no terminal (systemd, a script, or `--yes`) begins with the default
  settings instead of measuring first, and prints the one command to run the calibration later. A service no longer
  loops for ever when its start timeout ends the calibration. docs/SETUP.md has a systemd section (#690, thanks
  @ZackO2o for the report).
- **Experts read from the SSD (Linux):** how many reads run at once is now measured for each drive at its first start (about
  a second, then remembered), instead of guessed from the CPU. On an NVMe beside an 8 GB card, Qwen CYBER-FROST
  decoded up to 5% faster with its experts read from the SSD.

## v1.1.42 - 2026-10-30

Pictures on every model and every build: GLM-5.3-Flash now reads pictures on AMD GPUs too, Qwen-3.8-Flash-Next
(CYBER-FROST) reads pictures for the first time, and DeepSeek and MiniMax read them on every setup again. DeepSeek reads
long first prompts 16-24% faster and its two-GPU tensor-parallel prompts about 1.5x faster on PCs with few CPU cores,
GLM's automatic prompt chunk is up to 3.4x faster on cards where it chose too small a chunk, and an idle Maya no longer
keeps a CPU core busy. Against v1.1.41 on our test PCs, an 8K-token DeepSeek prompt
read about 1.27x faster on one GPU and 1.1x faster with `--tp 2`; decode, and the other models' speeds, are unchanged.

- **Pictures on every model and every build:**
  - GLM-5.3-Flash reads pictures on AMD GPUs - Linux and Windows, discrete cards and Ryzen AI Max APUs (#345, thanks
    @Ehco). The encoder's matmuls and attention now add up in float32 on the GPU, so its answers about a picture match
    the CPU encoder's on AMD (they drifted on a few image patches before) and on NVIDIA's newer cards too.
  - Qwen-3.8-Flash-Next (CYBER-FROST) reads pictures. The setup downloads its 0.9 GB image encoder (it asks first),
    and the encoder runs on the GPU in memory the engine lends it while a picture is read: on an 8 GB card the expert
    cache and, if need be, the draft layer's experts are lent and read back afterwards. On an 8 GB card a picture is
    read about 3.6x faster than with the CPU encoder; text answers before and after a picture are the same.
  - DeepSeek reads pictures with the EXL3 quants on V100s too, and the setup says why a PC short of RAM reads text
    only. A MiniMax setup run again turns pictures on (it kept an earlier "off" before). `--no-vision` keeps them off
    for every model.
- **DeepSeek:**
  - Long prompts (#691, thanks @needmorevram): the restore points near the end of a prompt are kept in one read, and
    a chunk streams only the experts it picks. A fresh 20-30K-token prompt read 16-24% faster on one and on two GPUs
    in our tests, with the same output. Lending EXL3 chunk buffers from the expert cache and reading host-summed
    pieces back by DMA are opt-in for now (`MAYA_DS4_EXL3_LEND=1`, `DS4_AR_BACK_DMA=1`).
  - GGUF `--tp 2`: each GPU of a tensor-parallel pair reads only its own part of an expert from the files - on a PC
    with few CPU cores its prompts took about two thirds of the time (#670, thanks @speja969 for the measurements).
  - A slow PCIe link (one expert takes 1 ms or more) streams one expert at a time, and the link is measured on the
    same stream as the loads, which gave back the decode speed lost on such a PC (#609, thanks @SeriousRising, who
    found the cause).
  - A GPU too small for the model's dense weights stops at the start with what it needs, instead of a memory error
    part way into the load.
- **GLM:** `--prefill auto` no longer falls back to 768-token chunks when the prestage buffer does not fit whole beside
  a bigger chunk; on such a card a 19K-token prompt read about 3.4x faster (#673, thanks @pietroMastro92).
- **An idle Maya stays idle** (#685, thanks @vbooka1 for the measurements): after a minute without a request the
  engines' CPU-lane threads sleep until the next request wakes them (DeepSeek's used up to 12% of a core doing
  nothing, now none), and the setup screen and the hardware sampler slow down when nothing runs. The first answer
  after a quiet spell comes as fast as before.
- **Settings:** a `MAYA_...` setting in a config's `"env"` now reaches the engine under either of its names (#686,
  thanks @mgabor3141).
- **Setup:** a discrete NVIDIA card without Resizable BAR is no longer called "shared with the system" (#671, thanks
  @speja969).
- **MiniMax:** a routing id outside the experts ends the request with an error that says where, instead of a crash.

## v1.1.41 - 2026-10-30

A community release: 36 contributions. DeepSeek reads the first long prompt after a start about 1.3x faster on
multi-GPU PCs and is faster on PCs short of RAM, agents get a loop guard that works across turns and a cancelled
prompt read that is resumed, Open WebUI can manage the model, and AMD GPUs get faster GLM prompts on Strix Halo, the
DFlash2 drafter on one Ryzen AI Max APU and early Instinct support. Against v1.1.40 on our test PCs, decode and
prompt speeds are otherwise unchanged (GLM, DeepSeek on one and on three GPUs, MiniMax, Qwen).

- **DeepSeek: the first long prompt after a start is read about 1.3x faster** on a multi-GPU PC (#612, thanks
  @Whamp): a short warm-up's timing no longer misleads the engine's choice of how to read a long prompt. On three
  V100s (`--tp 2`) the first 8K-token prompt now reads as fast as the ones after it.
- **DeepSeek on a PC short of RAM** (#669, thanks @xiongqiyang): with `--mmap-experts off`, the experts copied to RAM
  are now shared fairly between the layers, so no layer is left with none. On our RAM-limited test (32 GB, one GPU)
  a long answer took about 20% less time, with the later part about 2x faster, and the same tokens.
  `--mmap-experts auto` (the default) is unchanged. `DS4_COPY_FAIR=0` restores the old order.
- **Agents:**
  - The loop guard now works across turns (#640, thanks @merbanan): an agent that makes the same tool call with the
    same result a third time gets a short note in its prompt, and one that keeps going is steered off it. Normal
    agent conversations, and a call repeated after an edit, get exactly the same prompt as before. On by default.
  - GLM: a cancelled prompt read keeps its last checkpoint, so an agent that times out and retries goes on from there
    instead of reading the whole prompt again (#573, thanks @rushisan). Keeping the snapshot before the generation
    prompt, so a cancelled answer's next prompt continues it, is opt-in for now (`MAYA_GLM_SNAP_GEN=1`, #572, thanks
    @rushisan).
  - Very long streamed tool arguments no longer slow down as they grow, and streamed replies are sent without delay
    (#626, thanks @SeriousRising).
- **Server and clients:**
  - Open WebUI's Models page can load the model, free its GPU memory and watch its state through llama.cpp's
    `/models/load`, `/models/unload` and `/models/sse` - off unless `"model_management": true` is in the config
    (#678, thanks @rilken-star).
  - `/v1/models` lists each model's `context_length` for clients that size their requests by it (#649, thanks
    @NorwinYu).
  - `/status` asks for the API key when one is set, and a request during an engine start gets a retryable 503
    (#677, thanks @rilken-star). A reply that ends or is cancelled keeps its final statistics (#652, thanks
    @LostVector).
- **AMD GPUs:**
  - Strix Halo: GLM's prompt experts run with a better occupancy for GLM's shape - about 11% less time for that step
    on our Halo, the same results bit for bit (#634, thanks @mrsirg97-rgb).
  - One Ryzen AI Max APU (Strix / Gorgon Halo) is offered the DFlash2 drafter for GLM in the setup; MTP stays the
    default and the speed calibration keeps DFlash2 only where it is faster (#628, thanks @mrsirg97-rgb).
  - A hipBLASLt tuning preset for GLM's prompt GEMMs, and a warning when a shape is not in the table (#582, thanks
    @precisionailabsllc-collab).
  - EXL3: a specialised one-row decode kernel by default (#470), configurable prompt tiles on RDNA4 (#593) and RDNA4
    matrix prefill, opt-in with `EXL3_WMMA_GFX12=1` until we can measure it (#365) - thanks @LostVector.
  - Experimental Instinct MI300X / MI325X (gfx942) and MI350X / MI355X (gfx950) support, compile-checked (#468, #581,
    thanks @yagop).
  - DeepSeek on two AMD GPUs: an exact peer-VRAM transport for prompts, opt-in (`MAYA_DS4_HIP_PEER_AR=auto`) until a
    two-GPU AMD PC has measured it (#592, thanks @LostVector). The PCIe lane's timing uses the GPU clock on RDNA3 /
    RDNA4, and `DS4_IQ1S_ROWS` chooses IQ1_S's own CPU rows (#563, thanks @Cass67).
- **GLM:**
  - The expert pool keeps at least one route's experts of every size class, so mixed-recipe GGUFs no longer evict
    their own route (#660, thanks @Cass67).
  - IQ1_S / IQ2_S / IQ3_S prompts no longer read garbage on Windows with CUDA 13.2 (#487, thanks @Fringe210).
  - Prompt GEMVs over IQ weights read each weight block once per batch of columns, bit for bit as before (#560,
    thanks @precisionailabsllc-collab), and two small route-kernel optimisations (#451, #486, thanks @merbanan).
  - Windows + NVIDIA, one GPU: a balanced RAM-resident expert plan, opt-in with `MAYA_GLM_RAM_BALANCED=1` (#153,
    thanks @PhysicistLeon).
- **DeepSeek:**
  - Pictures: the vision tower no longer comes out zero when built with C++20 and CUDA 12.1 (#676, thanks
    @rilken-star).
  - DSpark's drafter skips the rows a round cannot keep, with the same drafts (#622, thanks @SeriousRising).
  - `--experts N`: a token takes the router's top N experts (lossy; an engine flag for experiments, #564, thanks
    @Cass67).
  - Drafter and verification timings in the STAT line and `/metrics/prometheus` (#647), and opt-in placements for
    a helper GPU with EXL3 experts and for DSpark's blocks (#527), with the helper's experts shown on their own in
    the Monitor (#599) - thanks @LostVector.
- **CPU lane:** IQ1_M and IQ2_XS rows that are faster on AVX2 CPUs (#623, thanks @SeriousRising) and an AVX-512 VNNI tier for EXL3 rows on
  CPUs that have it (#596, thanks @sbleschke), both bit for bit.
- **MiniMax:** reports include the engine's load summary and image setting (#667, thanks @tobitege).
- **Docs:** the Intel Arc page is up to date with stock llama.cpp (#674, thanks @Jumbomuffin777).
- The README's download badges are served from this repository, so they no longer time out.

## v1.1.40 - 2026-10-30

Experimental Mac support: GLM-5.3-Flash on Apple silicon, from a community contribution. DeepSeek's restore point
near a prompt's tail is now judged by its real cost on multi-GPU layer splits, and the dashboard got a few fixes.

- **Mac with Apple silicon, experimental (GLM-5.3-Flash only)** (#334, thanks @hotocoo): a new engine, `maya-metal`,
  runs GLM-5.3-Flash on the GPU of an M1 or newer Mac through Metal, with the same dashboard and the same OpenAI- and
  Anthropic-compatible API as on Linux and Windows (streamed answers, tool calls, thinking levels, pictures).
  `./maya.sh --backend metal --setup` builds it and downloads the model; on a Mac, `./maya.sh` picks Metal by itself.
  See [docs/METAL.md](docs/METAL.md).
  **This is experimental.** Every change is built and unit-tested on GitHub's Apple silicon runners, but we have no
  Mac and have not run it with the real model, so expect rough edges. If you try it, please tell us how it went in an
  issue: the chip and its memory, the model, the speed `./maya.sh --bench` prints, and anything that went wrong
  (`./maya.sh --report` collects the details). DeepSeek, MiniMax and Qwen stay NVIDIA / AMD only for now, and a Mac
  GPU without Apple silicon's matrix support (a virtual machine's, for example) gets a clear message instead of a
  crash. The Mac engine is only built on a Mac; for NVIDIA and AMD users the only changes from #334 are the
  server and download points at the end of these notes.
- **DeepSeek: the restore point near a prompt's tail is judged by its measured time per token** (#668, thanks
  @creslinux for the measurements): on a layer split (`--tp 1`) whose experts mostly stay outside VRAM, v1.1.39 kept
  that point although its short run cost several times more per token than the rest of the read - a few seconds
  added to every fresh prompt. The engine now compares the two runs' wall time per token and, after its first two
  measured reads, leaves the point out wherever it costs more than reading its tokens again. On three V100s fresh
  prompts then read about 1.3-1.4x faster with 10% of the experts in VRAM (4-8K tokens), and took 7-17% less time
  with the default placement, about three quarters of the experts in VRAM (4-16K tokens), where the point is now
  left out too. Editing the last message
  still restarts from the last user message, and the system-prompt reuse from v1.1.39 is unchanged. One GPU and the
  CPU lane decide as before. `MAYA_DS4_RESTORE_TAIL_AUTO=0` keeps the point on every read.
- **Dashboard:**
  - Monitor: the peak decode speed now follows the live Decode figure, so the live number no longer sits above the
    peak.
  - Monitor: the Context card says clearly what the VRAM window is (with KV streaming, how much of a chat's cache
    stays in GPU memory) and where it is set.
  - Models page: Engine settings > Context size is now the same slider as Settings > Model, up to each model's own
    trained context (1M for GLM, DeepSeek and MiniMax), with the same memory line under it.
  - Chat: the thinking-level picker highlights only the level that is chosen (Low no longer lit Off as well).
  - Models page: GLM on an AMD GPU says why Images is off (the setup does not build GLM's image encoder for AMD GPUs
    yet).
- Also from #334, for every platform: the server refuses requests that a web page of another site sends to it
  unless the config's `cors_origins` lists that page (curl, SDKs and apps like Open WebUI are not affected), a
  connection that stalls in the middle of a request is closed after 5 minutes (a long answer is not cut), and a
  downloaded model file whose checksum does not match is removed so the next setup fetches it again.

## v1.1.39 - 2026-10-29

DeepSeek reuses a shared system prompt across conversations (a new conversation's first token about 19x sooner),
GLM decodes about 22% faster when it reads experts from the SSD, more Strix Halo builds read GLM prompts with a tuned
table, and model files downloaded some other way can be imported.

- **DeepSeek: a new conversation behind the same system prompt and tools answers about 19x sooner** (#607): the
  engine keeps the state at the end of a system prompt and its tools apart from any one conversation (in RAM, about
  20 MB for an 11K-token prompt with 12 tools), so a new chat, a sub-agent or another `--parallel` conversation reads
  only its own part instead of the whole system prompt again. On one V100 with Maya-XS the first token came about 19x
  sooner after an unrelated chat, and about 25x sooner with `--parallel 2` while another conversation was answering.
- **GLM: decode about 22% faster when experts are read from the SSD** (#570, thanks @pschaub for the idea): an expert
  read from the SSD now goes to the GPU piece by piece as it arrives instead of after the whole read. Measured on a
  V100 PC whose RAM holds only part of the experts; a PC that holds every expert in RAM is unchanged.
  `MAYA_GLM_STREAM_FETCH=0` turns it off.
- **Strix Halo: GLM prompts about 1.9x faster on hipBLASLt 1.4.1 builds that rejected the shipped table** (#631,
  thanks @mrsirg97-rgb): a second tuned table ships beside the first, and the engine now uses, for each prompt shape,
  the first table the installed hipBLASLt accepts, so both AMD's Windows wheels and Linux ROCm 7.15 builds get a
  tuned table. On Windows a table on another drive than the engine now loads too.
- **Import model files downloaded some other way** (#666): `./maya.sh --import-model FOLDER` finds Maya's published
  model files in a folder (any names, subfolders, a Hugging Face cache) by size and sha256, moves them into place
  (`--copy` copies them instead) and says what is still missing. Files that do not match are never touched.
- **Chat: the text-file limit follows the context** (#635): instead of a fixed 512 KB it goes from 512 KB to 16 MB
  with the context; a file over it gets a message that names the context and where to raise it.
- **DeepSeek: the prompt ring is sized by each GPU's link** (#609), measured at load: a slow PCIe link gets a smaller
  ring and waits less for each batch, a fast link keeps the 400 MB it had.
- **MiniMax setup: the recommended context counts where its KV cache lives** (#663): on a Windows APU with about
  32 GB of RAM beside its GPU memory, and on a 24 GB card with 128 GB of RAM, it now suggests 128K instead of the 256K
  that about halved the speed there. Other PCs, GLM, DeepSeek and Qwen are unchanged; `--context` still wins.
- **Monitor on Windows + AMD** (#102): GPU load and memory are now shown, from Windows' own GPU counters.
- Every prompt read logs one line for GLM and DeepSeek with the tokens read, the time and the free VRAM before and
  after, per GPU (#648).
- A stopped request is now drained to its end with one conversation (#603, thanks @merbanan; #536): a client that
  went away during a long prompt read could make every later request get the answer of the one before.
- DeepSeek: the restore point near a prompt's tail now leaves itself out where it costs more than it saves, so a fresh
  long prompt on a PC with most experts outside VRAM is no longer about 13% slower than before v1.1.37; an edited last
  message still restarts from the last user message (#668).
- The thinking budget closes the reasoning at a sentence or line end instead of mid-word, and a malformed tool call
  no longer fails the whole answer (#613).
- DeepSeek on Radeon PRO V620 (gfx1030, 2 or 4 cards): short prompts no longer hang (#621).
- DeepSeek on Windows with NVIDIA: a PC with a lot of RAM for copied experts no longer fails to start with
  "cudaHostAlloc failed" (#589).
- Qwen on 2 GPUs: no more "illegal memory access" on the first long prompt (#645); GLM's prompt path got the same
  safety margin.
- `--dump-logits` writes every row with a native (IQ) pack too (#658).
- After an update, a shared build compiles the other model's engine before it starts (#665, thanks @tobitege; #664).
- Windows ROCm wheels: the GPU library links a long folder path left out are made after setup (#574).
- `CMAKE_CUDA_ARCHITECTURES=native` builds for the PC's compute GPUs, not a display card beside them (#56).
- The GLM engine names the GLM settings set in the environment in one start-up line, with the value used where one
  is clamped (#6).
- Docs: running the Maya-XS DeepSeek GGUFs on Intel Arc with llama.cpp's SYCL backend (#401, thanks @Jumbomuffin777).
- README: speed rows from community reports (#566 #577 #585 #586 #594 #614 #653 #659 #663) and a Sponsors section.

## v1.1.38 - 2026-10-29

Qwen CYBER-FROST runs faster on PCs with little RAM, a new chat shows the model that answers it, and the web app's
home-screen icon opens the page.

- **Qwen CYBER-FROST on RAM-short PCs: decode +26%** (#654): on Linux, the experts that neither the VRAM nor the RAM
  holds are now read straight from the model file, without the system's file cache, which such a PC cannot spare.
  Maya checks a sample of experts against the file when it opens it and keeps the old path on a file system that does
  not allow these reads (`MAYA_EXPERT_DIRECT=0` turns them off). On a PC where VRAM and RAM together cannot hold every
  expert, the setup now suggests a context that leaves more of the VRAM to experts (a 32 GB card with 30 GB of RAM:
  128K instead of 256K). Measured on a V100 32 GB with 30 GB of RAM; PCs that hold every expert are unchanged.
- **New chat** (#662, thanks @needmorevram): a card shows the model that answers it (ready or busy, its context, the
  last answer's speed, images, what it saved) with Switch model, and the ideas to start from stay below the message
  box. On a phone the card is one row that opens the Models page. An answer's numbers (tokens, speed) are now behind
  an info button next to Copy and Regenerate.
- **Web app on a phone's home screen** (#656, thanks @needmorevram): the saved icon opens the page instead of a
  not-found error; icons saved with an older version are sent to the page too.

## v1.1.37 - 2026-10-29

DeepSeek keeps restore points so an edited long chat answers about 50x sooner (#639), `--parallel` reads prompts
a piece at a time (#587), faster prompts on Strix Halo and less memory on Windows, and the loop guard never cuts a
tool call.

- **DeepSeek: editing the last message of a long chat gives the first token about 50x sooner** (#639, thanks @needmorevram):
  the engine now keeps restore points in a conversation (the start of the last user message, a point near the
  prompt's tail, the end of the system prompt and tools), so a chat edited at its end or a request that went away
  mid-read no longer reads the whole conversation again. On a 20K-token chat on our 3 V100s the first token after an
  edit came in 0.3 s instead of 18 s. A long cold prompt now takes a slightly different (equally valid) path
  through the last tokens, so its exact words can differ from v1.1.36 on long chats, as they already do after any
  cache hit; `MAYA_DS4_RESTORE_POINTS=0` turns the points off.
- **DeepSeek `--parallel` is faster and DeepSeek can answer with log-probabilities** (#639): a step's attention for all
  its requests runs in one launch (up to ~8% more total speed at four conversations; with DSpark 2 requests together
  +7% and 4 requests +8-12% on our 3 V100s; one request +4%); `logprobs` / `top_logprobs` work on the OpenAI API;
  faster IQ1/IQ2 GPU kernels (the persistent form is on by default for compute capability 8.0 and up) and AVX-512
  CPU rows; IQ4_NL experts; a RAM expert store (`--ssd-cache`, opt-in); `--pack-info`, `--version`, `--profile-run`
  and `tools/ds4/concurrent_check.py` (conversations at once must not see each other's state). Every exactness
  check (together vs alone, `--multi-check`) came out bit for bit. We fixed its Windows and AMD builds and kept the
  RAM copy of the CPU lane's experts on AMD APUs.
- **Web UI follow-ups** (#632, thanks @needmorevram): on a phone the chats slide in beside the page and push it aside,
  and Saved opens as a bottom sheet; the thinking chip opens a thick bar with a segment per level; the Models page
  can delete a model (and its downloaded files if asked, never a file another model runs from), delete the running
  model after a switch, list the files in the models folders that no model uses, and sets more DeepSeek and
  MiniMax engine options; the Monitor shows the peak decode speed since Maya started.
- **The loop guard never cuts inside a tool call:** repeated identical tool calls stay whole and are all delivered; a
  loop inside a single call only gets the repetition penalty.
- **Open WebUI streaming** (#605): the stream is at most 64 tokens behind the model (a block the model copies from its
  own draft streams as it is written, where it used to go silent for 10-20 s and then arrive in a burst); the guard
  closes the thinking far more rarely (more copies are needed inside the thinking, the first two loops only get the
  penalty, the output is never stopped while still thinking, and a stop before any answer says so in one line instead
  of ending empty); a client's `max_tokens` above 1024 no longer shortens the thinking budget; and MiniMax's second
  thinking block after an answer part or tool call is shown as thinking.
- DeepSeek `--tp 3`: new opt-in `DS4_TP_SHARES` gives each tensor-parallel member its own share of a group's expert
  FFN (for example `2,3,4`), so a member that also holds the DSpark drafter and has less free VRAM can hold a smaller
  part and the other members use their room; unset changes nothing (#591, thanks @NenadSteric).
- DeepSeek: a client that sends the same tools in a different order (an MCP hub's server order, plugins registered
  late) no longer makes the engine read the whole conversation again; the tools keep their first-seen order, as on
  the other models. In our test a reordered request went from 15.7 s to 0.3 s (#625, thanks @SeriousRising).
- DeepSeek `--parallel`: prompts are read a piece at a time on an event-driven scheduler, so answers keep writing
  and short questions no longer wait behind a long read. A mixed four-agent run's first token came 16% sooner on
  average and 1.4x sooner at the 95th percentile on our 3 V100s, total time unchanged (#587, thanks @mrsirg97-rgb).
- DeepSeek: a recent-slot fill that fails no longer publishes its slot, so a failed copy can never be used as an
  expert's weights; the expert is read over the RAM lane instead (#646, thanks @handmade0octopus).
- **DeepSeek prompts faster on Strix Halo** (#643, @mrsirg97-rgb): on gfx1151 the prompt's routed experts now go through the
  fused int8 experts GLM already uses (IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, IQ4_XS gate/up; where a layer's
  formats are not covered the old path stays): prompt speed up ~8% at 2K and 8K on Maya-S, decode unchanged, perplexity
  unchanged (the exact words of a greedy answer can differ, as the prompt's experts now use int8 activations); Maya-XS does not use it. `MAYA_DS4_PREFILL_FUSED=0` turns it off.
- **DeepSeek on Windows: the one-GPU decode path for NVIDIA** (#567, @TheSmokeDev): the single-GPU path that reads streamed
  experts in place now runs on Windows too, with the experts' copies pinned layer by layer as far as Windows allows
  (`MAYA_DS4_PIN_BY_NEED=0` turns that off). Measured by the author on an RTX 3090.
- **DeepSeek on Windows: far less memory held by the experts in the files** (#588, @handmade0octopus): the copied source pages
  leave the process's working set as soon as they are copied, and the in-place experts the CPU lane reads are trimmed back
  to the standby list past a ceiling (`MAYA_DS4_INPLACE_TRIM_GB`, default 6): the peak working set on a 32 GB Halo box fell
  by 42% on Maya-S and the lowest free RAM went from about 1-3 GB to about 9-10 GB, decode unchanged, prompts ~2% slower.

## v1.1.36 - 2026-10-28

DeepSeek drafting now works with `--parallel` (#611), MiniMax keeps its experts once in RAM instead of twice, the same
DeepSeek text whatever ran before, and a round of fixes from reports and pull requests.

- **DeepSeek `--parallel` with the drafter** (#611, @needmorevram): with two or more requests running, each request's
  DSpark drafter now proposes its tokens inside the shared step, and every request's token and drafts are checked in one
  pass, with a rollback of only that request's rejected rows; two or three requests' drafts share one drafter pass. One
  request alone is unchanged, token for token. Our fixes are part of it: on several GPUs a batched verify window of many rows
  took a different rounding path, and one pinned buffer was shared between requests' cache rows (a later request could
  overwrite rows another GPU had not copied yet), so answers could drift from the same request alone; both are fixed, and
  answers at `--parallel` 2 and 4 are the same as alone. With four requests the drafters do not fit one pass, so each
  round drafts only where they do and otherwise decodes a plain token of each as before. On three V100s two requests at
  once finished about 1.1x faster in all and four at the speed of before; on one V100 the gain is smaller.
- **MiniMax keeps its experts once in RAM** (#568, #578; reported by @talisp and @noahark): after an expert is copied
  to the pinned pool or a GPU, its pages in the mapped model files are dropped (Linux: `madvise`, Windows:
  `VirtualUnlock`). Resident memory after load went from 95 GB to 13 GB on a 3-GPU test with identical answers; the
  Windows pool is no longer under-sized by pages that were not "available". `MM3_LOAD_DROP_PAGES=0` keeps the old
  behaviour.
- **MiniMax single-conversation server reads prompts in the calibrated chunk** (#602, @talisp): `prefill-tune.txt`
  was applied to the model but the server kept the command line's chunk.
- **DeepSeek: the same text whatever ran before** (#608, @SeriousRising): windows of 9-16 rows (prompt lookup) took
  another rounding path than windows of 1-8, so a temperature-0 answer could change with earlier requests. All
  window sizes now use one path (`DS4_ASYNC_T` default 16). Logits of 48 prompt positions are bit-identical
  between window sizes 2 and 12; decode and prefill times unchanged or slightly better.
- **DeepSeek: a warning when `--dspark on` finds no drafter** in the pack (#590, @handmade0octopus), instead of
  silently decoding with prompt lookup only.
- **Windows AMD build** (#575, @sakugetu): a failed first configure no longer leaves the HIP Release flags empty
  (unoptimised kernels).
- **Server: a malformed tool call is logged whole** (#613, @NenadSteric), so the cut-off call can be seen; the call
  is carried on the error.
- **DeepSeek CPU lane** (@SeriousRising, #618, #619, #620): the worker pool fences its generation store before the
  sleeper load, so a batch could no longer be left unrun (#618); a sampler CUDA error ends the request instead of
  sending token -1 (#619); a pack whose layer the CPU lane cannot compute is refused at load instead of answering
  zeros (#620; every shipped pack loads as before).
- **CPU lane small calls** (#561, @Cass67): the down jobs read each token's activations from a stack copy
  (`DS4_ACT_COPY=0` is the old way); decode times unchanged here, the reported gain is on 2-socket machines.
- **Linux NUMA** (#562, @Cass67): maya-ds4 stays out of the kernel's NUMA balancing, and a failed `mbind` is said
  once (`DS4_NUMA_BALANCE=1` as before; an existing memory policy is left alone).
- **Windows** (#601, @pschaub): Ctrl+C and Ctrl+Break make the engine wait for QUIT, so stopping keeps the
  conversation.
- **Tests** (#526, @DailyDisco): the HIP report tests work with a home-local `TMPDIR`.

## v1.1.35 - 2026-10-28

GLM keeps every expert in VRAM on NVIDIA where they fit, tuned speed settings are measured again after a driver
change, GLM conversations with pictures keep their slots, faster long prompts for Qwen-3.8-Flash-Next (CYBER-FROST) on
three or more GPUs, and a MiniMax `--parallel` fix.

- **GLM: every expert stays in VRAM on NVIDIA where it fits** (#474, @walker2kk): where the GPUs' memory holds every
  expert, NVIDIA now gets the same complete expert pool as AMD, so a long prompt no longer pushes experts out of VRAM
  and the decode after it stays at full speed (the reporter's 4x A100 split: 2.5x decode). Where the experts do not all
  fit, nothing changes. `MAYA_GLM_COMPLETE_POOL=0` keeps the old pool.
- **A tuned speed setting is measured again after a driver or P2P change** (#419, @catonooka): the CPU-lane tuning
  (`MAYA_GLM_PCIE_SHARE`, `MAYA_GLM_CPU_LANE`, `MAYA_GLM_READ_CHUNKS`) now remembers the NVIDIA driver, the GPUs' slots
  and which GPUs reach each other directly (P2P). When one of them changes, Maya says what changed, drops the old values
  and the engine measures again at its start (the reporter's 2x RTX 3090 after a P2P driver: 1.6x decode back);
  `./maya.sh --calibrate` tunes it again. A setup no longer re-applies a stale stored tuning.
- **GLM: a conversation with pictures keeps its slot** (#331, @Cass67; also reported in #418 by @CoolJohn-lab): a chat
  with pictures is set aside in a conversation slot like a text chat, so a chat with one screenshot no longer reads its
  whole history again each time another chat pushed it out. Kept slots from an older version are removed once at start
  (the slot format changed).
- **GLM: an edited last message after switching chats** (#400, @Zcg2021): A -> B -> back to A with the last message
  edited no longer reads the whole conversation again (that step's prompt read about 10x faster on a 2K-token chat).
- **GLM calibration with more CPU-lane threads** (#178, @Jw88888): on one GPU whose CPUs show no SMT siblings (a VM's
  vCPUs) `--calibrate` also measures more CPU-lane threads than the engine's own count, and lower counts are no longer
  measured slower than they run; new setting `MAYA_GLM_CPU_LANE_MAX`.
- **MiniMax `--parallel`: a short question no longer evicts a long picture conversation:** the picture was read again
  at that conversation's next turn. A new request now takes an empty conversation (or the least recently used one);
  continued conversations are reused as before (that next turn 2.9x faster on 3x V100, its answer the same).
- **Qwen-3.8-Flash-Next (CYBER-FROST) on three or more GPUs reads long prompts with every GPU at once:** with the model
  split across GPUs, the second and third GPU used to take turns on each piece of the prompt; each now works on a
  different piece at the same time. On three V100s with most experts in VRAM, prompts of 4K, 8K and 16K tokens read
  about 1.14x, 1.35x and 1.5x faster (a 2K prompt is one piece: no change); with experts streamed from RAM the gain is
  small. Answers are the same; one GPU and two GPUs are unchanged.
- **Chat:** the attach menu's item reads "Select file".

## v1.1.34 - 2026-10-27

DeepSeek and MiniMax answer several requests at once, fixes for one-GPU and tensor-parallel speed, Qwen-3.8-Flash-Next
(CYBER-FROST) in the setup, each model's own defaults in the chat, your own EXL3 folders in the setup, and a new README.

- **DeepSeek-V4.1-Flash and MiniMax-M3.1-Flash answer several requests at once** (#332): `./maya.sh --parallel 4`
  (Windows: `START-MAYA.bat --parallel 4`; up to 8), or the Models page's Answers at once, as for GLM-5.3-Flash - for
  agents that run several sessions, or several people on one PC. Each conversation keeps its own state; two at once
  write about 1.6x the tokens per second of one after another on a V100 with most of DeepSeek Maya-XS's experts in
  RAM. MiniMax keeps every conversation's cache in VRAM, so KV streaming and its drafter
  are off while more than one runs, and the setup recommends a context that leaves them room.
- **DeepSeek on one GPU with every expert in RAM** (#551, @talisp): the one-GPU decode path now turns itself off where
  nothing would ever stream from the disk (`--mmap-experts off`, or auto with all of them copied); it cost 4-9% of
  decode on the reporter's L40S, and decode was 2.2x faster without it on a V100 test PC. Nothing to set.
- **MiniMax CPU lane** (#552, @talisp): `--cpu-threads N` now means N threads in all, as for DeepSeek (it started
  N + 1: about 1.3x faster decode with N = the cores); by default 2 cores stay free for the engine's other work on a
  CPU without SMT (a VM's vCPUs), and decode times the CPU lane again when the start-up timing was off.
- **MiniMax prompts on one GPU** (#552): the prompt calibration also tries larger chunks where most experts stay in
  RAM and keeps one only when it reads prompts clearly faster without slowing decode: 1.9x faster prompts on a V100
  with a quarter of the experts in VRAM.
- **MiniMax tensor parallel with DSpark** (#556, @bandito1): the VRAM the drafter's GPU uses alone was left empty on
  the other GPU of its group; that GPU now fills it with more experts (about 1.2x decode on a two-GPU test with half
  of the experts outside VRAM).
- **Qwen-3.8-Flash-Next (CYBER-FROST) in the setup:** its Q5_K_M GGUF, pinned and checked, with its MTP drafter built
  from the GGUF's own block (no second download), its experts read straight from the GGUF where RAM cannot hold them
  all, INT8 KV cache, and the model card's settings (temperature 1.0, top-p 0.95, top-k 20; reasoning efforts low,
  medium and xhigh). Text only.
- **Each model's own defaults:** the setup writes each model's card sampling and thinking level into its config every
  time, the chat uses them until you choose something, and switching models starts at the new model's own. GLM gets
  its Max level and penalty fields; Top-k and Min-p "off" take effect; a chat continued on another model takes its
  nearest thinking level; the same seed gives the same answer on DeepSeek and MiniMax. The chat follows a streaming
  answer until you scroll, and notifications get a close button.
- **Your own quants in the setup** (#545 @sebastian-burlacu, #547 @misty-soverin, #528 @woodsonl): `--model-dir`
  takes an EXL3 folder of GLM-5.3-Flash or MiniMax-M3.1-Flash at any bit rate, checked file by file, with the
  folder's own sampling and chat template; a GLM GGUF with another expert count (a pruned model) no longer gets the
  shipped 288-expert profile. The model list is shorter: unsloth's DeepSeek UD-Q4_K_XL and turboderp's GLM
  EXL3 sc_4.10bpw are no longer offered (named by hand, or set up before, they work as before).
- **DeepSeek expert tables** (#550, @handmade0octopus): two races closed - a streamed expert no longer lands in a
  recent slot while its swap is pending, and a table update waits for the device's earlier one.
- **DeepSeek tensor parallel: a stopped prompt no longer wedges the pair** (#539, @zhangxiaoxing): one GPU spinning
  and the other idle until a restart. A request that stops moving for 10 minutes now ends the engine with a report and
  the server starts it again; an engine stopped by a signal is no longer reported as out of RAM (#494).
- **More fixes:** DeepSeek on a 2-socket PC no longer gets killed for memory at load with RAM free (#507,
  @zhangxiaoxing); `--calibrate` measures the layer split too on two to four equal GPUs (#502, @noonghunna); one busy
  moment at MiniMax's start no longer moves every expert to the GPU for the session (#548, @noonghunna); `--bench`
  uses MiniMax's own chat format and engine on AMD and works with a `--parallel` config (#549 @noonghunna, #420
  @oscarlius, #509 @knoodrake); Linux setups warn when systemd-oomd could end the session during a large load (#497,
  @HansStrupat); the CPU expert kernel builds with an assembler without AVX512-VNNI (#535, @biochem-fan); a Windows
  AMD build with only Visual Studio 2026's Build Tools says what to install (#544, @wonder-sound).
- **README:** a shorter front page - quick start, one "pick your model" table with each quant under its own model
  (#554, @AutisticPancake), quality, and every user speed report back in a closed section per model; each model's
  files and options in [docs/MODELS.md](docs/MODELS.md).

## v1.1.33 - 2026-10-27

A quick fix for setting up MiniMax-M3.1-Flash from our GGUFs.

- **Fix: a new MiniMax GGUF setup finishes again** (#555, @GoKu-Gear6): the setup's pack step now finds the
  `-maya-indexers.bin` file beside the shards. Since v1.1.31 it stopped with "the sparse-attention indexer of layers
  0, 1, 2 is not in the set". A MiniMax already set up was not affected.

## v1.1.32 - 2026-10-27

A quick fix for the new web UI.

- **Fix: the "Saved" popover is whole again** (#553, @needmorevram): the glass sidebar sits above the main column,
  so the popover at the bottom of the sidebar is no longer cut off at its edge.

## v1.1.31 - 2026-10-27

MiniMax answers faster, DeepSeek reads copied text faster and keeps its pictures off the GPU, orcarouter's
abliterated DeepSeek in the setup, and a new web UI.

- **MiniMax-M3.1-Flash much faster:** decoding with the DSpark drafter is about 1.5x as fast as in v1.1.30 on
  three V100s and on a Strix Halo, plain decoding about 1.2x. A bug since v1.1.29 left the drafter guessing
  without the prompt, so it rarely helped; it now drafts from the whole conversation. Faster dense kernels too,
  with the same answers as before.
- **DeepSeek: copied text up to ~1.6x faster** (#540, #543, @needmorevram): when the answer repeats earlier text -
  a file sent back with small edits, quoted text - the engine checks the tokens that followed that text instead of
  only the drafter's guesses, with and without DSpark. On by default on NVIDIA and AMD (about +27% on an edit on
  a Strix Halo); `--suffix-draft 0` turns it off.
- **DeepSeek pictures take no VRAM between pictures:** the vision tower borrows its working memory only while a
  picture is read and gives it back, so about 320 MB more of the model stays on the GPU - as GLM and MiniMax.
- **orcarouter's abliterated DeepSeek-V4.1-Flash** (#520, @needmorevram): `--model deepseek` offers
  "orcarouter Abliterated Mix-128G" and "Mix-256G" below Maya's quants, the one that fits the PC recommended. They
  use Maya's own n-gram tables (their table is DeepSeek's own, so the higher-precision copy is shared with Maya's
  DeepSeek quants and their 110 GB table file is not downloaded). A gated repo: the setup uses your Hugging Face
  login. Also from #520: a tensor-parallel group reads long prompts much faster (~7x at `--tp 8` in his runs).
- **A new web UI** (#538, @needmorevram): a new chat, Settings, Monitor, a Models page to switch models and edit
  each one's engine settings, and "Saved" - what the model's own API would have charged. The new look is the
  default; the violet colours are a choice in Settings › General. Works offline: every library and font is in the
  repo.

From the community, thank you:

- **DeepSeek decode latency** (#495, @needmorevram): the CPU lane works in one batch per node, kernels launch
  dependent on the one before (newest GPUs), and the CPU answers its own requests - the same answers.
- **DeepSeek on more PCs and setups, all opt-in:** a series for tensor-parallel groups and a helper GPU - fused
  decode stages, exact trims checked at load, a pipelined all-reduce for two GPUs, uneven groups and the helper GPU in
  the prompt path, long prompts' exact skip and borrowed expert slots, a VRAM cap for the drafter's experts (#511-#519,
  @pepuscz); tensor parallel with uneven heads (#411, @NenadSteric); full zero-copy routes (#530) and a
  cache-budgeted Engram read-ahead (#524, @handmade0octopus); re-quantized attention matrices at load (#473) and RAM
  copies that follow the routes (#449, @cryptedy); direct I/O for the RAM copies (#421, @tctiger); packed cache
  transfers for EXL3 tensor parallel 2 (#493) and a refresh of the startup expert profiles (#399, @LostVector); a
  direction projection before each residual write, `--ablate` (#469, @JoshBolding).
- **Loading DeepSeek:** the experts' pages are dropped as they are copied, with the load's threads and file order set
  (#434, @Ross-Yang-94).
- **AMD:** RDNA 4 attention acceleration (#529, @nicholas631), MI50 / gfx906 as a build target (#505, @0c33), and
  opt-in prompt-path options: transfer pruning (#297, @LostVector) and int8 MMQ for routed experts (#354,
  @boxwrench).
- **NVIDIA DGX Spark (GB10) and Grace** on CUDA, and the prompt path recovers from a failed allocation (#471,
  @xcpatr9231975).

Fixes:

- **DeepSeek GGUF experts uploaded from NUMA-split RAM copies** are right again (#533, @tctiger).
- **Calibration** respects the effective tensor parallel (#435, @Whamp).
- **`--parallel` is refused for DeepSeek and MiniMax** before it is written into their configs (#525, @DailyDisco).
- **MiniMax quants on the Hub:** the dense layers' indexers file is now `<name>-maya-indexers.bin` (the same bytes;
  the Hub read the lone `.gguf` as the model). The setup renames a v1.1.30 download in place.

## v1.1.30 - 2026-10-27

MiniMax-M3.1-Flash in Project Maya's own quants: the whole model in a 128 GB PC.

- **MiniMax-M3.1-Flash-Maya-XS and -Maya-S on Maya:** `./setup.sh --model minimax-maya-xs` (or `minimax-maya-s`)
  runs our GGUF quants on `maya-mm3`: Maya-XS (~93 GB, the experts in about 1.6 bits) fits a 128 GB
  PC's memory, Maya-S (~122 GB, about 2.1 bits) is closer to the original. Run on Maya, both score as
  measured for their model card (the quality table there).
- **NVIDIA from V100 on and AMD, Linux and Windows**, with the experts VRAM cannot hold computed on the CPU from RAM,
  the INT8 K/V cache and the context fitted to the PC's memory, as for the EXL3 quant.
- **Pictures and DSpark drafting with the GGUFs**, on NVIDIA and AMD: the vision tower and the drafter come from the
  EXL3 repo's own files (about 4 GB), downloaded once beside the model.
- **MiniMax's EXL3 quant on AMD, experimental**, as GLM's EXL3 quants there: `maya-mm3` builds for AMD GPUs with its
  EXL3 kernels, and the setup offers the quant there too.
- **The same files work in llama.cpp:** the three dense layers' indexers are kept in a small file of their own, which
  Maya loads to run MiniMax's sparse attention in every layer.
- **One naming scheme for every model's quants:** the model first, then the tier by the experts' bits - XS (~1.6),
  S (~2-2.5), M (~3-3.5), L (4+). The setup and the dashboard show "GLM-5.3 · Maya-S24", "DeepSeek V4.1 · Maya-XS",
  "MiniMax M3.1 · Maya-XS (93 GB)", and `--model` takes setup ids: `glm-maya-s`, `deepseek-maya-xs`,
  `minimax-maya-xs` and so on. Every name used before (`--model Maya-S24`, `--ds4-quant maya-xs`, ...) still works,
  and installed models and their configs carry on as they are.
- **DeepSeek: a 256K context by default** wherever the PC's VRAM and RAM hold the model; PCs with less keep 64K. A
  setup again keeps the context you chose before.
- **A loop guard for every model:** an answer that falls into repeating one block over and over, in the thinking or
  the answer, is caught as it starts and steered out of it; on by default (`MAYA_LOOP_GUARD=0` turns it off).
- **MiniMax answers faster, on NVIDIA and AMD:** a token's attention keeps the whole GPU busy, the layers are shared
  out over the GPUs by what each one needs (more experts in VRAM), and where the experts fit in the GPU a token no
  longer waits for the host at every layer. On three V100s Maya-XS answers about twice as fast as the EXL3 quant with
  its drafter did in v1.1.29, and the EXL3 quant itself decodes about 13% faster.
- **MiniMax reads prompts faster, tuned to each PC:** a prefill calibration measures the best way to read prompts on
  this PC once (the setup, `./maya.sh --calibrate`, or the first start of an earlier setup) and the engine keeps it;
  on the Gorgon Halo Maya-XS reads a 1K-token prompt about 1.4x and 4K-16K-token prompts about 1.75x as fast, with the
  same quality on the reference prompts.
- **DeepSeek's EXL3 quants on AMD, experimental:** the setup offers them there with a warning, as GLM's and
  MiniMax's EXL3 quants.

From the community, thank you:

- **EXL3 faster on V100, T4 and RTX 20** (#461, #484, #485, @merbanan): decode cuts its trellis windows together and
  splits a routed expert's gate and up, prompts use the tensor cores for the light experts - the same results.
- **turboderp's self-calibrated GLM EXL3 quant at 4.10 bits in the setup** (#256, @needmorevram).
- **DeepSeek kept conversations hold pictures too** (#462, @markrussi), and **can be capped in RAM**
  (`MAYA_DS4_SLOT_RAM_GB`, the least recently used spilling to disk; #523, @handmade0octopus).
- **DeepSeek's Engram cache in the engine stats and `/metrics`** (#522, @handmade0octopus).
- **DeepSeek on more PCs:** the RAM budget counts ZFS's reclaimable cache (#492, @saltacc), the CPU lane uses up to
  four NUMA parts (#463, @geoffreybyers), the IQ1_S CPU lane runs fast on AMD Zen to Zen 2 (#489, from a post by
  @superalesha), the drafter can go on another GPU group (experimental, opt-in; #464, @LostVector), and
  `ds4_gguf_test` builds on Windows again (#465, @OscarSheng).
- **DeepSeek EXL3 packs** read the mixed-precision quant map as the quant made it (#481, @cgi-yf; closes #430).
- **DSML tool calls:** a value ending in part of a closing tag no longer swallows the rest of the call (#482,
  @catonooka).
- **GLM's thinking budget with a small `max_tokens`** keeps room for the reasoning (#488, @Fringe210).
- **The live decode rate is per request** in the Monitor with `--parallel` (#472, @asdacap).

Fixes:

- **EXL3 builds for Pascal again** (#479).
- **DeepSeek on Windows starts even when the OS will not pin all its RAM** - the same answers, with a warning (#521).
- **`--tp` follows the GPUs you choose now** when the setup runs again (#436).
- **DeepSeek's one-GPU decode path only where experts actually stream from RAM** (#496).
- **A DeepSeek GGUF drafter without its DSpark settings** gets them filled in, or is left out with a warning, instead
  of loading and never drafting (#500).
- **A streamed request waiting in the queue sends keep-alives**, so clients with an idle timeout wait for their turn
  (#483).
- **A request too long for the context gets OpenAI's context-overflow error**, so agents compact and retry (#432).

## v1.1.29 - 2026-10-26

MiniMax-M3.1-Flash joins GLM and DeepSeek as Maya's third model, on an engine of its own, with MiniMax's own
settings and fitted to each PC from the first release.

- **MiniMax-M3.1-Flash on its own engine** (#477, @needmorevram): `./setup.sh --model MiniMax-M3.1-Flash` downloads
  Mia-AiLab's EXL3 3.5 bpw quant (197 GB, every file checked) and runs it on `maya-mm3`, the engine @needmorevram
  built for this model: tensor parallel over groups of GPUs, the DSpark drafter (about 1.5x decode in his runs),
  pictures, a streamed K/V cache, and the experts VRAM cannot hold computed on the CPU from RAM. On an RTX 3090 with
  eight RTX 5060 Ti he measured 38.2 tokens/s writing code.
- **Prompts about 4.5x faster, decode up to 13% faster** (#490, @needmorevram, his figures): with tensor parallel the
  GPU groups now read a long prompt together, with the same perplexity. Also from him: a rotation that cuts the INT8
  K/V cache's quality loss by about three quarters (now on by default), and MiniMax tool calls keep a value that may
  be a string as a string.
- **Windows and V100-class GPUs:** MiniMax runs wherever DeepSeek's EXL3 does: NVIDIA from V100 up, on Linux and
  Windows (Windows is new: tell us how it goes), with the V100, T4 and RTX 20 tensor cores in use.
- **A compact K/V cache by default:** INT8 takes half of BF16's memory, so a longer context or more experts fit;
  `fp8`, `fp4` and mixes are options (`--kv-cache`).
- **Pictures on, never parked in VRAM:** the vision tower stays in RAM or on the SSD and comes to the GPU only while
  a picture is read, so the VRAM goes to the model.
- **Fitted to each PC:** the setup recommends the context (up to the model's 1M) by the PC's memory, the engine plans
  VRAM and RAM at start and steps down with a note instead of failing, the DSpark drafter stays only where it measures
  faster, and the CPU lane uses the PC's own cores.
- **MiniMax's own settings:** the model card's sampling (temperature 1.0, top-p 0.95), MiniMax's official prompt
  format, and its thinking levels in the chat: Auto (the default, the model decides whether to think), Low to Max,
  and Off.
- **Fix: an empty answer after an engine error** (DeepSeek and MiniMax): the request after a failed one (an
  unreadable picture, for example) now gets its own answer.
- **New setting:** `"reasoning_effort": "auto"` (any model) asks for the model's own default level.

**MiniMax runs on** NVIDIA V100 / RTX 20 and newer, Linux and Windows, with about 250 GB of VRAM and RAM together. Its license,
the MiniMax Community License (`LICENSE.minimax-m31`), allows non-commercial use freely; commercial use has conditions.

## v1.1.28 - 2026-10-25

DeepSeek decodes faster on a single NVIDIA GPU and reaches its full 1M context, the chat stays steady while it writes
long code, and AMD and multi-GPU setups get new opt-in options.

- **Faster DeepSeek decode on one NVIDIA GPU** (#476, @RubenTorney): when the PC's RAM holds the experts, the GPU now
  gets the experts it needs much more efficiently and keeps more of them close at hand. Up to about 2x decode reported
  on an RTX 3090, +14% on a V100 here and +35% with less VRAM free; prompts read at the same speed. It is on for Linux
  with one NVIDIA GPU; Windows, AMD and multi-GPU setups keep the previous path, and the start's log says which runs.
- **DeepSeek 512K and 1M contexts** (#442, @needmorevram): the setup and the dashboard now offer the model's full
  context. A longer context takes VRAM from the experts, and a prompt that long takes a while to read.
- **A steadier chat while it streams** (#478, @needmorevram): long code answers no longer flash or jump, scrolling up
  stops the chat following the answer, and the sidebar shows the model's logo and name.
- **GLM on AMD, opt-in** (#454, #455, #456, #457, @pepuscz): the expert helper now works beside a layer split, Maya-M's
  experts can run on a tuned RDNA kernel, the profiler shows attention in more detail, and alternative decode
  attention kernels give the same results to the bit. With these settings off nothing changes (`docs/SETTINGS.md`).
- **DeepSeek on several GPUs, opt-in** (#412, #415, @pepuscz): the DSpark drafter can be spread over a tensor-parallel
  group to free room on the first GPU, and a helper GPU can take the experts VRAM cannot hold (Linux). Off by default.

## v1.1.27 - 2026-10-24

DeepSeek decodes up to twice as fast on PCs whose RAM holds its experts, GLM reads whole videos, and fixes for a
Windows server that froze on the first picture and for DeepSeek's calibration.

- **DeepSeek decode where the RAM holds the experts:** `--mmap-experts auto` (the default) now copies a GGUF's CPU-lane
  experts into RAM when all of them fit with room to spare, instead of reading them from the model's files at every
  token. Reported on #339 and #441: +22% on four V100s with 504 GB, about 2x on a 20 GB card with 255 GB, +23% on four
  RTX 3090s with 126 GB; +23% on three V100s here. With less RAM nothing changes (a part copied was measured slower);
  `--mmap-experts on` keeps them in the files.
- **GLM reads whole videos** (#248, @kabachuha): send a video like a picture (OpenAI `video_url`, Anthropic `video`);
  its frames, 4 a second, go through the picture encoder. Needs ffmpeg and ffprobe on the PC (Linux: `sudo apt install
  ffmpeg`; Windows: `winget install Gyan.FFmpeg`). Only real video files reach ffmpeg (MP4, MOV, WebM, MKV, AVI), and a
  video longer than `MAYA_VIDEO_MAX_FRAMES` frames (240, a minute) is refused before it is decoded.
- **Windows: the server froze on the first picture** (#444, @DikiyLifter): the thread that watches the starting program
  held stdin's lock in a blocking read, and the first picture's `import numpy` waited on it forever. The pipe is now
  polled. Every version since v1.1.18 was affected.
- **GLM decode on Volta** (#459, @merbanan): the decode's sparse attention on the V100's tensor cores - its attention
  ~4x faster, decode a few percent. `MAYA_GLM_MLA_V70=0` turns it off.
- **DeepSeek calibration** tests the CPU-lane thread counts whenever the lane has threads, fewer than every CPU first
  (#433, @talisp: one GPU went from 15 to 41 tok/s with fewer threads, and `--calibrate` had skipped the test).
- **Fixes:** with `MAYA_DS4_TP_GGUF_UNEVEN` the busy experts' buffers are sized for the widest layer, so a 2-GPU tensor
  parallel no longer fails on the first request (#446, @pepuscz; #439); the busy experts' FP16 path takes only the
  experts already in VRAM; IQ1_M experts on the CPU lane up to 14x faster, the same results to the bit (#452, @Whamp);
  DeepSeek GGUFs with IQ2_XS experts (from #357, @OscarSheng), Q5_K dense weights or IQ4_NL Engram tables load, and
  an Engram table in a format the engine cannot read stops the load with a message instead of crashing.

## v1.1.26 - 2026-10-23

DeepSeek reads prompts up to 1.5x faster on NVIDIA V100s and faster on every NVIDIA GPU, its long tool calls stream,
and fixes for two-GPU Windows setups and the benchmark.

- **DeepSeek prompts on Volta and Turing** (V100, RTX 20): the sparse attention now runs on their FP16 tensor cores,
  like RTX 30 and newer since v1.1.24 - twice as fast as before on a V100, as exact as the old path (the held-out
  KL against the original model the same). `DS4_ATTN_V7=0` turns it off.
- **DeepSeek prompts on every NVIDIA GPU:** a prompt's busiest experts are decoded to FP16 once and multiplied by the
  library's GEMM instead of being decoded again for every 64 rows (`DS4_FAT_ROWS`, 64 rows by default; 0: off), and
  cards of 16-32 GB get wider attention slabs (fewer FP16 decodes of the same weights). Together on three V100s:
  2K-token prompts ~1.3x, 8K and 32K ~1.5x faster; decode the same.
- **DeepSeek prompts, more:** the expert copies keep going while the host waits for the routing (#360, @OscarSheng);
  prompt skip across a 2-GPU layer split (#343, @Gazelle221B); `--tp 1` on one GPU takes the chunked prompt path
  (#320, @pietroMastro92); tensor parallel 2 on Maya's quants, opt-in `MAYA_DS4_TP_GGUF_UNEVEN` (#408, @pepuscz).
- **DeepSeek decode:** two cores left free beside the CPU lane where it would take every logical CPU (#359,
  @OscarSheng; with SMT only where it helps, settable with `MAYA_DS4_LANE_FREE`); the CPU lane prefetches only the
  expert pages not already in RAM - Windows decode +6-7% (#425, @cryptedy).
- **GLM prompts on Volta:** the prompt attention in one pass (#335, @merbanan) - long prompts ~8% faster on a V100.
- **DeepSeek tool calls:** a long string argument (a file's content) streams as it is written, so clients with a
  stream-idle timeout no longer give up while the model is still writing (#403).
- **Windows, GLM on two GPUs:** the RAM tier is pinned in pieces of at most 1 GB, so the second GPU gets its share of
  Windows' pinning budget instead of 0 GB and the start no longer fails (#417). `MAYA_GLM_PIN_PIECE_MB=0`: one block.
- **Fixes:** the server's queue no longer stays blocked after an engine restart handed out a request number still in
  use (#330, @jettoblack); `--bench` sends DeepSeek its own chat format and `--report` shows DeepSeek's placement and
  speed lines and its pictures setting (#416); `--mmap-experts` is a setup option kept by later setups (#422).

## v1.1.25 - 2026-10-23

Several conversations at once on GLM get faster, DeepSeek keeps conversations across switches and restarts, and a
batch of fixes - among them DeepSeek's EXL3 setup and its thinking budget.

- **DeepSeek EXL3 setup** no longer stops at the pack step ("keeps dense weights in FP8", #376): since v1.1.21 every
  fresh EXL3 setup did. A download already made carries on from the pack step.
- **DeepSeek keeps conversations** across switches and restarts, so agents with sub-agents stop re-reading their
  prompts: opt-in, `MAYA_DS4_SLOT_KEEP=1` (#356, @mrsirg97-rgb). A retried prompt or a re-rendered turn is taken back at
  the conversation's checkpoints (#394, @mgabor3141).
- **DeepSeek's thinking budget** (`thinking_budget` in the config) is kept: the reasoning ends there and the answer
  follows (#363). A 1..100 `reasoning_effort` shows as itself at the start.
- **DeepSeek decode:** the PCIe lane is now judged over whole decode steps too, and streams only where the token gets
  faster (#329). `DS4_STREAM_TOKEN_AB=0` keeps the earlier check alone.
- **GLM, several conversations at once** (`--parallel`, @somnergy): MTP drafting in the batched steps
  (`MAYA_GLM_MULTI_MTP`, #385) and the rows' attention and small kernels in one launch (`MAYA_GLM_ROWS_ATTN` /
  `_ROWS_SMALL`, #391); running requests keep decoding while a new prompt is read (`MAYA_GLM_PREFILL_INTERLEAVE` /
  `_GAPS`, #386; `MAYA_GLM_PREFILL_MIN_CHUNKS`, #392); a conversation's system prompt is kept in a slot, so new
  conversations restore it instead of reading it again (#387). All opt-in except #387's, which is on with the slots.
- **GLM EXL3 prompts:** short chunks on the tensor cores, `MAYA_GLM_EXL3_MMA_ROWS_SHORT` - 2K-token prompts 28% faster
  on the contributor's PC (#388, @somnergy).
- **Fixes:** the server no longer answers "exceeds the context (0)" to every request after an engine that died could
  not start again - each request tries again, or says why it cannot (#351); `--speed-calibration` measures DeepSeek's
  DSpark drafter again (#393); the Monitor finds an engine started through a launcher (#398, @somnergy); the pack's MTP
  message no longer reads as drafting off (#327); docs on the CPU lane's threads (#390, @somnergy).

## v1.1.24 - 2026-10-22

A fix for DeepSeek on two NVIDIA GPUs under Windows, faster DeepSeek prompts on NVIDIA RTX 30 and newer and on RDNA 4,
and new options for large-RAM servers.

- **DeepSeek on two or more NVIDIA GPUs on Windows** no longer hangs on the first prompt (#384, #303): one GPU waited
  for a hand-off the other GPU's queued work had not yet sent. `DS4_WIN_SUBMIT=0` restores the old behaviour for
  comparison. Linux and one-GPU setups are unchanged.
- **Monitor: DeepSeek's "Where the experts live" card** shows VRAM, RAM and SSD apart again, each with its experts and
  GB, and the experts a token from RAM, from the SSD and promoted to VRAM - as GLM's - with the n-gram memory beside
  them (from the SSD: measured on Linux).
- **Prompt attention on NVIDIA tensor cores** (RTX 30 and newer, sm_80+; #369, @jinsoojinsoo) and **on RDNA 4's
  matrix cores** (#370, @Gazelle221B), like RDNA 3's since v1.1.21. `DS4_ATTN_MMA=0` / `DS4_ATTN_WMMA=0` turn them off.
- **Tensor parallel:** a GPU group member reads only its own expert rows from the RAM copy (#377, @NenadSteric).
- **Large-RAM Linux servers:** opt-in locked Engram tables (`DS4_ENGRAM_RAM=1`) and every CPU-lane expert held in RAM
  with no fallback to the SSD (`DS4_REQUIRE_ALL_RAM=1`) (#368, @tctiger). [docs/SETTINGS.md](docs/SETTINGS.md).
- **Options** (@pepuscz): `MAYA_DS4_Q8_BATCH=1` rounds a GEMV call's inputs in one launch, with the same bits (#382);
  `MAYA_DS4_LT_STORE` keeps AMD's measured hipBLASLt picks outside the pack (#381); `DS4_PROF` prints its line for each
  served request (#380); the kernel test fails on a NaN or Inf (#379).

## v1.1.23 - 2026-10-22

A fix for DeepSeek conversations past 32,768 tokens on AMD GPUs and NVIDIA RTX 20 cards.

- **DeepSeek past 32K tokens on AMD (Strix Halo, Radeon) and RTX 20:** since v1.1.22, once a conversation passed
  32,768 tokens the engine stopped (`ds4 gguf mv: invalid argument`). The long-context top-k now sizes its buffer from
  each GPU's shared memory - the same picks, the same speed. NVIDIA V100, RTX 30 and newer were not affected.
- **Maya-XS** is no longer labelled experimental in the setup.

## v1.1.22 - 2026-10-22

DeepSeek-V4.1-Flash gets Maya-XS - the whole model in a 128 GB PC's memory - tensor parallel on several GPUs, and much
faster drafting on one GPU.

- **Maya-XS (experimental): DeepSeek-V4.1-Flash in 120.6 GB of memory**, its own 2-bit DSpark drafter included - every
  one of its 384 experts a layer kept, in ~1.6 bits (IQ1_S), rounded with error feedback layer by layer on the
  quantized model itself, from three times Maya-S's calibration text (code, three.js and front-end, tool calls and
  agent traces, long reasoning). The Engram tables (161 GB, Q6_K) stay on the SSD as with Maya-S. On a 128 GB Ryzen AI
  Max every expert sits in GPU memory. `./setup.sh --setup --model DeepSeek-V4.1-Flash-1017 --ds4-quant maya-xs`.
  The engine now runs IQ1_S and IQ1_M experts on NVIDIA, on AMD and in the CPU lane.
- **DeepSeek tensor parallel on several GPUs** (#344, @needmorevram): prompts ~1.9x and answers ~1.55x faster at
  `--tp 8` than v1.1.21 on 9 GPUs, the same perplexity; `./maya.sh --calibrate` tunes `--tp`. On one GPU, drafting
  decodes **~1.85x faster** (Maya-S with DSpark on a Ryzen AI Max, the same drafts kept: each verify round about half
  the time).
- **AMD:** a 2-GPU DeepSeek hang on the first long prompt with hipBLASLt fixed (#367, @Gazelle221B; #366); the packed
  sparse attention's LDS swizzle, opt-in `DS4_HIP_ATTN_SWIZZLE=1` (#348, @LostVector); `--bench` takes the DeepSeek
  engine on HIP (#350, @GravLabs).
- **DeepSeek on Windows** (@OscarSheng): a 1 ms timer while the engine runs (#358; `MAYA_DS4_TIMER_MS=0` leaves it),
  64 Engram readers (#361) and unbuffered Engram reads (#362; `DS4_EG_DIRECT=0` = before) - faster prompts.
- **DeepSeek prompt report:** which experts a prompt found resident and the bytes it streamed, `DS4_PREFILL_STATS=1`
  (#364, @LostVector).
- **GLM:** a prompt's lending keeps the more recently used experts in RAM - 40% fewer disk reads after a fresh prompt
  (#352, @merbanan).
- **Fixes:** maya-ds4 builds with CUDA 12.0 / 12.1 (#336); the HIP build's test programs link (#355); `VERSION` holds
  the version again (v1.1.21's was empty).

## v1.1.21 - 2026-10-21

DeepSeek-V4.1-Flash reads prompts about 2.4x faster on AMD Strix Halo and Radeon RDNA 3 / 4, decodes faster where its
experts sit in GPU memory, and EXL3 runs faster on AMD.

- **DeepSeek prompts on AMD: about 2.4x faster** (Maya-S on a Ryzen AI Max through the server, 4K-18K-token prompts,
  2.35-2.53x), the answers the same within rounding:
  - the sparse attention and the low-bit experts run on the matrix cores (RDNA 3 / 4);
  - the dense projections run on hipBLASLt with the fastest kernel for each shape measured on your GPU, kept in the
    pack (`lt-picks.txt`) so only the first start pays the tuning;
  - larger attention slabs where the GPU has the memory (sized from it: 32 GB and smaller cards as before), leaner
    expert tiles, the router 8 tokens at a time, the experts reading the prompt's FP16 rows - the same results bit
    for bit.
  `DS4_LT=0`, `DS4_ATTN_WMMA=0`, `MAYA_DS4_NO_TC=1` and `DS4_ROUTER_ROWS=0` turn parts off.
- **DeepSeek on NVIDIA:** the same leaner expert tiles - a prompt's expert sets 5-11% faster on a V100.
- **DeepSeek decode: layers whose experts are all in GPU memory skip the host** (no mailbox, no wait) - about 9% more
  tokens a second with Maya-S on a Ryzen AI Max, the same tokens. `DS4_HOST_FREE=0` = before.
- **EXL3 on AMD: the experts about 30% faster**, and every EXL3 kernel rounds alike on AMD (drafting no longer changes
  answers there).
- **Tests:** the DeepSeek-V4.1 encoding fixtures stay byte for byte on Windows clones; DeepSeek's chat encoding checked
  against DeepSeek's official repository: identical.

## v1.1.20 - 2026-10-21

Stop now stops at once, DeepSeek-V4.1-Flash decodes ~6.5% faster on a Strix Halo and fits its GPU memory by itself,
Windows starts Maya again from `START-MAYA.bat`, the port can be changed from the dashboard, and a round of community
work.

- **Stop stops, even while a long prompt is read.** Pressing Stop while DeepSeek-V4.1-Flash read a long prompt let it
  read to the end first (a minute or more on a 28K prompt). It now stops within seconds, and the conversation before
  it stays loaded, so your next message does not read it again.
- **DeepSeek: faster drafting, ~6.5% more tokens a second** with the DSpark drafter on a Strix Halo, the same answers:
  a draft whose earlier drafts would likely be rejected is no longer verified (`MAYA_DS4_DSPARK_CUM`, 0.5; `0` = off).
- **DeepSeek on a Strix Halo / Windows APU: the GPU's memory, measured.** Windows' HIP counts an APU's GPU memory plus
  shared system memory (172 GB on a 160 GB Gorgon Halo); DeepSeek now keeps the model within the dedicated part, as
  GLM already did - no `--vram-experts` needed, the full speed by itself.
- **Windows: `START-MAYA.bat` works again** (#315, #322): since v1.1.18 its screen showed the setup and then did
  nothing; thanks to @kelheor for pinning down the cause.
- **Dashboard: Settings > Port** moves Maya to another port; it starts again there and the page follows.
- **Setup: you choose the drafter, or skip the speed test.** Before the 10-30 minute measurement the setup asks:
  measure, use the drafter, or no drafter (`--drafting measure|on|off`); the choice is kept.
- **Server:** a request whose client went away leaves the queue at once (#314, thanks @jettoblack); with `--parallel`
  each request reports its own timings and cached tokens (#310, thanks @pepuscz).
- **DeepSeek's EXL3 quants on the Tesla V100.** The setup now offers the EXL3 quants from the V100 on (text; pictures
  from the RTX 30 / A100 on): the prompt path runs portable kernels there, and the experts take the newer kernels
  where the older ones did not cover DeepSeek's width.
- **EXL3 on AMD GPUs:** the prompt path's expert kernels (#298 by @LostVector) - the groundwork for running EXL3 quants
  on a Strix Halo.
- **GLM:** fused prompt experts on the RX 7900 XT / XTX, +5-13% prompt speed (#304 by @Cass67); experimental AMD
  Instinct MI100 support (#245 by @nsaxploit); batched decode settings for several conversations, opt-in (#323, #324,
  #325 by @somnergy); prompt-path tools and the MMQ scratch's room counted (#284, #286, #289 by @gopinath87607); and
  `.gitignore` (#308 by @rkarp).

## v1.1.19 - 2026-10-20

DeepSeek-V4.1-Flash reads long prompts up to ~1.5x faster with the same answers, its prompts follow DeepSeek's own
format for agents and tools, and a round of community work.

- **DeepSeek: long prompts up to ~1.5x faster, same answers.** DeepSeek-V4.1-Flash's layers 20-39 see a prompt's
  earlier tokens only through what layer 20 keeps of them, so they need their full work only on the last ~2,700 tokens
  of a prompt. A long prompt now skips the rest: on one GPU, prompts of ~3,700 new tokens or more are read faster -
  the longer the prompt, the more it saves (+30% on a 32K prompt on one GPU, +43-54% on 8-16K prompts on a Strix
  Halo) - with the same answers. `MAYA_DS4_PROMPT_SKIP=2` goes further, as
  DeepSeek's own serving does (the last 128 tokens only; the answers shift a little); `0` turns it off. Learned from
  DeepSeek's DeepInfer.
- **DeepSeek: prompts in DeepSeek's reference format** (#300 by @Gazelle221B, fixes #299). Conversations are rendered
  with DeepSeek's own encoder: later system messages, message-local tools, parallel tool results and their call ids
  are kept, so agent sessions see what they sent.
- **DeepSeek: the engine ends at QUIT** (#306 by @loktar00), so stopping or swapping models no longer waits out a
  timeout; profiling (`DS4_PROF`) on several GPUs no longer disturbs inference (#307 by @cosmicnag).
- **EXL3: exllamav3 1.6's half-integer bitrates** (K 1.5 / 2.5 / 3.5) for self-calibrated community quants, every
  decoder bit for bit exllamav3's (#265 by @sociolog).
- **GLM:** the recurrent-state backup for drafting on a multi-GPU NextN split (#290), a warm-up prompt before READY,
  opt-in (`MAYA_GLM_WARM_PROMPT`, #285), and a docs fix (#287), all by @gopinath87607; the gfx1100 hipBLASLt table for
  ROCm 10 - faster prompts on RX 7900 cards with ROCm 10 (#292 by @Cass67).
- **README:** the community's speed reports since v1.1.12 in the speed table.

## v1.1.18 - 2026-10-20

DeepSeek-V4.1-Flash-1017 adapts to each PC (faster long prompts on one GPU, decode that uses the GPU's PCIe link when
it pays, a faster first answer), GLM-5.3-Flash on the Tesla P100, the Responses API, and a round of fixes - among them
the slower decode on Windows laptops with an AMD Strix Halo since v1.1.17.

- **Windows + AMD Strix Halo: decode back to its speed.** Since v1.1.17 the engines keep an integrated GPU's experts
  within what Windows can commit (RAM + page file), but they counted only what the page file holds now. Windows grows
  a page file it manages (its default) as more is committed, so on a laptop giving its GPU 96 GB, most experts stayed
  off the GPU (decode 17-22 -> 10-15 tokens/s, reported by a user). Both engines now count what the page file may still
  grow by.
- **DeepSeek: long prompts on one GPU borrow the experts' VRAM.** While a prompt runs, its buffers take room from the
  experts' slots, and the experts go back after it, so the decode keeps more experts in VRAM than v1.1.17 did. Prompt
  chunks grow to 16K or 32K tokens where the room allows, a short prompt's reads stay in the page cache for the answer
  after it, and after a long prompt the experts its last rows used are read ahead while the first tokens decode. On one
  GPU by default; several GPUs work as in v1.1.17.
- **DeepSeek: the PCIe lane.** Of a decoded token's experts that VRAM doesn't hold, some are streamed to the GPU and
  computed there while the CPU computes the rest, split by this PC's measured costs. Decode checks that streaming pays
  and turns it off where the CPU alone is faster (`MAYA_DS4_STREAM=0`: off). A low-bit expert now gives the same sums
  on the CPU and the GPU, so Maya-S's answers depend much less on where its experts are placed.
- **DeepSeek: tuned to the PC at start.** The CPU lane's way of running (cores, performance cores, one socket) is
  measured at load, a core's second thread stays with the readers unless asked for (`MAYA_DS4_CPU_LANE`), AMD desktop
  GPUs keep a VRAM reserve, and a new install starts from a shipped expert profile, so its first answers are faster.
- **DeepSeek: repetition, frequency and presence penalties**, drafting included, and `repeat_last_n` /
  `penalty_last_n` as in llama.cpp (0 = off, -1 = the whole context).
- **DeepSeek: repeatable answers with `--no-expert-cache`**: a request's answer no longer depends on what the process
  answered before. `MAYA_DS4_DISK_SKIP_AUTO=1` (experimental, with `MAYA_DS4_DISK_SKIP_W`) skips light experts only
  while the model is cold.
- **GLM on the Tesla P100** (#261 by @ruibeikaa). The engine builds for compute capability 6.0, and four GP100 decode
  kernels: Maya-M on four GP100 dies 21.0 -> 28.3 tokens/s, the same answers. Other cards run exactly as before.
- **The Responses API** (#217 by @Micro-ATP). `POST /v1/responses` with stored conversations, background jobs and
  structured output, each API key seeing only its own records (`docs/RESPONSES.md`).
- **GLM: prompts on two GPUs without NVLink, opt-in** (#270 by @sociolog, `MAYA_GLM_PREFILL_CALC=1`): the waiting GPU
  computes a share of the working one's experts over its own PCIe link (EXL3 experts; +25-48% prompt speed on an
  RTX 3090 + 3060).
- **Fixes:** `--calibrate` keeps the config's own CPU-lane settings unless it measures something clearly faster (#277,
  reported by @klvnblst); MSVC builds (#278, reported by @suprekt) and the Windows test build (#296, reported by
  @jerem91150); the setup finds ROCm installed from Fedora's own packages (#301, reported by @paxsali); the Monitor's
  and the CLI dashboard's live tokens/s no longer jump around with drafting (reported on X by @gabu3d_pl); stopping
  `maya.py` in any way stops the server and the engine too.
- **Smaller changes:** `--spec-follow` decodes a given continuation, for fair speed A/Bs (#266 by @sociolog);
  `--check` names the DeepSeek quant it checks (#294 by @boxwrench); the README shows each model's Hugging Face
  downloads.

## v1.1.17 - 2026-10-20

DeepSeek-V4.1-Flash-1017's prompts many times faster with Maya's quant, all of its files from Maya's repo, and for
GLM-5.3-Flash the repetition penalties, faster long prompts on Volta and a round of community work.

- **DeepSeek: prompts in chunks with Maya's quant** (#279 by @waxinz). The engine now gives Maya's quant 8192-token
  prompt chunks by itself. Before, the setup's tensor parallel fell back to one GPU per layer group (the experts'
  blocks don't split), which also turned the chunks off, so every prompt ran in windows of 16 tokens. On a 7x RTX 3090
  rig with Maya-S, a 32K-token prompt went from 99 to 1010 tokens/s. A warning at start says when chunks are off and
  why. The choice between the chunks and the windows for a prompt now follows what each actually took.
- **DeepSeek: Maya-S as one set of files from Maya's repo.** The model and its Engram tables in one GGUF split of 6
  files, 339.5 GB (was 381 GB): the Engram tables in Q6_K, made from DeepSeek's released ones, at no measurable cost in
  quality. Nothing comes from unsloth's repo any more. An install of v1.1.16 keeps its files and works as it is, and so
  does unsloth's GGUF in its earlier 10 files.
- **DeepSeek: long prompts read each expert once.** On one GPU, a prompt of several chunks takes each layer through all
  its chunks before the next layer, and keeps what that layer read from the files in RAM for its later chunks: a
  16K-token prompt reads 264 GB from the files instead of 736 GB.
- **DeepSeek: the 3.5-bit EXL3 quant, tensor parallel over 4 / 8 GPUs, cache swaps on threads** (#276 by
  @needmorevram). `--ds4-quant exl3-3.5` (LibertAIDAI's, 258 GB; RTX 30 / A100 or newer), the tensor parallel groups
  up to 8 GPUs, and the expert cache's swaps prepared on threads of their own (`DS4_CACHE_THREADS`, 1 by default).
- **DeepSeek: skipping light experts, opt-in.** `MAYA_DS4_DISK_SKIP_W` / `_FROM` leave out a routed expert that is
  not in RAM and carries little of the route's weight - faster on PCs whose RAM can't hold the experts, but it changes
  the answers, so it is off by default (experimental).
- **GLM: repetition, frequency and presence penalties** (#273, reported by @eliaskg). `repeat_penalty` /
  `repetition_penalty`, `frequency_penalty` and `presence_penalty` now apply on every GLM decode path, drafting
  included, and with `temperature` 0 too; requests without them give the same tokens as before.
- **GLM: long prompts faster on Volta** (#254, #233, #215, #199 by @merbanan). The heavily routed prompt experts as FP16
  cuBLAS GEMMs (a 27K prompt 41.3 -> 33.4 s), the DSA indexer's scores on the tensor cores, the shared expert in
  sub-batches, the KDA recurrence over 4 blocks a head.
- **GLM: short turns faster** (#232 and #253 by @merbanan). Prompts from 8 tokens take the batched path on one GPU
  (was 32), and the snapshot is kept as a checkpoint when the next turn starts right after it, so an edit of a short
  reply doesn't read the conversation again.
- **GLM: opt-ins** - CPU-expert deferral and routing affinity (#252 by @merbanan; +21-26% decode on one V100, changes
  the answers), conversation slots parked in RAM (#236 by @handmade0octopus), Volta grouped-WMMA prompt experts
  (#262 by @chenyb999-zhcn, `MAYA_PF_WMMA`), the DFlash verify window on one NUMA node (#197 by @merbanan).
- **Fixes:** the Monitor's NVML calls run in a reader process of their own, so a crashing call no longer ends the
  server (#136, reported by @cosmicnag); DFlash2's BF16 drafter no longer faults at its first draft (#173, reported by
  @waxinz); Windows builds work with a quoted `Path` entry such as `"C:\Program Files (x86)\..."` (#251, reported by
  @rizh42); GLM on a Windows APU keeps its expert pool within Windows' commit, as DeepSeek's engine does.
- **Smaller changes:** the server unloads the model after an idle time, opt-in (#250 by @cryptedy); AMD GPU readings
  from the amdgpu driver on Linux (#269 by @eugenrepp); Visual Studio Build Tools 2026 when the CUDA release supports
  it (#249 by @rkarp); more than 2 GPUs with HIP (#183 by @izabera); the HIP build only for the chosen cards' archs
  (#255 by @andrewprock); EXL3 tests against exllamav3 itself (#264 by @sociolog).

## v1.1.16 - 2026-10-19

DeepSeek-V4.1-Flash-1017 on Maya: Project Maya's own quant of the 748B model, on NVIDIA and AMD, Linux and Windows,
with any amount of memory.

- **DeepSeek-V4.1-Flash-1017.** The setup now asks which model to run: GLM-5.3-Flash as before, or DeepSeek's 748B
  model on an engine of its own (`maya-ds4`): its sliding-window and compressed sparse attention with the two-level
  indexer, hyper-connections, the Engram n-gram tables, the DSpark drafter and the vision tower.
- **Maya-S, Project Maya's own quant of it** (the default, `--ds4-quant maya-s`): the routed experts in 2-3 bits with
  error-feedback rounding, the attention, shared experts and dense weights in 8-bit - 178 GB, plus DeepSeek's Engram
  tables (203 GB, read a few rows a token from the SSD). It picks the same next token as the original model 92.6% of
  the time. [On Hugging Face](https://huggingface.co/peasantsmith/DeepSeek-V4.1-Flash-1017-Maya-GGUF).
- **Or unsloth's GGUFs** (`--ds4-quant gguf-q8` / `gguf-q4`: DeepSeek's own formats, 508 / 501 GB), **or LibertAIDAI's
  EXL3 quant** at 2.9 bits (`--ds4-quant exl3`; RTX 30 / A100 or newer; its basic path: #271 by @needmorevram).
- **Big models, little memory:** the most-used experts go to the GPUs, the others are read in place from the model's
  files - from RAM where they fit, from the SSD otherwise. Keep the files on an NVMe SSD. Runs from NVIDIA Volta
  (V100) on and on AMD (RX 7900, Radeon AI PRO R9700 / RX 9070, Strix Halo / Gorgon Halo).
- **The DSpark drafter:** the setup fetches the model's drafter (8 GB, from Maya's repo: DeepSeek's FP4 experts, its
  dense weights in 8-bit) and a speed calibration keeps it only on PCs where drafting makes answers faster.
- **Pictures:** with the GGUFs the setup fetches DeepSeek's vision tower (1 GB of its checkpoint); it runs on the
  first GPU in about 320 MB, on NVIDIA and AMD.
- **Several GPUs:** the layers are split across them; pairs of equal GPUs also split each layer where the files'
  formats allow.
- **Fast prompts:** a prompt streams only the experts it picks, read from the SSD side by side, and short prompts take
  whichever path is faster on the PC.
- **The attention caches as DeepSeek stores them:** the compressed cache and the index keys in FP4, the sliding window
  in FP8 - 3.5x smaller, the same answers, faster prompts, and more experts fit in VRAM at long contexts.
- **A faster CPU lane:** the experts the CPU computes use 8-bit integer dot products (AVX2, AVX-VNNI, AVX-512), each
  CPU picking its fastest; the results are the same on every one.
- **Windows on a Strix Halo / Gorgon Halo:** Windows counts the GPU's share of the memory against its page file. The
  setup asks for a page file big enough to keep every expert on the GPU and says the size; without it the engine keeps
  fewer experts there and never starves other programs (docs/AMD_MAYA.md).
- **The dashboard and the Chat per model:** with DeepSeek, the Monitor shows its own numbers (experts on the GPU and
  the CPU, SSD reads, the cache's moves, the n-gram memory), the Chat's thinking levels are DeepSeek's (Off, Low,
  High, Max) with its recommended settings, and each model keeps its own saved settings.
- **Model downloads that keep going:** four files at a time, each resumed where it stopped, and run again on its own
  when the connection drops or stalls.
- **GLM-5.3-Flash:** the expert usage record is saved on Windows again (it stopped updating after the first save
  there).
- **Name scope adjustments:** the programs and settings follow Maya's name; existing installs and settings keep
  working and are carried over at their next start.

## v1.1.15 - 2026-10-16

Faster EXL3 drafting across many GPUs, faster long prompts on V100s, a faster CPU lane on Zen 5, and EXL3 on AMD
(experimental).

- **EXL3 with DFlash2 on many GPUs, faster** (#225 by @needmorevram). Pipelined verify rounds, decode's attention on
  the tensor cores (RTX 30 series and newer) and a shorter drafter chain. On an RTX 3090 + 8x RTX 5060 Ti with
  EXL3-4.05bpw and DFlash2: decode 71 -> 81 tokens/s greedy, 69 -> 79 sampled, 46 -> 59 on 32K-token documents - the
  same tokens with drafts as without. Every part has a setting to turn it off (docs/SETTINGS.md).
- **Long prompts faster on Tesla V100 and RTX 20** (#196 by @merbanan). Their prompt attention runs in two passes with
  32 heads a block: on one V100, a 27K-token prompt's attention went from 22.5 to 11.8 s and the whole prompt from 78.4
  to 67.9 s. `STRATA_GLM_PREFILL_ATTN=tc` keeps the previous kernel.
- **A faster CPU lane on CPUs with AVX-512 VBMI** (#227 by @cryptedy). Where the CPU computes Maya-S / Maya-S24's
  experts, CPUs with AVX-512 VBMI (Zen 4, Zen 5, Xeons from Ice Lake on) use new rows that give exactly the same
  results: decode +6-8% on a Zen 5. Other CPUs are unchanged.
- **EXL3 on AMD GPUs, experimental** (#186 by @izabera). The setup offers turboderp's EXL3 quants with AMD GPUs too;
  the EXL3 kernels pass their tests on an RX 7900 XTX and a Radeon 8065S. Their speed on AMD is not tuned yet.
- **DFlash2 on a V100 with little free VRAM** (#195 by @merbanan). The drafting windows failed with "out of memory"
  once the expert pool had taken the VRAM down to a few hundred MB; they run now.
- **Skipping light experts, opt-in** (#228 by @cryptedy). `STRATA_GLM_MISS_SKIP_W` / `_FROM` leave out a routed expert
  that is not in VRAM and carries little of the route's weight - faster on RAM-bound PCs, but it changes the answers,
  so it is off by default.
- **Smaller changes:** the setup picks AMD on a first run on a PC with only supported AMD GPUs (#190 by @thomas9120);
  the image encoder skips its warm-up when it runs on the CPU (#184 by @izabera); docs/SETTINGS.md explains how to get
  the same answers after a restart with the CPU lane on (#224 by @hiroki-abe-58).

NVIDIA GPUs give v1.1.14's tokens with `STRATA_GLM_PREFILL_ATTN=tc`, `STRATA_GLM_DSA_ALL=0` and `STRATA_GLM_MLA_MMA=0`.
As shipped, three faster paths add the same numbers up in another order, so an answer can differ from v1.1.14's in
places: prompts on V100 and RTX 20 (#196), decode on RTX 30 and newer (#225), and conversations under 2048 tokens on
any GPU (#225; `STRATA_GLM_DSA_ALL=0` on AMD too). Otherwise AMD GPUs give v1.1.14's tokens.

## v1.1.14 - 2026-10-16

NVIDIA Pascal cards as experimental, and unsloth's Q6_K_XL / Q8_0 GGUFs of GLM-5.3-Flash.

- **NVIDIA Pascal, experimental** (#214 by @Jw88888, #212). The engine builds for compute capability 6.x - Tesla P40
  / P100, GTX 10 - and the setup accepts these cards, with a note that Maya has not been validated on them yet. On
  Pascal the engine caches global loads in L1 (`-Xptxas -dlcm=ca`), which its decode needs there: two Tesla P40s
  decode Maya-S24 at 28 tokens/s, with the same answers. Like the V100, Pascal needs CUDA 12.x.
- **unsloth's Q6_K_XL and Q8_0 GGUFs** (#216 by @Xar1c). unsloth's UD-Q6_K_XL and Q8_0 quants of GLM-5.3-Flash keep
  some layers' routed experts at Q8_0, and the engine stopped at start with `unresolved weights`. They run now.

NVIDIA GPUs of compute capability 7.0 and newer give v1.1.13's tokens.

## v1.1.13 - 2026-10-16

Faster decode where experts are fetched, a fix for long prompts on a Windows Strix Halo, a new chat with the same system
prompt reusing it, two kinds of AMD GPU together, AMD RDNA 2 builds and the current GLM-5.3-Flash chat template.

- **The expert spares keep what you used last** (#210 by @saatvik-viniak). With one conversation, when the GPU makes
  room for a fetched expert it now evicts the one used least recently, instead of the least used by the aged counts -
  those keep experts that were busy earlier in the chat. On 3x Tesla V100 16 GB with Maya-S at 128K: decode 24.8 ->
  26.5 tokens/s. With several conversations at once (`--parallel`) the old rule stays: there the recency let the latest
  conversation's experts push out what all of them use. `STRATA_GLM_VRAM_EVICT=lru` / `least` chooses either.
- **Long prompts on a Windows Strix Halo with a smaller GPU share** (#206). Under Windows, HIP counts a Ryzen AI Max's
  GPU memory as its dedicated share plus ~12 GB of shared system memory, and since v1.1.10 the expert pool could fill
  past the dedicated share into the shared part, which Windows pages - on a 96 GB share, 8K-token prompts were read at
  62 tokens/s instead of v1.1.9's 159. The pool now stays within the dedicated memory Windows reports for the GPU, and
  the engine's log says so at start. A share the model fits in (160 GB on a Radeon 8065S) is unchanged.
- **A new chat with the same system prompt reuses it** (#179). With the conversation slots on (the default), the end of
  a long system prompt is also set aside on disk, so another chat that starts with it - even after an unrelated
  conversation in between - reads only its own part instead of the whole system prompt again.
- **Two AMD GPUs of different kinds, and an experimental expert helper** (#201 by @zhedang). The setup accepts a
  discrete AMD GPU together with a Ryzen AI Max APU (the discrete one first). `STRATA_GLM_EXPERT_HELPER=1` makes the
  second GPU a helper that holds a share of the experts and computes them, instead of splitting the layers - see
  [the AMD guide](docs/AMD_MAYA.md#experimental-expert-helper).
- **Faster prompts on Strix Halo with ROCm 10** (#203 by @boxwrench). A tuning table for hipBLASLt 100500 (TheRock /
  ROCm 10): on a Radeon 8060S under Linux, prefill 204 -> 341 tokens/s at 2K tokens, 217 -> 373 at 8K.
- **AMD RDNA 2 builds** (#205). v1.1.12's experimental RX 6800 / 6900 and Radeon PRO V620 / W6800 (`gfx1030`) support
  now compiles: AMD's rocWMMA library, which knows only newer GPUs, is left out there.
- **The current GLM-5.3-Flash chat template.** zai-org updated it on 15 October: tool results are matched to their
  calls far faster. The server now renders with it instead of the older copy inside the model files - the same prompts
  (288 test conversations, byte for byte), and a chat with many tool calls is prepared up to ~15x faster. Nothing is
  downloaded again.

NVIDIA GPUs give v1.1.12's tokens with `STRATA_GLM_VRAM_EVICT=least` and with several conversations at once.
With one, the new rule keeps other experts in VRAM, and where the CPU computes the rest an answer can differ
from v1.1.12's in places.

## v1.1.12 - 2026-10-15

EXL3 on one GPU with the CPU's help, batched answers across a layer split (opt-in), the VRAM tier following the
conversation while DFlash2 drafts, and AMD RDNA 2 cards as experimental.

- **EXL3 on one GPU: the CPU takes more of the experts, faster** (#188 by @needmorevram). On one GPU most of an EXL3
  model's experts run on the CPU. The CPU now reads each expert in memory order (split-K pieces) and runs a route's
  experts as one batch of jobs, and CPUs without AVX-512 (Intel desktop chips, Zen 2 and 3) get an AVX2 tier of their
  own. Prompts give their least-used experts to the CPU too, as GGUF prompts already did. On an RTX 3090 alone:
  EXL3-3.05bpw decode 14.3-16.8 -> 20.9-22.1 tokens/s, 22.4 -> 29.2 with DFlash2; EXL3-4.05bpw 14.4-15.1 -> 18.9-19.6;
  an 8K prompt 384-396 -> 422-489 tokens/s. The answers are the same tokens as before.
- **Batched answers across a layer split, opt-in** (#204 by @zetlaw). With `STRATA_GLM_SPLIT_ROWS=k` (1-8) and several
  requests at once, each GPU of a split takes up to k requests in one step, reading the shared weights once for all of
  them. On four RTX 4000 Ada GPUs with Maya-M and eight requests, `k=2` gave 94.5 tokens/s in all against 74.4 (+27%);
  larger groups were slower there, so it stays off by default. See
  [several requests at once](docs/SETTINGS.md#several-requests-at-once).
- **The VRAM tier keeps following the conversation while DFlash2 drafts** (#198 by @merbanan). With the drafter's
  verify window on two CPU lanes, the background swaps that move the experts you use into VRAM had stopped. They run
  again: on a V100 with a Ryzen 9 7950X, the share of experts found in VRAM went from 49-62% to 54-67%.
- **AMD RDNA 2, experimental** (#205). The setup accepts the RX 6800 / 6900 series and the Radeon PRO V620 / W6800
  (`gfx1030`), and compiles for them only when one is chosen. Maya has not been validated on such a card yet: please
  tell us how it goes in an issue.

NVIDIA GPUs give v1.1.11's tokens.

## v1.1.11 - 2026-10-15

What fills each GPU, in the Monitor, and the expert pool's size as a setting in the config.

- **The Monitor shows what fills each GPU** (#191 by @needmorevram). The GPUs table becomes a card with a bar for each
  GPU: the model's weights, the KV cache, the activations, a drafter and its verify window, the expert pool, CUDA and
  the rest, other programs and the free memory, with that GPU's load, power, temperature and PCIe link under its bar.
  Pointing at a part picks it out in every bar; a click opens a GPU's row with a sentence for each part, and the advice
  names the GPU (other programs holding memory that a restart would give to the expert pool, say). On two Tesla V100s
  the parts add up to what the NVIDIA driver reports for each card; on AMD, where the driver does not list the memory a
  process holds, the card shows the engine's own measurement at its start.
- **The expert pool's size in the config** (#192 by @needmorevram). `--expert-cache auto`, the default the setup now
  writes, sizes the pool from the free VRAM less a reserve as before, and then checks that budget once it is written:
  under Windows the free figure read before an allocation can run about a gigabyte high, and a card filled past it
  pages GPU memory to system RAM, where decode runs several times slower. Short of the reserve, the pool gives back the
  difference. `--expert-cache N` caps the first GPU's pool at N experts; `--vram-reserve-mib N` (and
  `--vram-reserve-later-mib N` for a split's later GPUs) sets the reserve. At start the log says how much VRAM each GPU
  has free with everything loaded, and warns when it is low. The check is left out on the Ryzen AI Max APUs and where
  the free VRAM is far beyond what the pool can take.

NVIDIA GPUs give v1.1.10's tokens, and the Ryzen AI Max APUs keep v1.1.10's speed.

## v1.1.10 - 2026-10-14

The Strix Halo and Gorgon Halo release: faster answers and faster prompts on the Ryzen AI Max APUs, a faster cold start
from a hard disk, and prompts through both links of an NVLink pair.

- **Strix Halo and Gorgon Halo: faster answers.** On the Ryzen AI Max APUs (Radeon 8050S / 8060S / 8065S, `gfx115x`),
  answers come faster with the same text. On a Gorgon Halo (Radeon 8065S, Ryzen AI Max+ PRO 495 with 192 GB, 160 GB of
  it for the GPU, which holds all of Maya-S; Windows 11), 256-token answers to short prompts run at 28.4-29.6 tokens/s
  instead of 19.7, and at 23.2-23.5 instead of 18.4-18.5 after 3K- and 8K-token prompts; longer answers (about 600
  tokens) hold 26.3 and 23.4. It is on by default on these APUs only; every other GPU decodes as in v1.1.9. A 128 GB
  Strix Halo, whose GPU holds less of the model, is not measured yet. `STRATA_GLM_MTP_DECODE=0` turns it off, `1` tries
  it on one GPU elsewhere (usually slower where experts stream from RAM); it stays off with `--parallel` or a DFlash2
  drafter. Thanks to @boxwrench for testing on a Strix Halo.
- **Every expert stays in GPU memory where it fits** (#101). Where the GPU's budget holds every expert plus the room a
  prompt borrows (a 192 GB Gorgon Halo under Windows), the start loads all of them, and a long prompt no longer moves
  experts out to read them back from the SSD: answers after an 8K-token prompt +7%. On by default on AMD
  (`STRATA_GLM_COMPLETE_POOL`, `STRATA_GLM_RESIDENT_SKIP`); NVIDIA keeps its pool unless they are set to `1`.
- **Faster prompts on Strix Halo and Gorgon Halo** (#98 and #113 by @boxwrench). A prompt's routed experts run in one
  fused int8 pass, and the wave32 WMMA attention is the default on `gfx1151`: on the Radeon 8065S, 3K- and 8K-token
  prompts read at 271 and 278 tokens/s instead of 233 and 253. `STRATA_GLM_PREFILL_FUSED=0` and
  `STRATA_GLM_PREFILL_ATTN=f16q` give the old paths.
- **A faster cold start from a hard disk** (#182 by @hmartiniano). The start reads a GGUF pack's experts in the order
  they lie in its files, as it already did for EXL3, instead of the order the routes rank them, so a hard disk reads
  them in one pass instead of seeking from one to the next; how much faster depends on the disk. Which experts go to
  VRAM and RAM is unchanged.
- **Two GPUs over NVLink: prompts through both links** (#121 by @q8atnight). With `STRATA_GLM_DUAL_PREFILL=1`, in a
  prompt of one chunk the waiting GPU stages every second expert over its own PCIe link and NVLink (Maya-M on two RTX
  3090 with NVLink: 1K- and 4K-token prompts +28% and +24%). The start checks that peer copies arrive intact and keeps
  it off on a machine whose GPUs report peer access but deliver wrong bytes.

## v1.1.9 - 2026-10-14

EXL3 decodes on the tensor cores of RTX 30 and newer, EXL3 prompts are faster, and turboderp's 4.05-bit EXL3 quant is
in the setup.

- **EXL3 decode on the tensor cores** (#180 by @needmorevram). On RTX 30 and newer each 16 x 16 tile of an EXL3
  weight is decoded straight into a tensor-core product, up to 8 tokens at once: the expert kernels run about 1.8x as
  fast on an RTX 3090 and an RTX 5060 Ti, and decode is 14-30% faster on an RTX 3090 + eight RTX 5060 Ti
  (EXL3-4.05bpw, with the MTP block and with DFlash2). The answers are not bit for bit v1.1.8's there: the sums run in
  the tensor cores' order. NVIDIA GPUs before RTX 30 (V100, T4, RTX 20) keep v1.1.8's decode kernels, which are faster
  there - the same speed and output as v1.1.8. `STRATA_GLM_EXL3_DECODE=legacy` or `new` picks one on any GPU.
- **EXL3 prompts rotate the rows, not the weights** (#180). An expert that goes through cuBLAS has its rows of tokens
  rotated instead of its whole weight, so the weight is only decoded: an 8K prompt +6% on the nine GPUs above, +3% on
  a V100. `STRATA_GLM_EXL3_ROT=0` gives the old path.
- **EXL3-4.05bpw in the setup** (#180). turboderp's quant at 4.05 bits per weight (`--model EXL3-4.05bpw`, 165.2 GB
  with the MTP block): 4-bit routed experts and mostly 6-bit dense weights. Experimental and NVIDIA only, like the
  3.05-bit one; every file is checked against its sha256.
- **The layer split for EXL3 packs** (#180). `--layer-split auto` assumes flatter routes for an EXL3 pack (its experts
  are bigger, so a card holds fewer of a layer's): over an RTX 3090 and eight RTX 5060 Ti with DFlash2, the 3090 takes
  9 layers instead of 7 and decode is 24-27% faster. GGUF packs split as before.

## v1.1.8 - 2026-10-14

Contributors' fixes and additions: speculative sampling on Turing cards, AMD's tuning table for the installed library,
prompt-reading progress on the stream, and tool calls with templates that open thinking only when asked.

- **Speculative sampling on Turing cards** (#161 by @sociolog). With `STRATA_GLM_SPEC_SAMPLE=1`, its draft kernel
  needed more registers than a Turing card (RTX 20, T4) has for a block of 1024 threads, and the launch failed wherever
  the MTP block sat on such a card. The kernel is now bounded to fit; other cards run it as before.
- **AMD: the tuning table for the installed hipBLASLt** (#162 by @Tsury, #158). The setup passes every prompt tuning
  table made for the card's architecture, and the engine takes the one made for the installed hipBLASLt version.
  Before, it passed only the newest, and a PC with another library version fell back to plain hipBLAS. A Strix Halo
  (gfx1151) table for hipBLASLt 100401 is added: on @Tsury's Linux PC with that library, a tool-calling session's
  prompts took 29 s instead of 52 s.
- **Prompt progress on the stream** (#163 by @eliaskg). A streamed chat request with `"return_progress": true` gets
  `prompt_progress` chunks (`total`, `cache`, `processed`, `time_ms`) while a long prompt is read, so a client can draw
  a progress bar. Nothing changes unless a request asks; non-streamed requests and the Anthropic API are as before.
- **Tool calls with templates that open thinking only when asked** (#148 by @beverm2391). With Max thinking, a model
  whose chat template opens `<think>` only when `enable_thinking` is true returned its tool calls inside the reasoning
  text. The template now gets the thinking default the reply parser assumes. Maya's GLM template renders as before.
- **Docs:** the CPU image encoder with the AMD engine on Linux, set up by hand (#159 by @Tsury); what to install when
  the CUDA profiler or NVTX headers are missing (#160 by @Tsury).

## v1.1.7 - 2026-10-13

Faster prompts with the INT8 cache, Windows fixes for speed in the background and for the RAM tier, and a guard
against model files that change after setup.

- **Windows: full speed with Maya's window in the background.** The engine opts out of Windows' power throttling
  (EcoQoS) itself, which moved its CPU lane onto a hybrid Intel CPU's efficiency cores whenever every Maya window was
  in the background - decode dropped to a fraction of its focused speed (#114, thanks to @githug2024 for testing
  `powercfg`).
- **Windows: the RAM tier fills the pinning limit.** Where Windows refuses to pin a planned tier, the engine now
  measures how much still pins and sizes the tier to it, instead of stepping down 3% at a time - those steps left a
  128 GB PC's tier 2.5 GB under its limit, and the SSD served those experts (#129, measured by @anbow1). With several
  expert sizes each takes its planned share. The log's "RAM tier sits on node X of N seen" line now appears only on
  PCs with two NUMA nodes or more, where it means something (#114, #129, #147).
- **A guard against model files that change after setup** (#155, traced by @zhanghuqqqq). Before the setup builds a pack from a Maya model's
  files, it checks them against their published sha256 - files it downloaded were always checked, files already in the
  folder now are too, once each. The pack records the files it was built from, and a start refuses it if they changed
  afterwards (`./maya.sh --setup --repack` builds it again): a pack built while a download manager was still filling
  files it had created at full size answered garbage.
- **Prompts with the INT8 cache** (`--kv int8`). The prompt attention converts the cache's INT8 records to FP16 for
  less (a byte permute and FP16 arithmetic instead of int-to-float conversions), with the same bits as before - checked
  for every code and every scale on NVIDIA and AMD. On two V100s the attention's extra time over FP16 went from +18% to
  +11%, and INT8 prompts are now within about 2% of FP16's (8K-32K).

## v1.1.6 - 2026-10-13

Long contexts with the attention's cache in RAM, EXL3 with the DFlash2 drafter up to twice as fast, and faster experts
on the CPU and in prompts.

- **KV streaming for long contexts** (#86 by @needmorevram). Each attention layer's cache can live in pinned RAM with a
  window of it in VRAM (`--kv-streaming on`, `--kv-resident N`, 32,768 positions by default), so the VRAM it frees
  holds experts. It pays off where the cache is large beside the cards: at 512K on an RTX 3090, 1,176 expert slots
  instead of 714, prompts 4-9x faster and decode +5%; 1M context starts on one 3090; four conversations at 128K on two
  GPUs (`--parallel 4`) decode 37% faster together. New setups turn it on; an existing config gets it from
  `./maya.sh --setup` (Windows: `START-MAYA.bat --setup`). Decoding gives the same tokens as without it.
- **EXL3 with the DFlash2 drafter decodes each weight once per window** (#164 by @needmorevram). A verify window
  decoded every EXL3 weight once for each of its tokens; now once for all of them, with the same bits. On an RTX 3090 +
  5060 Ti, DFlash2 with EXL3 decodes about twice as fast and is now 15-29% ahead of the MTP block (which itself gains
  4-7%); on one RTX 3090 it is 13-30% faster than plain decode in chats.
- **Faster CPU experts for IQ3 quants** (#166 by @needmorevram). The CPU lane's one-token IQ3_S / IQ3_XXS rows - all
  of Maya-L's gate/up weights - decode 10-13% faster on a few cores, bit-identical to before (checked against ggml's
  own dot at first use; `STRATA_GLM_CPU_1T=0` turns it off).
- **Prompts: the experts' SwiGLU inside the MMQ product** (#166). On RTX 20 and newer, the prompt path's gate/up
  product writes the SwiGLU directly instead of handing a buffer twice its size to a separate kernel: that step -17% on
  an RTX 5060 Ti, -3.4% on an RTX 3090, the same values (`STRATA_GLM_MMQ_GLU=0` turns it off). It is applied to the
  pinned ggml when the engine is built; without git (a source zip) the engine builds as before.
- **Docs:** pages inherited from the code Maya grew out of, about other models, are gone; the README has five more
  machines measured by users.

## v1.1.5 - 2026-10-13

Conversations that share their start read only what is new, safer serving for agents, faster CPU experts, and opt-in
faster sampled answers on two GPUs.

- **Conversation checkpoints** (#146 by @needmorevram). A prompt that shares its start with an earlier one - the same
  system prompt and tools in another chat, or a chat whose last message was edited - now reads only what is new. The
  conversation's state is kept in RAM (about 150 MB each, four by default) at the end of the system prompt, at the
  start of the last turn and every 16K fresh tokens (`STRATA_GLM_CKPTS`, `STRATA_GLM_CKPT_EVERY`).
- **Special tags in messages stay text** (#146). A user quoting `</think>`, a file an agent reads that holds
  `<|user|>`, or a tool result printing `</tool_response>` no longer hands the model a real end of thinking, a new turn
  or the end of the result. A tool set sent in another order keeps the order it was first seen in, so the conversation
  is not read again from the system prompt.
- **Request API** (#146): OpenAI's `logit_bias` (-100 bans a token); sampling fields sent as numeric strings are read,
  and one that can't be read is a 400 naming the field; llama.cpp's `repeat_penalty` / `repeat_last_n`;
  `GET /metrics/prometheus` with llama-server's metric names and Maya's own.
- **Recovery and tuning** (#146): a conversation whose state turns non-finite is dropped and the request read again
  from the start; the CPU lane's split is timed again during decode (`STRATA_GLM_CPU_RETIME`), so a start that timed
  the CPU at a busy moment no longer keeps a skewed split; `STRATA_GLM_TRUNK_KEEP` applies in the DFlash2 windows too.
- **CPU experts shared by several tokens are decoded once** (#154 by @needmorevram). A prompt's CPU experts and a
  DFlash2 window's CPU pass read each weight row once for all the tokens that share it instead of once per token, with
  results bit-identical to before (checked against ggml's own dot at first use; `STRATA_GLM_CPU_MT=0` turns it off).
  Where the CPU computes a large share of the experts: decode +3.5-4.5% and prompts up to +15% on an RTX 3090 + RTX
  5060 Ti with 2x Xeon. For prompts with images, the MTP block's cache now reads the image's rows, as the model does.
- **Speculative sampling and cancelled wrong drafts for the two-GPU MTP decode** (#145 by @sociolog; opt-in:
  `STRATA_GLM_SPEC_SAMPLE=1`, `STRATA_GLM_SPEC_CANCEL=1`). Sampled answers keep exactly the same distribution of
  tokens and decode faster: +5% on an RTX 3090 + 3060.

## v1.1.4 - 2026-10-12

The EXL3 quant works with the DFlash2 drafter, and on Windows + AMD a conversation taken back from a slot answers its
own question again.

- **EXL3 with the DFlash2 drafter** (#141 by @needmorevram). The drafter's verify window read each token's embedding
  at the wrong row size for EXL3's BF16 table, so the answers turned to noise (#129, #140, #142), and on one and two
  GPUs the window had no kernels for EXL3's experts. Both are fixed. A prompt asked again no longer drafts from the
  previous answer, with every model. On two GPUs EXL3 decodes faster with the MTP block for now, and the speed
  calibration keeps it there.
- **Windows + AMD: a conversation taken back from a slot answered the other conversation's question, then looped**
  (#133). A new conversation's state was zeroed on the GPU's default queue, which nothing made the engine's own queues
  wait for, and Windows' HIP runtime holds queued work until the program waits on the GPU: the next conversation
  started from the previous one's state. The state is now zeroed on the engine's compute queue and waited for.

## v1.1.3 - 2026-10-12

DFlash2 kept only where it is faster (a speed calibration on your PC), an expert profile that fills VRAM well from
the first answer, an opt-in skip of the experts that would come from the SSD, and the EXL3 quant back in the Windows
setup.

- **The speed calibration** (DFlash2). DFlash2 makes some PCs faster and others slower: it depends on how much of the
  model fits in the GPUs' memory, and on the CPU and the RAM. When the setup drafts with DFlash2 - and once, at the
  first start after this update, for a config that already does - Maya measures it on your PC against the MTP block
  (one GPU: against no drafting) and keeps the faster. It loads the model twice; each time it reads a long document,
  writes about 1,500 tokens of answers after a warm-up (sampled as the dashboard samples, the same questions both
  times) and compares the sustained speeds; DFlash2 stays only when it is more than 5% faster. 5-20 minutes, every
  step shown. `./maya.sh --speed-calibration` measures again; `--draft dflash2 --no-speed-calibration` keeps DFlash2
  without measuring.
- **An expert profile for every GLM pack** (#88 by @needmorevram). A start fills VRAM by a profile built from
  Maya-L's routing - code, math, eight languages, long documents and tool calls, with MTP drafting on - instead of
  expert order or one machine's usage file: on 9 GPUs a fresh start decodes 35.4 -> 40.7 tok/s. The draft layer keeps
  at least its card's equal share of slots. Learning across starts is opt-in (`--expert-profile-save FILE`), and the
  usage file is read only when `STRATA_GLM_USAGE` names it. An installed config gets the profile at its next start.
- **Skip the experts that would come from the SSD** (#109 by @Ajay9o9; opt-in, changes the answers).
  `STRATA_GLM_DISK_SKIP_W=0.12`: where the model does not fit in VRAM + RAM, a routed expert held by neither and with
  less than 12% of the route's weight is left out of the token instead of read from the SSD - an RTX 3090 with 60 GB
  of RAM decoded 62% faster, an RX 7900 XTX 54%. The answers move away from the full model's, more the less RAM there
  is: KL 0.12 with 60 GB of RAM (the top token the same 88% of the time), 0.19-0.23 with 24-40 GB (75-78%). Unset,
  nothing changes.
- **The experimental EXL3 quant is offered on Windows again**, as v1.1.0 released it; v1.1.1 had hidden it there
  after one crash report (#129), which we are still looking into with its reporter.
- The setup's DFlash2 choice says what it gives again - the same answers, faster - now that the speed calibration
  checks it on every PC.

## v1.1.2 - 2026-10-12

Faster sampled answers, pictures in Claude Code's tool results, and a build fix for PCs with two CUDA toolkits.

- **Sampled answers decode faster** (#92 by @0xPreDa). The sampler now picks the top_k tokens in one pass instead of
  k scans of the vocabulary, and runs the draw chain on one thread: a sample takes about 0.07 ms instead of 1.5 ms at
  top_k 20 on an RTX 4090 (11 ms at top_k 64, which `top_k` 0 also means). Sampled decode on 4x RTX 4090: +13%. The
  picks are the same, token for token - checked on NVIDIA and AMD.
- **Pictures in tool results reach the model** (#120 by @CoolJohn-lab, fixes #119). A picture inside an Anthropic
  `tool_result` - how Claude Code's Read tool returns an image - was dropped before the image encoder; now the model
  sees it.
- **A second CUDA toolkit** (#124 by @fuad00): with `--nvcc` pointing at a toolkit other than the system's, the engine
  and the image encoder now link that toolkit's runtime and cuBLAS; they took the system's before. A build that
  already linked the wrong ones needs its `build/` folder removed once.
- README: a 2x RTX 4090 row (#123 by @fuad00).

## v1.1.1 - 2026-10-12

A guard for a crash reported on v1.1.0, and clearer advice on the DFlash2 drafter.

- **The setup no longer offers the experimental EXL3 quant on Windows** (#129). There an EXL3 model stopped at its
  first prompt with an illegal memory access, after a 125 GB download. `--model EXL3-3.05bpw` still asks for it, with a
  warning, while we look into it. Linux: nothing changes. Maya's own models are not affected.
- **DFlash2: measure yours.** Where most experts come from RAM, the MTP block can stay faster than the drafter: an
  RTX 3090 + 3060 with Maya-M decoded faster without it (#127). The setup and the docs say so; `--bench` tells.

## v1.1.0 - 2026-10-12

Three big additions at once: Maya answers several requests together, drafts with a DFlash2 drafter, and runs EXL3
quants. Also: web pages on another address can call the API (opt-in), and a clearer Windows start line.

- **Several requests at once** (`--parallel N`, up to 8; #115). `./maya.sh --parallel 4` (Windows:
  `START-MAYA.bat --parallel 4`) keeps it in the config. Maya then answers up to N requests at the same time instead of
  one after another.
  - **One GPU:** a token of each request goes through the model together. The dense weights (attention projections,
    shared experts, dense layers, the output head) are read once for all of them; the attention and the routed experts
    run per request. Each request gets exactly the tokens it would get alone.
  - Real chat questions on a V100 that streams most of Maya-S's experts from RAM: x1.41 in all for two at once, x1.35
    for four. With every expert in GPU memory (a Strix Halo with Maya-S): x1.67 for two and x2.02 for four. Each
    conversation brings its own experts, so the more of the model VRAM holds, the more it gains.
  - **Two GPUs or more:** each card works on a different request, one behind the other.
  - A request alone decodes as fast as before. Each extra request keeps its own context state in VRAM, taken from the
    expert pool at the start.
- **Faster answers with a DFlash2 drafter** (#106 by @needmorevram; NVIDIA RTX 30 / A100 and newer).
  `./maya.sh --setup --draft dflash2` downloads a small drafter (0.7 GB) that drafts up to 7 tokens a round; the model
  checks them and keeps the right ones, so the answers are the same. Maya-L: +44% on nine GPUs and +12% on two against
  the MTP block, +19% on one RTX 3090. The MTP block stays the default; older cards keep it (on Tesla V100s the drafter
  was slower). The drafter's license is CC BY-NC-ND 4.0: non-commercial use only. With `--parallel` above 1 the engine
  decodes without it.
- **EXL3 quants** (#75 by @needmorevram; experimental, NVIDIA). The engine reads EXL3 quants in place from their
  safetensors, with their own kernels for decode, prompts and the CPU lane. The setup can download one at 3.05 bits
  per weight (`./maya.sh --setup --model EXL3-3.05bpw`, 125.3 GB, with the MTP block). Tested on Linux; not measured
  against the FP8 model yet.
- **Web pages on another address can call the API** (#116 by @sociolog, opt-in). `"cors_origins":
  ["http://127.0.0.1:8090"]` in the config lets a browser page served elsewhere (a local chat client, say) use `/v1/*`;
  `["*"]` lets every page in - only with an `api_key`. Settings and tools stay for Maya's own page. Without it nothing
  changes.
- **Windows: which NUMA node the RAM tier landed on** (#112 by @noahark, from #56). On a two-socket PC the RAM tier
  lands on one socket or the other from start to start, and the far one is slower. The start now says
  `glm fast: CUDA0 the RAM tier sits on node 1 of 2 seen`, so you (or a launcher) can restart when it drew the far side.
  It replaces v1.0.34's longer NUMA line.
- **Fixed:** a rare crash in the CPU lane's thread pool when one batch of work started while a straggler of the
  previous one was still finishing (#75).

## v1.0.34 - 2026-10-11

Fixes from your reports: a two-GPU Windows PC whose RAM tier came out smaller in v1.0.33, and builds on CUDA 12.0-12.2.
Also two new opt-in settings for machines that read experts from the SSD.

- **A split's RAM tiers near Windows' pinning cap** (#110, from #105). A two-GPU Windows PC pinned 35.6 GB of its
  44 GB RAM budget on v1.0.33 against 42.6 on v1.0.32, so more experts came from the SSD. The fix:
  - the GPUs pin their tiers one after another, so only the one that reaches the cap gives way;
  - a tier leaves 512 MB of room under the cap;
  - the expert tables upload through the engine's own pinned buffer, so reaching the cap no longer stops the start with
    `the warmed expert tables did not upload`.
- **Windows with two NUMA nodes or more** (#110, from #56): the engine log says which processor group and node the
  engine runs in, and where the RAM tier's pages are, to find why some starts decode slower than others.
- **Builds on CUDA 12.0-12.2** (#107 by @Ajay9o9), the version Ubuntu's `nvidia-cuda-toolkit` installs.
- **New, opt-in: `STRATA_GLM_DISK_READERS=<n>`** (#108 by @Ajay9o9). Prompts read the experts that come from the SSD
  with n threads, each expert whole, handed on as soon as it lands. On a DRAM-less NVMe, prompts read 25-50% faster; an
  NVMe with its own DRAM can lose instead, so measure yours. The answers stay the same.
- **New, opt-in, it changes the answers: `STRATA_GLM_TRUNK_KEEP` / `_MAP`** (#103 by @sociolog), for a RAM-bound split
  that would rather be faster. The model's own layers leave out their lowest-ranked experts that aren't in VRAM, as the
  MTP draft does, layer by layer with the map. On an RTX 3090 + 3060, Maya-M decoded 23% faster with the map in the
  README's row, KL 0.018 from the full model.

## v1.0.33 - 2026-10-11

Several conversations at once on a layer split, a short request no longer waits for a long answer, `--calibrate` also
tunes the disk reads, and fixes from your reports: a start that failed on Windows, a speed that changed between
restarts, and ZFS.

- **Several conversations at once on a layer split** (#94 by @0xPreDa): with `STRATA_GLM_SEQS=<n>` on two GPUs or more,
  n conversations decode together, each card of the split working on a different one instead of waiting for the
  others. On 4x RTX 4090, four conversations at once give 225 tok/s in total. It's off by default; one GPU gains
  nothing from it. With it on, every conversation writes the same tokens as it would alone.
- **A short request no longer waits for a whole long answer** (#93 by @0xPreDa):
  - requests are served in arrival order;
  - with `STRATA_FAIR_SLICE_S=<s>`, a long answer that has been decoding for s seconds while another request waits
    lets it in, then continues from where it was.

  On 2x V100, a short question asked during a 700-token answer got its first word after 10 s instead of 60.
- **`--calibrate` tunes the disk reads too** (#99, from #67): when part of the model is read from the SSD while it
  answers, the tuning also tries reading each expert in fewer, larger pieces (`STRATA_GLM_READ_CHUNKS`), and keeps a
  size only when it is more than 3% faster. A Windows laptop on an Intel RST RAID decoded 15% faster with 4 pieces than
  with the default 8; a Linux NVMe stays fastest at 8.
- **Fixes from your reports** (#99):
  - **A start that failed right after the RAM tier** on Windows (`pinned disk staging did not allocate`, #67): the
    small pinned buffers are now set up before the tier, so a cap on pinned memory makes the tier slightly smaller
    instead of stopping the start.
  - **A speed that changed between restarts** (#56): the CPU lane's timing at start is now taken over half a second,
    and keeps the fastest round, so a burst of other work at that moment no longer slows the whole session.
  - **ZFS:** the memory ZFS's cache can give back now counts as free RAM when the RAM tier is sized (#89).
  - **The prefetch** says its settings in the engine log when it is on (#6).
- The README has rows for the new settings.

## v1.0.32 - 2026-10-11

A new model to download, Maya-M-Derisked. A split's GPUs now load at once, a split of three or more GPUs no longer
pins more RAM than the PC has, the Ryzen AI Max APUs decode faster, and the setup takes a first shard that holds
metadata only.

- **Maya-M-Derisked** (#97): Maya-M with a directional weight modification by Blackfrost_AI that reduces blanket
  refusals. It is not a new quant: the same IQ2_S files, tensors, MTP draft block and size as Maya-M (116 GB), with
  some of its weights changed. It is experimental, and it has a repo of its own on Hugging Face. `./setup.sh --setup
  --model Maya-M-Derisked` (Windows: `START-MAYA.bat --setup --model Maya-M-Derisked`); the setup's model menu offers
  it too. It reads pictures with Maya's vision files, like the other models.
- **A split's GPUs load at once** (#80 by @ksanislo): a thread per GPU instead of one GPU after another. On 4x Tesla T4
  with Maya-L, start to the first token went from 142 to 96 s. The GPUs plan their RAM tiers in turn, in GPU order, so
  each gets the same tier as before; then they pin and warm them at the same time. `STRATA_GLM_PARALLEL_LOAD=0` keeps
  the old order.
- **The CPU lane's timing, kept across starts** (#81 by @ksanislo): with `STRATA_GLM_CPU_CAL=<file>`, the CPU lane's
  timing at start (~1.7 s a GPU) is written once and read by later starts. A new build, another card or thread count
  times again, and a timing that decode finds off is dropped. Unset, nothing changes.
- **Three GPUs or more on Linux no longer freeze the desktop at start** (#85, issue #77): the later GPUs' prompt buffers
  are now set aside when the RAM tier is sized, each later GPU measures the free RAM again, and a warning says when
  what is left is short. A pinned split's CPU lane runs a thread a physical core.
- **Faster decode on Ryzen AI Max APUs** (#95, from the Gorgon Halo work): the decode's dense projections run in one
  RDNA3 kernel that keeps two blocks' loads in flight, bit for bit the original kernels' results. Radeon 8065S with
  Maya-S: decode +3.1%. On by default on gfx115x; `STRATA_GLM_MV_RDNA=1` on other gfx11 cards, `0` = off.
- **The setup accepts a first shard of metadata only** (#85, issue #83): a GGUF whose first shard holds only the
  tokenizer and settings (as unsloth's do) stopped the setup as incomplete.
- **For cache studies** (by @sociolog):
  - `STRATA_GLM_VRAM_EVICT=lru` refills a layer's spares by evicting its least recently used expert (#78);
  - `STRATA_GLM_ROUTE_LOG` writes each route's 16 near misses (#90);
  - `tools/glm_tier_replay.py` replays a route log through a model of the VRAM tier's rules and alternatives (#91).
- The README has rows for the new settings.

## v1.0.31 - 2026-10-10

The expert prefetch can now pay on a RAM-bound split: three settings choose when its copy starts, how many GPU blocks it
takes and which predictions it copies.

- **The expert prefetch, tunable** (#84 by @sociolog): `STRATA_GLM_PREFETCH_N` copies the next layer's predicted
  experts that are not in VRAM into that layer's spares while a layer computes. As it was, it cost more than it saved
  on a RAM-bound split. Three settings for it; the prefetch stays off by default and nothing else changes:
  - `STRATA_GLM_PREFETCH_AT=route|fetch|cpu`: when the copy starts - at once (as before), after this layer's own
    fetch, or after its CPU-lane answer;
  - `STRATA_GLM_PREFETCH_BLOCKS=<n>`: the copy's GPU blocks (half the SMs before);
  - `STRATA_GLM_PREFETCH_RANK=<n>`: copy only the prediction's first n ranks. Its first guesses are right almost
    always, its 5th-8th mostly not.

  Maya-M on an RTX 3090 + 3060 at 21K context: decode (writing the answer) 19.64 tok/s without the prefetch, 20.22
  with `N=1 AT=fetch BLOCKS=4 RANK=2`, and +4.7% over a 40-turn agent-like conversation. The README has a row for it.

## v1.0.30 - 2026-10-10

AMD GPUs set up on Windows too, Windows reads its experts from the SSD much faster, the CPU lane times itself
correctly at start, and the logs say Maya.

- **AMD on Windows, Strix Halo / Gorgon Halo included (experimental)** (#69): `START-MAYA.bat --backend hip` sets
  Maya up on Windows 10/11 as `./maya.sh --backend hip` does on Linux, for Ryzen AI Max 300 / 400 (Radeon 8050S /
  8060S / 8065S) and the RX 7900 XT / XTX and RX 9070 / R9700.
  - **Setup:**
    - finds the GPUs with ROCm's hipInfo, else Windows' display adapters;
    - uses AMD's HIP SDK for Windows, or offers AMD's ROCm SDK wheels in `.venv` when there is none;
    - compiles the engine with ROCm's clang and Ninja in Visual Studio's environment (2022 or 2026).

    The SDK's HIP runtime goes next to `strata.exe`, so the driver's own copy in System32 is not loaded instead.
  - **The engine on a Windows APU:**
    - The GPU's memory is a fixed carve-out that Windows does not count as RAM, so the engine sizes it like a
      discrete card.
    - An expert pool larger than one allocation goes into several.
    - The engine submits queued GPU work when no route has come for 1 ms, because Windows' HIP runtime holds
      launches until the host waits.

    Linux and NVIDIA are unchanged.
  - **Two HIP checks were wrong on every platform:** `hip_glm_handoff` passed a removed argument, and
    `hip_prefill_mmq_parity` raced its output sentinel.
  - **Checked on a Ryzen AI Max+ PRO 495 / Radeon 8065S under Windows 11:** setup builds and packs Maya-L, every
    expert stays in the GPU pool, answers, long prompts and thinking come out right, and all 15 HIP checks pass.
- **Windows reads experts from the SSD much faster** (#73 by @klvnblst, for #65 and #67):
  - **Unmapping:** while a model file stays mapped, Windows sends every direct read of it down a slower path. The
    engine now unmaps the files once they are loaded, moving the token table (~0.7 GB) to RAM. On the RTX 5090
    Laptop of #67, decode is about 3.5x as fast. `STRATA_GLM_KEEP_MAP=1` keeps the old behaviour.
  - **CPUs without SMT** (Intel Core Ultra 200 and others): the CPU lane uses two thirds of the cores, leaving the
    rest for the disk readers and the service thread.
  - **`--calibrate`:** the CPU lane threads it leaves out sleep instead of spinning.
- **The CPU lane times itself on experts read from RAM** (#70 by @merbanan): its start-up timing reused 4 experts
  that stayed in the CPU's cache, so it took the CPU for faster than it is during decode. It now goes through up to
  64. That gave +6% decode on a Ryzen 9 7950X with a Tesla V100, and the same plan at every start.
- **Maya's name in the logs** (#74):
  - the engine's lines say `maya generate:` / `maya serve:` / `maya:`, and the server's say `[maya]`;
  - the installer no longer shows Strata's name;
  - the Windows stall dump is `maya-stall-<pid>.dmp`.

  Settings (`STRATA_*`), flags, the API, file and binary names are unchanged, and the README keeps the credit.
- **A restart no longer mistakes the dashboard's saved settings for a model's config** (#69). After a settings
  change, `./maya.sh` / `START-MAYA.bat` could pick `maya-<model>.shared-settings.json` and fail.
- **The README's Windows setup, step by step** (#71 by @handmade0octopus):
  - the Visual Studio 2022 Build Tools components, and the difference between `nvidia-smi`'s CUDA version and an
    installed toolkit;
  - copy-paste commands for checking the PC, setting up on an NVMe drive and calibrating;
  - page-file advice that matches what `--check` asks for (about VRAM + 9 GB, since "System managed" can stay too
    small).
- The README has no speed figure under 12 tokens/s.
- Checked:
  - on 1x and 2x Tesla V100: the builds and the parity tests; identical greedy tokens with the CPU lane off, on one
    GPU and on two; decode and prefill (reading the prompt) the same within run-to-run variation (#70 picks a
    different CPU-lane plan at some starts); the CPU thread count changed per request, as `--calibrate` does; the
    server end to end; the setup screen in a real terminal; the logs saying `[maya]` and `maya ...`;
  - on Windows: the engine compiles (MSVC with CUDA 12.8);
  - AMD on Windows: on the author's Ryzen AI Max+ PRO 495;
  - the GitHub checks.

## v1.0.29 - 2026-10-09

Documentation: the README and v1.0.28's entry below describe the model files' new architecture name more plainly.

## v1.0.28 - 2026-10-09

The model files name their architecture with the standard GGUF name, `glm5-next`.

- **The model files say `glm5-next`.** Maya's quants had copied `glm5next`, an early spelling, from the file they
  were made from. Standard GGUF tools don't recognize that spelling.
  - On Hugging Face, the first file of each model is published again with only its header changed. Every byte of
    the model stays where it was, so installed models and their packs keep working, and nothing has to be
    downloaded again. Setup accepts both versions of that file.
  - `tools/gguf_fix_arch.py` renames a file downloaded earlier, in place (only the header).
  - The quantizer writes `glm5-next` from now on, and Maya reads either name.
- Checked:
  - Maya's greedy tokens are the same with the new header (Tesla V100, CPU lane off);
  - the four new files are the old ones byte for byte after the header;
  - the GitHub checks.

## v1.0.27 - 2026-10-09

The setup and Maya get a screen of their own in the terminal, the crash after a long prompt is fixed, a stuck prompt
ends with an error instead of waiting forever, and Maya builds on Windows again.

- **A screen of its own** (#58 by @needmorevram): in a terminal, the setup's 8 steps are tabs, the questions are
  arrow-key menus, and the compile and the download show their progress. Maya then runs on the same screen, with
  a loading view while the experts warm up, a dashboard of the Monitor's numbers and its log.
  - `./setup.sh`, `./maya.sh`, `START-MAYA.bat` and `run-maya-<model>.sh` all open it.
  - Its library (Textual) ships with Maya as 9 pure-Python wheels in `third_party/wheels`, byte-identical to
    PyPI's, and goes into `.venv` with nothing downloaded.
  - `--plain`, a pipe or a service keeps the plain text exactly as before, and `--yes` keeps a plain setup.
- **No more "illegal memory access" after a long prompt, and no more 0xC0000005 on Windows / AMD** (#39 by
  @boxwrench). Three races in the expert tiers, shared by NVIDIA and AMD:
  - the pinned buffers of the expert tables' updates were rewritten while their copy to the GPU was still in
    flight;
  - a background move of an expert could land in a VRAM slot that a prompt was borrowing at the same moment;
  - a route of NaN scores indexed the expert tables. It now fails that request with an error, and the engine
    goes on.
  - Confirmed on an RTX 3090 + 3060 with every RAM-tier expert on the CPU (#50: the 8K prompt after decoding
    crashed, now it runs), and on Windows with an RX 7900 XTX (#53: 5 crashes in 8 sessions, now 0 in 8).
- **A stuck prompt ends** (#40): the engine's watchdog now watches Maya's GLM engine too. A request that finishes
  no prompt layer and no token for 3 minutes ends the engine with an error that names where it stopped (on Windows
  also `strata-stall-<pid>.dmp`, every thread's stack), and the next request starts it again. Before, a prompt that
  stalled with the GPU idle waited forever. `STRATA_WATCHDOG_S` sets the time; `0` turns it off.
- **Windows builds again** (#54 by @jerem91150, the same fix as #55 by @noahark): `--kv int8` called `setenv`,
  which Windows' C library doesn't have.
- **Thinking off stays off** (#54): GLM-5.3's template opens a thinking block even when thinking is off, so the
  answer could land inside it. The server now closes it. This applies to every GPU.
- **AMD**:
  - Windows: `tools\hip\build_maya_windows.bat` builds the engine with TheRock's ROCm wheels (#54).
  - RDNA4 prompts +8-10% (#38 by @boxwrench): a wave32 WMMA attention kernel, the default on R9700 / RX 9070.
  - `STRATA_GLM_DISK_QD` goes up to 32 reads in flight (#54). The default stays 2.
- **One GPU skips the last layer's unused outputs in a prompt** (#61 by @boxwrench): +1% prefill (reading the
  prompt), the same output. `STRATA_GLM_PREFILL_TAIL_SKIP=0` turns it off.
- **Robustness** (#62 by @needmorevram):
  - The embedding rows of F16, BF16, Q2_K, Q4_1 and Q5_1 GGUFs are found (every token read row 0 before). The
    published Maya models are unaffected.
  - A prompt whose staging events cannot be created runs token by token instead of failing.
  - Damaged image-embedding files and vision measurements are refused instead of ending the engine or the
    server.
- **Switch chats while an answer is written** (#66 by @needmorevram): another chat, or a new one, opens; the
  answer goes on in its own chat and shows again when that chat is opened. Esc stops it only in its own chat.
- **A setup for another context keeps your tuning** (#57): it used the engine's defaults until `--calibrate` ran
  again. It now takes the settings tuned at the nearest context.
- Checked on 1x and 2x Tesla V100, and on Windows:
  - the builds and the parity tests, and the engine's Windows build (MSVC with CUDA 12.8);
  - identical greedy tokens with the CPU lane off, on one GPU and on two, and with the KDA graphs on;
  - #62's and #59's tests;
  - #50's sequence (every RAM-tier expert on the CPU, answers, then 2K and 8K prompts) on one GPU and on two;
  - the watchdog: a forced stall ends with an error that names its stage, and the next request is answered. A
    10-second watchdog never fired over long prompts and decode;
  - decode and prefill (reading the prompt) the same at the defaults, prefill +1% on one GPU (#61);
  - the server end to end: answers, a picture, thinking, context reloads, a graceful stop;
  - #58's screen in a real terminal, with its tests on Linux and Windows, and #66 in a browser against the
    running model;
  - the GitHub checks.

## v1.0.26 - 2026-10-09

Documentation.

- **A speed report in the README** (#63 by @npc97): Maya-L on an RTX 4090 24 GB with 192 GB of DDR5 under
  Windows 11. It holds every expert in VRAM or RAM and decodes at 16.4 tokens/s, with prefill at 1371 tokens/s
  on an 8K-token prompt. The Windows section no longer says Maya has not run on a Windows PC with an NVIDIA GPU,
  and notes that CUDA 13 works for RTX 20 and newer.
- **`STRATA_GLM_CPU_PIN` is Linux-only** (#64 by @needmorevram): on Windows the engine pins nothing, and one GPU is
  never pinned anywhere, so `0` changes nothing there. A two-socket Windows PC whose decode differs between starts
  differs in something else; compare the starts' `CPU lane:` log lines.

## v1.0.25 - 2026-10-09

Splits of three GPUs or more use all of the CPU; `--calibrate` measures realistic text and leaves your usage
profile alone; opt-in CUDA graphs for one GPU.

- **One CPU pool for a split of 3+ GPUs** (#60 by @needmorevram):
  - A decode token visits the cards in turn, so a pool per card (`cores / GPUs` threads each) left most of the
    CPU idle. One pool now serves them all, sized as one GPU's.
  - Maya-L on 9 GPUs (8x RTX 5060 Ti + RTX 3090, 2x Xeon Gold 6152): decode 25.7 -> 30.7 tokens/s, an 8K prompt
    1265 -> 1466 tokens/s.
  - One and two GPUs are unchanged. `STRATA_GLM_CPU_SHARED=1` shares the pool on two GPUs too; on 2x Tesla V100
    with one CPU socket it measured 30.0 against 28.5 tokens/s. `0` gives every GPU its own pool.
- **`--calibrate` on realistic text** (#60):
  - Every measurement answers prompts it has not seen before (60 of them). Repeating three prompts had left the
    experts in VRAM fitted to just those answers, so the CPU lane's settings were measured with almost nothing
    on the CPU.
  - The tuning's engine reads and writes a copy of your expert usage file, so its answers no longer change the
    order the next start warms up in.
- **Opt-in CUDA graphs for one GPU** (#59 by @handmade0octopus): `STRATA_GLM_KDA_GRAPH=1` replays the recurrent
  (KDA) decode layers as CUDA graphs. The output is bit-identical.
  - RTX 4090D: +2.35% decode.
  - Tesla V100: the same speed.
  - Off by default; two GPUs, MTP, profiling, prefetch and lookahead keep the direct path.
- Checked on 1x and 2x Tesla V100:
  - the builds and the parity tests;
  - identical greedy tokens with each change at its default, and with the graphs on (CPU lane on and off);
  - #59's two tests: 192 bitwise comparisons on the real model;
  - decode and prefill unchanged at the defaults;
  - a full `--calibrate` run (380 s) that left the usage file byte-for-byte unchanged;
  - the GitHub checks.

## v1.0.24 - 2026-10-09

Better drafts for speculative decode on two GPUs or more.

- **Better drafts** (#48 by @sociolog): the MTP draft block used to skip every expert it found missing from
  VRAM. It now computes the missing ones among its route's 2 best-scored experts and skips the rest. It is the
  default, set with `STRATA_GLM_MTP_KEEP=<n>`: `0` restores the old behaviour, `STRATA_GLM_MTP_MISS=1` fetches
  every missing expert.
  - On 2x Tesla V100: 82-83% of drafts accepted against 76-81%, decode at 128K 28.0-28.2 against
    27.0-28.1 tokens/s, and never slower.
  - On an RTX 3090 + 3060 split: 19.5 -> 20.6 tokens/s.
  - The output stays the model's own: greedy tokens are identical with the CPU lane off.
- Checked on 2x V100: the build, the tokens above, five draft settings over two rounds, and the GitHub checks.

## v1.0.23 - 2026-10-09

- **The AMD build can no longer break unseen.** `tools/check_hip_compat.py` lists every CUDA runtime and cuBLAS
  name the AMD (HIP) build compiles, and fails on one without a mapping in `include/strata/hip_compat/`. It
  flags v1.0.21's three.
- **Checks on every push and pull request:** a GitHub Actions workflow runs that check, the server tests and the
  installer and tools tests. No GPU is needed.
- **Every AMD target builds:** the debug tool `peer_probe` gets its three peer-copy mappings. The installer never
  built it.

## v1.0.22 - 2026-10-09

- **AMD builds again (#52 by @boxwrench):** v1.0.21 did not compile with HIP: #44's split search and CPU pinning use
  three device queries (`cudaDeviceGetPCIBusId`, the memory clock and the memory bus width) that had no HIP mapping.
  NVIDIA builds are unchanged. Built for gfx1100, gfx1201 and gfx1151 (ROCm 7.2.1) and run end to end on a Strix Halo.

## v1.0.21 - 2026-10-09

Eight community pull requests, with every default fitted to the machine it runs on:

- a split search and prompt chunks sized from the expert pool;
- a smaller KV cache at long context (decode 6-8% faster at 128K);
- pictures from the web app read again;
- a loop guard, and a thinking budget that leaves room for the answer;
- speculative decode on three GPUs or more;
- CPUs without AVX2;
- conversations that outlive a restart.

- **The layer split and prompt chunks (#44 by @needmorevram):**
  - `--layer-split auto` prices every placement on 2-4 GPUs before loading (each card's room for experts, the share
    of routes they cover, its memory bandwidth, the host's RAM speed) and takes the fastest.
  - `--prefill auto | N | 0` sizes the prompt chunk from what each card's expert pool can lend: up to 32768 tokens on
    one GPU and 8192 on a split, each the faster there. One V100 reads 26K-token prompts at 562 tok/s; two V100s at
    719.
  - On hosts with two NUMA nodes or more, each GPU's CPU lane runs on CPUs of its own (2x Xeon Gold 6152: decode
    15.4 -> 19.8 tok/s). On one node it stays off, which measured faster there. `STRATA_GLM_CPU_PIN=1/0` forces it.
  - The RAM tier takes huge pages only when free 2 MB blocks cover it, so a second card's tier no longer waits over
    10 minutes in kernel compaction.
- **A smaller KV cache (#41 by @merbanan):**
  - The attention indexer keeps its keys for the last 8,256 positions instead of the whole context, with the same
    output. At 128K that is 1.28 GB more VRAM for experts. Decode: one V100 with Maya-S24 18.2 -> 19.4 tok/s; two
    V100s with Maya-S 26.3 -> 28.2.
  - `STRATA_GLM_KV_INT8=1` (opt-in) stores the attention latents in INT8: another 0.65 GB, 19.6 tok/s on the same V100.
    Its KL divergence from the FP8 model is the same as without it (0.436 against 0.431).
  - Conversation slots saved by earlier versions are read again once.
- **Pictures from the web app (#42 by @tanutanu56):** the engine received them without their image data, so the model
  answered about a black picture. Fixed.
- **Loop guard (#33 by @ksanislo):** a reply that repeats one short pattern for 256 tokens while thinking (1024 while
  answering) is closed: the thinking ends, a tool call's argument ends, or else the turn. `STRATA_GLM_LOOP_THINK` /
  `_ANSWER` set the lengths; `0` = off.
- **Thinking budget (#34 by @ksanislo):** the budget leaves the answer at least a quarter of the request's `max_tokens`
  (and at least 1024 tokens), so a small `max_tokens` no longer ends in thinking with no answer.
- **Speculative decode on three GPUs or more (#31 by @ksanislo):** the parts split into a head and a tail group (4x
  Tesla T4: decode 15.0 -> 22.6 tok/s). Two GPUs: the same as before.
- **CPUs without AVX2 (#32 by @ksanislo):** the engine runs there; its CPU lane uses ggml's kernels. The setup warns
  instead of refusing.
- **Conversations that outlive a restart (#35 by @ksanislo):** `STRATA_GLM_SLOT_KEEP=1` (opt-in) keeps the saved
  conversations across starts. A stop (SIGTERM, Ctrl+C) lets the running answer finish its thinking and answer first.
- **Reloads wait for the GPUs:** a context reload, or a restart after a crash, starts the new engine only once the GPUs
  have freed the old one's memory. The driver releases it a few seconds after the process ends, and the split search
  measures the cards before anything loads. The server waits until no GPU lists the old engine (NVML), or 5 s where it
  can't tell.
- **INT8 shown as such:** with `STRATA_GLM_KV_INT8=1`, the engine reports `kv=int8` and About says "8-bit (INT8)
  attention cache".
- **Housekeeping:**
  - No third-party quant names in the tree.
  - `setup.py` keeps only the helpers Maya's installer uses.
  - The start line names the thinking level GLM gets.
- Checked on 1x and 2x Tesla V100:
  - the 239 Python tests and the CUDA parity tests;
  - greedy tokens, identical with and without the new KV cache;
  - prompts, decode, and KL against FP8;
  - the server end to end: an exact answer, a picture through the web app's path, a small `max_tokens` with
    thinking, the context reload 128K -> 16K -> 128K, and a graceful stop.

## v1.0.20 - 2026-10-09

- Housekeeping: Maya-S24's files on Hugging Face are labelled IQ2_XXS_S, apart from Maya-S; the files themselves are
  unchanged. A Maya-S24 already downloaded keeps its names; older setups need this update to download it.

## v1.0.19 - 2026-10-09

- Housekeeping: Maya-M's and Maya-L's files on Hugging Face are named with their quant label (IQ2_S, IQ3_S); the files
  themselves are unchanged. Models already downloaded keep their names; older setups need this update to download
  them.

## v1.0.18 - 2026-10-09

Maya-L, the closest quant yet to the full FP8 model; GLM's own thinking levels; an exact context meter; and an Update
button in the dashboard.

- **Maya-L** (156.3 GB, `./setup.sh --setup --model Maya-L`): Maya-M's recipe one step up - IQ3_S gate/up experts,
  IQ4_XS down projections, Q5_K in the most sensitive MoE layers, Q6_K attention and shared experts. It keeps 99.2% of
  the FP8 model's zero-shot accuracy (the same score on HellaSwag and PIQA), its KL divergence is 35% below Maya-M's on
  the same engine (0.188 vs 0.291), it picks the FP8 model's next token 90% of the time, and it wrote two
  14,000-token answers without a loop. The most demanding of the four: fastest when VRAM and RAM hold most of its
  156 GB. It is the setup's fourth download, in place of the 3.5-bit community file offered before.
  ([bench/results/MAYA-L.md](bench/results/MAYA-L.md))
- **Thinking levels are GLM's own:** Off / Low / High (the default) / Max in the dashboard, `none` / `low` / `high` /
  `max` in the API (OpenAI's `medium` is High and `xhigh` Max; Anthropic budgets under 2K tokens Low, under 8K High,
  else Max). The page's old "High" asked GLM for Max, its longest thinking; settings saved by the old page are
  translated once.
- **The context meter counts exactly:** while it answers, the request's prompt plus the tokens written so far (the
  server's live count); otherwise the chat and the draft through the model's own template and tokenizer
  (`POST /api/tokens`, nothing run; approximate only with pictures). Both numbers in the same K.
- **Updates from the dashboard:** About says when a new release is out, and its **Update** button downloads the new
  code (git), compiles only the engine files that changed and loads the same model with the same settings - the
  model is not downloaded again. A folder that can't update itself (not a git checkout, files changed by hand, a
  server not started by `./setup.sh` or `START-MAYA.bat`) says why and gives the steps by hand. The check asks GitHub
  for the latest release at most every six hours; `MAYA_UPDATE_CHECK=0` turns it off. (`GET`/`POST /api/update`)
- **Monitor shows each GPU figure once:** with two to four GPUs, every hardware card lists each GPU's own value under
  the total (the GPUs table repeated them); from five GPUs the table comes back and the cards keep their totals.
- **Fixed:** saving the Chat settings for other apps failed with a Min-p above 0 (the Creative preset), so nothing
  was shared.
- Checked: Maya-L against the FP8 model on one Tesla V100 (KL, zero-shot, loop test); the thinking levels rendered
  with GLM's own template; the update against a throwaway git origin (a real fast-forward, then the exit maya.py
  starts the new version on) and every case that blocks it; the dashboard in the browser at desktop and phone widths;
  the Python tests (82, 16 of them new) pass.

## v1.0.17 - 2026-10-09

A new dashboard: many chats, the context size from the settings (the model reloads with it), editing and regenerating,
code previews, and a Monitor and About that say what the server runs.

- **Chats:** every chat is kept in the browser (IndexedDB, with its pictures and attached files) and listed in the
  sidebar with search, rename and delete; "New chat" no longer wipes the last one. On a phone the list slides in from
  the left. A user message can be edited and sent again (the answers after it are replaced), the last answer
  regenerated, and an answer that was streaming when the page reloaded is taken back up from the server.
- **Context size in Settings:** a slider from 4K to the model's trained 1M tokens. It shows what the size costs here,
  from the engine's own measurement (2x V100: 3.3 GB at 128K; "about 249 more experts in VRAM" at 64K), warns past
  256K (untested) and asks before it reloads the model (about a minute on 2x V100). Requests from other apps get a
  503 with Retry-After meanwhile; a size that does not start puts the old one back; the new size is saved in the run
  config. (`GET`/`POST /api/context`, from the page itself only.)
- **Instructions per chat** (a system prompt, optionally for new chats too), sampling presets (Precise, Balanced,
  Creative), min-p, and a guard against closing Settings with unsaved changes.
- **While it answers:** a context meter for the chat, the prompt it read and reused under each answer, a "Latest"
  button, Esc to stop, Ctrl+Shift+O for a new chat.
- **Code:** syntax highlighting (bundled, no CDN), wrap, download, and a sandboxed preview for HTML pages.
- **Monitor:** a reading from the last answer is marked as such (it looked live), idle charts say so instead of a flat
  line, a GPUs table on multi-GPU PCs, the request's source (Chat / OpenAI / Anthropic) and prompt speed per row,
  banners for a reload or a hot GPU, tips from the expert tiers, and **Copy report** (versions, PC, engine settings,
  the engine log's telling lines; no API key).
- **About:** the version (it was missing), the trained context, the KV cache as it is (16-bit; it showed "32-bit"),
  each GPU's PCIe link, the model folder and its free disk, and copy-ready snippets for Claude Code, OpenAI-style
  apps and curl.
- **Phones and access:** no zoom into the message box on iOS, 44 px buttons, safe areas, a status word in the header,
  installable as an app (manifest); focus stays in Settings and dialogs, the arrow keys move the thinking level, and
  reduced motion is honoured. Removed: the Qwen-only speed-projection switch.
- Checked: on 2x Tesla V100 with Maya-S, the context changed 128K -> 96K -> 128K through the page's endpoint (about a
  minute each, chat requests answered 503 meanwhile, the run config's other keys unchanged), then answered exactly;
  the page tried in the browser at desktop and phone widths; the server's tests (102 + 9 new) pass.

## v1.0.16 - 2026-10-08

README: the users' speed table keeps the measurements on current versions.

- The 2x TITAN RTX row (v1.0.6, before the prefill and CPU-lane fixes since) is out of the users' table; the GPU
  requirements list 2x CMP 170HX among the users' machines instead.
- The RAM requirement names the machine it refers to (the 2x V100 in the speed table, 30 GB of RAM).

## v1.0.15 - 2026-10-08

Numbers in long prompts are read correctly, and Maya is faster on consumer GPUs: it measures the PCIe link the way
decode uses it and stops re-reading experts from the SSD that are already on their way to RAM; Maya-S24 for 24 GB
cards; Strix Halo; an idle engine no longer holds a CPU core per GPU.

- **Numbers in prompts are read the way GLM reads them (#27, found by @mab776):** the server split every number in a
  prompt into single digits (Qwen's pre-tokenizer) where GLM groups up to three (`504` is `50|4`, not `5|0|4`), so
  in long prompts numbers came back wrong (`504` -> `5504` from ~12K tokens on, every quant). The server now uses the
  pre-tokenizer the model's tokenizer names; numbers in a 32K and a 128K prompt came back exact (6 of 6, measured by
  @mab776). Nothing to redo: the setup already stored it.
- **Maya-S24 (94.7 GB), for cards of 24 GB or less:** Maya-S with its attention and shared experts in 4-bit (Q4_K)
  instead of 6-bit, so about 1.5 GB more of the card holds experts. One Tesla V100 limited to 24 GB: decode (writing
  the answer) 11.8 -> 13.5 tokens/s (+14%); on the full 32 GB 17.2 -> 19.1 (+11%); prefill the same. It keeps 97.7%
  of the FP8 model's zero-shot accuracy (Maya-S: 97.9%). The setup recommends it when every card has 24 GB or less;
  `--model Maya-S24` picks it anywhere. Details: bench/results/MAYA-S24.md.

- **The CPU / PCIe split on consumer GPUs (#24 by @tanutanu56, found in #13):** at start the engine times expert
  copies over PCIe to decide how many RAM-tier experts the CPU computes. It timed them on an idle GPU, and GeForce
  cards hold the link at Gen1 when idle, so the copies measured a link 3-8x slower than decode sees and the CPU took
  experts PCIe moves faster. The timing now runs with the GPU kept busy. (Tesla cards like the V100 keep the link up:
  unchanged there.)
- **Experts leaving VRAM are no longer read again from the SSD (#24):** an expert moved down to RAM at the end of one
  token, and needed again before its copy had landed, was read from the SSD; it now waits for the copy and reads RAM.
  2x Tesla V100 with Maya-S: decode (writing the answer) 28.4 / 28.9 -> 29.8 / 29.7 tokens/s, SSD reads per token
  3.0 -> 2.2.
- **CPU-lane threads per GPU (#24):** `STRATA_GLM_CPU_LANE<n>` sets GPU n's own (a card in a narrow slot wants more
  than one on a wide link). @tanutanu56's RTX 4070 Ti SUPER (PCIe 3.0 x4) + 5070 Ti, Maya-S: decode 8.1 -> 9.1
  tokens/s with the defaults, 14.6 with the threads and the split tuned for that pair.
- **An idle engine no longer holds a CPU core per GPU:** its tier threads spun waiting for work; they now sleep when
  nothing runs (2x V100: 200% -> 2% of a core idle, one V100: 100% -> 1%). Decode and prefill (reading the prompt)
  are unchanged (`STRATA_GLM_SERVICE_IDLE_MS`; 0 = spin always).
- **Sampled answers (temperature > 0)** draw their token on the engine's own GPU stream instead of syncing the whole
  GPU every token (which also waited for the tier copies in flight): 2x V100, temperature 1.0: 26.7 / 26.6 ->
  26.8 / 27.6 tokens/s. On Windows with an AMD GPU this also fixes answers that lost the prompt after their first
  token (#6, found by jerem91150).
- **The MTP draft block from a file:** `STRATA_GLM_MTP_GGUF=<file>` now also replaces a model's own draft block, and
  `tools/maya_quant/mtp_gguf.py` writes a model's draft block alone as a small GGUF - for quants published without
  one, on two GPUs.
- **RAM-resident tier (#25 by @handmade0octopus, opt-in):** `STRATA_GLM_RAM_RESIDENT=1` sizes the pinned RAM tier to
  hold every expert that isn't in VRAM, so nothing is ever evicted to the SSD (the start refuses if RAM can't hold
  them). On an RTX 4090 D with 128 GB RAM, Maya-S at 256K context, a 38K-token prompt: decode 11.8 -> 22.1 tokens/s
  (with #26's `STRATA_GLM_PROMOTE_MIN=6`), no SSD reads.
- **Fewer one-off promotions (#26 by @handmade0octopus, opt-in):** `STRATA_GLM_PROMOTE_MIN=N` keeps a fetched expert
  in VRAM only when it has been routed at least N times lately, so an expert used once no longer evicts a resident
  one. 4090 D: 49.8 -> 40.9 ms a token warm with N=6. On one AMD R9700 (measured by @boxwrench): decode 23.4 ->
  24.5 tokens/s with N=6, 25.1 with the RAM-resident tier too (no SSD reads), 25.55 with #24 as well (+9%).
- **AMD (experimental, by @boxwrench):** faster decode expert kernels on RDNA3 / RDNA3.5 (#19: RX 7900 XT +8.6%, the
  same tokens), the attention's prompt products in FP16 with a matrix-core attention kernel (#16: Strix Halo +16%
  prefill, RX 7900 XT +4% on 4K-token prompts), and `--bench` / `--report` on AMD (#18; `--config FILE` picks one
  installed setup). #24's PCIe timing runs with the GPU awake on AMD too.
- **Strix Halo (#17 by @boxwrench, experimental):** the Ryzen AI Max+ 395's GPU (gfx1151) shares the PC's memory, so
  the engine sizes its expert pool from the free RAM (MemAvailable less 16 GB) instead of the GPU's reported share.
  When the pool holds every expert it keeps them all and the RAM tier only stages reads, instead of holding a second
  copy. The setup finds a versioned ROCm (`/opt/rocm-X.Y.Z`); an APU runs as one GPU. Measured on a Strix Halo with
  Maya-S: prefill (reading the prompt) ~221 tokens/s on 3.5K-token prompts, decode (writing the answer) 17.4
  tokens/s. A HIP build configured by hand without `-DSTRATA_PREFILL_MMQ=ON` now stops at configure and says so,
  instead of failing to link.
- **Windows: the page file the RAM tier needs (#20, reported by @anbow1):** under Windows the card's memory and the
  pinned RAM tier are both charged to RAM + page file, so a small page file capped the RAM tier while RAM sat free
  (an RTX 5090 with 128 GB RAM: 2.4-4.1 tokens/s with an 8 GB page file, 8.4-11 with a 64 GB one). `--check` now
  asks for at least your VRAM + 8 GB and says what a smaller one costs, and the engine warns at start when the
  commit limit, not the free RAM, caps its RAM tier. (Built and run on Linux here; the Windows part is not compiled
  on Windows yet.)
- **API (from Strata 0.1.40):** the client's stop strings end the answer (OpenAI `stop`, Anthropic `stop_sequences`
  with `stop_reason: stop_sequence`; not inside the reasoning); `tool_choice` works - `none` offers no tools, and
  `required` or a named function (Anthropic `any` / `tool`) starts the answer with the call; `response_format:
  json_object` returns the JSON alone; an empty assistant turn is no longer put back into the prompt. The engine's
  start-up warnings show in the server window.
- **Activations can't turn into NaN in the quantizers (from Strata #1448):** the FP16 scale of an activation block
  overflowed to infinity past ~8.3 million; it is clamped at all four places it is written. Below that the output
  is bit-identical.
- **CONTRIBUTING.md:** what makes a useful speed report, bug report or pull request (#21). The README's table of
  speeds measured by users has 2x CMP 170HX with Maya-M at 61.3 tokens/s (#22, @ZackO2o).
- Checked on 2x Tesla V100: the engine's parity tests pass, the prompt attention matches F32 (1.5e-4), the server's
  102 tests and the 40 setup / tokenizer tests pass, tool calls work end to end; decode and prefill unchanged or faster. NVIDIA code is unchanged by the
  AMD pull requests.

## v1.0.14 - 2026-10-08

Maya runs on two AMD GPUs (experimental), with MTP drafting on the second card, contributed by @boxwrench.

- **Two AMD GPUs (#14 by @boxwrench):** `./maya.sh --backend hip --gpus 0,1` splits the layers across two supported
  cards (RX 7900 XT / XTX, Radeon AI PRO R9700 / RX 9070; they may be different models), and the second card runs
  the MTP block that drafts the next token. Setup puts the card with more VRAM first and writes one hipBLASLt tuning
  table per card's architecture (`STRATA_HIPBLASLT_TUNING` takes a `:`-separated list). Measured by @boxwrench with
  Maya-S on an R9700 + RX 7900 XT: decode (writing the answer) about 33 tokens/s against about 15 on the RX 7900 XT
  alone, prefill (reading the prompt) about 490 tokens/s on a 4K-token prompt against about 414. docs/AMD_MAYA.md has
  the details.
- NVIDIA: unchanged - the change is in the AMD setup and the AMD-only prompt GEMM code; every NVIDIA target builds.
- We have no AMD hardware; these results are the contributor's.

## v1.0.13 - 2026-10-08

Tool calls work: GLM-5.3-Flash's function calls reach your apps and coding agents instead of ending the request.

- **Tool calls (issue #5, reported by @xldistance):** GLM-5.x writes its calls in its own form
  (`<tool_call>NAME<arg_key>...</arg_key><arg_value>...</arg_value></tool_call>`), and the server read only another
  model family's form, so every tool call ended the request with "malformed tool call" - in the chat's tools, MCP,
  and every OpenAI or Anthropic client (Claude Code, agents). Both forms are read now, whole and streamed (the call's
  name as soon as it is written, then its arguments piece by piece), and checked end to end: the call, its result
  in the next turn, both APIs.
- **A call only quoted in the answer stays text:** a `<tool_call>` inside a code block or inline code (an example the
  model shows) is never run - an example `rm -rf` must not reach your shell (from Strata #1058).
- **A stuck engine is restarted:** an engine that says nothing for 90 s and in that time uses no CPU, disk or GPU is
  ended, the request gets an error, and the next request starts it again; a silent engine that is working (a long
  prompt on a slow PC) is never ended (`STRATA_ENGINE_STALL_S`, `0` = off; from Strata #1317).
- **The server, from Strata 0.1.41:** up to 256 connections wait to be accepted (an agent opening many at once got
  "connection reset"; `STRATA_HTTP_BACKLOG`); a request body is checked before it is read (a bad Content-Length is a
  400, one over 256 MiB a 413 - `STRATA_MAX_BODY_MIB`), relays' chunked bodies are read, and a malformed request is
  a 400 with its traceback in the server log instead of a dropped connection; `--api-key` takes several keys
  (`KEY1,KEY2`, or a list in the config).
- `--env` now also reaches these server settings (the rest still go to the engine).
- Checked: 92 server tests (12 new for tool calls, 10 for the rest); the engine is unchanged from v1.0.12.

## v1.0.12 - 2026-10-08

Maya runs on up to 16 GPUs and reads long prompts much faster on one GPU, contributed by @needmorevram.

- **Up to 16 GPUs (#12 by @needmorevram):** `--gpus 0,1,2,...` takes up to 16 GPUs (two before). Two GPUs split the
  layers in the middle as before; with more, each GPU takes a share sized to its free VRAM (`STRATA_GLM_SPLIT` or the
  config's `"layer_split"` pin it). Measured by @needmorevram on nine GPUs (8x RTX 5060 Ti 16 GB + an RTX 3090):
  decode (writing the answer) about 28 tokens/s, prefill (reading the prompt) 1,150-1,210 tokens/s on 8K-token
  prompts.
- **Prefill on one GPU:** bigger prompt chunks (up to 32,768 tokens, sized from the free VRAM); each expert's output
  added in as soon as it's computed, so a chunk holds about twice the tokens; the least-used RAM-tier experts computed
  on the CPU while the GPU loads the rest; experts copied to the GPU while the attention runs; faster sparse-attention
  scoring; a tensor-core attention kernel for Ampere and newer. 1x Tesla V100 with Maya-S: 2K / 8K / 16K-token prompts
  267 / 373 / 362 -> 263 / 578 / 620 tokens/s. 2x V100: 293 / 485 / 553 -> 313 / 541 / 674 tokens/s.
- **Decode with most experts in RAM:** hot RAM-tier experts move into VRAM in the background while it answers
  (`STRATA_GLM_PROMOTE`), and on two-socket machines the RAM tier is spread over both sockets' memory. On
  @needmorevram's RTX 3090: 15.3 -> 18.7-19.9 tokens/s. 1x and 2x V100 with Maya-S: unchanged
  (17.1 and about 27.6 tokens/s).
- **`./maya.sh --calibrate`:** measures decode with a few CPU-lane settings (how many RAM-tier experts go over PCIe
  instead of to the CPU, and how many CPU threads) and keeps one only if it is more than 3% faster (README > Tuning).
- **`--models-dir`** names the folder downloaded models go to (`--data-dir` before; older configs still work).
- **Our follow-ups to #12:**
  - Builds for Volta / Turing (V100, RTX 20) and Windows again: the new attention kernel compiles only for Ampere and
    newer, the page-cache drop is Linux-only, and the AVX-512 kernel converts FP16 in a way MSVC accepts.
  - One GPU beside a 6-core CPU kept its decode speed: the CPU's share of each token is cut into about 48 pieces in
    all (`STRATA_GLM_CPU_SPLIT`). One piece per thread, as submitted, took 1x V100 from 17.0 to 14.3 tokens/s; now
    17.1. Machines with 40 or more threads still stream one piece each.
  - The tokenizer remembers what it has read: a 220K-token agent prompt is tokenized in 0.23 s instead of 0.89 s, and
    in 0.013 s when it is sent again with a new turn (the same token ids).
- **Checked:** the engine's parity tests pass on 1x and 2x V100. On held-out text the model's loss stays within
  v1.0.11's run-to-run noise; on one GPU it now varies a little more from run to run.

## v1.0.11 - 2026-10-08

Maya runs on AMD GPUs (experimental): RX 7900 XT / XTX and Radeon AI PRO R9700 / RX 9070 on Linux, contributed by
@boxwrench.

- **AMD (experimental, #7 by @boxwrench):** `./maya.sh --backend hip` builds Maya's engine with ROCm 7 for RX 7900
  XT / XTX (gfx1100) and Radeon AI PRO R9700 / RX 9070 (gfx1201) - Linux, one GPU, text only. Measured by
  @boxwrench with Maya-S: on the R9700, decode (writing the answer) about 20 tokens/s and prefill (reading the
  prompt) up to 490-560 tokens/s; on the RX 7900 XT, prefill up to about 410 tokens/s. Setup and measurements:
  docs/AMD_MAYA.md.
- **Prefill on AMD:** the prompt projections go through hipBLASLt with tuning tables for each card, and the prompt
  runs in bigger sub-batches on cards with room (`STRATA_GLM_PREFILL_SUB`; NVIDIA keeps 256).
- The groundwork, the GPU-to-CPU signal fix, came in v1.0.7 (#2).
- NVIDIA: unchanged - the same tokens as v1.0.10 on 2x Tesla V100, every target builds.
- We have no AMD hardware ourselves; these results are the contributor's. Tell us how it runs on yours (`--bench`
  and `--report` learn AMD in a follow-up).

## v1.0.10 - 2026-10-08

RTX 20-series cards now read prompts on their tensor cores too.

- **Prefill on Turing (RTX 20-series, Titan RTX, Quadro RTX):** the tensor-core prompt attention now keeps its context
  in registers on these cards, so it fits their 64 KB of shared memory. They had been using the slower F32 kernel.
  2x TITAN RTX, an 18,476-token prompt: 62 s -> 58 s of prefill (reading the prompt). By @dummerjindabin (#10).
  Other cards keep the kernel they had; that variant's arithmetic is only correct on Turing and newer (a V100 got the
  attention wrong with it), so it is used only where the other one doesn't fit. A V100 gives the same tokens as
  before.
- **Prefill for 4-bit (Q4_K) weights:** they are converted for the tensor cores the fast way, as 5- and 6-bit ones
  already were - the same values. A test quant with 4-bit attention, 1x V100: 224 / 291 / 285 ->
  268 / 368 / 358 tokens/s at 2k / 8k / 16k, the same tokens. Maya-S and Maya-M are unchanged.

## v1.0.9 - 2026-10-08

Switching between conversations no longer re-reads them: Maya keeps the last few on the SSD.

- **Conversation slots:** when a request doesn't continue the conversation the engine holds - another chat, or an
  agent tool's side request (a title, a sub-task) - that conversation is first set aside in a file on the SSD (its
  attention caches and recurrent state, about 0.15 GB plus 21 KB a token: 0.2 GB at 3K tokens, 0.8 GB at 30K), and a
  later request that continues it takes it back instead of reading its whole prompt again. On 1x Tesla V100 with
  Maya-S, going back to a 3,500-token conversation after another one: prefill (reading the prompt) 10.2 s -> 2.2 s;
  setting it aside 0.09 s, taking it back 0.06 s. A conversation taken back from its file continues token for token as
  it would from memory, and each one still recalls its own details after others ran in between (one GPU, and the
  two-GPU split with the MTP block). Requests still run one at a time. Up to 4 conversations of 1,024 tokens or more, 16 GB on disk, never
  below 8 GB free; `STRATA_GLM_SLOTS=0` turns it off (README > Tuning).
- **Prefill:** the disk read-ahead buffer grows to 3% of free RAM, up to 96 experts (was 2%, up to 64): on two GPUs
  the second one no longer waits on the SSD between layers (2x Tesla V100 with 30 GB RAM: +1% prefill; with 64 GB RAM,
  which reads the SSD less, unchanged).
- README: a table of speeds measured by users (`--bench`), starting with 2x TITAN RTX.
- `--bench` / `--report`: a prompt-only request's engine line no longer shows a decode speed in every case it has
  none.

## v1.0.8 - 2026-10-08

`--bench` warms up before it measures prefill, so speed reports compare fairly.

- **`--bench`:** one unmeasured prompt is read before the 2k and 8k prefill measurements. The 2k figure used to be
  the first prompt after start-up, with the caches still cold, and could read lower than the 8k one (152 vs 238
  tokens/s on a Titan RTX pair). Reported by @dummerjindabin (#8).
- **`--bench` and `--report`:** a request that only read a prompt shows just its prefill time in the engine lines,
  not a meaningless decode speed (it read "96,000 tokens/s").

## v1.0.7 - 2026-10-08

RTX 20-series cards (Turing) no longer crash on long prompts.

- **Fix (RTX 20-series, Titan RTX, Quadro RTX):** any prompt longer than about a thousand tokens stopped the engine, and
  the server restarted it with the conversation lost. The prefill attention asked for 66 KB of shared memory, more
  than the 64 KB Turing GPUs have, and the failure went unnoticed until the engine crashed. Both attention kernels
  now keep the cached rows in shared memory in their 16-bit form - the same values and arithmetic in half the space
  (34 KB, which every NVIDIA card gives without asking). On Turing, decode at long context also gets the faster
  attention it was silently skipping. Other cards: the same answers token for token, the same speed. Found and
  diagnosed by @dummerjindabin (#8).
- **AMD groundwork (#2, by @boxwrench):** the signals the GPU sends the CPU (the expert requests the CPU lane answers,
  the doorbells) are published past AMD's GPU cache, as in Strata #697, and a new HIP test replays the real routing
  handoff. The Linux AMD setup itself is in review (#7). NVIDIA: unchanged.

## v1.0.6 - 2026-10-08

A standard speed test: `./maya.sh --bench` (Windows: `START-MAYA.bat --bench`).

- **`--bench`** measures the installed model on your machine in a few minutes, with Maya stopped: decode on the same
  three questions everywhere (after a warm-up answer) and prefill at 2k and 8k tokens, through the engine the way the
  dashboard starts it. It writes `maya-bench.txt` with the results and the engine's per-token breakdown; `--report`
  includes it - so speed reports from different machines can be compared.
- README: single-GPU speeds (1x Tesla V100 32 GB, 64 GB RAM: decode up to 19 tokens/s, prefill up to 370 tokens/s).

## v1.0.5 - 2026-10-08

Maya-M is out: the larger quant (116 GB), closer to the full model token by token.

- **Maya-M** on Hugging Face (`Maya-M/`): IQ2_S gate/up experts with error-feedback rounding, IQ3_XXS down
  projections, IQ3_S in the most sensitive layers, from Z.ai's FP8 release with the FP8 model's own statistics,
  calibrated toward tool calls and front-end code. Against the FP8 model: 23% lower KL divergence than Maya-S
  (0.329 vs 0.428), the same next token 86.2% of the time (Maya-S 83.3%), and 97.9% of its zero-shot accuracy - the
  same as Maya-S (multiple-choice tasks do not separate the two). Set it up with
  `./setup.sh --setup --model Maya-M` (Windows: `START-MAYA.bat --setup --model Maya-M`); downloads are checked
  against their published sha256. Maya-S stays the default. Results: `bench/results/MAYA-M.md`.
- **Decode:** the dense matrix-vector products run in kernels compiled for their weight type (Q6_K, Q8_0, Q4_K,
  Q5_K) - 10% faster per product, bit-identical results; it shows on cards that hold most experts in VRAM.
- `--model` now takes that download even when an earlier setup used `--gguf-dir`.

## v1.0.4 - 2026-10-08

Faster prefill again (how fast Maya reads your prompt), and more room for experts at long context.

- **Prefill attention on the tensor cores:** the prompt's attention (the absorbed MLA over each token's selected
  positions) now runs on the GPU's tensor cores, with the next positions loaded while the current ones compute - FP16
  operands with F32 accumulation, as flash attention (within ~1e-3 of before). RTX 20-series cards keep the previous
  kernel automatically. Prefill on 2x Tesla V100 32 GB, prompts of 2k / 8k / 16k / 30k tokens: 273 / 428 / 487 / 506
  -> 286 / 469 / 538 / 561 tokens/s (since v1.0.1: +19% / +42% / +56%). On one of those GPUs: 8k-token prompt 226 ->
  243 tokens/s.
- **The attention cache in FP16:** half the memory, so more of the GPU holds experts - at the default 32K context
  0.66 -> 0.47 GB a GPU, at 128K about 0.8 GB more for experts on each GPU (faster decode at long context).
- Quality: the same long texts scored before and after (6,000 tokens read, the next 400 scored) differ by +0.8% in
  likelihood, less than two runs of the same engine differ from each other (+1.4%) - no measurable change.

## v1.0.3 - 2026-10-07

A one-command report for problems and speeds: `./maya.sh --report` (Windows: `START-MAYA.bat --report`).

- **`--report`** writes `maya-report.txt` in the Maya folder: your GPUs (VRAM, driver, PCIe link), CPU, RAM, disks,
  your Maya setup (model, context, GPUs) and the engine log's speed lines - how the model is split across VRAM, RAM and
  the SSD, and where each token's time goes. Attach it when you report a problem or a speed, so the engine can be
  tuned for your machine. Nothing is sent anywhere; your home folder shows as `~` and no API key is included.

## v1.0.2 - 2026-10-07

Faster prefill (how fast Maya reads your prompt): up to 46% on two GPUs and up to 2x on one.

- **Prefill speed:** the experts a prompt needs that are neither in VRAM nor in RAM are read from the SSD ahead of
  time into a deeper buffer (the reader was keeping the NVMe at about a quarter of its speed), the next layer's
  experts are read while the current one computes, and the prompt is cut into bigger pieces on bigger cards (fewer
  times every expert is fetched). Prefill measured on 2x Tesla V100 32 GB with 30 GB RAM, prompts of 2k / 8k / 16k /
  30k tokens: 240 / 331 / 345 / - -> 273 / 428 / 487 / 506 tokens/s. On one of those GPUs: prefill of an 8k-token
  prompt 115 -> 226 tokens/s. Answers are unchanged in quality (the same text, up to rounding).
- **Support Project Maya:** [buymeacoffee.com/peasantsmith](https://buymeacoffee.com/peasantsmith) (README, and the
  Sponsor button on GitHub).
- README: how Maya grew out of Strata; exported chats are named `maya-chat-*.md`.

## v1.0.1 - 2026-10-07

- **Windows (experimental):** `START-MAYA.bat` sets Maya up the way `./maya.sh` does on Linux - Python, the
  engine and the image encoder compiled with Visual Studio 2022 Build Tools and CUDA 12.8, the model download, the
  dashboard. The engine reads the model's experts with unbuffered parallel reads and sizes its pinned RAM to what
  Windows allows (RAM + page file). It compiles on Windows; it has not been run on a Windows PC with an NVIDIA GPU
  yet - tell us how it runs. See README > Windows.
- **Fix:** a second GPU big enough to hold all of its experts (e.g. 64 GB cards) stopped the start with "the pinned
  RAM tier did not allocate".
- **Fix:** temperature 0 is greedy and deterministic again.

## v1.0.0 - 2026-10-07

The first release.

- **Maya-S** (96.5 GB): Project Maya's compact quant of GLM-5.3-Flash, made from Z.ai's FP8 release. It keeps
  97.9% of the FP8 model's zero-shot accuracy (ARC-Easy, ARC-Challenge, HellaSwag, WinoGrande, PIQA).
- The engine: the model's experts tiered across VRAM, pinned RAM and the SSD, one GPU or two that share the layers,
  MTP speculative decoding, a thinking budget, images on demand.
- The installer (`./setup.sh`), the dashboard and the OpenAI / Anthropic compatible API.

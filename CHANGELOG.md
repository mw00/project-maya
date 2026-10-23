# Changelog

Every release is on GitHub (Releases) with these notes; every published change moves the last number. Update: the
dashboard's About > Update (from v1.0.18), or `git pull`, then `./setup.sh` (Windows: `START-MAYA.bat`) - it recompiles
only what changed and starts; the model is not downloaded again.

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

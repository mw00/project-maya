# Tuning and settings

Everything on this page is optional: the engine sizes itself for the PC it runs on. Back to the
[README](../README.md).

The settings below are environment variables named `MAYA_*`: set them before a start, or keep them in the config
with `./maya.sh --setup --env MAYA_NAME=value`. Their earlier `STRATA_*` spellings still work; where both are set, the
`MAYA_` one counts.

Responses API cache, background queue, event/reader limits and owner quotas are
server settings under `MAYA_GLM_RESPONSES_*`, with defaults and units in
[Responses settings](RESPONSES.md#settings-and-metrics). They can be set in the same
config `env` block; response storage and background work remain request opt-ins.

The engine sizes itself: it splits the layers across the GPUs it is given (up to 16; with more than two, each GPU
takes layers in proportion to its free VRAM), fills each GPU's free VRAM with experts, sizes its RAM tier from the
free RAM, and measures the CPU against the PCIe link at start to decide how many RAM experts the CPU computes itself.

**Tuning for your PC** (`./maya.sh --calibrate`, also offered at the end of the setup): that start-up guess times one
expert each way; the tuning measures the real output speed instead, in one engine run (about 10-15 minutes, most of
it loading the model), with a few splits of the RAM-tier experts between the CPU and the PCIe link and with fewer CPU
threads - more threads than the memory can feed only wait, and on a hybrid CPU the efficiency cores can hold the rest
up - and, when part of the model is read from the SSD, with smaller disk reads (the best size depends on the drive and
its driver). Every measurement answers prompts it has not answered before, as a chat's text is new to the expert tiers (the
same few answers over and over left VRAM holding exactly their experts, and the speed it reported was one no chat
reached), and the engine saves no learned expert profile (`--expert-profile-save`) and works on a copy of a usage
file (`MAYA_GLM_USAGE`), so the tuning's answers do not change what the
next start loads first. A setting is kept when it is more than 3% faster than the engine's own choice. The result
goes into the config's `"env"` (`MAYA_GLM_PCIE_SHARE`, `MAYA_GLM_CPU_LANE`, `MAYA_GLM_READ_CHUNKS`) and into
`~/.config/project-maya/calibration.json` for this PC, model and context, so setting up again keeps it. Stop a
running server first: the tuning needs the GPU(s).

**The CPUs the engine runs on.** The CPU lane's threads share the engine's CPUs with the thread that issues every
decode step and with the expert tiers' service threads (one a GPU), which are busy most of the time while experts sit in
RAM; the lane's idle workers also spin for a while after each batch. The default thread count leaves a third of the
cores to them only when the engine may use every CPU of a PC whose cores have no SMT siblings. Confined to fewer CPUs
(`taskset`, a container's cpuset), the lane takes a thread for every core it may use, and a decode step then waits for
a time slice as well as for its experts. So give the engine all the CPUs you can, on a hybrid CPU the efficiency cores
as well as the performance cores, or, where it has to stay confined, set `MAYA_GLM_CPU_LANE` (`MAYA_GLM_CPU_LANE<n>`)
lower so a few of its cores stay free. The start log says how many threads each GPU's lane has and what an expert
costs there (`glm fast: CUDA0 CPU lane: 8 threads, an expert ... ms on the CPU vs ... ms over PCIe`).

These settings change what the engine chooses (put them in the config with `--env`, or into its `"env"` block):

| Setting | Default | |
| --- | --- | --- |
| `MAYA_GLM_EXPERT_HELPER` | unset (off) | Experimental native-GGUF expert helper: the visible GPU index (`1`, not a Boolean). GPU 0 keeps the trunk and MTP; the helper owns a complementary expert cache, with heat-coordinated replacement and GPU quantization/reduction. Use `MAYA_GLM_SPLIT=0`, without `--layer-split`; unset this variable to disable. UMA helpers alias the pinned RAM tier; discrete helpers use their VRAM. See [AMD helper setup and scope](AMD_MAYA.md#experimental-expert-helper) |
| `MAYA_GLM_RAM_HEADROOM_GB` | 6 | RAM left free for the system when the RAM tier is sized; it also takes one more GPU's prompt buffers, pinned after the measurement - a split of three or more GPUs sets the others' aside on top, and each later GPU measures again and leaves at least half of it free. The pinned tier can't be swapped out, so a Linux desktop whose kernel starts swapping with more free (several displays, a busy session) wants more: a 128 GB desktop with three GPUs and three displays stayed responsive at 16, and swapped at 12 (#77) |
| `MAYA_GLM_RAM_GB` | from free RAM | a fixed RAM-tier size in GB |
| `MAYA_GLM_COMPLETE_POOL` | AMD: on | where the GPU's expert budget holds every expert, the 3 spares a layer and the prompt path's borrowed slots (the Ryzen AI Max APUs' memory on Windows - a Radeon 8065S holds Maya-S with ~60 GB to spare), each layer keeps exactly that many: the start loads every expert, a prompt borrows only the slots beyond them (none moved out, none read from the SSD again), the decode never misses and the CPU lane has nothing to take (8065S, Maya-S: decode +2.6-4%, +7% after an 8K-token prompt, the same answers). `0` = off; `1` = on for NVIDIA too (not measured there). Linux APUs keep their unified-memory policy |
| `MAYA_GLM_RESIDENT_SKIP` | AMD: on | while every expert is in VRAM, the decode leaves out each route's wait and fetch launches, which would find nothing to do (two a layer and token); `0` = launch them anyway; `1` = on for NVIDIA too |
| `MAYA_GLM_FLUSH`, `MAYA_GLM_PREFILL_FLUSH` | off, off | Windows + AMD only, experimental: the HIP runtime holds queued kernels until the host waits; `n` = the decode submits its first layer as soon as it is queued and then every n layers, `1` = the prompt path each sub-batch and each layer's experts (`cudaStreamQuery`, which does not wait). On a Radeon 8065S every 4 layers was ~4% slower than without; prompts no faster |
| `MAYA_GLM_RAM_RESIDENT` | off | `1` (or `--glm-ram-resident`): the RAM tier holds every expert not in VRAM and nothing is evicted to disk. The start fails if it cannot. For PCs whose RAM holds everything off-card (a 4090 D with 128 GB: decode 1.9x as fast with `PROMOTE_MIN=6`). `MAYA_GLM_RAM_SLACK` is extra slots per MoE layer (default 16) |
| `MAYA_GLM_SPLIT` | `auto`: a split search on 2-4 GPUs (and on more under DFlash2's pipelined window), by free VRAM with more | the first layer of each later GPU (`24`, or `7,12,17,22,27,32,37,41` for nine); `0` = one GPU. The config's `"layer_split"` sets the same. `auto` prices every placement before loading: each card's room for experts after its layers' weights and state, the share of each layer's routes the experts it then holds take (your usage counts; without them a built-in coverage curve, 1 - (1 - f)^3 of the routes for a share f of a layer's experts, ^1.4 for an EXL3 pack - `MAYA_SPLIT_COVER_B` sets the exponent; EXL3-4.05bpw under DFlash2 over an RTX 3090 and eight 5060 Ti: the 3090 takes 9 layers instead of 7, 63.8 / 32.6 / 64.4 -> 79.8 / 40.4 / 81.9 tok/s), each layer's reads over the card's memory bandwidth and each miss over the host's RAM speed - and takes the fastest one every card can start (on a two-GPU split that drafts with the model's MTP block the two cards work on consecutive tokens, so the token costs the slower card: RTX 3090 + 5060 Ti drafting with a NextN block, 24 -> 26, the cards 25.3 / 32.2 -> 28.1 / 29.8 ms a token, decode +3-5%). Under DFlash2's pipelined window (three GPUs or more) the slowest card paces every draft: the search runs on any number of GPUs (from five, a DP over the layers for each cap on the slowest card), each layer also costs a fixed 0.4 ms (`MAYA_GLM_SPLIT_LAYER_MS`) and a miss is priced at the shared CPU pool's speed, half the host's (Maya-L over an RTX 3090 and eight 5060 Ti: the 3090 takes 8 layers instead of the 10 its VRAM share gave it, 57.1 / 58.9 -> 61.6 / 63.1 tok/s). `MAYA_GLM_SPLIT_LOG=1` prints every placement's prediction |
| `MAYA_GLM_PARALLEL_LOAD` | on | a split's GPUs load at once, a thread each: they plan their RAM tiers one after another in GPU order (each gets the tier a load one by one gives it), then pin and warm them at the same time (4 x Tesla T4, Maya-L from a cold page cache: start to the first token 142 -> 96 s). `0` = one GPU after another |
| `MAYA_GLM_CPU_LANE` | one thread per physical core (one GPU, or one pool shared by a split of 3+ GPUs: at most one NUMA node's; two GPUs: half each; cores without SMT: two thirds of them) | CPU threads for RAM-tier experts; `0` = off (the tuning sets it) |
| `MAYA_GLM_CPU_LANE<n>` | the setting above | the same for CUDA`<n>` alone (`MAYA_GLM_CPU_LANE0=6`, `MAYA_GLM_CPU_LANE1=2`): two cards on links of different speed want different counts - the slower the link, the more CPU threads help. RTX 4070 Ti SUPER on PCIe 3.0 x4 + RTX 5070 Ti on 4.0 x16, split 17: 6 / 2 threads beat the default 4 / 4 |
| `MAYA_GLM_CPU_PIN` | Linux: on with two GPUs on a host with two NUMA nodes or more. Windows: never | a split whose GPUs have a pool each (two GPUs, or `MAYA_GLM_CPU_SHARED=0`): every GPU's CPU lane on CPUs of its own, a thread a physical core of them (the SMT siblings stay free) - a whole NUMA node while there are enough (its GPU's node when the GPUs hang off different ones), else an even share of a node's physical cores - so one card's idle workers never spin on the cores the other's are using (2 x Xeon Gold 6152, RTX 3090 + 5060 Ti: decode 15.4 -> 19.8 tok/s); on one NUMA node it stays off - there the parts' pools sharing every core were faster (two V100s on one 14-core Xeon: 25.8 tok/s unpinned, 24.7 pinned). `1` = pinned on any Linux host, `0` = unpinned. Linux only: on Windows the engine pins nothing whatever this says, and one GPU is never pinned anywhere, so `0` changes nothing there |
| `MAYA_GLM_CPU_SHARED` | on with 3+ GPUs | a split of three or more GPUs runs its cards one after another, so one CPU pool serves them all in turn, sized as one GPU's; `0` = a pool per GPU with the cores divided between them (A/B), `1` = one pool for two GPUs too. A card given its own count (`MAYA_GLM_CPU_LANE<n>`) keeps a pool of its own |
| `MAYA_GLM_CPU_CONCURRENT` | on | the pool a split shares: several GPUs' routes on it at once, a later GPU's jobs first (under DFlash2's pipelined window each GPU holds an earlier token than the one before it); `0` = one route at a time, the others waiting for it whole (A/B). The same answers either way |
| `MAYA_GLM_CPU_CAL` | unset | a file (e.g. `~/.config/project-maya/cpu-cal.txt`): the CPU lane's timing at start (an expert on the CPU, one over PCIe; ~1.7 s a GPU) is kept there, a line a GPU, and later starts read it. A new engine build, another pack, card, slot or thread count times again; a line that decode finds more than 15% off five times in a row is dropped, and the next start times again |
| `MAYA_GLM_KDA_GRAPH` | off | experimental: `1` replays single-GPU KDA decode layers as CUDA graphs; DSA, MTP/split, profiling, dumps, prefetch and lookahead stay direct. Two thinking-Max 32K/2048 pairs on v1.0.21: 30.73 -> 31.45 tok/s (+2.35%). Current-main full native/quality validation is pending; see [scope and checks](GLM-KDA-GRAPHS.md) |
| `MAYA_GLM_CPU_SPLIT` | about 48 pieces in all | decode: each CPU-lane thread's share of a token's RAM-tier experts is cut into this many pieces (1-64), claimed in turn, so one thread held up by the disk readers doesn't hold up the token. Few threads take several pieces each (6 threads: 8 - one V100, Maya-S: 17.1 tok/s against 14.3 with one); 40 or more take one each, which streams the memory best |
| `MAYA_GLM_PCIE_SHARE` | measured at start | the share of a token's RAM-tier experts copied over PCIe and run on the GPU instead of on the CPU: `0` = the CPU takes every one, `1` = none (the tuning sets it; `MAYA_GLM_CPU_PLAN` = the split per expert count, digits for 0..8) |
| `MAYA_GLM_CPU_RETIME` | on | the split the start-up timing chose (neither of the two above set) is timed again in decode: over the first windows of 2,000 CPU-lane experts (at most eight; the first only warms up), when the lane's own cost an expert there is more than 15% from the one the split was planned with the same way in two windows in a row, the split is planned again from their mean, at most three times, with one `CPU lane timed again in decode` line each. A start on a busy CPU (one timed 0.611 ms an expert against the usual ~0.24) no longer keeps its split for the engine's life. `0` = off |
| `MAYA_GLM_DUAL_PREFILL` | off | `1`, two GPUs that reach each other's memory (NVLink): a prompt of one chunk runs the split's halves one after the other, so the waiting GPU and its PCIe link sit idle. With this on, the waiting GPU stages every second expert of the working half's staging groups: host to its own (idle) staging buffer over its own link, then peer to peer over NVLink. A group then arrives through both links. Longer prompts (several chunks) are pipelined across the GPUs and stage as before. The answers stay the same unless the prompt's CPU share plan, which learns from the copy times, gives the CPU different experts (`MAYA_GLM_SCORE_PREFILL=1`, 3000-token context: KL 0.000000). Maya-M on 2x RTX 3090 (x8 each + NVLink), Ryzen 3950X, 128 GB: prompt 1K 283 -> 362 tokens/s (+28 %), 4K 622 -> 772 (+24 %), 16K 1034 -> 1142 (+10 %), 64K unchanged. Over PCIe-only peer access the copy would cross the working GPU's link anyway, so expect no gain there. The start checks that peer copies arrive intact (a 4 MB pattern each way): a machine whose GPUs report peer access but deliver wrong bytes keeps it off and says so |
| `MAYA_GLM_PREFILL_CALC` | off | `1`, two GPUs without peer access (no NVLink; with it `MAYA_GLM_DUAL_PREFILL` is the one): while one GPU runs its half of a one-chunk prompt, the waiting GPU **computes** a share of each MoE layer's RAM-tier experts - their weights over its own PCIe link from the working GPU's pinned RAM tier, the chunk's inputs and routes over the host to it, the weighted sum of its rows back - so both links stream. The share is learned per GPU from both lanes' times (`MAYA_GLM_PREFILL_CALC_SHARE=0..0.9` fixes it); the GPU with fewer multiprocessors takes the lightest experts. EXL3 experts only; longer prompts (several chunks) are pipelined and stage as before. No VRAM beyond the prompt path's loan (the waiting GPU's buffers are carved from its own idle prompt buffers); a chunk's sum (T x n_embd floats) of pinned RAM more per GPU. Not bit-exact - the share's rows are summed apart and added, and cuBLAS may pick other kernels on the other GPU: `MAYA_GLM_SCORE_PREFILL=1`, 2048-token context, CPU lane off: KL 0.0126, the size of any re-blocking (`MAYA_GLM_PREFILL_SUB=512`: 0.0138). turboderp's sc_3.10bpw on RTX 3090 + RTX 3060 (PCIe 4.0 x8 each), Ryzen 9 5900X, most experts in RAM: prompt 512 tokens 98 -> 144 tokens/s (+48 %), 1K 155 -> 229 (+48 %), 2K 244 -> 349 (+43 %), 4K 352 -> 468 (+33 %), 7K 445 -> 557 (+25 %). `MAYA_GLM_PREFILL_CALC_CHECK=1` (debug, slow) computes the share again on the working GPU and prints how far the two sums differ, layer by layer |
| `MAYA_GLM_PREFETCH_N` | 0 | `1` or `2` (more is read as 2; the engine log says the settings at start): each layer copies up to this many of the next layer's predicted experts (its router on this layer's input) that are not in VRAM into that layer's spares while it computes. `MAYA_GLM_PREFETCH_AT`: when the copy starts - `route` (the default: at once, beside this layer's own PCIe fetch and CPU lane), `fetch` (after this layer's fetch) or `cpu` (after its CPU lane); `MAYA_GLM_PREFETCH_BLOCKS`: the copy's blocks (default half the SMs); `MAYA_GLM_PREFETCH_RANK`: copy only the prediction's first n ranks (its first guesses are right almost always, its 5th-8th mostly not). Maya-M on an RTX 3090 + 3060 with `RAM_RESIDENT=1`, decode at 21K: off 19.64, `N=1` 17.0, `N=1 AT=fetch BLOCKS=4` 19.75, `+ RANK=2` 20.22 tokens/s (VRAM hits 64 -> 69 %); Maya-L there (bigger experts, a busier link): no gain |
| `MAYA_GLM_NUMA` | on with 2+ NUMA nodes | `0` = allocate the RAM tier without spreading it page by page over the sockets' memory |
| `MAYA_GLM_READ_CHUNKS` | 8 | the experts read from the SSD while it answers (and a prompt's): each part of one in this many pieces at once, 1-16. The best size depends on the drive, its driver and the CPU - 8 on a Linux NVMe, 4 on a Windows laptop's Intel RST RAID (decode +15%) - and the tuning measures 4 and 2 against 8 when part of the model lives on the SSD |
| `MAYA_GLM_PREFILL_CHUNK` | unset | `--prefill N` from the environment (the older setting; it wins over the config's `--prefill`) |
| `MAYA_PREFILL_LEND_PCT`, `MAYA_GLM_PREFILL_MB` | 90 or 85, unset | `--prefill auto`: the share of each card's expert-pool slots a prompt may borrow for its buffers: 90 when at least 90% of the experts' bytes are held pinned (in VRAM or the RAM tier), else 85 - or a fixed budget in MB |
| `MAYA_GLM_LEND_PARK_LRU` | 1 | when a prompt borrows the pool's tail, an expert leaving VRAM is moved over a RAM-tier expert of a lower use count - and, at `1`, on equal counts over the one used less recently (the counts age, so most are equal). A V100, Maya-S24, 512 tokens after a fresh 24K prompt: disk reads 520-578 -> 309-337, decode +2-5%. `0` = a strictly lower count only (before v1.1.21). `MAYA_GLM_TIER_DIAG=1` counts the disk reads by cause (`MAYA_GLM_TIER_DIAG_EVERY`: every N boundaries, 64) |
| `MAYA_GLM_PREFILL_TAIL_SKIP` | on | single-device prompts without a loaded NextN/MTP block: skip the last layer's attention output projection and FFN after updating all its caches; `0` restores the full computation. Splits and `GLM_CB_DIR` seam dumps keep the full path |
| `MAYA_GLM_PREFILL_WINDOW` | the chunk's tokens | prompts: the expert output rows kept on the GPU at once - each expert set's rows are added into the layer's output as the window fills, instead of every routed row waiting for one combine (~40 KB a token instead of ~185, so a chunk holds ~2x the tokens); `0` = every row (the old layout) |
| `MAYA_GLM_PREFILL_FUSED` | on for gfx1151 and gfx1100 | HIP with MMQ built: fused int8 WMMA gate/up, clipped SwiGLU and down for GLM routed experts (4096/2048 geometry and covered IQ types). `0` retains MMQ and its scratch layout. Strix Halo measured +13.2/+7.5/+4.8% prompt speed at 4K/8K/16K on the v1.0.16-based prototype; with the wmma2 attention on a Radeon 8065S (Windows, Maya-S), 3K / 8K-token prompts 233 / 253 -> 271 / 278 tokens/s. 4x RX 7900 XT (gfx1100, Maya-S24, ROCm 10): +5-13% at 1.3K-16K-token prompts. Other AMD architectures retain MMQ. |
| `MAYA_PF_WMMA` | off | NVIDIA Volta (sm_70; other cards keep MMQ): the prompt path's routed-expert GEMMs run as a grouped WMMA kernel - weights dequantized to FP16 in shared memory, FP16 tensor-core tiles with FP32 accumulators - instead of MMQ's dp4a on the CUDA cores; decode is unchanged. The launcher checks each shape and falls back to MMQ whole-path from the first one it cannot run, so a device or format outside the covered set (the native packs' IQ3_XXS / IQ3_S / IQ2_S gate-up and IQ4_NL / Q2_0 down products, K a whole 256 values) costs nothing. Maya-M on 2x Tesla V100-32GB: prompt 2.6K tokens 8.18 -> 7.30 s (+11.6%), 7.8K 12.63 -> 11.90 s (+5.8%), 20K 26.14 -> 24.3 s (+7%); decode unchanged (~50 ms/token either way). The outputs differ from MMQ's within FP16 weight rounding and FP32 accumulation order (checked against a double reference over the same FP16 weights by the `pf_wmma_parity` test, max error <= 2e-5 of the matrix's max magnitude) |
| `MAYA_GLM_USAGE` | off | a file to keep your expert usage in between starts (written after every request; the warm-up then loads your experts ahead of the expert profile's). Off, a start goes by the expert profile alone (below); a pack's old `expert_usage.txt` is no longer read unless this names it |
| `MAYA_GLM_SLOTS` | 4 | conversations kept aside on the SSD, so switching back to one doesn't re-read its prompt; `0` = off |
| `MAYA_GLM_PREFIX_CUT` | `1` (on with the slots) | with the slots on and no checkpoints (`--parallel` above 1, or `MAYA_GLM_CKPTS=0`): a prompt read from token 0 also ends a call at the end of its system prompt (the first user turn, at least `MAYA_GLM_SLOT_MIN` tokens, 512 past where the read starts) and keeps the state there in a slot of its own - another conversation with the same system prompt and tools (a new conversation of the same client, another request of the same kind) restores it and reads only the rest, as the one-conversation path does through its checkpoints. Once per system prompt; such a slot goes only as the least recently used, after a restart with `MAYA_GLM_SLOT_KEEP` too. A restored read gives the same tokens as a read from token 0 with the cut; the cut itself moves a chunk boundary of that read, so a greedy answer can differ from a read without it (as `--prefill N` can). `0` = off (no cut) |
| `MAYA_GLM_CKPTS`, `MAYA_GLM_CKPT_EVERY` | 4, 16384 | checkpoints of the conversation in the model at earlier positions than the one the next turn continues from, in RAM (~150 MB each for GLM-5.3-Flash): at the end of the system prompt (the first user turn), at the start of the prompt's last turn, and every `_EVERY` freshly read prompt tokens (`0` = none of these), where at least 512 fresh tokens lie before. A prompt that leaves the conversation earlier - an edited or replaced last message, another chat with the same system prompt and tools, a client that rewrote an older turn - restores the longest checkpoint it starts with and reads only the rest, where it read everything from token 0 (`glm checkpoints` in the engine log). At most `MAYA_GLM_CKPTS` (up to 16; `0` = off), the least recently used replaced; one conversation at a time (not with `--parallel`), text prompts only. Measured 2026-10-15 (Maya-L split, 8192-token chunks, a 100,535-token fresh read): `_EVERY` is also how many pieces the read is cut into, and every piece pays a fixed ~14.6 s fill beside ~6.5 s a chunk, so fewer pieces is straight wall time - 16384 cut that read into 5 pieces (49152 + 16384x3 + 2230) read in 148.8 s, 32768 into four (32768x3 + tail) in 134.9 s (-9.3%), `0` into one of 13 chunks in 93.2 s (**-37.4%**) |
| `MAYA_GLM_SLOT_MIN`, `MAYA_GLM_SLOT_GB`, `MAYA_GLM_SLOT_DIR` | 1024, 16, `<pack>/slots` | the shortest conversation kept aside (tokens, down to 1), their total size on disk (GB), and the folder |
| `MAYA_GLM_SLOT_KEEP`, `MAYA_GLM_SLOT_DAYS` | off, 7 | `1`: the kept conversations outlive the engine - the folder is not emptied at start (a slot is kept when its `.meta` names this model, engine version and pack, else removed), the conversation in the model is kept when the engine ends, and a slot unused for `MAYA_GLM_SLOT_DAYS` days is removed. With a supervisor that stops the model when it idles (llama-swap), the next request after a restart reads only its new tokens (a 4K-token conversation back in 0.2 s instead of re-read) |
| `MAYA_GLM_SEQS`, `MAYA_GLM_SEQ_CTX` | 1, the context | the first is `--parallel` (above): n conversations at once, n sequence states - the first for the whole context, the others for `MAYA_GLM_SEQ_CTX` tokens. On a split a token of each goes through the GPUs one behind the other, so every GPU works on a different conversation (4x RTX 4090, Maya-S24: 110 tok/s alone, 132 in all for two, 225 for four); each extra state takes its size from the expert pool (three at 64K: 3.5 GB) |
| `MAYA_GLM_SPLIT_ROWS` | `0` (off) | Multi-GPU decode only: `1`–`8` batches up to this many concurrent requests on each part. Each group shares dense weight reads while sequence-specific attention and routed experts remain per-request; groups pipeline across the split. Each part allocates row activations before sizing its expert pool (about 8 MiB per GPU). Unset or `0` preserves the existing pipeline. On four RTX 4000 Ada GPUs with Maya-M and eight requests, `2` measured 94.5 tok/s aggregate (+27% over the 74.42 tok/s legacy median); larger groups were slower. Single-request decode and prefill stayed within ±5% in the matched check. |
| `MAYA_GLM_EVEN_GROUPS` | `0` (off) | With `MAYA_GLM_SPLIT_ROWS` = k: `1` cuts the running requests into max(2, ceil(n / k)) groups of near-equal size (k at most) instead of filling groups of k in turn, so k or fewer requests no longer make one group that leaves every GPU but one idle (with k = 4: two groups of 2 at four requests, two of 4 at eight). The groups are sized from all running requests, so a group that comes back from the split is not cut again. The multi bench's split-rows path uses the same groups |
| `MAYA_GLM_SKIP_EMPTY` | `0` (off) | Batched decode steps (`--parallel`: one GPU, or a split with `MAYA_GLM_SPLIT_ROWS`): `1` leaves out each row's disk-miss wait, PCIe fetch and CPU-lane wait in a layer whose every expert is in VRAM (as of the last tier update), and elsewhere runs the wait and the fetch as one launch. It changes which launches run, not their arithmetic. The single-request token path is unchanged (`MAYA_GLM_RESIDENT_SKIP` is its whole-model version) |
| `MAYA_GLM_GROUP_EXPERTS` | `0` (off) | Batched decode steps: `1` routes every row of a step's group first and runs each distinct routed expert once for all the rows that chose it (the verify window's batched MoE), with one merged CPU-lane request per group on the part's own CPU pool, instead of each row's whole MoE in turn. Each row's output is meant to be the per-row kernels' bit for bit (`glm_moe_rows_test` checks the kernels). Takes 8 scratch expert slots per extra row of a group (k - 1 rows on a split, `--parallel` - 1 on one GPU) and the verify window's buffers from VRAM before the expert pool is sized |
| `MAYA_GLM_MULTI_ROOM`, `MAYA_GLM_MULTI_BOUNDARY`, `MAYA_GLM_MULTI_SPEC` | 16384, 32, on | with `MAYA_GLM_SEQS`: the room a conversation state must leave after a prompt (a longer prompt waits for the first, large state; max tokens are cut to the state), how many tokens a conversation decodes between two updates of the expert tiers, and whether a request alone uses the speculative decode |
| `MAYA_GLM_PROMOTE` | 8 | when the CPU computes every RAM-tier expert (it beats the PCIe link), experts moved between VRAM and RAM in the background per token, so VRAM follows what you use; `0` = off |
| `MAYA_GLM_PROMOTE_MIN` | 0 | a fetched expert is kept in VRAM only when its aged route count is at least this. `0` or `1` is the old rule (keep it whenever a spare is free). A flat route table promotes one-off experts and then evicts them; `6` stopped that churn on one 4090 D with 128 GB of RAM; with little RAM it costs (2x V100, 30 GB: 29.6 -> 28.3 tokens/s) |
| `MAYA_GLM_PROMOTE_FILL` | 24 | while VRAM has free expert slots (after a prompt gives back what it borrowed), up to this many of the hottest RAM-tier experts are copied into them per token (several per layer) instead of `MAYA_GLM_PROMOTE`; `0` = the same pace as the moves. The refill runs whatever the CPU lane's plan (the moves between VRAM and RAM only with the all-CPU plan): on nine GPUs whose 5060 Tis send one of 7-8 RAM-tier experts over PCIe, ~13% of VRAM used to stay empty after a long prompt |
| `MAYA_GLM_VRAM_EVICT` | `lru` with one conversation, `least` with several | a layer's spare is refilled by evicting its least recently used expert; `least` (or `0`) evicts its least-used by the aged counts instead, recency breaking ties (the rule until this became the default). With several conversations at once (`--parallel`) the default stays `least`: they take turns, and the recency let the latest one's experts push out what all of them use. It matters as much as experts are promoted (fetched over PCIe rather than computed by the CPU lane): Maya-S over 3x Tesla V100 16 GB (62 GB of RAM, an NVMe tier, one conversation at 128K), interleaved `--bench` runs, decode 24.8 -> 26.1 tok/s; Maya-L on an RTX 3090 + 3060 with a RAM tier and an NVMe tier, decode at 21K +2.6% (VRAM hits 54.5 -> 55.9 %); Maya-M there with `MAYA_GLM_CPU_PLAN=012234567` (one PCIe fetch a route at most) unchanged |
| `MAYA_GLM_MV_RDNA` | AMD: on for the Ryzen AI Max APUs (gfx115x) | the decode's dense GEMVs (the Q6_K and BF16 projections) in one RDNA3 kernel that keeps two blocks' loads in flight, bit for bit the original kernels' (Radeon 8065S, Maya-S: decode +3.1%); `1` = on any gfx11 (an RX 7900: not measured yet), `0` = off. NVIDIA: no effect |
| `MAYA_GLM_MTP_GGUF` | the model's own | two GPUs or more: a GGUF holding the MTP draft block to draft with - for a model published without one, or a more precise block than its own (`tools/maya_quant/mtp_gguf.py` writes a model's block alone); `MAYA_GLM_NO_MTP=1` = no drafting |
| `MAYA_GLM_MTP_KEEP` | 2 | two GPUs or more: the draft block computes the experts it is missing from VRAM only among its route's n best-scored ones (CPU lane / PCIe as for any route) and skips the rest - a draft only has to be a good guess, but a rejected one costs the first GPU a redo. `2` drafts better than skipping all and was never slower (2x V100: 82-83% accepted against 76-81%; RTX 3090 + 3060: 19.5 -> 20.6 tok/s); `0` = skip every missing expert, `MAYA_GLM_MTP_MISS=1` = fetch them all |
| `MAYA_GLM_MTP_DECODE` | on for the Ryzen AI Max APUs (Strix / Gorgon Halo, gfx115x) | one GPU: faster answers where the GPU holds the whole model (a Radeon 8065S, Maya-S: 256-token answers +44-50% to short prompts, +26-27% after 3-8K-token prompts), the same answers as without it. `0` = off; `1` = on for one GPU elsewhere (usually slower where experts stream from RAM). Not with `--parallel` or a DFlash2 drafter |
| `MAYA_GLM_TRUNK_KEEP`, `MAYA_GLM_TRUNK_KEEP_MAP` | unset (8) | **changes the answers** - for a RAM-bound split that would rather be faster: the model's own layers leave out the experts they are missing from VRAM from route rank n on, as the draft block does; `_MAP` = one digit per layer (`8` = every expert). The lowest-ranked experts carry the least weight and are the coldest; GLM-5.3's early layers notice a left-out expert far more than its late ones (keep 5 in one band of layers at a time, Maya-M, KL to the full model on held-out text: layers 3-7 0.040, 23-29 0.012, 30-44 0.002-0.003). RTX 3090 + 3060, decode at 21K: Maya-M 20.0 -> 24.6 tok/s with `_MAP=8888888888888888888888855555554444444444444448` (keep 5 on layers 23-29, 4 on 30-44) and split 28, KL 0.018 (code 0.017), where a uniform `TRUNK_KEEP=7` gives 21.8 at 0.032; Maya-L +25% with the same map and split 27, KL 0.017. The verify windows (DFlash2, the batched window) leave out the same experts as the token path |
| `MAYA_GLM_DISK_SKIP_W`, `MAYA_GLM_DISK_SKIP_FROM` | off (0, 8) | **an approximation, for experts that do not fit in VRAM + RAM**: a routed expert in neither VRAM nor the RAM tier is left out of the token instead of read from the SSD when its share of the route's weights is below `MAYA_GLM_DISK_SKIP_W`, or its rank (0-7, the route's order: score + bias) is at least `MAYA_GLM_DISK_SKIP_FROM`. The weight rule leaves out what matters least: a 3090 with 60 GB of RAM and 74 GB of experts on a PCIe 4 NVMe, `--bench` decode 13.3 -> 21.6 tokens/s at `_W=0.12`; KL against the full computation (CPU lane off, 575 tokens): `_W=0.12` 0.12, `_W=0.14` 0.15, `_FROM=6` 0.08, `_FROM=4` 0.11 (Maya-S is 0.43 from FP8; the CPU lane's own arithmetic moves the output by ~0.06). The engine log's `glm skip` line counts the experts left out. Lookahead (`MAYA_GLM_AHEAD`) then reads only the ranks a route would wait for |
| `MAYA_GLM_MISS_SKIP_W`, `MAYA_GLM_MISS_SKIP_FROM` | off (0, 0) | **changes the answers** - for a RAM-bound decode: a routed expert that is NOT IN VRAM and carries less than this share of the route's weights is left out (no PCIe fetch, no CPU-lane work), on layers from `_FROM` on - `TRUNK_KEEP`'s idea by weight instead of rank, so a token routed to one or two strong experts drops more of its light misses than one spread evenly. GLM-5.3's early layers notice it far more than its late ones: keep `_FROM` at about 23 (on every layer, 0.04 already gave KL 0.041 for ~1% speed). RTX 3090 alone, Ryzen 7 9700X, DDR5-3600, `_FROM=23`, full suite (chat / agent tok/s, KL to the untrimmed model on 1,746 tokens): Maya-S24 `TRUNK_KEEP_MAP` stage 4 (`...5555555444...`) 27.4 / 27.3, KL 0.027; 0.12: 29.3 / 27.7, KL 0.037; 0.14: 30.1 / 29.0, KL 0.052; 0.16: 30.9 / 29.6, KL 0.075 (`TRUNK_KEEP_MAP` at the same KL: 29.7 / 28.8). Maya-M stage 4 19.9 / 18.1, KL 0.038; 0.12: 21.3 / 19.5, KL 0.047; 0.14: 21.8 / 21.0, KL 0.060. A request may set its own for a sweep without a restart: `"strata_tune": {"miss_skip_w": 0.14, "miss_skip_from": 23}` (0: off for it) |
| `MAYA_GLM_SPEC_HEAD` | half the GPUs | more than two GPUs: how many of the split's first GPUs form the head group of the speculative decode - the head group runs the draft's position while the rest finish the current token, as the two halves of a two-GPU split do (4x Tesla T4, Maya-S: decode 14.1 -> 21.0 tok/s against no drafting); `MAYA_GLM_NO_SPEC=1` = token by token |
| `MAYA_GLM_SPEC_CANCEL` | off | two GPUs (a head group of one GPU): `1` = when the tail's token shows that the draft was wrong, the head's speculative pass for it stops at its next layer (its heavy kernels do nothing, its routes go out empty) and the redo starts at once, while the tail drafts - instead of after the rest of a pass already known to be thrown away. The tokens are the same (greedy, CPU lane and layer map off: three requests identical to the token, 166 cancels in one of them). RTX 3090 + 3060, Maya-M: 5.4-6.5 ms of the first GPU's time saved per rejected draft (greedy, identical text: decode +0.9 to +2.6 %); at 21K with sampling it stayed within the run-to-run noise there (+0.6-1.2 %) - it pays more where the first GPU paces more clearly and more drafts are rejected. With `MAYA_GLM_SPEC_SAMPLE=1` the two add up: +5.3 % together on that box (the first GPU's average pass 31.6 -> 30.4 ms) |
| `MAYA_GLM_SPEC_SAMPLE` | off | two GPUs or more (the pipelined decode), sampled requests: `1` = speculative sampling - the draft block's draft is DRAWN from its own distribution q (the request's sampler chain on its logits, a Philox stream of its own) and the token is the draft with probability min(1, p/q), else a draw from max(0, p - q): the tokens keep the plain draw's distribution exactly, and a draft stands sum(min(p, q)) of the time instead of p(the draft block's top token). A fixed seed still repeats its answer - not the answer it gives without this. Greedy requests (and penalties, which this path does not take) are unchanged. RTX 3090 + 3060, Maya-M, 21K, default sampling: drafts kept 68.6 -> 74.7 %, decode +1.6-2.7 % alone, +5.3 % with `MAYA_GLM_SPEC_CANCEL=1` (24.30 -> 25.57 tok/s) |
| `MAYA_GLM_SPEC_FOLLOW` | off | speed A/B (`--spec-follow PATH`): a file of token ids (as `--tokens-file` takes them) - every request decodes that continuation instead of its own picks and ends with it, and the drafting decodes (the two-GPU draft decode, the one-GPU MTP block, DFlash2) keep exactly the drafts that match it: two builds or settings then process the same text with the same windows, drafts and routes, where a near-tie flip would otherwise change the text and with it the draft acceptance and the speed. The log's `glm follow:` line counts the tokens where the model's own pick differed (a check of the numbers: a placement change flips a few near-ties). One conversation at a time (not with `MAYA_GLM_SEQS`). A text from another model, or from the same model with other settings, favours the drafts that made it: compare a change on the same model and its own text |
| `MAYA_GLM_SPEC_FOLLOW_SAVE` | off | `--spec-follow-save PATH`: each request's decoded token ids are written to that file - the continuation `MAYA_GLM_SPEC_FOLLOW` reads |
| `MAYA_GLM_SERVICE_IDLE_MS` | 200 | the engine's tier threads (one per GPU) spin while a decode routes experts and sleep after this long without one, so an idle engine uses ~1% of a core instead of one core per GPU; `0` = spin always |
| `MAYA_GLM_PREFILL_CPU` | on | prompts: the least routed RAM-tier experts are computed on the CPU while the GPU loads the rest over PCIe, the split balanced each layer so both finish together; `0` = off. `MAYA_GLM_PREFILL_CPU_ROW_MS` fixes the CPU cost of a row it plans with (default: learned) |
| `MAYA_GLM_PRESTAGE` | 160 on one GPU, 0 on more | prompts: experts copied to the GPU while a layer's attention runs (the PCIe link is idle then), into a buffer of this many experts borrowed from the pool's tail - the next layer's most routed RAM-tier ones; `0` = off. `MAYA_GLM_PRESTAGE_ADAPT=1` (experimental, not yet measured) learns each layer's count from how long its attention takes |
| `MAYA_GLM_DISK_READERS` | off (0) | prompts: this many reader threads read the disk-only experts, each with one expert in flight (its three parts read whole) and each handed on as soon as it and those before it landed, instead of one reader reading `MAYA_GLM_DISK_QD` at a time in small pieces and handing the batch on when all of it landed. An NVMe whose random reads want large requests gains most: a DRAM-less Kingston NV3 (74 GB of experts, a 3090 with 60 GB of RAM), `--bench` with `8`: a 2K prompt +50%, an 8K one +25%; same output (KL against the old reader 0.019, within its own run-to-run 0.018). An NVMe with its own DRAM can lose instead (a Samsung PM981a beside 2x V100: prompts 15% slower with `8`), so measure yours |
| `MAYA_GLM_PREFILL_TRACE` | off | `1`: per layer of every prompt chunk, the attention time and when the prestage copies, the copy lane and the CPU lane ended (debug) |
| `MAYA_GLM_PREFILL_STAGES` | on | prompts: where a chunk's wall went, one line a part a chunk - the copy lane's span (the copies, the ring waits for a group's staging slot the compute stream is still reading, the gaps inside a layer's groups, the windows between two layers' groups where a lane that is going to idle, idles: the mixer and the host's plan) with its lead and tail beside it (lead + lane + tail = the chunk's wall), the bytes and experts each lane moved (with the pcie GB/s over the copy time), and the layer lane's max/min; plus one line a prompt with what each part's chunks walled and what the hand-overs between them cost (`glm prefill: 6500 tokens in 4802 ms (0.74 ms/token, 1353 tok/s), 1 chunk of 6500 | CUDA0 4523 ms, ...`). The `copies` also carry the CPU lane's uploads (same stream). It costs nothing measurable: the syncs it adds wait on copies the compute stream has already waited on (the last MoE layer prestages no next one). `0` = off |
| `MAYA_GLM_PREFILL_KEEP` | 13 (route + spares + 2) | prompts: the slots a layer of the expert pool keeps while the pool lends the prompt path its tail, so the chunk's buffers are sized by `(slots a layer - this) x the part's stride sum`. Lowering it lends more and buys a bigger chunk (`--prefill`) at the resident cache's expense - the tail still serves decode between prompts, but a layer's permanent slots shrink by the same amount. The floor is route + spares (11 here): the pool's own sizing needs that much left (`need = route + spares + lent`), so a smaller number is clamped up, not obeyed. Measured on the Maya-L four-way split (a 9-MoE-layer 5060 part, 32 slots a layer, 110 MB a slot column): the chunk's borrow is 830 MB + 0.119 MB a token + 96 weight scratch + 134-149 staging, so 6500 tokens borrow 1863 MB of the 2097 MB it can lend (19 slots) and 8192 needs 2319 - measured: at 11 with `MAYA_GLM_RESERVE_MB=400` (35 slots) the bind reaches 8192 and no further (`--prefill 16384` halves once and prints `chunks of 8192`); 16384 needs 3292 MB, which a 5060 holding three dense layers does not have to lend |
| `MAYA_GLM_MMQ_RESERVE_MB` | off (0) | prompts: VRAM kept out of the expert pool for the prompt path's MMQ scratch (llama.cpp's stream-k fixup tiles), subtracted by both the pool's sizing (`fast_setup`) and the chunk this part is sized for (`prefill_setup`); 0 is right with the slab, which carries the scratch in the prompt path's own loan (16 MB, a bound), so a value above 0 only takes room from the experts |
| `MAYA_GLM_WARM_PROMPT` | off (0) | `--serve`: one synthetic prompt of this many tokens (at least 32) through the whole part chain before READY, so a session's first request does not pay what only a first prompt pays (issue #8: the first prompt of every session read 95-133 ms/token against 23-44 warm - every kernel's first launch, the prompt path's own allocations, the tiers' first reads). It costs about one chunk of the smallest part's size (a few seconds to ~15 s on a four-way split of a big model, beside the tiers' own warm-up at a start), the ids are spread over the vocabulary, and the state is zeroed right after (`glm fast: the prompt path is warm (64 tokens in ...)`), so no conversation sees it. `0` or unset = off |
| `MAYA_GLM_KQ` | on | prompts' CPU experts: the multi-token AVX-512 Q3_K/Q2_K kernel (about 2x ggml's from 8 tokens an expert); `0` = ggml's dots only |
| `MAYA_GLM_CPU_MT` | on | CPU experts that several tokens share (a DFlash2 window's merged CPU pass, a prompt's CPU experts): each weight row decoded once for up to 8 tokens - IQ2_XXS/XS/S, IQ3_XXS, IQ3_S, IQ4_XS, Q4_K, Q5_K, on AVX2 or AVX-512 - with results bit-identical to ggml's one-pair dots (checked against them at the first use; a format that does not match keeps ggml's). Maya-L on a Xeon 6152, 40 threads, an expert shared by 2 / 4 / 8 tokens: 455 -> 333 / 889 -> 434 / 1794 -> 864 us. IQ2_XXS and IQ3_XXS rows build their signs in registers (one core, 2-8 tokens: 23-34% faster than the table lookups). One token: `MAYA_GLM_CPU_1T`; `0` = ggml's dots only |
| `MAYA_GLM_CPU_1T` | on | a decode step's CPU experts with IQ3_S or IQ3_XXS weights (Maya-L's gate/up), one token each: ggml-cpu's AVX2 dot with a cheaper decode - IQ3_XXS's signs built in registers, IQ3_S's grid indices in scalar registers - bit-identical to it (checked against it at the first use). Maya-L on a Xeon 6152, one expert's gate/up + down for one token, 1 / 12 / 40 threads: 3677 -> 3274 / 370 -> 333 / 250 -> 247 us - the fewer cores the CPU lane has, the more it gains. IQ2_XXS (Maya-S / Maya-S24's gate/up) on a CPU with AVX-512 VBMI and BMI2 (Zen 4 and later, Ice Lake and later): the grid as 2-bit codes in registers (every grid byte is 0x08 / 0x19 / 0x2b) and the signs built by pdep, bit-identical too - a Ryzen 7 9700X with an RTX 3090, Maya-S24, 8 threads: an expert 0.220 -> 0.175-0.196 ms, decode +6-8%. `0` = ggml's dot, `MAYA_NO_IQ512=1` = the AVX-2 tier only |
| `MAYA_GLM_PREFILL_SUB` | NVIDIA: up to 4096; AMD: 256 | prompts: the sub-batch the attention layers, the dense FFN and the shared expert run in (16-8192). Unset on NVIDIA, the attention layers' and the dense FFN's is 256 doubled while it fits in memory the chunk's MoE buffers already take (fewer, larger GEMMs, each weight dequantized once a sub-batch) and the shared expert's stays 256; `256` = the old sub-batches |
| `MAYA_GLM_KV_INT8` | off | `1` (or `--kv int8` in the config's `"args"`): the attention's latent cache in INT8 - 512 codes and one FP16 scale per 32 values, 544 bytes a token and layer instead of 1024. One V100, 128K context: state 1.78 -> 1.13 GB, 84 more expert slots; no measurable quality change (KL against FP16 within FP16's own run-to-run spread); prompts read within about 2% of FP16's speed (two V100s, 8K-32K) |
| `--kv-streaming on\|off`, `--kv-resident N` | on (the setup's configs), 32768 | KV streaming, as Strata's: the setup writes `"--kv-streaming", "on"` into every config's `"args"` - `off` there turns it off, `--kv-resident N` beside it sets the window (`MAYA_GLM_KV_RESIDENT` the same from the environment), and a setup again keeps what you set: each attention layer's latent cache lives in pinned RAM and N of its positions (at least 20480) in VRAM, so the VRAM the rest took holds experts. After the indexer picks a token's 2,048 positions, the blocks of 4 it names that are not in VRAM are copied in over PCIe (a clock sweep picks what leaves); prompts read a layer's whole cache from a staging copy, filled over PCIe while the layers before it compute (`MAYA_GLM_KV_PREFETCH=0`: on the compute stream, before the layer). A context of N positions or fewer is not streamed; with several conversations at once (`--parallel`) each keeps its own host copy. It pays off where the cache is large beside the cards: long contexts and several conversations on 1-3 GPUs (Maya-L with the DFlash2 drafter, an RTX 3090 and RTX 5060 Tis, 183 GB of RAM). At 512K on the 3090 the cache would take 6.4 GB of VRAM: with streaming 1,176 expert slots instead of 714, and prompts 4-9x faster (100K tokens 836 against 92 tok/s - without it the pool lends the prompt path only 768-token chunks), decode +5%. 1M on the 3090 starts only with streaming (a 900K-token prompt at 489 tok/s). Three GPUs at 1M decode 16% faster; two GPUs at 1M read prompts 26% faster with `--prefill 32768` (without streaming the split falls back to 16K chunks); four conversations at 128K (`--parallel 4`) on two GPUs decode 37% faster together. At 128K with one conversation it changes little (the 3090: expert slots 1,218 -> 1,302, decode within run-to-run noise; prompts as fast as before - 100K tokens 856 / 858 tok/s, without the prefetch 809). Decoding gives the same tokens as without it; a prompt's tokens can differ slightly, as with any change in which experts fit in VRAM |
| `MAYA_GLM_DSA_SCORE` | on | prompts: the sparse attention's indexer scores as a register-tiled FP32 GEMM (a 3090, 4096 tokens at position 10K: 78.8 -> 5.6 ms); `0` = the warp-per-pool kernel |
| `MAYA_GLM_DSA_ALL` | on | decode: while a context of at most the indexer's top-k positions (2048) leaves every pool in the selection, the sparse attention takes them all in position order without scoring them - the indexer's query GEMV, its scores and the selection left out (~40 us a DSA layer); every decode path the same, so the tokens match between them. `0` = scored and selected as at any length |
| `MAYA_GLM_MLA_MMA` | on (NVIDIA sm_80+) | decode: the sparse attention's latent scores and context on the tensor cores (FP16 parts with F32 sums, the products at F32's precision), a block running over up to 16 chunks of 32 cells; a verify window's tokens in a few launches. ~4x faster than the F32 chunks at 512-2048 cells (RTX 5060 Ti 49 -> 11 us a token and DSA layer at 512). `0` = the F32 chunks |
| `MAYA_GLM_MV_P100`, `MAYA_GLM_MV_P100_R` | on (compute capability 6.0 only) | decode on a Tesla P100 (GP100): the one-column Q6_K GEMVs - Maya-M's dense weights - read each row's blocks as aligned words handed to the lanes by warp shuffles, 2 rows a warp below 4096 rows in a launch, else 4, with the same sums bit for bit (four GP100 dies, Maya-M: decode +23%). `0` = the generic kernel; `MAYA_GLM_MV_P100_R` `2` or `4` forces the rows a warp |
| `MAYA_GLM_GU_STAGE` | on (compute capability 6.0 only) | decode on a Tesla P100 (GP100): the routed experts' gate/up kernel stages the layer's input in shared memory once a block (IQ2_S gate/up, 8 experts on a GP100 die: 274 -> 256 us), the same sums; `0` = it reads the input from global memory |
| `MAYA_GLM_PREFILL_ATTN` | tensor cores | prompts' sparse attention kernel: on Ampere and newer an mma.sync kernel with 32 heads a block (a 3090, 26.6K-token prompt: 832 -> 966 tok/s); `wmma` = the 91 KB shared-memory kernel Volta runs, `tcreg` = the register kernel Turing runs, `f32` = the FP32 kernel. AMD: `wmma2`, the default on gfx12 (R9700 / RX 9070: prompts +8-10%) and Strix Halo (gfx1151: prompts +4%), or `f16q`, the default on the other gfx11. `MAYA_GLM_PREFILL_ATTN_CHECK=1` compares the chosen kernel with the FP32 one (debug) |
| `MAYA_GLM_EXL3_MMA_ROWS` | 64 | EXL3 packs' prompts: an expert routed at most this many of a chunk's tokens runs in tensor-core kernels that decode its trellis as they go (RTX 30 and newer); a busier one is decoded to FP16 once and multiplied by cuBLAS. `0` = cuBLAS for every expert |
| `MAYA_GLM_EXL3_MMA_ROWS_SHORT` | unset (off) | `<rows>[,<max chunk tokens>]`: in a prompt chunk of up to `<max chunk tokens>` (4096) an EXL3 expert routed up to `<rows>` of the chunk's tokens goes through the tensor-core kernels instead of being reconstructed to FP16 for cuBLAS; longer chunks keep `MAYA_GLM_EXL3_MMA_ROWS`. In a short chunk (a short read, or a read cut into short chunks) the experts have tens to a few hundred rows each and the FP16 reconstruction, a fixed cost per expert and chunk, is most of the chunk's expert time; in a long chunk the FP16 product is cheaper. `192` is a good start. Not bit-exact: the two paths add in another order, so greedy answers can differ from the setting off. Prompts only; decode is unchanged |
| `MAYA_GLM_EXL3_DECODE` | by GPU | EXL3 packs' decode kernels: RTX 30 and newer decode each tile on the tensor cores (`new`); NVIDIA GPUs before them (V100, T4, RTX 20) keep v1.1.8's kernels (`legacy`), which decode about 10% faster there than the new ones on the CUDA cores, with v1.1.8's output. `legacy` or `new` picks one on any GPU (A/B); prompts take the same path either way |
| `MAYA_GLM_EXL3_ROT` | 1 | EXL3 packs' prompts, an expert through cuBLAS (A/B): its rows take the input and output rotations and its weight is only decoded to FP16 (`x W = H(H(x * suh) W') * svh`, as in decode); `0` = the weight rotated and scaled to FP16 instead (two 128-point transforms of every weight: an 8K prompt on 9 GPUs 5-7% slower) |
| `MAYA_GLM_MMQ_GLU` | on (NVIDIA, RTX 20 and newer) | prompts: the experts' SwiGLU applied in the MMQ gate/up product's write-back (`third_party/ggml/mmq-fused-swiglu.patch` on the pinned MMQ): it writes a chunk's T x n_ff values instead of T x 2 n_ff that a separate kernel reads back - the same values bit for bit where the plain product sums each output in one block, as at a chunk's size (`glm_mmq_glu_test`). Maya-L's shape, one layer of an 8K chunk: RTX 5060 Ti 43.5 -> 36.0 ms, RTX 3090 26.3 -> 25.4 ms; an RTX 3090 alone reads prompts as fast as before (they wait on the CPU and PCIe there). `0` = the separate kernel. A ggml checkout the patch does not apply to (`-DMAYA_GGML_DIR`) builds without it |
| `MAYA_GLM_DROP_CACHE` | on with 2+ NUMA nodes | before the RAM tier is pinned, the model files' clean page cache is dropped (the engine reads experts with O_DIRECT): a cached GGUF filling one node made the interleaved tier land 78% on the other, and the CPU lane read one socket's memory; `0` = keep it |
| `MAYA_GLM_PROFILE_WEIGHT` | 1 | with `MAYA_GLM_USAGE`: the weight of the pack's routing profile (`expert_counts.txt`) against your usage file in the start-up order of the expert tiers; `0` = your usage only |
| `MAYA_GLM_LOOP_THINK`, `MAYA_GLM_LOOP_ANSWER` | 256, 1024 | the loop guard: when the last this-many generated tokens are one pattern of at most 32 tokens repeated exactly (a quantized model can fall into `0 0 0 ...` until `max_tokens`), thinking is closed at once, and in the answer the model's current tool-call argument (or else the turn) is closed so an agent goes on; a second loop in the same answer ends it. `0` = off |
| `MAYA_GLM_WIN_PIPE` | three GPUs or more | DFlash2 on a split: `1` = the pipelined verify window (the GPUs on consecutive tokens at once), `0` = the batched one (the whole window a GPU at a time). Two GPUs take the batched one, three or more the pipelined one |
| `MAYA_GLM_WIN_THREADS`, `MAYA_GLM_WIN_SKIP` | on | the pipelined window (A/B): `0` = one host thread issuing every GPU's work (nine GPUs: 3-5 ms a token, longer than any GPU's share) / the tokens after a wrong draft run their work anyway |
| `MAYA_GLM_HEAD_SPLIT`, `MAYA_GLM_HEAD_SPLIT_CHECK` | on (EXL3 packs with DFlash2 on 3+ GPUs), off | the pipelined window's pick reads the output head's first half of the vocabulary on the last GPU and the second half on a helper GPU (the one of the fewest layers between the first and the last, which keeps its own copy: ~240 MB of its expert pool) at once, then joins them - the whole row's pick; `MAYA_GLM_HEAD_SPLIT=0` = the whole head on the last GPU. `MAYA_GLM_HEAD_SPLIT_CHECK=1` (debug, slower, greedy): the whole row as well, the picks compared - the count of picks and of differing ones in the engine log at exit |
| `MAYA_GLM_DFLASH_HEAD_SPLIT` | on (with the head split) | a DFlash2 draft's head reads the same halves at once - the vocabulary's first half on the last GPU, the second on the helper, each taking its own slices' top-k candidates (the helper idles while the drafter runs); ~0.5 ms off each round's drafting. `0` = the whole head on the last GPU |
| `MAYA_GLM_DFLASH_AHEAD` | on (pipelined window) | the next round's context injection and draft are queued from a thread of their own as soon as the round's last pick is in, beside the commit and the next anchor's begin (their ~1.3 ms of launches were behind those, the drafter's GPU idle meanwhile). `0` = at the top of the next round |
| `MAYA_GLM_DFLASH_INJ_OVERLAP` | on | the drafter's context injection (the kept positions' K / V into its ring, ~0.5 ms on an RTX 5060 Ti) on a stream of its own beside the next draft: the draft's first layer runs meanwhile, each layer's attention waiting only for that layer's rows. `0` = before the draft, on its stream |
| `MAYA_GLM_WIN_COMMIT_WAIT` | off | `1` = a pipelined window's commit waits until every GPU's restore and replay is queued before the next anchor is begun (by default each GPU's next begin queues behind its own commit, and the state is waited for before anything else reads it) |
| `MAYA_GLM_WIN_REPLAY` | on | a DFlash window's commit replays the kept tokens onto the recurrent states' backup through all of a GPU's KDA layers at once (two launches a token), then copies the backup to the live states - and the next pipelined window skips its backup. `0` = the live states restored, then replayed layer by layer |
| `MAYA_GLM_WIN_LANES`, `MAYA_GLM_WIN_CPU`, `MAYA_GLM_WIN_STAGGER`, `MAYA_GLM_WIN_ALLCPU` | 2, on, on, on | the batched window (one GPU, or two): its CPU lane on two pools - one GPU: a second pool on the other NUMA node (`2`: both unpinned, `1`: pinned to a node each, `0`: one pool); two GPUs: each GPU's window layer also on the other GPU's pool - with each layer's distinct experts read once and split between the two (`MAYA_GLM_WIN_CPU=0`: a token a pool); the window's tokens in two groups whose layer halves interleave, so the CPU answers one group while the GPU runs the other's attention (`MAYA_GLM_WIN_STAGGER=0`: all of them a layer at a time); and every RAM-tier expert of a window to the CPU (`MAYA_GLM_WIN_ALLCPU=0`: the calibrated CPU / PCIe plan). One single token keeps one pool |
| `MAYA_GLM_WIN_BATCH` | NVIDIA: on, AMD: off | the batched window's routed experts: each distinct expert once for all of the window's tokens (`1`), or each token's experts on the token path's kernels (`0`; AMD's default - its token-path expert kernels are tuned for the Ryzen AI Max APUs, and the window's tokens then also run a layer at a time). The same tokens either way |
| `MAYA_GLM_WIN_MULTI` | on | with `MAYA_GLM_WIN_BATCH` off (AMD): the window's tokens' routed experts in one launch for all of them (each token's arithmetic as on its own; an expert two tokens route is read from memory once - a Radeon 8065S's answers +10%); `0` = one token after another. The same tokens either way |
| `MAYA_GLM_WIN_X3`, `MAYA_GLM_WIN_X3_POOLS` | on, 2 | EXL3 packs, the batched window (A/B): `MAYA_GLM_WIN_X3=0` = each token's routed experts on the GPU one token after another instead of each distinct expert once for all its tokens (the same bits either way); `MAYA_GLM_WIN_X3_POOLS=1` = a merged window layer's CPU experts on the first pool alone instead of split over both (with one GPU both unpinned pools share the same cores; EXL3-4.05bpw with DFlash2 on one RTX 3090: the same speed either way) |
| `MAYA_GLM_DFLASH_STATS`, `MAYA_GLM_DFLASH_PROF` | off | DFlash2: a line a round to that file (drafts checked and kept, the draft's and the window's ms, the round's CPU-lane experts and VRAM misses, the drafts' confidences) / `1` = each request's round timing in the engine log |
| `MAYA_GLM_DFLASH_TL` | off | DFlash2's pipelined window: a line a round to that file - ms from the round's first token's start: the drafts ready and pushed, the host's begin steps, each pick as it comes out of the last GPU, then a `c` line with the commit done |
| `MAYA_GLM_DFLASH_PRIO` | on | the drafter's stream at the device's highest priority, so a draft is not held behind the last GPU's queued work (9 GPUs, EXL3-4.05bpw: a draft 6.4 -> 5.7 ms); `0` = the default priority |
| `MAYA_GLM_TIMING`, `MAYA_GLM_POOL_STATS` | off | `1` = timing and cache statistics in the engine log |
| `MAYA_GLM_ROUTE_LOG` | off | `<prefix>`: every decode route as a line in `<prefix>.<GPU>` - the layer, its experts, which were fetched, missed or computed on the CPU, then after a `\|` the 16 near misses (the next ranks by the same score) - for cache studies: `tools/glm_tier_replay.py` replays a log through a model of the VRAM tier's rules and alternatives |
| `MAYA_GLM_SCORE_PREFILL` | off | with `MAYA_GLM_SCORE=<c>` (score `--tokens` after the first c): `1` reads those c tokens through the prompt path instead of one by one, so a long context is scored in minutes (debug: comparing KV-cache formats at long context) |
| `MAYA_GLM_MULTI_BENCH` | off | `<n>,<prompt>,<tokens>` (with `MAYA_GLM_SEQS` >= n, `--glm-pack`, no `--serve`): n prompts, then that many greedy tokens each - sequence 0 alone through the token path, then all n through the pipelined decode; positive `MAYA_GLM_SPLIT_ROWS` groups up to k sequences per part and reports `split-rows`; prints both rates and how many of sequence 0's tokens agree (all of them with `MAYA_GLM_CPU_LANE=0`). `MAYA_GLM_MULTI_DUMP=<file>` also writes every sequence's tokens there, a line each (then the lone run's), to compare two builds or settings token for token |
| `MAYA_GLM_THINK_RESERVE` | 1024 | the server: a thinking block is closed early enough to leave the answer at least this many tokens, or a quarter of the request's `max_tokens` if that is more (the config's `"thinking_budget"` is lowered per request to fit) - a request whose `max_tokens` runs out while the model is still thinking otherwise ends with no answer at all |
| `MAYA_ENGINE_STALL_S` | 90 | the server: an engine silent this long that also used no CPU, disk or GPU in that time is stuck - it is ended, the request gets an error and the next request starts it again (a silent engine that is working is never ended); `0` = off |
| `MAYA_WATCHDOG_S` | 180 | the engine: a request that finishes no prompt layer and no token for this long has hung, even with a thread spinning (#40). The engine writes where it stopped to its log (on Windows also `maya-stall-<pid>.dmp`, every thread's stack: attach both to an issue) and ends; the request gets an error and the next request starts it again. `0` = off |
| `MAYA_HTTP_BACKLOG`, `MAYA_MAX_BODY_MIB` | 256, 256 | the server: connections that may wait to be accepted, and the largest request body in MiB (a larger one gets a 413 before it is read) |
| `MAYA_ENGINE_QUIT_S` | 50 | the server: how long the engine gets to end (and keep its conversation, with `MAYA_GLM_SLOT_KEEP`) after a stop - SIGTERM and SIGHUP are handled like Ctrl+C, a second one is ignored (a second Ctrl+C ends the server at once); a supervisor's own grace (llama-swap's `unloadTimeout`) should exceed it |
| `MAYA_ENGINE_WRAP_S` | 40 | the server: on a stop, a request in progress is wrapped up instead of dropped - its thinking is closed at once and it answers for up to this long (then it is cut), so the client gets an answer and the turn ends; requests still queued are refused. Keep it plus the engine's end inside the supervisor's grace (llama-swap's `unloadTimeout`); `0` = cut at once |
| `MAYA_FAIR_SLICE_S` | 0 (off) | the server: an answer that has been decoding this many seconds while another request waits gives the engine to it and continues afterwards, from where it was (the engine keeps its state with what it wrote: one token is read again). Concurrent chats and an agent's side requests no longer wait for a whole answer (4x RTX 4090, a short question during a 3000-token answer: first token after 34 s -> 13-17 s at 15-20). Each switch sets a conversation aside on disk (0.1-1 s up to 100K tokens on a SATA SSD array), so a slow disk wants a longer slice |

**The expert profile** (`data/expert-profile-glm.bin`). At start the engine fills VRAM with the experts a profile ranks
first. That profile ranks every (layer, expert) pair, and its order also decides how many VRAM slots each layer
gets - except the MTP draft layer on a split, which runs on every drafted token and so keeps at least an equal share
of its card's slots. Setup passes the shipped one, `data/expert-profile-glm.bin`, to every model (`--expert-profile` in the config's
`"args"`; a config written before it gets it at its next start, so an update needs no setup). It comes from the
router's routing, so it serves every quant of GLM-5.3-Flash: a routing trace of Maya-L with the MTP draft block
running, over 66 requests (code, math with thinking, chat, eight languages, tool calls, long documents). Maya-L on an
RTX 3090 + 8x RTX 5060 Ti, a fresh start: decode 40.7 tok/s on average against 35.4 without a profile, 96% of the
routes served from VRAM at first against 89%. Nothing is learned across starts unless you ask for it: while it runs,
the engine still moves the experts you use into VRAM, but the next start begins from the profile again. These engine
arguments go in the config's `"args"`; a setup again keeps them:

| Argument | What it does |
| --- | --- |
| `--expert-profile FILE` | start from this profile instead of the shipped one |
| `--expert-profile-save FILE` | save what the expert tiers learned as a profile: the experts in VRAM, then the routing counted, then the profile it started from. Written on exit and every `--expert-profile-save-every MIN` minutes between requests (default 10; `0` = on exit only). Point `--expert-profile` at it to start from there (a file of your own, not the shipped one: a setup again keeps it) |
| `--dump-routing FILE` | write every route (prompt and answer) as a routing trace, `tools/make_profile.py`'s input |

To build a profile for your own use: start the model once with `--dump-routing /tmp/trace.bin`, send it requests
typical of your use (code, docs, chat, languages), stop it, then run
`python tools/make_profile.py --glm --no-base /tmp/trace.bin --out my-profile.bin`. The tool also reads the engine's
text records - a usage file (`MAYA_GLM_USAGE`), a pack's `expert_counts.txt` - and `--equal` weighs several inputs
the same. Without any profile, the engine reads a pack's own `expert_counts.txt` / `expert_prior.txt`
(`tools/glm_expert_prior.py`), else fills in expert order.

**Prompt chunks** (`--prefill`). The engine reads a prompt in chunks, and every expert a chunk routes to is
copied into VRAM once per chunk, so larger chunks read long prompts faster; a chunk's buffers are borrowed from the
expert pool while the prompt runs. The setup writes `--prefill auto` into the config's `"args"` (a setup again keeps
what you changed there):

| Argument | What it does |
| --- | --- |
| `--prefill auto` | the largest chunk, in steps of 256 tokens, whose buffers each card's expert pool can lend - at most 90% of its slots (85% when under 90% of the experts' bytes are in VRAM or the pinned RAM tier), always keeping a route's experts and the spares - up to 32768 tokens on one GPU and 8192 on a split - each the faster there (one V100, 26K-token prompts: 559 tok/s at 32768, 367 at 8192; two V100s: 709 at 8192, 551 at 32768). On a split every card reads the same chunks, so the smallest card's sets it, and the log names a card that holds it below the first card's |
| `--prefill N` | chunks of N tokens (`32768`, say), halved until each card's pool can lend them - only a route's experts and the spares must stay; `0` = token by token. On a split with 96 GB of RAM or more the setup points to `32768` |

One card streams nearly every expert over PCIe for each chunk, so fewer, larger chunks pay off on long prompts. On a
split, a prompt that fits one chunk runs the cards one after the other, while two or more keep both busy.

## The expert pool's size

What each GPU's VRAM holds besides the model's dense weights, its context state and the drafter goes to experts (the
expert pool), less a reserve for the driver, the sampler and what is allocated later (the verify windows' graphs,
captured on first use). The setup writes `--expert-cache auto` into the config's `"args"`; a number of your own
there, and the reserves below, go there too, and a setup again keeps them (`MAYA_GLM_EXPERT_CACHE`,
`MAYA_GLM_RESERVE_MB` and `MAYA_GLM_RESERVE_LATER_MB` set the same from the environment; the config's args win):

| Argument | What it does |
| --- | --- |
| `--expert-cache auto` | the default: the pool takes the free VRAM less the reserve. Once that budget is written, the free VRAM is read again: under Windows an allocation counts only once it is touched, the free figure read before it can be about a gigabyte too high, and a card filled past it pages GPU memory to system RAM instead of failing (decode several times slower; a request can stall for good). Short of the reserve, the pool gives back the difference and is checked again; an allocation the driver refuses (Windows also charges VRAM to the commit, RAM + page file) ends it at what the driver gave. Not checked: an APU (Ryzen AI Max: its GPU memory is the system's), and a budget well beyond all the pool can take (every expert with the spares and a prompt's room - it cannot fill the card). `MAYA_GLM_POOL_CHECK=0` skips the check |
| `--expert-cache N` | your own size: the first GPU's pool holds at most N experts (N of its largest expert slots); a split's later GPUs stay on `auto`. It is not checked once written: a card it fills too far is only reported at start |
| `--vram-reserve-mib N` | the VRAM left free beside the pool: 700 MiB by default (1024 on a Ryzen AI Max APU under Linux). On a small card where that would leave the pool less than a route's experts and the spares in every layer, `auto` lowers it as far as 300 MiB and says so; a number set here is kept. The setup takes it too (`--vram-reserve-mib N`) |
| `--vram-reserve-later-mib N` | the same on a split's later GPUs (default: `--vram-reserve-mib`'s) - a card that drives no display needs less |

At start the engine log says how much VRAM each GPU has free with everything loaded (`maya serve: CUDA0: ... MiB of
VRAM free with everything loaded`). Under 256 MiB (512 on a split's later GPUs) it adds `LOW` and the reserve that would
leave room. The engine only says so: raise `--vram-reserve-mib` (`--vram-reserve-later-mib`), lower `--max-context`, or
close other programs that use the GPU. A drafter's verify windows allocate on first use, after the start: an RTX 3090
with DFlash2 at 32K had 431 MiB free at start and 77 MiB after the first request - fine on Linux, where a full card
fails an allocation instead of paging, but on Windows a reason to raise the reserve.

## Repeatable answers

A greedy request (`temperature` 0) returns the same tokens after a restart only if every expert is computed by the same
path each time. With the CPU lane on it is not: a RAM-tier expert goes to the lane or over PCIe to the GPU depending on
where it sits, and the two paths round differently - the GPU's kernels quantise the activations in blocks of 32 values,
ggml-cpu's dot (the lane) in blocks of 256, and an expert's output differs by about 2% (relative L2, median of 150 lane
calls over 42 layers of Maya-S, against a double-precision reference the lane is 1.7% off and the GPU 1.1%). The router's
top-8 amplifies that: once the lane computes one expert that the GPU did before, within a few tokens a quarter or more of the layers
pick a different set of 8 experts, and the greedy tokens part within some tens of tokens. What changes which
experts the lane computes from one start to the next:

| What | Why it differs | What fixes it |
| --- | --- | --- |
| the VRAM pool | `--expert-cache auto` takes the free VRAM less the reserve, so a few hundred MiB used by another program is some hundred experts fewer (the start's `tiers warm: N experts` and `only ... MiB free once the expert pool ... GB less` lines; 9,453-9,488 experts in five starts of one PC) | `--expert-cache N` in the config's `"args"` (they win over `MAYA_GLM_EXPERT_CACHE`), N a little under what `auto` gave |
| the lane's split | timed at start: where an expert costs about the same on the CPU and over PCIe (0.137 and 0.141 ms), starts land on neighbouring splits (`... 3 3 3` and `... 3 3 4`) | `MAYA_GLM_PCIE_SHARE` or `MAYA_GLM_CPU_PLAN` |
| the split timed again | `MAYA_GLM_CPU_RETIME` plans it again from decode's own timing | `MAYA_GLM_CPU_RETIME=0` |
| background moves between the tiers | they go live when their copies finish | `MAYA_GLM_PROMOTE=0` |
| the RAM tier | sized from the free RAM unless `MAYA_GLM_RAM_GB` is set | set it |

With those fixed (RTX 5090, Maya-S-v2-IQ2_XXS, a 42 GB RAM tier, `--expert-cache 3250`, `MAYA_GLM_PROMOTE=0`,
`MAYA_GLM_CPU_RETIME=0`, `MAYA_GLM_CPU_PLAN=001122333`; six prompts of 256 tokens, greedy, five engine starts) all
70 pairs of token streams were identical, also with 300 MiB less VRAM free at the start (105 of 105 pairs against the
first five starts) - decode 18.31 tok/s against 18.26 with the defaults (medians of five). With the same 300 MiB taken and
`--expert-cache auto`, none of 105 pairs matched. With the defaults, 17 of 70 pairs of starts matched.

`MAYA_GLM_CPU_LANE=0` takes the question away: every expert is computed by the GPU's kernels wherever it sits, so the
tokens are the same after every restart and with any RAM tier (identical for 26 and 42 GB here), at 4% (26 GB) to 10%
(42 GB) lower decode speed. Neither way makes two different settings agree: a different pool, RAM tier or split gives
different, equally valid tokens, and so does another build or driver. Measured on one PC; DFlash2, MTP, several GPUs, Linux
and AMD were not tried.

Drafting itself does not change the tokens: with a drafter loaded, decoding with it (DFlash2, the MTP block) and
without it (`"strata_tune": {"spec": 1}`) gave the same logits bit for bit - a verify window's rows are what the token
path computes (one V100, Maya-L, `MAYA_GLM_CPU_LANE=0` and the tiers' moves off, 48 tokens with 8-10 DFlash2 and 46
MTP verify rows; with penalties too). Loading a drafter does change them: it takes VRAM, so other experts sit in VRAM,
and the prompt path computes a layer's experts in sets by where they sit - the resident ones in one product, the staged
ones in groups, the rows summed as each set comes in - which rounds otherwise. With and without DFlash2 loaded the
tokens parted after 17 of 48. The decode does not depend on where the experts sit: with the prompt read token by token
(`MAYA_GLM_NO_PREFILL=1`, far slower prompts) both gave the same logits.

## Repetition penalties

A request's `repeat_penalty` (or `repetition_penalty`), `frequency_penalty` and `presence_penalty` work as in
llama.cpp, over the last `repeat_last_n` (or `penalty_last_n`) tokens - 64 when the request names no window, 0 for no
penalties at all, -1 for the whole context; at most 4096 tokens are counted, so -1 means the last 4096 on a longer
context - counted over the prompt's end and then what the answer has written so far. A token among them has its logit
divided by the repeat penalty (multiplied when the logit is negative), lowered by `frequency_penalty` for each time it
occurs there and by `presence_penalty` once. A run config's `sampling` block can set them for every request.

GLM applies them on every decode path: greedy requests too (`temperature` 0 picks the best token after the
penalties; the loop guard only catches a stretch repeated word for word, a penalty also pushes back on an answer that
keeps rewording itself), with drafting (MTP, DFlash2: each pick of a verify window counts the tokens before it, the
drafts among them, so a draft is kept only where it is the penalized pick - drafting does not change what is
penalized) and with several requests at once. DeepSeek-V4.1 applies them the same way, its DSpark drafts' verify
included (a sampled request's drafts are then kept by the speculative-sampling rule over the penalized distribution). A
repeat penalty of 1 and frequency and presence penalties of 0 are off: such a request decodes exactly as one without
them.

## Several requests at once

`--parallel N` (1-8, default 1). `./maya.sh --parallel 4` (Windows:
`START-MAYA.bat --parallel 4`) keeps it in the config's `"args"`. Maya then answers up to N requests at the same time,
and the ones after them wait their turn. The dashboard shows how many answers are being written, and `/v1/status`
reports N as its concurrency.
- **One GPU:** a token of each request goes through the model together. Every dense weight (the attention
  projections, the shared experts, the dense FFN layers and the output head) is read once for all of them, while the
  attention and the routed experts run per request. Each request gets exactly the tokens it would get alone (greedy,
  CPU lane off: checked with 2 and 4 at once). Real chat questions on a V100 32 GB that streams most of Maya-S's
  experts from RAM: 21.2 tok/s in all for two at once against 15.0 one after another (x1.41), x1.35 for four. Each
  conversation brings its own experts, so on a card that holds only part of them they share its VRAM; the more of the
  model VRAM holds, the less they compete. With every expert in GPU memory (a Strix Halo with Maya-S): x1.67 for two at
  once and x2.02 for four.
- **Two GPUs or more:** by default, each card works on a different request, one behind the other. With `MAYA_GLM_SPLIT_ROWS`, each part instead batches groups of requests, with several groups pipelined across the split.
- **A request alone:** with split rows enabled on the tested four-GPU Maya-M setup, decode and prefill stayed within ±5% of the legacy path. Where speculative decode is supported, the existing single-request speculative branch takes precedence.

At four requests on the same host, `k=1` was within run-to-run noise of legacy, while `k=2` and `k=4` reduced throughput. These short prompts did not fill the configured 256K contexts, so this measures decode throughput rather than full-context capacity.

Each extra request keeps its own context state in VRAM (the attention caches, the recurrent states and a snapshot),
taken from the expert pool at the start, so a larger N leaves fewer experts in VRAM; the log's `glm multi:` line gives
the size. With N above 1 the engine decodes without a [DFlash2 drafter](#drafting-with-dflash2) (the log says so).

## Drafting with DFlash2

`./maya.sh --setup --draft dflash2` downloads the drafter (0.7 GB,
[on Hugging Face](https://huggingface.co/Anbeeld/GLM-5.3-Flash-DFlash2-GGUF); its license is CC BY-NC-ND 4.0:
non-commercial use only) and passes it as `--dflash GGUF` in the config's `"args"`; NVIDIA GPUs, one request at a time
(`--parallel 1`). The setup offers it on RTX 30 / A100 and newer (sm_80): on Tesla V100s it decoded slower than the MTP
block, on one GPU and on two. How much it gains also depends on the PC - where most experts come from RAM, checking
the drafts can cost more than they save (an RTX 3090 + 3060 with Maya-M, #127) - so the setup measures it there
([the speed calibration](#the-speed-calibration)).

A DFlash2 drafter made for GLM-5.3-Flash - a small block drafter that reads
the model's hidden states at five of its layers and drafts up to 7 tokens in one pass, with the model's own embedding
and head - takes the place of the MTP block: with it the engine does not load the MTP block. Each round it drafts a
block, one verify window checks the drafts, and the longest right prefix is kept with the model's own next token - the
same tokens as decoding one at a time.

On three GPUs or more the window is a pipeline: the GPUs work on consecutive tokens of the window at once (one host
thread a GPU issues the work), the drafter drafts while the window's first token is already running, each token's pick
is read as it leaves the last GPU, and the tokens after a wrong draft skip their work (their experts are not fetched or
computed on the CPU either) while the next round starts behind them. The slowest GPU paces every draft, so the split
search balances the GPUs' time there, not only their VRAM (`MAYA_GLM_SPLIT`, above).

On one GPU, and on two, the window is a batched pass instead, built for a decode whose CPU lane sets the pace: its
tokens in two groups whose layers interleave (the CPU answers one group's experts while the GPU runs the other's
attention), each layer's experts read once for the tokens that share them and split between two CPU pools - a second
one on the other socket with one GPU, the other GPU's pool with two - and the window's RAM-tier experts all on the
CPU. Two GPUs check at least one draft a round and up to four while the product of their confidences stays at 0.9
(there a window of two costs only ~1.25 lone tokens); one GPU checks the drafts above confidence 0.95, and none below
it (~1.55 lone tokens).

| Argument | What it does |
| --- | --- |
| `--dflash GGUF` | the drafter (`MAYA_GLM_DFLASH` the same); it sits on the GPU with the model's head |
| `--spec T` | a round's window at most T tokens: the token and up to T - 1 drafts (default: the drafter's block, 8; four drafts on two GPUs) |
| `--spec-min-p P` | the drafts stop at the first whose selector confidence is below P (default 0.5 on three GPUs or more, where a wrong draft costs little; 0.95 on one GPU; on two GPUs the product of the confidences instead, below) |

A request can set its own window and cut (for tuning, without a restart): `"strata_tune": {"spec": 6,
"spec_min_p": 0.5}` (and `"spec_n_min"`: drafts kept whatever their confidence, `"spec_cum"`: a cut on the product of
the confidences - two GPUs default to 1 and 0.9; `MAYA_GLM_DFLASH_MIN` / `_CUM` set them for every request);
`"spec": 1` decodes without drafting.

Maya-L, greedy answers to five prompts - explanation, code, a list, a short story, a question about a
20,000-character document - in tokens/s, each pair measured back to back. Nine GPUs (an RTX 3090 and eight RTX 5060
Ti, the CPU lane on):

| | explanation | code | list | story | document | all |
| --- | --- | --- | --- | --- | --- | --- |
| the MTP block (two runs) | 43.7 | 44.7 | 46.1 | 38.3 | 42.0 | 42.6 |
| DFlash2 (three runs) | 60.1 | 76.2 | 90.8 | 48.0 | 51.8 | **61.4 (+44%)** |

One RTX 3090 (the MTP block is not loaded on one GPU: plain decode), and the 3090 with one 5060 Ti:

| | explanation | code | list | story | document | all |
| --- | --- | --- | --- | --- | --- | --- |
| 3090, plain decode (two runs) | 16.4 | 15.7 | 15.6 | 16.4 | 14.0 | 15.8 |
| 3090, DFlash2 (two runs) | 19.4 | 20.6 | 20.2 | 17.1 | 16.4 | **18.8 (+19%)** |
| 3090 + 5060 Ti, the MTP block (three runs) | 20.9 | 21.3 | 21.9 | 16.7 | 19.4 | 19.7 |
| 3090 + 5060 Ti, DFlash2 (three runs) | 23.2 | 24.2 | 25.8 | 19.4 | 19.2 | **22.1 (+12%)** |

### The speed calibration

When the setup drafts with DFlash2 - and once, at the first start after an update, for a config that already drafts
with it - Maya measures it on your PC against the MTP block (one GPU: against decoding without drafting) and keeps
the faster. It loads the model twice, the same config without and with `--dflash`. Each time it reads a long document
(about 8,000 tokens, the depth of a real conversation) and answers once to warm up - not counted - then answers fresh
questions about it, sampled as the dashboard samples (the same questions and seeds both times), until about 1,500
tokens are written. The sustained speed - all those tokens over all the time spent writing them - decides: DFlash2
stays only when it is more than 5% faster. It takes 5-20 minutes and shows each step as it goes. The result is kept in
the config (`installer.speed_calibration`); `./maya.sh --speed-calibration` measures again (new GPUs, more RAM).
DeepSeek-V4.1's DSpark drafter gets the same check against decoding without drafting (10-30 minutes).

You can also choose yourself and skip the measuring: the setup asks first (m = measure, y = draft with the drafter,
n = no drafting), or pass `--drafting on` / `--drafting off` (`--drafting measure` measures; with `--yes` the setup
measures). Your choice is kept in the config like a measured result, so a start does not measure by itself.
`--no-speed-calibration` is the same as `--drafting on`.

## DeepSeek: the PCIe lane

While answering, a token's routed experts that VRAM does not hold were all computed on the CPU (the CPU lane). Now,
as for GLM, some of them can go to the GPU instead: read into pinned memory, copied into a few staging slots on the
GPU and computed there, while the CPU computes the rest - so both finish together. How many is decided for every token
from this PC's own measured times: the CPU's time for an expert of the layer's format, the time to read and copy one
to the GPU, the GPU's own time, and whether the experts are in RAM or must come from the SSD. A PC with a strong GPU
link and a weaker CPU streams more; a fast CPU with a slow link streams little or nothing (its GPUs then keep their
slots for experts). Each GPU streams its own layers' experts over its own link; an integrated GPU (Strix Halo) reads
them from pinned memory where they lie when that measured faster. The times are measured at the first start and kept
in `lane-times.txt` beside the model's pack; later starts check them in a few milliseconds, and decode times both lanes
again under load. Both lanes share the PC's memory, SSD and cores, so while answering the first tokens it also checks
that streaming really is faster here, taking turns with the CPU lane alone (separately for tokens whose experts are all
in RAM and tokens that read some from the SSD); where it is not at least 3% faster, it stops streaming and the times
file keeps that verdict. It applies while answering (one token at a time), not while reading the prompt.

| Setting | Default | |
| --- | --- | --- |
| `MAYA_DS4_STREAM` | on (1) | `0`: every expert VRAM does not hold on the CPU, as before |

The engine's decode summary (`streamed to the GPU: N`) and the `pcie=` key of its STAT lines (experts a token
streamed) show it at work; the start's log prints each format's times and how many of a token's experts it streams.

## DeepSeek: long prompts read without the second half's work

DeepSeek-V4.1-Flash's layers 20-39 see the earlier tokens of a prompt only through what layer 20 keeps of them (its
compressed KV and index keys), and each of them looks back one window of 128 tokens. So for the answer and for what the
decode keeps, those layers need their full work only on the last ~2,700 tokens of a prompt: before that, layer 20
computes just what the later layers read, and layers 21-39 do nothing. The answers are the same (up to the order the
experts' outputs are added in), long prompts are read faster, and the experts are read as before. DeepSeek's own
serving goes further and runs layers 20-39 on the last 128 tokens only, which shifts the answers a little; `2` does
that. It applies to prompts of ~3,700 new tokens or more, on one GPU (several GPUs read prompts as before).

| Setting | Default | |
| --- | --- | --- |
| `MAYA_DS4_PROMPT_SKIP` | on (1) | `0`: every layer on every token of a prompt, as before; `2`: DeepSeek's own (the last 128 tokens only) |
| `MAYA_DS4_DSPARK_CUM` | 0.5 | DSpark drafting: a draft is verified only while the chance that it and every draft before it are kept (the product of their confidences) stays at least this, so a window whose later drafts would likely be lost is not verified whole. The same tokens; a Radeon 8065S with Maya-S decoded 12.04 -> 12.82 tok/s (+6.5%). `0` = off (only each draft's own floor, `--dspark-conf`); a request's `strata_tune` `spec_cum` sets it for that request |

## DeepSeek: skipping light experts the SSD would read (experimental)

When DeepSeek-V4.1-Flash's experts do not fit in VRAM and RAM, a token waits for the SSD to read the experts it was
routed to. `MAYA_DS4_DISK_SKIP_W` lets a token go without a routed expert that weighs little in its route **and** is
not in RAM at that moment; experts already in VRAM or RAM are always computed. **It changes the answers**, is off by
default and is experimental. It applies while answering (and to a draft's check), never while reading the prompt.

| Setting | Default | |
| --- | --- | --- |
| `MAYA_DS4_DISK_SKIP_W` | off (0) | leave out a routed expert below this share of its token's route weights when its pages are not in RAM (the page cache or the RAM tier; on Windows: sampled pages of the engine's working set). The weight it carried is lost. `0.10` changes the answers less, `0.14` leaves out more |
| `MAYA_DS4_DISK_SKIP_FROM` | 0 | the first layer it applies to |
| `MAYA_DS4_DISK_SKIP_AUTO` | 0 | `1` (with `MAYA_DS4_DISK_SKIP_W`): skip only while the model is cold, so warm answers are the unmodified model's. It starts on after a start, and switches off - in the middle of an answer too - once the share of experts read from the SSD stops falling (a level that depends on the PC), or after `MAYA_DS4_DISK_SKIP_BUDGET` (512) tokens. It switches on again only between requests, never during an answer and never just because a conversation moves to a new topic: when most of the experts the last answer left in RAM are gone (an idle while other programs took the RAM) |

It helps where answering waits on the SSD - a PC whose RAM holds only part of the experts, most on a first answer
about a topic - and does nothing where the experts sit in VRAM and RAM. Without `MAYA_DS4_DISK_SKIP_AUTO`, where the
RAM cannot hold the experts a conversation uses, it keeps skipping on later answers too. The engine's decode summary
(`decode left out ...`, `DISK_SKIP auto: ...`) and the `skipped=` / `skip_on=` keys of its STAT lines show what it
did.

## DeepSeek: explicit RAM residency

These independent settings are off by default. They are for Linux hosts with enough physical RAM to hold the
chosen model, its Engram tables and the other programs using the machine. They do not change tensor values or
skip any experts, and do not promise a speed increase over a warm page cache.

| Setting | Default | Effect |
| --- | --- | --- |
| `DS4_ENGRAM_RAM` | `0` | `1`: map and lock both complete Engram tables before serving. The reader copies rows from the locked mappings, including FP8 scales; an unregistered range fails instead of reading the file. Supports Q6_K, MXFP8 and native FP8 + separate scales. Linux; explicitly requesting it on Windows fails. |
| `DS4_REQUIRE_ALL_RAM` | `0` | `1`, with `--mmap-experts off`: copy every CPU-lane GGUF expert to anonymous RAM. Do not reduce the selection to the available-memory estimate or `DS4_COPY_GB`; insufficient allocation or an incomplete selection fails startup instead of silently leaving experts in their files. |

The second option applies to the GGUF expert-copy path. It does not lock the experts: **locked Engram tables and
CPU experts copied to RAM are different guarantees**. Ordinary expert RAM can still be swapped by the OS. Budget
the RAM, avoid swap pressure, and check the model process's `VmSwap` if the deployment requires zero swap.
The original automatic / file-backed modes, Windows and other models keep their existing behavior when these
options are unset or `0`. Restart the model after changing either option.

For an existing DeepSeek config, retain its other fields and add to `env`:

```json
{"DS4_ENGRAM_RAM": "1", "DS4_REQUIRE_ALL_RAM": "1"}
```

Also retain `--mmap-experts off` in the engine's `args`. Neither setting changes context length, DSpark, expert
precision, the GPU selection or the VRAM reserve.

**Locked-memory limit.** MXFP8 / native FP8 tables for 1017 occupy about 202.76 GB decimal; Q6_K depends on the pack.
The process's soft and hard `RLIMIT_MEMLOCK` must cover the page-rounded mappings. Set the limit for the account
or service that actually starts Maya (for systemd, the user manager's hard limit matters as well as the service's
`LimitMEMLOCK`). An unlimited limit does not supply RAM. A failed `mlock`, invalid range or unsupported table type
stops initialization with the reason; the engine does not silently fall back to disk.

**NUMA and cache pressure.** Locked file mappings can reuse pages already on one socket even with an interleave
policy. On a multi-socket host inspect `numastat -p PID`, `/proc/PID/numa_maps` and each node's free memory. Leave
headroom on every node. Unused tmpfs copies of another model consume RAM and possibly swap; they are not ordinary
reclaimable disk cache. These settings do not alter global cache limits, delete files or move pages between nodes.
Do not truncate or replace a mapped weight file while its model is running.

**Verify the result.** Look for `Engram RAM ONLY` and the count of experts copied to RAM in the engine log.
On Linux, check that the relevant table ranges in `/proc/PID/smaps` have `VmFlags: ... lo`, `Rss == Size`, and
`Swap: 0`; check the whole process's `VmSwap` separately. The expert-copy count must equal the CPU-lane expert
count. GPU-resident experts are separate. Do not infer these guarantees from total machine `MemAvailable` alone.

A small model-free regression test is available as `python tools/test_ds4_engram_ram.py` on Linux with a C++ compiler.
It checks file/RAM byte agreement, reads after unlink, split FP8 scales, MXFP8 extraction, bounds, and failure when
the memlock limit is zero. Run it as an ordinary user without `CAP_IPC_LOCK`.

## DeepSeek: the CPU lane's threads

DeepSeek's engine computes the routed experts that are not in VRAM on the CPU (the CPU lane), a thread per physical
core; a core's second thread (SMT) is left to the disk and Engram readers. Where the PC offers another way - the
performance cores alone on a hybrid CPU (Intel's P- and E-cores), the first socket's cores alone on two sockets or
more (the experts read in place) - a start times each, once the experts are placed, on the same calls a decode step
makes (several rounds, in turns), and keeps another way only when it is more than 5% faster. The choice never changes
an answer, only the speed; the log says what was measured (`ds4: CPU lane: node, 22 threads (measured, ...)`). It
needs at least 8 experts of one layer on the lane (otherwise the lane hardly runs). Only the CPUs the engine may run
on are used (`taskset`, a container's cpuset).

| Setting | Default | |
| --- | --- | --- |
| `--cpu-threads N` (the config's engine `args`) | a thread per core | at most N threads, a thread per core; nothing is measured |
| `MAYA_DS4_CPU_LANE` | measured | `cores`, `pcores`, `node` or `smt` (a thread per logical CPU): that way, nothing is measured |

## DeepSeek: a GEMV call's int8 inputs rounded in one launch

A GGUF pack's Q8_0 (and low-bit dense) GEMVs round their activations to int8 first, one small launch per job: a
decoded token runs a few hundred of them per GPU, each about 2 us of work behind a launch gap. With
`MAYA_DS4_Q8_BATCH=1` a call's jobs are rounded in one launch before its GEMVs, by the same per-block code, so the
int8 values, the scales and the answers are bit for bit the same. `MAYA_DS4_Q8_CHECK=1` checks the batched rounding
against the per-job one on every GPU at load (any difference fails the start).

| Setting | Default | |
| --- | --- | --- |
| `MAYA_DS4_Q8_BATCH` | off | `1`: a GEMV call's int8 rounding in one launch (the same bits) |
| `MAYA_DS4_Q8_CHECK` | off | `1`: the batched rounding checked against the per-job one on every GPU at load |

## DeepSeek: where the measured hipBLASLt picks are kept (AMD)

On AMD the engine measures hipBLASLt's kernels for each prompt GEMM shape at its first use on a GPU and keeps the
picks in `<pack>/lt-picks.txt`, so later starts reuse them. A pack whose files must not change (one shared between
installs, or checked by hash before each start) can keep them elsewhere.

| Setting | Default | |
| --- | --- | --- |
| `MAYA_DS4_LT_STORE` | `<pack>/lt-picks.txt` | the picks in this file instead; empty: none kept, each start measures its own |

## DeepSeek: the starting expert profile

The engine keeps how often each expert was routed in `<pack>/expert-profile.bin`: at a start the most-routed experts
go into VRAM and are read ahead into RAM first, and the counts grow with every answer (saved after each one). A new
pack has no counts yet, so the setup copies `data/expert-profile-ds4.bin` into it - DeepSeek's routing over a mix of
chat, code, multilingual text, reasoning and tool calls (`tools/ds4/profile.py`). Its counts weigh as much as about
2,000 tokens of answers, so what your PC routes soon outweighs them. A pack that already has a profile keeps its own;
an `--expert-profile` in the config's `args` (a file, or `none`) is used instead.

## DeepSeek: conversations kept (an agent's parent and workers on one model)

The engine holds one conversation: a prompt that continues it reuses its state, anything else starts over. An agent
whose parent and workers interleave - and whose prompts alternate between ~70k and ~185k tokens - then reads the
whole prompt again on every switch (257 of 632 requests in one log read from token 0; the 142 of them longer than 30k
tokens read at ~930 tok/s: a 70k-token prompt takes ~75 s, a 185k one ~200 s). With this on, the conversation in the
model is set aside when another one takes its place - its whole state: the position, every layer's 128-token window
ring, the compressors' partial groups and the four KV sources' compressed KV and index keys up to it - and a request
that continues a kept conversation takes it back, reading only its new tokens (a 12k-token conversation back in ~0.1 s
instead of ~12 s). Text only: a conversation that holds pictures is not set aside, and a prompt with pictures takes
none back. Off by default, so the engine is unchanged unless asked.

| Setting | Default | |
| --- | --- | --- |
| `MAYA_DS4_SLOT_KEEP` | off | `1`: keep the conversation when another one takes the model, and across runs in the folder; `0`: every prompt that does not continue the conversation in the model reads it from token 0, as before |
| `MAYA_DS4_KEEP_THINKING` | off | `1`: an earlier turn's reasoning (`reasoning_content`) stays in the prompt, where DeepSeek's own serving drops it at the next user turn. A tool-less chat needs this as well as `MAYA_DS4_SLOT_KEEP`: with thinking dropped, every turn renders a history that differs at each assistant turn, so its next turn continues no state at all (a Strix Halo, Ryzen AI Max: a 4,215-token follow-up behind a 4,197-token state read in 19.3 s; with both on, it reused 4,197 of 4,215 and read in 2.7 s). Requests that carry tools keep their thinking already (`serve/ds41_encoding.py`), so agent traffic is unchanged. The prompts are longer by the reasoning they carry - paid once as reuse, not as a read |
| `MAYA_DS4_SLOT_CAP` | 4 | the conversations kept in RAM (one is tens to hundreds of MB, the context's compressed KV being the most); past it the least recently used is written to the folder - RAM first, the folder past the cap - and read back from there (reading a 185k-token state off an NVMe: ~0.5 s, against ~200 s of re-reading); `0` = no conversations kept |
| `MAYA_DS4_SLOT_GB` | 16 | the folder's conversations' total size (GB, over their `.bin` files): at start, and after every conversation written to it, the least recently used in the folder is removed while they are over it - never the one just written, so a conversation bigger than the cap stays |
| `MAYA_DS4_SLOT_DIR` | `<pack>/slots` | the folder: a conversation is a `.bin` (its state) and a `.meta` (this model's identity, its tokens and its size), both written under a temporary name first |
| `MAYA_DS4_SLOT_MIN` | 1024 | the shortest conversation set aside (tokens): a shorter one is read again in about the time its slot would take |
| `MAYA_DS4_SLOT_DAYS` | 7 | a slot not used for this many days is removed at start; so is any file not this model's, another engine's, or incomplete. The engine's end (QUIT or the end of its input) writes the conversations in RAM to the folder, so a restart reads on from them (the server gives it `MAYA_ENGINE_QUIT_S`, 50 s, to do so) |

The client sends each assistant turn's `reasoning_content` back with the turn, as it got it: from `content` alone the
prompt differs from the state and nothing is reused - no error and no log line.

The engine's log says what happened (`ds4 slots: saved/took back/wrote ...`; the server shows those lines as
`[maya] sessions: ...`). A slot's identity is the engine's version, the model's geometry and files and whether the
DSpark blocks are loaded, so a slot from another model, another pack or an engine that changed the state's layout is
refused rather than read.

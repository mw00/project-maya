# Two-rig split for GLM-5.3-Flash (experimental)

One GLM-5.3-Flash model on **two machines** over an ordinary network: each machine runs one Maya process on its own
GPUs, and the model is split by layers at a boundary layer `K`:

- **front**: the embedding and layers `[0, K)`, the OpenAI-compatible server and the tokenizer;
- **back**: layers `[K, n_layers)`, the NextN (MTP) block, the output head and the sampler.

Only the residual of each token crosses the network: 4 × 4096 values. By default they cross as bf16, rounded to
nearest even and then packed losslessly to about 12 bits a value (~24 KB a token); `MAYA_GLM_LINK_BF16=0` sends
them as fp32 (64 KB, exact). 1 GbE is enough. Inside each machine the usual multi-GPU layer split still applies
(`--layer-split K:b1,b2,..`).

**It is off unless `--role` is given.** Without it nothing here runs and the engine is unchanged. It is
experimental: GLM-5.3-Flash packs only (not GLM-5.3), Linux, tested on CUDA (sm_86) only; see [Limits](#limits).

## When it helps

When neither machine can hold the model's experts on its own, but the two together can. GLM-5.3-Flash decodes at
the speed its expert reads allow: an expert in VRAM is read on the card, one in host RAM crosses PCIe on every use,
one on the SSD is slower again. Two machines give the model both machines' VRAM and RAM, at the cost of one network
hop per token.

Measured on two machines with 2× RTX 3060 12 GB each (PCIe 3.0, 1 GbE between them, one CPU with AVX2 and one
without; the cards power-limited to 130 W and 110 W), Maya-XS, `--prefill 2048`, one conversation, 128-token greedy
answers, on v1.1.75:

| | with MTP: short / summary / 2K-token prompt | without MTP: short / summary / 2K | expert reads from VRAM |
|---|---|---|---|
| one machine, 2× RTX 3060 | 18.8 / 16.3 / 13.5 tok/s | 12.8 / 11.8 / 11.2 tok/s | 60–71% |
| two machines, 2 + 2× RTX 3060 | 32.6 / 32.0 / 26.0 tok/s | 21.2 / 21.1 / 18.9 tok/s | 85–91% |

Over the link, 86–87% of the MTP drafts were accepted on the short prompts and 71% on the 2K prompt (one machine:
84–89% and 66%). Nothing was read from the SSD in either case. Earlier, on two machines with 4× RTX 3060 each and
Maya-S24, the split decoded at 38–42 tok/s against 42–45 tok/s for the same 8 GPUs in one machine: there the experts
fit in VRAM either way, and the hop costs the difference.

## Setup

Both machines need:

- the **same build** of Maya (both sides check the protocol version and the split's settings when they connect);
- the **same pack** (the same GGUF files and `pack/`; each process reads only the tensors of its own layers, so a
  copy of the whole model on each side is the simplest, but the other side's experts are never read). When they
  connect, both sides compare the pack's metadata - the geometry, a hash of the tensor index and the GGUF shards'
  names and sizes - and refuse a mismatch. The weights themselves are not hashed, so keep the files identical;
- the **same engine arguments and `MAYA_GLM_*` settings**, sampling arguments included, apart from `--role`,
  the local part of `--layer-split` and the memory each side has (`MAYA_GLM_RAM_GB`, the VRAM reserves,
  `--prefill`). What the link checks itself:
  - when the sides connect: `--max-context`, the boundary `K`, `MAYA_GLM_PREFILL_MIN_CHUNKS`,
    `MAYA_GLM_PREFILL_MIN` and the hop's format. A mismatch is refused;
  - at every request: that both decode the same way (with MTP or plainly, greedily or sampled) and continue
    from the same cached prompt;
  - at every prompt read: that both read the same piece of the prompt (settings that cut it differently -
    checkpoints, slots, `MAYA_GLM_PREFIX_CUT` - fail the request with a message that says so), and in which
    chunks: the smaller of what each side could lay out in its VRAM, or token by token if one side has no
    prompt path. Different `--prefill` sizes are fine for that reason;
- the same **link key file** (below).

### 1. The key file

The link authenticates every connection both ways with HMAC-SHA256 over a shared key. Make one key and copy it to
both machines, readable only by the user that runs Maya:

```sh
umask 077; head -c 48 /dev/urandom | base64 > ~/.config/maya/link.key
scp ~/.config/maya/link.key other-machine:.config/maya/link.key
```

Point both sides at it with `MAYA_GLM_LINK_KEY_FILE`. A key file that is set but missing or shorter than 32 bytes
stops the engine before it loads. With no key file the engine starts only when the link is on a numeric loopback
address (`127.0.0.1`; a name does not count), or with `MAYA_GLM_LINK_NO_AUTH=1` (a network you trust: the address
allowlist alone). The link is authenticated, **not
encrypted**: the residuals and the tokens cross in the clear.

### 2. The network

The back listens on one TCP port (7710 in the examples); the front opens two connections to it (data and control).
Open that port on the back for the front's address only. One of:

```sh
# ufw
sudo ufw allow from <front-ip> to any port 7710 proto tcp
# firewalld
sudo firewall-cmd --permanent --add-rich-rule='rule family="ipv4" source address="<front-ip>" port port="7710" protocol="tcp" accept'
sudo firewall-cmd --reload
# nftables (with an "inet filter" table and its "input" chain)
sudo nft add rule inet filter input ip saddr <front-ip> tcp dport 7710 accept
```

The front needs no open port for the link: it only connects out. Its server port (the API) is a separate matter.

The back accepts a front only from the addresses in `MAYA_GLM_LINK_ALLOW` (comma-separated IPv4; default: loopback
only). It listens on the `--link` host (`MAYA_GLM_LINK_BIND` overrides it). An empty host means loopback; it
listens on every interface only if you name `0.0.0.0` yourself.
If the front's server port is reachable from your network, give it an API key too (`MAYA_API_KEY`).

### 3. The back (start it first; it waits for the front)

```sh
MAYA_GLM_LINK_KEY_FILE=~/.config/maya/link.key MAYA_GLM_LINK_ALLOW=<front-ip> \
  build/maya --serve --glm-pack <pack> --max-context 65536 --expert-cache auto --prefill 2048 --kv-streaming on \
  --layer-split 24:35 --role back --link <back-ip>:7710
```

### 4. The front (the server, as in a normal Maya config)

Add to the config's `"args"` and `"env"`:

```json
"args": [ "...", "--layer-split", "24:12", "--role", "front", "--link", "<back-ip>:7710" ],
"env":  { "MAYA_GLM_LINK_KEY_FILE": "/home/you/.config/maya/link.key" }
```

`--layer-split K:b1,b2,..`: `K` is the boundary between the machines; `b1,b2,..` split each side's layers over its
local GPUs (the front's lie below `K`, the back's above). `24:12` on the front puts layers 0–11 on its first GPU and
12–23 on its second; `24:35` on the back puts 24–34 on its first GPU and 35–44 on its second. The back's last GPU also
carries the output head and the NextN block, so give it fewer layers. `--layer-split K` alone puts each side on its
first visible GPU.

### Choosing `K`

`K` is the only split decision; the model files are the same on both machines. Pick it from each side's memory.
In GLM-5.3-Flash, layers 3–44 are MoE layers with 288 routed experts each, and each side holds the experts of its
own layers:

| Pack | Experts per MoE layer | All 42 layers |
|---|---|---|
| Maya-S24 | 2.17 GB | 86.0 GB |
| Maya-XS | 1.57 GB (1.7–1.9 GB for the first and last layers) | 65.8 GB |

On each side the engine puts the most-used experts of its layers in its GPUs' expert pool (by the expert profile),
the next ones in pinned host RAM (`MAYA_GLM_RAM_GB`), and the rest on the SSD. Decode speed follows the share that
hits VRAM; reads from the SSD are what to avoid.

1. **A side's budget.** Each GPU's VRAM, less about 1.5–3 GB per GPU for the dense weights, the KV cache and
   scratch, plus the RAM you let that side pin. The back's last GPU also carries the output head and the NextN
   block (about 1–1.5 GB more).
2. **A side's need.** The front holds the MoE layers `3 … K−1` (`K − 3` layers), the back the layers `K … 44`
   (`45 − K`). Multiply by the per-layer size above.
3. **Pick `K`** so each side's need fits its budget. When both fit, balance the share of each side's experts that
   sits in VRAM, roughly in proportion to each side's VRAM.
4. **Check it.** Start both sides and read each one's start-up lines: `VRAM before the expert pool`,
   `layers [a, b) expert pool X GB, N slots/layer` and `RAM tier Y GB pinned`. During a request,
   `glm stat ... | disk 0.00 reads/tok` means nothing is read from the SSD.

Examples:
- **Two equal machines:** `K = 24` is about even. With Maya-XS, the front has 21 MoE layers (~33 GB) and the back
  21 layers (~34 GB) plus the head.
- **4× 12 GB next to 2× 12 GB:** give the bigger machine about two thirds of the MoE layers. As the front that is
  `K ≈ 31`; as the back, `K ≈ 17`.

Requests are bracketed (begin / end / ack), so a cancelled or failed request leaves both sides in step. If the link
drops, the front reconnects at its next request; the back stays up and accepts a front again
(`MAYA_GLM_BACK_STAY=0`: it exits instead).

To stop the back, send it `SIGTERM` or press Ctrl+C. (It takes no commands of its own, and while it stays up it
ignores a front's `QUIT`.) It fails the request in flight, accepts no other front and exits; a second signal ends
it at once.

## Link settings

| Setting | Default | Meaning |
|---|---|---|
| `--role front\|back` (`MAYA_GLM_ROLE`) | off | turn the split on |
| `--link host:port` (`MAYA_GLM_LINK`) | – | the back listens there, the front connects there |
| `MAYA_GLM_LINK_KEY_FILE` | – | the shared key (≥ 32 bytes), the same file on both sides |
| `MAYA_GLM_LINK_NO_AUTH` | 0 | 1: no key on a non-loopback link (the allowlist alone) |
| `MAYA_GLM_LINK_ALLOW` | loopback | the back: IPv4 addresses a front may connect from |
| `MAYA_GLM_LINK_BIND` | the `--link` host | the back: the address it listens on |
| `MAYA_GLM_LINK_BF16` | 1 | 1: the residual crosses as bf16 (rounded to nearest even: not exact, see [Checking it](#checking-it-single-process-vs-two-rigs)); 0: fp32, exact, ~2.7× the bytes |
| `MAYA_GLM_LINK_PACK` | 1 with bf16 | lossless packing of the bf16 rows (~12 bits a value) |
| `MAYA_GLM_LINK_WAIT` | 1800 s | the front: how long it waits for the back at start |
| `MAYA_GLM_RELINK_WAIT` | 5 s | the front: the time a command gives the reconnect after a link failure, all of it (connecting, authentication, the settings check); 1 s at least |
| `MAYA_GLM_LINK_DEAD_S` | 30 s | both sides: a peer that is gone without a word (its power, a cable) ends the request after this long (TCP keepalive and `TCP_USER_TIMEOUT`); 0: off |
| `MAYA_GLM_LINK_STALL_S` | 60 s | both sides: a peer that is connected but sends or takes nothing for this long (a stuck process) ends the request and the link; 0: off. Keep it under the server's `MAYA_ENGINE_STALL_S` (90 s) and the engine's `MAYA_WATCHDOG_S` (180 s), which end the engine instead |

## Speculative decoding over the link

The NextN (MTP) drafts work over the link, pipelined: while the back verifies the draft for one position, the front
already runs its layers for the next two (the confirmed token's and the draft's), so neither machine waits for the
other's whole step. The back sends each step's token and the next draft; the front sends its verdict and its rows in
the order the back reads them. A wrong draft costs the front one recomputed row, not a round trip.

A request is decoded this way when both sides would draft: the back has the NextN block (the default), neither
side runs with `MAYA_GLM_NO_SPEC=1`, and the request does not ask for a plain decode (`"spec": 1`). Otherwise both
decode it plainly, and a log line names the side that does not draft.
`MAYA_GLM_SPEC_WIN` (the verify window of a one-machine split) is not used over the link.

## Several conversations at once (`--parallel`)

`--parallel N` (2–8) works over the link when both sides start with it and with `MAYA_GLM_SPLIT_ROWS`, the same
values on both:

```sh
# the back's command line and the front's config ("args" and "env")
--parallel 4        MAYA_GLM_SPLIT_ROWS=4 MAYA_GLM_EVEN_GROUPS=1
```

- **What crosses the link.** The conversations decode in batched steps of up to `MAYA_GLM_SPLIT_ROWS` rows. The
  front runs a group's rows through its layers and sends one residual per conversation; the back finishes them and
  sends one token per conversation, each as soon as it is picked. While the back works on one group, the front
  runs the next.
- **Cut the requests into groups** (`MAYA_GLM_EVEN_GROUPS=1`). With one group the two machines take turns: the
  front waits while the back finishes the group. Cut evenly into two groups or more, one machine works on a
  group while the other works on the next. Measured on two machines with 2× RTX 3060 each (Maya-XS, 32K
  context, four conversations writing at once): 23.8 tok/s in all as one group of four, 38.9 tok/s as two groups
  of two; one conversation alone wrote 29 tok/s there.
- **Who decides.** Both sides run the engine's scheduler for several conversations, in step. The front decides
  what depends on its input, its timing or its own state, and tells the back: when it takes a new command and
  when a decode step, which conversation state a request goes to, and after each token whether the request goes
  on (a stop, a cancel, a pause, a forced token).
- **A request alone** still drafts with the pipelined MTP decode (above). From the second request on, all of them
  decode in batched steps, without drafts.
- **A new prompt is read whole** while the other conversations wait: `MAYA_GLM_PREFILL_GAPS`,
  `MAYA_GLM_PREFILL_INTERLEAVE` and `MAYA_GLM_MULTI_MTP` are not used over the link (a log line says so), and
  `MAYA_GLM_MULTI_BENCH` does not run with `--role`.
- **Settings both sides must share**, checked at connect: `--parallel`, `MAYA_GLM_SEQ_CTX`,
  `MAYA_GLM_SPLIT_ROWS`, `MAYA_GLM_EVEN_GROUPS`, `MAYA_GLM_MULTI_BOUNDARY`, `MAYA_GLM_MULTI_SPEC` and
  `MAYA_GLM_MULTI_ROOM`. A side that differs is refused with a message that names them. `--parallel` above 1
  with `--role` and without `MAYA_GLM_SPLIT_ROWS` does not start.
- **Stops.** A `STOP` for one request ends it on both sides, at its next token or in the middle of its prompt's
  read; the others go on. A request that is stopped before the engine takes it up, or while it waits for a free
  conversation state, ends without a token once it is taken up.
- **Failures.** A failure on either side, or a link that drops while requests run, ends every request that runs
  or waits with an error, and both sides reset the conversation states the engine holds; states kept in slots
  stay. The same happens to requests a front left behind when another front (or the same one, started again)
  connects. A link that is made again while nothing runs keeps the states. What each side still holds of a
  conversation is compared at its next request, as with one conversation: where they differ, both read the
  prompt from the start.
- **Memory.** Each conversation state takes VRAM on both sides, out of the expert pools (the start-up line
  `glm multi: ... sequence states` says how much), so more conversations mean fewer experts in VRAM.

## Checking it: single process vs two rigs

`tools/glm_split_check.py` runs the same build, pack and prompts once in one process (the reference) and once split,
and compares the greedy answers (token ids, with MTP on and off) and the teacher-forced logits (`MAYA_GLM_KL_DUMP`)
over each prompt plus the reference's answer (the prompt through the prompt path, the answer one token at a time).
Its four prompts are two short ones, about 2K tokens of code, and one with a 1.4K-token system prompt, whose read
the engine cuts at the system prompt's end (`--only NAME[,NAME]` runs some of them).

### The exact check (one machine, two GPUs)

On one machine with two GPUs or more, the split can run with exactly the reference's placement: the reference with
`--layer-split K` puts layers `[0, K)` on GPU 0 and the rest on the later GPUs; the split puts its front on GPU 0
and its back on GPU 1 over loopback. With the fp32 hop and the CPU lane off on every run, the split must be
**bit-identical** to the reference:

```sh
export MAYA_GLM_LINK_BF16=0 MAYA_GLM_CPU_LANE=0
ENG="build/maya --glm-pack <pack> --max-context 8192 --expert-cache auto --prefill 2048 --kv-streaming on"
python3 tools/glm_split_check.py prompts --pack <pack> --dir check
python3 tools/glm_split_check.py run --mode ref --dir check -- $ENG --layer-split 24
CUDA_VISIBLE_DEVICES=1 python3 tools/glm_split_check.py run --mode back --dir check -- $ENG \
    --layer-split 24 --role back --link 127.0.0.1:7711 &
CUDA_VISIBLE_DEVICES=0 python3 tools/glm_split_check.py run --mode front --dir check -- $ENG \
    --layer-split 24 --role front --link 127.0.0.1:7711
wait
python3 tools/glm_split_check.py compare --dir check --exact      # exit code 0 = PASS
```

Give each side enough `MAYA_GLM_RAM_GB` that no expert is read from the SSD (as the reference): an expert read from
the SSD instead of the RAM tier is rounded differently.

Several conversations at once are a step of their own (`--parallel N`): the engine runs as a server under the
script and `N` conversations over the same prompts in three turns: one alone first and the others while it
writes; a second turn of each that continues the first; then other prompts, half sampled with a seed and half
greedy with a token banned (`logit_bias`). A run counts only when the conversations decoded together. The
reference is one process with `--parallel N`:

```sh
export MAYA_GLM_SPLIT_ROWS=4       # with the two settings above
python3 tools/glm_split_check.py run --mode ref --parallel 4 --dir check -- $ENG --layer-split 24
CUDA_VISIBLE_DEVICES=1 python3 tools/glm_split_check.py run --mode back --parallel 4 --dir check -- $ENG \
    --layer-split 24 --role back --link 127.0.0.1:7711 &
CUDA_VISIBLE_DEVICES=0 python3 tools/glm_split_check.py run --mode front --parallel 4 --dir check -- $ENG \
    --layer-split 24 --role front --link 127.0.0.1:7711
wait
python3 tools/glm_split_check.py compare --dir check --parallel 4 --exact    # every conversation: the same tokens
```

`compare` also asks for `check/split/par4.back.ok`, which the back's run writes when it ends in order, with the
request ids of that run (on two machines, copy the back's `check/split/` to the front after each run).

### Across two machines

The same steps over the network, with the settings you will run with (`$ENG` as above):

```sh
# 1. the reference, on a machine that holds the whole model in VRAM + RAM (it can be the front): an ordinary
#    single-process start, with the --layer-split you would use there
python3 tools/glm_split_check.py prompts --pack <pack> --dir check
python3 tools/glm_split_check.py run --mode ref --dir check -- $ENG --layer-split 24

# 2. copy check/ to both machines (rsync -a check/ <other>:check/), then
#    on the back (start it first):
MAYA_GLM_LINK_KEY_FILE=link.key MAYA_GLM_LINK_ALLOW=<front-ip> \
  python3 tools/glm_split_check.py run --mode back --dir check -- $ENG \
    --layer-split 24:35 --role back --link <back-ip>:7710
#    on the front (it waits for the back):
MAYA_GLM_LINK_KEY_FILE=link.key \
  python3 tools/glm_split_check.py run --mode front --dir check -- $ENG \
    --layer-split 24:12 --role front --link <back-ip>:7710

# 3. the back wrote the logits and the front the answers: bring them together and compare
rsync -a <back>:check/split/ check/split/
python3 tools/glm_split_check.py compare --dir check          # exit code 0 = inside the tolerance below
```

Without `--exact`, `compare` allows a tolerance, because the two machines place the experts differently (more of
them in VRAM) and the default bf16 hop rounds the residual:

- the greedy answers must agree until a near-tie: where they first differ, the reference's own logits put the two
  tokens within `--tie` nats (0.5);
- the logits must stay within `--max-kl` (mean KL 2.5e-2) and `--min-top1` (93% top-1 agreement).

**These defaults are not a law of the model.** A low-bit quant moves more on its own, and long prompts move more
than short ones: with Maya-XS and a 2K-token prompt our own two-machine run fails all three defaults (see
[Measured](#measured)), and one process against itself with a single setting changed fails the KL and top-1
limits too. So measure the floor of your model first, with a second reference run that differs in one setting, and
pass limits a little above what it shows:

```sh
# a second reference run with the CPU lane off, and how far it is from the first
MAYA_GLM_CPU_LANE=0 python3 tools/glm_split_check.py run --mode ref --name ref2 --dir check -- $ENG --layer-split 24
python3 tools/glm_split_check.py compare --dir check --against ref2
# the split against the reference, with limits a little above that floor (these numbers are an example)
python3 tools/glm_split_check.py compare --dir check --max-kl 0.12 --min-top1 85 --tie 2
```

The exact check above is what shows the split computes what one process computes. This comparison shows what your
quant and your placement do to the answers.

Each check is one engine start (`maya generate`, no server): 4 a prompt for the reference, 3 for the split.
`--only code` runs one prompt. Run the script with the Python the setup made (`.venv/bin/python`):
`prompts` needs the tokenizer's packages and `compare` needs numpy; `run` needs neither.

### Measured

Maya-XS, 2K-token code prompt (and two short ones), on RTX 3060s. The exact check and the comparison without
`--role` ran on v1.1.76; the two-machine logit comparison on v1.1.74, the same split code before the strata →
maya rename.

- **The exact check** (one machine, 2 GPUs): bit-identical logits over the 2K-token prompt, and identical greedy
  answers with MTP (the pipelined decode) and without it (128 of 128 tokens each).
- **Across two machines** (2 + 2 GPUs, 1 GbE, bf16 hop):
  - **Short prompts:** the greedy answers first differ at near-ties (0.002–0.24 nats apart in the reference's
    logits), as two single-process runs do. KL is 9.9e-3–2.8e-2, top-1 95–96%.
  - **The 2K-token prompt:** they first differ at token 14, where the reference put the two tokens 1.7 nats apart.
    KL is 1.1e-1, top-1 86%.
  - **Why the 2K prompt moves more:** one process alone moves by KL 6.5e-2 (top-1 91%) on that prompt when only
    its CPU lane is switched on or off. The two machines place the experts differently (more of them in VRAM),
    Maya-XS's 1-bit experts flip near-ties in the top-8 choice easily, and the bf16 hop adds ~6e-3. The exact
    check above removes all three and is bit-identical.
- **Without `--role`:** greedy answers identical to an unmodified build (128 of 128 tokens on both prompts, CPU lane
  off so that runs repeat exactly), and the same decode speed (within 1% on both prompts, on v1.1.76).
- **Several conversations at once** (`--parallel 4`, v1.1.76):
  - **The exact check** (one machine, 2 GPUs, run on each of two machines): all 12 conversation turns of the
    `--parallel 4` step give the tokens of one process with `--parallel 4` (1,530 tokens: greedy, sampled with a
    seed, with a token banned). The same with a back that keeps no slots, a back without the NextN block, the
    requests in two groups of two, and two conversations. Without `--role`, one process writes the tokens of an
    unmodified build at the same speed.
  - **`tools/agent_bench.py`** (4 clients, 900 s, each setting twice, no failed request; a 32K context, so
    without the bench's 30K-token checks and with system prompts of 8K / 4K / 2K tokens;
    `MAYA_GLM_SPLIT_ROWS=4 MAYA_GLM_EVEN_GROUPS=1`):

    | | two machines, `--parallel 1` | two machines, `--parallel 4` | one machine, `--parallel 4` |
    |---|---|---|---|
    | tokens a second, all conversations | 5.4 | 9.2 | 1.5 |
    | a turn's first token, median | 123 s | 10 s | 175 s |
    | a turn's last token, median | 130 s | 64 s | 450 s |
    | decode of one request, median | 26.8 tok/s | 3.9 tok/s | 0.8 tok/s |
    | expert reads from VRAM | 87% | 77% | 35% |

    With `--parallel 1` the conversations take turns: each request decodes fast, with MTP, and waits for the
    others' whole reads and answers. With `--parallel 4` they share the engine. One machine alone cannot keep
    four conversations' states and the experts in its VRAM.

## Limits

- GLM-5.3-Flash packs only (`--glm-pack`). GLM-5.3 (glm-dsa) and the other models are refused with `--role`.
- Linux only for now, tested with CUDA 13.2 on RTX 3060 (sm_86) over 1 GbE. The link is plain POSIX sockets: HIP
  is untested, and on Windows `--role` ends with "glm link: not on Windows".
- Exactly two machines; the link carries the boundary between them only.
- Both sides must run the same build and the same pack. The protocol version, the split's settings and the model
  are checked at connect, and the per-request settings (decode mode, prompt chunks, reuse) at every request. A
  back that serves turns a front whose settings differ away and waits for the next one.
- The front decides how a prompt's read is cut into calls (where its conversation checkpoints and the prefix cut
  end one, where the snapshot is taken) and the back reads it the same way, whatever its own checkpoint and slot
  settings are. Two sides with different such settings (`MAYA_GLM_CKPTS`, `MAYA_GLM_SLOTS`, ...) keep different
  states, so they reuse less of a conversation: that costs prompt reads, not correctness.
- The prompt cache on both sides must agree; when they differ (one side restarted, a slot only one side has), both
  read the prompt again from the start, which costs time, not correctness.
- A failure on either side ends the running request on both (with `--parallel`, all of them). The link drops and
  the front reconnects at its next request.
- Text requests only: images (`GENI`), `LOGP` scoring and the vision lend (`VLEND` / `VRECLAIM`) are refused on
  the front with an error.
- `MAYA_GLM_WARM_PROMPT` is not used over the link (the first prompt warms both sides), and `--role` does not run
  with `MAYA_GLM_TP` or `MAYA_GLM_SLOW`.
- A peer that vanishes without closing the connection is noticed after `MAYA_GLM_LINK_DEAD_S` (30 s), not at
  once, and one that is up but stuck after `MAYA_GLM_LINK_STALL_S` (60 s). After a `STOP` the front waits on the
  link for 30 s more in all; then it drops the link, and the next request reads its prompt from the start.
- One front at a time. While one holds the link, the back answers no other (a second front's start fails). Two
  that start at once don't disturb each other: the one whose two connections arrive first gets the link. The back
  runs one connection's handshake at a time (10 s at most each, 2 s for one that sends nothing).
- The front resolves the `--link` host once, when it starts.
- `MAYA_GLM_SCORE` does not run with `--role` (`tools/glm_split_check.py` compares a split's logits).
- `--parallel` over the link needs `MAYA_GLM_SPLIT_ROWS` on both sides and reads a new prompt whole (see
  [Several conversations at once](#several-conversations-at-once---parallel)). A failure ends all the running
  requests, not one.
- No DFlash2 drafter over the link.

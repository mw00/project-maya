# Experimental EXL3 overflow expert execution

The helper keeps additional target-model experts on a GPU outside the target TP
group. When that group misses an expert, the helper executes its resident expert
while the CPU processes the remaining misses. Only activations and results cross
the helper link during decode; the helper weights remain resident.

This extends the existing GGUF helper path to uniform EXL3 layers, including
target layers sharded across two GPUs. The helper stores a complete expert;
the normal TP cache stores shards. Router selections and weights remain
authoritative. Moving a contribution from CPU to GPU changes floating-point
arithmetic, so bitwise output equivalence is not guaranteed.

## Selecting a helper and its budget

The native executable accepts the `DS4_*` names below; Maya's public settings
use the corresponding `MAYA_DS4_*` names. GPU IDs are visible IDs printed at
startup, after any visibility filtering. Budgets are explicit decimal GB;
there is no default GPU or pool size.

For a dedicated helper, choose a visible GPU outside the target's `--gpus`
selection. For example, with target GPUs 0 and 1 and helper GPU 2:

```sh
DS4_HELPER=2 DS4_HELPER_GB="$helper_gb" DS4_HELPER_EXL3=1 \
DS4_WARM_CAP_GB="$host_warm_gb" \
maya-ds4 --pack /path/to/pack --gpus 0,1 --tp 2
```

Set `helper_gb` and `host_warm_gb` to budgets suitable for the machine before
running this example. The helper executes existing router selections;
it does not predict experts or alter their routing weights.

## Sharing a helper GPU with DSpark

For three visible GPUs, target TP2 on GPUs 0 and 1, and DSpark on GPU 2,
also select GPU 2 and give its target group zero layers. Set `target_layers`
to the model's number of target layers (40 for the measured Flash model):

```sh
DS4_SPLIT="$target_layers,0" DS4_DRAFT_GPU=2 \
DS4_HELPER=2 DS4_HELPER_GB="$helper_gb" DS4_HELPER_EXL3=1 DS4_HELPER_SHARED=1 \
DS4_WARM_CAP_GB="$host_warm_gb" \
maya-ds4 --pack /path/to/pack --gpus 0,1,2 --tp 2 --dspark on
```

Shared mode requires loaded DSpark blocks, TP2, and zero target layers on the
helper. The loader reserves the requested helper pool before placing the
drafter, so subsequent memory estimates include that allocation. The example
uses decimal GB and requires enough remaining VRAM for drafter weights, scratch
space, context buffers, and the requested reserve. It is an experimental budget,
not an automatic recommendation for every model or GPU.

The measured three-R9700 configuration used `target_layers=40`, `helper_gb=20`
and `host_warm_gb=200`. These are benchmark settings, not new defaults or
universal recommendations. CPU worker count is independently configurable.
That configuration explicitly disables the recent-expert
cache with `DS4_RECENT=0`. The engine otherwise defaults to 32 recent slots on
eligible layers. That changes the primary/helper partition, RAM use, and copy
traffic; treat it as a separate tuning choice. Do not omit the setting when
reproducing the reported helper measurements.

Leave all helper settings unset to disable this feature. `DS4_HELPER_EXL3=1` is
required for EXL3. Mixed-format EXL3 layers are not selected for the helper.
The existing exclusive GGUF helper configuration remains available.
Upstream's dynamic helper slots (`DS4_HELPER_RECENT`) currently read GGUF
weights. EXL3 helpers use the static pool; a nonzero recent-slot count is
rejected at load rather than passed to the GGUF fill path.

CPU backing copies of helper experts are retained for fallback. This can
increase host RAM compared with the original placement. `DS4_WARM_CAP_GB`
controls warm copies in mmap mode; it does not cap all host allocations when
`--mmap-experts off` is used.

The helper uses the existing async CPU service path and its profile-based
selection of experts left outside the TP cache. This is resident overflow
execution, not predictive prefetching. Inspect `helper`, `helper_fb`,
`helper_ms`, `helper_held`, and `helper_dead` in STAT to confirm actual use.

## Validation

`DS4_HELPER_CHECK=1` reads back each uploaded helper expert and compares all
bytes. Before decode, it checks each represented EXL3 format against CPU
execution, compares grouped versus per-token GPU execution bitwise, and forces
the CPU fallback path with and without an existing CPU contribution. It uses
the existing helper tolerance of 1e-3 relative to each token's largest CPU
output. A failed execution check disables the helper; a failed upload byte
check fails loading. Do not interpret successful operator checks as broad
model-quality validation.

CPU fallback covers recoverable helper errors. A fatal error on a GPU also
hosting DSpark can affect the drafter, and is not guaranteed recoverable.

The optional HIP primitive test accepts a real-weight fixture and visible GPU:

```sh
cmake --build build-hip --target hip_ds4_exl3_helper
build-hip/hip_ds4_exl3_helper /path/to/full-layer.bin 2
```

Fixture header: 16 native-endian uint64 values: three K2 values, then
trellis/suh/svh byte offsets for each of three projections, full expert byte
size, expert count, embedding dimension (5120), and FFN dimension (2304).
The header is followed by full packed expert blobs; at least three are needed.
The test checks weight upload bytes, windows of 1–6 tokens, repeated experts,
unowned picks, and empty output rows. It needs external model weights, so it
is not registered as an unattended CTest.

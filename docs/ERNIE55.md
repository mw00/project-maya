# ERNIE-5.5-Flash on maya-ds4 (by hand, for now)

**[ERNIE-5.5-Flash-Preview](https://huggingface.co/baidu/ERNIE-5.5-Flash-Preview)** (Baidu, Apache 2.0; ~300 B
parameters, 512 routed experts a layer, 10 picked a token) runs on the DeepSeek engine, `maya-ds4` (#1232), from
[peasantsmith's Maya-S GGUF](https://huggingface.co/peasantsmith/ERNIE-5.5-Flash-Preview-Maya-GGUF) as it is. It is
not offered by the setup yet: a pack and a run config by hand, then `serve/server.py`.

## 1. The pack

```
python tools/ds4/pack_ernie55.py --model <folder of the two GGUF shards>
```

writes `<model>/maya-ds4/`: `ds4.txt` (`arch ernie55` and the model's dimensions), `tensors.txt` (where each tensor is
in the shards, read in place - nothing is converted), `ernie55.bin` (the mHC mixes transposed, a few KB) and
`tokenizer/` (the vocabulary with the `ernie5` pre-tokenizer, `tools/strata_tokenizer.py`, and
`tools/ds4/ernie55_chat_template.jinja`, by which the server knows the model).

## 2. Running it

```json
{"exe": "<repo>/build/maya-ds4",
 "args": ["--pack", "<model>/maya-ds4", "--gpus", "0", "--max-context", "131072", "--prefill-chunk", "2048"],
 "tokenizer": "<model>/maya-ds4/tokenizer", "model_name": "ernie-5.5-flash-preview",
 "sampling": {"temperature": 1.0, "top_p": 0.95}}
```

`python -m serve.server --engine maya --config ernie.json`. The sampling is the model card's (Baidu's
`generation_config.json`): without it the server samples with top_p 1.0, and the long tail sends answers off course.
The OpenAI and Anthropic endpoints, thinking on and off (`reasoning_effort` low / medium / high, or none; rendered by
Baidu's own encoder, `serve/ernie55_encoding.py`) and tool calls work as for the other models. An answer ends at
`<|ernie:im_end|>` or `<|ERNIE_MASK|>` (the card's two end tokens). One GPU, text only, one request at a time; the
DSpark drafter, the n-gram lookup drafter, `--threads` and tensor parallel are off for this model.

- **128K on a 12 GB card**: the head is held as Q8_0 (`DS4_ERNIE_HEAD_Q8=0`: BF16) and the indexer's keys in FP16,
  which leaves the chunked prompt path its buffers beside a 128K context's caches (1.1 GB).
- **Conversations kept** (`MAYA_DS4_SLOT_KEEP=1`,
  [SETTINGS.md](SETTINGS.md#deepseek-conversations-kept-an-agents-parent-and-workers-on-one-model)): a slot holds the
  MLA layers' latent caches and indexer keys beside V4.1's state, so an agent's short side requests (a title, a
  summary) no longer make its next turn read the whole prompt again (a 3091-token conversation taken back from RAM:
  18 new tokens read in 0.6 s instead of ~10 s).
- **Short prompts**: `DS4_CHUNK_MIN=128` takes prompts from 128 tokens through the chunked path (below 512 they run in
  windows of 16 by default, ~33 tok/s here).

### An AMD card beside the NVIDIA one (Linux, experimental)

`DS4_ERNIE_RX` loads an expert executor built for the AMD GPU's own runtime (`libmaya_ds4_ernie_rx.so`, from
`src/ds4/ds4_ernie_rx.cu` with the engine's own expert kernels; `HIP_ARCH=gfx1031 tools/ds4/build_ernie_rx.sh <dir>`
with ROCm's `hipcc`). At load it takes the CPU lane's most-routed experts by the pack's profile into the card's
memory (`DS4_ERNIE_RX_MIB`, 9216 by default; `DS4_ERNIE_RX_DEVICE`, its HIP ordinal, 0), the layers whose gate and up
differ in type excepted; a decoded token's picks it holds are computed there, side by side with the CPU lane (its
calls on a thread of their own), and the sum added into the lane's. A failure on the card: its experts on the CPU lane
from then on. Prompts are not helped (the chunked path's). The run's environment needs the ROCm libraries
(`LD_LIBRARY_PATH`). Answers can be worded differently from the CPU lane alone (the card sums in another order).

## 3. What the engine does for it

The architecture beside DeepSeek-V4.1's, as `maya-ds4` runs it (`Geo::ernie`; `layer_ernie` for decode and verify
windows, `layer_chunk_ernie` for chunked prompts):

| | ERNIE-5.5-Flash | in the engine |
| --- | --- | --- |
| Dense weights | FP8 E4M3, an F32 scale every 128 x 128 block | read in place (`gg::kFmtF8`): matrix-vector products in F32, prompts' GEMMs over FP16 rows |
| Attention, 37 layers | V4.1's compressed attention (ratio 128) + a 128-token window, a sigmoid gate on the output, the value heads' mix (I + V U^T) | V4.1's kernels, with the compressor's absolute position bias and the gate / mix (`vha_mix`, `sigmoid_mul`) |
| Attention, 6 layers (7, 14, ..., 42) | DeepSeek-V2 MLA (64 heads, kv_lora 512, RoPE 64, plain RoPE), a sink, the gate and mix; the DSA indexer (64 heads x 128, top 2048 older rows, then the window) | `mla_*`, `ix_*` (the latent cache in FP16, the indexer's keys in FP16; the absorbed queries) |
| mHC | 4 streams, the block's own pre (not V4.1's lagged one), the RMS epsilon outside the root | `hc_mix_*` with a negative epsilon |
| FFN | layer 0 dense (14336 wide); then a latent MoE: two routers (sigmoid + sigmoid, bias, top 10, renormalised, a per-expert scale), fc1 4096 -> 2048, the experts in 2048, latent norm, fc2, + a shared expert; SiTU-GLU (beta 4, linear 25) | the routed experts through the low-bit kernels with `iq::kActSitu` (GPU and CPU lane); gate and up of different types (layers whose gate is IQ2_XXS, up IQ1_M) in two passes |
| Head | the mHC head mix, then the logits' multimax | `head_ernie` (the head as Q8_0) |

Prompts: chunks of up to `--prefill-chunk` tokens (a token's ten picks as two rows of five, so the V4.1 paths' eight
picks a row stay as they are); `DS4_ERNIE_CHUNKS=0` reads them in windows of 16 instead. Decode does not wait for the
GPU at each layer's end (`DS4_ERNIE_SYNC=1` does, as a check).

## 4. Checked against

`tools/ds4/ref_ernie55.py`, a PyTorch reference that reads the same GGUF (`--sim-kv` rounds its caches as the engine
does: the window in FP8, the compressed rows in FP4):

- layer 0 the same to 1e-4 (relative); the rest drift by the int8 rounding of the experts' activations (about 0.5% a
  layer), as on DeepSeek;
- the logits over a 31-token prompt: median KL 0.008-0.012, the same top token 84-90% of the time (windows and
  chunks, BF16 and Q8_0 head alike);
- a 3454-token prompt (past the indexer's 2176) with a code word in its first line: every path answers it, at 16K
  and at 128K;
- greedy answers with and without the AMD card, with and without the per-layer wait, BF16 and Q8_0 head: all on
  topic and well formed.

## 5. Measured

One RTX 5070 (12 GB), an RX 6700 (10 GB), an EPYC 7302 (16 cores) with 128 GB of RAM (4 channels), Linux:

| | before (windows of 16) | now, 128K context |
| --- | ---: | ---: |
| A 1178-token prompt | 31.6 tok/s | 347 tok/s (16K context) |
| A 3454-token prompt | 30.6 tok/s (16K context) | 311 tok/s |
| A 12.4k-token prompt | | 283 tok/s |
| Decode, the RTX 5070 alone | 17.5-19.8 tok/s | 17.3-20.4 tok/s |
| Decode with the RX 6700 (2977 experts, 9.7 GB: 44% of the profile's picks) | | 20.4-21.5 tok/s |

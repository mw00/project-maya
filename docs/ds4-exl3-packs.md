# DeepSeek-V4.1 EXL3 packs: type-ordered shards, and Diffbot's 3 bpw

`tools/ds4/pack.py` packs another maker's EXL3 folder (`--model-dir` in the setup). Two layouts stop it; neither
needs the weights re-quantized.

## Type-ordered shards

The engine reads a layer's experts as one region an expert: `expert_layout` (`src/ds4/ds4_model.cu`) takes expert 0's
piece offsets as the layout of every other expert in the layer, so each expert's trellis / suh / svh pieces must lie
back to back in their file. A quant that writes its shards grouped by type - every expert's `.trellis`, then every
`.suh`, every `.svh`, ... - puts one expert's pieces megabytes apart, and the pack stops with `its pieces are not
together in their file`.

    python tools/ds4/reorder.py --model <model folder>

rewrites those shards in place: the non-expert tensors first (their order kept), then expert by expert, each expert's
w1 / w2 / w3 pieces back to back in the order LibertAIDAI's packs have - suh, svh, mul1 (or mcg, a drafter's
codebook), trellis. Only the tensor order and each `data_offsets` change: every tensor's dtype, shape and bytes stay
the same. A shard is written under a temporary name, every tensor's SHA-256 is checked against the bytes that were
read, then the name is swapped in; a run needs free space for one shard beside the model. Shards whose experts already
lie back to back (LibertAIDAI's) and dense / Engram shards are left alone. Naming shards instead of `--model` rewrites
just those; `--manifests DIR` writes every rewritten tensor's old and new offset and its SHA-256.

## Diffbot's 3 bpw

[Diffbot's 3 bpw](https://huggingface.co/diffbot/DeepSeek-V4.1-Flash-EXL3-3bpw-2x-RTX-PRO-6000) writes its expert
shards by type, keeps its dense weights in FP8, stores the Engram tables in its own INT4 layout, and quantizes the
MTP (DSpark) experts with the `mcg` codebook. To run it:

1. `python tools/ds4/reorder.py --model <folder>` - its shards 3-42 (main) and 44-46 (drafter) hold experts.
2. Replace its Engram shards `model-00047-of-00048.safetensors` / `model-00048-of-00048.safetensors` with DeepSeek's
   FP8 ones from `deepseek-ai/DeepSeek-V4.1-Flash-1017` (the same tables the 2.9 / 3.5 bpw setup already downloads).
3. `python tools/ds4/fp8_dense.py --model <folder>` - the FP8 dense weights as `maya-dense-bf16.safetensors`.
4. `python tools/ds4/pack.py --model <folder>`, then `--model-dir <folder>` in the setup.

DSpark: the DeepSeek engine decodes `mul1`; the 3 bpw repo's own drafter is `mcg` (issue #806's fourth gap, still
open), so the model runs with `--dspark none`. To keep drafting, pair it with the `mul1` drafter of an EXL3 pack that
has one: LibertAIDAI's 2.9 bpw shards `model-00042-of-00043.safetensors` / `model-00043-of-00043.safetensors` hold
only the `mtp.*` tensors - leave the 3 bpw drafter shards (44-46) out, put the 2.9's beside the rest, and give the
folder a `model.safetensors.index.json` whose `weight_map` lists the files that are there (the pack reads the file
list from its values; the keys are not read):

    python - <<'EOF'
    import json, pathlib
    m = pathlib.Path("<model folder>")
    files = sorted(p.name for p in m.glob("*.safetensors"))
    wm = {str(i): f for i, f in enumerate(files)}
    (m / "model.safetensors.index.json").write_text(json.dumps({"weight_map": wm}))
    EOF

The pack then reads the main model from the 3 bpw shards and the drafter from the 2.9's.

Measured, two RTX 3060 12 GB + two RTX 5060 8 GB, `DS4_SPLIT=6,6,7,21`, `--prefill-chunk 8384`: the mixed pack read
an 111,072-token prompt in 126.0 s (881.5 tok/s, the long-prompt skip on) and decoded at 9.36 tok/s over 785 tokens
(544 of 701 drafts kept). Without the drafter, 73,840 tokens read in 78,969 ms (935.1 tok/s). The split keeps the
drafter's extra ~140 MB off the card that holds layers 20-39 (the long-prompt skip's): at `5,5,6,24` that card ends
with 3 MiB free, and the first decode's lazy kernel load fails - one `out of memory` line, then `unknown error`s.

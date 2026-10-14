# Serving the experts-only community EXL3 quants

Community quants of GLM-5.3-Flash in the EXL3 format quantize the routed experts only
(`quantization_config.scope: "glm53_routed_experts_only"`; e.g. neko-legends' 4bpw): every expert tensor
arrives as `trellis` / `suh` / `svh`, while the attention and dense-FFN projections, the shared experts,
the dense layers and the head stay unquantized (BF16, sometimes F32 or F16) where Maya's own quants
quantize them. The packer and the engine accept both layouts.

## Packing

    python3 tools/exl3_pack.py --model <the model directory> --out <the model directory>/pack

The model directory is the EXL3 repo as downloaded (config.json, model.safetensors.index.json, the
shards, tokenizer.json). `--check <a GGUF of the same model>` verifies the conversion before writing;
`--no-mtp` leaves the NextN (MTP) draft block out. The packer writes `pack/` (index.txt, dense.bin,
exl3.txt, maya-exl3.gguf, tokenizer/):

- every EXL3 matrix (the experts) is referenced in place, byte for byte;
- every unquantized matrix becomes a BF16 float row (F32 and F16 sources convert here, at pack time);
  unfused q/k/v projections (no fused `qkv_proj` in the source) emit as three rows the engine serves
  the same way;
- an unquantized `eh_proj` packs the NextN block: its experts from exl3.txt, its float rows into the
  pack GGUF, and the engine quantizes the block's BF16 projections to Q8_0 at load for the draft.

At load the engine keeps one copy of each BF16 row: a row the fast path reads only through its
quantized matmul (`q()` - the attention/dense-FFN projections, the shared experts, the dense FFN and
`output.weight`) gets a Q8_0 copy and no BF16 copy; every other row (the routers, the hyper-connection
gates, the ssm gates, the indexer, `attn_k_b` / `attn_v_b`, and `attn_q_a` / `attn_kv_a_mqa`) keeps
only its exact BF16 copy. The freed VRAM goes to the expert pool.

## Serving

A serve config names the pack and the GPUs; adapt the paths:

    {
      "args": ["--glm-pack", "/path/to/EXL3-4bpw/pack", "--max-context", "131072", "--prefill", "auto"],
      "tokenizer": "/path/to/EXL3-4bpw/pack/tokenizer",
      "model_name": "glm-5.3-flash",
      "gpu": [0, 1, 2, 3, 4, 5, 6],
      "host": "127.0.0.1",
      "api_key": "set-a-real-key-if-not-on-localhost",
      "port": 8000
    }

    python3 serve/server.py --engine strata --config config.json --port 8000

The engine needs no quantization-specific flags: the pack's index says which rows are EXL3 and which
are floats. `STRATA_GLM_KDA_GRAPH=1` only helps single-GPU decode, so it is off by default and not
set here (a multi-GPU split does not use it).

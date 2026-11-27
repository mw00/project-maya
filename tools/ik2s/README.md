# maya-ik2s's tools (Inkling 2 Small on 2-4 AMD GPUs)

`maya-ik2s` (`src/ik2s`, docs/INKLING2_AMD.md) reads a pack of unsloth's GGUF set and drafts with a converted copy of
the model's DSpark drafter. The setup (`./maya.sh --setup --backend hip --model inkling-2-small`) runs these; they run
by hand as well:

- `download.json`: unsloth's UD-Q2_K_XL GGUF set of Inkling-2-Small and Thinking Machines' `mtp.safetensors` with its
  `config.json` (the DSpark drafter), each file's size and sha256 at a pinned revision.
- `pack_gguf.py --model <the set's folder>`: the pack (`<folder>/maya-ik2s`: the geometry, every tensor's place in the
  shards, read in place, and the tokenizer with `tools/ik2/chat_template.jinja`, thinkingmachines' own template).
- `convert_dspark.py --mtp <drafter folder> --tokenizer-gguf <first shard> --type q8_0`: the drafter's Q8_0 copy and
  its index (`maya-ik2s-dspark-q8_0`, docs/INKLING2_DSPARK.md).
- `ref/`: the llama.cpp reference dumps `maya-ik2s --parity` compares against (`run_dump.sh`, `compare_dumps.py`) and
  the drafter's reference run (`run_dspark_eval.sh`, for `ik2s_dspark_test`).

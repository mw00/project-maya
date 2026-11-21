# Inkling 2's chat template

`chat_template.jinja` is copied unchanged from thinkingmachines' release of
[Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (sha256
`e9cfd8e75d1272633c2e72b3cc76e99567e2bc8912b960483aed483ad35d54c7`). The GGUFs carry the llama.cpp conversion's own
template, which differs; a tokenizer folder for `maya-ik2` takes this one:

    python tools/strata_tokenizer.py --gguf <first shard> --out <folder> --chat-template tools/ik2/chat_template.jinja

Copyright (c) Thinking Machines Lab. Licensed under the Apache License, Version 2.0 (the model's license; see its
page). `serve/test_inkling2_frontend.py` checks that the server renders it as the model's own tokenizer does and reads
its answers.

## The AMD build's tools

`maya-ik2`'s AMD build (`src/ik2/hip`, docs/INKLING2_AMD.md) reads a pack and drafts with a converted drafter; the
setup (`./maya.sh --setup --backend hip --model inkling-2-small`) runs these, and they run by hand as well:

- `download-small.json`: unsloth's UD-Q2_K_XL GGUF set of Inkling-2-Small and Thinking Machines' `mtp.safetensors`
  (the DSpark drafter), each file's size and sha256 at a pinned revision (`download.json`: Inkling-2-Mini's).
- `pack_gguf.py --model <the set's folder>`: the pack (`<folder>/maya-ik2`: the geometry, every tensor's place in the
  shards, read in place, and the tokenizer with `chat_template.jinja`).
- `convert_dspark.py --mtp <drafter folder> --tokenizer-gguf <first shard>`: the drafter's Q8_0 copy and its index
  (`maya-ik2-dspark-q8_0`, docs/INKLING2_DSPARK.md).
- `ref/`: the llama.cpp reference dumps the AMD build's `--parity` compares against (`run_dump.sh`,
  `compare_dumps.py`) and the drafter's reference run (`run_dspark_eval.sh`).

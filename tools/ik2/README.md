# Inkling 2's chat template

`chat_template.jinja` is copied unchanged from thinkingmachines' release of
[Inkling-2-Small](https://huggingface.co/thinkingmachines/Inkling-2-Small) (sha256
`e9cfd8e75d1272633c2e72b3cc76e99567e2bc8912b960483aed483ad35d54c7`). The GGUFs carry the llama.cpp conversion's own
template, which differs; a tokenizer folder for `maya-ik2` takes this one:

    python tools/strata_tokenizer.py --gguf <first shard> --out <folder> --chat-template tools/ik2/chat_template.jinja

Copyright (c) Thinking Machines Lab. Licensed under the Apache License, Version 2.0 (the model's license; see its
page). `serve/test_inkling2_frontend.py` checks that the server renders it as the model's own tokenizer does and reads
its answers.

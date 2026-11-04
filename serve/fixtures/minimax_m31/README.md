# MiniMax-M3.1-Flash reference template

`chat_template.jinja` is copied unchanged from MiniMax's
[MiniMax-M3.1-Flash](https://huggingface.co/MiniMaxAI/MiniMax-M3.1-Flash/blob/54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e/chat_template.jinja),
revision `54e4e69dfc452b1e94a426c24a4fdeeaa8e3ad5e` (sha256 `961e3f96b2c08d26555efdfeb3b384d5e81181bf0cfc04edbadc31c6e507b431`).
It is the template MiniMax's own files and GGUF conversions of them carry.

Copyright (c) 2026 MiniMax. Distributed under the MiniMax Community License; see
[LICENSE.minimax-m31](../../../LICENSE.minimax-m31).

Maya renders its own copy, `tools/mm3/chat_template.jinja`: the same template plus MiniMax-M3's wording for thinking
turned off. `serve/test_minimax_frontend.py` checks that both write the same prompt wherever they share a mode
(MiniMax's adaptive thinking at every effort, with and without tools, pictures and tool results).

Run offline from the repository root:

```sh
python -m unittest serve.test_minimax_frontend -v
```

Preserve the original bytes when editing tests (the test checks the sha256 above).

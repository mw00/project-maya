# GLM chat-template fixtures

Test fixtures for `serve/test_glm_dsa_serve.py`, `serve/test_glm_dsa_api.py` and `serve/test_prompt_memo.py`: the
server's effort-dialect detection, rendering, tool-call ids, `clear_thinking` and history reuse are checked against real
GLM templates instead of an inline mock.

| File | sha256 | Source |
|---|---|---|
| `glm-5.3.official.jinja` | `69bb3ab52067898e2466b855407636de559568947f367945842aabcb7fcc1705` | GLM-5.3's `chat_template.jinja` as Z.ai publishes it (zai-org/GLM-5.3), byte for byte: the template a pack without one of its own gets (`serve/glm_dsa_product.py` TEMPLATE). |
| `glm-5.3.gguf-ud-iq2_m.jinja` | `15d2a7176beb599de0a59af8314b4869011e416cff7f65748b314940d3379b0e` | The template unsloth's GLM-5.3 UD-IQ2_M GGUF carries (`tokenizer.chat_template` of shard 1, revision `346b3591`; the pack's `tokenizer/chat_template.jinja` is the same bytes): the one the server renders that model with. It differs from the official one in five places - `[0]` for `.0` in four subscripts, and a tool call's arguments rendered only when they are a mapping. |
| `glm-5.2.jinja` | `172dc74a35e1752df75ecfb2b2cf9326d2852bb1379868ebeec9571654489679` | GLM-5.2 template (basetenlabs/basetenkenizer `vendored_tokenizers/glm-5.2/chat_template.jinja`). Effort high/max only plus `enable_thinking`: the dialect that Maya's exact-substring check missed. |

The templates are part of the Z.ai model releases (licence `glm-5.3` / GLM-5.2's MIT); they are kept here only as
test inputs.

`serve/test_glm_dsa_serve.py` picks up every `*.jinja` in this folder whose name starts with `glm-5.3` and checks that
low, high and max render three distinct system lines with it. To add the template of another GGUF:

```
python3 - <the GGUF's first shard> <<'EOF'
import sys, hashlib, pathlib
sys.path.insert(0, "third_party/llama.cpp/gguf-py")
from gguf import GGUFReader
f = GGUFReader(sys.argv[1]).fields["tokenizer.chat_template"]
t = bytes(f.parts[f.data[0]]).decode("utf-8")
pathlib.Path("data/glm-dsa/templates/glm-5.3.gguf-<quant>.jinja").write_text(t, encoding="utf-8", newline="\n")
print(hashlib.sha256(t.encode()).hexdigest())
EOF
```

then add its row above.

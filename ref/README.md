# Reference material

Upstream code kept for comparison, not built and not shipped.

| Path | What it is |
| --- | --- |
| `upstream-llama.cpp/` | A pinned copy of parts of [llama.cpp](https://github.com/ggml-org/llama.cpp) used to compare kernel behaviour and quantization formats. MIT. |
| `glm.py`, `load.py` | Small reference scripts used while porting the GLM model. |

Nothing here is compiled into Maya. The build uses `third_party/ggml` and Maya's own sources; these copies exist so a
change can be diffed against upstream. Keep the upstream licence with any copy you take from here.

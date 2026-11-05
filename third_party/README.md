# Third-party code and wheels

| Path | What it is | Licence |
| --- | --- | --- |
| `ggml/` | Pinned ggml / llama.cpp sources: quantization formats, CPU dot products, prompt MMQ kernels | MIT (ggml authors) |
| `exllamav3/` | EXL3 format reference: codebooks, CPU expert kernel, tensor-core tile decoding | MIT (Turboderp) |
| `beellama.cpp/` | DFlash2 drafter forward reference | MIT (ggml authors, Anbeeld) |
| `wheels/` | Vendored Python wheels for the setup TUI, so it runs offline before anything is installed | each wheel's own licence |

The wheels are Textual, Rich, Pygments, Markdown-it-py, mdit-py-plugins, linkify-it-py, mdurl, platformdirs and
typing_extensions. They are used only by the setup screen; the engine does not depend on them. Their licences ship
inside each wheel; the consolidated list is in [`../THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md).

Bump procedure: update the pinned commit or wheel, keep the licence file next to it, and note the change in the
changelog.

# Third-party notices

Maya is MIT-licensed. It builds on, links to, or ships the following components. Keep these notices with any copy
or binary distribution.

## Compiled or ported code

| Component | Location | Licence |
| --- | --- | --- |
| ggml / llama.cpp | `third_party/ggml/` | MIT, Copyright (c) 2023-2026 The ggml authors |
| exllamav3 | `third_party/exllamav3/` | MIT, Copyright (c) 2025 Turboderp |
| beellama.cpp (DFlash2 drafter forward) | `third_party/beellama.cpp/` | MIT, Copyright (c) 2023-2026 The ggml authors, Anbeeld |
| upstream reference copies | `ref/upstream-llama.cpp/` | MIT |

## Web UI

| Component | Location | Licence |
| --- | --- | --- |
| Preact + htm | `serve/web/preact.js` (notices bundled) | MIT / Apache-2.0 |
| Lucide icons | `serve/web/LICENSE.lucide.txt` | ISC |
| Outfit font | `serve/web/fonts/OFL.txt` | SIL Open Font License 1.1 |

## Setup TUI wheels

Vendored in `third_party/wheels/` so setup works offline:

| Package | Licence |
| --- | --- |
| Textual | MIT |
| Rich | MIT |
| Pygments | BSD-2-Clause |
| Markdown-it-py | MIT |
| mdit-py-plugins | MIT |
| linkify-it-py | MIT |
| mdurl | MIT |
| platformdirs | MIT |
| typing_extensions | PSF-2.0 |

Each wheel carries its own licence metadata; check it when updating a wheel.

## Models

Model weights are not part of Maya's MIT licence. See [`licenses/README.md`](licenses/README.md) and each model's
card on Hugging Face.

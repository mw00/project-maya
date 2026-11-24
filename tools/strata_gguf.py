"""Import llama.cpp's gguf-py for the glm-dsa tools: the copy on sys.path / PYTHONPATH, else where tools/_paths.py
looks.  Only a missing `gguf` package falls back to _paths; a gguf-py that is found but cannot import one of its own
dependencies (pyyaml, numpy, tqdm, requests: gguf-py's pyproject) is reported as that, not as "gguf-py was not found"."""
from __future__ import annotations

import sys
from pathlib import Path

GGUF_PY_DEPS = "numpy pyyaml tqdm requests"


def import_gguf():
    """The gguf module; exits with the real reason when it cannot be imported."""
    try:
        import gguf
        return gguf
    except ImportError as e:
        if getattr(e, "name", None) != "gguf":
            sys.exit(f"llama.cpp's gguf-py was found but does not import: {e} "
                     f"(gguf-py needs: pip install {GGUF_PY_DEPS})")
    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    from _paths import add_gguf_py
    add_gguf_py()
    try:
        import gguf
    except ImportError as e:
        sys.exit(f"llama.cpp's gguf-py at {sys.path[0]} does not import: {e} (gguf-py needs: pip install {GGUF_PY_DEPS})")
    return gguf

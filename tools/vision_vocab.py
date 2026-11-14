"""Export a local GLM GGUF's tokenizer for maya-vision's pinned llama.cpp.

The vision library does not know glm5-next/glm5next as text architectures. It
only needs their vocabulary: use a known architecture in a tensor-free sidecar,
preserving every tokenizer field byte for byte. Never change the source GGUF.
"""
from __future__ import annotations

import os
from pathlib import Path
import struct
import tempfile

from gguf_fix_arch import STRING, _str_of, _str_raw, read_header


def write_glm_vocab(model: Path, out: Path) -> Path:
    if out.absolute() == model.absolute() or (out.exists() and not out.is_symlink() and out.samefile(model)):
        raise ValueError("the vision vocabulary must be written separately from the source model")
    # Read headers rather than gguf-py's tensor reader: local quants can include
    # tensor encodings unknown to that reader. No tensor data is needed here.
    header = read_header(model)
    arch = header.get("general.architecture")
    if arch is None or arch[0] != STRING or _str_of(arch[1]) not in ("glm5-next", "glm5next"):
        raise ValueError(f"{model}: not a GLM-5.3-Flash GGUF")
    tokenizer = [(k, t, raw) for k, t, raw in header.kvs if k.startswith("tokenizer.")]
    if not header.get("tokenizer.ggml.tokens") or not header.get("tokenizer.ggml.model"):
        raise ValueError(f"{model}: no tokenizer (select the first GGUF shard)")
    kvs = [("general.architecture", STRING, _str_raw("llama")),
           ("general.name", STRING, _str_raw("GLM vocabulary for maya-vision")), *tokenizer]
    data = b"GGUF" + struct.pack("<IQQ", 3, 0, len(kvs))
    data += b"".join(_str_raw(k) + struct.pack("<I", t) + raw for k, t, raw in kvs)
    data += bytes(-len(data) % 32)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Replace atomically, including a stale symlink at the output path; do not
    # follow it and accidentally rewrite someone else's model/tokenizer.
    pending = None
    try:
        with tempfile.NamedTemporaryFile(dir=out.parent, prefix=out.name + ".", delete=False) as f:
            pending = Path(f.name)
            f.write(data)
        os.replace(pending, out)
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)
    return out

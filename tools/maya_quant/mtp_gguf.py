"""tools/maya_quant/mtp_gguf.py - GLM-5.3-Flash's NextN (MTP) draft block alone, as a small GGUF.

The engine drafts tokens with this block on two GPUs (speculative decoding).  A quant published without it - or one
whose block is less precise - takes it from this file instead: MAYA_GLM_MTP_GGUF=<the file> (the engine looks up
the draft block there first; it reads the trunk's hidden state and the embedding, so a block from any quant of the
same model fits).  This copies the blk.<block_count>.* tensors as they are (no requantization) from a Maya GGUF that
has them, with the model's hyperparameters (no tokenizer: the model's own GGUF carries it).

    python mtp_gguf.py MODEL-00001-of-0000N.gguf OUT.gguf
"""
from __future__ import annotations

import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "build/_deps/strata_llamacpp-src/gguf-py"))
import gguf  # noqa: E402


def shards(first: pathlib.Path) -> list[pathlib.Path]:
    m = re.match(r"(.*)-00001-of-(\d{5})\.gguf$", first.name)
    if not m:
        return [first]
    return [first.with_name(f"{m.group(1)}-{i:05d}-of-{m.group(2)}.gguf") for i in range(1, int(m.group(2)) + 1)]


def main() -> int:
    src, out = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
    files = shards(src)
    head = gguf.GGUFReader(files[0])
    arch = head.fields["general.architecture"].contents()
    readers = [head] + [gguf.GGUFReader(f) for f in files[1:]]
    blocks = {m.group(1) for r in readers for t in r.tensors
              if (m := re.match(r"blk\.(\d+)\.nextn\.eh_proj\.weight$", t.name))}
    if len(blocks) != 1:
        print(f"{src}: {'no' if not blocks else 'more than one'} draft block (blk.N.nextn.*) in this model",
              file=sys.stderr)
        return 1
    prefix = f"blk.{blocks.pop()}."
    tensors = [t for r in readers for t in r.tensors if t.name.startswith(prefix)]
    w = gguf.GGUFWriter(out, arch=arch, endianess=head.endianess)
    for f in head.fields.values():
        if f.name in ("general.architecture", "general.name") or f.name.startswith(("GGUF.", "split.", "tokenizer.")):
            continue
        vtype = f.types[0]
        sub = f.types[-1] if vtype == gguf.GGUFValueType.ARRAY else None
        w.add_key_value(f.name, f.contents(), vtype, sub_type=sub)
    w.add_string("general.name", "GLM-5.3-Flash NextN (MTP) draft block")
    for t in tensors:
        w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, t.tensor_type)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    for t in tensors:
        w.write_tensor_data(t.data, tensor_endianess=head.endianess)
    w.close()
    size = sum(t.n_bytes for t in tensors)
    kinds = sorted({t.tensor_type.name for t in tensors})
    print(f"{out}: {len(tensors)} tensors of {prefix[:-1]}, {size / 1e9:.2f} GB ({', '.join(kinds)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())

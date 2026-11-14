"""k31_dense_requant.py: a dense-override GGUF for maya-k31 --dense-override.

  python3 -I tools/k31_dense_requant.py REF SRC OUT [--from Q6_K[,..]] [--to Q4_K] [--copy] [--ggml LIB] [--gguf-py DIR]

REF is the model (its first shard: every shard is read): the tensors of REF whose type is one of --from (default
Q6_K), never the routed experts (names with "_exps"), are the ones written.  SRC holds each such tensor under the
same name and shape, in any type gguf-py can dequantize (the BF16 dense-only GGUF made from the original weights;
or REF itself for a test).  Each is quantized to --to (default Q4_K) by ggml's own quantize_chunk (the library
llama-quantize uses, no importance matrix), so the bytes are what llama-quantize writes for that type.  --copy
writes SRC's bytes unchanged instead (with SRC = REF: the override that must give the same bits as no override).

The output holds only those tensors (and general.architecture), one at a time from SRC to the file: never all of it in
memory."""
import argparse
import ctypes
import os
import re
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("ref")
ap.add_argument("src")
ap.add_argument("out")
ap.add_argument("--from", dest="frm", default="Q6_K")
ap.add_argument("--to", default="Q4_K")
ap.add_argument("--copy", action="store_true")
ap.add_argument("--ggml", default=os.path.expanduser("~/work/llama-up/build/bin/libggml-base.so"))
ap.add_argument("--gguf-py", default=os.path.expanduser("~/work/llama-up/gguf-py"))
a = ap.parse_args()
sys.path.insert(0, a.gguf_py)
import gguf  # noqa: E402
from gguf.quants import dequantize  # noqa: E402

QT = gguf.GGMLQuantizationType


def shards(first):
    m = re.match(r"^(.*)-(\d{5})-of-(\d{5})\.gguf$", first)
    if not m:
        return [first]
    n = int(m.group(3))
    return [f"{m.group(1)}-{i:05d}-of-{n:05d}.gguf" for i in range(1, n + 1)]


def tensors(first):   # name -> (reader tensor), over the set
    out, readers = {}, []
    for p in shards(first):
        r = gguf.GGUFReader(p)
        readers.append(r)
        for t in r.tensors:
            out[t.name] = t
    return out, readers


frm = {QT[x] for x in a.frm.split(",")}
to = QT[a.to]
ref, ref_r = tensors(a.ref)
src, src_r = (ref, ref_r) if os.path.abspath(a.src) == os.path.abspath(a.ref) else tensors(a.src)
arch = ref_r[0].fields["general.architecture"]
arch = bytes(arch.parts[arch.data[0]]).decode()

pick = [n for n, t in ref.items() if t.tensor_type in frm and "_exps" not in n]
missing = [n for n in pick if n not in src]
if missing:
    sys.exit(f"{len(missing)} tensors not in {a.src}, e.g. {missing[0]}")
for n in pick:
    if [int(x) for x in src[n].shape] != [int(x) for x in ref[n].shape]:
        sys.exit(f"{n}: shape {list(src[n].shape)} in SRC vs {list(ref[n].shape)} in REF")

lib = ctypes.CDLL(a.ggml)
lib.ggml_quantize_chunk.restype = ctypes.c_size_t
lib.ggml_quantize_chunk.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int64, ctypes.c_int64,
                                    ctypes.c_int64, ctypes.c_void_p]
lib.ggml_row_size.restype = ctypes.c_size_t
lib.ggml_row_size.argtypes = [ctypes.c_int, ctypes.c_int64]
lib.ggml_quantize_requires_imatrix.restype = ctypes.c_bool
lib.ggml_quantize_requires_imatrix.argtypes = [ctypes.c_int]
if not a.copy and lib.ggml_quantize_requires_imatrix(int(to)):
    sys.exit(f"{a.to} needs an importance matrix")


def ne(t):
    return [int(x) for x in t.shape]   # ggml order: ne0 (the row) first


def out_type(n):
    return src[n].tensor_type if a.copy else to


def out_bytes(n):
    t = src[n]
    if a.copy:
        return int(t.n_bytes)
    s = ne(t)
    return int(lib.ggml_row_size(int(to), s[0])) * int(np.prod(s[1:], dtype=np.int64))


w = gguf.GGUFWriter(a.out, arch)
w.add_string("k31.dense_override.ref", os.path.basename(a.ref))
w.add_string("k31.dense_override.from", a.frm)
w.add_string("k31.dense_override.to", "copy" if a.copy else a.to)
for n in pick:
    shape = list(reversed(ne(src[n])))   # gguf-py's writer takes numpy order
    if out_type(n) in (QT.F32, QT.F16, QT.BF16):
        w.add_tensor_info(n, shape, np.float32 if out_type(n) == QT.F32 else np.float16, out_bytes(n), raw_dtype=out_type(n))
    else:
        w.add_tensor_info(n, gguf.quant_shape_to_byte_shape(shape, out_type(n)), np.uint8, out_bytes(n), raw_dtype=out_type(n))
w.write_header_to_file()
w.write_kv_data_to_file()
w.write_ti_data_to_file()

tot_in = tot_out = 0
ROWS = 2048
for k, n in enumerate(pick):
    t = src[n]
    s = ne(t)
    if a.copy:
        data = np.ascontiguousarray(t.data).view(np.uint8).reshape(-1)
    else:
        rows = int(np.prod(s[1:], dtype=np.int64))
        flat = np.asarray(t.data)
        rs = int(lib.ggml_row_size(int(to), s[0]))
        data = np.empty(rows * rs, dtype=np.uint8)
        raw = np.ascontiguousarray(flat).view(np.uint8).reshape(rows, -1) if t.tensor_type not in (QT.F32,) else None
        for r0 in range(0, rows, ROWS):
            r1 = min(rows, r0 + ROWS)
            if t.tensor_type == QT.F32:
                f = np.ascontiguousarray(np.asarray(flat).reshape(rows, s[0])[r0:r1], dtype=np.float32)
            else:
                f = np.ascontiguousarray(dequantize(raw[r0:r1], t.tensor_type).reshape(r1 - r0, s[0]), dtype=np.float32)
            got = lib.ggml_quantize_chunk(int(to), f.ctypes.data, data[r0 * rs:].ctypes.data, 0, r1 - r0, s[0], None)
            if got != (r1 - r0) * rs:
                sys.exit(f"{n}: quantize_chunk wrote {got} bytes, want {(r1 - r0) * rs}")
    w.write_tensor_data(data)
    tot_in += int(ref[n].n_bytes)
    tot_out += data.nbytes
    if (k + 1) % 100 == 0 or k + 1 == len(pick):
        print(f"  {k + 1}/{len(pick)} tensors, {tot_in / 1e9:.2f} GB in REF -> {tot_out / 1e9:.2f} GB", flush=True)
w.close()
print(f"wrote {a.out}: {len(pick)} tensors ({', '.join(x.name for x in frm)} in REF -> {'copy' if a.copy else a.to}), "
      f"{tot_in / 1e9:.2f} -> {tot_out / 1e9:.2f} GB")

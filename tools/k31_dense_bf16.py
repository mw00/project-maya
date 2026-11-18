"""k31_dense_bf16.py: llama.cpp's convert_hf_to_gguf.py on the original K3.1 weights with the routed experts dropped
(not repacked): the dense-only GGUF that tools/k31_dense_requant.py quantizes into a --dense-override.

  python3 tools/k31_dense_bf16.py MODEL_DIR --outtype bf16 --outfile OUT.gguf [any convert_hf_to_gguf.py flag]

Everything else is the upstream converter as it is (the same names, shapes and metadata as a full conversion), from
the tree in $LLAMA_UP (default ~/work/llama-up).  The routed experts are MXFP4 in the original (compressed-tensors):
their packed and scale tensors are removed from the converter's tensor list unread, so only the dense tensors are
read from MODEL_DIR (~111 GB for K3.1).  An expert tensor in another form (a BF16 checkpoint) is skipped too."""
import logging
import os
import sys

LL = os.environ.get("LLAMA_UP", os.path.expanduser("~/work/llama-up"))
sys.path.insert(0, LL)
sys.path.insert(0, os.path.join(LL, "gguf-py"))
import convert_hf_to_gguf as C  # noqa: E402
from conversion import kimi_k3 as K  # noqa: E402

log = logging.getLogger("k31_dense_bf16")


def _drop_experts(self):   # in place of KimiK3Model._write_mxfp4_experts: the experts leave the list unread
    packed = [n for n in self.model_tensors if self._MXFP4_EXPERT_RE.match(n)]
    gone = 0
    for n in packed + [p.removesuffix("_packed") + "_scale" for p in packed]:
        if self.model_tensors.pop(n, None) is not None:
            gone += 1
    log.info(f"routed experts dropped: {gone} tensors (packed and scales), not read")


_get = K.KimiK3Model.get_tensors


def _get_dense(self):
    for name, data in _get(self):
        if ".block_sparse_moe.experts." in name:   # (an expert in a form the MXFP4 path did not take)
            continue
        yield name, data


K.KimiK3Model._write_mxfp4_experts = _drop_experts
K.KimiK3Model.get_tensors = _get_dense

if __name__ == "__main__":
    C.main()

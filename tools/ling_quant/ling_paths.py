"""tools/ling_quant/ling_paths.py - where the Ling tools look for things: these environment variables, else the
defaults below (an argument with no default left becomes required).

    LING_FP8        the inclusionAI/Ling-3.2-air checkpoint folder (FP8, as released)
    LING_WORK       the work folder: seqs/ (prep_calib.py), stats/ (ling_calib.py)       default ./ling-work
    LING_EVALDATA   the zero-shot task data (ARC, HellaSwag, WinoGrande, PIQA; tools/maya_quant/zs_tasks.py)
    MAYA_GGML_BASE  libggml-base.so of a llama.cpp build (quantisation and dequantisation through ggml)
    MAYA_GGUF_PY    llama.cpp's gguf-py                    default: the copy Maya's build fetches, else the gguf package
    MAYA_LING       the maya-ling binary                   default: build/maya-ling of this checkout
"""
import os
import pathlib

REPO = pathlib.Path(__file__).resolve().parents[2]
FP8 = os.environ.get("LING_FP8")
WORK = pathlib.Path(os.environ.get("LING_WORK", "ling-work"))
EVAL = str(WORK / "seqs" / "eval.npy")
REF = str(WORK / "stats" / "ref_topk.npz")
EVALDATA = os.environ.get("LING_EVALDATA")
GGML = os.environ.get("MAYA_GGML_BASE")
GGUF_PY = os.environ.get("MAYA_GGUF_PY", str(REPO / "build/_deps/strata_llamacpp-src/gguf-py"))
MAYA_LING = os.environ.get("MAYA_LING", str(REPO / "build/maya-ling"))

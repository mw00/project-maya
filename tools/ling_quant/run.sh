#!/bin/bash
# tools/ling_quant/run.sh - the Ling-3.2-air quants end to end: calibration sequences, the statistics and reference
# pass, the BF16 GGUF, then each tier quantised from its recipe and scored against the reference.
#
#   LING_FP8=/path/to/Ling-3.2-air MAYA_GGML_BASE=/path/to/libggml-base.so LLAMA_CPP=/path/to/llama.cpp \
#   LING_CALIB_DATA=/path/to/datasets [LING_WORK=ling-work] [TIERS="xs s m l"] [THREADS=64] tools/ling_quant/run.sh
#
# Every step is skipped when its output is already there, so a stopped run picks up where it was.
set -euo pipefail
: "${LING_FP8:?set LING_FP8 to the inclusionAI/Ling-3.2-air checkpoint folder}"
: "${MAYA_GGML_BASE:?set MAYA_GGML_BASE to libggml-base.so of a llama.cpp build}"
: "${LLAMA_CPP:?set LLAMA_CPP to a llama.cpp checkout (convert_hf_to_gguf.py)}"
: "${LING_CALIB_DATA:?set LING_CALIB_DATA to the calibration datasets (see README.md)}"
export LING_FP8 MAYA_GGML_BASE
export LING_WORK=${LING_WORK:-ling-work}
PY=${PY:-python}
THREADS=${THREADS:-64}
H=$(cd "$(dirname "$0")" && pwd)
W=$LING_WORK
mkdir -p "$W/seqs" "$W/stats" "$W/gguf" "$W/quants" "$W/results"

if [ ! -f "$W/seqs/eval.npy" ]; then
  echo "$(date +%T) 1/4 calibration and evaluation sequences"
  $PY "$H/prep_calib.py" --tok "$LING_FP8" --data "$LING_CALIB_DATA" --out "$W/seqs"
fi
if [ ! -f "$W/stats/ref_mtp_topk.npz" ]; then
  echo "$(date +%T) 2/4 statistics and reference (FP8 weights in fp32, layer by layer)"
  $PY "$H/ling_calib.py" --model "$LING_FP8" --calib "$W/seqs/calib.npy" --eval "$W/seqs/eval.npy" --out "$W/stats"
fi
BF16=$W/gguf/Ling-3.2-air-BF16.gguf
if [ ! -f "$BF16" ]; then
  echo "$(date +%T) 3/4 BF16 GGUF (the experts' source)"
  $PY "$LLAMA_CPP/convert_hf_to_gguf.py" "$LING_FP8" --outtype bf16 --outfile "$BF16"
fi
for t in ${TIERS:-xs s m l}; do
  T=$(echo "$t" | tr a-z A-Z)
  OUT=$W/quants/$t
  mkdir -p "$OUT"
  if [ ! -f "$OUT/quantized.ok" ]; then
    echo "$(date +%T) 4/4 quantising $T"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY "$H/ling_quantize.py" --bf16 "$BF16" --stats "$W/stats" \
      --recipe "$H/recipes/ling-$t.json" --out "$OUT/Ling-3.2-air-$T.gguf" --gptq --threads "$THREADS" \
      --name "Ling-3.2-air $T"
    touch "$OUT/quantized.ok"
  fi
  F=$(ls "$OUT"/Ling-3.2-air-"$T"*.gguf | head -1)   # the file, or its first shard
  echo "$(date +%T) scoring $T"
  $PY "$H/ling_eval.py" --gguf "$F" --threads "$THREADS" --label "Ling-3.2-air $T" --json "$W/results/eval.jsonl"
done
echo "$(date +%T) done: $W/quants, scores in $W/results/eval.jsonl"

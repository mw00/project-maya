#!/bin/bash
# Reference tensor dump for maya-ik2 (docs/INKLING2_AMD.md): llama.cpp's Inkling 2 (a checkout with
# examples/inkling-dump = this directory's inkling-dump.cpp, built as build-hip), run in a ROCm container.
#   run_dump.sh <name> <prompt file> [steps] [regex]   ->  $REF/<name>/
# Environment (all required): LLAMA the llama.cpp checkout, MODELS the folder holding Inkling-2-Small-GGUF/,
# REF where the dumps go, IMAGE a ROCm container image with podman.
#   LLAMA_ARGS replaces the attention flags (default "-fa on"), e.g. LLAMA_ARGS="-fa off -ctk f32 -ctv f32".
set -euo pipefail
: "${LLAMA:?set LLAMA to the llama.cpp checkout}" "${MODELS:?set MODELS}" "${REF:?set REF}" "${IMAGE:?set IMAGE}"
NAME=$1; PF=$(realpath "$2"); STEPS=${3:-4}
FILTER=${4:-'^(inp_embd|attn_out-|ffn_inp-|ffn_out-|l_out-|ffn_moe_topk-|ffn_moe_weights|ffn_moe_probs-|result_norm|result_output|(Qcur|Kcur|Vcur|Rcur|kq_b|kqv_out)-(0|5)$)'}
mkdir -p "$REF/$NAME"
exec podman run --rm --name inkling-dump-$NAME --network=host --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --ipc=host \
  -v "$LLAMA":/llama:z -v "$MODELS":/models:z -v "$REF":/ref:z -v "$(dirname "$PF")":/prompt:z \
  -e INK_DUMP_DIR=/ref/$NAME -e INK_DUMP_STEPS=$STEPS -e INK_DUMP_FILTER="$FILTER" \
  --entrypoint /bin/bash "$IMAGE" -c "
export LD_LIBRARY_PATH=/llama/build-hip/bin:\$LD_LIBRARY_PATH
/llama/build-hip/bin/llama-inkling-dump \
  -m /models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf \
  ${LLAMA_ARGS:--fa on} -c 4096 -b 4096 -ub 4096 -t 16 -f /prompt/$(basename "$PF")"

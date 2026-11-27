#!/bin/bash
# The DSpark drafter's acceptance in llama.cpp (docs/INKLING2_DSPARK.md): a checkout of llama.cpp PR #32218 with
# this directory's inkling-dspark-eval.cpp, built as build-hip, run in a ROCm container.
#   run_dspark_eval.sh <out name under $REF> <N> <prompts "name=file,.."> <variants file> [llama.cpp flags]
# The prompt files as the container sees them: under /llama, /models or /ref.
# Environment (all required): LLAMA the PR's checkout, MODELS the folder holding Inkling-2-Small-GGUF/ and
# Inkling-2-Small-MTP/, REF where the results go, IMAGE a ROCm container image with podman.
set -euo pipefail
: "${LLAMA:?set LLAMA to the PR 32218 checkout}" "${MODELS:?set MODELS}" "${REF:?set REF}" "${IMAGE:?set IMAGE}"
OUT=$1; N=$2; PROMPTS=$3; VARS=$(cat "$4"); shift 4
mkdir -p "$REF/$OUT"
exec podman run --rm --name inkling-dspark-eval --network=host --device /dev/kfd --device /dev/dri --group-add keep-groups \
  --security-opt seccomp=unconfined --ipc=host \
  -v "$LLAMA":/llama:z -v "$MODELS":/models:z -v "$REF":/ref:z \
  -e INK_OUT=/ref/$OUT -e INK_N=$N -e INK_PROMPTS="$PROMPTS" -e INK_VARIANTS="$VARS" -e INK_TAP_EXTRA=1 \
  --entrypoint /bin/bash "$IMAGE" -c "
export LD_LIBRARY_PATH=/llama/build-hip/bin:\$LD_LIBRARY_PATH
/llama/build-hip/bin/llama-inkling-dspark-eval \
  -m /models/Inkling-2-Small-GGUF/UD-Q2_K_XL/Inkling-2-Small-UD-Q2_K_XL-00001-of-00006.gguf \
  -md /models/Inkling-2-Small-MTP/Inkling-2-Small-DSpark-BF16.gguf --spec-type draft-dflash \
  -fa on -c 4096 -t 16 $*"

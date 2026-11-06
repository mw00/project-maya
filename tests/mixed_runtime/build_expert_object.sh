#!/usr/bin/env bash
# PB1 compilation gate only: no quantized kernel execution or engine integration.
set -euo pipefail
src=$(cd -- "$(dirname -- "$0")/../.." && pwd)
out=${1:?usage: build_expert_object.sh OUTPUT_DIRECTORY}
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
cd "$src"
"${HIPCC:-hipcc}" -x hip -std=c++20 -O2 --offload-arch=gfx1151 \
    -fPIC -ffunction-sections -fdata-sections -fvisibility=hidden \
    -DSTRATA_USE_HIP=1 -DSTRATA_HAS_WMMA=1 \
    -Iinclude/strata/hip_compat -Iinclude -Ithird_party/ggml \
    -include include/strata/hip_compat/cuda_runtime.h \
    -c src/kernels/cuda/glm_fast.cu -o "$out/glm_fast_hip.o"

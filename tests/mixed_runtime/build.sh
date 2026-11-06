#!/usr/bin/env bash
set -euo pipefail
src=$(cd -- "$(dirname -- "$0")" && pwd)
out=${1:?usage: build.sh OUTPUT_DIRECTORY}
mkdir -p "$out"
out=$(cd -- "$out" && pwd)
"${HIPCC:-hipcc}" -std=c++17 -O2 -fPIC -fvisibility=hidden --offload-arch="${HIP_ARCH:-gfx1151}" \
    -Wl,--version-script="$src/exports.map" -shared "$src/hip_helper.hip" -o "$out/libmaya_hip_probe.so"
"${NVCC:-/usr/local/cuda/bin/nvcc}" -std=c++17 -O2 -arch="${CUDA_ARCH:-sm_120}" \
    "$src/probe.cu" -ldl -o "$out/maya_mixed_probe"

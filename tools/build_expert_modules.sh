#!/usr/bin/env bash
# Build isolated executors from Maya's existing kernels. No dual-runtime engine.
set -euo pipefail
if [[ $# -lt 1 || $# -gt 2 || (${2:-} != '' && ${2:-} != --with-probe) ]]; then
    echo "usage: $0 OUTPUT_DIRECTORY [--with-probe]" >&2
    exit 2
fi
src=$(cd -- "$(dirname -- "$0")/.." && pwd)
mkdir -p "$1"
out=$(cd -- "$1" && pwd)
cd "$src"
sources=(src/core/expert_executor.cu src/kernels/cuda/glm_fast.cu)
exports=src/core/expert_executor.map
if [[ ${2:-} == --with-probe ]]; then
    sources+=(tests/mixed_runtime/expert_module.cu)
    exports=tests/mixed_runtime/expert_exports.map
fi
hipcc=${HIPCC:-hipcc}
nvcc=${NVCC:-/usr/local/cuda/bin/nvcc}
hipflags=(-std=c++20 -O2 -x hip --offload-arch="${HIP_ARCH:-gfx1151}"
    -DSTRATA_USE_HIP=1 -DSTRATA_HAS_WMMA=1
    -fPIC -ffunction-sections -fdata-sections -fvisibility=hidden
    -Iinclude/strata/hip_compat -Iinclude -Ithird_party/ggml
    -include include/strata/hip_compat/cuda_runtime.h)
cudaflags=(-std=c++20 -O2 -arch="${CUDA_ARCH:-sm_120}"
    -Xcompiler=-fPIC,-ffunction-sections,-fdata-sections,-fvisibility=hidden
    -Iinclude -Ithird_party/ggml)
hipobjects=()
cudaobjects=()
for file in "${sources[@]}"; do
    name=$(basename "$file" .cu)
    "$hipcc" "${hipflags[@]}" -c "$file" -o "$out/${name}_hip.o"
    hipobjects+=("$out/${name}_hip.o")
done
"$hipcc" -shared -Wl,--gc-sections -Wl,-z,defs -Wl,--version-script="$exports" \
    "${hipobjects[@]}" -o "$out/libmaya_expert_hip.so"
for file in "${sources[@]}"; do
    name=$(basename "$file" .cu)
    "$nvcc" "${cudaflags[@]}" -c "$file" -o "$out/${name}_cuda.o"
    cudaobjects+=("$out/${name}_cuda.o")
done
"$nvcc" -shared -Xlinker=--gc-sections,-z,defs,--version-script="$exports" \
    "${cudaobjects[@]}" -o "$out/libmaya_expert_cuda.so"

#!/usr/bin/env bash
# Builds libmaya_ds4_ernie_rx.so: maya-ds4's ERNIE-5.5 expert executor for an AMD helper GPU (HIP), loaded by the engine
# with DS4_ERNIE_RX=<this library> (DS4_ERNIE_RX_DEVICE=<HIP ordinal>, DS4_ERNIE_RX_MIB=<budget>).
# usage: HIP_ARCH=gfx1031 tools/ds4/build_ernie_rx.sh /absolute/output/dir     (hipcc from PATH or HIPCC=...)
set -euo pipefail
out=$(mkdir -p "$1" && cd "$1" && pwd)
src=$(cd -- "$(dirname -- "$0")/../.." && pwd)
cd "$src"
hipcc=${HIPCC:-hipcc}
flags=(-std=c++20 -O2 -x hip --offload-arch="${HIP_ARCH:-gfx1031}" -DMAYA_USE_HIP=1 -DMAYA_DS4_IQ_ONLY
       -fPIC -ffunction-sections -fdata-sections -fvisibility=hidden
       -Iinclude/maya/hip_compat -Iinclude -Isrc -Isrc/ds4 -Ithird_party/ggml
       -include include/maya/hip_compat/cuda_runtime.h)
objs=()
for f in src/ds4/ds4_ernie_rx.cu src/ds4/ds4_gguf.cu src/ds4/ds4_iq.cu ${EXTRA_SRC:-}; do
    o="$out/$(basename "$f" .cu).o"
    if [ ! -f "$o" ] || [ "$f" -nt "$o" ] || [ -n "${FORCE:-}" ]; then "$hipcc" "${flags[@]}" -c "$f" -o "$o"; fi
    objs+=("$o")
done
printf '{ global: ernx_get_api; local: *; };\n' > "$out/ernx.map"
rocm=${ROCM_PATH:-/opt/rocm}
"$hipcc" -shared -Wl,--gc-sections -Wl,-z,defs -Wl,--version-script="$out/ernx.map" "${objs[@]}" \
    -L"$rocm/lib" -lhipblas -lhipblaslt -o "$out/libmaya_ds4_ernie_rx.so"
echo "built: $out/libmaya_ds4_ernie_rx.so"

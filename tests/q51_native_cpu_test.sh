#!/usr/bin/env bash
# tests/q51_native_cpu_test.sh - build and run the Q5_1 native-path test on a host with no CUDA or
# ROCm toolkit (no nvcc, no hipcc on this box).
#
# WHY the transform step: native_mmvq.cu and dequant_bf16.cu are CUDA translation units, and g++
# cannot parse the kernel-launch syntax `kernel<<<grid, block, shared, stream>>>(args)`.  Stripping
# the launch config leaves a plain call with the kernel's own argument list, so the host functions
# (native_mmvq_supported, native_mmvq_weight_bytes, dequant_bf16_supported) and the device helpers
# the test calls directly (group32, small_q8_dot) compile and run unchanged.  Nothing else in the
# sources is touched, and the CPU test never launches a kernel, so no stripped launch is executed.
#
# usage: q51_native_cpu_test.sh <scratch-dir> <exit-stem> [--gate-only]
# writes <scratch-dir>/<exit-stem>.exit (plain int), .out and .err
#
# The red leg runs --gate-only, because parts B and C name the Q5_1 symbols that do not exist
# before the change.  The green leg runs everything.

set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${1:?usage: q51_native_cpu_test.sh <scratch-dir> <exit-stem> [--gate-only]}"
EXIT_STEM="${2:?usage: q51_native_cpu_test.sh <scratch-dir> <exit-stem> [--gate-only]}"
shift 2

GENERATED_DIR="$SCRATCH_DIR/gen"
mkdir -p "$GENERATED_DIR" || exit 1

# Strip every <<<...>>> launch config, including the ones that wrap across a line break.
for source in native_mmvq dequant_bf16; do
    perl -0777 -pe 's/<<<.*?>>>/ /gs' \
        "$REPO_ROOT/src/kernels/cuda/$source.cu" > "$GENERATED_DIR/${source}_cpu.cu" || exit 1
    # A leftover <<< means the transform missed one and the build would fail for the wrong reason.
    if grep -q '<<<' "$GENERATED_DIR/${source}_cpu.cu"; then
        echo "q51_native_cpu_test: launch-config strip left a <<< in $source.cu" >&2
        exit 1
    fi
done

# The gate-only build drops parts B and C at compile time, because they name the Q5_1 symbols that
# do not exist in the pre-change tree.  Without this the RED leg would be a build failure instead of
# a failing assertion, and a build failure proves nothing about the capability query.
EXTRA_DEFINES=()
for argument in "$@"; do
    if [ "$argument" = "--gate-only" ]; then
        EXTRA_DEFINES+=(-DSTRATA_Q51_GATE_ONLY=1)
    fi
done

# -I tests/cpu_cuda_shim comes first so <cuda_runtime.h> and <cuda_fp16.h> resolve to the CPU
# stand-ins; -I gen lets the test #include the transformed units.  -Wno-unknown-pragmas because
# the sources carry CUDA's `#pragma unroll`, which g++ does not act on but must not choke on.
g++ -std=c++20 -O1 -Wall -Wno-unused-function -Wno-unknown-pragmas \
    "${EXTRA_DEFINES[@]}" \
    -I "$REPO_ROOT/include" \
    -I "$REPO_ROOT/tests/cpu_cuda_shim" \
    -I "$GENERATED_DIR" \
    -o "$SCRATCH_DIR/q51_native_test" \
    "$REPO_ROOT/tests/q51_native_test.cpp" \
    "$REPO_ROOT/tests/cpu_cuda_shim/iq_kernels_cpu_stub.cpp" 2> "$SCRATCH_DIR/$EXIT_STEM.build.err"
build_exit=$?
if [ "$build_exit" -ne 0 ]; then
    echo "q51_native_cpu_test: build failed (exit $build_exit); see $EXIT_STEM.build.err" >&2
    echo "$build_exit" > "$SCRATCH_DIR/$EXIT_STEM.exit"
    exit "$build_exit"
fi

"$SCRATCH_DIR/q51_native_test" "$@" > "$SCRATCH_DIR/$EXIT_STEM.out" 2> "$SCRATCH_DIR/$EXIT_STEM.err"
run_exit=$?
echo "$run_exit" > "$SCRATCH_DIR/$EXIT_STEM.exit"
cat "$SCRATCH_DIR/$EXIT_STEM.out"
if [ -s "$SCRATCH_DIR/$EXIT_STEM.err" ]; then
    cat "$SCRATCH_DIR/$EXIT_STEM.err" >&2
fi
exit "$run_exit"

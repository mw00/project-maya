#!/bin/bash
# ==============================================================================
# Project Maya: GLM-5.3-Flash (Maya-S-v2-IQ2_XXS) 256K Context Runner
# 3x Tesla V100 16GB + dual Intel Xeon E5-2690 v4 (28 cores, AVX2)
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV="$SCRIPT_DIR/.venv"
PYTHON="$VENV/bin/python"
CONFIG="$SCRIPT_DIR/maya-maya-s-v2-iq2_xxs.json"
PORT="${PORT:-8080}"

# Export optimized environment variables
export LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/local/cuda-12.6/targets/x86_64-linux/lib:${LD_LIBRARY_PATH}"
export STRATA_GLM_SPLIT="19,32"
export STRATA_GLM_RAM_RESIDENT="1"
export STRATA_GLM_CPU_LANE="24"
export STRATA_GLM_RESERVE_MB="360"
export STRATA_GLM_RESERVE_LATER_MB="360"
export STRATA_GLM_PROMOTE_MIN="2"
export STRATA_GLM_KV_INT8="1"
export STRATA_GLM_SPEC_CANCEL="1"
export STRATA_GLM_SPEC_SAMPLE="1"
export STRATA_GLM_KDA_GRAPH="1"
export STRATA_GLM_USAGE="/home/svniak/Maya-data/usage-glm.txt"

print_usage() {
    echo "======================================================================"
    echo " Project Maya 256K Context Runner"
    echo "======================================================================"
    echo "Usage: ./run_256k.sh <command> [options]"
    echo ""
    echo "Commands:"
    echo "  bench [8k|12k|4k|all]  Run automated sustained benchmark (default: 8k)"
    echo "  server                 Start resident OpenAI API server on port $PORT"
    echo "  cli                    Start direct engine CLI in --serve mode"
    echo "  status                 Check running server and GPU VRAM usage"
    echo "  stop                   Cleanly stop any running strata / server processes"
    echo ""
    echo "Examples:"
    echo "  ./run_256k.sh bench 8k      # Benchmark 240K prompt + 8K decode"
    echo "  ./run_256k.sh bench 12k     # Benchmark 240K prompt + 12K decode (252K envelope)"
    echo "  ./run_256k.sh server        # Start HTTP API on http://localhost:$PORT"
    echo "  ./run_256k.sh status        # Check memory and server PID"
    echo "  ./run_256k.sh stop          # Shut down engine and free VRAM"
    echo "======================================================================"
}

cmd_status() {
    echo "=== Running Processes ==="
    if pgrep -a strata >/dev/null 2>&1; then
        pgrep -a strata
    else
        echo "No strata processes running."
    fi
    echo ""
    echo "=== GPU Memory Usage ==="
    nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
}

cmd_stop() {
    echo "Stopping any running strata or server processes..."
    pkill -15 strata 2>/dev/null || true
    sleep 2
    pkill -9 strata 2>/dev/null || true
    pkill -9 -f "serve.server" 2>/dev/null || true
    echo "Done."
    echo ""
    cmd_status
}

cmd_bench() {
    TARGET="${1:-8k}"
    echo "Starting sustained benchmark for target: $TARGET"
    echo "Config: $CONFIG (Max Context: 262,144)"
    echo ""
    "$PYTHON" "$SCRIPT_DIR/bench_sustained_suite.py" --target "$TARGET"
}

cmd_server() {
    echo "Starting Project Maya OpenAI API server on port $PORT (256K context) ..."
    exec "$PYTHON" -m serve.server --engine strata --config "$CONFIG" --port "$PORT" "$@"
}

cmd_cli() {
    echo "Starting direct strata engine in --serve mode (max-context 262144) ..."
    exec ./build/strata --serve \
      --glm-pack /home/svniak/Maya-data/models/Maya-S-v2-IQ2_XXS/pack \
      --max-context 262144 \
      --expert-cache auto \
      --prefill auto \
      --kv-streaming on \
      --kv int8 \
      --kv-resident 20480 \
      --vram-reserve-mib 360 \
      --vram-reserve-later-mib 360 \
      --expert-profile /home/svniak/project-maya/data/expert-profile-glm.bin
}

case "$1" in
    bench)
        shift
        cmd_bench "$@"
        ;;
    server)
        shift
        cmd_server "$@"
        ;;
    cli)
        cmd_cli
        ;;
    status)
        cmd_status
        ;;
    stop)
        cmd_stop
        ;;
    *)
        print_usage
        exit 1
        ;;
esac

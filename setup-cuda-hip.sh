#!/bin/sh
# Optional paired setup; reuse Maya's private Python environment and launcher.
cd "$(dirname "$0")" || exit 1
exec ./maya.sh --setup-cuda-hip "$@"

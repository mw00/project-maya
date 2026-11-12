#!/usr/bin/env bash
# Compatibility entry point: fixtures also need the stateless diagnostic export.
set -euo pipefail
src=$(cd -- "$(dirname -- "$0")/../.." && pwd)
exec bash "$src/tools/build_expert_modules.sh" "$@" --with-probe

#!/usr/bin/env bash
# Optional paired setup; only the requested Ubuntu package step needs sudo.
set -euo pipefail
cd "$(dirname "$0")" || exit 1
if [[ $# == 0 ]]; then
    set -- --interactive --install-deps
fi
install=0; plan=0; yes=0; readonly=0; interactive=0
for arg in "$@"; do
    case "$arg" in
        --install-deps) install=1 ;;
        --deps-plan) plan=1 ;;
        --yes) yes=1 ;;
        --check|--list-devices|--help|-h) readonly=1 ;;
        --interactive) interactive=1 ;;
    esac
done
if (( readonly )); then
    # Help and preflight never install packages, even with --install-deps.
    exec ./maya.sh --setup-cuda-hip "$@"
fi
if (( interactive )) && [[ ! -t 0 ]]; then
    echo 'Interactive setup needs a terminal. Pass --config, --hip-device and --helper-mib for scripted setup.' >&2
    exit 1
fi
if (( install || plan )); then
    # The distribution supplies both toolkits on this tested OS. Do not add
    # repositories for a different Ubuntu release or install graphics drivers.
    [[ -r /etc/os-release ]] && source /etc/os-release
    if [[ ${ID:-} != ubuntu || ${VERSION_ID:-} != 26.04 || $(uname -m) != x86_64 ]]; then
        echo 'Automatic system dependencies require Ubuntu 26.04 x86_64. On other Linux systems, install CUDA/HIP and rerun without --install-deps.' >&2
        exit 1
    fi
    packages=(build-essential python3 python3-venv git pkg-config
              cmake ninja-build cuda-toolkit-13-1 hipcc libamdhip64-dev)
    missing=()
    for pkg in "${packages[@]}"; do
        if [[ $(dpkg-query -W -f='${Status}' "$pkg" 2>/dev/null || true) != 'install ok installed' ]]; then
            missing+=("$pkg")
        fi
    done
    if (( ${#missing[@]} )); then
        echo "Missing Ubuntu build packages: ${missing[*]}"
        if (( install && !plan )); then
            if (( !yes )); then
                read -r -p 'Install these dependencies using sudo apt-get? [y/N] ' answer
                [[ $answer == y || $answer == Y ]] || exit 1
            fi
            sudo apt-get update
        fi
        proposal=$(LC_ALL=C apt-get --simulate --no-install-recommends install "${missing[@]}")
        printf '%s\n' "$proposal"
        if grep -Eq '^(Remv |Inst (nvidia-driver-|nvidia-dkms-|libnvidia-|linux-(image|modules)|xserver-xorg-video-|amdgpu-dkms))' <<< "$proposal"; then
            echo 'Package plan changes drivers or removes packages. Resolve those prerequisites separately, then rerun setup.' >&2
            exit 1
        fi
        if (( !plan )); then
            sudo apt-get install -y --no-remove --no-install-recommends "${missing[@]}"
        fi
    else
        echo 'Ubuntu build dependencies are already installed.'
    fi
    (( !plan )) || exit 0
fi
exec ./maya.sh --setup-cuda-hip "$@"

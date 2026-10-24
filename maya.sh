#!/bin/sh
# Project Maya for Linux and macOS: the first run checks the PC, compiles the engine, asks before it downloads the
# model, builds the pack and starts the dashboard; later runs just start it.  ./maya.sh --help lists the options.
# Nothing is installed system-wide: a missing Python is reported with the command that installs it.
cd "$(dirname "$0")" || exit 1
# Python 3.10+ that can make a venv WITH pip: Debian/Ubuntu ship `venv` without `ensurepip` (that is the separate
# python3-venv package), and a venv made without it has no pip
ok_py() { "$1" -c 'import sys, venv, ensurepip; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; }
# a .venv from an earlier run that failed half-way has a python but no pip: start it again
if [ -x .venv/bin/python ] && ! .venv/bin/python -m pip --version >/dev/null 2>&1; then
  rm -rf .venv
fi
if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3 python; do
    if command -v $c >/dev/null 2>&1 && ok_py $c; then
      PY=$c; break
    fi
  done
  if [ -z "$PY" ]; then
    echo "Python 3.10 or newer with venv is needed. Install it, then run ./maya.sh again:"
    echo "  Ubuntu/Debian: sudo apt-get install -y python3 python3-venv python3-pip"
    echo "  Fedora:        sudo dnf install -y python3 python3-pip"
    echo "  Arch:          sudo pacman -S python python-pip"
    echo "  macOS:         brew install python  (or python.org's installer)"
    exit 1
  fi
  # a private environment inside this folder (system Python stays untouched; newer distros refuse global pip)
  $PY -m venv .venv || { rm -rf .venv; echo "could not create .venv: sudo apt-get install -y python3-venv"; exit 1; }
fi
exec .venv/bin/python maya.py "$@"

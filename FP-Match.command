#!/bin/bash
# FP-Match - double-click to start on macOS (also works on Linux: ./FP-Match.command).
# The first run installs the Python packages into .venv (a few minutes, needs internet).
cd "$(dirname "$0")" || exit 1
export PATH="/opt/homebrew/bin:/usr/local/bin:$PATH"

fail() { echo; echo "$1"; echo; read -r -p "Press Return to close..."; exit 1; }

if [ ! -x .venv/bin/python ]; then
  PY=""
  for c in python3.13 python3.12 python3.11 python3.10 python3 /usr/bin/python3; do
    if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys; sys.exit(sys.version_info < (3, 9))' >/dev/null 2>&1; then
      PY="$c"; break
    fi
  done
  [ -n "$PY" ] || fail "Python 3.9 or newer is needed. Install it from https://www.python.org/downloads/
(or run: xcode-select --install), then double-click FP-Match.command again."
  echo "First run: setting up FP-Match with $("$PY" --version). This takes a few minutes..."
  rm -rf .venv
  "$PY" -m venv .venv || fail "Could not create the Python environment."
fi

if ! cmp -s requirements.txt .venv/requirements.installed; then
  echo "Installing / updating packages..."
  .venv/bin/python -m pip install --upgrade pip >/dev/null &&
  .venv/bin/python -m pip install -r requirements.txt ||
    fail "Package installation failed. Check your internet connection and try again."
  cp requirements.txt .venv/requirements.installed
fi

exec .venv/bin/python launcher.py

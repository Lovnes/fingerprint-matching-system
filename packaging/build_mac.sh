#!/bin/bash
# Builds dist/FP-Match.app and dist/FP-Match-macOS-<arch>.zip.
# The app runs on Macs with the same CPU type as this one (arm64 = Apple silicon, x86_64 = Intel).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PYTHON:-python3}
"$PY" -m venv .build-venv
.build-venv/bin/python -m pip install --upgrade pip
.build-venv/bin/python -m pip install -r requirements.txt pyinstaller
.build-venv/bin/pyinstaller --noconfirm --clean --distpath dist --workpath build packaging/fpmatch.spec
rm -rf dist/FP-Match            # intermediate folder; the .app is the deliverable
ZIP="dist/FP-Match-macOS-$(uname -m).zip"
rm -f "$ZIP"
ditto -c -k --keepParent dist/FP-Match.app "$ZIP"
echo "Built dist/FP-Match.app and $ZIP"

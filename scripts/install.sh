#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != --execute ]]; then
  echo "Preview: create .venv with Python 3.10, install pinned CUDA 12.8 Torch and this package."
  echo "Requires a suitable NVIDIA driver, build tools, ffmpeg and python3.10-venv."
  echo "Run: bash scripts/install.sh --execute"
  exit 0
fi
if [[ -e .venv ]]; then echo ".venv already exists; refusing to replace an environment" >&2; exit 2; fi
python3.10 -m venv .venv
.venv/bin/python -m pip install --upgrade pip 'setuptools<81' wheel ninja
.venv/bin/python -m pip install torch==2.7.1+cu128 torchvision==0.22.1+cu128 --index-url https://download.pytorch.org/whl/cu128
DS_BUILD_OPS=0 .venv/bin/python -m pip install -e .
.venv/bin/python -m pip check
.venv/bin/python -m pip freeze > .venv/installed-freeze.txt

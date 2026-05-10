#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

if ! command -v dvc >/dev/null 2>&1; then
  echo "Error: dvc is not installed. Install with 'pip install dvc'."
  exit 1
fi

if [[ ! -d ".dvc" ]]; then
  if [[ -d ".git" ]]; then
    dvc init -q
  else
    dvc init -q --no-scm
  fi
fi

mkdir -p "$ROOT_DIR/dvc-storage"

if dvc remote list | grep -q "^local"; then
  dvc remote modify local url "$ROOT_DIR/dvc-storage"
else
  dvc remote add -d local "$ROOT_DIR/dvc-storage"
fi

if [[ -d "experiments" ]]; then
  dvc add experiments
fi

if [[ -d "plots" ]]; then
  dvc add plots
fi

dvc status

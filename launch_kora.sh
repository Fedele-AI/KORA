#!/usr/bin/env bash
set -euo pipefail

# Resolve project root as the dir of this script
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$PROJECT_ROOT"

# Ensure RAG folder exists
mkdir -p RAG
mkdir -p .kora/index

# Install deps using uv
if command -v uv >/dev/null 2>&1; then
  uv pip install -e .
else
  echo "[KORA] 'uv' not found. Please install UV and re-run."
  exit 1
fi

# Run the app with UV's runtime to ensure correct environment
uv run -m kora.launcher

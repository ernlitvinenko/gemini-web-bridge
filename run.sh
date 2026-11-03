#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

if [[ ! -x ".venv/bin/python" ]]; then
  echo "No .venv found. Run: uv sync" >&2
  exit 1
fi

exec .venv/bin/gemini-web-bridge "$@"

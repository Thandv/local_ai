#!/usr/bin/env bash
# Run the stdlib unittest suite. The repo root is the `local_ai` package, so
# its parent directory goes on PYTHONPATH. No network, Ollama or model needed:
# tests/support.py blocks urlopen, subprocess and socket connections.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
parent_dir="$(dirname "$repo_root")"

cd "$repo_root"
PYTHONPATH="$parent_dir" PYTHONDONTWRITEBYTECODE=1 \
    python3 -m unittest discover -s tests "$@"

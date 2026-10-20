#!/usr/bin/env bash
# Run the stdlib unittest suite. The repo root is the `local_ai` package, so a
# temp dir holding a `local_ai` symlink to it goes on PYTHONPATH; this works
# whatever the checkout folder is called. No network, Ollama or model needed:
# tests/support.py blocks urlopen, subprocess and socket connections.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
import_root="$(mktemp -d)"
trap 'rm -rf "$import_root"' EXIT
ln -s "$repo_root" "$import_root/local_ai"

cd "$repo_root"
PYTHONPATH="$import_root" PYTHONDONTWRITEBYTECODE=1 \
    python3 -m unittest discover -s tests "$@"

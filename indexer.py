#!/usr/bin/env python3
"""
Repo Cloner — clones curated high-quality GitHub repos locally so the
vibe-coder agent can grep them for real working patterns at query time.
No embeddings or vector DB needed — just fast grep-based search.
"""

import json
import subprocess
import sys
from pathlib import Path

REPOS_JSON = Path(__file__).parent / "repos.json"  # bundled with package
CLONE_DIR  = Path.home() / ".local-ai" / "repos"


def clone_or_update(url: str, dest: Path) -> bool:
    if (dest / ".git").exists():
        print(f"  updating {dest.name} …", end=" ", flush=True)
        r = subprocess.run(
            ["git", "-C", str(dest), "pull", "--ff-only", "--quiet"],
            capture_output=True,
        )
        print("done" if r.returncode == 0 else "skipped (local changes)")
    else:
        print(f"  cloning  {url} …", end=" ", flush=True)
        dest.parent.mkdir(parents=True, exist_ok=True)
        r = subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", url, str(dest)],
            capture_output=True,
        )
        if r.returncode != 0:
            print(f"FAILED\n  {r.stderr.decode()[:200]}")
            return False
        print("done")
    return True


def build_index(repos_json: Path = REPOS_JSON, limit: int = 0):
    repos = json.loads(repos_json.read_text())["repos"]
    if limit:
        repos = repos[:limit]

    CLONE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Cloning {len(repos)} repos into {CLONE_DIR}\n")

    ok, failed = 0, []
    for repo_info in repos:
        url  = repo_info["url"]
        name = url.split("/")[-1]
        dest = CLONE_DIR / name
        if clone_or_update(url, dest):
            ok += 1
        else:
            failed.append(name)

    print(f"\nDone — {ok} repos ready in {CLONE_DIR}")
    if failed:
        print(f"Failed: {', '.join(failed)}")
    print("\nYour vibe coder will now search these repos for patterns when you ask it to build something.")


if __name__ == "__main__":
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    build_index(limit=limit)

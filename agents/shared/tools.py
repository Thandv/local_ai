"""
Shared tool implementations and Ollama tool-schema definitions.
All agents import from here so tool behaviour is consistent.
"""

import json
import os
import re
import shutil
import subprocess
import glob as _glob
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Optional

from . import guard
from .guard import wrap_untrusted  # re-exported for the stages

MAX_OUTPUT = 10_000  # chars — truncate large outputs

REPOS_CURATED   = Path.home() / ".local-ai" / "repos"
REPOS_DISCOVERED = Path.home() / ".local-ai" / "repos-discovered"


# ── Implementations ───────────────────────────────────────────────────────────

def read_file(path: str) -> str:
    """Read a file inside the sandbox root (symlinks are not followed)."""
    try:
        text = guard.read_text(path)
    except guard.Refused as e:
        return f"Refused: {e}"
    except FileNotFoundError:
        return f"File not found: {path}"
    except OSError as e:
        return f"Error reading {path}: {e}"
    if len(text) > MAX_OUTPUT:
        text = text[:MAX_OUTPUT] + f"\n[truncated — {len(text)} chars total]"
    return text


def write_file(path: str, content: str) -> str:
    """Write a file inside the sandbox root; '..', absolute paths outside it,
    symlinks, hard links and .git/.venv/node_modules/.npmrc are refused."""
    try:
        guard.write_text(path, content)
    except guard.Refused as e:
        return f"Refused: {e}"
    except OSError as e:
        return f"Error writing {path}: {e}"
    return f"Written: {path} ({len(content)} chars)"


def list_files(pattern: str) -> str:
    """Glob inside the sandbox root; matches that resolve outside it are dropped."""
    try:
        sandbox = guard.current()
        parts = guard.relative_parts(sandbox, pattern)
    except guard.Refused as e:
        return f"Refused: {e}"
    if not parts:
        return f"No files matching: {pattern}"
    root = str(sandbox.root)
    absolute = os.path.isabs(os.path.expanduser(pattern))
    matches = []
    for m in _glob.glob(os.path.join(*parts), root_dir=root, recursive=True):
        if guard.is_inside(sandbox, os.path.realpath(os.path.join(root, m))):
            matches.append(os.path.join(root, m) if absolute else m)
    return "\n".join(sorted(matches)) if matches else f"No files matching: {pattern}"


def run_command(command: str, working_dir: Optional[str] = None) -> str:
    """Run one command through the LA-3 guard (agents/shared/guard.py):
    shell=False, allowlist or the person's confirmation, cwd inside the root."""
    try:
        argv = guard.split_command(command)
    except guard.Refused as e:
        return f"Refused: {e}"
    outcome = guard.run_argv(argv, working_dir)
    if not outcome.ran:
        return f"Refused: {outcome.reason}" if outcome.status != "timeout" \
            else f"Error: command {outcome.reason}"
    out = outcome.output
    if len(out) > MAX_OUTPUT:
        out = out[:MAX_OUTPUT] + "\n[truncated]"
    out = out or "(no output)"
    if outcome.returncode:
        out += f"\n[exit code {outcome.returncode}]"
    return out


def search_code(pattern: str, directory: str = ".") -> str:
    """grep inside the sandbox root (grep -r does not follow symlinks)."""
    try:
        sandbox = guard.current()
        if not isinstance(pattern, str) or "\x00" in pattern:
            raise guard.Refused("pattern must be a string without NUL bytes")
        target = guard.resolve_dir(sandbox, directory)
    except guard.Refused as e:
        return f"Refused: {e}"
    try:
        result = subprocess.run(
            ["grep", "-rn", "-i",
             "--include=*.py", "--include=*.ts", "--include=*.tsx",
             "--include=*.js", "--include=*.jsx", "--include=*.json",
             "--include=*.css", "--include=*.html", "--include=*.md",
             "-e", pattern, "--", target],
            capture_output=True, text=True, timeout=10, cwd=str(sandbox.root),
            stdin=subprocess.DEVNULL,
        )
        out = result.stdout
        if not out.strip():
            return f"No matches for '{pattern}' in {directory}"
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + "\n[truncated]"
        return out
    except Exception as e:
        return f"Error: {e}"


def grep_repos(keywords: list, repos_dir: str = None) -> str:
    """Search reference repos for patterns matching any of the keywords.

    Searches both the curated set (~/.local-ai/repos/) and any repos the
    Researcher has discovered and shallow-cloned this run
    (~/.local-ai/repos-discovered/).
    """
    if repos_dir:
        # Explicit override (e.g. tests) — search only the given dir
        roots = [Path(repos_dir)]
    else:
        roots = [REPOS_CURATED, REPOS_DISCOVERED]
    roots = [r for r in roots if r.exists()]
    if not roots:
        return "No reference repos found. Run: ai-index"

    skip = {"node_modules", ".git", "__pycache__", ".next", "dist", "build",
            ".venv", "venv", "test_", "_test", ".min.", "migration"}

    snippets, seen = [], set()
    for kw in keywords[:5]:
        if len(snippets) >= 6:
            break
        for root in roots:
            if len(snippets) >= 6:
                break
            try:
                r = subprocess.run(
                    ["grep", "-rl", "--include=*.py", "--include=*.ts",
                     "--include=*.tsx", "--include=*.js", "-i", kw, str(root)],
                    capture_output=True, text=True, timeout=5
                )
                for fpath in r.stdout.splitlines():
                    if fpath in seen or len(snippets) >= 6:
                        break
                    if any(s in fpath for s in skip):
                        continue
                    seen.add(fpath)
                    lines = Path(fpath).read_text(encoding="utf-8", errors="ignore").splitlines()
                    start = next((max(0, i - 3) for i, l in enumerate(lines)
                                  if kw.lower() in l.lower()), 0)
                    chunk = "\n".join(lines[start: start + 35])
                    repo  = Path(fpath).relative_to(root).parts[0]
                    rel   = "/".join(Path(fpath).relative_to(root).parts[1:])
                    origin = "discovered" if root == REPOS_DISCOVERED else "curated"
                    snippets.append(f"### {repo} / {rel}  ({origin})\n```\n{chunk}\n```")
            except Exception:
                continue

    return "\n\n".join(snippets) if snippets else "No relevant patterns found in reference repos."


# ── On-demand GitHub research ─────────────────────────────────────────────────

def _gh_cli_available() -> bool:
    return shutil.which("gh") is not None


def search_github(query: str, language: Optional[str] = None,
                  min_stars: int = 200, n: int = 5) -> str:
    """Search GitHub for repos matching a query and return a short table.

    Use this when grep_repos returns little ("No relevant patterns found")
    and the prompt is in a domain that isn't covered by the curated repos
    (e.g. video generation, audio synthesis, scientific computing).

    Prefers the `gh` CLI if it's installed; falls back to the public
    GitHub Search API otherwise. Both have rate limits — don't call this
    in a tight loop.
    """
    n = max(1, min(int(n), 20))
    star_filter = f"stars:>={int(min_stars)}"
    lang_filter = f"language:{language}" if language else ""

    if _gh_cli_available():
        try:
            cmd = ["gh", "search", "repos", query,
                   "--limit", str(n), "--stars", f">={int(min_stars)}",
                   "--json", "fullName,description,stargazersCount,url,language"]
            if language:
                cmd += ["--language", language]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if r.returncode != 0:
                return f"gh search failed: {r.stderr.strip()[:300]}"
            data = json.loads(r.stdout or "[]")
        except Exception as e:
            return f"gh search error: {e}"
    else:
        # Fallback: public REST API (unauthenticated; lower rate limits)
        try:
            q = " ".join(p for p in [query, star_filter, lang_filter] if p)
            url = (f"https://api.github.com/search/repositories"
                   f"?q={urllib.parse.quote(q)}&per_page={n}&sort=stars")
            req = urllib.request.Request(
                url, headers={"User-Agent": "local-ai-dev/2.0",
                              "Accept": "application/vnd.github+json"})
            with urllib.request.urlopen(req, timeout=20) as resp:
                payload = json.loads(resp.read())
            data = [{
                "fullName":         it["full_name"],
                "description":      it.get("description") or "",
                "stargazersCount":  it["stargazers_count"],
                "url":              it["html_url"],
                "language":         it.get("language") or "",
            } for it in payload.get("items", [])[:n]]
        except urllib.error.HTTPError as e:
            return f"GitHub API error: HTTP {e.code} (likely rate-limited; install `gh` to authenticate)"
        except Exception as e:
            return f"GitHub API error: {e}"

    if not data:
        return f"No repos found for query: {query!r}"

    lines = [f"Top {len(data)} repos for {query!r} (min {min_stars}★):"]
    for d in data:
        stars = d.get("stargazersCount", 0)
        desc  = (d.get("description") or "").strip()[:120]
        lines.append(f"  {d['fullName']}  ★{stars}  ({d.get('language') or '?'})")
        if desc:
            lines.append(f"    {desc}")
        lines.append(f"    {d['url']}")
    return wrap_untrusted("\n".join(lines), f"search_github {query}")


_GH_NAME = re.compile(r"[A-Za-z0-9_.-]{1,100}")


def _owner_repo(repo_url: str) -> Optional[tuple[str, str]]:
    """Parse owner/repo from a URL or 'owner/repo'; None unless both are plain
    GitHub names (so neither can be '..', a path or an option)."""
    if not isinstance(repo_url, str):
        return None
    if repo_url.startswith("http"):
        parts = repo_url.rstrip("/").split("github.com/")[-1].split("/")
    else:
        parts = repo_url.strip().split("/")
    if len(parts) < 2:
        return None
    owner, repo = parts[0], parts[1]
    if repo.endswith(".git"):
        repo = repo[:-4]
    for name in (owner, repo):
        if not _GH_NAME.fullmatch(name) or name in (".", "..") or name.startswith("-"):
            return None
    return owner, repo


def peek_readme(repo_url: str) -> str:
    """Fetch a repo's README without cloning, wrapped as untrusted data."""
    return wrap_untrusted(_peek_readme(repo_url), f"peek_readme {repo_url}")


def _peek_readme(repo_url: str) -> str:
    """Fetch a repo's README without cloning. Use this to decide if a repo
    found via search_github is worth shallow-cloning. Owner/repo can be the
    full URL (https://github.com/owner/repo) or just owner/repo."""
    parsed = _owner_repo(repo_url)
    if parsed is None:
        return f"Could not parse owner/repo from {repo_url!r}"
    owner, repo = parsed

    if _gh_cli_available():
        try:
            r = subprocess.run(
                ["gh", "api", f"repos/{owner}/{repo}/readme",
                 "--jq", ".content", "-H", "Accept: application/vnd.github.raw"],
                capture_output=True, text=True, timeout=20,
            )
            if r.returncode == 0 and r.stdout.strip():
                # `--jq` strips the JSON wrapper; raw header gives plain text
                # but gh might still wrap; handle both
                content = r.stdout
                if not content.startswith("#") and len(content) > 500 and "\n" not in content[:200]:
                    # Looks base64
                    import base64
                    try:
                        content = base64.b64decode(content).decode("utf-8", errors="ignore")
                    except Exception:
                        pass
                return content[:MAX_OUTPUT] + (
                    f"\n[truncated — full README at {repo_url}]" if len(content) > MAX_OUTPUT else "")
        except Exception:
            pass

    # Fallback: raw README via GitHub API
    try:
        import base64
        req = urllib.request.Request(
            f"https://api.github.com/repos/{owner}/{repo}/readme",
            headers={"User-Agent": "local-ai-dev/2.0",
                     "Accept": "application/vnd.github+json"})
        with urllib.request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read())
        content = base64.b64decode(payload.get("content", "")).decode("utf-8", errors="ignore")
        return content[:MAX_OUTPUT] + (
            f"\n[truncated — full README at {repo_url}]" if len(content) > MAX_OUTPUT else "")
    except urllib.error.HTTPError as e:
        return f"Could not fetch README (HTTP {e.code}): {repo_url}"
    except Exception as e:
        return f"Could not fetch README: {e}"


def clone_shallow(repo_url: str) -> str:
    """Shallow-clone a repo (see _clone_shallow); the result is wrapped as
    untrusted data because it can carry git's stderr from a remote."""
    return wrap_untrusted(_clone_shallow(repo_url), f"clone_shallow {repo_url}")


def _clone_shallow(repo_url: str) -> str:
    """Shallow-clone a repo into ~/.local-ai/repos-discovered/ so grep_repos
    can search it for code patterns. Idempotent: re-runs are no-ops.

    Use this AFTER peek_readme has confirmed the repo is relevant. Don't
    clone repos you haven't peeked.
    """
    parsed = _owner_repo(repo_url)
    if parsed is None:
        return f"Could not parse owner/repo from {repo_url!r}"
    owner, repo = parsed
    canonical = f"https://github.com/{owner}/{repo}.git"

    REPOS_DISCOVERED.mkdir(parents=True, exist_ok=True)
    dest = REPOS_DISCOVERED / repo

    if (dest / ".git").exists():
        return f"Already cloned: {dest}"

    try:
        r = subprocess.run(
            ["git", "clone", "--depth=1", "--quiet", canonical, str(dest)],
            capture_output=True, text=True, timeout=120,
        )
        if r.returncode != 0:
            return f"Clone failed: {r.stderr.strip()[:300]}"
        return f"Cloned {owner}/{repo} → {dest} (grep_repos will now find it)"
    except subprocess.TimeoutExpired:
        return "Clone timed out after 120s"
    except Exception as e:
        return f"Clone error: {e}"


# ── Tool schemas (Ollama format) ──────────────────────────────────────────────

def make_tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
            }
        }
    }


READ_FILE   = make_tool("read_file",   "Read a file's contents.",
                        {"path": {"type": "string"}}, ["path"])

WRITE_FILE  = make_tool("write_file",
                        "Write content to a file inside the project directory (creates dirs). "
                        "Paths outside it, symlinks, and .git/.venv/node_modules/.npmrc are refused.",
                        {"path": {"type": "string"}, "content": {"type": "string"}},
                        ["path", "content"])

LIST_FILES  = make_tool("list_files",  "List files matching a glob pattern.",
                        {"pattern": {"type": "string"}}, ["pattern"])

RUN_COMMAND = make_tool(
    "run_command",
    ("Run ONE command (no shell: no pipes, redirects, '&&' or ';') inside the project "
     "directory and return its output. Runs without asking: `python3 -m pytest`, "
     "`python3 -m unittest`, `python3 -m py_compile <file>`, `pytest`, `python3 -m venv .venv`, "
     "`npm test`, `node <script>`, `ls`, `cat`, `git status|diff|init|add|commit -m`. "
     "`pip install` goes into the project's .venv (create it first) and, like any other "
     "command, needs the person's confirmation; it is refused when nobody can confirm."),
    {"command": {"type": "string"},
     "working_dir": {"type": "string",
                     "description": "Optional directory inside the project"}},
                        ["command"])

SEARCH_CODE = make_tool("search_code", "Search for a pattern across source files.",
                        {"pattern": {"type": "string"},
                         "directory": {"type": "string", "description": "Directory to search"}},
                        ["pattern"])

SEARCH_GITHUB = make_tool(
    "search_github",
    ("Search GitHub for repos relevant to the project's domain. Use this when "
     "grep_repos returns little — typically because the prompt is in a domain "
     "the curated reference set doesn't cover (e.g. video generation, audio "
     "processing, scientific computing). Returns a short list of top repos "
     "with descriptions and star counts."),
    {"query":     {"type": "string",
                   "description": "Search terms, e.g. 'video generation python' or 'text to speech api'"},
     "language":  {"type": "string",
                   "description": "Optional language filter, e.g. 'python', 'typescript'"},
     "min_stars": {"type": "integer",
                   "description": "Minimum star count (default 200)"},
     "n":         {"type": "integer",
                   "description": "Max results (default 5, capped at 20)"}},
    ["query"],
)

PEEK_README = make_tool(
    "peek_readme",
    ("Fetch a GitHub repo's README without cloning it. Use this to decide "
     "whether a repo from search_github is worth shallow-cloning."),
    {"repo_url": {"type": "string",
                  "description": "Either https://github.com/owner/repo or owner/repo"}},
    ["repo_url"],
)

CLONE_SHALLOW = make_tool(
    "clone_shallow",
    ("Shallow-clone a repo (depth 1) into ~/.local-ai/repos-discovered/ so "
     "grep_repos can search its code. Call this AFTER peek_readme has "
     "confirmed relevance, not speculatively."),
    {"repo_url": {"type": "string",
                  "description": "Either https://github.com/owner/repo or owner/repo"}},
    ["repo_url"],
)


HANDLERS = {
    "read_file":     read_file,
    "write_file":    write_file,
    "list_files":    list_files,
    "run_command":   run_command,
    "search_code":   search_code,
    "search_github": search_github,
    "peek_readme":   peek_readme,
    "clone_shallow": clone_shallow,
}

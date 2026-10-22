#!/usr/bin/env python3
"""
Vibe Coder — tell it your idea, it builds the full app.
Powered by Ollama or Claude API + live grep-based retrieval from
cloned open-source repos. No vector DB required — just fast, precise search.

Model: set LOCAL_AI_MODEL env var or use --model flag.
  Local (default): qwen2.5-coder:7b via Ollama
  Cloud:           a model name starting with "claude" (Claude API; needs the
                   [claude] extra and ANTHROPIC_API_KEY)
"""

import os
import sys
import subprocess
import glob as glob_module
import re
from pathlib import Path

from local_ai.agents.shared.llm import (
    LLMUnavailableError, run_tool_loop, set_model, get_model,
)
from local_ai.agents.shared import guard
from local_ai.agents.shared.guard import shown
from local_ai.agents.shared.tools import (
    HANDLERS, LIST_FILES, READ_FILE, RUN_COMMAND, SEARCH_CODE, WRITE_FILE, wrap_untrusted,
)

# ── Config ────────────────────────────────────────────────────────────────────

REPOS_DIR   = Path.home() / ".local-ai" / "repos"
RAG_SNIPPETS = 5      # max snippets to inject per query
SNIPPET_LINES = 40    # lines per snippet

SYSTEM_PROMPT = """You are an expert vibe-coder — you turn ideas into complete, working software.

When a user gives you an idea:
1. Think through the full architecture first
2. Scaffold the complete project structure
3. Write EVERY file needed to make it work immediately
4. Include setup instructions in the output
5. Make it look good — use modern patterns and clean code

You have tools to read/write files and run commands. You also receive REFERENCE CODE
from top open-source projects — use their patterns and conventions as your quality bar.

Rules:
- Write COMPLETE files, never leave TODOs or placeholders
- Make it runnable: always include requirements.txt or package.json
- Use modern patterns: async FastAPI, React hooks, TypeScript types
- Clean readable code over clever code
- When asked to build something, build the whole thing — not a skeleton
"""

# ── Repo retrieval (grep-based) ───────────────────────────────────────────────

def extract_keywords(query: str) -> list[str]:
    """Pull meaningful technical terms from the user's query."""
    # Remove common stop words, keep technical nouns
    stop = {"a", "an", "the", "and", "or", "with", "for", "that", "this",
            "to", "of", "in", "on", "is", "how", "do", "i", "want", "build",
            "create", "make", "app", "me", "my", "need", "can", "use", "using"}
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]{2,}", query.lower())
    return [w for w in words if w not in stop][:6]


def grep_repos(keywords: list[str], max_snippets: int = RAG_SNIPPETS) -> str:
    """Search cloned repos for files relevant to the keywords, return code snippets."""
    if not REPOS_DIR.exists() or not keywords:
        return ""

    snippets = []
    seen_files = set()

    for keyword in keywords:
        if len(snippets) >= max_snippets:
            break
        try:
            result = subprocess.run(
                ["grep", "-rl", "--include=*.py", "--include=*.ts", "--include=*.tsx",
                 "--include=*.js", "--include=*.jsx",
                 "-i", keyword, str(REPOS_DIR)],
                capture_output=True, text=True, timeout=5
            )
            for fpath in result.stdout.splitlines():
                if fpath in seen_files or len(snippets) >= max_snippets:
                    break
                # Skip test files, migrations, lock files
                if any(s in fpath for s in ["test_", "_test", "migration", "node_modules",
                                             "__pycache__", ".min.", "lock", "dist/"]):
                    continue
                seen_files.add(fpath)
                try:
                    lines = Path(fpath).read_text(encoding="utf-8", errors="ignore").splitlines()
                    # Find the most relevant section (around the keyword match)
                    match_line = 0
                    for i, line in enumerate(lines):
                        if keyword.lower() in line.lower():
                            match_line = max(0, i - 5)
                            break
                    chunk = "\n".join(lines[match_line : match_line + SNIPPET_LINES])
                    repo = Path(fpath).relative_to(REPOS_DIR).parts[0]
                    rel  = Path(fpath).relative_to(REPOS_DIR / repo)
                    snippets.append(f"# {repo} / {rel}\n{chunk}")
                except Exception:
                    continue
        except Exception:
            continue

    return "\n\n---\n\n".join(snippets)

# ── Tools (the shared implementations in agents/shared/tools.py) ──────────────

TOOLS = [READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND, SEARCH_CODE]
TOOL_HANDLERS = {
    name: HANDLERS[name]
    for name in ("read_file", "write_file", "list_files", "run_command", "search_code")
}

# ── Agent loop ────────────────────────────────────────────────────────────────

def _show_call(name: str, args: dict) -> None:
    if name == "run_command":
        print(f"\n    $ {shown(args.get('command', ''))}")
    elif name == "write_file":
        print(f"\n    write {shown(args.get('path', ''))}")
    else:
        print(f"\n  [{name}]", end="")


def run_agent(messages: list) -> str:
    """Run one REPL turn on the shared, bounded tool loop (history kept in messages)."""
    return run_tool_loop(messages, TOOLS, TOOL_HANDLERS, on_tool_call=_show_call)

# ── REPL ──────────────────────────────────────────────────────────────────────

def repos_status() -> str:
    if not REPOS_DIR.exists():
        return "No repos cloned yet — run: ai-index"
    repos = [d.name for d in REPOS_DIR.iterdir() if d.is_dir()]
    return f"{len(repos)} repos: {', '.join(repos)}" if repos else "No repos cloned yet — run: ai-index"

def _main() -> None:
    import argparse
    parser = argparse.ArgumentParser(prog="vibe", add_help=False)
    parser.add_argument("--model", "-m", default=None)
    parser.add_argument("--yes-unsafe", action="store_true")
    args, _ = parser.parse_known_args()
    if args.model:
        set_model(args.model)

    project_dir = Path.cwd()
    # File and command tools are confined to the current directory; only the
    # person's --yes-unsafe flag skips the confirmation for unlisted commands.
    guard.activate(project_dir, yes_unsafe=args.yes_unsafe)
    print("\n  Local Vibe Coder")
    print(f"  Model  : {get_model()}")
    print(f"  Project: {project_dir}")
    print(f"  Refs   : {repos_status()}")
    print("  Type 'exit' to quit, 'clear' to reset\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    existing = [f for f in glob_module.glob("**/*", recursive=True) if Path(f).is_file()][:20]
    if existing:
        messages.append({
            "role": "system",
            "content": f"Existing project files in {project_dir}:\n" + "\n".join(existing)
        })

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "exit":
            print("Bye!")
            break
        if user_input.lower() == "clear":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation reset.\n")
            continue

        # Grep-based RAG: find relevant code from cloned repos
        keywords = extract_keywords(user_input)
        if keywords:
            context = grep_repos(keywords)
            if context:
                messages.append({
                    "role": "system",
                    "content": (
                        f"REFERENCE CODE from open-source repos (keywords: {', '.join(keywords)}):\n\n"
                        + wrap_untrusted(context[:6000], "reference repos")
                    )
                })

        messages.append({"role": "user", "content": user_input})
        print("\nThinking", end="", flush=True)

        reply = run_agent(messages)
        print(f"\n\nAI:\n{reply}\n")


def main() -> None:
    try:
        _main()
    except LLMUnavailableError as e:
        sys.exit(str(e))  # prints the message to stderr, exit status 1


if __name__ == "__main__":
    main()

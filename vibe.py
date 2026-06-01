#!/usr/bin/env python3
"""
Vibe Coder — tell it your idea, it builds the full app.
Powered by Ollama or Claude API + live grep-based retrieval from
cloned open-source repos. No vector DB required — just fast, precise search.

Model: set LOCAL_AI_MODEL env var or use --model flag.
  Local (default): qwen2.5-coder:32b via Ollama
  Cloud  (best):   claude-opus-4-7 (set ANTHROPIC_API_KEY)
"""

import os
import sys
import json
import subprocess
import glob as glob_module
import re
from pathlib import Path
from typing import Optional

from local_ai.agents.shared.llm import chat as _llm_chat, set_model, get_model

# ── Config ────────────────────────────────────────────────────────────────────

REPOS_DIR   = Path.home() / ".local-ai" / "repos"
RAG_SNIPPETS = 5      # max snippets to inject per query
SNIPPET_LINES = 40    # lines per snippet
MAX_OUTPUT   = 8000

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

# ── Tools ─────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file, creating parent directories as needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files matching a glob pattern.",
            "parameters": {
                "type": "object",
                "properties": {"pattern": {"type": "string"}},
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command and return its output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "working_dir": {"type": "string"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a text pattern across project files.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "directory": {"type": "string"}
                },
                "required": ["pattern"]
            }
        }
    }
]

# ── Tool implementations ───────────────────────────────────────────────────────

def read_file(path: str) -> str:
    p = Path(path).expanduser()
    if not p.exists():
        return f"File not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > MAX_OUTPUT:
            content = content[:MAX_OUTPUT] + f"\n[truncated — {len(content)} total chars]"
        return content
    except Exception as e:
        return f"Error: {e}"

def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    print(f"    wrote {path}")
    return f"Written: {path}"

def list_files(pattern: str) -> str:
    matches = glob_module.glob(pattern, recursive=True)
    return "\n".join(sorted(matches)) if matches else f"No files: {pattern}"

def run_command(command: str, working_dir: Optional[str] = None) -> str:
    cwd = Path(working_dir).expanduser() if working_dir else Path.cwd()
    print(f"\n    $ {command}")
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=120
        )
        out = (result.stdout + result.stderr).strip()
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + "\n[truncated]"
        return out or "(no output)"
    except subprocess.TimeoutExpired:
        return "Timed out after 120s"
    except Exception as e:
        return f"Error: {e}"

def search_code(pattern: str, directory: str = ".") -> str:
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
             "--include=*.jsx", "--include=*.tsx", "--include=*.json", "--include=*.css",
             pattern, directory],
            capture_output=True, text=True, timeout=10
        )
        out = result.stdout
        if not out.strip():
            return f"No matches for '{pattern}'"
        if len(out) > MAX_OUTPUT:
            out = out[:MAX_OUTPUT] + "\n[truncated]"
        return out
    except Exception as e:
        return f"Error: {e}"

TOOL_HANDLERS = {
    "read_file": read_file,
    "write_file": write_file,
    "list_files": list_files,
    "run_command": run_command,
    "search_code": search_code,
}

def extract_json_tool_call(content: str) -> Optional[list]:
    """Extract one or more tool-call JSON objects embedded in the content field."""
    if not content:
        return None
    calls = []
    # Find all top-level {...} blocks in the text
    depth = 0
    start = None
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                fragment = content[start : i + 1]
                try:
                    obj = json.loads(fragment)
                    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                        calls.append({"function": {"name": obj["name"], "arguments": obj["arguments"]}})
                except json.JSONDecodeError:
                    pass
                start = None
    return calls if calls else None

# ── Agent loop ────────────────────────────────────────────────────────────────

def run_agent(messages: list) -> str:
    while True:
        msg = _llm_chat(messages, tools=TOOLS)
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or extract_json_tool_call(msg.get("content", ""))

        if not tool_calls:
            return msg.get("content", "")

        for call in tool_calls:
            fn_name = call["function"]["name"]
            raw_args = call["function"].get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    args = json.loads(raw_args)
                except json.JSONDecodeError:
                    args = {}
            else:
                args = raw_args

            handler = TOOL_HANDLERS.get(fn_name)
            if handler:
                print(f"\n  [{fn_name}]", end="")
                tool_result = handler(**args)
            else:
                tool_result = f"Unknown tool: {fn_name}"

            messages.append({"role": "tool", "content": tool_result})

# ── REPL ──────────────────────────────────────────────────────────────────────

def repos_status() -> str:
    if not REPOS_DIR.exists():
        return "No repos cloned yet — run: ai-index"
    repos = [d.name for d in REPOS_DIR.iterdir() if d.is_dir()]
    return f"{len(repos)} repos: {', '.join(repos)}" if repos else "No repos cloned yet — run: ai-index"

def main():
    import argparse
    parser = argparse.ArgumentParser(prog="vibe", add_help=False)
    parser.add_argument("--model", "-m", default=None)
    args, _ = parser.parse_known_args()
    if args.model:
        set_model(args.model)

    project_dir = Path.cwd()
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
                        + context[:6000]
                    )
                })

        messages.append({"role": "user", "content": user_input})
        print("\nThinking", end="", flush=True)

        reply = run_agent(messages)
        print(f"\n\nAI:\n{reply}\n")


if __name__ == "__main__":
    main()

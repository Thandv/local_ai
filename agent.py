#!/usr/bin/env python3
"""
Local AI Coding Agent — quick one-off coding tasks.

Model: set LOCAL_AI_MODEL env var or use --model flag.
  Local (default): qwen2.5-coder:32b via Ollama
  Cloud  (best):   claude-opus-4-7 (set ANTHROPIC_API_KEY)
"""

import os
import sys
import subprocess
import glob as glob_module
from pathlib import Path
from typing import Optional

from local_ai.agents.shared.llm import run_tool_loop, set_model, get_model

MAX_TOOL_OUTPUT = 8000  # chars — truncate large outputs so context stays clean

SYSTEM_PROMPT = """You are an expert AI coding assistant running locally on the user's machine.
You help write, debug, refactor, and build complete software applications.

You have access to these tools:
- read_file(path): Read a file's contents
- write_file(path, content): Write content to a file (creates dirs if needed)
- list_files(pattern): List files matching a glob pattern (e.g. "src/**/*.py")
- run_command(command): Execute a shell command and return its output
- search_code(pattern, directory): Search for a text pattern across files

Rules:
- Always read existing files before editing them so you understand the full context
- When writing code, think step by step before producing the final result
- When asked to build an app, scaffold the full structure — don't leave things half done
- Prefer simple, readable code over clever code
- When running commands, explain what you're about to do first
- If something fails, diagnose and fix it — don't give up
"""

# ── Tools ─────────────────────────────────────────────────────────────────────

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read the contents of a file at the given path.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to read"}
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file. Creates parent directories if needed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "File path to write"},
                    "content": {"type": "string", "description": "Content to write"}
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "List files matching a glob pattern. Use ** for recursive search.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Glob pattern e.g. 'src/**/*.py' or '*.json'"}
                },
                "required": ["pattern"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_command",
            "description": "Run a shell command and return its stdout and stderr.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string", "description": "Shell command to execute"},
                    "working_dir": {"type": "string", "description": "Directory to run in (optional, defaults to current)"}
                },
                "required": ["command"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_code",
            "description": "Search for a text pattern in files. Returns matching lines with file names.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text or regex pattern to search for"},
                    "directory": {"type": "string", "description": "Directory to search in (default: current dir)"}
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
        return f"Error: file not found: {path}"
    try:
        content = p.read_text(encoding="utf-8")
        if len(content) > MAX_TOOL_OUTPUT:
            content = content[:MAX_TOOL_OUTPUT] + f"\n... [truncated, {len(content)} total chars]"
        return content
    except Exception as e:
        return f"Error reading file: {e}"

def write_file(path: str, content: str) -> str:
    p = Path(path).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
    return f"Written {len(content)} chars to {path}"

def list_files(pattern: str) -> str:
    matches = glob_module.glob(pattern, recursive=True)
    if not matches:
        return f"No files found matching: {pattern}"
    return "\n".join(sorted(matches))

def run_command(command: str, working_dir: Optional[str] = None) -> str:
    cwd = Path(working_dir).expanduser() if working_dir else Path.cwd()
    print(f"\n  [running: {command}]")
    try:
        result = subprocess.run(
            command, shell=True, cwd=cwd,
            capture_output=True, text=True, timeout=60
        )
        out = result.stdout + result.stderr
        if len(out) > MAX_TOOL_OUTPUT:
            out = out[:MAX_TOOL_OUTPUT] + "\n... [truncated]"
        return out if out.strip() else "(no output)"
    except subprocess.TimeoutExpired:
        return "Error: command timed out after 60 seconds"
    except Exception as e:
        return f"Error: {e}"

def search_code(pattern: str, directory: str = ".") -> str:
    try:
        result = subprocess.run(
            ["grep", "-rn", "--include=*.py", "--include=*.js", "--include=*.ts",
             "--include=*.jsx", "--include=*.tsx", "--include=*.go", "--include=*.rs",
             "--include=*.java", "--include=*.cpp", "--include=*.c", "--include=*.h",
             "--include=*.html", "--include=*.css", "--include=*.json",
             pattern, directory],
            capture_output=True, text=True, timeout=15
        )
        out = result.stdout
        if not out.strip():
            return f"No matches for '{pattern}' in {directory}"
        if len(out) > MAX_TOOL_OUTPUT:
            out = out[:MAX_TOOL_OUTPUT] + "\n... [truncated]"
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

# ── Agent loop ────────────────────────────────────────────────────────────────

def _show_call(name: str, args: dict) -> None:
    print(f"\n  [{name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items())})]")


def run_agent(messages: list) -> str:
    """Run one REPL turn on the shared, bounded tool loop (history kept in messages)."""
    return run_tool_loop(messages, TOOLS, TOOL_HANDLERS, on_tool_call=_show_call)

# ── REPL ──────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(prog="ai", add_help=False)
    parser.add_argument("--model", "-m", default=None)
    args, _ = parser.parse_known_args()
    if args.model:
        set_model(args.model)

    project_dir = Path.cwd()
    print(f"\n  Local AI Coding Agent")
    print(f"  Model : {get_model()}")
    print(f"  Project: {project_dir}")
    print(f"  Type 'exit' or Ctrl+C to quit, 'clear' to reset conversation\n")

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    # Inject project context on first load
    py_files = glob_module.glob("**/*.py", recursive=True)[:10]
    js_files = glob_module.glob("**/*.js", recursive=True)[:5]
    ts_files = glob_module.glob("**/*.ts", recursive=True)[:5]
    all_files = py_files + js_files + ts_files
    if all_files:
        context = f"Current project directory: {project_dir}\nExisting files: {', '.join(all_files)}"
        messages.append({"role": "system", "content": context})

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
            print("Conversation cleared.\n")
            continue

        messages.append({"role": "user", "content": user_input})
        print("\nAI: ", end="", flush=True)

        reply = run_agent(messages)
        print(reply)
        print()

if __name__ == "__main__":
    main()

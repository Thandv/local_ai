#!/usr/bin/env python3
"""
Local AI Coding Agent — quick one-off coding tasks.

Model: set LOCAL_AI_MODEL env var or use --model flag.
  Local (default): qwen2.5-coder:7b via Ollama
  Cloud:           a model name starting with "claude" (Claude API; needs the
                   [claude] extra and ANTHROPIC_API_KEY)
"""

import os
import sys
import glob as glob_module
from pathlib import Path

from local_ai.agents.shared.llm import (
    LLMUnavailableError, run_tool_loop, set_model, get_model,
)
from local_ai.agents.shared.tools import (
    HANDLERS, LIST_FILES, READ_FILE, RUN_COMMAND, SEARCH_CODE, WRITE_FILE,
)

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

# ── Tools (the shared implementations in agents/shared/tools.py) ──────────────

TOOLS = [READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND, SEARCH_CODE]
TOOL_HANDLERS = {
    name: HANDLERS[name]
    for name in ("read_file", "write_file", "list_files", "run_command", "search_code")
}

# ── Agent loop ────────────────────────────────────────────────────────────────

def _show_call(name: str, args: dict) -> None:
    print(f"\n  [{name}({', '.join(f'{k}={repr(v)[:60]}' for k, v in args.items())})]")


def run_agent(messages: list) -> str:
    """Run one REPL turn on the shared, bounded tool loop (history kept in messages)."""
    return run_tool_loop(messages, TOOLS, TOOL_HANDLERS, on_tool_call=_show_call)

# ── REPL ──────────────────────────────────────────────────────────────────────

def _main() -> None:
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


def main() -> None:
    try:
        _main()
    except LLMUnavailableError as e:
        sys.exit(str(e))  # prints the message to stderr, exit status 1


if __name__ == "__main__":
    main()

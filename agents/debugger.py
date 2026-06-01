"""
Debugger Agent
Tries to start the generated app, captures startup errors, and loops with
the LLM to fix them — up to MAX_ATTEMPTS times.
Runs after the Coder and before the Tester.
"""

from .shared.llm import run_agent_loop
from .shared.tools import READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND, SEARCH_CODE, HANDLERS
from .shared.memory import Project

MAX_ATTEMPTS = 5

SYSTEM = """You are a senior engineer debugging a freshly generated application.

Your job:
1. Read the project files to understand the stack and entry points
2. Run a lightweight startup check:
   - Python: `python -c "import <main_module>"` or `python main.py --help`
   - Node/TS: `node -e "require('./dist/index')"` or `npx tsc --noEmit`
   - FastAPI: `python -c "from app.main import app; print('OK')"`
3. If it crashes, read the error carefully and identify the root cause
4. Apply a targeted fix using write_file (change only what's broken)
5. Re-run the check — repeat until clean

Focus on: import errors, missing deps, syntax errors, missing files, bad env vars.
Use `pip install <pkg>` or `npm install <pkg>` to fix missing packages.
Do NOT try to start a long-running server — use import checks and --help flags only.
"""

FIX_SYSTEM = """You are debugging a startup error in a generated application.

Read the error carefully, find the exact line/file causing it, fix it with write_file,
then re-run the startup check. Be surgical — change only what's broken.
"""

# Signals that indicate a real error vs expected output
ERROR_SIGNALS = [
    "Traceback", "SyntaxError", "ImportError", "ModuleNotFoundError",
    "NameError", "AttributeError", "TypeError", "IndentationError",
    "cannot find module", "Cannot find module", "Error:", "error TS",
    "ENOENT", "Failed to compile",
]
OK_SIGNALS = ["OK", "Usage:", "--help", "usage:", "version", "started", "ready"]


def run(project: Project) -> Project:
    print("  [Debugger] Checking app starts cleanly …")

    out = str(project.output_dir)
    files_ctx = "\n".join(project.files_written) if project.files_written else "(see output dir)"

    task = f"""Project: {project.instruction}
Output directory: {out}

Files written so far:
{files_ctx}

1. List files in {out} to understand the project structure
2. Run a startup check appropriate for the stack (import check, --help, or tsc --noEmit)
3. If it errors, diagnose and fix it
4. Confirm a clean startup with a final check
"""

    handlers = {
        "read_file":   HANDLERS["read_file"],
        "write_file":  HANDLERS["write_file"],
        "list_files":  HANDLERS["list_files"],
        "run_command": HANDLERS["run_command"],
        "search_code": HANDLERS["search_code"],
    }
    tools = [READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND, SEARCH_CODE]

    def on_call(name, args):
        if name == "write_file":
            print(f"    [write] {args.get('path', '')}")
        elif name == "run_command":
            print(f"    [run]   {args.get('command', '')[:80]}")
        else:
            print(f"    [{name}]")

    debug_output = run_agent_loop(
        system=SYSTEM,
        user=task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    has_errors = any(s in debug_output for s in ERROR_SIGNALS)

    for attempt in range(1, MAX_ATTEMPTS + 1):
        if not has_errors:
            break
        print(f"  [Debugger] Error detected — fix attempt {attempt}/{MAX_ATTEMPTS} …")

        fix_task = f"""Startup check output (last attempt):
{debug_output[-3000:]}

Project directory: {out}

Read the relevant source files, identify the root cause, apply a targeted fix,
then re-run the startup check. Report the final result.
"""
        debug_output = run_agent_loop(
            system=FIX_SYSTEM,
            user=fix_task,
            tools=tools,
            handlers=handlers,
            on_tool_call=on_call,
        )
        has_errors = any(s in debug_output for s in ERROR_SIGNALS)
        if not has_errors:
            print("  [Debugger] App starts cleanly!")

    project.debug_log = debug_output
    print("  [Debugger] Done.\n")
    return project

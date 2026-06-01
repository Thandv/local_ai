"""
Coder Agent
Reads PLAN.md and writes every source file for the project.
Uses tools: read_file, write_file, list_files, run_command.
"""

from pathlib import Path
from .shared.llm import run_agent_loop
from .shared.tools import READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND, HANDLERS
from .shared.memory import Project

SYSTEM = """You are an expert software engineer who writes complete, production-quality code.

You will be given a project plan (PLAN.md content is included in your task
prompt — you do NOT need to call read_file to get it). Implement every file
the plan specifies.

CRITICAL: You operate autonomously. There is no human you can ask for files
or clarifications. If you need to inspect a file you've already written,
call read_file yourself. NEVER respond with "please provide", "could you
share", "please paste the contents of", or any other phrasing that asks
the user to give you content. Take action with your tools instead.

Rules:
- Write COMPLETE files — no TODOs, no placeholders, no "add your code here".
- Every file must be immediately runnable / importable.
- Use modern, idiomatic patterns for the chosen stack.
- Include all necessary imports.
- Handle errors gracefully at system boundaries.
- After writing all source files, install dependencies with run_command
  (e.g. `pip install -r requirements.txt` or `npm install`), with
  working_dir set to the output directory.
- Do NOT write test files — a separate Tester agent handles that.

Start writing files immediately using write_file. The plan is below.
"""


def _load_plan(project: Project) -> str:
    """Return the project plan text. Prefer the in-memory copy populated by
    the Architect; fall back to PLAN.md on disk for resumed runs."""
    if (project.plan or "").strip():
        return project.plan
    plan_path = Path(project.output_dir) / "PLAN.md"
    if plan_path.exists():
        try:
            return plan_path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            pass
    return ""


def run(project: Project) -> Project:
    print("  [Coder] Reading plan and writing source files …")

    out = str(project.output_dir)
    plan_text = _load_plan(project)

    # Truncate enormous plans (rare) so we don't blow the model's context.
    if len(plan_text) > 16_000:
        plan_text = plan_text[:16_000] + (
            f"\n\n[plan truncated — {len(plan_text)} chars total; "
            f"call read_file('{out}/PLAN.md') for the remainder]")

    plan_block = (
        f"PLAN.md content (read it carefully — implement every file it describes):\n\n"
        f"---\n{plan_text}\n---\n\n"
        if plan_text else
        f"⚠ PLAN.md was not produced by the Architect. Use the project description "
        f"alone to design and write a complete minimum-viable implementation.\n\n"
    )

    task = f"""Project: {project.instruction}

Output directory: {out}

{plan_block}TASK:
1. Implement every file the plan describes by calling write_file for each one,
   with full paths under {out}/.
2. After all source files are written, install dependencies by calling
   run_command (e.g. command=`pip install -r requirements.txt`,
   working_dir=`{out}`).
3. When you're done, respond with a brief summary of what you built.

Do NOT write test files — a separate Tester agent handles that.
Do NOT ask the user for anything; use your tools.
"""

    # Intercept write_file to track created files
    written = []

    def tracked_write(path: str, content: str) -> str:
        result = HANDLERS["write_file"](path, content)
        written.append(path)
        return result

    handlers = {
        "read_file":   HANDLERS["read_file"],
        "write_file":  tracked_write,
        "list_files":  HANDLERS["list_files"],
        "run_command": HANDLERS["run_command"],
    }
    tools = [READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND]

    def on_call(name, args):
        if name == "write_file":
            print(f"    [write] {args.get('path', '')}")
        elif name == "run_command":
            print(f"    [run]   {args.get('command', '')[:80]}")
        else:
            print(f"    [{name}]")

    result = run_agent_loop(
        system=SYSTEM,
        user=task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    project.files_written.extend(written)
    print("  [Coder] Done.\n")
    return project

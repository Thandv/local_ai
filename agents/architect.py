"""
Architect Agent
Takes the research brief and produces a detailed project plan:
file structure, API design, data models, component breakdown.
Writes PLAN.md into the output directory.
"""

from pathlib import Path
from .shared.llm import run_agent_loop
from .shared.tools import WRITE_FILE, RUN_COMMAND, HANDLERS
from .shared.memory import Project

SYSTEM = """You are a senior software architect.
Given a research brief and an instruction, your job is to produce a detailed,
actionable project plan that a coder can follow file-by-file.

Your plan must include:
1. Complete file tree (every file that needs to be created)
2. For each file: its purpose and what it should contain
3. API endpoints (if applicable) with request/response shapes
4. Data models / database schema
5. Component hierarchy (if frontend)
6. Setup and run instructions

Be exhaustive. The coder will only create what you specify.
Write the plan as a markdown document and save it to PLAN.md.
"""


def run(project: Project) -> Project:
    print("  [Architect] Designing project structure …")

    out = str(project.output_dir)

    task = f"""Instruction: {project.instruction}

Research Brief:
{project.research}

Output directory: {out}

Create a complete PLAN.md at {out}/PLAN.md that covers:
1. Full file tree with every file to create
2. Purpose of each file
3. API routes (method, path, request body, response) if applicable
4. Data models / schema
5. Component breakdown if there is a frontend
6. Step-by-step setup instructions (how to install deps and run)

After writing the plan, respond with a short summary of the architecture.
"""

    handlers = {
        "write_file":  HANDLERS["write_file"],
        "run_command": HANDLERS["run_command"],
    }
    tools = [WRITE_FILE, RUN_COMMAND]

    def on_call(name, args):
        if name == "write_file":
            print(f"    [write] {args.get('path', '')}")
        else:
            print(f"    [{name}]")

    result = run_agent_loop(
        system=SYSTEM,
        user=task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    project.plan = result

    # Save PLAN.md. Some models return the plan as prose without calling
    # write_file — the response text IS the plan, so fall back to writing
    # it ourselves. Without this, the downstream Coder has no PLAN.md to
    # follow and produces stubs.
    plan_path = project.output_dir / "PLAN.md"
    if not plan_path.exists() and (result or "").strip():
        plan_path.write_text(result, encoding="utf-8")
        print(f"    [fallback-write] {plan_path}")
    if plan_path.exists() and str(plan_path) not in project.files_written:
        project.files_written.append(str(plan_path))

    print("  [Architect] Done.\n")
    return project

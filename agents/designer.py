"""
Designer Agent
Translates the user's instruction into a concrete UI/UX design spec:
screen list, component hierarchy, data flow, and theme tokens.
Runs before the Architect so the Coder gets richer UI context.
"""

from .shared.llm import run_agent_loop
from .shared.tools import WRITE_FILE, HANDLERS
from .shared.memory import Project

SYSTEM = """You are a senior product designer and frontend architect.

Given a description of an app to build, produce a concrete design spec:

1. **Screens / Views** — list every screen, its purpose, and key UI elements
2. **Component Tree** — hierarchy of components per screen (name + props/state hints)
3. **Data Flow** — what data each screen needs and where it comes from
4. **Theme Tokens** — color palette (hex values), typography (font + scale), spacing scale
5. **Navigation** — routing structure (React Router paths or Next.js App Router pages)
6. **Forms & Interactions** — key user actions, form fields, validation rules, loading/error states

Be concrete. Name components exactly as they will appear in code (e.g. <TaskCard>, <AuthForm>).
Output structured markdown. Save the spec as DESIGN.md in the output directory.
Do not write actual code — only the design specification.
"""


def run(project: Project) -> Project:
    print("  [Designer] Creating UI/UX design spec …")

    out = str(project.output_dir)

    task = f"""Instruction: {project.instruction}

Output directory: {out}

Produce a DESIGN.md at {out}/DESIGN.md covering:
1. Every screen and its key UI elements
2. Full component hierarchy with prop/state notes
3. Data flow per screen
4. Theme tokens (colors, typography, spacing)
5. Navigation / routing structure
6. Key interactions and form specs

Then give a one-paragraph summary of the overall design approach.
"""

    handlers = {"write_file": HANDLERS["write_file"]}
    tools = [WRITE_FILE]

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

    project.design = result

    design_path = project.output_dir / "DESIGN.md"
    if design_path.exists():
        project.files_written.append(str(design_path))

    print("  [Designer] Done.\n")
    return project

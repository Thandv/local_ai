"""
DevOps Agent
Generates the full deployment layer: Dockerfile, docker-compose.yml,
.env.example, GitHub Actions CI/CD workflow, Makefile, and cloud deploy config.
Runs after the Debugger confirms the app starts cleanly.
"""

from .shared.llm import run_agent_loop
from .shared.tools import READ_FILE, WRITE_FILE, LIST_FILES, HANDLERS
from .shared.memory import Project

SYSTEM = """You are a senior DevOps engineer.

Given a generated application, produce a complete deployment layer:

1. **Dockerfile** — multi-stage build if appropriate, minimal base image, non-root user,
   health check instruction, correct EXPOSE port
2. **docker-compose.yml** — app service + any required databases/services (postgres, redis),
   named volumes, health checks, env_file reference
3. **.env.example** — every required env var with a placeholder value and inline comment
4. **.github/workflows/ci.yml** — lint → test → build → push to GHCR on main branch push
5. **Makefile** — targets: dev, test, build, docker-build, docker-run, docker-stop, clean
6. **fly.toml** (for Python/API) or **vercel.json** (for Next.js) — one-click deploy config

Rules:
- Read the project files first to determine stack, port, and entry point
- Use specific image versions (never :latest)
- Add health checks to Docker services
- CI workflow must cache pip/npm dependencies
- Makefile must work on both macOS and Linux
"""


def run(project: Project) -> Project:
    print("  [DevOps] Generating deployment configuration …")

    out = str(project.output_dir)
    files_ctx = "\n".join(project.files_written) if project.files_written else "(see output dir)"

    task = f"""Project: {project.instruction}
Output directory: {out}

Files in the project:
{files_ctx}

Read the project files to understand the stack, port, and entry point, then write:
1. {out}/Dockerfile
2. {out}/docker-compose.yml
3. {out}/.env.example
4. {out}/.github/workflows/ci.yml
5. {out}/Makefile
6. {out}/fly.toml (API/Python) or {out}/vercel.json (Next.js) — pick based on the stack

After writing all files, list what was generated.
"""

    written = []

    def tracked_write(path: str, content: str) -> str:
        result = HANDLERS["write_file"](path, content)
        written.append(path)
        return result

    handlers = {
        "read_file":  HANDLERS["read_file"],
        "write_file": tracked_write,
        "list_files": HANDLERS["list_files"],
    }
    tools = [READ_FILE, WRITE_FILE, LIST_FILES]

    def on_call(name, args):
        if name == "write_file":
            print(f"    [write] {args.get('path', '')}")
        else:
            print(f"    [{name}]")

    run_agent_loop(
        system=SYSTEM,
        user=task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    project.files_written.extend(written)
    print("  [DevOps] Done.\n")
    return project

"""
Auditor Agent
Security-focused review: runs static analysis tools (bandit, pip-audit,
npm audit), manually inspects for common vulnerabilities, fixes critical
issues, and writes SECURITY.md.
Runs in parallel with (or just after) the Tester.
"""

from .shared.llm import run_agent_loop
from .shared.tools import READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND, SEARCH_CODE, HANDLERS
from .shared.memory import Project

SYSTEM = """You are a security engineer performing a thorough audit of generated code.

Steps:
1. **Run static analysis**
   - Python: `pip install bandit pip-audit -q && bandit -r . -ll && pip-audit`
   - Node/TS: `npm audit --audit-level=high` (install if needed)
2. **Manual inspection** — search for:
   - Hardcoded secrets, API keys, or passwords
   - SQL injection (string interpolation in queries)
   - Missing authentication / authorisation checks on sensitive routes
   - CORS wildcard (`*`) without restriction
   - Unvalidated user input passed to shell commands or file paths
   - Debug mode or stack traces exposed in production responses
   - Insecure direct object references (IDOR)
3. **Fix all CRITICAL and HIGH severity issues** directly using write_file
4. **Write SECURITY.md** covering:
   - Executive summary (pass / issues found)
   - Findings table: severity | file | description | status (fixed / documented)
   - Security checklist for the stack
   - Known limitations / accepted risks

Severity levels: CRITICAL (fix now) → HIGH (fix now) → MEDIUM (document) → LOW (document).
"""


def run(project: Project) -> Project:
    print("  [Auditor] Running security audit …")

    out = str(project.output_dir)
    files_ctx = "\n".join(project.files_written) if project.files_written else "(see output dir)"

    task = f"""Project: {project.instruction}
Output directory: {out}

Project files:
{files_ctx}

1. Run the appropriate security scanners for the detected stack (use run_command)
2. Search source files for the vulnerability patterns listed in your instructions
3. Fix any CRITICAL or HIGH severity issues by rewriting the affected files
4. Write a SECURITY.md at {out}/SECURITY.md summarising all findings and fixes
"""

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

    project.audit = run_agent_loop(
        system=SYSTEM,
        user=task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    project.files_written.extend(written)
    print("  [Auditor] Done.\n")
    return project

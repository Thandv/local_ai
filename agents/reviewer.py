"""
Reviewer Agent
Does a final pass over the written code:
- Security issues
- Missing error handling
- Code quality
- Completeness check against the plan
Writes REVIEW.md and applies any critical fixes.
"""

from pathlib import Path
from .shared.llm import run_agent_loop
from .shared.tools import READ_FILE, WRITE_FILE, LIST_FILES, SEARCH_CODE, HANDLERS
from .shared.memory import Project

SYSTEM = """You are a principal engineer doing a final code review.

Review the project for:
1. **Security** — SQL injection, hardcoded secrets, missing auth, CORS issues
2. **Correctness** — logic errors, unhandled edge cases, missing error handling
3. **Completeness** — does the code match what was planned? Anything missing?
4. **Code quality** — unclear naming, dead code, obvious anti-patterns
5. **Runability** — will it actually start and run without errors?

For each issue found:
- State the file and line/area
- Explain the problem
- Apply the fix directly using write_file

After fixing issues, write a REVIEW.md summarising:
- What was built
- Issues found and fixed
- How to run the project
- Any known limitations
"""


def run(project: Project) -> Project:
    print("  [Reviewer] Reviewing code quality and completeness …")

    out = str(project.output_dir)
    files_ctx = "\n".join(project.files_written) if project.files_written else ""

    task = f"""Project: {project.instruction}
Output directory: {out}

Files written:
{files_ctx}

Test results:
{project.test_results[-2000:] if project.test_results else 'No tests run'}

Review all source files, fix any critical issues, then write REVIEW.md to {out}/REVIEW.md.
"""

    def tracked_write(path: str, content: str) -> str:
        result = HANDLERS["write_file"](path, content)
        if path not in project.files_written:
            project.files_written.append(path)
        return result

    handlers = {
        "read_file":   HANDLERS["read_file"],
        "write_file":  tracked_write,
        "list_files":  HANDLERS["list_files"],
        "search_code": HANDLERS["search_code"],
    }
    tools = [READ_FILE, WRITE_FILE, LIST_FILES, SEARCH_CODE]

    def on_call(name, args):
        if name == "write_file":
            print(f"    [write] {args.get('path', '')}")
        else:
            print(f"    [{name}]")

    project.review = run_agent_loop(
        system=SYSTEM,
        user=task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    # Same fallback pattern as the Architect: if the model returned prose
    # without calling write_file, persist the review text as REVIEW.md so
    # the Packager has something to reference.
    review_path = project.output_dir / "REVIEW.md"
    if not review_path.exists() and (project.review or "").strip():
        review_path.write_text(project.review, encoding="utf-8")
        print(f"    [fallback-write] {review_path}")
    if review_path.exists() and str(review_path) not in project.files_written:
        project.files_written.append(str(review_path))

    print("  [Reviewer] Done.\n")
    return project

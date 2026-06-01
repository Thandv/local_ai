"""
Tester Agent
Reads the project files, writes tests, runs them, and attempts to fix
failures — up to MAX_ATTEMPTS times.
"""

from pathlib import Path
from .shared.llm import run_agent_loop
from .shared.tools import READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND, SEARCH_CODE, HANDLERS
from .shared.memory import Project

MAX_ATTEMPTS = 3

SYSTEM = """You are a senior QA engineer and software tester.

Your job:
1. Read the project's source files to understand what was built
2. Write comprehensive tests covering the main functionality
3. Run the tests
4. If tests fail, diagnose the failures, fix the source code or tests, and re-run
5. Repeat until all tests pass (max {max} attempts)

For Python projects: use pytest. Write tests in a tests/ folder.
For JS/TS projects: use the framework's built-in test runner (Jest/Vitest).
For APIs: test each endpoint with realistic requests.

Be thorough but practical — focus on the critical paths first.
""".format(max=MAX_ATTEMPTS)

FIX_SYSTEM = """You are a senior engineer debugging failing tests.

You have access to the source files and test output.
Your job: fix the failing tests by either:
  a) Correcting bugs in the source code, OR
  b) Correcting incorrect assumptions in the tests

Read the failure output carefully, find the root cause, and apply a targeted fix.
"""


def run(project: Project) -> Project:
    print("  [Tester] Writing and running tests …")

    out = str(project.output_dir)

    # List what's been written so the tester knows what to test
    files_context = "\n".join(project.files_written) if project.files_written else "(see output dir)"

    write_task = f"""Project: {project.instruction}
Output directory: {out}

Files already written:
{files_context}

1. Read the source files to understand the implementation
2. Write tests into {out}/tests/ (or appropriate location for the stack)
3. Run the tests and report results
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

    test_output = run_agent_loop(
        system=SYSTEM,
        user=write_task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    project.files_written.extend(written)
    project.test_results = test_output

    # Check if tests passed — look for obvious failure signals
    failed_signals = ["FAILED", "ERROR", "error", "failed", "AssertionError",
                      "ModuleNotFoundError", "ImportError", "exception"]
    pass_signals   = ["passed", "ok", "PASSED", "success", "✓", "✔"]

    has_failure = any(s in test_output for s in failed_signals)
    has_pass    = any(s in test_output for s in pass_signals)

    if has_failure and not has_pass:
        print(f"  [Tester] Tests failed — attempting fixes (max {MAX_ATTEMPTS}) …")
        for attempt in range(1, MAX_ATTEMPTS + 1):
            print(f"    Attempt {attempt}/{MAX_ATTEMPTS} …")

            fix_task = f"""The tests failed. Here is the output:

{test_output[-3000:]}

Project directory: {out}

Read the relevant source and test files, identify the root cause of each failure,
apply fixes, then re-run the tests. Report the final result.
"""
            test_output = run_agent_loop(
                system=FIX_SYSTEM,
                user=fix_task,
                tools=tools,
                handlers=handlers,
                on_tool_call=on_call,
            )
            project.test_results = test_output

            still_failing = any(s in test_output for s in failed_signals)
            now_passing   = any(s in test_output for s in pass_signals)
            if not still_failing or now_passing:
                print("  [Tester] Tests passing!")
                project.success = True
                break
    else:
        project.success = True

    print("  [Tester] Done.\n")
    return project

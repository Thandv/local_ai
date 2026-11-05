"""
Tester Agent
Reads the project files, writes tests, runs them, and attempts to fix
failures — up to MAX_ATTEMPTS times. Whether the tests passed is decided by
the exit code of the project's test command (run_test_gate), not by the
model's report.
"""

import json
import os
import re
import shlex
import stat
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

from .shared import guard
from .shared.guard import VENV_DIR
from .shared.llm import run_agent_loop
from .shared.tools import READ_FILE, WRITE_FILE, LIST_FILES, RUN_COMMAND, SEARCH_CODE, HANDLERS
from .shared.guard import shown
from .shared.memory import Project

MAX_ATTEMPTS = 3

SYSTEM = """You are a senior QA engineer and software tester.

Your job:
1. Read the project's source files to understand what was built
2. Write comprehensive tests covering the main functionality
3. Run the tests
4. If tests fail, diagnose the failures, fix the source code or tests, and re-run
5. Repeat until all tests pass (max {max} attempts)

For Python projects: use pytest. Write tests in a tests/ folder. Run them with
`python3 -m pytest -q` (or `python3 -m unittest discover -v`); one command per
call, no shell. Packages go into the build's .venv (`python3 -m venv .venv`,
then `pip install pytest`); installs need the person's confirmation.
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


# ── The test gate (LA-4): the real exit code of the project's test command ────

OUTPUT_TAIL = 3000
_SKIP_DIRS = {".git", VENV_DIR, "venv", "node_modules", "__pycache__", ".pytest_cache",
              "dist", "build"}
_NO_TESTS = re.compile(r"\bno tests (ran|found|collected)\b|(?<![\d.])0 tests?\b", re.IGNORECASE)
_UNITTEST_RAN = re.compile(r"^Ran (\d+) tests? in ", re.MULTILINE)
_UNITTEST_SKIPPED = re.compile(r"\bskipped=(\d+)")
# npm test runners' own summary lines (jest, vitest, mocha, ava, node --test
# in tap or spec form). The last one in the output is the runner's.
_NPM_PASSED = re.compile(
    r"^(?:Tests:\s.*?\b(\d+) passed\b.*\b\d+ total"     # jest
    r"|\s*Tests\s.*?\b(\d+) passed\b.*\(\d+\)"         # vitest
    r"|\s*(\d+) passing\b"                                # mocha
    r"|\s*(\d+) tests? passed\b"                          # ava
    r"|(?:#|\u2139) pass (\d+)\s*$)", re.MULTILINE)        # node --test (tap / spec)
_NPM_FAILED = re.compile(
    r"^(?:Tests:\s.*?\b[1-9]\d* failed\b"
    r"|\s*Tests\s.*?\b[1-9]\d* failed\b"
    r"|\s*[1-9]\d* failing\b"
    r"|\s*[1-9]\d* tests? failed\b"
    r"|(?:#|\u2139) fail [1-9]\d*\s*$)", re.MULTILINE)
JUNIT_LIMIT = 5_000_000
_PYTEST_IMPORT = re.compile(r"^\s*(import pytest\b|from pytest\b)", re.MULTILINE)


def _scan_python_tests(out: Path) -> tuple[bool, bool]:
    """(has_tests, needs_pytest) for the build dir, without following symlinks."""
    has = False  # a tests/ dir alone is not Python (it may hold JS tests)
    needs_pytest = False
    for dirpath, dirnames, filenames in os.walk(out):
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS]
        for name in filenames:
            if name in ("conftest.py", "pytest.ini"):
                needs_pytest = True
            # unittest discover's default pattern is test*.py; pytest's adds *_test.py
            if name.endswith(".py") and (name.startswith("test") or name.endswith("_test.py")):
                has = True
                if not needs_pytest:
                    text = _read_plain(Path(dirpath) / name)
                    needs_pytest = bool(text and _PYTEST_IMPORT.search(text))
    return has, needs_pytest


def _read_plain(path: Path, limit: int = 1_000_000) -> Optional[str]:
    """Read a regular file without following a symlink; None for symlinks,
    hard links (another name may be outside the build) and special files."""
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        return None
    with os.fdopen(fd, "rb") as f:
        st = os.fstat(f.fileno())
        if not stat.S_ISREG(st.st_mode) or st.st_nlink > 1:
            return None
        return f.read(limit).decode("utf-8", errors="ignore")


def _npm_test_script(out: Path) -> Optional[str]:
    text = _read_plain(out / "package.json")
    if text is None:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    scripts = data.get("scripts") if isinstance(data, dict) else None
    test = scripts.get("test") if isinstance(scripts, dict) else None
    if not isinstance(test, str) or not test.strip() or "no test specified" in test:
        return None  # npm init's default stub
    return test


def detect_test_command(out: Path) -> dict:
    """Pick the project's test command as a fixed argv.

    Returns {"runner", "argv"} to run, or {"runner", "status", "reason"} when
    nothing can be run (status "skipped" or "fail").
    """
    out = Path(out)
    venv = out / VENV_DIR
    python = venv / "bin" / "python"
    has_py, needs_pytest = _scan_python_tests(out)
    if has_py:
        if not (python.is_file() and (venv / "pyvenv.cfg").is_file()):
            return {"runner": "python", "status": "skipped",
                    "reason": f"no build venv ({VENV_DIR}) to run the Python tests in"}
        if (venv / "bin" / "pytest").is_file():
            return {"runner": "pytest", "argv": [str(python), "-m", "pytest"]}
        if needs_pytest:
            return {"runner": "pytest", "status": "skipped",
                    "reason": f"the tests use pytest, which is not installed in {VENV_DIR}"}
        tests = out / "tests"
        start = (["-s", "tests"] if tests.is_dir() and not tests.is_symlink()
                 and not (tests / "__init__.py").exists() else [])
        return {"runner": "unittest",
                "argv": [str(python), "-m", "unittest", "discover", *start]}
    script = _npm_test_script(out)
    if script:
        return {"runner": "npm", "argv": ["npm", "test"], "script": script}
    return {"runner": None, "status": "fail", "reason": "no tests found"}


def junit_counts(path: str) -> Optional[dict]:
    """{tests, failures, errors, skipped} summed over the JUnit XML report
    pytest wrote, or None when there is no readable report."""
    text = _read_plain(Path(path), JUNIT_LIMIT)
    if not text:
        return None
    try:
        root = ET.fromstring(text)
    except ET.ParseError:
        return None
    suites = [root] if root.tag == "testsuite" else root.findall("testsuite")
    if root.tag not in ("testsuite", "testsuites") or not suites:
        return None
    counts = dict.fromkeys(("tests", "failures", "errors", "skipped"), 0)
    for suite in suites:
        for key in counts:
            value = suite.get(key, "0")
            if not value.isdigit():
                return None
            counts[key] += int(value)
    return counts


def _judge_pytest(report: Optional[dict]) -> tuple[str, str]:
    if report is None:
        return "fail", ("pytest wrote no JUnit report (the run ended before its summary; "
                        "a conftest or plugin may have exited)")
    bad = report["failures"] + report["errors"]
    passed = report["tests"] - report["skipped"] - bad
    if bad:
        return "fail", f"{report['failures']} failed, {report['errors']} error(s)"
    if passed < 1:
        return "fail", f"no test passed ({report['tests']} collected, {report['skipped']} skipped)"
    return "pass", f"{passed} passed"


def _judge_npm(script: str, output: str, plain: bool = True) -> tuple[str, str]:
    if not plain:
        return "unverified", ("npm test is not one plain test-runner command (shell syntax, "
                              "a pre/post hook or another program): its output is its own")
    try:
        words = shlex.split(script)
    except ValueError:
        words = []
    if words[:1] == ["node"] and "--test" not in words:
        return "unverified", ("npm test runs a plain node script, whose output is its own: "
                              "there is no runner count to check")
    if _NPM_FAILED.search(output):
        return "fail", "the test runner reported failures"
    passed = [next(g for g in m.groups() if g) for m in _NPM_PASSED.finditer(output)]
    if not passed:
        return "unverified", "npm test exited 0 but printed no test count the gate can read"
    if int(passed[-1]) < 1:
        return "fail", "the test command ran no tests"
    return "pass", f"{passed[-1]} passed"


def judge(runner: str, returncode: int, output: str, report: Optional[dict] = None,
          script: str = "", plain: bool = True) -> tuple[str, str]:
    """(status, reason) for a test command that ran.

    pytest: exit 0 and, in the JUnit report the gate asked for (`report`,
    from junit_counts), at least one passed test and no failure or error; the
    console output is not read, so a printed summary line counts for nothing.
    unittest: exit 0 with 'Ran N tests', N >= 1, not all skipped.
    npm (`script` is package.json's test script): exit 0 and a test runner's
    own count of N >= 1 passed; no count, a plain `node <file>` script, or
    a script that is not one plain runner command (`plain` False: shell
    syntax, a pre/post hook, another program) is "unverified" (the verdict
    UNVERIFIED), never "pass"."""
    if runner == "unittest":
        ran = _UNITTEST_RAN.findall(output)
        count = int(ran[-1]) if ran else 0
        skipped = _UNITTEST_SKIPPED.findall(output)
        if returncode == 0 and count >= 1 and skipped and int(skipped[-1]) >= count:
            return "fail", "every test was skipped"
        if returncode == 0 and count >= 1:
            return "pass", f"{count} test(s) ran"
        if returncode in (0, 5) and count == 0:
            return "fail", "unittest ran no tests"
        return "fail", f"unittest exited {returncode}"
    if returncode == 5 and runner == "pytest":
        return "fail", "pytest collected no tests"
    if returncode != 0:
        return "fail", f"{runner} exited {returncode}"
    if runner == "pytest":
        return _judge_pytest(report)
    if _NO_TESTS.search(output):
        return "fail", "the test command ran no tests"
    if runner == "npm":
        return _judge_npm(script, output, plain)
    return "fail", f"no way to verify {runner}'s result"


def run_test_gate(project: Project) -> dict:
    """Run the detected test command through the LA-3 guard's run_argv (a
    fixed argv, no shell, its normal allowlist and confirmation) and return
    the project's test_results dict. A pytest run writes its JUnit report
    into a fresh temp dir outside the build, read before the dir is removed."""
    found = detect_test_command(project.output_dir)
    runner = found.get("runner")
    if "argv" not in found:
        return {"status": found["status"], "runner": runner, "returncode": None,
                "reason": found["reason"], "output_tail": ""}
    report = None
    with tempfile.TemporaryDirectory(prefix="local_ai-gate-") as tmp:
        if runner == "pytest":
            path = os.path.join(os.path.realpath(tmp), "junit.xml")
            outcome = guard.run_argv(found["argv"], pytest_report=path)
            report = junit_counts(path)
        else:
            outcome = guard.run_argv(found["argv"])
    plain = runner != "npm" or guard.npm_script_plain(project.output_dir)
    tail = outcome.output[-OUTPUT_TAIL:]
    if outcome.ran:
        status, reason = judge(runner, outcome.returncode, outcome.output, report,
                               found.get("script", ""), plain)
    elif outcome.status == "timeout":
        status, reason = "fail", outcome.reason
    else:  # refused, no TTY, declined, could not start
        status, reason = "skipped", outcome.reason
    return {"status": status, "runner": runner, "returncode": outcome.returncode,
            "reason": reason, "output_tail": tail}


def _report(gate: dict) -> str:
    return (f"status: {gate['status']}, runner: {gate['runner']}, "
            f"exit code: {gate['returncode']}, reason: {gate['reason']}\n"
            f"{gate['output_tail']}")


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
            print(f"    [write] {shown(args.get('path', ''))}")
        elif name == "run_command":
            print(f"    [run]   {shown(str(args.get('command', ''))[:80])}")
        else:
            print(f"    [{shown(name)}]")

    run_agent_loop(
        system=SYSTEM,
        user=write_task,
        tools=tools,
        handlers=handlers,
        on_tool_call=on_call,
    )

    # The gate is the real exit code of the test command, never the model's
    # own account of its run.
    gate = run_test_gate(project)
    attempt = 0
    while gate["status"] == "fail" and attempt < MAX_ATTEMPTS:
        attempt += 1
        print(f"  [Tester] Tests failed ({gate['reason']}) — fix attempt "
              f"{attempt}/{MAX_ATTEMPTS} …")
        fix_task = f"""The project's tests did not pass. Test gate result:

{_report(gate)}

Project directory: {out}

Read the relevant source and test files, identify the root cause of each failure,
apply fixes, then re-run the tests. Report the final result.
"""
        run_agent_loop(
            system=FIX_SYSTEM,
            user=fix_task,
            tools=tools,
            handlers=handlers,
            on_tool_call=on_call,
        )
        gate = run_test_gate(project)

    project.files_written.extend(written)
    project.test_results = gate
    project.success = gate["status"] == "pass"
    print(f"  [Tester] Test gate: {gate['status']}"
          + (f" — {gate['reason']}" if gate["reason"] else "") + "\n")
    return project

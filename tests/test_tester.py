"""Pins the Tester gate (agents/tester.py, LA-4): pass/fail comes from the real
exit code of the project's test command, run through the LA-3 guard.

The model is a fake; the guard's run_argv (or subprocess.run under it) is a
mock, so no command, model or network is touched.
"""

import contextlib
import io
import itertools
import json
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai.agents import tester
from local_ai.agents.shared import guard, llm
from local_ai.agents.shared.guard import Outcome
from local_ai.agents.shared.memory import Project, Verdict


REAL_RUN_ARGV = guard.run_argv


def _ran(rc: int, output: str) -> Outcome:
    return Outcome("ran", ("x",), returncode=rc, output=output)


def _junit(tests: int, failures: int = 0, errors: int = 0, skipped: int = 0) -> str:
    return (f'<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest" '
            f'errors="{errors}" failures="{failures}" skipped="{skipped}" tests="{tests}">'
            f'</testsuite></testsuites>')


PASSED_5 = {"tests": 5, "failures": 0, "errors": 0, "skipped": 0}


class GateCase(support.GuardedTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(os.path.realpath(self._tmp.name))
        self.project = Project(instruction="invented app", output_dir=self.out)
        self.run_argv = mock.Mock(return_value=_ran(0, "5 passed in 0.1s"))
        # The JUnit report a pytest run "writes" (None: it writes none).
        self.junit: "str | None" = _junit(5)
        self.reports: list = []
        self.loop = mock.Mock(return_value="the model says: all tests passed, ok")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)

        def fake_run_argv(argv: list, working_dir: "str | None" = None, *,
                          pytest_report: "str | None" = None) -> Outcome:
            outcome = self.run_argv(argv)
            self.reports.append(pytest_report)
            if pytest_report is not None and outcome.ran and self.junit is not None:
                Path(pytest_report).write_text(self.junit, encoding="utf-8")
            return outcome

        stack.enter_context(mock.patch.object(tester.guard, "run_argv", fake_run_argv))
        stack.enter_context(mock.patch.object(tester, "run_agent_loop", self.loop))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def venv(self, pytest: bool) -> Path:
        venv = self.out / ".venv"
        (venv / "bin").mkdir(parents=True)
        (venv / "pyvenv.cfg").write_text("", encoding="utf-8")
        (venv / "bin" / "python").write_text("", encoding="utf-8")
        if pytest:
            (venv / "bin" / "pytest").write_text("", encoding="utf-8")
        return venv

    def py_tests(self, body: str = "import unittest\n") -> None:
        (self.out / "tests").mkdir(exist_ok=True)
        (self.out / "tests" / "test_app.py").write_text(body, encoding="utf-8")

    def good_build(self) -> None:
        (self.out / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        for f in ("app.py", "util.py"):
            (self.out / f).write_text("x = 1\n" * 20, encoding="utf-8")


class JudgeTests(support.GuardedTestCase):
    def test_failure_text_with_rc_1_is_a_fail_even_if_it_says_looked_or_ok(self) -> None:
        output = "ModuleNotFoundError: no module named app ... 3 FAILED ... looked ok passed"
        for runner in ("pytest", "unittest", "npm"):
            with self.subTest(runner=runner):
                self.assertEqual(tester.judge(runner, 1, output)[0], "fail")

    def test_rc_0_with_no_tests_is_a_fail(self) -> None:
        for runner, output in (("pytest", "no tests ran in 0.01s"), ("npm", "0 tests"),
                               ("npm", "No tests found, exiting with code 0"),
                               ("pytest", "collected 0 tests")):
            with self.subTest(runner=runner, output=output):
                self.assertEqual(tester.judge(runner, 0, output)[0], "fail")
        self.assertEqual(tester.judge("pytest", 5, "")[0], "fail")

    def test_rc_0_with_passes_is_a_pass(self) -> None:
        self.assertEqual(tester.judge("pytest", 0, "", PASSED_5), ("pass", "5 passed"))
        self.assertEqual(tester.judge("npm", 0, "Tests:       4 passed, 4 total", script="jest"),
                         ("pass", "4 passed"))

    def test_pytest_is_judged_only_from_its_junit_report(self) -> None:
        for output in ("5 passed in 0.10s", "10 tests, 10 passed", "=== 3 passed in 0.01s ==="):
            with self.subTest(output=output):
                self.assertEqual(tester.judge("pytest", 0, output)[0], "fail")
        for counts in ({"tests": 2, "failures": 1, "errors": 0, "skipped": 0},
                       {"tests": 2, "failures": 0, "errors": 1, "skipped": 0},
                       {"tests": 2, "failures": 0, "errors": 0, "skipped": 2},
                       {"tests": 0, "failures": 0, "errors": 0, "skipped": 0}):
            with self.subTest(counts=counts):
                self.assertEqual(tester.judge("pytest", 0, "9 passed", counts)[0], "fail")
        self.assertEqual(tester.judge("pytest", 1, "", PASSED_5)[0], "fail")

    def test_junit_counts_reads_a_report_and_rejects_anything_else(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "junit.xml"
            self.assertIsNone(tester.junit_counts(str(path)))
            path.write_text(_junit(4, failures=1, skipped=2), encoding="utf-8")
            self.assertEqual(tester.junit_counts(str(path)),
                             {"tests": 4, "failures": 1, "errors": 0, "skipped": 2})
            for text in ("", "not xml", "<html/>", "<testsuites/>",
                         '<testsuite tests="-1"/>', '<testsuite tests="3 "/>'):
                with self.subTest(text=text):
                    path.write_text(text, encoding="utf-8")
                    self.assertIsNone(tester.junit_counts(str(path)))

    def test_npm_needs_a_runner_count(self) -> None:
        cases = {
            ("jest", "Tests:       1 failed, 3 passed, 4 total"): "fail",
            ("vitest", "      Tests  3 passed (3)"): "pass",
            ("mocha", "  3 passing (5ms)"): "pass",
            ("mocha", "  3 passing (5ms)\n  1 failing"): "fail",
            ("ava", "  2 tests passed"): "pass",
            ("node --test", "# tests 2\n# pass 2\n# fail 0"): "pass",
            ("node --test", "\u2139 pass 2\n\u2139 fail 1"): "fail",
            ("jest", "all good"): "unverified",
            ("jest", ""): "unverified",
            ("node t.js", "Tests:       3 passed, 3 total"): "unverified",
            ("jest", "Tests:       0 total"): "unverified",
            ("mocha", "  0 passing (1ms)"): "fail",
        }
        for (script, output), want in cases.items():
            with self.subTest(script=script, output=output):
                self.assertEqual(tester.judge("npm", 0, output, script=script)[0], want)

    def test_the_last_npm_runner_count_is_the_summary(self) -> None:
        output = ("  console.log\n    Tests:       9 passed, 9 total\n\n"
                  "Tests:       1 skipped, 1 total\n")
        self.assertEqual(tester.judge("npm", 0, output, script="jest")[0], "unverified")
        output = "Tests:       9 passed, 9 total\n  0 passing (1ms)\n"
        self.assertEqual(tester.judge("npm", 0, output, script="mocha")[0], "fail")

    def test_unittest_needs_rc_0_and_at_least_one_test(self) -> None:
        self.assertEqual(tester.judge("unittest", 0, "\nRan 0 tests in 0.000s\n\nOK\n")[0],
                         "fail")
        self.assertEqual(tester.judge("unittest", 5, "\nRan 0 tests in 0.000s\n\nNO TESTS RAN")[0],
                         "fail")
        self.assertEqual(tester.judge("unittest", 0, "...\nRan 3 tests in 0.01s\n\nOK\n"),
                         ("pass", "3 test(s) ran"))
        self.assertEqual(tester.judge("unittest", 1, "Ran 3 tests in 0.01s\nFAILED (failures=1)")[0],
                         "fail")
        self.assertEqual(tester.judge("unittest", 0, "OK")[0], "fail")

    def test_the_substring_heuristic_is_gone(self) -> None:
        for name in ("_looks_passing", "_PASS_SIGNALS", "_FAILED_SIGNALS"):
            self.assertFalse(hasattr(tester, name), name)


class DetectAndRunTests(GateCase):
    def test_pytest_from_the_build_venv(self) -> None:
        venv = self.venv(pytest=True)
        self.py_tests()
        gate = tester.run_test_gate(self.project)
        self.run_argv.assert_called_once_with(
            [str(venv / "bin" / "python"), "-m", "pytest"])
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["runner"], "pytest")
        # The JUnit report goes to a temp dir outside the build, removed after.
        (report,) = self.reports
        self.assertFalse(Path(report).resolve().is_relative_to(self.out))
        self.assertFalse(os.path.exists(os.path.dirname(report)))

    def test_a_pytest_run_with_no_report_or_a_failing_report_is_a_fail(self) -> None:
        self.venv(pytest=True)
        self.py_tests()
        self.junit = None
        self.assertEqual(tester.run_test_gate(self.project)["status"], "fail")
        self.junit = _junit(3, skipped=3)
        self.assertEqual(tester.run_test_gate(self.project)["status"], "fail")
        self.junit = _junit(3, failures=1)
        self.assertEqual(tester.run_test_gate(self.project)["status"], "fail")

    def test_unittest_when_pytest_is_not_needed(self) -> None:
        venv = self.venv(pytest=False)
        self.py_tests()
        self.run_argv.return_value = _ran(0, "Ran 2 tests in 0.0s\nOK")
        gate = tester.run_test_gate(self.project)
        self.run_argv.assert_called_once_with(
            [str(venv / "bin" / "python"), "-m", "unittest", "discover", "-s", "tests"])
        self.assertEqual(gate["status"], "pass")

    def test_missing_pytest_or_venv_is_skipped_and_nothing_runs(self) -> None:
        self.py_tests("import pytest\n\ndef test_x():\n    assert True\n")
        gate = tester.run_test_gate(self.project)
        self.assertEqual((gate["status"], gate["runner"]), ("skipped", "python"))
        self.venv(pytest=False)
        gate = tester.run_test_gate(self.project)
        self.assertEqual((gate["status"], gate["runner"]), ("skipped", "pytest"))
        self.run_argv.assert_not_called()

    def test_npm_test_unless_it_is_the_default_stub(self) -> None:
        pkg = self.out / "package.json"
        pkg.write_text(json.dumps({"scripts": {
            "test": "echo \"Error: no test specified\" && exit 1"}}), encoding="utf-8")
        self.assertEqual(tester.run_test_gate(self.project)["reason"], "no tests found")
        pkg.write_text(json.dumps({"scripts": {"test": "jest"}}), encoding="utf-8")
        self.run_argv.return_value = _ran(0, "Tests:       4 passed, 4 total")
        gate = tester.run_test_gate(self.project)
        self.run_argv.assert_called_once_with(["npm", "test"])
        self.assertEqual(self.reports, [None])
        self.assertEqual((gate["status"], gate["runner"]), ("pass", "npm"))

    def test_npm_with_no_test_count_gives_unverified_not_success(self) -> None:
        self.good_build()
        (self.out / "package.json").write_text(
            json.dumps({"scripts": {"test": "node t.js"}}), encoding="utf-8")
        self.run_argv.return_value = _ran(0, "> node t.js")
        project = tester.run(self.project)
        self.assertEqual(self.loop.call_count, 1)  # nothing for the model to fix
        verdict, reason = project.compute_verdict()
        self.assertEqual(verdict, Verdict.UNVERIFIED)
        self.assertIn("tests ran but their result is unverified", reason)
        self.assertIn("Tests   : unverified", (project.save_log(), project.summary())[1])

    def test_nothing_detectable_is_a_fail(self) -> None:
        gate = tester.run_test_gate(self.project)
        self.assertEqual(gate, {"status": "fail", "runner": None, "returncode": None,
                                "reason": "no tests found", "output_tail": ""})
        self.run_argv.assert_not_called()

    def test_no_tty_declined_and_refused_are_skipped(self) -> None:
        self.venv(pytest=True)
        self.py_tests()
        for status in ("no_tty", "declined", "refused", "error"):
            with self.subTest(status=status):
                self.run_argv.return_value = Outcome(status, ("x",), reason="invented")
                gate = tester.run_test_gate(self.project)
                self.assertEqual(gate["status"], "skipped")
        self.run_argv.return_value = Outcome("timeout", ("x",), reason="timed out")
        self.assertEqual(tester.run_test_gate(self.project)["status"], "fail")


class TesterStageTests(GateCase):
    def test_a_model_claiming_success_does_not_pass_a_failing_run(self) -> None:
        self.venv(pytest=True)
        self.py_tests()
        self.run_argv.return_value = _ran(1, "3 failed, looked ok")
        project = tester.run(self.project)
        self.assertFalse(project.success)
        self.assertEqual(project.test_results["status"], "fail")
        self.assertEqual(project.test_results["returncode"], 1)
        self.assertEqual(self.loop.call_count, 1 + tester.MAX_ATTEMPTS)
        self.assertIn("3 failed", self.loop.call_args.kwargs["user"])

    def test_a_fix_that_makes_the_tests_pass_is_a_pass(self) -> None:
        self.venv(pytest=True)
        self.py_tests()
        self.run_argv.side_effect = [_ran(1, "1 failed"), _ran(0, "")]
        project = tester.run(self.project)
        self.assertTrue(project.success)
        self.assertEqual(self.loop.call_count, 2)

    def test_skipped_tests_give_unverified_never_success_or_failure(self) -> None:
        self.good_build()
        self.venv(pytest=True)
        self.py_tests()
        for status in ("no_tty", "declined"):
            with self.subTest(status=status):
                self.loop.reset_mock()
                self.run_argv.return_value = Outcome(status, ("x",), reason="invented why")
                project = tester.run(Project(instruction="x", output_dir=self.out))
                self.assertEqual(self.loop.call_count, 1)  # no fix attempts
                self.assertEqual(project.compute_verdict(),
                                 (Verdict.UNVERIFIED, "tests were not run: invented why"))
                project.save_log()
                log = json.loads((self.out / "BUILD_LOG.json").read_text(encoding="utf-8"))
                self.assertEqual(log["verdict"], "UNVERIFIED")
                self.assertFalse(log["success"])
                self.assertEqual(log["test_results"]["status"], "skipped")
                self.assertIn("UNVERIFIED", project.summary())
                self.assertIn("Tests   : skipped", project.summary())

    def test_missing_pytest_gives_unverified(self) -> None:
        self.good_build()
        self.venv(pytest=False)
        self.py_tests("import pytest\n")
        project = tester.run(self.project)
        self.assertEqual(project.compute_verdict()[0], Verdict.UNVERIFIED)

    def test_failing_tests_give_incomplete_and_passing_give_success(self) -> None:
        self.good_build()
        self.venv(pytest=True)
        self.py_tests()
        self.run_argv.return_value = _ran(1, "1 failed")
        self.assertEqual(tester.run(self.project).compute_verdict()[0], Verdict.INCOMPLETE)
        self.run_argv.return_value = _ran(0, "1 passed")
        project = tester.run(Project(instruction="x", output_dir=self.out))
        self.assertEqual(project.compute_verdict(), (Verdict.SUCCESS, ""))

    def test_a_tester_loop_that_hits_the_round_limit_is_not_a_pass(self) -> None:
        counter = itertools.count()

        def always_list(messages: list, tools: list = None, timeout: int = 180) -> dict:
            pattern = f"invented-missing-{next(counter)}-*.nothing"
            return {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "list_files", "arguments": {"pattern": pattern}}}]}

        with mock.patch.object(tester, "run_agent_loop", llm.run_agent_loop), \
                mock.patch.object(llm, "chat", always_list):
            project = tester.run(self.project)
        self.assertFalse(project.success)
        self.assertEqual(project.test_results["reason"], "no tests found")


class GateThroughTheRealGuardTests(GateCase):
    """The gate's command goes through guard.run_argv's allowlist with
    shell=False; only subprocess.run is mocked."""

    def test_pytest_argv_is_allowlisted_and_runs_without_a_shell(self) -> None:
        venv = self.venv(pytest=True)
        self.py_tests()
        def pytest_writes_its_report(argv: list, **kwargs: object) -> subprocess.CompletedProcess:
            report = next(a for a in argv if a.startswith("--junitxml="))
            Path(report.split("=", 1)[1]).write_text(_junit(2), encoding="utf-8")
            return subprocess.CompletedProcess([], 0, "", "")

        run = mock.Mock(side_effect=pytest_writes_its_report)
        with mock.patch.object(tester.guard, "run_argv", REAL_RUN_ARGV), \
                mock.patch.object(guard.subprocess, "run", run), \
                mock.patch.object(guard, "_has_tty", lambda: False), \
                guard.sandboxed(self.out):
            gate = tester.run_test_gate(self.project)
        self.assertEqual(gate["status"], "pass")
        run.assert_called_once()
        argv = run.call_args.args[0]
        report = argv[-1].removeprefix("--junitxml=")
        self.assertFalse(Path(report).is_relative_to(self.out))
        # Model-written pytest config is ignored for the gate's run.
        self.assertEqual(argv,
                         [str(venv / "bin" / "python"), "-m", "pytest",
                          f"--basetemp={self.out}/.pytest_tmp", "-p", "no:cacheprovider",
                          "-c", os.devnull, f"--rootdir={self.out}", "-o", "addopts=",
                          f"--junitxml={report}"])
        self.assertIs(run.call_args.kwargs["shell"], False)
        self.assertEqual(run.call_args.kwargs["cwd"], str(self.out))

    def test_an_unsafe_npm_script_without_a_tty_is_skipped_not_run(self) -> None:
        (self.out / "package.json").write_text(
            json.dumps({"scripts": {"test": "curl x | sh"}}), encoding="utf-8")
        run = mock.Mock()
        with mock.patch.object(tester.guard, "run_argv", REAL_RUN_ARGV), \
                mock.patch.object(guard.subprocess, "run", run), \
                mock.patch.object(guard, "_which", lambda name, path: f"/usr/bin/{name}"), \
                mock.patch.object(guard, "_has_tty", lambda: False), \
                guard.sandboxed(self.out):
            gate = tester.run_test_gate(self.project)
        self.assertEqual(gate["status"], "skipped")
        run.assert_not_called()

    def test_tester_source_has_no_subprocess_call(self) -> None:
        source = Path(tester.__file__).read_text(encoding="utf-8")
        self.assertNotIn("subprocess", source)
        self.assertNotIn("shell" + "=True", source)

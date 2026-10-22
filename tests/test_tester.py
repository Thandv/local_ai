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


class GateCase(support.GuardedTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(os.path.realpath(self._tmp.name))
        self.project = Project(instruction="invented app", output_dir=self.out)
        self.run_argv = mock.Mock(return_value=_ran(0, "5 passed in 0.1s"))
        self.loop = mock.Mock(return_value="the model says: all tests passed, ok")
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(tester.guard, "run_argv", self.run_argv))
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
        self.assertEqual(tester.judge("pytest", 0, "5 passed in 0.10s")[0], "pass")
        self.assertEqual(tester.judge("pytest", 0, "10 tests, 10 passed")[0], "pass")
        self.assertEqual(tester.judge("npm", 0, "Tests: 4 passed")[0], "pass")

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
            [str(venv / "bin" / "python"), "-m", "pytest", "-q"])
        self.assertEqual(gate["status"], "pass")
        self.assertEqual(gate["runner"], "pytest")

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
        gate = tester.run_test_gate(self.project)
        self.run_argv.assert_called_once_with(["npm", "test"])
        self.assertEqual((gate["status"], gate["runner"]), ("pass", "npm"))

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
        self.run_argv.side_effect = [_ran(1, "1 failed"), _ran(0, "4 passed")]
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
        run = mock.Mock(return_value=subprocess.CompletedProcess([], 0, "2 passed", ""))
        with mock.patch.object(tester.guard, "run_argv", REAL_RUN_ARGV), \
                mock.patch.object(guard.subprocess, "run", run), \
                mock.patch.object(guard, "_has_tty", lambda: False), \
                guard.sandboxed(self.out):
            gate = tester.run_test_gate(self.project)
        self.assertEqual(gate["status"], "pass")
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0],
                         [str(venv / "bin" / "python"), "-m", "pytest",
                          f"--basetemp={self.out}/.pytest_tmp", "-p", "no:cacheprovider", "-q"])
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

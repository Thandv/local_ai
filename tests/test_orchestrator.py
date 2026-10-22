"""Pins orchestrator checkpoint save/load and resume-from-log.

Every path comes from a tempfile.TemporaryDirectory passed through the
functions' own parameters (output_dir / resume_dir); no global is patched.
All agent stages are replaced with fakes, so no model is ever called.
"""

import contextlib
import io
import json
import tempfile
from pathlib import Path
from typing import Callable
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai import build_cli, orchestrator
from local_ai.agents.shared.llm import LLMUnavailableError
from local_ai.agents.shared.memory import Project, Verdict

STAGES = ["designer", "researcher", "architect", "migrator", "coder", "debugger",
          "tester", "auditor", "devops", "reviewer", "packager"]


class CheckpointTests(support.GuardedTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def test_load_missing_log_returns_empty(self) -> None:
        self.assertEqual(orchestrator._load_checkpoint(self.out), {})

    def test_load_corrupt_log_returns_empty(self) -> None:
        (self.out / "BUILD_LOG.json").write_text("{not json", encoding="utf-8")
        self.assertEqual(orchestrator._load_checkpoint(self.out), {})

    def test_save_appends_stage_once_and_round_trips(self) -> None:
        project = Project(instruction="invented ledger tool", output_dir=self.out)
        project.files_written = ["a.py"]
        orchestrator._save_checkpoint(project, "designer")
        orchestrator._save_checkpoint(project, "researcher")
        orchestrator._save_checkpoint(project, "designer")
        log = orchestrator._load_checkpoint(self.out)
        self.assertEqual(log["completed_stages"], ["designer", "researcher"])
        self.assertEqual(log["instruction"], "invented ledger tool")
        self.assertEqual(log["files"], ["a.py"])
        self.assertEqual(log["started_at"], project.started_at)


class ResumeTests(support.GuardedTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.called: list[str] = []
        self.failing: set[str] = set()

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def _fake(self, name: str) -> Callable[[Project], Project]:
        def stage(project: Project) -> Project:
            self.called.append(name)
            if name in self.failing:
                raise RuntimeError(f"invented {name} failure")
            return project
        return stage

    def _run_resume(self, instruction: str = "") -> Project:
        with contextlib.ExitStack() as stack:
            for name in STAGES:
                module = getattr(orchestrator, name)
                stack.enter_context(mock.patch.object(module, "run", self._fake(name)))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return orchestrator.run(instruction=instruction, resume_dir=self.out)

    def test_resume_skips_completed_stages_and_restores_state(self) -> None:
        (self.out / "BUILD_LOG.json").write_text(json.dumps({
            "instruction": "invented recipe planner",
            "files": ["main.py", "util.py"],
            "completed_stages": ["designer", "researcher", "architect"],
        }), encoding="utf-8")

        project = self._run_resume()

        self.assertEqual(self.called, STAGES[3:])
        self.assertEqual(project.instruction, "invented recipe planner")
        self.assertEqual(project.files_written, ["main.py", "util.py"])
        self.assertEqual(project.output_dir, self.out)
        log = json.loads((self.out / "BUILD_LOG.json").read_text(encoding="utf-8"))
        self.assertEqual(log["instruction"], "invented recipe planner")

    def test_resume_checkpoints_each_stage_it_runs(self) -> None:
        (self.out / "BUILD_LOG.json").write_text(json.dumps({
            "instruction": "invented recipe planner",
            "completed_stages": STAGES[:9],
        }), encoding="utf-8")
        logged: list[list[str]] = []
        real_save = orchestrator._save_checkpoint

        def spy(project: Project, stage: str) -> None:
            real_save(project, stage)
            logged.append(orchestrator._load_checkpoint(self.out)["completed_stages"])

        with mock.patch.object(orchestrator, "_save_checkpoint", spy):
            self._run_resume()

        self.assertEqual(self.called, ["reviewer", "packager"])
        self.assertEqual(logged, [STAGES[:10], STAGES])

    def test_failed_stage_is_contained_and_not_checkpointed(self) -> None:
        (self.out / "BUILD_LOG.json").write_text(json.dumps({
            "instruction": "invented recipe planner",
            "completed_stages": STAGES[:8],
        }), encoding="utf-8")
        self.failing = {"devops"}
        saved: list[str] = []
        real_save = orchestrator._save_checkpoint

        def spy(project: Project, stage: str) -> None:
            saved.append(stage)
            real_save(project, stage)

        with mock.patch.object(orchestrator, "_save_checkpoint", spy), \
                contextlib.redirect_stderr(io.StringIO()):
            self._run_resume()

        self.assertEqual(self.called, ["devops", "reviewer", "packager"])
        self.assertEqual(saved, ["reviewer", "packager"])

    def test_explicit_instruction_overrides_logged_one(self) -> None:
        (self.out / "BUILD_LOG.json").write_text(json.dumps({
            "instruction": "old invented idea", "completed_stages": STAGES,
        }), encoding="utf-8")
        project = self._run_resume("new invented idea")
        self.assertEqual(self.called, [])
        self.assertEqual(project.instruction, "new invented idea")


class PipelineCase(support.GuardedTestCase):
    """Runs orchestrator.run with every stage replaced by a recording fake."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.called: list[str] = []
        self.failing: dict[str, BaseException] = {}

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def _fake(self, name: str) -> Callable[[Project], Project]:
        def stage(project: Project) -> Project:
            self.called.append(name)
            if name in self.failing:
                raise self.failing[name]
            return project
        return stage

    def _run(self, **kwargs: object) -> tuple[Project, str]:
        out = io.StringIO()
        with contextlib.ExitStack() as stack:
            for name in STAGES:
                module = getattr(orchestrator, name)
                stack.enter_context(mock.patch.object(module, "run", self._fake(name)))
            stack.enter_context(contextlib.redirect_stdout(out))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            project = orchestrator.run(**kwargs)
        return project, out.getvalue()

    def _build(self) -> tuple[Project, str]:
        return self._run(instruction="invented habit tracker", output_dir=self.out)

    def _log(self) -> dict:
        return json.loads((self.out / "BUILD_LOG.json").read_text(encoding="utf-8"))


class StageFailureTests(PipelineCase):
    """A required stage (architect, coder, tester) or an unavailable model
    stops the pipeline with verdict FAILED; an optional stage only warns."""

    def test_required_stage_raising_stops_with_failed(self) -> None:
        for stage in ("architect", "coder", "tester"):
            with self.subTest(stage=stage):
                self.called = []
                self.failing = {stage: RuntimeError(f"invented {stage} crash")}
                project, out = self._build()
                self.assertEqual(self.called, STAGES[:STAGES.index(stage) + 1])
                self.assertEqual(project.verdict, Verdict.FAILED)
                self.assertFalse(project.success)
                self.assertIn("FAILED", out)
                log = self._log()
                self.assertEqual(log["verdict"], "FAILED")
                self.assertFalse(log["success"])
                self.assertEqual(log["stage_errors"][-1]["stage"], stage)
                self.assertIn(f"invented {stage} crash", log["stage_errors"][-1]["error"])
                self.assertTrue(log["stage_errors"][-1]["fatal"])
                self.assertNotIn(stage, log["completed_stages"])

    def test_optional_stage_raising_warns_and_continues(self) -> None:
        self.failing = {"reviewer": RuntimeError("invented reviewer crash")}
        project, out = self._build()
        self.assertEqual(self.called, STAGES)
        self.assertIn("WARNING", out)
        self.assertIn("invented reviewer crash", out)
        self.assertNotEqual(project.verdict, Verdict.FAILED)
        log = self._log()
        self.assertEqual(log["verdict"], "INCOMPLETE")  # fakes write no PLAN.md
        self.assertEqual(len(log["stage_errors"]), 1)
        self.assertEqual(log["stage_errors"][0]["stage"], "reviewer")
        self.assertFalse(log["stage_errors"][0]["fatal"])

    def test_unavailable_model_in_optional_stage_stops_with_failed(self) -> None:
        self.failing = {"designer": LLMUnavailableError("invented: backend is down")}
        project, _ = self._build()
        self.assertEqual(self.called, ["designer"])
        self.assertEqual(project.verdict, Verdict.FAILED)
        log = self._log()
        self.assertEqual(log["verdict"], "FAILED")
        self.assertIn("backend is down", log["stage_errors"][0]["error"])

    def test_keyboard_interrupt_and_system_exit_propagate(self) -> None:
        for exc in (KeyboardInterrupt(), SystemExit(3)):
            with self.subTest(exc=type(exc).__name__):
                self.called = []
                self.failing = {"coder": exc}
                with self.assertRaises(type(exc)):
                    self._build()
                self.assertNotEqual(self._log().get("verdict"), "FAILED")

    def test_resuming_a_finished_build_runs_no_stage(self) -> None:
        self._build()
        self.assertEqual(self.called, STAGES)
        self.assertEqual(self._log()["completed_stages"], STAGES)
        self.called = []
        self._run(instruction="", resume_dir=self.out)
        self.assertEqual(self.called, [])
        self.assertEqual(self._log()["completed_stages"], STAGES)

    def test_resuming_a_failed_build_retries_the_failed_stage(self) -> None:
        self.failing = {"coder": RuntimeError("invented coder crash")}
        self._build()
        self.failing = {}
        self.called = []
        project, _ = self._run(instruction="", resume_dir=self.out)
        self.assertEqual(self.called, STAGES[STAGES.index("coder"):])
        self.assertNotEqual(project.verdict, Verdict.FAILED)


class ReviewFollowUpPipelineTests(PipelineCase):
    """Verdict on resume, the checkpoint/verdict contract and build-app's exit status."""

    def _fake(self, name: str) -> Callable[[Project], Project]:
        inner = super()._fake(name)

        def stage(project: Project) -> Project:
            project = inner(project)
            if name == "architect":
                (project.output_dir / "PLAN.md").write_text("# plan\n", encoding="utf-8")
            if name == "coder":
                for f in ("app.py", "util.py"):
                    (project.output_dir / f).write_text("x = 1\n" * 20, encoding="utf-8")
            if name == "tester":
                project.success = True
            return project
        return stage

    def test_resuming_a_finished_success_build_keeps_success(self) -> None:
        project, _ = self._build()
        self.assertEqual(project.verdict, Verdict.SUCCESS)
        self.called = []
        project, _ = self._run(instruction="", resume_dir=self.out)
        self.assertEqual(self.called, [])
        self.assertEqual(project.verdict, Verdict.SUCCESS)
        self.assertEqual(self._log()["verdict"], "SUCCESS")

    def test_checkpoint_drops_a_stale_verdict(self) -> None:
        self.failing = {"coder": RuntimeError("invented coder crash")}
        self._build()
        self.assertEqual(self._log()["verdict"], "FAILED")
        project = Project(instruction="x", output_dir=self.out)
        orchestrator._save_checkpoint(project, "coder")
        self.assertNotIn("verdict", self._log())

    def test_build_app_exits_nonzero_when_the_model_is_unavailable(self) -> None:
        self.failing = {name: LLMUnavailableError("invented: backend is down")
                        for name in STAGES}
        argv = ["build-app", "--out", str(self.out), "invented habit tracker"]
        with contextlib.ExitStack() as stack:
            for name in STAGES:
                stack.enter_context(mock.patch.object(
                    getattr(orchestrator, name), "run", self._fake(name)))
            stack.enter_context(mock.patch("sys.argv", argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
            with self.assertRaises(SystemExit) as cm:
                build_cli.main()
        self.assertIsInstance(cm.exception.code, str)  # str code -> exit status 1
        self.assertIn("FAILED", cm.exception.code)
        self.assertIn("backend is down", cm.exception.code)
        self.assertEqual(self.called, ["designer"])

    def test_build_app_returns_normally_when_not_failed(self) -> None:
        argv = ["build-app", "--out", str(self.out), "invented habit tracker"]
        with contextlib.ExitStack() as stack:
            for name in STAGES:
                stack.enter_context(mock.patch.object(
                    getattr(orchestrator, name), "run", self._fake(name)))
            stack.enter_context(mock.patch("sys.argv", argv))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            build_cli.main()
        self.assertEqual(self._log()["verdict"], "SUCCESS")

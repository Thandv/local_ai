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
from local_ai import orchestrator
from local_ai.agents.shared.memory import Project

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

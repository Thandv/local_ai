"""Pins Project.compute_final_success (agents/shared/memory.py)."""

import json
import tempfile
from pathlib import Path

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai.agents.shared.memory import MIN_SOURCE_FILES, MIN_SOURCE_LOC, Project, Verdict


def _write_lines(path: Path, n: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"line_{i} = {i}\n" for i in range(n)), encoding="utf-8")


class ComputeFinalSuccessTests(support.GuardedTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.project = Project(instruction="invented widget app", output_dir=self.out)
        self.project.success = True

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def _good_build(self) -> None:
        (self.out / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        _write_lines(self.out / "app.py", 20)
        _write_lines(self.out / "lib" / "util.py", 20)

    def test_passes_with_plan_two_sources_and_enough_loc(self) -> None:
        self._good_build()
        self.assertEqual(self.project.compute_final_success(), (True, ""))

    def test_fails_without_plan(self) -> None:
        _write_lines(self.out / "app.py", 40)
        _write_lines(self.out / "b.py", 40)
        passed, reason = self.project.compute_final_success()
        self.assertFalse(passed)
        self.assertEqual(
            reason, "PLAN.md was not produced — Architect stage failed or wrote no plan")

    def test_markdown_and_tooling_dirs_do_not_count(self) -> None:
        (self.out / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        _write_lines(self.out / "app.py", 40)
        _write_lines(self.out / "SECURITY.md", 40)
        _write_lines(self.out / "node_modules" / "dep.js", 40)
        _write_lines(self.out / ".venv" / "site.py", 40)
        passed, reason = self.project.compute_final_success()
        self.assertFalse(passed)
        self.assertEqual(reason, "only 1 source file(s) — need at least 2")

    def test_thresholds_are_two_files_and_thirty_lines(self) -> None:
        self.assertEqual((MIN_SOURCE_FILES, MIN_SOURCE_LOC), (2, 30))

    def test_every_tooling_dir_is_skipped_even_when_nested(self) -> None:
        (self.out / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        _write_lines(self.out / "app.py", 40)
        for d in (".git", "venv", ".venv", "node_modules", "__pycache__",
                  ".pytest_cache", "dist", "build"):
            _write_lines(self.out / "src" / d / "x.py", 40)
        self.assertEqual(
            self.project.compute_final_success(),
            (False, "only 1 source file(s) — need at least 2"),
        )

    def test_extension_match_is_case_insensitive(self) -> None:
        (self.out / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        _write_lines(self.out / "APP.PY", 20)
        _write_lines(self.out / "Index.HTML", 20)
        self.assertEqual(self.project.compute_final_success(), (True, ""))

    def test_fails_below_min_loc(self) -> None:
        (self.out / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        half = (MIN_SOURCE_LOC - 1) // 2
        _write_lines(self.out / "a.py", half)
        _write_lines(self.out / "b.py", MIN_SOURCE_LOC - 1 - half)
        passed, reason = self.project.compute_final_success()
        self.assertFalse(passed)
        self.assertEqual(
            reason,
            f"only {MIN_SOURCE_LOC - 1} lines across source files — "
            f"need at least {MIN_SOURCE_LOC}",
        )

    def test_exactly_min_loc_passes(self) -> None:
        (self.out / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        _write_lines(self.out / "a.py", MIN_SOURCE_LOC - 1)
        _write_lines(self.out / "b.py", 1)
        self.assertEqual(self.project.compute_final_success(), (True, ""))

    def test_fails_when_test_gate_failed(self) -> None:
        self._good_build()
        self.project.success = False
        self.assertEqual(
            self.project.compute_final_success(), (False, "test gate did not pass")
        )


class VerdictTests(support.GuardedTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.out = Path(self._tmp.name)
        self.project = Project(instruction="invented widget app", output_dir=self.out)
        self.project.success = True
        (self.out / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        _write_lines(self.out / "app.py", 20)
        _write_lines(self.out / "util.py", 20)

    def tearDown(self) -> None:
        self._tmp.cleanup()
        super().tearDown()

    def test_all_gates_pass_is_success(self) -> None:
        self.assertEqual(self.project.compute_verdict(), (Verdict.SUCCESS, ""))

    def test_failed_gate_is_incomplete(self) -> None:
        self.project.success = False
        self.assertEqual(self.project.compute_verdict(),
                         (Verdict.INCOMPLETE, "test gate did not pass"))

    def test_fatal_stage_error_is_failed_even_when_gates_pass(self) -> None:
        self.project.record_stage_error("coder", RuntimeError("invented"), fatal=True)
        verdict, reason = self.project.compute_verdict()
        self.assertEqual(verdict, Verdict.FAILED)
        self.assertEqual(reason, "coder stage raised RuntimeError: invented")

    def test_non_fatal_stage_error_leaves_the_gates_in_charge(self) -> None:
        self.project.record_stage_error("reviewer", RuntimeError("invented"), fatal=False)
        self.assertEqual(self.project.compute_verdict(), (Verdict.SUCCESS, ""))

    def test_save_log_and_summary_emit_the_verdict(self) -> None:
        self.project.record_stage_error("tester", ValueError("invented"), fatal=True)
        self.project.save_log()
        log = json.loads((self.out / "BUILD_LOG.json").read_text(encoding="utf-8"))
        self.assertEqual(log["verdict"], "FAILED")
        self.assertFalse(log["success"])
        self.assertEqual(log["success_reason"], "tester stage raised ValueError: invented")
        self.assertEqual(log["stage_errors"], [
            {"stage": "tester", "error": "ValueError: invented", "fatal": True}])
        self.assertIn("✗ FAILED", self.project.summary())
        self.assertIn("tester stage raised ValueError: invented", self.project.summary())

    def test_save_log_keeps_checkpoint_keys(self) -> None:
        (self.out / "BUILD_LOG.json").write_text(
            json.dumps({"completed_stages": ["designer", "coder"]}), encoding="utf-8")
        self.project.save_log()
        log = json.loads((self.out / "BUILD_LOG.json").read_text(encoding="utf-8"))
        self.assertEqual(log["completed_stages"], ["designer", "coder"])
        self.assertEqual(log["verdict"], "SUCCESS")
        self.assertIn("✓ SUCCESS", self.project.summary())

"""Pins Project.compute_final_success (agents/shared/memory.py)."""

import tempfile
from pathlib import Path

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai.agents.shared.memory import MIN_SOURCE_LOC, Project


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
        self.assertIn("PLAN.md", reason)

    def test_markdown_and_tooling_dirs_do_not_count(self) -> None:
        (self.out / "PLAN.md").write_text("# plan\n", encoding="utf-8")
        _write_lines(self.out / "app.py", 40)
        _write_lines(self.out / "SECURITY.md", 40)
        _write_lines(self.out / "node_modules" / "dep.js", 40)
        _write_lines(self.out / ".venv" / "site.py", 40)
        passed, reason = self.project.compute_final_success()
        self.assertFalse(passed)
        self.assertEqual(reason, "only 1 source file(s) — need at least 2")

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

"""Pins the Tester stage's pass/fail reading (agents/tester.py).

The model is a fake that never stops calling a harmless tool, so every
tester loop hits its round limit; nothing runs a command or touches a model.
"""

import contextlib
import io
import itertools
import tempfile
from pathlib import Path
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai.agents import tester
from local_ai.agents.shared import llm
from local_ai.agents.shared.memory import Project


class TesterRoundLimitTests(support.GuardedTestCase):
    def test_a_tester_loop_that_hits_the_round_limit_is_not_a_pass(self) -> None:
        counter = itertools.count()

        def always_list(messages: list, tools: list = None, timeout: int = 180) -> dict:
            pattern = f"invented-missing-{next(counter)}-*.nothing"
            return {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "list_files", "arguments": {"pattern": pattern}}}]}

        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(llm, "chat", always_list), \
                contextlib.redirect_stdout(io.StringIO()):
            project = Project(instruction="invented app", output_dir=Path(tmp))
            project = tester.run(project)
        self.assertFalse(project.success)
        self.assertTrue(llm.stopped_at_round_limit(project.test_results))

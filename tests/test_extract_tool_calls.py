"""Pins llm._extract_tool_calls (agents/shared/llm.py)."""

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai.agents.shared.llm import _extract_tool_calls


class ExtractToolCallsTests(support.GuardedTestCase):
    def test_well_formed_call(self) -> None:
        content = 'Sure. {"name": "read_file", "arguments": {"path": "a.py"}}'
        self.assertEqual(
            _extract_tool_calls(content),
            [{"function": {"name": "read_file", "arguments": {"path": "a.py"}}}],
        )

    def test_multiple_fragments_in_order(self) -> None:
        content = (
            '{"name": "list_files", "arguments": {"pattern": "*.py"}}\n'
            'then {"name": "read_file", "arguments": {"path": "b.py"}}'
        )
        self.assertEqual(
            _extract_tool_calls(content),
            [
                {"function": {"name": "list_files", "arguments": {"pattern": "*.py"}}},
                {"function": {"name": "read_file", "arguments": {"path": "b.py"}}},
            ],
        )

    def test_malformed_json_is_skipped_and_later_call_kept(self) -> None:
        content = (
            '{"name": "read_file", "arguments": {path: oops}} '
            '{"name": "write_file", "arguments": {"path": "c.py", "content": "x"}}'
        )
        self.assertEqual(
            _extract_tool_calls(content),
            [{"function": {"name": "write_file",
                           "arguments": {"path": "c.py", "content": "x"}}}],
        )

    def test_missing_name_or_arguments_is_ignored(self) -> None:
        self.assertIsNone(_extract_tool_calls('{"name": "read_file"}'))
        self.assertIsNone(_extract_tool_calls('{"arguments": {"path": "a"}}'))

    def test_empty_and_plain_text(self) -> None:
        self.assertIsNone(_extract_tool_calls(""))
        self.assertIsNone(_extract_tool_calls("no tool call here"))

    def test_nested_braces_parse_as_one_object(self) -> None:
        content = '{"name": "run_command", "arguments": {"command": "echo {x}"}}'
        self.assertEqual(
            _extract_tool_calls(content),
            [{"function": {"name": "run_command",
                           "arguments": {"command": "echo {x}"}}}],
        )

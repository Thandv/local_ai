"""Pins tool-call dispatch in run_agent_loop (agents/shared/llm.py).

The model is replaced by a scripted fake `chat`; handlers are plain
in-memory functions, so nothing touches the disk, a model or a subprocess.
"""

import contextlib
import copy
import io
from typing import Any
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai.agents.shared import llm


def _tool_call(name: str, arguments: Any) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


def _text(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": None}


class ScriptedChat:
    """Fake llm.chat: returns scripted replies and records what it was sent."""

    def __init__(self, replies: list[dict]) -> None:
        self.replies = list(replies)
        self.sent: list[list[dict]] = []

    def __call__(self, messages: list, tools: list = None, timeout: int = 180) -> dict:
        self.sent.append(copy.deepcopy(messages))
        return self.replies.pop(0)


class DispatchToolCallTests(support.GuardedTestCase):
    def setUp(self) -> None:
        self.written: list[tuple[str, str]] = []

        def write_file(path: str, content: str) -> str:
            self.written.append((path, content))
            return f"Written: {path}"

        self.handlers = {"write_file": write_file}

    def test_valid_json_string_args_call_the_handler(self) -> None:
        result = llm.dispatch_tool_call(
            "write_file", '{"path": "a.py", "content": "x"}', self.handlers)
        self.assertEqual(result, "Written: a.py")
        self.assertEqual(self.written, [("a.py", "x")])

    def test_dict_args_call_the_handler(self) -> None:
        result = llm.dispatch_tool_call(
            "write_file", {"path": "b.py", "content": "y"}, self.handlers)
        self.assertEqual(result, "Written: b.py")

    def test_invalid_json_is_an_error_result(self) -> None:
        result = llm.dispatch_tool_call("write_file", "not json", self.handlers)
        self.assertTrue(result.startswith("Error: invalid arguments for write_file:"),
                        result)
        self.assertEqual(self.written, [])

    def test_non_object_json_is_an_error_result(self) -> None:
        for raw in ("[1]", [1], "3", None):
            with self.subTest(raw=raw):
                result = llm.dispatch_tool_call("write_file", raw, self.handlers)
                self.assertTrue(
                    result.startswith("Error: invalid arguments for write_file:"), result)
        self.assertEqual(self.written, [])

    def test_args_that_do_not_bind_are_an_error_result(self) -> None:
        for raw in ({"bogus": 1}, {"path": "a.py"},
                    {"path": "a.py", "content": "x", "extra": 2}):
            with self.subTest(raw=raw):
                result = llm.dispatch_tool_call("write_file", raw, self.handlers)
                self.assertTrue(
                    result.startswith("Error: invalid arguments for write_file:"), result)
        self.assertEqual(self.written, [])

    def test_unknown_tool_keeps_existing_text(self) -> None:
        self.assertEqual(llm.dispatch_tool_call("nope", {}, self.handlers),
                         "Unknown tool: nope")

    def test_type_error_inside_handler_body_propagates(self) -> None:
        def broken(path: str) -> str:
            raise TypeError("bug inside the handler")

        with self.assertRaisesRegex(TypeError, "bug inside the handler"):
            llm.dispatch_tool_call("broken", {"path": "a"}, {"broken": broken})


class RunAgentLoopBadArgsTests(support.GuardedTestCase):
    def setUp(self) -> None:
        self.written: list[tuple[str, str]] = []

        def write_file(path: str, content: str) -> str:
            self.written.append((path, content))
            return f"Written: {path}"

        self.handlers = {"write_file": write_file}
        self.seen_args: list[Any] = []

    def _run(self, fake: ScriptedChat) -> str:
        def on_tool_call(name: str, args: dict) -> None:
            self.seen_args.append(args.get("path"))

        with mock.patch.object(llm, "chat", fake), \
                contextlib.redirect_stdout(io.StringIO()):
            return llm.run_agent_loop(
                system="sys", user="build it", tools=[], handlers=self.handlers,
                on_tool_call=on_tool_call)

    def _tool_messages(self, sent: list[dict]) -> list[str]:
        return [m["content"] for m in sent if m["role"] == "tool"]

    def test_not_json_args_return_error_to_the_model(self) -> None:
        fake = ScriptedChat([_tool_call("write_file", "not json"), _text("done")])
        self.assertEqual(self._run(fake), "done")
        tool_msgs = self._tool_messages(fake.sent[1])
        self.assertEqual(len(tool_msgs), 1)
        self.assertTrue(tool_msgs[0].startswith("Error: invalid arguments for write_file:"))
        self.assertEqual(self.written, [])

    def test_bogus_keyword_returns_error_to_the_model(self) -> None:
        fake = ScriptedChat([_tool_call("write_file", {"bogus": 1}), _text("done")])
        self.assertEqual(self._run(fake), "done")
        self.assertTrue(self._tool_messages(fake.sent[1])[0].startswith(
            "Error: invalid arguments for write_file:"))
        self.assertEqual(self.written, [])

    def test_non_dict_json_args_do_not_crash_the_callback(self) -> None:
        fake = ScriptedChat([_tool_call("write_file", "[1]"), _text("done")])
        self.assertEqual(self._run(fake), "done")
        self.assertEqual(self.seen_args, [None])
        self.assertTrue(self._tool_messages(fake.sent[1])[0].startswith(
            "Error: invalid arguments for write_file:"))

    def test_bogus_args_on_the_no_progress_path(self) -> None:
        bad = _tool_call("write_file", {"bogus": 1})
        fake = ScriptedChat([bad, bad, bad, _text("gave up")])
        self.assertEqual(self._run(fake), "gave up")
        final_sent = fake.sent[-1]
        self.assertTrue(self._tool_messages(final_sent)[-1].startswith(
            "Error: invalid arguments for write_file:"))
        self.assertEqual(self.written, [])

    def test_valid_call_runs_and_its_result_reaches_the_model(self) -> None:
        fake = ScriptedChat([
            _tool_call("write_file", '{"path": "a.py", "content": "x"}'),
            _text("done"),
        ])
        self.assertEqual(self._run(fake), "done")
        self.assertEqual(self.written, [("a.py", "x")])
        self.assertEqual(self._tool_messages(fake.sent[1]), ["Written: a.py"])
        self.assertEqual(self.seen_args, ["a.py"])

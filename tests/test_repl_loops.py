"""Pins the agent.py and vibe.py REPL tool loops.

The model is a scripted fake `chat`; input() is scripted; the working
directory and vibe's reference-repo dir are temporary, so no model,
subprocess or real project is touched.
"""

import contextlib
import copy
import io
import os
import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any, Iterator
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai import agent, vibe
from local_ai.agents.shared import tools

REPLS: list[ModuleType] = [agent, vibe]
RUNAWAY_CAP = 100  # replies before the fake model runs dry (StopIteration)


def _text(content: str) -> dict:
    return {"role": "assistant", "content": content, "tool_calls": None}


def _tool_call(name: str, arguments: Any) -> dict:
    return {"role": "assistant", "content": "",
            "tool_calls": [{"function": {"name": name, "arguments": arguments}}]}


class RecordingChat:
    """Fake chat: replies come from an iterator; every call is recorded."""

    def __init__(self, replies: Iterator[dict]) -> None:
        self.replies = replies
        self.sent: list[list[dict]] = []

    def __call__(self, messages: list, tools: list = None, timeout: int = 180) -> dict:
        self.sent.append(copy.deepcopy(messages))
        return next(self.replies)


class ReplTestCase(support.GuardedTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name)
        self._cwd = os.getcwd()
        os.chdir(self.dir)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()
        super().tearDown()

    def _patch_chat(self, fake: RecordingChat) -> Any:
        return mock.patch("local_ai.agents.shared.llm.chat", fake)

    def _run_agent(self, module: ModuleType, fake: RecordingChat,
                   messages: list) -> tuple[str, str]:
        out = io.StringIO()
        with self._patch_chat(fake), contextlib.redirect_stdout(out):
            reply = module.run_agent(messages)
        return reply, out.getvalue()

    def _run_main(self, module: ModuleType, fake: RecordingChat,
                  inputs: list[str]) -> str:
        out = io.StringIO()
        with self._patch_chat(fake), \
                mock.patch("builtins.input", side_effect=inputs), \
                mock.patch("sys.argv", [module.__name__]), \
                mock.patch.object(vibe, "REPOS_DIR", self.dir / "no-repos"), \
                contextlib.redirect_stdout(out):
            module.main()
        return out.getvalue()


class ReplBehaviourTests(ReplTestCase):
    """Written before the loops moved onto agents/shared/llm.py."""

    def test_embedded_content_tool_call_runs_and_result_is_fed_back(self) -> None:
        (self.dir / "notes.txt").write_text("invented note body", encoding="utf-8")
        for module in REPLS:
            with self.subTest(module=module.__name__):
                fake = RecordingChat(iter([
                    _text('{"name": "read_file", "arguments": {"path": "notes.txt"}}'),
                    _text("all read"),
                ]))
                messages = [{"role": "system", "content": "sys"},
                            {"role": "user", "content": "read it"}]
                reply, _ = self._run_agent(module, fake, messages)
                self.assertEqual(reply.strip(), "all read")
                tool_msgs = [m["content"] for m in fake.sent[1] if m["role"] == "tool"]
                self.assertEqual(len(tool_msgs), 1)
                self.assertIn("invented note body", tool_msgs[0])
                # The caller's list holds the whole exchange.
                self.assertEqual([m["role"] for m in messages],
                                 ["system", "user", "assistant", "tool", "assistant"])

    def test_history_is_kept_across_turns_and_reply_printed(self) -> None:
        for module in REPLS:
            with self.subTest(module=module.__name__):
                fake = RecordingChat(iter([_text("first answer"),
                                           _text("second answer")]))
                out = self._run_main(module, fake, ["first question",
                                                    "second question", "exit"])
                self.assertEqual(len(fake.sent), 2)
                second = [(m["role"], m["content"]) for m in fake.sent[1]]
                self.assertIn(("user", "first question"), second)
                self.assertIn(("assistant", "first answer"), second)
                self.assertEqual(second[-1], ("user", "second question"))
                self.assertIn("first answer", out)
                self.assertIn("second answer", out)
                self.assertIn("Bye!", out)


class ReplLoopBoundTests(ReplTestCase):
    def test_always_calling_model_stops_after_max_rounds(self) -> None:
        for module in REPLS:
            with self.subTest(module=module.__name__):
                # Distinct args each round so the no-progress detector never
                # fires; capped so an unbounded loop fails instead of hanging.
                calls = (_tool_call("read_file", {"path": f"missing-{n}.txt"})
                         for n in range(RUNAWAY_CAP))
                fake = RecordingChat(calls)
                messages = [{"role": "user", "content": "loop forever"}]
                reply, out = self._run_agent(module, fake, messages)
                self.assertEqual(len(fake.sent), 20)
                self.assertIn("[agent] stopped after 20 rounds", out)
                self.assertIn("[agent] stopped after 20 rounds", reply)

    def test_bad_args_become_an_error_result(self) -> None:
        for module in REPLS:
            with self.subTest(module=module.__name__):
                for bad in ("not json", {"bogus": 1}, "[1]"):
                    fake = RecordingChat(iter([_tool_call("write_file", bad),
                                               _text("recovered")]))
                    reply, _ = self._run_agent(
                        module, fake, [{"role": "user", "content": "go"}])
                    self.assertEqual(reply.strip(), "recovered")
                    tool_msgs = [m["content"] for m in fake.sent[1]
                                 if m["role"] == "tool"]
                    self.assertTrue(tool_msgs[0].startswith(
                        "Error: invalid arguments for write_file:"), tool_msgs)
                self.assertEqual(list(self.dir.iterdir()), [])


class ReplToolSourceTests(support.GuardedTestCase):
    """The REPLs use the one tool implementation in agents/shared/tools.py."""

    REPL_TOOLS = ["read_file", "write_file", "list_files", "run_command", "search_code"]

    def test_handlers_and_schemas_are_the_shared_ones(self) -> None:
        shared_schemas = {t["function"]["name"]: t for t in
                          (tools.READ_FILE, tools.WRITE_FILE, tools.LIST_FILES,
                           tools.RUN_COMMAND, tools.SEARCH_CODE)}
        for module in REPLS:
            with self.subTest(module=module.__name__):
                self.assertEqual(sorted(module.TOOL_HANDLERS), sorted(self.REPL_TOOLS))
                for name, handler in module.TOOL_HANDLERS.items():
                    self.assertIs(handler, tools.HANDLERS[name])
                self.assertEqual(len(module.TOOLS), len(self.REPL_TOOLS))
                for schema in module.TOOLS:
                    self.assertIs(schema, shared_schemas[schema["function"]["name"]])

    def test_no_own_tool_code_left_in_the_repls(self) -> None:
        for module in REPLS:
            with self.subTest(module=module.__name__):
                source = Path(module.__file__).read_text(encoding="utf-8")
                for marker in ("shell=True", "json.loads", "def run_command",
                               "def write_file", "def read_file", "handler(**"):
                    self.assertNotIn(marker, source)

"""An unavailable model backend raises LLMUnavailableError (agents/shared/llm.py)
instead of calling sys.exit, and each CLI turns it into exit 1 with the message.

urlopen and time.sleep are mocked and `anthropic` is a fake module, so no
network, Ollama or API key is ever used.
"""

import contextlib
import io
import os
import sys
import types
import urllib.error
from typing import Any
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai import agent, build_cli, vibe
from local_ai.agents.shared import llm

INVENTED_KEY = "sk-invented-not-a-real-key"
MESSAGES = [{"role": "user", "content": "hi"}]


def _fake_anthropic(recorded: dict) -> types.ModuleType:
    """A stand-in `anthropic` module whose client records its constructor kwargs."""

    class _Stream:
        text_stream: list[str] = []

        def __enter__(self) -> "_Stream":
            return self

        def __exit__(self, *exc: Any) -> None:
            return None

        def get_final_message(self) -> Any:
            return types.SimpleNamespace(content=[])

    class Anthropic:
        def __init__(self, **kwargs: Any) -> None:
            recorded.update(kwargs)
            self.messages = types.SimpleNamespace(stream=lambda **kw: _Stream())

    module = types.ModuleType("anthropic")
    module.Anthropic = Anthropic  # type: ignore[attr-defined]
    return module


class OllamaUnreachableTests(support.GuardedTestCase):
    def test_unreachable_ollama_raises_after_retries(self) -> None:
        refused = urllib.error.URLError("connection refused")
        with mock.patch.object(llm.urllib.request, "urlopen", side_effect=refused) as urlopen, \
                mock.patch.object(llm.time, "sleep") as sleep, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(llm.LLMUnavailableError) as cm:
                llm._ollama_chat(MESSAGES, [], "qwen2.5-coder:7b", timeout=5)
        self.assertEqual(urlopen.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertIn("Cannot reach Ollama after 3 attempts", str(cm.exception))
        self.assertNotIsInstance(cm.exception, SystemExit)


class ClaudeBackendTests(support.GuardedTestCase):
    def test_missing_anthropic_package_raises(self) -> None:
        with mock.patch.dict(sys.modules, {"anthropic": None}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": INVENTED_KEY}):
            with self.assertRaises(llm.LLMUnavailableError) as cm:
                llm._claude_chat(MESSAGES, [], "claude-x", timeout=7)
        self.assertIn("anthropic package not installed", str(cm.exception))
        self.assertNotIn(INVENTED_KEY, str(cm.exception))

    def test_missing_api_key_raises(self) -> None:
        recorded: dict = {}
        env = {k: v for k, v in os.environ.items() if k != "ANTHROPIC_API_KEY"}
        with mock.patch.dict(sys.modules, {"anthropic": _fake_anthropic(recorded)}), \
                mock.patch.dict(os.environ, env, clear=True):
            with self.assertRaises(llm.LLMUnavailableError) as cm:
                llm._claude_chat(MESSAGES, [], "claude-x", timeout=7)
        self.assertIn("ANTHROPIC_API_KEY not set", str(cm.exception))
        self.assertEqual(recorded, {})

    def test_client_gets_the_timeout(self) -> None:
        recorded: dict = {}
        with mock.patch.dict(sys.modules, {"anthropic": _fake_anthropic(recorded)}), \
                mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": INVENTED_KEY}), \
                contextlib.redirect_stdout(io.StringIO()):
            reply = llm._claude_chat(MESSAGES, [], "claude-x", timeout=7)
        self.assertEqual(recorded.get("timeout"), 7)
        self.assertEqual(reply["content"], "")


class EntryPointTests(support.GuardedTestCase):
    """Each CLI main() exits 1 with the message when the backend is unavailable."""

    def _assert_exits_1(self, main: Any, patch: Any) -> None:
        err = llm.LLMUnavailableError("invented: backend is down")
        stderr = io.StringIO()
        with patch(side_effect=err), \
                mock.patch("builtins.input", side_effect=["build a thing", "exit"]), \
                contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(stderr):
            with self.assertRaises(SystemExit) as cm:
                main()
        self.assertEqual(cm.exception.code, "invented: backend is down")
        # sys.exit(<str>) prints the string to stderr and exits with status 1.
        self.assertIsInstance(cm.exception.code, str)

    def test_build_app(self) -> None:
        with mock.patch("sys.argv", ["build-app", "an invented app"]):
            self._assert_exits_1(build_cli.main,
                                 lambda **kw: mock.patch.object(build_cli.orchestrator, "run", **kw))

    def test_vibe(self) -> None:
        with mock.patch("sys.argv", ["vibe"]), \
                mock.patch.object(vibe, "REPOS_DIR", vibe.Path("/nonexistent-invented")), \
                mock.patch.object(vibe.glob_module, "glob", return_value=[]):
            self._assert_exits_1(vibe.main,
                                 lambda **kw: mock.patch.object(vibe, "run_agent", **kw))

    def test_agent(self) -> None:
        with mock.patch("sys.argv", ["ai"]), \
                mock.patch.object(agent.glob_module, "glob", return_value=[]):
            self._assert_exits_1(agent.main,
                                 lambda **kw: mock.patch.object(agent, "run_agent", **kw))

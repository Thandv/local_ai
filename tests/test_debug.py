"""LOCAL_AI_DEBUG=1 prints debug lines to stderr; off prints nothing extra;
secrets are redacted. urlopen, the guard and subprocess are mocked."""

import contextlib
import io
import json
import os
import sys
import tempfile
import types
import urllib.error
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai.agents.shared import debug, guard, llm, tools

SENTINEL = "PROMPT-SENTINEL-7f3a"
MESSAGES = [{"role": "user", "content": SENTINEL}]


class _Resp:
    def __init__(self, lines):
        self._lines = lines

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return None

    def __iter__(self):
        return iter(self._lines)


def _ok_reply():
    return _Resp([json.dumps({"message": {"content": "hi"}, "done": True}).encode()])


class DebugLogTests(support.GuardedTestCase):
    def _run(self, switch, fn):
        """Run fn with the switch set (or unset) and return captured stderr."""
        err = io.StringIO()
        with mock.patch.dict(os.environ), contextlib.redirect_stderr(err), \
                contextlib.redirect_stdout(io.StringIO()):
            os.environ.pop("LOCAL_AI_DEBUG", None)
            if switch is not None:
                os.environ["LOCAL_AI_DEBUG"] = switch
            debug.setup_debug()
            try:
                fn()
            finally:
                os.environ["LOCAL_AI_DEBUG"] = "0"
                debug.setup_debug()
        return err.getvalue()

    def _chat(self):
        with mock.patch.object(llm.urllib.request, "urlopen", return_value=_ok_reply()):
            llm._ollama_chat(MESSAGES, [], "qwen2.5-coder:7b", timeout=5)

    def test_off_prints_nothing(self):
        self.assertEqual(self._run(None, self._chat), "")
        self.assertEqual(self._run("0", self._chat), "")

    def test_on_logs_model_call_without_prompt(self):
        err = self._run("1", self._chat)
        self.assertIn("[local_ai debug] local_ai.llm ollama chat start model=qwen2.5-coder:7b", err)
        self.assertIn("ollama chat end status=ok", err)
        self.assertNotIn(SENTINEL, err)

    def test_failed_call_logs_end_line(self):
        def fail():
            with mock.patch.object(llm.urllib.request, "urlopen",
                                   side_effect=urllib.error.URLError("refused")), \
                    mock.patch.object(llm.time, "sleep"):
                with self.assertRaises(llm.LLMUnavailableError):
                    llm._ollama_chat(MESSAGES, [], "m", timeout=5)
        err = self._run("1", fail)
        self.assertEqual(err.count("ollama chat end status=error model=m error=URLError"), 3)
        self.assertNotIn(SENTINEL, err)

    def test_claude_call_logs_start_and_end_without_prompt_or_key(self):
        class _Stream:
            text_stream = ["hi"]

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return None

            def get_final_message(self):
                return types.SimpleNamespace(content=[], stop_reason="end_turn")

        fake = types.ModuleType("anthropic")
        fake.Anthropic = lambda **kw: types.SimpleNamespace(  # type: ignore[attr-defined]
            messages=types.SimpleNamespace(stream=lambda **kw: _Stream()))
        key = "sk-ant-invented-key-0000"
        msgs = [{"role": "system", "content": SENTINEL}] + MESSAGES

        def call():
            with mock.patch.dict(sys.modules, {"anthropic": fake}), \
                    mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": key}):
                llm._claude_chat(msgs, [], "claude-x", timeout=5)
        err = self._run("1", call)
        self.assertIn("claude chat start model=claude-x messages=1", err)
        self.assertIn("claude chat end status=ok model=claude-x", err)
        self.assertNotIn(SENTINEL, err)
        self.assertNotIn(key, err)

    def test_run_command_logs_status_not_output(self):
        outcome = guard.Outcome("ran", ("ls", "-la"), returncode=0, output=SENTINEL)
        with mock.patch.object(tools.guard, "run_argv", return_value=outcome):
            err = self._run("1", lambda: tools.run_command("ls -la"))
        self.assertIn("run_command argv0=ls args=1 cwd=. status=ran rc=0", err)
        self.assertNotIn(SENTINEL, err)

    def test_swallowed_errors_are_logged(self):
        def swallow():
            with mock.patch.object(llm.urllib.request, "urlopen", side_effect=OSError("down")):
                self.assertEqual(llm._list_ollama_models(), [])
            loop: dict = {}
            loop["self"] = loop
            llm._call_signature("t", loop)
            with tempfile.TemporaryDirectory() as d, \
                    mock.patch.object(tools.subprocess, "run", side_effect=OSError("x")):
                tools.grep_repos(["kw"], repos_dir=d)
        err = self._run("1", swallow)
        self.assertIn("ollama tags end status=error error=OSError", err)
        self.assertIn("call signature not JSON tool=t error=ValueError", err)
        self.assertIn("grep_repos skipped error=OSError", err)

    def test_secrets_are_redacted(self):
        shapes = ("key=v1secret", '{"api_key": "v2secret"}',
                  "Authorization: Bearer v3secret", "sk-v4secret")
        err = self._run("1", lambda: [debug.log.debug("x %s", s) for s in shapes])
        self.assertEqual(err.count("[REDACTED]"), 4)
        for s in ("v1secret", "v2secret", "v3secret", "v4secret"):
            self.assertNotIn(s, err)

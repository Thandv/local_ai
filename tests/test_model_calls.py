"""Model calls fail cleanly (LAI-E3 t2).

A fake Ollama: a stub HTTP server on 127.0.0.1 with a random port (never
11434), started and stopped by each test. Only for those tests are urlopen
and socket connections let through, and llm.OLLAMA_URL points at the stub.
No real model or Ollama is used; time.sleep (the retry back-off) is a no-op.
"""

import contextlib
import io
import json
import os
import socket
import tempfile
import threading
import time
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Callable, Iterator
from unittest import mock

import support
from local_ai import agent, vibe
from local_ai.agents.shared import guard, llm

MESSAGES = [{"role": "user", "content": "hi"}]


def _line(content: str = "", done: bool = False) -> bytes:
    return (json.dumps({"message": {"role": "assistant", "content": content},
                        "done": done}) + "\n").encode()


class StubOllama:
    """A local stand-in for Ollama's /api/chat. Each request is answered by
    the next behaviour in `script` (the last one repeats)."""

    def __init__(self, script: list[str]) -> None:
        self.script = list(script)
        self.requests = 0
        self.stop = threading.Event()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.0"  # the body ends when the connection closes

            def log_message(self, *args: object) -> None:
                pass

            def do_POST(self) -> None:
                self.rfile.read(int(self.headers.get("Content-Length", "0")))
                stub.requests += 1
                mode = stub.script[min(stub.requests, len(stub.script)) - 1]
                if mode == "http500":
                    self.send_error(500, "invented server error")
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.end_headers()
                try:
                    stub.serve(mode, self.wfile)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.port = self.server.server_address[1]
        assert self.port != 11434, "the stub must never use Ollama's real port"
        self.url = f"http://127.0.0.1:{self.port}/api/chat"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def serve(self, mode: str, out: "io.BufferedIOBase") -> None:
        if mode == "ok":
            out.write(_line("hello ") + _line("there") + _line(done=True))
        elif mode == "stall":  # one token, then silence
            out.write(_line("partial"))
            out.flush()
            self.stop.wait(10)
        elif mode == "trickle":  # a token every 20 ms (stops after 10 s at most)
            end = time.monotonic() + 10
            while not self.stop.is_set() and time.monotonic() < end:
                out.write(_line("."))
                out.flush()
                time.sleep(0.02)
        elif mode == "cut":  # the connection closes before "done"
            out.write(_line("partial"))
        else:
            raise AssertionError(mode)

    def __enter__(self) -> "StubOllama":
        self.thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(5)


@contextlib.contextmanager
def fake_ollama(*script: str) -> Iterator[StubOllama]:
    """Run a stub Ollama, let urlopen reach it, and point the client at it."""
    real_urlopen = support.REAL["urllib.request.urlopen"]
    real_connect = support.REAL["socket.create_connection"]

    def connect(address: tuple, *args: object, **kwargs: object) -> socket.socket:
        assert address[0] == "127.0.0.1" and address[1] != 11434, address
        return real_connect(address, *args, **kwargs)

    with StubOllama(list(script)) as stub, \
            mock.patch.object(urllib.request, "urlopen", real_urlopen), \
            mock.patch.object(socket, "create_connection", connect), \
            mock.patch.object(llm, "OLLAMA_URL", stub.url), \
            mock.patch.object(llm, "_resolve_active_model", lambda: None), \
            mock.patch.object(llm.time, "sleep", lambda s: None):
        yield stub


class OllamaStreamTests(support.GuardedTestCase):
    def call(self, timeout: float = 0.3) -> dict:
        with contextlib.redirect_stdout(io.StringIO()):
            return llm._ollama_chat(MESSAGES, [], "invented-model", timeout)

    def test_control_a_normal_stream_returns_the_message(self) -> None:
        with fake_ollama("ok") as stub:
            reply = self.call()
        self.assertEqual(reply["content"], "hello there")
        self.assertEqual(stub.requests, 1)

    def test_a_mid_stream_read_timeout_is_model_unavailable(self) -> None:
        with fake_ollama("stall") as stub:
            with self.assertRaises(llm.LLMUnavailableError) as cm:
                self.call(timeout=0.3)
        self.assertEqual(stub.requests, 3)  # retried like a connection error
        self.assertIn("Ollama", str(cm.exception))

    def test_a_retry_after_a_stall_starts_the_reply_afresh(self) -> None:
        with fake_ollama("stall", "ok") as stub:
            reply = self.call(timeout=0.3)
        self.assertEqual(reply["content"], "hello there")  # not "partialhello there"
        self.assertEqual(stub.requests, 2)

    def test_a_stream_cut_before_done_is_model_unavailable(self) -> None:
        with fake_ollama("cut") as stub:
            with self.assertRaises(llm.LLMUnavailableError):
                self.call()
        self.assertEqual(stub.requests, 3)

    def test_an_http_error_is_model_unavailable(self) -> None:
        with fake_ollama("http500"):
            with self.assertRaises(llm.LLMUnavailableError):
                self.call()

    def test_a_trickle_that_never_ends_stops_at_the_call_deadline(self) -> None:
        t0 = time.monotonic()
        with fake_ollama("trickle") as stub, mock.patch.object(llm, "CALL_DEADLINE_S", 0.5, create=True):
            with self.assertRaises(llm.LLMUnavailableError) as cm:
                self.call(timeout=5)
        self.assertLess(time.monotonic() - t0, 3)
        self.assertEqual(stub.requests, 1)  # the deadline covers the whole call
        self.assertIn("deadline", str(cm.exception))

    def test_chat_passes_the_deadline_down(self) -> None:
        with fake_ollama("stall"), mock.patch.object(llm, "CALL_DEADLINE_S", 0.5, create=True):
            t0 = time.monotonic()
            with self.assertRaises(llm.LLMUnavailableError), \
                    contextlib.redirect_stdout(io.StringIO()):
                llm.chat(MESSAGES, timeout=60)
        self.assertLess(time.monotonic() - t0, 3)


class ReplSurvivesABadTurnTests(support.GuardedTestCase):
    """vibe and agent keep the session after a turn that fails."""

    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self._cwd = os.getcwd()
        os.chdir(self._tmp.name)
        self.addCleanup(os.chdir, self._cwd)

    def run_repl(self, main: Callable[[], None], argv0: str,
                 inputs: list[str]) -> tuple[str, str]:
        out, err = io.StringIO(), io.StringIO()
        with mock.patch("sys.argv", [argv0]), \
                mock.patch.object(vibe, "REPOS_DIR", Path("/nonexistent-invented")), \
                mock.patch("builtins.input", side_effect=inputs), \
                contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            main()
        guard._ACTIVE.set(None)
        return out.getvalue(), err.getvalue()

    def test_a_model_timeout_mid_turn_keeps_the_session(self) -> None:
        for module, argv0 in ((vibe, "vibe"), (agent, "ai")):
            with self.subTest(repl=argv0):
                sent: list[list[dict]] = []
                real_chat = llm.chat

                def recording_chat(messages: list, tools: list = None, timeout: float = 0.3,
                                   sent: list = sent, real_chat: Callable = real_chat) -> dict:
                    sent.append([m["content"] for m in messages if m["role"] == "user"])
                    return real_chat(messages, tools, timeout=0.3)

                with fake_ollama("stall", "stall", "stall", "ok") as stub, \
                        mock.patch.object(llm, "chat", recording_chat):
                    out, err = self.run_repl(module.main, argv0,
                                             ["first question", "second question", "exit"])
                self.assertEqual(stub.requests, 4)
                self.assertIn("Cannot reach Ollama", err)
                self.assertIn("hello there", out)  # the second turn was answered
                self.assertIn("Bye!", out)
                # The failed turn is dropped from the history, not left dangling.
                self.assertEqual(sent[-1], ["second question"])

    def test_any_error_in_a_turn_keeps_the_session(self) -> None:
        for module, argv0 in ((vibe, "vibe"), (agent, "ai")):
            with self.subTest(repl=argv0):
                replies = iter([RuntimeError("invented tool crash"), "second answer"])

                def run_agent(messages: list, replies: Iterator = replies) -> str:
                    reply = next(replies)
                    if isinstance(reply, Exception):
                        raise reply
                    return reply

                with mock.patch.object(module, "run_agent", run_agent):
                    out, err = self.run_repl(module.main, argv0, ["one", "two", "exit"])
                self.assertIn("invented tool crash", err)
                self.assertIn("second answer", out)
                self.assertIn("Bye!", out)

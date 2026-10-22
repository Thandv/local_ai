"""Test guard: blocks network and subprocess calls for the whole test run.

Import this module before any `local_ai` import. On import it replaces
urllib.request.urlopen, subprocess.run, subprocess.Popen and
socket.create_connection with stubs that record the call and raise
GuardViolation. GuardViolation derives from BaseException, so the product
code's `except Exception` / `except URLError` handlers cannot swallow it,
retry it, sleep on it or turn it into sys.exit.

Every test case derives from GuardedTestCase, whose tearDown fails the test
if any guarded call has been recorded so far in the run.
"""

import socket
import subprocess
import sys
import unittest
import urllib.request
from typing import Any, Callable


class GuardViolation(BaseException):
    """Raised when code under test tries to reach the network or a subprocess."""


CALLS: list[str] = []

GUARDED: dict[str, tuple[Any, str]] = {
    "urllib.request.urlopen":   (urllib.request, "urlopen"),
    "subprocess.run":           (subprocess, "run"),
    "subprocess.Popen":         (subprocess, "Popen"),
    "socket.create_connection": (socket, "create_connection"),
}


def _make_stub(name: str) -> Callable[..., Any]:
    def stub(*args: Any, **kwargs: Any) -> Any:
        CALLS.append(f"{name} args={args!r} kwargs={kwargs!r}")
        raise GuardViolation(f"blocked {name} during tests: {args!r}")
    stub.guard_name = name  # type: ignore[attr-defined]
    return stub


def install() -> None:
    """Replace every guarded callable with a recording stub (idempotent)."""
    for name, (module, attr) in GUARDED.items():
        if getattr(getattr(module, attr), "guard_name", None) != name:
            setattr(module, attr, _make_stub(name))


def is_installed() -> bool:
    return all(
        getattr(getattr(module, attr), "guard_name", None) == name
        for name, (module, attr) in GUARDED.items()
    )


class GuardedTestCase(unittest.TestCase):
    """Base class for every test: fails if any guarded call was recorded."""

    def tearDown(self) -> None:
        # A REPL main() confines its tools for the rest of the process; drop
        # that sandbox so it cannot leak into the next test.
        guard = sys.modules.get("local_ai.agents.shared.guard")
        if guard is not None:
            guard._ACTIVE.set(None)
        self.assertEqual(CALLS, [], "a guarded network/subprocess call was made")
        super().tearDown()


install()

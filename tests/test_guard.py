"""The guard in support.py is active, complete, and fails tests that trip it."""

import io
import socket
import subprocess
import unittest
import urllib.request

import support

EXPECTED = [
    "urllib.request.urlopen",
    "subprocess.run",
    "subprocess.Popen",
    "socket.create_connection",
]


class GuardTests(support.GuardedTestCase):
    def test_guard_covers_exactly_the_expected_callables(self) -> None:
        self.assertEqual(list(support.GUARDED), EXPECTED)
        live = {
            "urllib.request.urlopen": urllib.request.urlopen,
            "subprocess.run": subprocess.run,
            "subprocess.Popen": subprocess.Popen,
            "socket.create_connection": socket.create_connection,
        }
        for name, fn in live.items():
            self.assertEqual(getattr(fn, "guard_name", None), name)
        self.assertTrue(support.is_installed())

    def test_guarded_calls_raise_and_are_recorded_by_name(self) -> None:
        calls = [
            ("urllib.request.urlopen",
             lambda: urllib.request.urlopen("http://127.0.0.1:9/")),
            ("subprocess.run", lambda: subprocess.run(["false"])),
            ("subprocess.Popen", lambda: subprocess.Popen(["false"])),
            ("socket.create_connection",
             lambda: socket.create_connection(("127.0.0.1", 9))),
        ]
        live = {
            "urllib.request.urlopen": urllib.request.urlopen,
            "subprocess.run": subprocess.run,
            "subprocess.Popen": subprocess.Popen,
            "socket.create_connection": socket.create_connection,
        }
        # Check every live callable is a stub before calling any of them, so a
        # broken guard never runs a real process or opens a real connection.
        unguarded = [n for n, fn in live.items()
                     if getattr(fn, "guard_name", None) != n]
        self.assertEqual(unguarded, [], "refusing to call unguarded functions")
        saved = list(support.CALLS)
        try:
            support.CALLS.clear()
            for _, call in calls:
                with self.assertRaises(support.GuardViolation):
                    call()
            self.assertEqual([c.split(" ", 1)[0] for c in support.CALLS],
                             [name for name, _ in calls])
        finally:
            support.CALLS[:] = saved

    def test_guard_violation_escapes_except_exception(self) -> None:
        self.assertFalse(issubclass(support.GuardViolation, Exception))

    def test_teardown_fails_a_test_once_a_call_is_recorded(self) -> None:
        class Probe(support.GuardedTestCase):
            def test_noop(self) -> None:
                pass

        saved = list(support.CALLS)
        try:
            support.CALLS.append("subprocess.run args=('fake',) kwargs={}")
            result = unittest.TextTestRunner(stream=io.StringIO()).run(
                Probe("test_noop"))
        finally:
            support.CALLS[:] = saved
        self.assertEqual(len(result.failures), 1)
        self.assertIn("guarded network/subprocess call", result.failures[0][1])

    def test_no_guarded_call_recorded_so_far(self) -> None:
        # Covers tests that ran before this one; every later test is covered
        # by GuardedTestCase.tearDown.
        self.assertEqual(support.CALLS, [])

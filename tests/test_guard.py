"""The guard in support.py is active and nothing in the run tripped it."""

import socket
import subprocess
import urllib.request

import support


class GuardTests(support.GuardedTestCase):
    def test_guard_is_installed(self) -> None:
        self.assertTrue(support.is_installed())

    def test_guarded_calls_raise_and_are_recorded(self) -> None:
        saved = list(support.CALLS)
        try:
            calls = [
                lambda: urllib.request.urlopen("http://127.0.0.1:9/"),
                lambda: subprocess.run(["true"]),
                lambda: subprocess.Popen(["true"]),
                lambda: socket.create_connection(("127.0.0.1", 9)),
            ]
            for call in calls:
                with self.assertRaises(support.GuardViolation):
                    call()
            self.assertEqual(len(support.CALLS), len(saved) + len(calls))
        finally:
            support.CALLS[:] = saved

    def test_guard_violation_escapes_except_exception(self) -> None:
        self.assertFalse(issubclass(support.GuardViolation, Exception))

    def test_no_guarded_call_recorded_in_run(self) -> None:
        self.assertEqual(support.CALLS, [])

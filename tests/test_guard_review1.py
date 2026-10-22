"""LA-3 review round 1 (file confinement): each test pins a gap the
independent reviewer reproduced. subprocess is a recording mock throughout.
"""

import contextlib
import io
import os
import subprocess
import tempfile
from pathlib import Path
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai.agents import tester
from local_ai.agents.shared import guard, tools


def _completed(argv, rc=0, out="ok"):
    return subprocess.CompletedProcess(argv, rc, out, "")


class ReviewRound1Tests(support.GuardedTestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        base = Path(os.path.realpath(self._tmp.name))
        self.root = base / "root"
        self.outside = base / "outside"
        self.root.mkdir()
        self.outside.mkdir()
        self.run_mock = mock.Mock(side_effect=lambda argv, **kw: _completed(argv))
        stack = contextlib.ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(mock.patch.object(guard.subprocess, "run", self.run_mock))
        stack.enter_context(mock.patch.object(tools.subprocess, "run", self.run_mock))
        stack.enter_context(mock.patch.object(guard, "_which",
                                              lambda name, path: f"/usr/bin/{name}"))
        stack.enter_context(mock.patch.object(guard, "_has_tty", lambda: False))
        stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
        self._token = guard._ACTIVE.set(guard.make_sandbox(self.root))

    def tearDown(self) -> None:
        guard._ACTIVE.reset(self._token)
        self._tmp.cleanup()
        super().tearDown()

    def test_a_model_written_bare_repo_layout_is_never_used_by_git(self) -> None:
        w = tools.write_file
        w("evil/HEAD", "ref: refs/heads/main\n")
        w("evil/objects/info/keep", "")
        w("evil/refs/heads/keep", "")
        w("evil/config", '[core]\n\tworktree = ..\n[filter "x"]\n\tclean = touch PWNED\n')
        w(".gitattributes", "* filter=x\n")
        # No ROOT/.git: nothing to run git on without confirmation.
        outcome = guard.run_argv(["git", "add", "-A"], working_dir="evil")
        self.assertNotEqual(outcome.status, "ran")
        self.run_mock.assert_not_called()
        # With ROOT/.git, git is pinned to it and to the root work tree.
        (self.root / ".git").mkdir()
        outcome = guard.run_argv(["git", "add", "-A"], working_dir="evil")
        self.assertEqual(outcome.status, "ran")
        argv, env = self.run_mock.call_args.args[0], self.run_mock.call_args.kwargs["env"]
        self.assertEqual(env["GIT_DIR"], str(self.root / ".git"))
        self.assertEqual(env["GIT_WORK_TREE"], str(self.root))
        self.assertIn("safe.bareRepository=explicit", argv)

    def test_git_init_takes_no_directory(self) -> None:
        for argv in (["git", "init", "evil"], ["git", "init", "."]):
            with self.subTest(argv=argv):
                self.assertNotEqual(guard.run_argv(argv).status, "ran")
        self.run_mock.assert_not_called()

    def test_list_files_never_scans_symlinked_dirs_outside_the_root(self) -> None:
        (self.outside / "deep").mkdir()
        (self.outside / "deep" / "s.py").write_text("x")
        os.symlink(self.outside, self.root / "lnk")
        scanned = []
        real_scandir = os.scandir

        def recording(path=".", *a, **kw):
            if isinstance(path, (str, bytes, os.PathLike)):
                scanned.append(os.path.realpath(path))
            return real_scandir(path, *a, **kw)

        with mock.patch("os.scandir", recording):
            out = tools.list_files("**")
        self.assertNotIn("s.py", out)
        self.assertEqual([p for p in scanned if not guard.is_inside(guard.current(), p)], [])

    def test_list_files_does_not_loop_through_a_symlink_to_an_ancestor(self) -> None:
        (self.root / "sub").mkdir()
        (self.root / "a.py").write_text("x")
        os.symlink("..", self.root / "sub" / "up")
        self.assertEqual(tools.list_files("**/a.py").splitlines(), ["a.py"])

    def test_list_files_glob_semantics(self) -> None:
        for rel in ("a.py", "src/b.py", "src/deep/c.py", ".hidden.py", "src/x.txt"):
            (self.root / rel).parent.mkdir(parents=True, exist_ok=True)
            (self.root / rel).write_text("")
        self.assertEqual(tools.list_files("*.py"), "a.py")
        self.assertEqual(tools.list_files("**/*.py").splitlines(),
                         ["a.py", "src/b.py", "src/deep/c.py"])
        self.assertEqual(tools.list_files("src/*"), "src/b.py\nsrc/deep\nsrc/x.txt")
        self.assertEqual(tools.list_files(".*.py"), ".hidden.py")
        self.assertEqual(tools.list_files(str(self.root / "src" / "?.py")),
                         str(self.root / "src" / "b.py"))

    def test_search_code_skips_a_hard_linked_outside_file(self) -> None:
        secret = self.outside / "secret.py"
        secret.write_text("token = 'invented'\n")
        os.link(secret, self.root / "h.py")
        self.assertNotIn("invented", tools.search_code("token"))

    def test_search_code_does_not_follow_a_dir_swapped_for_a_symlink(self) -> None:
        (self.root / "d").mkdir()
        (self.outside / "s.py").write_text("token = 'invented'\n")
        real_resolve = guard.resolve_dir

        def racing(sandbox, path):
            target = real_resolve(sandbox, path)
            os.rename(self.root / "d", self.root / "d.old")
            os.symlink(self.outside, self.root / "d")
            return target

        with mock.patch.object(guard, "resolve_dir", racing):
            self.assertNotIn("invented", tools.search_code("token", "d"))

    def test_run_argv_refuses_a_cwd_swapped_for_a_symlink(self) -> None:
        (self.root / "w").mkdir()
        real_resolve = guard.resolve_dir

        def racing(sandbox, path):
            target = real_resolve(sandbox, path)
            os.rename(self.root / "w", self.root / "w.old")
            os.symlink(self.outside, self.root / "w")
            return target

        with mock.patch.object(guard, "resolve_dir", racing):
            outcome = guard.run_argv(["ls"], working_dir="w")
        self.assertEqual(outcome.status, "refused")
        self.run_mock.assert_not_called()

    def test_trailing_slash_is_refused(self) -> None:
        for path in ("newdir/", "newdir/."):
            with self.subTest(path=path):
                self.assertTrue(tools.write_file(path, "x").startswith("Refused"))
        self.assertFalse((self.root / "newdir").exists())

    def test_an_overlong_name_is_refused_before_any_mkdir(self) -> None:
        result = tools.write_file("k1/k2/k3/" + "n" * 300, "x")
        self.assertTrue(result.startswith("Refused"), result)
        self.assertFalse((self.root / "k1").exists())

    def test_tester_ignores_a_hard_linked_package_json(self) -> None:
        pkg = self.outside / "package.json"
        pkg.write_text('{"scripts": {"test": "jest"}}')
        os.link(pkg, self.root / "package.json")
        self.assertNotEqual(tester.detect_test_command(self.root).get("runner"), "npm")

    def test_files_other_tools_run_on_their_own_are_protected(self) -> None:
        for path in (".envrc", ".vscode/tasks.json", ".VSCode/settings.json"):
            with self.subTest(path=path):
                self.assertTrue(tools.write_file(path, "x").startswith("Refused"))

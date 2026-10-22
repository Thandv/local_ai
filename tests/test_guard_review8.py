"""Review round 8: pip never reads a model-written file and never sees an
ambiguous argument. Only strict PyPI names reach pip, as separate argv items,
with fixed isolation options, a fresh empty cwd outside the build and the
scrubbed env. No real process runs."""
import os
from unittest import mock

import support  # noqa: F401
from local_ai.agents.shared import guard, tools

from test_tools_guard import SandboxCase, _completed

FIXED_INSTALL = ["--isolated", "--no-input", "--disable-pip-version-check",
                 "--index-url", "https://pypi.org/simple", "--only-binary=:all:"]


class _PipCase(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.venv = self.make_venv()
        self.set_sandbox(yes_unsafe=True)
        self.calls = []

        def record(argv, **kw):
            cwd = kw["cwd"]
            self.calls.append({"argv": list(argv), "cwd": cwd, "env": dict(kw["env"]),
                               "entries": os.listdir(cwd), "isdir": os.path.isdir(cwd)})
            return _completed(argv)

        self.run_mock.side_effect = record

    def py(self) -> str:
        return str(self.venv / "bin" / "python")

    def req(self, data) -> None:
        p = self.root / "requirements.txt"
        p.write_bytes(data) if isinstance(data, bytes) else p.write_text(data, encoding="utf-8")

    def refused_without_run(self, command: str) -> None:
        self.run_mock.reset_mock()
        self.assertNotRun(command)


class CommandLineTests(_PipCase):
    def test_every_option_is_refused(self) -> None:
        for command in (
                "pip install -r other.txt", "pip install -c c.txt six", "pip install -e .",
                "pip install --editable=.", "pip install --requirement other.txt",
                "pip install --constraint=c.txt six", "pip install -i https://pypi.org/simple six",
                "pip install --index-url=https://pypi.org/simple six",
                "pip install --extra-index-url https://x six", "pip install -f /tmp six",
                "pip install --find-links=/tmp six", "pip install --trusted-host x six",
                "pip install -U six", "pip install --pre six", "pip install --user six",
                "pip install -q six", "pip install -- six", "pip install six --no-deps",
                "pip install -r=requirements.txt", "pip install -r requirements.txt six",
                "pip install -r ./requirements.txt", "pip install -r requirements.txt -r requirements.txt",
                "pip uninstall -y six", "pip list --format=json", "pip show -f six",
                "pip install --config-settings=a=b six", "pip install --isolated six"):
            with self.subTest(command=command):
                self.refused_without_run(command)
        self.assertIn("pip option", tools.run_command("pip install -U six"))

    def test_anything_but_a_strict_name_is_refused(self) -> None:
        for spec in ("pkg @ https://example.com/p-1.0.tar.gz", "pkg@https://x/a.whl",
                     "git+https://github.com/a/b.git", "./pkg", ".", "..", "pkg/sub",
                     "/abs", "file:///etc", "https://x/a.whl", "six; python_version>'3'",
                     "six;python_version>'3'", "six ==1.0", "six== 1.0", "six\\", "x:y",
                     "sïx", "six\t", "${X}", "six[a b]", "six[", "six==", "six==1,",
                     ".six", "_six", "six#x", "six==1 --hash=sha256:ab"):
            with self.subTest(spec=spec):
                self.refused_without_run(f"pip install {guard.shlex.quote(spec)}")

    def test_names_pass_as_separate_argv_items_with_fixed_options(self) -> None:
        tools.run_command("pip install 'flask[async]>=3,<4' requests==2.32.3 six")
        self.assertEqual(self.calls[-1]["argv"],
                         [self.py(), "-I", "-m", "pip", "install", *FIXED_INSTALL,
                          "flask[async]>=3,<4", "requests==2.32.3", "six"])
        for command in ("python3 -m pip install six", ".venv/bin/pip install six",
                        f"{self.venv}/bin/pip3 install six"):
            with self.subTest(command=command):
                tools.run_command(command)
                self.assertEqual(self.calls[-1]["argv"],
                                 [self.py(), "-I", "-m", "pip", "install", *FIXED_INSTALL, "six"])

    def test_other_subcommands(self) -> None:
        common = ["--isolated", "--no-input", "--disable-pip-version-check"]
        for command, tail in (("pip list", ["list", *common]), ("pip freeze", ["freeze", *common]),
                              ("pip check", ["check", *common]),
                              ("pip show six", ["show", *common, "six"]),
                              ("pip uninstall six", ["uninstall", *common, "--yes", "six"])):
            with self.subTest(command=command):
                tools.run_command(command)
                self.assertEqual(self.calls[-1]["argv"], [self.py(), "-I", "-m", "pip", *tail])
        for command in ("pip list six", "pip install", "pip show", "pip download six",
                        "pip config set global.index-url x", "pip"):
            with self.subTest(command=command):
                self.refused_without_run(command)

    def test_cwd_is_fresh_empty_outside_the_build_and_removed(self) -> None:
        (self.root / "https:" / "pypi.org" / "simple" / "six").mkdir(parents=True)
        (self.root / "pip").mkdir()
        (self.root / "pip" / "__init__.py").write_text("raise SystemExit('planted')\n")
        sub = self.root / "sub"
        sub.mkdir()
        for working_dir in (None, "sub"):
            with self.subTest(working_dir=working_dir):
                tools.run_command("pip install six", working_dir)
                call = self.calls[-1]
                self.assertTrue(call["isdir"])
                self.assertEqual(call["entries"], [])
                real = os.path.realpath(call["cwd"])
                self.assertFalse(guard.is_inside(guard.current(), real), real)
                self.assertFalse(os.path.exists(call["cwd"]))  # cleaned up
        self.assertNotEqual(self.calls[0]["cwd"], self.calls[1]["cwd"])

    def test_no_temp_dir_outside_the_build_refuses(self) -> None:
        with mock.patch.object(guard, "_empty_dir_outside", lambda sandbox, stack: None):
            self.assertTrue(tools.run_command("pip install six").startswith("Refused"))
        self.run_mock.assert_not_called()

    def test_temp_dir_inside_the_build_is_not_used(self) -> None:
        inside = self.root / "tmp"
        inside.mkdir()
        real = guard.tempfile.TemporaryDirectory

        def fake(prefix, dir=None):
            return real(prefix=prefix, dir=str(inside) if dir is None else dir)

        with mock.patch.object(guard.tempfile, "TemporaryDirectory", fake):
            tools.run_command("pip install six")
        self.assertFalse(self.calls[-1]["cwd"].startswith(str(self.root)), self.calls[-1]["cwd"])

    def test_env_has_no_pip_or_python_variables(self) -> None:
        with mock.patch.dict(os.environ, {"PIP_INDEX_URL": "file:///outside",
                                          "PIP_CONFIG_FILE": "/outside/pip.conf",
                                          "PIP_FIND_LINKS": "/outside", "PYTHONPATH": "/outside",
                                          "PYTHONSTARTUP": "/outside/x.py"}):
            tools.run_command("pip install six")
        env = self.calls[-1]["env"]
        # review9: the one PIP_ variable is the guard's own PIP_CONFIG_FILE=/dev/null.
        self.assertEqual([k for k in env if k.startswith(("PIP_", "PYTHON"))],
                         ["PIP_CONFIG_FILE"], env)
        self.assertEqual(env["PIP_CONFIG_FILE"], os.devnull)
        self.assertEqual(env["VIRTUAL_ENV"], str(self.venv))

    def test_still_needs_confirmation_without_yes_unsafe(self) -> None:
        self.set_sandbox(yes_unsafe=False)
        self.assertIn("confirmation", tools.run_command("pip install six"))
        self.run_mock.assert_not_called()  # no TTY: not run


class RequirementsFileTests(_PipCase):
    def test_guard_reads_the_file_and_passes_names(self) -> None:
        self.req("# top\n\nsix==1.16.0\n   \n  requests>=2,<3  \n    # via x\nflask[async]\n")
        for command in ("pip install -r requirements.txt", "pip install -rrequirements.txt",
                        "pip install --requirement requirements.txt",
                        "pip install --requirement=requirements.txt"):
            with self.subTest(command=command):
                tools.run_command(command)
                self.assertEqual(self.calls[-1]["argv"],
                                 [self.py(), "-I", "-m", "pip", "install", *FIXED_INSTALL,
                                  "six==1.16.0", "requests>=2,<3", "flask[async]"])
                self.assertNotIn("-r", self.calls[-1]["argv"])

    def test_leading_utf8_bom_is_accepted(self) -> None:
        self.req(b"\xef\xbb\xbfsix\n")
        tools.run_command("pip install -r requirements.txt")
        self.assertEqual(self.calls[-1]["argv"][-1], "six")

    def test_any_other_line_refuses_the_whole_file(self) -> None:
        for body in (b"-e .\n", b"-r other.txt\n", b"-c c.txt\n", b"--index-url https://x\n",
                     b"six --hash=sha256:ab\n", b"six==1 \\\n  --hash=sha256:ab\n",
                     b"six \\\n", b"pkg @ https://x/p.tar.gz\n", b"git+https://x/r.git\n",
                     b"./pkg\n", b".\n", b"../outside\n", b"/etc\n", b"file:///etc\n",
                     b"six; python_version>'3'\n", b"six  # trailing comment\n", b"six ==1\n",
                     b"${HOME}\n", b"six\r\nrequests\r\n", b"six\rrequests\n", b"six\x0c\n",
                     b"six\x0b\n", b"six\x00\n", b"\tsix\n", b"six\t\n", b"# caf\xc3\xa9\nsix\n",
                     b"s\xc3\xafx\n", b"\xff\xfes\x00i\x00x\x00\n\x00", b"\xfe\xff\x00s\x00i\x00x",
                     b"\xef\xbb\xbf\xef\xbb\xbfsix\n", b"\xc3\n", b"# x \\\n../outside\n",
                     b"# coding: utf-7\n# x+AAo--e /etc\nsix\n" + b"+AAo-", b"six\x1c\n",
                     b"\n# only comments\n", b""):
            with self.subTest(body=body):
                self.req(body)
                self.refused_without_run("pip install -r requirements.txt")

    def test_utf7_comment_is_just_a_comment(self) -> None:
        # pip never sees the file, so a coding line has no effect.
        self.req(b"six\x0c\n".replace(b"\x0c", b"") + b"# coding: utf-7\n# x+AAo--e /etc\n")
        tools.run_command("pip install -r requirements.txt")
        self.assertEqual(self.calls[-1]["argv"][-1], "six")

    def test_unsafe_files_are_refused(self) -> None:
        (self.outside / "r.txt").write_text("six\n")
        os.symlink(self.outside / "r.txt", self.root / "requirements.txt")
        self.refused_without_run("pip install -r requirements.txt")
        os.remove(self.root / "requirements.txt")
        os.link(self.outside / "r.txt", self.root / "requirements.txt")
        self.refused_without_run("pip install -r requirements.txt")
        os.remove(self.root / "requirements.txt")
        self.refused_without_run("pip install -r requirements.txt")  # missing
        (self.root / "requirements.txt").mkdir()
        self.refused_without_run("pip install -r requirements.txt")
        os.rmdir(self.root / "requirements.txt")
        self.req("six\n" * 300_000)
        self.refused_without_run("pip install -r requirements.txt")  # too large
        self.req("six\n" * 501)
        self.refused_without_run("pip install -r requirements.txt")  # too many names

    def test_reads_the_working_dirs_file(self) -> None:
        (self.root / "sub").mkdir()
        (self.root / "sub" / "requirements.txt").write_text("requests\n")
        self.req("six\n")
        tools.run_command("pip install -r requirements.txt", "sub")
        self.assertEqual(self.calls[-1]["argv"][-1], "requests")

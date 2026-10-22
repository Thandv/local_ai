"""Review round 9: two hardenings of the structural pip path.

1. pip never loads a config file: PIP_CONFIG_FILE=/dev/null is set in pip's
   env (pip skips every config file, including ROOT/.venv/pip.conf, when it
   is os.devnull; --isolated alone still loads the global and venv files).
2. A name ending in an archive/wheel extension is refused: pip treats
   `x.whl`, `x.tar.gz`, `foo==1.zip`, ... as a file relative to its cwd, not
   as a PyPI name (pip 26 is_archive_file, case-insensitive). `.egg` is
   refused too.

No real process runs."""
import os

import support  # noqa: F401
from local_ai.agents.shared import guard, tools

from test_guard_review8 import _PipCase

ARCHIVE_NAMES = (
    "a.whl", "a-1.0-py3-none-any.whl", "evil.zip", "a.tar", "a.tar.gz", "a.tgz",
    "a.tar.bz2", "a.tbz", "a.tar.xz", "a.txz", "a.tlz", "a.tar.lz", "a.tar.lzma",
    "a.egg", "A.WHL", "a.Tar.Gz", "foo==1.whl", "foo>=1.tar.gz", "foo==1.0+x.zip",
    "foo!=1.TGZ", "foo<2,>=1.egg",
    # pip strips trailing [extras] before its archive check: b.zip[x] -> ./b.zip
    "b.zip[x]", "c.tar.gz[a,b]", "a-1.0-py3-none-any.whl[x]", "E.TGZ[x]==1")


class ArchiveNameTests(_PipCase):
    def test_archive_names_are_refused_on_the_command_line(self) -> None:
        for name in ARCHIVE_NAMES:
            for sub in ("install", "uninstall", "show"):
                with self.subTest(name=name, sub=sub):
                    self.refused_without_run(f"pip {sub} {guard.shlex.quote(name)}")
        self.assertIn("archive", tools.run_command("pip install a.whl"))

    def test_archive_names_refuse_a_requirements_file(self) -> None:
        for name in ARCHIVE_NAMES:
            with self.subTest(name=name):
                self.req(f"six\n{name}\n")
                self.refused_without_run("pip install -r requirements.txt")

    def test_names_that_merely_contain_a_dot_still_install(self) -> None:
        for name in ("zope.interface", "backports.zoneinfo", "ruamel.yaml", "six==1.16.0",
                     "jaraco.text>=3.11", "tar", "whl", "py-zipfile", "tarfile-ext",
                     "a.gz", "eggs", "a.wheel", "foo[a.whl]", "foo[zip]==1"):
            with self.subTest(name=name):
                tools.run_command(f"pip install {guard.shlex.quote(name)}")
                self.assertEqual(self.calls[-1]["argv"][-1], name)


class PipConfigTests(_PipCase):
    def test_pip_config_file_is_devnull_for_every_pip_command(self) -> None:
        self.req("six\n")
        for command in ("pip install six", "pip install -r requirements.txt", "pip list",
                        "pip freeze", "pip check", "pip show six", "pip uninstall six",
                        "python3 -m pip install six", ".venv/bin/pip install six"):
            with self.subTest(command=command):
                tools.run_command(command)
                env = self.calls[-1]["env"]
                self.assertEqual(env.get("PIP_CONFIG_FILE"), os.devnull, env)
                self.assertEqual([k for k in env if k.startswith(("PIP_", "PYTHON"))],
                                 ["PIP_CONFIG_FILE"], env)

    def test_parent_pip_config_file_is_overridden(self) -> None:
        from unittest import mock
        with mock.patch.dict(os.environ, {"PIP_CONFIG_FILE": str(self.outside / "pip.conf")}):
            tools.run_command("pip install six")
        self.assertEqual(self.calls[-1]["env"]["PIP_CONFIG_FILE"], os.devnull)

    def test_other_commands_do_not_get_pip_config_file(self) -> None:
        (self.root / "t.py").write_text("")
        tools.run_command("python -m unittest t")
        env = self.run_mock.call_args.kwargs["env"]
        self.assertNotIn("PIP_CONFIG_FILE", env)


class RequirementsSpecialFileTests(_PipCase):
    def test_fifo_requirements_is_refused_without_blocking(self) -> None:
        os.mkfifo(self.root / "requirements.txt")
        self.refused_without_run("pip install -r requirements.txt")

    def test_symlinked_parent_dir_is_refused(self) -> None:
        (self.outside / "requirements.txt").write_text("six\n")
        os.symlink(self.outside, self.root / "link")
        self.refused_without_run_in("pip install -r requirements.txt", "link")

    def refused_without_run_in(self, command: str, working_dir: str) -> None:
        self.run_mock.reset_mock()
        self.assertNotRun(command, working_dir)

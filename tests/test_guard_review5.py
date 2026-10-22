"""Review round 5 (independent): probes of 965d57e. No real process runs."""
import os
from unittest import mock

import support  # noqa: F401
from local_ai.agents.shared import tools

from test_tools_guard import SandboxCase


class ReqFileRound5(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.make_venv()
        self.set_sandbox(yes_unsafe=True)
        (self.outside / "setup.py").write_text("from setuptools import setup\nsetup()\n")

    def req(self, data) -> None:
        p = self.root / "requirements.txt"
        if isinstance(data, bytes):
            p.write_bytes(data)
        else:
            p.write_text(data, encoding="utf-8")

    def test_a_double_backslash_continuation(self) -> None:
        # pip: line.strip("\\") drops BOTH backslashes: "." + "./../outside"
        # == "../outside"; the checker keeps one: ".\./../outside" -> inside.
        self.req(".\\\\\n./../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_b_env_var_split_by_double_backslash(self) -> None:
        # pip joins to "${OUT}/.." then expands it; the checker sees "$\{OUT}".
        with mock.patch.dict(os.environ, {"OUT": str(self.outside)}):
            self.req("$\\\\\n{OUT}\n")
            self.assertNotRun("pip install -r requirements.txt")

    def test_c_file_url_semicolon_in_path(self) -> None:
        # pip: path "./ok;/../../outside" (no '; ' marker split for URLs);
        # the checker stops the URL at ';' and checks "./ok".
        (self.root / "ok;").mkdir()
        self.req("file:./ok;/../../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_d_editable_path_with_semicolon(self) -> None:
        # pip's -e takes the whole value as a path: isdir("./ok;/../../outside").
        (self.root / "ok;").mkdir()
        self.req("-e ./ok;/../../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_e_pep263_coding_line(self) -> None:
        # pip decodes per a '# coding:' line in the first two lines; utf-7
        # "+AC4ALgAv-" is "../".
        self.req(b"# -*- coding: utf-7 -*-\n+AC4ALgAv-outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_f_utf16_bom(self) -> None:
        self.req("﻿../outside\n".encode("utf-16-le"))
        self.assertNotRun("pip install -r requirements.txt")

    # regressions for ordinary files
    def test_h_file_url_space_in_path(self) -> None:
        # pip: Link("file:./ok /../../outside") -> <outside>; the checker stops at ' '.
        (self.root / "ok ").mkdir()
        self.req("file:./ok /../../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_i_editable_file_url_semicolon(self) -> None:
        (self.root / "ok;").mkdir()
        self.req("-e file:./ok;/../../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_j_uppercase_file_scheme(self) -> None:
        # pip's is_url lowercases the scheme: FILE:../outside is <outside>.
        self.req("FILE:../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_k_editable_vcs_file_url(self) -> None:
        self.req(f"-e git+file://{self.outside}#egg=x\n")
        self.assertNotRun("pip install -r requirements.txt")

    # pins for surviving mutants
    def test_l_bare_dotdot(self) -> None:
        self.req("..\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_m_localhost_inside_and_config_settings_and_lowercase_var(self) -> None:
        # Narrowed after round 5: file: URLs (even to a path inside the build)
        # and any ${...} on a line are refused outright rather than parsed.
        (self.root / "pkg").mkdir()
        for text in (f"file://localhost{self.root}/pkg\n", "x ; extra == '${lower}'\n"):
            with self.subTest(text=text):
                self.req(text)
                self.assertTrue(tools.run_command("pip install -r requirements.txt")
                                .startswith("Refused:"))


class PipCommandLineGrammarTests(SandboxCase):
    """The same allowlist grammar applies to specs on the pip command line."""

    def test_unusual_specs_are_refused_even_unsafe(self) -> None:
        self.make_venv()
        self.set_sandbox(yes_unsafe=True)
        for command in ("pip install FILE:../outside", "pip install file:./x",
                        "pip install -e git+file:///tmp/r#egg=x",
                        "pip install 'x @ file:///tmp/x'", "pip install 'x @ ftp://h/x'",
                        "pip install './ok;/../../outside'", "pip install -e './a b'",
                        "pip install ../outside"):
            with self.subTest(command=command):
                self.assertNotRun(command)

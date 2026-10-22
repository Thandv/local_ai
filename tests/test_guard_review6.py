"""Review round 6 (independent): probes of 253ad2a. No real process runs."""
import os

import support  # noqa: F401
from local_ai.agents.shared import tools

from test_tools_guard import SandboxCase


class Round6(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.make_venv()
        self.set_sandbox(yes_unsafe=True)
        (self.outside / "setup.py").write_text("from setuptools import setup\nsetup()\n")

    def req(self, text: str) -> None:
        (self.root / "requirements.txt").write_text(text, encoding="utf-8")

    def refused(self, command: str) -> bool:
        return tools.run_command(command).startswith("Refused")

    # --- bypasses -------------------------------------------------------
    def test_a_comment_line_ending_in_backslash(self) -> None:
        # pip's join_lines: a line matching COMMENT_RE is never a continuation,
        # so line 2 stands alone; the checker glues it onto the comment.
        self.req("# pinned below \\\n../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_b_indented_comment_mid_continuation(self) -> None:
        self.req("six \\\n    # note \\\n../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_c_subdirectory_absolute(self) -> None:
        # Link.subdirectory_fragment is joined onto the unpacked dir unchecked:
        # os.path.join(tmp, "/abs") == "/abs" -> builds <outside>/setup.py.
        self.assertNotRun(f"pip install 'pkg @ https://example.com/a.zip#subdirectory={self.outside}'")
        self.req(f"git+https://github.com/a/b.git#egg=b&subdirectory={self.outside}\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_d_index_url_file_scheme(self) -> None:
        self.req(f"--index-url file://{self.outside}\nsix\n")
        self.assertNotRun("pip install -r requirements.txt")
        self.assertNotRun(f"pip install -i file://{self.outside} six")

class Round6Pins(SandboxCase):
    """Pass on 253ad2a; each kills a surviving mutant of the new lines."""

    def setUp(self) -> None:
        super().setUp()
        self.make_venv()
        self.set_sandbox(yes_unsafe=True)

    def req(self, data) -> None:
        p = self.root / "requirements.txt"
        p.write_bytes(data) if isinstance(data, bytes) else p.write_text(data, encoding="utf-8")

    def test_m4_utf7_version_that_is_a_path(self) -> None:
        # utf-7 "a==1+AC8ALgAuAC8ALgAuAC8-outside" is "a==1/../../outside": a
        # valid name spec as UTF-8, an escaping path for pip once ./a==1 exists.
        (self.root / "a==1").mkdir()
        self.req(b"# coding: utf-7\na==1+AC8ALgAuAC8ALgAuAC8-outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_m6_editable_text_after_extras(self) -> None:
        # pip's _strip_extras needs a trailing ']', so the path is the whole
        # "./x[a]/../../outside".
        (self.root / "x[a]").mkdir()
        self.assertNotRun("pip install -e './x[a]/../../outside'")

    def test_m8_trailing_continuation(self) -> None:
        self.req("../outside \\")
        self.assertNotRun("pip install -r requirements.txt")

    def test_m11_vcs_http(self) -> None:
        self.assertNotRun("pip install git+http://example.com/r.git")

    def test_m12_oversized_file(self) -> None:
        self.req("six\n" * 260_000 + "../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_m2_var_in_option_value(self) -> None:
        self.req("--index-url ${IDX}\nsix\n")
        self.assertNotRun("pip install -r requirements.txt")


class Round6Allowed(SandboxCase):
    def test_relative_subdirectory_and_https_index_still_run(self) -> None:
        self.make_venv()
        self.set_sandbox(yes_unsafe=True)
        for command in ("pip install 'pkg @ https://example.com/a.zip#subdirectory=src/pkg'",
                        "pip install -i https://pypi.org/simple six"):
            with self.subTest(command=command):
                self.assertFalse(tools.run_command(command).startswith("Refused"))
        for command in ("pip install 'pkg @ https://example.com/a.zip#subdirectory=a/../..'",
                        "pip install -i http://pypi.org/simple six"):
            with self.subTest(command=command):
                self.assertTrue(tools.run_command(command).startswith("Refused"))

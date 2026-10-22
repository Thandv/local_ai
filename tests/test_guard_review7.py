"""Review round 7 (independent): probes of 9c3ef51. No real process runs.

Each bypass was checked offline against pip 26.1.1's own parser
(parse_req_from_line, parse_requirements, the install option parser and
PackageFinder.find_all_candidates with the network blocked)."""
import os

import support  # noqa: F401
from local_ai.agents.shared import tools

from test_tools_guard import SandboxCase, _completed


class Round7(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.make_venv()
        self.set_sandbox(yes_unsafe=True)
        (self.outside / "setup.py").write_text("from setuptools import setup\nsetup()\n")

    def req(self, data, name: str = "requirements.txt") -> None:
        p = self.root / name
        p.write_bytes(data) if isinstance(data, bytes) else p.write_text(data, encoding="utf-8")

    # --- bypasses -------------------------------------------------------
    def test_a_short_option_with_equals(self) -> None:
        # optparse gives `-r=req.txt` the value "=req.txt": pip reads a file the
        # checker never opened (the checker read req.txt).
        self.req("six\n")
        self.req(f"-e {self.outside}\n", name="=requirements.txt")
        self.assertNotRun("pip install -r=requirements.txt")
        self.assertNotRun("pip install -c=requirements.txt six")
        self.req("-r=requirements.txt\n", name="outer.txt")
        self.assertNotRun("pip install -r outer.txt")

    def test_b_name_at_url_with_dot_dot_is_a_local_path_to_pip(self) -> None:
        # Not is_url(), so pip tries normpath(abspath("pkg @ https://h/../../../outside"))
        # first: a lexical path that leaves the cwd; an installable dir there is built.
        spec = "pkg @ https://h/../../../outside"
        self.assertNotRun(f"pip install '{spec}'")
        self.req(spec + "\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_c_https_dir_in_cwd_is_never_pips_cwd(self) -> None:
        # build_source() tries os.path.exists("https://pypi.org/simple/six/")
        # first; a model-written tree there would become the (default) index.
        # Since review8 pip runs in a fresh empty dir outside the build.
        page = self.root / "https:" / "pypi.org" / "simple" / "six"
        page.mkdir(parents=True)
        (page / "index.html").write_text(
            f'<a href="file://{self.outside}/six-9.0.tar.gz">six-9.0.tar.gz</a>\n')
        seen = []
        self.run_mock.side_effect = lambda argv, **kw: seen.append(
            (kw["cwd"], os.listdir(kw["cwd"]))) or _completed(argv)
        self.assertFalse(tools.run_command("pip install six").startswith("Refused"))
        cwd, entries = seen[0]
        self.assertFalse(cwd.startswith(str(self.root)), cwd)
        self.assertEqual(entries, [])
        self.run_mock.reset_mock()
        self.assertNotRun("pip install -i https://pypi.org/simple six")

    def test_d_coding_line_hidden_from_the_checker(self) -> None:
        # pip looks for PEP 263 in data.split(b"\n")[:2]; the checker looked at
        # splitlines()[:2], which a \x0c or a lone \r shifts. As utf-7 the
        # comment is "# x\n-e <outside>".
        for head in (b"six\x0c\n", b"\r\r\n"):
            with self.subTest(head=head):
                self.req(head + b"# coding: utf-7\n# x+AAo--e " + bytes(self.outside) + b"\n")
                self.assertNotRun("pip install -r requirements.txt")

    def test_e_name_at_plain_http(self) -> None:
        self.assertNotRun("pip install 'pkg @ http://example.com/pkg-1.0.tar.gz'")
        self.req("pkg @ http://example.com/pkg-1.0.tar.gz\n")
        self.assertNotRun("pip install -r requirements.txt")


class Round7Allowed(SandboxCase):
    def test_only_plain_names_still_run(self) -> None:
        self.make_venv()
        self.set_sandbox(yes_unsafe=True)
        (self.root / "requirements.txt").write_text("# pinned\nsix==1.16\n", encoding="utf-8")
        for command in ("pip install -r requirements.txt",
                        "pip install --requirement=requirements.txt", "pip install six"):
            with self.subTest(command=command):
                self.assertFalse(tools.run_command(command).startswith("Refused"))
        (self.root / "requirements.txt").write_text(
            "six==1.16 \\\n    --hash=sha256:abc\n", encoding="utf-8")
        self.run_mock.reset_mock()
        self.assertNotRun("pip install -r requirements.txt")
        self.assertNotRun("pip install 'pkg @ https://example.com/pkg-1.0.tar.gz'")

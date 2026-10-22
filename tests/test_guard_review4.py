"""Review round 4 (independent): failing tests for 215a123. No real process runs."""

import contextlib
import io
import json
from pathlib import Path
from unittest import mock

import support  # noqa: F401
from local_ai import orchestrator
from local_ai.agents import tester
from local_ai.agents.shared import tools
from local_ai.agents.shared.memory import Project, Verdict

from test_tools_guard import SandboxCase
from test_guard_review2 import ResumeForgeryTests as _Base, STAGES


class ReqFileRound4(SandboxCase):
    def setUp(self) -> None:
        super().setUp()
        self.make_venv()
        self.set_sandbox(yes_unsafe=True)
        (self.outside / "setup.py").write_text("from setuptools import setup\nsetup()\n")

    def req(self, text: str) -> None:
        (self.root / "requirements.txt").write_text(text, encoding="utf-8")

    def test_a_continuation_is_joined_without_a_space_like_pip(self) -> None:
        # pip: "." + "./../outside" == "../outside" (outside); the checker sees
        # ". ./../outside", which folds to <root>/outside.
        self.req(".\\\n./../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_b_scheme_separator_in_a_marker_skips_the_path_check(self) -> None:
        # pip installs ../outside (marker true); '://' anywhere returns None.
        self.req('../outside ; sys_platform != "x://"\n')
        self.assertNotRun("pip install -r requirements.txt")

    def test_c_file_url_with_localhost_authority(self) -> None:
        # pip's url_to_path drops a 'localhost' netloc: this is <outside>.
        self.req(f"file://localhost{self.outside}\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_d_percent_encoded_file_url(self) -> None:
        # pip unquotes the path: file:%2E%2E/outside is ../outside.
        self.req("file:%2E%2E/outside\n")
        self.assertNotRun("pip install -r requirements.txt")

class EchoRound4(_Base):
    def test_f_resume_does_not_echo_build_log_text_raw(self) -> None:
        # BUILD_LOG.json is model-writable (the premise of the resume fix).
        (self.out / "BUILD_LOG.json").write_text(json.dumps({
            "instruction": "x\x1b[2J\r[ok] fake", "files": ["y\x1b]0;t\x07"],
            "completed_stages": STAGES + ["z\x1b[31m"]}), encoding="utf-8")
        buf = io.StringIO()
        with contextlib.ExitStack() as stack:
            for name in STAGES:
                stack.enter_context(mock.patch.object(getattr(orchestrator, name), "run",
                                                      lambda p: p))
            stack.enter_context(mock.patch.object(orchestrator.tester, "run_test_gate",
                                                  lambda p: {"status": "fail", "runner": None,
                                                             "returncode": None, "reason": "",
                                                             "output_tail": ""}))
            stack.enter_context(contextlib.redirect_stdout(buf))
            orchestrator.run(instruction="", resume_dir=self.out)
        self.assertNotIn("\x1b", buf.getvalue())

    def test_g_summary_does_not_echo_model_paths_raw(self) -> None:
        # tracked_write records the model's path even when the write is refused.
        p = Project(instruction="i", output_dir=self.out)
        p.files_written.append("a\x1b[2Jb.py")
        p.verdict = Verdict.INCOMPLETE
        self.assertNotIn("\x1b", p.summary())


class ResumeGateRaises(_Base):
    def test_h_a_raising_gate_on_resume_is_a_recorded_failure(self) -> None:
        (self.out / "BUILD_LOG.json").write_text(json.dumps({
            "instruction": "i", "completed_stages": STAGES}), encoding="utf-8")
        with mock.patch.object(orchestrator.tester, "run_test_gate",
                               side_effect=OSError("invented")):
            project = self._run({}, instruction="", resume_dir=self.out)
        self.assertEqual(project.verdict, Verdict.FAILED)
        self.assertEqual(json.loads((self.out / "BUILD_LOG.json").read_text())["verdict"],
                         "FAILED")


class Round4MutationPins(SandboxCase):
    """Pass on 215a123; each kills a surviving mutant of its new lines."""

    def setUp(self) -> None:
        super().setUp()
        self.make_venv()
        self.set_sandbox(yes_unsafe=True)

    def req(self, text: str, name: str = "requirements.txt") -> None:
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def test_nested_file_editable_path_is_from_the_cwd(self) -> None:
        self.req("-e ../outside\n", "a/b/inner.txt")  # from a/ it would be <root>/outside
        self.req("-r b/inner.txt\n", "a/req.txt")
        self.assertNotRun("pip install -r a/req.txt")

    def test_bare_path_with_an_inner_slash(self) -> None:
        (self.root / "x").mkdir()
        self.req("x/../../outside\n")
        self.assertNotRun("pip install -r requirements.txt")

    def test_editable_symlink_name_without_a_slash(self) -> None:
        import os
        os.symlink(self.outside, self.root / "link")
        self.req("-e link\n")
        self.assertNotRun("pip install -r requirements.txt")

class EveryStageEchoTests(support.GuardedTestCase):
    """Every stage's tool-call echo escapes the model's tool name, path and command."""

    def test_no_stage_echoes_control_characters(self) -> None:
        import importlib
        import tempfile
        from local_ai.agents.shared.memory import Project

        def loop(system, user, tools, handlers, on_tool_call=None):
            for name, args in (("x\x1b[8m", {}), ("write_file", {"path": "a\x1b[2J"}),
                               ("run_command", {"command": "ls\r\x1b[8m"}),
                               ("peek_readme", {"repo_url": "a\x1b/b"}),
                               ("clone_shallow", {"repo_url": "a\x1b/b"}),
                               ("search_github", {"query": "q\x1b"})):
                on_tool_call(name, args)
            return "done"

        for stage in STAGES:
            module = importlib.import_module(f"local_ai.agents.{stage}")
            if not hasattr(module, "run_agent_loop"):
                continue
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as tmp:
                buf = io.StringIO()
                project = Project(instruction="x", output_dir=Path(tmp))
                with mock.patch.object(module, "run_agent_loop", loop), \
                        mock.patch.object(tester, "run_test_gate",
                                          lambda p: {"status": "fail", "runner": None,
                                                     "returncode": None, "reason": "r",
                                                     "output_tail": ""}), \
                        mock.patch.object(module, "grep_repos", lambda kw: "", create=True), \
                        contextlib.redirect_stdout(buf):
                    module.run(project)
                self.assertNotIn("\x1b", buf.getvalue())
                self.assertNotIn("\r", buf.getvalue())

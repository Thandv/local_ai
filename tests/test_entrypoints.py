"""The console-script targets in pyproject.toml import and start cleanly."""

import contextlib
import io
import tomllib
from pathlib import Path
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai import build_cli, indexer, vibe

PYPROJECT = Path(__file__).resolve().parent.parent / "pyproject.toml"


class EntryPointTests(support.GuardedTestCase):
    def test_console_scripts_point_at_callable_mains(self) -> None:
        with PYPROJECT.open("rb") as f:
            scripts = tomllib.load(f)["project"]["scripts"]
        self.assertEqual(scripts, {
            "build-app": "local_ai.build_cli:main",
            "vibe": "local_ai.vibe:main",
            "ai-index": "local_ai.indexer:main",
        })
        for module in (build_cli, vibe, indexer):
            self.assertTrue(callable(module.main))

    def test_build_app_help_exits_zero_without_running_anything(self) -> None:
        out = io.StringIO()
        with mock.patch("sys.argv", ["build-app", "--help"]), \
                mock.patch.object(build_cli.orchestrator, "run") as run, \
                mock.patch.object(build_cli.orchestrator, "fix") as fix, \
                contextlib.redirect_stdout(out):
            with self.assertRaises(SystemExit) as cm:
                build_cli.main()
        self.assertEqual(cm.exception.code, 0)
        self.assertIn("usage: build-app", out.getvalue())
        run.assert_not_called()
        fix.assert_not_called()

    def test_ai_index_main_passes_limit_from_argv(self) -> None:
        with mock.patch("sys.argv", ["ai-index", "3"]), \
                mock.patch.object(indexer, "build_index") as build_index:
            indexer.main()
        build_index.assert_called_once_with(limit=3)

    def test_ai_index_main_defaults_to_no_limit(self) -> None:
        with mock.patch("sys.argv", ["ai-index"]), \
                mock.patch.object(indexer, "build_index") as build_index:
            indexer.main()
        build_index.assert_called_once_with(limit=0)

"""pyproject.toml packaging config, and the console-script targets it names."""

import contextlib
import importlib
import io
import tomllib
from pathlib import Path
from unittest import mock

import support  # noqa: F401  (installs the guard before local_ai imports)
from local_ai import build_cli, indexer, vibe

REPO_ROOT = Path(__file__).resolve().parent.parent

with (REPO_ROOT / "pyproject.toml").open("rb") as _f:
    PYPROJECT = tomllib.load(_f)


class PackagingConfigTests(support.GuardedTestCase):
    def test_console_scripts_resolve_to_the_real_mains(self) -> None:
        scripts = PYPROJECT["project"]["scripts"]
        expected = {"build-app": build_cli.main, "vibe": vibe.main,
                    "ai-index": indexer.main}
        self.assertEqual(set(scripts), set(expected))
        for name, target in scripts.items():
            module_name, attr = target.split(":")
            self.assertIs(getattr(importlib.import_module(module_name), attr),
                          expected[name])

    def test_packages_match_every_package_dir_in_the_repo(self) -> None:
        on_disk = sorted(
            ".".join(("local_ai",) + p.parent.relative_to(REPO_ROOT).parts)
            for p in REPO_ROOT.rglob("__init__.py")
            if not {".git", ".venv", "build"} & set(p.relative_to(REPO_ROOT).parts)
        )
        cfg = PYPROJECT["tool"]["setuptools"]
        self.assertEqual(sorted(cfg["packages"]), on_disk)
        self.assertEqual(on_disk, ["local_ai", "local_ai.agents", "local_ai.agents.shared"])
        self.assertEqual(cfg["package-dir"], {"local_ai": "."})

    def test_repos_json_ships_as_package_data(self) -> None:
        self.assertEqual(PYPROJECT["tool"]["setuptools"]["package-data"],
                         {"local_ai": ["repos.json"]})
        self.assertTrue((REPO_ROOT / "repos.json").is_file())

    def test_no_runtime_deps_and_anthropic_is_an_extra(self) -> None:
        self.assertEqual(PYPROJECT["project"]["dependencies"], [])
        self.assertEqual(PYPROJECT["project"]["optional-dependencies"],
                         {"claude": ["anthropic"]})


class MainTests(support.GuardedTestCase):
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

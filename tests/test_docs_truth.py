"""README.md and STATUS.md name only commands, files and modules that exist.

Checked: relative links and #anchors; backticked paths under a repo dir
(agents/, tests/, scripts/, .github/); backticked `local_ai.x[:attr]`
references and `-m local_ai.x`; and every command in a ```bash block (its
program must be one of this package's console scripts, a .venv/bin program
the install creates, a repo file, or a listed system tool). A backticked
`word --option` span must start with a known command too.
"""

import importlib
import re
import shlex
import tomllib
from pathlib import Path

import support  # noqa: F401  (installs the guard before local_ai imports)

REPO_ROOT = Path(__file__).resolve().parent.parent
DOCS = ("README.md", "STATUS.md")
REPO_DIRS = ("agents", "tests", "scripts", ".github")

with (REPO_ROOT / "pyproject.toml").open("rb") as _f:
    SCRIPTS = set(tomllib.load(_f)["project"]["scripts"])

# Programs a doc may tell the reader to run that are not part of this repo.
SYSTEM_TOOLS = {"bash", "cd", "cat", "chmod", "echo", "git", "ln", "ls", "mkdir", "printf", "rm", "ruff",
                "python3", "python", "pip", "brew", "ollama", "test", "uv"}
# Programs a doc may mention (`npm test`, `node --test`) as what a build runs.
MENTIONED_TOOLS = SYSTEM_TOOLS | {"pytest", "npm", "node", "jest", "vitest", "mocha"}
VENV_PROGRAMS = SCRIPTS | {"python", "python3", "pip", "pytest"}  # pytest: a build's venv


def _slug(heading: str) -> str:
    return re.sub(r"[^\w\- ]", "", heading.strip().lower()).replace(" ", "-")


def _module_ok(ref: str) -> bool:
    name, _, attr = ref.partition(":")
    try:
        module = importlib.import_module(name)
    except ImportError:
        return False
    return not attr or hasattr(module, attr)


def _path_problem(token: str, root: Path) -> str:
    """A problem with a path-like token, or "". Text with spaces or quotes
    inside (a printf format, say) is not a path."""
    token = token.strip("\"'").replace("$PWD/", "").replace("$HOME/", "~/")
    if re.search(r"[\s\"']", token):
        return ""
    token = re.sub(r":\d+$", "", token)  # file.py:129
    if "/.venv/bin/" in f"/{token}" or token.startswith(".venv/bin/"):
        name = token.rsplit("/", 1)[1]
        return "" if name in VENV_PROGRAMS else f"{token}: no such program in the venv"
    if "/.local/bin/" in token:
        name = token.rsplit("/", 1)[1]
        return "" if name in SCRIPTS | {"ai"} else f"{token}: not a command this repo installs"
    first = token.split("/", 1)[0]
    exists = any(root.glob(token)) if "*" in token else (root / token).exists()
    if first in REPO_DIRS and not exists:
        return f"{token}: no such file in the repo"
    return ""


def _command_problems(line: str, root: Path) -> list[str]:
    line = re.sub(r"\s+#.*$", "", line).strip()
    if not line or line.startswith("#"):
        return []
    try:
        words = shlex.split(line)
    except ValueError as e:
        return [f"{line!r}: cannot parse ({e})"]
    while words and re.fullmatch(r"[A-Z_][A-Z0-9_]*=.*", words[0]):
        words = words[1:]  # VAR=value prefixes
    if not words:
        return []
    program, problems = words[0], []
    if "/" in program:
        problems.append(_path_problem(program, root))
    elif program not in SCRIPTS | SYSTEM_TOOLS:
        problems.append(f"{program!r}: not a console script or a known tool")
    for i, word in enumerate(words[1:], 1):
        if words[i - 1] == "-m" and word.startswith("local_ai") and not _module_ok(word):
            problems.append(f"-m {word}: no such module")
        elif "/" in word and not word.startswith(("-", "http")):
            problems.append(_path_problem(word, root))
    return [p for p in problems if p]


def doc_problems(text: str, root: Path = REPO_ROOT) -> list[str]:
    """Every command, file or module `text` names that does not exist."""
    problems: list[str] = []
    anchors = {_slug(m) for m in re.findall(r"^#+\s+(.+)$", text, re.MULTILINE)}
    for target in re.findall(r"\]\(([^)\s]+)\)", text):
        if target.startswith(("http://", "https://")):
            continue
        path, _, anchor = target.partition("#")
        if path and not (root / path).exists():
            problems.append(f"link {target}: no such file")
        elif not path and anchor not in anchors:
            problems.append(f"link {target}: no such heading")
    for block in re.findall(r"```bash\n(.*?)```", text, re.DOTALL):
        for line in block.splitlines():
            problems += _command_problems(line, root)
    prose = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    for span in re.findall(r"`([^`\n]+)`", prose):
        ref = re.fullmatch(r"(local_ai(?:\.\w+)+(?::\w+)?)", span)
        if ref and not _module_ok(ref.group(1)):
            problems.append(f"`{span}`: no such module or attribute")
        for word in span.split():
            if "/" in word and not word.startswith(("-", "http")):
                problem = _path_problem(word, root)
                if problem:
                    problems.append(problem)
        lead = re.match(r"([a-z][\w.-]*) --?[a-z]", span)
        if lead and lead.group(1) not in SCRIPTS | MENTIONED_TOOLS:
            problems.append(f"`{span}`: {lead.group(1)!r} is not a known command")
    return problems


class DocsTruthTests(support.GuardedTestCase):
    def test_readme_and_status_name_only_things_that_exist(self) -> None:
        for name in DOCS:
            with self.subTest(doc=name):
                path = REPO_ROOT / name
                self.assertTrue(path.is_file(), f"{name} is missing")
                self.assertEqual(doc_problems(path.read_text(encoding="utf-8")), [])

    def test_the_checker_catches_each_kind_of_wrong_name(self) -> None:
        bad = {
            "a missing repo file": "Run `bash scripts/install.sh` first.",
            "a missing linked file": "See [the plan](docs/PLAN.md).",
            "a missing anchor": "See [Safety](#no-such-section).",
            "a missing module": "Entry point `local_ai.no_such_module:main`.",
            "a missing attribute": "Entry point `local_ai.build_cli:no_such_main`.",
            "an unknown command in a bash block": "```bash\nlocal-ai-doctor --fix\n```",
            "an unknown venv program": "```bash\n.venv/bin/ai-doctor --help\n```",
            "an unknown -m module": "```bash\npython3 -m local_ai.nothing\n```",
            "a missing script in a bash block": "```bash\nbash scripts/nothing.sh\n```",
            "an unknown linked command": "```bash\nln -sf \"$PWD/.venv/bin/ai-x\" ~/.local/bin/ai-x\n```",
            "an unknown command in prose": "Then run `build-apps --help`.",
            "a missing file with a line number": "See `agents/shared/nothing.py:129`.",
            "a glob that matches nothing": "See `tests/test_nothing*.py`.",
        }
        for what, text in bad.items():
            with self.subTest(what=what):
                self.assertNotEqual(doc_problems(text), [], text)

    def test_the_checker_accepts_what_exists(self) -> None:
        good = ("Run `bash scripts/test.sh -v`; see `agents/shared/guard.py`, "
                "`local_ai.vibe:main` and [Safety](#safety).\n## Safety\n"
                "```bash\n.venv/bin/build-app --help\npython3 -m local_ai.agent\n"
                "ln -sf \"$PWD/.venv/bin/vibe\" ~/.local/bin/vibe\n```\n")
        self.assertEqual(doc_problems(good), [])

# local_ai

A stdlib-only multi-agent "build an app" pipeline (`build-app`) and a `vibe`
coding REPL that run on a local Ollama model (or the Claude API).

**Status: prototype.** Read [Safety](#safety) before running it: the model can
run commands on this machine. A guard limits which ones and where, but an
allowlisted test runner still runs model-written code.

## What it is

- `build-app` runs a pipeline of agents over one instruction: Designer →
  Researcher → Architect → Migrator → Coder → Debugger → Tester → Auditor →
  DevOps → Reviewer → Packager. Each stage is a model with tools (read, write
  and list files, run commands, search code; the Researcher also searches
  GitHub and shallow-clones repos). Output goes to `~/builds/<name>/` unless
  you pass `--out`.
- `vibe` is an interactive REPL in the current directory. When the cloned
  reference repos (see `ai-index`) match words in your prompt, snippets from
  them are added to it.
- `python -m local_ai.agent` is a smaller REPL for one-off coding tasks (not
  installed as a command).

The code uses only the Python standard library. The optional `anthropic`
package is needed only for the Claude backend.

## Model

The default model is **`qwen2.5-coder:7b`** on a local Ollama at
`localhost:11434` (`DEFAULT_MODEL` in `agents/shared/llm.py`). Change it with
the `LOCAL_AI_MODEL` environment variable or `--model` / `-m`. If the
requested Ollama model is not pulled, the largest pulled tag of the same
family is used instead (or else the first pulled model), with a notice. A
model name starting with `claude` selects the Claude API, which needs the
`[claude]` extra and `ANTHROPIC_API_KEY`.

If Ollama cannot be reached after 3 attempts (2 s and 4 s apart), or the
Claude backend is missing its package or key, the model call raises
`LLMUnavailableError`. `build-app` then stops with the verdict `FAILED` (see
[Build verdict](#build-verdict)); `build-app`, `vibe` and `agent` exit with
status 1 and print the reason to stderr.

## Install

Packaging is defined in `pyproject.toml` (setuptools, no runtime
dependencies). From the repo root, inside a virtualenv:

```bash
pip install -e .            # Ollama backend only
pip install -e ".[claude]"  # also installs `anthropic` for the Claude API backend
```

The `[claude]` extra is optional and off by default. Either form installs
three commands:

| Command     | Entry point                | What it does                                                       |
|-------------|----------------------------|--------------------------------------------------------------------|
| `build-app` | `local_ai.build_cli:main`  | Multi-agent app builder (`build-app --help` for options)           |
| `vibe`      | `local_ai.vibe:main`       | Interactive vibe-coding REPL in the current directory              |
| `ai-index`  | `local_ai.indexer:main`    | Clones the reference repos in `repos.json` (`ai-index N` = first N) |

The "run: ai-index" hints printed by `vibe` and the Researcher's repo search
refer to this command.

## Safety

**This tool CAN run commands on this machine**, with your user's permissions.
Every file and command tool goes through one guard, `agents/shared/guard.py`
(the handlers are in `agents/shared/tools.py`):

- **Confinement.** Each run has one allowed root: the output dir for
  `build-app`, the current directory for `vibe` and `agent`. `write_file`,
  `read_file`, `list_files` and `search_code` refuse paths outside it,
  including `..`, absolute paths and symlinks (paths are opened one component
  at a time with `O_NOFOLLOW`, so a symlink swapped in later is refused too).
  `write_file` also refuses hard links and anything under `.git`, `.venv`,
  `node_modules`, `.npmrc`, `.envrc`, `.vscode` or `BUILD_LOG.json`
  (case-insensitively).
  `list_files` and `search_code` (a case-insensitive literal search) walk the
  root without entering symlinked dirs and read files through the same checks. With no root configured,
  every tool refuses.
- **No shell.** `run_command` splits the command with `shlex` and runs it with
  `shell=False`, in a working dir inside the root, with a scrubbed environment
  (API keys, `PYTHON*`, `NODE_OPTIONS`, `GIT_*` and `npm_config_*` are not
  passed on). Pipes, redirects, `&&`, `;`, `$(...)` and backticks are refused.
- **Allowlist.** These run without asking, when every path argument stays in
  the root: `python3 -m pytest`, `python3 -m unittest`, `python3 -m py_compile`,
  `pytest`, `python3 -m venv .venv`, `node <script>` and `node --test`, `ls`,
  `cat`, `git init|status|diff|add|commit -m` (always on `ROOT/.git` via
  `GIT_DIR`/`GIT_WORK_TREE`, with hooks, fsmonitor, external diff and pager
  turned off), and `npm test` / `npm run <name>` when that
  package.json script is one plain test-runner command (jest, vitest, mocha,
  ava, tap or `node <script>`) with no pre/post hook. Options that write or
  read elsewhere (`pytest --basetemp`, `git -c`, `git diff --output`,
  `node -e`, `python -c`, ...) are not on it. Every allowlisted pytest run also gets
  `--basetemp=ROOT/.pytest_tmp -p no:cacheprovider`, so `addopts` in a
  model-written `pytest.ini` cannot point those outside the root.
- **Confirmation.** Anything else is shown to you with a `y/N` prompt when
  stdin is a TTY, and refused when it is not. `--yes-unsafe` on the
  `build-app`, `vibe` or `agent` command line skips that prompt; the model
  cannot set it.
- **Installs.** pip never reads a file the model wrote and never gets an
  option or path from it. The model may run `pip install <names>`,
  `pip uninstall/show <names>`, `pip list/freeze/check`, or
  `pip install -r requirements.txt`; every other pip option is refused. The
  guard reads that one `requirements.txt` itself. It must be plain ASCII with
  `\n` line ends, and every line must be blank, a full-line `#` comment or a
  plain PyPI name (`name`, `name[extra]`, `name==1.2,<2`; no spaces, markers,
  URLs, paths, hashes or options). Any other line refuses the whole file.
  The names go to pip as separate arguments:
  `ROOT/.venv/bin/python -I -m pip install --isolated --no-input
  --disable-pip-version-check --index-url https://pypi.org/simple
  --only-binary=:all: <names>`, run in a fresh empty temp dir outside the
  build, with the scrubbed environment (no `PIP_*` or `PYTHON*` variables).
  Wheels only: a package that publishes only an sdist will not install.
  Hash-pinned, editable (`-e .`), URL and local-path installs are not
  available to the model; run those yourself.
  `npm install` runs only next
  to a package.json in the working dir; `-g`, `-C`, `--global`, `--prefix`
  and similar options are refused in any npm command. Both always need the confirmation (or
  `--yes-unsafe`), so a build with no TTY installs nothing.
- **Untrusted text.** `peek_readme`, `clone_shallow`, `search_github` and the
  repo snippets are wrapped as `<untrusted-data>` blocks, and the Researcher's
  brief is wrapped again when the Architect and Migrator receive it. The
  Architect has no `run_command`; the Migrator's goes through the guard.

**What the guard does not do.** An allowlisted test runner (pytest, unittest,
`npm test`, `node <script>`) runs code the model wrote, and that code can do
anything your user can, including reach the network or write outside the
root. The guard decides which program starts and where, not what it does
next. Text from the Researcher can also reach later stages through `PLAN.md`.
Run `build-app`, `vibe` and `agent` in a throwaway directory, VM or container,
ideally with the network off.

What is in place: every agent tool loop, in `build-app` and in both REPLs,
runs on one shared loop (`run_tool_loop` in `agents/shared/llm.py`). It stops
after 20 model rounds with `[agent] stopped after 20 rounds without a final
answer.`; the same tool call repeated 3 times in a row ends it early; and a
malformed tool call (invalid JSON, arguments that are not an object or do not
match the tool, or a non-string value for a string argument) goes back to the
model as `Error: invalid arguments for <tool>: ...` instead of crashing.

## Build verdict

At the end, `build-app` prints a verdict and writes it to `BUILD_LOG.json`
(`verdict`, plus `success`, `success_reason`, `stage_errors` and
`completed_stages`). `Project.compute_verdict()` in `agents/shared/memory.py`
decides it:

| Verdict      | Meaning |
|--------------|---------|
| `SUCCESS`    | `PLAN.md` exists; there are at least 2 source files with at least 30 lines between them (docs, `node_modules`, venvs and build dirs do not count); and the test gate passed. |
| `UNVERIFIED` | Every other gate passed, but the tests were not run: no TTY to confirm the test command, the person declined it, no build venv, or pytest/npm missing. |
| `INCOMPLETE` | The pipeline ran, but one of those gates did not pass (including tests that ran and failed, or no tests found). |
| `FAILED`     | A required stage (Architect, Coder or Tester) raised, or any stage hit `LLMUnavailableError`. The pipeline stops there and no later stage runs. |

Any other stage that raises prints a warning, is recorded in `stage_errors`,
and the pipeline continues. A build or `--resume` run of `build-app` exits
with status 1 when the verdict is `FAILED` and 0 otherwise. `build-app --resume DIR` skips the stages listed
in `completed_stages` (when the tester had finished, its test gate is run again rather than read
from `BUILD_LOG.json`, which model-run code could have rewritten); a stage that raised is not listed, so a resume retries
it.

**The test gate.** After the Tester agent writes and runs its tests, the
Tester stage (`run_test_gate` in `agents/tester.py`) runs the project's test
command itself, as a fixed argv through the guard, and reads its exit code;
the model's own report is not used. It records `test_results` (`status`
pass/fail/skipped, `runner`, `returncode`, `reason`, `output_tail`) in
`BUILD_LOG.json` and prints a `Tests` line in the summary.

- Python tests (`test*.py` or `*_test.py` files) run with the build
  venv's interpreter: `.venv/bin/python -m pytest` when `.venv/bin/pytest`
  exists, else `-m unittest discover`. Tests that import pytest, or a
  `conftest.py`/`pytest.ini`, with no pytest in `.venv` are skipped; so are
  tests with no `.venv`.
- Otherwise `npm test`, when package.json has a `test` script that is not
  npm's "no test specified" stub. It needs a confirmation unless the script is
  a plain test-runner command (see [Safety](#safety)).
- pytest passes only on exit 0 with at least one `N passed` (exit 5, nothing
  collected, fails). unittest passes only on exit 0 with `Ran N tests`,
  N >= 1, not all of them skipped. **`npm test` passes on
  exit 0 without counting tests**; only output that says no tests ran (for
  example `0 tests`) fails it.
- No test command found is a fail ("no tests found"). A failing gate gives
  the model up to 3 fix attempts, each followed by a new run of the gate.
- No TTY, a declined confirmation, a refusal or a missing venv/pytest/npm
  is `skipped`, which gives `UNVERIFIED`, never `SUCCESS`.

## Tests

```bash
bash scripts/test.sh
```

Runs the stdlib `unittest` suite in `tests/` (Python 3.11+). It needs no network, no Ollama
and no model: `tests/support.py` replaces `urllib.request.urlopen`,
`subprocess.run`, `subprocess.Popen` and `socket.create_connection` with
stubs that raise, and every test fails if any of them was called. Tests write
only to temporary directories. Extra arguments are passed to `unittest`
(e.g. `bash scripts/test.sh -v`, or `-k resume` to run matching tests).

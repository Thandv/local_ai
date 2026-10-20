# local_ai

A stdlib-only multi-agent "build an app" pipeline (`build-app`) and a `vibe`
coding REPL that run on a local Ollama model (or the Claude API).

> **Safety:** `build-app` and `vibe` let the model run shell commands
> (`shell=True`) and write files anywhere your user can. Run them only in a
> throwaway directory or VM until the command/write guard lands.

## Install

Packaging is defined in `pyproject.toml` (setuptools, no runtime
dependencies). From the repo root, inside a virtualenv:

```bash
pip install -e .            # Ollama backend only
pip install -e ".[claude]"  # also installs `anthropic` for the Claude API backend
```

This installs three commands:

| Command     | Entry point                | What it does                                                       |
|-------------|----------------------------|--------------------------------------------------------------------|
| `build-app` | `local_ai.build_cli:main`  | Multi-agent app builder (`build-app --help` for options)           |
| `vibe`      | `local_ai.vibe:main`       | Interactive vibe-coding REPL in the current directory              |
| `ai-index`  | `local_ai.indexer:main`    | Clones the reference repos in `repos.json` (`ai-index N` = first N) |

The "run: ai-index" hints printed by `vibe` and the Researcher's repo search
refer to this command.

## Tests

```bash
bash scripts/test.sh
```

Runs the stdlib `unittest` suite in `tests/` (Python 3.11+). It needs no network, no Ollama
and no model: `tests/support.py` replaces `urllib.request.urlopen`,
`subprocess.run`, `subprocess.Popen` and `socket.create_connection` with
stubs that raise, and every test fails if any of them was called. Tests write
only to temporary directories. Extra arguments are passed to `unittest`
(e.g. `bash scripts/test.sh -v`).

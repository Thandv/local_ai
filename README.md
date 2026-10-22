# local_ai

A stdlib-only multi-agent "build an app" pipeline (`build-app`) and a `vibe`
coding REPL that run on a local Ollama model (or the Claude API).

**Status: prototype.** Read [Safety](#safety) before running it: the model can
run shell commands on this machine, and the guard for that has not landed yet.

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

**This tool CAN run commands on this machine.** Today nothing guards that:

- The `run_command` tool (`agents/shared/tools.py`) runs whatever command the
  model sends through the shell (`shell=True`), with your user's permissions,
  no allowlist and no confirmation prompt. `build-app`'s Architect, Migrator,
  Coder, Debugger, Tester, Auditor and Packager stages have it, and so do
  `vibe` and `agent`.
- `write_file` and `read_file` accept any path your user can reach; they are
  not confined to the output directory.
- Dependencies are installed by whatever command the model chooses (e.g.
  `pip install -r requirements.txt`), not into a per-build venv.
- The Researcher's text, which includes READMEs and code fetched from GitHub,
  is pasted into the Architect's and Migrator's prompts as-is, not marked as
  untrusted data.

A guarded runner (a `shell=False` allowlist, confirmation on a TTY,
output-dir confinement, per-build venv installs and untrusted-data wrapping)
is planned but not merged. Until it is, run `build-app`, `vibe` and `agent`
only in a throwaway directory, VM or container with nothing you would mind
losing.

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
| `SUCCESS`    | `PLAN.md` exists; there are at least 2 source files with at least 30 lines between them (docs, `node_modules`, venvs and build dirs do not count); and the tester did not report a failure. |
| `INCOMPLETE` | The pipeline ran, but one of those gates did not pass. |
| `FAILED`     | A required stage (Architect, Coder or Tester) raised, or any stage hit `LLMUnavailableError`. The pipeline stops there and no later stage runs. |

Any other stage that raises prints a warning, is recorded in `stage_errors`,
and the pipeline continues. `build-app` exits with status 1 when the verdict
is `FAILED` and 0 otherwise. `build-app --resume DIR` skips the stages listed
in `completed_stages`; a stage that raised is not listed, so a resume retries
it.

**`SUCCESS` does not yet mean the tests passed.** The tester gate still reads
the Tester agent's own text and looks for pass/fail words (e.g. "passed",
"ok", "FAILED"), so a failing run can be reported as passing. (A Tester
loop that hits the 20-round limit counts as not passing.) A gate based on
the real exit code of the project's test command is planned; it needs the
guarded runner above.

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

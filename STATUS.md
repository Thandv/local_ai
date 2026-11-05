# local_ai status

Last updated 2026-11-05 (LAI-E3). Keep this file current with each landing.

**State: prototype.** The code and its guard are tested without a model.
Nobody has yet recorded a full `build-app` run, or real use of `vibe`, on a
real model.

## What works (tested without a model)

- **Test suite.** `bash scripts/test.sh` runs 322 hermetic tests
  (unittest). They need no network, Ollama or model. The real-pytest gate
  tests skip themselves when no local Python has pytest.
- **Guard.** Every file and command tool is confined to one root, runs
  with no shell and checks an allowlist. Commands off the allowlist need a
  y/N on a TTY (see README "Safety"). Probed by the 2026-11-02 assessor.
- **Test gate** (LAI-E3 t1). pytest is judged from a JUnit report the gate
  writes outside the build, with model-written pytest config ignored. A
  printed `N passed` line, `addopts = -s` / `--co` / `-k` / `-m`, a
  conftest that exits 0 and a zero-test run all fail, checked against a
  real pytest in `tests/test_gate_fooling.py`. `npm test` with no runner
  count is `unverified`, which gives `UNVERIFIED`, never `SUCCESS`.
- **Model calls** (LAI-E3 t2). These all become `LLMUnavailableError`
  after 3 attempts: an Ollama read timeout, a dropped stream, an HTTP error,
  or a stream that ends early. A call that trickles past `CALL_DEADLINE_S`
  (900 s) stops. In `vibe` and `agent`, a failed turn is reported and
  dropped, and the session goes on. Tested against a stub Ollama on
  127.0.0.1 in `tests/test_model_calls.py`.
- **Install steps** (LAI-E3 t3). The README "Install" steps were run in a
  fresh venv on this Mac (Python 3.15). `.venv/bin/build-app --help` exits 0.
  `vibe` and `python -m local_ai.agent` start and exit cleanly. Launchers
  symlinked into a scratch `PATH` dir also work from an empty environment.
- **Docs.** `tests/test_docs_truth.py` fails when this file or README.md
  names a command, repo file, module or link target that does not exist.
- **CI.** `.github/workflows/test.yml` runs the tests and ruff. Ruff checks
  syntax errors and undefined names only.

## Untested

- Any real model run. Nobody knows yet whether `qwen2.5-coder:7b` can
  drive the 11 stages to a `SUCCESS` build, or whether it obeys text
  injected into a README.
- The CI workflow has never run, because nothing was pushed. The suite has
  only run on macOS.
- The npm count parsing has only been tested on fixture strings. It has
  not run against real jest, vitest, mocha, ava or `node --test` output.
- The GitHub research tools against the network, the Claude backend, the
  interactive TTY confirmation, and `ai-index` cloning for real.

## Owner-held

- **LA-7:** one network-off `build-app` run on this Mac, with its verdict,
  stages and test gate result recorded here.
- **Real use:** one `vibe` session on a real model, recorded here.
- Whether to run the allowlisted test runners in a sandbox (container,
  VM or `sandbox-exec`), or keep the documented "use a throwaway VM" limit.
  The gate stops fooling by config and console output. It does not stop
  hostile code inside pytest: a `conftest.py` hook can still rewrite
  results.
- Whether to keep fixing this repo minimally or to archive it.

## Known open items (not in LAI-E3)

- `indexer.py` does not validate `repos.json`, passes the URL to
  `git clone` without `--`, and sets no timeout.
- `ruff check --select F,E9,B` reports 32 findings (unused imports,
  `raise ... from`). An unclosed file at `agents/shared/memory.py:129` gives
  a ResourceWarning.
- The `tests/test_guard_review*.py` files are named after review rounds,
  not behaviour.

## Repairing the launchers on this Mac (operator)

The `~/.local/bin` shims (`build-app`, `vibe`, `ai-index` and `ai`) and the
main checkout's `.venv` still point at the deleted `~/Claude/repos` path.
Pip has to reach PyPI once, to fetch setuptools for the build. Run:

```bash
cd ~/Projects/local_ai
python3 -m venv --clear .venv
.venv/bin/python -m pip install -e .
.venv/bin/build-app --help
ln -sf "$PWD/.venv/bin/build-app" ~/.local/bin/build-app
ln -sf "$PWD/.venv/bin/vibe" ~/.local/bin/vibe
ln -sf "$PWD/.venv/bin/ai-index" ~/.local/bin/ai-index
printf '#!/bin/sh\nexec "%s/.venv/bin/python" -m local_ai.agent "$@"\n' "$PWD" > ~/.local/bin/ai
chmod +x ~/.local/bin/ai
build-app --help
```

`~/.local/bin/ai` runs the `agent` REPL, which has no console script, so
it stays a small wrapper. If the checkout moves again, run the block again
from the new path.

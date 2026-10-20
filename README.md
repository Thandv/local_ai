# local_ai

A stdlib-only multi-agent "build an app" pipeline (`build-app`) and a `vibe`
coding REPL that run on a local Ollama model (or the Claude API).

> **Safety:** `build-app` and `vibe` let the model run shell commands
> (`shell=True`) and write files anywhere your user can. Run them only in a
> throwaway directory or VM until the command/write guard lands.

## Tests

```bash
bash scripts/test.sh
```

Runs the stdlib `unittest` suite in `tests/`. It needs no network, no Ollama
and no model: `tests/support.py` replaces `urllib.request.urlopen`,
`subprocess.run`, `subprocess.Popen` and `socket.create_connection` with
stubs that raise, and every test fails if any of them was called. Tests write
only to temporary directories. Extra arguments are passed to `unittest`
(e.g. `bash scripts/test.sh -v`).

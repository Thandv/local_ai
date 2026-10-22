"""
Multi-backend LLM client: Ollama (local) or Claude API (cloud).

Select via LOCAL_AI_MODEL env var or set_model() / --model flag:
  Ollama (default):  qwen2.5-coder:7b (if it is not pulled, a pulled tag of
                     the same family is used instead, with a notice)
  Claude:            a model name starting with "claude" (requires the
                     anthropic package and ANTHROPIC_API_KEY)

Both backends stream tokens to stdout and return the same message dict.
"""

import inspect
import json
import os
import time
import urllib.request
import urllib.error
from typing import Any, Callable, Mapping, Optional

ToolHandlers = Mapping[str, Callable[..., str]]


# run_tool_loop() returns text starting with this when it hits max_rounds.
ROUND_LIMIT_NOTICE = "[agent] stopped after"


class LLMUnavailableError(RuntimeError):
    """The model backend cannot be used (Ollama unreachable, the anthropic
    package missing, or no API key). Raised instead of exiting so callers can
    contain it; each CLI main() turns it into exit status 1."""

# ── Backend selection ─────────────────────────────────────────────────────────

OLLAMA_URL    = "http://localhost:11434/api/chat"
OLLAMA_TAGS   = "http://localhost:11434/api/tags"

# The installer pulls qwen2.5-coder:7b by default (works on 16 GB Macs).
# We match that here so a fresh install "just works" without --model.
DEFAULT_MODEL = os.environ.get("LOCAL_AI_MODEL", "qwen2.5-coder:7b")
_ACTIVE_MODEL = DEFAULT_MODEL
_MODEL_RESOLVED = False  # have we verified the active model exists in Ollama?


def set_model(model: str):
    """Override the active model at runtime (e.g. from --model CLI flag)."""
    global _ACTIVE_MODEL, _MODEL_RESOLVED
    _ACTIVE_MODEL   = model
    _MODEL_RESOLVED = False  # re-check on next chat() call


def get_model() -> str:
    return _ACTIVE_MODEL


def _is_claude(model: str) -> bool:
    return model.startswith("claude")


def _list_ollama_models() -> list[str]:
    """Return the list of model tags pulled in the local Ollama. Empty on error."""
    try:
        with urllib.request.urlopen(OLLAMA_TAGS, timeout=5) as resp:
            data = json.loads(resp.read())
        return [m.get("name", "") for m in data.get("models", []) if m.get("name")]
    except Exception:
        return []


def _resolve_active_model() -> None:
    """Verify the active model is pulled. If not, fall back to a same-family
    model that IS pulled, with a clear notice. Claude models skip this check."""
    global _ACTIVE_MODEL, _MODEL_RESOLVED
    if _MODEL_RESOLVED:
        return
    if _is_claude(_ACTIVE_MODEL):
        _MODEL_RESOLVED = True
        return

    available = _list_ollama_models()
    if not available:
        # Ollama unreachable — let chat() emit its own connection error
        _MODEL_RESOLVED = True
        return

    if _ACTIVE_MODEL in available:
        _MODEL_RESOLVED = True
        return

    # Fall back: prefer the same family (e.g. another pulled qwen2.5-coder:* tag)
    family = _ACTIVE_MODEL.split(":")[0]
    family_matches = [m for m in available if m.startswith(family + ":")]
    if family_matches:
        # Pick the largest tag that's pulled. Sort by size hint in tag string.
        family_matches.sort(key=lambda m: (
            -int("".join(ch for ch in m.split(":")[-1] if ch.isdigit()) or "0")
        ))
        chosen = family_matches[0]
    else:
        chosen = available[0]  # arbitrary — better than 404

    print(f"\n[LLM] Requested model {_ACTIVE_MODEL!r} is not pulled in Ollama. "
          f"Falling back to {chosen!r}. "
          f"To pull the requested model: `ollama pull {_ACTIVE_MODEL}`.\n")
    _ACTIVE_MODEL   = chosen
    _MODEL_RESOLVED = True


# ── Ollama backend ─────────────────────────────────────────────────────────────

def _ollama_chat(messages: list, tools: list, model: str, timeout: int) -> dict:
    """Streaming Ollama call. Prints tokens live; returns assembled message dict."""
    payload = json.dumps({
        "model":    model,
        "messages": messages,
        "tools":    tools or [],
        "stream":   True,
    }).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload,
        headers={"Content-Type": "application/json"}, method="POST",
    )
    content    = ""
    tool_calls = None
    retries    = 3
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                for raw in resp:
                    line = raw.decode("utf-8").strip()
                    if not line:
                        continue
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    msg_chunk = chunk.get("message", {})
                    delta = msg_chunk.get("content") or ""
                    if delta:
                        print(delta, end="", flush=True)
                        content += delta
                    if msg_chunk.get("tool_calls"):
                        tool_calls = msg_chunk["tool_calls"]
                    if chunk.get("done"):
                        if content:
                            print()
                        break
            break  # success
        except urllib.error.URLError as e:
            if attempt == retries:
                raise LLMUnavailableError(
                    f"[LLM] Cannot reach Ollama after {retries} attempts — "
                    f"is it running? (`brew services start ollama`)\n{e}"
                ) from e
            wait = 2 ** attempt
            print(f"\n[LLM] Connection error (attempt {attempt}/{retries}), retrying in {wait}s …")
            time.sleep(wait)

    return {"role": "assistant", "content": content, "tool_calls": tool_calls}


# ── Claude backend ─────────────────────────────────────────────────────────────

def _to_claude_messages(messages: list) -> tuple[str, list]:
    """Convert internal Ollama-format messages to Claude API format.

    System messages → top-level `system` string.
    Tool results    → user content blocks with tool_use_id.
    Assistant msgs  → content blocks (text + tool_use).
    """
    system_parts: list[str] = []
    out: list[dict] = []

    for m in messages:
        role    = m["role"]
        content = m.get("content", "")

        if role == "system":
            system_parts.append(content)

        elif role == "tool":
            block = {
                "type":        "tool_result",
                "tool_use_id": m.get("tool_use_id", "unknown"),
                "content":     content,
            }
            if out and out[-1]["role"] == "user" and isinstance(out[-1]["content"], list):
                out[-1]["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})

        elif role == "assistant":
            blocks: list[dict] = []
            if content:
                blocks.append({"type": "text", "text": content})
            for tc in (m.get("tool_calls") or []):
                blocks.append({
                    "type":  "tool_use",
                    "id":    tc.get("id", f"call_{len(blocks)}"),
                    "name":  tc["function"]["name"],
                    "input": tc["function"].get("arguments", {}),
                })
            out.append({"role": "assistant", "content": blocks or [{"type": "text", "text": ""}]})

        else:  # user
            out.append({"role": "user", "content": content})

    return "\n".join(system_parts), out


def _openai_tools_to_claude(tools: list) -> list:
    return [
        {
            "name":         t["function"]["name"],
            "description":  t["function"].get("description", ""),
            "input_schema": t["function"].get("parameters", {"type": "object", "properties": {}}),
        }
        for t in (tools or [])
        if "function" in t
    ]


def _claude_chat(messages: list, tools: list, model: str, timeout: int) -> dict:
    """Streaming Claude API call. Prints tokens live; returns normalized message dict."""
    try:
        import anthropic
    except ImportError as e:
        raise LLMUnavailableError(
            "[LLM] anthropic package not installed. Run: pip install anthropic") from e

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise LLMUnavailableError(
            "[LLM] ANTHROPIC_API_KEY not set. Export it to use the Claude backend.")

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    system, claude_msgs = _to_claude_messages(messages)
    claude_tools = _openai_tools_to_claude(tools)

    kwargs: dict = {
        "model":      model,
        "max_tokens": 8192,
        "messages":   claude_msgs,
    }
    if system:
        kwargs["system"] = system
    if claude_tools:
        kwargs["tools"] = claude_tools

    accumulated = ""
    tool_calls: list = []

    with client.messages.stream(**kwargs) as stream:
        for text in stream.text_stream:
            print(text, end="", flush=True)
            accumulated += text
        final = stream.get_final_message()

    if accumulated:
        print()

    for block in final.content:
        if block.type == "tool_use":
            tool_calls.append({
                "id":       block.id,
                "function": {"name": block.name, "arguments": block.input},
            })

    return {
        "role":       "assistant",
        "content":    accumulated,
        "tool_calls": tool_calls or None,
    }


# ── Public API ────────────────────────────────────────────────────────────────

def chat(messages: list, tools: list = None, timeout: int = 180) -> dict:
    """
    Single LLM call. Routes to Ollama or Claude based on the active model.
    Streams tokens to stdout and returns the assembled message dict:
      {"role": "assistant", "content": str, "tool_calls": list | None}
    """
    _resolve_active_model()
    model = _ACTIVE_MODEL
    if _is_claude(model):
        return _claude_chat(messages, tools, model, timeout)
    return _ollama_chat(messages, tools, model, timeout)


def _extract_tool_calls(content: str) -> Optional[list]:
    """Pull tool-call JSON objects that some models embed in the content field."""
    if not content:
        return None
    calls, depth, start = [], 0, None
    for i, ch in enumerate(content):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(content[start: i + 1])
                    if isinstance(obj, dict) and "name" in obj and "arguments" in obj:
                        calls.append({
                            "function": {"name": obj["name"], "arguments": obj["arguments"]}
                        })
                except json.JSONDecodeError:
                    pass
                start = None
    return calls or None


def _decode_tool_args(raw_args: Any) -> tuple[Optional[dict], str]:
    """Decode a tool call's arguments (a JSON string or an already-decoded
    value). Returns (args, "") for a JSON object, else (None, reason)."""
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return None, f"not valid JSON ({e.msg})"
    if not isinstance(raw_args, dict):
        return None, f"expected a JSON object, got {type(raw_args).__name__}"
    return raw_args, ""


def dispatch_tool_call(name: str, raw_args: Any, handlers: ToolHandlers) -> str:
    """Run one model-requested tool call and return the tool result text.

    A malformed call (unknown tool, arguments that are not a JSON object, that
    do not bind to the handler's signature, or a non-string value for a `str`
    parameter) becomes an error result for
    the model instead of an exception, so one bad call cannot kill a stage.
    Exceptions raised inside the handler body still propagate.
    """
    handler = handlers.get(name)
    if handler is None:
        return f"Unknown tool: {name}"
    args, problem = _decode_tool_args(raw_args)
    if args is None:
        return f"Error: invalid arguments for {name}: {problem}"
    signature = inspect.signature(handler)
    try:
        signature.bind(**args)
    except TypeError as e:
        return f"Error: invalid arguments for {name}: {e}"
    for key, value in args.items():
        param = signature.parameters.get(key)  # absent when bound via **kwargs
        if param is not None and param.annotation is str and not isinstance(value, str):
            return (f"Error: invalid arguments for {name}: "
                    f"{key} must be a string, got {type(value).__name__}")
    return handler(**args)


def _call_signature(name: str, args: Any) -> str:
    """Stable string signature of a tool call, used by the no-progress detector."""
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)[:400]}"
    except Exception:
        return f"{name}:{str(args)[:400]}"


def run_agent_loop(
    system: str,
    user: str,
    tools: list,
    handlers: ToolHandlers,
    history: list = None,
    on_tool_call: Optional[Callable[[str, dict], None]] = None,
    max_rounds: int = 20,
    repeat_limit: int = 3,
) -> str:
    """
    One-shot agentic task: system + optional history + user prompt, then
    run_tool_loop() until the model answers in plain text.
    """
    messages = list(history or [])
    messages.insert(0, {"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return run_tool_loop(messages, tools, handlers, on_tool_call=on_tool_call,
                         max_rounds=max_rounds, repeat_limit=repeat_limit)


def run_tool_loop(
    messages: list,
    tools: list,
    handlers: ToolHandlers,
    on_tool_call: Optional[Callable[[str, dict], None]] = None,
    max_rounds: int = 20,
    repeat_limit: int = 3,
) -> str:
    """
    Bounded tool loop on a caller-owned message list (appended in place, so a
    REPL keeps its history across turns):
      1. Call the model (streaming tokens to stdout)
      2. If it returns tool calls (native or embedded in content), run each
         through dispatch_tool_call() and feed the results back
      3. Repeat until a plain-text response is returned

    Tool call IDs (present when using Claude) are forwarded in tool result
    messages so the Claude backend can reconstruct its conversation format.

    No-progress detector: if the model issues the same tool call (same name +
    same args) `repeat_limit` times in a row, the loop returns early. This
    prevents pathological loops like running `pytest tests/` 15 times when
    there are no tests to find.

    Round limit: after `max_rounds` model calls without a plain-text answer
    the loop stops, prints why, and returns that notice; it never raises.
    """
    recent_signatures: list[str] = []

    for _ in range(max_rounds):
        msg = chat(messages, tools=tools)
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or _extract_tool_calls(msg.get("content", ""))
        if not tool_calls:
            return msg.get("content", "").strip()

        for index, call in enumerate(tool_calls):
            name        = call["function"]["name"]
            raw         = call["function"].get("arguments", {})
            args, _     = _decode_tool_args(raw)
            tool_use_id = call.get("id")  # present for Claude tool calls

            # ── No-progress detector ─────────────────────────────────────
            sig = _call_signature(name, raw if args is None else args)
            recent_signatures.append(sig)
            recent_signatures = recent_signatures[-repeat_limit:]
            if (len(recent_signatures) == repeat_limit
                    and len(set(recent_signatures)) == 1):
                print(f"\n    [agent] no-progress detector: {name!r} called "
                      f"{repeat_limit} times with the same args — ending loop early.")
                # Tell the model so it stops requesting the same call
                stop_note = (
                    f"\n[loop guard] You called {name} with the same arguments "
                    f"{repeat_limit} times in a row. That tool will not advance the "
                    f"task further. Stop calling it and respond with a plain text "
                    f"summary of what you've accomplished and what's blocking you.")
                # Answer this call and skip the rest of the message, so every
                # tool call has a result right after it (the Claude API
                # requires that), then add the note.
                messages.append(_tool_message(dispatch_tool_call(name, raw, handlers),
                                              tool_use_id))
                for skipped in tool_calls[index + 1:]:
                    messages.append(_tool_message(
                        "Skipped: the loop guard ended this turn.", skipped.get("id")))
                messages.append({"role": "user", "content": stop_note})
                # One more round, no tools, to extract a text answer
                final = chat(messages, tools=[])
                messages.append(final)
                return (final.get("content") or "").strip()

            if on_tool_call:
                on_tool_call(name, args or {})

            messages.append(_tool_message(dispatch_tool_call(name, raw, handlers),
                                          tool_use_id))

    notice = f"{ROUND_LIMIT_NOTICE} {max_rounds} rounds without a final answer."
    print(f"\n    {notice}")
    return notice


def stopped_at_round_limit(reply: str) -> bool:
    """True when run_tool_loop() returned its round-limit notice, i.e. the
    model never gave a final answer (callers must not read it as a result)."""
    return reply.startswith(ROUND_LIMIT_NOTICE)


def _tool_message(result: str, tool_use_id: Optional[str]) -> dict:
    tool_msg: dict = {"role": "tool", "content": result}
    if tool_use_id:
        tool_msg["tool_use_id"] = tool_use_id
    return tool_msg

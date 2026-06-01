"""
Multi-backend LLM client: Ollama (local) or Claude API (cloud).

Select via LOCAL_AI_MODEL env var or set_model() / --model flag:
  Ollama (default):  qwen2.5-coder:32b
  Claude (best):     claude-opus-4-7   (requires ANTHROPIC_API_KEY)

Both backends stream tokens to stdout and return the same message dict.
"""

import json
import os
import sys
import time
import urllib.request
import urllib.error
from typing import Optional

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

    # Fall back: prefer same family (e.g. qwen2.5-coder:* if asked for qwen2.5-coder:32b)
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
                print(
                    f"\n[LLM] Cannot reach Ollama after {retries} attempts — "
                    f"is it running? (`brew services start ollama`)\n{e}"
                )
                sys.exit(1)
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
    except ImportError:
        print("\n[LLM] anthropic package not installed. Run: pip install anthropic")
        sys.exit(1)

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("\n[LLM] ANTHROPIC_API_KEY not set. Export it to use the Claude backend.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)
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


def _call_signature(name: str, args: dict) -> str:
    """Stable string signature of a tool call, used by the no-progress detector."""
    try:
        return f"{name}:{json.dumps(args, sort_keys=True, default=str)[:400]}"
    except Exception:
        return f"{name}:{str(args)[:400]}"


def run_agent_loop(
    system: str,
    user: str,
    tools: list,
    handlers: dict,
    history: list = None,
    on_tool_call=None,
    max_rounds: int = 20,
    repeat_limit: int = 3,
) -> str:
    """
    Full agentic loop:
      1. Call the model with system + user prompt (streaming tokens to stdout)
      2. If it returns tool calls, execute them and feed results back
      3. Repeat until a plain-text response is returned

    Tool call IDs (present when using Claude) are forwarded in tool result
    messages so the Claude backend can reconstruct its conversation format.

    No-progress detector: if the model issues the same tool call (same name +
    same args) `repeat_limit` times in a row, the loop returns early. This
    prevents pathological loops like running `pytest tests/` 15 times when
    there are no tests to find.
    """
    messages = list(history or [])
    messages.insert(0, {"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    recent_signatures: list[str] = []

    for round_n in range(max_rounds):
        msg = chat(messages, tools=tools)
        messages.append(msg)

        tool_calls = msg.get("tool_calls") or _extract_tool_calls(msg.get("content", ""))
        if not tool_calls:
            return msg.get("content", "").strip()

        for call in tool_calls:
            name        = call["function"]["name"]
            raw         = call["function"].get("arguments", {})
            args        = json.loads(raw) if isinstance(raw, str) else raw
            tool_use_id = call.get("id")  # present for Claude tool calls

            # ── No-progress detector ─────────────────────────────────────
            sig = _call_signature(name, args)
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
                messages.append({"role": "user", "content": stop_note})
                # Append the result one last time so the conversation is consistent
                handler = handlers.get(name)
                result  = handler(**args) if handler else f"Unknown tool: {name}"
                tool_msg: dict = {"role": "tool", "content": result}
                if tool_use_id:
                    tool_msg["tool_use_id"] = tool_use_id
                messages.append(tool_msg)
                # One more round, no tools, to extract a text answer
                final = chat(messages, tools=[])
                return (final.get("content") or "").strip()

            if on_tool_call:
                on_tool_call(name, args)

            handler = handlers.get(name)
            result  = handler(**args) if handler else f"Unknown tool: {name}"

            tool_msg: dict = {"role": "tool", "content": result}
            if tool_use_id:
                tool_msg["tool_use_id"] = tool_use_id
            messages.append(tool_msg)

    return messages[-1].get("content", "").strip()

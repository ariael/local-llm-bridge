#!/usr/bin/env python3
r"""
local-llm-bridge — MCP server exposing a local Qwen model (on the GPU) as a set
of delegation tools for an *online* Claude Code session.

Design intent
-------------
The online Claude (real Anthropic API) stays the planner/orchestrator. When a
subtask is bounded and mechanical — summarize, classify, extract fields, rewrite
boilerplate, transform a file — Claude calls one of these tools instead of doing
the work itself. The heavy lifting (and the bulky output) happens on the local
GPU, so it costs zero Anthropic tokens.

Token-saving rule of thumb
--------------------------
An MCP tool RESULT still flows back into Claude's context and costs tokens. The
real saving comes from keeping that result *small*:
  - `delegate`        -> returns the model's answer (use for compact outputs:
                         a summary, a JSON object, a label).
  - `transform_file`  -> writes the model's (potentially large) output straight
                         to disk and returns only a short status line. Use this
                         for anything bulky (generated code, a rewritten doc):
                         the big text never enters Claude's context at all.

Transport: stdio (registered in Claude Code via .mcp.json).
Backend  : llama-server's OpenAI-compatible endpoint. NO LiteLLM needed here —
           this server speaks OpenAI Chat Completions directly. (LiteLLM is only
           needed for the separate "run Claude Code itself on Qwen" path in
           C:\AI\local-llm.)

Dependencies: only `mcp` (pip install mcp). HTTP uses the standard library so
there is nothing else to install.
"""

import json
import os
import time
import urllib.error
import urllib.request

from mcp.server import MCPServer

# Make the sibling module importable no matter how this server is launched.
import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import local_agent

# --- Configuration (override via environment in .mcp.json) ------------------
# Base URL of llama-server's OpenAI-compatible API. 127.0.0.1 on purpose, not
# "localhost": on this machine localhost resolves IPv6 (::1) first and wastes
# ~2s per request failing over to IPv4 (documented in C:\AI\local-llm).
LLAMA_BASE = os.environ.get("LOCAL_LLM_BASE", "http://127.0.0.1:8001/v1")
# Model alias — must match the --alias passed to llama-server.
MODEL = os.environ.get("LOCAL_LLM_MODEL", "local-model")
# Per-request timeout (seconds). Long default: a big prefill on a 35B MoE plus
# generation can take a while on a 24 GB card.
TIMEOUT = float(os.environ.get("LOCAL_LLM_TIMEOUT", "300"))

mcp = MCPServer("local-llm")


def _chat(system, user, max_tokens, temperature):
    """Call llama-server's /chat/completions and return the assistant text.

    Raises RuntimeError with a readable message if the backend is unreachable or
    returns an error — the caller (a tool) turns that into a tool error so the
    online Claude sees a clear signal instead of a silent empty result.
    """
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "stream": False,
        # Qwen3 is a "thinking" model: left on, it spends the whole token budget
        # in a <think> block (which llama.cpp routes to message.reasoning_content)
        # and message.content comes back EMPTY. These tools are a fast worker path
        # where we want the answer, not the reasoning — so disable thinking.
        "chat_template_kwargs": {"enable_thinking": False},
    }
    data = json.dumps(payload).encode("utf-8")
    url = LLAMA_BASE.rstrip("/") + "/chat/completions"
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )

    started = time.time()
    try:
        with urllib.request.urlopen(req, timeout=TIMEOUT) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Cannot reach local llama-server at %s (%s). "
            "Is it running? Start it with scripts\\Start-LlamaServer.ps1 -Commit."
            % (url, exc)
        )

    choices = body.get("choices") or []
    if not choices:
        raise RuntimeError("llama-server returned no choices: %s" % json.dumps(body)[:400])
    message = choices[0].get("message") or {}
    # Prefer the real answer. Fall back to reasoning_content only if a model/template
    # ignored enable_thinking and put everything in the reasoning channel — better a
    # thinking dump than a silent empty string.
    text = message.get("content") or message.get("reasoning_content") or ""
    usage = body.get("usage") or {}
    elapsed = time.time() - started
    # Attach a compact telemetry footer so Claude (and the user) can see the
    # local call actually happened and how big it was — without a separate tool.
    footer = "\n\n---\n[local-llm] %d in / %d out tok, %.1fs" % (
        usage.get("prompt_tokens", 0),
        usage.get("completion_tokens", 0),
        elapsed,
    )
    return text, footer


@mcp.tool()
def delegate(task: str, input_text: str = "", max_tokens: int = 1024, temperature: float = 0.2) -> str:
    """Offload a bounded subtask to the local Qwen model on the GPU.

    Use for tasks with a COMPACT result that you want back in context:
    summarizing, classifying, extracting fields to JSON, short rewrites, quick
    Q&A over a snippet. For bulky output (generated files, long rewrites) use
    `transform_file` instead so the big text never enters your context.

    Args:
        task: The instruction for the local model (what to do).
        input_text: Optional payload the task operates on (the text to summarize,
            classify, etc.). Kept separate from `task` for clarity.
        max_tokens: Cap on the local model's output length.
        temperature: Sampling temperature. Keep low (0.0-0.3) for deterministic
            extraction/classification.
    """
    system = (
        "You are a fast local worker model. Do exactly the requested task and "
        "nothing more. No preamble, no explanation, no apologies. If asked for "
        "JSON, output only valid JSON."
    )
    user = task if not input_text else (task + "\n\n---\n" + input_text)
    text, footer = _chat(system, user, max_tokens, temperature)
    return text + footer


@mcp.tool()
def transform_file(instruction: str, path: str, output_path: str = "", max_tokens: int = 8192, temperature: float = 0.2) -> str:
    """Read a file, have the local Qwen model transform it, write the result to
    disk, and return only a SHORT status line.

    This is the token-saving workhorse: the model's (possibly large) output goes
    straight to a file and never enters your context. Use it for generated code,
    rewritten documents, bulk reformatting, translation of a whole file, etc.

    Args:
        instruction: What transformation to apply to the file's contents.
        path: Absolute path of the input file to read.
        output_path: Where to write the result. Defaults to `path` (in-place).
        max_tokens: Cap on the local model's output length.
        temperature: Sampling temperature.
    """
    if not os.path.isfile(path):
        raise RuntimeError("Input file not found: %s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    target = output_path or path
    system = (
        "You are a fast local worker model that transforms file contents. "
        "Output ONLY the full transformed file content — no preamble, no code "
        "fences, no commentary. Preserve everything not covered by the "
        "instruction."
    )
    user = "INSTRUCTION:\n%s\n\n---FILE CONTENT---\n%s" % (instruction, content)
    text, footer = _chat(system, user, max_tokens, temperature)

    with open(target, "w", encoding="utf-8") as fh:
        fh.write(text)
    return "OK: wrote %d chars to %s%s" % (len(text), target, footer)


@mcp.tool()
def run_local_agent(task: str, workdir: str = "", max_steps: int = 12, timeout_s: int = 180, allow_shell: bool = False, allow_web: bool = False) -> str:
    """Run the local Qwen model as a confined SUB-AGENT for a bounded task.

    Unlike `delegate` (a single completion), this runs a small ReAct loop: the
    local model can read/write files and list dirs inside a sandbox, then calls
    `finish`. Everything is hard-confined to LOCAL_AGENT_ROOT — the model cannot
    touch the rest of the disk. Runs fully local (zero Anthropic tokens); you get
    back only a compact summary + list of changed files to VERIFY.

    Use for small, well-scoped work: scaffold files, mechanical multi-file edits,
    reformat/transform within a folder. Keep tasks tight — an A3B model is not a
    reliable open-ended autonomous agent; always review the result.

    Args:
        task: The bounded task, described clearly and self-contained.
        workdir: Optional subdirectory (under the sandbox root) to work in.
        max_steps: Tool-call rounds allowed (default 12, capped at 30).
        timeout_s: Wall-clock limit (default 180, capped at 600).
        allow_shell: Enable a cwd-confined run_command tool. OFF by default —
            only enable for trusted, reviewed tasks that truly need it.
        allow_web: Enable an http(s) GET fetch_url tool. OFF by default — fetched
            content is untrusted (prompt-injection risk with a small model).
    """
    result = local_agent.run_agent(
        task=task, workdir=(workdir or None), max_steps=max_steps,
        timeout_s=timeout_s, allow_shell=allow_shell, allow_web=allow_web,
    )
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def health() -> str:
    """Check whether the local llama-server backend is up and which model it serves."""
    url = LLAMA_BASE.rstrip("/") + "/models"
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            body = json.loads(resp.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        return "DOWN: %s unreachable (%s)" % (url, exc)
    ids = [m.get("id") for m in (body.get("data") or [])]
    return "UP: %s serving %s" % (LLAMA_BASE, ", ".join(ids) or "(no models listed)")


if __name__ == "__main__":
    mcp.run()

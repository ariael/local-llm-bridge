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
import backend
import model_update
import stats

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
    ok, msg = backend.ensure()  # lazily spin up llama-server if it's not running
    if not ok:
        raise RuntimeError("local backend unavailable: %s" % msg)

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
    return text, footer, {
        "prompt_tokens": usage.get("prompt_tokens", 0),
        "completion_tokens": usage.get("completion_tokens", 0),
        "elapsed_s": elapsed,
    }


_WORKER_SYSTEM = (
    "You are a fast local worker model. Do exactly the requested task and "
    "nothing more. No preamble, no explanation, no apologies. If asked for "
    "JSON, output only valid JSON."
)


@mcp.tool()
def delegate(task: str, input_text: str = "", max_tokens: int = 1024, temperature: float = 0.2, kind: str = "") -> str:
    """Offload a bounded subtask to the local Qwen model on the GPU.

    Use for tasks with a COMPACT result that you want back in context:
    summarizing, classifying, extracting fields to JSON, short rewrites, quick
    Q&A over a snippet. For bulky output (generated files, long rewrites) use
    `transform_file` instead so the big text never enters your context. For MANY
    small items, use `delegate_batch` (one call, one compact result array).

    Args:
        task: The instruction for the local model (what to do).
        input_text: Optional payload the task operates on (the text to summarize,
            classify, etc.). Kept separate from `task` for clarity.
        max_tokens: Cap on the local model's output length.
        temperature: Sampling temperature. Keep low (0.0-0.3) for deterministic
            extraction/classification.
        kind: Optional short label for the task shape (e.g. "summarize",
            "classify", "extract", "rewrite"). Only used for statistics grouping —
            it makes `local_stats` show which task shapes work vs. fail.
    """
    user = task if not input_text else (task + "\n\n---\n" + input_text)
    try:
        text, footer, u = _chat(_WORKER_SYSTEM, user, max_tokens, temperature)
    except Exception:
        stats.record("delegate", task, "error", elapsed_s=0, kind=kind)
        raise
    stats.record("delegate", task, "ok", u["prompt_tokens"],
                 u["completion_tokens"], u["elapsed_s"], kind=kind)
    return text + footer


@mcp.tool()
def delegate_batch(tasks: list, shared_instruction: str = "", max_tokens: int = 512, temperature: float = 0.2, kind: str = "") -> str:
    """Run MANY small tasks in ONE call — the efficient path for bulk work.

    Instead of N separate `delegate` calls (N tool round-trips, N results into your
    context), this processes a list locally and returns one compact JSON array.
    Ideal for "classify each of these 40 lines", "summarize each of these snippets",
    "extract fields from each record". Each item is independent.

    Args:
        tasks: A list of items. If `shared_instruction` is given, each item is the
            INPUT that the shared instruction operates on (e.g. instruction=
            "sentiment as one word", items=[review1, review2, ...]). If it's empty,
            each item is treated as a complete standalone task string.
        shared_instruction: One instruction applied to every item (optional).
        max_tokens: Per-item cap on the local model's output length.
        temperature: Sampling temperature. Keep low for classification/extraction.
        kind: Optional task-shape label for statistics (see `delegate`).

    Returns compact JSON: {results:[{i, result}|{i, error}], n, failed, tokens...}.
    Keep per-item outputs short — the point of batching is many small results.
    """
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("`tasks` must be a non-empty list.")
    results = []
    tot_in = tot_out = failed = 0
    started = time.time()
    for i, item in enumerate(tasks):
        item = "" if item is None else str(item)
        user = item if not shared_instruction else (shared_instruction + "\n\n---\n" + item)
        try:
            text, _footer, u = _chat(_WORKER_SYSTEM, user, max_tokens, temperature)
            results.append({"i": i, "result": text.strip()})
            tot_in += u["prompt_tokens"]
            tot_out += u["completion_tokens"]
        except Exception as exc:
            results.append({"i": i, "error": str(exc)[:200]})
            failed += 1
    elapsed = time.time() - started
    status = "ok" if failed == 0 else ("partial" if failed < len(tasks) else "error")
    stats.record("delegate_batch", shared_instruction or (str(tasks[0])[:120]),
                 status, tot_in, tot_out, elapsed, kind=kind,
                 extra={"n": len(tasks), "failed": failed})
    return json.dumps({
        "results": results, "n": len(tasks), "failed": failed,
        "tokens_in": tot_in, "tokens_out": tot_out, "elapsed_s": round(elapsed, 1),
    }, ensure_ascii=False, indent=2)


@mcp.tool()
def transform_file(instruction: str, path: str, output_path: str = "", max_tokens: int = 8192, temperature: float = 0.2, kind: str = "") -> str:
    """Read a file, have the local Qwen model transform it, write the result to
    disk, and return only a SHORT status line.

    This is the token-saving workhorse: the model's (possibly large) output goes
    straight to a file and never enters your context. Use it for generated code,
    rewritten documents, bulk reformatting, translation of a whole file, etc.

    Truncation safety: if the model hits `max_tokens` the output is likely cut off.
    In that case the result is written to `<path>.partial` (NOT over the original)
    and a WARNING is returned — re-run with a higher max_tokens. Bump max_tokens for
    large files; the output must fit in it.

    Args:
        instruction: What transformation to apply to the file's contents.
        path: Absolute path of the input file to read.
        output_path: Where to write the result. Defaults to `path` (in-place).
        max_tokens: Cap on the local model's output length. Must exceed the
            expected output size, or the file will be truncated.
        temperature: Sampling temperature.
        kind: Optional task-shape label for statistics (see `delegate`).
    """
    if not os.path.isfile(path):
        raise RuntimeError("Input file not found: %s" % path)
    with open(path, "r", encoding="utf-8") as fh:
        content = fh.read()

    system = (
        "You are a fast local worker model that transforms file contents. "
        "Output ONLY the full transformed file content — no preamble, no code "
        "fences, no commentary. Preserve everything not covered by the "
        "instruction."
    )
    user = "INSTRUCTION:\n%s\n\n---FILE CONTENT---\n%s" % (instruction, content)
    try:
        text, footer, u = _chat(system, user, max_tokens, temperature)
    except Exception:
        stats.record("transform_file", instruction, "error", elapsed_s=0,
                     kind=kind, extra={"path": path})
        raise

    # If the model spent the whole budget, the output was almost certainly cut
    # off. Never silently overwrite the source with a truncated file: divert to a
    # .partial sibling and warn instead.
    truncated = u["completion_tokens"] >= max_tokens
    target = output_path or path
    diverted = truncated and not output_path
    if diverted:
        target = path + ".partial"
    with open(target, "w", encoding="utf-8") as fh:
        fh.write(text)

    stats.record("transform_file", instruction, "truncated" if truncated else "ok",
                 u["prompt_tokens"], u["completion_tokens"], u["elapsed_s"], kind=kind,
                 extra={"path": path, "output_path": target, "out_chars": len(text),
                        "truncated": truncated})
    if truncated:
        where = ("%s (original left intact)" % target) if diverted else target
        return ("WARNING: output likely TRUNCATED — hit max_tokens=%d. Wrote %d chars to "
                "%s. Re-run with a higher max_tokens.%s" % (max_tokens, len(text), where, footer))
    return "OK: wrote %d chars to %s%s" % (len(text), target, footer)


@mcp.tool()
def run_local_agent(task: str, workdir: str = "", max_steps: int = 12, timeout_s: int = 180, allow_shell: bool = False, allow_web: bool = False, kind: str = "") -> str:
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
    ok, msg = backend.ensure()  # spin up llama-server on demand
    if not ok:
        stats.record("run_local_agent", task, "backend_error", elapsed_s=0, kind=kind)
        return json.dumps({"status": "backend_error", "summary": msg}, ensure_ascii=False)
    backend.touch()
    result = local_agent.run_agent(
        task=task, workdir=(workdir or None), max_steps=max_steps,
        timeout_s=timeout_s, allow_shell=allow_shell, allow_web=allow_web,
    )
    backend.touch()
    stats.record("run_local_agent", task, result.get("status", "?"),
                 result.get("prompt_tokens", 0), result.get("completion_tokens", 0),
                 result.get("elapsed_s", 0), kind=kind,
                 extra={"steps_used": result.get("steps_used"),
                        "files_changed": result.get("files_changed")})
    return json.dumps(result, ensure_ascii=False, indent=2)


@mcp.tool()
def mark_outcome(outcome: str, note: str = "", kind: str = "") -> str:
    """Record whether the LAST local-model result was actually usable.

    Call this right after you VERIFY a `delegate` / `transform_file` /
    `run_local_agent` result, so the statistics reflect real usefulness — not just
    whether the call completed. This is the signal that answers "what does the local
    model actually handle well?" and drives future tuning of what to delegate.

    Args:
        outcome: "accepted" (kept the result as-is), "rejected" (unusable, did it
            myself), or "redone" (kept the idea but had to fix/redo it).
        note: Optional one-line reason (what was wrong / why it worked).
        kind: Optional task-shape label matching the call you're grading.
    """
    stats.record_outcome(outcome, note=note, kind=kind)
    return "recorded outcome: %s" % (outcome or "").strip().lower()


@mcp.tool()
def copy_in(source_path: str, dest: str = "") -> str:
    """Copy a file from anywhere on disk INTO the agent sandbox.

    Use to stage inputs before a `run_local_agent` task: the agent is hard-confined
    to LOCAL_AGENT_ROOT and cannot read the rest of the disk, so anything it needs
    must be copied in first. The destination is confined under the sandbox root —
    it cannot escape via `..`, absolute paths, or symlinks.

    Args:
        source_path: Absolute path of the file to copy in (read from anywhere).
        dest: Destination path RELATIVE to the sandbox root. Defaults to the
            source file's basename at the sandbox root.
    """
    import shutil
    if not os.path.isfile(source_path):
        raise RuntimeError("Source file not found: %s" % source_path)
    rel = dest or os.path.basename(source_path)
    try:
        target = local_agent._confine(local_agent.AGENT_ROOT, rel)
    except ValueError as exc:
        raise RuntimeError("Refused: %s" % exc)
    os.makedirs(os.path.dirname(target) or local_agent.AGENT_ROOT, exist_ok=True)
    shutil.copy2(source_path, target)
    return "OK: copied %s -> %s (in sandbox)" % (source_path, target)


@mcp.tool()
def copy_out(source: str, dest_path: str, overwrite: bool = False) -> str:
    """Copy a result file OUT of the agent sandbox to a chosen path on disk.

    Use to retrieve what `run_local_agent` produced. The source is confined to the
    sandbox root; the destination is any path you choose. Refuses to overwrite an
    existing destination unless overwrite=true, so a stray call can't clobber your
    files.

    Args:
        source: Path RELATIVE to the sandbox root (the file the agent produced).
        dest_path: Absolute destination path to copy it to.
        overwrite: Allow overwriting an existing destination file. Default False.
    """
    import shutil
    try:
        src = local_agent._confine(local_agent.AGENT_ROOT, source)
    except ValueError as exc:
        raise RuntimeError("Refused: %s" % exc)
    if not os.path.isfile(src):
        raise RuntimeError("Sandbox file not found: %s" % source)
    if os.path.exists(dest_path) and not overwrite:
        raise RuntimeError("Destination exists (pass overwrite=true to replace): %s" % dest_path)
    os.makedirs(os.path.dirname(dest_path) or ".", exist_ok=True)
    shutil.copy2(src, dest_path)
    return "OK: copied %s -> %s (out of sandbox)" % (src, dest_path)


@mcp.tool()
def local_stats(limit: int = 0) -> str:
    """Report telemetry on local-model delegations: success rates, tokens burned
    locally, estimated Anthropic tokens saved, and a per-tool breakdown.

    Every `delegate`, `transform_file`, and `run_local_agent` call is logged. Use
    this to see what the local model handles well vs. where it fails — the data to
    tune what gets delegated. Cost is free (local GPU); this reads a local file.

    Args:
        limit: If > 0, summarize only the last N calls. 0 = all history.
    """
    return json.dumps(stats.summary(limit=limit or None), ensure_ascii=False, indent=2)


@mcp.tool()
def start_backend() -> str:
    """Start the local llama-server on the GPU (loads the model into VRAM).

    You normally don't need this — the delegation tools auto-start the backend on
    first use. Call it to pre-warm before a batch of local work. The backend also
    auto-stops after an idle period, and can be stopped explicitly with
    `stop_backend` to free the GPU (e.g. for gaming).
    """
    ok, msg = backend.ensure()
    return ("OK: " if ok else "FAILED: ") + msg + "\n" + backend.status()


@mcp.tool()
def stop_backend(force: bool = False) -> str:
    """Stop the local llama-server and free the GPU (e.g. before playing a game).

    Only stops a backend this tool started, unless force=true. The delegation
    tools will transparently start it again next time they're used.
    """
    return backend.stop(force=force)


@mcp.tool()
def check_model_update(force: bool = False) -> str:
    """Report whether a newer GGUF of the tracked model is available.

    Returns the cached result of the server's own ~monthly self-check (no
    external scheduler involved). Pass force=true to re-check Hugging Face right
    now. Downloads nothing — actually fetching a newer model is a deliberate,
    ~18 GB step done via `python server/model_update.py apply --commit`.
    """
    r = model_update.maybe_check(force=force)
    if r["fresh"]:
        return "[checked just now] " + r["verdict"]
    if r["age_days"] is None:
        return r["verdict"]
    return "[cached, checked %.0f days ago] %s" % (r["age_days"], r["verdict"])


@mcp.tool()
def health() -> str:
    """Report whether the local backend is up, whether we own it, and idle time.

    Does NOT start anything — safe to call while gaming to check the GPU is free.
    """
    return backend.status()


# Self-run the model-update check in the background at startup. maybe_check()
# throttles itself to ~monthly via a state file, so this is a no-op most starts
# and never blocks startup or a tool call. No external scheduler needed.
def _bg_model_check():
    try:
        model_update.maybe_check()
    except Exception:
        pass


import threading as _threading
_threading.Thread(target=_bg_model_check, name="model-update-selfcheck", daemon=True).start()


if __name__ == "__main__":
    mcp.run()

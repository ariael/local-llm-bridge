# local-llm-bridge

Expose a **local Qwen model running on the GPU** as MCP tools that an **online
Claude Code** session can call. Claude stays the planner; the local model does
the bounded, mechanical subtasks — saving Anthropic tokens.

This is the productionized "delegation" layer. It is a sibling to the existing
`C:\AI\local-llm\` stack, which does a *different* thing (runs Claude Code
itself on Qwen via `ANTHROPIC_BASE_URL` + LiteLLM). See
[Relationship to `C:\AI\local-llm`](#relationship-to-cailocal-llm).

---

## Architecture

```
online Claude Code  (real Anthropic API — planner/orchestrator)
   │  calls MCP tool: delegate / transform_file / health
   ▼
local-llm MCP server  (server/local_llm_mcp.py, stdio)
   │  OpenAI Chat Completions  (NO LiteLLM on this path)
   ▼
llama-server  :8001  (Vulkan build, scripts/Start-LlamaServer.ps1)
   ▼
Radeon RX 7900 XTX  →  Qwen A3B (GGUF)
```

Key point: the MCP server speaks the OpenAI protocol directly to llama-server,
so **LiteLLM is not needed here**. LiteLLM only exists in `C:\AI\local-llm` to
translate Anthropic↔OpenAI for the "run Claude *on* Qwen" path.

---

## The tools

| Tool | Returns | Use for |
|---|---|---|
| `delegate(task, input_text?, max_tokens?, temperature?)` | the model's answer (goes into Claude's context) | **compact** results: summaries, classification, field extraction to JSON, short rewrites, Q&A over a snippet |
| `transform_file(instruction, path, output_path?, ...)` | a short status line only | **bulky** output: generated code, whole-file rewrites, bulk reformatting, translating a file — the big text is written to disk and never enters Claude's context |
| `run_local_agent(task, workdir?, max_steps?, timeout_s?, allow_shell?, allow_web?)` | compact JSON: summary + files_changed + step log | a bounded **multi-step** subtask where the model reads/writes files itself in a sandbox (scaffold files, mechanical multi-file edits) — see [Agent policy](#agent-policy) |
| `start_backend()` | status line | pre-warm the model before a batch (optional — delegation tools auto-start it) |
| `stop_backend(force?)` | status line | **free the GPU** immediately (e.g. before gaming) |
| `health()` | up/down, whether we own it, idle time | check without starting anything |

### On-demand GPU (shares the card with games)

The model is **not** an always-on service — the GPU is shared with games. Instead:

- **Auto-start on first use.** The card stays free until Claude actually calls
  `delegate` / `transform_file` / `run_local_agent`; the first call spins up
  llama-server (loads the model, ~15s) and reuses it after.
- **Idle auto-stop.** After `LOCAL_LLM_IDLE_STOP_S` seconds (default 600) with no
  delegation, the backend stops itself and releases VRAM.
- **Explicit control.** `stop_backend` frees the GPU right now; `start_backend`
  pre-warms it. `health` reports status without starting anything.

The backend is launched detached and tracked by a PID file, so `stop_backend`
works even from a later session. Set `LOCAL_LLM_AUTOSTART=0` to require a manual
`scripts\Start-LlamaServer.ps1` instead.

### Why two tools — the token-saving rule

An MCP tool **result still costs Claude tokens** (it flows back into context).
So the saving is real only when the result is small:

- Use `delegate` when you *want* the answer back and it's short.
- Use `transform_file` when the output is large — Qwen writes it to disk, Claude
  gets only `OK: wrote N chars to <path>`. This is where the biggest token
  savings come from.

Good candidates to delegate: summarization, classification/labelling, JSON
extraction, log parsing, boilerplate/scaffold generation, mechanical
refactors, whole-file translation. Keep architecture, hard reasoning, and final
review on Claude.

---

## Agent policy

`run_local_agent` is the only tool that lets the local model *act* — a small
ReAct loop (`server/local_agent.py`) where Qwen calls tools itself and iterates.
Because a ~3B-active model is not a reliable autonomous agent, and because it
runs without a human approving each step, the policy is deliberately tight.
These defaults are chosen for a driver (the online Claude) that hands off small,
well-scoped work and then verifies the result.

**Hard sandbox (the real guardrail).** Every file/dir/command path is resolved
and must stay under `LOCAL_AGENT_ROOT` (default `C:\AI\agent-sandbox`).
Traversal (`..`), absolute paths, and symlink escapes are rejected **in Python**,
so confinement does not rely on the model behaving. To point the agent at a real
project, set `LOCAL_AGENT_ROOT` to that project's folder — the model still cannot
escape whatever root the human configured.

**What it can do**

| Capability | Default | Notes |
|---|---|---|
| `read_file` / `write_file` / `list_dir` | **on** | text files, inside the sandbox only |
| `finish` | on | ends the loop with a summary |
| `run_command` (shell) | **off** | `allow_shell=true` — cwd-confined, per-command timeout, destructive-pattern deny-list. Only for trusted, reviewed tasks. |
| `fetch_url` (web GET) | **off** | `allow_web=true` — http(s) only, size/time capped. Fetched text is **untrusted** (prompt-injection risk with a small model). |
| anything outside the sandbox | **never** | rejected before the tool runs |

**Limits**

| | Default | Hard cap | Why |
|---|---|---|---|
| `max_steps` (tool rounds) | 12 | 30 | past ~15 steps an A3B model's tool-calling degrades and it tends to loop |
| `timeout_s` (wall clock) | 180 | 600 | loop is killed if exceeded |

**Contract.** Runs fully local (zero Anthropic tokens). Returns a compact JSON
result — `status`, `summary`, `files_changed`, `steps_used`, `log` — not a
transcript. The driver is expected to **review `files_changed` before trusting
them**. Good tasks: scaffold a few files, mechanical multi-file edits, reformat a
folder. Bad tasks: open-ended "figure it out", anything needing judgement, or
anything that must be right without review.

## Setup

1. **Install the MCP server dependency** (only `mcp`; HTTP uses the stdlib):
   ```bash
   pip install -r server/requirements.txt
   ```

2. **Start the local backend** (leave it running in its own window):
   ```powershell
   .\scripts\Start-LlamaServer.ps1 -Commit
   ```
   Dry-run first (`-DryRun`, the default) to see the exact command. Update
   `-ModelPath` when the Qwen model version changes.

3. **Register the MCP server** with whatever project you want to delegate from:
   copy the block in [`.mcp.json.example`](.mcp.json.example) into that
   project's `.mcp.json`, or run:
   ```bash
   claude mcp add local-llm -- python C:\GitHub\local-llm-bridge\server\local_llm_mcp.py
   ```

4. **Verify** from a Claude Code session: call the `health` tool — it should
   report `UP: ... serving local-model`.

### Smoke test

With the backend running, check the whole chain (health, delegate, transform_file)
without needing a Claude Code session:

```bash
python scripts/smoke_test.py
```

Exit code 0 means all three tools answered correctly (a summary, a `positive`
classification, a JSON extraction, and an uppercased temp file).

---

## Production status & backlog

Done:
- [x] MCP server with `delegate` / `transform_file` / `health` (`server/local_llm_mcp.py`)
- [x] Standalone persistent backend launcher (`scripts/Start-LlamaServer.ps1`)
- [x] Registration example (`.mcp.json.example`)

To do:
- [ ] **Model check/update** — confirm the current Qwen A3B version on the box
      (LM Studio / HF). As of 2026-08 there is no "Qwen3.8"; current A3B-class
      releases are `Qwen3-30B-A3B-Instruct-2507`, `Qwen3-Coder-30B-A3B`, and
      `Qwen3-Next-80B-A3B`. Pick one, download the GGUF, update `-ModelPath`,
      then re-run `C:\AI\local-llm\Test-ToolCalling.ps1` to confirm tool-calling.
- [ ] **Run the backend as a service** (Windows scheduled task at logon, or NSSM)
      so the MCP tools always have a backend without a manual window.
- [ ] **Smoke test** the full chain: `health`, then a `delegate` summarize, then
      a `transform_file` on a scratch file; confirm token footer looks sane.
- [ ] **Delegation policy** — optionally add a short skill / CLAUDE.md snippet in
      consumer projects telling Claude *when* to reach for these tools.
- [ ] Consider a `classify`/`extract_json` convenience tool if `delegate` prompts
      get repetitive.

---

## Relationship to `C:\AI\local-llm`

| | `C:\AI\local-llm` (existing) | this repo (`local-llm-bridge`) |
|---|---|---|
| What runs the agent loop | **Qwen** (Claude Code pointed at local model) | **Claude** (real API) |
| Local model's role | *is* the assistant | a callable worker tool |
| Needs LiteLLM | yes (Anthropic→OpenAI translation) | no (MCP server speaks OpenAI) |
| Launch | `local-ai -Commit` | `Start-LlamaServer.ps1 -Commit` + MCP registration |
| Token cost | ~zero (fully offline) | Claude tokens for planning + tool round-trips; Qwen does the bulk work |

Both share the same llama.cpp Vulkan build and model files. The benchmark
rationale (measure prefill, Vulkan > HIP) lives in
`C:\AI\local-llm\reports\phase2-summary.md`.

---

## Notes / gotchas (learned the hard way)

- **Qwen3 thinking must be off for worker tasks.** Qwen3 is a reasoning model;
  left on, it spends the whole token budget in a `<think>` block (llama.cpp puts
  that in `message.reasoning_content`) and `message.content` comes back empty.
  The server sends `chat_template_kwargs.enable_thinking=false` on every call.
- **`mcp` 2.x renamed `FastMCP` → `MCPServer`** (same `@tool()` decorator, same
  `run()` defaulting to stdio). Requires `mcp>=2.0.0`.
- **Use `127.0.0.1`, never `localhost`.** On this box `localhost` resolves IPv6
  (`::1`) first and wastes ~2s per request failing over to IPv4.
- The Vulkan `llama-server.exe` is a ~9 KB thin loader next to a
  `llama-server-impl.dll` — that small size is normal, not a broken build.

## License

MIT — see [LICENSE](LICENSE).
